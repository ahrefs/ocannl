open Base
module Lazy = Utils.Lazy
module Nd = Ndarray
module Tn = Tnode

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_LOW_LEVEL=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_LOW_LEVEL"]

module Scope_id = struct
  type t = { tn : Tn.t; scope_id : int } [@@deriving sexp_of, equal, hash, compare]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type scope_id = Scope_id.t = { tn : Tn.t; scope_id : int }
[@@deriving sexp_of, equal, hash, compare]

let get_scope =
  let uid = ref 0 in
  fun tn ->
    Int.incr uid;
    { tn; scope_id = !uid }

(** How a loop's iterations map to hardware; see docs/proposals/axis-types-for-loops.md. [Serial] is
    an ordinary for-loop. [Grid] / [Workgroup] bind the loop index to a GPU grid / workgroup (block,
    threadgroup) hardware index instead of looping; [Workgroup_reduce] is a [Workgroup] axis
    participating in a workgroup-cooperative reduction (see the contract below). [Unrolled] is
    emitted as the repeated body with substituted constants. [Vectorized] renders as a serial loop
    annotated with the backend's vectorization pragmas when it provides them (gh-ocannl-164; a plain
    serial loop otherwise) — like the hardware kinds, the annotating pass asserts iteration
    independence. Hardware slots are positional: among a kernel's loops of one kind, the innermost
    binds [.x], then [.y], [.z]. Annotated loops must have [from_ = 0] and iterations with no
    cross-iteration dependencies. [Workgroup_reduce] is the labelled exception; its body must either
    stage its communication explicitly through workgroup-shared nodes and barriers (rendered by
    binding the index like [Workgroup]), or be a single accumulation statement
    [acc = op(acc, contrib)] over an associative-commutative [op] with the accumulator's indices
    free of the loop index — the renderer then owns the communication: warp/simdgroup shuffles on
    GPU backends (gh-ocannl-462), the plain serial loop on CPU backends. Like [Vectorized], the
    annotation licenses reassociating the (floating-point) reduction. *)
type axis_type = Serial | Grid | Workgroup | Workgroup_reduce | Unrolled | Vectorized
[@@deriving sexp, compare, equal]

(** Loop keyword used by the human-readable printers: plain [for] for [Serial] (unchanged legacy
    output), [for@<axis>] otherwise. *)
let axis_type_label = function
  | Serial -> "for"
  | Grid -> "for@grid"
  | Workgroup -> "for@workgroup"
  | Workgroup_reduce -> "for@workgroup_reduce"
  | Unrolled -> "for@unrolled"
  | Vectorized -> "for@vectorized"

(** Which pass minted a [Local_scope] (gh-ocannl-687). The construct has two producers, and a
    consumer that walks the IR looking for schedulable structure means only one of them:

    - [Inlined_computation] -- virtualization's inline of a virtual node's computation at a read
      site ([virtual_llc], and the CSE / simplification rewrites that carry those scopes along). The
      loops inside such a body are the inlined node's own iteration space, re-instantiated per use
      site; no [Schedule] op has ever targeted them.
    - [Schedule_minted] -- the accumulator localization built by [Schedule]'s materializing [Unroll]
      and by [Partition] (gh-ocannl-639), and by [C_syntax.try_localize_serial_reduce]: a running
      value for a MATERIALIZED cell, whose body holds the very per-step / per-segment loops
      [Schedule.rewrite_loop] retargets.

    This is the fact [Autotune.collect_loops] needs: it enumerates loops inside [Schedule_minted]
    scopes only, so the action menu does not spend its per-unit budget proposing splits, swaps,
    unrolls and vectorize retypes for inlined interpolation and reduction loops that no schedule op
    has ever been able to reach.

    Deliberately NOT the mechanism behind {!scope_target_rejection}: that one asks whether a scope
    was in the program a given [optimize] call was HANDED, which is a per-call fact -- a
    virtualizer-minted scope handed back into a second [optimize] still carries
    [Inlined_computation] and still is not that call's to retract -- and hand-built IR has no honest
    way to spell "not mine". See {!input_scope_ids}. *)
type scope_mint = Inlined_computation | Schedule_minted [@@deriving sexp, compare, equal]

(** The iteration order of a {!t.Scan_loop} (gh-ocannl-696): [Forward] runs the index from [from_]
    up to [to_], [Backward] from [to_] down to [from_]. Part of the construct from day one because
    every adjoint of a forward scan is a backward scan over the same range. *)
type scan_direction = Forward | Backward [@@deriving sexp, compare, equal]

type t =
  | Noop
  | Comment of string
  | Staged_compilation of ((unit -> PPrint.document)[@equal.ignore] [@compare.ignore])
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; axis : axis_type }
  | Scan_loop of {
      index : Indexing.symbol;
      from_ : int;
      to_ : int;
      direction : scan_direction;
      carried : carried list;
      body : t;
    }
      (** A loop with declared loop-carried scalar state (gh-ocannl-696): the minimal recurrence
          construct behind cumulative ops, online softmax and top-k. Semantics, for [carried] =
          [c_1 .. c_n]: each [c.prev] is set to [c.init] once, before the first iteration; then for
          every value of [index] in [from_ .. to_] taken in [direction], [body] runs reading the
          previous iteration's state through [Get_local c.prev] and producing the next through
          [Set_local c.next]; after the body, every [c.prev] takes its [c.next] simultaneously
          (phi-style rotation, so a body may read any old value after any new one is written). A
          dead range ([to_ < from_]) is malformed and refused by the validator: a producer with no
          iterations emits [Noop], so no walker needs a dead-scan convention. The final state is not
          readable after the loop: a body that wants a trajectory or a final value writes it to a
          tensor node itself.

          Contract, enforced by {!validate_scan_loops} at both ends of the pipeline: [c.prev] and
          [c.next] are ids, pairwise distinct across [carried], over one node DECLARED virtual
          ([c.prev.tn == c.next.tn]), which names and types the state and is never a buffer;
          [c.next] is written exactly once, as a top-level statement of [body] (not under a guard or
          a nested loop), and read only by statements after that write; nothing writes [c.prev];
          [c.init] reads no carried local and does not mention [index]. The state is scalars only,
          of unbounded arity -- small fixed extents unroll into it; there is no dynamic indexing
          into state.

          Placement contract: a virtualization candidate whose captured computation contains a scan
          -- one written inside the body, one fed by a scan through a scope local, or one merely
          enclosing a sibling scan -- is refused ([Non_virtual 148]): a cell's value depends on the
          whole prefix through state the index does not parameterize, so per-cell recomputation at a
          read site is unbounded and wrong. Reads inside the body inline as usual (an independent
          producer replays soundly at any position of the scan). Schedule transforms neither target
          the scan's own index nor reach the loops inside its body (the loop is opaque to
          [Schedule.find_loops_env] / [rewrite_loop], so an op naming either symbol declines with
          the usual no-such-loop [Invalid_argument]); enclosing loops keep their full menu, since
          the state is per-iteration-of-the-enclosing-loop local scratch. The index is an ordinary
          affine loop symbol for footprint purposes ({!loop_bounds}, {!affine_accesses}, interval
          analysis) -- a scan's accesses stay affine and dense even though its values are serial. *)
  | Zero_out of Tn.t
  | Set of { tn : Tn.t; idcs : Indexing.axis_index array; llsc : scalar_t; mutable debug : string }
  | Set_dynamic of {
      tn : Tn.t;
      idcs : Indexing.axis_index array;
          (** Static everywhere except [dyn_axis] (a [Fixed_idx 0] placeholder there). *)
      dyn_axis : int;  (** Which [idcs] slot is replaced by [dyn_value] at codegen time. *)
      dyn_value : scalar_arg;
          (** Integer-valued index spliced into the row-major offset at [dyn_axis]. A nested scalar:
              all recursive scalar traversals must descend into it. *)
      llsc : scalar_t;
      mutable debug : string;
    }
      (** A scatter: like [Set] but the write lands at a runtime row of axis [dyn_axis] — the write
          counterpart of {!Get_dynamic}. gh-466: produced only by [rewrite_one_hot_reductions]
          (transposed one-hot pattern, the embedding-table gradient); never constructed by
          [Assignments] lowering. The enclosing guard (when interval analysis has not discharged it)
          guarantees [dyn_value] is in range before the write executes. Loops whose index reaches
          [dyn_value] carry a cross-iteration write dependency, so schedule analyses must treat this
          write as statically unknown (never parallelize/align over it). *)
  | Set_from_vec of {
      tn : Tn.t;
      idcs : Indexing.axis_index array;
      length : int;
      vec_unop : Ops.vec_unop;
      arg : scalar_arg;
      mutable debug : string;
    }
  | Set_local of scope_id * scalar_t
  | Declare_local of { id : scope_id; needs_init : bool }
  | Workgroup_barrier
      (** Workgroup-scoped synchronization ([__syncthreads()] / [threadgroup_barrier]). An opaque
          effectful statement: no CSE, hoisting, or code motion across it. Grid-scoped
          synchronization is deliberately not representable. *)
  | If of { cond : scalar_arg; body : t }
      (** Guarded statement: [body] executes iff [cond] is nonzero (renders as
          [if (cond != 0) { body }], mirroring how [Where] renders its condition). Introduced by
          launch-extent guards on hardware-annotated loops whose extent is smaller than the per-slot
          launch dimension (docs/proposals/axis-types-for-loops.md §2, construct-then-fold:
          [simplify_llc] erases a guard whose condition an interval proves), and available to
          hand-built IR. A conditional write is never a definite write; virtualization treats
          guarded computations as non-inlineable in v1. *)
  | Tile_mma of {
      d : Tn.t * Indexing.axis_index array;  (** Accumulator block base. *)
      a : Tn.t * Indexing.axis_index array;
      b : Tn.t * Indexing.axis_index array;
      ta : bool;
          (** [a] is stored transposed: its tile axes are [k, i]-major rather than [i, k]. *)
      tb : bool;
          (** [b] is stored transposed: its tile axes are [j, k]-major rather than [k, j]. *)
      m : int;
      n : int;
      k : int;  (** Covered block extents (multiples of the backend's intrinsic tile). *)
      ldd : int;
      lda : int;
      ldb : int;
          (** Leading-dimension strides in elements: the address distance between consecutive
              tile-major-axis lines of each operand. Historically each tile spanned its tnode's last
              two axes, making the stride the minor dim; with batched sites (gh-ocannl-528) the tile
              major axis may sit further out (an interior batch axis between the row and column
              axes), so the stride is recorded explicitly by [Schedule.Tensorize]. *)
      lane : Indexing.symbol;  (** The cooperating [Workgroup] axis (extent = SIMD width). *)
      tile : Register_tile.t option;
          (** The C-tile geometry the register-tiled CPU rendering must use (gh-ocannl-619), carried
              from {!Schedule.optop.Tensorize}; [None] lets the renderer choose. Ignored by the
              intrinsic (tensor-core) emissions. *)
      fallback : t;  (** Semantically equivalent scalar micro-kernel over fresh serial symbols. *)
    }
      (** Cooperative tile multiply-accumulate (docs/proposals/tensorize-mma.md):
          [d[i,j] += Σ_{l<k} a[i,l] * b[l,j]] for [i < m], [j < n], relative to the operands' base
          index vectors, executed jointly by the threads of the [lane] axis (tensor cores /
          [simdgroup_matrix]). The per-lane ownership of tile elements is architecture-defined and
          deliberately opaque — the [lane] index must not occur in the base indices. Each operand's
          tile is a 2-D slice with the minor tile axis at the tnode's last axis (stride 1) and the
          major tile axis at the recorded leading-dimension stride ([ldd]/[lda]/[ldb]) — the tnode's
          second-to-last axis in the plain case, further out for batched sites. With [ta] (resp.
          [tb]) the stored layout of [a] (resp. [b]) is the transpose of its role — emissions load
          tiles with the hardware transpose flag ([simdgroup_load]'s [transpose_matrix], wmma's
          [col_major]) and swap the tile-offset arithmetic; the scalar [fallback] carries the
          original indexing and is unaffected. Backends without an MMA hook render [fallback] once
          per simdgroup, under an [if (lane == 0)] guard — the renderer's obligation, keyed off
          [lane]. The statement validates like {!Workgroup_barrier} (it is one for code-motion and
          divergence purposes) plus a write of [d] for the coverage rule. *)
[@@deriving sexp_of, equal]

and scalar_t =
  | Local_scope of {
      id : scope_id;
      body : t;
      orig_indices : Indexing.axis_index array;
      mint : scope_mint;  (** Which pass built this scope; see {!scope_mint}. *)
    }
  | Get_local of scope_id
  | Get of Tn.t * Indexing.axis_index array
  | Get_dynamic of {
      tn : Tn.t;  (** The gathered table; treated as a read of [tn], like [Get]. *)
      idcs : Indexing.axis_index array;  (** Static everywhere except [dyn_axis]. *)
      dyn_axis : int;  (** Which [idcs] slot is replaced by [dyn_value] at codegen time. *)
      dyn_value : scalar_arg;
          (** Integer-valued index spliced into the row-major offset at [dyn_axis]. A nested scalar:
              all recursive scalar traversals must descend into it. gh-343: produced only by
              [rewrite_one_hot_reductions], never escapes [Low_level] / backend codegen. *)
    }
  | Get_merge_buffer of Tn.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_arg * scalar_arg * scalar_arg
  | Binop of Ops.binop * scalar_arg * scalar_arg
  | Unop of Ops.unop * scalar_arg
  | Constant of float
  | Constant_bits of int64  (** Direct bit representation, primarily for uint4x32 *)
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

and scalar_arg = scalar_t * Ops.prec [@@deriving sexp_of, equal, compare]

and carried = { prev : scope_id; next : scope_id; init : scalar_t }
[@@deriving sexp_of, equal, compare]
(** One loop-carried scalar of a {!t.Scan_loop}: read as [prev], written as [next], rotated
    [prev := next] after each iteration, [prev := init] before the first. Both ids name the same
    virtual node, whose precision is the state's precision. *)

(* gh-563: the one canonical rendering of lowered code, shared by both digest consumers —
   [analysis_digest] (the analysis cache, consulted inside [optimize]) and
   [Schedule_cache.canonicalize] (schedule replay across sessions). The walk is the same for both:
   index / scalar / statement emission, loop-binder tokens, local-scope alpha renaming, comment
   skipping, opaque-statement handling. What deliberately differs is the {i identity policy} — the
   analysis cache keys tensor nodes and static symbols by identity (a hit reuses the stored code
   verbatim), the schedule cache alpha-renames everything (a hit replays a schedule onto a
   different-but-isomorphic lowering) — and that is exactly what {!Canonical_render.policy}
   parameterizes. Both digests are correctness-critical, so keep the split honest: a new [t] /
   [scalar_t] construct is rendered here and only here (the matches are exhaustive, so it breaks the
   build); a new digest-relevant {i fact} enters here if it belongs to the code itself, or in one
   policy field / one consumer preamble if it is an identity choice or a consumer-specific
   companion. *)
module Canonical_render = struct
  (** How [Tile_mma] enters the rendering. *)
  type mma_policy =
    | Opaque_mma
        (** Mark the rendering incomplete and emit a placeholder — the consumer's guarantees do not
            extend to the construct. *)
    | Structural_mma  (** Render operands, extents, lane and fallback body. *)

  type policy = {
    emit_tn : Tn.t -> unit;  (** Render a tensor node reference. *)
    emit_free_sym : Indexing.symbol -> unit;
        (** Render a symbol that neither an enclosing loop binder nor {!initial_tokens} bound. *)
    on_bind_loop : Indexing.symbol -> id:int -> shadowed:bool -> unit;
        (** Called when a [For_loop] binder mints the token ["b<id>"]. [shadowed] iff the symbol
            already had a token: a duplicated binder makes symbol references ambiguous. *)
    mark_incomplete : unit -> unit;
        (** Called when an opaque construct makes the rendering an unfaithful summary of the code.
        *)
    mma : mma_policy;
    initial_tokens : (Indexing.symbol * string) list;
        (** Symbols pre-bound to a rendering token before the walk — the static indices, for the
            consumer that renders them positionally. *)
  }

  (** Appends the canonical rendering of [llc] to [buf]. Deterministic and allocation-light: the
      caller digests [buf] (usually after its own preamble and companion sections). *)
  let emit ~(buf : Buffer.t) (p : policy) (llc : t) : unit =
    let add = Buffer.add_string buf in
    (* The in-scope rendering token per symbol (shadow-aware: inner binders overwrite). *)
    let tokens = Hashtbl.create (module Indexing.Symbol) in
    List.iter p.initial_tokens ~f:(fun (key, data) -> Hashtbl.set tokens ~key ~data);
    let base_count = ref 0 in
    let bind_loop s =
      let id = !base_count in
      Int.incr base_count;
      p.on_bind_loop s ~id ~shadowed:(Hashtbl.mem tokens s);
      let tok = "b" ^ Int.to_string id in
      Hashtbl.set tokens ~key:s ~data:tok;
      tok
    in
    let emit_sym s =
      match Hashtbl.find tokens s with Some tok -> add tok | None -> p.emit_free_sym s
    in
    let emit_tn = p.emit_tn in
    (* Local scope ids come from a process-global counter freshly on each lowering (like loop
       symbols), so render their first-occurrence alpha index, not the raw id — otherwise sibling
       lowerings of one local-heavy routine would never agree on a digest. *)
    let scope_alpha = Hashtbl.create (module Scope_id) in
    let emit_scope (id : scope_id) =
      emit_tn id.tn;
      let a = Hashtbl.find_or_add scope_alpha id ~default:(fun () -> Hashtbl.length scope_alpha) in
      add ("." ^ Int.to_string a)
    in
    let emit_idx = function
      | Indexing.Fixed_idx i -> add ("#" ^ Int.to_string i)
      | Indexing.Iterator s -> emit_sym s
      | Indexing.Affine { symbols; offset } ->
          add "(";
          List.iter symbols ~f:(fun (c, s) ->
              add (Int.to_string c ^ "*");
              emit_sym s;
              add "+");
          add (Int.to_string offset ^ ")")
      | Indexing.Sub_axis -> add "_"
      | Indexing.Concat syms ->
          add "cat(";
          List.iter syms ~f:(fun s ->
              emit_sym s;
              add ",");
          add ")"
    in
    let emit_idcs idcs =
      add "[";
      Array.iter idcs ~f:(fun idx ->
          emit_idx idx;
          add ",");
      add "]"
    in
    let rec emit_scalar (sc : scalar_t) =
      match sc with
      | Local_scope { id; body; orig_indices; mint } ->
          (* gh-ocannl-687: the mint is part of the program's identity -- it decides which loops the
             action menu may target -- but only the newer, schedule-minted form takes a marker, so
             digests of the code the optimizer produces are unchanged. *)
          add (match mint with Inlined_computation -> "scope(" | Schedule_minted -> "sscope(");
          emit_scope id;
          add ")";
          emit_idcs orig_indices;
          add "{";
          emit body;
          add "}"
      | Get_local id ->
          add "getl(";
          emit_scope id;
          add ")"
      | Get (tn, idcs) ->
          add "get ";
          emit_tn tn;
          emit_idcs idcs
      | Get_dynamic { tn; idcs; dyn_axis; dyn_value } ->
          add "getd ";
          emit_tn tn;
          emit_idcs idcs;
          add ("/" ^ Int.to_string dyn_axis ^ ":");
          emit_arg dyn_value
      | Get_merge_buffer (tn, idcs) ->
          add "getm ";
          emit_tn tn;
          emit_idcs idcs
      | Ternop (op, a, b, c) ->
          add (Sexp.to_string (Ops.sexp_of_ternop op));
          add "(";
          emit_arg a;
          add ",";
          emit_arg b;
          add ",";
          emit_arg c;
          add ")"
      | Binop (op, a, b) ->
          add (Sexp.to_string (Ops.sexp_of_binop op));
          add "(";
          emit_arg a;
          add ",";
          emit_arg b;
          add ")"
      | Unop (op, a) ->
          add (Sexp.to_string (Ops.sexp_of_unop op));
          add "(";
          emit_arg a;
          add ")"
      | Constant f -> add (Float.to_string f)
      | Constant_bits b -> add ("bits:" ^ Int64.to_string b)
      | Embed_index idx ->
          add "ix:";
          emit_idx idx
    and emit_arg ((sc, prec) : scalar_arg) =
      emit_scalar sc;
      add ("@" ^ Sexp.to_string (Ops.sexp_of_prec prec))
    and emit (llc : t) =
      match llc with
      | Noop -> add "nop;"
      (* Comment text is presentational (routine names, debug names): identical code under different
         names shares one digest. *)
      | Comment _ -> add "c;"
      | Staged_compilation _ ->
          p.mark_incomplete ();
          add "staged;"
      | Seq (a, b) ->
          emit a;
          emit b
      | For_loop { index; from_; to_; body; axis } ->
          let tok = bind_loop index in
          add
            (Printf.sprintf "for %s=%d..%d@%s{" tok from_ to_
               (Sexp.to_string (sexp_of_axis_type axis)));
          emit body;
          add "}"
      | Scan_loop { index; from_; to_; direction; carried; body } ->
          (* gh-ocannl-696: the inits render before the binder, in the order they evaluate; a
             carried pair renders as [prev<-next], so swapping the two ids changes the digest. *)
          List.iter carried ~f:(fun { prev; next; init } ->
              add "carry ";
              emit_scope prev;
              add "<-";
              emit_scope next;
              add ":=";
              emit_scalar init;
              add ";");
          let tok = bind_loop index in
          add
            (Printf.sprintf "scan %s=%d..%d@%s{" tok from_ to_
               (Sexp.to_string (sexp_of_scan_direction direction)));
          emit body;
          add "}"
      | Zero_out tn ->
          add "zero ";
          emit_tn tn;
          add ";"
      | Set { tn; idcs; llsc; debug = _ } ->
          add "set ";
          emit_tn tn;
          emit_idcs idcs;
          add ":=";
          emit_scalar llsc;
          add ";"
      | Set_dynamic { tn; idcs; dyn_axis; dyn_value; llsc; debug = _ } ->
          add "setdyn ";
          emit_tn tn;
          emit_idcs idcs;
          add ("@" ^ Int.to_string dyn_axis ^ "=");
          emit_arg dyn_value;
          add ":=";
          emit_scalar llsc;
          add ";"
      | Set_from_vec { tn; idcs; length; vec_unop; arg; debug = _ } ->
          add ("setv" ^ Int.to_string length ^ " ");
          emit_tn tn;
          emit_idcs idcs;
          add (":=" ^ Sexp.to_string (Ops.sexp_of_vec_unop vec_unop));
          add "(";
          emit_arg arg;
          add ");"
      | Set_local (id, sc) ->
          add "setl ";
          emit_scope id;
          add ":=";
          emit_scalar sc;
          add ";"
      | Declare_local { id; needs_init } ->
          add "decl ";
          emit_scope id;
          add (if needs_init then "0;" else ";")
      | Workgroup_barrier -> add "bar;"
      | If { cond; body } ->
          add "if(";
          emit_arg cond;
          add "){";
          emit body;
          add "}"
      | Tile_mma { d; a; b; ta; tb; m; n; k; ldd; lda; ldb; lane; tile; fallback } -> (
          match p.mma with
          | Opaque_mma ->
              p.mark_incomplete ();
              add "mma;"
          | Structural_mma ->
              let operand (tn, idcs) =
                emit_tn tn;
                emit_idcs idcs
              in
              add "mma ";
              operand d;
              operand a;
              operand b;
              add (Printf.sprintf " %b %b %d %d %d %d %d %d " ta tb m n k ldd lda ldb);
              (* A requested geometry is part of the statement's identity; the renderer's own choice
                 is not, so [None] renders as before gh-ocannl-619. *)
              Option.iter tile ~f:(fun t ->
                  add (Printf.sprintf "tile{%s} " (Register_tile.to_string t)));
              emit_sym lane;
              add "{";
              emit fallback;
              add "}")
    in
    emit llc
end

(** Extract the precision from a scalar value by examining its origin tensor node *)
let scalar_precision = function
  | Get (tn, _) -> Lazy.force tn.Tn.storage_prec
  | Get_dynamic { tn; _ } -> Lazy.force tn.Tn.storage_prec
  | Get_merge_buffer (tn, _) -> Lazy.force tn.Tn.storage_prec
  | Get_local { tn; _ } -> Lazy.force tn.Tn.storage_prec
  | Local_scope { id; _ } -> Lazy.force id.tn.Tn.storage_prec
  | Constant _ ->
      (* Single is the most widely supported precision, so we use it as the default. *)
      Ops.single
  | Constant_bits _ -> Ops.int64
  | Embed_index _ -> Ops.index_prec ()
  | Ternop (_, (_, prec), _, _) -> prec
  | Binop (_, (_, prec), _) -> prec
  | Unop (_, (_, prec)) -> prec

(** Helper to construct binary/ternary/unary ops with proper precision *)
let mk_binop op arg1 arg2 = Binop (op, (arg1, scalar_precision arg1), (arg2, scalar_precision arg2))

let mk_ternop op arg1 arg2 arg3 =
  Ternop
    (op, (arg1, scalar_precision arg1), (arg2, scalar_precision arg2), (arg3, scalar_precision arg3))

let mk_unop op arg = Unop (op, (arg, scalar_precision arg))

let apply_op op args =
  match (op, args) with
  | Ops.Binop Ops.Arg1, [| rhs1; _ |] -> rhs1
  | Binop Arg2, [| _; rhs2 |] -> rhs2
  | Unop Identity, [| rhs |] -> rhs
  | Ternop op, [| rhs1; rhs2; rhs3 |] -> mk_ternop op rhs1 rhs2 rhs3
  | Binop op, [| rhs1; rhs2 |] -> mk_binop op rhs1 rhs2
  | Unop op, [| rhs |] -> mk_unop op rhs
  | _ -> invalid_arg "Low_level.op: invalid number of arguments"

let rec flat_lines ts =
  List.concat_map ts ~f:(function Seq (t1, t2) -> flat_lines [ t1; t2 ] | t -> [ t ])

let rec unflat_lines = function
  | [] -> Noop
  | [ llc ] -> llc
  | Noop :: tl -> unflat_lines tl
  | llc :: tl -> Seq (llc, unflat_lines tl)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_inline_reduction : int;
  mutable max_inline_fanin : int;
  mutable inline_scalar_constexprs : bool;
  mutable inline_simple_computations : bool;
  mutable inline_complex_computations : bool;
}

let virtualize_settings =
  let max_visits =
    Int.of_string @@ Utils.get_global_arg ~arg_name:"virtualize_max_visits" ~default:"1"
  in
  let max_inline_reduction =
    Int.of_string @@ Utils.get_global_arg ~arg_name:"virtualize_max_inline_reduction" ~default:"16"
  in
  let max_inline_fanin =
    Int.of_string @@ Utils.get_global_arg ~arg_name:"virtualize_max_inline_fanin" ~default:"8"
  in
  let enable_device_only = Utils.get_global_flag ~default:true ~arg_name:"enable_device_only" in
  let inline_scalar_constexprs =
    Utils.get_global_flag ~default:true ~arg_name:"inline_scalar_constexprs"
  in
  let inline_simple_computations =
    Utils.get_global_flag ~default:true ~arg_name:"inline_simple_computations"
  in
  let inline_complex_computations =
    Utils.get_global_flag ~default:true ~arg_name:"inline_complex_computations"
  in
  {
    enable_device_only;
    max_visits;
    max_inline_reduction;
    max_inline_fanin;
    inline_scalar_constexprs;
    inline_simple_computations;
    inline_complex_computations;
  }

type traced_array = {
  tn : Tn.t;
  mutable has_assignment : bool;
      (** The code contains a [Set] or [Set_from_vec] of the node ([Zero_out] is tracked separately
          as [zeroed_out]). Structural replacement (gh-554) for the retired concrete tracer's
          per-cell assignment table; per-cell facts are answered by the affine queries instead. *)
  mutable zero_initialized_by_code : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
  mutable is_accessing : bool;
  mutable is_complex : bool;
  mutable prefers_virtual_one_hot : bool;
  mutable has_non_one_hot_setter : bool;
  mutable is_range_producer : bool;
  mutable inline_reduction_extent : int;
      (** The largest product of trip counts of loops that enclose one of the node's setters without
          appearing in its indices (i.e. reduction loops). Inlining the computation replays these
          loops at every read site, so large extents make virtualization pathological; see
          [virtualize_max_inline_reduction]. *)
  mutable read_by_other : bool;
      (** True when some statement other than the node's own setters reads the node. Unlike
          [accesses], same-cell reads count (they are exempt from the visit cap), while a setter's
          own read-modify-write does not. A node never read in the routine has no inlining cost, so
          the recompute-cost guard must not materialize it (it may instead be dropped as a committed
          virtual computation, or inlined by a later routine in the lineage). *)
  mutable setter_reads : Set.M(Tnode).t list;
      (** Per setter statement ([Set]/[Set_from_vec]), the tensor nodes its right-hand side reads —
          including reads inside [Local_scope] bodies in the right-hand side (their loads execute
          per evaluation of the setter), and excluding the node's own read-modify-write self-reads
          (when inlined they become the local accumulator, not a load). Decision-independent
          analysis fact behind the transitive inline-fanin guard (gh-573): a read of a cell executes
          one setter's computation, so the guard takes the per-setter maximum, not the union — a
          Block/concat node written by one range-guarded setter per component costs one component
          per read. The per-setter maximum is chosen before downstream unions, so a consumer
          overlapping one alternative but not another can be undercounted — accepted: an exact
          treatment is combinatorial across multi-setter dependencies, and both error directions of
          this heuristic prior are benign (a miss reproduces the pre-guard placement, which stays a
          [`Materialize] flip candidate with a fanin-charged cost). *)
  mutable inline_fanin : int;
      (** The transitive inline fan-in the guard computed for this node under the current
          placements: the number of distinct materialized nodes the node's fully-inlined computation
          would load (at least 1). Decision-dependent (written by [decide_placements] on the
          candidate's private store copy); multiplies into [fc_recompute_cost] so the search and the
          memory-budget planner see the true cost of re-inlining a node the fanin cap materialized.
      *)
}
[@@deriving sexp_of]

type optimize_ctx = {
  computations : (Tnode.t, (Indexing.axis_index array option * t) list) Base.Hashtbl.t;
  placements : Tnode.Placements.t;
      (** Per-compilation-lineage memory-mode resolution
          (docs/proposals/context-scoped-memory-modes.md): decisions land here, not on the tnode.
          Copied per backend [compile] (see {!copy_optimize_ctx}) so sibling compiles are hermetic.
      *)
  alias_candidates : Hash_set.M(Tnode).t;
      (** gh-ocannl-489 liveness-based buffer aliasing: nodes the memory planner may place at
          overlapping byte ranges within the routine's working pool (decided per compile, before
          codegen). Codegen must not emit the [restrict] qualifier for these parameters — whether a
          candidate pair actually shares bytes is settled only at link time ([allocate_delta]), and
          an aliased [restrict] pair is a miscompile. Another per-compilation-lineage decision kind,
          same species as {!field-placements}. *)
  inline_preferences : Hash_set.M(Tnode).t;
      (** gh-555: the [Inline] half of the per-lineage inlining decision vector. A node recorded
          here is exempt from the heuristic virtualization caps ([virtualize_max_visits],
          [virtualize_max_inline_reduction], [virtualize_max_inline_fanin]) in {!decide_placements}
          — the caps are priors of the default policy, not legality. Legality is unaffected:
          [check_and_store_virtual] / [inline_computation] can still reject the node
          ([Never_virtual] with their provenances). The [Materialize] half of the vector is a
          pre-seeded [On_device] decision in {!field-placements} (see
          [Context.decide_materialized]). *)
}
[@@deriving sexp_of]

let empty_optimize_ctx () =
  {
    computations = Hashtbl.create (module Tnode);
    placements = Tnode.Placements.create ();
    alias_candidates = Hash_set.create (module Tnode);
    inline_preferences = Hash_set.create (module Tnode);
  }

(** A shallow-copy fork of the lineage state: the copy sees everything decided so far, and neither
    the original nor sibling copies observe its later mutations. Backend [compile] forks the
    incoming context's [optimize_ctx] through this, which is what makes sibling candidate compiles
    from one frontier hermetic. *)
let copy_optimize_ctx { computations; placements; alias_candidates; inline_preferences } =
  {
    computations = Hashtbl.copy computations;
    placements = Tnode.Placements.copy placements;
    alias_candidates = Hash_set.copy alias_candidates;
    inline_preferences = Hash_set.copy inline_preferences;
  }

(** Records an [On_device] decision in the lineage state for each node of [tns] this lineage has not
    already resolved otherwise — the "materialize this node" move of the placement lattice, and the
    [Materialize] half of the inlining decision vector.

    Nodes already resolved to [Virtual] / [Local] / [Effectively_constant] keep their resolution:
    decisions are final within a lineage, since already-compiled routines depend on them.

    {!Context.decide_materialized} is the context-level form (it forks the lineage first), and is
    what ordinary [Assignments] callers want. This raw form serves the two paths that hold an
    [optimize_ctx] directly: the analyze-only entry points, and hand-built {!optimize} calls in
    tests — for which no context-level form can work, since the [?prelowered] seam replaces the
    context's lineage state with the optimized record's own [optimize_ctx]. *)
let decide_materialized ?(provenance = 31) (optim_ctx : optimize_ctx) tns =
  List.iter tns ~f:(fun tn ->
      match Tnode.Placements.get optim_ctx.placements tn with
      | None | Some ((Tnode.Never_virtual | Tnode.On_device), _) ->
          Tnode.Placements.update optim_ctx.placements tn Tnode.On_device provenance
      | Some ((Tnode.Virtual | Tnode.Local | Tnode.Effectively_constant), _) -> ())

type traced_store = (Tn.t, traced_array) Base.Hashtbl.t [@@deriving sexp_of]

(** Granularity of the XOR remap applied to a swizzled node's minor axis (gh-ocannl-481 item 3, D1).
    Both flavors are per-row bijections of the minor axis, so the IR-level semantics are identical;
    they differ in the unit the XOR permutes and therefore in which access pattern they de-conflict.
*)
type swizzle_kind =
  | Swizzle_elem
      (** Element-granularity XOR: [P*C + col] renders as [P*C + (col lxor (P land (C-1)))]. Spreads
          same-column scalar reads of consecutive rows across banks; the flavor the scalar and
          register-blocktiled staged kernels want. *)
  | Swizzle_b128
      (** 16-byte-unit XOR: the column's 16-byte-unit index is XORed with the low bits of the row
          prefix, leaving the offset within the unit alone. This is the CUTLASS-style layout
          [ldmatrix] wants — its 8 per-phase row addresses are 16-byte-aligned, so only a remap that
          keeps 16-byte units intact can both de-conflict them and stay loadable. Requires the row's
          byte length to be a multiple of 16 and a power of two in 16-byte units. *)
[@@deriving sexp, compare, equal]

type flip_candidate = {
  fc_tn : Tnode.t;
  fc_flip : [ `Materialize | `Inline ];
  fc_recompute_cost : int;
}
[@@deriving sexp_of]
(** gh-555: one searchable inlining decision dimension of a compile — a node whose placement the
    default policy decided, together with the flip a search can try and the recompute-cost bound of
    the virtual placement (reduction extent × per-cell read multiplicity × transitive inline fan-in
    — the cost the flip trades against memory traffic; without the fan-in factor a node the fanin
    cap materialized would rank among the cheapest to re-inline, and the memory-budget planner would
    prefer undoing exactly the guard's decision). [`Materialize] flips a node the policy left
    virtual (via [Context.decide_materialized]); [`Inline] flips a node materialized by the
    heuristic caps (provenance 1, 39 or 41 — never by legality or observability, which are not
    decisions), via [Context.decide_inline]. An [`Inline] flip's legality is settled only when the
    virtualizer replays ([check_and_store_virtual]): a rejected flip reproduces the materialized
    placement. *)

type pipelined_tile = { pt_depth : int; pt_rotor : Indexing.symbol } [@@deriving sexp_of]
(** gh-487: a software-pipelined (double-buffered) staged tile — codegen allocates [pt_depth]
    rotating copies of the tile and renders every access with a buffer-selection term rotated by the
    [pt_rotor] loop counter: reads select copy [rotor mod depth], writes copy
    [(rotor + 1) mod depth] (the schedule emits the loads one iteration ahead), and writes outside
    the rotor loop (the prologue load) select copy 0. The IR keeps the tile's single-copy dims and
    indices — the rotation is a physical-layout choice like {!type-swizzle_kind}, invisible to
    IR-level semantics — so the pipelined rendering is bitwise identical to the unpipelined one. *)

type optimized = {
  traced_store : traced_store;
  optimize_ctx : optimize_ctx;
  llc : t;
  merge_node : Tnode.t option;
  workgroup_shared : Set.M(Tnode).t;
      (** [Local]-memory-mode nodes to be placed in workgroup-shared memory ([__shared__] /
          [threadgroup]) instead of kernel-local arrays. Populated by schedule transforms; empty for
          unscheduled code. See docs/proposals/axis-types-for-loops.md. *)
  simdgroup_fragments : Set.M(Tnode).t;
      (** Per-simdgroup accumulator tiles synthesized by [Schedule.Tensorize] when it contracts an
          enclosing serial reduction. Metal renders their marked lifetime as persistent
          [simdgroup_matrix] fragments; scalar fallback backends retain the local-array meaning. *)
  swizzled : swizzle_kind Map.M(Tnode).t;
      (** Nodes stored in an XOR-swizzled layout (docs/proposals/tensorize-mma.md, "Swizzled
          staging"), keyed by the remap's granularity ({!type-swizzle_kind}): codegen remaps every
          element access [flat = P*C + col] (with [C] the minor dim, [P] the linearized prefix) to a
          per-row permutation of [col] — a bijection on the buffer, so the IR-level semantics are
          unchanged; only the physical layout differs, spreading same-column accesses across
          shared-memory banks. Populated by [Schedule.Stage ~swizzle]; requires at least 2 axes and
          a power-of-two minor dim (plus, for [Swizzle_b128], a 16-byte-multiple row). Renderings
          that assume a row-major layout (vectorized/contiguous multi-element accesses, the
          register-tiled path) must decline swizzled nodes; the tile-MMA intrinsic arms decline
          [Swizzle_elem] and may consume [Swizzle_b128] through [ldmatrix]-style loads. *)
  pipelined : pipelined_tile Map.M(Tnode).t;
      (** gh-487: workgroup-shared staged tiles rendered as [pt_depth] rotating buffer copies (see
          {!type-pipelined_tile}). Populated by [Schedule.Stage ~pipeline_depth] with depth > 1;
          codegen multiplies the tile's allocation by the depth and rotates a buffer-selection
          offset with the [pt_rotor] loop counter. Renderings that assume single-copy storage
          (vectorized/contiguous multi-element accesses) must decline pipelined nodes. *)
  zero_fringe : Set.M(Tnode).t;
      (** Schedule-minted staged tiles whose whole index space is safe to read: slots outside the
          staged source region (edge tiles of a non-dividing or padded staging, gh-ocannl-485) hold
          0 — the add-reduce accumulation identity — written by the load nest's [Where]-form edge
          guards or by the host-side constant packing. [Schedule.Tensorize] consults this to
          discharge pad guards on the intrinsic path. *)
  flip_candidates : flip_candidate list;
      (** gh-555: the searchable inlining decision dimensions of this compile, most expensive first,
          as decided at the whole-routine specialization (schedule-transform copies inherit the
          whole-routine list). Excluded: nodes never assigned or never read (no decision to make),
          scalar constexprs and pure one-hot selector producers (must stay virtual for their
          rewrites), and nodes placed by legality, intent or observability rather than the heuristic
          policy. *)
  spliced_rbw : Set.M(Tnode).t;
      (** gh-610 review round 6: the nodes whose [read_before_write] was set by the FINAL-code
          reconciliation (a spliced read preceding, or not definitely covered by, the routine's own
          writes) — as opposed to the raw analysis' uncovered-read classification, which also flags
          every pure input. [Backends]' prior-context demand keys on this set. *)
}
[@@deriving sexp_of]

type footprint = {
  fp_total : int;
  fp_working : int;
  fp_constants : int;
  fp_dedicated : int;
  fp_planned : int;
  fp_nodes : int;
}
[@@deriving sexp_of, equal]

let get_node store tn =
  Hashtbl.find_or_add store tn ~default:(fun () ->
      {
        tn;
        has_assignment = false;
        zero_initialized_by_code = false;
        zeroed_out = false;
        read_before_write = false;
        read_only = false;
        is_scalar_constexpr = false;
        is_accessing = false;
        is_complex = false;
        prefers_virtual_one_hot = false;
        has_non_one_hot_setter = false;
        is_range_producer = false;
        inline_reduction_extent = 1;
        read_by_other = false;
        setter_reads = [];
        inline_fanin = 1;
      })

let is_constexpr_comp traced_store llsc =
  let rec loop llsc =
    match llsc with
    | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Get (tn, _) ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Get_dynamic _ -> false (* a runtime gather is never a scalar constexpr *)
    | Get_merge_buffer (tn, _) ->
        let traced = get_node traced_store tn in
        traced.is_scalar_constexpr
    | Ternop (_, (v1, _), (v2, _), (v3, _)) -> loop v1 && loop v2 && loop v3
    | Binop (_, (v1, _), (v2, _)) -> loop v1 && loop v2
    | Unop (_, (v, _)) -> loop v
    | Constant _ | Constant_bits _ -> true
    | Embed_index _ -> false
  in
  loop llsc

let is_accessing_comp traced_store llsc =
  let rec loop llsc =
    match llsc with
    | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
        let traced = get_node traced_store tn in
        traced.is_accessing
    | Get (tn, _) ->
        let traced = get_node traced_store tn in
        not traced.is_scalar_constexpr
    | Get_dynamic _ -> true (* accesses the table at a runtime index *)
    | Get_merge_buffer (tn, _) ->
        let traced = get_node traced_store tn in
        traced.is_accessing <- true;
        true
    | Ternop (_, (v1, _), (v2, _), (v3, _)) -> loop v1 || loop v2 || loop v3
    | Binop (_, (v1, _), (v2, _)) -> loop v1 || loop v2
    | Unop (_, (v, _)) -> loop v
    | Constant _ | Constant_bits _ -> false
    | Embed_index _ -> false
  in
  loop llsc

let is_complex_comp traced_store llsc =
  let accessing = is_accessing_comp traced_store in
  match llsc with
  | Get_local { tn; _ } | Local_scope { id = { tn; _ }; _ } ->
      let traced = get_node traced_store tn in
      traced.is_complex
  | Get _ -> false
  | Get_dynamic { dyn_value = v, _; _ } -> accessing v
  | Get_merge_buffer _ -> false
  | Ternop (_, (v1, _), (v2, _), (v3, _)) -> accessing v1 || accessing v2 || accessing v3
  | Binop (_, (v1, _), (v2, _)) -> accessing v1 || accessing v2
  | Unop (_, (v, _)) -> accessing v
  | Constant _ | Constant_bits _ -> false
  | Embed_index _ -> false

let is_scalar_dims tn = Array.for_all ~f:(( = ) 1) @@ Lazy.force tn.Tn.dims

(* Records [tn] as a candidate owner of each loop symbol appearing in its assignment indices.
   Multiple tensors may share a symbol (e.g. Block/concat lowering): all are recorded, in first-seen
   (trace) order and deduplicated, so that [virtual_llc] can store one computation per candidate.
   See #134. *)
let track_symbol reverse_node_map tn idcs =
  let add s =
    let existing = Hashtbl.find reverse_node_map s |> Option.value ~default:[] in
    if not (List.mem existing tn ~equal:Tn.equal) then
      Hashtbl.set reverse_node_map ~key:s ~data:(existing @ [ tn ])
  in
  Array.iter idcs ~f:(function
    | Indexing.Fixed_idx _ -> ()
    | Indexing.Sub_axis -> ()
    | Indexing.Iterator s -> add s
    | Indexing.Affine { symbols; _ } -> List.iter symbols ~f:(fun (_, s) -> add s)
    | Indexing.Concat syms -> List.iter syms ~f:add)

(* gh-343 / task-73617488: helpers for the one-hot reduction rewrite and the virtualizer exemption.
   Placed before [trace_node_facts] so [is_one_hot_selector_assignment] is available during
   tracing. *)

let axis_index_mentions_symbol = Indexing.axis_index_mentions_symbol

let rec scalar_mentions_symbol (s : Indexing.symbol) (llsc : scalar_t) : bool =
  match llsc with
  | Embed_index idx -> axis_index_mentions_symbol s idx
  | Get (_, idcs) | Get_merge_buffer (_, idcs) ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s)
  | Get_dynamic { idcs; dyn_value = v, _; _ } ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s) || scalar_mentions_symbol s v
  | Local_scope { orig_indices; body; _ } ->
      Array.exists orig_indices ~f:(axis_index_mentions_symbol s) || proc_mentions_symbol s body
  | Ternop (_, (v1, _), (v2, _), (v3, _)) ->
      scalar_mentions_symbol s v1 || scalar_mentions_symbol s v2 || scalar_mentions_symbol s v3
  | Binop (_, (v1, _), (v2, _)) -> scalar_mentions_symbol s v1 || scalar_mentions_symbol s v2
  | Unop (_, (v, _)) -> scalar_mentions_symbol s v
  | Get_local _ | Constant _ | Constant_bits _ -> false

and proc_mentions_symbol (s : Indexing.symbol) (llc : t) : bool =
  match llc with
  | Set { idcs; llsc; _ } ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s) || scalar_mentions_symbol s llsc
  | Set_dynamic { idcs; dyn_value = v, _; llsc; _ } ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s)
      || scalar_mentions_symbol s v || scalar_mentions_symbol s llsc
  | Set_from_vec { idcs; arg = v, _; _ } ->
      Array.exists idcs ~f:(axis_index_mentions_symbol s) || scalar_mentions_symbol s v
  | Set_local (_, llsc) -> scalar_mentions_symbol s llsc
  | Seq (a, b) -> proc_mentions_symbol s a || proc_mentions_symbol s b
  | For_loop { body; _ } -> proc_mentions_symbol s body
  | Scan_loop { carried; body; _ } ->
      List.exists carried ~f:(fun c -> scalar_mentions_symbol s c.init)
      || proc_mentions_symbol s body
  | If { cond = c, _; body } -> scalar_mentions_symbol s c || proc_mentions_symbol s body
  | Tile_mma { d = _, d_idcs; a = _, a_idcs; b = _, b_idcs; lane; fallback; _ } ->
      Indexing.equal_symbol s lane
      || Array.exists d_idcs ~f:(axis_index_mentions_symbol s)
      || Array.exists a_idcs ~f:(axis_index_mentions_symbol s)
      || Array.exists b_idcs ~f:(axis_index_mentions_symbol s)
      || proc_mentions_symbol s fallback
  | Zero_out _ | Declare_local _ | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier ->
      false

(* Count occurrences of [Iterator s] in [idcs], and report whether every occurrence is a plain
   [Iterator s] (no [Affine]/[Concat] use). Returns [(count, axis_of_last_plain_occurrence)]. *)
let count_plain_iterator (s : Indexing.symbol) (idcs : Indexing.axis_index array) :
    int * int option * bool =
  let count = ref 0 and axis = ref None and only_plain = ref true in
  Array.iteri idcs ~f:(fun i idx ->
      if axis_index_mentions_symbol s idx then (
        Int.incr count;
        match idx with Indexing.Iterator _ -> axis := Some i | _ -> only_plain := false));
  (!count, !axis, !only_plain)

(* task-73617488: true when [expr] is an embedded loop iterator [k] in either plain or unit-affine
   form. Shared by [match_one_hot_contribution] and [is_one_hot_selector_assignment]. *)
let is_embedded_range_iterator (k : Indexing.symbol) (expr : scalar_t) : bool =
  match expr with
  | Embed_index (Indexing.Iterator k') -> Indexing.equal_symbol k k'
  | Embed_index (Indexing.Affine { symbols = [ (1, k') ]; offset = 0 }) ->
      Indexing.equal_symbol k k'
  | _ -> false

(* task-73617488: true when a [Set] assignment is a one-hot selector producer. Recognises two forms:
   - Direct: the [llsc] is [Cmpeq(Embed_index (Iterator k), expr_free_of_k)] (or unit-affine). -
   Indirect: one side is [Get(range_tn, [|Iterator k|])] where [range_tn.is_range_producer] is true,
   meaning it was set from a bare [Embed_index] (the [Range_over_offsets] lowering). [virtual_llc]
   will inline such a tensor to [Embed_index k] so the gather rewrite fires. [traced_store] is
   consulted to classify the indirect case. [Set_from_vec] is never a one-hot selector (the caller
   sets [has_non_one_hot_setter] instead). *)
let is_one_hot_selector_assignment traced_store ~(idcs : Indexing.axis_index array)
    (llsc : scalar_t) : bool =
  let is_range_side k expr =
    is_embedded_range_iterator k expr
    ||
    match expr with
    | Get (rtn, inner_idcs) when Array.length inner_idcs = 1 -> (
        match inner_idcs.(0) with
        | Indexing.Iterator k' when Indexing.equal_symbol k k' ->
            (* Only accept a tensor that was itself assigned from Embed_index (a real range
               producer). Read-only inputs and arbitrary computed tensors are NOT range producers
               even if they happen to be non-complex and non-accessing. *)
            let rt = get_node traced_store rtn in
            rt.is_range_producer
        | _ -> false)
    | _ -> false
  in
  Array.exists idcs ~f:(function
    | Indexing.Iterator k -> (
        match llsc with
        | Binop (Ops.Cmpeq, (a, _), (b, _)) ->
            (is_range_side k a && not (scalar_mentions_symbol k b))
            || (is_range_side k b && not (scalar_mentions_symbol k a))
        | _ -> false)
    | _ -> false)

(* The affine form of an access position component as the retired concrete tracer evaluated it:
   statics contribute 0 (they are fixed per routine invocation, so equal statics cancel and distinct
   statics are conflated — mirroring the tracer's statics-pinned-to-0 environment). [None] =
   uninterpretable component. Shared by the read-modify-write exemption of [trace_node_facts],
   [read_multiplicity_query] and [reads_covered_query]. *)
let norm_position_comp ~statics_set (idx : Indexing.axis_index) =
  let drop_statics symbols =
    List.filter (Indexing.coalesce_affine_terms symbols) ~f:(fun (_, s) ->
        not (Set.mem statics_set s))
    |> List.sort ~compare:[%compare: int * Indexing.symbol]
  in
  match idx with
  | Indexing.Fixed_idx c -> Some ([], c)
  | Indexing.Iterator s -> Some (drop_statics [ (1, s) ], 0)
  | Indexing.Affine { symbols; offset } -> Some (drop_statics symbols, offset)
  | Indexing.Sub_axis | Indexing.Concat _ -> None

(* Whether two access positions denote the same cell at every iteration: per-axis affine-form
   equality with statics zeroed. The [inline_complex_computations] exemption ("a read at the
   enclosing statement's write position is a read-modify-write self-read, not a visit") compares
   positions with this — whichever nodes the two accesses touch, exactly as the retired tracer
   compared concrete index vectors. *)
let same_position ~statics_set (m : Indexing.axis_index array) (m' : Indexing.axis_index array) =
  Array.length m = Array.length m'
  && Array.for_all2_exn m m' ~f:(fun a b ->
      match (norm_position_comp ~statics_set a, norm_position_comp ~statics_set b) with
      | Some fa, Some fb -> [%equal: (int * Indexing.symbol) list * int] fa fb
      | _ -> false)

(* Structural facts pass (gh-554): a single non-enumerating program-order traversal computing the
   per-node facts of {!traced_array} that are not affine questions — setter/reader structure,
   scalar-constexpr and one-hot classifications, reduction extents, loop-symbol ownership and the
   merge node. Each statement is visited exactly once (the retired concrete-index tracer's
   [first_visit] pass); the per-cell visit counts and read-before-write facts the tracer sampled by
   enumerating loop iterations are answered exactly by the affine queries instead
   ({!read_multiplicity_query}, {!reads_covered_query}, consumed by {!decide_placements}). *)
let trace_node_facts traced_store ~merge_node_ref reverse_node_map ~static_indices llc =
  let statics_set =
    Set.of_list (module Indexing.Symbol)
    @@ List.map static_indices ~f:(fun s -> s.Indexing.static_symbol)
  in
  (* Nodes with a non-exempt read seen so far, in program order — the first-touch state behind
     [is_scalar_constexpr] and [zero_initialized_by_code] (the tracer's accesses-empty test). *)
  let read_seen = Hash_set.create (module Tnode) in
  (* Concat indices are eliminated during lowering (concatenation lowers to sequenced components);
     one reaching this pass would survive to codegen, which has no rendering for it. Reject loudly,
     as the retired tracer's concrete-index evaluator did. *)
  let check_no_concat idcs =
    Array.iter idcs ~f:(function
      | Indexing.Concat _ ->
          invalid_arg
            "BUG: Concat index encountered during virtualization - should have been eliminated \
             during lowering"
      | _ -> ())
  in
  (* [loop_ranges] maps each enclosing loop's index symbol to its full trip count; at a setter, the
     fiber cardinality of the LHS map over that box — the product over symbols absent from the LHS
     indices — is the recompute cost (per read site) of inlining that setter's computation (gh-494:
     {!Affine.fiber_cardinality}; the exact-vs-lower-bound distinction is not needed here, the
     absent-symbol product is the cost either way). *)
  let reduction_extent loop_ranges idcs =
    match Affine.fiber_cardinality ~domain:(Map.to_alist loop_ranges) idcs with
    | `Exact n | `At_least n -> n
  in
  (* [scope_reads] is the enclosing setter's identity and read sinks when this traversal is inside a
     [Local_scope] body in that setter's right-hand side (gh-573): the body's loads execute per
     evaluation of the setter, so they belong to its [setter_reads], with the setter's own
     read-modify-write self-reads still excluded. The payload carries TWO sinks, (self, statement
     sink, current sink): scope bodies hoist to just before the enclosing statement and execute
     unconditionally (both [Where] arms' bodies really run — see the operand-conditionality notes),
     so they record into the statement sink, while directly conditional arm expressions record into
     per-arm current sinks maxed by the [Ternop] arm below. [None] at top level. *)
  let rec loop_proc ~loop_ranges ~scope_reads llc =
    let loop = loop_proc ~loop_ranges ~scope_reads in
    match llc with
    | Noop -> ()
    | (Seq (c1, c2) : t) ->
        loop c1;
        loop c2
    | For_loop { index; from_; to_; body; axis = _ } ->
        (* A dead loop ([to_ < from_]) never executes its body: record no facts from it, like the
           retired tracer, which never enumerated such loops. *)
        if to_ >= from_ then
          loop_proc ~scope_reads
            ~loop_ranges:(Map.set loop_ranges ~key:index ~data:(to_ - from_ + 1))
            body
    | Scan_loop { index; from_; to_; carried; body; _ } ->
        (* gh-ocannl-696: the inits are recorded once and the body's facts under the scan index
           exactly as under a loop index. (A dead range never reaches here: the validator refuses it
           ahead of the analysis, so no walker needs a dead-scan convention.) *)
        List.iter carried ~f:(fun c -> loop_scalar ~loop_ranges ~lhs:None ~reads:scope_reads c.init);
        loop_proc ~scope_reads
          ~loop_ranges:(Map.set loop_ranges ~key:index ~data:(to_ - from_ + 1))
          body
    | Zero_out tn ->
        let traced : traced_array = get_node traced_store tn in
        if (not traced.has_assignment) && not (Hash_set.mem read_seen tn) then (
          traced.zero_initialized_by_code <- true;
          traced.is_accessing <- false;
          traced.is_complex <- false;
          if is_scalar_dims tn then traced.is_scalar_constexpr <- true);
        traced.zeroed_out <- true
    | Set { tn; idcs; llsc; debug = _ } ->
        check_no_concat idcs;
        let reads = Hash_set.create (module Tnode) in
        loop_scalar ~loop_ranges ~lhs:(Some (tn, idcs)) ~reads:(Some (tn, reads, reads)) llsc;
        let traced : traced_array = get_node traced_store tn in
        traced.setter_reads <-
          Set.of_list (module Tnode) (Hash_set.to_list reads) :: traced.setter_reads;
        traced.inline_reduction_extent <-
          max traced.inline_reduction_extent (reduction_extent loop_ranges idcs);
        if (not traced.has_assignment) && (not (Hash_set.mem read_seen tn)) && is_scalar_dims tn
        then
          (* An assignment re-executed by an enclosing loop is not a single constant assignment (the
             retired tracer cleared the flag on the statement's second enumerated visit). *)
          traced.is_scalar_constexpr <-
            Map.for_all loop_ranges ~f:(fun w -> w <= 1) && is_constexpr_comp traced_store llsc
        else if traced.has_assignment then traced.is_scalar_constexpr <- false;
        traced.is_accessing <- traced.is_accessing || is_accessing_comp traced_store llsc;
        traced.is_complex <- traced.is_complex || is_complex_comp traced_store llsc;
        traced.has_assignment <- true;
        (* task-73617488: track range producers (assigned purely from Embed_index) so the indirect
           arm of is_one_hot_selector_assignment can identify them precisely. *)
        (match llsc with
        | Embed_index _ -> traced.is_range_producer <- true
        | _ -> ());
        (* task-73617488: track whether all setters are one-hot selector assignments so the
           virtualizer can exempt this tensor from the visit-count Never_virtual rule. *)
        if is_one_hot_selector_assignment traced_store ~idcs llsc then
          traced.prefers_virtual_one_hot <- true
        else traced.has_non_one_hot_setter <- true;
        (* Track which tensors use which loop symbol. Multiple tensors may legitimately share a
           symbol (e.g. Block/concat lowering); all of them are recorded as candidate owners in
           first-seen (trace) order. Sharing a symbol no longer marks tensors [is_complex] -- only
           genuine computation complexity does (see #134). *)
        track_symbol reverse_node_map tn idcs
    | Set_from_vec { tn; idcs; length = _; vec_unop = _; arg = arg, _; debug = _ } ->
        check_no_concat idcs;
        let reads = Hash_set.create (module Tnode) in
        loop_scalar ~loop_ranges ~lhs:(Some (tn, idcs)) ~reads:(Some (tn, reads, reads)) arg;
        let traced : traced_array = get_node traced_store tn in
        traced.setter_reads <-
          Set.of_list (module Tnode) (Hash_set.to_list reads) :: traced.setter_reads;
        traced.inline_reduction_extent <-
          max traced.inline_reduction_extent (reduction_extent loop_ranges idcs);
        (* Vector operations cannot be scalar constexpr or one-hot selectors. *)
        traced.is_scalar_constexpr <- false;
        traced.has_non_one_hot_setter <- true;
        traced.is_accessing <- traced.is_accessing || is_accessing_comp traced_store arg;
        traced.is_complex <- traced.is_complex || not (is_constexpr_comp traced_store arg);
        traced.has_assignment <- true;
        track_symbol reverse_node_map tn idcs
    | Set_local (_, llsc) -> loop_scalar ~loop_ranges ~lhs:None ~reads:scope_reads llsc
    | Declare_local _ -> ()
    | Comment _ -> ()
    | Staged_compilation _ -> ()
    | Workgroup_barrier -> ()
    | Set_dynamic _ ->
        (* gh-466: [rewrite_one_hot_reductions] constructs [Set_dynamic] after fact tracing. *)
        invalid_arg "Low_level.trace_node_facts: Set_dynamic reached fact tracing"
    | Tile_mma _ ->
        (* Schedule transforms construct [Tile_mma] after the optimization pipeline ran; it never
           reaches fact tracing. *)
        invalid_arg "Low_level.trace_node_facts: Tile_mma reached the optimization pipeline"
    | If { cond = c, _; body } ->
        loop_scalar ~loop_ranges ~lhs:None ~reads:scope_reads c;
        loop body
  and loop_scalar ~loop_ranges ~lhs ~reads llsc =
    let loop = loop_scalar ~loop_ranges ~lhs ~reads in
    match llsc with
    | Constant _ | Constant_bits _ -> ()
    | Get (ptr, indices) ->
        check_no_concat indices;
        let traced : traced_array = get_node traced_store ptr in
        if not (Option.exists lhs ~f:(fun (tn, _) -> Tn.equal ptr tn)) then
          traced.read_by_other <- true;
        (* The collector carries the setter's identity separately from [lhs], which a [Local_scope]
           body does not inherit: the read-modify-write self-read exclusion must follow the OUTER
           setter through the body, or a self-accumulating scope records a phantom contributor that
           can flip a decision at the cap boundary. *)
        Option.iter reads ~f:(fun (self, _stmt, cur) ->
            if not (Tn.equal ptr self) then Hash_set.add cur ptr);
        (* The read-modify-write exemption: a read at the enclosing statement's write position is
           not a visit ([inline_complex_computations]), whichever node it reads. *)
        let exempt =
          virtualize_settings.inline_complex_computations
          && Option.exists lhs ~f:(fun (_, w_idcs) -> same_position ~statics_set w_idcs indices)
        in
        if not exempt then Hash_set.add read_seen ptr
    | Get_dynamic { dyn_value = v, _; _ } ->
        (* gh-343: [Get_dynamic] is produced after this tracing pass, so this arm is defensive; the
           dynamic index sub-expression is still traversed for completeness. *)
        loop v
    | Local_scope { body; _ } ->
        (* The body hoists to statement level, outside any conditional arm of the enclosing
           expression: its reads record into the statement sink unconditionally. *)
        loop_proc ~loop_ranges
          ~scope_reads:(Option.map reads ~f:(fun (self, stmt, _cur) -> (self, stmt, stmt)))
          body
    | Get_local _ -> ()
    | Get_merge_buffer (source, _) ->
        Option.iter !merge_node_ref ~f:(fun merge_node ->
            if not (Tn.equal merge_node source) then
              raise
              @@ Utils.User_error
                   [%string
                     "Low_evel.optimize_proc: currently only one merge buffer per routine is \
                      allowed, found nodes %{Tn.debug_name source} and %{Tn.debug_name merge_node}"]);
        merge_node_ref := Some source
        (* Not recorded in [reads]: a merge-buffer read loads a materialized copy, never inlining
           the source's computation, and the source's placement here says nothing about that copy.
           The fanin guard undercounts the one load — the safe direction for a heuristic cap. *)
    | Embed_index _ -> ()
    | Binop (Arg1, (llv1, _), _llv2) -> loop llv1
    | Binop (Arg2, _llv1, (llv2, _)) -> loop llv2
    | Ternop (op, (llv1, _), (llv2, _), (llv3, _)) -> (
        loop llv1;
        match (Ops.ternop_conditionality op, reads) with
        | Ops.All_three, _ | Ops.Cond_and_one_arm, None ->
            loop llv2;
            loop llv3
        | Ops.Cond_and_one_arm, Some (self, stmt, cur) ->
            (* Exactly one arm's EXPRESSION evaluates per visit ([Ops.ternop_conditionality]): the
               setter's fan-in charges the wider arm, not the union — two arms each within the cap
               must not jointly trip it. [Local_scope] bodies inside the arms are the exception:
               they hoist to statement level and both execute, so their reads flow to the statement
               sink via the [Local_scope] arm above rather than into the per-arm sinks. The other
               traced facts still record from both arms (their subjects are rendering-level, and
               both arms are rendered). *)
            let r2 = Hash_set.create (module Tnode) and r3 = Hash_set.create (module Tnode) in
            loop_scalar ~loop_ranges ~lhs ~reads:(Some (self, stmt, r2)) llv2;
            loop_scalar ~loop_ranges ~lhs ~reads:(Some (self, stmt, r3)) llv3;
            let wider = if Hash_set.length r3 > Hash_set.length r2 then r3 else r2 in
            Hash_set.iter wider ~f:(Hash_set.add cur))
    | Binop (_, (llv1, _), (llv2, _)) ->
        loop llv1;
        loop llv2
    | Unop (_, (llsc, _)) -> loop llsc
  in
  loop_proc ~loop_ranges:(Map.empty (module Indexing.Symbol)) ~scope_reads:None llc

let%diagn2_sexp check_and_store_virtual (optim_ctx : optimize_ctx) ~guarded ~in_scan ~enclosing
    traced static_indices top_llc =
  let exception Non_virtual of int in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  let at_idcs = ref None in
  let has_setter = ref false in
  let top_tn = traced.tn in
  let check_idcs loop_ranges indices =
    (match !at_idcs with
    | None -> at_idcs := Some indices
    | Some at ->
        if not @@ [%equal: Indexing.axis_index array] at indices then raise @@ Non_virtual 4);
    (* gh-133 Stage A: repeated non-static symbols (diagonal [i;i] / partially-diagonal [i;j;i]) and
       covered single-symbol affine positions are supported. gh-133 Stage B: multi-symbol affine
       positions ([stride*oh+kh], [K*i+k], triangular [(s1, s1+s2)]) are also supported, but only
       when the whole LHS index map is proven injective over the producer loop widths -- otherwise
       dropping the producer loops in [inline_computation] would lose fold contributions. [Concat]
       ([Non_virtual 52]) remains rejected. *)
    let symbol_range s = Map.find loop_ranges s |> Option.value ~default:1 in
    (* Non-static symbols per position; a position with more than one non-static affine symbol is
       only admissible when the whole vector is injective. *)
    let has_multi_affine = ref false in
    let syms =
      Array.fold indices
        ~init:(Set.empty (module Indexing.Symbol))
        ~f:(fun acc -> function
          | Indexing.Fixed_idx _ | Indexing.Sub_axis -> acc
          | Indexing.Iterator s -> if Set.mem static_indices s then acc else Set.add acc s
          | Indexing.Affine { symbols; offset = _ } ->
              let nonstatic =
                List.filter_map symbols ~f:(fun (_, s) ->
                    Option.some_if (not @@ Set.mem static_indices s) s)
              in
              (match nonstatic with [] | [ _ ] -> () | _ -> has_multi_affine := true);
              List.fold nonstatic ~init:acc ~f:Set.add
          | Indexing.Concat _syms ->
              (* Concat indices should be eliminated before virtualization *)
              raise @@ Non_virtual 52)
    in
    let injective = lazy (Indexing.affine_injective ~symbol_range indices) in
    (* A multi-symbol affine position is sound to drop only if the LHS map is injective. *)
    if !has_multi_affine && not (Lazy.force injective) then raise @@ Non_virtual 51;
    (* Non-static symbols appearing in a bare [Iterator] position. [inline_computation]'s pass 1
       binds only such positions from the call args; pass 2 grounds any affine occurrence via
       [subst]. *)
    let iter_syms =
      Set.of_array (module Indexing.Symbol)
      @@ Array.filter_map indices ~f:(function
        | Indexing.Iterator s when not @@ Set.mem static_indices s -> Some s
        | _ -> None)
    in
    (* Coverage check (replaces the old uniqueness check): every non-static symbol used must be
       groundable by [inline_computation]. A symbol that appears in a bare [Iterator] position is
       bound from the call args; otherwise (a symbol that occurs only inside affine positions, Stage
       B) the whole map must be affine-injective so its symbols are pinned/solvable. Repeated
       [Iterator] positions are allowed and produce equality guards. [inline_computation]
       re-validates per call site, so over-accepting here only risks a later [Non_virtual 13] fall
       back to materialization. *)
    if (not (Lazy.force injective)) && not (Set.is_subset syms ~of_:iter_syms) then
      raise @@ Non_virtual 5
  in
  (* A sibling tensor's access is fine as long as its indices are bound within the candidate's
     computation; only an escaping (unbound) non-static symbol disqualifies (see #134). gh-133 Stage
     B: inspect symbols hidden inside [Affine]/[Concat] too, not just bare [Iterator], so the
     relaxed affine path cannot admit an escaping symbol concealed in an affine index. *)
  let check_sibling_escaping ~env_dom ~code idcs =
    Array.iter idcs ~f:(fun idx ->
        let syms =
          match idx with
          | Indexing.Iterator s -> [ s ]
          | Indexing.Affine { symbols; _ } -> List.map symbols ~f:snd
          | Indexing.Concat syms -> syms
          | Indexing.Fixed_idx _ | Indexing.Sub_axis -> []
        in
        List.iter syms ~f:(fun s ->
            if (not (Set.mem static_indices s)) && not (Set.mem env_dom s) then (
              [%log2
                "INFO: Inlining candidate has an escaping variable",
                (idx : Indexing.axis_index),
                (top_llc : t)];
              raise @@ Non_virtual code)))
  in
  (* Traverse the float code too, for completeness / future use-cases. *)
  let rec loop_proc ~env_dom ~loop_ranges llc =
    let loop = loop_proc ~env_dom ~loop_ranges in
    match llc with
    | Noop -> ()
    | (Seq (c1, c2) : t) ->
        loop c1;
        loop c2
    | For_loop { index; body; from_; to_; axis = _ } ->
        loop_proc ~env_dom:(Set.add env_dom index)
          ~loop_ranges:(Map.set loop_ranges ~key:index ~data:(to_ - from_ + 1))
          body
    | Scan_loop { index; from_; to_; carried; body; _ } ->
        (* gh-ocannl-696: a scan anywhere in the captured computation refuses the candidate. Writing
           the candidate inside it is the obvious case -- cell [i]'s value depends on the whole
           prefix through state the index does not parameterize, so replaying instance [i] cannot
           reproduce it -- but a scan that feeds the candidate through a scope local is the same
           recurrence one level down, and a sibling scan is dropped by the inline filter, which
           cannot keep it without re-minting its carried locals per replay. One verdict for the
           three, decided where the computation is captured. *)
        ignore (index, from_, to_, carried, body);
        raise @@ Non_virtual 148
    | Zero_out tn -> if Tn.equal tn top_tn then has_setter := true
    | Set { tn; idcs; llsc; debug = _ } ->
        if Tn.equal tn top_tn then (
          check_idcs loop_ranges idcs;
          has_setter := true)
        else check_sibling_escaping ~env_dom ~code:7 idcs;
        loop_scalar ~env_dom ~loop_ranges llsc
    | Set_from_vec { tn; idcs; length = _; vec_unop = _; arg = arg, _; debug = _ } ->
        if Tn.equal tn top_tn then (
          check_idcs loop_ranges idcs;
          has_setter := true)
        else check_sibling_escaping ~env_dom ~code:7 idcs;
        loop_scalar ~env_dom ~loop_ranges arg
    | Set_local (_, llsc) -> loop_scalar ~env_dom ~loop_ranges llsc
    (* #296: defensive/unreachable on the fresh-lowering path. [Declare_local] is produced only by
       [hoist_cross_statement_cse], which runs last in [optimize_proc]; [check_and_store_virtual]
       captures computations during [virtual_llc] (before hoisting), so a stored computation never
       contains one. The arm guards the not-currently-exercised case of a hoisted program
       re-entering virtualization. *)
    | Declare_local _ -> raise @@ Non_virtual 19
    | Comment _ -> ()
    | Staged_compilation _ -> raise @@ Non_virtual 8
    (* A barrier is an opaque effect: a computation containing one cannot be inlined/recomputed.
       Defensive: schedule transforms run after virtualization, so this should be unreachable. *)
    | Workgroup_barrier -> raise @@ Non_virtual 141
    (* Cooperative statements are barrier-strength opaque effects; defensive like the barrier. *)
    | Tile_mma _ -> raise @@ Non_virtual 143
    (* gh-466: a dynamically-indexed write is never a definite write of a virtual candidate.
       Defensive: [rewrite_one_hot_reductions] runs after virtualization. *)
    | Set_dynamic _ -> raise @@ Non_virtual 144
    (* Conservative in v1: a guarded computation is not inlined (a conditional write is not a
       definite write, so recomputing it at the read site could read an unset scope). This arm sees
       guards INTERIOR to the captured subtree; a guard ENCLOSING it is reported by the caller via
       [~guarded] (gh-651). Reachable pre-virtualization: [Assignments.to_low_level] emits interval
       guards for clamped-window pooling (gh-504) and extent guards for symbolic extents (gh-490),
       besides the launch-extent guards introduced at backend-compile time. *)
    | If _ -> raise @@ Non_virtual 142
  and loop_scalar ~env_dom ~loop_ranges llsc =
    match llsc with
    | Constant _ | Constant_bits _ -> ()
    | Get (tn, idcs) ->
        if Tn.equal tn top_tn then check_idcs loop_ranges idcs
        else check_sibling_escaping ~env_dom ~code:9 idcs
    | Get_dynamic { dyn_value = v, _; _ } ->
        (* gh-343: defensive -- [Get_dynamic] is produced after virtualization analysis. *)
        loop_scalar ~env_dom ~loop_ranges v
    | Local_scope { body; _ } -> loop_proc ~env_dom ~loop_ranges body
    | Get_local _ -> ()
    | Get_merge_buffer (tn, idcs) ->
        if Tn.equal tn top_tn then check_idcs loop_ranges idcs
        else check_sibling_escaping ~env_dom ~code:9 idcs
    | Embed_index (Fixed_idx _ | Sub_axis) -> ()
    | Embed_index (Iterator s) ->
        if not @@ Set.mem env_dom s then (
          if not (Set.mem static_indices s) then
            [%log2
              "Inlining candidate has an escaping variable", (s : Indexing.symbol), (top_llc : t)];
          raise @@ Non_virtual 10)
    | Embed_index (Affine { symbols; _ }) ->
        List.iter symbols ~f:(fun (_, s) ->
            if not @@ Set.mem env_dom s then (
              if not (Set.mem static_indices s) then
                [%log2
                  "Inlining candidate has an escaping variable",
                  (s : Indexing.symbol),
                  (top_llc : t)];
              raise @@ Non_virtual 10))
    | Embed_index (Concat syms) ->
        List.iter syms ~f:(fun s ->
            if not @@ Set.mem env_dom s then (
              if not (Set.mem static_indices s) then
                [%log2
                  "Inlining candidate has an escaping variable",
                  (s : Indexing.symbol),
                  (top_llc : t)];
              raise @@ Non_virtual 10))
    | Ternop (_, (llv1, _), (llv2, _), (llv3, _)) ->
        loop_scalar ~env_dom ~loop_ranges llv1;
        loop_scalar ~env_dom ~loop_ranges llv2;
        loop_scalar ~env_dom ~loop_ranges llv3
    | Binop (_, (llv1, _), (llv2, _)) ->
        loop_scalar ~env_dom ~loop_ranges llv1;
        loop_scalar ~env_dom ~loop_ranges llv2
    | Unop (_, (llsc, _)) -> loop_scalar ~env_dom ~loop_ranges llsc
  in
  try
    if Tn.Placements.known_non_virtual optim_ctx.placements traced.tn then raise @@ Non_virtual 11;
    (* gh-651: the candidate's whole nest sits inside an enclosing [If], which is NOT part of
       [top_llc] — the walk below would never see it, and the stored computation would replay
       unguarded at every read site. Same verdict as the interior-guard arm, decided here because
       only the caller knows the context the subtree was captured from. *)
    if guarded then raise @@ Non_virtual 142;
    (* gh-ocannl-696: the candidate's setter sits inside a [Scan_loop], which is outside [top_llc]
       like an enclosing guard, so only the caller can report it. Same verdict as a scan met inside
       the nest by the walk above. *)
    if in_scan then raise @@ Non_virtual 148;
    loop_proc ~env_dom:static_indices ~loop_ranges:(Map.empty (module Indexing.Symbol)) top_llc;
    if not !has_setter then raise @@ Non_virtual 12;
    (* gh-651 (loop half): an enclosing [For_loop] whose symbol the candidate's index map does not
       mention replays the whole captured nest into the SAME cells — it is a reduction/repetition
       loop, and it is outside [top_llc], so inlining would replay the setter once instead of
       [width] times. [virtual_llc] captures at the outermost loop whose symbol occurs in the
       candidate's indices, so a reduction loop BELOW that point does ride along inside [top_llc]
       (the ordinary [x[t] += a[s]] shape, whose cost the [max_inline_reduction] cap governs); one
       ABOVE it, or any loop at all when the index map is symbol-free, does not. Only widths above 1
       matter: replaying a single-iteration loop once is exact. *)
    List.iter enclosing ~f:(fun (s, width) ->
        if
          width > 1
          && not (Option.exists !at_idcs ~f:(Array.exists ~f:(axis_index_mentions_symbol s)))
        then raise @@ Non_virtual 147);
    let current_computations =
      Hashtbl.find optim_ctx.computations traced.tn |> Option.value ~default:[]
    in
    Hashtbl.set optim_ctx.computations ~key:traced.tn
      ~data:((!at_idcs, top_llc) :: current_computations)
  with Non_virtual i -> Tn.Placements.update optim_ctx.placements traced.tn Never_virtual i

(* Whether the computation stored for [self] would replay a merge-buffer read when inlined: a
   shared-loop stored body carries SIBLING setters that [inline_computation] filters out, so only
   [self]'s own setters' right-hand sides (and shared control scalars — [If] conditions — which
   inlining keeps) can taint (review round 6). *)
let rec computation_reads_merge ~self : t -> bool = function
  | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ ->
      false
  | Seq (c1, c2) -> computation_reads_merge ~self c1 || computation_reads_merge ~self c2
  (* A dead loop's body replays zero times: no taint from it (review round 10), mirroring
     [drop_dead_loop_accesses] and the fan-in collector's dead-loop skip. *)
  | For_loop { from_; to_; body; _ } -> to_ >= from_ && computation_reads_merge ~self body
  | Scan_loop { from_; to_; carried; body; _ } ->
      List.exists carried ~f:(fun c -> scalar_reads_merge_buffer ~self c.init)
      || (to_ >= from_ && computation_reads_merge ~self body)
  | Set { tn; llsc; _ } -> Tn.equal tn self && scalar_reads_merge_buffer ~self llsc
  | Set_local (_, llsc) -> scalar_reads_merge_buffer ~self llsc
  | Set_from_vec { tn; arg = s, _; _ } -> Tn.equal tn self && scalar_reads_merge_buffer ~self s
  | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
      Tn.equal tn self && (scalar_reads_merge_buffer ~self v || scalar_reads_merge_buffer ~self llsc)
  | If { cond = c0, _; body } ->
      scalar_reads_merge_buffer ~self c0 || computation_reads_merge ~self body
  | Tile_mma { fallback; _ } -> computation_reads_merge ~self fallback

and scalar_reads_merge_buffer ~self : scalar_t -> bool = function
  | Constant _ | Constant_bits _ | Embed_index _ | Get_local _ | Get _ -> false
  | Get_merge_buffer _ -> true
  | Get_dynamic { dyn_value = v, _; _ } -> scalar_reads_merge_buffer ~self v
  | Local_scope { body; _ } -> computation_reads_merge ~self body
  | Ternop (_, (a, _), (b, _), (d, _)) ->
      scalar_reads_merge_buffer ~self a || scalar_reads_merge_buffer ~self b
      || scalar_reads_merge_buffer ~self d
  | Binop (op, (a, _), (b, _)) -> (
      (* A projection's discarded operand is never rendered, hence never reads anything: it must not
         taint (review round 3). A gated second operand may evaluate, so it counts. *)
      match Ops.binop_conditionality op with
      | Ops.Only_first -> scalar_reads_merge_buffer ~self a
      | Ops.Only_second -> scalar_reads_merge_buffer ~self b
      | Ops.Both_operands | Ops.Gated_second ->
          scalar_reads_merge_buffer ~self a || scalar_reads_merge_buffer ~self b)
  | Unop (_, (a, _)) -> scalar_reads_merge_buffer ~self a

let%track7_sexp inline_computation ~id ~inherited_merge_tainted ~inherited_tns
    (optim_ctx : optimize_ctx) (traced : traced_array)
    (static_indices : Indexing.static_symbol list) (call_args : Indexing.axis_index array) :
    t option =
  let exception Non_virtual of int in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  let computations =
    Hashtbl.find optim_ctx.computations traced.tn
    |> Option.value_or_thunk ~default:(fun () ->
        raise
        @@ Utils.User_error
             [%string
               "Stale optimize_ctx: No computations found for #%{traced.tn.Tn.id#Int}: \
                %{Tn.debug_name traced.tn}"])
  in
  (* Review round 2 of the gh-610/611 PR: a computation INHERITED from an earlier routine of the
     lineage that reads a merge buffer must not be consumed here — merge-buffer contents are
     transient to the routine receiving the transfer, and this routine declaring the SAME merge
     source does not rescue the splice: the deferred read would observe this routine's transfer, not
     the one it was written against. Detection is at consumption time because only the entry-time
     snapshot (taken in [virtual_llc]) can tell inherited components from ones stored during this
     routine's own walk; a post-hoc scan of the final code cannot. *)
  if Hash_set.mem inherited_merge_tainted traced.tn then
    raise
    @@ Utils.User_error
         [%string
           "the deferred computation of %{Tn.debug_name traced.tn} was stored by an earlier \
            routine of this compilation lineage and reads a merge buffer: merge-buffer contents \
            are transient to the routine receiving the transfer, so the computation cannot be \
            consumed across routines. Mark %{Tn.debug_name traced.tn} as materialized (e.g. via \
            Train.set_materialized) in the routine that computes it."];
  (* gh-509 task 4: a packed-uniform producer ([Set_from_vec]) is inlined via the lane-extract
     scalar form [vec_convert(counter[flat / lanes]).v[flat mod lanes]], where [flat] is the read
     cell's flat offset -- bitwise-identical to the vectorized stores (the lane builtins index the
     same converted block). The value stream depends only on the element index, so a single builder
     serves every peeled computation (interior and tail write the same stream). The counter is read
     at a runtime-computed block index ([Get_dynamic]), so it is committed to a materialized
     placement; its own (inlineable) chain stays virtual inside its setter. *)
  let set_from_vec_def =
    List.find_map computations ~f:(fun (_, def) ->
        let rec find = function
          | Set_from_vec { tn; idcs; length = _; vec_unop; arg = arg_scalar, _; debug = _ }
            when Tn.equal tn traced.tn ->
              Some (idcs, vec_unop, arg_scalar)
          | For_loop { body; _ } -> find body
          | Seq (a, b) -> ( match find a with Some _ as r -> r | None -> find b)
          | _ -> None
        in
        find def)
  in
  let lane_extract_body (idcs, vec_unop, arg_scalar) : t =
    (* [Set_from_vec] assumes an unpadded dense target; lowering rejects padded targets, re-check
       defensively (the flat-offset arithmetic below would be wrong under padding). *)
    (match Tn.get_padding traced.tn with
    | None -> ()
    | Some (pads, _) when Array.for_all pads ~f:(fun p -> p.Ops.left = 0 && p.Ops.right = 0) -> ()
    | Some _ -> raise @@ Non_virtual 145);
    let prec = Lazy.force traced.tn.Tn.storage_prec in
    let lanes = Ops.vec_unop_lanes prec in
    let dims = Lazy.force traced.tn.Tn.dims in
    (* Flat row-major offset of an index vector over [dims], as affine terms plus a constant. *)
    let flat_of arr =
      if Array.length arr <> Array.length dims then raise @@ Non_virtual 145;
      let terms = ref [] and offset = ref 0 and stride = ref 1 in
      for i = Array.length arr - 1 downto 0 do
        (match arr.(i) with
        | Indexing.Fixed_idx k -> offset := !offset + (k * !stride)
        | Indexing.Sub_axis -> ()
        | Indexing.Iterator s -> terms := (!stride, s) :: !terms
        | Indexing.Affine { symbols; offset = o } ->
            offset := !offset + (o * !stride);
            List.iter symbols ~f:(fun (c, s) -> terms := (c * !stride, s) :: !terms)
        | Indexing.Concat _ -> raise @@ Non_virtual 145);
        stride := !stride * dims.(i)
      done;
      (Indexing.coalesce_affine_terms !terms, !offset)
    in
    (* The producer's flat store offset must be [lanes * counter_iterator] (a single-block target
       collapses to a constant 0: its dim-1 counter axis has no iterator in the projections). *)
    let ctr_sym =
      match flat_of idcs with
      | [], 0 -> None
      | [ (c, s) ], 0 when c = lanes -> Some s
      | _ -> raise @@ Non_virtual 145
    in
    let flat_idx =
      match flat_of call_args with
      | [], offset -> Indexing.Fixed_idx offset
      | terms, offset -> Indexing.Affine { symbols = terms; offset }
    in
    let iprec = Ops.index_prec () in
    let flat_sc = Embed_index flat_idx in
    let lanes_sc = Embed_index (Indexing.Fixed_idx lanes) in
    let mentions_nonstatic = function
      | Indexing.Iterator s -> not (Set.mem static_indices s)
      | Indexing.Affine { symbols; _ } ->
          List.exists symbols ~f:(fun (_, s) -> not (Set.mem static_indices s))
      | Indexing.Fixed_idx _ | Indexing.Sub_axis -> false
      | Indexing.Concat syms -> List.exists syms ~f:(fun s -> not (Set.mem static_indices s))
    in
    let ctr_read =
      match arg_scalar with
      | Get (ctr, ctr_idcs) ->
          (* The counter is about to gain a dynamically-indexed read, which recomputation cannot
             serve: it must stay materialized. Bail out (keeping the uniform result materialized, as
             before gh-509 task 4) if the counter's placement is already committed the other way. *)
          (match Tn.Placements.get optim_ctx.placements ctr with
          | Some ((Virtual | Effectively_constant), _) -> raise @@ Non_virtual 146
          | _ -> ());
          let ctr_read =
            match ctr_sym with
            | None ->
                (* Single-block target: the counter read is at static indices; keep the plain [Get]
                   (cleanup commits the surviving read to [Never_virtual]). *)
                if Array.exists ctr_idcs ~f:mentions_nonstatic then raise @@ Non_virtual 146;
                Get (ctr, ctr_idcs)
            | Some c ->
                let dyn_positions =
                  Array.filter_mapi ctr_idcs ~f:(fun i idx ->
                      match idx with
                      | Indexing.Iterator s when Indexing.equal_symbol s c -> Some i
                      | idx when mentions_nonstatic idx -> Some (-1)
                      | _ -> None)
                in
                let dyn_axis =
                  match Array.to_list dyn_positions with
                  | [ i ] when i >= 0 -> i
                  | _ -> raise @@ Non_virtual 146
                in
                let idcs' =
                  Array.mapi ctr_idcs ~f:(fun i idx ->
                      if i = dyn_axis then Indexing.Fixed_idx 0 else idx)
                in
                let block_sc = Binop (Ops.Div, (flat_sc, iprec), (lanes_sc, iprec)) in
                Get_dynamic { tn = ctr; idcs = idcs'; dyn_axis; dyn_value = (block_sc, iprec) }
          in
          Tn.Placements.update optim_ctx.placements ctr Never_virtual 146;
          ctr_read
      | _ ->
          (* Argument is not a plain counter read (e.g. the counter chain was materialized away or
             the computation is exotic); keep the vector store materialized. *)
          raise @@ Non_virtual 140
    in
    let lane_sc =
      match ctr_sym with
      | None -> flat_sc (* Single block: [flat < lanes] already. *)
      | Some _ -> Binop (Ops.Mod, (flat_sc, iprec), (lanes_sc, iprec))
    in
    let lane_op =
      match vec_unop with Ops.Uint4x32_to_prec_uniform -> Ops.Uint4x32_to_prec_uniform_lane
    in
    Set_local (id, Binop (lane_op, (ctr_read, scalar_precision ctr_read), (lane_sc, iprec)))
  in
  (* gh-133 Stage A: a guard's else-branch reads [Get_local id] -- the zero/init local produced by a
     [Zero_out] computation. If the producer has no [Zero_out] (e.g. a surjective producer that was
     previously materialized), that local is never initialized, so we must NOT emit a guard; such
     reads fall back to materialization via [Non_virtual 13] below, preserving prior behavior. *)
  let has_zero_init =
    let rec contains = function
      | Zero_out tn -> Tn.equal tn traced.tn
      | Seq (a, b) -> contains a || contains b
      | For_loop { body; _ } | Scan_loop { body; _ } -> contains body
      | _ -> false
    in
    List.exists computations ~f:(fun (_, def) -> contains def)
  in
  (* For Block/Concat virtual nodes with multiple Set computations (each writing a distinct slice),
     a single init local is shared across all component computations so that each guarded update
     sees the preceding component's value via [Get_local id] rather than a per-component reset to 0.
     [None] entries are [Zero_out] computations and are excluded from the count. *)
  let set_computation_count =
    List.count computations ~f:(fun (def_args, _) -> Option.is_some def_args)
  in
  let global_needs_init = ref false in
  (* In the order of computation. *)
  let loop_proc ((def_args : Indexing.axis_index array option), (def : t)) : t option =
    (* One substitution step: replace env-bound symbols, folding nested affine/fixed
       contributions. *)
    let subst_step env (idx : Indexing.axis_index) : Indexing.axis_index =
      match idx with
      | Indexing.Iterator s when Map.mem env s -> Map.find_exn env s
      | Indexing.Affine { symbols; offset } ->
          (* We need to substitute each symbol in the affine expression. If a symbol maps to a
             non-Iterator, we need to handle it specially. *)
          let expand_symbol (coeff, s) =
            match Map.find env s with
            | Some (Indexing.Iterator new_s) -> [ (coeff, new_s) ]
            | Some (Indexing.Fixed_idx _ | Indexing.Sub_axis) ->
                [] (* Fixed index contributes to offset *)
            | Some (Indexing.Affine { symbols = inner_symbols; offset = _ }) ->
                (* Expand nested affine: coeff * (inner_symbols + inner_offset) *)
                List.map inner_symbols ~f:(fun (inner_coeff, inner_s) ->
                    (coeff * inner_coeff, inner_s))
            | Some (Indexing.Concat _) ->
                (* Concat should not appear in affine substitution *)
                failwith "BUG: Concat in affine substitution not supported"
            | None -> [ (coeff, s) ]
          in
          let all_terms = List.concat_map symbols ~f:expand_symbol in
          (* Calculate the new offset by adding contributions from Fixed_idx substitutions *)
          let offset_additions =
            List.fold symbols ~init:0 ~f:(fun acc (coeff, s) ->
                match Map.find env s with
                | Some (Indexing.Fixed_idx i) -> acc + (coeff * i)
                | Some (Indexing.Affine { offset = inner_offset; _ }) -> acc + (coeff * inner_offset)
                | _ -> acc)
          in
          let new_offset = offset + offset_additions in
          Indexing.Affine { symbols = all_terms; offset = new_offset }
      | idx -> idx
    in
    (* gh-133 Stage B: a solved producer symbol can be bound to an affine expression that still
       mentions other producer symbols (e.g. [wh := t - 2*oh] when [oh]'s loop is kept). Those inner
       symbols are themselves env-bound (to a freshened loop variable), so substitution must be
       applied transitively to a fixpoint. Bindings are acyclic and finite, so this terminates. *)
    let rec subst env idx =
      let stepped = subst_step env idx in
      if Indexing.equal_axis_index stepped idx then stepped else subst env stepped
    in
    (* Canonical [(coeff, symbol) list, offset] view of an affine-like position. *)
    let canon idx =
      match idx with
      | Indexing.Iterator s -> Some ([ (1, s) ], 0)
      | Indexing.Affine { symbols; offset } -> Some (Indexing.coalesce_affine_terms symbols, offset)
      | Indexing.Fixed_idx i -> Some ([], i)
      | Indexing.Sub_axis | Indexing.Concat _ -> None
    in
    (* Per-symbol loop width of this producer computation, for range guards on solved symbols. *)
    let def_loop_ranges =
      let rec scan acc = function
        | For_loop { index; from_; to_; body; _ } | Scan_loop { index; from_; to_; body; _ } ->
            scan (Map.set acc ~key:index ~data:(to_ - from_ + 1)) body
        | Seq (a, b) -> scan (scan acc a) b
        | _ -> acc
      in
      scan (Map.empty (module Indexing.Symbol)) def
    in
    let symbol_range s = Map.find def_loop_ranges s |> Option.value ~default:1 in
    (* gh-133 Stage A/B: build the substitution environment and guards in several passes, so a
       producer index vector that repeats a non-static symbol (diagonal [i;i] / partially-diagonal
       [i;j;i]) does not crash on duplicate keys, and so multi-symbol affine positions (Stage B) can
       be solved. *)
    let def_args_arr = Option.value def_args ~default:[||] in
    let n = Array.length def_args_arr in
    if n > Array.length call_args then
      failwith
        [%string
          "inline_computation: call_args too short, maybe stale optimization context? Tnode: \
           %{Tn.debug_name traced.tn} #%{traced.tn.Tn.id#Int} n: %{n#Int}"];
    (* [bound_pos.(i)] marks positions that DEFINE bindings (no consistency guard for them).
       [range_guards] collects [(solved_symbol, range)] pairs whose value must fall in [0,
       range). *)
    let bound_pos = Array.create ~len:n false in
    let env = ref (Map.empty (module Indexing.Symbol)) in
    let range_guards = ref [] in
    (* Pass 1: bind the first bare non-static [Iterator] occurrence of each symbol. For Block/Concat
       virtual nodes with multiple Set computations, also emit a range guard: the producer's loop
       covers [0, range) while the consumer iterates a wider range, so out-of-range consumer
       iterations must fall back to [Get_local id]. *)
    Array.iteri def_args_arr ~f:(fun i lhs_ind ->
        match lhs_ind with
        | Indexing.Iterator s when (not (Set.mem static_indices s)) && not (Map.mem !env s) ->
            bound_pos.(i) <- true;
            env := Map.add_exn !env ~key:s ~data:call_args.(i);
            if set_computation_count > 1 then
              range_guards :=
                (1, Indexing.Fixed_idx 0, call_args.(i), symbol_range s) :: !range_guards
        | _ -> ());
    (* gh-133 Stage B: structural affine match -- producer [Σ cₖ·sₖ + off] read at a call-site
       affine with the same canonical (distinct) coefficient list and equal offset binds producer
       symbols pairwise to the call-site symbols, no inversion and no guard. *)
    let try_structural_match i lhs_ind =
      if bound_pos.(i) then false
      else
        match (lhs_ind, canon call_args.(i)) with
        | Indexing.Affine { symbols = psyms; offset = poff }, Some (cterms, coff) when poff = coff
          ->
            let pterms = Indexing.coalesce_affine_terms psyms in
            let pcoeffs = List.map pterms ~f:fst in
            let ccoeffs = List.map cterms ~f:fst in
            let distinct l =
              List.length (List.dedup_and_sort ~compare:Int.compare l) = List.length l
            in
            let unbound (_, s) = (not (Set.mem static_indices s)) && not (Map.mem !env s) in
            if
              (not (List.is_empty pterms))
              && List.for_all pterms ~f:unbound && distinct pcoeffs
              && List.equal Int.equal
                   (List.sort ~compare:Int.compare pcoeffs)
                   (List.sort ~compare:Int.compare ccoeffs)
            then (
              List.iter pterms ~f:(fun (c, ps) ->
                  let _, cs = List.find_exn cterms ~f:(fun (cc, _) -> cc = c) in
                  env := Map.set !env ~key:ps ~data:(Indexing.Iterator cs));
              bound_pos.(i) <- true;
              true)
            else false
        | _ -> false
    in
    (* gh-133 Stage B: unit-coefficient solving -- after substituting already-bound symbols, if
       exactly one unbound producer symbol has coefficient ±1, bind it to the residual affine
       expression and emit a range guard. Other unbound symbols stay free; their producer loops are
       kept (and range-guarded indirectly via injectivity), guaranteeing exactly one matching
       iteration. *)
    let try_unit_solve i lhs_ind =
      if bound_pos.(i) then false
      else
        match lhs_ind with
        | Indexing.Affine { symbols; offset } -> (
            let terms = Indexing.coalesce_affine_terms symbols in
            let unbound =
              List.filter terms ~f:(fun (_, s) ->
                  (not (Set.mem static_indices s)) && not (Map.mem !env s))
            in
            match List.filter unbound ~f:(fun (c, _) -> abs c = 1) with
            | [ (uc, us) ] -> (
                match canon call_args.(i) with
                | None -> false
                | Some (rterms, roff) ->
                    (* us = uc * (rhs − offset − Σ_{other terms} c·s). uc = ±1 so uc⁻¹ = uc. Other
                       producer terms are left symbolic and resolved by [subst] (transitively). *)
                    let value_terms =
                      List.map rterms ~f:(fun (rc, rs) -> (uc * rc, rs))
                      @ List.filter_map terms ~f:(fun (c, s) ->
                          if Indexing.equal_symbol s us then None else Some (-uc * c, s))
                    in
                    let value =
                      Indexing.Affine { symbols = value_terms; offset = uc * (roff - offset) }
                    in
                    env := Map.set !env ~key:us ~data:value;
                    (* Range guard [0 <= us < range], reformulated with NON-NEGATIVE operands so it
                       is correct under unsigned index precision (a negative [rhs - rest] would
                       underflow): [rest := Σ_{s≠us} c·s + offset], [rhs := call index]. uc=+1 needs
                       [rest <= rhs < rest+range]; uc=-1 needs [rhs <= rest < rhs+range]. *)
                    let rest_axis =
                      Indexing.Affine
                        {
                          symbols =
                            List.filter terms ~f:(fun (_, s) -> not (Indexing.equal_symbol s us));
                          offset;
                        }
                    in
                    range_guards := (uc, rest_axis, call_args.(i), symbol_range us) :: !range_guards;
                    bound_pos.(i) <- true;
                    true)
            | _ -> false)
        | _ -> false
    in
    (* Run the Stage B binding rounds to a fixpoint, in pinning order (structural match before
       unit-coefficient solving). *)
    let progress = ref true in
    while !progress do
      progress := false;
      Array.iteri def_args_arr ~f:(fun i lhs_ind ->
          if (not bound_pos.(i)) && (try_structural_match i lhs_ind || try_unit_solve i lhs_ind)
          then progress := true)
    done;
    let env = !env in
    (* Remaining non-binding positions become consistency guards: [subst(producer_pos) = call_site].
       [Fixed_idx]/[Sub_axis] carry no symbol, so a static mismatch there is a genuine
       sparse-producer mismatch and stays materialized via [Non_virtual 13]. Guards are materialized
       at the [Set] node with the live (freshened) env. *)
    let depends_on_symbol (idx : Indexing.axis_index) =
      match idx with
      | Indexing.Iterator s -> not (Set.mem static_indices s)
      | Indexing.Affine { symbols; _ } ->
          List.exists symbols ~f:(fun (_, s) -> not (Set.mem static_indices s))
      | Indexing.Fixed_idx _ | Indexing.Sub_axis -> false
      | Indexing.Concat _ -> false (* unreachable: rejected by check_idcs *)
    in
    let guards =
      Array.foldi def_args_arr ~init:[] ~f:(fun i guards lhs_ind ->
          if bound_pos.(i) then guards
          else
            let rhs_ind = call_args.(i) in
            let lhs' = subst env lhs_ind in
            if Indexing.equal_axis_index lhs' rhs_ind then guards
            else if depends_on_symbol lhs_ind then (lhs_ind, rhs_ind) :: guards
            else raise @@ Non_virtual 13)
    in
    let guards = List.rev guards in
    let range_guards = List.rev !range_guards in
    (* Set when a guard is introduced but the producer emitted no [Zero_out]: an explicit init local
       is then prepended before the (possibly loop-nested) guarded updates. *)
    let needs_init = ref false in
    let rec loop env llc : t option =
      match llc with
      | Noop -> None
      | Seq _ ->
          let body = List.filter_map ~f:(loop env) @@ flat_lines [ llc ] in
          if List.is_empty body then None else Some (unflat_lines body)
      | For_loop { index; body; _ } when Map.mem env index -> loop env body
      | For_loop { from_; to_; _ } when to_ < from_ ->
          (* A dead loop replays zero times: drop it from the spliced body (review round 10) — it
             would otherwise RENDER in the consumer (renderers emit dead loops), and a merge read
             inside it would reference a parameter the consumer cannot declare. Mirrors the tracer's
             and the fan-in collector's dead-body skips. *)
          Some Noop
      | For_loop { index; from_; to_; body; axis } ->
          (* Freshen the binding. *)
          let fresh = Indexing.get_symbol () in
          let env = Map.add_exn ~key:index ~data:(Indexing.Iterator fresh) env in
          Option.map ~f:(fun body : t -> For_loop { index = fresh; from_; to_; body; axis })
          @@ loop env body
      | Scan_loop _ ->
          (* gh-ocannl-696: a computation containing a scan never reaches storage
             ([check_and_store_virtual]'s [Non_virtual 148] on the scan itself), so this arm is a
             consumption-time backstop with the same verdict, never a silent drop: the filter cannot
             keep a scan without re-minting its carried locals per replay, and dropping one that
             feeds the value through a scope local would return the pre-scan value. *)
          raise @@ Non_virtual 148
      | Zero_out tn when Tn.equal tn traced.tn -> Some (Set_local (id, Constant 0.0))
      | Set { tn; idcs; llsc; debug = _ } when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some idcs) def_args);
          let inlined = loop_scalar env llsc in
          let value_prec = Lazy.force traced.tn.Tn.storage_prec in
          let index_prec = Ops.index_prec () in
          (* gh-133 Stage A: consistency (equality) guards for repeated / covered single-symbol
             affine positions -- the substituted producer index must equal the call-site index.
             Indices are resolved with the live (freshened) env so kept-loop symbols match the loop
             body. Index comparison uses index precision (Cmpeq is homogeneous); the [Where] keeps
             value precision on its then/else arms. *)
          let eq_conds =
            List.map guards ~f:(fun (lhs_ind, rhs_ind) ->
                Binop
                  ( Ops.Cmpeq,
                    (Embed_index (subst env lhs_ind), index_prec),
                    (Embed_index rhs_ind, index_prec) ))
          in
          (* gh-133 Stage B: range guards -- a unit-solved symbol's value must fall within its
             producer loop range [0, range). The guard never forms a negative intermediate: we
             compare [rest] and [rhs] (both non-negative) rather than [rhs - rest]. uc=+1: [rest <=
             rhs] & [rhs < rest+range]; uc=-1: [rhs <= rest] & [rest < rhs+range] -- one canonical
             shape per role, a direct [Cmple] lower bound and a strict [Cmplt] upper bound. *)
          let add_offset (idx : Indexing.axis_index) d : Indexing.axis_index =
            if d = 0 then idx
            else
              match idx with
              | Indexing.Iterator s -> Indexing.Affine { symbols = [ (1, s) ]; offset = d }
              | Indexing.Affine { symbols; offset } ->
                  Indexing.Affine { symbols; offset = offset + d }
              | Indexing.Fixed_idx i -> Indexing.Fixed_idx (i + d)
              | Indexing.Sub_axis | Indexing.Concat _ -> idx
          in
          let cmp op a b = Binop (op, (Embed_index a, index_prec), (Embed_index b, index_prec)) in
          let le = cmp Ops.Cmple and lt = cmp Ops.Cmplt in
          let range_conds =
            List.map range_guards ~f:(fun (uc, rest_axis, rhs_axis, range) ->
                let rest = subst env rest_axis and rhs = subst env rhs_axis in
                let lower, upper =
                  if uc >= 0 then (le rest rhs, lt rhs (add_offset rest range))
                  else (le rhs rest, lt rest (add_offset rhs range))
                in
                Binop (Ops.And, (lower, index_prec), (upper, index_prec)))
          in
          let conds = eq_conds @ range_conds in
          (* Off-condition reads fall back to [Get_local id] -- the init local emitted by the
             producer's [Zero_out] ([has_zero_init]) or, when absent (an injective+surjective
             scatter that skipped neutral init -- Stage B), the explicit init prepended below. The
             no-[Zero_out] case implies the map is surjective, so every read cell IS written by
             exactly one iteration (injectivity) and the init value is always overwritten -- 0. is a
             safe neutral. *)
          if (not (List.is_empty conds)) && not has_zero_init then needs_init := true;
          (* task-9658aac9: the then-arm [acc] structurally contains the producer read at the
             unit-solved index, so for a non-matching kept-loop iteration that index (hence the flat
             offset) can be out of range. This is sound today: every backend lowers [Where] to a C/
             CUDA/Metal ternary that short-circuits, so arithmetic execution never dereferences the
             discarded then-branch. The one path that would dereference it unconditionally is debug
             value-logging, which is made safe by [C_syntax.debug_float]'s [Where] arm gating each
             branch read on its (negated) condition. Full structural soundness -- index clamping or
             an IR branch so the discarded read is never even constructed -- is deferred until an
             eager/predicated backend that evaluates both [Where] arms actually lands. *)
          let guarded =
            List.fold conds ~init:inlined ~f:(fun acc cond ->
                Ternop (Ops.Where, (cond, index_prec), (acc, value_prec), (Get_local id, value_prec)))
          in
          Some (Set_local (id, guarded))
      | Set_from_vec { tn; idcs; length = _; vec_unop = _; arg = _; debug = _ }
        when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some idcs) def_args);
          (* Unreachable since gh-509 task 4: computations containing a [Set_from_vec] setter of
             this node are diverted to [lane_extract_body] before the generic path. *)
          raise @@ Non_virtual 140
      | Zero_out _ -> None
      | Set _ -> None
      | Set_from_vec _ -> None
      | Set_local (id, llsc) -> Some (Set_local (id, loop_scalar env llsc))
      | Declare_local _ -> None
      | Comment _ -> Some llc
      | Staged_compilation _ -> Some llc
      (* Unreachable: [check_and_store_virtual] rejects computations containing barriers,
         cooperative tile statements, guarded statements, and (gh-466) dynamic scatters. *)
      | Workgroup_barrier | Tile_mma _ | If _ | Set_dynamic _ -> assert false
    and loop_scalar env llsc : scalar_t =
      match llsc with
      | Constant _ | Constant_bits _ -> llsc
      | Get (tn, indices) when Tn.equal tn traced.tn ->
          assert ([%equal: Indexing.axis_index array option] (Some indices) def_args);
          Get_local id
      | Get (tn, indices) -> Get (tn, Array.map ~f:(subst env) indices)
      | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
          (* gh-343: defensive -- [Get_dynamic] is produced after virtualization. *)
          Get_dynamic
            {
              tn;
              idcs = Array.map ~f:(subst env) idcs;
              dyn_axis;
              dyn_value = (loop_scalar env v, prec);
            }
      | Local_scope { id; body; orig_indices; mint } ->
          Local_scope
            {
              id;
              body = Option.value_exn ~here:[%here] @@ loop env body;
              orig_indices = Array.map ~f:(subst env) orig_indices;
              mint;
            }
      | Get_local _ -> llsc
      | Get_merge_buffer (tn, indices) -> Get_merge_buffer (tn, Array.map ~f:(subst env) indices)
      | Embed_index idx -> Embed_index (subst env idx)
      | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
          Ternop
            ( op,
              (loop_scalar env llv1, prec1),
              (loop_scalar env llv2, prec2),
              (loop_scalar env llv3, prec3) )
      | Binop (op, (llv1, prec1), (llv2, prec2)) ->
          Binop (op, (loop_scalar env llv1, prec1), (loop_scalar env llv2, prec2))
      | Unop (op, (llsc, prec)) -> Unop (op, (loop_scalar env llsc, prec))
    in
    match loop env def with
    | Some body ->
        if !needs_init then global_needs_init := true;
        Some body
    | None -> None
  in
  try
    match set_from_vec_def with
    | Some def -> Some (lane_extract_body def)
    | None ->
        let body = List.rev_filter_map ~f:loop_proc computations in
        if List.is_empty body then raise @@ Non_virtual 14
        else
          (* Prepend a single init local when any component computation has guards but the producer
             has no [Zero_out] to supply the initial value. For multi-computation nodes
             (Block/Concat) this emits exactly one reset rather than one per component, so each
             component's guarded update sees the preceding component's value via [Get_local id]
             rather than 0. *)
          let body =
            if !global_needs_init && not has_zero_init then Set_local (id, Constant 0.0) :: body
            else body
          in
          Some (unflat_lines body)
  with Non_virtual i ->
    (* Review round 11: an INHERITED computation has no materialization fallback — the deferring
       routine already dropped the setters from its schedule, so committing [Never_virtual] here
       would either conflict with the lineage's [Virtual] commitment (a cryptic provenance-collision
       error) or commit the consumer to a buffer no routine writes. Fail actionably instead. *)
    if Hash_set.mem inherited_tns traced.tn then
      raise
        (Utils.User_error
           [%string
             "the deferred computation of %{Tn.debug_name traced.tn}, stored by an earlier routine \
              of this compilation lineage, could not be inlined at a read site of this routine \
              (rejection %{i#Int}: unsupported indexing or vector form for inlining): no routine \
              writes the node's buffer, so the read cannot fall back to a materialized access. \
              Mark %{Tn.debug_name traced.tn} as materialized (e.g. via Train.set_materialized) in \
              the routine that computes it."]);
    Tn.Placements.update optim_ctx.placements traced.tn Never_virtual i;
    None

let optimize_integer_pow = ref true

let rec unroll_pow ~(base : scalar_t) ~(exp : int) : scalar_t =
  if exp < 0 then
    unroll_pow
      ~base:(Binop (Div, (Constant 1., Ops.single), (base, scalar_precision base)))
      ~exp:(Int.neg exp)
  else if exp = 0 then Constant 1.
  else
    Fn.apply_n_times ~n:(exp - 1)
      (fun accu -> Binop (Mul, (base, scalar_precision base), (accu, scalar_precision accu)))
      base

(* gh-509 task 4: packed-uniform ([Set_from_vec]) producers are stored RAW -- without the phase-1
   rewriting that inlines virtual providers into the argument -- so that [inline_computation]'s
   lane-extract builder sees the argument as a plain [Get] of the counter tensor (which it turns
   into a dynamically-indexed read of block [flat / lanes]). The stored computation of such a node
   is consumed exclusively by the lane-extract builder, so skipping the rewrite loses nothing; the
   phase-2 emitted statement (used when the node stays materialized) is rewritten as before. *)
let rec proc_contains_set_from_vec tn = function
  | Set_from_vec { tn = tn2; _ } -> Tn.equal tn tn2
  | For_loop { body; _ } | Scan_loop { body; _ } -> proc_contains_set_from_vec tn body
  | Seq (a, b) -> proc_contains_set_from_vec tn a || proc_contains_set_from_vec tn b
  | If { body; _ } -> proc_contains_set_from_vec tn body
  | _ -> false

(** gh-ocannl-734: a dynamic-gather table that is already committed [Virtual] is unsatisfiable — the
    gather index is only known at runtime, so recomputation cannot serve the read, while a [Virtual]
    node owns no buffer to load from. Both [Get_dynamic] arms of the virtualization pipeline report
    it with this message instead of letting [Tn.Placements.update] answer with a bare provenance
    collision ("update 152 -> 17 for ... is already virtual"), which named neither the node's two
    readings nor anything the author could act on.

    The situation is out of contract for the ordinary pipeline — [Assignments] lowering builds no
    [Get_dynamic], and the one the pipeline does produce comes from [rewrite_one_hot_reductions],
    which runs after both arms — so it is reached only from hand-built IR through [optimize] /
    [Ll_test], which is a supported input class for the analysis probes. *)
let virtualized_gather_table_rejection (tn : Tn.t) ~(decided_by : string) : string =
  [%string
    "the program reads %{Tn.debug_name tn} as a dynamic-gather table (Get_dynamic), but \
     %{Tn.debug_name tn} is virtual in this routine (%{decided_by}). A dynamically-indexed read \
     cannot be served by inlining -- the gathered row is only known at runtime, so there is no \
     computation to replay at the read site -- while a virtual node has no buffer to load from, so \
     the two readings cannot both hold. Materialize %{Tn.debug_name tn} (e.g. via \
     Train.set_materialized, or Ll_test's ~materialized for hand-built IR) instead of declaring it \
     virtual; a table whose placement is merely undecided is materialized for you by this arm."]

let virtual_llc (optim_ctx : optimize_ctx) traced_store reverse_node_map static_indices (llc : t) :
    t * Tnode.t Hash_set.t =
  let plc = optim_ctx.placements in
  (* Every array read inside the inlined body of an INHERITED computation (present in the lineage
     table at routine entry — the snapshot above): cross-routine splicing introduces reads the raw
     analysis never saw, so the reconcile-time strict coverage verdicts apply exactly to these
     nodes, and a raw-positioned or locally-inlined read keeps the raw verdicts (review round 7: a
     routine-wide gate dragged unrelated raw reads into strict re-judging, a has-local-assignment
     provenance test missed consumption through an update of an inherited virtual, and recording
     LOCAL splices re-broke the init flows — local bodies' reads were already judged by the raw
     analysis at their producer positions). Nested inherited consumption inside a local body still
     records: the nested [Get] routes through the same arm while the local setter is processed for
     storage. *)
  let spliced_reads = Hash_set.create (module Tnode) in
  let rec record_spliced_reads (c : t) =
    match c with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ ->
        ()
    | Seq (c1, c2) ->
        record_spliced_reads c1;
        record_spliced_reads c2
    | For_loop { body; _ } -> record_spliced_reads body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> record_spliced_scalar c.init);
        record_spliced_reads body
    | Set { llsc; _ } | Set_local (_, llsc) -> record_spliced_scalar llsc
    | Set_from_vec { arg = sc, _; _ } -> record_spliced_scalar sc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        record_spliced_scalar v;
        record_spliced_scalar llsc
    | If { cond = c0, _; body } ->
        record_spliced_scalar c0;
        record_spliced_reads body
    | Tile_mma { fallback; _ } -> record_spliced_reads fallback
  and record_spliced_scalar (sc : scalar_t) =
    match sc with
    | Constant _ | Constant_bits _ | Embed_index _ | Get_local _ | Get_merge_buffer _ -> ()
    | Get (tn, _) -> Hash_set.add spliced_reads tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        Hash_set.add spliced_reads tn;
        record_spliced_scalar v
    | Local_scope { body; _ } -> record_spliced_reads body
    | Ternop (_, (a, _), (b, _), (d, _)) ->
        record_spliced_scalar a;
        record_spliced_scalar b;
        record_spliced_scalar d
    | Binop (op, (a, _), (b, _)) -> (
        (* A projection's discarded operand is never evaluated: its reads are not spliced (review
           round 8) — same dispatch as the reconcile and merge-taint walkers. *)
        match Ops.binop_conditionality op with
        | Ops.Only_first -> record_spliced_scalar a
        | Ops.Only_second -> record_spliced_scalar b
        | Ops.Both_operands | Ops.Gated_second ->
            record_spliced_scalar a;
            record_spliced_scalar b)
    | Unop (_, (a, _)) -> record_spliced_scalar a
  in
  (* The entry-time snapshot of merge-tainted deferred computations: everything in the table at this
     point was stored by an earlier routine of the lineage (this routine's own computations are
     stored during the walk below), so a node found here with a merge-buffer-reading body is exactly
     a cross-routine merge splice waiting to happen — [inline_computation] rejects its consumption.
     A node with local setters ON TOP of an inherited tainted component (the
     update-an-inherited-virtual pattern) is tainted all the same: inlining would replay the
     inherited component. *)
  let inherited_tns = Hash_set.create (module Tnode) in
  let inherited_merge_tainted =
    let tainted = Hash_set.create (module Tnode) in
    Hashtbl.iteri optim_ctx.computations ~f:(fun ~key ~data ->
        Hash_set.add inherited_tns key;
        if List.exists data ~f:(fun (_, body) -> computation_reads_merge ~self:key body) then
          Hash_set.add tainted key);
    tainted
  in
  (* [process_for] holds tensors whose [Get]s must be left untouched (self/recursive references,
     replaced by [Get_local] during the tensor's own [inline_computation]). [owned] holds tensors
     whose whole-loop computation is captured at an enclosing [For_loop]: their per-statement
     auto-store is suppressed and they are excluded from nested candidate lists, but -- unlike
     [process_for] -- reads of them are still inlined, so surviving sibling readers can inline a
     virtualized provider. [in_storage_pass] is set within a per-candidate storage sub-pass so it
     does not recursively re-store nested-loop candidates. See #134. *)
  let rec loop_proc ~process_for ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing (llc : t) : t
      =
    let loop = loop_proc ~process_for ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing in
    match llc with
    | Noop -> Noop
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Seq (c1, c2)
    (* Review round 13: dead loops are dropped at virtualization — they replay zero times, and
       descending into them could reject valid programs (a merge-tainted splice in code that never
       executes) or mint phantom parameters for identifiers only dead code renders. Aligns the
       consumer side with [inline_computation]'s spliced-body elision (round 10); stored
       computations lose their dead sub-loops at store time for the same reason. *)
    | For_loop { from_; to_; _ } when to_ < from_ -> Noop
    | Scan_loop ({ index; from_; to_; carried; body; _ } as scan_config) ->
        (* gh-ocannl-696: no candidate capture at the scan's binder. A node written in the body is
           refused wherever its store is attempted -- per statement here, through [in_scan] reaching
           [check_and_store_virtual], or at an enclosing loop's capture, whose validity walk meets
           the scan -- so inside a scan only reads inline. The inits evaluate once, outside the
           loop, in the enclosing context. *)
        let carried =
          List.map carried ~f:(fun c ->
              {
                c with
                init =
                  loop_scalar ~process_for ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing
                    c.init;
              })
        in
        Scan_loop
          {
            scan_config with
            carried;
            body =
              loop_proc ~process_for ~owned ~in_storage_pass ~guarded ~in_scan:true
                ~enclosing:((index, to_ - from_ + 1) :: enclosing)
                body;
          }
    | For_loop ({ index; body; from_; to_; _ } as for_config) -> (
        (* What an inner candidate's capture would be replaying: this loop is outside any subtree
           stored below it (gh-651 loop half). *)
        let enclosing' = (index, to_ - from_ + 1) :: enclosing in
        if in_storage_pass then
          For_loop
            {
              for_config with
              body =
                loop_proc ~process_for ~owned ~in_storage_pass:true ~guarded ~in_scan
                  ~enclosing:enclosing' body;
            }
        else
          let tns = Hashtbl.find reverse_node_map index |> Option.value ~default:[] in
          let candidates =
            (* First-seen (trace) order is preserved by [track_symbol], so a forward provider is
               stored before its consumer below. *)
            List.filter tns ~f:(fun tn ->
                (not @@ Set.mem process_for tn)
                && (not @@ Set.mem owned tn)
                && (not @@ Tn.Placements.known_non_virtual plc tn))
          in
          match candidates with
          | [] ->
              For_loop
                {
                  for_config with
                  body =
                    loop_proc ~process_for ~owned ~in_storage_pass:false ~guarded ~in_scan
                      ~enclosing:enclosing' body;
                }
          | _ ->
              let owned' = List.fold candidates ~init:owned ~f:Set.add in
              (* Phase 1 -- store, sequentially in source order. For candidate [k], its stored loop
                 is processed with [process_for] containing [k] AND every later (not-yet-stored)
                 candidate. Keeping the not-yet-stored candidates in [process_for] leaves their
                 [Get]s intact, so a sibling setter (e.g. an in-loop materialized consumer) that
                 reads a later candidate does NOT trigger [inline_computation] before that candidate
                 is stored (which would raise the stale optimize_ctx error). Earlier candidates are
                 already stored and are left OUT, so [k]'s own setter can inline them (forward
                 provider chains). [owned'] suppresses per-statement auto-store for every
                 shared-loop candidate; [in_storage_pass] stops nested re-storage.
                 [check_and_store]/[inline_computation] filter the stored body to [k]'s own setters,
                 so the irrelevant sibling setters left un-rewritten here are dropped. *)
              List.iteri candidates ~f:(fun k tn ->
                  let node : traced_array = get_node traced_store tn in
                  let store_pf = List.fold (List.drop candidates k) ~init:process_for ~f:Set.add in
                  let stored =
                    (* gh-509 task 4: vector-store producers are stored raw, see
                       [proc_contains_set_from_vec]. *)
                    if proc_contains_set_from_vec tn body then For_loop { for_config with body }
                    else
                      For_loop
                        {
                          for_config with
                          body =
                            loop_proc ~process_for:store_pf ~owned:owned' ~in_storage_pass:true
                              ~guarded ~in_scan ~enclosing:enclosing' body;
                        }
                  in
                  (* The stored subtree is rooted AT this loop, so [enclosing] (not [enclosing']) is
                     what it fails to contain. *)
                  check_and_store_virtual optim_ctx ~guarded ~in_scan ~enclosing node static_indices
                    stored);
              (* Phase 2 -- emit. Candidates are NOT in [process_for], so surviving readers
                 (materialized siblings, and later virtual siblings, all now stored) inline the
                 provider; [owned'] still suppresses candidate auto-store; each candidate setter
                 keeps its own self-references via the per-Set [next]. Candidate setters are emitted
                 intact and removed later by [cleanup_virtual_llc]. *)
              For_loop
                {
                  for_config with
                  body =
                    loop_proc ~process_for ~owned:owned' ~in_storage_pass:false ~guarded ~in_scan
                      ~enclosing:enclosing' body;
                })
    | Zero_out tn ->
        let traced : traced_array = get_node traced_store tn in
        if
          (not @@ Set.mem process_for tn)
          && (not @@ Set.mem owned tn)
          && (not @@ Tn.Placements.known_non_virtual plc traced.tn)
        then
          check_and_store_virtual optim_ctx ~guarded ~in_scan ~enclosing traced static_indices llc;
        llc
    | Set { tn; idcs; llsc; debug } ->
        let traced : traced_array = get_node traced_store tn in
        let next =
          if Tn.Placements.known_non_virtual plc traced.tn then process_for
          else Set.add process_for tn
        in
        let result =
          Set
            {
              tn;
              idcs;
              llsc =
                loop_scalar ~process_for:next ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing
                  llsc;
              debug;
            }
        in
        if
          (not @@ Set.mem process_for tn)
          && (not @@ Set.mem owned tn)
          && (not @@ Tn.Placements.known_non_virtual plc traced.tn)
        then
          check_and_store_virtual optim_ctx ~guarded ~in_scan ~enclosing traced static_indices
            result;
        result
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
        let traced : traced_array = get_node traced_store tn in
        let next =
          if Tn.Placements.known_non_virtual plc traced.tn then process_for
          else Set.add process_for tn
        in
        let result =
          Set_from_vec
            {
              tn;
              idcs;
              length;
              vec_unop;
              arg =
                ( loop_scalar ~process_for:next ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing
                    arg_scalar,
                  arg_prec );
              debug;
            }
        in
        if
          (not @@ Set.mem process_for tn)
          && (not @@ Set.mem owned tn)
          && (not @@ Tn.Placements.known_non_virtual plc traced.tn)
        then
          (* gh-509 task 4: store the raw statement (argument not rewritten), see
             [proc_contains_set_from_vec]. The emitted statement remains [result]. *)
          check_and_store_virtual optim_ctx ~guarded ~in_scan ~enclosing traced static_indices llc;
        result
    | Set_local (id, llsc) ->
        Set_local
          (id, loop_scalar ~process_for ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing llsc)
    | Declare_local _ -> llc
    | Comment _ -> llc
    | Staged_compilation _ -> llc
    | Workgroup_barrier -> llc
    (* Unreachable pre-schedule (visit tracing already rejected it); kept opaque. *)
    | Tile_mma _ -> llc
    (* gh-466: unreachable — [Set_dynamic] is produced after virtualization; kept opaque. *)
    | Set_dynamic _ -> llc
    (* gh-651: a candidate captured inside the guard would be stored without it and replayed
       unguarded at every read site. The flag rides down into [check_and_store_virtual], which
       rejects such a candidate as [Non_virtual 142] — the same verdict the walk's own [If] arm
       gives an interior guard. The CONDITION is evaluated whenever this statement is reached, so it
       inherits the enclosing flag rather than the one this [If] establishes. Lifting this
       (prepending the guard to the stored computation) is adjacent to the Block virtualizer's
       range-guard machinery. *)
    | If { cond = c, prec; body } ->
        If
          {
            cond =
              (loop_scalar ~process_for ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing c, prec);
            body =
              loop_proc ~process_for ~owned ~in_storage_pass ~guarded:true ~in_scan ~enclosing body;
          }
  and loop_scalar ~process_for ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing
      (llsc : scalar_t) : scalar_t =
    let loop = loop_scalar ~process_for ~owned ~in_storage_pass ~guarded ~in_scan ~enclosing in
    match llsc with
    | Constant _ -> llsc
    | Constant_bits _ -> llsc
    | Get (tn, _) when Set.mem process_for tn ->
        (* [Get_local] will replace this [Get] during [inline_computation] if [tn] remains
           virtual. *)
        llsc
    | Get (tn, indices) ->
        let traced = get_node traced_store tn in
        if Tn.Placements.known_non_virtual plc traced.tn then (
          (* Review round 13: [Local] is non-virtual yet does NOT persist across routines — an
             inherited node the earlier routine materialized as routine-local scratch has no buffer
             a later routine can read. Only persistent placements may pass through. *)
          if Hash_set.mem inherited_tns tn && Tn.Placements.known_not_materialized plc tn then
            raise
              (Utils.User_error
                 [%string
                   "the node %{Tn.debug_name tn} was computed as routine-local scratch by an \
                    earlier routine of this compilation lineage: its buffer does not persist, so a \
                    later routine cannot read it. Mark %{Tn.debug_name tn} as materialized (e.g. \
                    via Train.set_materialized) in the routine that computes it."]);
          llsc)
        else
          let id = get_scope tn in
          Option.value ~default:llsc
          @@ Option.map
               (inline_computation ~id ~inherited_merge_tainted ~inherited_tns optim_ctx traced
                  static_indices indices) ~f:(fun body ->
                 if Hash_set.mem inherited_tns tn then record_spliced_reads body;
                 Local_scope { id; body; orig_indices = indices; mint = Inlined_computation })
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        (* Review round 12: a dynamically-indexed read cannot be served by recomputation (the gather
           index is only known at runtime), so a LOCAL table materializes — but an INHERITED table
           has no setter left in any schedule, and letting it reach cleanup produces the cryptic
           already-virtual provenance collision. Fail actionably here. *)
        if
          Hash_set.mem inherited_tns tn
          && ((not (Tn.Placements.known_non_virtual plc tn))
             (* Round 13: [Local] passes [known_non_virtual] yet does not persist across routines —
                only a persistent materialized table can serve the gather. *)
             || Tn.Placements.known_not_materialized plc tn)
        then
          raise
            (Utils.User_error
               [%string
                 "the deferred computation of %{Tn.debug_name tn}, stored by an earlier routine of \
                  this compilation lineage, is read as a dynamic-gather table in this routine: \
                  dynamically-indexed reads cannot be served by inlining, and no routine writes a \
                  persistent buffer for the node. Mark %{Tn.debug_name tn} as materialized (e.g. \
                  via Train.set_materialized) in the routine that computes it."]);
        (* gh-ocannl-734: the LOCAL table's materialization, which the paragraph above always
           asserted but never performed. Leaving it undecided here let the node stay a
           virtualization candidate: its setter then reached [cleanup_virtual_llc]'s [Set] arm
           first, was committed [Virtual 152] and dropped as dead, and cleanup's own [Get_dynamic]
           arm hit the resulting [152 -> 17] collision. Deciding it here is what the sibling
           lane-extract gather already does for its counter ([Never_virtual 146]). *)
        if Tn.Placements.known_virtual plc tn then
          raise
            (Utils.User_error
               (virtualized_gather_table_rejection tn
                  ~decided_by:"declared virtual, or committed virtual before this read"));
        Tn.Placements.update plc tn Never_virtual 17;
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop v, prec) }
    | Local_scope opts ->
        Local_scope
          {
            opts with
            body =
              loop_proc ~process_for:(Set.add process_for opts.id.tn) ~owned ~in_storage_pass
                ~guarded ~in_scan ~enclosing opts.body;
          }
    | Get_local _ -> llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index _ -> llsc
    | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
        Ternop (op, (loop llv1, prec1), (loop llv2, prec2), (loop llv3, prec3))
    (* Review round 9: a projection's discarded operand is never evaluated — do not descend into it
       (attempting to inline a merge-tainted inherited virtual there would raise for code that never
       runs); collapse to the selected operand, mirroring [simplify_llc]'s Arg1/Arg2 arms — the
       renderers emit the selected operand alone either way. *)
    | Binop (Arg1, (llv1, _), _) -> loop llv1
    | Binop (Arg2, _, (llv2, _)) -> loop llv2
    | Binop (op, (llv1, prec1), (llv2, prec2)) -> Binop (op, (loop llv1, prec1), (loop llv2, prec2))
    | Unop (op, (llsc, prec)) -> Unop (op, (loop llsc, prec))
  in
  let result =
    loop_proc
      ~process_for:(Set.empty (module Tnode))
      ~owned:(Set.empty (module Tnode))
      ~in_storage_pass:false ~guarded:false ~in_scan:false ~enclosing:[] llc
  in
  (result, spliced_reads)

(** gh-ocannl-805: the decision-coverage seam between [virtual_llc] and cleanup. Every explicit
    tensor-buffer read that will survive cleanup must already have a placement verdict. Cleanup may
    resolve a [Never_virtual] verdict to [Local] / [On_device], and may commit setter-only
    candidates [Virtual] while dropping their stores; it must not be the first phase to decide a
    surviving read. Otherwise its source-order walk can drop an undecided node's setter as [Virtual]
    before reaching the read that asks for [Never_virtual], turning one missing virtualizer arm into
    an order-dependent provenance collision.

    The survival qualification is load-bearing. [virtual_llc] intentionally leaves a virtual
    candidate's setter in the intermediate IR, including its self-read, for cleanup to drop as a
    whole without visiting the right-hand side. This walk mirrors that keep/drop decision without
    mutating placements, and inspects a guard condition only when some statement in its body will
    survive (cleanup elides an empty guard before visiting its condition).

    [Local_scope] is not itself a tensor-buffer read -- it is the replacement for an inlined one --
    but its body may read other tensor buffers, so descend into it. [Get_merge_buffer] reads the
    transient merge buffer rather than its tnode's context buffer and therefore carries no placement
    obligation. The match is intentionally exhaustive: a new IR constructor has to state whether and
    where it can contain a surviving read. *)
let validate_virtualization_decision_coverage (plc : Tn.Placements.t) (llc : t) : unit =
  let undecided = ref (Set.empty (module Tn)) in
  let check tn =
    if Option.is_none (Tn.Placements.get plc tn) then undecided := Set.add !undecided tn
  in
  let rec proc (llc : t) : bool =
    match llc with
    | Noop -> false
    | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ -> true
    | Seq (a, b) ->
        let a_survives = proc a in
        let b_survives = proc b in
        a_survives || b_survives
    | For_loop { body; _ } -> proc body
    | Scan_loop { carried; body; _ } ->
        (* Kept whole by cleanup: the body's setters are all non-virtual and the rotation is an
           effect on the carried locals. *)
        List.iter carried ~f:(fun c -> scalar c.init);
        ignore (proc body : bool);
        true
    | Zero_out tn -> Tn.Placements.known_non_virtual plc tn
    | Set { tn; llsc; _ } ->
        let survives = Tn.Placements.known_non_virtual plc tn in
        if survives then scalar llsc;
        survives
    | Set_from_vec { tn; arg = llsc, _; _ } ->
        let survives = Tn.Placements.known_non_virtual plc tn in
        if survives then scalar llsc;
        survives
    | Set_local (_, llsc) ->
        scalar llsc;
        true
    | Set_dynamic { dyn_value = dyn, _; llsc; _ } ->
        scalar dyn;
        scalar llsc;
        true
    | If { cond = c, _; body } ->
        let survives = proc body in
        if survives then scalar c;
        survives
    | Tile_mma { fallback; _ } ->
        ignore (proc fallback : bool);
        true
  and scalar (llsc : scalar_t) : unit =
    match llsc with
    | Get (tn, _) -> check tn
    | Get_dynamic { tn; dyn_value = dyn, _; _ } ->
        check tn;
        scalar dyn
    | Local_scope { id; body; _ } ->
        if not (Tn.Placements.known_non_virtual plc id.tn) then ignore (proc body : bool)
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
    | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  in
  ignore (proc llc : bool);
  if not (Set.is_empty !undecided) then
    let names = List.map (Set.to_list !undecided) ~f:Tn.debug_name |> String.concat ~sep:", " in
    invalid_arg
      [%string
        "Low_level.virtual_llc decision-coverage seam: the post-pass code retains a tensor read \
         with no virtualization decision for %{names}. Every surviving Get / Get_dynamic must be \
         decided before cleanup."]

(** gh-ocannl-681: the scope-TARGET contract, companion to the scope-BODY contract
    {!validate_scope_bodies} enforces. A [Local_scope] over [X] denotes THE INLINED COMPUTATION OF
    [X], so it means something only while [X] is virtual. Over a materialized [X] the body is not
    what a read of [X] yields — [X]'s own setter writes the buffer — so the optimizer cannot honour
    it, and used to answer by discarding it: [Set (X, Local_scope ...)] became [Set (X, Get X)],
    silently. That is a false green waiting to happen, and it happened twice. Two
    [test/operations/accum_width.ml] legs ran kernels literally spelling [acc[0] = acc[0]] while
    claiming to pin an accumulation width — the identity copy reproduced the expected value. And a
    hand-built scope over a FRESH node collapses the same way, because a node with no setter is
    decided non-virtual: [test/operations/affine_extraction.ml]'s sibling-operand probe said in its
    own comment that its scope nodes were "fresh (virtualizable) ... so the program below also
    survives [specialize_proc]", and they did not.

    The shape IS legal on the far side of the pipeline — [Schedule]'s materializing [Unroll] and
    [Partition] mints and [C_syntax.try_localize_serial_reduce] build exactly it over a materialized
    accumulator, and codegen renders it. One IR, two meanings, with nothing saying which side of
    [optimize] a program was on. This is the statement of which side: localizing a materialized
    accumulator is codegen's accumulator peel (gh-ocannl-693), and rejecting here is what keeps that
    the ONE route. IR already in the scope form reaches a backend past the optimizer, through
    [Context.compile ?prelowered] ([Ll_test.optimize_scoped]).

    The optimizer's OWN scopes are exempt, and only they — the exemption is a retraction of its own
    decision, not of the caller's program. [virtual_llc] mints a scope at a [Get] of a node still
    virtual at that point; a later refusal can commit that node [Never_virtual], and then the
    surviving setter writes the very value the body recomputes, so rewriting back to a [Get] is
    sound. Hence the ids the pass was HANDED are threaded in: a scope in that set may not be
    rewritten away.

    gh-ocannl-704 settled that the retraction is not hypothetical. [virtual_llc] walks statements in
    SOURCE ORDER while a node's placement is a single mutable cell shared by the whole walk, so a
    refusal decided at a statement reached AFTER a read that already minted flips the node under an
    existing scope, and nothing revisits that scope in between. Both rejection families reach it:
    store time ([check_and_store_virtual]'s [Non_virtual 142] on a guarded LATER setter of a node an
    earlier statement already read) and consumption time ([inline_computation]'s [Non_virtual 13] at
    a second read the producer's index map cannot serve). Dropping the exemption would make
    [optimize] refuse IR it built itself; [test/operations/scope_over_materialized.ml] pins one
    witness of each family, with executed parity against the materialized reading. *)
let scope_target_rejection (id : scope_id) : string =
  [%string
    "the program wraps a computation of %{Tn.debug_name id.tn} in a Local_scope \
     (v%{id.scope_id#Int}), but %{Tn.debug_name id.tn} is materialized in this routine. A \
     Local_scope over a node denotes the INLINED computation of that node, which is meaningful \
     only while the node is virtual: materialized, its own setter writes the buffer, the scope \
     body is not what a read of it yields, and honouring the scope would mean discarding the body. \
     Note that a node with no setter at all is decided non-virtual, so a scope over a freshly \
     created node lands here too -- declare such a scope's node virtual. Localizing a MATERIALIZED \
     accumulator is codegen's accumulator peel (gh-ocannl-693), not the virtualizer's business: \
     hand that form to a backend past the optimizer, through the Context.compile ?prelowered seam \
     (Ll_test.optimize_scoped)."]

(** The [Local_scope] ids occurring anywhere in [llc], at any depth: the scopes the optimization was
    handed, as opposed to the ones it mints. See {!scope_target_rejection}. Ids are collected whole
    rather than by their integer, because hand-built IR can mint two locals sharing an integer while
    naming different tensor nodes. *)
let input_scope_ids (llc : t) : Set.M(Scope_id).t =
  let acc = ref (Set.empty (module Scope_id)) in
  let rec proc (llc : t) : unit =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ ->
        ()
    | Seq (a, b) ->
        proc a;
        proc b
    | For_loop { body; _ } -> proc body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> scalar c.init);
        proc body
    | If { cond = c, _; body } ->
        scalar c;
        proc body
    | Set { llsc; _ } -> scalar llsc
    | Set_local (_, llsc) -> scalar llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        scalar v;
        scalar llsc
    | Set_from_vec { arg = a, _; _ } -> scalar a
    | Tile_mma { fallback; _ } -> proc fallback
  and scalar (llsc : scalar_t) : unit =
    match llsc with
    | Local_scope { id; body; _ } ->
        acc := Set.add !acc id;
        proc body
    | Get_dynamic { dyn_value = v, _; _ } -> scalar v
    | Ternop (_, (a, _), (b, _), (d, _)) ->
        scalar a;
        scalar b;
        scalar d
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  in
  proc llc;
  !acc

let cleanup_virtual_llc plc ~input_scopes ~static_indices (llc : t) : t =
  (* The current position is within scope of the definitions of the process_for virtual arrays. *)
  let rec loop_proc ~balanced ~env_dom (llc : t) : t option =
    let loop = loop_proc ~balanced ~env_dom in
    match llc with
    | Noop -> None
    | Seq _ ->
        let body = List.filter_map ~f:loop @@ flat_lines [ llc ] in
        if List.is_empty body then None else Some (unflat_lines body)
    | For_loop ({ index; body; _ } as for_config) ->
        (* Recurse into the loop body. A shared loop may compute several tensors: the per-statement
           cases below drop (and force [Virtual]) the setters of virtual tensors and keep those of
           non-virtual tensors, so we must not drop the whole loop just because its index has a
           virtual owner. The loop is elided only when its cleaned body is empty. See #134. *)
        let env_dom = Set.add env_dom index in
        Option.map ~f:(fun body : t -> For_loop { for_config with body })
        @@ loop_proc ~balanced ~env_dom body
    | Scan_loop ({ index; carried; body; _ } as scan_config) ->
        (* gh-ocannl-696: kept whole -- every setter in the body is non-virtual ([Non_virtual 148])
           and the rotation is an effect on the carried locals; the state node is committed
           [Virtual] like a scope local's. *)
        let carried =
          List.map carried ~f:(fun c ->
              assert (not @@ Tn.Placements.known_non_virtual plc c.prev.tn);
              Tn.Placements.update plc c.prev.tn Virtual 16;
              { c with init = loop_scalar ~balanced ~env_dom c.init })
        in
        let env_dom = Set.add env_dom index in
        let body = Option.value ~default:Noop (loop_proc ~balanced ~env_dom body) in
        Some (Scan_loop { scan_config with carried; body })
    | Zero_out tn ->
        if not @@ Tn.Placements.known_non_virtual plc tn then (
          (* #296: a tnode still not [known_non_virtual] by cleanup was never forced [Never_virtual]
             during tracing/virtualization, so it has no materialized reader left -- its only uses
             were inlined into [Local_scope] bodies. We therefore commit it to [Virtual] and drop
             this now-dead initializer. Provenance 151 = dropped from the [Zero_out] cleanup arm. *)
          Tn.Placements.update plc tn Virtual 151;
          None)
        else Some llc
    | Set { tn; idcs; llsc; debug } ->
        if not @@ Tn.Placements.known_non_virtual plc tn then (
          (* #296: same default-to-[Virtual] policy as the [Zero_out] arm above -- an undecided
             tnode has no materialized reader left after inlining, so commit it [Virtual] and drop
             the store. Provenance 152 = dropped from the [Set]/[Set_from_vec] cleanup arms. *)
          Tn.Placements.update plc tn Virtual 152;
          None)
        else (
          assert (
            Array.for_all idcs ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
          Some (Set { tn; idcs; llsc = loop_scalar ~balanced ~env_dom llsc; debug }))
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
        if not @@ Tn.Placements.known_non_virtual plc tn then (
          (* #296: same default-to-[Virtual] policy as [Set]. A vector op that genuinely cannot be
             scalar-inlined was already forced [Never_virtual] (via [Non_virtual 140] in
             [inline_computation]), so reaching here means the node stayed virtual-eligible and its
             vector store is dead -- drop it. Provenance 152. *)
          Tn.Placements.update plc tn Virtual 152;
          None)
        else (
          assert (
            Array.for_all idcs ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
          Some
            (Set_from_vec
               {
                 tn;
                 idcs;
                 length;
                 vec_unop;
                 arg = (loop_scalar ~balanced ~env_dom arg_scalar, arg_prec);
                 debug;
               }))
    | Set_local (id, llsc) ->
        assert (not @@ Tn.Placements.known_non_virtual plc id.tn);
        Tn.Placements.update plc id.tn Virtual 16;
        Some (Set_local (id, loop_scalar ~balanced ~env_dom llsc))
    | Declare_local _ -> Some llc
    | Comment _ -> Some llc
    | Staged_compilation _ -> Some llc
    | Workgroup_barrier -> Some llc
    (* Unreachable pre-schedule; kept opaque. *)
    | Tile_mma _ -> Some llc
    (* gh-466: defensive — [Set_dynamic] is produced after cleanup; a scatter target is always
       materialized. Keep, recursing like the [Set] arm. *)
    | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec; llsc; debug } ->
        Tn.Placements.update plc tn Never_virtual 17;
        Some
          (Set_dynamic
             {
               tn;
               idcs;
               dyn_axis;
               dyn_value = (loop_scalar ~balanced ~env_dom v, prec);
               llsc = loop_scalar ~balanced ~env_dom llsc;
               debug;
             })
    | If { cond = c, prec; body } ->
        (* The guard is elided when its cleaned body is empty, like an empty loop. *)
        Option.map (loop_proc ~balanced ~env_dom body) ~f:(fun body : t ->
            If { cond = (loop_scalar ~balanced ~env_dom c, prec); body })
  and loop_scalar ~balanced ~env_dom (llsc : scalar_t) : scalar_t =
    let loop = loop_scalar ~balanced ~env_dom in
    match llsc with
    | Constant _ -> llsc
    | Constant_bits _ -> llsc
    | Get (a, indices) ->
        (* #296: keep [update_memory_mode] rather than [assert (Tn.known_non_virtual a)]. A [Get]
           surviving into cleaned code reads a materialized array, but its target's mode is not
           guaranteed finalized before this point: cleanup is itself the phase that commits
           surviving reads to [Never_virtual] (a node read here but only written under a virtualized
           setter is decided right now), so this update is the commitment point, not a redundant
           re-assertion. Mirrors the [Get_dynamic] arm just below. Provenance 17. *)
        Tn.Placements.update plc a Never_virtual 17;
        assert (
          Array.for_all indices ~f:(function Indexing.Iterator s -> Set.mem env_dom s | _ -> true));
        llsc
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        (* gh-343: defensive -- the table is a materialized read; recurse into the dynamic index. *)
        (* gh-ocannl-734: [virtual_llc]'s own [Get_dynamic] arm decides every table it walks, so a
           table still virtual here was never seen by that arm (a [Get_dynamic] minted after
           virtualization over a node this cleanup then virtualized). Report it rather than letting
           the update below answer with a bare provenance collision. *)
        if Tn.Placements.known_virtual plc tn then
          raise
            (Utils.User_error
               (virtualized_gather_table_rejection tn
                  ~decided_by:"its setter was inlined into the read sites and dropped as dead"));
        Tn.Placements.update plc tn Never_virtual 17;
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop v, prec) }
    | Local_scope { id; body; orig_indices; mint } ->
        assert (
          Array.for_all orig_indices ~f:(function
            | Indexing.Iterator s -> Set.mem env_dom s
            | _ -> true));
        if Tn.Placements.known_non_virtual plc id.tn then (
          (* gh-ocannl-681: the pass may retract a scope IT minted -- the node was a virtualization
             candidate at that read site and a later refusal committed it [Never_virtual], so the
             setter that now survives writes the very value the body recomputes -- but never one it
             was handed. See {!scope_target_rejection}. *)
          if Set.mem input_scopes id then
            invalid_arg ("Low_level.cleanup_virtual_llc: " ^ scope_target_rejection id);
          Get (id.tn, orig_indices))
        else
          let body = Option.value_exn ~here:[%here] @@ loop_proc ~balanced ~env_dom body in
          Tn.Placements.update plc id.tn Virtual 18;
          Local_scope { id; orig_indices; body; mint }
    | Get_local id ->
        assert (not @@ Tn.Placements.known_non_virtual plc id.tn);
        Tn.Placements.update plc id.tn Virtual 16;
        llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index (Fixed_idx _ | Sub_axis) -> llsc
    | Embed_index (Iterator s) ->
        assert (Set.mem env_dom s);
        llsc
    | Embed_index (Affine { symbols; _ }) ->
        List.iter symbols ~f:(fun (_, s) -> assert (Set.mem env_dom s));
        llsc
    | Embed_index (Concat syms) ->
        List.iter syms ~f:(fun s -> assert (Set.mem env_dom s));
        llsc
    | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
        Ternop (op, (loop llv1, prec1), (loop llv2, prec2), (loop llv3, prec3))
    | Binop (op, (llv1, prec1), (llv2, prec2)) -> Binop (op, (loop llv1, prec1), (loop llv2, prec2))
    | Unop (op, (llsc, prec)) -> Unop (op, (loop llsc, prec))
  in
  let static_indices =
    Set.of_list (module Indexing.Symbol)
    @@ List.map ~f:(fun s -> s.Indexing.static_symbol) static_indices
  in
  (* gh-611: a routine whose every statement virtualizes away is legal, not a crash — its runtime
     schedule is empty while its stored computations and placements persist in the lineage, awaiting
     consumption by later routines (the incremental flows: [Context.decide_inline], the
     [?prelowered] seam, the documented cross-routine computation sharing). *)
  Option.value ~default:Noop @@ loop_proc ~balanced:false ~env_dom:static_indices llc

let rec substitute_float ~var ~value llsc =
  let loop_scalar = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  if equal_scalar_t var llsc then value
  else
    match llsc with
    | Constant _ -> llsc
    | Constant_bits _ -> llsc
    | Get (_ptr, _indices) -> llsc
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop_scalar v, prec) }
    | Local_scope opts -> Local_scope { opts with body = loop_proc opts.body }
    | Get_local _ -> llsc
    | Get_merge_buffer (_, _) -> llsc
    | Embed_index _ -> llsc
    | Ternop (op, (llv1, prec1), (llv2, prec2), (llv3, prec3)) ->
        Ternop (op, (loop_scalar llv1, prec1), (loop_scalar llv2, prec2), (loop_scalar llv3, prec3))
    | Binop (op, (llv1, prec1), (llv2, prec2)) ->
        Binop (op, (loop_scalar llv1, prec1), (loop_scalar llv2, prec2))
    | Unop (op, (llsc, prec)) -> Unop (op, (loop_scalar llsc, prec))

and substitute_proc ~var ~value llc =
  let loop_scalar = substitute_float ~var ~value in
  let loop_proc = substitute_proc ~var ~value in
  match llc with
  | Noop -> Noop
  | Seq (c1, c2) ->
      let c1 = loop_proc c1 in
      let c2 = loop_proc c2 in
      Seq (c1, c2)
  | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
  | Scan_loop sc ->
      Scan_loop
        {
          sc with
          carried = List.map sc.carried ~f:(fun c -> { c with init = loop_scalar c.init });
          body = loop_proc sc.body;
        }
  | Zero_out _ -> llc
  | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = loop_scalar llsc; debug }
  | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, vprec; llsc; debug } ->
      Set_dynamic
        { tn; idcs; dyn_axis; dyn_value = (loop_scalar v, vprec); llsc = loop_scalar llsc; debug }
  | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
      Set_from_vec { tn; idcs; length; vec_unop; arg = (loop_scalar arg_scalar, arg_prec); debug }
  | Set_local (id, llsc) -> Set_local (id, loop_scalar llsc)
  | Declare_local _ -> llc
  | Comment _ -> llc
  | Staged_compilation _ -> llc
  | Workgroup_barrier -> llc
  | Tile_mma ({ fallback; _ } as tm) -> Tile_mma { tm with fallback = loop_proc fallback }
  | If { cond = c, prec; body } -> If { cond = (loop_scalar c, prec); body = loop_proc body }

(** {2 Interval analysis over [scalar_t]}

    Phase A of docs/proposals/interval-analysis-scalar-t.md: [interval_of] computes machine-value
    bounds of a scalar expression as consumed at a given precision, threading a total symbol
    environment (every in-scope symbol comes from a [For_loop] or a static binding) -- the abstract
    twin of the retired concrete tracer's [symbol -> int] env. Results carry the set of tensor nodes
    whose {e proposed} (unsettled) bounds candidates were consulted; any rewrite that consumes a
    result must settle those sources ({!Tnode.settle_bounds}, binding constraint 2) -- facts derived
    purely from precisions and loop extents have no sources and need no settlement. *)

type interval_result = { ival : Interval.t; srcs : Set.M(Tn).t }

let no_srcs = Set.empty (module Tn)

(* Physical-identity memo keyed per expression node ([Stdlib.Hashtbl.hash] is total, including on
   the closures inside [Tnode.t]; physically equal keys are structurally equal, so the hash is
   consistent). One table per symbol-environment scope, shared across queries within the scope
   (binding constraint 8); the same subtree can be physically shared under different loop nests with
   different intervals, so the memo must not outlive its scope. *)
module Phys_memo = Stdlib.Hashtbl.Make (struct
  type t = scalar_t

  let equal = phys_equal
  let hash = Stdlib.Hashtbl.hash
end)

type ienv = {
  sym_env : Interval.t Map.M(Indexing.Symbol).t;
  memo : (Ops.prec * interval_result) list Phys_memo.t;
}

let interval_of_symbol ienv s =
  match Map.find ienv.sym_env s with Some iv -> iv | None -> Interval.top

(* [from_ > to_] (a dead loop) yields an empty range; give the singleton [from_] -- the loop body
   never executes, so any interval is sound for its occurrences. *)
let interval_of_loop_range ~from_ ~to_ =
  if from_ <= to_ then Interval.of_int_range from_ to_ else Interval.of_int from_

let ienv_of_static_indices (static_indices : Indexing.static_symbol list) =
  let sym_env =
    List.fold static_indices
      ~init:(Map.empty (module Indexing.Symbol))
      ~f:(fun env { Indexing.static_symbol; static_range; used_as_extent; used_as_slice = _ } ->
        (* [static_range] is a declared-bounds slot ([None] = unbounded, hence top); see
           docs/proposals/signed-index-precision.md on bind-time validation. A symbolic extent
           (gh-490) is a size, bind-validated inclusively: its value ranges over [0, range]. *)
        let iv =
          match static_range with
          | Some range when range > 0 && used_as_extent -> Interval.of_int_range 0 range
          | Some range when range > 0 -> Interval.of_int_range 0 (range - 1)
          | _ -> Interval.top
        in
        Map.set env ~key:static_symbol ~data:iv)
  in
  { sym_env; memo = Phys_memo.create 64 }

let ienv_extend ienv sym ~from_ ~to_ =
  {
    sym_env = Map.set ienv.sym_env ~key:sym ~data:(interval_of_loop_range ~from_ ~to_);
    memo = Phys_memo.create 16;
  }

(** {3 Narrowing the environment from a statement guard (gh-ocannl-566)}

    An [If] condition holds on every execution that reaches its body, so it is a fact about the
    body's symbols just as a loop range is. Without it the environment only knows loop extents, and
    a guard the enclosing condition proves survives as a per-element ternary -- the pipelined
    prefetch's zero-fringe edge guard (gh-ocannl-487 phase 1), whose [k := k+1] substitution widens
    the index interval past the extent that the enclosing [If (k < to_)] excludes.

    Recognized conditions are conjunctions of integer-affine index comparisons -- the same shapes
    {!Schedule.partition_breakpoints} reads. Everything else contributes nothing. Only loop-range /
    static-index facts are consulted, never tensor-node bounds candidates, so a narrowing never
    creates a source needing settlement (binding constraint 2). A comparison contributes only when
    its machine evaluation is faithful to the mathematical integers over the incoming bounds
    (binding constraint 5): each side must survive [Interval.at_prec] unchanged at both the index
    precision and the condition's evaluation precision.

    The result is guard-relative: a statement simplified under a condition is valid only where that
    condition holds, exactly as a statement simplified under a loop range is valid only within it.
    Nothing in [optimize_proc] moves statements across an [If] afterwards
    ([hoist_cross_statement_cse] descends only into [Seq]/[For_loop] and never lifts a guarded
    body), and a schedule transform that relocates code owes the same re-derivation it already owes
    for loop extents (docs/proposals/schedule-ir-optops.md §2). *)

(* Integer-affine view of a scalar as [(coefficient, symbol) terms, constant offset]; [None] for
   anything that is not an integer index expression. *)
let affine_terms_of_scalar (sc : scalar_t) : ((int * Indexing.symbol) list * int) option =
  match sc with
  | Constant c when Float.is_integer c && Float.(abs c < Interval.exact_int_cutoff) ->
      Some ([], Float.to_int c)
  | Embed_index (Indexing.Fixed_idx i) -> Some ([], i)
  | Embed_index Indexing.Sub_axis -> Some ([], 0)
  | Embed_index (Indexing.Iterator s) -> Some ([ (1, s) ], 0)
  | Embed_index (Indexing.Affine { symbols; offset }) ->
      Some (Indexing.coalesce_affine_terms symbols, offset)
  | _ -> None

(* Exact integer bounds of a symbol under [sym_env]; [None] when unbounded or inexact (loop ranges
   and bounded static indices always qualify). *)
let sym_int_bounds sym_env s =
  match Map.find sym_env s with
  | Some { Interval.integral = true; exact = true; lo; hi }
    when Float.is_finite lo && Float.is_finite hi ->
      Some (Float.to_int lo, Float.to_int hi)
  | _ -> None

(* Floor resp. ceiling division, for a nonzero divisor of either sign ([/] truncates towards zero
   and [rem] takes the dividend's sign). *)
let fdiv_int a b =
  let q = a / b and r = Int.rem a b in
  if r <> 0 && Bool.( <> ) (r > 0) (b > 0) then q - 1 else q

let cdiv_int a b =
  let q = a / b and r = Int.rem a b in
  if r <> 0 && Bool.equal (r > 0) (b > 0) then q + 1 else q

(* Narrows [sym_env] by the integer constraint [Σ terms + offset <= 0] (with [terms] coalesced, so
   each symbol occurs once): rewriting it as [k*s <= -rest] for one term at a time, the minimum of
   the remaining terms bounds [s] from above (positive [k]) or below (negative [k]). Every symbol is
   narrowed against the {e incoming} bounds, so the result is a conjunction of independently sound
   facts. Requires every symbol of the constraint to be bounded -- an unbounded one leaves the
   remainder unbounded too, and the constraint yields nothing. *)
let narrow_sym_env_le sym_env ~terms ~offset =
  match
    List.map terms ~f:(fun (c, s) -> Option.map (sym_int_bounds sym_env s) ~f:(fun b -> (c, s, b)))
    |> Option.all
  with
  | None -> sym_env
  | Some bounded ->
      List.fold bounded ~init:sym_env ~f:(fun env (k, s, (lo, hi)) ->
          (* [rest] is minimized by taking each other term's extreme in the direction of its
             sign. *)
          let rest_lo =
            List.fold bounded ~init:offset ~f:(fun acc (c, s', (lo', hi')) ->
                if Indexing.equal_symbol s s' then acc
                else acc + if c >= 0 then c * lo' else c * hi')
          in
          let bound = -rest_lo in
          let lo, hi =
            if k > 0 then (lo, min hi (fdiv_int bound k)) else (max lo (cdiv_int bound k), hi)
          in
          (* An empty range means the guard is unsatisfiable and the body dead; any environment is
             sound there, so keep the incoming one rather than fabricating an empty interval. *)
          if lo > hi then env else Map.set env ~key:s ~data:(Interval.of_int_range lo hi))

let ienv_narrow_from_cond ienv ~(cprec : Ops.prec) (cond : scalar_t) : ienv =
  (* The machine evaluates each comparison side at the index precision (the [Embed_index] boundary,
     cf. [interval_of]) and converts it to [cprec], the condition's evaluation precision. The
     comparison outcome is the mathematical-integer fact read off below only when both steps are
     faithful over the side's whole range under the incoming bounds -- e.g. a single-precision guard
     [k <= 2^24] is also true at [k = 2^24 + 1], which rounds down. Integer wrap is modular, so an
     in-range final value suffices; float rounding is not, so [Interval.at_prec] widening to top
     declines the narrowing. (Both checks are conservative for a pure-constant side, which skips the
     index-precision step; declining is always sound.) *)
  let side_faithful sym_env (terms, offset) =
    let range =
      List.fold terms
        ~init:(Some (offset, offset))
        ~f:(fun acc (c, s) ->
          match (acc, sym_int_bounds sym_env s) with
          | Some (lo, hi), Some (slo, shi) ->
              let a = c * slo and b = c * shi in
              Some (lo + min a b, hi + max a b)
          | _ -> None)
    in
    match range with
    | None -> false
    | Some (lo, hi) ->
        let iv = Interval.of_int_range lo hi in
        let faithful prec = Interval.equal (Interval.at_prec prec iv) iv in
        faithful (Ops.index_prec ()) && faithful cprec
  in
  let le sym_env a b ~shift =
    (* On integers [a < b] is [a - b + 1 <= 0] and [a <= b] is [a - b <= 0]. *)
    match Option.both (affine_terms_of_scalar a) (affine_terms_of_scalar b) with
    | Some ((ta, oa), (tb, ob)) when side_faithful sym_env (ta, oa) && side_faithful sym_env (tb, ob)
      ->
        let terms = Indexing.coalesce_affine_terms (ta @ List.map tb ~f:(fun (c, s) -> (-c, s))) in
        narrow_sym_env_le sym_env ~terms ~offset:(oa - ob + shift)
    | _ -> sym_env
  in
  let rec narrow sym_env (sc : scalar_t) =
    match sc with
    (* Both conjuncts hold in the body; a disjunction implies neither. *)
    | Binop (Ops.And, (a, _), (b, _)) -> narrow (narrow sym_env a) b
    | Binop (Ops.Cmplt, (a, _), (b, _)) -> le sym_env a b ~shift:1
    | Binop (Ops.Cmple, (a, _), (b, _)) -> le sym_env a b ~shift:0
    | Binop (Ops.Cmpeq, (a, _), (b, _)) -> le (le sym_env a b ~shift:0) b a ~shift:0
    | _ -> sym_env
  in
  let sym_env = narrow ienv.sym_env cond in
  (* The memo is scoped to a symbol environment (binding constraint 8): keep it only when nothing
     narrowed. *)
  if Map.equal Interval.equal sym_env ienv.sym_env then ienv
  else { sym_env; memo = Phys_memo.create 16 }

(* Exact integral intervals of index expressions; machine validity (the unsigned index precision,
   binding constraint 7's crosses-zero widening) is applied by [interval_of] via [Interval.at_prec
   (Ops.index_prec ())] at the [Embed_index] boundary. *)
let interval_of_index ienv (idx : Indexing.axis_index) : Interval.t =
  match idx with
  | Indexing.Fixed_idx i -> Interval.of_int i
  | Iterator s -> interval_of_symbol ienv s
  | Sub_axis -> Interval.of_int 0
  | Affine { symbols; offset } ->
      List.fold symbols ~init:(Interval.of_int offset) ~f:(fun acc (coeff, s) ->
          Interval.add acc (Interval.mul (Interval.of_int coeff) (interval_of_symbol ienv s)))
  | Concat _ -> Interval.top (* Eliminated during lowering; conservative if ever reached. *)

(* Bounds of a tensor-node read: the machine range of the stored precision, narrowed by the node's
   bounds candidate when one exists. The node becomes a source only when the candidate actually
   narrows -- folds justified by the dtype range alone are static facts requiring no settlement. *)
let interval_of_node ~prec (tn : Tn.t) : interval_result =
  let stored_prec = Lazy.force tn.Tn.storage_prec in
  let dtype = Interval.dtype_range stored_prec in
  let ival, srcs =
    match Tn.bounds_candidate tn with
    | Some c when not (Interval.is_top c) ->
        let narrowed = Interval.inter c dtype in
        if Interval.equal narrowed dtype then (dtype, no_srcs)
        else (narrowed, Set.singleton (module Tn) tn)
    | _ -> (dtype, no_srcs)
  in
  { ival = Interval.at_prec prec ival; srcs }

(** [interval_of ~ienv ~prec llsc] bounds the machine value of [llsc] evaluated at (converted into)
    precision [prec] -- matching [C_syntax.pp_scalar]'s convention that homogeneous operations
    evaluate their arguments at the consumer's precision, while [Where] conditions keep their
    annotation precision. Every rule pipes its real-arithmetic result through
    [Interval.at_prec prec], accounting for wrapping/rounding of the precision actually computed in
    (binding constraint 5). Currently-unhandled operations return top explicitly (exhaustive match:
    adding an op to [Ops] forces a conscious interval-rule decision). *)
let rec interval_of ~ienv ~(prec : Ops.prec) (llsc : scalar_t) : interval_result =
  let cached =
    match Phys_memo.find_opt ienv.memo llsc with
    | Some entries -> List.Assoc.find entries ~equal:Ops.equal_prec prec
    | None -> None
  in
  match cached with
  | Some r -> r
  | None ->
      let r = interval_of_uncached ~ienv ~prec llsc in
      let entries = Option.value (Phys_memo.find_opt ienv.memo llsc) ~default:[] in
      Phys_memo.replace ienv.memo llsc ((prec, r) :: entries);
      r

and interval_of_uncached ~ienv ~(prec : Ops.prec) (llsc : scalar_t) : interval_result =
  let at iv = Interval.at_prec prec iv in
  let pure iv = { ival = at iv; srcs = no_srcs } in
  let bool_undecided = { ival = Interval.bool_range; srcs = no_srcs } in
  let decided ~srcs b = { ival = (if b then Interval.true_ else Interval.false_); srcs } in
  match llsc with
  | Constant c -> pure (Interval.point c)
  | Constant_bits _ -> pure Interval.top
  | Embed_index idx ->
      (* The symbol crosses from the index world into the value world: computed at the index
         precision, then converted to the consumer's precision. *)
      pure (Interval.at_prec (Ops.index_prec ()) (interval_of_index ienv idx))
  | Get (tn, _) | Get_merge_buffer (tn, _) | Get_dynamic { tn; _ } -> interval_of_node ~prec tn
  | Get_local _ | Local_scope _ -> pure Interval.top (* Phase A: locals are unanalyzed. *)
  | Binop (Arg1, (v1, _), _) -> interval_of ~ienv ~prec v1
  | Binop (Arg2, _, (v2, _)) -> interval_of ~ienv ~prec v2
  | Binop (op, (v1, _), (v2, _)) -> (
      let r1 () = interval_of ~ienv ~prec v1 in
      let r2 () = interval_of ~ienv ~prec v2 in
      let lift2 rule =
        let a = r1 () and b = r2 () in
        { ival = at (rule a.ival b.ival); srcs = Set.union a.srcs b.srcs }
      in
      let cmp decides =
        let a = r1 () and b = r2 () in
        match decides a.ival b.ival with
        | Some v -> decided ~srcs:(Set.union a.srcs b.srcs) v
        | None -> bool_undecided
      in
      match op with
      | Ops.Arg1 | Ops.Arg2 -> assert false
      | Ops.Add -> lift2 Interval.add
      | Ops.Sub -> lift2 Interval.sub
      | Ops.Mul -> lift2 Interval.mul
      | Ops.Div -> lift2 Interval.div
      | Ops.Mod -> lift2 Interval.mod_
      | Ops.Max -> lift2 Interval.max_
      | Ops.Min -> lift2 Interval.min_
      | Ops.Relu_gate | Ops.Satur01_gate ->
          (* Hull of [0] (gate shut, also the NaN-condition outcome) and the gated argument. *)
          let b = r2 () in
          { ival = at (Interval.join Interval.false_ b.ival); srcs = b.srcs }
      | Ops.Cmplt -> cmp Interval.cmplt_decides
      | Ops.Cmple -> cmp Interval.cmple_decides
      | Ops.Cmpeq -> cmp Interval.cmpeq_decides
      | Ops.Cmpne -> cmp (fun a b -> Option.map (Interval.cmpeq_decides a b) ~f:not)
      | Ops.And ->
          let a = r1 () and b = r2 () in
          if Interval.definitely_false a.ival then decided ~srcs:a.srcs false
          else if Interval.definitely_false b.ival then decided ~srcs:b.srcs false
          else if Interval.definitely_true a.ival && Interval.definitely_true b.ival then
            decided ~srcs:(Set.union a.srcs b.srcs) true
          else bool_undecided
      | Ops.Or ->
          let a = r1 () and b = r2 () in
          if Interval.definitely_true a.ival then decided ~srcs:a.srcs true
          else if Interval.definitely_true b.ival then decided ~srcs:b.srcs true
          else if Interval.definitely_false a.ival && Interval.definitely_false b.ival then
            decided ~srcs:(Set.union a.srcs b.srcs) false
          else bool_undecided
      | Ops.ToPowOf | Ops.Threefry4x32_crypto | Ops.Threefry4x32_light
      | Ops.Uint4x32_to_prec_uniform_lane ->
          pure Interval.top)
  | Ternop (op, ((v1, p1) as _a1), (v2, _), (v3, _)) -> (
      match op with
      | Ops.Where ->
          (* Heterogeneous: the condition is evaluated at its annotation precision. Branch selection
             propagates the condition's sources (binding constraint 2). *)
          let c = interval_of ~ienv ~prec:p1 v1 in
          if Interval.definitely_true c.ival then
            let t = interval_of ~ienv ~prec v2 in
            { t with srcs = Set.union c.srcs t.srcs }
          else if Interval.definitely_false c.ival then
            let e = interval_of ~ienv ~prec v3 in
            { e with srcs = Set.union c.srcs e.srcs }
          else
            let t = interval_of ~ienv ~prec v2 and e = interval_of ~ienv ~prec v3 in
            { ival = at (Interval.join t.ival e.ival); srcs = Set.union t.srcs e.srcs }
      | Ops.FMA ->
          let a = interval_of ~ienv ~prec v1
          and b = interval_of ~ienv ~prec v2
          and c = interval_of ~ienv ~prec v3 in
          {
            ival = at (Interval.add (Interval.mul a.ival b.ival) c.ival);
            srcs = Set.union a.srcs (Set.union b.srcs c.srcs);
          }
      | Ops.Mul3 ->
          let a = interval_of ~ienv ~prec v1
          and b = interval_of ~ienv ~prec v2
          and c = interval_of ~ienv ~prec v3 in
          {
            ival = at (Interval.mul (Interval.mul a.ival b.ival) c.ival);
            srcs = Set.union a.srcs (Set.union b.srcs c.srcs);
          })
  | Unop (op, (v, _)) -> (
      let r () = interval_of ~ienv ~prec v in
      let lift1 rule =
        let a = r () in
        { ival = at (rule a.ival); srcs = a.srcs }
      in
      match op with
      | Ops.Identity -> interval_of ~ienv ~prec v
      | Ops.Relu -> lift1 Interval.relu
      | Ops.Satur01 -> lift1 Interval.satur01
      | Ops.Neg -> lift1 Interval.neg
      | Ops.Trunc -> lift1 Interval.trunc
      | Ops.Exp | Ops.Exp2 -> lift1 Interval.exp_like
      | Ops.Sin | Ops.Cos | Ops.Tanh_approx -> lift1 Interval.abs_le_1
      | Ops.Sqrt -> lift1 Interval.sqrt_
      | Ops.Not ->
          let a = r () in
          if Interval.definitely_false a.ival then decided ~srcs:a.srcs true
          else if Interval.definitely_true a.ival then decided ~srcs:a.srcs false
          else bool_undecided
      | Ops.Log | Ops.Log2 | Ops.Recip | Ops.Recip_sqrt | Ops.Uint4x32_to_prec_uniform1 ->
          pure Interval.top)

(** Settles the bounds of every source node consumed by a fold decision (binding constraint 2:
    settlement is transitive and fires at every consumption point). *)
let settle_srcs { srcs; _ } = Set.iter srcs ~f:Tn.settle_bounds

(* Interval-driven comparison folding for [simplify_llc]: when the interval of a comparison /
   logical connective is decided, replace it by the constant and settle the consumed bounds. Only
   these [0,1]-valued operations are folded -- their decided intervals are exact points by
   construction. *)
let try_interval_fold ~ienv ~prec (llsc : scalar_t) : scalar_t option =
  match llsc with
  | Binop ((Ops.Cmplt | Ops.Cmple | Ops.Cmpeq | Ops.Cmpne | Ops.And | Ops.Or), _, _) ->
      let r = interval_of ~ienv ~prec llsc in
      if Interval.is_singleton r.ival then (
        settle_srcs r;
        Some (Constant r.ival.Interval.lo))
      else None
  | _ -> None

let simplify_llc static_indices llc =
  (* Implements top-down rewriting. The interval environment [ienv] tracks every in-scope symbol's
     bounds (seeded from the static indices, extended per [For_loop]) for the interval-driven
     comparison folds. *)
  let rec loop_proc ~ienv (llc : t) : t =
    let loop = loop_proc ~ienv in
    let loop_scalar = loop_scalar ~ienv in
    match llc with
    | Noop -> Noop
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Seq (c1, c2)
    | For_loop for_config ->
        For_loop
          {
            for_config with
            body =
              loop_proc
                ~ienv:
                  (ienv_extend ienv for_config.index ~from_:for_config.from_ ~to_:for_config.to_)
                for_config.body;
          }
    | Scan_loop ({ index; from_; to_; carried; body; _ } as scan_config) ->
        Scan_loop
          {
            scan_config with
            carried =
              List.map carried ~f:(fun c ->
                  { c with init = fst (loop_scalar (c.init, Lazy.force c.prev.tn.Tn.storage_prec)) });
            body = loop_proc ~ienv:(ienv_extend ienv index ~from_ ~to_) body;
          }
    | Zero_out _ -> llc
    | Set { tn; idcs; llsc; debug } ->
        Set { tn; idcs; llsc = fst (loop_scalar (llsc, Lazy.force tn.Tn.storage_prec)); debug }
    | Set_dynamic { tn; idcs; dyn_axis; dyn_value; llsc; debug } ->
        (* gh-466: reached via [Schedule.apply]'s simplify of post-rewrite code. The scatter itself
           is never folded; its index value and RHS are. *)
        Set_dynamic
          {
            tn;
            idcs;
            dyn_axis;
            dyn_value = loop_scalar dyn_value;
            llsc = fst (loop_scalar (llsc, Lazy.force tn.Tn.storage_prec));
            debug;
          }
    | Set_from_vec { tn; idcs; length; vec_unop; arg; debug } ->
        Set_from_vec { tn; idcs; length; vec_unop; arg = loop_scalar arg; debug }
    | Set_local (id, llsc) ->
        Set_local (id, fst (loop_scalar (llsc, Lazy.force id.tn.Tn.storage_prec)))
    | Declare_local _ -> llc
    | Comment _ -> llc
    | Staged_compilation _ -> llc
    | Workgroup_barrier -> llc
    (* The base indices and block extents are already static; only the scalar fallback benefits from
       simplification. The statement itself is never folded. *)
    | Tile_mma ({ fallback; _ } as tm) -> Tile_mma { tm with fallback = loop fallback }
    | If { cond = c, cprec; body } -> (
        (* Construct-then-fold (axis-types proposal §2): a guard whose condition the interval
           environment decides is erased -- provably true folds to the body, provably false to
           [Noop]. The comparison folds happen inside [loop_scalar]. *)
        let c', _ = loop_scalar (c, cprec) in
        match c' with
        | Constant f when Float.(f = 0.) -> Noop
        | Constant _ -> loop body
        | _ ->
            (* gh-ocannl-566: a surviving guard is a fact about its body -- narrow the environment
               with it, so the interval queries inside can discharge what only it proves. *)
            If
              {
                cond = (c', cprec);
                body = loop_proc ~ienv:(ienv_narrow_from_cond ienv ~cprec c') body;
              })
  and loop_scalar ~ienv ((llsc, prec) : scalar_t * Ops.prec) : scalar_t * Ops.prec =
    let loop_scalar = loop_scalar ~ienv in
    let loop_proc = loop_proc ~ienv in
    let local_scope_body, llsc' =
      match llsc with
      | Local_scope opts ->
          ( opts.body,
            Local_scope
              {
                opts with
                body =
                  unflat_lines
                  @@ List.filter ~f:(function Comment _ -> false | _ -> true)
                  @@ flat_lines [ opts.body ];
              } )
      | _ -> (Noop, llsc)
    in
    match llsc' with
    | Constant _ -> (llsc, prec)
    | Constant_bits _ -> (llsc, prec)
    | Get (tn, _indices) -> (llsc, Lazy.force tn.Tn.storage_prec)
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, vprec } ->
        (* gh-343: defensive -- simplify runs before the one-hot rewrite, so this is unreachable in
           practice; still simplify the dynamic index sub-expression and never fold to a
           constant. *)
        let v', vprec' = loop_scalar (v, vprec) in
        (Get_dynamic { tn; idcs; dyn_axis; dyn_value = (v', vprec') }, Lazy.force tn.Tn.storage_prec)
    (* Collapsing a single-assignment scope into the enclosing expression demotes its hoisted reads
       to in-expression reads, i.e. moves them relative to any sibling scope body's effects. That is
       unconditionally sound under scope purity (gh-ocannl-584, {!validate_scope_bodies}): sibling
       bodies touch only their own locals, so no read can observe where it was placed. *)
    | Local_scope { id; body = Set_local (id2, v); _ } when equal_scope_id id id2 ->
        ignore (Lazy.force id.tn.Tn.dims);
        loop_scalar (v, Lazy.force id.tn.Tn.storage_prec)
    | Local_scope { id; body = Seq (Set_local (id1, v1), Set_local (id2, v2)); _ }
      when equal_scope_id id id1 && equal_scope_id id id2 ->
        ignore (Lazy.force id.tn.Tn.dims);
        let result = substitute_float ~var:(Get_local id) ~value:v1 v2 in
        loop_scalar (result, Lazy.force id.tn.Tn.storage_prec)
    | Local_scope opts ->
        ( Local_scope { opts with body = loop_proc local_scope_body },
          Lazy.force opts.id.tn.Tn.storage_prec )
    | Get_local id -> (llsc, Lazy.force id.tn.Tn.storage_prec)
    | Get_merge_buffer (tn, _) -> (llsc, Lazy.force tn.Tn.storage_prec)
    | Embed_index (Fixed_idx i) -> (Constant (Float.of_int i), prec)
    | Embed_index Sub_axis -> (Constant 0., prec)
    | Embed_index (Iterator _) -> (llsc, prec)
    | Embed_index (Affine _) -> (llsc, prec) (* Cannot simplify affine expressions to constants *)
    | Embed_index (Concat _) -> (llsc, prec) (* Cannot simplify concat to constants *)
    | Binop (Arg1, (llv1, prec1), _) -> loop_scalar (llv1, prec1)
    | Binop (Arg2, _, (llv2, prec2)) -> loop_scalar (llv2, prec2)
    | Binop ((Threefry4x32_crypto | Threefry4x32_light | Uint4x32_to_prec_uniform_lane), _, _) ->
        (llsc, prec)
    | Binop (op, (Constant c1, prec1), (Constant c2, prec2)) ->
        (Constant (Ops.interpret_binop op c1 c2), Ops.promote_prec prec1 prec2)
    | Binop (Add, (llsc, prec1), (Constant 0., _))
    | Binop (Sub, (llsc, prec1), (Constant 0., _))
    | Binop (Add, (Constant 0., _), (llsc, prec1)) ->
        loop_scalar (llsc, prec1)
    | Binop (Sub, (Constant 0., _), (llsc, prec1)) ->
        loop_scalar (Binop (Mul, (Constant (-1.), prec1), (llsc, prec1)), prec1)
    | Binop (Mul, (llsc, prec1), (Constant 1., _))
    | Binop (Div, (llsc, prec1), (Constant 1., _))
    | Binop (Mul, (Constant 1., _), (llsc, prec1)) ->
        loop_scalar (llsc, prec1)
    | Binop (Mul, (_, prec1), (Constant 0., _))
    | Binop (Div, (Constant 0., _), (_, prec1))
    | Binop (Mul, (Constant 0., _), (_, prec1)) ->
        (Constant 0., prec1)
    | Binop
        ( Add,
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) )
    | Binop
        ( Add,
          (Constant c1, prec1),
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ) ) ->
        loop_scalar (Binop (Add, (Constant (c1 +. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop
        ( Sub,
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) ) ->
        loop_scalar (Binop (Add, (Constant (c2 -. c1), Ops.promote_prec prec2 prec1), llsc), prec3)
    | Binop
        ( Sub,
          (Constant c1, prec1),
          ( Binop (Add, (Constant c2, prec2), llsc), prec3
          | Binop (Add, llsc, (Constant c2, prec2)), prec3 ) ) ->
        loop_scalar (Binop (Add, (Constant (c1 -. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop (Add, llv1, (Binop (Sub, llv2, llv3), prec3))
    | Binop (Add, (Binop (Sub, llv2, llv3), prec3), llv1) ->
        loop_scalar (Binop (Sub, (Binop (Add, llv1, llv2), prec), llv3), prec3)
    | Binop (Sub, llv1, (Binop (Sub, llv2, llv3), prec3)) ->
        loop_scalar (Binop (Sub, (Binop (Add, llv1, llv3), prec), llv2), prec3)
    | Binop (Sub, (Binop (Sub, llv1, llv2), prec1), llv3) ->
        loop_scalar (Binop (Sub, llv1, (Binop (Add, llv2, llv3), prec1)), prec1)
    | Binop
        ( Mul,
          ( Binop (Mul, (Constant c2, prec2), llsc), prec3
          | Binop (Mul, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) )
    | Binop
        ( Mul,
          (Constant c1, prec1),
          ( Binop (Mul, (Constant c2, prec2), llsc), prec3
          | Binop (Mul, llsc, (Constant c2, prec2)), prec3 ) ) ->
        loop_scalar (Binop (Mul, (Constant (c1 *. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop
        ( Div,
          ( Binop (Mul, (Constant c2, prec2), llsc), prec3
          | Binop (Mul, llsc, (Constant c2, prec2)), prec3 ),
          (Constant c1, prec1) )
      when Ops.is_float prec ->
        loop_scalar (Binop (Mul, (Constant (c2 /. c1), Ops.promote_prec prec2 prec1), llsc), prec3)
    | Binop (Div, (Constant c1, prec1), (Binop (Mul, (Constant c2, prec2), llsc), prec3))
    | Binop (Div, (Constant c1, prec1), (Binop (Mul, llsc, (Constant c2, prec2)), prec3))
      when Ops.is_float prec ->
        (* TODO: this might worsen the conditioning in hand-designed formula cases. *)
        loop_scalar (Binop (Div, (Constant (c1 /. c2), Ops.promote_prec prec1 prec2), llsc), prec3)
    | Binop (Mul, llv1, (Binop (Div, llv2, llv3), prec23))
    | Binop (Mul, (Binop (Div, llv2, llv3), prec23), llv1)
      when Ops.is_float prec ->
        loop_scalar (Binop (Div, (Binop (Mul, llv1, llv2), prec), llv3), prec23)
    | Binop (Div, llv1, (Binop (Div, llv2, llv3), prec23)) when Ops.is_float prec ->
        loop_scalar (Binop (Div, (Binop (Mul, llv1, llv3), prec), llv2), prec23)
    | Binop (Div, (Binop (Div, llv1, llv2), prec12), llv3) when Ops.is_float prec ->
        (* (a / b) / c = a / (b * c). *)
        loop_scalar (Binop (Div, llv1, (Binop (Mul, llv2, llv3), prec)), prec12)
    | Binop (ToPowOf, llv1, llv2) -> (
        let ((v1_scalar, _) as v1) = loop_scalar llv1 in
        let v2 = loop_scalar llv2 in
        let result = (Binop (ToPowOf, v1, v2), prec) in
        if not !optimize_integer_pow then result
        else
          match v2 with
          | Constant c, _ when Float.is_integer c ->
              loop_scalar (unroll_pow ~base:v1_scalar ~exp:(Float.to_int c), prec)
          | _ -> result)
    | Binop (Add, (Binop (Mul, llv1, llv2), prec12), llv3)
    | Binop (Add, llv3, (Binop (Mul, llv1, llv2), prec12))
      when Ops.is_float prec ->
        (* TODO: this is tentative. *)
        loop_scalar @@ (Ternop (FMA, llv1, llv2, llv3), Ops.promote_prec prec12 prec)
    | Binop (op, llv1, llv2) ->
        let v1 = loop_scalar llv1 in
        let v2 = loop_scalar llv2 in
        let result = (Binop (op, v1, v2), prec) in
        if equal_scalar_arg llv1 v1 && equal_scalar_arg llv2 v2 then
          (* At the rewriting fixpoint, try the interval-driven comparison fold. *)
          match try_interval_fold ~ienv ~prec (fst result) with
          | Some c -> (c, prec)
          | None -> result
        else loop_scalar result
    | Ternop (Where, (Binop (Cmpeq, (Embed_index a, _), (Embed_index b, _)), _), then_, _)
      when Indexing.equal_axis_index a b ->
        (* gh-133 Stage A: a repeated-symbol equality guard whose two embedded indices are
           syntactically identical is always taken; fold it to its then-branch. *)
        loop_scalar then_
    | Ternop (op, llv1, llv2, llv3) ->
        let v1 = loop_scalar llv1 in
        let v2 = loop_scalar llv2 in
        let v3 = loop_scalar llv3 in
        let result = (Ternop (op, v1, v2, v3), prec) in
        if equal_scalar_arg llv1 v1 && equal_scalar_arg llv2 v2 then
          match op with
          | Ops.Where ->
              (* Interval-decided condition: fold to the taken branch, settling any tensor-node
                 bounds the decision consumed (binding constraint 2). *)
              let c = interval_of ~ienv ~prec:(snd v1) (fst v1) in
              if Interval.definitely_true c.ival then (
                settle_srcs c;
                loop_scalar v2)
              else if Interval.definitely_false c.ival then (
                settle_srcs c;
                loop_scalar v3)
              else result
          | _ -> result
        else loop_scalar result
    | Unop (Identity, llsc) -> loop_scalar llsc
    | Unop (op, (Constant c, _)) -> (Constant (Ops.interpret_unop op c), prec)
    | Unop (op, llsc) ->
        let v = loop_scalar llsc in
        let result = (Unop (op, v), prec) in
        if equal_scalar_arg llsc v then result else loop_scalar result
  in
  let check_constant tn c =
    (* Prevent triggering over-eager guard against forcing precision. *)
    ignore (Lazy.force tn.Tn.dims);
    if Ops.exceeds_fp16_cutoff c && Ops.is_up_to_fp16 (Lazy.force tn.Tn.storage_prec) then
      raise
      @@ Utils.User_error
           ("Constant " ^ Float.to_string c
          ^ " is too big for FP16 aka. half precision, risk of overflow; increase precision of \
             tensor node " ^ Tn.debug_name tn)
  in
  let rec check_proc llc =
    let loop = check_proc in
    match llc with
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { body; _ } -> loop body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> check_float c.prev.tn c.init);
        loop body
    | Zero_out _ -> ()
    | Set { tn; llsc; _ } -> check_float tn llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        check_float tn v;
        check_float tn llsc
    | Set_from_vec { tn; arg = arg_scalar, _; _ } -> check_float tn arg_scalar
    | Set_local (id, llsc) -> check_float id.tn llsc
    | If { body; _ } -> loop body
    | Tile_mma { fallback; _ } -> loop fallback
    | Declare_local _ | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier -> ()
  and check_float tn llsc =
    let loop = check_float tn in
    match llsc with
    | Constant c -> check_constant tn c
    | Constant_bits _ -> () (* No check needed for bit constants *)
    | Local_scope { body; _ } -> check_proc body
    | Ternop (_, (v1, _), (v2, _), (v3, _)) ->
        loop v1;
        loop v2;
        loop v3
    | Binop (_, (v1, _), (v2, _)) ->
        loop v1;
        loop v2
    | Unop (_, (v, _)) -> loop v
    | Embed_index (Indexing.Fixed_idx i) -> check_constant tn (Float.of_int i)
    | Get_dynamic { dyn_value = v, _; _ } -> loop v
    | Embed_index _ | Get_local _ | Get_merge_buffer (_, _) | Get (_, _) -> ()
  in
  let result = loop_proc ~ienv:(ienv_of_static_indices static_indices) llc in
  if Option.is_some Utils.settings.check_half_prec_constants_cutoff then check_proc result;
  result

(** Alpha-equivalence comparison for CSE: compare two [scalar_t] trees ignoring concrete [scope_id]
    integers and fresh iterator symbols BOUND WITHIN the compared trees, but verifying
    cross-reference consistency via renaming maps. Symbols and scope ids that are free in the
    compared trees (e.g. enclosing loop iterators) must be exactly equal: renaming them would
    conflate distinct runtime values -- the backward stale-local bug (a nested recomputation of a
    virtual node at inner loop indices was judged equal to the enclosing iteration's recomputation
    and replaced by a [Get_local] of the outer, stale local). *)
let cse_equal_scalar s1 s2 =
  (* The renaming maps must be partial bijections, not just functions: alpha-equivalence requires an
     injective correspondence between bound variables. We therefore keep a reverse map alongside
     each forward map and reject when a target is already claimed by a different source (Bug 1: a
     forward-only map judged [t[i;j]] equal to [t[i;i]]). Renamings are registered ONLY at binder
     sites ([For_loop] indices; [Local_scope] / [Declare_local] scope ids); use sites look the
     mapping up and fall back to requiring exact equality for unmapped (free) names (Bug 2 -- the
     stale-local bug above). The maps are persistent for the whole comparison (no scope push/pop on
     entering binders): [Indexing.get_symbol] and [get_scope] are global counters, so symbols and
     scope ids are globally unique and no binder shadows another within a single tree. If the IR
     ever starts reusing symbol/scope ids, this assumption breaks and the maps would need
     scoping. *)
  let scope_renaming = Hashtbl.create (module Int) in
  let scope_renaming_rev = Hashtbl.create (module Int) in
  let sym_renaming = Hashtbl.create (module Indexing.Symbol) in
  let sym_renaming_rev = Hashtbl.create (module Indexing.Symbol) in
  let ids_bind (id1 : scope_id) (id2 : scope_id) =
    Tn.equal id1.tn id2.tn
    &&
    match
      (Hashtbl.find scope_renaming id1.scope_id, Hashtbl.find scope_renaming_rev id2.scope_id)
    with
    | Some mapped, _ -> Int.equal mapped id2.scope_id
    | None, Some _ -> false (* id2 already claimed by a different source scope id *)
    | None, None ->
        Hashtbl.set scope_renaming ~key:id1.scope_id ~data:id2.scope_id;
        Hashtbl.set scope_renaming_rev ~key:id2.scope_id ~data:id1.scope_id;
        true
  in
  let ids_equal (id1 : scope_id) (id2 : scope_id) =
    Tn.equal id1.tn id2.tn
    &&
    match Hashtbl.find scope_renaming id1.scope_id with
    | Some mapped -> Int.equal mapped id2.scope_id
    | None ->
        (* Free scope id: no renaming -- requires the identical id, which must not be claimed as a
           bound id of the other tree. *)
        (not (Hashtbl.mem scope_renaming_rev id2.scope_id)) && Int.equal id1.scope_id id2.scope_id
  in
  let sym_bind (s1 : Indexing.symbol) (s2 : Indexing.symbol) =
    match (Hashtbl.find sym_renaming s1, Hashtbl.find sym_renaming_rev s2) with
    | Some mapped, _ -> Indexing.equal_symbol mapped s2
    | None, Some _ -> false (* s2 already claimed by a different source symbol *)
    | None, None ->
        Hashtbl.set sym_renaming ~key:s1 ~data:s2;
        Hashtbl.set sym_renaming_rev ~key:s2 ~data:s1;
        true
  in
  let sym_equal (s1 : Indexing.symbol) (s2 : Indexing.symbol) =
    match Hashtbl.find sym_renaming s1 with
    | Some mapped -> Indexing.equal_symbol mapped s2
    | None ->
        (* Free symbol (bound outside the compared trees, e.g. an enclosing loop iterator): no
           renaming -- requires the identical symbol, not claimed as a bound symbol of the other
           tree. *)
        (not (Hashtbl.mem sym_renaming_rev s2)) && Indexing.equal_symbol s1 s2
  in
  (* [orig_indices] carry call-site metadata, not value semantics: [inline_computation] already
     substituted the call indices into the body, so body equality (with free symbols exact) alone
     guarantees value equality. Free symbols in [orig_indices] therefore keep the legacy bijective
     renaming -- consistently-renamed call sites of a producer whose body ignores an index (e.g. a
     broadcast) still merge -- in maps of their own, so an orig-position renaming never justifies
     renaming a free symbol inside a body. Bound symbols still follow the binder map. *)
  let orig_sym_renaming = Hashtbl.create (module Indexing.Symbol) in
  let orig_sym_renaming_rev = Hashtbl.create (module Indexing.Symbol) in
  let orig_sym_equal (s1 : Indexing.symbol) (s2 : Indexing.symbol) =
    match Hashtbl.find sym_renaming s1 with
    | Some mapped -> Indexing.equal_symbol mapped s2
    | None -> (
        (not (Hashtbl.mem sym_renaming_rev s2))
        &&
        match (Hashtbl.find orig_sym_renaming s1, Hashtbl.find orig_sym_renaming_rev s2) with
        | Some mapped, _ -> Indexing.equal_symbol mapped s2
        | None, Some _ -> false (* s2 already claimed by a different source symbol *)
        | None, None ->
            Hashtbl.set orig_sym_renaming ~key:s1 ~data:s2;
            Hashtbl.set orig_sym_renaming_rev ~key:s2 ~data:s1;
            true)
  in
  let idx_equal_gen sym_equal (i1 : Indexing.axis_index) (i2 : Indexing.axis_index) =
    match (i1, i2) with
    | Indexing.Iterator s1, Indexing.Iterator s2 -> sym_equal s1 s2
    | Fixed_idx n1, Fixed_idx n2 -> Int.equal n1 n2
    | Sub_axis, Sub_axis -> true
    | Affine { symbols = syms1; offset = o1 }, Affine { symbols = syms2; offset = o2 } ->
        Int.equal o1 o2
        && List.equal (fun (c1, s1) (c2, s2) -> Int.equal c1 c2 && sym_equal s1 s2) syms1 syms2
    | Concat ss1, Concat ss2 -> List.equal sym_equal ss1 ss2
    | _ -> false
  in
  let idx_equal = idx_equal_gen sym_equal in
  let orig_idx_equal = idx_equal_gen orig_sym_equal in
  let rec equal_t (a : t) (b : t) : bool =
    match (a, b) with
    | Noop, Noop -> true
    | Comment s1, Comment s2 -> String.equal s1 s2
    | Seq (a1, a2), Seq (b1, b2) -> equal_t a1 b1 && equal_t a2 b2
    | ( For_loop { index = i1; from_ = f1; to_ = t1; body = bd1; axis = ax1 },
        For_loop { index = i2; from_ = f2; to_ = t2; body = bd2; axis = ax2 } ) ->
        Int.equal f1 f2 && Int.equal t1 t2 && equal_axis_type ax1 ax2 && sym_bind i1 i2
        && equal_t bd1 bd2
    | Zero_out tn1, Zero_out tn2 -> Tn.equal tn1 tn2
    | Set { tn = tn1; idcs = i1; llsc = s1; _ }, Set { tn = tn2; idcs = i2; llsc = s2; _ } ->
        Tn.equal tn1 tn2 && Array.equal idx_equal i1 i2 && equal_scalar s1 s2
    | Set_local (id1, s1), Set_local (id2, s2) -> ids_equal id1 id2 && equal_scalar s1 s2
    | Declare_local { id = id1; _ }, Declare_local { id = id2; _ } -> ids_bind id1 id2
    (* Conservative: computations containing barriers are never judged alpha-equivalent --
       deduplicating them would delete a synchronization point. *)
    | Workgroup_barrier, _ -> false
    (* Conservative in v1: guarded statements are not deduplicated. *)
    | If _, _ -> false
    | _ -> false
  and equal_scalar (a : scalar_t) (b : scalar_t) : bool =
    match (a, b) with
    | ( Local_scope { id = id1; body = b1; orig_indices = oi1; mint = m1 },
        Local_scope { id = id2; body = b2; orig_indices = oi2; mint = m2 } ) ->
        (* Record the binder mapping through the checked path (Bug 3) before comparing the body, so
           the binder and its nested [Set_local] / [Get_local] uses all agree via [ids_equal]. *)
        equal_scope_mint m1 m2 && ids_bind id1 id2
        && Array.equal orig_idx_equal oi1 oi2
        && equal_t b1 b2
    | Get_local id1, Get_local id2 -> ids_equal id1 id2
    | Get (tn1, i1), Get (tn2, i2) -> Tn.equal tn1 tn2 && Array.equal idx_equal i1 i2
    | ( Get_dynamic { tn = tn1; idcs = i1; dyn_axis = da1; dyn_value = v1 },
        Get_dynamic { tn = tn2; idcs = i2; dyn_axis = da2; dyn_value = v2 } ) ->
        Tn.equal tn1 tn2 && Int.equal da1 da2 && Array.equal idx_equal i1 i2 && equal_arg v1 v2
    | Get_merge_buffer (tn1, i1), Get_merge_buffer (tn2, i2) ->
        Tn.equal tn1 tn2 && Array.equal idx_equal i1 i2
    | Ternop (op1, a1, a2, a3), Ternop (op2, b1, b2, b3) ->
        Ops.equal_ternop op1 op2 && equal_arg a1 b1 && equal_arg a2 b2 && equal_arg a3 b3
    | Binop (op1, a1, a2), Binop (op2, b1, b2) ->
        Ops.equal_binop op1 op2 && equal_arg a1 b1 && equal_arg a2 b2
    | Unop (op1, a1), Unop (op2, b1) -> Ops.equal_unop op1 op2 && equal_arg a1 b1
    | Constant c1, Constant c2 -> Float.equal c1 c2
    | Constant_bits i1, Constant_bits i2 -> Int64.equal i1 i2
    | Embed_index idx1, Embed_index idx2 -> idx_equal idx1 idx2
    | _ -> false
  and equal_arg ((s1, p1) : scalar_arg) ((s2, p2) : scalar_arg) : bool =
    Ops.equal_prec p1 p2 && equal_scalar s1 s2
  in
  equal_scalar s1 s2

(** Eliminates common subexpressions within each statement's scalar expression tree. Replaces
    duplicate [Local_scope] nodes (structurally identical modulo [scope_id]) with [Get_local]
    references to the first occurrence. *)
let eliminate_common_subexpressions llc =
  (* CSE operates within a single scalar expression tree per statement. *)
  let cse_scalar llsc =
    (* Association list: (representative Local_scope scalar, its scope_id) *)
    let seen : (scalar_t * scope_id) list ref = ref [] in
    let rec loop_scalar (llsc : scalar_t) : scalar_t =
      match llsc with
      | Local_scope { id; body; orig_indices; mint } -> (
          (* Save seen list: inner definitions must not leak to sibling subtrees *)
          let saved_seen = !seen in
          (* First CSE within the body (bottom-up: inner scopes first) *)
          let body = loop_proc body in
          (* Restore: discard inner definitions, keep only those visible at this level *)
          seen := saved_seen;
          let result = Local_scope { id; body; orig_indices; mint } in
          (* Search for an alpha-equivalent Local_scope already seen at this level *)
          let found =
            List.find_map !seen ~f:(fun (prev_scalar, prev_id) ->
                if cse_equal_scalar prev_scalar result then Some prev_id else None)
          in
          match found with
          | Some existing_id -> Get_local existing_id
          | None ->
              seen := (result, id) :: !seen;
              result)
      | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
          Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop_scalar v, prec) }
      | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          llsc
      | Ternop (op, (s1, p1), (s2, p2), (s3, p3)) ->
          Ternop (op, (loop_scalar s1, p1), (loop_scalar s2, p2), (loop_scalar s3, p3))
      | Binop (op, (s1, p1), (s2, p2)) -> Binop (op, (loop_scalar s1, p1), (loop_scalar s2, p2))
      | Unop (op, (s1, p1)) -> Unop (op, (loop_scalar s1, p1))
    and loop_proc (llc : t) : t =
      match llc with
      | Noop -> Noop
      | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> llc
      | Seq (c1, c2) -> Seq (loop_proc c1, loop_proc c2)
      | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
      | Scan_loop sc ->
          (* Each init is its own statement for scope-sharing purposes, like a [Set_local]. *)
          let carried =
            List.map sc.carried ~f:(fun c ->
                let saved = !seen in
                let init = loop_scalar c.init in
                seen := saved;
                { c with init })
          in
          Scan_loop { sc with carried; body = loop_proc sc.body }
      | Set { tn; idcs; llsc; debug } ->
          (* Each statement gets its own scope: codegen wraps in { } when local defs exist, so
             sibling statements can't reference each other's Local_scope declarations. *)
          let saved = !seen in
          let llsc = loop_scalar llsc in
          seen := saved;
          Set { tn; idcs; llsc; debug }
      | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, vprec; llsc; debug } ->
          let saved = !seen in
          let v = loop_scalar v in
          let llsc = loop_scalar llsc in
          seen := saved;
          Set_dynamic { tn; idcs; dyn_axis; dyn_value = (v, vprec); llsc; debug }
      | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
          let saved = !seen in
          let arg_scalar = loop_scalar arg_scalar in
          seen := saved;
          Set_from_vec { tn; idcs; length; vec_unop; arg = (arg_scalar, arg_prec); debug }
      | Set_local (id, llsc) ->
          let saved = !seen in
          let llsc = loop_scalar llsc in
          seen := saved;
          Set_local (id, llsc)
      | If { cond = c, prec; body } ->
          let saved = !seen in
          let c = loop_scalar c in
          seen := saved;
          If { cond = (c, prec); body = loop_proc body }
      | Tile_mma _ -> llc (* Opaque: no CSE into the cooperative statement or its fallback. *)
      | Declare_local _ -> llc
    in
    loop_scalar llsc
  in
  let rec loop_proc (llc : t) : t =
    match llc with
    | Noop -> Noop
    | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
    | Tile_mma _ ->
        llc
    | Seq (c1, c2) -> Seq (loop_proc c1, loop_proc c2)
    | For_loop for_config -> For_loop { for_config with body = loop_proc for_config.body }
    | Scan_loop sc ->
        Scan_loop
          {
            sc with
            carried = List.map sc.carried ~f:(fun c -> { c with init = cse_scalar c.init });
            body = loop_proc sc.body;
          }
    | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = cse_scalar llsc; debug }
    | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, vprec; llsc; debug } ->
        Set_dynamic
          { tn; idcs; dyn_axis; dyn_value = (cse_scalar v, vprec); llsc = cse_scalar llsc; debug }
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
        Set_from_vec { tn; idcs; length; vec_unop; arg = (cse_scalar arg_scalar, arg_prec); debug }
    | Set_local (id, llsc) -> Set_local (id, cse_scalar llsc)
    | If { cond = c, prec; body } -> If { cond = (cse_scalar c, prec); body = loop_proc body }
  in
  loop_proc llc

(** Collect all top-level [Local_scope] nodes from a scalar expression tree. Returns a list of
    [(local_scope_scalar, scope_id)] pairs. Does not recurse into nested [Local_scope] bodies. *)
let collect_local_scopes_in_scalar (llsc : scalar_t) : (scalar_t * scope_id) list =
  let acc = ref [] in
  let rec loop (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; _ } -> acc := (llsc, id) :: !acc
    | Get_dynamic { dyn_value = v, _; _ } -> loop v
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        loop s1;
        loop s2;
        loop s3
    | Binop (_, (s1, _), (s2, _)) ->
        loop s1;
        loop s2
    | Unop (_, (s, _)) -> loop s
  in
  loop llsc;
  List.rev !acc

(** Collect all [Local_scope] candidates from a statement's scalar trees. *)
let collect_local_scopes_in_stmt (stmt : t) : (scalar_t * scope_id) list =
  match stmt with
  | Set { llsc; _ } -> collect_local_scopes_in_scalar llsc
  | Set_from_vec { arg = arg_scalar, _; _ } -> collect_local_scopes_in_scalar arg_scalar
  | Set_local (_, llsc) -> collect_local_scopes_in_scalar llsc
  | _ -> []

(** Replace all [Local_scope] nodes alpha-equivalent to [target] with [Get_local replacement] in a
    scalar expression tree. Also remaps [Get_local] nodes whose [scope_id] is in [stale_ids] to
    point to [replacement], since their original [Local_scope] is being hoisted. *)
let replace_local_scope_in_scalar ~target ~(replacement : scope_id) ~(stale_ids : scope_id list)
    (llsc : scalar_t) : scalar_t =
  let rec loop (llsc : scalar_t) : scalar_t =
    match llsc with
    | Local_scope _ -> if cse_equal_scalar llsc target then Get_local replacement else llsc
    | Get_local id ->
        if List.exists stale_ids ~f:(fun stale -> equal_scope_id id stale) then
          Get_local replacement
        else llsc
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, prec } ->
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop v, prec) }
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> llsc
    | Ternop (op, (s1, p1), (s2, p2), (s3, p3)) ->
        Ternop (op, (loop s1, p1), (loop s2, p2), (loop s3, p3))
    | Binop (op, (s1, p1), (s2, p2)) -> Binop (op, (loop s1, p1), (loop s2, p2))
    | Unop (op, (s, p)) -> Unop (op, (loop s, p))
  in
  loop llsc

(** Replace matching [Local_scope] nodes in a statement's scalar children, and remap stale
    [Get_local] references. *)
let replace_local_scope_in_stmt ~target ~replacement ~stale_ids (stmt : t) : t =
  let repl = replace_local_scope_in_scalar ~target ~replacement ~stale_ids in
  match stmt with
  | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = repl llsc; debug }
  | Set_from_vec { tn; idcs; length; vec_unop; arg = arg_scalar, arg_prec; debug } ->
      Set_from_vec { tn; idcs; length; vec_unop; arg = (repl arg_scalar, arg_prec); debug }
  | Set_local (id, llsc) -> Set_local (id, repl llsc)
  | other -> other

(** Collect all tensor nodes read via [Get(tn, _)] in a statement tree. *)
let reads_of_body (body : t) : Set.M(Tn).t =
  let acc = ref (Set.empty (module Tn)) in
  let rec loop_proc (llc : t) =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
        ()
    | Seq (c1, c2) ->
        loop_proc c1;
        loop_proc c2
    | For_loop { body; _ } -> loop_proc body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> loop_scalar c.init);
        loop_proc body
    | Set { llsc; _ } -> loop_scalar llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        (* The RMW read of the scatter target surfaces via the [Get_dynamic] inside [llsc]. *)
        loop_scalar v;
        loop_scalar llsc
    | Set_from_vec { arg = arg_scalar, _; _ } -> loop_scalar arg_scalar
    | Set_local (_, llsc) -> loop_scalar llsc
    | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; _ } ->
        (* [d] is read-modify-written; [a]/[b] are reads. The fallback touches the same nodes. *)
        acc := Set.add (Set.add (Set.add !acc d_tn) a_tn) b_tn
    | If { cond = c, _; body } ->
        loop_scalar c;
        loop_proc body
  and loop_scalar (llsc : scalar_t) =
    match llsc with
    | Get (tn, _) -> acc := Set.add !acc tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        (* gh-343: the table is read at [tn]; the dynamic index reads its own tensor inside
           [dyn_value], so recurse to count it too. *)
        acc := Set.add !acc tn;
        loop_scalar v
    | Get_merge_buffer (tn, _) -> acc := Set.add !acc tn
    | Local_scope { body; _ } -> loop_proc body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        loop_scalar s1;
        loop_scalar s2;
        loop_scalar s3
    | Binop (_, (s1, _), (s2, _)) ->
        loop_scalar s1;
        loop_scalar s2
    | Unop (_, (s, _)) -> loop_scalar s
  in
  loop_proc body;
  !acc

(** Collect all tensor nodes written by a statement, recursing into [Seq] and [For_loop] bodies.

    The recursion into [For_loop] is load-bearing for hoisting safety (Bug 2): [flat_lines] keeps
    [For_loop] opaque, so [hoist_shared_locals]'s hazard check relies on this function to see writes
    performed *inside* a sibling loop sitting between two users of a hoisted [Local_scope]. A
    non-recursive version reported no writes for such a loop, which could permit an unsound hoist
    above it (later users would then read the pre-loop value). Recursing can only enlarge the hazard
    set, so it only ever narrows what is hoisted -- safe by construction. [Set_local] writes a
    [scope_id] local rather than a materialized [Tn], so it contributes nothing here. *)
let writes_of_stmt (stmt : t) : Set.M(Tn).t =
  let acc = ref (Set.empty (module Tn)) in
  let rec loop (s : t) =
    match s with
    | Set { tn; _ } | Set_dynamic { tn; _ } | Set_from_vec { tn; _ } -> acc := Set.add !acc tn
    | Zero_out tn -> acc := Set.add !acc tn
    | Tile_mma { d = tn, _; _ } -> acc := Set.add !acc tn
    | Seq (a, b) ->
        loop a;
        loop b
    | For_loop { body; _ } | Scan_loop { body; _ } -> loop body
    | If { body; _ } -> loop body
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Set_local _ | Workgroup_barrier ->
        ()
  in
  loop stmt;
  !acc

(** The scope locals a [Local_scope] body READS, at any depth. The companion of {!reads_of_body} for
    the other half of a body's inputs: [reads_of_body] tracks tensor nodes and ignores [Get_local]
    entirely, which left cross-statement hoisting blind to local-valued dependencies (gh-ocannl-584
    review round 2). Over-approximate on purpose — it includes locals the body itself owns, which no
    sibling statement can write, and an enlarged hazard set only narrows hoisting. *)
let local_reads_of_body (body : t) : scope_id list =
  let acc = ref [] in
  let rec loop_proc (llc : t) =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
        ()
    | Seq (c1, c2) ->
        loop_proc c1;
        loop_proc c2
    | For_loop { body; _ } -> loop_proc body
    | Scan_loop { carried; body; _ } ->
        (* The rotation reads every [next]; the body's [Get_local prev]s are collected below. *)
        List.iter carried ~f:(fun c ->
            acc := c.next :: !acc;
            loop_scalar c.init);
        loop_proc body
    | Set { llsc; _ } | Set_local (_, llsc) -> loop_scalar llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        loop_scalar v;
        loop_scalar llsc
    | Set_from_vec { arg = a, _; _ } -> loop_scalar a
    | Tile_mma { fallback; _ } -> loop_proc fallback
    | If { cond = c, _; body } ->
        loop_scalar c;
        loop_proc body
  and loop_scalar (llsc : scalar_t) =
    match llsc with
    | Get_local id -> acc := id :: !acc
    | Local_scope { body; _ } -> loop_proc body
    | Get_dynamic { dyn_value = v, _; _ } -> loop_scalar v
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        loop_scalar s1;
        loop_scalar s2;
        loop_scalar s3
    | Binop (_, (s1, _), (s2, _)) ->
        loop_scalar s1;
        loop_scalar s2
    | Unop (_, (s, _)) -> loop_scalar s
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  in
  loop_proc body;
  !acc

(** The scope locals a statement writes, mirroring {!writes_of_stmt}: recursive through [Seq],
    [For_loop] and [If] bodies, and deliberately NOT into [Local_scope] bodies within its
    expressions. Scope purity is what makes that omission sound — a body only writes locals it owns,
    which by construction no other statement shares (gh-ocannl-584). *)
let local_writes_of_stmt (stmt : t) : scope_id list =
  let acc = ref [] in
  let rec loop (s : t) =
    match s with
    | Set_local (id, _) -> acc := id :: !acc
    | Seq (a, b) ->
        loop a;
        loop b
    | For_loop { body; _ } -> loop body
    | Scan_loop { carried; body; _ } ->
        (* The rotation writes every [prev]; the body's [Set_local next]s are collected below. *)
        List.iter carried ~f:(fun c -> acc := c.prev :: !acc);
        loop body
    | If { body; _ } -> loop body
    | Set _ | Set_dynamic _ | Set_from_vec _ | Zero_out _ | Tile_mma _ | Noop | Comment _
    | Staged_compilation _ | Declare_local _ | Workgroup_barrier ->
        ()
  in
  loop stmt;
  !acc

(** Returns [true] if the given [scope_id] is read (via [Get_local]) before the first definitely
    executed [Set_local] to that id in [body]. Used to decide whether a [Local_scope] or hoisted
    [Declare_local] needs a zero initializer. A loop body write is only considered definite when the
    loop bounds guarantee at least one iteration ([from_ <= to_]); reads inside loops always count
    conservatively. Nested [Local_scope] binders introduce distinct [scope_id]s, so there is no
    shadowing to handle. *)
let reads_scope_before_set (target : scope_id) (body : t) : bool =
  let rec scalar_has_read (llsc : scalar_t) : bool =
    match llsc with
    | Get_local id -> equal_scope_id id target
    | Local_scope { body; _ } -> proc_has_read body
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        scalar_has_read s1 || scalar_has_read s2 || scalar_has_read s3
    | Binop (_, (s1, _), (s2, _)) -> scalar_has_read s1 || scalar_has_read s2
    | Unop (_, (s, _)) -> scalar_has_read s
    | Get_dynamic { dyn_value = v, _; _ } -> scalar_has_read v
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> false
  and proc_has_read (llc : t) : bool =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
        false
    | Seq (a, b) -> proc_has_read a || proc_has_read b
    | For_loop { body; _ } -> proc_has_read body
    | Scan_loop { carried; body; _ } ->
        List.exists carried ~f:(fun c -> scalar_has_read c.init) || proc_has_read body
    | If { cond = c, _; body } -> scalar_has_read c || proc_has_read body
    | Set { llsc; _ } -> scalar_has_read llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar_has_read v || scalar_has_read llsc
    | Set_from_vec { arg = s, _; _ } -> scalar_has_read s
    | Set_local (_, llsc) -> scalar_has_read llsc
    | Tile_mma { fallback; _ } -> proc_has_read fallback
  in
  (* Three-valued scan: Read (found a get before first definite set), Written (found a definite set
     before any get), Neither. *)
  let rec scan (llc : t) : [ `Read | `Written | `Neither ] =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
        `Neither
    | Set { llsc; _ } -> if scalar_has_read llsc then `Read else `Neither
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        if scalar_has_read v || scalar_has_read llsc then `Read else `Neither
    | Set_from_vec { arg = s, _; _ } -> if scalar_has_read s then `Read else `Neither
    | Set_local (id, llsc) ->
        if scalar_has_read llsc then `Read
        else if equal_scope_id id target then `Written
        else `Neither
    | Tile_mma { fallback; _ } -> if proc_has_read fallback then `Read else `Neither
    | Seq (a, b) -> (
        match scan a with `Read -> `Read | `Written -> `Written | `Neither -> scan b)
    | For_loop { body; from_; to_; _ } -> (
        match scan body with
        | `Read -> `Read
        | `Written -> if from_ <= to_ then `Written else `Neither
        | `Neither -> `Neither)
    | Scan_loop { carried; body; from_; to_; _ } -> (
        if List.exists carried ~f:(fun c -> scalar_has_read c.init) then `Read
        else
          match scan body with
          | `Read -> `Read
          | `Written -> if from_ <= to_ then `Written else `Neither
          | `Neither -> `Neither)
    | If { cond = c, _; body } -> (
        if
          (* A guarded write is never a definite write. *)
          scalar_has_read c
        then `Read
        else match scan body with `Read -> `Read | `Written | `Neither -> `Neither)
  in
  match scan body with `Written -> false | `Read | `Neither -> true

(** Whether a statement tree contains a [Workgroup_barrier] anywhere, including inside [For_loop]
    bodies and [Local_scope] bodies of its scalar expressions. Used to delimit code motion:
    [flat_lines] keeps [For_loop] opaque and [writes_of_stmt] treats barriers as writing no tensor,
    so this recursive check is what makes a barrier nested in a sibling statement visible to
    hoisting. *)
let rec contains_barrier (llc : t) : bool =
  match llc with
  | Workgroup_barrier -> true
  (* A cooperative tile statement is barrier-strength: all lanes must reach it together, and no code
     motion may cross it. Treating it as a barrier also makes [validate_parallel] apply the
     workgroup-extent-uniformity and no-If-guard rules to kernels containing it. *)
  | Tile_mma _ -> true
  | Seq (a, b) -> contains_barrier a || contains_barrier b
  | For_loop { body; _ } -> contains_barrier body
  | Scan_loop { carried; body; _ } ->
      List.exists carried ~f:(fun c -> scalar_contains_barrier c.init) || contains_barrier body
  | Set { llsc; _ } -> scalar_contains_barrier llsc
  | Set_dynamic { dyn_value = v, _; llsc; _ } ->
      scalar_contains_barrier v || scalar_contains_barrier llsc
  | Set_from_vec { arg = s, _; _ } -> scalar_contains_barrier s
  | Set_local (_, llsc) -> scalar_contains_barrier llsc
  | If { cond = c, _; body } -> scalar_contains_barrier c || contains_barrier body
  | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ -> false

and scalar_contains_barrier (llsc : scalar_t) : bool =
  match llsc with
  | Local_scope { body; _ } -> contains_barrier body
  | Get_dynamic { dyn_value = v, _; _ } -> scalar_contains_barrier v
  | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
      scalar_contains_barrier s1 || scalar_contains_barrier s2 || scalar_contains_barrier s3
  | Binop (_, (s1, _), (s2, _)) -> scalar_contains_barrier s1 || scalar_contains_barrier s2
  | Unop (_, (s, _)) -> scalar_contains_barrier s
  | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> false

(* Whether the tree carries a read-modify-write accumulation: some [Set] (resp. [Set_local]) reads
   its own target — a loop-carried dependency through memory when the written cell does not vary
   with an enclosing loop. Conservative: [Local_scope] contents count as reading anything, and
   [Tile_mma] accumulates by construction. Used by the autotune menu and by codegen fallbacks that
   must not assert iteration independence (e.g. vectorization pragmas) over an accumulating body
   (gh-ocannl-468). *)
let has_accumulation (llc : t) : bool =
  let rec scalar_reads ~read (sc : scalar_t) =
    let arg (s, _prec) = scalar_reads ~read s in
    match sc with
    | Get (tn2, _) -> ( match read with `Tn tn -> Tnode.equal tn tn2 | `Local _ -> false)
    | Get_local id2 -> ( match read with `Local id -> equal_scope_id id id2 | `Tn _ -> false)
    | Local_scope _ -> true (* Conservative: opaque nested computation. *)
    | Get_dynamic { tn = tn2; dyn_value; _ } ->
        (match read with `Tn tn -> Tnode.equal tn tn2 | `Local _ -> false) || arg dyn_value
    | Get_merge_buffer _ -> false
    | Ternop (_, a, b, c) -> arg a || arg b || arg c
    | Binop (_, a, b) -> arg a || arg b
    | Unop (_, a) -> arg a
    | Constant _ | Constant_bits _ | Embed_index _ -> false
  in
  let rec loop (llc : t) =
    match llc with
    | Seq (a, b) -> loop a || loop b
    | If { body; _ } -> loop body
    | For_loop { body; _ } -> loop body
    (* gh-ocannl-696: a loop-carried dependency by construction. *)
    | Scan_loop _ -> true
    | Set { tn; llsc; _ } -> scalar_reads ~read:(`Tn tn) llsc
    (* gh-466: a dynamic scatter accumulates by construction (its RHS reads the target row) and its
       write location is not statically known — never assert iteration independence over it. *)
    | Set_dynamic _ -> true
    | Set_local (id, sc) -> scalar_reads ~read:(`Local id) sc
    | Tile_mma _ -> true
    | Set_from_vec _ | Zero_out _ | Declare_local _ | Workgroup_barrier | Noop | Comment _
    | Staged_compilation _ ->
        false
  in
  loop llc

exception Impure_scope of string

(** gh-ocannl-584: the scope-purity contract. A [Local_scope] body's ONLY effect is on the locals it
    owns — its own scope id, plus ids [Declare_local]d lexically within it. Raises
    [Invalid_argument] on anything else inside a body, at any nesting depth: a tensor-node write
    ([Set], [Set_from_vec], [Set_dynamic], [Zero_out], [Tile_mma]); a [Set_local] of a sibling or
    enclosing scope's local; a [Workgroup_barrier]; and a [Staged_compilation], whose callback emits
    code this cannot inspect.

    The rule exists because a scope body does not execute where it is written. [C_syntax.pp_scalar]
    returns it as a local definition, which [pp_local_defs] emits ahead of the enclosing statement,
    ordered by [scope_id] rather than by the operand's syntactic position; [simplify_llc] collapses
    a single-assignment scope into the expression, moving its reads the other way; and
    [hoist_cross_statement_cse] lifts a body shared by sibling statements out of the statement
    altogether, to run ONCE ahead of the first user. An effect in a body is therefore placed by none
    of the rules a reader of the expression would expect. Confining bodies to the locals they own
    makes all three placements unobservable to everything OUTSIDE the body, which is what
    {!Affine.path_before} already assumes when it declines to order sibling [Arg] positions, and
    what makes the collapse sound.

    Purity is about a body's effects, and it is not on its own enough for the hoist, which also
    needs the body's INPUTS unchanged across the statements it is lifted over — the obligation of
    [hoist_shared_locals]'s hazard check, which covers tensor reads ({!reads_of_body}) and scope
    locals ({!local_reads_of_body}) alike.

    The statement match is deliberately exhaustive with no catch-all: a new {!t} constructor breaks
    this build until someone classifies it as body-legal or not.

    The optimizer satisfies the contract by construction: [inline_computation] drops the inlined
    computation's [Set]s and [Zero_out]s, rewriting only the setters of the node being inlined, and
    into [Set_local] of the scope's own id. So this binds hand-built and future-pass IR, at both
    ends of the pipeline — [optimize_proc] on the way in (before any pass can launder a violation
    into a shape later gates would accept) and [C_syntax.compile_proc] on the way out (catching
    anything a later pass constructs). The raw analysis entry points [analyze_proc] and
    [specialize_proc] deliberately do NOT validate: they are the probes that must stay conservative
    on IR they may not trust (see [test/operations/affine_extraction.ml]). *)

let scope_purity_violation_gen (root : [ `Proc of t | `Scalar of scalar_t ]) : string option =
  let reject scope what =
    raise
      (Impure_scope
         (what ^ " inside the body of local scope v" ^ Int.to_string scope.scope_id ^ "_"
        ^ Tn.debug_name scope.tn
        ^ " -- a scope body is hoisted out of the position it is written in, so its only effect \
           may be on the locals it owns (gh-ocannl-584)"))
  in
  let wrote construct tn () = construct ^ " writes the tensor node " ^ Tn.debug_name tn in
  (* [owned] carries the ids the enclosing body may write, extended lexically by [Declare_local];
     [proc] threads it along a [Seq] and returns it. Declarations inside a loop or guard body do not
     escape it. Membership is by whole {!scope_id} ([equal_scope_id]), not by the [scope_id]
     integer: hand-built IR can mint two locals sharing an integer but naming different tensor
     nodes, and codegen renders those as DIFFERENT C variables ([pp_scope_id] concatenates both), so
     an integer-keyed set would let a body write a local it does not own. A list is the right shape
     here -- a body owns one id plus whatever it declares. *)
  let owns owned id = List.mem owned id ~equal:equal_scope_id in
  let rec proc ~scope ~owned (llc : t) : scope_id list =
    let ban what = Option.iter scope ~f:(fun s -> reject s (what ())) in
    match llc with
    | Seq (a, b) -> proc ~scope ~owned:(proc ~scope ~owned a) b
    | For_loop { body; _ } ->
        ignore (proc ~scope ~owned body : scope_id list);
        owned
    | Scan_loop { carried; body; _ } ->
        (* gh-ocannl-696: the carried pair is declared by the construct, so the body's [Set_local]s
           of a [next] are the scan's own; the declarations do not escape the scan. *)
        List.iter carried ~f:(fun c -> scalar ~scope ~owned c.init);
        let owned' = List.fold carried ~init:owned ~f:(fun acc c -> c.next :: c.prev :: acc) in
        ignore (proc ~scope ~owned:owned' body : scope_id list);
        owned
    | If { cond = c, _; body } ->
        scalar ~scope ~owned c;
        ignore (proc ~scope ~owned body : scope_id list);
        owned
    | Declare_local { id; _ } -> id :: owned
    | Set_local (id, llsc) ->
        if Option.is_some scope && not (owns owned id) then
          ban (fun () ->
              "a Set_local of v" ^ Int.to_string id.scope_id ^ "_" ^ Tn.debug_name id.tn
              ^ ", a local the body does not own,");
        scalar ~scope ~owned llsc;
        owned
    | Zero_out tn ->
        ban (wrote "Zero_out" tn);
        owned
    | Set { tn; llsc; _ } ->
        ban (wrote "Set" tn);
        scalar ~scope ~owned llsc;
        owned
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        ban (wrote "Set_dynamic" tn);
        scalar ~scope ~owned v;
        scalar ~scope ~owned llsc;
        owned
    | Set_from_vec { tn; arg = a, _; _ } ->
        ban (wrote "Set_from_vec" tn);
        scalar ~scope ~owned a;
        owned
    | Tile_mma { d = d_tn, _; fallback; _ } ->
        ban (wrote "Tile_mma" d_tn);
        ignore (proc ~scope ~owned fallback : scope_id list);
        owned
    (* Neither is scope-local state, and neither can be placed by the hoisting rules: a barrier is a
       synchronization effect, and a staged callback emits code this walk cannot see. *)
    | Workgroup_barrier ->
        ban (fun () -> "a Workgroup_barrier, a synchronization effect,");
        owned
    | Staged_compilation _ ->
        ban (fun () -> "a Staged_compilation, whose callback emits code that cannot be inspected,");
        owned
    | Noop | Comment _ -> owned
  and scalar ~scope ~owned (llsc : scalar_t) : unit =
    match llsc with
    (* A nested body starts from its own id alone: it owns neither its parent's local nor a
       sibling's, so it cannot make their emission order observable. *)
    | Local_scope { id; body; _ } ->
        ignore (proc ~scope:(Some id) ~owned:[ id ] body : scope_id list)
    | Get_dynamic { dyn_value = v, _; _ } -> scalar ~scope ~owned v
    | Ternop (_, (s1, _), (s2, _), (s3, _)) ->
        scalar ~scope ~owned s1;
        scalar ~scope ~owned s2;
        scalar ~scope ~owned s3
    | Binop (_, (s1, _), (s2, _)) ->
        scalar ~scope ~owned s1;
        scalar ~scope ~owned s2
    | Unop (_, (s, _)) -> scalar ~scope ~owned s
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  in
  match
    match root with
    | `Proc llc -> ignore (proc ~scope:None ~owned:[] llc : scope_id list)
    | `Scalar llsc -> scalar ~scope:None ~owned:[] llsc
  with
  | () -> None
  | exception Impure_scope msg -> Some msg

let scope_purity_violation (llc : t) : string option = scope_purity_violation_gen (`Proc llc)

(** The scalar form: pass a [Local_scope] to ask whether ITS body is pure — [scope_purity_violation]
    on a bare body would walk it with no enclosing scope and find nothing. *)
let scope_purity_violation_scalar (llsc : scalar_t) : string option =
  scope_purity_violation_gen (`Scalar llsc)

let validate_scope_bodies (llc : t) : unit =
  Option.iter (scope_purity_violation llc) ~f:(fun msg ->
      invalid_arg ("Low_level.validate_scope_bodies: " ^ msg))

(** {2 Hardware axis analyses}

    Phase B of docs/proposals/axis-types-for-loops.md: pure analyses of hardware-annotated loops.
    Hardware slot assignment is positional, not stored in the IR: among a kernel's loops of one
    kind, the innermost binds [.x] (slot 0), the next [.y], then [.z] — the slot of a loop is the
    maximum nesting depth of same-kind annotated loops strictly inside it, so sibling nests align
    positionally. [Grid] occupies the grid dimensions; [Workgroup] and [Workgroup_reduce] share the
    block/threadgroup dimensions.

    [Grid] slots beyond the three hardware dimensions are legal (gh-ocannl-643: a rank-4 batched
    matmul's chain is two batch loops + row + column, four grid-annotated loops): slots [>= 2] share
    the hardware [.z] dimension by {e folding} — the launch's [.z] extent is the product of the
    per-slot maxima ({!launch_dims}), and a slot-[s] loop binds [(z / stride) % cap] where [stride]
    is the product of the slot maxima below it and [cap] its own slot maximum ({!grid_fold};
    rendered by [C_syntax.hardware_binding]). The decode is a bijection between [.z] values and slot
    coordinate tuples, so thread identity and the coverage rule are unchanged; a nest whose extent
    at a slot is below the slot maximum keeps the ordinary [If] guard ({!guard_annotated_extents}).
    [Workgroup] slots stay capped at 3: threadgroup shape interacts with barriers and thread-id
    semantics, and no annotator emits deeper workgroup nests. *)

type launch_dims = { grid : int array; block : int array } [@@deriving sexp_of, equal]

let hardware_kind_of_axis = function
  | Grid -> Some `Grid
  | Workgroup | Workgroup_reduce -> Some `Workgroup
  | Serial | Unrolled | Vectorized -> None

let hardware_kind_label = function `Grid -> "Grid" | `Workgroup -> "Workgroup"

(* Max nesting depth of [kind]-annotated loops within [llc]. Statement tree only: annotated loops
   inside [Local_scope] bodies are rejected by [validate_parallel]. *)
let rec hardware_depth kind (llc : t) : int =
  match llc with
  | For_loop { axis; body; _ } ->
      let d = hardware_depth kind body in
      if Option.value_map (hardware_kind_of_axis axis) ~default:false ~f:(Poly.equal kind) then
        d + 1
      else d
  | Seq (a, b) -> max (hardware_depth kind a) (hardware_depth kind b)
  | If { body; _ } | Scan_loop { body; _ } -> hardware_depth kind body
  | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_dynamic _ | Set_from_vec _
  | Set_local _ | Declare_local _ | Workgroup_barrier | Tile_mma _ ->
      0

type hardware_axis_info = {
  ha_index : Indexing.symbol;
  ha_kind : [ `Grid | `Workgroup ];
  ha_axis : axis_type;
      (** The loop's own annotation: [Workgroup] and [Workgroup_reduce] share the kind (and the slot
          space) but not the rendering, and the binding-legality check reads the difference. *)
  ha_slot : int;  (** Positional: the innermost same-kind loop binds [.x] = slot 0. *)
  ha_from_ : int;
  ha_extent : int;  (** [to_ - from_ + 1]. *)
}

(** All hardware-annotated loops of [llc] in pre-order, with their positional slots. *)
let hardware_axes (llc : t) : hardware_axis_info list =
  let acc = ref [] in
  let rec walk llc =
    match llc with
    | For_loop { axis; body; index; from_; to_; _ } ->
        Option.iter (hardware_kind_of_axis axis) ~f:(fun kind ->
            acc :=
              {
                ha_index = index;
                ha_kind = kind;
                ha_axis = axis;
                ha_slot = hardware_depth kind body;
                ha_from_ = from_;
                ha_extent = to_ - from_ + 1;
              }
              :: !acc);
        walk body
    | Seq (a, b) ->
        walk a;
        walk b
    | If { body; _ } | Scan_loop { body; _ } -> walk body
    (* The fallback's loops are fresh serial symbols; the statement binds no hardware axes. *)
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_dynamic _ | Set_from_vec _
    | Set_local _ | Declare_local _ | Workgroup_barrier | Tile_mma _ ->
        ()
  in
  walk llc;
  List.rev !acc

let slot_max_extent axes kind slot =
  List.fold axes ~init:1 ~f:(fun m a ->
      if Poly.equal a.ha_kind kind && a.ha_slot = slot then max m a.ha_extent else m)

(** Launch dimensions: per-slot maximum extents over the kernel's annotated loops ([.x], [.y], [.z];
    all-1s for all-[Serial] code). [Grid] slots [>= 2] fold onto the hardware [.z] dimension (see
    the section comment), so [grid.(2)] is the {e product} of their per-slot maxima. Smaller-extent
    sibling bindings are wrapped in [If] guards by {!guard_annotated_extents}. *)
let launch_dims (llc : t) : launch_dims =
  let axes = hardware_axes llc in
  let grid = [| 1; 1; 1 |] and block = [| 1; 1; 1 |] in
  let max_grid_slot =
    List.fold axes ~init:(-1) ~f:(fun m a ->
        match a.ha_kind with `Grid -> max m a.ha_slot | `Workgroup -> m)
  in
  List.iter axes ~f:(fun a ->
      match a.ha_kind with
      | `Grid -> if a.ha_slot < 2 then grid.(a.ha_slot) <- max grid.(a.ha_slot) a.ha_extent
      | `Workgroup -> if a.ha_slot < 3 then block.(a.ha_slot) <- max block.(a.ha_slot) a.ha_extent);
  for s = 2 to max_grid_slot do
    grid.(2) <- grid.(2) * slot_max_extent axes `Grid s
  done;
  { grid; block }

(** The binding arithmetic of a [Grid] loop at [slot >= 2] under the [.z] fold (see the section
    comment): [(stride, cap)] such that the loop's index is [(z / stride) % cap] — [stride] the
    product of the per-slot maxima of grid slots in [\[2, slot)], [cap = Some m] with [m] the loop's
    own slot maximum when a higher grid slot exists in the kernel, [None] (no modulo needed) when
    this is the topmost folded slot, since then [z / stride < m] already. For the common
    single-slot-2 case this is [(1, None)]: the binding is the bare [.z] register. *)
let grid_fold (axes : hardware_axis_info list) ~(slot : int) : int * int option =
  assert (slot >= 2);
  let stride = ref 1 in
  for s = 2 to slot - 1 do
    stride := !stride * slot_max_extent axes `Grid s
  done;
  let has_higher =
    List.exists axes ~f:(fun a ->
        match a.ha_kind with `Grid -> a.ha_slot > slot | `Workgroup -> false)
  in
  (!stride, if has_higher then Some (slot_max_extent axes `Grid slot) else None)

(* Whether any [Local_scope] body within [llc]'s scalars contains a hardware-annotated loop. *)
let rec scalar_scopes_have_annotated (llc : t) : bool =
  let scalar (llsc : scalar_t) : bool =
    let rec go = function
      | Local_scope { body; _ } ->
          (not (List.is_empty (hardware_axes body))) || scalar_scopes_have_annotated body
      | Get_dynamic { dyn_value = v, _; _ } -> go v
      | Ternop (_, (s1, _), (s2, _), (s3, _)) -> go s1 || go s2 || go s3
      | Binop (_, (s1, _), (s2, _)) -> go s1 || go s2
      | Unop (_, (s, _)) -> go s
      | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          false
    in
    go llsc
  in
  match llc with
  | Seq (a, b) -> scalar_scopes_have_annotated a || scalar_scopes_have_annotated b
  | For_loop { body; _ } -> scalar_scopes_have_annotated body
  | Scan_loop { carried; body; _ } ->
      List.exists carried ~f:(fun c -> scalar c.init) || scalar_scopes_have_annotated body
  | If { cond = c, _; body } -> scalar c || scalar_scopes_have_annotated body
  | Set { llsc; _ } -> scalar llsc
  | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar v || scalar llsc
  | Set_from_vec { arg = s, _; _ } -> scalar s
  | Set_local (_, llsc) -> scalar llsc
  | Tile_mma { fallback; _ } -> scalar_scopes_have_annotated fallback
  | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
      false

(** Backend-independent well-formedness of hardware annotations (axis-types proposal §2). A no-op
    for all-[Serial] code. Raises [Invalid_argument] on: annotated loops with [from_ <> 0]; more
    than 3 [Workgroup] slots ([Grid] slots [>= 2] fold onto [.z], see the hardware-axis section
    comment); annotated loops inside [Local_scope] bodies; a kernel containing barriers whose
    same-slot workgroup extents differ (a barrier under divergent control flow is UB) or with a
    barrier lexically under an [If] guard; and writes to materialized tensor nodes lexically outside
    all annotated loops (every hardware thread would execute them, racing with the annotated writes
    — there is no grid-wide synchronization). Cannot prove iteration independence; that is the
    annotating pass's obligation. *)
let validate_parallel plc (llc : t) : unit =
  let axes = hardware_axes llc in
  if not (List.is_empty axes) then (
    List.iter axes ~f:(fun a ->
        if a.ha_from_ <> 0 then
          invalid_arg
            ("Low_level.validate_parallel: annotated loop " ^ Indexing.symbol_ident a.ha_index
           ^ " must start at 0, starts at " ^ Int.to_string a.ha_from_);
        (* Grid slots [>= 2] fold onto the hardware [.z] dimension (see the hardware-axis section
           comment), so any number of grid axes is renderable; workgroup slots stay capped at the
           three threadgroup dimensions. *)
        if a.ha_slot > 2 && Poly.equal a.ha_kind `Workgroup then
          invalid_arg
            ("Low_level.validate_parallel: more than 3 " ^ hardware_kind_label a.ha_kind
           ^ " axes in one kernel (loop " ^ Indexing.symbol_ident a.ha_index ^ " needs slot "
           ^ Int.to_string a.ha_slot ^ ")"));
    if scalar_scopes_have_annotated llc then
      invalid_arg "Low_level.validate_parallel: hardware-annotated loop inside a Local_scope body";
    if contains_barrier llc then (
      List.iter axes ~f:(fun a ->
          match a.ha_kind with
          | `Workgroup ->
              let m = slot_max_extent axes `Workgroup a.ha_slot in
              if a.ha_extent <> m then
                invalid_arg
                  ("Low_level.validate_parallel: kernel contains barriers but workgroup extents \
                    differ at slot " ^ Int.to_string a.ha_slot
                 ^ " (a barrier under divergent control flow is UB): " ^ Int.to_string a.ha_extent
                 ^ " vs " ^ Int.to_string m)
          | `Grid -> ());
      let rec no_guarded_barrier llc =
        match llc with
        | If { body; _ } ->
            if contains_barrier body then
              invalid_arg "Low_level.validate_parallel: barrier under an If guard is divergent"
        | Seq (a, b) ->
            no_guarded_barrier a;
            no_guarded_barrier b
        | For_loop { body; _ } | Scan_loop { body; _ } -> no_guarded_barrier body
        | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_dynamic _
        | Set_from_vec _ | Set_local _ | Declare_local _ | Workgroup_barrier | Tile_mma _ ->
            ()
      in
      no_guarded_barrier llc);
    (* Launch dimensions are global to the kernel, so a materialized write must be nested under
       annotated loops covering EVERY active (non-unit) hardware dimension -- a statement covering
       only some of them executes once per hardware index of each uncovered dimension, racing or
       repeating read-modify-write updates (PR #89 review). A whole-node [Zero_out] is never
       distributed across threads by nesting, so in a multi-threaded kernel it is rejected outright;
       distribute zeroing as ordinary per-element [Set]s instead. *)
    let pair_equal (k1, s1) (k2, s2) = Poly.equal k1 k2 && s1 = s2 in
    let active =
      List.filter_map axes ~f:(fun a ->
          if slot_max_extent axes a.ha_kind a.ha_slot > 1 then Some (a.ha_kind, a.ha_slot) else None)
      |> List.dedup_and_sort ~compare:Poly.compare
    in
    let slot_of_index = List.map axes ~f:(fun a -> (a.ha_index, (a.ha_kind, a.ha_slot))) in
    let describe_pair (kind, slot) = hardware_kind_label kind ^ " slot " ^ Int.to_string slot in
    let check_covered_write ~covered tn =
      let missing = List.filter active ~f:(fun p -> not (List.mem covered p ~equal:pair_equal)) in
      if (not (List.is_empty missing)) && Tn.Placements.is_materialized_force plc tn 160 then
        invalid_arg
          ("Low_level.validate_parallel: write to materialized node " ^ Tn.debug_name tn
         ^ " is not nested under annotated loops covering all active hardware dimensions (missing: "
          ^ String.concat ~sep:", " (List.map missing ~f:describe_pair)
          ^ "): every hardware index of an uncovered dimension executes it, racing or repeating \
             the update"
          ^
          (* gh-ocannl-633: a constant's in-kernel init normally moves to a link-time [Host_inits]
             upload before this check ([hosted_constant_inits_to_link_time]); when a bail-out kept
             it (e.g. a padded constant), name the actual culprit — the schedule is not at fault.
             The flag advice is scoped honestly (review round 2): [Tensor.constant_fill]'s 1-element
             arm never consults the limit — deliberately, since routing a 1-element literal to the
             host-backed path would pin its element count and break broadcast shape inference — so
             for such literals the flag changes nothing, and this frame cannot tell the literal's
             length (a broadcast scalar's node has the consumer's numel). *)
          if Tn.known_host_constant tn then
            ". The write is the in-kernel initialization of a constant that could not be moved to \
             link time; for literals of at least two elements, --ocannl_limit_constant_fill_size=0 \
             forces host-side initialization (one-element literals always initialize in kernel, \
             keeping broadcast shape inference)"
          else "")
    in
    let rec check_writes ~covered ~enclosing llc =
      match llc with
      | For_loop { index; body; _ } ->
          let covered =
            match List.Assoc.find slot_of_index ~equal:Indexing.equal_symbol index with
            | Some pair -> pair :: covered
            | None -> covered
          in
          check_writes ~covered ~enclosing:(index :: enclosing) body
      | Scan_loop { index; body; _ } -> check_writes ~covered ~enclosing:(index :: enclosing) body
      | Seq (a, b) ->
          check_writes ~covered ~enclosing a;
          check_writes ~covered ~enclosing b
      | If { body; _ } -> check_writes ~covered ~enclosing body
      | Zero_out tn ->
          if (not (List.is_empty active)) && Tn.Placements.is_materialized_force plc tn 160 then
            invalid_arg
              ("Low_level.validate_parallel: Zero_out of materialized node " ^ Tn.debug_name tn
             ^ " in a multi-threaded kernel: whole-node zeroing is not distributed across hardware \
                threads; zero via per-element writes under the annotated loops instead \
                (Schedule.Expand_zero expands the init into a loop nest the same geometry ops can \
                annotate)")
      | Set { tn; _ } | Set_dynamic { tn; _ } | Set_from_vec { tn; _ } ->
          check_covered_write ~covered tn
      | Tile_mma { d = d_tn, d_idcs; a = _, a_idcs; b = _, b_idcs; lane; _ } ->
          (* The cooperating [lane] axis must be a real enclosing [Workgroup]-typed loop: the launch
             needs its threads, hardware backends reach the intrinsic on all of them together, and
             serial renderers key their once-per-simdgroup guard off its index. *)
          (match List.Assoc.find slot_of_index ~equal:Indexing.equal_symbol lane with
          | Some (`Workgroup, _) when List.mem enclosing lane ~equal:Indexing.equal_symbol -> ()
          | _ ->
              invalid_arg
                ("Low_level.validate_parallel: Tile_mma lane " ^ Indexing.symbol_ident lane
               ^ " must be bound by an enclosing Workgroup-typed loop"));
          (* The tile is jointly owned by the simdgroup: per-lane element ownership is
             architecture-opaque, so no base index may mention the lane. *)
          Array.iter
            (Array.concat [ d_idcs; a_idcs; b_idcs ])
            ~f:(fun idx ->
              if axis_index_mentions_symbol lane idx then
                invalid_arg
                  ("Low_level.validate_parallel: Tile_mma base indices must not mention the lane \
                    axis " ^ Indexing.symbol_ident lane ^ " (the tile is jointly owned)"));
          (* The tile store covers the lane slot by decree (the statement is nested under the lane
             loop, adding its slot to [covered]); other active slots follow the ordinary rule. *)
          check_covered_write ~covered d_tn
      | Noop | Comment _ | Staged_compilation _ | Set_local _ | Declare_local _ | Workgroup_barrier
        ->
          ()
    in
    check_writes ~covered:[] ~enclosing:[] llc)

let validate_parallel_classified plc llc =
  match validate_parallel plc llc with
  | () -> ()
  | exception Invalid_argument detail ->
      raise
        (Schedule_outcome.Cause_at
           ( Schedule_outcome.Backend_codegen,
             Schedule_outcome.Illegal_schedule { check = "Low_level.validate_parallel"; detail } ))

(** Wraps the body of each hardware-annotated loop whose extent is smaller than its slot's launch
    dimension in an [If (index < extent)] guard, for the kinds [should_guard] selects (backends
    binding the axis in hardware; the serial fallback iterates the true extent and needs no guard).
    Construct-then-fold: extents are non-negative ints at the signed index width, so every emitted
    conjunct is correct unfolded; the common equal-extent case emits nothing. *)
let guard_annotated_extents ~(should_guard : [ `Grid | `Workgroup ] -> bool) (llc : t) : t =
  let axes = hardware_axes llc in
  let iprec = Ops.index_prec () in
  let rec walk llc =
    match llc with
    | For_loop ({ axis; index; to_; body; _ } as fc) -> (
        let body = walk body in
        match hardware_kind_of_axis axis with
        | Some kind when should_guard kind ->
            let slot = hardware_depth kind body in
            let m = slot_max_extent axes kind slot in
            if to_ + 1 < m then
              let cond =
                Binop
                  ( Ops.Cmplt,
                    (Embed_index (Indexing.Iterator index), iprec),
                    (Constant (Float.of_int (to_ + 1)), iprec) )
              in
              For_loop { fc with body = If { cond = (cond, iprec); body } }
            else For_loop { fc with body }
        | _ -> For_loop { fc with body })
    | Seq (a, b) -> Seq (walk a, walk b)
    | If { cond; body } -> If { cond; body = walk body }
    | Scan_loop sc -> Scan_loop { sc with body = walk sc.body }
    | ( Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_dynamic _
      | Set_from_vec _ | Set_local _ | Declare_local _ | Workgroup_barrier | Tile_mma _ ) as other
      ->
        other
  in
  walk llc

(** Hoists shared [Local_scope] computations from sibling statements to the enclosing scope.
    Operates on a flat list of sibling statements. *)
let rec hoist_shared_locals (stmts : t list) : t list =
  (* No code motion across a barrier: hoist within each barrier-delimited segment separately. The
     write-hazard check below cannot see barriers (they write no tensor), so splitting here is what
     enforces the barrier's full-fence contract for cross-statement CSE. A statement merely
     *containing* a barrier (e.g. a loop with a barrier in its body) is a boundary too: hoisting a
     shared computation from after it to before it would move work across the barrier. *)
  match List.findi stmts ~f:(fun _ stmt -> contains_barrier stmt) with
  | Some (i, _) ->
      let before, rest = List.split_n stmts i in
      let boundary = List.hd_exn rest in
      hoist_shared_locals_segment before @ (boundary :: hoist_shared_locals (List.tl_exn rest))
  | None -> hoist_shared_locals_segment stmts

and hoist_shared_locals_segment (stmts : t list) : t list =
  (* Step 1: Collect all Local_scope candidates with their statement indices *)
  let candidates =
    List.concat_mapi stmts ~f:(fun stmt_idx stmt ->
        List.map (collect_local_scopes_in_stmt stmt) ~f:(fun (scalar, id) -> (stmt_idx, scalar, id)))
  in
  (* Step 2: Group by alpha-equivalence *)
  (* Each group is: (representative_scalar, representative_id, list of stmt indices,
     all scope_ids in the group) *)
  let groups : (scalar_t * scope_id * int list * scope_id list) list ref = ref [] in
  List.iter candidates ~f:(fun (stmt_idx, scalar, cand_id) ->
      let found =
        List.find_mapi !groups ~f:(fun group_idx (rep_scalar, _rep_id, _indices, _all_ids) ->
            if cse_equal_scalar rep_scalar scalar then Some group_idx else None)
      in
      match found with
      | Some group_idx ->
          groups :=
            List.mapi !groups ~f:(fun i (s, id, idxs, all_ids) ->
                if i = group_idx then (s, id, stmt_idx :: idxs, cand_id :: all_ids)
                else (s, id, idxs, all_ids))
      | None -> groups := (scalar, cand_id, [ stmt_idx ], [ cand_id ]) :: !groups);
  (* Keep only groups with 2+ members *)
  let shared_groups =
    List.filter_map !groups ~f:(fun (scalar, id, indices, all_ids) ->
        let indices = List.dedup_and_sort indices ~compare:Int.compare in
        if List.length indices >= 2 then Some (scalar, id, indices, all_ids) else None)
  in
  if List.is_empty shared_groups then stmts
  else
    (* Step 3: Safety check + rewrite *)
    let stmts = Array.of_list stmts in
    let insertions : (int * t list) list ref = ref [] in
    List.iter shared_groups ~f:(fun (target_scalar, canonical_id, user_indices, all_ids) ->
        let first_user = List.hd_exn user_indices in
        let last_user = List.last_exn user_indices in
        (* Collect reads of the Local_scope body -- tensor nodes AND scope locals. Both are inputs
           to the body, so a write to either between the hoist point and a later user would make
           that user read a stale value (gh-ocannl-584 review round 2: the pipeline does produce
           bodies reading a local declared outside them, e.g. layer_norm_divided_mean, via
           [eliminate_common_subexpressions] and this pass's own [Get_local] rewrites). *)
        let body_reads =
          match target_scalar with
          | Local_scope { body; _ } -> reads_of_body body
          | _ -> Set.empty (module Tn)
        in
        let body_local_reads =
          match target_scalar with Local_scope { body; _ } -> local_reads_of_body body | _ -> []
        in
        (* Check for writes between first_user and last_user that could invalidate hoisting. Include
           writes from ALL statements (including user statements) from first_user up to but not
           including last_user. User statements perform tensor writes after evaluating their
           Local_scope, so earlier users' writes can affect what later users would read. *)
        (* gh-ocannl-584 review round 3: the pass guards its OWN precondition rather than trusting a
           gate at every door into it. This is the one pass that can move an effect out of a
           [Local_scope] -- lifting the body to a top-level [Declare_local] + body -- so an impure
           body reaching it would be laundered into a shape the codegen gate no longer recognizes,
           and the routine would compile silently changed (the write executing once ahead of the
           first user instead of once per user). Declining to hoist leaves the impure body inside
           its scope, where the exit gate refuses it: the program is REJECTED, never rewritten. This
           keeps the raw [analyze_proc] / [specialize_proc] probes usable on out-of-contract IR,
           which is why they do not validate. *)
        let pure_body = Option.is_none (scope_purity_violation_scalar target_scalar) in
        let safe =
          let hazard_writes = ref (Set.empty (module Tn)) in
          let hazard_local_writes = ref [] in
          for i = first_user to last_user - 1 do
            hazard_writes := Set.union !hazard_writes (writes_of_stmt stmts.(i));
            hazard_local_writes := local_writes_of_stmt stmts.(i) @ !hazard_local_writes
          done;
          Set.is_empty (Set.inter body_reads !hazard_writes)
          && not
               (List.exists !hazard_local_writes ~f:(fun w ->
                    List.mem body_local_reads w ~equal:equal_scope_id))
        in
        if safe && pure_body then (
          (* Extract body from canonical Local_scope *)
          let body =
            match target_scalar with Local_scope { body; _ } -> body | _ -> assert false
          in
          (* Replace all occurrences in all user statements, also remapping stale Get_local
             references that were created by intra-statement CSE pointing at Local_scope nodes that
             are now being hoisted away. *)
          List.iter user_indices ~f:(fun idx ->
              stmts.(idx) <-
                replace_local_scope_in_stmt ~target:target_scalar ~replacement:canonical_id
                  ~stale_ids:all_ids stmts.(idx));
          (* Record insertion: Declare_local + body before first user *)
          let needs_init = reads_scope_before_set canonical_id body in
          insertions :=
            (first_user, [ Declare_local { id = canonical_id; needs_init }; body ]) :: !insertions));
    (* Apply insertions (sorted by position, last first to preserve indices) *)
    let insertions = List.sort !insertions ~compare:(fun (a, _) (b, _) -> Int.descending a b) in
    let result = Array.to_list stmts in
    let result =
      List.fold insertions ~init:result ~f:(fun acc (pos, prefix) ->
          let before = List.take acc pos in
          let after = List.drop acc pos in
          before @ prefix @ after)
    in
    result

(** Hoists shared [Local_scope] computations from sibling statements to the enclosing scope. When
    two or more sibling statements share an alpha-equivalent [Local_scope] node, the computation is
    extracted as a [Declare_local] + body preceding the first user, and all occurrences are replaced
    with [Get_local]. *)
let hoist_cross_statement_cse llc =
  let rec loop_proc (llc : t) : t =
    match llc with
    | Seq _ ->
        let stmts = flat_lines [ llc ] in
        let stmts = List.map stmts ~f:loop_proc in
        hoist_shared_locals stmts |> unflat_lines
    | For_loop fc -> For_loop { fc with body = loop_proc fc.body }
    (* gh-ocannl-696: hoisting within a scan body is the per-iteration case of the loop-body one;
       the rotation assigns the carried [prev]s after every body statement, so no statement of the
       body can be lifted across such a write. *)
    | Scan_loop sc -> Scan_loop { sc with body = loop_proc sc.body }
    | _ -> llc
  in
  loop_proc llc

let input_and_output_nodes optimized =
  let plc = optimized.optimize_ctx.placements in
  ( Hashtbl.fold optimized.traced_store
      ~init:(Set.empty (module Tn), Set.empty (module Tn))
      ~f:(fun ~key ~data (inputs, outputs) ->
        let materialized = Tn.Placements.is_materialized_force plc key 50 in
        let inputs =
          if
            materialized
            && (not (Tn.Placements.known_constant plc key))
            && (data.read_only || data.read_before_write)
          then Set.add inputs key
          else inputs
        in
        let outputs =
          if materialized && (data.zeroed_out || data.has_assignment) then Set.add outputs key
          else outputs
        in
        (inputs, outputs)),
    optimized.merge_node )

(** All [For_loop] bindings within [llc] (loop symbols are unique within a routine), with inclusive
    iteration bounds — the box environment for {!Affine} queries over the routine's accesses. *)
let loop_bounds (llc : t) : (Indexing.symbol * (int * int)) list =
  let acc = ref [] in
  let rec go = function
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ ->
        ()
    | Seq (a, b) ->
        go a;
        go b
    | For_loop { index; from_; to_; body; _ } ->
        acc := (index, (from_, to_)) :: !acc;
        go body
    | Scan_loop { index; from_; to_; carried; body; _ } ->
        (* gh-ocannl-696: the scan index bounds its body's accesses like a loop index does; what
           differs about a scan is the order its values are produced in, not their footprint. *)
        acc := (index, (from_, to_)) :: !acc;
        List.iter carried ~f:(fun c -> go_sc c.init);
        go body
    | If { cond = c, _; body } ->
        go_sc c;
        go body
    | Tile_mma { fallback; _ } -> go fallback
    | Set { llsc; _ } | Set_local (_, llsc) -> go_sc llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        go_sc v;
        go_sc llsc
    | Set_from_vec { arg = a, _; _ } -> go_sc a
  and go_sc = function
    | Local_scope { body; _ } -> go body
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get_dynamic { dyn_value = v, _; _ } -> go_sc v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        go_sc a;
        go_sc b;
        go_sc c
    | Binop (_, (a, _), (b, _)) ->
        go_sc a;
        go_sc b
    | Unop (_, (a, _)) -> go_sc a
  in
  go llc;
  List.rev !acc

(* Loop symbols a scalar expression's value depends on, syntactically, resolving scalar scope-locals
   through [locals] — accumulated per-scope-id assignment symbols (see {!scope_value_syms}). *)
let rec scalar_value_syms ~(locals : (int, Indexing.symbol list) Hashtbl.t) (llsc : scalar_t) :
    Indexing.symbol list =
  let idx_syms (idx : Indexing.axis_index) =
    match idx with
    | Indexing.Iterator s -> [ s ]
    | Indexing.Affine { symbols; _ } -> List.map symbols ~f:snd
    | Indexing.Concat syms -> syms
    | Indexing.Fixed_idx _ | Indexing.Sub_axis -> []
  in
  let scalar_syms = scalar_value_syms ~locals in
  match llsc with
  | Local_scope { body; _ } -> body_value_syms ~locals body
  | Get_local id -> Hashtbl.find locals id.scope_id |> Option.value ~default:[]
  | Get_merge_buffer _ | Constant _ | Constant_bits _ -> []
  | Embed_index idx -> idx_syms idx
  | Get (_, idcs) -> List.concat_map (Array.to_list idcs) ~f:idx_syms
  | Get_dynamic { idcs; dyn_value = v, _; _ } ->
      List.concat_map (Array.to_list idcs) ~f:idx_syms @ scalar_syms v
  | Ternop (_, (a, _), (b, _), (c, _)) -> scalar_syms a @ scalar_syms b @ scalar_syms c
  (* The dead operand of a projection is never evaluated (codegen renders only the used one). *)
  | Binop (Ops.Arg1, (a, _), _) -> scalar_syms a
  | Binop (Ops.Arg2, _, (b, _)) -> scalar_syms b
  | Binop (_, (a, _), (b, _)) -> scalar_syms a @ scalar_syms b
  | Unop (_, (a, _)) -> scalar_syms a

and body_value_syms ~locals (llc : t) : Indexing.symbol list =
  let scalar_syms = scalar_value_syms ~locals in
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ -> []
  | Seq (a, b) -> body_value_syms ~locals a @ body_value_syms ~locals b
  | For_loop { body; _ } -> body_value_syms ~locals body
  | Scan_loop { carried; body; _ } ->
      List.concat_map carried ~f:(fun c -> scalar_syms c.init) @ body_value_syms ~locals body
  | If { cond = c, _; body } -> scalar_syms c @ body_value_syms ~locals body
  | Set { llsc; _ } | Set_local (_, llsc) -> scalar_syms llsc
  | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar_syms v @ scalar_syms llsc
  | Set_from_vec { arg = a, _; _ } -> scalar_syms a
  | Tile_mma { fallback; _ } -> body_value_syms ~locals fallback

(** The value-dependence symbols of statement-level scalar scope-locals, whole-code: per scope id,
    the union of the symbols its statement-level assignments depend on, transitively through
    [Get_local] references (such a local may be assigned in one statement and read in another — the
    shared-definition / CSE-hoisted pattern). Fixpoint; consumed by the value scans of
    [affine_accesses] and [Schedule.scan_accesses] so that a value routed through a scope-local is
    not laundered of its symbols (gh-494 per-thread value-variance). Assignments inside
    [Local_scope] bodies are deliberately NOT recorded: a scope id is re-instantiated at every use
    site with per-site loop symbols, so a global union would import foreign statements' symbols into
    unrelated setters; scope-internal flow is covered lexically instead — the value of a
    [Local_scope] is over-approximated by the union of all its body setters' symbols. *)
let scope_value_syms (llc : t) : (int, Indexing.symbol list) Hashtbl.t =
  let locals = Hashtbl.create (module Int) in
  let changed = ref true in
  let record id llsc =
    let s = scalar_value_syms ~locals llsc in
    let old = Hashtbl.find locals id.scope_id |> Option.value ~default:[] in
    let merged = List.dedup_and_sort ~compare:Indexing.compare_symbol (s @ old) in
    if List.length merged > List.length old then (
      Hashtbl.set locals ~key:id.scope_id ~data:merged;
      changed := true)
  in
  let rec stmt ~depth (llc : t) =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ ->
        ()
    | Seq (a, b) ->
        stmt ~depth a;
        stmt ~depth b
    | For_loop { body; _ } -> stmt ~depth body
    | Scan_loop { carried; body; _ } ->
        (* gh-ocannl-696: [prev] is assigned by the init and, through the rotation, by [next] --
           recorded as the two statement-level assignments they are, so a value routed through the
           carried state keeps the symbols the body's update depends on. *)
        List.iter carried ~f:(fun c ->
            if depth = 0 then (
              record c.prev c.init;
              record c.prev (Get_local c.next));
            scalar ~depth c.init);
        stmt ~depth body
    | If { cond = c, _; body } ->
        scalar ~depth c;
        stmt ~depth body
    | Set_local (id, llsc) ->
        if depth = 0 then record id llsc;
        scalar ~depth llsc
    | Set { llsc; _ } -> scalar ~depth llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        scalar ~depth v;
        scalar ~depth llsc
    | Set_from_vec { arg = a, _; _ } -> scalar ~depth a
    | Tile_mma { fallback; _ } -> stmt ~depth fallback
  and scalar ~depth (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> stmt ~depth:(depth + 1) body
    | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ | Get _ -> ()
    | Get_dynamic { dyn_value = v, _; _ } -> scalar ~depth v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar ~depth a;
        scalar ~depth b;
        scalar ~depth c
    | Binop (_, (a, _), (b, _)) ->
        scalar ~depth a;
        scalar ~depth b
    | Unop (_, (a, _)) -> scalar ~depth a
  in
  while !changed do
    changed := false;
    stmt ~depth:0 llc
  done;
  locals

(** gh-494 waypoint 1: the routine's tensor-node accesses as explicit affine relations
    ({!Affine.access}), extracted from (typically optimized) code — the queryable artifact behind
    the affine legality queries. Fires in program order: a statement's right-hand-side reads precede
    its write, [Local_scope] bodies are descended into at their use site. [Tile_mma] is traversed
    through its scalar [fallback] (the fallback is the statement's access footprint, as in
    [C_syntax.iter_local_accesses]). Not represented: scope-locals ([Get_local]/[Set_local] carry no
    index map), merge-buffer reads (a separate read-only input buffer), and [Staged_compilation]
    (opaque) — callers needing exhaustiveness must check for the latter separately. *)
let affine_accesses (llc : t) : Tn.t Affine.access list =
  let rec reads_tn uid (llsc : scalar_t) =
    match llsc with
    | Get (tn, _) -> tn.Tn.uid = uid
    | Get_dynamic { tn; dyn_value = v, _; _ } -> tn.Tn.uid = uid || reads_tn uid v
    | Get_merge_buffer _ | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> false
    | Local_scope { body; _ } -> body_reads uid body
    | Ternop (_, (a, _), (b, _), (c, _)) -> reads_tn uid a || reads_tn uid b || reads_tn uid c
    | Binop (op, (a, _), (b, _)) -> (
        (* The dead operand of a projection is never evaluated (codegen renders only the used one),
           per {!Ops.binop_conditionality}. A gated operand can still evaluate, so it counts. *)
        match Ops.binop_conditionality op with
        | Ops.Only_first -> reads_tn uid a
        | Ops.Only_second -> reads_tn uid b
        | Ops.Both_operands | Ops.Gated_second -> reads_tn uid a || reads_tn uid b)
    | Unop (_, (a, _)) -> reads_tn uid a
  and body_reads uid (llc : t) =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ ->
        false
    | Seq (a, b) -> body_reads uid a || body_reads uid b
    | For_loop { body; _ } -> body_reads uid body
    | Scan_loop { carried; body; _ } ->
        List.exists carried ~f:(fun c -> reads_tn uid c.init) || body_reads uid body
    | If { cond = c, _; body } -> reads_tn uid c || body_reads uid body
    | Set { llsc; _ } | Set_local (_, llsc) -> reads_tn uid llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } -> reads_tn uid v || reads_tn uid llsc
    | Set_from_vec { arg = a, _; _ } -> reads_tn uid a
    | Tile_mma { fallback; _ } -> body_reads uid fallback
  in
  let acc = ref [] in
  let add ~loops ~path ~guarded ?(dynamic = false) ?(whole = false) ?(vec_len = 0) ?(rmw = false)
      ?(val_syms = []) ?stmt_write ~write tn map =
    acc :=
      {
        Affine.a_tn = tn;
        a_map = map;
        a_write = write;
        a_dynamic = dynamic;
        a_whole = whole;
        a_vec_last = vec_len > 0;
        a_vec_len = vec_len;
        a_guarded = guarded;
        a_rmw = rmw;
        a_val_syms = val_syms;
        a_stmt_write = stmt_write;
        a_loops = List.rev loops;
        a_path = List.rev path;
      }
      :: !acc
  in
  let local_syms = scope_value_syms llc in
  let scalar_syms = scalar_value_syms ~locals:local_syms in
  (* [stmt_write] threads the enclosing [Set]-family statement's write map through its right-hand
     side, and only there: [If] conditions and [Set_local] pass [None], and a [Local_scope] body's
     statements establish their own (its reads are subordinate to the inner setters, not to the
     statement the scope is inlined into). *)
  let rec code ~loops ~path ~guarded (llc : t) =
    (* gh-561: paths gain an intra-statement component at each statement the traversal descends into
       — [Cond]/[Body] for [If], [Rhs]/[Write] for the [Set] family — so lexicographic path order is
       program order within a statement too: a guarded body's write no longer shares its condition's
       path, and a statement's write orders after its right-hand side (including [Local_scope]
       bodies inlined there, which used to need a prefix-exclusion rule in
       [Affine.read_covered_before]). Each statement's scalar tree carries an evaluation-position
       counter ([arg_c], shared across [Set_dynamic]'s dynamic index and value): every [Local_scope]
       occurrence extends the path with a distinct [Arg] component, so sibling scope bodies inlined
       into one statement never interleave their interior components — and [path_before]
       deliberately does not order across sibling [Arg]s. *)
    let arg_c = ref 0 in
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ -> ()
    | Seq _ ->
        List.iteri (flat_lines [ llc ]) ~f:(fun k stmt ->
            code ~loops ~path:(Affine.Stmt k :: path) ~guarded stmt)
    | For_loop { index; from_; to_; body; _ } ->
        code ~loops:((index, (from_, to_)) :: loops) ~path ~guarded body
    | Scan_loop { index; from_; to_; carried; body; _ } ->
        (* gh-ocannl-696: the inits are statement [0] of the construct, one [Set_local]-shaped
           position each, and the body statement [1] -- so every init read orders before every body
           access, and the scan index bounds the body like a loop index. *)
        List.iteri carried ~f:(fun j c ->
            scalar ~loops
              ~path:(Affine.Rhs :: Affine.Stmt j :: Affine.Stmt 0 :: path)
              ~guarded ~arg_c ?stmt_write:None c.init);
        code ~loops:((index, (from_, to_)) :: loops) ~path:(Affine.Stmt 1 :: path) ~guarded body
    | If { cond = c, _; body } ->
        scalar ~loops ~path:(Affine.Cond :: path) ~guarded ~arg_c ?stmt_write:None c;
        code ~loops ~path:(Affine.Body :: path) ~guarded:true body
    | Zero_out tn -> add ~loops ~path:(Affine.Write :: path) ~guarded ~whole:true ~write:true tn [||]
    | Set { tn; idcs; llsc; _ } ->
        scalar ~loops ~path:(Affine.Rhs :: path) ~guarded ~arg_c ~stmt_write:idcs llsc;
        add ~loops ~path:(Affine.Write :: path) ~guarded ~rmw:(reads_tn tn.Tn.uid llsc)
          ~val_syms:(scalar_syms llsc) ~write:true tn idcs
    | Set_dynamic { tn; idcs; dyn_value = v, _; llsc; _ } ->
        scalar ~loops ~path:(Affine.Rhs :: path) ~guarded ~arg_c ~stmt_write:idcs v;
        scalar ~loops ~path:(Affine.Rhs :: path) ~guarded ~arg_c ~stmt_write:idcs llsc;
        add ~loops ~path:(Affine.Write :: path) ~guarded ~dynamic:true
          ~rmw:(reads_tn tn.Tn.uid llsc || reads_tn tn.Tn.uid v)
          ~val_syms:(scalar_syms v @ scalar_syms llsc)
          ~write:true tn idcs
    | Set_from_vec { tn; idcs; length; arg = a, _; _ } ->
        scalar ~loops ~path:(Affine.Rhs :: path) ~guarded ~arg_c ~stmt_write:idcs a;
        add ~loops ~path:(Affine.Write :: path) ~guarded ~vec_len:length ~rmw:(reads_tn tn.Tn.uid a)
          ~val_syms:(scalar_syms a) ~write:true tn idcs
    | Set_local (_, llsc) ->
        scalar ~loops ~path:(Affine.Rhs :: path) ~guarded ~arg_c ?stmt_write:None llsc
    | Tile_mma { fallback; _ } -> code ~loops ~path ~guarded fallback
  and scalar ~loops ~path ~guarded ~arg_c ?stmt_write (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } ->
        let k = !arg_c in
        Int.incr arg_c;
        code ~loops ~path:(Affine.Arg k :: path) ~guarded body
    | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get (tn, idcs) -> add ~loops ~path ~guarded ?stmt_write ~write:false tn idcs
    | Get_dynamic { tn; idcs; dyn_value = v, _; _ } ->
        add ~loops ~path ~guarded ~dynamic:true ?stmt_write ~write:false tn idcs;
        scalar ~loops ~path ~guarded ~arg_c ?stmt_write v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar ~loops ~path ~guarded ~arg_c ?stmt_write a;
        scalar ~loops ~path ~guarded ~arg_c ?stmt_write b;
        scalar ~loops ~path ~guarded ~arg_c ?stmt_write c
    | Binop (op, (a, _), (b, _)) -> (
        (* The dead operand of a projection is never evaluated (codegen renders only the used one),
           so it contributes no access; a gated operand can evaluate, and these accesses are the
           guards-taken upper bound — both per {!Ops.binop_conditionality}. *)
        match Ops.binop_conditionality op with
        | Ops.Only_first -> scalar ~loops ~path ~guarded ~arg_c ?stmt_write a
        | Ops.Only_second -> scalar ~loops ~path ~guarded ~arg_c ?stmt_write b
        | Ops.Both_operands | Ops.Gated_second ->
            scalar ~loops ~path ~guarded ~arg_c ?stmt_write a;
            scalar ~loops ~path ~guarded ~arg_c ?stmt_write b)
    | Unop (_, (a, _)) -> scalar ~loops ~path ~guarded ~arg_c ?stmt_write a
  in
  code ~loops:[] ~path:[] ~guarded:false llc;
  List.rev !acc

(** Calls [touch] on every tnode whose context buffer the statement reads or writes, [on_opaque] on
    [Staged_compilation] (its accesses cannot be enumerated). Scope-local reads/writes
    ([Get_local]/[Set_local]) own no buffer and are skipped; [Local_scope] bodies are descended
    into, since inlined virtual computations read materialized nodes; [Get_merge_buffer] reads the
    merge buffer, not the node's context buffer. The single source of buffer-access truth for the
    gh-ocannl-489 liveness passes ({!buffer_access_spans}, {!sink_zero_outs}). *)
let iter_buffer_accesses ~(touch : Tn.t -> unit) ~(on_opaque : unit -> unit) (c : t) : unit =
  let rec scal (sc : scalar_t) : unit =
    match sc with
    | Local_scope { body; id = _; orig_indices = _; mint = _ } -> stmt body
    | Get_local _ -> ()
    | Get (tn, _) -> touch tn
    | Get_dynamic { tn; dyn_value = dv, _; idcs = _; dyn_axis = _ } ->
        touch tn;
        scal dv
    | Get_merge_buffer (_, _) -> (* Reads the merge buffer, not the node's context buffer. *) ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scal a;
        scal b;
        scal c
    | Binop (_, (a, _), (b, _)) ->
        scal a;
        scal b
    | Unop (_, (a, _)) -> scal a
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
  and stmt (c : t) : unit =
    match c with
    | Noop | Comment _ -> ()
    | Staged_compilation _ -> on_opaque ()
    | Seq (t1, t2) ->
        stmt t1;
        stmt t2
    | For_loop { body; _ } -> stmt body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> scal c.init);
        stmt body
    | Zero_out tn -> touch tn
    | Set { tn; llsc; idcs = _; debug = _ } ->
        touch tn;
        scal llsc
    | Set_dynamic { tn; dyn_value = dv, _; llsc; idcs = _; dyn_axis = _; debug = _ } ->
        touch tn;
        scal dv;
        scal llsc
    | Set_from_vec { tn; arg = a, _; idcs = _; length = _; vec_unop = _; debug = _ } ->
        touch tn;
        scal a
    | Set_local (_, sc) -> scal sc
    | Declare_local _ | Workgroup_barrier -> ()
    | If { cond = cond, _; body } ->
        scal cond;
        stmt body
    | Tile_mma { d = d, _; a = a, _; b = b, _; fallback; _ } ->
        touch d;
        touch a;
        touch b;
        stmt fallback
  in
  stmt c

(** gh-ocannl-489 liveness-based buffer aliasing: per-tnode access span over the final (post-
    schedule, post-fission) code of a routine, as a closed interval of positions. [segments] are the
    routine's kernels in execution order (a singleton list when the routine was not fissioned).

    Position granularity is the soundness crux. With [stmt_serial:true] every top-level statement of
    every segment gets its own position: valid only for backends where consecutive top-level
    statements of one compiled procedure are fully synchronized (the C backends — parallel [Grid]
    dispatches join before the next statement). With [stmt_serial:false] all statements of a segment
    share one position: on GPU backends a kernel's top-level statements have no grid-wide
    synchronization between them (that lack is exactly where fission cuts), so only segment
    boundaries — device-ordered by the fissioned routine's event chain — separate lifetimes.

    Returns [None] when the code contains [Staged_compilation]: its accesses are opaque to
    {!iter_buffer_accesses}, so no aliasing plan can be trusted. *)
let buffer_access_spans ~stmt_serial (segments : t list) : (Tn.t, int * int) Base.Hashtbl.t option =
  let spans = Hashtbl.create (module Tn) in
  let opaque = ref false in
  let pos = ref 0 in
  let touch tn =
    Hashtbl.update spans tn ~f:(function None -> (!pos, !pos) | Some (lo, _) -> (lo, !pos))
  in
  List.iter segments ~f:(fun seg ->
      List.iter (flat_lines [ seg ]) ~f:(fun line ->
          iter_buffer_accesses ~touch ~on_opaque:(fun () -> opaque := true) line;
          if stmt_serial then Int.incr pos);
      if not stmt_serial then Int.incr pos);
  if !opaque then None else Some spans

(** gh-ocannl-489 follow-up: sink each top-level [Zero_out] to just before the first later top-level
    statement that accesses the zeroed node. [Train.grad_update] emits every gradient's [Zero_out]
    in one up-front block ([loss.zero_grads]), which starts all the gradients' live spans at that
    block, so the backprop chain's intervals nest instead of being disjoint and
    {!buffer_access_spans} finds nothing for the arena planner to overlay; after sinking, a
    gradient's span starts at its first accumulation.

    Reordering soundness: a [Zero_out] commutes with any statement that does not access the zeroed
    node's buffer (per {!iter_buffer_accesses}, the same fold the liveness planner trusts). It never
    moves past an access of the node, a [Staged_compilation] (opaque accesses), or a
    [Workgroup_barrier] (cross-workgroup ordering). A [Zero_out] never re-accessed in-routine (an
    export initializing the node for later routines) stays in place. Runs on the whole-routine code
    BEFORE scheduling/fission, so segment cuts and cross-nest merges see the sunk order. *)
let sink_zero_outs (llc : t) : t =
  let lines = Array.of_list (flat_lines [ llc ]) in
  let n = Array.length lines in
  let accesses =
    Array.map lines ~f:(fun line ->
        let s = ref (Set.empty (module Tn)) in
        let opaque = ref false in
        iter_buffer_accesses line
          ~touch:(fun tn -> s := Set.add !s tn)
          ~on_opaque:(fun () -> opaque := true);
        (!s, !opaque))
  in
  let barrier i =
    snd accesses.(i) || match lines.(i) with Workgroup_barrier -> true | _ -> false
  in
  (* [inserts.(j)]: sunk [Zero_out] lines re-emitted just before line [j], in original order. *)
  let inserts = Array.create ~len:(n + 1) [] in
  let moved = Array.create ~len:n false in
  Array.iteri lines ~f:(fun i line ->
      match line with
      | Zero_out tn -> (
          let rec find j =
            if j >= n then None
            else if Set.mem (fst accesses.(j)) tn || barrier j then Some j
            else find (j + 1)
          in
          match find (i + 1) with
          | Some j when j > i + 1 ->
              moved.(i) <- true;
              inserts.(j) <- line :: inserts.(j)
          | _ -> ())
      | _ -> ());
  if not (Array.exists moved ~f:Fn.id) then llc
  else
    unflat_lines
      (List.concat
         (List.init (n + 1) ~f:(fun j ->
              List.rev inserts.(j) @ if j < n && not moved.(j) then [ lines.(j) ] else [])))

(* gh-343: recognize the in-range guard's reduction body. Matches the two semantically-equivalent
   one-hot selectors over loop variable [k]: - [Where (Cmpeq (Embed_index (Iterator k), index_expr),
   table_get, Constant 0.)] (either operand order of [Cmpeq]); - the multiply form [Binop (Mul,
   <cmpeq 0/1>, table_get)] (either factor order). On success returns [Some (table, table_idcs,
   index_expr)] where [index_expr] is the scalar value used as the dynamic index. *)
(* Match a Cmpeq comparing [Embed_index (Iterator k)] against an index expression free of [k].
   Returns the index expression (with its precision). Uses [is_embedded_range_iterator] defined
   above, sharing the recognition logic with [is_one_hot_selector_assignment]. *)
let match_one_hot_cmpeq (k : Indexing.symbol) : scalar_t -> scalar_arg option = function
  | Binop (Ops.Cmpeq, (a, pa), (b, pb)) ->
      if is_embedded_range_iterator k a && not (scalar_mentions_symbol k b) then Some (b, pb)
      else if is_embedded_range_iterator k b && not (scalar_mentions_symbol k a) then Some (a, pa)
      else None
  | _ -> None

let match_one_hot_contribution (k : Indexing.symbol) (contribution : scalar_t) :
    (Tn.t * Indexing.axis_index array * scalar_arg) option =
  let match_cmpeq = match_one_hot_cmpeq k in
  let as_table_get = function Get (table, table_idcs) -> Some (table, table_idcs) | _ -> None in
  match contribution with
  | Ternop (Ops.Where, cond, (then_, _), (Constant 0., _)) -> (
      match (match_cmpeq (fst cond), as_table_get then_) with
      | Some index_expr, Some (table, table_idcs) -> Some (table, table_idcs, index_expr)
      | _ -> None)
  | Binop (Ops.Mul, (a, _), (b, _)) -> (
      (* one factor is the comparison, the other is the table read *)
      match (match_cmpeq a, as_table_get b) with
      | Some index_expr, Some (table, table_idcs) -> Some (table, table_idcs, index_expr)
      | _ -> (
          match (match_cmpeq b, as_table_get a) with
          | Some index_expr, Some (table, table_idcs) -> Some (table, table_idcs, index_expr)
          | _ -> None))
  | _ -> None

(* gh-466: recognize the transposed one-hot contribution — the embedding-table gradient. Matches the
   same two selector forms as [match_one_hot_contribution], but where the selected operand is an
   arbitrary scalar [g] (the incoming gradient) instead of a table read: - [Where (Cmpeq
   (Embed_index (Iterator k), index_expr), g, Constant 0.)]; - the multiply form [Binop (Mul, <cmpeq
   0/1>, g)] (either factor order). On success returns [Some (index_expr, g_arg)]; the caller checks
   that [g] is free of [k] and does not read the written tensor. *)
let match_transposed_one_hot_contribution (k : Indexing.symbol) (contribution : scalar_t) :
    (scalar_arg * scalar_arg) option =
  let match_cmpeq = match_one_hot_cmpeq k in
  match contribution with
  | Ternop (Ops.Where, cond, then_, (Constant 0., _)) ->
      Option.map (match_cmpeq (fst cond)) ~f:(fun index_expr -> (index_expr, then_))
  | Binop (Ops.Mul, a, b) -> (
      match match_cmpeq (fst a) with
      | Some index_expr -> Some (index_expr, b)
      | None -> Option.map (match_cmpeq (fst b)) ~f:(fun index_expr -> (index_expr, a)))
  | _ -> None

(* gh-343: in-range guard conjuncts for the dynamic index of a gather ([build_guarded_gather]) or,
   gh-466, a scatter ([build_guarded_scatter]). [class_count] is the size of the table axis being
   dynamically indexed.

   The guard is constructed generically -- a lower-bound conjunct, an upper-bound conjunct, and an
   integrality conjunct -- and interval analysis erases each conjunct it can prove
   (docs/proposals/interval-analysis-scalar-t.md, the designated re-expression target). The
   precision facts alone reproduce the previous hand-written flavors as emergent behavior: an
   unsigned index precision proves the lower bound (its machine range starts at 0), and any integer
   precision proves integrality. Settled [Tnode] bounds can additionally discharge the upper bound
   (and the lower bound for signed indices), leaving a bare gather.

   Fail-safe construction (binding constraint 1): folding is an optimization, never a correctness
   obligation. Every conjunct that can be emitted is correct unfolded at its guard precision, at
   every guard precision -- no conjunct depends on being folded away.

   Guard precision by index flavor (unchanged from the hand-written version): unsigned compares in
   uint64 (zero-extension is lossless; casting to int64 instead could map huge values to negatives
   and pass the upper bound); signed compares in int64 (exact and native on all backends, including
   Metal which lacks double); float compares in double (exact for the integer-valued indices in
   scope). Guards take one canonical shape per role: the lower bound is a direct [0 <= idx], the
   upper bound the strict [idx < class_count]; integrality is [idx == trunc(idx)] (the backend casts
   [idx] to int when gathering, so for non-integer indices, where every [k == idx] is false and the
   reduction is 0, the gather must not fire). *)
let guard_conjuncts ~ienv ~(index_expr : scalar_arg) ~class_count : scalar_t list * Ops.prec =
  let iv, iprec = index_expr in
  let guard_prec, unsigned_guard, prec_proves_integral =
    match iprec with
    | Ops.Byte_prec _ | Ops.Uint16_prec _ | Ops.Uint32_prec _ | Ops.Uint64_prec _ ->
        (Ops.uint64, true, true)
    | Ops.Int32_prec _ | Ops.Int64_prec _ -> (Ops.int64, false, true)
    | _ -> (Ops.double, false, false)
  in
  (* Bounds of the index value as evaluated at the guard precision (the conjuncts are homogeneous
     binops, so this is exactly what the comparisons see). *)
  let ivr = interval_of ~ienv ~prec:guard_prec iv in
  let lower_proved =
    Float.(ivr.ival.Interval.lo >= 0.)
    (* outward endpoints: sound *)
  in
  (* The gather compares against [class_count], an exact small integer. *)
  let upper_proved = Float.(ivr.ival.Interval.hi < Float.of_int class_count) in
  let integral_proved = prec_proves_integral || ivr.ival.Interval.integral in
  (* An unsigned guard precision proves the lower bound by construction (its machine range starts at
     0), so the lower conjunct is always erased there rather than emitted vacuously. *)
  assert ((not unsigned_guard) || lower_proved);
  let lower =
    if lower_proved then None
    else Some (Binop (Ops.Cmple, (Constant 0., guard_prec), (iv, guard_prec)))
  in
  let upper =
    if upper_proved then None
    else
      Some (Binop (Ops.Cmplt, (iv, guard_prec), (Constant (Float.of_int class_count), guard_prec)))
  in
  let is_integral =
    if integral_proved then None
    else
      Some (Binop (Ops.Cmpeq, (iv, guard_prec), (Unop (Ops.Trunc, (iv, guard_prec)), guard_prec)))
  in
  let conjuncts = List.filter_opt [ lower; upper; is_integral ] in
  let dropped_any = Option.is_none lower || Option.is_none upper || Option.is_none is_integral in
  (* Any conjunct erased while tensor-node bounds narrowed the interval settles those bounds
     (binding constraint 2; over-settling when a drop was justified by the precision alone is
     harmless -- it only locks in bounds that later writers must respect). *)
  if dropped_any && not (Set.is_empty ivr.srcs) then settle_srcs ivr;
  (conjuncts, guard_prec)

let conjoin ~guard_prec conjuncts =
  match conjuncts with
  | [] -> None
  | first :: rest ->
      Some
        (List.fold rest ~init:first ~f:(fun acc c ->
             Binop (Ops.And, (acc, guard_prec), (c, guard_prec))))

let build_guarded_gather ~ienv ~table ~table_idcs ~dyn_axis ~(index_expr : scalar_arg) ~class_count
    ~value_prec : scalar_t =
  let gather = Get_dynamic { tn = table; idcs = table_idcs; dyn_axis; dyn_value = index_expr } in
  let conjuncts, guard_prec = guard_conjuncts ~ienv ~index_expr ~class_count in
  match conjoin ~guard_prec conjuncts with
  | None -> gather (* Fully discharged: the (settled) bounds prove every access in range. *)
  | Some in_range ->
      Ternop (Ops.Where, (in_range, guard_prec), (gather, value_prec), (Constant 0., value_prec))

(* gh-466: build the guarded scatter-accumulate replacing a matched transposed one-hot reduction:
   [if in_range(e) then tn[.., e @dyn_axis, ..] += g]. The write reads back its own cell via
   [Get_dynamic] (an explicit read-modify-write), so the accumulation is visible to read-tracking
   and [has_accumulation]. When interval analysis discharges every guard conjunct the [If] wrapper
   is omitted. *)
let build_guarded_scatter ~ienv ~tn ~idcs ~dyn_axis ~(index_expr : scalar_arg)
    ~(grad_arg : scalar_arg) ~class_count ~debug : t =
  let value_prec = Lazy.force tn.Tn.storage_prec in
  let gather = Get_dynamic { tn; idcs; dyn_axis; dyn_value = index_expr } in
  let scatter =
    Set_dynamic
      {
        tn;
        idcs;
        dyn_axis;
        dyn_value = index_expr;
        llsc = Binop (Ops.Add, (gather, value_prec), grad_arg);
        debug;
      }
  in
  let conjuncts, guard_prec = guard_conjuncts ~ienv ~index_expr ~class_count in
  match conjoin ~guard_prec conjuncts with
  | None -> scatter
  | Some in_range -> If { cond = (in_range, guard_prec); body = scatter }

(* gh-343: peel a possible zero-initializer off a reduction body, returning the inner [For_loop]. *)
let strip_zero_init_for_local (id : scope_id) (body : t) : t option =
  match body with
  | Seq (Set_local (id', Constant 0.), (For_loop _ as fl)) when equal_scope_id id id' -> Some fl
  | For_loop _ -> Some body
  | _ -> None

(* gh-ocannl-639: whether a scalar (resp. a statement body) reads or writes [tn] anywhere — the
   accumulation recognizers below use it to certify that an update's contribution is free of the
   accumulator's node, which is what licenses holding the accumulator out of memory across a
   reduction. *)
let rec scalar_touches_tn tn (llsc : scalar_t) =
  match llsc with
  | Get (tn2, _) -> Tn.equal tn tn2
  | Get_merge_buffer _ ->
      (* A node's merge buffer is a SEPARATE read-only staging buffer (the transfer source's copy),
         never the node's own storage — reading it is independent of the accumulator's cell, so [p
         =+ p.merge]-style updates stay recognizable as accumulations. *)
      false
  | Get_dynamic { tn = tn2; dyn_value = v, _; _ } -> Tn.equal tn tn2 || scalar_touches_tn tn v
  | Local_scope { body; _ } -> code_touches_tn tn body
  | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> false
  | Ternop (_, (a, _), (b, _), (c, _)) ->
      scalar_touches_tn tn a || scalar_touches_tn tn b || scalar_touches_tn tn c
  | Binop (_, (a, _), (b, _)) -> scalar_touches_tn tn a || scalar_touches_tn tn b
  | Unop (_, (a, _)) -> scalar_touches_tn tn a

and code_touches_tn tn (llc : t) =
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ -> false
  | Seq (a, b) -> code_touches_tn tn a || code_touches_tn tn b
  | For_loop { body; _ } -> code_touches_tn tn body
  | Scan_loop { carried; body; _ } ->
      List.exists carried ~f:(fun c -> scalar_touches_tn tn c.init) || code_touches_tn tn body
  | If { cond = c, _; body } -> scalar_touches_tn tn c || code_touches_tn tn body
  | Zero_out tn2 -> Tn.equal tn tn2
  | Set { tn = tn2; llsc; _ } -> Tn.equal tn tn2 || scalar_touches_tn tn llsc
  | Set_dynamic { tn = tn2; dyn_value = v, _; llsc; _ } ->
      Tn.equal tn tn2 || scalar_touches_tn tn v || scalar_touches_tn tn llsc
  | Set_from_vec { tn = tn2; arg = a, _; _ } -> Tn.equal tn tn2 || scalar_touches_tn tn a
  | Set_local (_, llsc) -> scalar_touches_tn tn llsc
  (* The fallback is what a backend without an MMA hook executes, so what it touches is touched. *)
  | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; fallback; _ } ->
      Tn.equal tn d_tn || Tn.equal tn a_tn || Tn.equal tn b_tn || code_touches_tn tn fallback

(* gh-ocannl-639: the accumulation-update statement shape [tn[idcs] = op(tn[idcs], contrib)] (or its
   FMA form) over an associative-commutative [op], with [contrib] free of [tn]. The single source of
   truth for [C_syntax]'s widened renderings (the serial-fallback nest rewrite and its siblings) and
   for [Schedule.Unroll ~materialize:true]'s scope-form unrolling — sharing it is what keeps "what
   counts as an accumulation" from drifting between the schedule transform and the emission that
   must honor it. *)
let accum_update_parts ~tn ~idcs (llsc : scalar_t) : (Ops.binop * scalar_t) option =
  let is_acc s = equal_scalar_t s (Get (tn, idcs)) in
  let reduce_op = function Ops.Add | Ops.Mul | Ops.Max | Ops.Min -> true | _ -> false in
  match llsc with
  | Binop (op, (a, _), (b, _)) when reduce_op op && is_acc a && not (scalar_touches_tn tn b) ->
      Some (op, b)
  | Binop (op, (a, _), (b, _)) when reduce_op op && is_acc b && not (scalar_touches_tn tn a) ->
      Some (op, a)
  | Ternop (Ops.FMA, (a, pa), (b, pb), (c, _))
    when is_acc c && (not (scalar_touches_tn tn a)) && not (scalar_touches_tn tn b) ->
      Some (Ops.Add, Binop (Ops.Mul, (a, pa), (b, pb)))
  | _ -> None

(* Retarget an accumulation update's read of [tn[idcs]] to the scope local [id] — the shapes
   [accum_update_parts] admits only carry the accumulator read as a direct operand of the top
   operator. *)
let subst_accum_read ~tn ~idcs ~id (llsc : scalar_t) : scalar_t =
  let subst ((s, p) : scalar_arg) : scalar_arg =
    if equal_scalar_t s (Get (tn, idcs)) then (Get_local id, p) else (s, p)
  in
  match llsc with
  | Binop (op, a, b) -> Binop (op, subst a, subst b)
  | Ternop (op, a, b, c) -> Ternop (op, subst a, subst b, subst c)
  | _ ->
      (* [accum_update_parts] admits only the two shapes above. *)
      assert false

(* A guard reading only embedded indices and constants — the gh-490 symbolic-extent shape [If (i <
   s)] and its constant-bound sibling (Schedule.Pad's leaf guards). Such a guard commutes with an
   accumulator's init and store: it gates which updates run, and for a cell whose guard never fires
   a widen/narrow round-trip is exact on every narrow-float value. Data-dependent guards are NOT of
   this shape and stay opaque to the peel below. *)
let pure_index_guard (llsc : scalar_t) =
  let operand = function Embed_index _ | Constant _ -> true | _ -> false in
  match llsc with
  | Binop ((Ops.Cmplt | Ops.Cmple), (a, _), (b, _)) -> operand a && operand b
  | _ -> false

(* Whether a scalar reads the scope local [id], descending into nested scope bodies — certifies that
   an update's contribution is free of the local it updates (the scope-local counterpart of
   {!scalar_touches_tn}). *)
let rec scalar_reads_scope ~id (s : scalar_t) =
  match s with
  | Get_local id' -> Scope_id.equal id id'
  | Local_scope { body; _ } -> code_reads_scope ~id body
  | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> false
  | Get_dynamic { dyn_value = v, _; _ } -> scalar_reads_scope ~id v
  | Ternop (_, (a, _), (b, _), (c, _)) ->
      scalar_reads_scope ~id a || scalar_reads_scope ~id b || scalar_reads_scope ~id c
  | Binop (_, (a, _), (b, _)) -> scalar_reads_scope ~id a || scalar_reads_scope ~id b
  | Unop (_, (a, _)) -> scalar_reads_scope ~id a

and code_reads_scope ~id (llc : t) =
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _ ->
      false
  | Seq (a, b) -> code_reads_scope ~id a || code_reads_scope ~id b
  | For_loop { body; _ } -> code_reads_scope ~id body
  | Scan_loop { carried; body; _ } ->
      List.exists carried ~f:(fun c -> Scope_id.equal id c.next || scalar_reads_scope ~id c.init)
      || code_reads_scope ~id body
  | If { cond = c, _; body } -> scalar_reads_scope ~id c || code_reads_scope ~id body
  | Set { llsc; _ } | Set_local (_, llsc) -> scalar_reads_scope ~id llsc
  | Set_dynamic { dyn_value = v, _; llsc; _ } ->
      scalar_reads_scope ~id v || scalar_reads_scope ~id llsc
  | Set_from_vec { arg = a, _; _ } -> scalar_reads_scope ~id a
  (* The fallback is what a backend without an MMA hook executes, so a local it reads is read. *)
  | Tile_mma { fallback; _ } -> code_reads_scope ~id fallback

(* The reduce-shaped update of a scope LOCAL: [local = op(local, contrib)] (or its FMA form) with
   [contrib] free of the local — [subst_accum_read]'s output shape. The [`Scope] arm of the peel
   below accepts a base only when every update fits this grammar, because hoisting an enclosing loop
   into a scope is licensed by the reduction reading alone: a general recurrence through the local
   (a subtraction, a scaled update) narrows per enclosing iteration by the source's own semantics,
   and holding it wide would change values outside the accumulator-width policy. *)
let accum_local_update_parts ~id (llsc : scalar_t) =
  let is_acc = function Get_local id' -> Scope_id.equal id id' | _ -> false in
  let reads_local = scalar_reads_scope ~id in
  let reduce_op = function Ops.Add | Ops.Mul | Ops.Max | Ops.Min -> true | _ -> false in
  match llsc with
  | Binop (op, (a, _), (b, _)) when reduce_op op && is_acc a && not (reads_local b) -> Some (op, b)
  | Binop (op, (a, _), (b, _)) when reduce_op op && is_acc b && not (reads_local a) -> Some (op, a)
  | Ternop (Ops.FMA, (a, pa), (b, pb), (c, _))
    when is_acc c && (not (reads_local a)) && not (reads_local b) ->
      Some (Ops.Add, Binop (Ops.Mul, (a, pa), (b, pb)))
  | _ -> None

(* A scalar reading only embedded indices and constants — the semantic notion behind
   {!pure_index_guard}, closed over the index arithmetic ([And]-joined range conditions, [Cmpeq]
   unit-solve conditions) that virtualization's guarded reads build. Such an expression cannot
   observe any precision residency. *)
let rec index_only_scalar (llsc : scalar_t) =
  match llsc with
  | Embed_index _ | Constant _ | Constant_bits _ -> true
  | Get _ | Get_local _ | Get_merge_buffer _ | Get_dynamic _ | Local_scope _ -> false
  | Ternop (_, (a, _), (b, _), (c, _)) ->
      index_only_scalar a && index_only_scalar b && index_only_scalar c
  | Binop (_, (a, _), (b, _)) -> index_only_scalar a && index_only_scalar b
  | Unop (_, (a, _)) -> index_only_scalar a

(* The reduction operator of a scope-local update in EITHER spelling: the plain
   [accum_local_update_parts] form, or virtualization's guarded-read form — [Set_local (id, Where
   (index-only cond, update, Get_local id))], possibly nested per condition ([inline_computation]
   folds one [Where] per range/unit-solve guard). The guarded form is a reduction for RESIDENCY
   purposes (gh-ocannl-663): the condition observes no precision, the off-condition arm carries the
   accumulator through unchanged, and the on-condition arm is the reduce-shaped update — so the
   accumulator-scope census accepts it where treating the guarded self-read as a recurrence would
   leave a virtualized reduction narrow while its materialized serial twin widens
   (placement-dependent width). Deliberately NOT merged into {!accum_local_update_parts}: its [(op,
   contrib)] decomposition licenses consumers (the SIMD folding, [subst_accum_read]-style rewrites)
   to rebuild an unguarded [op(local, contrib)], which the guarded form is not; and not into
   {!scope_updates_reduce_op}: that is the HOIST license, and hoisting a guarded update across
   further levels is a separate question from what width its accumulator resides at. *)
let accum_local_update_op ~id (llsc : scalar_t) : Ops.binop option =
  let rec go llsc =
    match llsc with
    | Ternop (Ops.Where, (c, _), (t, _), (Get_local id', _))
      when Scope_id.equal id id' && index_only_scalar c ->
        go t
    | _ -> Option.map (accum_local_update_parts ~id llsc) ~f:fst
  in
  go llsc

(* Whether a scope body's update statements (everything after the opening init) fit the grammar the
   scope-form mint emits: Serial/[Unrolled]/[Vectorized] loops, pure-index-guarded [If]s, comments,
   and reduce-shaped [Set_local]s of the scope's own local — all carrying ONE reduction operator
   (the FMA form counting as [Add]). Anything else — another node's write, a nested scope, a
   non-reduction recurrence, or individually-valid updates under MIXED operators (e.g. [local += x;
   local *= y], which is not a reduction and whose per-iteration narrowing is the source's
   semantics) — disqualifies the base from hoisting. Returns the uniform operator. *)
let scope_updates_reduce_op ~id (llc : t) : Ops.binop option =
  let rec go op_acc llc =
    match llc with
    | Noop | Comment _ -> Some op_acc
    | Seq (a, b) -> Option.bind (go op_acc a) ~f:(fun acc -> go acc b)
    | For_loop { body; axis = Serial | Unrolled | Vectorized; _ } -> go op_acc body
    | If { cond = gc, _; body } -> if pure_index_guard gc then go op_acc body else None
    | Set_local (id', v) when Scope_id.equal id id' -> (
        match accum_local_update_parts ~id v with
        | Some (op, _) -> (
            match op_acc with
            | None -> Some (Some op)
            | Some op0 -> if Ops.equal_binop op0 op then Some op_acc else None)
        | None -> None)
    | _ -> None
  in
  match go None llc with Some (Some op) -> Some op | _ -> None

(* gh-ocannl-639: peel a single-statement reduction nest down to its accumulation base. Levels are
   Serial/[Unrolled] loops and {!pure_index_guard}ed [If]s, each containing nothing else (comments
   aside); the base is either a raw accumulation update ([`Update]: an {!accum_update_parts}-shaped
   [Set] whose cell is invariant across the peeled levels) or the scope form a previous rewrite
   minted ([`Scope]: a [Set] whose value is a [Local_scope] opening with the init [Set_local (id,
   Get (tn, idcs))], returned as the id and the update statements after the init — reusing the id is
   what lets a consumer hoist the scope through further levels). Returns [(tn, idcs, base, debug,
   rebuild)] where [rebuild] re-wraps a replacement base statement in the peeled levels. The ONE
   definition shared by [C_syntax]'s widened serial fallback and the scope-form mints of
   [Schedule.Unroll ~materialize:true] and [Schedule.Partition], so the transforms and the emission
   cannot drift in what nests they recognize. Guard legality — whether the peeled levels may include
   a given [If] at all — is asked of {!Affine.peel_guard} and {!Affine.separates} rather than
   decided here (gh-ocannl-722); [loop_bounds] is {!loop_bounds} of the enclosing program, which is
   what classifies a guard symbol as an enclosing level, a peeled one, or an index bound outside
   every loop.

   Deliberately single-statement: a fused body updating several distinct accumulators is out of
   scope — no lowering or schedule op produces one (each [Assignments] accumulation lowers to its
   own nest; [Fuse_epilogue] folds elementwise tails, not sibling reductions), so no candidate pair
   can diverge on it; and a multi-accumulator hoist could not be a [Local_scope] value at all (scope
   purity forbids a sibling [Set] inside a scope body) — it would need the [Declare_local] statement
   form. A transform that starts minting fused reduction bodies must extend this peel alongside. *)
(* The scope-form accumulation base {!peel_accum_nest} recognizes: a [Local_scope] over [tn.idcs]
   whose body OPENS by loading that very cell into the scope local, whose remaining statements carry
   one reduction operator, and which touches the node nowhere else. Returns those statements, which
   are what a consumer re-wraps. Factored out so that the census's notion of a reduction site
   (gh-ocannl-733) is the peel's own, and cannot drift from it. *)
let scope_accum_updates ~tn ~idcs ~id sbody =
  match
    List.filter (flat_lines [ sbody ]) ~f:(function Noop | Comment _ -> false | _ -> true)
  with
  | Set_local (id', Get (tn', idcs')) :: (_ :: _ as rest)
    when Scope_id.equal id id' && Tn.equal tn tn'
         && Array.length idcs = Array.length idcs'
         && Array.for_all2_exn idcs idcs' ~f:Indexing.equal_axis_index
         && (not (List.exists rest ~f:(code_touches_tn tn)))
         && Option.is_some (scope_updates_reduce_op ~id (unflat_lines rest)) ->
      Some rest
  | _ -> None

(* gh-ocannl-733: whether a tree holds a SELF-RECURRENCE — some [Set] whose value reads the very
   cell it writes. This is the census's notion of "localization was a live question here", and it is
   deliberately narrower than {!has_accumulation} in two ways: the recurrence is on the CELL rather
   than on the node, and a [Local_scope] counts only when its body actually reads that cell, where
   [has_accumulation] counts every scope by conservative assumption (it must, being the predicate
   that decides whether iteration independence may be asserted). Censusing on the conservative one
   recorded every non-reduction virtualized scope in a loop as a declined reduction site, which
   inflates the decline count and lets a "the census is non-empty and nothing localized" claim pass
   over a routine with no reduction in it (Codex P2, round 2). *)
(* Whether a value reads the cell [tn[idcs]] — the recurrence test of {!has_accumulating_cell}. *)
let rec reads_cell ~tn ~idcs (sc : scalar_t) =
  let arg (s, _prec) = reads_cell ~tn ~idcs s in
  match sc with
  | Get (tn', idcs') ->
      Tnode.equal tn tn'
      && Array.length idcs = Array.length idcs'
      && Array.for_all2_exn idcs idcs' ~f:Indexing.equal_axis_index
  (* A scope NESTED inside a larger value — [a[i] = f(scope { … a[i] … })] — is a recurrence like
     any other read; the scope that IS the written value is the case above, judged by its shape. *)
  | Local_scope { body; _ } -> stmt_reads_cell ~tn ~idcs body
  (* The runtime slot cannot prove disjointness, but unequal fixed slots elsewhere can
     (gh-ocannl-960). Symbolic slots may alias even when their expressions differ; a rank mismatch
     is unknown too. The selector can read the written cell independently of the gathered cell. *)
  | Get_dynamic { tn = tn'; idcs = idcs'; dyn_axis; dyn_value } ->
      Tnode.equal tn tn'
      && (Array.length idcs <> Array.length idcs'
         || not
              (Array.existsi idcs ~f:(fun k a ->
                   k <> dyn_axis
                   &&
                   match (a, idcs'.(k)) with
                   | Indexing.Fixed_idx x, Indexing.Fixed_idx y -> x <> y
                   | _ -> false)))
      || arg dyn_value
  | Ternop (_, a, b, c) -> arg a || arg b || arg c
  | Binop (op, a, b) -> (
      (* A projection's discarded operand is never rendered, hence never reads the cell (the same
         rule [scalar_reads_merge_buffer] applies); a gated second operand may evaluate. *)
      match Ops.binop_conditionality op with
      | Ops.Only_first -> arg a
      | Ops.Only_second -> arg b
      | Ops.Both_operands | Ops.Gated_second -> arg a || arg b)
  | Unop (_, a) -> arg a
  | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> false

and stmt_reads_cell ~tn ~idcs (llc : t) =
  match llc with
  | Seq (a, b) -> stmt_reads_cell ~tn ~idcs a || stmt_reads_cell ~tn ~idcs b
  (* A guard's condition reads too: [If (a[i] == 0) local = 1] inside the scope that writes [a[i]]
     is a read of the cell the scope writes back. *)
  | If { cond = c, _; body } -> reads_cell ~tn ~idcs c || stmt_reads_cell ~tn ~idcs body
  | For_loop { body; _ } -> stmt_reads_cell ~tn ~idcs body
  (* A scan's carried initializers are reads the loop performs before its first iteration. *)
  | Scan_loop { carried; body; _ } ->
      List.exists carried ~f:(fun c -> reads_cell ~tn ~idcs c.init)
      || stmt_reads_cell ~tn ~idcs body
  | Set { llsc; _ } | Set_local (_, llsc) -> reads_cell ~tn ~idcs llsc
  | _ -> false

let has_accumulating_cell (llc : t) : bool =
  let rec loop (llc : t) =
    match llc with
    | Seq (a, b) -> loop a || loop b
    | If { body; _ } | For_loop { body; _ } -> loop body
    (* A scope over the written cell is judged by {!scope_accum_updates}, not by whether it reads
       that cell: EVERY scope opens by loading the cell it will write back, so a plain virtualized
       computation reads its own cell in its initializer and would count as a recurrence on that
       basis alone (Codex P2, round 3). What makes it one is the rest of the body carrying a
       scope-local accumulation, which is exactly what the recognizer asks. *)
    | Set { tn; idcs; llsc = Local_scope { id; body = sbody; orig_indices = _; mint = _ }; _ } ->
        Option.is_some (scope_accum_updates ~tn ~idcs ~id sbody)
    | Set { tn; idcs; llsc; _ } -> reads_cell ~tn ~idcs llsc
    | _ -> false
  in
  loop llc

(* gh-ocannl-959: the legality of a hardware binding, as a query on the written cell. Binding a
   loop's index to a hardware register makes its iterations threads, and thread identity is one
   coordinate per ACTIVE slot — a (kind, slot) of the launch some bound loop of extent above one
   occupies; cc's pool renders one outermost Grid loop at a time as chunks, a binding whose threads
   exist only under that loop. A store under a binding to storage the threads share is race-free
   exactly when no two threads own the same cell: every active slot must have a bound loop enclosing
   the store (a slot without one executes the store on each of its coordinates —
   [validate_parallel]'s coverage rule, here for workgroup-shared storage too), and the cell's index
   map must SEPARATE the enclosing thread symbols ({!Affine.separates}: two instances of the
   statement, varying every loop symbol in scope independently, address a common cell only if they
   agree on every thread symbol). Mentioning a thread symbol is not separating it: [acc[i + j]]
   mentions both of two bound axes and [(0, 1)] and [(1, 0)] share [acc[1]] (the issue's example),
   and [acc[i + k]] under a bound [i] and a serial [k] collides the same way — which is why every
   loop symbol in scope is concurrent, not only the bound ones. What separates a thread symbol is
   the injectivity the engine proves ([acc[2 * i + j]] with [j < 2]), or the guards the store sits
   under: the environment is narrowed by each enclosing [If] exactly as gh-ocannl-566's simplifier
   narrows it ({!ienv_narrow_from_cond}: conjunctions of affine comparisons), so [If (i < 16)]
   shrinks [i]'s range for the radix argument and [If (i == 0)] PINS [i] to a width-one range, which
   the engine already reads as one thread along that axis — and along that axis alone, since the
   other active slots keep their coordinates. The six ways a pin was found not to be one thread on
   staging#674 are each another coordinate the cell must still separate: a grid of blocks is a bound
   [Grid] axis (of device storage; workgroup-shared storage is per block, so [Grid] axes are not its
   threads), a second workgroup dimension is a bound [Workgroup] axis, a pin to a loop index or a
   memory read narrows nothing.

   What this judges is the store's OWN threads. The cell a pinned store writes may be read or
   written by other threads from OTHER statements — the tensorized pipeline zeroes its output on
   every lane and stores the tile back from lane 0, ordered by the barrier the intrinsic block ends
   with — and whether such a pair is ordered is dependence analysis over barrier regions: the
   schedule's obligation where it mints the pin ([Stage], [Tensorize]) and the annotating pass's
   where it mints the binding, exactly as for every unpinned store before this rule existed. It is
   deliberately not modelled here.

   Reads play no part: a plain store every thread performs to one cell is a write-write race whether
   or not the values agree, so the sound single-thread form is a pinned store or a per-thread cell.
   A dead level ([to_ < from_]) and a statically false guard execute nothing. A [Set_dynamic] is
   judged by its static slots (the dynamic one separates nothing: masked opaque). A [Set_from_vec]
   is a run of [length] cells from its base: where the base's last component is a multiple of
   [length] the runs are aligned blocks and the quotient map is what must separate, otherwise the
   component is opaque and the other slots must. A [Zero_out] is a store of every cell. A [Tile_mma]
   is judged through its [fallback] — the scalar nest that spells the tile's stores — with its own
   cooperating [lane] excused (the tile is jointly owned by that axis and by construction mentions
   it nowhere), so a tile whose base omits an outer bound axis is refused where the plain store
   would be (gh-ocannl-960). [Set_local] and [Declare_local] are thread-private by nature.

   [active] is the launch's slots; [thread s] says whether the loop [s] is bound and to which slot;
   [deferred s] marks a bound loop whose separation another call judges (a [Workgroup_reduce] lane
   the warp shuffle may still own) — it covers its slot like any bound loop but is not a symbol a
   store must separate here; [storage] classifies a node as device-resident, workgroup-shared or
   thread-private. The whole kernel is walked, so every loop symbol a store can mention is in scope.
   Returns the first offending store's node and cell with the reason. *)
type thread_storage = [ `Device | `Shared | `Thread ]
type thread_slot = [ `Grid | `Workgroup ] * int

let unseparated_thread_write ~(active : thread_slot list)
    ~(thread : Indexing.symbol -> thread_slot option) ~(deferred : Indexing.symbol -> bool)
    ~(storage : Tnode.t -> thread_storage) (kernel : t) :
    (Tnode.t * Indexing.axis_index array * string) option =
  let equal_slot (k1, s1) (k2, s2) = Poly.equal k1 k2 && s1 = s2 in
  let slot_matters cls (kind, _) =
    match (cls, kind) with
    | `Shared, `Grid -> false
    | (`Device | `Shared), (`Grid | `Workgroup) -> true
  in
  let describe (kind, slot) =
    (match kind with `Grid -> "Grid" | `Workgroup -> "Workgroup") ^ " slot " ^ Int.to_string slot
  in
  let mask_dynamic idcs dyn_axis =
    let m = Array.copy idcs in
    m.(dyn_axis) <- Indexing.Sub_axis;
    m
  in
  (* The aligned-block quotient a vector store's separation is judged by. *)
  let vec_blocks idcs length =
    let last = Array.length idcs - 1 in
    if length <= 1 || last < 0 then idcs
    else
      let m = Array.copy idcs in
      m.(last) <-
        (match idcs.(last) with
        | Indexing.Fixed_idx c when c % length = 0 -> Indexing.Fixed_idx (c / length)
        | Indexing.Affine { symbols; offset }
          when offset % length = 0 && List.for_all symbols ~f:(fun (c, _) -> c % length = 0) ->
            Indexing.Affine
              {
                symbols = List.map symbols ~f:(fun (c, s) -> (c / length, s));
                offset = offset / length;
              }
        | _ -> Indexing.Sub_axis);
      m
  in
  (* [threads]: the bound loops enclosing the store, innermost first; [excused]: the cooperating
     lanes of the tiles it sits in. *)
  let judge ~env ~threads ~excused tn idcs =
    match storage tn with
    | `Thread -> None
    | (`Device | `Shared) as cls -> (
        let covering = List.filter threads ~f:(fun (_, sl) -> slot_matters cls sl) in
        match
          List.find active ~f:(fun sl ->
              slot_matters cls sl
              && not (List.exists covering ~f:(fun (_, sl') -> equal_slot sl sl')))
        with
        | Some sl ->
            Some
              ( tn,
                idcs,
                "no enclosing bound loop of " ^ describe sl
                ^ " tells its threads apart: every coordinate of the slot executes the store" )
        | None ->
            let syms =
              List.filter_map covering ~f:(fun (s, _) ->
                  if deferred s || List.mem excused s ~equal:Indexing.equal_symbol then None
                  else Some s)
            in
            (* A Grid symbol is shared by the threads of one block, hence equal across two instances
               on workgroup-shared storage. *)
            let concurrent s =
              Map.mem env.sym_env s
              && (Poly.equal cls `Device
                 || not (Option.exists (thread s) ~f:(fun (k, _) -> Poly.equal k `Grid)))
            in
            Option.map
              (Affine.separation_failure ~range:(sym_int_bounds env.sym_env) ~concurrent ~syms ~idcs)
              ~f:(fun why -> (tn, idcs, why)))
  in
  let rec go ~env ~threads ~excused (st : t) =
    let go' = go ~env ~threads ~excused in
    let judge = judge ~env ~threads ~excused in
    let enter index ~from_ ~to_ body =
      let threads = match thread index with Some sl -> (index, sl) :: threads | None -> threads in
      go ~env:(ienv_extend env index ~from_ ~to_) ~threads ~excused body
    in
    match st with
    | Seq (a, b) -> ( match go' a with Some _ as r -> r | None -> go' b)
    | (For_loop { from_; to_; _ } | Scan_loop { from_; to_; _ }) when to_ < from_ -> None
    | If { cond = Constant c, _; _ } when Float.equal c 0. -> None
    | For_loop { index; from_; to_; body; _ } | Scan_loop { index; from_; to_; body; _ } ->
        enter index ~from_ ~to_ body
    | If { cond = c, cprec; body } ->
        go ~env:(ienv_narrow_from_cond env ~cprec c) ~threads ~excused body
    | Set { tn; idcs; _ } -> judge tn idcs
    | Set_dynamic { tn; idcs; dyn_axis; _ } -> judge tn (mask_dynamic idcs dyn_axis)
    | Set_from_vec { tn; idcs; length; _ } -> judge tn (vec_blocks idcs length)
    | Zero_out tn -> judge tn [||]
    | Tile_mma { lane; fallback; _ } -> go ~env ~threads ~excused:(lane :: excused) fallback
    | Noop | Comment _ | Staged_compilation _ | Set_local _ | Declare_local _ | Workgroup_barrier ->
        None
  in
  go
    ~env:{ sym_env = Map.empty (module Indexing.Symbol); memo = Phys_memo.create 16 }
    ~threads:[] ~excused:[] kernel

type peel_guard_verdict = Guard_confined | Guard_lane_private | Guard_lane_private_unresolved
[@@deriving sexp, equal, compare]

type peel_refusal =
  | Refused_not_a_nest
  | Refused_dead_level
  | Refused_guard_fixed of string
  | Refused_cell_varies
  | Refused_cell_shared
[@@deriving sexp, equal, compare]

type peel_report = { levels : int; guards : peel_guard_verdict list; refusal : peel_refusal option }
[@@deriving sexp_of]

let peel_accum_nest ?(extra_level = fun _ _ -> false) ?report ~loop_bounds ~free_of body :
    (Tn.t
    * Indexing.axis_index array
    * [ `Update of scalar_t | `Scope of scope_id * t list ]
    * string
    * (t -> t))
    option =
  let strip stmts = List.filter stmts ~f:(function Noop | Comment _ -> false | _ -> true) in
  let idx_mentions sym (idx : Indexing.axis_index) =
    match idx with
    | Indexing.Iterator s -> Indexing.equal_symbol s sym
    | Indexing.Affine { symbols; _ } ->
        List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s sym)
    | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
  in
  (* Guard legality is {!Affine.peel_guard}'s, not this function's (gh-ocannl-722). [rebuild] keeps
     a guard around the accumulating update only, so the localized form performs its opening load
     and its closing store OUTSIDE it; the engine states, in one place, when that matches the
     original and when it invents accesses or races an enclosing lane. Five review rounds of
     gh-ocannl-693 re-derived the rule here as an ad hoc predicate before it moved there.

     Two halves of the answer arrive at different times. What the guard mentions is known on the way
     DOWN, so {!Affine.peel_guard} is asked at each guard and rejects there; the enclosing loop
     symbols it can only admit conditionally are carried in [pending] until the base, where the
     accumulated cell is finally in hand and {!Affine.separates} decides whether each of those
     instances owns a distinct cell. *)
  let range s =
    List.fold loop_bounds ~init:None ~f:(fun acc (s', (blo, bhi)) ->
        if Indexing.equal_symbol s s' then
          match acc with None -> Some (blo, bhi) | Some (lo, hi) -> Some (min lo blo, max hi bhi)
        else acc)
  in
  let loop_bound s = Option.is_some (range s) in
  let peeled ~free_of s = List.exists free_of ~f:(Indexing.equal_symbol s) in
  let guard_symbols (llsc : scalar_t) =
    let of_index (idx : Indexing.axis_index) =
      match idx with
      | Indexing.Iterator s -> [ s ]
      | Indexing.Affine { symbols; _ } -> List.map symbols ~f:snd
      | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> []
    in
    let rec go acc (s : scalar_t) =
      match s with
      | Embed_index idx -> of_index idx @ acc
      | Binop (_, (a, _), (b, _)) -> go (go acc a) b
      | Unop (_, (a, _)) -> go acc a
      | Ternop (_, (a, _), (b, _), (c, _)) -> go (go (go acc a) b) c
      | _ -> acc
    in
    go [] llsc
  in
  (* The deferred half, in two parts. [concurrent] is every non-peeled loop symbol rather than only
     the [pending] ones: two instances may differ in ANY of them, and holding one equal would let a
     cell like [acc[w1 + w2]] "prove" a separation it does not have.

     And separation is distinctness, not validity. The guard being hoisted past may be what keeps
     the cell IN BOUNDS -- with a one-element [acc] and [If (w + k < 1) (acc[w] += ...)], [acc[w]]
     separates [w] while lanes 1-3 address cells that do not exist -- so the cell must also sit
     inside the node's box over the enclosing symbols' full ranges, judged without the guard. Only
     the escape needs this: a confined guard mentions no symbol the cell mentions. *)
  let cell_admits ~free_of ~pending tn idcs =
    List.is_empty pending
    || Affine.separates ~range
         ~concurrent:(fun s -> loop_bound s && not (peeled ~free_of s))
         ~syms:pending ~idcs
       && Affine.within_box ~range ~dims:(Lazy.force tn.Tn.dims) idcs
  in
  let cell_invariant ~free_of idcs =
    not (Array.exists idcs ~f:(fun idx -> List.exists free_of ~f:(fun s -> idx_mentions s idx)))
  in
  (* A DEAD level ([to_ < from_]) is never peeled. Its body performs no accesses at all — the
     interface walker propagates [live:(live && to_ >= from_)], so a node reached only under a dead
     loop is absent from the routine's parameters and may not even be allocated — while the
     localized form this peel licenses reads and writes the accumulator cell OUTSIDE the level,
     unconditionally. Hoisting there would invent accesses the program does not make, on an
     identifier the interface never declared. Refusing here covers every consumer at once: codegen's
     localization and [Schedule]'s [Unroll ~materialize:true] / [Partition] mints. The level then
     renders as the plain (access-free) dead loop it is. *)
  (* Every exit reports what it decided as well as whether it succeeded (gh-ocannl-733): the levels
     peeled so far, the verdict of each guard peeled through, and the refusal where there is one.
     [guards] is accumulated innermost-first and reversed at the exit, so a reader sees the nest
     order. *)
  let rec peel ~free_of ~pending ~levels ~guards ~rebuild body =
    (* A [Lane_private_if_separated] guard is only ADMITTED once the base's cell has been shown to
       separate the enclosing symbols it mentions, and that check happens at the base — so on a
       refusal the guard's own question was never settled, and reporting it as admitted beside a
       refusal would make the report contradict itself (Codex P2, round 1). The descent carries the
       optimistic tag and each exit resolves it: the base resolves it to admitted, every refusal to
       {!Guard_lane_private_unresolved}. *)
    let unresolved =
      List.map ~f:(function
        | Guard_lane_private -> Guard_lane_private_unresolved
        | (Guard_confined | Guard_lane_private_unresolved) as g -> g)
    in
    let refuse refusal =
      (None, { levels; guards = unresolved (List.rev guards); refusal = Some refusal })
    in
    let reached result = (Some result, { levels; guards = List.rev guards; refusal = None }) in
    match strip (flat_lines [ body ]) with
    | [ For_loop ({ index; body = ibody; axis; _ } as r) ]
      when match axis with Serial | Unrolled | Vectorized -> true | _ -> extra_level index axis ->
        (* [Vectorized] levels ride into the scope: the SIMD reduction rendering recognizes the
           [Set_local] update form and folds its chains into the scope local, so the whole nest
           keeps one accumulator residency even when an inner reduction axis is vectorized (autotune
           proposes Retype-[Vectorized] over reductions); where that rendering declines, the loop
           renders serially over the scope local — same values either way.

           Beyond those annotation-free kinds are the levels the CALLER vouches for: codegen passes
           a predicate accepting a hardware-annotated reduction loop its backend will serialize (no
           hardware index for the slot), so e.g. a nested [Workgroup_reduce] on cc keeps the
           whole-nest residency. The schedule mints pass nothing: wrapping a hardware-annotated loop
           in a scope at transform time would break the schedule on backends that do bind the
           hardware dimension. *)
        if r.to_ < r.from_ then refuse Refused_dead_level
        else
          peel ~free_of:(index :: free_of) ~pending ~levels:(levels + 1) ~guards
            ~rebuild:(fun b -> rebuild (For_loop { r with body = b }))
            ibody
    | [ If { cond = gc, gp; body = gbody } ] when pure_index_guard gc -> (
        let rebuild b = rebuild (If { cond = (gc, gp); body = b }) in
        match
          Affine.peel_guard ~loop_bound ~peeled:(peeled ~free_of) ~guard_syms:(guard_symbols gc)
        with
        | Affine.Not_peelable why -> refuse (Refused_guard_fixed why)
        | Affine.Confined_to_peel ->
            peel ~free_of ~pending ~levels ~guards:(Guard_confined :: guards) ~rebuild gbody
        | Affine.Lane_private_if_separated enclosing ->
            peel ~free_of ~pending:(enclosing @ pending) ~levels
              ~guards:(Guard_lane_private :: guards) ~rebuild gbody)
    | [ Set { tn; idcs; llsc; debug } ] -> (
        (* The base's SHAPE is settled first, and only then the cell (Codex P2, round 2). The cell
           refusals describe an accumulation base — a cell that varies across the peeled levels, a
           cell the lanes an admitted guard selects among share — and a statement that is not an
           accumulation at all has neither property: reporting [Refused_cell_varies] for an ordinary
           [out[k] = x[k]] would hand a [~report] consumer a reason its own contract denies. Which
           of the three it was is precisely what a form claim cannot see (gh-ocannl-733), so the
           three stay distinct once the shape admits them. *)
        let base =
          match llsc with
          | Local_scope { id; body = sbody; orig_indices = _; mint = _ } ->
              Option.map (scope_accum_updates ~tn ~idcs ~id sbody) ~f:(fun rest ->
                  `Scope (id, rest))
          | _ when Option.is_some (accum_update_parts ~tn ~idcs llsc) -> Some (`Update llsc)
          | _ -> None
        in
        match base with
        | None -> refuse Refused_not_a_nest
        | Some base ->
            if not (cell_invariant ~free_of idcs) then refuse Refused_cell_varies
            else if not (cell_admits ~free_of ~pending tn idcs) then refuse Refused_cell_shared
            else reached (tn, idcs, base, debug, rebuild))
    | _ -> refuse Refused_not_a_nest
  in
  let result, rep = peel ~free_of ~pending:[] ~levels:0 ~guards:[] ~rebuild:(fun b -> b) body in
  Option.iter report ~f:(fun f -> f rep);
  result

(* gh-343: extract the per-iteration one-hot contribution from an accumulation [acc] in which the
   running total is recognized by [acc_is]. Handles the [Binop (Add, total, contribution)] form
   (either operand order) and the fused [Ternop (FMA, a, b, total)] form, where FMA(a,b,total) = a*b
   + total so the contribution is the product [a*b]. *)
let accumulation_contribution ~(acc_is : scalar_t -> bool) (acc : scalar_t) : scalar_t option =
  match acc with
  | Binop (Ops.Add, (total, _), (contribution, _)) when acc_is total -> Some contribution
  | Binop (Ops.Add, (contribution, _), (total, _)) when acc_is total -> Some contribution
  | Ternop (Ops.FMA, (a, pa), (b, pb), (total, _)) when acc_is total ->
      Some (Binop (Ops.Mul, (a, pa), (b, pb)))
  | _ -> None

(* gh-343: shared core -- given a reduction over [k] with bounds [\[from_, to_\]] and a
   per-iteration [contribution], check the narrow one-hot side conditions and build the guarded
   gather. *)
let gather_of_reduction ~ienv ~(k : Indexing.symbol) ~from_ ~to_ (contribution : scalar_t) :
    scalar_t option =
  Option.bind (match_one_hot_contribution k contribution) ~f:(fun (table, table_idcs, index_expr) ->
      let count, axis, only_plain = count_plain_iterator k table_idcs in
      match if count = 1 && only_plain then axis else None with
      | None -> None
      | Some dyn_axis ->
          let class_count = from_ + (Lazy.force table.Tn.dims).(dyn_axis) in
          (* the loop must span exactly [0, class_count) over the gathered axis, and the index
             expression must be free of the reduction variable *)
          if (not (from_ = 0)) || to_ <> class_count - 1 then None
          else if scalar_mentions_symbol k (fst index_expr) then None
          else
            let value_prec = Lazy.force table.Tn.storage_prec in
            (* Neutralize the now-dead loop symbol at the dynamic axis. *)
            let table_idcs = Array.copy table_idcs in
            table_idcs.(dyn_axis) <- Indexing.Fixed_idx 0;
            Some
              (build_guarded_gather ~ienv ~table ~table_idcs ~dyn_axis ~index_expr ~class_count
                 ~value_prec))

(* gh-343: scalar-local form -- Local_scope { id; body = [init;] For k { Set_local (id, acc) } }. *)
let try_rewrite_local_scope ~ienv (id : scope_id) (body : t) : scalar_t option =
  match strip_zero_init_for_local id body with
  | Some (For_loop { index = k; from_; to_; body = Set_local (id', acc); _ })
    when equal_scope_id id id' ->
      let acc_is = function Get_local id' -> equal_scope_id id id' | _ -> false in
      Option.bind (accumulation_contribution ~acc_is acc) ~f:(fun contribution ->
          gather_of_reduction ~ienv ~k ~from_ ~to_ contribution)
  | _ -> None

(* gh-343: materialized form -- a reduction loop [For k { Set lhs idcs acc }] at any nesting depth,
   where [acc] accumulates the one-hot contribution into [lhs\[idcs\]] (and [k] does not index
   [lhs]). The loop is replaced with a single read-accumulate of the guarded gather, dropping the
   vocabulary loop. Reading-and-adding [lhs\[idcs\]] keeps the rewrite sound regardless of any
   preceding zero-init: sum_k contribution == gather, so [lhs += gather] equals the original [lhs +
   sum_k contribution]. *)
let try_rewrite_materialized_loop ~ienv (llc : t) : t option =
  match llc with
  | For_loop { index = k; from_; to_; body = Set { tn; idcs; llsc; _ }; _ }
    when not (Array.exists idcs ~f:(axis_index_mentions_symbol k)) ->
      let acc_is = function
        | Get (g, gi) -> Tn.equal g tn && [%equal: Indexing.axis_index array] gi idcs
        | _ -> false
      in
      Option.bind (accumulation_contribution ~acc_is llsc) ~f:(fun contribution ->
          Option.map (gather_of_reduction ~ienv ~k ~from_ ~to_ contribution) ~f:(fun gather ->
              let value_prec = scalar_precision gather in
              Set
                {
                  tn;
                  idcs;
                  llsc = Binop (Ops.Add, (Get (tn, idcs), value_prec), (gather, value_prec));
                  debug = "";
                }))
  | _ -> None

(* gh-466: conservative "mentions [tn]" scan over a scalar, descending into [Local_scope] bodies and
   [Get_dynamic] index values. *)
let rec scalar_mentions_tn (tn : Tn.t) (llsc : scalar_t) : bool =
  match llsc with
  | Get (g, _) | Get_merge_buffer (g, _) -> Tn.equal g tn
  | Get_dynamic { tn = g; dyn_value = v, _; _ } -> Tn.equal g tn || scalar_mentions_tn tn v
  | Local_scope { body; _ } -> proc_mentions_tn tn body
  | Ternop (_, (a, _), (b, _), (c, _)) ->
      scalar_mentions_tn tn a || scalar_mentions_tn tn b || scalar_mentions_tn tn c
  | Binop (_, (a, _), (b, _)) -> scalar_mentions_tn tn a || scalar_mentions_tn tn b
  | Unop (_, (a, _)) -> scalar_mentions_tn tn a
  | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> false

and proc_mentions_tn (tn : Tn.t) (llc : t) : bool =
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> false
  | Seq (a, b) -> proc_mentions_tn tn a || proc_mentions_tn tn b
  | For_loop { body; _ } -> proc_mentions_tn tn body
  | Scan_loop { carried; body; _ } ->
      List.exists carried ~f:(fun c -> scalar_mentions_tn tn c.init) || proc_mentions_tn tn body
  | Zero_out g -> Tn.equal g tn
  | Set { tn = g; llsc; _ } -> Tn.equal g tn || scalar_mentions_tn tn llsc
  | Set_dynamic { tn = g; dyn_value = v, _; llsc; _ } ->
      Tn.equal g tn || scalar_mentions_tn tn v || scalar_mentions_tn tn llsc
  | Set_from_vec { tn = g; arg = a, _; _ } -> Tn.equal g tn || scalar_mentions_tn tn a
  | Set_local (_, llsc) -> scalar_mentions_tn tn llsc
  | If { cond = c, _; body } -> scalar_mentions_tn tn c || proc_mentions_tn tn body
  | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; fallback; _ } ->
      Tn.equal d_tn tn || Tn.equal a_tn tn || Tn.equal b_tn tn || proc_mentions_tn tn fallback

(* gh-466: transposed (scatter) form -- the embedding-table gradient. A loop [For k { Set { tn;
   idcs; llsc } }] where [k] indexes [tn] itself (once, as a plain [Iterator] -- the table axis) and
   each iteration accumulates the one-hot-selected contribution into its own row:

   for k in [0, class_count): tn[.., k, ..] += (k == e) * g

   with [e] and [g] free of [k] and [g] not touching [tn]. Every iteration adds 0 except [k = e]
   (when [e] is in range and integral -- one-hot semantics), so the loop equals a single guarded
   scatter-accumulate at the dynamic row [e]:

   if in_range(e): tn[.., e, ..] += g

   -- O(1) rows touched per position instead of O(class_count), llm.c's deterministic no-atomics
   encoder backward (docs/research/llmc-lessons.md B5). Enclosing position loops execute in their
   original serial order, so for every cell the accumulation order matches the dense form. Like the
   forward rewrite, contributions that the dense form multiplies by 0 are dropped rather than added
   (a NaN/Inf [g] at a non-selected or out-of-range position no longer poisons the table). *)
let try_rewrite_transposed_loop ~ienv (llc : t) : t option =
  match llc with
  | For_loop { index = k; from_; to_; body = Set { tn; idcs; llsc; _ }; _ } -> (
      let count, axis, only_plain = count_plain_iterator k idcs in
      match if count = 1 && only_plain then axis else None with
      | None -> None
      | Some dyn_axis ->
          let acc_is = function
            | Get (g, gi) -> Tn.equal g tn && [%equal: Indexing.axis_index array] gi idcs
            | _ -> false
          in
          Option.bind (accumulation_contribution ~acc_is llsc) ~f:(fun contribution ->
              Option.bind (match_transposed_one_hot_contribution k contribution)
                ~f:(fun (index_expr, grad_arg) ->
                  let class_count = from_ + (Lazy.force tn.Tn.dims).(dyn_axis) in
                  (* The loop must span exactly [0, class_count) over the written axis; the index
                     expression is free of [k] by construction ([match_one_hot_cmpeq]); the
                     contribution must be loop-invariant and must not read (or re-write, via a
                     [Local_scope]) the scattered tensor, else dropping the per-row iterations could
                     change what it observes. *)
                  if from_ <> 0 || to_ <> class_count - 1 then None
                  else if scalar_mentions_symbol k (fst grad_arg) then None
                  else if scalar_mentions_tn tn (fst grad_arg) then None
                  else if scalar_mentions_tn tn (fst index_expr) then None
                  else
                    (* Neutralize the now-dead loop symbol at the dynamic axis. *)
                    let idcs = Array.copy idcs in
                    idcs.(dyn_axis) <- Indexing.Fixed_idx 0;
                    Some
                      (build_guarded_scatter ~ienv ~tn ~idcs ~dyn_axis ~index_expr ~grad_arg
                         ~class_count ~debug:""))))
  | _ -> None

let rewrite_one_hot_reductions ?(static_indices = []) (llc : t) : t =
  let rec loop_proc ~ienv (llc : t) : t =
    match try_rewrite_materialized_loop ~ienv llc with
    | Some replacement -> replacement
    | None -> (
        match try_rewrite_transposed_loop ~ienv llc with
        | Some replacement -> replacement
        | None -> loop_unmatched ~ienv llc)
  and loop_unmatched ~ienv (llc : t) : t =
    match llc with
    | Seq (a, b) -> Seq (loop_proc ~ienv a, loop_proc ~ienv b)
    | For_loop fc ->
        For_loop
          {
            fc with
            body = loop_proc ~ienv:(ienv_extend ienv fc.index ~from_:fc.from_ ~to_:fc.to_) fc.body;
          }
    | Scan_loop sc ->
        Scan_loop
          {
            sc with
            carried = List.map sc.carried ~f:(fun c -> { c with init = loop_scalar ~ienv c.init });
            body = loop_proc ~ienv:(ienv_extend ienv sc.index ~from_:sc.from_ ~to_:sc.to_) sc.body;
          }
    | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = loop_scalar ~ienv llsc; debug }
    | Set_dynamic { tn; idcs; dyn_axis; dyn_value = v, p; llsc; debug } ->
        (* Only produced by this pass; recurse for exhaustiveness/idempotence. *)
        Set_dynamic
          {
            tn;
            idcs;
            dyn_axis;
            dyn_value = (loop_scalar ~ienv v, p);
            llsc = loop_scalar ~ienv llsc;
            debug;
          }
    | Set_from_vec { tn; idcs; length; vec_unop; arg = s, p; debug } ->
        Set_from_vec { tn; idcs; length; vec_unop; arg = (loop_scalar ~ienv s, p); debug }
    | Set_local (id, llsc) -> Set_local (id, loop_scalar ~ienv llsc)
    | If { cond = c, p; body } ->
        If { cond = (loop_scalar ~ienv c, p); body = loop_proc ~ienv body }
    | ( Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
      | Tile_mma _ ) as other ->
        other
  and loop_scalar ~ienv (llsc : scalar_t) : scalar_t =
    let loop_scalar = loop_scalar ~ienv in
    match llsc with
    | Local_scope { id; body; orig_indices; mint } -> (
        (* Recurse into the body first so inner reductions are handled, then try to collapse this
           scope itself. *)
        let body = loop_proc ~ienv body in
        match try_rewrite_local_scope ~ienv id body with
        | Some gather -> gather
        | None -> Local_scope { id; body; orig_indices; mint })
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, p } ->
        Get_dynamic { tn; idcs; dyn_axis; dyn_value = (loop_scalar v, p) }
    | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
        Ternop (op, (loop_scalar a, pa), (loop_scalar b, pb), (loop_scalar c, pc))
    | Binop (op, (a, pa), (b, pb)) -> Binop (op, (loop_scalar a, pa), (loop_scalar b, pb))
    | Unop (op, (a, pa)) -> Unop (op, (loop_scalar a, pa))
    | (Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _) as
      other ->
        other
  in
  loop_proc ~ienv:(ienv_of_static_indices static_indices) llc

(* Phase B v1 execution anchoring (interval-analysis proposal, binding constraint 3): every tensor
   node written by compiled code has its bounds candidate pinned to top at lowering time, BEFORE any
   interval consultation, so guard folds only ever rely on host-initialized, never-device-written
   data -- sidestepping the writer-runs-never/-later/-repeatedly hazards and read-modify-write
   cycles wholesale. Pinning a node whose bounds a previously-compiled reader already settled (to a
   non-top interval) raises, as required: that reader's generated code discharged an in-range guard
   the new writer could invalidate. Walks the unoptimized code so writes that later become virtual
   are included (conservative). *)
let rec pin_device_written_bounds (llc : t) : unit =
  let pin tn = Tn.pin_bounds_top ~what:"compiled device write" tn in
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> ()
  | Seq (c1, c2) ->
      pin_device_written_bounds c1;
      pin_device_written_bounds c2
  | For_loop { body; _ } -> pin_device_written_bounds body
  | Scan_loop { carried; body; _ } ->
      List.iter carried ~f:(fun c -> pin_scalar_written_bounds c.init);
      pin_device_written_bounds body
  | Zero_out tn -> pin tn
  | Set { tn; llsc; _ } ->
      pin tn;
      pin_scalar_written_bounds llsc
  (* gh-466: defensive -- [Set_dynamic] is produced after lowering-time pinning. *)
  | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
      pin tn;
      pin_scalar_written_bounds v;
      pin_scalar_written_bounds llsc
  | Set_from_vec { tn; arg = s, _; _ } ->
      pin tn;
      pin_scalar_written_bounds s
  | Set_local (_, llsc) -> pin_scalar_written_bounds llsc
  | Tile_mma { d = d_tn, _; fallback; _ } ->
      pin d_tn;
      pin_device_written_bounds fallback
  | If { cond = c, _; body } ->
      pin_scalar_written_bounds c;
      pin_device_written_bounds body

and pin_scalar_written_bounds (llsc : scalar_t) : unit =
  match llsc with
  | Local_scope { body; _ } -> pin_device_written_bounds body
  | Get_dynamic { dyn_value = v, _; _ } -> pin_scalar_written_bounds v
  | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  | Ternop (_, (v1, _), (v2, _), (v3, _)) ->
      pin_scalar_written_bounds v1;
      pin_scalar_written_bounds v2;
      pin_scalar_written_bounds v3
  | Binop (_, (v1, _), (v2, _)) ->
      pin_scalar_written_bounds v1;
      pin_scalar_written_bounds v2
  | Unop (_, (v, _)) -> pin_scalar_written_bounds v

let statics_set_of static_indices =
  Set.of_list
    (module Indexing.Symbol)
    (List.map static_indices ~f:(fun s -> s.Indexing.static_symbol))

(* The [inline_complex_computations] read-modify-write exemption over access records: a read at its
   enclosing statement's write position (same {!same_position} map as the write the read is
   subordinate to — [a_stmt_write] — whichever nodes are involved) is a self-read of the
   statement's own store, not a visit — exactly the reads the retired concrete tracer never
   recorded. Statement subordination, not program-path matching: an [If] condition's read shares
   its path with the guarded body's write but executes before it, so it is never exempt. Shared by
   {!reads_covered_query} and {!read_multiplicity_query} — but the two consumers read it
   differently (gh-ocannl-618): for per-cell visit counting the exemption is always right (the
   statement's own store is not a visit), while for the routine INTERFACE it is not a coverage
   fact — the cells an exempt read touches still carry their incoming values unless a prior write
   definitely covers them, so the coverage query reports exemption-dependent coverage as its own
   verdict rather than folding it into [`Covered]. *)
(* Inclusive value bounds of a static symbol, matching the runtime validation of static bindings
   ([Indexing]): a symbol [used_as_extent] takes values in [0, r] (extents size buffers), while a
   plain static is a strict index in [0, r) — universalizing a read over the impossible cell [r]
   would spuriously decline coverage of fully-covered static-slice routines. *)
let static_bounds (static_indices : Indexing.static_symbol list) s =
  List.find_map static_indices ~f:(fun ss ->
      if Indexing.equal_symbol ss.Indexing.static_symbol s then
        Option.map ss.static_range ~f:(fun r -> if ss.used_as_extent then (0, r) else (0, r - 1))
      else None)

let rmw_exempt ~statics_set (r : _ Affine.access) =
  virtualize_settings.inline_complex_computations
  && Option.exists r.Affine.a_stmt_write ~f:(fun w -> same_position ~statics_set w r.a_map)

(* gh-494 waypoint 2 / gh-554: the read-before-write fact of the retired concrete-index tracer (its
   [Recurrent] classification), computed as containment queries — every read covered by prior
   writes. The tracer's semantics is mirrored: [If] guards are taken (guarded writes count as
   assignments, as the tracer traced [If] bodies unconditionally); a read at the enclosing
   statement's write position is exempt ({!rmw_exempt}); and static symbols are universalized over
   their declared ranges — the query is exact where the tracer sampled (loops truncated at the
   retired [virtualize_max_tracing_dim], statics pinned to 0).

   [affine_accesses] is not exhaustive on [Staged_compilation] (opaque), so in its presence the
   verdicts describe the visible accesses only — the same blindness the tracer had: a write hidden
   in staged code makes the query decline coverage (pessimize — the safe direction), a hidden read
   is invisible to both analyses. A node without affine accesses is vacuously covered
   ([affine_accesses] and [trace_node_facts] walk the same tree, so such a node has no traced reads
   either). *)
(* Whether every [If] guard enclosing [write] also encloses [read]: the write's path prefix up
   to its LAST [Body] component is a prefix of the read's path, so whenever the read executes,
   every guard admitting the write held (review round 8). Positional before-ness is the coverage
   query's separate job. *)
let write_guards_dominate ~(write : Tn.t Affine.access) ~(read : Tn.t Affine.access) : bool =
  match
    List.foldi write.Affine.a_path ~init:None ~f:(fun i acc comp ->
        match comp with Affine.Body -> Some i | _ -> acc)
  with
  | None -> true
  | Some i ->
      let prefix = List.take write.Affine.a_path (i + 1) in
      List.is_prefix read.Affine.a_path ~prefix ~equal:Affine.equal_path_comp

let reads_covered_query ?(write_eligible = fun ~read:_ ~write:_ -> true)
    (static_indices : Indexing.static_symbol list) (accs : Tn.t Affine.access list) :
    Tn.t -> [ `Covered | `Covered_rmw_exempt of string | `Unknown of string ] =
  let by_tn = Hashtbl.create (module Tn) in
  List.iter accs ~f:(fun a -> Hashtbl.add_multi by_tn ~key:a.Affine.a_tn ~data:a);
  let statics_set = statics_set_of static_indices in
  let static_range = static_bounds static_indices in
  (* The three-way verdict is the gh-ocannl-618 split, decided in one pass: [`Covered] holds without
     leaning on the read-modify-write exemption; [`Covered_rmw_exempt] means the only uncovered
     reads are {!rmw_exempt} ones — covered for the tracer-mirroring placement and multiplicity
     consumers, NOT covered for the routine interface ([read_before_write]), where a same-position
     read is a genuine read-modify-write whose cells require their entry values unless a prior
     definite write covers them (review round 6 of the gh-610/611 PR established this reading for
     spliced reads; the raw-side split followed). A strictly-uncovered read dominates an
     exemption-dependent one. *)
  fun tn ->
    match Hashtbl.find by_tn tn with
    | None -> `Covered
    | Some accs ->
        let accs = List.rev accs in
        let writes = List.filter accs ~f:(fun a -> a.Affine.a_write) in
        let exempt_witness = ref None in
        let rec go = function
          | [] -> ( match !exempt_witness with None -> `Covered | Some w -> `Covered_rmw_exempt w)
          | r :: rest when r.Affine.a_write -> go rest
          | r :: rest -> (
              let writes = List.filter writes ~f:(fun w -> write_eligible ~read:r ~write:w) in
              match Affine.read_covered_before ~static_range ~read:r ~writes () with
              | `Covered -> go rest
              | `Unknown w when rmw_exempt ~statics_set r ->
                  if Option.is_none !exempt_witness then exempt_witness := Some w;
                  go rest
              | `Unknown w -> `Unknown w)
        in
        go accs

(* gh-554: per-cell read multiplicity upper bound — the abstract replacement for the retired
   tracer's sampled per-cell visit counts ([Visits i > max_visits]). A read site's own per-cell
   contribution is bounded by {!Affine.fiber_cardinality_ub} over its loop box; sites that can touch
   a common cell ({!Affine.may_touch_same_cell}) can stack, so a cell's total is bounded by a site's
   own bound plus the bounds of every site overlapping it — maximized over sites. Exemptions mirror
   the tracer: reads at the enclosing statement's write position are skipped ({!rmw_exempt});
   dynamic reads carry no interpretable map and are skipped like the tracer's defensive
   [Get_dynamic] arm (the construct postdates this analysis); guarded reads count (guards-taken).
   The bound over-approximates where the engine cannot prove per-site exactness or pairwise
   disjointness — erring toward materialization, the safe direction for the visit cap. *)
let read_multiplicity_query (static_indices : Indexing.static_symbol list)
    (accs : Tn.t Affine.access list) : Tn.t -> int =
  let statics_set = statics_set_of static_indices in
  let static_range = static_bounds static_indices in
  let exempt = rmw_exempt ~statics_set in
  let reads_by_tn = Hashtbl.create (module Tn) in
  List.iter accs ~f:(fun a ->
      if (not a.Affine.a_write) && (not a.a_dynamic) && not (exempt a) then
        Hashtbl.add_multi reads_by_tn ~key:a.a_tn ~data:a);
  fun tn ->
    match Hashtbl.find reads_by_tn tn with
    | None -> 0
    | Some sites ->
        let sites = Array.of_list sites in
        let bounds =
          Array.map sites ~f:(fun a ->
              let domain = List.map a.Affine.a_loops ~f:(fun (s, (lo, hi)) -> (s, hi - lo + 1)) in
              match Affine.fiber_cardinality_ub ~domain a.a_map with `Exact n | `At_most n -> n)
        in
        Array.foldi sites ~init:0 ~f:(fun i acc a ->
            let total =
              Array.foldi sites ~init:bounds.(i) ~f:(fun j acc' b ->
                  if j = i || not (Affine.may_touch_same_cell ~static_range a b) then acc'
                  else acc' + bounds.(j))
            in
            max acc total)

(* An access under a dead loop ([to_ < from_]: the body never executes) never happens: drop it from
   the metric views, like the retired tracer, which never enumerated such loops — a dead write must
   not supply coverage and a dead read must not demand it or count as a visit. *)
let drop_dead_loop_accesses (accs : Tn.t Affine.access list) : Tn.t Affine.access list =
  List.filter accs ~f:(fun a -> List.for_all a.Affine.a_loops ~f:(fun (_, (lo, hi)) -> hi >= lo))

(* The placement decision procedure over the traced facts and affine metrics — the tail of the
   retired [visit_llc], factored out (gh-554; the analysis/decision split of gh-555 step 1). Writes
   decisions into the lineage's placements table; the metrics are forced only when a decision
   actually consults them. The heuristic caps ([max_visits], [max_inline_reduction],
   [max_inline_fanin]) are priors of this default policy, not legality: a node in
   [optim_ctx.inline_preferences] (gh-555) is exempt from both, like one-hot selector producers
   always were, while the legality rejections ([check_and_store_virtual] / [inline_computation]) and
   the observability pessimizations (read-only, read-before-write) apply regardless. *)
let decide_placements (optim_ctx : optimize_ctx) traced_store ~max_visits ~reads_covered
    ~read_multiplicity =
  let plc = optim_ctx.placements in
  (* task-73617488: one-hot selector producers are exempt from the heuristic caps. The ordinary
     [virtual_llc]/[cleanup_virtual_llc] path will inline them so that [rewrite_one_hot_reductions]
     can fire at default [max_visits = 1]. gh-555: an explicit [Inline] decision
     ([inline_preferences]) is the same kind of exemption, searchable. *)
  let cap_exempt traced =
    (traced.prefers_virtual_one_hot && not traced.has_non_one_hot_setter)
    || Hash_set.mem optim_ctx.inline_preferences traced.tn
  in
  Hashtbl.iter traced_store ~f:(fun traced ->
      let tn = traced.tn in
      (* The gh-ocannl-618 split of the read-modify-write exemption by consumer: the placement
         decisions here accept exemption-dependent coverage ([`Covered_rmw_exempt]) as covered,
         mirroring the retired tracer — a statement's read of its own store position is not a visit.
         The routine-INTERFACE classification does not, but it cannot run here: placements are not
         settled (the fan-in guard below and [check_and_store_virtual]'s legality rejections still
         flip candidates non-virtual), so the strict verdict is applied by [reconcile_traced_store]
         over the FINAL code, once every node's standing is known. *)
      let covered =
        lazy
          (match (Lazy.force reads_covered) tn with
          | `Covered | `Covered_rmw_exempt _ -> true
          | `Unknown _ -> false)
      in
      let cap_exempt = cap_exempt traced in
      if
        virtualize_settings.inline_scalar_constexprs && traced.is_scalar_constexpr
        && not (Tn.Placements.known_non_virtual plc tn)
      then Tn.Placements.update plc tn Virtual 40;
      (* Recompute-cost guard: inlining a computation replays its reduction loops (loops enclosing a
         setter without appearing in its indices) at every read site, and the cost multiplies
         through chains of virtual consumers -- reads of the consumers replay the producer's
         reduction too. Cap the tolerated extent. One-hot selector producers are exempt like for the
         visit cap: they must stay virtual so [rewrite_one_hot_reductions] can fire (the rewrite
         itself removes the recompute cost). *)
      if
        virtualize_settings.max_inline_reduction >= 0
        && traced.inline_reduction_extent > virtualize_settings.max_inline_reduction
        && traced.read_by_other
        && Option.is_none (Tn.Placements.get plc tn)
        && not cap_exempt
      then Tn.Placements.update plc tn Never_virtual 39;
      let skip_simple =
        virtualize_settings.inline_simple_computations && (not traced.is_complex)
        && not (Tn.Placements.known_non_virtual plc tn)
      in
      (* Visit cap (gh-554): the retired tracer's sampled per-cell counts, as an exact-or-upper
         cardinality bound; an uncovered read (the tracer's [Recurrent]) also trips the cap. *)
      if
        (not skip_simple)
        && Option.is_none (Tn.Placements.get plc tn)
        && ((Lazy.force read_multiplicity) tn > max_visits || not (Lazy.force covered))
        && not cap_exempt
      then Tn.Placements.update plc tn Never_virtual 1;
      if (not traced.zeroed_out) && not traced.has_assignment then (
        (* The tensor node is read-only/recurrent for this computation, but maybe computed or
           specified as virtual by another routine (in this compilation lineage). However, if the
           placement is unspecified, we assume this will be the first computation involving the
           tensor node. *)
        traced.read_only <- true;
        if Tn.Placements.mode_is_unspecified plc tn then Tn.Placements.update plc tn On_device 37
        else if Tn.Placements.known_not_materialized plc tn then (
          if Tn.Placements.known_non_virtual plc tn then
            raise
              (Utils.User_error
                 [%string
                   "Mark %{Tn.debug_name tn} as materialized (e.g. via Train.set_materialized) \
                    before the first routine using it gets compiled; another routine re-uses that \
                    computation. Debug: %{Tn.Placements.debug plc tn}"]))
        else if Tn.Placements.known_non_virtual plc tn then Tn.Placements.update plc tn On_device 35);
      (* We allow sharing virtual nodes across routines. A node with an uncovered read (read before
         write within this routine — the tracer's [Recurrent]) must own a device buffer whose prior
         contents are preserved: it is an input of the routine. *)
      if (not (Tn.Placements.known_virtual plc tn)) && not (Lazy.force covered) then (
        traced.read_before_write <- true;
        Tn.Placements.update plc tn On_device 36));
  (* Transitive inline-fanin guard (gh-573): the per-node caps above cannot see chains. A running
     sum such as a transformer's residual stream has per-cell read multiplicity within the visit cap
     (its consumers' copy-position reads are read-modify-write-exempt) and no reduction loops, yet
     inlining it replays the entire prefix of the chain at every consumer — quadratic in depth. The
     per-evaluation cost that actually grows is the fan-in of the fully-inlined computation: the
     number of distinct materialized nodes it loads (the issue's triangular kernel signatures).
     Bottom-up over the setter-reads graph, a node still headed for inlining accumulates its virtual
     dependencies' fan-in sets; when the set outgrows the cap, the node is materialized (provenance
     41 — a heuristic policy decision, [`Inline]-flippable like the other caps), which resets the
     fan-in of everything downstream: the chain materializes once per ~cap contributors instead of
     once per consumer. Per-setter maximum, not union across setters — a read of one cell executes
     one setter's computation (Block/concat range-guarded setters). *)
  if virtualize_settings.max_inline_fanin >= 0 then
    (* The reads a stored computation replays when [inline_computation] inlines it: [Get]s of nodes
       other than [self], including inside [Local_scope] bodies. Mirrors [trace_node_facts]'
       collection for this routine's own setters; used for a dependency this routine only READS
       whose definition an earlier routine in the lineage committed [Virtual] — its traced entry
       here has no setters, but its computation will be replayed all the same. Merge-buffer reads
       stay uncounted (a materialized copy), and projections charge only the evaluated operand. *)
    (* [reads_of_proc acc code] is the statement-level sink; [reads_of_scalar (stmt, cur) v]
       threads the statement sink alongside the current (possibly per-arm) expression sink:
       [Local_scope] bodies hoist to statement level and execute unconditionally — both [Where]
       arms' bodies really run — so their reads join [stmt], while directly conditional arm
       expressions collect into fresh [cur] sinks maxed by the [Cond_and_one_arm] case. *)
    let rec reads_of_proc ~self acc (c : t) =
      match c with
      | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ | Zero_out _
        ->
          acc
      | Tile_mma _ -> acc (* post-optimization construct; never in stored computations *)
      | Seq (c1, c2) -> reads_of_proc ~self (reads_of_proc ~self acc c1) c2
      | For_loop { from_; to_; body; _ } ->
          (* A dead loop ([to_ < from_]) replays zero times: charge nothing, mirroring
             [trace_node_facts] (which records no facts from dead-loop bodies). *)
          if to_ >= from_ then reads_of_proc ~self acc body else acc
      | Scan_loop { from_; to_; carried; body; _ } ->
          let acc = List.fold carried ~init:acc ~f:(fun acc c -> scalar_into ~self acc c.init) in
          if to_ >= from_ then reads_of_proc ~self acc body else acc
      | Set { llsc; _ } -> scalar_into ~self acc llsc
      | Set_dynamic { dyn_value = v, _; llsc; _ } ->
          scalar_into ~self (scalar_into ~self acc v) llsc
      | Set_from_vec { arg = v, _; _ } -> scalar_into ~self acc v
      | Set_local (_, llsc) -> scalar_into ~self acc llsc
      | If { cond = c0, _; body } -> reads_of_proc ~self (scalar_into ~self acc c0) body
    and scalar_into ~self acc v =
      let stmt, cur = reads_of_scalar ~self (acc, Set.empty (module Tnode)) v in
      Set.union stmt cur
    and reads_of_scalar ~self ((stmt, cur) as acc) (sc : scalar_t) =
      match sc with
      | Constant _ | Constant_bits _ | Embed_index _ | Get_local _ | Get_merge_buffer _ -> acc
      | Get (q, _) -> if Tn.equal q self then acc else (stmt, Set.add cur q)
      | Get_dynamic { tn = q; dyn_value = v, _; _ } ->
          reads_of_scalar ~self (stmt, if Tn.equal q self then cur else Set.add cur q) v
      | Local_scope { body; _ } -> (reads_of_proc ~self stmt body, cur)
      | Ternop (op, (v1, _), (v2, _), (v3, _)) -> (
          let acc = reads_of_scalar ~self acc v1 in
          match Ops.ternop_conditionality op with
          | Ops.All_three -> reads_of_scalar ~self (reads_of_scalar ~self acc v2) v3
          | Ops.Cond_and_one_arm ->
              (* Exactly one arm's expression evaluates per visit: charge the wider arm, not the
                 union. The arms' hoisted scope bodies flowed to [stmt] above regardless. *)
              let stmt, cur = acc in
              let stmt, s2 = reads_of_scalar ~self (stmt, Set.empty (module Tnode)) v2 in
              let stmt, s3 = reads_of_scalar ~self (stmt, Set.empty (module Tnode)) v3 in
              (stmt, Set.union cur (if Set.length s3 > Set.length s2 then s3 else s2)))
      | Binop (op, (v1, _), (v2, _)) -> (
          match Ops.binop_conditionality op with
          | Ops.Only_first -> reads_of_scalar ~self acc v1
          | Ops.Only_second -> reads_of_scalar ~self acc v2
          | Ops.Both_operands | Ops.Gated_second ->
              (* A gated second operand still charges: the worst-case evaluation runs both, and with
                 a single alternative the per-arm maximum degenerates to the union. *)
              reads_of_scalar ~self (reads_of_scalar ~self acc v1) v2)
      | Unop (_, (v, _)) -> reads_of_scalar ~self acc v
    in
    let memo = Hashtbl.create (module Tnode) in
    let rec fanin tn : Set.M(Tnode).t =
      match Hashtbl.find memo tn with
      | Some s -> s
      | None ->
          (* Cycle guard: a node re-entered during its own expansion counts as one contributor, as
             if materialized — a cyclically-read node has an uncovered read and was placed
             [On_device] by the coverage rule above anyway. *)
          Hashtbl.set memo ~key:tn ~data:(Set.singleton (module Tnode) tn);
          let expand acc p =
            (* Recurse first: the dependency's own decision lands before its placement is consulted,
               making the result traversal-order-independent. An undecided dependency expands as
               though it will be virtual, though inlining legality is only settled later
               ([check_and_store_virtual], inside the virtualizer this pass feeds) — an
               over-approximation erring toward materialization, the safe direction shared with the
               multiplicity bound; the opposite default would let a chain of undecided links through
               the cap on every first compile. A consumer spuriously materialized this way stays an
               [`Inline] flip candidate, the search's channel for undoing it. *)
            let s_p = fanin p in
            if Tn.Placements.known_non_virtual plc p then Set.add acc p else Set.union acc s_p
          in
          (* Per-setter maximum over the per-setter read sets (see [setter_reads]). *)
          let max_expansion read_sets =
            List.fold read_sets
              ~init:(Set.empty (module Tnode))
              ~f:(fun best reads ->
                let s = Set.fold reads ~init:(Set.empty (module Tnode)) ~f:expand in
                if Set.length s > Set.length best then s else best)
          in
          (* A node committed [Virtual] by an earlier routine in the lineage: its stored
             computation's reads stand in for setters this routine does not see (one stored
             component ≈ one setter, so the same per-setter maximum applies). Computed even when
             this routine sets the node too — a read replays the stored definition AND the local
             updates, and the local update's excluded self-read must not hide the prefix — so the
             inherited maximum unions with the local one. Only prior routines' components exist
             here: this routine's are stored later, by the virtualizer this pass feeds. *)
          let inherited_reads () =
            if not (Tn.Placements.known_virtual plc tn) then []
            else
              match Hashtbl.find optim_ctx.computations tn with
              | Some comps ->
                  List.map comps ~f:(fun (_, code) ->
                      reads_of_proc ~self:tn (Set.empty (module Tnode)) code)
              | None -> []
          in
          let s =
            match Hashtbl.find traced_store tn with
            | None -> (
                match inherited_reads () with
                | [] -> Set.singleton (module Tnode) tn
                | read_sets -> max_expansion read_sets)
            | Some traced ->
                let s =
                  Set.union (max_expansion traced.setter_reads) (max_expansion (inherited_reads ()))
                in
                traced.inline_fanin <- max 1 (Set.length s);
                if
                  Set.length s > virtualize_settings.max_inline_fanin
                  && traced.has_assignment && traced.read_by_other
                  && Option.is_none (Tn.Placements.get plc tn)
                  && not (cap_exempt traced)
                then Tn.Placements.update plc tn Never_virtual 41;
                s
          in
          Hashtbl.set memo ~key:tn ~data:s;
          s
    in
    Hashtbl.iter_keys traced_store ~f:(fun tn -> ignore (fanin tn : Set.M(Tnode).t))

type analysis = {
  an_llc : t;
  an_static_indices : Indexing.static_symbol list;
  an_traced_store : traced_store;
      (** The decision-independent facts. [specialize_proc] hands each candidate a record-copied
          view, because [decide_placements] writes the placement-dependent [read_only] /
          [read_before_write] flags under the candidate's own placements. *)
  an_reverse_node_map : (Indexing.symbol, Tnode.t list) Hashtbl.t;
  an_merge_node : Tnode.t option;
  an_reads_covered :
    (Tn.t -> [ `Covered | `Covered_rmw_exempt of string | `Unknown of string ]) Lazy.t;
  an_read_multiplicity : (Tn.t -> int) Lazy.t;
}

(* Decision-independent analysis of a lowered routine (gh-555 step 1): the structural facts pass,
   the lazily-materialized affine metrics, and the written-bounds pinning — everything
   [specialize_proc] consumes that does not depend on the lineage's placement decisions. Computed
   once per routine; per-candidate compiles replay only [specialize_proc]. *)
let%diagn2_sexp analyze_proc (static_indices : Indexing.static_symbol list) (llc : t) : analysis =
  let traced_store = Hashtbl.create (module Tnode) in
  (* Identifies the computations that the code block associated with the symbol belongs to. *)
  let reverse_node_map = Hashtbl.create (module Indexing.Symbol) in
  [%log "tracing"];
  pin_device_written_bounds llc;
  let merge_node_ref = ref None in
  trace_node_facts traced_store ~merge_node_ref reverse_node_map ~static_indices llc;
  (* The affine metrics are lazily-materialized views over the routine's access relations: only the
     facts the decisions actually consult get computed (gh-554). *)
  let accs = lazy (drop_dead_loop_accesses (affine_accesses llc)) in
  {
    an_llc = llc;
    an_static_indices = static_indices;
    an_traced_store = traced_store;
    an_reverse_node_map = reverse_node_map;
    an_merge_node = !merge_node_ref;
    an_reads_covered = lazy (reads_covered_query static_indices (Lazy.force accs));
    an_read_multiplicity = lazy (read_multiplicity_query static_indices (Lazy.force accs));
  }

(* A candidate-private view of the analysis' traced store: record-level copies, so one candidate's
   [decide_placements] writes ([read_only], [read_before_write] under its own placements) do not
   leak into siblings replaying the same analysis. *)
let copy_traced_store (store : traced_store) : traced_store =
  Hashtbl.map store ~f:(fun t -> { t with tn = t.tn })

(* gh-610: reconcile the traced store with the FINAL optimized code. The traced store is the
   routine's node registry — the kernel parameter list ([C_syntax.compile_proc]), context allocation
   ([Backends.allocate_delta]) and the routine interface ([input_and_output_nodes]) all enumerate it
   — but [analyze_proc] builds it from the RAW code, and cross-routine inlining
   ([inline_computation] splicing a computation an earlier routine of the lineage committed
   [Virtual]) makes the final code diverge from the raw code in both directions. Three
   reconciliations, all against the final code (review round 1 of the gh-610/611 PR):

   - Nodes the raw code never mentions get fresh entries, or C-family codegen emits an identifier no
   parameter declares. A stored computation's setters target the virtual node itself, so splices
   only contribute reads and fresh entries are [read_only]; writes are handled all the same in case
   a future pass splices differently. - A spliced read of an ALREADY-TRACED node can invalidate the
   raw analysis' write-covers-reads conclusion, leaving the node output-only with its incoming value
   silently ignored. A read at a walk position before the node's first write is flipped to
   [read_before_write] directly (a raw pre-write read would already have set it via the coverage
   query); a read AFTER a write is NOT thereby covered — the write may touch only some cells or sit
   under a guard — so such reads are re-judged with the same per-cell machinery the raw pipeline
   uses, [reads_covered_query] over the final code's affine accesses (review round 2: syntactic
   priority is not coverage). The query is built lazily: routines without a
   read-after-write-of-a-traced-node pattern never pay for it. - The gh-ocannl-618 strict interface
   classification closes over the settled placements: a read-modify-write-exempt read still consumes
   the entry values of a node that owns a buffer ([ell[0] = 5000; out[i] = ell[i]] reads ell's
   incoming cells 1.. at the copy position, an accumulation with no preceding definite
   initialization reads its own), so every non-virtual written node whose RAW-analysis coverage is
   exemption-dependent flips to [read_before_write] and is promoted [On_device]. This cannot run in
   [decide_placements] (review round 1 of the gh-617/618 PR): an undecided node can become
   non-virtual AFTER it — the fan-in guard, a [check_and_store_virtual] legality rejection — and
   only here is every node's standing known. Nodes that stay virtual are exempt by construction: a
   virtual node has no interface, and exemption-dependent coverage is exactly the shape of the
   virtualizer's partial-write producers (an injective scatter emits no neutral init; inlining
   prepends the init fallback). - Read-only entries whose node the final code never touches are
   dropped (an all-virtual routine's deferred-computation leaves): they would otherwise read back as
   phantom inputs, parameters, allocations and dependencies of a schedule that does not touch them.
   The prune is deliberately no wider than that genre: an entry recording a raw WRITE keeps its
   place even when no access survives — the vanished write is not splicing's doing (out-of-contract
   scope writes probed at the analysis level per gh-584, optimizer elisions), and the raw
   decision-level facts are the deliverable there. Committed-[Virtual] entries stay too — they are
   not interface material (no parameter, no allocation) but remain introspectable.

   Merge-buffer reads reconcile separately, via the return value: the result says whether the final
   code still reads the merge buffer, so the caller can drop a raw-declared [merge_node] whose read
   was deferred away (keeping it would make linking demand a transfer the schedule never consumes).
   A spliced merge read the routine does NOT declare raises instead: merge buffers hold the payload
   of the transfer preceding THIS routine's run, so deferring a merge read into a later routine
   changes which transfer it observes — that computation must not be shared across routines. *)
let reconcile_traced_store (plc : Tn.Placements.t) (traced_store : traced_store)
    ~(spliced_reads : Tnode.t Hash_set.t) ~(static_indices : Indexing.static_symbol list)
    ~(merge_node : Tnode.t option)
    ~(raw_coverage :
       (Tn.t -> [ `Covered | `Covered_rmw_exempt of string | `Unknown of string ]) Lazy.t)
    ~(cap_inline_flips : Tnode.t Hash_set.t) (llc : t) : bool * Set.M(Tnode).t =
  let accessed = Hash_set.create (module Tnode) in
  let written_seen = Hash_set.create (module Tnode) in
  let fresh_read = Hash_set.create (module Tnode) in
  let fresh_written = Hash_set.create (module Tnode) in
  let suspects = Hash_set.create (module Tnode) in
  let flipped_rbw = Hash_set.create (module Tnode) in
  let uses_merge = ref false in
  (* [live] is false under a dead loop ([to_ < from_]): a dead access still REGISTERS (renderers
     emit dead-loop bodies, so their identifiers need parameters, and its entry must survive the
     prune) but never executes, so it neither supplies coverage (a dead write does not enter
     [written_seen]) nor demands it (a dead read flips no flag), and a node mentioned only in dead
     code gets a flagless entry — registered for rendering, absent from the interface — mirroring
     the raw analysis and [drop_dead_loop_accesses] (review rounds 3-4). *)
  let read ~live tn =
    Hash_set.add accessed tn;
    match Hashtbl.find traced_store tn with
    | None -> if live then Hash_set.add fresh_read tn
    | Some traced ->
        (* Flag flips are gated PER NODE on [spliced_reads] — the nodes read inside inlined bodies
           (review rounds 6-7): splicing is what moves reads to positions the raw analysis never
           judged, so only those nodes get the strict coverage verdicts, while raw-positioned reads
           keep the raw verdicts, whose lenient contracts (rmw exemption, guards-taken) deliberately
           classify patterns whose initialization lives in an earlier routine of the program —
           routine-wide strictness broke real flows across the suite. [zeroed_out] counts as
           written: a [Zero_out]-only node (a [Fetch] of constant 0.) records no [has_assignment],
           yet a spliced read before it still needs the entry value. *)
        if
          live && Hash_set.mem spliced_reads tn
          && (traced.has_assignment || traced.zeroed_out)
          && not traced.read_before_write
        then
          if not (Hash_set.mem written_seen tn) then (
            traced.read_before_write <- true;
            Hash_set.add flipped_rbw tn)
          else Hash_set.add suspects tn
  in
  let written ~live tn =
    Hash_set.add accessed tn;
    if live then (
      Hash_set.add written_seen tn;
      if not (Hashtbl.mem traced_store tn) then Hash_set.add fresh_written tn)
  in
  let rec proc ~live (c : t) =
    match c with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ -> ()
    | Seq (c1, c2) ->
        proc ~live c1;
        proc ~live c2
    | For_loop { from_; to_; body; _ } -> proc ~live:(live && to_ >= from_) body
    | Scan_loop { from_; to_; carried; body; _ } ->
        List.iter carried ~f:(fun c -> scalar ~live c.init);
        proc ~live:(live && to_ >= from_) body
    | Zero_out tn -> written ~live tn
    | Set { tn; llsc; _ } ->
        scalar ~live llsc;
        written ~live tn
    | Set_from_vec { tn; arg = s, _; _ } ->
        scalar ~live s;
        written ~live tn
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        scalar ~live v;
        scalar ~live llsc;
        written ~live tn
    | Set_local (_, llsc) -> scalar ~live llsc
    | If { cond = c0, _; body } ->
        scalar ~live c0;
        proc ~live body
    (* Pre-schedule construct, unreachable here; defensively scan the semantically-equivalent
       fallback. *)
    | Tile_mma { fallback; _ } -> proc ~live fallback
  and scalar ~live (sc : scalar_t) =
    match sc with
    | Constant _ | Constant_bits _ | Embed_index _ | Get_local _ -> ()
    | Get_merge_buffer (source, _) ->
        (* Dead merge reads mirror the raw tracer's dead-body skip (review round 9): the read never
           executes, so it neither validates against the declared merge node nor keeps the
           declaration alive. The SOURCE deliberately does not enter [accessed] in either case: the
           merge buffer is the parameter, and an ordinary traced entry for the source would mint a
           duplicate buffer through [C_syntax.compile_proc]/[allocate_delta] — the raw tracer
           records only [merge_node] (review round 5). *)
        if live then (
          (match merge_node with
          | Some m when Tn.equal m source -> ()
          | _ ->
              raise
                (Utils.User_error
                   [%string
                     "an inlined cross-routine computation reads the merge buffer of \
                      %{Tn.debug_name source}, which is not this routine's declared merge node: \
                      merge-buffer contents are transient to the routine receiving the transfer, \
                      so a computation reading them must not be deferred across routines. Mark the \
                      node computed from the merge buffer as materialized (e.g. via \
                      Train.set_materialized) in the routine that reads the transfer."]));
          uses_merge := true)
    | Get (tn, _) -> read ~live tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        scalar ~live v;
        read ~live tn
    | Local_scope { body; _ } -> proc ~live body
    | Ternop (_, (a, _), (b, _), (d, _)) ->
        scalar ~live a;
        scalar ~live b;
        scalar ~live d
    | Binop (op, (a, _), (b, _)) -> (
        (* The discarded operand of a projection is never rendered, hence never evaluated:
           registering its reads would invent phantom parameters — dispatch through the operand
           conditionality classifier like the affine and tracing walkers (review round 3). A gated
           second operand IS rendered, so it registers. *)
        match Ops.binop_conditionality op with
        | Ops.Only_first -> scalar ~live a
        | Ops.Only_second -> scalar ~live b
        | Ops.Both_operands | Ops.Gated_second ->
            scalar ~live a;
            scalar ~live b)
    | Unop (_, (a, _)) -> scalar ~live a
  in
  proc ~live:true llc;
  let final_accs = lazy (drop_dead_loop_accesses (affine_accesses llc)) in
  (* Reads that follow a write of their node in program order: whether the write actually covers
     them is a per-cell, guard-aware question, answered by the same query that judged the raw reads.
     Lazy so routines without the pattern skip the affine pass over the final code. *)
  (if not (Hash_set.is_empty suspects) then
     (* A guarded write may suppress [read_before_write] only for reads it DOMINATES: when the read
        executes, a same-guard write executed too, but a guard that can be false while the read
        still runs leaves the entry value required (review rounds 4 and 8; the query's own
        guards-taken contract stays for its placement-decision consumers). Guarded READS still
        demand coverage — conservative in the same direction. *)
     let covered =
       reads_covered_query
         ~write_eligible:(fun ~read ~write ->
           (not write.Affine.a_guarded) || write_guards_dominate ~write ~read)
         static_indices (Lazy.force final_accs)
     in
     Hash_set.iter suspects ~f:(fun tn ->
         let traced = Hashtbl.find_exn traced_store tn in
         if not traced.read_before_write then
           match covered tn with
           | `Covered -> ()
           | `Covered_rmw_exempt _ | `Unknown _ ->
               traced.read_before_write <- true;
               Hash_set.add flipped_rbw tn));
  Hash_set.iter fresh_written ~f:(fun tn -> (get_node traced_store tn).has_assignment <- true);
  Hash_set.iter fresh_read ~f:(fun tn ->
      let traced = get_node traced_store tn in
      if traced.has_assignment || traced.zeroed_out then (
        traced.read_before_write <- true;
        Hash_set.add flipped_rbw tn)
      else traced.read_only <- true);
  (* The gh-ocannl-618 strict interface classification, over the SETTLED placements (see the header
     comment): every non-virtual written node whose reads are covered only thanks to the
     read-modify-write exemption consumes its entry values, so it must not classify output-only
     (aliasing-eligible, absent from link-time input verification). Judged on the RAW analysis'
     verdict ([raw_coverage] — review round 4): the fact being closed over is a property of the
     program as analyzed, and the final code can only obscure it — [rewrite_one_hot_reductions]
     turns a raw copy-position self-read into [Get_dynamic], whose coverage is uninterpretable, so a
     final-code query both loses real exemption facts (an uninitialized embedding-gradient
     accumulation) and mints spurious [`Unknown]s (round 3's threefry materialization). ONLY the
     [`Covered_rmw_exempt] verdict flips: this pass closes the exemption split, it does not
     re-litigate coverage — genuinely uncovered raw reads were already flipped by
     [decide_placements], and spliced reads have their own strict path above. A flipped node is also
     promoted [On_device], like [decide_placements]' own rule (same provenance 36): a late-rejected
     candidate is otherwise only [Never_virtual], which [is_materialized_force] would default to
     [Local] — routine scratch with no incoming contents, contradicting the entry values the reads
     consume. Two bookkeeping consequences of promoting (round 4): a cap-provenance entry (1/39/41)
     is recorded in [cap_inline_flips] before being overwritten, so the node keeps its [`Inline]
     flip candidacy (a virtual reading needs no interface classification — the search remains free
     to try it); and a node an EARLIER routine of the lineage committed [Local] cannot be promoted —
     its scratch buffer does not persist, so the in-place update is rejected with the
     materialize-before-first-use error rather than [Placements.update]'s internal transition
     failure. Deliberately NOT recorded in [flipped_rbw]: these are raw-analysis-genre facts, and
     the prior-context demand override is for splice-created flips only (a raw pattern's entry
     values arrive through the assignments layer's curated flows). *)
  Hashtbl.iteri traced_store ~f:(fun ~key:tn ~data:traced ->
      if
        (traced.has_assignment || traced.zeroed_out)
        && (not traced.read_before_write)
        && not (Tn.Placements.known_virtual plc tn)
      then
        match (Lazy.force raw_coverage) tn with
        | `Covered | `Unknown _ -> ()
        | `Covered_rmw_exempt _ ->
            if Tn.Placements.known_not_materialized plc tn then
              raise
                (Utils.User_error
                   [%string
                     "the node %{Tn.debug_name tn} was placed as routine-local scratch by an \
                      earlier routine of this compilation lineage, but this routine updates it in \
                      place (its reads consume the entry values), and a scratch buffer does not \
                      persist between routines. Mark %{Tn.debug_name tn} as materialized (e.g. via \
                      Train.set_materialized) before the first routine using it gets compiled."]);
            (match Tn.Placements.raw_entry plc tn with
            | Some (Never_virtual, (1 | 39 | 41)) -> Hash_set.add cap_inline_flips tn
            | _ -> ());
            traced.read_before_write <- true;
            Tn.Placements.update plc tn On_device 36);
  (* A node mentioned ONLY in dead code still needs a registry entry (its identifier renders, so a
     parameter must declare it and the prune must not drop it) but no interface flags: it neither
     reads nor writes at runtime, and advertising either would create phantom dependencies for
     schedules that execute nothing (review round 4). [get_node] creates the flagless entry; for
     already-entried nodes it is a no-op lookup. *)
  Hash_set.iter accessed ~f:(fun tn -> ignore (get_node traced_store tn : traced_array));
  let stale =
    Hashtbl.fold traced_store ~init:[] ~f:(fun ~key ~data acc ->
        if
          (not (Hash_set.mem accessed key))
          && (not data.has_assignment) && (not data.zeroed_out)
          && not (Tn.Placements.known_virtual plc key)
        then key :: acc
        else acc)
  in
  List.iter stale ~f:(Hashtbl.remove traced_store);
  (!uses_merge, Set.of_list (module Tnode) (Hash_set.to_list flipped_rbw))

(* gh-ocannl-633: a small constant's in-kernel initialization (the unrolled [Constant_fill] fetch)
   is the recipe virtualization inlines; once placement materializes the node, the initialization
   belongs to link time — [Host_inits] uploads the same values into each context — not to the
   kernel, where straight-line whole-node writes are rejected beside hardware-annotated loops
   ([validate_parallel]'s coverage rule) and re-execute on every run. Drop the writes and flip the
   traced facts to the read-only-input shape that above-threshold ndarray literals have, so I/O
   classification and buffer pooling treat both regimes identically.

   Conservative bail-outs keep the in-kernel init whenever: the node's placement is not a context
   buffer ([Local] scratch is fresh per launch, link uploads cannot reach it; [Virtual] definitions
   were already dissolved by cleanup), the node is padded (its init includes padding-region loops),
   or any write to the node is not a literal-constant store. Eligibility is the HOST-CONSTANT
   contract, [Tn.known_host_constant] — values declared forever equal to the registered host-init
   data — not the [Effectively_constant] intent: an explicitly materialized literal
   ([Train.set_materialized] flips the intent to [On_device]) keeps the [host_constant] marker
   (review of gh-ocannl-633, P2). The same contract is what makes any literal-constant write form
   droppable regardless of its indexing — every write such a node carries comes from its own fetch
   lowering and stores the registered values — which covers all three forms the lowering emits:
   unrolled fixed-index [Set]s ([Constant_fill]), loop-borne [Set]s (a broadcast [Constant]), and
   whole-node [Zero_out] ([Constant 0.], whose 1-element literals the
   [--ocannl_limit_constant_fill_size=0] escape cannot reach).

   Only constants the routine also READS convert: in-routine reads are what motivate materializing
   an operand beside its init, and every gh-633 face has them. A write-only constant — a literal
   that IS the routine's root, the [Train.forward_once]-then-print pattern — is an explicit "compute
   this constant into the context" request: converting it would optimize the routine to [Noop] and
   push observation onto the [Host_inits]/for-print-proxy fallbacks, churning behavior for no
   legality gain (no reads, so nothing races the annotated loops... the init alone can, but such a
   routine carries no hardware loops to race with).

   Walker conventions (gh-ocannl-630): the read scan descends operands exhaustively WITHOUT
   [Ops.binop_conditionality] dispatch — counting a projection's discarded read keeps a node
   un-converted, the status-quo direction, and a write-only constant read only by a discarded
   operand is not a real pattern; writes under dead loops need no special-casing — treating a dead
   write as candidate or disqualifier is sound either way (dropping dead code, or keeping live
   init). *)
let hosted_constant_inits_to_link_time (plc : Tn.Placements.t) (traced_store : traced_store)
    (llc : t) : t =
  let candidates : (Tn.t, bool) Hashtbl.t = Hashtbl.create (module Tnode) in
  let reads = Hash_set.create (module Tnode) in
  let eligible tn =
    Tn.known_host_constant tn && Host_inits.mem tn && Option.is_none (Tn.get_padding tn)
  in
  let note_write tn ok =
    if eligible tn then
      Hashtbl.update candidates tn ~f:(function None -> ok | Some prev -> prev && ok)
  in
  let rec scan_scalar (sc : scalar_t) =
    match sc with
    | Constant _ | Constant_bits _ | Embed_index _ | Get_local _ | Get_merge_buffer _ -> ()
    | Get (tn, _) -> Hash_set.add reads tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        Hash_set.add reads tn;
        scan_scalar v
    | Local_scope { body; _ } -> scan body
    | Ternop (_, (a, _), (b, _), (d, _)) ->
        scan_scalar a;
        scan_scalar b;
        scan_scalar d
    | Binop (_, (a, _), (b, _)) ->
        scan_scalar a;
        scan_scalar b
    | Unop (_, (a, _)) -> scan_scalar a
  and scan (c : t) =
    match c with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _ -> ()
    | Seq (c1, c2) ->
        scan c1;
        scan c2
    | For_loop { body; _ } -> scan body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> scan_scalar c.init);
        scan body
    | If { cond = c0, _; body } ->
        scan_scalar c0;
        scan body
    | Set_local (_, llsc) -> scan_scalar llsc
    | Zero_out tn -> note_write tn true
    | Set { tn; llsc; _ } ->
        scan_scalar llsc;
        note_write tn (match llsc with Constant _ -> true | _ -> false)
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        scan_scalar v;
        scan_scalar llsc;
        note_write tn false
    | Set_from_vec { tn; arg = a, _; _ } ->
        scan_scalar a;
        note_write tn false
    (* Pre-schedule construct, unreachable here; defensively treat the target as disqualified. *)
    | Tile_mma { d = tn, _; fallback; _ } ->
        scan fallback;
        note_write tn false
  in
  scan llc;
  let hosted =
    Hashtbl.fold candidates ~init:[] ~f:(fun ~key:tn ~data:ok acc ->
        if ok && Hash_set.mem reads tn && Tn.Placements.is_materialized_peek plc tn then tn :: acc
        else acc)
  in
  if List.is_empty hosted then llc
  else
    let hosted = Set.of_list (module Tnode) hosted in
    let rec rewrite (c : t) : t =
      match c with
      | Seq (c1, c2) -> (
          match (rewrite c1, rewrite c2) with Noop, c | c, Noop -> c | c1, c2 -> Seq (c1, c2))
      | For_loop ({ body; _ } as f) -> (
          match rewrite body with Noop -> Noop | body -> For_loop { f with body })
      | If ({ body; _ } as i) -> (
          match rewrite body with Noop -> Noop | body -> If { i with body })
      (* gh-ocannl-696: the scan's inits are kept as they are (they only read); its body loses the
         hosted writes like any loop body, and an emptied body still rotates, so the scan stays. *)
      | Scan_loop ({ body; _ } as sc) -> Scan_loop { sc with body = rewrite body }
      | Set { tn; _ } when Set.mem hosted tn -> Noop
      | Zero_out tn when Set.mem hosted tn -> Noop
      | c -> c
    in
    let llc = rewrite llc in
    Set.iter hosted ~f:(fun tn ->
        match Hashtbl.find traced_store tn with
        | None -> ()
        | Some traced ->
            traced.has_assignment <- false;
            traced.zeroed_out <- false;
            traced.zero_initialized_by_code <- false;
            traced.read_only <- true);
    llc

let%diagn2_sexp specialize_proc (input_ctx : optimize_ctx) (an : analysis) : optimized =
  let static_indices = an.an_static_indices in
  let llc = an.an_llc in
  let traced_store = copy_traced_store an.an_traced_store in
  decide_placements input_ctx traced_store ~max_visits:virtualize_settings.max_visits
    ~reads_covered:an.an_reads_covered ~read_multiplicity:an.an_read_multiplicity;
  [%log "optimizing"];
  (* gh-ocannl-681: taken BEFORE virtualization, so cleanup can tell the scopes this pass was handed
     from the ones it mints and only retracts its own. *)
  let input_scopes = input_scope_ids llc in
  let virtual_llc_result, spliced_reads =
    virtual_llc input_ctx traced_store an.an_reverse_node_map static_indices llc
  in
  validate_virtualization_decision_coverage input_ctx.placements virtual_llc_result;
  let llc =
    hoist_cross_statement_cse @@ eliminate_common_subexpressions
    @@ rewrite_one_hot_reductions ~static_indices
    @@ hosted_constant_inits_to_link_time input_ctx.placements traced_store
    @@ simplify_llc static_indices
    @@ cleanup_virtual_llc input_ctx.placements ~input_scopes ~static_indices
    @@ virtual_llc_result
  in
  let cap_inline_flips = Hash_set.create (module Tnode) in
  let uses_merge, spliced_rbw =
    reconcile_traced_store input_ctx.placements traced_store ~spliced_reads ~static_indices
      ~merge_node:an.an_merge_node ~raw_coverage:an.an_reads_covered ~cap_inline_flips llc
  in
  (match llc with
  | Noop -> [%log "routine optimized to an empty schedule: every target virtualized (gh-611)"]
  | _ -> ());
  (* The searchable decision dimensions (gh-555), read off the now-committed placements: cleanup has
     committed the surviving virtual candidates, and the backend-compile finalization
     ([default_to_most_local]) has not yet rewritten the cap provenances. *)
  let flip_candidates =
    let plc = input_ctx.placements in
    Hashtbl.fold traced_store ~init:[] ~f:(fun ~key:tn ~data:traced acc ->
        let one_hot = traced.prefers_virtual_one_hot && not traced.has_non_one_hot_setter in
        if
          (not traced.has_assignment) || one_hot || traced.is_scalar_constexpr
          || not traced.read_by_other
        then acc
        else
          let flip =
            (* The raw lineage entry (no intent fallback): only policy decisions are flippable. A
               node whose virtuality is declared intent (e.g. Mixed_prec.Twin_virtual) would make
               [Context.decide_materialized] a no-op and waste a search slot; reading the raw entry
               also keeps tnode-level intent provenances from coincidentally matching the cap
               provenances on the [`Inline] side. *)
            match Tn.Placements.raw_entry plc tn with
            | Some (Virtual, _) when not (Tn.known_virtual tn || Tn.known_constant tn) ->
                Some `Materialize
            | Some (Never_virtual, (1 | 39 | 41)) -> Some `Inline
            (* A cap-selected node the reconcile-stage interface classification promoted [On_device
               36] (gh-618 round 4): the promotion is the interface consequence of the cap's own
               materialization, not a legality/intent decision, so the [`Inline] flip stays
               searchable — a virtual reading has no interface to classify. *)
            | Some (On_device, 36) when Hash_set.mem cap_inline_flips tn -> Some `Inline
            | _ -> None
          in
          match flip with
          | None -> acc
          | Some fc_flip ->
              let mult = max 1 ((Lazy.force an.an_read_multiplicity) tn) in
              {
                fc_tn = tn;
                fc_flip;
                fc_recompute_cost = traced.inline_reduction_extent * mult * traced.inline_fanin;
              }
              :: acc)
    |> List.sort ~compare:(fun a b ->
        match Int.compare b.fc_recompute_cost a.fc_recompute_cost with
        | 0 -> Tn.compare a.fc_tn b.fc_tn
        | c -> c)
  in
  let optimize_ctx = input_ctx in
  {
    traced_store;
    optimize_ctx;
    llc;
    (* A raw-declared merge node whose read was deferred away is dropped: keeping it would make
       linking demand a transfer the final schedule never consumes. *)
    merge_node = (if uses_merge then an.an_merge_node else None);
    workgroup_shared = Set.empty (module Tnode);
    simdgroup_fragments = Set.empty (module Tnode);
    swizzled = Map.empty (module Tnode);
    pipelined = Map.empty (module Tnode);
    zero_fringe = Set.empty (module Tnode);
    flip_candidates;
    spliced_rbw;
  }

(* gh-560: the identity of a routine's analysis inputs — a canonical rendering of the raw lowered
   code plus everything else [analyze_proc] (and the lazy queries it closes over) consults,
   digested. The code walk is [Canonical_render.emit]; what follows is its identity policy plus the
   analysis-specific preamble. Opposite identity choices to [Schedule_cache.canonicalize]'s: tensor
   nodes and static symbols enter by IDENTITY ([Tn.uid]; the symbol's unique ident, with the mutable
   range facts that feed coverage universalization) because a cache hit reuses the stored analysis'
   code verbatim — a same-shape routine over different nodes or differently-bound statics must not
   share an entry — while loop binders and local-scope ids (minted fresh on every lowering) are
   alpha-renamed by first occurrence so sibling lowerings of one routine agree. Comment text is
   skipped: identical code under different names shares one analysis. [None] = not cacheable: an
   opaque statement ([Staged_compilation]) or a duplicated binder would make the rendering
   unfaithful. *)
let analysis_digest (static_indices : Indexing.static_symbol list) (llc : t) : string option =
  let buf = Buffer.create 4096 in
  let add = Buffer.add_string buf in
  let cacheable = ref true in
  (* The read-modify-write exemption setting changes which reads the coverage and multiplicity
     queries count, so it is part of the analysis identity. *)
  if virtualize_settings.inline_complex_computations then add "icc;";
  List.iter static_indices ~f:(fun ss ->
      add
        (Printf.sprintf "%s=[%s%s];"
           (Indexing.symbol_ident ss.Indexing.static_symbol)
           (Option.value_map ss.static_range ~default:"" ~f:Int.to_string)
           (if ss.used_as_extent then ";ext" else "")));
  Canonical_render.emit ~buf
    {
      (* Tensor nodes enter by identity: the stored analysis' code is reused verbatim. *)
      emit_tn = (fun tn -> add ("t" ^ Int.to_string tn.Tn.uid));
      (* Statics and any other free symbols render by their raw ident — see the preamble above. *)
      emit_free_sym = (fun s -> add (Indexing.symbol_ident s));
      on_bind_loop = (fun _ ~id:_ ~shadowed -> if shadowed then cacheable := false);
      mark_incomplete = (fun () -> cacheable := false);
      (* Schedule transforms construct [Tile_mma] after the optimization pipeline ran; it never
         reaches analysis. Defensive. *)
      mma = Canonical_render.Opaque_mma;
      initial_tokens = [];
    }
    llc;
  if not !cacheable then None
  else Some (Stdlib.Digest.to_hex (Stdlib.Digest.string (Buffer.contents buf)))

(* gh-560: sibling candidate compiles of one routine (the placement A/B arms, flip refinements and
   schedule candidates of [Train.tune_placements] / [Autotune.tune] each re-lower the routine from
   the same assignments) share one [analyze_proc] result and replay only [specialize_proc]. Safe by
   the gh-555 hermeticity contract: the analysis is decision-independent, [specialize_proc]
   record-copies the traced store per candidate and only reads the rest. Bounded, process-global,
   move-to-front on hit; entries keyed by stale [Tn.uid]s can never alias fresh nodes (uids are
   never reused), they just age out. *)
let analysis_cache_capacity = 8
let analysis_cache : (string * analysis) list ref = ref []
let analysis_cache_hits = ref 0
let analysis_cache_misses = ref 0
let analysis_cache_stats () = (!analysis_cache_hits, !analysis_cache_misses)
let clear_analysis_cache () = analysis_cache := []

(* Entries retain their routines' code, hence their tensor nodes: drop them before an accessibility
   snapshot — the diagnostic reports user-code liveness, not cache retention. *)
let () =
  Tn.before_accessibility_snapshot := clear_analysis_cache :: !Tn.before_accessibility_snapshot

let cached_analyze_proc (static_indices : Indexing.static_symbol list) (llc : t) : analysis =
  match analysis_digest static_indices llc with
  | None -> analyze_proc static_indices llc
  | Some key -> (
      match List.Assoc.find !analysis_cache key ~equal:String.equal with
      | Some an ->
          Int.incr analysis_cache_hits;
          analysis_cache := (key, an) :: List.Assoc.remove !analysis_cache key ~equal:String.equal;
          (* Re-run the (cheap) written-bounds pinning: it is idempotent on this routine's nodes,
             but its raising guard — a writer compiled after a reader settled the written node's
             bounds — must fire regardless of the cache (see [pin_device_written_bounds]). *)
          pin_device_written_bounds an.an_llc;
          an
      | None ->
          Int.incr analysis_cache_misses;
          let an = analyze_proc static_indices llc in
          analysis_cache := List.take ((key, an) :: !analysis_cache) analysis_cache_capacity;
          an)

(** gh-ocannl-696: the well-formedness contract of {!t.Scan_loop}, as a message rather than an
    exception so both pipeline gates can phrase it. Checked per scan, at any nesting depth: the
    carried pair names one node DECLARED virtual ([known_virtual], not merely undecided -- an
    undecided node another statement reads as a buffer would otherwise surface as a cleanup
    assertion instead of a verdict here), with ids pairwise distinct across the whole list (the
    renderer declares one C local per id); the [init]s read no carried local and do not mention the
    scan's own index, which the renderer binds only after they have evaluated; every [next] is
    written exactly once, as a top-level statement of the body -- a guarded or nested write would
    make the carried update conditional or repeated -- and is read, at any depth, only by statements
    AFTER that write, since the local is declared without a value; and nothing writes a [prev],
    which only the rotation assigns. *)
let scan_loop_violation (plc : Tn.Placements.t) (root : t) : string option =
  let exception Malformed of string in
  let name (id : scope_id) = "v" ^ Int.to_string id.scope_id ^ "_" ^ Tn.debug_name id.tn in
  (* Whether a subtree holds a [Staged_compilation]: a callback emitting code none of the counts
     below can inspect, so a scan containing one has no checkable contract. *)
  let rec has_staged (llc : t) : bool =
    match llc with
    | Staged_compilation _ -> true
    | Seq (a, b) -> has_staged a || has_staged b
    | For_loop { body; _ } | Tile_mma { fallback = body; _ } -> has_staged body
    | If { cond = c, _; body } -> scalar_has_staged c || has_staged body
    | Scan_loop { carried; body; _ } ->
        List.exists carried ~f:(fun c -> scalar_has_staged c.init) || has_staged body
    | Set { llsc; _ } | Set_local (_, llsc) -> scalar_has_staged llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar_has_staged v || scalar_has_staged llsc
    | Set_from_vec { arg = a, _; _ } -> scalar_has_staged a
    | Noop | Comment _ | Zero_out _ | Declare_local _ | Workgroup_barrier -> false
  and scalar_has_staged (llsc : scalar_t) : bool =
    match llsc with
    | Local_scope { body; _ } -> has_staged body
    | Get_dynamic { dyn_value = v, _; _ } -> scalar_has_staged v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar_has_staged a || scalar_has_staged b || scalar_has_staged c
    | Binop (_, (a, _), (b, _)) -> scalar_has_staged a || scalar_has_staged b
    | Unop (_, (a, _)) -> scalar_has_staged a
    | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
        false
  in
  (* Every scope id a subtree mentions -- binders, reads and writes -- at any depth. *)
  let rec scope_ids_in (llc : t) : scope_id list =
    match llc with
    | Declare_local { id; _ } -> [ id ]
    | Set_local (id, llsc) -> id :: scalar_scope_ids llsc
    | Seq (a, b) -> scope_ids_in a @ scope_ids_in b
    | For_loop { body; _ } | Tile_mma { fallback = body; _ } -> scope_ids_in body
    | If { cond = c, _; body } -> scalar_scope_ids c @ scope_ids_in body
    | Scan_loop { carried; body; _ } ->
        List.concat_map carried ~f:(fun c -> c.prev :: c.next :: scalar_scope_ids c.init)
        @ scope_ids_in body
    | Set { llsc; _ } -> scalar_scope_ids llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar_scope_ids v @ scalar_scope_ids llsc
    | Set_from_vec { arg = a, _; _ } -> scalar_scope_ids a
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> []
  and scalar_scope_ids (llsc : scalar_t) : scope_id list =
    match llsc with
    | Get_local id -> [ id ]
    | Local_scope { id; body; _ } -> id :: scope_ids_in body
    | Get_dynamic { dyn_value = v, _; _ } -> scalar_scope_ids v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar_scope_ids a @ scalar_scope_ids b @ scalar_scope_ids c
    | Binop (_, (a, _), (b, _)) -> scalar_scope_ids a @ scalar_scope_ids b
    | Unop (_, (a, _)) -> scalar_scope_ids a
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> []
  in
  let routine_scope_ids = lazy (scope_ids_in root) in
  (* Every scope id a subtree binds: statement-level [Declare_local]s and [Local_scope] binders in
     scalar positions, at any depth. *)
  let rec binders_in (llc : t) : scope_id list =
    match llc with
    | Declare_local { id; _ } -> [ id ]
    | Seq (a, b) -> binders_in a @ binders_in b
    | For_loop { body; _ } | If { body; _ } | Tile_mma { fallback = body; _ } -> binders_in body
    | Scan_loop { carried; body; _ } ->
        List.concat_map carried ~f:(fun c -> scalar_binders c.init) @ binders_in body
    | Set { llsc; _ } | Set_local (_, llsc) -> scalar_binders llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar_binders v @ scalar_binders llsc
    | Set_from_vec { arg = a, _; _ } -> scalar_binders a
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> []
  and scalar_binders (llsc : scalar_t) : scope_id list =
    match llsc with
    | Local_scope { id; body; _ } -> id :: binders_in body
    | Get_dynamic { dyn_value = v, _; _ } -> scalar_binders v
    | Ternop (_, (a, _), (b, _), (c, _)) -> scalar_binders a @ scalar_binders b @ scalar_binders c
    | Binop (_, (a, _), (b, _)) -> scalar_binders a @ scalar_binders b
    | Unop (_, (a, _)) -> scalar_binders a
    | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> []
  in
  let rec count_writes id (llc : t) =
    match llc with
    | Set_local (id', _) -> if equal_scope_id id id' then 1 else 0
    | Seq (a, b) -> count_writes id a + count_writes id b
    | For_loop { body; _ }
    | If { body; _ }
    | Scan_loop { body; _ }
    | Tile_mma { fallback = body; _ } ->
        count_writes id body
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
    | Set _ | Set_dynamic _ | Set_from_vec _ ->
        0
  in
  (* Syntactic occurrences of a scope id -- [Get_local], [Set_local], binders -- at any depth. The
     carried ids are declared inside the scan's own block by the renderer, so an occurrence outside
     the scan is an undeclared identifier: the count over the routine must equal the count over the
     scan (its inits and body). *)
  let rec count_refs id (llc : t) : int =
    match llc with
    | Set_local (id', llsc) -> (if equal_scope_id id id' then 1 else 0) + scalar_refs id llsc
    | Declare_local { id = id'; _ } -> if equal_scope_id id id' then 1 else 0
    | Seq (a, b) -> count_refs id a + count_refs id b
    | For_loop { body; _ } | Tile_mma { fallback = body; _ } -> count_refs id body
    | If { cond = c, _; body } -> scalar_refs id c + count_refs id body
    | Scan_loop { carried; body; _ } ->
        List.sum (module Int) carried ~f:(fun c -> scalar_refs id c.init) + count_refs id body
    | Set { llsc; _ } -> scalar_refs id llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar_refs id v + scalar_refs id llsc
    | Set_from_vec { arg = a, _; _ } -> scalar_refs id a
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> 0
  and scalar_refs id (llsc : scalar_t) : int =
    match llsc with
    | Get_local id' -> if equal_scope_id id id' then 1 else 0
    | Local_scope { id = id'; body; _ } ->
        (if equal_scope_id id id' then 1 else 0) + count_refs id body
    | Get_dynamic { dyn_value = v, _; _ } -> scalar_refs id v
    | Ternop (_, (a, _), (b, _), (c, _)) -> scalar_refs id a + scalar_refs id b + scalar_refs id c
    | Binop (_, (a, _), (b, _)) -> scalar_refs id a + scalar_refs id b
    | Unop (_, (a, _)) -> scalar_refs id a
    | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> 0
  in
  let rec proc (llc : t) =
    match llc with
    | Scan_loop { index; from_; to_; carried; body; _ } ->
        let where = "Scan_loop over " ^ Indexing.symbol_ident index in
        let reject what = raise (Malformed (where ^ ": " ^ what)) in
        (* A dead range is refused rather than given a meaning: every walker would otherwise need
           its own "dead scan" convention (renderer, tracer, censuses, footprint, cost, launch
           geometry), and each omission is a divergence. Refusing it here keeps the construct's only
           shape the live one; a producer with an empty range emits [Noop]. *)
        if to_ < from_ then
          reject
            (Printf.sprintf
               "the range %d..%d is empty; a scan over no iterations is malformed -- emit Noop"
               from_ to_);
        if has_staged body || List.exists carried ~f:(fun c -> scalar_has_staged c.init) then
          reject "a Staged_compilation inside the scan emits code the contract cannot inspect";
        let ids = List.concat_map carried ~f:(fun c -> [ c.prev; c.next ]) in
        List.iteri ids ~f:(fun k id ->
            if List.mem (List.take ids k) id ~equal:equal_scope_id then
              reject ("the id " ^ name id ^ " occurs twice among the carried pairs"));
        (* The carried locals live in the scan's block: a reference after (or before) the scan is an
           undeclared identifier in the rendered source, and the contract says the final state is
           not readable after the loop. *)
        List.iter ids ~f:(fun id ->
            let inside =
              List.sum (module Int) carried ~f:(fun c -> scalar_refs id c.init) + count_refs id body
            in
            if count_refs id root <> inside then
              reject ("the carried id " ^ name id ^ " is referenced outside its scan"));
        (* The per-local censuses of codegen -- accumulator residency and its volatility plumbing
           among them -- key locals by the integer alone, an invariant the pipeline's scope-id
           counter guarantees and hand-built IR can break. The contract guarantees it for carried
           ids: no other local in the routine may share a carried id's integer over another node, so
           no census can conflate carried state with an unrelated local. *)
        List.iter ids ~f:(fun id ->
            List.iter (Lazy.force routine_scope_ids) ~f:(fun other ->
                if other.scope_id = id.scope_id && not (Tn.equal other.tn id.tn) then
                  reject
                    ("the carried id " ^ name id ^ " shares its integer with the unrelated local "
                   ^ name other ^ "; scope-id integers must be unique in the routine")));
        (* A binder of a carried id inside the scan -- a [Declare_local] or a [Local_scope] over it
           -- would render a shadowing C local, resetting the state every iteration. *)
        List.iter
          (binders_in body @ List.concat_map carried ~f:(fun c -> scalar_binders c.init))
          ~f:(fun id ->
            if List.mem ids id ~equal:equal_scope_id then
              reject ("the carried id " ^ name id ^ " is redeclared inside the scan"));
        List.iter carried ~f:(fun { prev; next; init } ->
            if not (Tn.equal prev.tn next.tn) then
              reject ("the carried pair " ^ name prev ^ " / " ^ name next ^ " names two nodes");
            if not (Tn.Placements.known_virtual plc prev.tn) then
              reject
                ("the state node " ^ Tn.debug_name prev.tn
               ^ " is not declared virtual -- carried state lives in per-iteration locals, never a \
                  buffer");
            (* The node names and types the state; a tensor access to it anywhere in the routine
               would ask placement to give it a buffer, against the virtual declaration. *)
            if code_touches_tn prev.tn root then
              reject
                ("the state node " ^ Tn.debug_name prev.tn
               ^ " is accessed as a tensor buffer in the routine; it only names carried state");
            if scalar_mentions_symbol index init then
              reject
                ("the init of " ^ name prev ^ " mentions the scan index "
               ^ Indexing.symbol_ident index ^ ", which is bound only after the inits evaluate");
            List.iter carried ~f:(fun c ->
                if scalar_reads_scope ~id:c.prev init || scalar_reads_scope ~id:c.next init then
                  reject ("the init of " ^ name prev ^ " reads carried state")));
        let top = flat_lines [ body ] in
        List.iter carried ~f:(fun { prev; next; _ } ->
            let at_top =
              List.count top ~f:(function
                | Set_local (id, _) -> equal_scope_id id next
                | _ -> false)
            in
            let anywhere = count_writes next body in
            if at_top <> 1 || anywhere <> 1 then
              reject
                (name next
               ^ " must be written exactly once, as a top-level statement of the body; found "
               ^ Int.to_string at_top ^ " top-level and " ^ Int.to_string anywhere ^ " total");
            if count_writes prev body > 0 then
              reject (name prev ^ " is written in the body; only the rotation assigns a prev"));
        (* A [next] is declared without a value, so a read of it -- at any depth, the defining
           statement's own right-hand side included -- is meaningful only after its top-level write;
           the ordered walk over the body's statements is what decides "after". *)
        let defined = ref [] in
        List.iter top ~f:(fun stmt ->
            List.iter carried ~f:(fun c ->
                if
                  (not (List.mem !defined c.next ~equal:equal_scope_id))
                  && code_reads_scope ~id:c.next stmt
                then reject (name c.next ^ " is read before the statement that writes it"));
            match stmt with
            | Set_local (id, _) when List.exists carried ~f:(fun c -> equal_scope_id c.next id) ->
                defined := id :: !defined
            | _ -> ());
        List.iter carried ~f:(fun c -> scalar c.init);
        proc body
    | Seq (a, b) ->
        proc a;
        proc b
    | For_loop { body; _ } | Tile_mma { fallback = body; _ } -> proc body
    | If { cond = c, _; body } ->
        scalar c;
        proc body
    | Set { llsc; _ } | Set_local (_, llsc) -> scalar llsc
    | Set_dynamic { dyn_value = v, _; llsc; _ } ->
        scalar v;
        scalar llsc
    | Set_from_vec { arg = a, _; _ } -> scalar a
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
        ()
  and scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> proc body
    | Get_dynamic { dyn_value = v, _; _ } -> scalar v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
    | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
  in
  match proc root with () -> None | exception Malformed msg -> Some msg

let validate_scan_loops plc (llc : t) : unit =
  Option.iter (scan_loop_violation plc llc) ~f:(fun msg ->
      invalid_arg ("Low_level.validate_scan_loops: " ^ msg))

let optimize_proc (input_ctx : optimize_ctx) static_indices llc =
  (* gh-ocannl-584: the pipeline's entry gate for scope purity, ahead of the analysis cache so a
     digest hit cannot skip it. Codegen's gate alone would not do: [hoist_cross_statement_cse] lifts
     a body shared by sibling statements to a top-level [Declare_local] + body, which moves an
     impure body's write out of any [Local_scope] -- past the later check, and executing once
     instead of once per user statement. Catch it while it is still visibly a body. *)
  validate_scope_bodies llc;
  (* gh-ocannl-696: the scan contract, same two-gate discipline (codegen holds the exit gate). *)
  validate_scan_loops input_ctx.placements llc;
  specialize_proc input_ctx (cached_analyze_proc static_indices llc)

let code_hum_margin = ref 100

open Indexing.Doc_helpers

let function_header_doc ?name ?static_indices () =
  let open PPrint in
  match (name, static_indices) with
  | Some name, Some static_indices ->
      !^name ^^ space
      ^^ parens (separate comma_sep (List.map ~f:pp_static_symbol static_indices))
      ^^ colon ^^ space
  | Some name, None -> !^name ^^ colon ^^ space
  | _ -> empty

let get_ident_within_code ?no_dots ?(blacklist = []) llcs =
  let ident_style = Tn.get_style ~arg_name:"ll_ident_style" ?no_dots () in
  let nograd_idents = Hashtbl.create (module String) in
  let grad_idents = Hashtbl.create (module String) in
  List.iter blacklist ~f:(fun b_ident ->
      (* Consider blacklisted items as already seen with a placeholder ID like -1 to avoid
         clashes *)
      Hashtbl.set nograd_idents ~key:b_ident ~data:(Set.singleton (module Int) (-1));
      Hashtbl.set grad_idents ~key:b_ident ~data:(Set.singleton (module Int) (-1)));
  let visit tn =
    let is_grad, ident = Tn.no_grad_ident_label tn in
    let idents = if is_grad then grad_idents else nograd_idents in
    Option.iter ident
      ~f:
        (Hashtbl.update idents ~f:(fun old ->
             Set.add (Option.value ~default:Utils.no_ints old) tn.uid))
  in
  let rec loop (c : t) =
    match c with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier -> ()
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | For_loop { body; _ } -> loop body
    | Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c ->
            visit c.prev.tn;
            loop_scalar c.init);
        loop body
    | Zero_out la -> visit la
    | Set { tn; llsc; _ } ->
        visit tn;
        loop_scalar llsc
    | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        visit tn;
        loop_scalar v;
        loop_scalar llsc
    | Set_from_vec { tn; arg = arg_scalar, _; _ } ->
        visit tn;
        loop_scalar arg_scalar
    | Set_local ({ tn; _ }, llsc) ->
        visit tn;
        loop_scalar llsc
    | If { cond = c, _; body } ->
        loop_scalar c;
        loop body
    | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; fallback; _ } ->
        visit d_tn;
        visit a_tn;
        visit b_tn;
        loop fallback
    | Declare_local { id = { tn; _ }; _ } -> visit tn
  and loop_scalar fc =
    match fc with
    | Local_scope { id = { tn; _ }; body; orig_indices = _; mint = _ } ->
        visit tn;
        loop body
    | Get_merge_buffer (la, _) -> visit la
    | Get (la, _) -> visit la
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        visit tn;
        loop_scalar v
    | Ternop (_, (f1, _), (f2, _), (f3, _)) ->
        loop_scalar f1;
        loop_scalar f2;
        loop_scalar f3
    | Binop (_, (f1, _), (f2, _)) ->
        loop_scalar f1;
        loop_scalar f2
    | Unop (_, (f, _)) -> loop_scalar f
    | Get_local { tn; _ } -> visit tn
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
  in
  Array.iter ~f:loop llcs;
  let repeating_nograd_idents =
    Hashtbl.filter nograd_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  let repeating_grad_idents =
    Hashtbl.filter grad_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  fun tn ->
    let ident = Tn.styled_ident ~repeating_nograd_idents ~repeating_grad_idents ident_style tn in
    Tn.update_code_name tn ident;
    ident

let to_doc_cstyle ?name ?static_indices () llc =
  let ident_label = get_ident_within_code [| llc |] in
  let open PPrint in
  let doc_ident la =
    let base = string (ident_label la) in
    if Utils.get_global_flag ~default:false ~arg_name:"output_prec_in_ll_files" then
      let prec_str = Ops.prec_string (Lazy.force la.storage_prec) in
      base ^^ string ("<" ^ prec_str ^ ">")
    else base
  in
  let doc_local { tn; scope_id } = string ("v" ^ Int.to_string scope_id ^ "_") ^^ doc_ident tn in

  let rec doc_of_code c =
    match c with
    | Noop -> empty
    | Seq (c1, c2) ->
        let docs =
          List.filter_map [ c1; c2 ] ~f:(function Noop -> None | c -> Some (doc_of_code c))
        in
        separate hardline docs
    | For_loop { index = i; from_; to_; body; axis } ->
        let header =
          string (axis_type_label axis ^ " ")
          ^^ pp_symbol i ^^ string " = " ^^ int from_ ^^ string " to " ^^ int to_ ^^ string " {"
        in
        let body_doc = nest 2 (break 1 ^^ doc_of_code body) in
        group (header ^^ body_doc ^^ break 1 ^^ string "}")
    | Scan_loop { index = i; from_; to_; direction; carried; body } ->
        let dir = match direction with Forward -> "" | Backward -> " backward" in
        let carry { prev; next; init } =
          doc_local prev ^^ string " <- " ^^ doc_local next ^^ string " := "
          ^^ doc_of_float (Lazy.force prev.tn.storage_prec) init
        in
        let header =
          string ("scan" ^ dir ^ " ")
          ^^ pp_symbol i ^^ string " = " ^^ int from_ ^^ string " to " ^^ int to_
          ^^ string " carrying ["
          ^^ separate (string "; ") (List.map carried ~f:carry)
          ^^ string "] {"
        in
        let body_doc = nest 2 (break 1 ^^ doc_of_code body) in
        group (header ^^ body_doc ^^ break 1 ^^ string "}")
    | Workgroup_barrier -> string "workgroup_barrier;"
    | Tile_mma
        { d = d_tn, d_idcs; a = a_tn, a_idcs; b = b_tn, b_idcs; ta; tb; m; n; k; lane; fallback; _ }
      ->
        let transposed t = if t then string "^T" else empty in
        let header =
          string (Printf.sprintf "tile_mma<%dx%dx%d>@" m n k)
          ^^ pp_symbol lane ^^ string " " ^^ doc_ident d_tn
          ^^ brackets (pp_indices d_idcs)
          ^^ string " += " ^^ doc_ident a_tn
          ^^ brackets (pp_indices a_idcs)
          ^^ transposed ta ^^ string " * " ^^ doc_ident b_tn
          ^^ brackets (pp_indices b_idcs)
          ^^ transposed tb ^^ string " fallback {"
        in
        group (header ^^ nest 2 (break 1 ^^ doc_of_code fallback) ^^ break 1 ^^ string "}")
    | If { cond = c, cprec; body } ->
        let header = string "if " ^^ doc_of_float cprec c ^^ string " != 0 {" in
        group (header ^^ nest 2 (break 1 ^^ doc_of_code body) ^^ break 1 ^^ string "}")
    | Zero_out tn -> string "zero_out " ^^ doc_ident tn ^^ string ";"
    | Set p ->
        let prec = Lazy.force p.tn.storage_prec in
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ string " := " ^^ doc_of_float prec p.llsc ^^ string ";")
        in
        if not (String.is_empty p.debug) then (
          let b = Buffer.create 100 in
          PPrint.ToBuffer.pretty 0.7 100 b result;
          p.debug <- Buffer.contents b);
        result
    | Set_dynamic p ->
        let prec = Lazy.force p.tn.storage_prec in
        let v, vprec = p.dyn_value in
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ string (Printf.sprintf "@dyn[%d]=" p.dyn_axis)
            ^^ parens (doc_of_float vprec v)
            ^^ string " := " ^^ doc_of_float prec p.llsc ^^ string ";")
        in
        if not (String.is_empty p.debug) then (
          let b = Buffer.create 100 in
          PPrint.ToBuffer.pretty 0.7 100 b result;
          p.debug <- Buffer.contents b);
        result
    | Set_from_vec p ->
        let prec = Lazy.force p.tn.storage_prec in
        let prefix, postfix = Ops.vec_unop_c_syntax prec p.vec_unop in
        (* TODO: this assumes argument is generated from the high-level code, which means it is
           either Get or Local_scope -- they don't need precision. *)
        let arg_scalar, _arg_prec = p.arg in
        let vec_result = string prefix ^^ doc_of_float Ops.Void_prec arg_scalar ^^ string postfix in
        let length_doc = string ("<" ^ Int.to_string p.length ^ ">") in
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ length_doc ^^ string " := " ^^ vec_result ^^ string ";")
        in
        if not (String.is_empty p.debug) then (
          let b = Buffer.create 100 in
          PPrint.ToBuffer.pretty 0.7 100 b result;
          p.debug <- Buffer.contents b);
        result
    | Comment message -> string ("/* " ^ message ^ " */")
    | Staged_compilation callback -> callback ()
    | Set_local (id, llsc) ->
        let prec = Lazy.force id.tn.storage_prec in
        group (doc_local id ^^ string " := " ^^ doc_of_float prec llsc ^^ string ";")
    | Declare_local { id; _ } -> group (string "declare " ^^ doc_local id ^^ string ";")
  and doc_of_float prec value =
    match value with
    | Local_scope { id; body; _ } ->
        group
          (doc_local id ^^ string " {"
          ^^ nest 2 (break 1 ^^ doc_of_code body)
          ^^ break 1 ^^ string "}")
    | Get_local id -> doc_local id
    | Get_merge_buffer (source, idcs) ->
        group (doc_ident source ^^ string ".merge" ^^ brackets (pp_indices idcs))
    | Get (tn, idcs) -> group (doc_ident tn ^^ brackets (pp_indices idcs))
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, vprec } ->
        group
          (doc_ident tn
          ^^ brackets (pp_indices idcs)
          ^^ string (Printf.sprintf "@dyn[%d]=" dyn_axis)
          ^^ parens (doc_of_float vprec v))
    | Constant c -> string (Utils.decimal_float_literal c)
    | Constant_bits i -> string (Printf.sprintf "0x%LX" i)
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        if PPrint.is_empty idx_doc then string "0" else idx_doc
    | Ternop (op, (v1, _), (v2, _), (v3, _)) ->
        let prefix, comma1, comma2, postfix = Ops.ternop_c_syntax prec op in
        group
          (string prefix ^^ doc_of_float prec v1 ^^ string comma1 ^^ space ^^ doc_of_float prec v2
         ^^ string comma2 ^^ space ^^ doc_of_float prec v3 ^^ string postfix)
    | Binop (Arg1, (v1, _), _v2) -> doc_of_float prec v1
    | Binop (Arg2, _v1, (v2, _)) -> doc_of_float prec v2
    | Binop (op, (v1, _), (v2, _)) ->
        let prefix, infix, postfix = Ops.binop_c_syntax prec op in
        group
          (string prefix ^^ doc_of_float prec v1 ^^ string infix ^^ space ^^ doc_of_float prec v2
         ^^ string postfix)
    | Unop (Identity, (v, _)) -> doc_of_float prec v
    | Unop (op, (v, _)) ->
        let prefix, postfix = Ops.unop_c_syntax prec op in
        string prefix ^^ doc_of_float prec v ^^ string postfix
  in
  hardline ^^ nest 2 (function_header_doc ?name ?static_indices () ^^ doc_of_code llc)

let to_doc ?name ?static_indices () llc =
  let ident_label = get_ident_within_code [| llc |] in
  let open PPrint in
  let doc_ident la =
    let base = string (ident_label la) in
    if Utils.get_global_flag ~default:false ~arg_name:"output_prec_in_ll_files" then
      let prec_str = Ops.prec_string (Lazy.force la.storage_prec) in
      base ^^ string ("<" ^ prec_str ^ ">")
    else base
  in
  let doc_local { tn; scope_id } = string ("v" ^ Int.to_string scope_id ^ "_") ^^ doc_ident tn in

  let rec doc_of_code c =
    match c with
    | Noop -> empty
    | Seq (c1, c2) ->
        let docs =
          List.filter_map [ c1; c2 ] ~f:(function Noop -> None | c -> Some (doc_of_code c))
        in
        separate hardline docs
    | For_loop { index = i; from_; to_; body; axis } ->
        let header =
          string (axis_type_label axis ^ " ")
          ^^ pp_symbol i ^^ string " = " ^^ int from_ ^^ string " to " ^^ int to_ ^^ string " {"
        in
        let body_doc = nest 2 (break 1 ^^ doc_of_code body) in
        group (header ^^ body_doc ^^ break 1 ^^ string "}")
    | Scan_loop { index = i; from_; to_; direction; carried; body } ->
        let dir = match direction with Forward -> "" | Backward -> " backward" in
        let carry { prev; next; init } =
          doc_local prev ^^ string " <- " ^^ doc_local next ^^ string " := " ^^ doc_of_float init
        in
        let header =
          string ("scan" ^ dir ^ " ")
          ^^ pp_symbol i ^^ string " = " ^^ int from_ ^^ string " to " ^^ int to_
          ^^ string " carrying ["
          ^^ separate (string "; ") (List.map carried ~f:carry)
          ^^ string "] {"
        in
        let body_doc = nest 2 (break 1 ^^ doc_of_code body) in
        group (header ^^ body_doc ^^ break 1 ^^ string "}")
    | Workgroup_barrier -> string "workgroup_barrier;"
    | Tile_mma
        { d = d_tn, d_idcs; a = a_tn, a_idcs; b = b_tn, b_idcs; ta; tb; m; n; k; lane; fallback; _ }
      ->
        let transposed t = if t then string "^T" else empty in
        let header =
          string (Printf.sprintf "tile_mma<%dx%dx%d>@" m n k)
          ^^ pp_symbol lane ^^ string " " ^^ doc_ident d_tn
          ^^ brackets (pp_indices d_idcs)
          ^^ string " += " ^^ doc_ident a_tn
          ^^ brackets (pp_indices a_idcs)
          ^^ transposed ta ^^ string " * " ^^ doc_ident b_tn
          ^^ brackets (pp_indices b_idcs)
          ^^ transposed tb ^^ string " fallback {"
        in
        group (header ^^ nest 2 (break 1 ^^ doc_of_code fallback) ^^ break 1 ^^ string "}")
    | If { cond = c, _; body } ->
        let header = string "if " ^^ doc_of_float c ^^ string " != 0 {" in
        group (header ^^ nest 2 (break 1 ^^ doc_of_code body) ^^ break 1 ^^ string "}")
    | Zero_out tn -> string "zero_out " ^^ doc_ident tn ^^ string ";"
    | Set p ->
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ string " := " ^^ doc_of_float p.llsc ^^ string ";")
        in
        let b = Buffer.create 100 in
        PPrint.ToBuffer.pretty 0.7 100 b result;
        p.debug <- Buffer.contents b;
        result
    | Set_dynamic p ->
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ string (Printf.sprintf "@dyn[%d]=" p.dyn_axis)
            ^^ parens (doc_of_float (fst p.dyn_value))
            ^^ string " := " ^^ doc_of_float p.llsc ^^ string ";")
        in
        let b = Buffer.create 100 in
        PPrint.ToBuffer.pretty 0.7 100 b result;
        p.debug <- Buffer.contents b;
        result
    | Set_from_vec p ->
        let length_doc = string ("<" ^ Int.to_string p.length ^ ">") in
        let result =
          group
            (doc_ident p.tn
            ^^ brackets (pp_indices p.idcs)
            ^^ length_doc ^^ string " := "
            ^^ string (Ops.vec_unop_cd_syntax p.vec_unop)
            ^^ string "("
            ^^ doc_of_float (fst p.arg)
            ^^ string ", " ^^ length_doc ^^ string ");")
        in
        let b = Buffer.create 100 in
        PPrint.ToBuffer.pretty 0.7 100 b result;
        p.debug <- Buffer.contents b;
        result
    | Comment message -> string ("/* " ^ message ^ " */")
    | Staged_compilation callback -> callback ()
    | Set_local (id, llsc) ->
        group (doc_local id ^^ string " := " ^^ doc_of_float llsc ^^ string ";")
    | Declare_local { id; _ } -> group (string "declare " ^^ doc_local id ^^ string ";")
  and doc_of_float value =
    match value with
    | Local_scope { id; body; _ } ->
        group
          (doc_local id ^^ string " {"
          ^^ nest 2 (break 1 ^^ doc_of_code body)
          ^^ break 1 ^^ string "}")
    | Get_local id -> doc_local id
    | Get_merge_buffer (source, idcs) ->
        group (doc_ident source ^^ string ".merge" ^^ brackets (pp_indices idcs))
    | Get (tn, idcs) -> group (doc_ident tn ^^ brackets (pp_indices idcs))
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, _ } ->
        group
          (doc_ident tn
          ^^ brackets (pp_indices idcs)
          ^^ string (Printf.sprintf "@dyn[%d]=" dyn_axis)
          ^^ parens (doc_of_float v))
    | Constant c -> string (Utils.decimal_float_literal c)
    | Constant_bits i -> string (Printf.sprintf "0x%LX" i)
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        if PPrint.is_empty idx_doc then string "0" else idx_doc
    | Ternop (op, (v1, _), (v2, _), (v3, _)) ->
        let prefix = Ops.ternop_cd_syntax op in
        group
          (string prefix
          ^^ parens
               (doc_of_float v1 ^^ string "," ^^ space ^^ doc_of_float v2 ^^ string "," ^^ space
              ^^ doc_of_float v3))
    | Binop (Arg1, (v1, _), _v2) -> doc_of_float v1
    | Binop (Arg2, _v1, (v2, _)) -> doc_of_float v2
    | Binop (op, (v1, _), (v2, _)) ->
        if Ops.is_binop_nice_infix op then
          let infix = Ops.binop_cd_syntax op in
          group (parens (doc_of_float v1 ^^ space ^^ string infix ^^ space ^^ doc_of_float v2))
        else
          let prefix = Ops.binop_cd_fallback_syntax op in
          group (string prefix ^^ parens (doc_of_float v1 ^^ string "," ^^ space ^^ doc_of_float v2))
    | Unop (Identity, (v, _)) -> doc_of_float v
    | Unop (op, (v, _)) ->
        let prefix = Ops.unop_cd_syntax op in
        string prefix ^^ parens (doc_of_float v)
  in

  hardline ^^ nest 2 (function_header_doc ?name ?static_indices () ^^ doc_of_code llc)

let%diagn2_sexp optimize (input_ctx : optimize_ctx) ~unoptim_ll_source ~ll_source ~(name : string)
    (static_indices : Indexing.static_symbol list) (llc : t) : optimized =
  Option.iter unoptim_ll_source ~f:(fun callback -> callback (to_doc ~name ~static_indices () llc));
  let result = optimize_proc input_ctx static_indices llc in
  Option.iter ll_source ~f:(fun callback -> callback (to_doc ~name ~static_indices () result.llc));
  result

let loop_over_dims dims ~body =
  let rec for_loop rev_idcs : _ -> t = function
    | [] -> body @@ Array.of_list_rev rev_idcs
    | d :: product when not @@ Indexing.iterated d ->
        for_loop (Indexing.Fixed_idx 0 :: rev_idcs) product
    | d :: product ->
        let index = Indexing.get_symbol () in
        For_loop
          {
            index;
            from_ = 0;
            to_ = d - 1;
            body = for_loop (Indexing.Iterator index :: rev_idcs) product;
            axis = Serial;
          }
  in
  for_loop [] (Array.to_list dims)

let unroll_dims dims ~body =
  if Array.is_empty dims then body [||] ~offset:0
  else
    (* Calculate strides for each dimension (rightmost changes fastest) *)
    let strides = Array.create ~len:(Array.length dims) 1 in
    for i = Array.length dims - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * dims.(i + 1)
    done;

    (* Generate all combinations of indices *)
    let rec generate_all_combinations indices_so_far offset dim_index =
      if dim_index >= Array.length dims then
        (* We have a complete combination, call the body *)
        body (Array.of_list_rev indices_so_far) ~offset
      else
        (* Generate all values for current dimension *)
        let results = ref [] in
        for i = 0 to dims.(dim_index) - 1 do
          let new_offset = offset + (i * strides.(dim_index)) in
          let result =
            generate_all_combinations
              (Indexing.Fixed_idx i :: indices_so_far)
              new_offset (dim_index + 1)
          in
          results := result :: !results
        done;
        unflat_lines (List.rev !results)
    in
    generate_all_combinations [] 0 0

let loop_over_padding_region ~dims ~(padding : Ops.axis_padding array) ~body =
  (* Generate loops that iterate ONLY over the padding margins (NOT the data region).

     The padding region is the union of "strips" where at least one dimension's index is in the
     padding range [0, left) or [dim-right, dim).

     For each dimension with padding, we generate: 1. Left padding strip: index in [0, left) -
     iterate ALL remaining dims 2. Middle: index in [left, dim-right) - recurse to find padding in
     other dims 3. Right padding strip: index in [dim-right, dim) - iterate ALL remaining dims

     For dimensions with NO padding, we just iterate the full range while recursing.

     The recursion stops when we've processed all dimensions. If we reach the end without any
     dimension having contributed padding, we DON'T call body (that's data). *)
  let rec build_loops ~any_padding_so_far dim_idx rev_idcs =
    if dim_idx >= Array.length dims then
      (* Only generate body if we're actually in a padding region *)
      if any_padding_so_far then body @@ Array.of_list_rev rev_idcs else Noop
    else
      let dim = dims.(dim_idx) in
      let pad = padding.(dim_idx) in
      let index = Indexing.get_symbol () in
      let has_padding = pad.left > 0 || pad.right > 0 in
      if not has_padding then
        (* No padding on this dimension - iterate full range, keep looking for padding *)
        For_loop
          {
            index;
            from_ = 0;
            to_ = dim - 1;
            body =
              build_loops ~any_padding_so_far (dim_idx + 1) (Indexing.Iterator index :: rev_idcs);
            axis = Serial;
          }
      else
        (* Has padding - generate left strip, middle (recurse), right strip *)
        let left_loop =
          if pad.left > 0 then
            For_loop
              {
                index;
                from_ = 0;
                to_ = pad.left - 1;
                body =
                  (* In left padding - iterate ALL remaining dims (they're all in padding region) *)
                  loop_over_dims
                    (Array.sub dims ~pos:(dim_idx + 1) ~len:(Array.length dims - dim_idx - 1))
                    ~body:(fun rest_idcs ->
                      body
                      @@ Array.concat
                           [ Array.of_list_rev rev_idcs; [| Indexing.Iterator index |]; rest_idcs ]);
                axis = Serial;
              }
          else Noop
        in
        let middle_loop =
          let middle_from = pad.left in
          let middle_to = dim - pad.right - 1 in
          if middle_from <= middle_to then
            For_loop
              {
                index;
                from_ = middle_from;
                to_ = middle_to;
                body =
                  (* In middle - NOT in padding for this dim, recurse to find other padded dims *)
                  build_loops ~any_padding_so_far (dim_idx + 1) (Indexing.Iterator index :: rev_idcs);
                axis = Serial;
              }
          else Noop
        in
        let right_loop =
          if pad.right > 0 then
            let right_index = Indexing.get_symbol () in
            For_loop
              {
                index = right_index;
                from_ = dim - pad.right;
                to_ = dim - 1;
                body =
                  (* In right padding - iterate ALL remaining dims *)
                  loop_over_dims
                    (Array.sub dims ~pos:(dim_idx + 1) ~len:(Array.length dims - dim_idx - 1))
                    ~body:(fun rest_idcs ->
                      body
                      @@ Array.concat
                           [
                             Array.of_list_rev rev_idcs;
                             [| Indexing.Iterator right_index |];
                             rest_idcs;
                           ]);
                axis = Serial;
              }
          else Noop
        in
        unflat_lines [ left_loop; middle_loop; right_loop ]
  in
  build_loops ~any_padding_so_far:false 0 []
