(* The serial-rendered forms of ONE reduction, enumerated mechanically (gh-ocannl-664).

   The set of forms a single reduction loop can reach at codegen is defined implicitly, by the union
   of the schedule ops that can rewrite it and the codegen arms that can render the result. Three
   optimizer arcs enumerated its members one review round at a time — gh-ocannl-639 took six rounds
   over [Unroll] (both representations), sequential double materialization, [Pad]-guarded copies,
   [Partition] segments, Retype-[Vectorized] and Retype-[Workgroup_reduce]; gh-ocannl-663 added
   schedule-minted vs codegen-minted scopes, [Tile_mma] fallbacks, the rng-exclusion granularity,
   virtualization's [Where]-guarded update spelling and init-vs-update rendering; gh-ocannl-693
   added the localized-at-identity-precision form and the peeled-guard shapes. Every one of those
   was a point in the same product, found by a reviewer rather than by a test.

   This test defines the set EXTENSIONALLY: a table of (composition x storage precision), each
   member naming the form it claims to reach, executed and read back. Three claims per member, and
   each makes the one before it trustworthy:

   - VALUE. A localizing member must agree BITWISE with the serial baseline's execution; a
   read-modify-write member must agree bitwise with the host's per-step-narrowed reference. Both
   references are exact: the operands are storage-exact multiples of 1/8 whose PARTIAL SUMS leave
   the storage format's exactness range (see {!cells}), so the two references differ at bf16 and f16
   — proven here rather than assumed, by the "the whole-nest and per-step references differ" claims.
   At f32 they coincide, which is the identity-precision leg gh-ocannl-693 added: there the value
   claim pins substitution and iteration coverage rather than accumulator width. - FORM. The emitted
   kernel is classified — localized scope, per-step read-modify-write, SIMD accumulator grid,
   warp-shuffle tree, [Tile_mma] scalar fallback — and must be the form the member claims. Without
   this a composition that silently stopped reaching its form would keep passing its value claim by
   falling back to another one, which is exactly the false green the issue is about: agreement
   between two renderings is worthless if they are the same rendering. - DECISION (gh-ocannl-733).
   The form does not determine WHICH code path produced it: a nest whose accumulated cell is free of
   the enclosing index is peeled at the enclosing level under a confined guard, while one whose cell
   mentions it peels the reduction level alone under a lane-private guard the cell separates — and
   both emit one localized scope with one closing store. So each member also declares what codegen's
   reduction peel decided, checked against the routine's own peel census ({!Context.routine.peel})
   rather than against emitted text. Without it a member can be green over a kernel the code path it
   is named for never touched, which is the same false green one level down: agreement between two
   decisions is worthless if they produce the same rendering.

   The member list itself is printed, so a form added to codegen without a table entry shows up as a
   golden diff rather than as silence. Members a backend cannot evaluate (SIMD grids off the C
   backends, warp shuffles off the GPUs) stay in the table and report {!Verdict.skipped}.

   The reduction is hand-built {!Ir.Low_level} (via [ll_test]) so that the nest shape is the test's,
   not shape inference's, and so that the loop symbols are in hand: every composition names its axes
   directly instead of re-deriving them from the lowered nest.

   The table is also the WIDTH-PARITY corpus gh-ocannl-754 asks for: every rendering that holds an
   accumulator somewhere other than its storage cell — the localized scope, the SIMD grid, the
   shuffle tree — takes its "does this accumulator widen?" answer from one shared decision
   ([C_syntax.accum_width]), and the members over the awkward bodies (the RNG-bearing contribution,
   whose accumulator is pinned to storage; a sibling statement; a data-dependent guard) under each
   retype kind are what pin that agreement: a retype must equal the serial rendering bitwise at a
   narrow storage precision where the residency is wider, which is where a width divergence shows.
   Two census claims say which way the decision went: {!Widened_elsewhere} (the decision was wide
   and the grid or the tree took it) and {!Pinned_to_storage} (the decision pinned, and nothing
   widened).

   Two points of the space are deliberately absent, and saying so is part of defining the set:

   - The [Grid] arm, whose fallback is the plain serial loop and NOT the localizing one — the one
   arm that can serialize without localizing. No schedule op in the tree produces an unbindable,
   non-parallel-eligible [Grid] level over a reduction axis, and one that existed would be a
   cross-thread race rather than a width question, so a member here could only be built by hand out
   of a shape the pipeline cannot reach. - The HONOURED register-tile and tensor-core renderings.
   Those hold their accumulator in a C-tile rather than in a serial nest, so they are not serial
   forms of this reduction; [tile_mma_narrow] pins their width. What is a member is the [Tile_mma]
   SCALAR FALLBACK, whose reduction is a serial nest like any other. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Cs = Ir.C_syntax
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Ops = Ir.Ops
module Sched = Ir.Schedule
module Numerics = Ir.Numerics
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Generated.init ~backend_name
let on_cpu = Sched.backend_is_cpu backend_name
let codegen_capabilities = Context.codegen_capabilities (Context.auto ())

open Verdict.Claims

let skipped = Verdict.skipped ~backend:backend_name

(* {1 The reduction}

   [out[r] = sum_k x[r, k]] over [rows] x [cols]. Small on purpose: every member compiles its own
   kernel, and the table is a cross product. *)

let rows = 3

(* A multiple of the GPU backends' warp size, so that the [Workgroup_reduce] member can reach the
   shuffle-tree rendering where the backend has one: that rendering REFUSES (loudly, by design — a
   plain hardware binding would race the accumulator update) an extent it cannot cover a warp at a
   time.

   TWO warps rather than one, so that splitting the axis can leave a 32-wide inner half under a
   surviving outer loop. A 4-wide one is not portable: [simd_reduce_lanes_for] filters the lane
   ladder by [extent >= lanes], and the ladder starts at 8 for f32 on a 32-byte-vector x86 target —
   so the inner half vectorized on this NEON machine and silently declined to the localizing peel on
   the Linux CI runner, which is a form claim that holds on the author's hardware and nowhere else.
   32 clears every ladder a real target has. *)
let cols = 64

(* The operand cells: [(flat mod 67 + 180) / 8], i.e. the multiples of 1/8 between 22.5 and 30.75.

   Three properties, each arithmetic rather than a rule of thumb (see {!Ll_test.cycle}).

   EXACT in storage: in units of 1/8 every cell is an integer in [180, 246], and a format with [p]
   significand bits holds every integer up to [2^p] — 246 < 256 = 2^8 — so the cells are exact in
   bf16, in f16 and in f32. That is what lets a host reference reproduce the kernel bitwise, and it
   is why the offset is 180 rather than something larger: 67 + offset must stay under 256.

   DRIFTING out of it: the running sum passes 2^8 units within two terms and 2^11 within ten, so a
   per-step narrowing to bf16 or f16 visibly diverges from a whole-nest accumulation over both the
   full 64-term axis and the 20-term guarded prefix, while the total (~13732 units, ~1716.5) stays
   exact in f32.

   APERIODIC over the reduction axis (Codex P2, round 8). [flat = 64 r + k], so a modulus [m] makes
   [cell r (k + d) = cell r k] exactly when [m | d]: any [m <= 63] therefore has an in-range period,
   and the previous 13 meant an [Unroll], [Split] or [Partition] regression that dropped one
   iteration and replayed a copy 13 positions away preserved every executed value AND every emitted
   count — defeating the value arm this suite leans on to pin substitution and iteration coverage.
   67 exceeds the extent, so no in-range step is a period; being odd it is coprime to the row stride
   64, so no in-range row step is one either, and 64 consecutive [k] give 64 distinct residues. It
   still varies with BOTH symbols, which {!Ll_test.cycle} raises rather than let lapse. *)
let cells = Ll_test.cycle ~dims:[| rows; cols |] ~modulus:67 ~offset:180. ~stride:0.125
let cell r k = cells [| r; k |]
let x_values = Array.init (rows * cols) ~f:(fun n -> cell (n / cols) (n % cols))

(* The accumulator's INCOMING contents: distinct per row and nonzero (Codex P2, round 1). With every
   [out] cell and both references starting at zero, a regression that initialized a localized
   accumulator to zero instead of loading [out[r]] would satisfy every form and every value claim —
   while the same [out[r] += x[r,k]] IR computes the wrong thing for any caller whose destination
   already holds data, which is what an accumulation into a pre-zeroed or previously-written node
   is. In units of 1/8 these are 100, 107, 114: distinct, well inside bf16's exact range, and in the
   same units as the operands so the sums stay f32-exact. *)
let seed_value r = (100. +. (7. *. Float.of_int r)) /. 8.
let out_seed = Array.init rows ~f:seed_value

(* Host references, per storage precision. [round] is the library's own conversion, so these are the
   kernel's roundings and not an approximation of them. *)
let round (prec : Ops.prec) v =
  match prec with
  | Ops.Bfloat16_prec _ -> Ops.bfloat16_to_single (Ops.single_to_bfloat16 v)
  | Ops.Half_prec _ -> Ops.half_to_single (Ops.single_to_half v)
  | Ops.Fp8_prec _ -> Ops.fp8_to_single (Ops.single_to_fp8 v)
  | _ -> v

(* How many of the [cols] terms a guarded shape's guard admits. INTERIOR, not the whole axis (Codex
   P2, round 2): a guard bound to the full extent is true on every iteration, so a schedule mint
   that DROPPED or broadened the predicate while rewriting the nest would still produce the
   localized form and still compute the baseline's value — both claims green over a guard that had
   ceased to exist. With an interior extent the reference is a partial sum, and a lost predicate or
   a lost iteration is a different number. *)
let guard_terms = 20

(* The whole-nest reference: open the cell, accumulate wide, narrow once at the store. [~from] is
   what the accumulator loads — the seed for every shape that reads [out], zero for the virtual
   accumulator, which owns its cell and initializes it from a [Zero_out]. [~terms] is a function of
   the ROW rather than a count: {!Mixed_guard}'s predicate mixes the output index into the bound, so
   its rows admit different prefixes of the reduction axis — which is also what makes a regression
   that substituted the wrong row into the guard a different number. *)
let whole_nest_ref ~terms ~from prec =
  Array.init rows ~f:(fun r ->
      let acc = ref (from r) in
      for k = 0 to terms r - 1 do
        acc := !acc +. cell r k
      done;
      round prec !acc)

(* The per-step reference: narrow to storage at every accumulation step, which is what a
   read-modify-write rendering does. The addition itself happens at compute precision, and the
   operands are storage-exact, so one rounding per step is the whole of it. *)
let per_step_ref ~terms ~from prec =
  Array.init rows ~f:(fun r ->
      let acc = ref (from r) in
      for k = 0 to terms r - 1 do
        acc := round prec (!acc +. cell r k)
      done;
      !acc)

let precs = [ ("f32", Ops.single); ("bf16", Ops.bfloat16); ("f16", Ops.half) ]

(* Whether an op is a pure LOOP REWRITE, hence safe to apply a second time as a probe. The ops
   [apply_op] handles rewrite the nest and mint nothing; the ones [apply_opt_op] handles need the
   whole [optimized] record and can MINT NODES — [Split_reduce] creates a partials node, so probing
   it and then applying it for real produced two nodes with one name and a C function with a
   duplicated parameter. Exhaustive, so a new constructor has to answer the question. *)
let is_loop_rewrite (op : Sched.optop) =
  match op with
  | Sched.Split _ | Sched.Swap _ | Sched.Retype _ | Sched.Unroll _ | Sched.Partition _ | Sched.Pad _
    ->
      true
  | Sched.Split_reduce _ | Sched.Tensorize _ | Sched.Stage _ | Sched.Privatize _
  | Sched.Expand_zero _ | Sched.Fuse_epilogue _ ->
      false

(* One root context for the whole run: [Context.compile] forks the lineage per compile, so members
   do not observe each other's placement decisions. It is also what answers the target question
   below, so it is created before the policy is resolved rather than at the first compile. *)
let base_ctx = lazy (Context.auto ())
let selected_backend = lazy (Context.backend_name (Lazy.force base_ctx))
let native_fp16 = lazy (Context.hardware_limits (Lazy.force base_ctx)).native_fp16_arithmetic

(* {1 Which reference the backend's policy selects}

   A baseline claim of the form "the executed value is ONE OF the two host references" cannot fail
   in the direction that matters (Codex P2, round 3): if [C_syntax.scope_prec_of] or a backend's
   [accum_prec] regressed to storage precision, the baseline and every member referencing it would
   shift TOGETHER, every structure would still read as localized, and the only thing to change would
   be a label on stderr. So the reference is SELECTED here, from a statement of the policy that is
   independent of anything this run compiled or executed.

   The selected backend exposes the exact [C_syntax_config.accum_prec] resolution through
   [Context.codegen_capabilities]. The numerical members select their reference through that query,
   while one independent policy table below first checks the query itself. That table intentionally
   matches the canonical backend identity: deriving the oracle from the queried capability would be
   circular and would silently bless a regression in the capability and renderer together. *)

type residency =
  | Wider  (** The accumulator resides above storage: the whole-nest reference. *)
  | At_storage  (** It resides at storage width: the per-step-narrowed reference. *)

let same_prec a b = String.equal (Ops.prec_string a) (Ops.prec_string b)
let of_resolved prec resolved = if same_prec resolved prec then At_storage else Wider

let independent_residency prec =
  let policy = Numerics.get () in
  let narrow = policy.Numerics.narrow_compute_f32 in
  let wide_f16 = Numerics.fp16_accum_wide () in
  match Lazy.force selected_backend with
  | "cc" | "multidev_cc" -> (
      match prec with
      | Ops.Half_prec _ when wide_f16 -> Wider
      | Ops.Half_prec _
        when Numerics.equal_fp16_mode policy.fp16_arithmetic Numerics.Fp16_narrow
             && Lazy.force native_fp16 ->
          At_storage
      | _ when Ops.is_narrow_float prec && narrow -> Wider
      | _ -> At_storage)
  | "metal" -> (
      match prec with
      | Ops.Half_prec _ when wide_f16 -> Wider
      | Ops.Fp8_prec _ -> Wider
      | _ -> At_storage)
  | "cuda" -> (
      match prec with
      | Ops.Bfloat16_prec _ -> Wider
      | Ops.Half_prec _ when wide_f16 -> Wider
      | Ops.Fp8_prec _ when narrow -> Wider
      | _ -> At_storage)
  | "hip" -> (
      match prec with
      | Ops.Half_prec _ when wide_f16 -> Wider
      | Ops.Fp8_prec _ when narrow -> Wider
      | _ -> At_storage)
  | other -> failwith ("no independent accumulator policy recorded for backend " ^ other)

let expected_residency prec =
  of_resolved prec (codegen_capabilities.Ir.Backend_intf.accum_prec prec)

let () =
  p_all "the queried accumulator capability matches the independent backend policy table"
    [ Ops.single; Ops.bfloat16; Ops.half; Ops.fp8 ] ~f:(fun prec ->
      match (expected_residency prec, independent_residency prec) with
      | Wider, Wider | At_storage, At_storage -> true
      | Wider, At_storage | At_storage, Wider -> false)

let residency_name = function
  | Wider -> "wider than storage (whole-nest)"
  | At_storage -> "storage width (per-step)"

(* Which storage precisions the warp-shuffle rendering of a [Workgroup_reduce] loop accepts on a GPU
   backend (gh-ocannl-682): it stages the value at the accumulator RESIDENCY, so it takes f32 itself
   and every narrow precision the backend's policy resolves to f32 — bf16 on CUDA. A residency that
   stays narrow is refused by raising rather than by falling back to a serial loop, so those legs
   stay unevaluable off the C backends, where [warp_size = 0] serializes them all. *)
let shuffle_takes prec_name =
  String.equal prec_name "f32"
  ||
  match List.Assoc.find precs ~equal:String.equal prec_name with
  | Some prec -> ( match expected_residency prec with Wider -> true | _ -> false)
  | None -> false

(* {1 Program shapes}

   Six spellings of one reduction. The guarded ones admit an INTERIOR prefix of the reduction axis —
   the runtime extent is bound to {!guard_terms} of [cols] at launch, and the mask admits the same
   prefix — so they compute a smaller sum than the unguarded ones and are judged against a reference
   of their own. A guard true on every iteration would have made their claims unfailable, which is
   the whole of {!guard_terms}'s comment. Beyond the value, what the guards change is which peel
   decision codegen and the schedule mints reach, which is the point of having them. *)

type shape =
  | Plain  (** [for r: for k: out[r] += x[r,k]] — the recognizer's canonical nest. *)
  | Runtime_guard
      (** The gh-490 symbolic-extent shape: [If (k < s)] with [s] a STATIC symbol, a kernel
          parameter bound at launch. Its bound is outside every loop, so the peel must take it
          (gh-ocannl-693); the mints' agreement with the serial baseline under it is gh-ocannl-715.
          Bound to {!guard_terms} of [cols], so the predicate SELECTS. *)
  | Data_guard
      (** [If (mask[k] < 1)] — a guard that is not [pure_index_guard], so the peel refuses and the
          nest renders as per-step read-modify-writes. The mask admits the first {!guard_terms}
          iterations, for the same reason the runtime extent is interior. *)
  | Side_write
      (** The reduction level holds a SECOND statement (a write to another node), one of
          [peel_accum_nest]'s structural refusals. The sibling does not feed [out], so the value is
          the same and the form is not. *)
  | Virtual_acc
      (** The accumulator is a VIRTUAL node the virtualizer inlines at its read site: the
          [Local_scope] with [mint = Inlined_computation], the other producer of the scope form. *)
  | Mixed_guard
      (** The gh-ocannl-721 shape: [If (r + k < s)], a guard mixing the PEELED reduction index with
          the ENCLOSING output loop's, over a cell each of that loop's instances owns ([out[r]]).
          The mix is what used to stop the peel — [r] selects among enclosing lanes — and the
          separation of [r] by the cell is what now lets it through: every lane owning a distinct
          cell makes the hoisted load/store pair private and idempotent. Its rows admit 20, 19 and
          18 terms, so a regression that dropped the predicate, or substituted the wrong row into
          it, is a different number in every row. *)
  | Where_scope
      (** Virtualization's guarded-update spelling: an already-scoped nest whose update is
          [Set_local (id, Where (cond, acc + x, Get_local id))] — an expression-spelled guard whose
          else-arm carries the accumulator through. Post-optimize IR, so it reaches the backend
          through {!Ll_test.optimize_scoped} rather than through [LL.optimize]. Same interior
          runtime extent, so the else-arm is taken on real iterations rather than on none. *)
  | Rng_contrib
      (** [out[r] += x[r,k] * uniform1(key)]: a contribution mentioning an RNG conversion, which
          pins the accumulator to STORAGE precision (gh-ocannl-517: the conversion picks its result
          type and which random bits it consumes from the precision it renders at, so the whole
          update renders at storage precision and localizing it would change the draw). Every
          rendering must then narrow the accumulator on every step — the one body class where the
          serial form and the shuffle were found deciding differently (gh-ocannl-682), and the
          natural first member of the gh-ocannl-754 corpus. Built from a [Constant_bits] key rather
          than a uint4x32 node: the pin fires from the contribution's SHAPE at codegen, and a
          constant key needs no host representation. The factor is loop-invariant, which is the
          shape a broadcast scalar uniform virtualized into a reduction takes — and exactly the
          operand the vector renderer would splat across its lanes, rendering the draw at compute
          precision and accumulating wide, if it decided the width for itself. *)

(* How many terms a shape's reduction admits, PER ROW — which reference its value is judged against.
   Only {!Mixed_guard} varies with the row; the others admit the same prefix in each. *)
let terms_of_shape shape r =
  match shape with
  | Runtime_guard | Data_guard | Where_scope -> guard_terms
  | Mixed_guard -> guard_terms - r
  | Plain | Side_write | Virtual_acc | Rng_contrib -> cols

let shape_name = function
  | Plain -> "plain"
  | Runtime_guard -> "runtime-extent guard"
  | Mixed_guard -> "mixed lane guard"
  | Data_guard -> "data-dependent guard"
  | Side_write -> "sibling statement"
  | Virtual_acc -> "virtual accumulator"
  | Where_scope -> "Where-guarded scope"
  | Rng_contrib -> "RNG-bearing contribution"

(* The RNG conversion's key, the same constant [hardware_warp_shuffle]'s RNG leg draws from. *)
let rng_key = Int64.of_int 0x9E3779B9

type prog = {
  llc : LL.t;
  raw : LL.t option;  (** The unscoped twin {!Ll_test.optimize_scoped} needs, for [Where_scope]. *)
  r : Idx.symbol;
  k : Idx.symbol;
  out : Tn.t;
  materialized : Tn.t list;
  seed : (Tn.t * float array) list;
  bindings : Idx.unit_bindings;
  bind : Idx.lowered_bindings -> unit;
  verify : (string * Tn.t * float array) list;
      (** Other materialized nodes the shape writes, with what every cell must hold. Reading back
          only the accumulator is not enough (Codex P2, round 10): a schedule op rewrites the WHOLE
          level, so a regression that dropped the sibling stores of [Side_write], or substituted the
          wrong [k] into them, would leave the accumulator's value, its form classification and all
          its site counts intact while corrupting an observable result. *)
}

(* Ids are the test's own range, bumped per program so that nodes stay distinguishable in debug
   output and no two compiles share a tensor node. *)
let next_id = ref 964_000_000

(* A launch parameter standing for a symbolic DIMENSION rather than an index: its bound value is a
   size in [0, cols] INCLUSIVE, the standing [Row.get_sym_dim] gives a gh-490 extent symbol. Without
   [used_as_extent] the bind-time validation rejects the one binding these members want — the extent
   that covers the whole axis, which is what makes the guard vacuous and the value comparable to the
   unguarded baseline. *)
let extent_symbol () =
  let s, bindings = Idx.get_static_symbol ~static_range:cols Idx.Empty in
  s.Idx.used_as_extent <- true;
  (s, bindings)

let make ~(prec : Ops.prec) ~(shape : shape) () : prog =
  let node ?(dims = [| rows; cols |]) label =
    next_id := !next_id + 1;
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()
  in
  let out = node ~dims:[| rows |] "rfout" in
  let x = node "rfxs" in
  Ll_test.materialize out;
  Ll_test.materialize x;
  let r = Ll_test.sym () and k = Ll_test.sym () in
  let iprec = Ops.index_prec () in
  let ri = Idx.Iterator r and ki = Idx.Iterator k in
  let acc_cell = [| ri |] in
  let get_acc = LL.Get (out, acc_cell) in
  let get_x = LL.Get (x, [| ri; ki |]) in
  let update = LL.Binop (Ops.Add, (get_acc, prec), (get_x, prec)) in
  let plain_body = LL.Set { tn = out; idcs = acc_cell; llsc = update; debug = "" } in
  let seed = [ (out, out_seed); (x, x_values) ] in
  let base =
    {
      llc = LL.Noop;
      raw = None;
      r;
      k;
      out;
      materialized = [ out; x ];
      seed;
      bindings = Idx.Empty;
      bind = (fun _ -> ());
      verify = [];
    }
  in
  let nest ?(body = plain_body) () =
    LL.For_loop
      {
        index = r;
        from_ = 0;
        to_ = rows - 1;
        axis = LL.Serial;
        body = LL.For_loop { index = k; from_ = 0; to_ = cols - 1; axis = LL.Serial; body };
      }
  in
  match shape with
  | Plain -> { base with llc = nest () }
  | Runtime_guard ->
      let s, bindings = extent_symbol () in
      let cond =
        LL.Binop
          ( Ops.Cmplt,
            (LL.Embed_index ki, iprec),
            (LL.Embed_index (Idx.Iterator s.Idx.static_symbol), iprec) )
      in
      {
        base with
        llc = nest ~body:(LL.If { cond = (cond, iprec); body = plain_body }) ();
        bindings;
        bind = (fun lowered -> Idx.find_exn lowered s := guard_terms);
      }
  | Mixed_guard ->
      (* [r + k < s]. [s] is the same launch-bound extent the {!Runtime_guard} shape uses, so the
         bound is outside every loop and only [r] is an enclosing index — which is the whole of what
         the peel has to decide. *)
      let s, bindings = extent_symbol () in
      let cond =
        LL.Binop
          ( Ops.Cmplt,
            (LL.Embed_index (Idx.Affine { symbols = [ (1, r); (1, k) ]; offset = 0 }), iprec),
            (LL.Embed_index (Idx.Iterator s.Idx.static_symbol), iprec) )
      in
      {
        base with
        llc = nest ~body:(LL.If { cond = (cond, iprec); body = plain_body }) ();
        bindings;
        bind = (fun lowered -> Idx.find_exn lowered s := guard_terms);
      }
  | Data_guard ->
      let mask = node ~dims:[| cols |] "rfmask" in
      Ll_test.materialize mask;
      let cond = LL.Binop (Ops.Cmplt, (LL.Get (mask, [| ki |]), prec), (LL.Constant 1.0, prec)) in
      {
        base with
        llc = nest ~body:(LL.If { cond = (cond, prec); body = plain_body }) ();
        materialized = [ out; x; mask ];
        seed = (mask, Array.init cols ~f:(fun k -> if k < guard_terms then 0.0 else 1.0)) :: seed;
      }
  | Side_write ->
      (* Indexed by BOTH symbols, so the sibling is not loop-invariant: a hoistable one would leave
         the level holding a single statement again and the member would silently become a
         localizing one. *)
      let side = node "rfside" in
      Ll_test.materialize side;
      let sibling =
        LL.Set
          {
            tn = side;
            idcs = [| ri; ki |];
            llsc = LL.Binop (Ops.Add, (get_x, prec), (LL.Constant 1.0, prec));
            debug = "";
          }
      in
      {
        base with
        llc = nest ~body:(LL.Seq (plain_body, sibling)) ();
        materialized = [ out; x; side ];
        seed = (side, Array.create ~len:(rows * cols) Ll_test.sentinel) :: seed;
        verify =
          [
            ( "the sibling stores cover every cell with the value of their own iteration",
              side,
              Array.init (rows * cols) ~f:(fun n -> cell (n / cols) (n % cols) +. 1.0) );
          ];
      }
  | Virtual_acc ->
      let tmp = node ~dims:[| rows |] "rftmp" in
      Ll_test.virtualize tmp;
      let tmp_update =
        LL.Set
          {
            tn = tmp;
            idcs = acc_cell;
            llsc = LL.Binop (Ops.Add, (LL.Get (tmp, acc_cell), prec), (get_x, prec));
            debug = "";
          }
      in
      let accumulate =
        LL.For_loop
          {
            index = r;
            from_ = 0;
            to_ = rows - 1;
            axis = LL.Serial;
            body =
              LL.For_loop
                { index = k; from_ = 0; to_ = cols - 1; axis = LL.Serial; body = tmp_update };
          }
      in
      let copy_sym = Ll_test.sym () in
      let copy =
        LL.For_loop
          {
            index = copy_sym;
            from_ = 0;
            to_ = rows - 1;
            axis = LL.Serial;
            body =
              LL.Set
                {
                  tn = out;
                  idcs = [| Idx.Iterator copy_sym |];
                  llsc = LL.Get (tmp, [| Idx.Iterator copy_sym |]);
                  debug = "";
                };
          }
      in
      {
        base with
        llc = LL.Seq (LL.Zero_out tmp, LL.Seq (accumulate, copy));
        seed = [ (x, x_values) ];
      }
  | Rng_contrib ->
      let draw =
        LL.Unop (Ops.Uint4x32_to_prec_uniform1, (LL.Constant_bits rng_key, Ops.uint4x32))
      in
      let contrib = LL.Binop (Ops.Mul, (get_x, prec), (draw, prec)) in
      let body =
        LL.Set
          {
            tn = out;
            idcs = acc_cell;
            llsc = LL.Binop (Ops.Add, (get_acc, prec), (contrib, prec));
            debug = "";
          }
      in
      { base with llc = nest ~body () }
  | Where_scope ->
      let s, bindings = extent_symbol () in
      let cond =
        LL.Binop
          ( Ops.Cmplt,
            (LL.Embed_index ki, iprec),
            (LL.Embed_index (Idx.Iterator s.Idx.static_symbol), iprec) )
      in
      let id = LL.get_scope out in
      let body =
        LL.Seq
          ( LL.Set_local (id, get_acc),
            LL.For_loop
              {
                index = k;
                from_ = 0;
                to_ = cols - 1;
                axis = LL.Serial;
                body =
                  LL.Set_local
                    ( id,
                      LL.Ternop
                        ( Ops.Where,
                          (cond, iprec),
                          (LL.Binop (Ops.Add, (LL.Get_local id, prec), (get_x, prec)), prec),
                          (LL.Get_local id, prec) ) );
              } )
      in
      let scoped =
        LL.For_loop
          {
            index = r;
            from_ = 0;
            to_ = rows - 1;
            axis = LL.Serial;
            body =
              LL.Set
                {
                  tn = out;
                  idcs = acc_cell;
                  llsc =
                    (* [Schedule_minted], not [Inlined_computation], even though the UPDATE is
                       spelled the way [inline_computation] spells a guarded one: the scope is over
                       a MATERIALIZED node, which the virtualizer never produces (gh-ocannl-681) —
                       this is post-optimize IR handed to the backend the way the schedule mints
                       hand theirs. The mint records which side built the scope; the borrowed
                       spelling is what the member is about. *)
                    LL.Local_scope { id; body; orig_indices = acc_cell; mint = LL.Schedule_minted };
                  debug = "";
                };
          }
      in
      let raw =
        nest
          ~body:
            (LL.Set
               {
                 tn = out;
                 idcs = acc_cell;
                 llsc = LL.Ternop (Ops.Where, (cond, iprec), (update, prec), (get_acc, prec));
                 debug = "";
               })
          ()
      in
      {
        base with
        llc = scoped;
        raw = Some raw;
        bindings;
        bind = (fun lowered -> Idx.find_exn lowered s := guard_terms);
      }

(* {1 Reading the rendered form off the emitted kernel}

   Classification is textual, over the artifact {!Test_utils.Generated} guarantees this run emitted.
   It is deliberately phrased in terms of the STORED node rather than of any particular local's
   name: the localizing forms differ in where the accumulator lives (a codegen-minted scope, a
   schedule-minted one, a SIMD register grid, a warp's registers) but agree on what they do to the
   node — read it at most once, write it at most once, never both in one statement. The
   read-modify-write forms are exactly the ones with a statement doing both.

   Statements are split on [;], not on newlines: the pretty-printer breaks a long value expression
   across lines, so a line-based reading of "does this statement touch the node twice" answers no
   for precisely the wide read-modify-write it is meant to catch. *)

type form = Localized | Partials_combine | Rmw | Simd | Warp | Mma_fallback | Unrecognized

let form_name = function
  | Localized -> "localized scope"
  | Partials_combine -> "scoped block partials + combine"
  | Rmw -> "per-step read-modify-write"
  | Simd -> "SIMD accumulator grid"
  | Warp -> "warp-shuffle tree"
  | Mma_fallback -> "Tile_mma scalar fallback"
  | Unrecognized -> "unrecognized"

(* {1 The peel DECISION a member claims}

   The form above is what was RENDERED. It does not determine what was DECIDED, and for the
   localizing peel the gap is the one gh-ocannl-733 is about: a nest whose accumulated cell is free
   of the enclosing index is peeled at the ENCLOSING level, taking both levels under a confined
   guard, while one whose cell mentions it peels the reduction level alone under a lane-private
   guard admitted because the cell separates the lanes (gh-ocannl-721) — and both emit one localized
   scope with one closing store. Every claim of [mixed-guard] would be green over the first kernel,
   in which the escape it exists to exercise never ran.

   So each member also declares what codegen's peel decided, read off the routine's own peel census
   ({!Context.routine.peel}, [C_syntax.peel_census]) rather than off the emitted text. The
   declaration separates three provenances the form claim cannot: the peel produced this scope, the
   scope was already in the IR when codegen met it, and nothing localized at all. *)

type peel_claim =
  | Peeled of Cs.peel_verdict
      (** Codegen's peel produced this member's accumulator scope, and every site that localized
          decided exactly this: that many levels, those guard verdicts. *)
  | Minted_upstream
      (** The scope this member renders was in the IR before codegen: a [Schedule] mint
          ([Unroll ~materialize], [Partition], [Pad]) or virtualization's inline. Codegen's peel was
          consulted and localized nothing — which is a different fact about the kernel than
          "localized", and the one this member is really about. *)
  | No_localization
      (** Nothing localized: the form is a decline (the per-step read-modify-write members) or a
          rendering holding the accumulator elsewhere (the SIMD grid, the shuffle tree). The census
          must still hold a site — "nothing localized" is worth nothing where the peel was never
          consulted. *)
  | Never_a_site
      (** The census holds NO site: at no level does this member accumulate into the cell it writes,
          so localization was never a question. Virtualization's inlined accumulator is the case —
          it owns its cell and opens from a zero-init rather than from a load of the node, which is
          why nothing here is a self-recurrence through memory. Distinct from {!Minted_upstream},
          where the peel was consulted at a real reduction site and refused. *)
  | Widened_elsewhere of Cs.peel_verdict
      (** The shared width decision (gh-ocannl-754) reached the base and said WIDE, deciding exactly
          this, and the rendering holding the accumulator at that width is not a scope but the SIMD
          grid or the shuffle tree: the census carries {!Cs.Peel_ceded} with this verdict, and no
          site localized. What separates it from {!No_localization} is that the width was decided by
          the same call the serial form consults, not by a recognizer of the rendering's own — which
          is the whole of what gh-ocannl-754 changed. *)
  | Pinned_to_storage
      (** The shared width decision reached the base and PINNED it to storage width
          ({!Cs.Skip_accum_pinned}: the RNG carve-out), so no rendering widened: no site localized,
          none ceded, and at least one recorded the pin. The claim of every RNG-bearing member,
          whatever retype it is under. *)

let guard_verdict_name = function
  | LL.Guard_confined -> "a confined guard"
  | LL.Guard_lane_private -> "a lane-private guard"
  | LL.Guard_lane_private_unresolved ->
      (* Unreachable in a member's declaration: separation is decided at the base, so only a
         REFUSING report leaves a lane-private guard unresolved, and no site that localized carries
         one. Named rather than wildcarded, so a verdict added to the peel must be given a word
         here. *)
      "a lane-private guard the peel never settled"

let peel_claim_name = function
  | Peeled { Cs.levels; guards } ->
      Printf.sprintf "codegen's peel takes %s%s"
        (match levels with 1 -> "one level" | n -> Printf.sprintf "%d levels" n)
        (match guards with
        | [] -> ", unguarded"
        | gs -> ", through " ^ String.concat ~sep:" then " (List.map gs ~f:guard_verdict_name))
  | Minted_upstream -> "the scope is minted upstream; codegen's peel localizes nothing"
  | No_localization -> "codegen's peel localizes nothing"
  | Never_a_site -> "no cell recurrence at any level: the peel is never consulted"
  | Widened_elsewhere { Cs.levels; guards } ->
      Printf.sprintf "the shared width decision is wide (%s%s) and the grid or tree holds it"
        (match levels with 1 -> "one level" | n -> Printf.sprintf "%d levels" n)
        (match guards with
        | [] -> ", unguarded"
        | gs -> ", through " ^ String.concat ~sep:" then " (List.map gs ~f:guard_verdict_name))
  | Pinned_to_storage -> "the shared width decision pins the accumulator to storage: nothing widens"

(* Checking a member's peel claim against its routine's census, shared by the table members and the
   [Tile_mma] fallback leg so the two cannot read the census differently. *)
let check_peel ~name ~claim (peel : Cs.peel_summary) (want : peel_claim) =
  let localized_sites =
    List.filter_map peel.Cs.sites ~f:(fun (_, site) ->
        match site with Cs.Peel_localized v -> Some v | _ -> None)
  in
  let ceded_sites =
    List.filter_map peel.Cs.sites ~f:(fun (_, site) ->
        match site with Cs.Peel_ceded v -> Some v | _ -> None)
  in
  let verdicts vs =
    String.concat ~sep:", " (List.map vs ~f:(fun v -> Sexp.to_string (Cs.sexp_of_peel_verdict v)))
  in
  match want with
  | Peeled want ->
      if not (List.for_all localized_sites ~f:(Cs.equal_peel_verdict want)) then
        Stdio.eprintf "  %s: localized sites decided [%s], declared [%s]\n" name
          (verdicts localized_sites) (verdicts [ want ]);
      Verdict.p_all claim localized_sites ~f:(Cs.equal_peel_verdict want)
  | Minted_upstream | No_localization ->
      (* Over the whole census, so an EMPTY one fails too: "nothing localized" says nothing where
         the peel was never consulted, and every composition here reaches at least one
         self-recurrent serial site on both backend families. The member that does NOT is declared
         {!Never_a_site} and checked the other way. *)
      if not (List.is_empty localized_sites) then
        Stdio.eprintf "  %s: declared no localization, got [%s]\n" name (verdicts localized_sites);
      Verdict.p_empty claim ~over:peel.Cs.sites localized_sites
  | Never_a_site ->
      if not (List.is_empty peel.Cs.sites) then
        Stdio.eprintf "  %s: declared no peel site, got %s\n" name (Cs.peel_summary_string peel);
      p claim (List.is_empty peel.Cs.sites)
  | Widened_elsewhere want ->
      (* Non-emptiness is part of the claim: a rendering that stopped consulting the shared decision
         would leave no ceded site, and "nothing localized" alone would still hold. *)
      let ok =
        List.length localized_sites = 0
        && (not (List.is_empty ceded_sites))
        && List.for_all ceded_sites ~f:(Cs.equal_peel_verdict want)
      in
      if not ok then
        Stdio.eprintf "  %s: declared ceded [%s]; localized [%s], ceded [%s]\n" name
          (verdicts [ want ]) (verdicts localized_sites) (verdicts ceded_sites);
      p claim ok
  | Pinned_to_storage ->
      let pinned =
        List.count peel.Cs.sites ~f:(fun (_, site) ->
            match site with Cs.Peel_not_attempted Cs.Skip_accum_pinned -> true | _ -> false)
      in
      let ok = pinned > 0 && List.length localized_sites = 0 && List.length ceded_sites = 0 in
      if not ok then
        Stdio.eprintf "  %s: declared pinned to storage, got %s\n" name
          (Cs.peel_summary_string peel);
      p claim ok

let is_ident_char c = Char.is_alphanum c || Char.equal c '_'

(* The code name codegen derived for a node is not predictable from its label (it goes through a
   blacklist and a dot-stripping pass), so it is read off the source: an identifier CONTAINING the
   label and followed by a subscript. Several can match — [Split_reduce] mints a [partials_<parent>]
   node, whose name contains the parent's — so the SHORTEST is taken, which is the label itself
   wherever codegen kept it. *)
let ident_for src ~label =
  let n = String.length src in
  String.substr_index_all src ~may_overlap:false ~pattern:label
  |> List.filter_map ~f:(fun at ->
      let b = ref at and e = ref (at + String.length label) in
      while !b > 0 && is_ident_char src.[!b - 1] do
        Int.decr b
      done;
      while !e < n && is_ident_char src.[!e] do
        Int.incr e
      done;
      if !e < n && Char.equal src.[!e] '[' then Some (String.sub src ~pos:!b ~len:(!e - !b))
      else None)
  |> List.min_elt ~compare:(fun a b -> Int.compare (String.length a) (String.length b))

let statements src =
  List.map (String.split src ~on:';') ~f:(fun st ->
      String.concat ~sep:" " (String.split_on_chars st ~on:[ '\n'; '\t' ]))

(* Whether [tok] is a scope-local name: [v<digits>_<node ident>], {!C_syntax.pp_scope_id}'s
   spelling. Which node it belongs to is not constrained — the accumulator of a [Virtual_acc] member
   is a different node from the one stored. *)
let is_scope_local tok =
  String.length tok > 2
  && Char.equal tok.[0] 'v'
  &&
  match String.index tok '_' with
  | None | Some 1 -> false
  | Some us -> String.for_all (String.sub tok ~pos:1 ~len:(us - 1)) ~f:Char.is_digit

type reading = {
  rmw_statements : int;  (** Statements reading AND writing the stored node: the RMW fingerprint. *)
  stores_from_local : int;
      (** [node[...] = <conversion>(v<n>_...)]: the once-per-cell narrowing store that ends a
          localized nest. The conversion is why the right-hand side is searched for a scope-local
          TOKEN rather than compared to one: at bf16 the store reads
          [rfout[i] = single_to_bfloat16(v5_rfout)], and a test looking for a bare local would
          report every narrow-storage kernel as unlocalized — which is the reading the whole
          accumulator-width policy is about. *)
  foreign_local_stores : int;
      (** The same shape into some OTHER node: [Split_reduce]'s block partials, each of which
          localizes while the combine that folds them into the target does not. *)
  node_accesses : int;
  foreign_accesses : int;
      (** Subscripts of the nodes a foreign closing store writes — [Split_reduce]'s partials. The
          COUNT of foreign stores is not enough (Codex P2, round 13): a regression could keep one
          localized closing store into the partials while routing another segment through direct
          read-modify-writes of that same node, leaving one foreign store, one target RMW and two
          target accesses intact, and the f32-only execution exact throughout. *)
  has_simd : bool;
  has_warp : bool;
}

(* Whether [st] mentions a scope-local name anywhere. *)
let mentions_scope_local st =
  let n = String.length st in
  let rec go i =
    if i >= n then false
    else if is_ident_char st.[i] then begin
      let j = ref i in
      while !j < n && is_ident_char st.[!j] do
        Int.incr j
      done;
      if is_scope_local (String.sub st ~pos:i ~len:(!j - i)) then true else go !j
    end
    else go (i + 1)
  in
  go 0

(* Erase the expression-level [volatile] pointer casts a backend may put on the READ side of a
   device read-modify-write: a read of [rfout] parenthesized and prefixed with a cast to [device
   volatile float *] becomes a plain [rfout] again. Metal is the backend that spells accumulating
   reads this way ({!Ir.C_syntax.pp_device_read_ptr}, guarded by [volatile_serial_accumulation]); cc
   and the other CPU renderings do not, so without this the same kernel reads as two different forms
   on two backends.

   Erasing rather than teaching {!subscripts} a second pattern, and erasing here rather than at each
   count, because the cast is a SPELLING of the access and every reading has to agree about that:
   [subscripts], [assigns_node] and [foreign_accesses] each decide from the same text, and a node
   access counted by one and missed by another is a reading of nothing in particular.

   The failure direction is what makes this load-bearing rather than cosmetic. An uncounted read
   costs a statement its second subscript, so a genuine per-step read-modify-write reads as zero RMW
   statements — which is [Unrecognized] for a member claiming {!Rmw} (a loud failure, and how this
   was found), but is [Localized] for any member that also closes a cell from a scope local. That
   second case is a kernel still hammering the device cell every step, classified as the localized
   form this table exists to certify. *)
let strip_volatile_casts src =
  let n = String.length src in
  let buf = Buffer.create n in
  let i = ref 0 in
  while !i < n do
    (* Two opening parens, a cast to a volatile pointer that itself contains no nested parens, then
       a bare identifier and the outer close: the shape the emitter builds around a device read.
       Anything that does not match all of that is copied through. *)
    let erased =
      if !i + 1 < n && Char.equal src.[!i] '(' && Char.equal src.[!i + 1] '(' then
        match String.substr_index src ~pos:(!i + 2) ~pattern:"*)" with
        | None -> None
        | Some close ->
            let inner = String.sub src ~pos:(!i + 2) ~len:(close - (!i + 2)) in
            if
              String.is_substring inner ~substring:"volatile "
              && (not (String.contains inner '('))
              && not (String.contains inner ')')
            then begin
              let j = close + 2 in
              let k = ref j in
              while !k < n && is_ident_char src.[!k] do
                Int.incr k
              done;
              if !k > j && !k < n && Char.equal src.[!k] ')' then
                Some (String.sub src ~pos:j ~len:(!k - j), !k + 1)
              else None
            end
            else None
      else None
    in
    match erased with
    | Some (ident, next) ->
        Buffer.add_string buf ident;
        i := next
    | None ->
        Buffer.add_char buf src.[!i];
        Int.incr i
  done;
  Buffer.contents buf

(* Occurrences of [ident] as a WHOLE identifier followed by a subscript. Substring counting is wrong
   here: [partials_rfout[..]] contains [rfout[], so a reduction whose partials node is named after
   its target would read as touching the target twice. *)
let subscripts st ~ident =
  List.count
    (String.substr_index_all st ~may_overlap:false ~pattern:(ident ^ "["))
    ~f:(fun at -> at = 0 || not (is_ident_char st.[at - 1]))

(* The statements that assign a scope local: [v<digits>_<ident> = ...]. What {!member}'s [extra]
   substrings are searched in — a builtin helper's own locals are not named this way, so its
   ternaries and casts cannot satisfy a claim about the reduction's update. *)
let scope_update_statements src =
  List.filter (statements src) ~f:(fun st ->
      match String.substr_index st ~pattern:" = " with
      | None -> false
      | Some at ->
          let lhs =
            match List.rev (String.split_on_chars (String.prefix st at) ~on:[ ' '; '('; ')' ]) with
            | tok :: _ -> tok
            | [] -> ""
          in
          is_scope_local lhs)

(* The array a statement assigns to: the identifier ending at the first subscript before [ = ]. *)
let lhs_array st =
  match String.substr_index st ~pattern:"] = " with
  | None -> None
  | Some at -> (
      let prefix = String.prefix st at in
      match String.index prefix '[' with
      | None -> None
      | Some br ->
          let b = ref br in
          while !b > 0 && is_ident_char prefix.[!b - 1] do
            Int.decr b
          done;
          if !b < br then Some (String.sub prefix ~pos:!b ~len:(br - !b)) else None)

let read_form src ~label =
  let src = strip_volatile_casts src in
  match ident_for src ~label with
  | None -> None
  | Some ident ->
      let sts = statements src in
      let accesses st = subscripts st ~ident in
      (* A store of a scope local: the assignment's left-hand side is an array subscript, and its
         right-hand side mentions a local. *)
      let stores_a_local st =
        match String.substr_index st ~pattern:"] = " with
        | None -> false
        | Some at -> mentions_scope_local (String.drop_prefix st (at + 4))
      in
      let assigns_node st =
        match
          (String.substr_index st ~pattern:"] = ", String.substr_index st ~pattern:(ident ^ "["))
        with
        | Some at, Some lhs -> lhs < at && (lhs = 0 || not (is_ident_char st.[lhs - 1]))
        | _ -> false
      in
      let foreign_idents =
        List.filter_map sts ~f:(fun st ->
            if accesses st = 0 && stores_a_local st then lhs_array st else None)
        |> List.dedup_and_sort ~compare:String.compare
      in
      Some
        {
          rmw_statements = List.count sts ~f:(fun st -> accesses st >= 2);
          stores_from_local =
            List.count sts ~f:(fun st -> accesses st = 1 && assigns_node st && stores_a_local st);
          foreign_local_stores = List.count sts ~f:(fun st -> accesses st = 0 && stores_a_local st);
          node_accesses = List.sum (module Int) sts ~f:accesses;
          foreign_accesses =
            List.sum
              (module Int)
              foreign_idents
              ~f:(fun ident -> List.sum (module Int) sts ~f:(subscripts ~ident));
          has_simd = String.is_substring src ~substring:"Vectorized reduction rendering";
          has_warp = String.is_substring src ~substring:"ocannl_shfl_xor";
        }

(* What the reading says the kernel rendered. The order matters, and not merely for tidiness: a SIMD
   grid's epilogue is TEXTUALLY a read-modify-write ([out[i] = out[i] + vred_total]) — one per nest
   rather than one per step, which is the entire difference — so testing the RMW fingerprint first
   would classify the vector rendering as the form it exists to avoid. The more specific form wins,
   and the per-nest count is checked separately against the member's declared sites. *)
let form_of reading =
  if reading.has_warp then Warp
  else if reading.has_simd then Simd
  else if reading.rmw_statements = 0 && reading.stores_from_local > 0 then Localized
  else if
    reading.rmw_statements > 0 && reading.stores_from_local = 0 && reading.foreign_local_stores > 0
  then Partials_combine
  else if reading.rmw_statements > 0 && reading.stores_from_local = 0 then Rmw
  else
    (* NOT a catch-all (Codex P2, round 3). A reading with neither fingerprint — a declining
       renderer that moved to an unrecognized accumulator spelling, or to a once-per-nest direct
       store — used to land here as [Rmw], and at f32 it can compute the per-step reference's value
       exactly, so form and value would both have passed over a kernel containing no
       read-modify-write at all. An unrecognized reading is a failure to classify, and matches no
       member's claim. *)
    Unrecognized

(* {1 Executing a member} *)

let execute ~name ~(prog : prog) ~(sched : Sched.schedule) =
  (* The launch parameters have to reach BOTH halves: [LL.optimize]'s walk asserts that every symbol
     it meets is in scope (a runtime-extent guard mentions one that no loop binds), and
     [Sched.apply] folds guards against their declared ranges. *)
  let static_indices = Idx.bound_symbols prog.bindings in
  let o =
    match prog.raw with
    | None -> Ll_test.optimize ~materialized:prog.materialized ~static_indices ~name prog.llc
    | Some raw ->
        Ll_test.optimize_scoped ~materialized:prog.materialized ~static_indices ~name ~raw prog.llc
  in
  (* The transform's INPUT and OUTPUT are both kept. A coverage entry that only checks its member
     exists says nothing about whether that member still exercises the op it is credited with (Codex
     P2, round 10): were [Unroll]'s annotated rewrite to regress to leaving the loop [Serial], the
     kernel would keep the same localized classification, the same closing store and the same value,
     and both the member and its coverage entry would stay green. Comparing the two is what binds a
     cited op to an observable effect. *)
  let before = ref None and after = ref None and per_op = ref [] in
  let ctx, routine =
    Context.compile ~name ~prelowered:o
      ~lowered_transform:(fun opt ->
        before := Some opt.LL.llc;
        (* EVERY op, not just the schedule as a whole (Codex P2, round 11). Comparing only the two
           ends credits a composition to all of its ops: in [split-then-swap] the [Split] alone
           makes the whole-schedule comparison true, so a [Swap] that became a no-op would leave the
           member's form, value and site checks passing and the sole [Swap] coverage entry green
           over an op that never ran. This is a PROBE beside the real transform, not a replacement
           for it -- [Sched.apply] simplifies and re-runs CSE at the end of the fold, so applying
           the ops one at a time would not produce the same IR, and only the whole-schedule result
           is compiled. *)
        (* Only for a COMPOSITION of loop rewrites: with one op the whole-schedule comparison
           already is the per-op one, and a node-minting op cannot be applied twice. *)
        (if List.length sched > 1 && List.for_all sched ~f:is_loop_rewrite then
           per_op :=
             let rec step acc state = function
               | [] -> List.rev acc
               | op :: rest ->
                   let next = Sched.apply ~static_indices [ op ] state in
                   let moved =
                     not (Sexp.equal (LL.sexp_of_t state.LL.llc) (LL.sexp_of_t next.LL.llc))
                   in
                   step (moved :: acc) next rest
             in
             step [] opt sched);
        let result = Sched.apply ~static_indices sched opt in
        after := Some result.LL.llc;
        [ result ])
      (Lazy.force base_ctx) Ir.Assignments.empty_comp prog.bindings
  in
  prog.bind routine.Context.bindings;
  let ctx = List.fold prog.seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) in
  let ctx = Context.run ctx routine in
  let read tn = Context.get_values ctx tn in
  ( read prog.out,
    List.map prog.verify ~f:(fun (c, tn, want) -> (c, read tn, want)),
    (!before, !after, !per_op),
    (* What the reduction peel DECIDED while emitting this kernel (gh-ocannl-733), as against what
       the classifier below reads off the emitted text. *)
    routine.Context.peel )

(* The axis kinds a statement's loops carry. What binds a member to the constructor its coverage
   entry credits it with: an op that names an axis kind must leave that kind in the IR it
   produced. *)
let rec axis_kinds (llc : LL.t) : LL.axis_type list =
  match llc with
  | LL.For_loop { axis; body; _ } -> axis :: axis_kinds body
  | LL.Seq (a, b) -> axis_kinds a @ axis_kinds b
  | LL.If { body; _ } -> axis_kinds body
  | LL.Set { llsc; _ } -> scalar_axis_kinds llsc
  | LL.Set_local (_, llsc) -> scalar_axis_kinds llsc
  | _ -> []

and scalar_axis_kinds (llsc : LL.scalar_t) : LL.axis_type list =
  match llsc with
  | LL.Local_scope { body; _ } -> axis_kinds body
  | LL.Ternop (_, (a, _), (b, _), (c, _)) ->
      scalar_axis_kinds a @ scalar_axis_kinds b @ scalar_axis_kinds c
  | LL.Binop (_, (a, _), (b, _)) -> scalar_axis_kinds a @ scalar_axis_kinds b
  | LL.Unop (_, (a, _)) -> scalar_axis_kinds a
  | _ -> []

(* {1 The member table}

   Each member is one point of the product the issue asks to sweep: a schedule composition over one
   of the {!shape}s, the form it claims to reach, and the reference its value must match. *)

(* Every value claim is BITWISE, the reassociating compositions included. A swapped split, a
   block-partial reduction and a shuffle tree all change the order of the additions, so they would
   normally earn a tolerance — but each of them is declared over f32 only (or, for the shuffle,
   available at f32 only), and there these operands' partial sums are exact, so every association
   gives the same float. That is not an assumption: the "at f32 the whole-nest and per-step
   references coincide" claim asserts exactly the exactness the argument rests on. A tolerance would
   have hidden a real narrowing at a seam behind an allowance for a reassociation that cannot cost
   anything here. *)

type reference =
  | Baseline  (** The serial rendering of the plain nest, executed at this precision: 32 terms. *)
  | Guarded_baseline
      (** The serial rendering of the runtime-extent-GUARDED nest, executed at this precision:
          {!guard_terms} terms. A schedule mint that dropped or broadened the predicate while
          rewriting the nest computes the unguarded sum instead, which is a different number — that
          is what makes the guarded members' value claims able to fail (Codex P2, round 2). *)
  | Per_step
      (** The host's per-step-narrowed reference over the member's own term count: what a
          read-modify-write form computes, opening from the seeded cell. *)
  | Mixed_baseline
      (** The serial rendering of the MIXED-guarded nest, executed at this precision: 20, 19 and 18
          terms by row. Its own agreement with the policy's host reference is asserted beside the
          other baselines, and that is the claim gh-ocannl-721 moves — before the localization the
          run narrowed at every step where the policy asks for a wider accumulator. *)
  | Owned_cell
      (** The host reference the POLICY selects, over the full axis, starting from ZERO: the virtual
          accumulator owns its cell and initializes it from a [Zero_out] rather than loading [out],
          so the incoming contents are not part of what it computes. *)
  | Rng_baseline
      (** The serial rendering of the RNG-bearing nest, executed at this precision: the
          storage-pinned per-step form. No host model — the draw is the kernel's to make — so the
          RNG members claim PARITY with it, and the control beside the baselines proves the
          RNG-scaled operands discriminate accumulator width. *)

type member = {
  slug : string;  (** Short name; also the routine name's stem, so artifacts are per member. *)
  what : string;  (** What the composition is, in words. *)
  shape : shape;
  sched : prog -> Sched.schedule;
  expect : form;  (** The form this backend must reach. *)
  claimed : string;
      (** How the form is NAMED in the printed table. Backend-uniform, unlike [expect]: the
          [.expected] golden is one file for every backend, so a member whose form is
          backend-dependent names both readings here. *)
  reference : reference;
  store_sites : int;
      (** How many CLOSING STORES of the accumulated node the localized form may emit: one per
          textual copy of the surviving output loop's body, which is one unless the composition
          duplicated that loop ([Unroll ~materialize] or [Partition] of the OUTPUT axis). Pinned
          exactly, not as "at least one" (Codex P2, round 1): a regression that gave each
          [Partition] segment or each unrolled copy its OWN scope and closing store would still read
          as localized under a positive-count test, and on a backend whose accumulator already
          resides at storage width the extra store/reload seams need not change the executed value
          either — a false green in both claims at once. The node's total subscript count is bounded
          by [2 * store_sites] for the same reason: each site may open the cell and close it, and
          nothing else may touch it. *)
  foreign_sites : int;
      (** For a member claiming {!Partials_combine}: how many closing stores into a node OTHER than
          the target the composition emits — one per textual copy of the block loop's body, which
          [Split_reduce] leaves at one. Pinned for the same reason the other two counts are (Codex
          P2, round 7): the form alone says only that SOME partial localized and SOME statement
          read-modify-writes the target, so partials closing through several scopes, or a combine
          expanded into several textual sites, would still read as this form — and this leg's f32
          arithmetic is exactly representable, so the extra seams or the reassociation need not move
          the value either. *)
  foreign_accesses : int;
      (** For {!Partials_combine}: how many subscripts the foreign node carries in total — the
          partials' own zero-init, scope-open read and closing store, plus one read per block in the
          combine. *)
  rmw_sites : int;
      (** For a member claiming {!Rmw}: how many read-modify-write statements the composition emits,
          textually — one per repetition where the level is [Unrolled], one where it is a serial
          loop. Pinned exactly, because the COUNT is what distinguishes the two declining forms
          (Codex P2, round 5): if the [Unrolled] renderer regressed to [serial_loop], or the axis
          annotation were dropped, the source would still hold one read-modify-write, [form_of]
          would still answer [Rmw], and execution would still equal the per-step reference — serial
          and unrolled loops visit the same terms — so a member advertised as "one per copy" would
          be green without the Unrolled fallback ever running. *)
  expect_axis : LL.axis_type option;
      (** The axis kind this member's schedule must leave in the IR it produced. Checked against the
          post-schedule loop nest, so a retype or an unroll that silently became a no-op fails here
          rather than passing on a form claim its own regression preserves. [None] where the op
          changes structure without naming a kind — the "the schedule changed the IR" claim covers
          those. *)
  extra : string list;
      (** Substrings that must appear in a statement ASSIGNING A SCOPE LOCAL — the scope's own
          updates — ANDed into the form claim. For the one distinguishing feature the node-access
          classifier cannot see: which spelling the update took inside the scope.

          Scoped to those statements rather than searched for in the whole artifact (Codex P2, round
          2): at f16 the kernel pulls in [half_to_float_emulated], whose body is full of unrelated
          ternaries, so [" ? "] anywhere in the source is true whatever the update renders as. *)
  peel : peel_claim;
      (** What codegen's reduction peel must have DECIDED while emitting this member's kernel,
          checked against the routine's peel census. Pinned in full — levels and guard verdicts —
          because a decision that swallowed one more level, or admitted a guard on the other ground,
          renders the same localized scope this member's form claim accepts (gh-ocannl-733). *)
  peel_claimed : string;
      (** How that decision is NAMED in the printed table. Backend-uniform, like {!claimed}: the two
          members whose rendering arm is backend-dependent decide differently on a GPU than on the C
          backends, and the golden is one file for both. *)
  precisions : string list;
  available : string -> bool;
      (** Whether this backend can evaluate the member at the given storage precision. Per
          PRECISION, not merely per backend: the warp-shuffle rendering exists only for single- and
          double-precision accumulators, so the same member is evaluable at f32 and vacuous at bf16
          on the very same GPU. Legs it excludes stay in the printed table and report
          {!Verdict.skipped}, so the golden is backend-uniform and [grep SKIPPED] enumerates what
          this hardware did not check. *)
}

let no_ops _ = []

(* The [Vectorized] arm has a DEFINED exit on every backend, and skipping the GPUs hid it (Codex P2,
   round 10): [try_vectorize_reduce] declines the GPU backends' [`Packed_struct] style, and the
   renderer then routes an accumulating body through [try_localize_serial_reduce] rather than
   through the pragma'd loop. That fallback is as much a member of this set as the SIMD grid, so the
   expectation is backend-dependent instead of the leg being marked unavailable. *)
let simd_or_localized = if on_cpu then Simd else Localized
let simd_claimed = "SIMD accumulator grid, or the localized scope where no vector rendering exists"

(* And the DECISION behind that exit: where the vector rendering takes the level the accumulator
   lives in its grid and codegen's peel localizes nothing, while where it declines the peel is what
   produced the scope (gh-ocannl-733). *)
let simd_or_localized_peel =
  if on_cpu then Widened_elsewhere { Cs.levels = 1; guards = [] }
  else Peeled { Cs.levels = 1; guards = [] }

let simd_peel_claimed =
  "the shared width decision is wide and the SIMD grid holds it, or codegen's peel takes the \
   reduction level where no vector rendering exists"

let member ?(shape = Plain) ?(sched = no_ops) ?(expect = Localized) ?claimed ?(reference = Baseline)
    ?(store_sites = 1) ?(rmw_sites = 1) ?(foreign_sites = 0) ?(foreign_accesses = 0) ?expect_axis
    ?(extra = []) ?(peel = Peeled { Cs.levels = 1; guards = [] }) ?peel_claimed
    ?(precisions = [ "f32"; "bf16"; "f16" ]) ?(available = fun _ -> true) slug what =
  let claimed = Option.value claimed ~default:(form_name expect) in
  let peel_claimed = Option.value peel_claimed ~default:(peel_claim_name peel) in
  {
    slug;
    what;
    shape;
    sched;
    expect;
    claimed;
    reference;
    store_sites;
    rmw_sites;
    foreign_sites;
    foreign_accesses;
    expect_axis;
    extra;
    peel;
    peel_claimed;
    precisions;
    available;
  }

let members =
  [
    (* --- the baseline itself: gh-ocannl-693's identity-precision localization, which before it
       left every f32 reduction doing one global read-modify-write per step. --- *)
    member "serial" "no schedule ops (the reference rendering)";
    (* --- the two [Unroll] representations autotune proposes over small reduction loops --- *)
    member "unroll-annot" "Unroll (annotated: codegen repeats the body)" ~expect_axis:LL.Unrolled
      ~sched:(fun g -> [ Sched.Unroll { axis = g.k; materialize = false } ]);
    (* The mint's own scope reaches codegen already built, so the peel localizes NOTHING here — it
       hoists a schedule-minted scope through the levels above it, or, as here, meets one that
       already spans the whole reduction. The form is the same localized scope [serial] renders
       (gh-ocannl-733). *)
    member "unroll-mat" "Unroll ~materialize (the schedule mints the scope)" ~peel:Minted_upstream
      ~sched:(fun g -> [ Sched.Unroll { axis = g.k; materialize = true } ]);
    (* The output loop is gone, so each of its [rows] copies closes its own cell -- which is not a
       seam but the whole of that cell's reduction. *)
    member "unroll-outer-mat" "Unroll ~materialize of the OUTPUT axis" ~store_sites:rows
      ~sched:(fun g -> [ Sched.Unroll { axis = g.r; materialize = true } ]);
    (* --- [Partition]: an index-set specialization of one reduction, so its segment seams must not
       become narrowing points — one scope spans every segment. --- *)
    member "partition" "Partition of the reduction axis into three segments" ~peel:Minted_upstream
      ~sched:(fun g ->
        let pt, _ = Sched.partition ~axis:g.k ~breakpoints:[ 4; 12 ] in
        [ pt ]);
    member "partition-then-unroll" "Partition, then Unroll one segment (the seam stays addressable)"
      ~peel:Minted_upstream ~sched:(fun g ->
        let pt, segs = Sched.partition ~axis:g.k ~breakpoints:[ 4; 12 ] in
        [ pt; Sched.Unroll { axis = List.hd_exn segs; materialize = false } ]);
    (* Two segment loops over the output axis, each carrying a whole reduction: two sites. *)
    member "partition-outer" "Partition of the OUTPUT axis (the reduction is inside a segment)"
      ~store_sites:2 ~sched:(fun g ->
        let pt, _ = Sched.partition ~axis:g.r ~breakpoints:[ 1 ] in
        [ pt ]);
    (* --- [Split], with and without a reordering --- *)
    member "split-then-unroll" "Split the reduction axis, then Unroll ~materialize the inner half"
      ~sched:(fun g ->
        let sp, _outer, inner = Sched.split ~axis:g.k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
        [ sp; Sched.Unroll { axis = inner; materialize = true } ]);
    (* Two levels into the scope: after the split the reduction is a two-level nest, and the peel
       takes the inner half along with the outer one it is rendering. *)
    member "split-then-swap" "Split the reduction axis, then Swap the halves" ~precisions:[ "f32" ]
      ~peel:(Peeled { Cs.levels = 2; guards = [] })
      ~sched:(fun g ->
        let sp, outer, inner = Sched.split ~axis:g.k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
        [ sp; Sched.Swap { outer; inner } ]);
    (* --- [Pad]: guarded copies, whose constant-bounded guard the mint must peel into the scope
       --- *)
    member "pad-then-unroll-mat" "Pad the reduction axis, then Unroll ~materialize (guarded copies)"
      ~peel:Minted_upstream ~sched:(fun g ->
        [
          Sched.Pad { axis = g.k; to_multiple_of = 6 };
          Sched.Unroll { axis = g.k; materialize = true };
        ]);
    (* --- [Split_reduce]: block partials plus a combine nest. The reduction scopes do NOT coincide
       (each partial narrows at its own store), so this member claims f32 only, where the sums are
       exact. --- *)
    member "split-reduce"
      "Split_reduce into four block partials plus a combine nest"
      (* Seven subscripts of the partials node, and each one is accounted for: the whole-node
         zero-init, the scope-opening read and the closing store of the per-block accumulation (the
         partials are themselves localized), then one read per block in the combine. Counted off the
         emitted kernel rather than predicted — my arithmetic said five and the init and the open
         are the two it forgot. *)
      ~expect:Partials_combine ~foreign_sites:1 ~foreign_accesses:7 ~precisions:[ "f32" ]
      ~sched:(fun g ->
        let op, _b, _i, _c = Sched.split_reduce ~axis:g.k ~target:g.out ~num_blocks:4 in
        [ op ]);
    (* --- the [Vectorized] arm: a SIMD accumulator grid plus its scalar tail, and the same
       rendering NESTED inside a surviving serial reduction level. --- *)
    member "retype-vectorized" "Retype the reduction axis to Vectorized" ~expect:simd_or_localized
      ~claimed:simd_claimed ~peel:simd_or_localized_peel ~peel_claimed:simd_peel_claimed
      ~expect_axis:LL.Vectorized ~sched:(fun g ->
        [ Sched.Retype { axis = g.k; ty = LL.Vectorized } ]);
    (* The peel localizes here on EVERY backend, the SIMD one included: a [Vectorized] level rides
       into the scope (the vector rendering folds its chains into the scope local), so the surviving
       outer half is peeled with the inner one inside it. *)
    member "split-then-vectorize-inner" "Split, then Retype the INNER half to Vectorized"
      ~expect:simd_or_localized ~claimed:simd_claimed
      ~peel:(Peeled { Cs.levels = 2; guards = [] })
      ~expect_axis:LL.Vectorized
      ~sched:(fun g ->
        (* 32 wide, which every lane ladder a real target has can cover; see {!cols}. *)
        let sp, _outer, inner =
          Sched.split ~axis:g.k ~factor:32 ~outer:LL.Serial ~inner:LL.Serial
        in
        [ sp; Sched.Retype { axis = inner; ty = LL.Vectorized } ]);
    (* The Vectorized arm's OTHER exit: a loop-carried accumulation that no vector rendering
       accepted falls back to the localizing peel, never to the pragma'd loop — the pragmas assert
       iteration independence, which an accumulation does not satisfy. A two-wide inner half is
       below the profitability gate, so the SIMD grid declines and the peel takes it. *)
    member "split-then-vectorize-narrow"
      "Split into a two-wide inner half, then Retype it Vectorized" ~expect_axis:LL.Vectorized
      ~peel:(Peeled { Cs.levels = 2; guards = [] })
      ~sched:(fun g ->
        let sp, _outer, inner = Sched.split ~axis:g.k ~factor:2 ~outer:LL.Serial ~inner:LL.Serial in
        [ sp; Sched.Retype { axis = inner; ty = LL.Vectorized } ]);
    (* --- a hardware reduction kind: the warp-shuffle tree where the backend binds one, and the
       serialized fallback (which is a serial level to the peel) where it does not. --- *)
    member "retype-workgroup-reduce" "Retype the reduction axis to Workgroup_reduce"
      ~expect:(if on_cpu then Localized else Warp)
      ~claimed:"warp-shuffle tree, or the localized scope where no lane index is bound"
        (* The shuffle tree takes accumulators whose RESIDENCY is single or double precision
           (gh-ocannl-682), and refuses anything else by raising rather than by falling back — so
           off the C backends this member is evaluable at f32 and at whichever narrow precisions the
           backend's [accum_prec] widens, which is bf16 on CUDA and nothing elsewhere. Its value
           claim stays bitwise there: the tree reassociates, but these operands' f32 partial sums
           are exact whatever the association, and both the shuffle and the localized serial
           baseline narrow to bf16 exactly once. *)
      ~available:(fun prec_name -> on_cpu || shuffle_takes prec_name)
      ~peel:
        (if on_cpu then Peeled { Cs.levels = 1; guards = [] }
         else Widened_elsewhere { Cs.levels = 1; guards = [] })
      ~peel_claimed:
        "codegen peels the serialized level, or the shared width decision is wide and the shuffle \
         tree holds it"
      ~expect_axis:LL.Workgroup_reduce
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Workgroup_reduce } ]);
    (* The plain [Workgroup] arm: a hardware binding where the backend has an index for the slot,
       and otherwise the localizing peel. Only the second half is a member here — binding a
       REDUCTION axis to a workgroup dimension is a cross-lane race wherever the binding exists, so
       the leg runs on the backends that serialize it and is skipped where it would not be. *)
    member "retype-workgroup" "Retype the reduction axis to Workgroup (no index bound: serialized)"
      ~available:(fun _ -> on_cpu)
      ~expect_axis:LL.Workgroup
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Workgroup } ]);
    (* --- the gh-490 runtime-extent guard, whose bound is a kernel parameter rather than a
       constant. The peel must see through it (gh-ocannl-693) and the mints must agree with the
       serial baseline under it (gh-ocannl-715): a refused mint round-trips the accumulator per
       copy, which on narrow storage is a different number. --- *)
    member "runtime-guard" "a runtime-extent guard, unscheduled" ~shape:Runtime_guard
      ~peel:(Peeled { Cs.levels = 1; guards = [ LL.Guard_confined ] })
      ~reference:Guarded_baseline;
    member "runtime-guard-unroll-mat" "a runtime-extent guard under Unroll ~materialize"
      ~shape:Runtime_guard ~peel:Minted_upstream ~reference:Guarded_baseline ~sched:(fun g ->
        [ Sched.Unroll { axis = g.k; materialize = true } ]);
    member "runtime-guard-partition" "a runtime-extent guard under Partition of the reduction axis"
      ~shape:Runtime_guard ~peel:Minted_upstream ~reference:Guarded_baseline ~sched:(fun g ->
        let pt, _ = Sched.partition ~axis:g.k ~breakpoints:[ 4; 12 ] in
        [ pt ]);
    (* --- gh-ocannl-721: a guard mixing the enclosing output index with the peeled reduction one.
       The output loop cannot be peeled here (the cell mentions its index), so the guard's [r] is a
       genuinely ENCLOSING symbol at the reduction level — and localization turns on whether the
       cell separates it. Declining leaves one storage read-modify-write per term, which on a
       backend whose policy widens the accumulator is a different number, not a slower one. --- *)
    (* The verdict is the whole point of this member (gh-ocannl-733): [Guard_lane_private] at ONE
       level. Were the cell free of [r] — the obvious way to write the shape — the peel would swallow
       both levels at the outer one under [Guard_confined], and every form and site claim below would
       still pass over a kernel in which the gh-ocannl-721 escape never ran. *)
    member "mixed-guard" "a guard mixing the enclosing output index with the peeled one"
      ~shape:Mixed_guard
      ~peel:(Peeled { Cs.levels = 1; guards = [ LL.Guard_lane_private ] })
      ~reference:Mixed_baseline;
    (* The same shape with the output axis actually BOUND to hardware, which is the form the issue
       states it in: [Workgroup r -> Serial k -> If (r + k < s) (out[r] += x[r,k])]. Where the
       backend binds the dimension this is the lane-private hoist itself — each lane loads and
       stores only its own [out[r]] — and where it does not, the axis serializes and the reduction
       level localizes exactly as above. Binding the OUTPUT axis is safe in a way binding the
       reduction axis is not, which is why this member runs everywhere and [retype-workgroup] does
       not. *)
    member "mixed-guard-workgroup" "the same guard with the output axis bound to a workgroup"
      ~shape:Mixed_guard
      ~peel:(Peeled { Cs.levels = 1; guards = [ LL.Guard_lane_private ] })
      ~reference:Mixed_baseline ~expect_axis:LL.Workgroup
      ~sched:(fun g -> [ Sched.Retype { axis = g.r; ty = LL.Workgroup } ]);
    (* --- the OTHER producer of the scope form: virtualization's inline at a read site --- *)
    (* The one member with no peel site at all: virtualization gives the accumulator its own scope
       and opens it from a zero-init, so nothing here reads the cell it writes and localization was
       never a question (gh-ocannl-733). *)
    member "virtual-accumulator" "a virtual accumulator inlined at its read site" ~shape:Virtual_acc
      ~peel:Never_a_site ~reference:Owned_cell;
    (* Virtualization's other spelling, and the same peel story: the accumulator lives in the
       inlined scope rather than in a cell, so no level holds a self-recurrence and the peel is
       never consulted. Both virtualization members are {!Never_a_site}, which is the fact that
       separates them from the schedule mints — those DO meet the peel, at a real reduction site,
       and are refused there. *)
    member "where-guarded-update" "virtualization's Where-guarded update spelling"
      ~shape:Where_scope ~peel:Never_a_site ~reference:Guarded_baseline ~extra:[ " ? " ];
    (* --- the two read-modify-write forms, i.e. the declines. Without these the localized claims
       could not fail: a classifier that answered "localized" for everything would pass every other
       member, and the whole table would be measuring nothing. --- *)
    member "decline-data-guard" "a data-dependent guard (not a pure index guard)" ~shape:Data_guard
      ~expect:Rmw ~peel:No_localization ~reference:Per_step;
    member "decline-sibling-statement" "a second statement in the reduction level" ~shape:Side_write
      ~expect:Rmw ~peel:No_localization ~reference:Per_step;
    member "decline-sibling-unrolled" "the same level Unrolled: one read-modify-write per copy"
      ~shape:Side_write ~expect:Rmw ~peel:No_localization ~reference:Per_step ~rmw_sites:cols
      ~expect_axis:LL.Unrolled ~sched:(fun g ->
        [ Sched.Unroll { axis = g.k; materialize = false } ]);
    (* --- gh-ocannl-754: the width-parity corpus. The awkward bodies under each retype kind, at
       the narrow precisions where the residency is wider than storage on the widening backends —
       which is where a rendering deciding the width for itself would show. Every value claim is
       parity with the body's own serial rendering. --- *)
    (* The storage-pinned accumulator under every kind. [f32] is left out on purpose: there the
       residency IS the storage width, the pin changes nothing, and the SIMD grid and the shuffle
       tree legitimately take the statement — reassociating f32 products that are not exact, so
       parity would not be bitwise and the claim would say nothing about width. *)
    member "rng" "an RNG-bearing contribution, unscheduled (the storage-pinned accumulator)"
      ~shape:Rng_contrib ~expect:Rmw ~peel:Pinned_to_storage ~reference:Rng_baseline
      ~precisions:[ "bf16"; "f16" ];
    member "rng-unroll" "the RNG-bearing level Unrolled: one read-modify-write per copy"
      ~shape:Rng_contrib ~expect:Rmw ~peel:Pinned_to_storage ~reference:Rng_baseline ~rmw_sites:cols
      ~expect_axis:LL.Unrolled ~precisions:[ "bf16"; "f16" ] ~sched:(fun g ->
        [ Sched.Unroll { axis = g.k; materialize = false } ]);
    (* The schedule mint DOES take an RNG-bearing nest into a scope — and the scope is then pinned
       to storage precision by the rng census ([C_syntax.rng_scope_ids]), so its local narrows on
       every update exactly as the read-modify-write does. A localized FORM whose accumulator is
       storage-width: the member that shows the width is a property of the decision and not of the
       form. *)
    member "rng-unroll-mat" "the RNG-bearing level Unroll ~materialize (a storage-width scope)"
      ~shape:Rng_contrib ~peel:Minted_upstream ~reference:Rng_baseline ~precisions:[ "bf16"; "f16" ]
      ~sched:(fun g -> [ Sched.Unroll { axis = g.k; materialize = true } ]);
    (* The Vectorized arm over a pinned body: the SIMD grid, which would render the loop-invariant
       draw at compute precision and accumulate its chains wide, takes the shared decision's pin and
       declines — the third sighting of the genre, had it been left to decide for itself. *)
    member "rng-vectorized" "the RNG-bearing axis Retyped to Vectorized: the SIMD grid declines"
      ~shape:Rng_contrib ~expect:Rmw ~peel:Pinned_to_storage ~reference:Rng_baseline
      ~expect_axis:LL.Vectorized ~precisions:[ "bf16"; "f16" ] ~sched:(fun g ->
        [ Sched.Retype { axis = g.k; ty = LL.Vectorized } ]);
    (* Where a lane index is bound the shuffle REFUSES a pinned body at a wider residency (loudly:
       [hardware_warp_shuffle]'s RNG leg pins the message), so the member is evaluable only where
       the level serializes. *)
    member "rng-workgroup-reduce"
      "the RNG-bearing axis Retyped to Workgroup_reduce (serialized: no lane index bound)"
      ~shape:Rng_contrib ~expect:Rmw ~peel:Pinned_to_storage ~reference:Rng_baseline
      ~expect_axis:LL.Workgroup_reduce ~precisions:[ "bf16"; "f16" ]
      ~available:(fun _ -> on_cpu)
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Workgroup_reduce } ]);
    (* The peel-declining bodies under the two retype kinds whose renderings hold the accumulator
       outside the cell. Under a bound lane index a declined [Workgroup_reduce] level is REFUSED
       loudly rather than handed to the hardware binding, which would race the shared cell: the
       lane-invariant self-updating [Set] is gh-ocannl-950's race criterion, and
       [hardware_warp_shuffle] pins the refusal on the GPUs. A refusal has no value to compare, so
       those two are evaluable where the level serializes; the Vectorized arm's decline has a
       defined exit on every backend. *)
    member "sibling-vectorized" "a second statement in the level, Retyped to Vectorized"
      ~shape:Side_write ~expect:Rmw ~peel:No_localization ~reference:Per_step
      ~expect_axis:LL.Vectorized ~sched:(fun g ->
        [ Sched.Retype { axis = g.k; ty = LL.Vectorized } ]);
    member "sibling-workgroup-reduce"
      "a second statement in the level, Retyped to Workgroup_reduce (serialized)" ~shape:Side_write
      ~expect:Rmw ~peel:No_localization ~reference:Per_step ~expect_axis:LL.Workgroup_reduce
      ~available:(fun _ -> on_cpu)
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Workgroup_reduce } ]);
    member "data-guard-vectorized" "a data-dependent guard, Retyped to Vectorized" ~shape:Data_guard
      ~expect:Rmw ~peel:No_localization ~reference:Per_step ~expect_axis:LL.Vectorized
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Vectorized } ]);
    member "data-guard-workgroup-reduce"
      "a data-dependent guard, Retyped to Workgroup_reduce (serialized)" ~shape:Data_guard
      ~expect:Rmw ~peel:No_localization ~reference:Per_step ~expect_axis:LL.Workgroup_reduce
      ~available:(fun _ -> on_cpu)
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Workgroup_reduce } ]);
  ]

(* {1 Coverage ratchets}

   Printing the member list says what the table SWEEPS; on its own it cannot say what the table
   MISSES. A schedule op or an axis kind that no member reaches changes neither the list nor the
   output, which leaves exhaustiveness as an unenforced advertisement and puts the next form back on
   the review-only discovery path this test exists to replace (Codex P2, round 1).

   So the two variant types that DEFINE the space are matched exhaustively below, and the compiler
   is the ratchet: adding a constructor to [Schedule.optop] or to [Low_level.axis_type] fails to
   build this file until someone says which member reaches it, or why it is out of scope. The
   classification is printed, and every member name it cites is checked to exist — so the coverage
   claim cannot drift from the table by renaming either side.

   The limit is worth stating rather than glossing: this catches a new schedule OP and a new axis
   KIND. It does not catch a new rendering ARM inside an axis kind the table already reaches — if
   such an arm fires for a member the form claim moves and the golden diffs, but an arm reachable
   only through a shape no member builds is invisible here. What the peel census
   ([C_syntax.peel_census], gh-ocannl-733) closes is the neighbouring gap: for the localizing peel,
   which DECISION each member's kernel earned, so an arm reached through a shape the table does
   build cannot be credited to a member that never took it. An arm no member's shape reaches at all
   still needs a census over the RENDERINGS, in the shape of [C_syntax.mma_census]. *)

type coverage =
  | Covered of string list  (** The members that reach it, by slug. *)
  | Out_of_scope of string  (** Why it is not a serial-rendered form of a reduction. *)

let optop_coverage (op : Sched.optop) : coverage =
  match op with
  | Sched.Split _ ->
      Covered
        [
          "split-then-unroll";
          "split-then-swap";
          "split-then-vectorize-inner";
          "split-then-vectorize-narrow";
        ]
  | Sched.Swap _ -> Covered [ "split-then-swap" ]
  | Sched.Retype _ ->
      Covered
        [
          "retype-vectorized";
          "split-then-vectorize-inner";
          "split-then-vectorize-narrow";
          "retype-workgroup-reduce";
          "retype-workgroup";
          "mixed-guard-workgroup";
          "rng-vectorized";
          "rng-workgroup-reduce";
          "sibling-vectorized";
          "sibling-workgroup-reduce";
          "data-guard-vectorized";
          "data-guard-workgroup-reduce";
        ]
  | Sched.Unroll _ ->
      Covered [ "unroll-annot"; "unroll-mat"; "unroll-outer-mat"; "rng-unroll"; "rng-unroll-mat" ]
  | Sched.Partition _ -> Covered [ "partition"; "partition-then-unroll"; "partition-outer" ]
  | Sched.Pad _ -> Covered [ "pad-then-unroll-mat" ]
  | Sched.Split_reduce _ -> Covered [ "split-reduce" ]
  | Sched.Tensorize _ -> Covered [ "tile-mma-fallback" ]
  | Sched.Expand_zero _ -> Covered [ "tile-mma-fallback" ]
  | Sched.Stage _ ->
      Out_of_scope
        "stages an OPERAND into shared memory: it changes where the reduction reads from, never \
         where its accumulator lives"
  | Sched.Privatize _ ->
      Out_of_scope
        "gives a scratch node a per-thread copy, which is a parallelism decision about a different \
         node than the one being accumulated"
  | Sched.Fuse_epilogue _ ->
      Out_of_scope
        "splices a consumer AFTER the reduction's closing store, so the form it follows is \
         whichever one this table already pins"

let axis_coverage (ty : LL.axis_type) : coverage =
  match ty with
  | LL.Serial -> Covered [ "serial"; "decline-data-guard"; "decline-sibling-statement" ]
  | LL.Unrolled -> Covered [ "unroll-annot"; "decline-sibling-unrolled"; "rng-unroll" ]
  | LL.Vectorized ->
      Covered
        [
          "retype-vectorized";
          "split-then-vectorize-inner";
          "split-then-vectorize-narrow";
          "rng-vectorized";
          "sibling-vectorized";
          "data-guard-vectorized";
        ]
  | LL.Workgroup_reduce ->
      Covered
        [
          "retype-workgroup-reduce";
          "rng-workgroup-reduce";
          "sibling-workgroup-reduce";
          "data-guard-workgroup-reduce";
        ]
  | LL.Workgroup -> Covered [ "retype-workgroup"; "mixed-guard-workgroup" ]
  | LL.Grid ->
      Out_of_scope
        "its fallback is the plain serial loop and NOT the localizing one — the one arm that \
         serializes without localizing. No schedule op in the tree produces an unbindable, \
         non-parallel-eligible Grid level over a reduction axis, and one that existed would be a \
         cross-thread race rather than a width question"

(* Sample values, one per constructor, purely to drive the printed table: the classification itself
   is the exhaustive [match] above, which is what the compiler checks. A constructor added without a
   sample here still fails the build at that match — add both. *)
let coverage_sym = Ll_test.sym ()

let coverage_node =
  Int.incr next_id;
  Tn.create (Tn.Specified Ops.single) ~id:!next_id ~label:[ "rfcov" ]
    ~unpadded_dims:(lazy [| 1 |])
    ~padding:(lazy None)
    ()

let optop_samples : Sched.optop list =
  let s = coverage_sym in
  [
    Sched.Split
      {
        axis = s;
        factor = 1;
        outer = LL.Serial;
        inner = LL.Serial;
        outer_index = s;
        inner_index = s;
      };
    Sched.Swap { outer = s; inner = s };
    Sched.Retype { axis = s; ty = LL.Serial };
    Sched.Unroll { axis = s; materialize = false };
    Sched.Partition { axis = s; breakpoints = []; segment_indices = [] };
    Sched.Pad { axis = s; to_multiple_of = 1 };
    Sched.Stage
      {
        source = coverage_node;
        tile_loops = [];
        shared = false;
        cooperative = None;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 0;
        tile_prec = None;
      };
    Sched.Privatize { target = coverage_node; over = s };
    Sched.Expand_zero { tn = coverage_node; indices = [] };
    Sched.Tensorize { i = s; j = s; k = s; lane = s; simd_width = 1; tile = None };
    Sched.Fuse_epilogue { target = coverage_node; shared = false };
    Sched.Split_reduce
      {
        axis = s;
        target = coverage_node;
        num_blocks = 1;
        block_index = s;
        inner_index = s;
        combine_indices = [];
      };
  ]

let axis_samples : LL.axis_type list =
  [ LL.Serial; LL.Unrolled; LL.Vectorized; LL.Workgroup; LL.Workgroup_reduce; LL.Grid ]

(* The constructor's own name, off its sexp — so the printed table cannot disagree with the value it
   describes. *)
let constructor_name sexp =
  match sexp with Sexp.List (Sexp.Atom name :: _) -> name | Sexp.Atom name -> name | _ -> "?"

(* {1 Running the table} *)

let same_form a b = String.equal (form_name a) (form_name b)

(* Non-empty by construction (gh-ocannl-746): [Array.for_all2_exn] answers [true] on two empty
   arrays, and a readback and the reference it is compared against go empty TOGETHER -- the
   reference went through the same path. {!Verdict.p_all2} is the claim-shaped form; the sites
   reached through this helper keep a boolean, so the guard lives here. *)
let agrees got want =
  (not (Array.is_empty got))
  && Array.length got = Array.length want
  && Array.for_all2_exn got want ~f:Float.equal

let show vs = String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%h")))
let routine_stem slug = String.tr slug ~target:'-' ~replacement:'_'

(* The coverage tables, printed and asserted: every constructor is reached by a member that EXISTS,
   or exempted with a reason. A slug naming no member fails the claim, so renaming a member without
   updating its coverage entry is a failure rather than a quiet gap. *)
let () =
  let slugs =
    "tile-mma-fallback" :: List.map members ~f:(fun m -> m.slug) |> Set.of_list (module String)
  in
  let report ~what ~name_of ~classify samples =
    let ok = ref true in
    List.iter samples ~f:(fun sample ->
        let name = name_of sample in
        match classify sample with
        | Covered [] ->
            ok := false;
            Stdio.printf "  %-18s covered by nothing\n" name
        | Covered members_of ->
            let missing = List.filter members_of ~f:(fun slug -> not (Set.mem slugs slug)) in
            if not (List.is_empty missing) then begin
              ok := false;
              Stdio.eprintf "  %s cites members that do not exist: %s\n" name
                (String.concat ~sep:", " missing)
            end;
            Stdio.printf "  %-18s covered by %s\n" name (String.concat ~sep:", " members_of)
        | Out_of_scope why -> Stdio.printf "  %-18s out of scope: %s\n" name why);
    p
      (Printf.sprintf
         "every %s constructor is reached by a member that exists, or exempted with a reason" what)
      !ok
  in
  Stdio.printf "Schedule.optop coverage:\n";
  report ~what:"Schedule.optop"
    ~name_of:(fun op -> constructor_name (Sched.sexp_of_optop op))
    ~classify:optop_coverage optop_samples;
  Stdio.printf "Low_level.axis_type coverage:\n";
  report ~what:"Low_level.axis_type"
    ~name_of:(fun ty -> constructor_name (LL.sexp_of_axis_type ty))
    ~classify:axis_coverage axis_samples;
  (* A duplicated sample would print one constructor twice and leave another unprinted, while the
     exhaustive matches above stayed satisfied. *)
  let distinct l = List.length (List.dedup_and_sort l ~compare:String.compare) = List.length l in
  p "the coverage samples name distinct constructors"
    (distinct (List.map optop_samples ~f:(fun op -> constructor_name (Sched.sexp_of_optop op)))
    && distinct (List.map axis_samples ~f:(fun ty -> constructor_name (LL.sexp_of_axis_type ty))))

(* The table's VALUE claims follow the numerics policy wherever it goes — that is what
   {!expected_residency} is for. Its FORM claims do not: whether the SIMD reduction rendering fires
   for a narrow storage precision is itself policy-dependent (with [narrow_compute_f32] off, a bf16
   Vectorized reduction declines to the localizing peel instead), and modelling that per member
   would be a wider claim than this table makes. So the premise is stated rather than assumed: under
   a non-default policy the forms below are not the ones to expect, and this says so in one line
   instead of leaving four members to fail obscurely. *)
let () =
  let policy = Numerics.get () in
  p "the numerics policy is the default the member table's forms are stated for"
    (policy.Numerics.narrow_compute_f32
    && Numerics.equal_fp16_mode policy.Numerics.fp16_arithmetic Numerics.Fp16_auto)

(* The [Tile_mma] fallback's own reduction is a serial nest like any other, and codegen's peel is
   what localizes it — the leg below checks that against the routine's peel census beside the
   [Tile_mma] one. *)
let mma_fallback_peel = Peeled { Cs.levels = 1; guards = [] }
let mma_fallback_peel_claimed = peel_claim_name mma_fallback_peel

let () =
  Stdio.printf "reduction: out[%d] = sum of %d terms, hand-built Low_level, %d members\n" rows cols
    (List.length members + 1);
  List.iteri members ~f:(fun i m ->
      Stdio.printf "  %02d %-27s [%-12s] over %-21s %-70s -> %s\n" (i + 1) m.slug
        (String.concat ~sep:" " m.precisions)
        (shape_name m.shape) m.what m.claimed;
      (* The DECISION on its own line under the form: what codegen's peel must have decided while
         emitting this member's kernel (gh-ocannl-733). A member that stopped exercising its own
         code path shows up here as a golden diff, whether or not its form moved. *)
      Stdio.printf "     %-27s %s\n" "" m.peel_claimed);
  Stdio.printf "  %02d %-27s [%-12s] over %-21s %-70s -> %s\n"
    (List.length members + 1)
    "tile-mma-fallback" "f32 bf16 f16" "small contraction"
    "Tensorize whose emission preconditions fail (transposed-B operands)" (form_name Mma_fallback);
  Stdio.printf "     %-27s %s\n" "" mma_fallback_peel_claimed

(* The two declarations are not independent, and pinning the relation is what keeps the peel claim
   from being a free-floating label: a member whose peel localizes nothing must render its
   accumulator somewhere else — a decline, a SIMD grid, a shuffle tree — unless the scope reached
   codegen already built, which is exactly what {!Minted_upstream} says and {!No_localization}
   denies. So the pair (form, decision) is checked for coherence over the whole table, and a member
   that declares "localized" beside "nothing localized" without saying where the scope came from
   fails here rather than on some backend. *)
let () =
  Verdict.p_all "every member's peel claim is coherent with the form it claims" members ~f:(fun m ->
      match m.peel with
      | Peeled _ -> true
      | Minted_upstream | Never_a_site ->
          same_form m.expect Localized || same_form m.expect Partials_combine
      | No_localization -> not (same_form m.expect Localized)
      | Widened_elsewhere _ -> same_form m.expect Simd || same_form m.expect Warp
      | Pinned_to_storage -> same_form m.expect Rmw)

(* The baselines: the plain nest with no schedule ops, and the runtime-extent-guarded nest with no
   schedule ops, one of each per precision. Every localizing member is compared against one of them,
   so a regression that moved a BASELINE fails its own host-reference claim below rather than
   silently shifting the whole table. *)
let baselines =
  List.map precs ~f:(fun (prec_name, prec) ->
      let prog = make ~prec ~shape:Plain () in
      let values, _, _, _ = execute ~name:("rf_baseline_" ^ prec_name) ~prog ~sched:[] in
      (prec_name, values))

let guarded_baselines =
  List.map precs ~f:(fun (prec_name, prec) ->
      let prog = make ~prec ~shape:Runtime_guard () in
      let values, _, _, _ = execute ~name:("rf_guarded_baseline_" ^ prec_name) ~prog ~sched:[] in
      (prec_name, values))

let mixed_baselines =
  List.map precs ~f:(fun (prec_name, prec) ->
      let prog = make ~prec ~shape:Mixed_guard () in
      let values, _, _, _ = execute ~name:("rf_mixed_baseline_" ^ prec_name) ~prog ~sched:[] in
      (prec_name, values))

(* The RNG-bearing nest's own serial rendering, at the precisions its members run at. *)
let rng_precs = List.filter precs ~f:(fun (prec_name, _) -> not (String.equal prec_name "f32"))

let rng_baselines =
  List.map rng_precs ~f:(fun (prec_name, prec) ->
      let prog = make ~prec ~shape:Rng_contrib () in
      let values, _, _, _ = execute ~name:("rf_rng_baseline_" ^ prec_name) ~prog ~sched:[] in
      (prec_name, values))

let baseline prec_name = List.Assoc.find_exn baselines ~equal:String.equal prec_name
let guarded_baseline prec_name = List.Assoc.find_exn guarded_baselines ~equal:String.equal prec_name
let mixed_baseline prec_name = List.Assoc.find_exn mixed_baselines ~equal:String.equal prec_name
let rng_baseline prec_name = List.Assoc.find_exn rng_baselines ~equal:String.equal prec_name

(* The discrimination control for the RNG members (gh-ocannl-754). Their value claims are parity
   with {!rng_baselines}, and parity is worth nothing where the width cannot show — so the draw the
   pinned rendering actually makes is read back from a one-cell kernel at the storage precision, and
   the RNG-scaled operands are shown to discriminate accumulator width the way {!cells} does:
   narrowing the running sum at every step differs, in every row, from narrowing it once. The
   products are rounded to storage in both models, so the two differ only in where the SUM narrows,
   which is the property the members are about.

   The draw is also the cross-backend pin gh-ocannl-951 asked for. A conversion picks which bits of
   the key it consumes from the precision it renders at (gh-ocannl-517) -- per PRECISION, not per
   backend -- and the scalar bf16 and half conversions both take the single draw of the first 32
   bits, narrowed, on every backend (cc's half used to consume the low 16 bits instead, so one key
   drew a different half on cc than on the GPUs). The draws are exact by construction (a bit
   selection and one rounding, never a reduction), so they belong on stdout: a backend that consumed
   other bits prints another number, and the claim that each narrow draw IS the single draw narrowed
   by the library's own conversion fails on it. The host copy of the conversion ([Ops]'s stubs over
   [builtins.c], the hand-synced twin of the cc builtins -- gh-ocannl-656) is held to the same draw,
   which is the first mechanical check that the pair has not drifted. *)
let device_draw (prec_name, prec) =
  Int.incr next_id;
  let uvals =
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ "rfdraw" ]
      ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None)
      ()
  in
  Ll_test.materialize uvals;
  let prog =
    {
      llc =
        LL.Set
          {
            tn = uvals;
            idcs = [| Idx.Fixed_idx 0 |];
            llsc = LL.Unop (Ops.Uint4x32_to_prec_uniform1, (LL.Constant_bits rng_key, Ops.uint4x32));
            debug = "";
          };
      raw = None;
      r = Ll_test.sym ();
      k = Ll_test.sym ();
      out = uvals;
      materialized = [ uvals ];
      seed = [];
      bindings = Idx.Empty;
      bind = (fun _ -> ());
      verify = [];
    }
  in
  let values, _, _, _ = execute ~name:("rf_rng_draw_" ^ prec_name) ~prog ~sched:[] in
  values.(0)

(* The key's four lanes, as the host conversions consume them: what [LL.Constant_bits] widens to on
   the device. *)
let rng_key_lanes = Ops.int64_to_uint4x32 rng_key

let single_draw =
  let draw = device_draw ("f32", Ops.single) in
  Stdio.printf "the f32 draw of the RNG members' key: %s\n" (Test_utils.hex_float draw);
  draw

let () =
  List.iter rng_precs ~f:(fun (prec_name, prec) ->
      let draw = device_draw (prec_name, prec) in
      Stdio.printf "the %s draw of the RNG members' key: %s\n" prec_name (Test_utils.hex_float draw);
      p
        (Printf.sprintf
           "the %s draw is the f32 draw narrowed to %s: the conversion consumes the same 32 bits \
            of the key on every backend (gh-ocannl-951)"
           prec_name prec_name)
        (Float.equal draw (round prec single_draw));
      let host_draw =
        match prec with
        | Ops.Bfloat16_prec _ ->
            Ops.bfloat16_to_single (Ops.uint4x32_to_bfloat16_uniform rng_key_lanes)
        | Ops.Half_prec _ -> Ops.half_to_single (Ops.uint4x32_to_half_uniform rng_key_lanes)
        | _ -> assert false
      in
      p
        (Printf.sprintf
           "the host copy of the %s conversion draws what the backend draws (the builtins.c twin \
            has not drifted, gh-ocannl-656)"
           prec_name)
        (Float.equal draw host_draw);
      let term r k = round prec (cell r k *. draw) in
      let per_step =
        Array.init rows ~f:(fun r ->
            let acc = ref (seed_value r) in
            for k = 0 to cols - 1 do
              acc := round prec (!acc +. term r k)
            done;
            !acc)
      in
      let once =
        Array.init rows ~f:(fun r ->
            let acc = ref (seed_value r) in
            for k = 0 to cols - 1 do
              acc := !acc +. term r k
            done;
            round prec !acc)
      in
      (* "Differs", as the plain operands' control above says it: in SOME row. Which rows coincide
         depends on the draw, which the pins above hold equal across backends but which any future
         change to the conversion moves, so a per-row demand would tie this control to one draw,
         while a single differing row is what a width divergence needs to show. *)
      p
        (Printf.sprintf
           "at %s the RNG-scaled operands discriminate accumulator width: the draw is a factor in \
            (0, 1) and per-step narrowing differs from a once-narrowed whole-nest sum"
           prec_name)
        (Float.(draw > 0. && draw < 1.) && not (Array.for_all2_exn per_step once ~f:Float.equal)))

let () =
  List.iter precs ~f:(fun (prec_name, prec) ->
      (* One pair of host references per term count: the full axis, and what a guard admitting
         [guard_terms] of it leaves. *)
      let refs ?(from = seed_value) terms =
        (whole_nest_ref ~terms ~from prec, per_step_ref ~terms ~from prec)
      in
      let full = fun (_ : int) -> cols in
      let guarded_terms = fun (_ : int) -> guard_terms in
      let mixed_terms r = guard_terms - r in
      let wide, stepped = refs full in
      let got = baseline prec_name in
      let differ = not (Array.for_all2_exn wide stepped ~f:Float.equal) in
      (* The discrimination control. Without it every value claim in the table could hold because
         the operands never leave the storage format's exactness range, in which case a form that
         narrowed at every step would be indistinguishable from one that narrows once. *)
      let g_wide, g_stepped = refs guarded_terms in
      let differ_guarded = not (Array.for_all2_exn g_wide g_stepped ~f:Float.equal) in
      if String.equal prec_name "f32" then
        let coincide =
          (not (Array.is_empty wide))
          && (not (Array.is_empty g_wide))
          && (not differ) && not differ_guarded
        in
        p "at f32 the whole-nest and per-step references coincide (the identity-precision leg)"
          coincide
      else
        (* At BOTH term counts: the guarded baseline is asserted against a selected reference too,
           and that assertion is only worth something where the two references are distinguishable
           over the guard's shorter run. *)
        p
          (Printf.sprintf
             "at %s the whole-nest and per-step references differ over the full axis and over the \
              guarded prefix (the operands discriminate accumulator width)"
             prec_name)
          (differ && differ_guarded);
      (* Which reference the policy selects, and the assertion that the run took THAT one. Not "one
         of the two": see {!expected_residency}. *)
      let selected ~terms ~terms_desc values ~what =
        let wide, stepped = refs terms in
        let claim =
          Printf.sprintf "the %s %s matches the reference its backend's accumulator policy selects"
            prec_name what
        in
        let residency = expected_residency prec in
        let want = match residency with Wider -> wide | At_storage -> stepped in
        let ok = Array.for_all2_exn values want ~f:Float.equal in
        if not ok then
          Stdio.eprintf "  %s %s over %s: got [%s] want [%s] (policy says %s)\n" prec_name what
            terms_desc (show values) (show want) (residency_name residency);
        p_all2 claim values want ~f:Float.equal;
        Some residency
      in
      let residency =
        selected ~terms:full ~terms_desc:"the full axis" got ~what:"serial baseline"
      in
      let guarded = guarded_baseline prec_name in
      ignore
        (selected ~terms:guarded_terms
           ~terms_desc:(Printf.sprintf "%d terms" guard_terms)
           guarded ~what:"guarded baseline");
      (* The gh-ocannl-721 claim, and the one the localization moves: before it, the mixed-guarded
         nest fell back to a storage read-modify-write per term, so on a backend whose policy widens
         the accumulator this read the per-step reference instead of the whole-nest one. *)
      let mixed = mixed_baseline prec_name in
      ignore
        (selected ~terms:mixed_terms ~terms_desc:"20, 19 and 18 terms by row" mixed
           ~what:"mixed-guard baseline");
      (* And the guards SELECT: without this the guarded members could not fail at all, since a mint
         or a renderer that discarded the predicate would land on the unguarded sum and still match.
         Both references the guarded members use are covered — the executed guarded baseline (the
         localizing members) and the host per-step reference at the guarded term count (the
         data-guarded decline). *)
      let _, stepped_guarded = refs guarded_terms in
      p
        (Printf.sprintf
           "the %s guards are load-bearing: %d of %d terms differs from the full axis in both the \
            executed baseline and the per-step reference"
           prec_name guard_terms cols)
        ((not (Array.for_all2_exn guarded got ~f:Float.equal))
        && not (Array.for_all2_exn stepped_guarded stepped ~f:Float.equal));
      (* And the mixed guard's ROW DEPENDENCE is load-bearing: its rows must not all admit the same
         prefix, or a regression that read the guard as [k < s] — dropping the very symbol that made
         it mixed — would land on the runtime-extent baseline and pass. *)
      p
        (Printf.sprintf
           "the %s mixed guard admits a different prefix in every row: it differs from both the \
            full-axis and the runtime-extent baselines"
           prec_name)
        ((not (Array.for_all2_exn mixed got ~f:Float.equal))
        && not (Array.for_all2_exn mixed guarded ~f:Float.equal));
      (* Which regime this backend is in, on stderr so the golden stays backend-uniform: whether the
         accumulator resolves wider than storage decides whether the localized and read-modify-write
         forms are also distinguishable BY VALUE here, or only structurally. *)
      Stdio.eprintf "accumulator residency at %s: %s\n%!" prec_name
        (match residency with
        | Some r -> residency_name r
        | None -> residency_name (expected_residency prec)))

let () =
  List.iter members ~f:(fun m ->
      List.iter precs ~f:(fun (prec_name, prec) ->
          if List.mem m.precisions prec_name ~equal:String.equal then begin
            let form_claim =
              Printf.sprintf "%s @ %s renders the form its composition claims" m.slug prec_name
            in
            let value_claim =
              Printf.sprintf "%s @ %s agrees with its reference value" m.slug prec_name
            in
            let evidence_claim =
              Printf.sprintf "%s @ %s: its schedule ops left their mark on the lowered IR" m.slug
                prec_name
            in
            let peel_claim =
              Printf.sprintf "%s @ %s: codegen's peel decided what the composition declares" m.slug
                prec_name
            in
            (* Built before the availability check: a shape's extra VERIFY claims are printed on the
               skipped path too, and only the program knows what they are. *)
            let prog = make ~prec ~shape:m.shape () in
            if not (m.available prec_name) then begin
              (* Every claim an evaluable leg prints, in the same order: the golden is one file for
                 all backends, so a claim emitted only on the available path leaves a hole rather
                 than a skip. *)
              List.iter prog.verify ~f:(fun (claim, _, _) ->
                  skipped (Printf.sprintf "%s @ %s: %s" m.slug prec_name claim));
              skipped evidence_claim;
              skipped form_claim;
              skipped peel_claim;
              skipped value_claim
            end
            else begin
              let name = Printf.sprintf "rf_%s_%s" (routine_stem m.slug) prec_name in
              let sched = m.sched prog in
              let got, verified, (before, after, per_op), peel = execute ~name ~prog ~sched in
              (* The peel census of this kernel, on stderr: it names backend kernels and counts
                 sites the surrounding structure decides, so it is a diagnostic; the CLAIM it
                 supports is on stdout. *)
              Stdio.eprintf "  %s peel census: %s\n%!" name (Cs.peel_summary_string peel);
              (* Every node the shape writes, not only the accumulator. *)
              List.iter verified ~f:(fun (claim, values, want) ->
                  let ok = agrees values want in
                  if not ok then
                    Stdio.eprintf "  %s: %s -- got [%s] want [%s]\n" name claim (show values)
                      (show want);
                  p (Printf.sprintf "%s @ %s: %s" m.slug prec_name claim) ok);
              (* The op is bound to an observable effect: a non-empty schedule must have CHANGED the
                 lowered IR, and one that names an axis kind must have left that kind in it. *)
              (match (before, after) with
              | Some b, Some a ->
                  let changed =
                    (List.is_empty sched || not (Sexp.equal (LL.sexp_of_t b) (LL.sexp_of_t a)))
                    && (List.is_empty per_op
                       || (List.length per_op = List.length sched && List.for_all per_op ~f:Fn.id))
                  in
                  let axis_ok =
                    match m.expect_axis with
                    | None -> true
                    | Some ty -> List.exists (axis_kinds a) ~f:(fun k -> LL.equal_axis_type k ty)
                  in
                  if not (changed && axis_ok) then
                    (* Phrased with [%s] rather than a trailing [%b] on purpose: this is a
                       failure-only diagnostic beside the real claim, and a format ending on a bare
                       boolean behind a separator is exactly the self-decided-verdict shape
                       [verdict_ratchet] hunts. Better to say it in words than to earn an
                       exemption. *)
                    Stdio.eprintf
                      "  %s: the schedule left no mark -- IR changed: %s, named axis kind present: \
                       %s\n"
                      name (Bool.to_string changed) (Bool.to_string axis_ok);
                  p evidence_claim (changed && axis_ok)
              | _ ->
                  Stdio.eprintf "  %s: the lowered transform was never invoked\n" name;
                  p evidence_claim false);
              let src = Generated.read name in
              let updates = scope_update_statements src in
              let extra_ok =
                List.for_all m.extra ~f:(fun sub ->
                    List.exists updates ~f:(fun st -> String.is_substring st ~substring:sub))
              in
              if not extra_ok then
                Stdio.eprintf "  %s: no scope-local assignment among %d contains all of [%s]\n" name
                  (List.length updates) (String.concat ~sep:"; " m.extra);
              (match read_form src ~label:"rfout" with
              | None ->
                  Stdio.eprintf "  %s: no accumulator identifier in the emitted kernel\n" name;
                  p form_claim false
              | Some reading ->
                  let rendered = form_of reading in
                  (* The coarse form is not the whole claim: a localizing composition must close the
                     cell exactly as often as it opens the output loop's body, and touch the node no
                     more than that (Codex P2, round 1). The declining and partial forms are
                     identified by what they do to the node in the first place, so the count adds
                     nothing there. *)
                  let sites_ok =
                    match m.expect with
                    | Localized ->
                        reading.stores_from_local = m.store_sites
                        && reading.node_accesses <= 2 * m.store_sites
                        (* And NOTHING closes a partial into another node on the way (Codex P2,
                           round 11). A split, partition or unroll regression that started closing
                           intermediate partials elsewhere before the single final store would keep
                           the Localized reading and every count above — and on a backend whose
                           accumulator already resides at storage width, the added store/reload
                           seams need not move the executed value either. Foreign stores belong to
                           the explicit partials form, which declares how many it has. *)
                        && reading.foreign_local_stores = m.foreign_sites
                    | Simd | Warp ->
                        (* A vector or shuffle epilogue closes the cell ONCE per nest, and may spell
                           that as [out[i] = out[i] + vred_total] rather than as a store from a
                           scope local — textually a read-modify-write, but one per nest instead of
                           one per step, which is the whole distinction. Either spelling counts, and
                           the access bound is what refuses a regression that closed once per
                           chain. *)
                        reading.stores_from_local + reading.rmw_statements = m.store_sites
                        && reading.node_accesses <= 2 * m.store_sites
                        (* And no partial closed elsewhere on the way, exactly as for an ordinary
                           localized member (Codex P2, round 12 — the round-11 fix belonged here too
                           and I applied it to one arm only). A SIMD or shuffle renderer that began
                           closing intermediate partials into another node would keep its marker,
                           its single target epilogue and these target-only counts, and the f32
                           legs' exactly-representable arithmetic would absorb the added
                           store/reload seams. *)
                        && reading.foreign_local_stores = m.foreign_sites
                    | Rmw ->
                        reading.rmw_statements = m.rmw_sites
                        && reading.node_accesses <= 2 * m.rmw_sites
                    | Partials_combine ->
                        reading.foreign_local_stores = m.foreign_sites
                        && reading.rmw_statements = m.rmw_sites
                        && reading.stores_from_local = 0
                        && reading.node_accesses <= 2 * m.rmw_sites
                        && reading.foreign_accesses = m.foreign_accesses
                    | Mma_fallback ->
                        (* No table member claims it: the contraction leg is separate and does its
                           own site check, against a bound of its own (the einsum lowering's
                           whole-node zero-init is a third touch the members do not have). *)
                        true
                    | Unrecognized ->
                        (* No member claims it; [same_form] already fails. *)
                        false
                  in
                  if not sites_ok then
                    Stdio.eprintf
                      "  %s: %d closing store(s), %d rmw statement(s), %d foreign store(s), %d \
                       node subscript(s); expected %d store / %d rmw / %d foreign sites\n"
                      name reading.stores_from_local reading.rmw_statements
                      reading.foreign_local_stores reading.node_accesses m.store_sites m.rmw_sites
                      m.foreign_sites;
                  if not (same_form rendered m.expect) then
                    Stdio.eprintf
                      "  %s: rendered %s, claimed %s (node subscripts %d, rmw statements %d, \
                       stores from a local %d, foreign local stores %d)\n"
                      name (form_name rendered) (form_name m.expect) reading.node_accesses
                      reading.rmw_statements reading.stores_from_local reading.foreign_local_stores;
                  p form_claim (same_form rendered m.expect && sites_ok && extra_ok));
              (* And the DECISION, which the form above cannot deliver (gh-ocannl-733). Read off the
                 routine's peel census rather than off the emitted text: a member declaring a peel
                 its kernel did not perform fails here even when every textual claim holds, which is
                 the whole of what this instrument adds. *)
              check_peel ~name ~claim:peel_claim peel m.peel;
              let want =
                match m.reference with
                | Baseline -> baseline prec_name
                | Guarded_baseline -> guarded_baseline prec_name
                | Mixed_baseline -> mixed_baseline prec_name
                | Per_step -> per_step_ref ~terms:(terms_of_shape m.shape) ~from:seed_value prec
                | Owned_cell -> (
                    let from _ = 0.0 in
                    let full (_ : int) = cols in
                    match expected_residency prec with
                    | Wider -> whole_nest_ref ~terms:full ~from prec
                    | At_storage -> per_step_ref ~terms:full ~from prec)
                | Rng_baseline -> rng_baseline prec_name
              in
              let ok = agrees got want in
              if not ok then Stdio.eprintf "  %s: got [%s] want [%s]\n" name (show got) (show want);
              p value_claim ok
            end
          end))

(* {1 The [Tile_mma] fallback}

   The register-tiled micro-kernel holds its C-tile across the whole k extent, and when its emission
   preconditions fail it renders the lane-0 SCALAR fallback — whose reduction is a serial nest like
   any other, and therefore another member of this set (gh-ocannl-663 review round 2, "scopes inside
   [Tile_mma] fallbacks"). It needs an i/j/k triple, so it rides a small contraction rather than the
   row sum, and it is read through the routine's own [Tile_mma] census rather than through the
   textual classifier: [Scalar_fallback] is exactly "every [Tile_mma] statement declined", which is
   the fact the member claims.

   The transposed-B operand layout is the decline the C backends actually have (the gradient-GEMM
   shape); on a backend whose renderer does not decline there, the member reports skipped. *)

let mma_n = 16

(* The contraction's operands. The moduli are coprime to the row stride and LARGER than the extent
   (Codex P2, round 6): with [t = i * 16 + k], a modulus [m] gives [A[i+d,k] = A[i,k]] exactly when
   [16 d = 0 mod m], which for [m] coprime to 16 means [m | d] — so any [m <= 15] has an in-range
   row period, and the previous 7 and 5 made rows 7 apart and columns 5 apart interchangeable. A
   [Tile_mma] fallback regression that swapped or substituted such rows stayed bitwise equal to the
   plain run, which neither the census nor the accumulator-form check inspects. 17 and 19 are
   coprime to 16 and exceed 15, so no in-range step is a period on either axis. *)
let mma_a = Ll_test.cycle ~dims:[| mma_n; mma_n |] ~modulus:17 ~offset:1. ~stride:0.25
let mma_b = Ll_test.cycle ~dims:[| mma_n; mma_n |] ~modulus:19 ~offset:1. ~stride:0.5

let mma_matmul ~tag ~prec =
  (* The operands take the LEG's precision explicitly, through [NTDSL.init] rather than
     [TDSL.ndarray] — which has no [~prec] and so reads [default_prec] (Codex P2, round 8). That
     left the bf16 and f16 legs contracting f32 operands into a narrow output, a three-precision
     sweep in name only, and made what this leg compiles depend on a configuration key. *)
  let ma = NTDSL.init ~l:(tag ^ "_a") ~prec ~o:[ mma_n; mma_n ] ~f:mma_a () in
  let mb = NTDSL.init ~l:(tag ^ "_b") ~prec ~o:[ mma_n; mma_n ] ~f:mma_b () in
  let%op mc = ma +* "ik;jk=>ij" mb in
  Tn.update_prec mc.Tensor.value prec;
  mc

let mma_nest_syms (opt : LL.optimized) =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Idx.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  let paths =
    List.filter_map (LL.flat_lines [ opt.LL.llc ]) ~f:(fun stmt ->
        match path stmt with [] -> None | pth -> Some pth)
  in
  match List.find_exn paths ~f:(fun pth -> List.length pth = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let mma_run ~name ~out ~tensorize comp =
  let transform (opt : LL.optimized) =
    if not tensorize then opt
    else
      let i, j, k = mma_nest_syms opt in
      let ez, zsyms = Sched.expand_zero ~tn:out in
      let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
      let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:mma_n () in
      Sched.apply [ ez; Sched.Retype { axis = zj; ty = LL.Workgroup }; tz ] opt
  in
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun o -> [ transform o ])
      (Lazy.force base_ctx) comp Idx.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx out, routine.Context.mma, routine.Context.peel)

let () =
  List.iter precs ~f:(fun (prec_name, prec) ->
      let form_claim =
        Printf.sprintf "tile-mma-fallback @ %s renders the form its composition claims" prec_name
      in
      let value_claim =
        Printf.sprintf "tile-mma-fallback @ %s agrees with its reference value" prec_name
      in
      let peel_claim =
        Printf.sprintf "tile-mma-fallback @ %s: codegen's peel decided what the leg declares"
          prec_name
      in
      if not on_cpu then begin
        skipped form_claim;
        skipped peel_claim;
        skipped value_claim
      end
      else begin
        let plain = mma_matmul ~tag:("rfmma_p_" ^ prec_name) ~prec in
        let want, _, _ =
          mma_run ~name:("rf_mma_plain_" ^ prec_name) ~out:plain.Tensor.value ~tensorize:false
            (Train.forward plain)
        in
        let tiled = mma_matmul ~tag:("rfmma_t_" ^ prec_name) ~prec in
        let got, census, peel =
          mma_run ~name:("rf_mma_fallback_" ^ prec_name) ~out:tiled.Tensor.value ~tensorize:true
            (Train.forward tiled)
        in
        Stdio.eprintf "  rf_mma_fallback_%s peel census: %s\n%!" prec_name
          (Cs.peel_summary_string peel);
        (* The census says every [Tile_mma] statement declined; it says NOTHING about how the
           nested fallback reduction then rendered (Codex P2, round 3). If localization inside the
           fallback regressed to per-step read-modify-writes the census would be unchanged, and
           these small exactly-representable products give the same value either way — so the
           fallback's own accumulator is read with the same classifier the row-sum members use.

           Three node touches rather than two: the einsum lowering emits a whole-node zero-init
           before the contraction, which the hand-built members do not have (they seed their
           accumulator from the host instead). It writes the cell once and mentions no local, so it
           cannot be confused with a closing store. *)
        (* [%op mc = ...] labels the node ["*._mc"] — the operator and the binding name — while
           codegen names it [mc]. Take the trailing identifier of the label, which is the part the
           code name is derived from. *)
        let label =
          let l = Tn.label tiled.Tensor.value in
          let b = ref (String.length l) in
          while !b > 0 && is_ident_char l.[!b - 1] do
            Int.decr b
          done;
          String.lstrip ~drop:(Char.equal '_') (String.drop_prefix l !b)
        in
        let localized =
          match read_form (Generated.read ("rf_mma_fallback_" ^ prec_name)) ~label with
          | None ->
              Stdio.eprintf "  rf_mma_fallback_%s: no accumulator identifier in the kernel\n"
                prec_name;
              false
          | Some reading ->
              let ok =
                same_form (form_of reading) Localized
                && reading.stores_from_local = 1 && reading.node_accesses <= 3
                (* And nothing closed elsewhere on the way, as every other localized form in the
                   table now requires (Codex P2, round 13). A fallback that began closing
                   intermediate partials into another node would keep its single [mc] store, its
                   Scalar_fallback census and this leg's exact f32 comparison. *)
                && reading.foreign_local_stores = 0
                && reading.foreign_accesses = 0
              in
              if not ok then
                Stdio.eprintf
                  "  rf_mma_fallback_%s: fallback rendered %s (%d subscripts, %d rmw, %d local \
                   stores)\n"
                  prec_name
                  (form_name (form_of reading))
                  reading.node_accesses reading.rmw_statements reading.stores_from_local;
              ok
        in
        p form_claim
          (Cs.equal_tensorization census.Cs.tensorization Cs.Scalar_fallback
          && census.Cs.statements > 0
          && census.Cs.scalar_fallbacks = census.Cs.statements
          && localized);
        (* And, as for every table member, WHICH decision localized that fallback reduction
           (gh-ocannl-733): codegen's peel, at the contraction's own reduction level. The textual
           reading above says a scope is there; only the census says the peel put it there. *)
        check_peel ~name:("rf_mma_fallback_" ^ prec_name) ~claim:peel_claim peel mma_fallback_peel;
        let ok = agrees got want in
        if not ok then
          Stdio.eprintf "  rf_mma_fallback_%s: got [%s] want [%s]\n" prec_name (show got)
            (show want);
        p value_claim ok
      end)

(* {1 The reading is backend-independent}

   Everything above reads whichever kernel the run's backend emitted, so a spelling only one backend
   produces is only ever exercised where that backend runs — and Metal is off the per-PR path. When
   Metal's accumulation workaround moved from a volatile pointer DECLARATION to expression-level
   casts on the read, the table's counter stopped seeing the read half of a read-modify-write and
   every Metal decline leg went [Unrecognized], on a machine nobody was watching.

   So the normalizer is checked here directly, on both spellings of one statement, on every backend:
   the reading a kernel gets must not depend on how its backend spells a device read. Synthetic text
   rather than a compiled kernel, deliberately — the point is to hold on the backends that cannot
   produce the second spelling at all. *)
let () =
  let decl = "  device float* __restrict rfout = (device float*)(__pools[0]);\n" in
  let plain = "  rfout[i] = (rfout[i] + rfxs[(i) * 64 + k]);\n" in
  let cast =
    "  rfout[i] = (((device volatile float*)rfout)[i] + ((device volatile float*)rfxs)[(i) * 64 + \
     k]);\n"
  in
  let read src = read_form (decl ^ src) ~label:"rfout" in
  match (read plain, read cast) with
  | Some plain_reading, Some cast_reading ->
      p "a read-modify-write reads as one, in either spelling of its device read"
        (same_form (form_of plain_reading) Rmw && same_form (form_of cast_reading) Rmw);
      p "the volatile-cast spelling counts the same node accesses as the plain one"
        (cast_reading.node_accesses = plain_reading.node_accesses
        && cast_reading.rmw_statements = plain_reading.rmw_statements
        && cast_reading.stores_from_local = plain_reading.stores_from_local
        && cast_reading.foreign_local_stores = plain_reading.foreign_local_stores)
  | _ ->
      Stdio.eprintf "  normalizer check: no accumulator identifier in the synthetic kernel\n";
      p "a read-modify-write reads as one, in either spelling of its device read" false;
      p "the volatile-cast spelling counts the same node accesses as the plain one" false
