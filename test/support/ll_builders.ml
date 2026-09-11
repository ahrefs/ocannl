(** Pure IR builders shared by both packages (gh-ocannl-954). This library depends only on
    [arrayjit.ir] and [base]; execution and optimization helpers stay in [Ll_test]. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing

let single = Ops.single

(** {1 Tensor nodes} *)

(** [node_factory ~first_id ~dims ()] returns a maker of fresh single-precision tensor nodes with
    consecutive ids above [first_id] and default dimensions [dims] (overridable per node). Each test
    executable picks an id range of its own, so nodes stay distinguishable in debug output. *)
let node_factory ?(prec = single) ~first_id ~dims () =
  let next_id = ref first_id in
  fun ?(dims = dims) label ->
    Int.incr next_id;
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

(** Declares [tn] materialized and observable: the executed legs seed and read back exactly these
    nodes, and observability is also what forbids the buffer-aliasing planner from handing their
    bytes to another node. Both are declared intent, settled before optimization, so neither
    perturbs a structural pin — see {!virtualize} for what "declared intent" reaches. *)
let materialize tn =
  Tn.update_memory_mode tn Tn.On_device 99;
  Tn.set_observable tn

(** Declares [tn] virtual — the standing of the scope-local scalars a virtualizer-emitted
    [Local_scope] owns.

    This and {!materialize} write the tnode's DECLARED INTENT ([Tn.update_memory_mode], the
    [memory_mode_intent] field), not a lineage decision. Placement decisions live on the
    [optimize_ctx]'s placements table, and [Tn.Placements.get] falls back to the declared intent for
    a node the lineage has not decided — which is the whole reason a test can hand [optimize] a node
    that is ALREADY virtual (or already materialized) before the analyses run, and the reason the
    passes read it back as such. *)
let virtualize tn = Tn.update_memory_mode tn Tn.Virtual 99

(** {1 Index and statement builders} *)

let sym () = Idx.get_symbol ()
let iter s : Idx.axis_index = Idx.Iterator s
let fixed n : Idx.axis_index = Idx.Fixed_idx n

(** [aff terms offset] is the affine index [sum (coeff * symbol) + offset]. *)
let aff terms offset : Idx.axis_index = Idx.Affine { symbols = terms; offset }

let set ?(debug = "") tn idcs llsc : LL.t = LL.Set { tn; idcs; llsc; debug }
let get tn idcs : LL.scalar_t = LL.Get (tn, idcs)
let zero tn : LL.t = LL.Zero_out tn
let seq a b : LL.t = LL.Seq (a, b)

(** [if_ cond body] guards [body] on [cond] being nonzero ({!Ir.Low_level.If}). The condition is
    read at index precision only when it is an index expression; a value read (the usual flag
    tensor) keeps the node's precision, which is what {!single} is here. *)
let if_ cond body : LL.t = LL.If { cond = (cond, single); body }

(** [loop ~upto s body] iterates [s] over [0 .. upto] INCLUSIVE, mirroring
    {!Ir.Low_level.For_loop}'s own bounds; [upto < 0] is a dead loop, which is a case worth
    building. [~axis] declares the loop's hardware axis ({!Ir.Low_level.Serial} by default): the
    tests that judge a binding — a [Grid] block loop, a [Workgroup] lane, a [Workgroup_reduce]
    accumulation — name it here rather than spelling the record. *)
let loop ?(from_ = 0) ?(axis = LL.Serial) ~upto s body : LL.t =
  LL.For_loop { index = s; from_; to_ = upto; body; axis }

(** [loop_n s n body] iterates [s] over a range of WIDTH [n], i.e. [0 .. n-1]. *)
let loop_n ?axis s n body : LL.t = loop ?axis ~upto:(n - 1) s body

(** [set_at tn idx llsc] writes the single-axis cell [idx] — [set] over a one-element index array,
    which is the shape of every hand-built one-dimensional case. *)
let set_at tn idx llsc : LL.t = set tn [| idx |] llsc

(** {2 Scan loops}

    {!Ir.Low_level.Scan_loop} (gh-ocannl-696): a loop with declared loop-carried scalar state. A
    carried scalar is a pair of scope ids over one VIRTUAL node -- the state's name and precision,
    never a buffer -- read as [prev] and written as [next] inside the body, rotated [prev := next]
    after every iteration. The builders below mint the pair from a node the test declares
    {!virtualize}d, so a case cannot spell the two ids over different nodes or forget the
    declaration. *)

(** [carry ~init tn] is one carried scalar over the state node [tn], starting at [init] (a scalar
    that may read tensor nodes but no carried state). *)
let carry ~init tn : LL.carried = { prev = LL.get_scope tn; next = LL.get_scope tn; init }

(** [prev cr] reads the carried scalar's value from the previous iteration (its [init] on the
    first). *)
let prev (cr : LL.carried) : LL.scalar_t = LL.Get_local cr.prev

(** [next cr] reads the value the CURRENT iteration already assigned with {!set_next}: the rotation
    is phi-style, so old and new values coexist inside one body. *)
let next (cr : LL.carried) : LL.scalar_t = LL.Get_local cr.next

(** [set_next cr v] assigns the carried scalar's next value -- exactly once per carried scalar, as a
    top-level statement of the body, which is the contract {!Ir.Low_level.validate_scan_loops}
    enforces. *)
let set_next (cr : LL.carried) v : LL.t = LL.Set_local (cr.next, v)

(** [scan ~upto s ~carried body] iterates [s] over [from_ .. upto] INCLUSIVE like {!loop}, carrying
    [carried] across iterations; [~direction:Backward] counts down instead. *)
let scan ?(from_ = 0) ?(direction = LL.Forward) ~upto s ~carried body : LL.t =
  LL.Scan_loop { index = s; from_; to_ = upto; direction; carried; body }

(** {2 Dynamic indexing}

    The gather/scatter pair ({!Ir.Low_level.Get_dynamic} / {!Ir.Low_level.Set_dynamic}): a read or
    write whose row along ONE axis is a runtime value rather than an index expression. The ordinary
    pipeline never hands these to [optimize] — [Assignments] lowering emits neither, and the ones
    the pipeline does mint come from [rewrite_one_hot_reductions], which runs after both
    virtualization arms — so hand-built IR is the only way to put one in front of the analyses
    (gh-ocannl-734).

    Their [idcs] array is static everywhere except [dyn_axis], where the type's contract asks for a
    [Fixed_idx 0] placeholder standing in for the runtime row. The builders below PLANT that
    placeholder themselves: pass the static indices of the other axes at full array width (whatever
    sits at [dyn_axis] is overwritten) and a slot cannot be spelled wrong, nor a [dyn_axis] pointed
    outside the array. *)

(* Whatever the caller put at [dyn_axis] is replaced by the placeholder: the runtime row is
   [dyn_value]'s to supply, and a leftover index expression there would be silently ignored by
   codegen while still being reported as a read by the index-walking analyses. *)
let dyn_idcs ~idcs ~dyn_axis =
  if dyn_axis < 0 || dyn_axis >= Array.length idcs then
    invalid_arg
      ("Ll_test: dyn_axis " ^ Int.to_string dyn_axis ^ " is outside the "
      ^ Int.to_string (Array.length idcs)
      ^ " index slots of the node");
  Array.mapi idcs ~f:(fun i idx -> if i = dyn_axis then fixed 0 else idx)

(** [gather ~tn ~idcs ~dyn_axis ~dyn_value] reads [tn] at [idcs] with the [dyn_axis] row taken from
    the runtime value [dyn_value] (an index-valued scalar paired with the precision it is read at —
    [iprec] for an index computation, the node's own precision for a row number stored in a tensor).
    Counts as a read of [tn], like {!get}. *)
let gather ~tn ~idcs ~dyn_axis ~(dyn_value : LL.scalar_arg) : LL.scalar_t =
  LL.Get_dynamic { tn; idcs = dyn_idcs ~idcs ~dyn_axis; dyn_axis; dyn_value }

(** [scatter ~tn ~idcs ~dyn_axis ~dyn_value llsc] writes [llsc] into that same cell: {!set} with the
    [dyn_axis] row supplied at runtime. Loops whose index reaches [dyn_value] carry a
    cross-iteration write dependency, so schedule analyses must treat the write as statically
    unknown — which is much of what makes this shape worth building by hand. *)
let scatter ~tn ~idcs ~dyn_axis ~(dyn_value : LL.scalar_arg) llsc : LL.t =
  LL.Set_dynamic { tn; idcs = dyn_idcs ~idcs ~dyn_axis; dyn_axis; dyn_value; llsc; debug = "" }

(** [scatter_add ~tn ~idcs ~dyn_axis ~dyn_value addend] is the accumulating form
    [tn[.., dyn_value, ..] += addend] — the shape [rewrite_one_hot_reductions] actually mints for
    the embedding-table gradient. The read-back is an explicit {!gather} of the written cell at the
    node's storage precision, which is what makes the accumulation visible to read-tracking and to
    [has_accumulation]; [addend] carries its OWN precision, as the matched gradient argument does
    there — a mixed-precision accumulation (an [f32] gradient into a [bf16] table) is a shape worth
    building, and relabelling the addend with the target's precision would build different IR from
    the one the pipeline mints. *)
let scatter_add ~tn ~idcs ~dyn_axis ~(dyn_value : LL.scalar_arg) (addend : LL.scalar_arg) : LL.t =
  let value_prec = Lazy.force tn.Tn.storage_prec in
  scatter ~tn ~idcs ~dyn_axis ~dyn_value
    (LL.Binop (Ops.Add, (gather ~tn ~idcs ~dyn_axis ~dyn_value, value_prec), addend))

(** {2 Cooperative tile multiply-accumulate}

    {!Ir.Low_level.Tile_mma}: [d[i,j] += Σ_{l<k} a[i,l] * b[l,j]] over a block of the declared
    extents, executed jointly by the threads of a [Workgroup] lane axis (tensor cores /
    [simdgroup_matrix] / the register-tiled CPU GEBP kernel), carrying a scalar micro-kernel
    [fallback] the renderer falls back to when it declines the block. Hand-built IR is the only way
    to put one in front of a pass: schedule transforms mint [Tile_mma] AFTER the optimization
    pipeline, and [Ir.Low_level.optimize] rejects one outright — so a test that wants a tile in a
    routine builds the scalar twin, optimizes THAT, and substitutes the tile into the result.

    [ldd]/[lda]/[ldb] default to the declared extents read as a contiguous row-major block —
    [ldd = n], [lda = if ta then m else k], [ldb = if tb then k else n] — a purely syntactic default
    off the tile's own geometry, NOT a read of the operands' dimensions: an operand whose tile is a
    window into a wider array, or whose tile major axis sits outside its minor two (a batched site,
    gh-ocannl-528), passes its stride explicitly. [lane] defaults to a fresh symbol and [tile] to
    [None], the renderer's own choice of C-tile geometry (gh-ocannl-619). *)
let tile_mma ?(ta = false) ?(tb = false) ?(m = 2) ?(n = 2) ?(k = 2) ?ldd ?lda ?ldb ?tile ?lane
    ~(d : Tn.t * Idx.axis_index array) ~(a : Tn.t * Idx.axis_index array)
    ~(b : Tn.t * Idx.axis_index array) (fallback : LL.t) : LL.t =
  LL.Tile_mma
    {
      d;
      a;
      b;
      ta;
      tb;
      m;
      n;
      k;
      ldd = Option.value ldd ~default:n;
      lda = Option.value lda ~default:(if ta then m else k);
      ldb = Option.value ldb ~default:(if tb then k else n);
      lane = (match lane with Some s -> s | None -> sym ());
      tile;
      fallback;
    }

(** {1 Scalar builders} *)

let c x : LL.scalar_t = LL.Constant x
let embed s : LL.scalar_t = LL.Embed_index (iter s)
let binop op a b : LL.scalar_t = LL.Binop (op, (a, single), (b, single))
let add a b = binop Ops.Add a b
let mul a b = binop Ops.Mul a b

(** {2 Index-precision scalars}

    The builders above are single-precision, which is what a value computation is. A GUARD is not: a
    comparison and its conjunctions are read at index precision, the same as an {!Ir.Low_level.If}'s
    condition and a [Where]'s selector, and building one at [single] misstates what the pass under
    test sees. Index precision is read at build time rather than captured once, because it is a
    configured setting. *)

let iprec () = Ops.index_prec ()

(** [embed_idx idx] embeds an arbitrary index expression, where {!embed} takes a symbol. *)
let embed_idx idx : LL.scalar_t = LL.Embed_index idx

(** [ic n] is the integer constant [n] as a scalar. *)
let ic n : LL.scalar_t = LL.Constant (Float.of_int n)

(** [cmp op a b] applies an index-precision binary operator — a comparison ([Cmplt], [Cmple],
    [Cmpeq], [Cmpne]) or a connective ([And], [Or]). *)
let cmp op a b : LL.scalar_t = LL.Binop (op, (a, iprec ()), (b, iprec ()))

let lt a b = cmp Ops.Cmplt a b
let le a b = cmp Ops.Cmple a b
let eq a b = cmp Ops.Cmpeq a b
let ne a b = cmp Ops.Cmpne a b
let conj a b = cmp Ops.And a b
let disj a b = cmp Ops.Or a b

(** [where_ cond then_ else_] is the [Where] ternop, its condition read at index precision and its
    arms at [single] — the shape a zero-fringe guard renders as. *)
let where_ cond then_ else_ : LL.scalar_t =
  LL.Ternop (Ops.Where, (cond, iprec ()), (then_, single), (else_, single))

(** [if_idx cond body] is {!if_} with the condition read at INDEX precision: the standing of a
    launch-extent or fringe guard, whose condition is an index expression rather than a value read.
*)
let if_idx cond body : LL.t = LL.If { cond = (cond, iprec ()); body }
