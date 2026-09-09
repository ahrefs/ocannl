(** Hand-built {!Ir.Low_level} test harness: builders, exhaustive traversals, and the executed-leg
    machinery, shared by the tests that construct IR the [Assignments] pipeline never emits
    (gh-ocannl-600).

    Why hand-built IR at all: high-level lowering gives each assignment its own loop nest, so shapes
    like two tensors sharing one for-loop, a [Get] of a diagonal at two distinct call-site symbols,
    or sibling [Local_scope]s reading a node a later statement overwrites are unreachable through
    [Assignments] — yet they are exactly the shapes the virtualizer's guards exist for. Such a case
    is built as an {!Ir.Low_level.t}, run through {!optimize} (the same
    [analyze_proc]/[specialize_proc] pipeline the backends use), asserted on structurally, and then
    EXECUTED through the [?prelowered] seam (gh-ocannl-562) — because virtualization rewrites what
    value a cell holds, which no structural pin can reach.

    Four doctrines are built into the helpers here, each learned from a leg that passed for the
    wrong reason:

    - The oracle must DISCRIMINATE. A producer value has to vary with every symbol of its iteration
      and stay off the init value, or a too-wide range guard replays an identical assignment, a
      wrong substitution stays invisible along the axis whose symbol the value omits, and a dropped
      first iteration hides in the zero-init. Hence {!tick} / {!tag} / {!ramp} rather than
      constants, and {!drift} where what has to discriminate is numeric (a narrow storage precision
      the running sums must leave behind) rather than which iteration wrote the cell.
    - Cells no writer covers must carry a {!sentinel}, so "wrote the wrong cells" fails the value
      check instead of reading whatever the buffer happened to hold.
    - A claim must be able to FAIL. Every {!p} is an assertion, not a recorded observation: a run
      with a false claim exits nonzero, so a regression cannot be [dune promote]d into the golden
      (gh-ocannl-601). Claims are therefore phrased so [true] is the passing reading.
    - A node to be read back must be declared {!materialize}d: [known_non_virtual] does NOT mean
      "has a context buffer" — a node written and read within one routine and never observed is
      placed [Local], routine-scoped scratch, and host access to it raises (gh-ocannl-599).

    This library links [ocannl], which is why it lives beside {!Test_utils} rather than inside it:
    [test_utils] deliberately depends on [arrayjit.ir] alone, for tests that link no more than that.
*)

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

let set tn idcs llsc : LL.t = LL.Set { tn; idcs; llsc; debug = "" }
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
    {!iprec} for an index computation, the node's own precision for a row number stored in a
    tensor). Counts as a read of [tn], like {!get}. *)
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

(** {1 Discriminating producer values}

    A producer that writes a constant makes an executed leg blind to WHICH iteration supplied a cell
    — and that is precisely what the guards under test decide. These three write values that
    identify the iteration and stay clear of the zero-init. *)

(** [tick s] is [1 + s]: identifies a one-symbol producer's iteration, and the [1 +] keeps even the
    [s = 0] write distinct from the zero-init, so a dropped first iteration is visible. *)
let tick s = add (c 1.) (embed s)

(** [tag outer inner] is [1 + 10*outer + inner]: identifies both symbols of a two-symbol producer,
    so a substitution wrong on either axis shows up. *)
let tag outer inner = add (c 1.) (add (mul (c 10.) (embed outer)) (embed inner))

(** [ramp base s] is [base + s]: a per-cell value for a sibling provider, distinguished from other
    providers by [base]. [Embed_index] is not an array access, so this does not make the provider
    [is_complex]. *)
let ramp base s = add (c base) (embed s)

(** {1 Discriminating storage values}

    The counterpart of {!tick} / {!tag} / {!ramp} for tests that seed real tensor nodes rather than
    build producers: host-side [~f] initializers for [NTDSL.init], indexed by the cell's
    multi-index.

    Narrow-storage accumulator tests (gh-ocannl-639) need a sharper property than "varies with every
    symbol": the cells must be EXACT in the storage precision while their running sums are not, or
    the leg passes for the wrong reason. A zero-mean operand random-walks small enough that every
    bf16 partial sum stays bf16-exact, so per-step narrowing is invisible and a schedule-dependent
    accumulator width reads as parity — the zero-mean trap that
    docs/agent-notes/backend-precision-and-simd.md's gh-ocannl-639 entry records (trap 2), found the
    hard way by this test's first draft. Hence a cycle whose cells are exact and whose running sums
    DRIFT out of the storage format's exactness range.

    Both halves of that are arithmetic, not rules of thumb, and {!cycle} states each as a condition
    a caller has to check rather than assume. *)

(** [flat ~dims idcs] is the row-major flat index of [idcs] in a [dims]-shaped array. The leading
    dimension does not enter, so [dims.(0)] may be any extent the caller finds convenient. *)
let flat ~dims idcs =
  Array.foldi idcs ~init:0 ~f:(fun ax acc i -> if ax = 0 then i else (acc * dims.(ax)) + i)

(** [blind_axis ~dims ~modulus] is the outermost axis whose index {!cycle} would ignore, if any:
    stepping along axis [ax] moves {!flat} by that axis's row-major stride, so the cycle is constant
    along [ax] exactly when [modulus] divides that stride. *)
let blind_axis ~dims ~modulus =
  let found = ref None and stride = ref 1 in
  for ax = Array.length dims - 1 downto 0 do
    if !stride % modulus = 0 then found := Some (ax, !stride);
    stride := !stride * dims.(ax)
  done;
  !found

(** [cycle ~dims ~modulus ~offset ~stride idcs] is [(flat idcs mod modulus + offset) * stride]: the
    values [k * stride] for [k] cycling through [offset .. offset + modulus - 1] with period
    [modulus]. Reusing it means checking two conditions, neither of which the obvious phrasings
    imply:

    - {b It must vary with every index.} Coprimality with the reduction EXTENT is not the condition
      — [dims = [|2; 4; 3|]] with [modulus = 3] has row-major strides [12; 3; 1], so despite 3 and 4
      being coprime the value depends on the innermost index alone and a wrong substitution on
      either other axis stays invisible. The condition is on the STRIDES: no axis's stride may be a
      multiple of [modulus] ({!blind_axis}, which this function raises on — a [modulus] coprime to
      every [dims.(1 ..)] satisfies it, since the strides are their products).
    - {b The partial sums must actually leave exactness.} Count in units of [stride], which makes
      every cell and every partial sum an integer [k]: with [stride] a negative power of two, a
      format with [p] significand bits holds [k * stride] exactly for every [|k| <= 2^p], and above
      that only for the [k] whose trailing zeros make up the difference. So cells are exact when the
      whole range [offset .. offset + modulus - 1] fits within [2^p], and a reduction of [n] terms
      leaves exactness only once its running [k] clears [2^p] (and lands off the sparser multiples
      still representable above it) — a statement about the extent, not just about the mean being
      nonzero. A test relying on inexact partials has to EXHIBIT that crossing rather than infer it;
      [test/operations/discriminating_values] does so for {!drift}, and a nonzero mean over too few
      terms is the zero-mean trap wearing a different hat. *)

(** [cycle_flat ~dims ~modulus ~offset ~stride i] is {!cycle} over an already-FLATTENED offset — the
    form an [Array.init (rows * cols) ~f:…] operand is written in, which is most of them.

    The flat spelling is where this goes wrong in the field (gh-ocannl-640), and it goes wrong
    invisibly: [Array.init (m * k) ~f:(fun i -> Float.of_int (i % 13) *. 0.25)] LOOKS like it varies
    with both indices, and stops doing so at exactly the sizes where the modulus divides the row
    stride — the value collapses to [col mod p] and every row becomes identical. An operand constant
    along a row makes a whole class of bugs invisible: a transform that substitutes or repeats the
    wrong row, panel or K-block computes the correct output, so no whole-output check, checksum
    included, can see it. It was found four times across three review rounds of one PR.

    Passing the real [~dims] is what buys the guard, and the guard is the reason to convert a site:
    the arithmetic is identical to the idiom it replaces, so no golden moves, and the day someone
    widens a size onto a multiple of the modulus the run raises instead of quietly passing.

    What this does NOT give is aperiodicity. The values repeat with period [modulus] in the flat
    offset, so a shift by [modulus] is a symmetry, and where the BLOCKING factors are searchable a
    packed panel can repeat under [k -> k + p] and hide a panel-substitution bug just as thoroughly.
    That needs a mixer with no shift symmetry at any lag; [bin/narrow_gebp_bench.ml]'s [mix] is the
    worked recipe, measured over lags 1..256 and over every block width from 1 to 64. *)
let cycle_flat ~dims ~modulus ~offset ~stride i =
  (match blind_axis ~dims ~modulus with
  | Some (ax, s) ->
      raise
        (Invalid_argument
           (Printf.sprintf
              "Ll_test.cycle: modulus %d is blind to axis %d of %s (row-major stride %d is a \
               multiple of it), so the value would not vary with that index"
              modulus ax
              (Sexp.to_string (Array.sexp_of_t Int.sexp_of_t dims))
              s))
  | None -> ());
  (Float.of_int (i % modulus) +. offset) *. stride

let cycle ~dims ~modulus ~offset ~stride idcs =
  cycle_flat ~dims ~modulus ~offset ~stride (flat ~dims idcs)

(** [drift ~dims idcs] is {!cycle} at [13/20/(1/64)], the accumulator-width tests' operand: cells
    are the multiples of 1/64 between 0.3125 and 0.5, i.e. [k * (1/64)] for [k] in [20 .. 32], with
    mean [k = 26]. In {!cycle}'s units: cells are exact in bf16 ([p = 8], [32 < 256]) and stay exact
    in f32 ([p = 24]) along with every partial sum a test of this size can build, which is what lets
    an f64 host-side reference reproduce the widened kernel bitwise; while the running sum passes
    [2^8 = 256] units — the value 4, bf16's last guaranteed-exact multiple of 1/64 — at the eleventh
    term, landing on the bf16-unrepresentable [275/64], so a reduction longer than that visibly
    diverges if the accumulator narrows per step instead of once at the store (the legs using this
    reduce 16 and 36 terms). 13 is prime and divides none of those shapes' strides, so every index
    discriminates — and {!cycle} raises rather than let a later [~dims] silently break that.
    [test/operations/discriminating_values] pins each of those numbers. *)
let drift ~dims = cycle ~dims ~modulus:13 ~offset:20. ~stride:0.015625

(** {1 Optimization} *)

(** [optimize_in ctx ~name llc] runs the backends' own pipeline ([analyze_proc] ->
    [specialize_proc]: structural facts -> placements -> [virtual_llc] -> cleanup -> simplify -> CSE
    -> hoist) over hand-built code, retaining its decisions and stored computations in [ctx] for a
    follow-on routine. This is the cross-routine counterpart of {!optimize}: use it when the case
    under test is a lineage transition rather than one isolated routine.

    [~materialized] pre-decides those nodes' placement in the lineage state the optimization reads,
    which is what {!Context.decide_materialized} does for the [Assignments] pipeline — and the only
    way to do it for this path, since the [?prelowered] seam replaces the context's lineage with the
    optimized record's own [optimize_ctx], so a decision recorded on a context never reaches the
    optimization. Re-specializing the SAME [LL.t] with a virtualization candidate materialized gives
    a case its differential arm: the inlined and materialized readings of one program must agree
    cell for cell, which is what pins a guard.

    [~static_indices] are the launch parameters the code may mention outside every loop: the bound
    symbols of the [Indexing.bindings] the routine will be compiled with. Hand-built IR carrying a
    gh-490 runtime-extent guard ([If (i < s)], as [Assignments.extent_guard] emits) needs them — the
    virtualization walk asserts that every [Embed_index (Iterator s)] it meets is in scope, and a
    launch parameter is in scope only because the caller declared it. Empty by default, which is
    right for every nest whose indices are all loop indices. *)
let optimize_in ?(materialized = []) ?(static_indices = []) (ctx : LL.optimize_ctx) ~name llc :
    LL.optimized =
  LL.decide_materialized ~provenance:589 ctx materialized;
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name static_indices llc

(** [optimize ~name llc] is {!optimize_in} in a fresh lineage. *)
let optimize ?materialized ?static_indices ~name llc =
  optimize_in ?materialized ?static_indices (LL.empty_optimize_ctx ()) ~name llc

(** [optimize_scoped ~name ~raw scoped] is the supported route for hand-built IR in the
    POST-optimize [Local_scope] form — a scope over a MATERIALIZED node, which
    {!Ir.Low_level.optimize} rejects (gh-ocannl-681) and which only [Schedule]'s materializing
    [Unroll] / [Partition] mints and [C_syntax.try_localize_serial_reduce] produce. [raw] is a twin
    spelling the same nodes, reads and writes WITHOUT the scope: optimizing it builds the traced
    store and the placements the record needs, and [scoped] then replaces the schedule wholesale,
    reaching the backend through the [?prelowered] seam.

    Optimizing [scoped] itself is what used to collapse it into an identity copy of the accumulator
    — green for the wrong reason, which is how two [accum_width.ml] legs passed while executing
    [acc[0] = acc[0]]. This seam is what gh-ocannl-693's codegen accumulator peel will replace as
    the one localization route. *)
let optimize_scoped ?materialized ?static_indices ~name ~raw scoped : LL.optimized =
  let o = optimize ?materialized ?static_indices ~name raw in
  { o with LL.llc = scoped }

(** Post-optimization placement probes. Decisions live on the [optimize_ctx]'s placements
    (context-scoped memory modes), not on the tnode, which holds only declared intent. *)
let known_virtual (o : LL.optimized) tn =
  Tn.Placements.known_virtual o.LL.optimize_ctx.placements tn

let known_non_virtual (o : LL.optimized) tn =
  Tn.Placements.known_non_virtual o.LL.optimize_ctx.placements tn

(** Whether backend compilation finalized [tn] to routine-scoped [Local] scratch. Unlike
    {!Ir.Tnode.Placements.known_not_materialized}, this excludes [Virtual], so a post-finalization
    test cannot accidentally pin the wrong arm of a guard shared by both placements. *)
let known_local (o : LL.optimized) tn =
  match Tn.Placements.raw_entry o.LL.optimize_ctx.placements tn with
  | Some (Local, _) -> true
  | _ -> false

(** The [Non_virtual] code the virtualizer recorded for [tn], as the leading factor of its
    placement's provenance.

    Provenances COMPOSE: {!Ir.Tnode.Placements.default_to_most_local} folds the prior provenance in
    as [1000 * prior + its own] when it resolves a [Never_virtual] decision into a concrete
    placement, so the rejection code is not the whole number. Stripping the trailing factors is
    sound only while every code is below 1000, which they are — {!Ir.Low_level}'s two [Non_virtual]
    exceptions use disjoint two- and three-digit codes, which is also what lets a reader tell the
    store-time verdicts from the consumption-time ones. [None] means no decision was recorded (the
    node is still undecided, which after a full {!optimize} means it was never a candidate). *)
let rejection_code (o : LL.optimized) tn =
  Option.map (Tn.Placements.get o.LL.optimize_ctx.placements tn) ~f:(fun (_, prov) ->
      let rec strip p = if p >= 1000 then strip (p / 1000) else p in
      strip prov)

(** {1 The executed leg} *)

(* One root context per executable: [Context.compile] forks the lineage for each compile, so sibling
   executions do not observe each other's decisions. *)
let base_ctx = lazy (Context.auto ())

(** [link ~name o] compiles the optimized record AS WRITTEN through the [?prelowered] seam,
    returning the advanced context and the compiled routine — so a test can assert on the
    [inputs]/[outputs] the link actually computed, read straight off the routine record, instead of
    re-deriving them via [input_and_output_nodes] (gh-ocannl-590).

    The identity [lowered_transform] takes the place of the default schedule annotator, which would
    otherwise parallelize or fission the hand-built loop nest — the point of the case is usually the
    nest's exact shape.

    [~bindings] are the routine's launch parameters, for code that mentions a symbol no loop binds:
    a gh-490 symbolic extent, a batch-slice index. Pass the same [unit_bindings] whose bound symbols
    were declared as {!optimize}'s [~static_indices], and set each parameter's value through
    [Context.bindings] on the RETURNED routine — [Idx.find_exn routine.Context.bindings sym := v] —
    before {!run_linked}; the unit bindings themselves carry no cell to write. ({!run} and
    {!execute}, which return no routine, take those values as [~launch] instead.) Empty by default,
    which is right for every nest whose indices are all loop indices. *)
let link ?ctx ?(bindings = Idx.Empty) ~name (o : LL.optimized) =
  let ctx = match ctx with Some ctx -> ctx | None -> Lazy.force base_ctx in
  Context.compile ~name ~prelowered:o
    ~lowered_transform:(fun x -> [ x ])
    ctx Ir.Assignments.empty_comp bindings

(** [link_finalized ~placements ~name o] links through the real backend, then checks that every
    requested node has left the undecided placement classes. Backend code generation is the
    supported finalization step: it resolves a retained [Never_virtual] computation to [Local] (or
    [On_device]) in [o]'s shared lineage, making that settled decision visible to a later
    {!optimize_in} call. The returned pair can be passed to {!run_linked} when the producer itself
    also needs an executed leg.

    This deliberately does not seed the placements table by hand: overwriting a committed entry
    would bypass the provenance/conflict checks whose behavior the lineage test is meant to cover.
*)
let link_finalized ?ctx ~placements ~name (o : LL.optimized) =
  let linked = link ?ctx ~name o in
  List.iter placements ~f:(fun tn ->
      if Tn.Placements.mode_is_unspecified o.LL.optimize_ctx.placements tn then
        raise
          (Invalid_argument
             (Printf.sprintf "Ll_test.link_finalized: backend compile left %s undecided"
                (Tn.debug_name tn))));
  linked

(** [run_linked (ctx, routine) ~seed] drives the executed leg of an ALREADY-LINKED pair: uploads
    [seed], runs, and returns the context the values can be read from. It is the half of {!run} that
    survives keeping the routine — a test asserting on the [inputs]/[outputs] the link actually
    computed (gh-ocannl-590) holds the pair {!link} returned, so it needs the execution without
    compiling a second time. Every node in [seed] must have been {!materialize}d: host access to a
    node the pipeline placed [Local] raises (gh-ocannl-599). *)
let run_linked (ctx, routine) ~(seed : (Tn.t * float array) list) =
  let ctx = List.fold seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) in
  Context.run ctx routine

(** Refuses a [launch] list that does not give EXACTLY ONE value to EACH bound symbol of [bindings].
    That invariant, not a list of the ways to break it, is what the check tests: it counts, so no
    entry can be silently absorbed. Backend linking initializes every launch parameter to [ref 0],
    so an uncovered symbol would launch at a zero the test never chose and could not distinguish in
    the result from one it did; a repeated symbol would launch at whichever value the assignment
    loop wrote last; a stray entry is a value that reaches no launch at all. All three are the
    wrong-value class the executed legs exist to catch, so all three raise rather than default, and
    they raise BEFORE the routine is compiled: a misuse costs no codegen, and a test can pin the
    refusal for the price of the check. *)
let check_launch ~bindings ~launch =
  let bound = Idx.bound_symbols bindings in
  let ident (sym : Idx.static_symbol) = Idx.symbol_ident sym.Idx.static_symbol in
  let refuse what syms =
    if not (List.is_empty syms) then
      raise
        (Invalid_argument
           (Printf.sprintf "Ll_test: ~launch %s: %s" what
              (String.concat ~sep:", " (List.map syms ~f:ident))))
  in
  let values sym = List.count launch ~f:(fun (s, _) -> Idx.equal_static_symbol s sym) in
  refuse "does not give exactly one value to the launch parameter(s)"
    (List.filter bound ~f:(fun sym -> values sym <> 1));
  refuse "names symbol(s) the bindings do not bind"
    (List.filter_map launch ~f:(fun (sym, _) ->
         Option.some_if (not (List.mem bound sym ~equal:Idx.equal_static_symbol)) sym))

(** [run ~name o ~seed] is {!link} followed by {!run_linked}, for the tests that have no use for the
    routine itself. Same materialization requirement on [seed].

    [~launch] carries what a {!link} caller would write into the returned routine's
    [Context.bindings]: these wrappers never hand the routine back, so it is their only way to set a
    launch parameter, and {!check_launch} requires it to give each of [~bindings]' bound symbols
    exactly one value. A case that also needs [~static_indices] on the optimization, or the routine
    itself, calls {!link} directly. *)
let run ?ctx ?(bindings = Idx.Empty) ?(launch = []) ~name (o : LL.optimized) ~seed =
  check_launch ~bindings ~launch;
  let ctx, routine = link ?ctx ~bindings ~name o in
  List.iter launch ~f:(fun (sym, v) -> Idx.find_exn routine.Context.bindings sym := v);
  run_linked (ctx, routine) ~seed

(** {!run}, then read back [read] in order. Same materialization requirement, on both lists. *)
let execute ?ctx ?bindings ?launch ~name (o : LL.optimized) ~seed ~(read : Tn.t list) =
  let ctx = run ?ctx ?bindings ?launch ~name o ~seed in
  List.map read ~f:(Context.get_values ctx)

(** Whether [f] was refused because the node it touched is placed [Local] — routine-scoped scratch
    whose values never reach a context buffer (gh-ocannl-599). *)
let refused_as_local f =
  try
    ignore (f ());
    false
  with Utils.User_error msg -> String.is_substring msg ~substring:"placed Local"

(** [optimize_and_execute] is {!optimize} followed by {!execute} under the same name, returning both
    the optimized record (for structural probes) and the values read back. A case whose [~bindings]
    also have to be declared to the optimization as [~static_indices] calls {!optimize} and
    {!execute} separately: this helper's optimize half takes no [~static_indices]. *)
let optimize_and_execute ?ctx ?bindings ?launch ?materialized ~name llc ~seed ~read =
  let o = optimize ?materialized ~name llc in
  (o, execute ?ctx ?bindings ?launch ~name o ~seed ~read)

(** The value seeded into cells no writer is supposed to cover: distinct from every producer value
    the builders above generate, and from the zero-init. *)
let sentinel = -1.

(** [blank n] is [n] cells of {!sentinel}. *)
let blank n = Array.create ~len:n sentinel

(** Whether two value arrays match elementwise within [tol] (and have the same length). *)
let close ?(tol = 1e-5) values expected =
  Array.length values = Array.length expected
  && Array.for_alli values ~f:(fun i v -> Float.(abs (v -. expected.(i)) <= tol))

(** [same got expected] is {!close} over the list {!execute} returns. Raises if the lists differ in
    length — a mismatched [~read] is a test bug, not a failed assertion. *)
let same ?tol got expected = List.for_all2_exn got expected ~f:(close ?tol)

(** {1 Claims}

    A hand-built-IR test reaches the claim surface the same way every other test does, by
    [open Verdict.Claims]: [p] for a named boolean, the quantified forms for a claim over a
    collection the walk derived (over an empty one, [p] applied to a [for_all] reads as a pass while
    checking nothing — gh-ocannl-729), and [p_all2] for the executed-parity form, whose two
    readbacks a placement regression empties TOGETHER (gh-ocannl-746). This module re-exported six
    of those names when [Verdict] had no open-able surface; the copy could only go stale against the
    original, and it did (gh-ocannl-815). *)

(** {1 Structural probes}

    One exhaustive pair of traversals over [Low_level.t] / [scalar_t], from which every counter
    below is derived. Every new IR constructor is handled HERE, once — three hand-maintained copies
    of this pair is what gh-ocannl-600 retired. *)

(* One record of callbacks rather than eight labelled arguments: the traversal below threads them
   through every recursive call, and a record keeps adding a hook from being an edit to forty call
   sites. Defaults are the do-nothing hooks, so a caller names only what it cares about. *)
type hooks = {
  h_set : Tn.t -> unit;
  h_get : Tn.t -> unit;
  h_binop : Ops.binop -> unit;
  h_ternop : Ops.ternop -> unit;
  h_scope : LL.scope_id -> unit;
  h_if : LL.scalar_t -> unit;
  h_stmt : LL.t -> unit;  (** Every statement, in preorder — the generic hook. *)
  h_scalar : LL.scalar_t -> unit;  (** Every scalar expression, in preorder. *)
  h_exit : LL.t -> unit;
      (** Every statement again, on the way out. What makes an enclosing-context query (which loops
          a statement sits inside) derivable from this traversal instead of a second one. *)
  h_in_scopes : bool;
      (** Whether to descend from a scalar into the statement body of a [Local_scope]. The one
          scalar-to-statement edge in the IR, and the one a hand-written walk gets wrong: a query
          that wants "reachable through statement positions alone" (what [Schedule.find_loop] saw
          before gh-ocannl-668) sets this false, and every other query leaves it true. Deciding it
          here once is the point — six copies in one test file each decided it independently. *)
}

let ignore1 _ = ()

let no_hooks =
  {
    h_set = ignore1;
    h_get = ignore1;
    h_binop = ignore1;
    h_ternop = ignore1;
    h_scope = ignore1;
    h_if = ignore1;
    h_stmt = ignore1;
    h_scalar = ignore1;
    h_exit = ignore1;
    h_in_scopes = true;
  }

let rec walk_t h (llc : LL.t) =
  h.h_stmt llc;
  (match llc with
  | LL.Noop | LL.Declare_local _ | LL.Comment _ | LL.Staged_compilation _ | LL.Workgroup_barrier ->
      ()
  | LL.Tile_mma { fallback; _ } ->
      (* Through the fallback, which is a semantically equivalent scalar micro-kernel over the same
         tensor nodes: reporting the tile's own [d]/[a]/[b] as well would count every operand of a
         tensorized nest twice. Descending at all is what a walk that stopped at [Tile_mma] missed —
         the loops, guards and accesses of the fallback are statements like any others. *)
      walk_t h fallback
  | LL.Seq (a, b) ->
      walk_t h a;
      walk_t h b
  | LL.For_loop { body; _ } -> walk_t h body
  | LL.Scan_loop { carried; body; _ } ->
      List.iter carried ~f:(fun c -> walk_s h c.LL.init);
      walk_t h body
  | LL.Zero_out tn -> h.h_set tn
  | LL.Set { tn; llsc; _ } ->
      h.h_set tn;
      walk_s h llsc
  | LL.Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
      h.h_set tn;
      walk_s h v;
      walk_s h llsc
  | LL.Set_from_vec { tn; arg = s, _; _ } ->
      h.h_set tn;
      walk_s h s
  | LL.Set_local (_, s) -> walk_s h s
  | LL.If { cond = c, _; body } ->
      h.h_if c;
      walk_s h c;
      walk_t h body);
  h.h_exit llc

and walk_s h (s : LL.scalar_t) =
  h.h_scalar s;
  match s with
  | LL.Constant _ | LL.Constant_bits _ | LL.Get_local _ | LL.Embed_index _ | LL.Get_merge_buffer _
    ->
      ()
  | LL.Get (tn, _) -> h.h_get tn
  | LL.Get_dynamic { tn; dyn_value = v, _; _ } ->
      h.h_get tn;
      walk_s h v
  | LL.Local_scope { id; body; _ } ->
      h.h_scope id;
      if h.h_in_scopes then walk_t h body
  | LL.Ternop (op, (a, _), (b, _), (d, _)) ->
      h.h_ternop op;
      walk_s h a;
      walk_s h b;
      walk_s h d
  | LL.Binop (op, (a, _), (b, _)) ->
      h.h_binop op;
      walk_s h a;
      walk_s h b
  | LL.Unop (_, (a, _)) -> walk_s h a

let hooks_of ?(on_set = ignore1) ?(on_get = ignore1) ?(on_binop = ignore1) ?(on_ternop = ignore1)
    ?(on_scope = ignore1) ?(on_if = ignore1) ?(on_stmt = ignore1) ?(on_scalar = ignore1)
    ?(on_exit = ignore1) ?(in_scopes = true) () =
  {
    h_set = on_set;
    h_get = on_get;
    h_binop = on_binop;
    h_ternop = on_ternop;
    h_scope = on_scope;
    h_if = on_if;
    h_stmt = on_stmt;
    h_scalar = on_scalar;
    h_exit = on_exit;
    h_in_scopes = in_scopes;
  }

(** [walk] over a statement, with only the callbacks the caller cares about. *)
let walk ?on_set ?on_get ?on_binop ?on_ternop ?on_scope ?on_if ?on_stmt ?on_scalar ?on_exit
    ?in_scopes llc =
  walk_t
    (hooks_of ?on_set ?on_get ?on_binop ?on_ternop ?on_scope ?on_if ?on_stmt ?on_scalar ?on_exit
       ?in_scopes ())
    llc

(** [walk_scalar] over a scalar expression, likewise. *)
let walk_scalar ?on_set ?on_get ?on_binop ?on_ternop ?on_scope ?on_if ?on_stmt ?on_scalar ?on_exit
    ?in_scopes s =
  walk_s
    (hooks_of ?on_set ?on_get ?on_binop ?on_ternop ?on_scope ?on_if ?on_stmt ?on_scalar ?on_exit
       ?in_scopes ())
    s

let count f =
  let n = ref 0 in
  f (fun () -> Int.incr n);
  !n

(** How many setters (including [Zero_out]) of [tn] survive in the optimized form. *)
let count_set (o : LL.optimized) tn =
  count (fun bump -> walk o.LL.llc ~on_set:(fun t -> if Tn.equal t tn then bump ()))

(** How many array READS of [tn] survive — [0] is what "the producer was inlined" means. *)
let count_get (o : LL.optimized) tn =
  count (fun bump -> walk o.LL.llc ~on_get:(fun t -> if Tn.equal t tn then bump ()))

(** How many guarded statements ({!Ir.Low_level.If}) survive in the optimized form. A guard whose
    condition an interval proves is erased by [simplify_llc], so this counts the ones that still
    decide something at run time. *)
let count_if (o : LL.optimized) = count (fun bump -> walk o.LL.llc ~on_if:(fun _ -> bump ()))

(** How many [Where] ternops survive: the shape a virtualization equality/range guard renders as. *)
let count_where (o : LL.optimized) =
  count (fun bump -> walk o.LL.llc ~on_ternop:(function Ops.Where -> bump () | _ -> ()))

(** [(wheres, cmples, cmplts)] in the optimized form. A range guard emitted by unit-coefficient
    solving renders as [Where (And (Cmple _, Cmplt _), value, Get_local)] — one comparison shape per
    role, a non-strict lower bound and a strict upper bound — whereas a pure structural affine match
    introduces none of the three. *)
let count_guard_ops (o : LL.optimized) =
  let wh = ref 0 and le = ref 0 and lt = ref 0 in
  walk o.LL.llc
    ~on_ternop:(function Ops.Where -> Int.incr wh | _ -> ())
    ~on_binop:(function Ops.Cmple -> Int.incr le | Ops.Cmplt -> Int.incr lt | _ -> ());
  (!wh, !le, !lt)

(** How many [Local_scope]s a RAW statement carries, nested ones included — [0] after
    [hoist_cross_statement_cse] lifted a shared body out of its users. Takes a statement rather than
    an [optimized] record, because the passes it observes are run directly. *)
let count_scopes (llc : LL.t) = count (fun bump -> walk llc ~on_scope:(fun _ -> bump ()))

(** {2 Counting a raw statement}

    The counters above take an [optimized] record, which is what a virtualization test holds. A test
    of a pass called directly ([simplify_llc], [Schedule.apply]) holds a bare {!Ir.Low_level.t}, and
    counts whatever shape its case is about — so these take a predicate rather than naming one
    construct, and both walk the same traversal above. *)

(** How many statements satisfy [f]. *)
let count_stmt ?in_scopes ~f (llc : LL.t) =
  count (fun bump -> walk ?in_scopes llc ~on_stmt:(fun s -> if f s then bump ()))

(** How many scalar expressions satisfy [f], including those inside statement positions. *)
let count_scalar ?in_scopes ~f (llc : LL.t) =
  count (fun bump -> walk ?in_scopes llc ~on_scalar:(fun s -> if f s then bump ()))

(** [census llc] is [(ifs, wheres, loops)] of a raw statement: guarded statements, [Where] ternops,
    and [For_loop]s. The three a schedule transform's structural pin is usually about. *)
let census (llc : LL.t) =
  let ifs = ref 0 and wheres = ref 0 and loops = ref 0 in
  walk llc
    ~on_stmt:(function LL.If _ -> Int.incr ifs | LL.For_loop _ -> Int.incr loops | _ -> ())
    ~on_ternop:(function Ops.Where -> Int.incr wheres | _ -> ());
  (!ifs, !wheres, !loops)

(** {1 Loop queries}

    Locating a loop in an optimized or scheduled nest is the other thing every such test writes for
    itself, and the two ways it goes wrong are not visible in the result: a walk that stops at
    statement positions misses the loops an accumulation mints inside a [Local_scope], and a
    first-match on extent alone silently picks an initialization loop that happens to share the
    extent of the reduction nest one axis out (gh-ocannl-608). Both are decided once here: the scope
    descent by {!hooks.h_in_scopes}, the disambiguation by making the enclosing loops part of what a
    query sees. *)

type loop_site = {
  ls_index : Idx.symbol;  (** The symbol the loop binds. *)
  ls_extent : int;  (** [to_ - from_ + 1], the iteration count. *)
  ls_stmt : LL.t;  (** The [For_loop] statement itself, as a standalone routine. *)
  ls_ancestors : loop_site list;
      (** The loops enclosing it, innermost first — what tells a reduction nest from an
          initialization loop of the same extent. *)
}

(** Every [For_loop], in preorder, with its enclosing loops. Derived from the one traversal above
    via its exit hook, so a new IR constructor is still handled in exactly one place. *)
let loop_sites ?in_scopes (llc : LL.t) : loop_site list =
  let stack = ref [] and found = ref [] in
  walk ?in_scopes llc
    ~on_stmt:(fun s ->
      match s with
      | LL.For_loop { index; from_; to_; _ } ->
          let site =
            { ls_index = index; ls_extent = to_ - from_ + 1; ls_stmt = s; ls_ancestors = !stack }
          in
          found := site :: !found;
          stack := site :: !stack
      | _ -> ())
    ~on_exit:(fun s -> match s with LL.For_loop _ -> stack := List.tl_exn !stack | _ -> ());
  List.rev !found

(** The first loop (in preorder) satisfying [f]. *)
let find_loop ?in_scopes ~f llc = List.find (loop_sites ?in_scopes llc) ~f

(** The first loop with the given iteration count, as the symbol it binds. Prefer {!find_nest} where
    an initialization loop can share the extent: this answers "some loop of that extent", which is
    the weaker question. *)
let find_loop_with_extent ?in_scopes ~n llc =
  Option.map (find_loop ?in_scopes ~f:(fun s -> s.ls_extent = n) llc) ~f:(fun s -> s.ls_index)

(** The first loop of extent [inner_n] enclosed by one of extent [outer_n], as the pair of symbols
    (outer, inner) — the reduction nest of a pooling forward, past the initialization loop over the
    same extent. The outer is the {e innermost} enclosing loop of that extent. *)
let find_nest ?in_scopes ~outer_n ~inner_n llc =
  List.find_map (loop_sites ?in_scopes llc) ~f:(fun site ->
      if site.ls_extent <> inner_n then None
      else
        Option.map
          (List.find site.ls_ancestors ~f:(fun a -> a.ls_extent = outer_n))
          ~f:(fun outer -> (outer.ls_index, site.ls_index)))

(** Whether any loop binds [sym]. With [~in_scopes:false], whether one does through statement
    positions alone. *)
let binds_loop ?in_scopes sym llc =
  List.exists (loop_sites ?in_scopes llc) ~f:(fun s -> Idx.equal_symbol s.ls_index sym)

(** How many loops bind [sym] — the copies a materializing [Unroll] leaves behind. *)
let count_loops ?in_scopes sym llc =
  List.count (loop_sites ?in_scopes llc) ~f:(fun s -> Idx.equal_symbol s.ls_index sym)

(** The first loop binding [sym], as a standalone routine: the slice of the code a first-match probe
    speaks for. *)
let first_binding ?in_scopes sym llc =
  Option.map
    (find_loop ?in_scopes ~f:(fun s -> Idx.equal_symbol s.ls_index sym) llc)
    ~f:(fun s -> s.ls_stmt)

(** The [is_complex] fact recorded for [tn] by the structural facts pass. *)
let is_complex (o : LL.optimized) tn = (Hashtbl.find_exn o.LL.traced_store tn).LL.is_complex

(** Whether the facts pass classified [tn] as read before written within the routine — hence a
    routine input whose incoming buffer contents must be preserved. *)
let read_before_write (o : LL.optimized) tn =
  (Hashtbl.find_exn o.LL.traced_store tn).LL.read_before_write

(** Whether [tn] carries no assignment in this routine. *)
let read_only (o : LL.optimized) tn = (Hashtbl.find_exn o.LL.traced_store tn).LL.read_only
