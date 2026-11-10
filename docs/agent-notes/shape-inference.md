# Shape inference and einsum

The solver, einsum notation, row and dim variables, padding-mode windows.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Operator gotchas: unary sum-reduction is `++` (einsum1, e.g. `a ++ "i => 0"`); `+++` is the
  BINARY `outer_sum`, not a reduction. `m ++ "ii => i"` extracts diagonals. The `%op` `O` scope
  has the full comparison set `<`, `<=`, `>`, `>=`, `=`, `<>` (plus `not`, `where`) since the
  `Cmple` primitive (staging#216); `<=` is IEEE-exact (NaN-false), not a `not <` composite.
  A NUMBER in an axis spec is a fixed-index placement (size k+1 axis, value at slot k), not "axis
  N": per-axis index grids are `range n ++ "i=>i0"` ([n,1]) and `range n ++ "j=>0j"` ([1,n]),
  and comparing them broadcasts to an [n,n] grid; `range_of_shape` gives row-major flattened
  offsets, not per-axis indices. In `%op` only numeric literals auto-lift to tensors — a
  let-bound float needs `!.x`.
- Solver deferral discipline: `unify_dim` solves eagerly (recursing internally), so any
  constraint it RETURNS is a deferral that cannot progress in the current env — e.g. a conv axis
  whose kernel dim is still a variable. Deferrals are idempotent: re-solving one in place
  livelocks; hand it to the driver's next pass, where substitution can make it actionable
  (`solve_inequalities` is the pattern). Executable probe for output-only windowed specs:
  `test/einsum/test_conv_output_only.ml`. A `Shape` call site of `unify_row` outside the driver
  pushes what it returns onto `active_constraints`, never asserts it empty: `deduce_within`
  (`~deduced:Input_equals_output`) defers an unpadded window over an open kernel and a held window
  pair (gh-ocannl-2030, `test/einsum/input_equals_output_deferral.ml`).
- `Row.Concat` dimensions (block tensors, `^` in specs) are solved as ARITHMETIC SUMS, not
  structural lists: equality lives in `unify_dim` (single-component unwrap, `Concat=Dim`
  subtraction, cancel-common residuals, nested-`Concat` flattening), and the equal-arity
  all-`Var` pairing must fire at ANY solver stage or incremental Stage-1 solving livelocks.
  Solver-level facts are testable directly via `Row.solve_inequalities` on hand-built
  constraints (`test/operations/test_concat_dim_solver.ml`).
- `Shape.compute_row_product` must not latch a product while the row still ends in an open
  `Row_var` — the historical bug silently resolved xavier/kaiming fan products to 1 for every
  `{ w }` param with an inferred input row (fixed; regression `test/operations/layer_norm_1d.ml`,
  `test_default_init_std.ml`). Fixed-index output axes (`=> ...|0`, recognized via their
  `At_least_dim`) must not be equated by the stage-2 singleton-bounds heuristic, so that it
  broadcasts at stage-7.
- The row singleton arm (`solve_row_ineq`: a recorded `r ⊑ o` re-processed from Stage2 on, each
  row the other's only bound) is a GUESS, and load-bearing: it is how a parameter whose row only its
  consumer bounds takes that consumer's row (without it `test_param_shape_error` and
  `test_default_init_std` regress). A guess waits for the call's fixpoint (gh-ocannl-2039): the arm
  holds the bare pair and `release_sole_bound` equates it there, unless one row's GLB is open in
  the other (`<r>.[2] ⊑ o`, an infinite row). Made mid-pass, it refused stores a solution existed
  for, by where a repeated pair sat in the list. A closing call's fixpoint is final: constraints fed
  in a later stage-2 call find the pair already equated. Two things it still does not do: fire
  independently of the repeat (it reads re-processed pairs only, so a repeat can change what closes,
  never whether the store solves; `test_row_bound_emissions` counts it), and equating every
  sole-bound pair at the fixpoint instead pre-empts the stage-3 closure guess
  (`test_row_closure_guess`) and the rank refusals (`test_row_rank_cycle`).
- Implicit (omitted) row specs are shared and safe-to-guess; an explicit `...` stays strict; row
  variables in kernel slots need the `|` prefix. Symbolic extents (`Row.Sym`, gh-490) are rigid:
  equal only to themselves, materialized at their declared maximum; `Total_elems` against them
  fails fast, and `Total_elems` cannot resolve conv/affine dims either — give leaves explicit
  dims (e.g. `Operation.init ~o:[n] ~grad_spec:Require_grad`) instead of `Tensor.term_init` when
  the consumer is a windowed spec.
- `einsum_n_constraints` (the shared constraint-generation helper for binary Einsum, Block/concat
  and ternary Einsum) takes `?lhs_constraints_first` because the branches need OPPOSITE orders and
  no global choice works: binary Einsum passes `true` (the LHS constraint must precede RHS Affine
  constraints so a conv `over` variable resolves against an LHS-specified axis), while Block/concat
  and ternary keep the `false` default (RHS before LHS, or concat rows get prematurely solved to a
  concrete dimension). If a change makes one family's tests go green and another's red, the fix is
  the per-caller flag, not flipping the default.
- A capture reference (`delayed_var_ref`) passed by hand to several operations names ONE dimension
  (gh-ocannl-1545). Each operation's `bind_delayed_vars_to_envs` rebinds `var` to its own
  variable, so it also ties the new variable to the previous binding, solved or not: `Dim_eq` for
  dims, `Row_eq` for rows, equal products (`Total_elems` over a `Strided_var`, as `set_equal`) for
  a mixed dim/row pair. Tying rows by product instead looks consistent with the solved arm
  (`solved_dim` -> `Total_elems`) but `Total_elems` refuses rows holding a `Sym`, so two captures
  of one symbolic row would fail. Across a closure `solved_dim`/`solved_sym` carry the value, and a
  row reference's `carried_row` its axes (gh-ocannl-1689): `update_delayed_var_refs` publishes a
  dim variable's solution of either kind (`Sym` too, through `Row.get_dim_from_env`,
  gh-ocannl-1616) and a row's numeric product (none for a row holding a `Sym`). The carried row is
  published at the CLOSURE only (`~closure:true`, from `finish_inference`), from the binding the
  reference has then: a `Row` binding whose substituted row is `Broadcastable` with only `Dim`/`Sym`
  axes writes it (even over a held product, e.g. `set_dim` on a row reference), a `Dim` binding
  clears it. A step holding a surviving `Concat` (a symbolic component) or `Affine` axis cannot close
  at all (`padding_of_row`), so only a scheme instance's internal row reaches `carried_row_of` with
  one (no update step, no projections): pin such a case through `Tensor.Scheme.instantiate`. One
  over a `Sym` stays uncarried: carried into the next inference it would crash there. Do not publish it as a row resolves mid-inference: within one inference a row captured
  after a dimension binding is product-tied (row, dim, row ties the last row to the first by
  product only), and an early publication would make the carried value depend on whether the row
  resolved at a registration's Stage1 or at the closure. The parity table in
  `capture_per_call.ml` (`rebind_scenarios`) runs each scenario within one inference and across
  closures and asserts the outcomes equal. The product is written only to a reference with
  neither number set, and a held one is never adjudicated: every
  held solution has a constraint on the current variable (`set_dim`/`set_sym_dim` push one, a later
  binding emits the `solved` arm), so a dim meeting a symbol is the solver's `Shape_error` first, and
  the `Ir.Indexing` second-kind refusal is an unreached backstop (pinned directly, and every meeting
  order, in `capture_per_call.ml`). After the closure `var` names a variable of the reset environment,
  free in the open work, so the tie to it constrains nothing: a binding drops it (`previous_of`) and
  lets the `solved` arm carry the value -- for a row binding, `Row_eq` against the carried row,
  standing in for the dead previous row binding; `Total_elems` remains for a closed dimension (the
  tie is not transitive, maintainer's call in gh-ocannl-1969: row, dimension, row stays a product
  tie, and identifying the rows takes an explicit `set_equal`). `var` is always a binding inference
  made: `bind_ref` skips `set_var` for projections (gh-ocannl-1969), whose variables nothing reads
  through it; rewriting it left a closed reference with the kind of its oldest step, which
  `set_scale`'s row refusal and `set_equal`'s `row_numbered_refs` marking read backwards. A closed reference holding no number (a `Sym` row) still has its
  carried row for the `as_row` uses (a row binding, a Row-Row `set_equal`), but has lost its
  constraint for every use needing a number, which refuses it naming the closure (gh-ocannl-1655,
  `refuse_closed_unsolved_ref`): a dimension binding, `set_dim`, `set_sym_dim`, a mixed
  `set_equal`, `set_scale`. Refuse at the use, not at the closure: a `Sym` row that is never reused
  is a valid program. The checks run BEFORE `set_var`,
  which touches the reference into `open_refs` and would make it read as open. An unbound
  reference `set_equal` relates to a row reference is marked (`row_numbered_refs`), and binding it
  to a row later is refused (product-only tie; binding it first gives `Row_eq`). Pinned by the shared-reference cases of
  `test/operations/capture_per_call.ml`, which assert the reference unsolved after both
  registrations: an operand of fixed size solves it at that step's Stage1 and exercises only the
  solved arm.
- A capture reference publishes a bare size (`Ir.Indexing.variable_ref.solved_dim`), and projection
  derivation re-solves each step with it (`projection_sizing_operation`), re-emitted as
  `get_default_dim`: that refused every capture whose dimension solved to another tag, a free one
  (closed to `1_(bcast_if_1)`) or a named one (gh-ocannl-2033). `solve_with_sizing` adds those
  sizes only after a first Stage1 solve, dropping the ones the closed axes already fix at that
  size, so the size claims no basis there. Carrying the tag on the reference does not replace this:
  the tag is per BINDING, and a dim binding product-tied to a later row binding has its own, which
  the reference never sees (`product_tie_cancellation`'s "nothing sizes" case). Across a closure the
  reference does carry it (`Delayed_ref.solved_basis`, written with the size by
  `publish_ref_solutions`), and a later dimension capture takes it exactly, as `carried_row` keeps a
  row's tags: a named dimension is captured again as itself, and a free one closed to
  `1_(bcast_if_1)` as that top, so a plain `1` captured afterwards is refused though one inference
  would accept it (both pinned in `free_capture_basis.ml`). `set_dim`, `set_equal` and a row's
  product stay `default`.
- `Shape.set_equal` on two row references is `Row_eq` in every solved state (gh-ocannl-1617): its
  Row-Row arm matches on `var` BEFORE the `solved_sym`/`solved_dim` arms. A row reference's
  `solved_dim` is only its product, and it is set early (by `set_dim`, or by
  `update_delayed_var_refs` at the registering step's Stage1 when an operand is fixed), so deciding
  on it first tied two rows by product, or compared two products, depending on what was solved
  by the time of the call. `Total_elems` is for a row against a number or a dimension variable only
  (`set_dim` on a row, the Dim-Row arm). In a Row-Row pair a reference bound before the last
  closure (its uid no longer in `open_refs`, its `var` dead) contributes its `carried_row`
  (`row_for_set_equal`, gh-ocannl-1689). With both closed the rows hold no variable, so `row_row`
  asks the solver at the call (gh-ocannl-1969, `Row.solve_inequalities` over `Row.empty_env`): the
  rank, size, basis and `Sym` rules stay the solver's, the closed-rows arm of `unify_row` writes
  nothing, and a refusal leaves no constraint behind, so it taints nothing and needs no rewind.
  Pinned under both registration and argument orders by `test/operations/set_equal_rows.ml`.
- Which leaves take margins (gh-ocannl-1982): `Keep_shape_no_padding` commits an unpadded layout at
  birth (`Tnode.create_from_padded` forces `padding`, and `Shape.fresh_proj_ids` then registers
  zeros as resolved), so `wrap`, `wrap_param` and a loaded node refuse a margin demand. Everything
  whose `padding` is a lazy over the shape takes margins: `init` and `reshape` copy their array
  into the padded buffer at link time (`Tnode.create_with_reshape`), `range_of_shape` and params
  fetch into it. The price of the lazy is that reading the node's `dims` or `padding` closes the
  session, as for any other leaf, so `init` sits on the `Reshape` route only because it owns its
  array; `test/operations/padded_conv_init_leaf` holds the executed values and both refusals.
- Padded (`=`-mode) windows: max/tropical-family accumulations lower with clamped window bounds
  and demand NO margins (gh-504: `Row.proj_env.clamp_padded`; interval-based range guards in
  `Assignments.to_low_level`); add-family keeps physical halos committed as tensor-node identity
  (padded convs lower offset-free). The padded dim equation requires the input divisible by the
  stride, and a stride-2 window only clips on the right for window ≥ 4. A padded window in the
  RESULT slot is refused before elaboration (gh-ocannl-1550, `refuse_result_side_padded_window`,
  called from `einsum_n_constraints` and the `Permute` branch); a padded window's backward writes
  through the same window but reuses the forward projections, so it never comes through there.
  The refusal stays (gh-ocannl-1619, maintainer's call): its message gives the two-step recipe,
  `<+` then a crop by a slice (`"s, p^o^q => s, o"`), executed against PyTorch's
  `ConvTranspose1d` in `result_side_affine_forms` beside the older one-hot-window crop.
- An unsized padded kernel has two refusals with one message (`Row.unsized_kernel_error`).
  `Shape.refuse_unsized_kernels`, in `propagate_shapes` beside `refuse_discarded_operands`, takes
  what spec and captures decide, before any write: `get_inequalities` is too late, its einsum
  branches `unmark_unused` first, which would taint the session (gh-ocannl-1983). It sits on the
  registration path, so projection derivation never runs it. `to_proj` takes the rest: a capture
  nothing set, and a kernel an axis carries that closure guessed to `1_(bcast_if_1)`. That guess
  used to surface as `different bases` at strides above 1 and pass silently at stride 1, by
  `s_dim_one`'s padded arm merging two reserved tags. A broadcast consumer does not size such an
  axis, an einsum consumer does (`scheme_unsized_kernel`).
- Extraction and replacement (`p^o^q => o`, `b => a^b^c`, gh-ocannl-1656) are a concat TARGET
  exactly one of whose components another access names. `solve_proj_equations` gives such a target
  a SHIFT index (the named component's index plus the preceding extents) and the unnamed
  components no iterator; as `Concat` segments, a width-1 named component (indexed `Fixed_idx 0`
  like every dimension-1 axis) selected no segment and every segment hit the one cell. Concat
  targets never get an ordinary product iterator: an extraction's target is an operand axis, which
  is `Iterated`. A middle form's split is not carried by the closed shapes, so captured dims reach
  `derive_projections`' local re-solve as size-only constraints, filtered out of
  `get_proj_equations` (a dimension-1 size there would pin the axis at 0). Closure refuses to guess
  a concat component to 1 (`Row.add_concat_component`); `++^`'s discardable components still
  guess 0. Executed: `test/einsum/concat_extraction`.
- No route through `Shape` chains windows in one step: a tensor axis has one role per step (plain
  label, window, concat or fixed index), operands that alias are de-aliased, and a window ranges
  over a label, so a target is never what a window ranges over. `solve_proj_equations` still goes
  in dependency order (gh-ocannl-1971, hand-built in `test/operations/proj_window_verify`), for
  windows among themselves only: a window over a concat target is still an unknown projection.
  Likewise a later step sees a concatenation as a plain `Dim` (`test/einsum/concat_axis_consumers`
  executes slices, fixed indices and windows over one): `subst_dim` sums concrete components with
  a variable behind them or not, so `Concat` stays in a closed row only for a symbolic component,
  refused by `symbolic_concat_error`. Only `keep_affine` keeps concrete parts, for projections
  (integer components alone, `2^3 =>`: `test/einsum/concat_integer_components`, gh-ocannl-2019).
  The solver holds such an axis as a `Dim`, so a hand-built case reaching a `Concat` arm of
  `unify_dim` or the GLB merge needs a free component.
- A concat's segments are alternatives, and a block that cannot substitute an access is silently
  empty, so a form with no meaning used to compute all zeros or partly zeros: `p^q => p, q`,
  `a, b => a^b`. A result concat no operand names was iterated like a plain axis, each cell taking
  the reduced operand (`a => 2^3`, `p^q => r^s`), though an integer is a skipped segment everywhere
  else (`2^b => 2^b` is `[0 0 3 4 5]`). `Shape.refuse_meaningless_concats` decides these from the
  spec, since closure sums the axis and a width-1 axis indexes at `Fixed_idx 0`, and `loop_accum`
  refuses an operand no block reached; refusing empty BLOCKS or unreached segments would be wrong,
  replacements leave segments unwritten (gh-ocannl-2017). Only a single LHS `Concat` of one segment
  per operand selects operands by position, and only if each operand names its own segment (or none,
  at width 1): `block_selector` otherwise lets substitution decide, so two viable operands are
  refused as ambiguous rather than filling a segment from the wrong one.
- An affine axis's constant offset is sized for by nothing: the dim formulas omit it, so an offset
  at or past the stride, or on an unpadded window, indexes past the extent. On the result side
  those writes were silently dropped (stride 1 died in `projection_of_solved_dims`); on an
  operand they read past it. `refuse_out_of_extent_offset` (gh-ocannl-1618) refuses them in
  `einsum_slot_spec_to_dims_bio`, so every slot of every route; padded windows keep offsets.
  `test/einsum/result_side_affine_forms` executes every accepted result-side form against an
  oracle — extend it when adding one. The `%op`/`%cd` renderer (`Ppx_shared`) re-renders specs, so
  a grammar form it renders wrong is a parse error of a spec the user never wrote.
- A padded window's margins follow the DILATED kernel extent, `left = (span - 1) / 2` and
  `right = (span + 1) / 2` (the `Conv_input` arm of `Row`'s projection derivation): PyTorch's
  `padding='same'`, the extra tap of an even span on the right (gh-ocannl-1692; the
  `padded_operand_offset_windows` oracle's numbers are torch's); sizing them by the undilated
  kernel left dilated windows off-center, a wrong value no structural test saw (gh-ocannl-1657).
  Taps past the margins are
  range-guarded in `Assignments` (`clamp_conds`), so any padded offset reads zero there.
  `refuse_nonpositive_coefficient` refuses a stride or dilation below 1 at the same spot as the
  offset refusal. `test/einsum/padded_operand_offset_windows` is the operand-side oracle audit.
- A one-operand window (`x @^^ "s, 2*o=+k => s, o" ["k"]`, no kernel tensor) gets its kernel's
  extent from the capture alone, in two places that re-elaborate the spec with FRESH variables
  (gh-ocannl-1691): `derive_projections`' local solve (the projection-sizing constraints, which
  gh-ocannl-1656 added for concat splits and which fixed the forward as a side effect) and
  `product_space_shape`, the `*_pspace` gate proxy of a max-family backward, which ties each
  captured label to the operation's binding (`capture_ties_for_proxy`; untied, the kernel closed to
  1 and lowering died in `Indexing.prod_project_for`). A new fresh re-elaboration of a spec needs
  the same tie: operand equalities reach only labels an operand row carries. The audit of every fresh elaboration (gh-ocannl-1734) found four environments, all tied:
  `einsum_n_constraints` (einsum, ternary, block) and the `Permute` arm rebind captures through
  `bind_delayed_vars_to_envs` (in a projection re-solve, the closure-published value as a sizing
  constraint), `product_space_shape` adds `capture_ties_for_proxy`, and `of_spec` takes no captures;
  the binary route with a kernel neither operand carries executes in `padded_operand_offset_windows`.
  `Row.dim_to_proj` reads a window's kernel extent through `proj_env.v_env` and refuses an unsized
  or symbolic one as `get_proj_equations` does (`unsized_kernel_error`); no `Shape` route is known
  to hand it a variable kernel, so `test/operations/affine_dim_index_kernel` builds the environment
  by hand.
- Use-site row resolution is opt-in (gh-ocannl-544): an operation result's open row variables
  close DOWN to its arguments' shapes; a use site broadcasts the result in but cannot widen it.
  The resolve-at-use mark (`Row.add_resolve_at_use`, propagated through unification) grants the
  old widening behavior and is registered for: terminal rows (leaves and params), parameter
  initialization cones (`Tensor.mark_init_cone` — init intermediates must FILL the param's shape,
  or broadcasting repeats random values), sampler results (`uint4x32_to_prec_uniform{,1}` — a
  draw fills whatever shape context demands), and the explicit `stretch` identity wrapper (which
  replaced the `0.5 + 0.5` shape-inferred-constant idiom, e.g. pooling kernels `stretch 0.0`).
  The motivating trap (gh-ocannl-540): `Mixed_prec`'s cast twin read as the weight operand of a
  *batched* matmul was widened by the use site to `[batch, out, in]` — a per-batch-row copy of
  one weight. Nothing catches that downstream: every slice holds the same value, so results,
  gradients and loss trajectories are all correct, and the only symptoms are `batch`x memory/work
  and the row symbol appearing in an operand index. A parity oracle whose model has no batch axis
  cannot see a batch-broadcast bug — `mixed_prec_parity.ml` had covered this recipe for a whole
  release; `mixed_prec_twin_shape.ml` and `stretch_resolution.ml` pin the shapes directly.
  Dim variables follow the same policy (`Row.add_resolve_at_use_dim`): unmarked
  `Unconstrained_dim` axes are guessed minimal instead of widening to the use-site GLB — but only
  at stage 7, because stage-6 row closings still push concrete dims through einsum equality
  chains and an eager stage-6 guess-to-1 conflicts with a dim arriving in the same stage
  (box_muller's interior axes were the canary). `At_least_dim` axes always meet their use sites
  (direct indexing is a dim-carrying use).
- An exact `Strided_var` whose variable is also a factor of the constrained product cancels
  (`Row.cancel_strided_var`, gh-ocannl-1960): `(coeff * v / denom) / (v * rest) =
  (coeff / denom) / rest`, so `v` stays free and only `rest` is checked. This is what a product tie
  means for a row holding its own dimension, in either form (`v` among the row's axes or, after
  reduction, in `divided_by`). A numerator variable is never solved from such a constraint, and a
  fractional `coeff / denom` is a refusal, not a wait: no dimension can make it whole. The closed-row
  path, row elimination and `row_conjunction` all go through it, and the closed-row path marks `v`
  as if it were in `divided_by`, or the leftover `has_uniq_constr_unless` would keep it from ever
  closing.
- `divided_by` is a multiset, and a repeat in it is an equation: `n / w` over `[w, ..r..]` leaves
  `n / (w * w)` on `r` (`Row.reduce_row_constraint` appends the consumed variables as they are), and
  the closed-row check takes any number of `divided_by` variables. A shortcut that treats a repeat
  as unsolvable (dropping the constraint, or waiting) silently stops enforcing the total: a wrong
  size passes and a size the total fixes stays unsolved (gh-ocannl-2034; `divided_by_repeat.ml`
  solves in two calls, as the incremental solver does, since a single call re-reads the original
  constraint and hides the drop).
- The packed sampler's round-up total (`Total_elems` over a `Strided_var { round_up = true }`,
  minted by `Uint4x32_to_prec`) has to wait for the same use-site bound, in
  `Row.eliminate_row_constraint`: an arm that settles the counter from the already-closed axes
  alone is a guess, valid only when no bound gives the leftover row variable axes
  (gh-ocannl-1868). The stage-6 arm for `denom % coeff = 0` took any GLB for absent, so a param
  whose bound arrives late — bigram's `{ w } + 1`, reaching the matmul through a pointwise op —
  drew 28 values for `28 -> 28` whenever the closed axes' product was a multiple of the block size
  (4 at single precision), repeated along the axis constraint order happened to leave open. A GLB
  the caller passes in counts as pending while its dims are unresolved, like one read from the
  environment. Symptom to recognize: `range_over_offsets` smaller than the param by a whole axis;
  `test_default_init_std.ml` claims distinctness on the executed values.
- A draw's closure must not trust what the environment says mid-pass (gh-ocannl-1961). A bound
  solved this pass has already left its operands' `res` while the `Row_ineq` carrying it is still in
  the list, so a sibling reads as boundless for one pass; adding `res = []` to `closeable` did not
  fix it, and neither did it once the rest below was in, so it is not there. A variable shared with
  another row's `Total_elems` shows no bound at all, because a multi-row constraint is stored on no
  row. Hence round-up siblings close from stage 5 only, and `solve_inequalities` runs draws at
  quiescence at stages 5 and 6. A parameter never had this: its rows carry `is_in_param`.
  Non-parameter draws do, since `operation.ml` marks them resolve-at-use; `row_closure_guess.ml`
  compares them to explicit twins, with shape and count claimed too, since a smaller draw is distinct
  as well. Non-round-up totals keep the stage-3 closing: gating them at stage 5 with quiescence, as
  draws are, refused 8 claims in 3 tests (gh-ocannl-2025), since a reshape sized at stage 3 is what
  bounds a draw or parameter beside it. A fixed total closing a sibling cannot fail silently, only
  refuse or pick another valid split; the one false refusal found, an output row bounded by a closed
  use site short of the total, takes that bound and leaves the rest to its one sibling.
- A `Row_ineq` that reaches stage 6 unsolved closes its row variable to empty
  (`solve_row_ineq`'s first arm): the operand broadcasts in. So a size that arrives at stage 6
  through a chain does not widen a draw; only one arriving by stage 5 does.
- A balanced mixed-flank equation (`begS.<v> = <v2>.dimsR`, `|begS| = |dimsR|`) waits until
  stage 6, then commits the least overlap whose commit raises nothing, trying each in turn under
  `capture_globals`/`restore_globals` (gh-ocannl-2037). Refuting overlaps pair by pair missed two
  occurrences of one variable forced apart (`[x; x]` against `[3; 5]`), and `Affine`/`Concat`
  pairs: a loud refusal of a store satisfiable at a larger overlap. A refused trial must restore
  the module-level sets, not only drop its environment: `unify_dim` shares a use-site mark across a
  `Var = Var` merge before a later pair conflicts. Every pair the old check refuted, `unify_dim`
  refuses too, so a store accepted before keeps its overlap.
- That spurious axis is also how a shape defect reaches the SCHEDULER, which is where it actually
  got noticed: `Tensorize`'s role check rejects an operand mentioning the third micro symbol, and
  `Stage`'s insertion point L\* is "the deepest loop carrying an outer-part symbol", so an operand
  that spuriously depends on the row loop pins the cooperative load nest INSIDE it and breaks the
  perfect nest. One shape bug, two unrelated-looking decline families, split exactly by whether the
  seed stages operands (`sk_bk > 0`) or not. When a whole autotune family declines structurally,
  check the operands' index expressions before suspecting the scheduler.

- The solver's fixpoint is list equality over CONSECUTIVE iterations, so a constraint is deferred by
  re-emitting the SAME value, never a freshly built one: re-emitting a fresh `Rows_constr` from an
  `eliminate_*` arm churns substitutions, defeats that equality and livelocks the
  per-stage loop (gh-ocannl-509 task 5) — defer through the stored-constraint path
  (`keep_constr`/`apply_rows_constraint`) instead. Same discipline the `unify_dim` deferral rule
  above states for dim constraints. The equality is up to origins
  (`Row.equal_constraint_up_to_origin`, gh-ocannl-2000): they explain a constraint and say nothing.
  The one re-emission is the stage-7 pending round-up arm, whose value is stable: it wants the
  fixpoint, since there `Shape.refuse_unsolved` reports what stayed stuck, and the stored path
  re-fires on the constraint's own variables, not its GLB's.
- `Shape_row` constraints (added in `finish_inference` from every update row) are the ONLY mechanism
  that finalizes NON-terminal variables — `Terminal_dim`/`Terminal_row` cover leaf tensors alone. So
  a `process_shape_row` arm that closes a row var via its GLB must RE-EMIT the `Shape_row` unless it
  is the final stage (row.ml's `keep`), or a dim var in that row whose elimination is deferred (a
  fixed-index `=> …|0` axis: `At_least_dim 1`, guessed 1 at stage 7) is silently orphaned and
  lowering dies with "Not enough shape information: unresolved variable". Recipe for that message:
  dump `unsolved` (`[%sexp_of: Row.environment]`) after each stage of `finish_inference` and find the
  stage where the variable's `Shape_row` vanishes. The same orphaning surfaces instead as "You forgot
  to specify the hidden dimension(s)" when the var `is_in_param` — which is also the correct message
  for a genuinely underdetermined param dim, so the message does not tell you which you have. One
  cause is told apart (gh-ocannl-1508): a param dim held open only by a fixed-index lower bound
  (`At_least_dim`, minted solely for fixed-index spec axes) reports "Unresolved parameter dimension"
  naming the spec, via `unresolved_param_dim_error` (`tensor/row.ml`) at the stage-7
  `process_shape_row` arm; a new raise site for an open param dim should go through it.
- `unify_dim` and `solve_dim_ineq` end in CATCH-ALL arms, so adding a `Row.dim` constructor does not
  break the build. The equal-constructor arm in `unify_dim`, the inequality's final mismatch arm, and
  `join_dim` in `solve_row_ineq`'s GLB merge (whose wildcards keep a side instead of demoting to the
  top) each need a hand-written arm; that audit is the checklist a new dim constructor comes with
  (learned adding `Row.Sym`, gh-ocannl-490).
- Projection ids live on the `Dim`s of a shape's rows, and `fresh_proj_ids` freshens a shape IN
  PLACE, so one tensor that is several operands of one step (`(w, w) ++^ ...`, `m +* "a, b; b, c
  => a, c" m`, `m * m`) used to give every occurrence the same ids: the solver equated the axes the
  spec keeps apart. A self-product silently computed its diagonal; a self-concatenation lowered one
  segment twice and died "Ambiguous indices in concatenation" (gh-ocannl-1476).
  `Shape.dealias_operands` gives each repeated occurrence a shallow copy (same `id`, own rows) and
  `merge_dealiased_padding` folds the copy's inferred padding back. Pointwise self-operands are
  unaffected: their equations unify the copies' ids again. Executed parity:
  `test/operations/self_concat.ml`, and library-wide `test/operations/self_operand_sites.ml`
  (gh-ocannl-1497): it derives the library's own self-operand sites from the ppxlib parse of
  `lib/` and `tensor/` (all pointwise or same-spec today, so none depends on the de-aliasing) and
  runs each against a host oracle, and runs `*` plus every binary einsum operator `Einsum_parser`
  generates specs for on one matrix twice -- the cases that fail with `dealias_operands` reverted.
  A new self-operand site in `%op`/`%cd` library code fails that test until it gets a case.
- Closed shapes are write-guarded (gh-ocannl-1301): `Shape.t.closed` is set by a SUCCESSFUL
  `finish_inference` (after every derivation), and every write of rows, paddings and `padding_elem`
  checks equality first — rows modulo projection ids (`Row.equal_modulo_proj`; `fresh_shape`
  re-mints them per derivation). Since gh-ocannl-1506 the compiler enforces the routing: `Shape.t`
  is a `private` record, inside `shape.ml` too, whose only writers are the sealed `Shape_record`
  submodule's guarded setters (`set_rows`, `set_paddings`, `set_padding_elem`, plus `of_rows`,
  `copy`, `close`, `mark_discarded`); `shape_private_compile_control` holds the interface to it. A
  new kind of write is a new setter there. The privacy is shallow: a padding array read off a shape
  is still mutable. Since gh-ocannl-1573 the rest of gh-ocannl-903's frontier is private the same
  way: an `update_step` is built only by `make_update_step` (and `dealias_operands`' same-id
  `with_logic` copy) and its projections published once, by `set_projections`; a capture reference
  (`delayed_var_ref`) is written only by the sealed `Delayed_ref`'s setters, each of which refuses a
  rewound reference and counts the write (its open/discarded tracking is keyed by the reference's
  `uid`, so it stays top-level state the scan reads); and the `Ir.Indexing.variable_ref` it carries
  is private in `indexing.mli`, solved once through `set_solved_dim`/`set_solved_sym` (same value a
  no-op, a different one or a second kind refused). Since gh-ocannl-1624 those setters, and
  `commit_role` (a `static_symbol`'s `used_as_extent`/`used_as_slice`; the record is private too),
  take an `Ir.Indexing.shape_writer`: one per process, which `Shape` claims as it initializes, so
  `Delayed_ref` and `finish_inference`'s role commit are their only callers and any other claim
  raises (`indexing_shape_writer` claims it in a Shape-free process; `capture_per_call` holds the
  refusal). Hand-built IR gives a symbol its role at birth (`make_static_symbol ~used_as`). A
  `Tnode`'s delayed precision is sealed in `Tnode.Prec_cell`: only `update_prec`/`update_infer_prec`
  write it, and each changing write calls `Tnode.prec_write_hook`, which `Tensor` sets to
  `Row.note_mutation` — callers no longer count precision writes themselves. Reads of committed
  state
  (`to_dims`/`to_padding` of a closed shape, `get_projections` of a committed step) skip
  `finish_inference`: a failed closure of OTHER work would otherwise poison a closed tensor's lazies
  (OCaml lazies re-raise their first exception). Tests that deliberately continue after an
  expected shape error must `Tensor.rewind_to_last_closure ()` when `Shape.taint_provenance ()` is
  set (`test_symbolic_extents.ml`), and a test that closes an operand to inspect it (e.g. via
  `Shape.to_padding`) can no longer then give it a padded consumer — read the fields instead
  (`neutral_elem_guard.ml`).
- Every write to a module-level mutable binding of `row.ml`, `shape.ml` or `tensor.ml` (a `ref`,
  `Hashtbl`, `Hash_set`, ..., or since gh-ocannl-1573 a mutable field of a module-level record value
  such as `Tensor.session_state` or `checkpoint`) must count as a session mutation — the definition writing it calls
  `note_mutation ()`, usually as `if not (mem ...) then (note_mutation (); add ...)` so a no-op write
  does not taint — or carry an in-place classification: `[@@mutation_serial.exempt "<reason>"]` on
  the binding (fresh-id counters, the serial itself, a solver pass's transient flags, user
  configuration — the one kind of binding the `.mli` may export) or, for a record value, on the
  field's label declaration in its type (`session_state.next_id`), `[@@mutation_serial.fingerprinted]`
  on a ref `Shape.mutated_since` compares physically, or `[@@mutation_serial.boundary "<reason>"]`
  on a commit, rewind or reset function, or a scoped restore (`with_unchanged_roots`' `finally`).
  `mutation_serial_scan` (gh-ocannl-1506) enforces it; its header in
  `test/support/mutation_serial_scan.ml` lists the forms it reads and what it refuses, including a
  table passed to a helper or exported by the `.mli` (export a function instead, as
  `Row.mutation_serial` and `Shape.set_on_closure` do). It does not check branch dominance: a
  definition that notes on one path is credited for all its writes.
- Explicit checkpoints (gh-ocannl-903) snapshot the open frontier; their completeness is held two
  ways. Module-level state: every tracked binding of `row.ml`, `shape.ml`, `tensor.ml` that is not
  `exempt` must be captured, as a pointer, by its file's top-level `capture_globals` (a persistent
  value in a ref, or a field of a tracked record), or carry `[@@mutation_serial.kept "<reason>"]`
  (a registry rewinds extend: `discarded_nodes_table`, the handout record, the closure's
  checkpoint) -- the same `mutation_serial_scan`, so a new `Hashtbl`/`Hash_set` of session
  state is refused (hold a `Set`/`Map` in a ref). `restore_globals` destructures the capture
  exhaustively, so a field it forgets is an unused-variable error. Objects: each sealed owner's
  `snapshot` (`Shape_record`, `Update_step_record`, `Delayed_ref`, `Ir.Indexing.variable_ref`,
  `Tnode.snapshot_open`, `Prec_cell`) destructures its record with no wildcard: a new mutable field
  is a compile error there. The frontier is the active steps' shapes and steps (every in-place
  shape write goes through a step's `iter_shapes`; the rewind refuses a stray older shape), every
  LIVE capture reference (`Delayed_ref`'s weak registry: a caller may hold one no step reaches),
  and `Tensor.open_nodes`. What a rewind cannot reset makes it refuse: a raising committed-state
  read of a frontier shape (`failed_reads`; OCaml lazies memoize the exception) and a precision
  whose settling began (`Prec_cell`'s flag, set as the force starts, so a raising force counts).
  Oracles independent of the snapshots, and a negative control per restore category
  (`Tensor.For_testing.rewind_omitting`): `test/operations/checkpoint_rewind.ml`. A window op that
  rewrites an older shape's rows in place needs an equation: pointwise ops only add broadcasting
  inequalities, an einsum (`+*`) equates.
  The "steps" category restores only projections, which only a closure writes, so its control
  needs a closure failing PARTWAY through derivation (gh-ocannl-1827): `finish_inference` derives
  newest step first, so the step that keeps stale projections sits between the window's step and
  an older one that the window makes fail -- there a sum window whose operand's margins a window
  op with another FINITE neutral (a product accumulation; the max family is clamped, gh-504)
  committed first -- with the closure forced through a window shape, which leaves the rewind
  available. Nothing but `get_projections` compares a step's projections with the shapes they index
  (a closure derives none for a step that has them), so it refuses disagreeing ones on the way out
  of every path (gh-ocannl-1970).
- What a rewind discarded is recorded where it lasts only as long as someone can present it
  (gh-ocannl-1789): a shape's, a capture reference's and an update step's own `discarded_by` field,
  and a node's entry in `Tensor`'s weak `discarded_nodes` (an ephemeron table). A schemed closure's
  trace rewinds once, and a REPL rewinds per re-run cell, so a record keyed by id in a strong table
  grows with the session. Marking needs the window's objects: references come from `Delayed_ref`'s
  weak `live` registry (by the uids in `open_refs`), steps from `Update_step_record`'s weak `made`,
  the steps made since the last closure (a rewind cannot cross one; registered or not). A mark is
  set once: the provenance reported is the rewind that discarded the object, not a later enclosing
  one. `test/operations/rewind_registries_bounded.ml` pins the counts flat over 1400 rewinds and
  traces, with held stale handles as the control.
- Shape schemes (gh-ocannl-404, `Shape.Scheme`): replaying a trace's constraints with renamed
  variables is unsound alone. The solver mints variables mid-solve (row flank splits, the template
  rule's `more_dims`), `apply_env_step` writes them into shapes, and LATER constraints mention them,
  so the scheme also carries each constraint variable's solution under the trace's final
  environment. Export everything else substituted under that environment, so it holds only unsolved
  variables. A reference made before the trace may already be in `open_refs`, which lists each once
  per inference: the trace records its own touches (`trace_touches`).
- A closure retires the solver's per-variable tables (gh-ocannl-1859, `Row.retire_closed_variables`):
  the marks, the rank graph and the template cache, ahead of `on_closure` so the closure checkpoint
  does not pin them, and `Shape.unused_shapes` beside them. They were about 96% of what a session
  building fresh graphs retained per graph. A closed variable is therefore unknown to the solver, so
  a scheme applied after a closure generalizes even the variables from before its trace, and its
  anchors pin the fresh ones to the now-ground shapes: without that, shared-parameter variables
  re-entered the solver asking about facts that were gone. `debug_check_retired_solver_vars`, on
  across the suite, refuses any read of a retired variable, the only way to see the emptied table
  answer "no"; a new read of these tables goes through `refuse_retired_read`.
- External capture references in schemes (gh-ocannl-1791, `ref_inputs`): a reference made before
  the trace that the body binds is an input, not a refusal. During the trace it must read as unbound
  AND unsolved in every branch of `binding_constraints` -- the `unsolved` flag behind the mixed
  ties and the row arm's symbolic refusal too -- and `publish_ref_solutions` skips it, or the scheme
  bakes in the trace-time state. Instantiation relates the FIRST binding to the reference's current
  state; an explicit instance binds it to the LAST (what the body leaves it bound to). Setters on
  such a reference refuse at entry (`refuse_external_in_trace`): a satisfied `set_dim` and the
  row/row `set_equal` touch nothing, so `trace_touches` alone would miss them. A dim/row mixed
  binding traces (gh-ocannl-1890): both product ties (`Shape.product_tie`) have a constant
  coefficient (`Utils.safe_value`), which Stage1's cancellation may read early -- the
  `is_safe_val` disjunct of its `can_force` exists for constants.
- Random draws in schemes (gh-ocannl-1890): a packed conversion's `Strided_var` coefficient is a
  lazy over its result node's `storage_prec`, set top-down by consumers as they register
  (`update_infer_prec`). Every arm that reads a lazy coefficient is gated on Stage2+ (`late` in the
  merge arms, `is_stage2_up`/`is_stage5_up` elsewhere), so it is forced at closure, at its call
  site's precision. A scheme never forces it (trace, check and instance are Stage1): the trace
  swaps it for a hole that raises if read (`Row.map_draw_coefficients`; the kept `env` too, or the
  scheme pins the trace's nodes), and the body's own conversion node fills it per application. An
  explicit `instantiate` has no node and refuses. A new Stage1 arm reading a lazy coefficient
  would break this, and inline precision too. A round-up `Strided_var` is the draw marker: only
  `Uint4x32_to_prec` mints one.
- Printing a scheme (gh-ocannl-1753, `Shape.Scheme.to_string`): substituted boundary rows alone
  say almost nothing -- after a trace an argument's and its result's rows are distinct unsolved
  variables, related only by broadcasting bounds -- so the scheme keeps the trace's final
  environment (`env`) and the printer walks `Bounds_row`/`Bounds_dim` monotone chains through the
  intermediate variables (`res` side: refines; `opnd` side: refined). Names go by first appearance
  in reading order, never by id: a trace's ids come from the reserved range, and so do the ids in
  its errors' constraint sexps, which a golden must not pin. The chains stop at the printed
  variables -- the boundary's and those a lower bound names -- and at the fixed ones, and the
  relations are computed structurally first, then ordered by a name-independent key (a boundary
  variable by its header position; a fixed one by the sorted list of its relations to named
  variables, then by its pre-trace id) and only then rendered, so the naming follows the reading
  order and the text does not follow the solver's traversal (gh-ocannl-1988; before, the
  `nn_blocks_definitions` golden churned on solver-only changes). An intermediate that refines two
  printed variables with none refining it prints as `_ ⊑ a, b` -- a common refinement, dropped
  when a printed variable's own group witnesses it (`test/operations/scheme_printed_relations.ml`
  is the pin, with `cross_entropy_loss` in the inventory as the library's case).
- The results of an automatic scheme (gh-ocannl-1801, `Tensor.Scheme.traced_results`): the ppx
  cannot name a function's result type, so its tracer hands the body's value over untyped and the
  runtime finds the tensors in it by physical identity, among the tensors `add_fwd_root` saw while
  the body ran (`made_by_traced_body`) and the tracers. Not by the forward roots a body leaves: the
  backprop `%cd` of `@^^`/`@^+` makes inline tensors (`cond_pspace`, `sum_rhs1`) that stay roots
  nothing forward consumes, a tuple's intermediate is no root at all, and roots are weak, so a
  dropped tensor's presence would depend on the GC. A `{ t with forward = ... }` copy
  (`Nn_blocks`' batch norm in training) is matched by holding `t`'s node in the same field.
  `Scheme.check` passes `~results:None` to `instantiate_core`: results only print and instantiate.
- Printing a shape error (gh-ocannl-1799, `Shape.error_to_string_hum`): `Printexc.to_string`
  (the toplevel) and `Exn.to_string` render a `Row.Shape_error` readably by default, naming
  variables and reserved-range tensors (`n'1`) by first appearance, also inside the message text
  (`$<id>`, `n<id>`, `#<id>`); `shape_error_full_sexp=true` restores the raw sexp. Build every
  part in reading order with explicit `let`s: a `[%string]` or list literal evaluates its parts
  back to front, which would name the variables right to left. A golden prints the rendering, not
  the sexp (`scheme_error_rendering` pins it with a reserved-id control).
- A check that reports at the definition needs its verdict to hold for EVERY application, and for
  whatever else holds the function's tensors (gh-ocannl-1803). Reachability from the boundary rows
  does not decide that: an inline `{ w }` is an anchor built before the trace, and a padded kernel
  sits in the argument's row (inside the `Affine`) yet is certain, because the error concerns a
  variable each `derive_projections` re-mints, whose trace counterpart Stage 7 would guess
  silently. So decide such a check from what fixes the verdict (`Shape.unsized_padded_kernels`
  reads the spec and captures), never from trace variables; a check that only needs that much
  need not be a trace's at all (gh-ocannl-1983). Underspecification (a forgotten hidden
  dimension, a fixed-index parameter) stays at the closure by design: early reporting of it is out
  of the core calculus, and an analysis, if ever done, covers the whole class, not one case.
- Which origins an error keeps (gh-ocannl-1903): `Row.merge_origins` interleaves the incoming
  constraint's origins with the entry's in arrival order (never sorted: a sort by name made the
  `max_shape_error_origins` cap keep the alphabetically first), then ranks by the focus stack
  `Row.with_origin_focus` -- `Scheme.trace` and `instantiate_core` push the scheme's tensors
  (`Shape.traced_names`, filled by `Shape.make`/`of_spec` while a trace runs, and a scheme's
  `tensor_names`) -- before the cap. A new window that should own its errors' origins pushes a
  focus; ranking at a rethrow handler instead would come too late, after the cap has trimmed.
- Origins are diagnostics, and `max_shape_error_origins` must stay execution-neutral
  (gh-ocannl-2000): never decide anything from an origin list a solve has touched, since
  `merge_origins` may have trimmed it. Two places read origins outside error text. The solver's
  fixpoint used to compare them (now it does not; on 49 shape-heavy tests, 269k continuations, no
  pass had continued for an origin alone, at cap 0 or 20). `Shape.is_projection_sizing` finds a
  capture's sizing constraint by its origin tag, soundly because `derive_projections` filters
  `get_inequalities`' own list, before any solve; drop that filter and every concatenation case
  raises `to_proj: Dim without proj_id`. A filter over a solver's OUTPUT needs a constructor field
  instead. The `-origins_cap_0` aliases (`concat_extraction`, `transformer_test`, `resnet_block`)
  rerun at cap 0 against the default run's golden; what cap 0 may move is error text
  (`test_param_shape_error` loses its spec, by design) and allocation (`scheme_payoff`'s ratio).
- Scheme cost (gh-ocannl-1752, `test/operations/scheme_payoff.ml`, `Shape.registration_times`):
  a scheme only checks; it cannot stand in for the body's registration. An instantiation carries
  about as many constraints as the body's inline registration produces (C' + E' vs the steps'
  inequalities) and costs about as long, and constraints left pending beside later ties are
  re-solved by every tie solve. Count before claiming a solver win: `registration_counts` splits the store by
  source, and `closure_constraints` is the store at closure. Error locality is the payoff that
  holds (`scheme_error_locality` is the catalogue).
- Nested applications (gh-ocannl-1814, `Scheme.For_ppx.in_checked_body`): `apply_variants` runs
  plain inside the body of an application whose check really ran, set only after it did, so an
  outer that is refused, opted out or could not checkpoint leaves its applications checked. It
  moves no error (the outer check always came first, so a nested check could only speak after it
  passed), and a body breaking the determinism premise still registers inline. Nested applications
  skip `due` too, but two traces stay, each holding what no outer window does: a multi-pattern
  function's first application traces its other patterns, and `closure_variants` its eager trace,
  for a closure built in a body and never applied. On a schemed decoder, tracing is about two thirds
  of check's time, and those eager per-layer traces are most of it.
  `Tensor.For_testing.with_nested_rechecks` is the control, and `scheme_payoff --bench`'s `recheck`
  row its A/B in one process.
- A rank-cycle error is usually a consequence: find the commitment that made it (gh-ocannl-1955).
  Two sit in the closing path. An equation left with `beg_handled = false` solves neither variable,
  and `value` there is the rest of the other side, so `visit_rows_for_row_solution` (which reads
  `v = value` into every bound through `v`) runs only on the arm recording the solution. And closing
  a row to its GLB is a choice, so every close reads the entry through `rank_refutes_glb`, a
  read-only query of the rank graph: `entry_for_closing` for the terminal close and
  `process_shape_row`, `eliminate_row_constraint` when it re-reads the GLB from the environment
  (`collect_factors` looks at flanks only, so `4,4,..w..` counts as settled). A refuted GLB is
  absent, not an error (gh-ocannl-1961), except its trailing flank: it bounds the row's last axes
  at any rank, so a row that must take axes reads it (gh-ocannl-2035, `refuted_flank`), a draw
  filling it and a fixed total taking it when it holds the total. The bound itself was true: `pointdiv`'s `grad_asn` builds `-1 *. t1 /. (t2 **. 2)` on the primal, a pointwise result
  over both operands, so dividing `h, w, ..c..` by `..c..` bounds the statistic by a spatial row.
  A fresh evaluation never meets that GLB: its bound stays between variables, and stage 2 discharges
  `Total_elems` with no GLB. A compile before training hides both, as the rows arrive solved.
  `batch_norm_closure` pins it; a leaf closes rows at once, a result only bounds them, so test both.
- Where inference time goes: profile before blaming a term, per trace kind. `scheme_payoff`'s
  `tracing` column is per-block traces; a trace of a whole body is `--stacks`, a window of its own
  (gh-ocannl-1891). The rank-cycle check is `Row.add_rank_edge` over `global_rank_potentials`,
  checkpointed with the edges: `p v >= p w + k` on every edge is a proof no cycle is positive, so an
  edge that satisfies it needs no search, and one that breaks it lowers what must move, Dijkstra-style
  over reduced costs. The work is `registration_counts.rank_search_visits`, which doubles per depth
  doubling (before this it was the largest solver term left: 4x visits, 18.3 s of 42.9 s at decoder
  depth 64). `Row.with_unbounded_rank_search` is the old walk, and compares verdicts edge by edge.
  `solve_dim_ineq`'s `cyclic` was the exponential one: unsized dims stay `Bounds_dim`, so a trace's
  end validation re-fed each inequality over a graph that residual connections fill with diamonds.
  It keeps a visited set, a recorded edge skips it (a merge re-emits bounds as inequalities, so every
  edge that adds a path is still checked), `Row.with_unmemoized_cycle_search` is the old walk, and
  `dim_cycle_visits` still grows 4x with a small constant. What is left past depth 16 is not the
  solver's: tensor construction's set unions (schemes off too), and each closure's trace
  checkpointing the session's open frontier (quadratic in closures, 11.7 s of 43.9 s tracing at
  decoder depth 128). Allocation (`OCAMLRUNPARAM=v=0x400`) is a deterministic proxy when the box is noisy.
  `scheme_payoff`'s allocation bands assume the default `max_shape_error_origins` and the
  cross-check off (gh-ocannl-1968; its header): cap 0 drops the origin lists, a term linear in the
  merges, and the decoder trace's deepest doubling reads 2.88 against its 2.9 ceiling; under the
  cross-check they are a skipped leg.
- What a row solution owes the entries it bounds (gh-ocannl-1977, `s_row_one_in_entry`'s comment):
  rows have no GLB transport but the visit, so `value ⊑ opnd` (a value with axes) and `glb ⊑ value`
  are load-bearing, and for a long time nothing pinned them; dropping both passed the whole shape
  suite (`test_row_bound_emissions` does now). A bare variable carries no axes, so it stops at the
  entry: its emissions stored the transitive closure, 94% of the volume in a resnet, and duplicates
  were not the cost. The pairs it stored moved two things a test can see. A store closes differently
  in a corner, where an alias before a solution reached a third hop (an accident of equation order,
  pinned), and the stage-2 singleton arm fires about 3% more often, since it needs a recorded pair
  and a short list. `Row.with_transitive_row_emissions` is the old walk; compare against it on
  random stores before touching this. Row GLBs reach two hops, dims all the way: closing that gap
  would change what ordinary programs close to.
- Substitution is indexed (gh-ocannl-1904): a solution visits the entries mentioning its variable
  and, for a dimension solution, the pending ones (`environment`'s doc comment), not the whole
  environment. Every write goes through `set_dim`/`set_row`, which index it: a `Tree_map.add` on
  `dim_env`/`row_env` elsewhere, or a mention a visit reads that `dim_entry_mentions`/
  `row_entry_*_mentions` do not list, is a missed visit. The whole map also changed entries that
  do not mention the variable -- it summed a stored `Concat` of solved parts and collapsed
  windows -- which is why written entries stay pending until an unrelated visit leaves them
  unchanged. A `Total_elems` row no longer does: a solution reaching no divisor is no change, where
  it used to re-emit and stay pending, rows times solutions (gh-ocannl-1977).
  Verify a change here with `OCANNL_SHAPE_SUBSTITUTION_CROSS_CHECK=true` over the shape tests (it
  fails on the first solution whose indexed visit differs from the whole map's), and
  `test/operations/shape_substitution_chains` for the chains. Under it `scheme_payoff` runs one
  depth shallower in every table (gh-ocannl-1968): at its full depths the cross-check took over 12
  minutes, nearly all in the depth-16 decoder stacks, where the default run takes about 25 s; one
  depth shallower it takes under 3 minutes. The counts it claims are the default run's at every
  depth, so the check loses nothing but the deepest doubling. The eager, resolved representation
  stays quadratic on an alias chain solved oldest-first (`x1:=x2, x2:=x3, ...`); no model workload
  showed one.
- A trace solves symbolically what inline construction solves numerically (gh-ocannl-1820): a
  scheme's tracer widths are unsolved variables, so an `Affine` the `Dim` arms would collapse to a
  number stays a term, and each symbolic arm of `unify_dim` must agree with the concrete
  arithmetic. Two windows over one extent relate through `window_pair_equation` (`tensor/row.ml`;
  applied only as a closing fallback since gh-ocannl-2038, below), which writes both as `stride * (o - 1) + span`: a padded window or plain affine of stride `s`
  has span `s`, not 1 (read as 1, a padded stride-4 shortcut against a stride-2 pool derived
  `2*o - 1` where concrete solving gives `2*o`, gh-ocannl-1823; equal strides hid it). A span
  difference `gcd s1 s2` does not divide has no solution and is refused, not truncated. Strides
  that do not divide each other (2 and 3) have solutions but no functional relation, so the pair
  stays a `Dim_eq` until a side is solved, then the solved arms decide it (gh-ocannl-1972); a
  rigid `Sym` on both sides is refused, since nothing would ever solve one. No width at all still
  fails: closure sizes each output at 1 on its own, and `2 = 3` fails in the plain arms; from
  Stage5 `solve_inequalities`' handler rewords a size failure of a pair stored with both outputs
  open (`unsized_window_pair_message`) to name both windows, the sizes closing chose, and the
  widths that fit both (gh-ocannl-2040). It rewords only what closing's sizes broke, so a width
  that arrives and does not fit keeps its own `incompatible stride`. The
  helper's span can be below 1 (`o2 = o1 - 2`), and `s_dim_one` refuses a helper output below 1,
  which is the window not fitting; without that, a width arriving after the pair gave `o2 = 0`
  or a negative extent where solving concretely refuses. Reorienting the helper would not do:
  at unequal strides no orientation has a positive span.
  A stride-1 valid window of span 1 over an unsolved extent is the identity and collapses in
  `unify_dim`'s stride-1 arm and in `s_dim_one` (`is_identity_window`); kept, a residual add
  equating its `over` with it reads as an occurs-check cycle. Collapsed, nothing constrains the
  extent's basis later, so the identity passes the basis through over a solved `Dim` too
  (gh-ocannl-1824): the solved arms' basis rule turned the other reserved tag into the
  kernel's (`bcast_if_1` into `default`), and `conv(v + kernel 1) = 4_(bcast_if_1)` solved before
  `v = 4_(bcast_if_1)` but not after. `s_dim_one` applies `is_identity_window` before its solved
  arms for the same reason, and `unify_dim`'s kernel-inference arm (a window over a solved extent
  whose kernel is a variable, against a solved `Dim`) re-unifies the equation once it has bound
  the kernel: it picks the kernel from the sizes alone, and used to skip the basis check a kernel
  solved earlier would have met. A named kernel tag still converts (its window is no identity).
  The occurs check is unchanged. Pinned by
  `test/operations/scheme_strided_residual`: `check` against `off` on schemed blocks with equal
  and unequal strides, the window-pair equations solved directly in both orders, and the identity's
  basis over every reserved tag pair, constraint order and a kernel solved late.
- A valid window's basis has one forward rule, `window_input_basis` (gh-ocannl-1968); the reverse
  arm and the kernel-inference arm enumerate candidates through it (`window_bases`) instead of
  inverting it by hand, which is how reverse came to accept what forward refused. The rule is not
  injective: `default` and `1_(bcast_if_1)` both map to the kernel's tag, so an equation with
  several surviving tags stays a `Dim_eq`, and `solve_inequalities` settles the first one on
  `default` at a fixpoint from Stage2 up. At a fixpoint, not at the stage: `set_dim` and the other
  trailing setters push constraints no Stage1 pass read, so a stage gate would settle a window ahead
  of an older `set_dim` in the same pass. `default` is in every ambiguous set, so it satisfies every
  open window over an output at once, which `merge_derived_basis`'s answer does not; a bound on the
  variable or a nested window can still rule it out, so a candidate is vetted by solving what it
  emits, and one that binds nothing settles nothing, or the fixpoint would repeat. `bcast_if_1`
  above size 1 is an atom in the rule, or the whole conv path would be ambiguous and every window
  output would leave Stage1; contested outputs are size 1 and those of a named kernel, and such an
  output is unknown until closure if nothing else solves it. Padded windows take the extent's basis,
  whatever the kernel. A window pair with a kernel that claims a tag stays a `Dim_eq` after its size
  equation, since that equation carries no basis; once the reduction changes nothing, the pair alone
  stays, or a deferred reduction would pile up a copy a pass. Pinned by
  `test/operations/scheme_strided_residual`'s matrix over tags, window shapes and orders.
- A window pair's size equation is a whole-dim equation, so applied while an output is open it
  equates the outputs' tags too (gh-ocannl-2038). The pair says only that the inputs coincide, and
  the rule is not injective: `1_(bcast_if_1)` and `1_default` under `default` kernels, or `1_h` and
  `1_default` under `h` kernels, derive one input, and were refused pair-first. The helper window's
  `default` kernel also carried a tag across sizes: an inert `2_(bcast_if_1)` became the top at
  size 1, accepted pair-first where the windows refuse it. So `unify_dim`'s `window_pair` holds an
  open pair: once one output is solved, the solved arms decide the other's size and its tag
  candidates exactly. The size equation applies only from `settle_window`, at a closing fixpoint
  and in a second pass after no open tag settled (`merge_pairs`): closure sizes each open output on
  its own, and `2*(o1-1)+3 = (o2-1)+2` closed at 1 and 1 is a mismatch. Merging first was itself
  order-dependent, since a pair merged before a tag settled bound an output a solved input had left
  free. The pair stays with its equation, to check the tags once closure solves the merged output;
  rigid outputs are decided at once, as before. A traced scheme now carries the held pair and two
  output variables (traces solve Stage1 only); an application replays it. Pinned by the same test's
  "a window pair's verdict and tags do not depend on constraint order" (both orientations, solved
  and late kernels, padded against valid), the solved-input case and the closing case.
- Every benchmark model builds under `check` with `off`'s dims (gh-ocannl-1908):
  `test/operations/bench_model_schemes` builds each conv and gpt workload of `benchmarks/workloads`
  from a synthetic zero-valued fixture, through `Bench_conv_model`/`Bench_gpt_model` as the runners
  do, plus `Nn_blocks.resnet_block` at stride 2, and compares every graph tensor's label and dims.
  A new workload JSON is covered without an edit; a new model kind needs its fixture layout in
  the test's `fixture_tensors`. The mlp workloads are not covered: `bench_mlp`'s model lives in its
  executable.
