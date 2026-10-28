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
  `test/einsum/test_conv_output_only.ml`.
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
  of one symbolic row would fail. Across a closure only `solved_dim`/`solved_sym` carry the value:
  `update_delayed_var_refs` publishes a dim variable's solution of either kind (`Sym` too, through
  `Row.get_dim_from_env`, gh-ocannl-1616) and a row's numeric product (none for a row holding a
  `Sym`). It writes only a reference with neither field set and never adjudicates a held one: every
  held solution has a constraint on the current variable (`set_dim`/`set_sym_dim` push one, a later
  binding emits the `solved` arm), so a dim meeting a symbol is the solver's `Shape_error` first, and
  the `Ir.Indexing` second-kind refusal is an unreached backstop (pinned directly, and every meeting
  order, in `capture_per_call.ml`). After the closure `var` names a variable of the reset environment (or a
  projection-local one, `derive_projections` rebinds `var`), free in the open work, so the tie to
  it constrains nothing: a binding drops it and lets the `solved` arm carry the value. A closed
  reference holding NO value (a `Sym` row) has lost its constraint, so every later relating use
  refuses it naming the closure (gh-ocannl-1655, `refuse_closed_unsolved_ref`): the binding's
  pre-pass, `set_dim`, `set_sym_dim`, `set_equal`, `set_scale`. Refuse at the use, not at the
  closure: a `Sym` row that is never reused is a valid program. The checks run BEFORE `set_var`,
  which touches the reference into `open_refs` and would make it read as open. An unbound
  reference `set_equal` relates to a row reference is marked (`row_numbered_refs`), and binding it
  to a row later is refused (product-only tie; binding it first gives `Row_eq`). Pinned by the shared-reference cases of
  `test/operations/capture_per_call.ml`, which assert the reference unsolved after both
  registrations: an operand of fixed size solves it at that step's Stage1 and exercises only the
  solved arm.
- `Shape.set_equal` on two row references is `Row_eq` in every solved state (gh-ocannl-1617): its
  Row-Row arm matches on `var` BEFORE the `solved_sym`/`solved_dim` arms. A row reference's
  `solved_dim` is only its product, and it is set early (by `set_dim`, or by
  `update_delayed_var_refs` at the registering step's Stage1 when an operand is fixed), so deciding
  on it first tied two rows by product, or compared two products, depending on what was solved
  by the time of the call. `Total_elems` is for a row against a number or a dimension variable only
  (`set_dim` on a row, the Dim-Row arm). A Row-Row pair with a reference bound before the last
  closure is refused, not product-tied: such a reference's uid is no longer in `open_refs`, its
  `var` is dead. Pinned under both registration and argument orders by
  `test/operations/set_equal_rows.ml`.
- Padded (`=`-mode) windows: max/tropical-family accumulations lower with clamped window bounds
  and demand NO margins (gh-504: `Row.proj_env.clamp_padded`; interval-based range guards in
  `Assignments.to_low_level`); add-family keeps physical halos committed as tensor-node identity
  (padded convs lower offset-free). The padded dim equation requires the input divisible by the
  stride, and a stride-2 window only clips on the right for window ≥ 5. A padded window in the
  RESULT slot is refused before elaboration (gh-ocannl-1550, `refuse_result_side_padded_window`,
  called from `einsum_n_constraints` and the `Permute` branch); a padded window's backward writes
  through the same window but reuses the forward projections, so it never comes through there.
  The refusal stays (gh-ocannl-1619, maintainer's call): its message gives the two-step recipe,
  `<+` then a crop by an operand-side valid window over a one-hot literal, executed against
  PyTorch's `ConvTranspose1d` in `result_side_affine_forms`. The crop is not a concat slice
  because operand-side concat extraction (`p^o^q => o`, `p^o => o`) dies in projection inference
  and integer concat components do not parse, whatever `syntax_extensions.md` once advertised.
- An affine axis's constant offset is sized for by nothing: the dim formulas omit it, so an offset
  at or past the stride, or on an unpadded window, indexes past the extent. On the result side
  those writes were silently dropped (stride 1 died in `projection_of_solved_dims`); on an
  operand they read past it. `refuse_out_of_extent_offset` (gh-ocannl-1618) refuses them in
  `einsum_slot_spec_to_dims_bio`, so every slot of every route; padded windows keep offsets.
  `test/einsum/result_side_affine_forms` executes every accepted result-side form against an
  oracle — extend it when adding one. The `%op`/`%cd` renderer (`Ppx_shared`) re-renders specs, so
  a grammar form it renders wrong is a parse error of a spec the user never wrote.
- A padded window's margins total the DILATED kernel extent, `left = span / 2` (the `Conv_input`
  arm of `Row`'s projection derivation); sizing them by the undilated kernel left dilated windows
  off-center, a wrong value no structural test saw (gh-ocannl-1657). Taps past the margins are
  range-guarded in `Assignments` (`clamp_conds`), so any padded offset reads zero there.
  `refuse_nonpositive_coefficient` refuses a stride or dilation below 1 at the same spot as the
  offset refusal. `test/einsum/padded_operand_offset_windows` is the operand-side oracle audit.
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
- That spurious axis is also how a shape defect reaches the SCHEDULER, which is where it actually
  got noticed: `Tensorize`'s role check rejects an operand mentioning the third micro symbol, and
  `Stage`'s insertion point L\* is "the deepest loop carrying an outer-part symbol", so an operand
  that spuriously depends on the row loop pins the cooperative load nest INSIDE it and breaks the
  perfect nest. One shape bug, two unrelated-looking decline families, split exactly by whether the
  seed stages operands (`sk_bk > 0`) or not. When a whole autotune family declines structurally,
  check the operands' index expressions before suspecting the scheduler.

- The solver's fixpoint is list equality over CONSECUTIVE iterations, so a constraint is deferred by
  re-emitting the SAME value, never a freshly built one: re-emitting a fresh `Rows_constr` from an
  `eliminate_*` arm churns origins and substitutions, defeats that equality and livelocks the
  per-stage loop (gh-ocannl-509 task 5) — defer through the stored-constraint path
  (`keep_constr`/`apply_rows_constraint`) instead. Same discipline the `unify_dim` deferral rule
  above states for dim constraints.
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
  no-op, a different one or a second kind refused). Those two Indexing setters are public — a write
  through them skips the session accounting; `Delayed_ref` is their only caller. Reads of committed
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
