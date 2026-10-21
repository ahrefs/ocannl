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
- Padded (`=`-mode) windows: max/tropical-family accumulations lower with clamped window bounds
  and demand NO margins (gh-504: `Row.proj_env.clamp_padded`; interval-based range guards in
  `Assignments.to_low_level`); add-family keeps physical halos committed as tensor-node identity
  (padded convs lower offset-free). The padded dim equation requires the input divisible by the
  stride, and a stride-2 window only clips on the right for window ≥ 5.
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
  for a genuinely underdetermined param dim, so the message does not tell you which you have.
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
  `test/operations/self_concat.ml`.
- Closed shapes are write-guarded (gh-ocannl-1301): `Shape.t.closed` is set by a SUCCESSFUL
  `finish_inference` (after every derivation), and every write site of rows, paddings and
  `padding_elem` (`set_rows`, `set_padding`, `set_padding_elem` in `shape.ml`) checks equality first
  — rows modulo projection ids (`Row.equal_modulo_proj`; `fresh_shape` re-mints them per
  derivation). A new write site on a shape must go through those helpers. Reads of committed state
  (`to_dims`/`to_padding` of a closed shape, `get_projections` of a committed step) skip
  `finish_inference`: a failed closure of OTHER work would otherwise poison a closed tensor's lazies
  (OCaml lazies re-raise their first exception). Tests that deliberately continue after an
  expected shape error must `Tensor.rewind_to_last_closure ()` when `Shape.taint_provenance ()` is
  set (`test_symbolic_extents.ml`), and a test that closes an operand to inspect it (e.g. via
  `Shape.to_padding`) can no longer then give it a padded consumer — read the fields instead
  (`neutral_elem_guard.ml`).
