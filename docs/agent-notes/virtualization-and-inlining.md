# Virtualization and inlining

Which candidates inline, where each rejection is decided, the policy caps, and how to test a
value-rewriting pass on hand-built IR.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Value-rewriting passes need executed parity tests, not just structural pins (see AGENTS.md).
  To exercise a virtualized affine-LHS producer end-to-end, hand-build an `Assignments.comp`
  (einsum result-side scatter specs don't parse; gradients accumulate → stay materialized): pass
  `~name` to `Context.compile` (or wrap in `Asgns.Block_comment` for labeled debug dumps), set
  `embedded_nodes`, force the output materialized, seed inputs with `Context.set_values`, then
  compile→run→`get_values`. When the shape under test is NOT reachable through `Assignments` at
  all (the optimizer never emits it), build the `Low_level.optimized` directly — `LL.optimize` over
  a hand-written `LL.t`, or `analyze_proc`+`specialize_proc` — and pass it as
  `Context.compile ?prelowered` with `~name` and `Ir.Assignments.empty_comp` (gh-ocannl-562).
  It replaces the compile's lowering wholesale, so the analysis layer and the kernels see one IR;
  add `~lowered_transform:(fun o -> [ o ])` to keep the default schedule annotator off hand-built
  code.
  Doc-check `prelowered_transform`: `let compile = let open Ocannl.Nn_blocks.DSL_modules in Context.compile ~lowered_transform:(fun o -> [ o ])`.
  See `test/operations/prelowered_seam.ml`, and mind the scope-purity contract below.
- Do not re-derive that harness: `test/support/ll_test.ml` (library `ll_test`, links `ocannl`) holds
  the LL builders, ONE exhaustive `Low_level.t`/`scalar_t` traversal with the counters derived from
  it, and `optimize`/`run`/`execute` — add `ll_test` to the test's `(libraries ...)`. A new IR
  constructor is handled there once instead of in every copy. `test_utils` stays separate on purpose:
  it depends on `arrayjit.ir` alone, for tests that link no more than that.
- Such a hand-built case gets its differential arm for free: pre-decide the placement in the
  `optimize_ctx` you hand `LL.optimize` (`Low_level.decide_materialized`, which is what
  `Context.decide_materialized` records for the `Assignments` pipeline; `Ll_test.optimize
  ~materialized` wraps it) and re-specialize the SAME `LL.t`. The inlined and materialized readings
  of one program must agree cell for cell, which is what pins a virtualization guard;
  `Context.decide_materialized` on the context itself cannot do this, because `?prelowered` replaces
  the lineage state with the record's own `optimize_ctx`.
  Three traps: (a) `known_non_virtual` does NOT mean "has a context buffer" — a node written and read
  within one routine and never observed is placed `Local`, routine-scoped scratch whose values never
  reach a context buffer. Host access to a node this lineage placed `Local` raises in BOTH
  directions (gh-ocannl-599; `test/operations/local_host_access.ml`), so the mistake fails loudly
  rather than reporting the uploaded copy back as a computed value. Read back only nodes you
  declared `On_device`, and mark them `Tn.set_observable` so the aliasing planner cannot hand their
  bytes to another node. (b) Producer/consumer indices that run past a node's dims are
  invisible while the node is virtual (the access is inlined away) and become real out-of-bounds
  traffic the moment the case executes or the materialized arm runs — size hand-built arrays for the
  materialized reading, and seed outputs with a sentinel so "wrote the wrong cells" fails the value
  check instead of reading whatever the buffer held. (c) The oracle has to discriminate, not merely
  exist: a producer must write a value that varies with EVERY symbol of its iteration and stays off
  the init value (`1 + i`, `1 + 10*outer + inner` — the `tick`/`tag` helpers in
  `test/operations/virtual_diagonal.ml`), because a constant producer just replays an identical
  assignment under a too-wide range guard, a value omitting a symbol is constant along that axis
  under a wrong substitution, and a value colliding with the zero-init hides a dropped first
  iteration.
- A `Local_scope` body's ONLY effect is on the locals it owns — its own scope id plus ids
  `Declare_local`d lexically within it (gh-ocannl-584). Not a tensor node, not a sibling's or
  enclosing scope's local, no `Workgroup_barrier`, no `Staged_compilation`. The reason is that a body
  does not execute where it is written, in three different ways: `C_syntax.pp_scalar` returns it as
  a local definition that `pp_local_defs` emits ahead of the enclosing statement in its own order;
  `simplify_llc` collapses a single-assignment scope into the expression, moving its
  reads the other way; and `hoist_cross_statement_cse` lifts a body shared by sibling statements to a
  top-level `Declare_local` + body, running ONCE ahead of the first user. Purity makes all three
  placements unobservable from outside the body, which is exactly what `Affine.path_before` assumes
  when it refuses to order sibling `Arg` positions; the two would otherwise disagree. Purity governs
  a body's EFFECTS, not its inputs — the hoist separately needs the body's reads untouched across
  the statements it is lifted over, and its hazard check must cover scope locals (`Get_local` /
  `Set_local`) as well as tensor nodes, or a shared body is lifted above a `Set_local` of a local it
  reads and later users read a stale value (a real miscompile, pinned by `prelowered_seam` phase 5).
  Bodies reading a local declared OUTSIDE them are ordinary pipeline output — CSE and the hoist
  itself create them — so "a body may only read locals it owns" is not available as a rule.
  `hoist_cross_statement_cse` is the ONLY pass that can move an effect out of a `Local_scope`
  (`simplify_llc`'s collapse cannot match an impure body; CSE's dedup leaves the surviving
  occurrence impure), so it guards its own precondition with `scope_purity_violation` and declines
  to hoist an impure body, instead of every public door into `specialize_proc` needing a gate. That
  is what keeps the raw analysis probes usable: an impure body reaching the pass would otherwise be
  laundered to a top-level `Declare_local` + body, and `Context.compile ?prelowered` would compile a
  silently changed routine. Rejected, never rewritten — pinned by `prelowered_seam` phase 6.
  The contract governs a body's EFFECTS only; it deliberately says nothing about the ORDER of its
  reads of other locals. A rule for that was written and reverted (gh-ocannl-584 review rounds 4-5):
  deciding "is this local emitted before me?" means replicating codegen's emission algorithm
  (`pp_local_defs` sorts by `scope_id`, per-statement def blocks, `Set_dynamic` concatenating two
  operands' defs), and every divergence is a FALSE REJECTION of valid IR — three surfaced in one
  review round, one of them rejecting CSE output inside a scope body, which the pipeline really
  produces. The guarantee lives in `pp_local_defs` instead, which HAS the emission order in hand and
  so cannot diverge from it (gh-ocannl-1895): definitions run by ascending `scope_id`, except that
  one follows the siblings its body reads (`Low_level.local_reads`), so a kernel whose ids already
  respect the reads keeps its text. CSE produces the other order whenever a collapsed relay leaves
  an older scope beside a fresh one: a read's id is minted when it is walked, so a gradient chain's
  `v25_d.grad` reads `v27_rw_x`. The hoist has the same obligation one level up: its stale
  `Get_local` remap has to reach scope bodies, or a body reads a local nothing declares. A local read
  by its own single assignment is a different failure (`simplify_llc` collapses it out from under
  its declaration); that one is refused earlier, as `158`. `local_scope_sibling_reads` pins all of
  this but the last.
  `Low_level.validate_scope_bodies` enforces it at BOTH ends of the pipeline: `optimize_proc` on the
  way in (ahead of the analysis cache, so a digest hit cannot skip it — and before the hoist can
  launder a body write into a top-level statement that no later gate would recognize) and
  `C_syntax.compile_proc` on the way out (catching what a schedule transform constructs), the latter
  ahead of `validate_parallel_classified` and NOT transported as an `Illegal_schedule` — no schedule
  choice can rescue malformed IR. Its statement match is exhaustive with no catch-all, so a new
  `Low_level.t` constructor breaks the build until someone classifies it as body-legal or not. The
  pipeline complies by construction: `inline_computation` drops the inlined computation's `Set`s and
  `Zero_out`s. The raw analysis entry points `analyze_proc`/`specialize_proc` deliberately do NOT
  validate — they are the probes that must stay conservative on IR they may not trust
  (`test/operations/affine_extraction.ml`); everything past them does.
- **A scope's collapse is exact, or it does not happen** (gh-ocannl-1788). A `Local_scope` rounds
  its value to its own precision (codegen declares its local at the backend's `compute_prec` of it),
  so `simplify_llc` collapses a float scope into a float slot only when the slot evaluates at the
  scope's precision, or when the value is a load whose storage the scope's precision widens exactly
  (`Ops.widens_exactly`). The slot's precision is `loop_scalar`'s `~eval`, threaded as
  `walk_operand_precisions` threads it: a homogeneous operand keeps its parent's, a heterogeneous
  one takes its own annotation (the annotation alone cannot tell, since a scope operand is annotated
  at its own precision). The collapse used to be unconditional, so a virtualized bf16 cast read in
  an f32 sum lost its rounding wherever the local is declared bf16: Metal, CUDA and HIP by default,
  cc under `narrow_compute_f32=false`. cc's default declares it f32, which is why nothing on the
  default CPU path noticed. A wide scope over arithmetic collapsed into a narrow slot computed the
  arithmetic narrow. `test/operations/scope_collapse_rounding` executes each shape against the
  uncollapsed `?prelowered` form under both policies; `top_down_prec`'s goldens show it end to end
  (a half sum read in a bfloat16 product was computed at bfloat16 on Metal). A scope or slot of an
  integer precision still collapses unconditionally: the gh-343 gather rewrite recognizes a one-hot
  comparison only once its scope has collapsed into the float product (`test_bounds_folded_gather`
  and the tune-cache tests went red when it did not). A kept scope is what the pattern-matching
  passes then see: the explicit-SIMD renderings decline a loop holding one (`scalar_mentions`).
- The scope-TARGET contract, companion of the body contract above (gh-ocannl-681): a `Local_scope`
  over X denotes THE INLINED COMPUTATION OF X, so `LL.optimize` accepts one only while X is virtual,
  and REJECTS it over a materialized X (`scope_target_rejection` in `low_level.ml`, raised from
  `cleanup_virtual_llc`) instead of the silent normalization to a plain `Get` it used to do. The
  trap that makes this bite: a node with NO SETTER is decided non-virtual, so a hand-built scope
  over a freshly created node is a scope over a materialized node — declare it virtual
  (`Ll_test.virtualize`). Two shapes were green-by-collapse before the rejection: `accum_width.ml`'s
  gh-639 legs ran kernels literally spelling `acc[0] = acc[0]` (the identity copy reproduced the
  expected value), and `affine_extraction.ml`'s sibling-operand probe lost scope B's write while its
  own comment claimed the scopes survived `specialize_proc`.
  Exactly one exemption, and it is a retraction of the optimizer's OWN decision rather than of the
  caller's program: `virtual_llc` mints a scope at a `Get` of a still-virtual node, a later refusal
  can commit that node `Never_virtual`, and rewriting back to a `Get` is then sound because the
  surviving setter writes the value the body recomputed. `input_scope_ids`, taken before
  virtualization, is what tells the two apart — a scope in that set may not be rewritten away.
  The retraction is REACHABLE, and structurally so rather than by accident (gh-ocannl-704):
  `virtual_llc` walks statements in SOURCE ORDER while a node's placement is one mutable cell shared
  by the whole walk, so a refusal decided at a statement reached AFTER a read that already minted
  flips the node under an existing scope, and nothing revisits that scope in between. Both rejection
  families do it — store time (`check_and_store_virtual`, e.g. `Non_virtual 142` on a guarded LATER
  setter of an already-read node) and consumption time (`inline_computation`, `Non_virtual 13` at a
  second read the producer's index map cannot serve). Deleting the exemption therefore makes
  `LL.optimize` refuse IR it built itself; `test/operations/scope_over_materialized.ml` pins one
  witness of each family, each with executed parity against the same program's materialized reading.
  Both witnesses are hand-built, which is the exemption's honest standing: it is load-bearing for IR
  `optimize` accepts, not a mechanism user programs are known to hit — instrumented builds (the
  gh-ocannl-681 PR's, and a repeat over the targeted virtualization tests) recorded hits only on
  out-of-contract INPUT scopes.
  The SAME shape is legal and means the opposite AFTER `optimize`: `Schedule`'s materializing
  `Unroll` / `Partition` mints and `C_syntax.try_localize_serial_reduce` localize a materialized
  accumulator this way and codegen renders it. That asymmetry is the point — **materialized-accumulator
  localization belongs to codegen's accumulator peel (gh-ocannl-693) and to nothing else**; a second
  route through the virtualizer would restore the gh-639 "whichever schedule happened to run"
  problem. The peel is unconditional as of gh-ocannl-693 — every recognized serial reduction nest is
  localized, not only those whose storage precision the numerics policy wants widened — so ordinary
  lowering now produces this shape at f32 routinely; `test/operations/reduction_accumulator_residency.ml`
  pins it. Hand-built IR in that form still reaches a backend past the optimizer via
  `Ll_test.optimize_scoped` (optimize a scope-free raw twin for the traced store and placements,
  then swap in the scoped `llc`) and `Context.compile ?prelowered`. Pinned by
  `test/operations/scope_over_materialized.ml`.
  Since gh-ocannl-687 the node also RECORDS which side it came from — `Local_scope`'s `mint` field,
  `Inlined_computation` vs `Schedule_minted` — but that flag is deliberately not what decides this
  rejection, and claiming the schedule's provenance does not buy a program past the optimizer
  (pinned in the same test). The mint says which pass BUILT a scope, a durable fact consumers such
  as `Autotune.collect_loops` need; the rejection is about which side of a PARTICULAR `optimize`
  call a program was handed to, which only `input_scope_ids` can answer. Conflating them would let
  hand-built IR label its way back into the silent collapse. When building this shape by hand, spell
  the honest mint anyway — the canonical digest distinguishes the two, so an inlined scope wearing
  the schedule's label would key a different cache entry.
- A node-level "what happened at first touch" flag (`zero_initialized_by_code` and friends) cannot
  soundly drive a PER-OCCURRENCE codegen decision, because nothing clears it across the traversal: a
  guard keyed on it alone collapses `Zero_out; Set; Zero_out` to one zero and drops a `Zero_out`
  inside a `For_loop` on every iteration. The shape that works is per-traversal state — a `seen` set
  cleared at the single reset point (`compile_proc`) plus a positional `~in_loop` threaded through
  the recursion, defaulting to `true` for mutually-recursive callers that don't carry it. When a
  codegen decision consults a `traced_array`-style boolean, ask whether it is node-level or
  occurrence-level; they coincide only at first touch on the linear path.
- The codegen localizer and `Schedule.Privatize` forward a preceding whole-node `Zero_out` (or
  its unconditional covering `Expand_zero` nest) directly into the accumulator via the shared
  `Low_level.zero_seed_candidate` proof (gh-ocannl-1175): `Low_level.affine_accesses` must find exactly one same-cell RMW pair and
  `Affine.covers_box` proves its closing stores cover the whole node (gh-ocannl-821). The zero store
  is dropped only after `try_localize_serial_reduce` actually accepts, every loop that repeats a
  cell is inside that accepted scope (inside `over` for `Privatize`), no enclosing loop is statically dead, and the covering write
  is unconditional. Those clauses are distinct from geometric coverage: an enclosing reduction
  loop can repeat an otherwise covering output nest, a dead loop executes no closing store, and a
  symbolic-extent guard can make only a prefix execute. A localizer/SIMD decline, partial or
  conditional coverage, an opaque effect, or an accumulation with no preceding zero retains the
  original opening value.
- **The same proof drops a GPU routine's zeros when the zero and its accumulation are kernels apart**
  (gh-ocannl-1257, `Low_level.elide_covered_zeros`, gate `gpu_elide_covered_zeros`). The
  accumulation then opens from zero itself: a constant opening, the localized scope form, or the
  opening read guarded by the first iteration of the repeating loops. The scope form must sit where
  code generation would have localized (`peel_accum_nest`, asked with and without the hardware
  levels): guarding the read of an update the localizer would have taken narrows a bf16 cell on
  every step instead of once, which is a different value wherever accumulators widen. A guard is
  placed only where its loops iterate no more times than the node has cells. Guarding the bias and
  layer-norm gradients' whole-batch reductions saved a negligible zero, yet cost the b256
  gpt2_mini_train step 1.2% on CUDA (staging#1113). It runs in `Backends.compile`
  on the final kernel list, after the transform that produced it. So schedule transforms, sketch
  site detection and the schedule cache never see the rewritten accumulations. A pre-fission rewrite
  would hide every gradient contraction from the sketches, and would change the unit lists that
  recorded segmentations replay onto. `?prelowered` compiles are exempt. The forms are
  value-identical, because the opening is `Constant 0.`, which is what `Zero_out` stores. The sign
  of a zero is kept only where the backend compiler honours signed zeros: never fold `0 + x` to
  `x` in OCANNL's own passes, but a compiler under a no-signed-zeros licence (HIP's
  `-ffast-math`) may fold the literal, as it may any other addition of a zero. Trap for its tests: a fresh buffer is usually zero,
  so a dropped seed passes a routine's first run. Run the routine twice and compare the second run
  too. In `zero_elision.ml`, an unseeded guard failed only there.
- **A virtualization candidate under an `If` is rejected, and the guard's position decides which
  arm rejects it** (gh-ocannl-651). `check_and_store_virtual`'s walk sees only the subtree it is
  handed, so a guard INTERIOR to that subtree hits its own `If` arm while a guard ENCLOSING it is
  invisible there — `virtual_llc` threads a `~guarded` flag down its walk and reports it, and both
  paths land on `Non_virtual 142`. Guards are NOT confined to the backend-compile-time launch-extent
  pass: `Assignments.to_low_level` emits interval guards for clamped-window pooling (gh-ocannl-504)
  and extent guards for symbolic extents (gh-ocannl-490), both before virtualization. When adding a
  pass that captures a subtree for later replay, ask what context the subtree was captured FROM —
  a walk of the subtree alone cannot see it.
- **The same question for loops: a candidate is captured at the outermost `For_loop` whose index
  occurs in its assignment indices** (`track_symbol` / `reverse_node_map`), so a reduction loop
  BELOW that point is part of the stored computation (the ordinary `x[t] += a[s]`, priced by
  `virtualize_max_inline_reduction`) while a repetition loop ABOVE it is not — and a symbol-free
  (all-`Fixed_idx`) index map has no capture site at all. Such a candidate is rejected as
  `Non_virtual 147` (gh-ocannl-674); width-1 loops stay exempt, since replaying one iteration once
  is exact. Two arms hide most of this shape and neither is a guarantee: an array reduction
  `x[0] += a[s]` is rejected because the sibling read escapes (`Non_virtual 9`), and an accumulator
  read more than `virtualize_max_visits` times is capped (`1:visit-cap`) — a flippable policy
  prior, decided in `decide_placements` before any legality question is asked.
- **A committed-virtual node has no fallback, so nothing commits one ahead of the walk**
  (gh-ocannl-1690, gh-ocannl-1735): a `Non_virtual` against a node already `Virtual` cannot decline
  to materialization — `Tn.Placements.update` raises "already virtual", at store time and at
  consumption time alike. `decide_placements` therefore only PREFERS a constant scalar virtual: it
  records it in `specialize_proc`'s `tentative_constexpr`, which the caps and the fan-in guard
  treat as decided (the exemption is the point of the rule) while the coverage rule still applies,
  and `specialize_proc` commits the survivors `40:scalar-constexpr` right after `virtual_llc`,
  ahead of the walked-placements snapshot and cleanup. A new early commitment needs the same
  split. Pinned by the `constexpr_*` rows of `test/operations/virtual_rejection_boundary.ml` (a
  guarded setter refused as 142, a flattened read served since gh-ocannl-1898 — a one-cell node
  leaves no consumption-time refusal to pin — and the cap exemption), with executed parity.
- **A fixed-index producer position is guarded only over a proved fallback** (gh-ocannl-1735,
  gh-ocannl-1795). On an extent-1 coordinate a producer `Fixed_idx 0` read at a symbolic index is
  the same cell (`same_unit_cell`, no guard). On a wider one `instantiate_computations` guards
  `k == index`, and where the guard is false the replay's shared local carries on, which is the
  materialized value only if that local is defined: `fixed_guardable` requires a `Zero_out` of the
  node or `union_covers` — the components' writes TOGETHER cover the node, the guarded component
  included (`Affine.union_covers_box`: each write a box over its live loops, a diagonal or strided
  write proving nothing). The proof is about the node, not the read: a union covering only the
  cells read still refuses as 13 (`fixed_component_read_covered_only`), since the read's range is
  unknown where the template is instantiated. Declared-virtual and inherited nodes skip the
  coverage rule, so for them the proof is all there is. A component fixed at a cell other than a
  fixed read's is omitted at that read site before any guard is built (`Omitted`; all omitted is
  still 13). A host-constant literal (`Tn.known_host_constant`) takes neither the union nor the
  omission: its per-cell `Constant_fill` components would replay as a guard per cell, and a later
  `Context.set_values` must reach its readers, so a multi-cell literal read off its own cell stays
  a buffer read (`recompute_on_read`, `test_one_hot_embedding_lookup` pin what that keeps). Rows: `fixed_component_*` and `declared_virtual_union` in
  `virtual_rejection_boundary.ml`.
- **Every guard on a node that skipped the coverage rule needs that proof** (gh-ocannl-1897).
  Equality guards (a repeated symbol `x[i; i]`, an unsolved affine `x[2s]` with its loop kept) and
  range guards (a unit solve, a binding that needs one) fall back on the same shared
  local. For a node `decide_placements` decides, the coverage rule is the proof: a node read where
  its writes do not reach owns a buffer instead. A node already `Virtual` at routine entry —
  declared, or committed by an earlier routine of the lineage, a recompiled step's own nodes
  included — skipped it, so `inline_computation` passes `~coverage_exempt` and
  `instantiate_computations` refuses any guard on such a node without `fallback_defined` as
  `157:guard-fallback-unproven` (the fixed-index guard folds into the same check there). It cannot
  materialize instead — `Virtual -> Never_virtual` is the "already virtual" collision, and an
  inherited node's setters are gone — so the refusal is a `User_error`
  (`declared_uninlinable_rejection`, `inherited_uninlinable_rejection`). The flag is the caller's,
  never `known_virtual` read inside: `instantiate_at_synthetic_read` marks the node it prices
  virtual in its private copy. The proof is the node's, so a guard sound only for the cells read (a
  block whose reads avoid its hole) refuses too. A write that reads the node builds on the invented
  init where the materialized program builds on the entry value (gh-ocannl-1954), so on such a node
  only the plain prefix proves: the components, oldest first, ahead of the first one reading it. Its
  covering the node defines every cell before any accumulation replays, which is what admits an
  update of an inherited virtual and a non-zero neutral's fill ahead of its reduction, recompiles
  included; a whole-union rule refused those. A component that reads the node ends the prefix even
  where it stores first (`dva_mixed_components`, refused though sound). With no guard no init is
  prepended, so an opening read refuses as `158:accumulation-without-init`, where it was an
  undeclared local and a backend compile failure. Mind the fold's order there: `List.rev_filter_map`
  runs a component newest first, though the replay it returns is oldest first, so a decision about
  the replay's order waits until after it. Rows: `dvg_*`, `dva_*` and `inh_*` in
  `virtual_rejection_boundary.ml`; the real `@^^` reduction in `lineage_recompile_step.ml`.
- **The inliner solves index positions pairwise, so a mismatched pair is first put in one
  coordinate layout** (gh-ocannl-1795, gh-ocannl-1898). Solved raw, a `Sub_axis` on one side only
  bound a producer symbol to the run placeholder, which miscomputed in guards, affine reads and
  transposed operand reads (`flattened_read_*` rows). `instantiate_computations` now re-expresses
  such a pair through `Affine.common_layout`: both views coarsened to the grouping either side
  needs, each coordinate written back at the node's rank as a `Sub_axis` run and its composed
  row-major index (same flat cells, now position for position the same coordinates). A flattened
  read of a per-axis producer `x[i; k]` meets `K·i + k` (unit-solved under a range guard, the `i`
  loop kept; a non-unit `3s` is an equality guard over the kept loop); a per-axis read of a
  flattened producer binds the flat symbol to `K·h + e`. The frame is NEVER refined below a side's
  own grouping — splitting `[Sub_axis; 2u + v]` into `[u; v]` would assert a decomposition the
  read does not have, the gh-ocannl-1896 hole — so a multi-term match on normalized coordinates
  still meets the containment rule below (`normalized-wide`/`-contained` in `virtual_affine.ml`).
  Only mismatched pairs are normalized (agreeing layouts have identical frames), on copies:
  `def_args`, which the replay's setters are asserted against, is untouched. Composing also folds
  away what a per-axis reading checks — `x[1; k]` becomes `2 + k`, so its `Fixed_idx` never meets
  `fixed_guardable`. Its guards are answered as any guard is: the coverage rule
  (`flattened_read_past_partial`, `flattened_read_written_cells`), or the gh-ocannl-1897 check on a
  coverage-exempt node (`dvu_flat_*`). A binding without a guard is the
  next bullet's question. Still 13: no common frame (a map of another rank, a `Concat`
  component). Rows: `flattened_read_*`,
  `flattened_producer_*`, `partial_frame`, `trailing_sub_axis_read` in
  `virtual_rejection_boundary.ml`. Range guards bound a solved symbol by its
  loop's `[from_, from_ + width)`, not `[0, width)` (`block_components_offset_*`), and a
  structural affine match (`s + 1` read at `t + 1`) gets one too in a multi-component replay, as a
  pass-1 binding does (`block_components_structural_*`).
- **A binding replays the iteration it names, whether or not the producer's loop has it**
  (gh-ocannl-1953). `needs_range_guard` decides. A multi-component replay always guards. With one
  component the coverage rule is the proof the read stays in the loop (a symbol pinned by its
  position has no other iteration to write the cell), but it proves nothing in two cases: a
  `Zero_out` satisfies it at cells no iteration writes, where the unguarded replay computed the
  producer's expression over the zeros, and a coverage-exempt node skipped it, where the guard then
  asks the fallback proof (157). Both guard unless `call_range` shows the read within the loop's
  `[from_, from_ + width)`, where the guard could never be false: a whole-node zeroed write, the
  usual reduction, stays guard-free, and so does a read inside a partial one.
  `instantiate_at_synthetic_read` passes no range, so it builds the guard, and its pipeline folds
  it against the reader's static ranges where the write sweeps the coordinate (`zeroed-synthetic`).
  Measured reach when this landed: no existing read in those families or the training and autotune
  integrations gained a guard when inlined. Rows: `zeroed-*` in `virtual_affine.ml`, `dvu_{partial,zeroed}_*`,
  `flattened_producer_zero_partial`, `flattened_read_zero_structural` in
  `virtual_rejection_boundary.ml`.
- **A multi-term structural match binds pairwise only within the producer's bounds**
  (gh-ocannl-1896): `x[2s + t]` read at `x[2u + v]` replays the producer at `(u, v)`, which —
  the write map being injective (store-time `51`) — is the iteration writing the read cell exactly
  when it lies in the producer's loops. A `v` ranging wider than `t` reads cells written at
  `(u + 1, v - 2)`, so the pairwise binding miscomputed silently with one component, and a range
  guard on it would suppress cells the component writes. The match therefore requires every
  call-site symbol's range (`~call_range`: the read site's `enclosing` loops, a pricing read's
  `a_loops`, a footprint's fresh loops; `None` for statics, which declines) to lie within its
  producer loop's `[from_, from_ + width)`, with one component or several; otherwise unit solving
  keeps a producer loop under a guard. Rows: `structural-wide*` / `structural-contained-offset` in
  `virtual_affine.ml`, `block_components_two_terms{,_contained}` in
  `virtual_rejection_boundary.ml`.
- **Where a virtualization candidate is refused is readable off its placement PROVENANCE, and four
  phases write into the same table** (gh-ocannl-658, pinned row by row in
  `test/operations/virtual_rejection_boundary.ml`): `decide_placements` applies the heuristic caps
  (`Visit_cap` / uncovered read, `Inline_reduction_cap`, `Inline_fanin_cap`) BEFORE any legality
  question, so a shape capped there may be perfectly inlineable; `check_and_store_virtual` rejects
  at store time and `inline_computation` at consumption time — which codes each one mints is
  derived, not listed: `test/operations/provenance_tag_inventory.expected` groups them by function,
  and refuses a tag minted in two (the consumption-time backstop behind `148:scan-recurrence` is
  `149:scan-at-inline`, so the provenance says which phase spoke) — which is why two setters with
  different index maps as separate statements store fine as components and only fail once a read
  site cannot be served; and `cleanup_virtual_llc` commits a surviving read as `Surviving_read`,
  which is the absence of a rejection rather than one.
  Two properties of the store-time set are worth knowing before you chase one, and BOTH are read
  off pipeline order rather than off the arm. First, some of those arms cannot fire: nothing emits
  `Staged_compilation` (`Non_virtual 8`) today, and the passes minting barriers
  (`Non_virtual 141`), cooperative tiles (`Non_virtual 143`) and dynamic scatters
  (`Non_virtual 144` — `rewrite_one_hot_reductions`) all run after `virtual_llc`. Do not group arms
  by what they match on to predict this — `Scan_loop` (`Non_virtual 148`) and `If`
  (`Non_virtual 142`) are refused by constructor exactly as those are, and both fire on ordinary
  code.
  Second, **`Non_virtual 19` is live, and a note or comment telling you otherwise is stale.**
  `Assignments.lower` runs the algebraic rewrite tier — `Rewrites.apply`, gh-ocannl-483 — BEFORE
  `Low_level.optimize`, and its member `online_softmax` declares its cached probability cell as a
  `Declare_local` (`Online_softmax.hoist`, the consumer-side read hoist), so with that key on the
  arm fires for real; `row_hoisted_local` in the boundary test pins it on a plain hoisted local,
  since what the arm refuses is any `Declare_local` in the captured nest and not that rewrite's
  shape. Two qualifications: `online_softmax` is enabled by the `approximate` profile only —
  `performance` deliberately leaves the algebraic-rewrite gates alone as a numerics axis, and
  `reproducible` pins them off — and the rewrite's OTHER `Declare_local`, inside `emit_normalizer`'s
  scan body, never reaches 19, because an enclosing `Scan_loop` is refused as 148 first. (The
  running max/sum are `Scan_loop.carried` values, not locals.) Claims that
  `hoist_cross_statement_cse` is the only producer of `Declare_local` predate that tier.
  Both 142 and 148 arrive by two routes, and the pair is the general shape: the offending
  construct ENCLOSES the candidate's nest, so it is outside the captured subtree and only the
  caller can report it (`~guarded` / `~in_scan`, decided before the walk starts), or it sits
  INSIDE and the walk's own arm finds it. Same verdict either way — for 148 that one verdict also
  covers a scan the candidate merely reads from or sits beside (gh-ocannl-696).
  Pipeline order is what falsified 19's "cannot fire", and nothing tied the claim's copies
  together. So before moving a pass across `virtual_llc`, or adding one ahead of
  `Low_level.optimize`, read the golden of `dune build
  @test/operations/runtest-provenance_tag_inventory`: it lists every file that names a code — the
  raise-site comments, `docs/lowering_and_inlining.md`, this note, the tests — which is the
  checklist that change lacked (gh-ocannl-1015). Cite a code as `Non_virtual N` or by its tag, and
  any provenance by its tag or as `provenance N` — the spellings the inventory reads, and refuses
  when no source mints what they name (gh-ocannl-1081). `Non_virtual 52` is enforced earlier still (`trace_node_facts`
  raises `invalid_arg` on a `Concat` write index, so the virtualizer's arm never sees one). The
  tags themselves, and how they compose, are the TAG entry below.
- **A placement provenance is a TAG, and which kind it is tells you whether code reads it**
  (gh-ocannl-609): `Tnode.provenance` splits two ways. A decision nothing interrogates is a
  `Site "<code>:<kebab-reason>"` explaining itself — some sixty of those, minted across nine modules
  from the C renderer's storage queries to the scheduler's tile placements. A tag some other code
  reads back is a `provenance` CONSTRUCTOR: `Visit_cap`, `Inline_reduction_cap`,
  `Inline_fanin_cap`, `Read_before_write`, `Scope_local`, `Surviving_read`. The split is a layering
  decision, not a style one — the type lives in `tnode.ml` (that is where `Placements` is), so a
  constructor per site would make the bottom module enumerate the vocabulary of every module above
  it, which is why the field was an unstructured `int` for years in the first place. The rule for a
  new tag: `Site` unless something matches on it.
  - The code is the integer the provenance used to be, so older issues and comments citing
    `Non_virtual 13` or "provenance 39" still resolve. Codes are NOT unique: `176`/`178` each name
    two different schedule sites, told apart by their reasons — collisions that predate
    `provenance_tag_inventory`, which pins them and refuses a new one (gh-ocannl-1081). A new tag
    takes a number its golden does not list; a test's own tag is spelled `N:test-<reason>`, the
    one spelling allowed to reuse a library number.
  - They compose STRUCTURALLY: `default_to_most_local` records
    `Refined (Inline_reduction_cap, Site "432:is-local-materialized-query")`, rendered
    `39:inline-reduction-cap -> 432:is-local-materialized-query` where the retired arithmetic wrote
    `39432`. `Tnode.leading_provenance` walks back to the first tag (what `Ll_test.rejection_code`
    returns); `provenance_to_string` is the only place the ` -> ` separator is spelled.
  - The two readers are `Low_level.is_cap_provenance` and `cap_provenance_setting`, both exhaustive
    matches with no wildcard: a fourth cap does not compile until its policy and its config key are
    stated. `cap_provenance_setting` is what lets the `Local` host-access refusal offer "raise the
    cap" instead of only "materialize the node" — the distinction the integer could not express —
    and the advice is HEDGED, because `decide_placements` records only the first cap that fires
    (each arm is guarded on the placement still being undecided) and the legality rejections run
    afterwards, so raising the named setting can merely expose the next obstacle.
- **A dynamic-gather table (`Get_dynamic`) materializes at the read, and a table declared `Virtual`
  is refused** (gh-ocannl-734, `test/operations/gather_table_placement.ml`): the gathered row is
  only known at runtime, so no computation can be replayed at the read site — `virtual_llc`'s
  `Get_dynamic` arm therefore commits an undecided table `Never_virtual 17` right there, exactly as
  the sibling lane-extract gather does for its packed-uniform counter (`Never_virtual 146`), and a
  table that is already `Virtual` gets a `User_error` naming the node, both readings and
  `set_materialized`. Both `Get_dynamic` arms (`virtual_llc`'s and `cleanup_virtual_llc`'s) carry
  that check, so neither can answer a hand-built gather with a bare provenance collision. Only
  hand-built IR reaches this: `Assignments` lowering emits no `Get_dynamic`, and the pipeline's own
  comes from `rewrite_one_hot_reductions`, downstream of both arms.
- **`virtual_llc` owns every placement decision for a tensor read it leaves behind**
  (gh-ocannl-805, `test/operations/virtual_decision_coverage.ml`). The always-on
  `validate_virtualization_decision_coverage` seam runs immediately after the pass and before
  cleanup: every `Get` / `Get_dynamic` in a statement cleanup will keep, including tensor-buffer
  reads nested in a kept inlined `Local_scope` body, must have an effective placement state (a
  lineage decision or declared intent). The intermediate IR still contains a virtual candidate's
  setter and its self-read, but cleanup drops that statement whole without visiting its RHS, so the
  seam mirrors cleanup's keep/drop decision and excludes it. `Never_virtual` counts -- it is the
  virtualizer's materialization verdict even though cleanup/backend finalization later resolves it
  to `Local` / `On_device`; a scope's `Get_local` and a `Get_merge_buffer` do not, because neither
  reads a tensor context buffer. This is deliberately a second exhaustive walk rather than an
  assertion copied into each read arm: forgetting a future arm fails at the pass boundary, before
  cleanup can make the result depend on whether it walks the undecided node's setter (`Virtual
  151/152`) or reader (`Never_virtual 17`) first.
- **A flip alternative's `fa_recompute_cost` is the cost model's count, not the traced proxy**
  (gh-ocannl-637 Part 2, gh-ocannl-1011): `Low_level.specialize_proc` prices each candidate through
  `Low_level.recompute_pricer`, a seam `Cost_model` registers at initialization (it sits above
  `Low_level`, so a hook is the only way the virtualizer can reach it — a program linking no cost
  model prices everything with the proxy). The price is `Cost_model.analyze` over CODE the inliner
  produced, never a rewrite of it: `Low_level.instantiate_at_synthetic_read` runs
  `inline_computation`'s own core (`instantiate_computations`) at a synthetic read — a fresh symbol
  per axis ranging over the axis, a fixed or static position mirrored — then the SAME
  `post_virtualization_pipeline` `specialize_proc` runs (cleanup through the cross-statement CSE),
  on a placement copy in which the node is virtual. `specialize_proc` hands the pricer a WORLD per
  node: a virtual node's stored computations under the placements as the walk left them (snapshot
  before cleanup — cleanup's commitments are the default reading's, and a flip re-makes them), or,
  for a cap-materialized node (never stored — the cap fires before the walk), `walked_computations`:
  the walk itself (`virtual_llc`) re-run over the node's RAW setter statements in a scratch copy of
  the lineage with the node undecided. The re-run decides everything the flip changes about the
  node's own computation — capture point, refusals, the raw storage of a packed-uniform producer, a
  footprint read its setter no longer hosts retracting — so imitating any of those rules is the
  wrong fix (an imitation of the capture rules missed both the raw storage and the retraction). No nested
  expansion: a stored template already carries the producers inlined when it was stored. The
  scalar price memo (gh-ocannl-1113) belongs to that analysis, bounded to 128 resolved worlds:
  `recompute_price_key` captures canonical computations, referenced nodes' effective placements
  and bounds/host facts, and the mutable static-domain snapshot. Validation and refusal still run
  in each specialization; the memo retains no private placement table. Both exact counts and
  inexact `None` results are reusable, and replacing the registered pricer clears the memo. Three
  traps: (a) pricing must not draw on `Indexing.get_symbol` or `get_scope` — generated code prints
  those counters, so a pricer that drew from them would renumber every later routine's goldens;
  `virtual_llc` and `inline_computation` take `?fresh_symbol`/`?fresh_scope`, and pricing passes the
  negative `pricing_symbol`/`pricing_scope`; (b) exactness is per leg (`rc_flops_approx` /
  `rc_bytes_approx`) and the seam gates on the op count only — a read under a guard's arm makes bytes
  a bound while ops stay exact (a diagonal producer's consistency guard). Pure range guards use
  eager 0/1 multiplication of signed-index comparisons (gh-ocannl-1094) when the scoped simplifier
  proves every affine multiplication and partial sum total over the active loop/static domain.
  Their guard overhead is exact too (affine write positions, multi-setter components); unknown or
  potentially overflowing domains retain `And` and a bound. Negative indices are allowed by the
  proof; it does not assume nonnegative substituted coordinates. `Where` still gates the operand
  read, whose solved index can be negative or out of range on unmatched iterations.
  `Schedule.partition_breakpoints` uses `Low_level.is_pure_index_conjunction` to descend these
  products too; numeric or floating products stay opaque. The shared classifier identifies the
  Boolean representation, while the arithmetic-totality proof remains private to simplification.
  Ordinary short-circuit conjunctions and conditional arm arithmetic remain bounds. Exactly one
  matching
  arm alone cannot prove uniform `&&` overhead: the upper comparison may execute a different
  number of times at different read indices; (c) a flip the store refuses (a scalar reduction's
  operand read escapes the setter it would be captured at, `9:`) is refused by the re-run walk and
  carries the proxy — and, since gh-ocannl-1093, the refusal itself as `fa_refused`, which
  `Autotune.rank_flip_candidates` and `Memory_budget.fit` exclude (the proxy of a scalar reduction
  is its whole extent, so it used to rank first and take budget slots). The walk tags who refused:
  its own store verdict refuses the `Footprint` flip too, a read site the inliner cannot serve
  (`13:`) only the `Inline` one. Stored worlds validate every live read too (gh-ocannl-1152):
  a footprint may retract inside a virtual consumer's storage pass before instantiating that
  read, leaving a cap-materialized node with a retained template and an unservable inline flip
  (`footprint_materialization.ml`, stored-refusal row). Validation uses a per-world placement copy
  because packed-uniform instantiation can materialize its counter. The ordering is otherwise as
  before: `specialize_proc` multiplies the one-read
  count by the instantiations (read multiplicity, or `per_cell` for a footprint reading) and falls
  back to the proxy (reduction extent × multiplicity × transitive fan-in, `fa_modeled = false`) when
  the op count is only a bound. A test that stages a "decoy" or a budget cut on cost ORDER must build
  the order from modeled costs, not from fan-in counts; and a test that wants the visit cap needs a
  read off the reader's write position (a copy-position read is rmw-exempt, so two identity readers
  do not trip it). The relationship pinned is PRICE = EMITTED READ, per shape and through the seam,
  in `test/operations/cost_model_template.ml`.
- **The three caps have a cheaper landing spot than `Never_virtual`: footprint-scoped
  materialization** (gh-ocannl-616, `virtualize_footprint_materialization`, default on). When a cap
  would materialize a node whole but every read of it is an affine sub-image read — a diagonal
  `a[i, i]`, a slice, a cell read under loops its index does not mention — and the readers'
  iteration boxes together hold fewer cells than the node (`footprint_eligibility_query`, the
  profitability rule: both forms pay the same per-instantiation cost, so only the instantiation
  counts compare), `decide_placements` leaves the node undecided and records it in the
  specialization-local `footprint_scoped` table instead; `virtual_llc` then serves each read from a
  fresh scratch node in the `footprint` namespace (`Low_level.footprint_namespace`, how a test tells
  them apart) shaped like the READER's box, filled by a prologue that is the ordinary
  `inline_computation` instantiation at the read's indices under fresh loops. WHERE the prologue
  runs is the semantics: right after the top-level statement of a local producer's last write —
  the position its materialized buffer would have been complete at, so the scratch snapshots
  exactly what that buffer would have held even when a later statement rewrites a template input
  (review round 1 found the reader-side placement diverging from the cap's reading there) — and
  ahead of the reader's statement for an inherited template, the recompute-at-read reading
  inlining gives it, which is why an inherited template reading the reader's own target declines
  (`template_leaves`), why a reader placed before the producer's last write is ineligible, and why nothing but the producer may be written between its first and last write statements (a shared loop rewriting an input after the producer, a tensor OR A LOCAL rewritten between two
  accumulating components — local effects come from the routine's `Affine.statement_effect` rows,
  a local write inside the producer's own statement included, and a dead loop is no writer for it
  or for the prologue's position: the prologue replays every component after the last write;
  rounds 2-5). A read under a SCALAR gate (a `Where` arm, a gated operand) is not an `If` guard:
  `a_gated`, not `a_guarded`, marks it, and such a node is ineligible (round 7: the gate may be
  what keeps an instance in range, and the prologue is unconditional).
  A reader statement that writes a local is ineligible too (an inherited prologue, ahead of the
  statement, would see the local as it was; round 10). An explicit preference exempts its node from
  the caps even where the form cannot serve the reads; the node then inlines. Flip pricing is per
  read CELL in both readings (`per_cell`: the sites' fiber cardinalities, 1 per injective site),
  and an inherited footprint-scoped node offers its `` `Inline`` flip only. A
  consumption-time rejection after a scratch was minted is harmless: cleanup's scope-target
  retraction turns the stranded prologue into an n-cell gather of the buffer
  (`case_rejection_after_footprint`).
  Four more things that are easy to get wrong: (a) the decision is per ROUTINE, not a placement — a
  consumer routine footprint-scopes a node an earlier routine left `Virtual` on the template's own
  reduction extent (`template_facts`, over a SNAPSHOT of the traced store: reading the template
  registers operands the routine never mentions), the gh-573 corner, and the node stays
  `Virtual 152` in the lineage; (b) a read may be footprinted only in a MATERIALIZED
  consumer's setter, in a single-writer top-level statement, unguarded, outside a scan and outside
  a storage pass — anywhere else the read would land in a stored template (replayed by a later
  routine where the scratch does not exist), so the decision RETRACTS at that read: to the cap's
  own materialization for a local producer (recorded in `footprint_retracted`, so the decision
  surface stops offering the `` `Footprint`` flip), to plain inlining for an inherited node or an
  explicit `Context.decide_footprint` preference (exclusive with `decide_inline` per node —
  `prefer_inline` / `prefer_footprint` withdraw each other, so a search trying a node's sibling
  flips does not accumulate them); (c) the scratch's traced entry and its
  `Never_virtual 153` placement are minted in the virtualizer, ahead of `reconcile_traced_store`,
  which would otherwise read the scratch's write-then-read as a fresh node's read-before-write and
  demand it from a prior context — and that placement is the node's declared INTENT
  (`Tn.update_memory_mode`), not a lineage-table entry (gh-ocannl-1481): every autotune candidate
  compiles the base lowering's code over its own compile's fresh `optimize_ctx`
  (`Autotune.compile_candidate_gated`'s `rebase`), whose lowering minted a different scratch, so a
  table entry left the base's scratch with no placement there and `is_materialized_peek` asserted.
  Any node a lowering mints for its own code takes its placement as intent for the same reason
  (`Online_softmax` does); (d) `flip_candidates` is ONE record per node (gh-ocannl-1017):
  `fc_default` (the reading the policy chose) and `fc_alternatives` (the mutually exclusive
  readings a search may take — `` `Materialize``/`` `Inline`` for a footprint-scoped node, only
  `` `Inline`` for an inherited one, `` `Inline``/`` `Footprint`` for a cap-materialized node with
  a strictly smaller footprint), built in `specialize_proc` where both facts are known. The flat
  list it replaced cost nine review rounds across five consumers re-deriving the grouping, the last
  one an inference of the default from which records exist (an inherited footprint-scoped node and
  a cap-materialized one both offer a lone `` `Inline``). Consumers decide a candidate whole — keep
  the default or take one alternative — top-N cuts count candidates, and certainty bounds read
  `fc_default`, never the alternatives' shapes; `Autotune.placement_floor_withheld` is the one
  place a candidate's standing gates the floor.
  Structural probe for "inlined": `count_get` of the node, never `count_scopes` — the simplifier
  collapses a single-assignment scope into its expression. Pinned row by row, with executed parity
  against the materialized and (where the cap alone stands in the way) the inlined reading, by
  `test/operations/footprint_materialization.ml`; end to end through the default GPU schedule (the
  scratch crosses the prologue/reader statement boundary, so fission lifts it `On_device`, the
  two-launch form the issue ships first) by `test/operations/footprint_diagonal_einsum.ml`.
- **A stored template carries only the operands that evaluate** (gh-ocannl-625,
  `Low_level.elide_unevaluated_operands`, applied by `check_and_store_virtual` after its legality
  walk, so acceptance is unchanged): the arm of a `Where` whose condition is decided, and the
  operand behind an `And` / `Or` / `Relu_gate` decided shut, become `elided_operand` (low_level.ml:
  `Constant 0.` annotated at the constant's own `single`, descriptive in every slot, gh-ocannl-1739;
  `virtual_llc`'s decided-`Where` arm and `live_view` write the same form). Every reader of
  `optimize_ctx.computations` — the merge-taint (`computation_reads_merge`) and spliced-read
  (`record_spliced_reads`) scans at a later routine's entry, the fan-in guard, footprint
  eligibility, flip pricing — then sees only reads that execute, so such a read neither taints nor
  registers provenance. The issue proposed storing the
  `simplify_llc` residual instead, which would carry interval decisions taken under the
  producer's loop extents and bounds candidates, and those need not hold at the replay site; the
  consumer's own simplify runs where the template replays, threefry operands included
  (gh-ocannl-1498). Elision changes no evaluated expression, so it is
  exact wherever the template replays; the consumer's simplify still folds the decided `Where` in
  every ordinary position. Decisions must hold wherever the template replays: the loops of the write
  index's symbols do not extend the interval environment (the inliner substitutes the read site's
  index, which need not lie in the producer's extent; a kept reduction loop keeps its extent, so it
  decides), no bounds candidate is consulted (the decision would settle it at store time,
  refusing a later writer, for a template that may never be consumed), and a test is decided only
  if it comes out the same at every precision it may be evaluated at — its slot's precision as the
  renderer reads it (a homogeneous operator's result precision, a `Where` condition's own
  annotation) through a `compute_prec` unknown at store time, so every float precision for a float
  slot (review round 1: `2049 = 2048` is false in single and true in half). The float `0 * x` the issue
  also named is not dead under exact IEEE (`0 * inf` is NaN); under `fp_finite_math`, `Fp_algebra`
  folds it on the raw lowering, ahead of storage. Pinned, with executed parity against the
  materialized reading, by `test/operations/stored_template_elision.ml`.
- **The walk itself skips the dead arm of a decided `Where`** (gh-ocannl-1469, the consumer's
  half of the entry above): `virtual_llc`'s `Where` arm asks the injected `~decide_where`
  (`Low_level.where_condition_decided`, the same every-precision test) and walks only the live arm,
  the dead one becoming `Constant 0.` — so a routine's own constant-condition `Where` around an
  inherited merge-tainted read is no longer refused for code that never runs (`Arg1`/`Arg2`
  already collapsed this way, review round 9 of gh-610/611). Its environment is the static
  indices plus the enclosing loops, minus the `~opaque` ones: the phase-1 capture of a `For_loop`
  candidate makes the candidate's write-index symbols opaque, since that walk IS the stored
  template; any other loop's symbol in stored code gets the store refused (`Non_virtual 9`/`10`),
  so its extent decides soundly. A guard (`If`) narrows nothing here, and `And`/`Or`/`Relu_gate`
  operands are still walked (their decision needs the operator's result precision, which the walk
  does not track) — a merge-tainted read behind a shut gate in the consumer is still refused.
  The capture walk's opacity can still meet such a read in an arm only the candidate's own
  write-index loop decides (gh-ocannl-1514): there the walk does not inline it but records a hit
  against the code's OWNER (the setter whose right-hand side holds it, else the candidate), and a
  store whose own node has a hit is refused as `154:capture-reads-merge` — the candidate
  materializes, and the emit walk, holding the loop's extent, skips the arm or raises the
  merge-splice error if it is live. A sibling setter's hit refuses nothing (consumption drops
  sibling setters), and a node already committed virtual keeps the merge-splice message. Recording
  instead of raising is the point: an aborted walk would strand `spliced_reads` and placement
  side effects that nothing rolls back (`stored_template_elision.ml`, the capture cases).
  gh-ocannl-1551 generalized the hit (`capture_hit`) to the two other inherited reads that refuse a
  routine where they execute: routine-local scratch (`155:capture-reads-local`) and a template that
  cannot be instantiated at the read's indices (`156:capture-read-uninlinable`, reported through
  `inline_computation ?on_inherited_unservable`, which commits nothing).
- **Routine-local scratch an earlier routine placed is judged on the code that executes**
  (gh-ocannl-1620). `Local` is only ever an earlier routine's finalization decision, so a `Local`
  node at `decide_placements` is inherited, with or without a template in the table. It has no
  incoming contents: every read the routine EVALUATES must be strictly covered by the routine's own
  earlier writes. The raw facts cannot say that — `trace_node_facts` and the raw coverage verdicts
  see both `Where` arms, take guards as taken, and exempt copy-position reads — so
  `Low_level.live_view` (the raw code with dead loops dropped and every arm the emit walk decides
  dead made `Constant 0.`; analysis only, never stored) feeds `an_live_reads` and a strict
  `an_live_coverage` (a guarded write counts only where its guards dominate the read, as for spliced
  reads). A condition that reads a buffer is never decided there: the emit walk decides it after
  inlining, which the interval analysis cannot see through. Rules: no live read decides nothing (a
  read-only node in a dead arm is no input, its placement untouched); a covered read executes on the
  routine's scratch; any other read is refused with `inherited_local_read_rejection`, in
  `decide_placements`, ahead of the `Local -> On_device` read-before-write promotion (an
  inconsistent transition). **No stored template may hold such a read**, covered or not: it would
  replay wherever the template does, in later routines too. The capture walk's `155` hit stands for
  covered reads, and outside a capture the emit walk records the setter owning the read
  (`local_dependents`), refusing its per-statement store with the same `155`; a packed-uniform
  producer's RAW capture, which no walk reads, is scanned for such a read (`raw_local_read`). **One
  exception** (gh-ocannl-1645, `carried_local_read`): an owner already committed virtual has no
  materialization fallback, and recompiling a training step in the lineage that ran it is the
  ordinary case — the first compile stores the virtual `w.grad`'s template while `y.grad` is
  undecided, and only its finalization places `y.grad` `Local`. Such an owner's read is admitted
  (recorded as no hit) when this routine reads the node live and strictly covers it; an undecided
  owner still materializes (the cheaper outcome, pushing no refusal onto later routines). The
  uncovered case never reaches it (`decide_placements` refuses it first), so the coverage half only
  adds the read-in-a-dead-arm (`Unread`) exclusion. **Replay is judged on the emitted code**
  (gh-ocannl-1646): a stored template's scratch reads are covered at replay or refused at replay,
  never silently uninitialized. `refuse_uncovered_local_reads`, run by `specialize_proc` on the
  final code just ahead of `reconcile_traced_store`, applies `live_local`'s judgment (over
  `live_view`, strict guard-dominating coverage, per node) to every live read of a `Local` node,
  however it got there: an inherited splice (whose operand the raw code need not mention, so it
  would otherwise only reach the reconcile bookkeeping and read uninitialized scratch), a footprint
  prologue, or the replay of a template this routine just stored. That is what lets the exception
  admit a dependency the lineage did not carry (a declared-virtual consumer's first template, a
  re-store whose inherited template did not read the node): a later routine replaying it covers the
  read or is refused. Gating on `spliced_reads` instead would miss the in-routine replays. Pinned by
  the `carried, ...`, `replayed, ...` and `new dependency` cases and `lc5` of
  `stored_template_elision.ml` (base: the uncovered replays compiled; mutant with the check off:
  they and the later uncovered replays compile again), and by `lineage_recompile_step.ml` (three
  SGD steps in one lineage, bitwise against a `decide_materialized` lineage). The rules above are
  pinned, executed against the materialized reading, by the `local, ...` cases of `stored_template_elision.ml`
  (lineages with and without a stored template; partial, guarded, read-before-write and
  uninitialized-accumulation controls). Boundary: the rules apply to `Local` nodes only — a fresh
  node read only in a dead arm still gets the raw `37:read-only-input` placement (over-materializes,
  values correct).
- **A store replaces the components it supersedes** (gh-ocannl-1653). A template replays as one
  scope local written by each component in turn, oldest first, and `check_and_store_virtual` used to
  prepend unconditionally, so a recompile in the lineage that ran a step added a whole copy of each
  re-stored node's components (measured on `lineage_recompile_step`: 29 → 42 → 55 … stored
  components, the recompiled routine 53 → 124 → 218 … 2950 statements over eleven compiles; a
  single-setter node also gained the range guards of a multi-component one). `supersede` now runs at
  store time, so the recompiled routine's own reads already replay the reduced list: a component
  whose first statement is `Zero_out self` replaces the whole template (older components wrote only
  the local it resets), and a PLAIN component (no read of `self`, no statement-level local) drops
  its newest alpha-equivalent twin (CSE's comparator, `alpha_comparators`) when every component in
  between is plain too: a masked assignment with the same mask and value overwrites its twin, and an
  intervening accumulation must not end up replaying ahead of the local's first write. An
  accumulation is never dropped (two identical ones both contribute). Boundary: a re-trace whose
  template TEXT differs still prepends. A test counting components or reads of a template after a
  same-code re-store must expect the count to stay, not grow (`stored_template_elision`'s carried
  cases read the template's head instead). Guards: `lineage_template_supersession` (counts, sizes,
  executed parity against materialized lineages, and the controls each mutant fails),
  `lineage_recompile_step` (flat totals and routine sizes over eleven compiles).
- **What a capture walk does to code its template drops** (gh-ocannl-1551). Phase 1 of a shared
  loop walks every setter of the body, though the stored template keeps only the candidate's. Two
  divergences from the emit walk were real (`capture_pass_side_effects.ml`): (1) a statement of
  another node stored on its own inside the walk (only under width-1 enclosing loops, else `147`)
  was stored once per phase-1 walk and again by the emit walk — duplicated components, so an
  accumulation replayed twice; such stores are now provisional, restored away when the walk ends
  (a later setter of the same walk still inlines them, so they cannot be skipped); (2) a sibling
  setter's right-hand side was walked with the CANDIDATE's write-index loops opaque, entering arms
  the emitted code never evaluates — committing an unservable local producer `Never_virtual` and
  marking an inherited template's operands spliced; a setter the walk will not store, not the
  candidate's, now decides `Where` with every loop's extent, as the emit walk does. **Stated
  boundary** (gh-ocannl-1620): a node whose own loop nests inside a width-1 capture gets the walk's
  per-statement store (`in_storage_pass` suppresses the nested capture), which lacks its inner
  binder and refuses a loop-varying operand (`9:sibling-escaping-read-index`); that refusal is a
  placement, which restoring the provisional computations does not retract, so the node
  materializes where the emit walk alone would have captured it at its loop. A lost
  virtualization, values correct; `capture_pass_side_effects.ml`'s `nested loop under a width-1
  capture` pins it (`zero y; for i<1 { w[i] := x[i] + 1; for j<4 { y[j] += k[j] } }`). Retracting
  the placement alone is unsound: phase 1's template already holds a raw read of the node, which
  would then read a virtual node with no buffer — a fix needs transactional nested captures. Shared
  loops are hand-built IR (`Ll_test`); `Assignments` lowering gives each assignment its own nest.
- Big-reduction producers are forced `Never_virtual` by `virtualize_max_inline_reduction`
  (default 16) — remember it when a structural expectation assumes inlining.
- Wide-fanin producers are forced `Never_virtual 41` by `virtualize_max_inline_fanin` (default 8,
  gh-573): a node whose fully-inlined computation would load more than that many distinct
  materialized nodes — accumulated through chains of virtual producers, per setter — materializes,
  resetting the fan-in downstream. This is the guard that breaks residual-stream-style running sums
  (per-cell multiplicity passes the visit cap because copy-position reads are rmw-exempt, yet each
  consumer re-sums the whole prefix); a structural expectation assuming a deep all-virtual chain
  must disable it (`Low_level.virtualize_settings.max_inline_fanin <- -1`). Like the other caps it
  is a flippable policy prior, not legality (`test/operations/virtual_chain_fanin.ml`). **The cap's
  bite depends on how many distinct transitive materialized inputs a chain accumulates — which varies
  with depth AND with graph shape at constant depth — so a cap conclusion holds for the graph it was
  measured on and not for a size class.** On gpt2_mini specifically (4 layers, gfx1151,
  `report-gh612-hip.md`), caps 16, 32 and −1 all emit a **135-kernel** arm A, yet only cap 32 is
  actually placement-identical: at cap 16 one node's worth of placement difference appears (the final
  layer norm gains a materialized `n792`) *behind an unchanged kernel count*. Node counts proxy guard
  firings; nothing logs provenance-41 decisions, so they are not firing counts. **Equal fission width can absorb a changed
  materialization decision, so a kernel count cannot establish that a cap did nothing — compare the
  emitted PARAMETER-SIGNATURE multisets and the materialized-node sets** (`benchmarks/gh612_cells.sh
  diff`, which needs only a snapshot). Three distinct levels, and conflating them is easy: a kernel's
  pointer parameters are exactly the materialized nodes it touches, so signature multisets track
  PLACEMENT and are insensitive to the crowned tile; kernel BODIES also move with the tile, so a body
  diff is not evidence of a placement change; and the count of newly materialized NODES is the proxy
  for guard firings, not the count of changed signatures — one materialization changes several
  consumers' parameter lists (on gpt2_mini, cap 8's 16/17 exclusive signatures come from 4 nodes:
  0/1/4/9/23 for caps 32/16/8/4/2). **A zero placement difference does NOT prove the guard was silent,
  so no fan-in bound follows from it**: `decide_placements` assigns provenance 41 only to a node not
  already placed, and `virtual_llc` afterwards rejects inlining for its own legality reasons, so with
  the cap disabled a different mechanism can materialize the same node and yield an identical source.
  The observable statement is all there is, and only at the caps actually swept (2, 4, 8, 16, 32, −1):
  placement differed from cap −1 at 2, 4, 8 and 16, and matched at 32. Nothing is established for
  caps between or above those. Cap 4 beat the default 8 by a
  non-overlapping 5.7% in a block order-balanced in BOTH the searches and the pass-2 replays (5.5%
  was the same six artifacts replayed in an unbalanced order; three replay sets of them spanned
  5.5-6.5%, so an identical schedule varies ~1pp run to run -- all three non-overlapping), and balancing that order matters: a fixed order
  confounds the cap with session position, which was worth ~1.4pp of an apparent 7.1%.
- **The fan-in guard charges a value select as the backend renders it** (gh-ocannl-1775). Where
  `backend_facts.blends_value_selects` holds (cc, HIP), a `Where` that `speculable_select` admits is
  a blend evaluating both arms, so a setter is charged the UNION of the arms' loads (distinct nodes,
  never the sum of sizes); elsewhere (CUDA, Metal) one arm -- the one whose loads are the most
  AFTER transitive expansion (gh-ocannl-1874): an arm loading one virtual node can replay more than
  an arm loading three materialized ones, so a choice on direct loads under-counts. Hence
  `setter_reads` keeps each setter's `read_charge` -- unconditional loads plus every select's arms
  and its `speculable` admission, unresolved -- and `decide_placements` resolves it
  (`resolve_charge`) with its transitive `expand` under the backend's flag; the cached analysis
  stays backend-independent. The flag reaches it only through `Backends.lower_assignments
  ~facts` -- the compile and `Context.lowered_for_decisions` alike; `Low_level.optimize`, and so
  `Ll_test`, default to the short-circuiting reading, so a hand-built test of a select decision says
  which reading it asks for. Admission is judged before inlining at the arms' annotations, while the
  renderer judges it after virtualization at the compute precision (and HIP keeps `?:` over a NaN
  literal): the guard can charge the union where the select ends up branching, never the reverse --
  the materializing direction. The per-setter maximum across components is untouched: each
  component's update sits under an index-only `Where` no backend blends. Known under-counts that
  remain: overlapping components (two bodies run where both guards hold), and several selects on
  one path, where the arm choice is greedy (each select joined to what is already charged), still
  a path some visit runs. Pins: `virtual_chain_fanin` phases 1c, 1d (the arm chosen after
  expansion) and 2b/2c (two-sided on the running backend), `access_fold`, `backend_facts`, and
  `op_table_compile` against the emitted kernels.
- **The fan-in traversal must not enter a node already materialized** (gh-ocannl-1777). A whole
  step's setter-reads graph is cyclic through its parameters (a parameter reads its gradient, which
  reads activations that read the parameter), and the cycle guard memoizes whatever it computed under
  a placeholder, so entering
  them made `inline_fanin` (the proxy price's multiplier) depend on where the `traced_store`
  iteration started: on uid hashes, hence on how many nodes the session minted before. Any
  decision read off a Tnode-keyed table's iteration order has the same trap; equal-cost flip
  candidates are ordered by first access in the raw code, never by uid, and consumers sort stably.
  `test/operations/flip_order_minting.ml` is the shift-the-uids pattern for checking a decision.
- **A cycle of covered reads through nodes still headed for inlining is cut where the fan-in
  traversal enters it, so the traversal runs in program order** (gh-ocannl-1877). Not entering
  materialized nodes removed the cycles through parameters, not every cycle: a node with several
  setters can read, in one of them, a node that reads the cells another of its setters wrote (`a`
  over its first half, `b` from it, `a`'s second half from `b`), with every read covered and neither
  node materialized. The set memoized for the node inside the cut is computed under the entry
  node's placeholder (`b` counts `a` once when entered at `a`, and all of `a`'s inputs when entered
  at `b`), which moved `b` across `virtualize_max_inline_fanin` with the uids. The top-level
  iteration is sorted by `compare_program_order` over `first_access_positions`; what remains is
  that the cut node is under-counted by design, and that within one setter's loads the recursion
  follows the `Set`'s uid order (creation order, which a uid shift keeps). Pin: part 2b of
  `flip_order_minting`. The audit behind it (every hash iteration in `low_level`, `schedule`,
  `autotune` and their neighbours, classified) is the gh-ocannl-1877 issue comment; the other live
  site was `C_syntax.parallel_grid_safe`, which privatized a pool chunk's locals in the order of a
  uid-keyed table: the chunk's declarations and which of two equal-sized tiles kept the stack
  followed the uids (now first-access order; `heap_local_arrays`' two-tile leg).
- **The shift-the-uids check reads what a compile ships, not only the decision surface**
  (gh-ocannl-1877): `flip_order_minting` part 1 compares, across builds minted at different uid
  offsets, the emitted source up to minted identifiers (`Generated.normalize_minted`) and every
  program node's placement with its provenance, for the untuned compile and for the compile from a
  mid-range `Bytes` budget's plan. A dependence planted downstream of the surface (kernel
  parameters in table order) fails the source claims alone; one planted in `decide_placements`
  fails the placement claims too.
- **A candidate whose captured computation contains a `Scan_loop` is refused, and the refusal has
  two doors** (gh-ocannl-696, `test/operations/scan_loop.ml` legs 5 and 5b): `virtual_llc` threads
  `~in_scan` beside `~guarded`, so a per-statement store inside the scan hits `Non_virtual 148` in
  `check_and_store_virtual`, and a capture whose nest contains a scan anywhere — a sibling, or one
  feeding the value through a scope local — hits the same code from the validity walk's
  `Scan_loop` arm. The inline filter's own `Scan_loop` arm is a backstop raising 148, never a drop:
  dropping a value-producing scan returned the pre-scan value (Codex P1, staging#660 round 5). Reads inside the body inline as usual; the state
  node of a carried pair must be declared virtual (`Ll_test.virtualize`; the validator names it
  otherwise) and cleanup commits it `Virtual 16` like a scope local's node. The contract itself —
  one node DECLARED virtual per pair, never accessed as a tensor buffer, ids pairwise distinct,
  rebound by no `Declare_local`/`Local_scope` inside the scan and referenced nowhere outside it, inits free of carried state and of
  the scan index, `next` written exactly once at the body's top level and read only by later
  statements (it is declared without a value), no write of `prev`, no `Staged_compilation` inside,
  and a NON-EMPTY range (a dead scan is refused rather than given a meaning: review rounds 4–8 on
  staging#660 found a fresh walker each round that needed its own "dead scan is a no-op"
  convention, so the class was closed by making the shape unreachable) — is
  `Low_level.validate_scan_loops`, run at both gates like scope purity. Codegen's three per-local
  censuses (rng precision, accumulator residency, controlled accumulators) walk the implicit
  `prev = init` / `prev = next` assignments as the `Set_local`s they render as
  (`C_syntax.scan_implicit_set_locals`), and the carried ids are pinned to their node's STORAGE
  precision in `scope_prec_of` (`carried_state_scope_ids`, rng-carve-out precedence): a half state
  rounds every step — `scan_loop.ml` leg 6b holds 2048 through six +1 steps where a single state
  reaches 2054. Build scans through `Ll_test.carry`/`scan`/`prev`/`next`/`set_next`.
- **A host read of a virtual node recompiles, and its classifier reads the same code the inliner
  replays** (gh-ocannl-777). `Context.observability` walks the lineage's stored computations without
  compiling, through `Low_level.replay_facts ~self`: it keeps exactly the statements
  `instantiate_computations`' `loop_proc` keeps (dead loops and a shared-loop body's sibling setters
  dropped) and visits every operand -- never `evaluation_reads_of_proc`, the cost collector, which
  skips merge reads and charges one `Where` arm. A new statement or scalar constructor, or a change to
  what instantiation keeps, must update `replay_facts` too: its match arms are exhaustive for that
  reason. A materialized node with a buffer ends the walk (it shields a `Local` dependency behind
  it); answers rank `Unobservable_local` > `Unobservable_aliased` (a buffer the `buffer_aliasing`
  planner shares, which `to_host` and linking refuse) > `Unobservable_merge` > `Launch_bound` >
  `Needs_context`.
  `Launch_bound` exists because `check_and_store_virtual` walks with `env_dom = static_indices`, so a
  template can keep a launch symbol a bindings-free recompute cannot supply; it is conservative (the
  inliner could bind a write-index occurrence). `Needs_context` for a virtual node's LEAF is
  unreachable through ordinary linking -- `verify_prior_context` requires every leaf at the link
  that stores the template, and descendants inherit buffers -- so tests pin it on a materialized
  root. `Context.to_host` compiles `proxy =: tn` (plain Assignments, identity projections, `proxy` in
  namespace `observe` with `tn`'s unpadded dims and padding), runs it, transfers, releases the child
  and retires its private ledger records (`rollback_execution` plus the routine name) -- a release
  failing before it frees is queued and retried by the next recompute, since no caller holds the
  child; a padded
  read's fringe is refilled host-side with the padding value. Printers ask the query first and show
  `observability_tag`; a failure after a readable answer propagates (no catch-all). A printer names
  every node it shows up front (`to_dag` renders twice: a recording pass, then the real one) and
  reads the readable ones through `Context.to_host_all`, whose `Recomputable` nodes share ONE
  throwaway routine, one sequenced copy per node (gh-ocannl-1737). `print_tree`'s `depth` shapes
  only the layout (levels below it are drawn as an indented tree, values included), so no depth
  narrows the read. Count routines with
  `Context.backend_compiles` against a unit measured by a single-node read, never an assumed 1: a
  Metal routine also counts its link-time library compile. Guards: `observability_query`,
  `recompute_on_read` (bitwise parity against a materialized run, ledger and pool counts, an
  injected transfer failure), `for_print_test`, `print_tree_one_recompute`.
