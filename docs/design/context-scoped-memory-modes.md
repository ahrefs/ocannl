# Context-scoped memory modes: split observability facts from placement decisions

**Date**: 2026-07-05
**Status**: Implemented 2026-07-05 (core split). `Tnode.memory_mode` is now declared
intent only (monotone, side-effect free to read; the tnode-level forcing family was
removed); decisions land in `Tnode.Placements` tables riding `Low_level.optimize_ctx`,
forked per backend `compile` (`Low_level.copy_optimize_ctx`) so sibling candidate
compiles are hermetic. All pipeline force sites (low_level virtualizer, c_syntax
codegen/`ptr_params` incl. the restrict/alias assert, `validate_parallel`, schedule
tiles and peeks, backends allocation/verify, `Assignments.context_nodes` and slice-alias
eligibility) resolve per-lineage; `Context.placements` exposes the resolution.
Same day, the `is_observed` intent bit landed as `Tnode.is_observable` /
`set_observable` (monotone-up, side-effect free), replacing the last intent-level
`Never_virtual` requester — parameter gradients (provenance 26). Enforcement is the
minimal pair of guards in `Placements.default_to_most_local` / `is_virtual_force`: an
observable node never resolves `Local`, and an observable node still **undecided at a
forcing point** materializes instead of defaulting `Virtual` (an undecided node has no
tracked computation — the virtualizer commits every stored candidate at cleanup — so
`Virtual` would be unobservable in practice; this is the "observable nodes without
computations are materialized" heuristic). Observable nodes **with** tracked
computations stay virtualizable, and cross-routine recomputation genuinely serves them:
`test/operations/observable_grads.ml` pins that a raw split grad-update/sgd flow works
with the gradient `Virtual` in both lineages (sgd inlines the gradient's computation),
while the fused flow virtualizes the gradient the old regime forced `Local` (the
unobservable class — the old default actively contradicted observation intent for small
gradients). Still not done: candidate masking; scoped constancy;
`Materialize`/`Devirtualize` optops; recomputation-backed *host reads* of virtual nodes
outside `Train.printf`'s for-print proxies (e.g. `printf_tree ~with_grad` of a fused
lineage's virtual gradient shows `<not-in-context>` rather than recomputing).

## Motivation

`Tnode.memory_mode` conflates two things that have different natural scopes:

1. **Observability facts** — is the node read by the host, shared across routines or
   streams, persistent, effectively constant? These are properties of how the *program*
   (the set of comps plus host code) uses the node. They are inherently relative to a
   routine set — which today is approximated by "the whole process" via a global mutable
   field with provenance-tracked settlement.
2. **Placement decisions** — inlined ([Virtual]), routine-local scratch ([Local]), or a
   device buffer ([On_device]/hosted). These are per-compilation choices *constrained* by
   the observability facts. The contract already hints at the split: nothing says a
   [Local] node cannot be inlined — Virtual-vs-Local is a pure performance choice for any
   node that is unobservable, yet today it is settled once, globally, per node.

Now that contexts are values (`Context.t` carries the compile frontier, execution ledger,
and per-context buffers), observability questions always have a context in hand — and the
buffer side is *already* per-context: whether a node has storage is a fact about a
context, not about the node. The decision layer just doesn't match yet. (gh-ocannl-164
added a small illustration from the other direction: the 32-byte within-pool offset
padding landed entirely inside `allocate_delta`, which is already per-context — the
buffer machinery keeps demonstrating that per-context is its natural scope.)

## The forcing function: autotuning isolation

The BEAM/search plan (schedule-ir-optops §7, gh-ocannl-261) compiles sibling candidates
from one frontier. With a global `memory_mode`, a candidate compile that forces or settles
a mode **poisons its sibling candidates** through shared mutable state — mode updates are
process-global, and conflicting settlements raise. Contexts-as-values was supposed to make
candidate compiles independent; the memory-mode global is the remaining shared-state leak
(the same class of problem as `Tnode.bounds_state`, which the interval work handled by
pinning at lowering time). Per-context placement resolution is what actually makes
sibling candidate compiles hermetic.

Secondary benefits: per-context specialization (the same tensor virtual inside a fused
inference kernel but device-resident in the training lineage), and making Virtual↔Local
officially a per-compilation choice — the "unobservable middle" where a future
`Materialize`/`Devirtualize` optop could operate without touching global state.

## Sketch: what stays on `Tnode.t`, and why

Move the memory-mode lattice wholesale under the (optimization) context: the resolution
(Virtual | Local | On_device | hosted-materialization) becomes per-compilation-lineage
state — a `placements : placement Map.M(Tnode).t` on `Context.t` threaded through
compile (matching how buffers already live per-context), or on the routine/`optimized`
result keyed by the context that produced it. The `is_virtual_force` /
`is_materialized_force ~provenance` call sites (c_syntax, validate, schedule, backend
pool/transfer logic, host-init upload paths) become lookups against the resolution
carried by the compile in progress — and crucially, the *forcing* (settlement) those
sites perform today moves into the per-context resolver; reads elsewhere become
side-effect free.

What remains on `Tnode.t` is a deliberately small residue, in two categories:

1. **Semantic facts that define program meaning** — same category as shapes and
   precision, out of scope for context-scoping entirely:
   - `alias_of` / `is_alias`: defines the *write semantics* (a write through the view is
     a write to the parent's sub-range). Not a placement decision; settled
     deterministically at assignments lowering, before schedule-level candidates diverge.
   - Op-support constraints: a primitive that cannot consume or produce an inlined
     operand states a fact about the operation, not about any lineage's preference.
   - Global constancy (semantic-adjacent): "no routine in the program writes this node"
     is a fact about the program's use of the node, like alias-ness, and it anchors
     shared infrastructure — `constant_buffer_cache` dedups constant pools **per device
     across contexts**, so per-lineage divergence on read-only-ness would let one lineage
     write a buffer another lineage registered as shared-constant. Keep it on `Tnode.t`.
     Optimization-relevant *scoped* constancy — constant within a routine or lineage
     (invariant hoisting, per-lineage constant pooling) — is a distinct concept to
     bifurcate as per-context state when a consumer needs it, not a weakening of the
     global bit.

2. **Declarative intent bits protecting cross-context shared substrates**:
   - `is_observed` (host-observation intent): the host ndarray mirror is a **per-tnode
     singleton** — one logical tensor has one host buffer regardless of how many lineages
     compile it; it is what `Host_inits` and ndarray literals feed, what `set_values`
     writes, what printing reads. Configuration that guards a shared substrate has to key
     on the thing that is shared.

   Intent bits are **monotone-upward only** (a node can start being observed, never
   stop) and **side-effect free to read**. This is the crisp version of the motivation:
   today's settlement raises come from *decisions* colliding, and decisions collide
   because they are two-sided. One-directional facts cannot conflict, so keeping them
   global does not reintroduce the poisoning problem the split is meant to fix.

Notably absent from the residue: cross-routine and cross-stream sharing. These are
*derivable* — the lineage's frontier discovers a second reader at its compile and the
device/stream layer sees cross-stream use — so they need no declared bit at all.

### Observation is not materialization: recomputation, and the `Never_virtual` drift

`is_observed` intent does **not** map to "must materialize". The design intent is that
virtual nodes remain observable *via recomputation* — the inlined computation can be
re-run on demand to produce values — **except when they depend on [Local] nodes**
(routine-scoped scratch is gone after the routine, unobservable by contract, and that
unobservability is contagious to virtual computations reading it). So the placement
constraint `is_observed` imposes is only: do not resolve the node (or its recomputation
closure) into the Local-dependent unobservable class. Materialization is one way to
serve observation; recomputation is another; the resolver picks.

`Never_virtual` has diverged from its original meaning ("someone will read this, so it
must not be inlined away") — under the recomputation intent, observability alone never
requires `Never_virtual`. It decomposes as:

- observation-flavored uses → subsumed by `is_observed` + recomputation (no
  materialization requirement);
- op-support uses ("this primitive cannot take a virtual operand") → stay tnode-level
  semantic constraints (category 1 above);
- performance pinning ("keep this materialized because the schedule wants it") →
  per-context, the `Materialize`/`Devirtualize` optop territory.

**Audit item**: some call sites may still assume the old meaning (observable ⇒
never-virtual). Sweep `Never_virtual` requests and consumers during the refactor and
classify each into the three buckets; sites found relying on the old meaning are bugs
against the recomputation intent, not constraints to preserve.

### Candidate masking

To serve the autotuning forcing function, contexts can **mask** intent: candidate
compiles in a search serve no host reads, so they compile with observation suppressed
(and hence maximal freedom to virtualize/localize) and stay hermetic; the winning
schedule recompiles unmasked. Masking is sound precisely because intent reads are
side-effect free — a masked compile leaves no trace on the tnode.

### Enforcement

Enforcement moves entirely per-context, on the bounds-settlement model: observing a node
through a lineage consults that lineage's resolution. Virtual resolves are served by
recomputation; the raise is reserved for what recomputation cannot serve — [Local]
placements and Local-dependent virtual closures. Intent strengthened *after* a lineage
compiled does not invalidate that lineage; observation through it may fail while new
compiles honor the stronger intent.

## Honest caveats

- **Temporal monotonicity does not go away.** A later compile in the same lineage can
  still discover a reader of a node an earlier compile already resolved as Local — the
  frontier only knows the past, so late materialization must still raise (as bounds
  settlement does). Context scoping shrinks the blast radius to a lineage; it does not
  make placement retroactively revisable.
- **Wide touch surface**: the provenance-int force sites are scattered (c_syntax 333–338,
  validate 160, schedule 172/175/176, backend pool/transfer logic, slice aliasing,
  host-init upload paths). This is a refactor to schedule deliberately, not to fold into
  a feature PR — in particular, `Privatize` did not need it (its tiles are born
  per-kernel with fresh ids, so global `Local` mode is correct for them).
- **gh-ocannl-164 raised the stakes at the c_syntax site** (landed 2026-07-05, PR #92):
  the `ptr_params` force-site cluster now also emits the `restrict` qualifier and asserts
  `not (Tn.is_alias tn)` per parameter. Both must ride along with the migrated lookup,
  and a placement-resolution bug that ever produced overlapping kernel parameters is now
  miscompile-grade (restrict makes it UB) rather than a redundant pointer. Also one more
  hand-built-IR test pins global-mode behavior and joins the refactor's sweep:
  `arrayjit/test/test_vectorized_codegen.ml` (hand-settles `On_device`/`Local` to drive
  `compile_proc` directly, like `test_zero_out_codegen.ml`).

## Relations

[schedule-ir-optops](schedule-ir-optops.md) (§7 search needs hermetic sibling compiles),
[gh-ocannl-261](../../archive/docs/proposals/gh-ocannl-261.md) (autotuning consumer),
[memory-mode-streamlined; Local contract] (the 2026-07-03 lattice this splits),
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (the bounds-settlement
precedent for lineage-scoped monotonic facts),
[gh-ocannl-164](gh-ocannl-164.md) (restrict on kernel parameters — the alias assert and
qualifier that the migrated `ptr_params` lookup must preserve).
