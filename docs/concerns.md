# OCANNL's concerns

What the project is working on, organised by concern rather than by release.
[GitHub milestones](https://github.com/ahrefs/ocannl/milestones) say by when an issue is due, at the
latest; this map says what the work is about. [CHANGES.md](../CHANGES.md) lists what shipped, and
[release-history.md](release-history.md) records what each release set out to do.

The map names no issues and carries no dates or counts, so it changes only when the shape of the
project does. Each branch's status is reviewed at release prep. Some nodes belong to more than one
branch; [Shared nodes](#shared-nodes) lists them, and [Cross-cutting axes](#cross-cutting-axes) lists
the questions that recur everywhere.

## What the system means

### Shape inference

Shape schemes make a tensor function reusable at different shapes and report a contradictory
application at its call site. They settled as a check, not a registration path. The open question
is how far the row solver can be trusted: reading it closely after that work raised suspected
defects that are not all confirmed by execution yet. Reference:
[shape_inference.md](shape_inference.md).

- Solver correctness: window and stride unification, concatenation, closing unresolved rows,
  row-capture product ties, projection derivation
- Solver cost: constraint emission growth, checkpointing the open frontier, scheme laziness
- Refusals and diagnostics: padded and unsized kernels, scheme rendering, origin tracking

### Surface language

`%op` and `%cd` are stable in daily use; what remains is typing corners and rules worth fixing once,
such as operand evaluation order. Reference: [syntax_extensions.md](syntax_extensions.md).

- Syntax-extension typing and evaluation-order rules
- The ppx's dependency surface
- DSL and leaf-type API hygiene

### Lineage and lifetime

Routine composition is checked per lineage, and a virtual node reads by recomputation. What remains
sits at the edges of those rules: who owns gradient zeroing once a consumer is reclaimed, and what a
host write means for a node the next forward pass recomputes. Reference:
[tensors_and_contexts.md](tensors_and_contexts.md).

- Ownership across contexts: fragments, gradient zeroing, reclaimed consumers
- Host-visible values vs recomputation
- Update rules on computed nodes
- Session-level heap growth
- Composing contexts

### Numerics

Numerics-changing work goes behind the `approximate` profile, with `reproducible` pinning the
defaults. Mixed precision is the active front: some rounding defects are witnessed by execution and
others found by reading. Reference: [precision_inference.md](precision_inference.md).

- Policy: profiles, what may change default results, placement neutrality
- Mixed and integer precision: double rounding, collapses, folds vs C promotion, cast rendering
- Fast math and contraction per backend
- Special-value agreement across backends
- Builtins: host and kernel parity, parameter guards

## How it compiles

### Pipeline architecture

The pipeline is a fixed phase sequence, and a fact found late cannot feed an earlier opportunity.
What makes the order rigid is the placement lattice, not the pass list; the loop that exists is one
level up, in placement tuning. Splitting off arrayjit as a standalone loop-nest compiler is decided
and follows the mid-November release. Reference: [compilation_manifesto.md](compilation_manifesto.md).

- Phase order vs feedback: the placement lattice and sealing
- Layering: arrayjit below the cut, the Context tier above it
- One legality model in place of three checkers

### Lowering and virtualization soundness

Value-changing passes need executed parity against a materialized run, not only structural tests.
The next step is an exhaustive oracle for the inliner, so later changes run under it. References:
[lowering_and_inlining.md](lowering_and_inlining.md),
[virtualization notes](agent-notes/virtualization-and-inlining.md).

- Inliner correctness and its oracle
- IR well-formedness gates
- Determinism: dependence on creation or evaluation order
- Observability of writes: the blind spots of filling fresh outputs
- Placement priors and proofs

### Rewrites

The algebraic rewrite tier runs ahead of the analyses, one member behind each config key, with online
softmax as the exemplar. Its open question applies to the whole tier: what a rewrite owes a node
someone has asked to observe.

- What a rewrite owes an observed node
- Pattern-directed folds

### Recurrence and scans

`Scan_loop` is a real IR construct, reached by the online-softmax fold and by prefix-scan recognition
under `approximate`. The high-level story is not there yet: which operations produce scans, under
what licence to reorder their combine, and how state carried through memory is declared.

- Producers: recognition, primitives, or a first-class construct
- Combine licences and carried-through-memory state
- Parallel and backward forms

### Parallel legality

A store's own threads are checked for separation. Reasoning across statements and barrier regions is
done per schedule, with a conservative refusal, rather than by a general analysis.

- Separation of stores
- Barrier-region dependence
- Control-flow uniformity

## How it runs fast

### Schedule search

Measured search works, and much of the recent effort makes it cheaper and better informed. Most of a
search's progress arrives late, from recombination, which gives its budget rules little to act on.
Reference: [schedules_and_autotuning.md](schedules_and_autotuning.md).

- Economy: budgets, beam, composites, rounds, abandonment, timing windows
- Ranking: the cost model's fill, traffic and compute terms
- Seeding: candidate admission, pre-filters, the model's default pick

### Untuned defaults

Most users never run a search, so the untuned path carries real weight. GPU training defaults now
seed register-tiled plans on CUDA and HIP; Metal waits for its measurement. CPU thresholds are being
reconciled onto one work measure.

- GPU seeded defaults
- CPU thresholds and the work measure

### Schedule vocabulary

The transformations search can choose from. Convolutions and padded sites are where the vocabulary
still has gaps, and Winograd's backward leg was tried and reverted on memory and compile cost.

- Tiling, staging and geometry
- Epilogue fusion and split-reduce
- Padding-related forms
- Convolution-specific transforms

### Persistence and identity of results

Tuned results are cached per concrete device, with the numerics policy in the key. The known gap is
provenance: driver and runtime versions that can move timings are not all observable.

- Schedule and placement stores
- Device and toolchain keys
- Replay and invalidation

### Backends

All backends generate through a shared driver, with differences made explicit. Reference:
[anatomy_of_a_backend.md](anatomy_of_a_backend.md).

- GPU execution model: lanes, all-reduce, tensor-core formats, async copy
- GPU memory: local reservation, launch-time backoff, host registration, transfers, eviction,
  zero-copy
- CPU: vector-file assumptions, register tiles, workgroup-staging rendering
- Vendor toolchain behaviour
- Binding libraries: cudajit, ocaml-metal, hipjit
- Hardware beyond the current fleet, which waits on access, and training across networked machines

## How we know

### Test truthfulness

The aim is that a green result means what it says. `Verdict` makes a failed claim fail the test; the
remaining work is claims that pass with nothing checked and bypasses that leave no record.

- Vacuity: claims that pass with nothing checked
- Verdict states: skip, bypass, gated, owned aggregation
- Controls: negative controls, mutation runs, forced declines

### Goldens

Goldens still pay for changes that mean nothing, chiefly node ids and source locations.

- Noise: ids, `line:col`
- Portable output

### Repository scans

Scans enforce much of the repository's discipline, so their false refusals and blind spots matter.
Most of them still resolve names by spelling rather than scope; one shared scope model is the
direction. Reference: [build and test notes](agent-notes/build-and-test.md).

- Name resolution: spelling, scope or types
- Corpus membership
- Scan-case validity
- Emitter attribution
- Diagnostic and refusal identity

### Test support

Shared helpers that keep each test from restating what another part of the system owns.

- IR builders
- Skip lists turned into legs
- Shared readers and helpers

### Measurement

Performance claims rest on checked-in [reports](../benchmarks/) and record their nulls. Many
comparisons are owed but need an exclusive window on a quiet machine.

- Protocol: timing modes, drift control, exclusivity
- Attribution: per-kernel profiling, step diagnostics
- Instruments: readers, calibration, hardware ceilings
- Coverage: benchmark legs, unowned performance leads, deferred comparisons
- Toolchain fingerprinting

### Value parity across platforms

Hosted CI has no GPU, so backend coverage comes from the cross-machine sweep and from legs run on the
boxes themselves.

- Every-backend value legs
- Compiler and architecture census matrices

## What it is for

The consumers that test the compiler against real demands. They pull on everything above.

- Model demos and reproductions
- Sampling and top-k
- Experiment and inference tooling
- Standalone ArrayJIT

## How work gets done

### Repository tooling

- Test runner: locks, queued launches, slots, run records
- Scope reach: which tests a change touches, and what a pre-push run covers
- Harness outcomes: leg vocabulary, failure reading, signal handling
- CI: workflow structure, shards, dependency pinning, durations
- Sweep: telling the environment from a defect
- Reds and flakes
- Platform portability: Windows paths, line endings and DLLs; macOS

### Agent machinery

Parallel development runs on [ludics](https://github.com/lukstafi/ludics), which owns the landing
gate, execution reservations, fleet management and wave coordination, with
[flotilla](https://github.com/lukstafi/flotilla) as the fleet's dashboard. Their work is tracked in
those repositories.

## Cross-cutting axes

- **Claimed vs actual coverage:** does a pass mean something was checked? This holds for tests, scans,
  harnesses and verdict reads alike.
- **One owner, derive don't restate:** a fact kept in two places drifts, whether it is legality,
  a constant in prose or a list of claims.
- **Inference vs reading the fact:** a value guessed from a proxy, a capped read taken as the whole
  answer, a finding reasoned out but not executed.
- **Identity:** what keys a thing, such as cache entries, node ids, fixture digests, diagnostic sites,
  processes and commits.
- **Lifetime and ownership:** who releases fragments, buffers, candidates, locks and temporary state.
- **Refusal quality:** silently wrong, falsely refused or usefully refused. The same trade-off appears
  in the solver, the scans and the backends.
- **Bounded reader vs real parser:** state a reader's boundary, or keep chasing spellings.
- **Determinism:** order dependence in the compiler, timing dependence in claims, id churn in goldens.
- **Hardware and platform variation:** unified vs discrete memory, vector widths, three GPU vendors,
  three operating systems.

## Shared nodes

- **Cache identity:** schedule search, numerics policy, pipeline sealing and measurement.
- **Legality:** parallel legality, search moves and backend barrier rules.
- **`Scan_loop`:** the IR, numerics licences, attention performance and top-k.
- **Fast math and special values:** numerics, backends and test claims.
- **Measurement windows:** measurement, and execution reservations.
- **Bypass and skip aggregation:** test truthfulness, and the sweep.
- **Windows:** portability defects, and rebooting a box to witness them.
