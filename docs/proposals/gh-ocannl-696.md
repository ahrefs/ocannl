# Loop-carried recurrences: the `Scan_loop` construct

Issue: [#696](https://github.com/ahrefs/ocannl/issues/696) (split out of #483, whose task 1 this is)

**Date**: 2026-09-07
**Status**: **landed** (v1) — the construct, its pipeline contracts, C-family rendering on every
backend, and the hand-built-IR test `test/operations/scan_loop.ml`. The design comment on the
issue records the wider loop taxonomy this is the first constructor of; the sections below say
what v1 commits to and what it deliberately leaves open.

## The decision

The IR route, not the structural encoding. Encoding a recurrence as mutually-indexed scope locals
inside a plain `For_loop` leaves every analysis believing the loop's iterations are independent:
`has_accumulation` sees no self-read (the update reads `prev` and writes `next`), so a hardware
annotation or a vectorization pragma could be asserted over a body that is serial by
construction, and the virtualizer could replay one iteration of the "loop" at a read site of its
output. The declaration has to be first-class so that the analyses can refuse by matching, not by
convention.

## The construct

```ocaml
| Scan_loop of {
    index : Indexing.symbol; from_ : int; to_ : int;
    direction : scan_direction;         (* Forward | Backward *)
    carried : carried list;             (* { prev; next; init } — scope ids over one virtual node *)
    body : t;
  }
```

Semantics: every `prev` takes its `init` once, before the first iteration; for each value of
`index` over `from_ .. to_` in `direction`, the body runs reading the previous state through
`Get_local prev` and producing the next through `Set_local next`; after the body every `prev`
takes its `next` simultaneously. The rotation is phi-style, so old and new values coexist inside
one body — online softmax needs the previous *and* the new running max in one expression. The
state is scalars only, of unbounded arity; a body that wants a trajectory or a final value writes
it to a tensor node itself. A dead range is a no-op, like a dead `For_loop`: its inits are
unobservable once nothing may reference the carried locals outside the scan.

Rotation rather than in-place update is the load-bearing choice: it is SSA form for carried
state, which is what keeps body rewrites (CSE, hoisting within the body) order-insensitive and
what a future adjoint generator reads per scalar instead of reconstructing from clobbered memory.

## Contracts, and where each is enforced

- **Well-formedness** (`Low_level.validate_scan_loops`, at both pipeline gates like scope purity:
  `optimize_proc` on the way in, `C_syntax.compile_proc` on the way out): a carried pair names one
  node declared virtual, with ids pairwise distinct across the list and rebound by no `Declare_local`
  or `Local_scope` inside the scan and referenced nowhere outside it (the locals live in the scan's
  block); the state node is accessed as a tensor buffer nowhere in the routine; inits read no carried state and do not mention the scan index; each `next` is written
  exactly once, as a top-level statement of the body, and read only by later statements; nothing
  writes a `prev`.
- **Placement** (`check_and_store_virtual`, `Non_virtual 148`): a virtualization candidate whose
  captured computation contains a scan is refused — one written inside the body (the `~in_scan`
  flag threaded through `virtual_llc` for the per-statement store), one fed by a scan through a
  scope local, or one merely enclosing a sibling scan (the validity walk meets the scan at an
  enclosing loop's capture). Per-cell replay of a scan is what the construct forbids, and the
  inline filter could not keep one without re-minting its carried locals per replay. Reads inside
  the body inline as usual. The state node is committed
  `Virtual` at cleanup like a scope local's, so it reaches no routine parameter list.
- **Schedule opacity**: `Schedule.find_loops_env` and `rewrite_loop` do not enter a scan, so an op
  naming the scan's index or a loop inside its body declines with the usual "no For_loop with
  index" refusal (locate and rewrite agree, per gh-ocannl-668). `Pad` guards the whole scan; `Stage`,
  `Privatize` and `Split_reduce` refuse a scan in their region by name. Enclosing loops keep their
  full menu, since the state is per-iteration scratch of the enclosing loop — the default
  annotator descends into scan bodies for exactly that reason.
- **Footprint**: the scan index is an ordinary affine loop symbol for `loop_bounds`,
  `affine_accesses` (inits at path `Stmt 0`, body at `Stmt 1`), interval analysis and the cost
  model. A scan's accesses stay affine and dense; only the order its values are produced in is
  serial.
- **Digest identity**: `Canonical_render` renders the inits, the direction and the carried pairs,
  so a fresh lowering digests identically while `Forward`/`Backward` and a swapped pair differ.

## Rendering

One rendering in the `C_syntax` functor, hence identical on cc, CUDA, HIP and Metal: the carried
pair as two locals declared and initialized ahead of a serial `for` (counting down for
`Backward`) whose body ends with the rotation. Declarations, inits and the rotation go through
the `Declare_local` / `Set_local` arms, so the state's type, precision conversions and runtime
logging are a scope local's.

## Deliberately out of v1

- The rest of the loop taxonomy from the issue's design comment — `Forall` as a sibling
  constructor, `Scatter_accum`, the closed polymorphic-variant mapping rows per constructor.
- An associative-combine license on the scan (what makes flash-decoding and top-k merges
  parallel), and carried-through-memory nodes (`carried_nodes`). Both are additive fields.
- Carried-state residency as a *choice*: the state is pinned to its node's storage precision in
  codegen (`carried_state_scope_ids`, precedence like the rng carve-out), so a half state rounds
  per step and an fp32 state under fp16 compute stays wide; the arithmetic feeding an update runs
  at compute precision. Letting a cost model or a policy widen the state is not offered yet.
- Any high-level surface: no `Assignments` or `Operation` produces a scan yet. Cumulative ops and
  top-k as primitives, and a first-class scan in `Assignments`, are a separate issue.
- The #483 rewrite itself, which this unblocks.

## Acceptance criteria

- [x] `Ir.Low_level.Scan_loop` with printing (both human printers, the canonical rendering) and
      traversal in every walker — exhaustive matches by the compiler, catch-alls audited with
      OCaml's fragile-match warning.
- [x] `optimize`'s contract for the state scalars and for scan-body writes.
- [x] `C_syntax` rendering on the C-family backends, exercised on cc and Metal locally.
- [x] Schedule-op contract: refuse loudly, with the standard no-such-loop refusal.
- [x] Hand-built cumsum (both directions), the online-softmax pair, a per-row scan under a Grid
      loop, the placement contract, both validation gates, and digest identity, all in
      `test/operations/scan_loop.ml`.
