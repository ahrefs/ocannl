# Bring up + test the OCANNL CUDA backend on minipc-wsl

## Goal

OCANNL's CUDA backend now **builds and the pool-allocator path works** on the
federation's only CUDA host (minipc-wsl, RTX 3050 Ti, CUDA 12.8, cudajit
region-API commit pinned locally) — that landed via the pool-allocator region
rewrite (ocannl-staging PR #68, merge `0614f397`), whose own AC verification
already exercised build-clean + the pool-allocator runtest on the device.

Two gaps remain, and this task closes them:

1. **Broader test sweep.** Only the pool-allocator unit test has actually run on
   the device. The full `OCANNL_BACKEND=cuda dune runtest` suite (operations,
   einsum, training, ppx) has *not* been run on a real GPU. This task runs it,
   enumerates and triages every break beyond the pool-allocator path, fixes the
   mechanical ones in scope, and files the design-involving ones.

2. **Reproducibility gap.** The committed tree declares `cudajit` only as a
   `depopt (>= 0.7.0)` with **no `pin-depends`**, yet `cuda_backend.ml` now
   consumes the *unreleased* cudajit region API (non-owning `Deviceptr.region`
   view + device-side byte-offset copy/memset/kernel-arg primitives). A fresh
   clone + `opam install` would therefore fail to build the CUDA backend. The
   GitHub pin currently exists only as a minipc-wsl-local opam pin. This task
   commits that pin to `arrayjit.opam.template` so the CUDA backend builds
   reproducibly.

This is a verification/triage umbrella, not a feature task: the heavy lifting
(making the backend build + the core allocator path work) already merged. It is
also the first real `gpu:nvidia` slot launch end-to-end (controller→worker
intent stamp + local launch); treat the deferred AC 4 of task-7eab5162 as
likely-already-satisfied rather than a fresh goal.

## Acceptance Criteria

1. **CUDA build is clean.** On minipc-wsl, with the worktree fetched to current
   `origin/master` and cudajit pinned to the region-API commit,
   `OCANNL_BACKEND=cuda dune build @check` (or `dune build`) completes with no
   errors. Evidence: the build command and its exit-0 output.

2. **`sync_cc` baseline established first.** Before the CUDA sweep, record the
   set of tests that already fail at the same HEAD under
   `OCANNL_BACKEND=sync_cc dune runtest` on minipc-wsl. Pre-existing,
   non-CUDA-specific failures present in this baseline are **out of scope** for
   this task. Evidence: the `sync_cc` runtest output (the baseline failure set,
   empty or otherwise).

3. **CUDA suite introduces no new failures vs. the `sync_cc` baseline.**
   `OCANNL_BACKEND=cuda dune runtest` on minipc-wsl is green **modulo**:
   - the **slow** tests (already gated behind the dune `(alias slow)` and thus
     not run by plain `dune runtest` — the conv/training demos
     `cifar_conv`, `circles_conv`, `mlp_names`, `mlp_bn_names`, etc.), and
   - any test already failing in the AC-2 `sync_cc` baseline at the same HEAD.

   Net: the CUDA sweep must introduce **no new failures** relative to the
   `sync_cc` baseline (slow-excluded). Evidence: the `cuda` runtest output, and
   a diff/comparison of its failure set against the AC-2 baseline.

4. **Residual breaks are dispositioned per the agreed heuristic.** Each break the
   CUDA sweep surfaces beyond the known pool-allocator path is classified and
   handled:
   - **Mechanical / golden re-bless / trivial** → fixed **in scope** in this PR.
     A CUDA-specific golden that legitimately changed under the allocator rewrite
     (different emitted addressing in a `.cu`, etc.) is re-blessed here; the
     existing CUDA goldens (`test/operations/test_where_precision.cu.expected`,
     `top_down_prec.cu.expected`, `zero_out_local_decl.cu.expected`,
     `micrograd_demo_logging-cuda-0-0.log.expected`) are confirmed-matching or
     re-blessed with the diff shown in the PR.
   - **Behavioral / design-involving** (e.g. a genuine on-device numerical or
     correctness regression needing investigation) → **filed** as a follow-up
     task or GitHub issue, not fixed here.

   Evidence: an enumeration of residual breaks with each one's disposition
   (fixed-here vs filed, with the follow-up link).

5. **cudajit GitHub pin is committed.** `arrayjit.opam.template` gains a
   `pin-depends` entry pinning `cudajit.dev` to the ocaml-cudajit region-API
   commit (`git+https://github.com/lukstafi/ocaml-cudajit.git#<commit>`), so a
   fresh clone builds the CUDA backend reproducibly. The pin lives in the
   **`.opam.template`**, not the dune-regenerated `arrayjit.opam` (per the
   dataprep precedent in `neural_nets_lib.opam.template`); the regenerated
   `arrayjit.opam` is allowed to update as a build artefact of
   `generate_opam_files`. Evidence: the template diff and the regenerated
   `.opam` pin-depends block.

6. **Single-GPU caveats are reported, not claimed as passes.** Paths that cannot
   execute on a single-GPU host (`memcpy_peer` / cross-device / data-parallel
   peer copy) are reported as **not-exercised** rather than passed. Evidence: a
   note in the PR/retro listing the not-exercised arms.

## Context

### Test backend selection

Tests do not hard-code a backend; they read the `OCANNL_BACKEND` env var. In
`test/operations/dune`, `test/einsum/dune`, and `test/training/dune` every rule
declares `(env_var OCANNL_BACKEND)` and resolves the backend name via
`test/operations/config/ocannl_read_config.ml` (`--read=backend`, which calls
`Utils.get_global_arg ~arg_name:"backend"`). Running the CUDA suite means
`OCANNL_BACKEND=cuda dune runtest`; the default (env unset) is the host C
backend, which mac-studio CI already runs green.

Valid backend names are registered in `arrayjit/lib/backends.ml`
(`fresh_backend`): `sync_cc`, `cuda`, `metal`. So the AC-2 baseline backend is
`sync_cc` and the sweep backend is `cuda`.

### Slow tests are already alias-gated

The slow conv/training demos are already declared behind `(alias slow)` in
`test/training/dune` (see the `cifar_conv` / `mlp_names` / `circles_conv` /
`mlp_bn_names` rules, each using `no-infer` so plain `dune build`/`runtest` does
not run them — only `dune build @slow` does). So AC-3's slow exclusion is
satisfied by the existing dune structure with no extra flags: plain
`dune runtest` already skips them. The worker should *not* invoke `@slow` on the
RTX 3050 Ti.

### CUDA-specific goldens to confirm or re-bless

Existing CUDA-specific golden files:
`test/operations/test_where_precision.cu.expected`,
`test/operations/top_down_prec.cu.expected`,
`test/operations/zero_out_local_decl.cu.expected`,
`test/operations/micrograd_demo_logging-cuda-0-0.log.expected`. Hex-address
noise is already scrubbed by `tools/minised.exe` in the dune rules, so address
churn should not cause spurious diffs. Distinguish "expected diff to re-bless"
(addressing legitimately changed by the allocator rewrite → AC-4 mechanical)
from "real regression" (→ AC-4 behavioral, file it).

### Why the demos exercise sub-region addressing broadly

`arrayjit/lib/backends.ml` distinguishes `allocate` (offset 0) from
`allocate_delta` (bump-packed non-zero offsets); the training/operations tests
allocate many pooled tnodes via the bump-packing path, so non-zero region
offsets are exercised well beyond the single pool-allocator unit test. The
rewritten surface under test lives in `arrayjit/lib/cuda_backend.ml`
(`Slab.resolve_pool` returning the slab base, `memset_zero` taking `~offset`,
the `Tensor_at` region-view sites, `from_host`/`to_host`, `device_to_device`).

### cudajit pin

The cudajit region API was added in ocaml-cudajit
(`https://github.com/lukstafi/ocaml-cudajit.git`): commit `ee69a0a` added the
non-owning `Deviceptr.region` view + offset params, and PR #11 merge `fb2b552`
added the offset test coverage. `fb2b552` is the appropriate pin target — it is
the region-API tip that the merged pool-allocator proposal
(`docs/proposals/cuda-pool-allocator-region-addressing.md`) references as
carrying the consumed API. The pin is added to `arrayjit.opam.template`'s
existing `pin-depends` block (which currently pins only `ppx_minidebug` and
`notty-community`), as a sibling entry:

```
["cudajit.dev" "git+https://github.com/lukstafi/ocaml-cudajit.git#fb2b552"]
```

Then regenerate with `dune build` (`generate_opam_files true` rewrites
`arrayjit.opam` from the template). Per MEMORY, edit only the `.opam.template`,
never the wiped-on-rebuild `.opam` by hand.

> Note: the in-progress **task-06f7a4cb** lands a partial-data-integrity
> (`host_offset`/length-semantics) memcpy fix in ocaml-cudajit. The task's launch
> gate notes a preference to pin a cudajit commit that already includes it, for
> pin-coherence. If task-06f7a4cb has merged into ocaml-cudajit `main` by the
> time this runs, prefer the post-fix tip over `fb2b552`; otherwise `fb2b552`
> (the region-API tip) is correct. Confirm the chosen commit is reachable from
> the cudajit clone actually installed on minipc-wsl
> (`git merge-base --is-ancestor <commit> HEAD`).

### Worker / deploy hygiene (pre-build checklist)

Per MEMORY (`reference_worker_deploy_stale_ref_trap`, `remote_slot_recovery`):
- `git fetch` minipc-wsl's local `~/ocannl-staging` to current `origin/master`
  before building — a stale checkout would rebuild the *old* backend (verify
  PR #68 merge `0614f397` is an ancestor of the built HEAD via
  `git merge-base --is-ancestor 0614f397 HEAD`).
- The local cudajit clone/pin must carry the region-API commit
  (`git merge-base --is-ancestor <pin-commit> HEAD` in `~/ocaml-cudajit`).
- Re-apply the minipc-wsl on-stop jq PATH patch after any `ludics init` on the
  worker (per `reference_minipc_onstop_jq_patch`).

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. On minipc-wsl: fetch `~/ocannl-staging` to `origin/master`, confirm `0614f397`
   ancestry; confirm the cudajit install carries the region-API commit.
2. Build clean: `OCANNL_BACKEND=cuda dune build @check` (AC-1).
3. Run the `sync_cc` baseline: `OCANNL_BACKEND=sync_cc dune runtest`, record the
   failure set (AC-2).
4. Run the CUDA sweep: `OCANNL_BACKEND=cuda dune runtest` (plain — slow alias
   auto-excluded), diff its failure set against the baseline (AC-3).
5. Triage residuals per the AC-4 heuristic: re-bless legitimate CUDA golden
   diffs in scope; file behavioral regressions as follow-ups.
6. Commit the cudajit `pin-depends` to `arrayjit.opam.template` and regenerate
   `arrayjit.opam` via `dune build` (AC-5).
7. Note single-GPU not-exercised arms (`memcpy_peer`/cross-device) (AC-6).

## Scope

**In scope:** the full `dune runtest` CUDA sweep (operations + einsum + training
+ ppx, slow-excluded), the `sync_cc` baseline, mechanical/golden-rebless fixes,
the committed cudajit `pin-depends`, and reporting of not-exercised single-GPU
arms.

**Out of scope:**
- Slow conv/training demos (`@slow` alias) — RTX 3050 Ti, deliberately skipped.
- Tests already red at the same HEAD under `sync_cc` (pre-existing non-CUDA
  failures).
- Behavioral / design-involving on-device regressions — *filed* as follow-ups,
  not fixed here.
- `memcpy_peer` / cross-device correctness — not exercisable on a single-GPU
  host; reported as not-exercised.
- Performance — single-threaded CUDA kernels (`grid_dim=1, block_dim=1`) make
  perf irrelevant; only build + correctness are in scope.

**Dependencies / relations:** descends from PR #68 (task-6abfb6a9, merged) and
the cudajit region-API work (task-66a3bbff, task-55a11fa3, both done). Pin
coherence relates to task-06f7a4cb (cudajit partial-data-integrity fix). Closes
the deferred AC 4 of task-7eab5162 (first real `gpu:nvidia` launch) if not
already claimed by task-6abfb6a9's slot run. Must run on **minipc-wsl** (sole
CUDA host); serialized against other remote orchestrations on that host.
