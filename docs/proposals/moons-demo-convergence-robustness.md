# Proposal: moons_demo CI robustness — convergence check + seed retry (drop the exact-loss golden)

**Task:** task-850e1b37
**Project:** OCANNL (`test/training/moons_demo`)
**Effort:** small

## Goal

Stop `test/training/moons_demo` from reddening OCANNL CI's `dune runtest` for a
purely environmental reason. The demo trains a half-moons MLP to
`epoch loss=0.00` locally under `sync_cc` but the GitHub ubuntu runner renders
`0.23`; because the test diffs the exact printed `%.2f` loss against a `0.00`
golden, it fails on every PR (and on master), masking genuine PR-side failures.

The fix follows the resolved direction (2026-06-20, user): **robustify the run,
do not defend an exact golden.** The test should assert that training
*demonstrably converges* — not that it reproduces a precise loss trajectory
across hosts. Exact cross-host reproducibility is explicitly **not** the aim.

Out of scope: the C-codegen portability lever (`-march=native` / `FP_CONTRACT
OFF`) — that is reserved for a separate "highly-deterministic, C-backend-only"
test category, tracked as **task-72bae3c8**. This proposal must not touch
`cc_backend.ml` defaults or `test/config/ocannl_config` arch flags. moons_demo is
a general, multi-backend test and gets run-robustness instead.

## Root cause (established, not re-litigated here)

Host-specific C codegen: `sync_cc` builds the generated C with `-O3
-march=native` and the host compiler (clang locally, gcc on CI) and emits no
`#pragma STDC FP_CONTRACT OFF`. FMA-contraction and SIMD differences perturb each
SGD step; over `80 epochs × ~80 steps ≈ 6400` updates the trajectories separate
enough that the printed `epoch loss=%.2f` lands at `0.00` on one host and `0.23`
on another. Backend selection is identical (both run `sync_cc`) and the PRNG
(threefry4x32) is platform-stable, so neither is the divergence source. The
in-source comment already flags this: *"for as-yet unknown reason, this test can
lead to different results on different versions of dependencies."*

## Acceptance Criteria

1. **moons_demo no longer pins the exact `%.2f` epoch loss as a golden.** The
   per-step `Epoch=…, epoch loss=%.2f` lines (the host-dependent output) are
   removed from both the program's stdout and `moons_demo.expected`. No numeric
   loss value is diffed against a golden.

2. **The test asserts convergence.** After training, the final epoch loss is
   compared against a small threshold `epsilon`; a converged run prints a single
   **stable** signal line (no numeric loss, no attempt index — both are
   host-dependent), e.g. `moons_demo: converged (final epoch loss < epsilon)`.

3. **The run is robust to per-host nondeterminism via seed retry.** If an attempt
   does not converge (final loss ≥ `epsilon`), the test retries the *entire*
   training run with a **different seed**, up to **K = 3** attempts total. It
   succeeds as soon as any attempt converges, and **fails (non-zero exit) only if
   none of the K attempts converge.** Both the failure exit code and the
   stdout/`.expected` diff enforce correctness.

4. **CI is green for the right reason.** `dune runtest` (which builds, runs
   `moons_demo.exe`, and diffs stdout against `moons_demo.expected`) passes for
   `moons_demo` on every host in the ubuntu/macos/windows × OCaml 5.3 matrix,
   because the only diffed output is now host-stable (the convergence line plus
   the deterministic learning-rate plot and `%op` debug-name lines). Evidence:
   wrapper-pipeline — `eval $(opam env) && dune runtest` exits 0 with
   `moons_demo` included, and the CI matrix is green on the PR.

5. **No production/codegen behavior changes.** No edits to `cc_backend.ml`
   compiler-flag defaults, to `test/config/ocannl_config`, or to other tests
   (`bigram`, `embedding_ids`, `fsm_transformer`, `transformer_names`). The
   change is confined to `test/training/moons_demo.ml` and
   `test/training/moons_demo.expected`.

## Approach

The test is a `(test (name moons_demo) …)` stanza in `test/training/dune`, which
uses dune's implicit run-exe-and-diff-`.expected` convention. Keep that harness;
make the program's stdout host-stable.

1. **Extract a single-run function.** Refactor `main` so the per-run work becomes
   `let train_once ~seed () : float`, returning the final epoch's accumulated
   loss. It must, per attempt: set `Utils.settings.fixed_state_for_init <- Some
   seed; Tensor.unsafe_reinitialize ()`, build the dataset with
   `Dataprep.Half_moons.Config.{ noise_range = 0.1; seed = Some seed }`, build the
   MLP / loss / sgd routines, run the `epochs × steps` loop, and return
   `!epoch_loss` from the last epoch. (Everything currently in `main` from the
   `seed` binding through the epoch loop moves inside, parameterized by `seed`.)

2. **Convergence + retry driver.** Replace the single training run with:
   ```
   let epsilon = 0.5 in           (* see "epsilon" note below *)
   let max_attempts = 3 in
   let seeds = [ 1; 2; 3 ] in     (* distinct per attempt *)
   let converged =
     List.find_map (List.take seeds max_attempts) ~f:(fun seed ->
       let final = train_once ~seed () in
       if Float.(final < epsilon) then Some seed else None)
   in
   match converged with
   | Some _ -> Stdio.printf "moons_demo: converged (final epoch loss < %.2g)\n%!" epsilon
   | None ->
       Stdio.eprintf "moons_demo: FAILED to converge in %d attempts\n%!" max_attempts;
       Stdlib.exit 1
   ```
   The printed success line contains only `epsilon` (a compile-time constant), not
   the loss or the winning seed/attempt — so it is identical on every host.

3. **Stop printing the per-step loss.** Remove the in-loop
   `Stdio.printf "Epoch=%d … epoch loss=%.2f\n"` and the progress-dot prints (or
   drop just the host-dependent loss field and the dots). The deterministic
   pieces that the test still exercises as a regression check — the learning-rate
   plot and the two `%op` debug-name lines (`mlp_result's name: …`,
   `(mlp moons_input) name: …`) — stay as-is and remain in the golden, since they
   are seed- and host-independent. (The `losses`/`log_losses` accumulators that
   feed the currently-commented-out loss plots can stay or be dropped; they are
   not printed.)

4. **Trim `moons_demo.expected`** to exactly the new stable stdout: the config
   log-level preamble, the `moons_demo: converged …` line, the learning-rate
   plot, and the two debug-name lines. Remove the entire `Epoch=… epoch loss=…`
   block and the `....` progress run.

5. **Regenerate and verify** with `dune runtest --auto-promote` locally under
   `sync_cc`, then confirm the CI matrix is green.

### Note on `epsilon` (the one tunable to settle in review)

`epoch_loss` is the **sum** of per-step `scalar_loss` over the final epoch. A
well-trained run sums to ~`0.00` locally and ~`0.23` on the CI host — both
represent a demonstrably-converged classifier. A recommended `epsilon = 0.5`
treats both as converged (so CI goes green on the first attempt on every host)
while still catching a genuinely stuck run (initial sums are ~18–21), with the
seed retry as insurance against a pathological initialization. A tighter
`epsilon = 0.1` would instead force the `0.23` host to retry until it hits a
seed that converges to ~`0.00`; this is closer to the literal "retry until below
epsilon" framing but risks a host where no seed clears `0.1` in 3 tries. The
recommendation is the generous `0.5` (aligns with "demonstrably converges"); the
exact value is the coder/reviewer's call.

## Risks / Open items

- **`epsilon` choice** (above) — the single judgment call; recommended `0.5`.
- **Cost of retries.** Each `train_once` is the full ~6400-step run (a few
  seconds). With the generous `epsilon`, healthy hosts converge on attempt 1, so
  the retry cost is only paid on a genuinely flaky run. Worst case is 3× runtime
  before a hard failure — acceptable for a non-`slow` test.
- **lr-plot stability assumption.** The learning-rate schedule is a pure function
  of the step index (no training-data dependence) and already matches across
  hosts in the current golden, so it is kept as the stable regression anchor. If
  it ever proves host-sensitive, drop it from the golden too.
