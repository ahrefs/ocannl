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

2. **The test asserts convergence against `epsilon = 0.1`.** After training, the
   final epoch loss is compared against the threshold `epsilon = 0.1`
   (load-bearing — see the epsilon note: it is small enough that a marginally-
   converged host like the 0.23 one must retry, not pass on the first attempt). A
   converged run prints a single **stable** signal line (no numeric loss, no
   attempt index — both are host-dependent), e.g.
   `moons_demo: converged (final epoch loss < epsilon)`.

3. **The run is robust to per-host nondeterminism via a 3-seed retry, then a
   hyperparameter escalation.** Resolution order (2026-06-20, user):
   - **Tier 1 — 3 preselected seeds.** Retry the *entire* training run across
     **3 fixed, preselected seed candidates** (a deterministic seed list, not
     RNG-at-runtime). Succeed as soon as any of the 3 yields final loss
     `< epsilon`.
   - **Tier 2 — hyperparameter escalation.** If none of the 3 preselected seeds
     clears `epsilon = 0.1`, the demo's hyperparameters are too marginal — choose
     stronger ones (learning rate, model size, epoch count, …) so the 3-seed
     check converges robustly below `0.1` on **every** CI host. This is primarily
     a **development-time** tuning mandate: the shipped test must converge within
     the 3 preselected seeds on all of ubuntu/macos/windows. (A runtime
     hyperparameter-sweep fallback is permissible but not required; the simpler,
     preferred outcome is well-chosen fixed hyperparameters that make 3 seeds
     sufficient.)
   - **Failure.** The test **fails (non-zero exit) only if**, with the chosen
     hyperparameters, none of the 3 preselected seeds converges below `epsilon` —
     which, after Tier-2 tuning, should not happen on a healthy host. Both the
     failure exit code and the stdout/`.expected` diff enforce correctness.

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
   let epsilon = 0.1 in           (* load-bearing — see "epsilon" note below *)
   let seeds = [ 1; 2; 3 ] in     (* 3 preselected, fixed candidates *)
   let converged =
     List.find_map seeds ~f:(fun seed ->
       let final = train_once ~seed () in
       if Float.(final < epsilon) then Some seed else None)
   in
   match converged with
   | Some _ -> Stdio.printf "moons_demo: converged (final epoch loss < %.2g)\n%!" epsilon
   | None ->
       (* Tier-2: if this fires on a CI host, the hyperparameters are too
          marginal — strengthen lr / model size / epochs so the 3 preselected
          seeds clear epsilon=0.1 on every host (development-time mandate). *)
       Stdio.eprintf "moons_demo: FAILED to converge in %d seeds\n%!" (List.length seeds);
       Stdlib.exit 1
   ```
   The printed success line contains only `epsilon` (a compile-time constant), not
   the loss or the winning seed/attempt — so it is identical on every host. The
   seed list is **preselected and fixed** (deterministic across hosts), so the
   only variation between hosts is FP-contraction noise, which the 3 candidates +
   well-chosen hyperparameters absorb.

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

### Note on `epsilon` (resolved 2026-06-20, user: `0.1`, load-bearing)

`epoch_loss` is the **sum** of per-step `scalar_loss` over the final epoch. A
well-trained run sums to ~`0.00` locally and ~`0.23` on the CI host. **The user
chose `epsilon = 0.1` (load-bearing), not the generous `0.5`.** This is the
literal "retry until below epsilon" framing: at `0.1` the `0.23` host does **not**
pass on its first seed — it must retry across the preselected seeds until one
converges to ~`0.00`. The earlier risk of "a host where no seed clears `0.1` in
3 tries" is now resolved by the **Tier-2 escalation** (AC 3): the answer to "3
seeds insufficient" is **not** to loosen `epsilon` back to `0.5`, but to
**strengthen the hyperparameters** (learning rate, model size, epochs) so the
demo converges well below `0.1` and 3 preselected seeds reliably suffice on every
host. The retry is therefore genuinely load-bearing (it fires for the known
`0.23` host), and a marginal model is fixed by tuning, not by a lax threshold.

## Risks / Open items

- **`epsilon` choice** — resolved to `0.1` (load-bearing), with Tier-2
  hyperparameter escalation as the answer to "3 seeds insufficient" (above). No
  longer an open judgment call.
- **Cost of retries.** Each `train_once` is the full ~6400-step run (a few
  seconds). At `epsilon = 0.1` the `0.23` host (and any marginal host) **will**
  retry, so the common cost is up to 3× runtime on those hosts — acceptable for a
  non-`slow` test. After Tier-2 tuning the model should clear `0.1` on the first
  seed on most hosts, bringing the cost back down. A hard failure (all 3 seeds
  miss) means the hyperparameters still need strengthening, not a flaky test.
- **lr-plot stability assumption.** The learning-rate schedule is a pure function
  of the step index (no training-data dependence) and already matches across
  hosts in the current golden, so it is kept as the stable regression anchor. If
  it ever proves host-sensitive, drop it from the golden too.
