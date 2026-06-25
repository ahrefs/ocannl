# Fix broken test suite: centered-init (#73) + PDSL-retire (#70) fallout

**Task:** task-059ebb82 (subtask of task-bb30d0be)
**Project:** OCANNL (`lukstafi/ocannl-staging:master`)
**Effort:** medium

## Goal

Restore a green `dune runtest` on `lukstafi/ocannl-staging:master`. Three PRs
merged 2026-06-25 left the suite red:

- **#70** `f9f5ed10` "Refactor inline parameter initialization semantics"
  (retires PDSL; inline `%op` param init becomes forward-only via
  `Prohibit_grad`) — creates extra IR nodes during param construction, so
  node-ids / shape-ids (`sh_id`, `id`) renumber and some param tensors gain a
  `*._` label prefix.
- **#73** `a5f2c273` "Center default param init" — changes `default_param_init`
  from `uniform1` (uniform `[0,1)`, mean 0.5) to a centered, scaled
  `default_uniform1_param_init` producing `0.5 * (uniform1 − 0.5)` = uniform
  `[-0.25, 0.25)` (mean 0). Any test printing initialized param values or
  downstream training output drifts.
- **#74** `d071ebf7` (test-only embedded-init regression test) — confirmed to
  cause no behavioral change to any failure below.

`dune runtest` produces **10 expect-test diffs** (all benign — promote) plus
**one real regression**: `test/training/moons_demo` no longer converges
(final loss ~0.3 → ~17). This proposal directs both the mechanical promotions
and the one judgment call.

## The (a)-vs-(b) decision on `moons_demo` (resolved: **(b) adapt the test**)

Task Q1 (resolved 2026-06-25) instructs the worker to first inspect the #73
diff and choose: **(a)** fix the centered initializer if it is *scale-wrong*,
else **(b)** adapt `moons_demo` to the new default — and to **document the
reasoning in the proposal**. Inspection of `a5f2c273` makes the answer clearly
**(b)**. Reasoning:

1. **The centered init is the intentional, standard-practice default — not a
   defect.** #73 does not invent an arbitrary range; it *promotes into the
   global default the exact recentering that two transformer tests already
   performed by hand*. Both `test/training/fsm_transformer.ml` and
   `test/training/transformer_names.ml` previously ran a post-init host-side
   loop `vals.(i) <- 0.5 *. (v -. 0.5)` with the comment: *"OCANNL's default
   uniform1 init produces all-positive weights; through the transformer's
   Q·Kᵀ attention scores this causes extreme values and exp overflow. Centered
   initialization (e.g. xavier/normal) is standard for transformers but not yet
   available as a built-in default_param_init."* #73 deletes both workarounds
   because the new default now does that recentering for everyone. Centering a
   param init to mean 0 is what every framework (Xavier/He/normal) does; the
   *old* mean-0.5 init was the anomaly.

2. **The new range is not scale-wrong by the user's own yardstick.** Q1's
   example of *scale-wrong* is a too-**wide** range (`uniform(-1,1)` rather than
   a sound `uniform(-0.5,0.5)`). #73's actual range is `[-0.25, 0.25)` — even
   **narrower** than the user's "sound" example, i.e. firmly on the
   conservative side, not the too-wide side. There is no over-wide-variance
   defect to fix.

3. **The author already applied the (b) treatment to the other training tests
   and simply missed `moons_demo`.** In the same commit, `fsm_transformer.ml`
   was retuned for the new default: `fixed_state_for_init` seed `3 → 14`,
   learning-rate scale `1.0 → 0.5`, and the first-epoch loss limit `16.0 →
   16.2`; `transformer_names.expected` was updated. `moons_demo` (a hinge-loss
   half-moons MLP) was the one training test not re-tuned in that pass.

**Mechanistic note for the coder (why moons_demo specifically broke).** The
centered default lowers the weights' second moment from `E[w²] = 1/12 + 0.5² =
0.333` (old `uniform[0,1)`) to `(0.5)²/12 = 0.0208` (new `uniform[-0.25,0.25)`)
— ~16× less forward-signal energy per layer, with the mean shifted off the
positive offset the relu MLP previously rode on. `moons_demo`'s SVM-style margin
loss `relu(1 − y·f(x))` needs `|f(x)| ≈ 1` to separate classes; with the
lower-energy centered init the output starts near 0 and the `lr = 0.1` schedule
no longer escapes the all-loss-≈1 regime within 80 epochs. This is a
*per-recipe sensitivity of one test*, not a global defect — the transformer
training tests pass under the new default (after their retune), and centering is
the correct default. The remedy is to give `moons_demo` an initialization /
hyperparameters appropriate to its recipe, exactly as the conv training tests
already do (`cifar_conv.ml`, `circles_conv.ml`, `mnist_conv.ml` each set
`TDSL.default_param_init := NTDSL.xavier ~scale_sq:0.06 TDSL.O.uniform1` at the
top of their run).

**Pre-existing marginality (orthogonal, but bounds the AC).** Even *before* #73,
none of seeds 1/2/3 dipped under `epsilon = 0.1` on this Mac (best `0.227`). So
restoring old-init behavior alone would not make the test pass here — the
adaptation must make `moons_demo` *genuinely* converge below `epsilon` (Tier-2
"strengthen the recipe, don't loosen epsilon" mandate from the prior
`moons-demo-convergence-robustness.md` proposal, task-850e1b37).

## Acceptance Criteria

1. **The 10 benign expect-test diffs are promoted.** After the fix, the
   following `.expected` files reflect the new (verified-sane) output and
   `dune runtest` shows no diff for them:
   - `test/operations/attention_test.expected` (sh_id/id renumber — #70)
   - `test/operations/decoder_only_test.expected` (sh_id renumber — #70)
   - `test/operations/layer_norm_test.expected` (sh_id renumber — #70)
   - `test/operations/rope_test.expected` (sh_id renumber — #70)
   - `test/operations/transformer_test.expected` (sh_id renumber — #70)
   - `test/operations/test_param_shape_error.expected` (node-id + `*._` label + value drift — #70/#73)
   - `test/operations/test_block_tensor.expected` (centered param-value drift — #73)
   - `test/einsum/test_conv_padding.expected` (node-id shift + value drift — #70/#73)
   - `test/einsum/moons_demo_variant.expected` (param-init structure change — #70)
   - `test/training/bigram.expected` (loss-trajectory drift — #73)

   Each promotion must be eyeballed for sanity (id renumbering and/or
   centered-init value drift consistent with the table above), not blind-promoted.

2. **`moons_demo` converges and the test passes (exit 0) on this Mac.**
   `test/training/moons_demo` reaches a final epoch loss below its convergence
   threshold within its preselected seeds and exits 0 (no "FAILED to converge"),
   with `moons_demo.expected` matching the program's stable stdout. The fix
   strengthens the `moons_demo` recipe (initialization and/or
   learning-rate/epochs/seeds) rather than weakening the convergence gate — do
   not loosen `epsilon` to paper over non-convergence.

3. **The global centered default is preserved — the regression is fixed
   test-side, not by reverting #73.** `Operation.Make_DSL.default_param_init`
   remains `default_uniform1_param_init` (centered `[-0.25, 0.25)`). Any
   initialization override is local to `test/training/moons_demo.ml` (e.g.
   setting `TDSL.default_param_init := …` inside the run, or pinning an explicit
   `?param_init`/`~values` on the params), following the existing per-test
   pattern in the conv training tests.

4. **No production / codegen behavior changes, and no unrelated tests touched.**
   No edits to `tensor/operation.ml`'s `default_uniform1_param_init` /
   `default_param_init` definitions, to `cc_backend.ml`, to
   `test/config/ocannl_config`, or to the transformer tests already retuned by
   #73 (`fsm_transformer.ml`, `transformer_names.ml`). The change is confined to
   the 10 promoted `.expected` files and `test/training/moons_demo.{ml,expected}`.

5. **Full suite is green.** `cd ~/ocannl-staging && dune build && dune runtest`
   exits 0 with no remaining diffs or non-zero-exit tests. (Scope is the CPU /
   `sync_cc` path on this Mac; CUDA/minipc-wsl backends are unrelated and out of
   scope.)

## Context — how things work now

- **Default param init**: `tensor/operation.ml`, functor `Make_DSL`. The ref
  `default_param_init` now holds `default_uniform1_param_init`, defined just
  above it: `pointmul (number 0.5) (sub (uniform1 …) (number 0.5))` — i.e.
  `0.5 · (uniform1 − 0.5)`, all `~grad_spec:Prohibit_grad`. `param ?value
  ?values ?param_init` calls `!default_param_init ()` only when no explicit init
  is supplied; it also accepts an explicit `?param_init` (a forward-only init
  tensor) or `?values`.
- **Per-test init override pattern** (the template for the `moons_demo` fix):
  `test/training/cifar_conv.ml`, `circles_conv.ml`, `mnist_conv.ml` each do
  `TDSL.default_param_init := NTDSL.xavier ~scale_sq:0.06 TDSL.O.uniform1;` at
  the top of their run, decoupling themselves from the global default. `NTDSL`
  also exposes `kaiming` (see `docs/tensors_and_contexts.md`).
- **`moons_demo`**: `test/training/moons_demo.ml`. `train_once ~seed ()` sets
  `Utils.settings.fixed_state_for_init <- Some seed; Tensor.unsafe_reinitialize
  ()`, builds the half-moons dataset (`Dataprep.Half_moons`), the 3-layer MLP
  `{w3} * relu({b2}+{w2}*relu({b1}+{w1}*x))` (hidden width 16), the margin loss
  `relu(1 - moons_class *. mlp x)`, runs `epochs = 80` with
  `learning_rate = 0.1 * ((2*steps) - step_n)/steps` and `weight_decay =
  0.0001`, and returns the final epoch's summed loss. `main` retries
  `seeds = [1;2;3]` against `epsilon = 0.1`, prints
  `moons_demo: converged (final epoch loss < %.2g)` on success or
  `eprintf "moons_demo: FAILED to converge in N seeds"; exit 1` on failure.
  The params `w1/w2/w3/b1/b2` take the global `default_param_init`.
- **Prior related proposal** (already landed, not to be redone):
  `docs/proposals/moons-demo-convergence-robustness.md` (task-850e1b37)
  produced the current `train_once` / 3-seed / `epsilon=0.1` shape to absorb
  host-FP nondeterminism. Its **Tier-2 mandate** — "if the preselected seeds
  don't clear epsilon, strengthen the hyperparameters, don't loosen the
  threshold" — is the governing principle for AC 2 here.
- **#73 removed two workarounds**: the post-init recentering loops in
  `fsm_transformer.ml` and `transformer_names.ml` (now redundant under the
  centered default). This is the strongest evidence the centered default is
  intentional and correct.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. **Promote the 10 benign diffs.** Run `dune build && dune runtest`; for each of
   the 10 `.expected` files in AC 1, confirm the diff is only id renumbering
   (#70) and/or centered-init value drift (#73) per the table, then
   `dune promote` (or `dune runtest --auto-promote` scoped to those files after
   review). Do **not** auto-promote without eyeballing.

2. **Fix `moons_demo` (option b).** Decouple `moons_demo` from the global
   centered default by giving its params a higher-signal-energy initialization
   appropriate to the hinge-loss recipe — the cleanest, most future-proof
   option is to set `TDSL.default_param_init := …` at the top of `train_once`
   (mirroring the conv tests), e.g. a fan-scaled `NTDSL.xavier`/`kaiming` over
   `uniform1`, or an explicit wider centered/`uniform1`-style init — and/or
   retune `lr` / `epochs` / the preselected seeds so the 3-seed check clears
   `epsilon` robustly on this Mac. Pinning an explicit init is preferred over
   loosening `epsilon`, and over leaving the test hostage to future default-init
   changes. Verify convergence empirically across seeds 1/2/3; then regenerate
   `moons_demo.expected` to the new stable stdout and confirm exit 0.

3. **Verify.** `dune build && dune runtest` is green end-to-end. Open the fix PR
   against `lukstafi/ocannl-staging:master` (`gh pr create --repo
   lukstafi/ocannl-staging`).

## Scope

- **In scope:** the 10 `.expected` promotions and `test/training/moons_demo.{ml,expected}`.
- **Out of scope:** reverting or modifying #73's centered default
  (`default_uniform1_param_init` / `default_param_init`); codegen / compiler
  flags (`cc_backend.ml`, `ocannl_config`); the already-retuned transformer
  tests; CUDA / minipc-wsl backends. `~/ocannl-staging` stays a permanent fork
  of `ahrefs/ocannl` — do not detach it.
- **Dependencies:** none blocking; subtask of task-bb30d0be. Builds on the
  landed task-850e1b37 robustness work (do not redo it).
