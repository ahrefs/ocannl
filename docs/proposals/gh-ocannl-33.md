# Demonstrate that model surgery is straightforward

GitHub issue: [ahrefs/ocannl#33](https://github.com/ahrefs/ocannl/issues/33)
Milestone: v1.0

## Status update (2026-06-12)

- Issue #33 is OPEN, milestone v1.0 (consistent with ROADMAP.md's v1.0, "End of October 2026" — this remains an unstarted v1.0 task).
- Not started: `test/training/model_surgery.ml` does not exist, and the README hedge with the `starightforward` typo is still present at `README.md:27`.
- All building blocks re-verified in the current tree: `stop_gradient` (`tensor/operation.ml:608`), `Tnode.set_values`/`get_values` (`arrayjit/lib/tnode.ml:752`/`758`), `Tensor.params` (`tensor/tensor.mli:22`), `Train.sgd_one` (`lib/train.ml:99`), `Nn_blocks.mlp` (`lib/nn_blocks.ml:89`). The `test/training/` reference tests (`moons_demo.ml`, `mlp_names.ml`, `mlp_bn_names.ml`, etc.) all exist.
- Tensor persistence (`lib/persistence.ml`, #373) has landed, confirming the proposal's non-goal note; disk-based transfer learning remains out of scope here.
- Interaction to watch: proposal/issue #333 (remove hosted memory mode) plans to replace `Tn.get_values`/`set_values` with context-aware versions. If #333 lands first, scenario B's API calls change shape (a `ctx` argument) but the demo's substance is unaffected.
- Remains to do: everything in the Approach section (single `test/training/model_surgery.ml` covering A-D, `.expected` file, dune stanza, README edit).

## Goal

Replace the hedged claim in `README.md` ("Model surgery should be straightforward (not sure if we are there yet)") with a concrete, runnable demonstration that the four canonical model-surgery workflows compose cleanly out of OCANNL's existing primitives, despite OCANNL being fully compiled. Where the demonstration uncovers genuine friction in current code, surface it as a follow-up issue rather than papering over it in the demo.

The four workflows in scope (from issue #33 / elaboration):

- **A. Layer freezing** via `stop_gradient` (e.g. fine-tuning a pre-trained backbone).
- **B. Weight transplantation** via `Tn.set_values` / `Tn.get_values` (copying parameters between models with matching shape).
- **C. New head on a frozen backbone** — composition of A and B in a transfer-learning shape.
- **D. Per-parameter-group learning rates** — selective optimization built from `Train.sgd_one` and the `loss.Tensor.params` set.

## Acceptance Criteria

Each criterion must be verifiable from CI output (the `.expected` file) or by running the resulting test, not by reading prose alone.

- [ ] **A. Layer freezing — frozen gradients are zero.** A demo trains a small MLP, then wraps the first layer's output in `stop_gradient` and runs at least one further training step. The test asserts (and prints, so it lands in `.expected`) that the gradient tnodes of the frozen layer's parameters are exactly zero (or are absent from the recompiled `sgd_update`'s param set), while the trainable layer's gradient is non-zero.
- [ ] **B. Weight transplantation — outputs match bit-for-bit.** Two structurally-identical models are constructed independently. After running `Tn.set_values dst (Tn.get_values src)` over each parameter, both models produce numerically identical outputs on the same input batch. The test asserts equality (within zero tolerance for the same backend; an exactly-equal print line lands in `.expected`).
- [ ] **C. New head on frozen backbone — only the head's params train.** Starting from the trained model in (A), the demo discards the original output layer, attaches a new head with fresh parameters, freezes the backbone via `stop_gradient`, and runs a few training steps on a different objective. The test asserts that backbone parameter values are unchanged (compared to a snapshot taken via `Tn.get_values` before surgery) and head parameters have moved.
- [ ] **D. Per-group learning rates — gradient steps reflect the schedule.** The demo partitions `loss.Tensor.params` into two groups (e.g. by label / depth) and composes `Train.sgd_one` per group with two distinct learning rates into a single update via `Asgns.sequence`. The test runs one step starting from a known parameter and gradient state and asserts that each group's parameter delta equals `-lr_group * grad` (within float tolerance).
- [ ] **E. README hedge removed.** The line `Model surgery should be starightforward (not sure if we are there yet).` in `README.md` is rewritten to an unhedged statement that links to the new demo file (and along the way the typo `starightforward` is fixed). If the implementation discovers genuine blockers and (E) cannot honestly be claimed, the implementer instead files a follow-up issue describing the gap and adjusts the README to point at it — but does not silently leave the hedge.
- [ ] **F. Builds and runs in CI.** The new file(s) are wired into `test/training/dune` as `(test ...)` stanzas with a corresponding `.expected` file, and `dune runtest` passes locally on the default backend.
- [ ] **G. Gaps surfaced as issues.** Any rough edge encountered (verbose patterns, missing helpers like `freeze_params`, partial-recompilation surprises, shape-mismatch ergonomics, serialization gaps for transplant-from-disk) is filed as a separate GitHub issue and linked from the demo file's header comment. The proposal explicitly does *not* require shipping new helper APIs as part of this task — those are follow-ups.

Non-goals (explicitly out of scope):

- Saving/loading parameters to/from disk for transfer learning. `lib/persistence.ml` exists; whether it covers surgery use-cases cleanly is a separate question. Demos do all surgery in-process.
- Adding new convenience APIs (`freeze_params`, `named_params`, `sgd_update_with_schedule`). These belong in follow-up tasks if the demo reveals they are warranted.
- Verifying that compilation is *incremental* (i.e. that frozen layers don't recompile). That is a property of the compilation pipeline; the demo only needs to show that surgery *works* end-to-end. Compilation-cost claims, if made, belong in a separate benchmarking task.
- Demonstrating these patterns on large / realistic models. A small MLP (and possibly a tiny transformer block, if cheap) is sufficient.

## Context

### Building blocks — verified to exist in the staging tree

- **`stop_gradient`** in `tensor/operation.ml` — identity on the forward pass, `Asgns.empty_comp` on the backward pass, with `~grad_spec:Prohibit_grad`. Re-exported through `Initial_NTDSL` / `NTDSL` so it is visible at the `let%op` level.
- **`Tnode.set_values` / `Tnode.get_values`** in `arrayjit/lib/tnode.ml` — flat-array read/write that goes through the host-side memory mode (`Hosted Nonconstant`) and `do_read` / `do_write` plumbing. Already used in `test/training/{moons_demo,bigram,mlp_names,fsm_transformer,transformer_names,mlp_bn_names}.ml` for input data; using it on parameters is the same call.
- **`Tensor.params`** field in `tensor/tensor.mli` — a `Base.Set.t` of all `Tensor.t` descendants whose `diff` is not `None`. This is what `Train.sgd_update` iterates: `Set.to_list loss.Tensor.params |> List.map ~f:sgd_one |> Asgns.sequence` (`lib/train.ml`). The same enumeration is the natural hook for per-group learning rates.
- **`Train.sgd_one`** in `lib/train.ml` — the per-parameter SGD-with-momentum step, factored out of `sgd_update` for exactly this kind of composition. Already public.
- **`Train.grad_update`** and **`Train.run_once`** in `lib/train.ml` — used by all existing training demos; the surgery demo should use the same idioms.
- **`Nn_blocks.mlp`** at `lib/nn_blocks.ml` ~line 88 (`let%op mlp ~label ~hid_dims () = ...`) — a ready-made small architecture suitable for the demo. The file's header explicitly says "Free to copy-paste and modify as needed."

### How existing training tests are structured

`test/training/dune` registers each `*.ml` as a `(test ...)` stanza with `package neural_nets_lib`, depends on `ocannl_config` plus `OCANNL_BACKEND`, and uses `(pps ppx_here ppx_ocannl)`. Each test has a sibling `*.expected` file that captures stdout. Examples to model on:

- `mlp_names.ml` — `Nn_blocks.mlp`-based, manageable size, plenty of `Tn.set_values` usage.
- `moons_demo.ml` — smallest end-to-end example, two-layer MLP, fast.

### README site of the hedge

`README.md` (around the bullet list near line 27, after "Should be easily extensible."):

> * Model surgery should be starightforward (not sure if we are there yet).

The replacement should be a concrete claim of the form "Model surgery is straightforward — see `test/training/<demo>.ml`." (and fix the typo).

### Code pointers

- `tensor/operation.ml` — `stop_gradient`
- `tensor/tensor.mli` — `params` field documentation
- `arrayjit/lib/tnode.ml` — `set_values`, `get_values`
- `lib/train.ml` — `sgd_one`, `sgd_update`, `grad_update`, `init_params`, `run_once`
- `lib/nn_blocks.ml` — `mlp`, `multi_head_attention`, `transformer_encoder_block`
- `lib/persistence.ml` / `.mli` — exists; relevant for follow-ups, not this task
- `arrayjit/lib/assignments.ml` — `Asgns.sequence`, `Asgns.Block_comment`
- `test/training/dune` — where the new test(s) get registered
- `test/training/moons_demo.ml`, `mlp_names.ml` — style and structure references
- `README.md` — the hedge to remove

## Approach (suggested)

*Suggested approach — agents may deviate if they find a better path.*

**Single demo file** rather than four. Putting all four scenarios in one `test/training/model_surgery.ml` makes the "look, surgery is easy and these all compose" point more vividly than four scattered tests, and amortizes dataset/setup boilerplate. Each scenario is a clearly-commented section with its own asserts; the `.expected` file captures the assertion summary (one line per scenario: "A: frozen gradients zero — OK", "B: transplanted outputs equal — OK", etc.) plus enough numeric output to be informative without being brittle.

Use `Nn_blocks.mlp ~hid_dims:[ ... ]` over a tiny synthetic dataset (e.g. random inputs and a deterministic target derived from them) to keep runtime negligible and avoid pulling in `Dataprep`. This also keeps assertions cleanly numerical.

For (A) and (D), it's worth comparing two strategies for "drop a parameter from the update": (i) wrap with `stop_gradient` and let `Tensor.params` exclude it via the `Prohibit_grad` flag, (ii) manually filter `loss.Tensor.params` before passing to `sgd_one`. Whichever the existing infrastructure supports cleanly is the documented pattern; if both work, mention both.

For (B), do `Tn.get_values` / `Tn.set_values` over the entire param set in a single pass (iterating `loss.Tensor.params`), not per-named-parameter — that's the more idiomatic OCANNL move and it's what would scale to real models.

For (C), reuse the model from (A) rather than training from scratch a second time.

If during implementation any scenario reveals that a building block is missing or awkward (e.g. `Tensor.params` doesn't expose what's needed for a clean `freeze_params`, or `Asgns.sequence` of `sgd_one` partials hits an unexpected error), file a GH issue, link it from the demo's header comment, and either work around it in the demo or, if the workaround would obscure the point, mark that scenario as `(* TODO #NNN: ... *)` and exit with a non-zero status so CI catches it. Don't silently fudge.

## Scope

**In scope:**

- One new file `test/training/model_surgery.ml` covering scenarios A–D.
- Companion `test/training/model_surgery.expected`.
- Dune stanza in `test/training/dune`.
- README.md edit to replace the hedge (criterion E).
- Any small follow-up GH issues filed for friction encountered (criterion G).

**Out of scope:**

- New helper APIs in `Train` or elsewhere.
- Persistence / disk-based transfer learning.
- Compilation-cost or partial-recompilation benchmarking.
- Large-model demonstrations.
- Any change to `stop_gradient` itself, `Tn.set_values`, `Tensor.params`, or `sgd_one` — they're treated as fixed primitives.

**Dependencies:** none. Building blocks are all present today.

**Adjacent tasks (not blocking):** gh-ocannl-275 (LLM101n replication) and gh-ocannl-308 (Tensor Puzzles) may eventually reuse patterns established here, but neither blocks nor is blocked by this task.
