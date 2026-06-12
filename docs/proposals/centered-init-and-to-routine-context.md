# Add centered param init support and return updated context from Train.to_routine

## Status update (2026-06-12)

- gh-ocannl-116 (the Karpathy FSM transformer tutorial, source of discovery) is CLOSED/COMPLETED (milestone v0.6.4); the tutorial landed as `test/training/fsm_transformer.ml`.
- Neither improvement is implemented yet: there is no `centered_uniform1` (or similar) in `tensor/operation.ml`, and `Train.to_routine` (now `lib/train.ml:139`) still discards the context — with an explicit in-code comment "Return just the routine for backward compatibility - ctx is discarded here".
- Demand for centered init is confirmed in-tree: both `test/training/fsm_transformer.ml` (lines ~153-163) and `test/training/transformer_names.ml` recenter all params post-init with a host-side loop, commented "Centered initialization (e.g. xavier/normal) is standard for transformers but not yet available as a built-in default_param_init in OCANNL".
- `param` now accepts an explicit `?param_init` argument (`tensor/operation.ml:752`), an additional channel through which a centered init function could be supplied per-param.
- Line references in Context drifted and have been corrected below: `default_param_init` is now at line 747, `uniform1` at 669-675, `param` at 752-770; `sgd_update` is still at `lib/train.ml:110-113`.
- The caller list drifted: `test/training/bigram_mlp.ml` no longer exists (the MLP examples are now `mlp_names.ml` / `mlp_bn_names.ml`); `test/operations/zero2hero_1of7.ml` is an additional caller. Re-enumerate callers at implementation time.
- The core analysis (grad-carrying init intermediates iterated by `sgd_update`) still matches the current code. *(Update 2026-06-12: the invariant — init subgraphs must be grad-free — is confirmed empirically, but the mechanism stated here is stale: intermediates never enter `loss.Tensor.params` (only `Tensor.param` populates `params`), so `sgd_update` does not iterate them. The actual failure today is a hard `Utils.User_error "The linked context lacks node _N.grad"` at routine-link time. See the Design review section below.)*

## Goal

Two improvements discovered during the Karpathy FSM transformer tutorial (gh-ocannl-116):

1. Neural networks need centered parameter initialization (values in [-0.5, 0.5) or [-1, 1) rather than [0, 1)). Users who try to build centered init from existing primitives (e.g., `2 * uniform1 - 1`) inadvertently create gradient-carrying intermediate nodes that cause NaN during SGD updates, because `sgd_update` iterates over all `loss.Tensor.params` including those stale intermediates. *(Update 2026-06-12: mechanism corrected — the intermediates do not enter `loss.Tensor.params`; instead, their `zero_grads` get baked into the param's `diff.zero_grads` at construction, so the training step references grad tnodes the init context never created, and `Train.to_routine` fails with `User_error "The linked context lacks node _N.grad"` (verified by repro). The fix direction — grad-free init subgraphs — is unchanged.)*

2. `Train.to_routine` discards the updated `Context.t` returned by `Context.compile`, preventing users from threading context through multiple routine compilations that share tensor nodes. *(Update 2026-06-12: overstated — the routine itself carries the updated context, recoverable via `Context.context : routine -> t` (in `context.mli` since 2025-09), and in-tree callers already thread it that way, e.g. `Train.to_routine (Context.context sgd_step) ...` in `test/training/mlp_names.ml`, `bigram.ml`, `cifar_conv.ml`, `moons_demo.ml`. See the Design review section.)*

Related: gh-ocannl-116, task-28c898b7.

## Acceptance Criteria

1. A convenience function (e.g., `centered_uniform1`) is available in the DSL that produces uniform random values centered around zero, suitable as a drop-in `default_param_init` replacement.
2. The centered init function does not create gradient-carrying intermediate nodes -- only the final param tensor itself should have `Require_grad`. This prevents the NaN issue when used with `sgd_update`.
3. `Train.to_routine` returns `Context.t * Context.routine` (the updated context alongside the routine).
4. All in-tree callers of `Train.to_routine` are updated to destructure the new return type.
5. Existing tests pass after the changes.

## Context

### Centered init

- `tensor/operation.ml` line 747 *(Update 2026-06-12: was 710)*: `default_param_init` defaults to `uniform1 ~grad_spec:Require_grad`.
- `tensor/operation.ml` lines 669-675 *(Update 2026-06-12: was 632-638)*: `uniform1` chains `embed_self_id`, `threefry4x32`, and `uint4x32_to_prec_uniform1` -- each intermediate node inherits the passed `grad_spec`.
- `tensor/operation.ml` lines 752-770 *(Update 2026-06-12: was 715-733)*: `param` function uses `!default_param_init ()` when no explicit init is given (it now also accepts an explicit `?param_init` argument).
- `lib/train.ml` lines 110-113: `sgd_update` iterates `loss.Tensor.params` and calls `sgd_one` on every param with a `diff` field, including init-time intermediates if they carry `Require_grad`. *(Update 2026-06-12: incorrect — `params` is populated only by `Tensor.param` (`tensor/tensor.ml:678` sets `params = Set.singleton t`); op nodes merely union subtensor `params` (`tensor.ml:372`), so init intermediates never appear there. The real leak is via `zero_grads`: `Tensor.op` folds subtensor `zero_grads` into the new node's `diff.zero_grads` (`tensor.ml:438-446`), so a composite param's training-step zeroing references the intermediates' grad tnodes, which fail `verify_prior_context` at link (`arrayjit/lib/backends.ml:301-306`).)*
- The key insight: init-time arithmetic intermediates should use `Prohibit_grad`; only the outermost param tensor needs `Require_grad`.

### to_routine context threading

- `lib/train.ml` lines 139-155 *(Update 2026-06-12: was 186-203)*: `to_routine` calls `Context.compile ctx comp bindings` which returns `(Context.t * Context.routine)`, but discards the context with `let _ctx, routine = ...`.
- `lib/train.ml` line 161 *(Update 2026-06-12: was 208)*: `init_params` already returns `Context.t` after compile+run, showing the pattern.
- Callers that need updating (all in-tree uses of `to_routine`) *(Update 2026-06-12: list refreshed; `bigram_mlp.ml` was replaced by `mlp_names.ml`/`mlp_bn_names.ml`)*:
  - `test/einsum/moons_demo_variant.ml`, `test/einsum/test_padding_reset.ml`
  - `test/operations/test_where_simple.ml`, `test/operations/zero2hero_1of7.ml`, `zero2hero_1of7_exec.ml`, `test_execution_deps.ml`
  - `test/training/bigram.ml`, `mlp_names.ml`, `mlp_bn_names.ml`, `circles_conv.ml`, `cifar_conv.ml`, `mnist_conv.ml`, `moons_demo.ml`
  - `test/operations/primitive_ops.ml`, `check_slice_shapes.ml`
  - `bin/compilation_speed.ml`

## Approach

*Suggested approach -- agents may deviate if they find a better path.*

**Centered init**: Add `centered_uniform1` (and optionally `centered_uniform`) functions in `operation.ml` that build the same threefry PRNG chain as `uniform1` but with `Prohibit_grad` on all intermediate nodes, then apply `2 * result - 1` arithmetic also with `Prohibit_grad`, wrapping only the final output with the caller's `grad_spec`. Expose through `Make_DSL` alongside existing `uniform1`.

**to_routine**: Change return type from `Context.routine` to `Context.t * Context.routine`. Callers that don't need the context can destructure as `let _ctx, routine = ...`.

## Scope

**In scope:**
- New centered init convenience function(s) in `operation.ml` and the DSL module
- Changing `to_routine` return type and updating all in-tree callers
- Updating any affected `.mli` signatures

**Out of scope:**
- Changes to the IR/codegen layer or `fetch_op` type
- Adding new PRNG variants (normal distribution centering, etc.) beyond the basic centered uniform
- Out-of-tree caller migration (they will get a clear type error)

**Dependencies:** None blocking. Related to gh-ocannl-116 (source of discovery) and task-28c898b7.

## Design review (2026-06-12)

**Verdict: sound-with-changes.** Part (a)'s invariant and mechanism are right
(empirically validated, see below), but the surface should be parameterized and
a deeper root fix considered. Part (b) as specified (breaking `to_routine`)
should be dropped — its motivating gap does not exist.

**Empirical validation** (throwaway repro, removed after running):

- *Failure reproduced*: `TDSL.param ~param_init:(PDSL.sub PDSL.O.(!.2. *.
  uniform1 ()) PDSL.O.(!.1.)) "w"` compiles, `init_params` succeeds, but
  `Train.to_routine ctx bindings (sequence [grad_update; sgd_update])` raises
  `User_error "The linked context lacks node _1.grad"`. Mechanism: inherited
  `zero_grads` chains (see corrected Context above), not `sgd_update`
  iteration, and a hard error today rather than NaN.
- *Proposed mechanism validated*: the same composite with `Prohibit_grad`
  intermediates and a grad-carrying outermost node — `PDSL.sub NTDSL.O.(!.2.
  *. uniform1 ()) NTDSL.O.(!.1.)` — trains correctly (5 SGD steps, param
  updates, values centered). So `centered_uniform1` is packaging of an
  already-working pattern; `%cd`-generated `grad_asn`s Option-guard grads of
  diff-less subtensors, and `uniform1`'s own final node (Require_grad over a
  grad-free chain, empty `grad_asn`) is in-tree precedent.

**Recommendations:**

1. *Consider the root fix in `Tensor.param`*: override the captured tensor's
   `diff.zero_grads` to only `fetch_zeros` its own grad. A param's backprop is
   already terminal at consumption sites (`tensor.ml:462-466`), so the init
   intermediates' grads are referenced *only* through the inherited
   `zero_grads` chain — pruning it makes **any** grad-carrying init expression
   harmless, turning `centered_uniform1` from a correctness requirement into a
   convenience. Verify nothing else references the pruned grads.
2. *Parameterize instead of multiplying names*: the in-tree demand
   (`fsm_transformer.ml`, `transformer_names.ml`) recenters to [-0.25, 0.25),
   not [-1, 1), so a fixed `centered_uniform1` would not remove those
   workaround loops. Prefer `uniform1 ?(low = 0.) ?(high = 1.)` (or
   `centered_uniform1 ?scale`), exposed through `Make_DSL` so the `%op` record
   syntax (`PDSL.`-qualification of the init head) picks it up.
3. *Strengthen acceptance criteria*: (i) replace the host-side recentering
   loops in `test/training/fsm_transformer.ml` and
   `test/training/transformer_names.ml` with the new init — that is the real
   acceptance test; (ii) add a test that a naive composite Require_grad init
   either works (if rec. 1 lands) or fails with an actionable error message
   instead of "The linked context lacks node _1.grad".
4. *Precision caveat for the composition approach*: `uniform1`'s final
   conversion node uses `top_down_prec:true` so the param's precision flows
   into the PRNG conversion; with `sub` as the new outermost node that flow
   changes. Either put the affine transform inside the final op's `op_asn` or
   add a non-default-precision param test.
5. *Part (b): no breaking change.* `Context.context : routine -> t` returns
   exactly the updated context (`context.ml:201` stores `updated_ctx` in the
   routine), and the idiom `Train.to_routine (Context.context prev_routine)
   ...` is already standard in-tree (16 caller files checked). Returning
   `Context.t * routine` would churn all of them to expose redundant
   information. Do instead: fix the misleading comment at `lib/train.ml:155`
   ("ctx is discarded here" — it is not, it is inside the routine) and
   document the idiom in `to_routine`'s docstring. If tuple-consistency with
   `Context.compile` is still wanted, add a new function rather than breaking
   `to_routine`.

**Open decision points for Łukasz:**

- Root fix in `Tensor.param` (zero_grads pruning), convenience-only
  `centered_uniform1`, or both?
- Surface choice: range-parameterized `uniform1 ~low ~high` vs a separate
  `centered_uniform1`; and should the *default* `default_param_init` become
  centered (standard for transformers, but churns many `.expected` files)?
- Part (b): accept "document `Context.context`, no API change", or still
  prefer the tuple return for discoverability despite the redundancy?
