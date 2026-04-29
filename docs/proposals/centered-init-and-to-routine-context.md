# Add centered param init support and return updated context from Train.to_routine

## Goal

Two improvements discovered during the Karpathy FSM transformer tutorial (gh-ocannl-116):

1. Neural networks need centered parameter initialization (values in [-0.5, 0.5) or [-1, 1) rather than [0, 1)). Users who try to build centered init from existing primitives (e.g., `2 * uniform1 - 1`) inadvertently create gradient-carrying intermediate nodes that cause NaN during SGD updates, because `sgd_update` iterates over all `loss.Tensor.params` including those stale intermediates.

2. `Train.to_routine` discards the updated `Context.t` returned by `Context.compile`, preventing users from threading context through multiple routine compilations that share tensor nodes.

Related: gh-ocannl-116, task-28c898b7.

## Acceptance Criteria

1. A convenience function (e.g., `centered_uniform1`) is available in the DSL that produces uniform random values centered around zero, suitable as a drop-in `default_param_init` replacement.
2. The centered init function does not create gradient-carrying intermediate nodes -- only the final param tensor itself should have `Require_grad`. This prevents the NaN issue when used with `sgd_update`.
3. `Train.to_routine` returns `Context.t * Context.routine` (the updated context alongside the routine).
4. All in-tree callers of `Train.to_routine` are updated to destructure the new return type.
5. Existing tests pass after the changes.

## Context

### Centered init

- `tensor/operation.ml` line 710: `default_param_init` defaults to `uniform1 ~grad_spec:Require_grad`.
- `tensor/operation.ml` lines 632-638: `uniform1` chains `embed_self_id`, `threefry4x32`, and `uint4x32_to_prec_uniform1` -- each intermediate node inherits the passed `grad_spec`.
- `tensor/operation.ml` lines 715-733: `param` function uses `!default_param_init ()` when no explicit init is given.
- `lib/train.ml` lines 110-113: `sgd_update` iterates `loss.Tensor.params` and calls `sgd_one` on every param with a `diff` field, including init-time intermediates if they carry `Require_grad`.
- The key insight: init-time arithmetic intermediates should use `Prohibit_grad`; only the outermost param tensor needs `Require_grad`.

### to_routine context threading

- `lib/train.ml` lines 186-203: `to_routine` calls `Context.compile ctx comp bindings` which returns `(Context.t * Context.routine)`, but discards the context with `let _ctx, routine = ...`.
- `lib/train.ml` line 208: `init_params` already returns `Context.t` after compile+run, showing the pattern.
- Callers that need updating (all in-tree uses of `to_routine`):
  - `test/einsum/moons_demo_variant.ml`, `test/einsum/test_padding_reset.ml`
  - `test/operations/test_where_simple.ml`, `test/operations/zero2hero_1of7_exec.ml`, `test/operations/test_execution_deps.ml`
  - `test/training/bigram.ml`, `bigram_mlp.ml`, `circles_conv.ml`, `cifar_conv.ml`, `mnist_conv.ml`, `moons_demo.ml`
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
