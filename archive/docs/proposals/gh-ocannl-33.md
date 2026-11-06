# Demonstrate model surgery

Issue: [#33](https://github.com/ahrefs/ocannl/issues/33)
ROADMAP: v1.0 completeness

## Goal

Back the claim that compiled OCANNL models remain easy to modify with one
small, runnable demonstration:

- freeze a backbone;
- transplant weights between compatible models;
- attach and train a new head;
- apply different optimizer settings to parameter groups.

## Current building blocks

`Operation.stop_gradient`, `Tensor.params`, and `Train.sgd_one` provide the
graph/optimizer pieces. Values are no longer stored on `Tnode.t`; all reads and
writes must use `Context.get_values`/`set_values`. Persistence also supports
namespaced loads, and `test_tnode_namespaces` already demonstrates loading one
checkpoint twice, so this task need not duplicate persistence coverage.

## Direction

Use a tiny named MLP and make parameter identity explicit through stable label
paths. Do not pair parameters by `Set` iteration order: surgery should fail
clearly on missing or mismatched names/shapes.

Freezing should be shown as a graph-construction choice (`stop_gradient`) and
verified by unchanged backbone values, not by assuming every frozen gradient
buffer exists and contains zero. Per-group learning rates should compose
`Train.sgd_one` over an explicit partition of the parameter set.

## Completion criteria

- After a training step, frozen backbone values are unchanged and the new
  head's values changed.
- Copying named, shape-compatible parameters makes two models produce matching
  outputs; a mismatch produces a useful error in the demo/helper.
- Two optimizer groups exhibit the expected different update magnitudes.
- The example uses context-mediated value access and runs in the standard test
  harness.
- README's model-surgery claim links to the demonstration.

Keep convenience APIs out of scope unless the example reveals a repeated,
nontrivial pattern. Disk-based transfer learning and incremental compilation
performance are separate concerns.
