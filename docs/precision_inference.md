# Bidirectional precision inference

OCANNL features a rudimentary bidirectional precision inference. It is much less powerful than the constraints-based shape and projections inference. It is somewhat prominent because it contributes the `top_down_prec` flag to the central `Tensor.t` type.

Tensors that choose `top_down_prec=true` "detach" themselves from their defining tensor expression as far as precision goes, and get precision propagated into them from the use sites. By default, terminal tensors are `top_down_prec=true`, and non-terminal tensors are `top_down_prec=false`, except for all the parameter tensors (created via `Tensor.param`), and results of the operation `uint4x32_to_prec_uniform`, that are also `top_down_prec=true`. When a tensor precision is set by the user via `Tnode.update_prec`, this setting takes precedence over any inferences. When a `top_down_prec=true` tensor has its precision set by the user, it contributes this precision in the bottom up inference (together with all `top_down_prec=false` subtensors).

The core algorithm is a few dozen lines of the `Tensor.op` function (`tensor/tensor.ml`) over the three-valued `Tnode.delayed_prec` a node is created with: `Default prec` (nothing has constrained it), `Inferred (lazy prec)` (a promotion over other nodes, forced on first read) or `Specified prec` (set by the user through `Tnode.update_prec`, overriding any inference). A node's `Tn.storage_prec` is the lazy that forces whichever of the three the node holds at that time; once forced it no longer changes, so a conflicting write is a `User_error` naming the node. The cell holding the value is `Tnode.Prec_cell`, which seals these transitions and reports every write that changes the value to the session through `Tn.prec_write_hook` (gh-ocannl-1624).

The bottom-up pass is `delayed_prec_for default get` in `Tensor.op`, called once for the value node with `Tensor.default_value_prec` and `(fun t -> Some t.value)`, and once for the gradient node with `Tensor.default_grad_prec` and the subtensors' `grad` nodes. For a `top_down_prec=true` result it produces `Default default` without looking at the operands. Otherwise it produces `Inferred` of a lazy promotion (`Ir.Ops.promote_prec`) over the operands in `ordered_ts`, each contributing: `Tn.bottom_up_prec` of its node when the operand is `top_down_prec=false`, and `Tn.get_specified_prec` of its node when it is `top_down_prec=true`, so a top-down operand takes part only where the user set its precision. An empty promotion (all operands top-down and unset) settles at `default`.

The top-down pass runs after the operation is registered as an update step, deferred so a construction that fails before that point leaves no link on its operands (gh-ocannl-1301). For each operand in `top_down_ts` (the `top_down_prec=true` ones), `Tensor.op`'s local `update_infer_prec ti tn prec` links the operand's node `tn` to the result's `v.Tn.storage_prec`, and later the operand's gradient to the result's gradient `g.Tn.storage_prec`. The local wrapper is a guard around `Tn.update_infer_prec`: it skips only an operand whose shape is `closed`, since a closed operand's committed state does not change after its closure and a rewind discarding the consumer would leave the link dangling (gh-ocannl-1301). `Tn.update_infer_prec` itself turns a `Default` node into `Inferred` of the link, promotes an `Inferred` node with it, and leaves a `Specified` node unchanged. Layout readiness (`dims` being forced) does not decide whether a precision link is allowed: the precision cell itself refuses a link that could change an already-read precision (gh-ocannl-2026).

A tensor whose data fixes its precision never takes this link: `init`, `reshape`, `wrap`, `wrap_padded` and the other data-backed terms create their node `Specified` at the data's precision, so a top-down update of one changes nothing, even after that precision has been read (gh-ocannl-2011). Only a node the update could change refuses it once forced: a term with no data of its own, or a result, read before its consumer was built.

## Initializers follow their parameter

A parameter's initialization expression fills the parameter, so its float tensors are at least as precise as the parameter (gh-ocannl-1967): otherwise a `double` parameter over the `single` default would start from f32-quantized draws, since the bottom-up pass stops at the first operation none of whose operands states a precision (the default initializer's centering `sub`), which settles at `Tensor.default_value_prec`. `Tensor.param` therefore links every float tensor of the expression (stopping at other parameters) to the parameter's precision, whether that one is set by the user, pinned by `Mixed_prec.with_master_weights ~master_prec`, or inferred top-down from a consumer.

The link promotes and never narrows: a `half` parameter still rounds a `single` draw once. Narrowing an initializer to a reduced parameter precision is `Precision_policy.apply`'s assignment.

A parameter's precision can depend on a consumer that reads one of these tensors bottom-up (a tensor shared between an initializer and the model), so the link is `Tnode.follow_prec` rather than `update_infer_prec`: a linked tensor keeps answering bottom-up readers with the precision it had before the link (`Tnode.bottom_up_prec`), which keeps the lazy computation acyclic and independent of forcing order.

## Storage, compute and placement

Inference settles a node's **storage** precision. The arithmetic over it runs at the backend's **compute** precision, and a reduction's accumulator at its **accumulator** precision; `Context.codegen_capabilities` answers both for a storage precision. Among float formats, the ones that can compute wider than they store are the narrow floats: `cc` computes f16, bf16 and fp8 in f32 under `narrow_compute_f32`, and Metal computes fp8 in f32 always. This section is about floats; an integer node keeps its scope wherever collapsing it could change the value. Accumulators are their own axis, so a backend can compute a format at storage width and accumulate it wider, as CUDA does for bf16.

That leaves one question for placement, the compiler's choice between keeping a node's value in a buffer (materialized) and recomputing it at each read (virtual): does anything still round the value to storage precision?

| What produces the value | Materialized | Virtual |
| --- | --- | --- |
| A narrowing read: a narrow node reading a tensor node or virtual node of a precision that does not widen exactly into it, such as a cast twin or a narrow slice of f32 data | rounds | rounds |
| An RNG draw | storage precision | storage precision |
| A constant | rounds at lowering | rounds at lowering |
| Arithmetic | rounds at the store | rounds only where compute precision equals storage |
| An accumulation | rounds at the store | rounds only where accumulator precision equals storage |

The rows compose. An accumulation opened by a narrowing read, such as `s =: src` followed by `s =+ x` over a narrow `s` and a wider `src`, rounds that read in either placement, and only what the accumulation adds after it stays at accumulator precision (gh-ocannl-2076).

A constant is a value of the precision its slot evaluates it at, so at a narrow float the compiler rounds it while lowering, after folding constant expressions in double (gh-ocannl-2046): every backend and placement reads the same literal, at no run-time cost. Two details. An integer power exponent is an operator parameter, not a value of the slot, so `x **. 9` stays a ninth power at fp8. And `check_half_prec_constants_cutoff` reads the literal before it is rounded, so a literal past fp16's range is refused rather than becoming an infinity. On a backend computing a narrow format wider, a literal too small for the format (an `eps` of `1e-8` at f16) is now 0 there as well, as it already was on the backends computing at storage width.

So where a backend computes wider than a node stores, a virtual narrow node with arithmetic in it has no store, and it keeps the wider value until something downstream rounds. For example, take bf16 inputs `1` and `2^-9`, both exactly representable, summed into a bf16 node and read in an f32 expression. On a backend computing bf16 in f32, a materialized node rounds the sum to `1`, because bf16's spacing at 1 is `2^-7`, while a virtual one hands on `1 + 2^-9`.

The two values come from the same program, and both are correct under this contract. This is the narrow-float counterpart of C's excess precision, as an analogy only: what is promised here is the table above. Particular inputs may still agree, and where they differ no bound on the difference is promised: a rounding can overflow, and later operations can amplify what it would have removed.

Consequences to plan for, all of them only where compute or accumulator precision is wider than storage:

- Anything that changes a node's placement can move a narrow model's results. That covers `Train.set_materialized`, `Context.decide_materialized`, the `virtualize_*` limits, and materializing a node in order to inspect it.
- `Train.tune_placements` chooses among placements by timing, so the arm that wins decides the values, and a flip accepted by `tune_inline_flips` does too. The `reproducible` profile keeps `tune_inline_flips` at 0 for this reason, and does not search.

To decide a value yourself:

- Materialize the node, and the store rounds it.
- Compute at a wider precision and cast to the narrow one. A narrowing read rounds in either placement, on every backend. The source computation keeps its own placement behaviour, and it can change the input the cast rounds.
- A cast to the precision its argument already has is the identity, and rounds nothing.
- `narrow_compute_f32=false` restores per-operator arithmetic on the CPU backends, the rule the GPU backends follow for the formats they compute at storage width. It gives up the accuracy of computing in f32 and, for a format without native vector arithmetic, the vectorized renderings. It does not decide accumulators, which have their own settings (`fp16_arithmetic`, `bf16_arithmetic`), or fp8 on Metal. Reference: `ocannl_config.reference`.

This contract is about placement alone. It says nothing of schedules, which can introduce stores of their own (a k-blocked reduction keeps storage-precision partials at its block boundaries), of the C compiler's contraction, or of agreement between backends. Which reading a deferred computation observes across routines is a separate matter: see [Recompute-at-read](lowering_and_inlining.md#recompute-at-read-the-semantics-of-virtual-gh-617). `test/operations/narrow_placement_contract.ml` executes the table on every backend, predicting from the backend's own precisions.
