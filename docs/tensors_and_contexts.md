# Tensors and Contexts

This document describes how to work with tensors and execution contexts in OCANNL. It covers the core `Tensor` module, the `Context` API for backend management, and the `Train` module for training workflows.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Tensors](#tensors)
  - [Tensor Structure](#tensor-structure)
  - [Tensor Node Namespaces](#tensor-node-namespaces)
  - [Roots, Embedded Nodes, and Params](#roots-embedded-nodes-and-params)
  - [Creating Tensors](#creating-tensors)
  - [Operation Functions and Shape Parameters](#operation-functions-and-shape-parameters)
  - [Gradient Specifications](#gradient-specifications)
  - [Tensor Precision](#tensor-precision)
- [Contexts](#contexts)
  - [Context Creation](#context-creation)
  - [Compilation and Execution](#compilation-and-execution)
  - [Node Initialization Tracking](#node-initialization-tracking)
  - [Execution Dependency Tracking](#execution-dependency-tracking)
- [The Train Module](#the-train-module)
  - [Parameter Initialization](#parameter-initialization)
  - [Forward and Backward Passes](#forward-and-backward-passes)
  - [Training Loops](#training-loops)
  - [Stopping and Resuming Training](#stopping-and-resuming-training)
- [Reusable Tensor Programs](#reusable-tensor-programs)
  - [One Function, Two Shapes](#one-function-two-shapes)
  - [A Shape Mistake, Named](#a-shape-mistake-named)
  - [Recovering a Session from a Failed Construction](#recovering-a-session-from-a-failed-construction)
- [Randomness and Parameter Initialization](#randomness-and-parameter-initialization)
  - [Counter-Based PRNG](#counter-based-prng)
  - [Initialization Functions](#initialization-functions)
  - [Kaiming and Xavier Initialization](#kaiming-and-xavier-initialization)
  - [Configuring Default Initialization](#configuring-default-initialization)
- [Memory Management](#memory-management)

## Core Concepts

OCANNL separates the *definition* of computations from their *execution*:

1. **Tensors** (`Tensor.t`) describe computations and their structure, including forward code, gradients, and backpropagation logic.
2. **Contexts** (`Context.t`) manage execution on a specific backend (CPU, CUDA, Metal), handling device memory and compiled routines.
3. **Routines** (`Context.routine`) are compiled computations ready for efficient repeated execution.

This separation enables:
- Defining models once, running on multiple backends
- Compiling training loops once, executing many times efficiently
- Clear separation between model definition and execution concerns

## Tensors

### Tensor Structure

A tensor in OCANNL (`Tensor.t`) contains:

```ocaml doc-skip illustrative partial Tensor record; see tensor/tensor.ml
type t = {
  params : (t, comparator_witness) Base.Set.t;  (* Learnable parameters *)
  forward : comp;                                (* Forward computation *)
  diff : diff option;                            (* Gradient info if differentiable *)
  value : tn;                                    (* The underlying tensor node *)
  shape : Shape.t;                               (* Shape with inference state *)
  children : subtensor list;                     (* Sub-tensors in the computation *)
  (* ... *)
}
```

Key fields:
- **`value`**: The tensor node (`Ir.Tnode.t`) holding the actual data; the tensor's identifiers
  live here: `value.id` is the consecutive session id, `value.namespace` qualifies it
  (see below), and `value.uid` is the process-unique identity used for comparison
- **`diff`**: Contains gradient node and backpropagation code (if differentiable)
- **`params`**: Set of tensors requiring separate initialization (see [Roots, Embedded Nodes, and Params](#roots-embedded-nodes-and-params))
- **`forward`**: Computation to produce this tensor's value, including embedded subtensor computations
- **`shape`**: Shape information, progressively refined during inference

### Tensor Node Namespaces

Every tensor node carries a `namespace` string alongside its session `id` (gh-ocannl-372).
Within a session all nodes share the ambient namespace — `"ocannl"` by default — and only
`Tensor.unsafe_reinitialize ~namespace:"my_ns" ()` changes it. Namespaces must be legal
identifiers (`[A-Za-z_][A-Za-z0-9_]*`); non-default namespaces show up as a prefix in debug
names and generated code (`my_ns__n42` instead of `n42`).

Namespaces exist for "model surgery": `Persistence.load ~prefix_namespace` rewrites the
namespaces of the tensors stored in a checkpoint file (to `prefix ^ "__" ^ file_namespace`)
while keeping their session ids, so weights of independently developed models — or two
snapshots of the same model — can coexist in one registry and one routine without id
clashes. Since `Embed_self_id` embeds only the session id, its values are invariant under
namespace prefixing.

### Roots, Embedded Nodes, and Params

Understanding how OCANNL manages computation inclusion is essential for advanced usage.

#### Forward Roots and Embedding

When you build a computation graph, OCANNL tracks **forward roots** - tensors whose forward computation code hasn't yet been included in another tensor's forward code. When a tensor T uses subtensor S:

1. If S is a forward root, S's forward code is **embedded** into T's forward code
2. S is then removed from the forward roots set
3. S's `value` node is added to T's `forward.embedded_nodes`

This ensures each tensor's initialization code appears exactly once - in the first tensor that uses it. The `embedded_nodes` set tracks which tensor nodes' computations are included in a given forward computation.

```ocaml
(* When creating: let%op result = a + b *)
(* If 'a' is a forward root:
   - a.forward code is included in result.forward
   - a.value is added to result.forward.embedded_nodes
   - 'a' is removed from forward_roots *)
```

#### The `params` Field

The `params` field of a tensor is slightly misnamed from a user perspective. It contains tensors that:
1. Need initialization (their forward code computes initial values)
2. Are NOT embedded in the parent tensor's forward code

This typically corresponds to "parameters" in the ML sense (learnable weights), but more precisely it means "tensors requiring separate initialization". When you call `Train.init_params`, it:
1. Collects all tensors in `t.params` recursively
2. Gathers their `forward.embedded_nodes`
3. Compiles and runs their initialization code

A tensor becomes a "param" when created via `Tensor.param` or inline `{ name }` syntax - its forward code is kept separate rather than embedded into the computation that uses it.

#### Why This Matters

This design enables efficient execution:
- Initialization code runs once, not every forward pass
- Forward pass code only includes the actual computation
- Context tracks which nodes are initialized to remind the programmer of potential bugs
- Tensors can be computed in one routine and used in another routine

**Potential pitfall**: If a tensor T1 embeds another tensor S (capturing S's forward code), but T1 is never used in the final computation, S's initialization may be "lost" - it won't appear in `init_params` because T1 isn't in anyone's `params`. OCANNL handles the common case of `random_seed` specially, but custom initialization patterns may need care.

#### Backprop Roots

Similarly, **backprop roots** track tensors whose backpropagation code hasn't been consumed. When you call `Train.grad_update` or similar, the backprop code flows from loss backward through the graph, with each tensor's backprop code included exactly once.

### Creating Tensors

OCANNL provides multiple ways to create tensors:

**Using DSL modules**:
```ocaml
open Ocannl.Operation.DSL_modules

(* Differentiable tensor with gradient tracking when needed *)
let x = TDSL.term ~output_dims:[10] ()

(* Non-differentiable tensor (no gradient) *)
let const = NTDSL.number 3.14

(* Parameter with gradient (for model weights) *)
let w = TDSL.param "weights" ~output_dims:[hidden_dim] ()
```

**Using syntax extensions** (see [syntax_extensions.md](syntax_extensions.md)):

```ocaml
open Ocannl.Operation.DSL_modules

(* %op creates differentiable tensors *)
let%op layer x = { w } * x + { b = 0.; o = [hidden_dim] }

(* %cd creates assignment code with non-differentiable intermediate tensors *)
let%cd update = p =- learning_rate *. p.grad
```

### Operation Functions and Shape Parameters

Most tensor-creating operations in OCANNL return an **operation function** (`op_fun`) rather than a tensor directly. This allows you to specify shape constraints and other options at the call site:

```ocaml
(* param_op_fun: the innermost layer of optional parameters *)
type param_op_fun =
  ?input_dims:int list ->
  ?output_dims:int list ->
  ?input_axes:(string * int) list ->
  ?output_axes:(string * int) list ->
  ?deduced:Shape.deduce_within_shape ->
  unit ->
  t

(* op_fun: extends param_op_fun with label and precision options *)
type op_fun =
  ?label:string list ->
  ?top_down_prec:bool ->
  ?batch_dims:int list ->
  ?batch_axes:(string * int) list ->
  param_op_fun
```

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `~input_dims:[n; m]` | Set input axis dimensions (rightmost in memory layout) |
| `~output_dims:[p; q]` | Set output axis dimensions |
| `~batch_dims:[b]` | Set batch axis dimensions (leftmost in memory layout) |
| `~input_axes:[("hidden", 64)]` | Input axes with labeled dimensions |
| `~output_axes:[("classes", 10)]` | Output axes with labeled dimensions |
| `~label:["layer1"]` | Labels for debugging/printing |
| `~top_down_prec:true` | Force precision from parent operation |
| `~deduced:Input_equals_output` | Shape constraint (input = output dims) |
| `()` | **Required**: finalizes the tensor creation |

**Usage examples:**

```ocaml
(* Operations like pointmul return op_fun, allowing shape specification *)
let scaled = pointmul x scale ~output_dims:[hidden_dim] ()

(* The kaiming/xavier functions return op_fun for deferred shape binding *)
let weights = kaiming uniform ~input_dims:[784] ~output_dims:[256] ()

(* Term creates a terminal tensor with shape *)
let input = TDSL.term ~batch_dims:[batch_size] ~input_dims:[784] ()

(* Params use the same interface *)
let w = TDSL.param "weights" ~input_dims:[in_dim] ~output_dims:[out_dim] ()
```

**Why this design?**

1. **Deferred shape binding**: Shape can be specified at the use site rather than definition site
2. **Shape inference integration**: Unspecified dimensions are inferred from context
3. **Flexible composition**: Functions like `kaiming` can wrap other functions while preserving the interface

For example, `kaiming_at init_f counter` returns an `op_fun`, so you can write:
```ocaml
(* Shape specified at call site, not in kaiming_at definition *)
let w = kaiming_at uniform_at counter ~input_dims:[100] ~output_dims:[40] ()
```

### Gradient Specifications

The `grad_spec` parameter controls gradient behavior:

```ocaml
type grad_spec = Require_grad | Prohibit_grad | If_needed
```

- **`Require_grad`**: Always create gradient nodes (for learnable parameters)
- **`Prohibit_grad`**: Never create gradients (for constants, inputs)
- **`If_needed`**: Create gradients only if required by the computation graph

The DSL modules set appropriate defaults:

- `TDSL` uses `If_needed` (automatic gradient propagation; `TDSL.param` adds the final parameter gradient)
- `NTDSL` uses `Prohibit_grad` (non-differentiable)

### Tensor Precision

OCANNL supports multiple numeric precisions:

```ocaml
Tensor.default_value_prec := Ir.Ops.single  (* float32, the default *)
Tensor.default_grad_prec := Ir.Ops.single

(* Available precisions *)
Ir.Ops.half    (* float16 *)
Ir.Ops.single  (* float32 *)
Ir.Ops.double  (* float64 *)
```

Individual tensor nodes can have their precision updated:
```ocaml
Ir.Tnode.update_prec tensor.value Ir.Ops.double
```

See [Precision Inference](docs/precision_inference.md) for details.

## Contexts

The `Context` module provides a simplified interface for backend management, introduced in v0.6.1.

### Context Creation

```ocaml doc-check=context_creation
open Ocannl.Nn_blocks.DSL_modules

(* Automatic backend selection (respects OCANNL_BACKEND env var) *)
let ctx = Context.auto ()

(* Explicit backend selection *)
let ctx = Context.cuda ~ordinal:0 ()  (* NVIDIA GPU *)
let ctx = Context.metal ~ordinal:0 () (* Apple Metal *)
let ctx = Context.cpu ~threads:4 ()     (* Multi-threaded CPU *)
```

### Compilation and Execution

```ocaml doc-check=context_compile
open Ocannl.Nn_blocks.DSL_modules

let compile_and_run ctx computation bindings =
  (* Compile a computation *)
  let ctx, routine = Context.compile ctx computation bindings in

  (* Execute the routine *)
  let ctx = Context.run ctx routine in

  (* Routine metadata is read straight off the record (private: readable, not constructible) *)
  let bindings = routine.Context.bindings in
  let routine_context = routine.Context.context in
  ctx, bindings, routine_context
```

### Node Initialization Tracking

Contexts track which tensor nodes have been initialized:

```ocaml
(* Check if a node is initialized *)
let is_init = Context.is_initialized ctx tensor.value

(* Initialize from host memory *)
let ctx = Context.init_from_host_deprecated ctx tensor.value

(* Copy between contexts *)
Context.copy ~src:ctx1 ~dst:ctx2 tensor.value
```

### Execution Dependency Tracking

When multiple routines are compiled from the same context lineage, OCANNL tracks **execution dependencies** — a compile-time DAG that records which routines must execute before others based on tensor-node read/write hazards.

#### Data Hazards

Three classic data hazards are tracked between routines in the same compilation lineage:

| Hazard | Rule | Example |
|--------|------|---------|
| **RAW** (Read After Write) | Routine reads a node last written by an earlier routine | `grad_routine` writes gradients, `sgd_routine` reads them |
| **WAR** (Write After Read) | Routine writes a node last read by an earlier routine | `forward_routine` reads params, `sgd_routine` writes params |
| **WAW** (Write After Write) | Routine writes a node last written by an earlier routine | Two routines both updating the same accumulator |

#### Dependency Scope Mirrors Compilation Lineage

Execution dependencies are scoped to compilation lineage, not to the root context. Compiling two routines from the *same* `Context.t` produces **independent siblings** — even if both routines write the same tensor node. Only routines compiled from the *returned* (child) context of a prior compilation see that prior routine in their frontier and can depend on it.

```ocaml
(* Sequential: b sees a in its frontier, may depend on a *)
let ctx1, routine_a = Context.compile ctx0 comp_a bindings in
let ctx2, routine_b = Context.compile ctx1 comp_b bindings in

(* Sibling: b does NOT see a — independent even on shared nodes *)
let ctx1, routine_a = Context.compile ctx0 comp_a bindings in
let ctx2, routine_b = Context.compile ctx0 comp_b bindings in

(* Via Train.to_routine: chains through the returned post-compile context *)
let ctx_a, routine_a = Train.to_routine ctx bindings comp_a in
let ctx_b, routine_b = Train.to_routine ctx_a bindings comp_b in
(* routine_b depends on routine_a — sequential via the child context *)
```

#### API

`Context.routine` is a `private` record: only `Context.compile` constructs routines, but every
field is readable, including the per-routine metadata:

```ocaml
routine.Context.routine_id      (* unique integer identifying the routine *)
routine.Context.name            (* name of the routine, from backend compilation *)
routine.Context.execution_deps  (* Set of routine IDs that must execute before this routine *)
routine.Context.inputs          (* materialized nodes read before written, if written at all *)
routine.Context.outputs         (* materialized nodes the routine writes *)

(* Whether all execution dependencies have been satisfied *)
val can_run : t -> routine -> bool
```

#### Enforcement

`Context.run` checks execution dependencies before running a routine. If any prerequisite routine has not been executed, it raises `Failure` with the missing dependency names. After successful execution, the routine is marked as executed in the shared ledger.

Re-running a routine that has already executed is allowed — its dependencies remain satisfied.

```ocaml
(* Typical pattern: grad_update then sgd_update *)
let grad_ctx, grad_routine = Train.to_routine ctx bindings grad_comp in
let _, sgd_routine = Train.to_routine grad_ctx bindings sgd_comp in

(* grad has no deps (first in lineage), sgd depends on grad *)
assert (Context.can_run ctx grad_routine);
assert (not (Context.can_run ctx sgd_routine));

let ctx = Context.run ctx grad_routine in
assert (Context.can_run ctx sgd_routine);
let ctx = Context.run ctx sgd_routine in

(* Re-execution works — deps are still satisfied *)
let ctx = Context.run ctx grad_routine in
let _ctx = Context.run ctx sgd_routine in
```

#### Shared Gradients

A tensor shared by two losses -- an optimized `Require_grad` input under a content and a style
objective, a parameter, a trunk -- receives each loss's contribution to its gradient from that
loss's backprop, but only its first-constructed consumer carries its gradient zeroing (and, for a
trunk, the backprop propagating from it). Two `Train.grad_update` routines over it in one lineage
would compute neither loss's gradient, so `Context.compile` refuses the second, naming the gradient
and both routines (gh-ocannl-1744). Compute the objectives in one routine, as `Train.grad_update` of
a loss over their sum; or compile `Train.zero_params_grads ~params` first and then a
`Train.grad_update ~accum_steps:1` per objective, neither of which zeroes the shared leaf: after the
reset and both runs, its gradient is the sum's.

A routine that accumulates into a gradient without zeroing it sums every run's contribution until
something resets the gradient. `Context.run` therefore refuses its run once it has run as many times
as its computation allows since the gradient's last reset -- once, or `k` times for
`Train.grad_update ~accum_steps:k`, the explicit opt-in for accumulating over micro-steps
(gh-ocannl-1781). A reset is a run of a routine zeroing the gradient (`Train.zero_params_grads`, a
plain `Train.grad_update` of the loss carrying the zeroing) or a write from outside a run
(`Context.set_values`, `Context.from_host`, `Context.copy`). The refusal names the gradient and any
routine of the lineage that zeroes it but has not run since. Counting is per routine, so the common
reset above -- one reset, then one run of each objective's update -- stays legal.

## The Train Module

The `Train` module provides high-level utilities for training workflows.

### Parameter Initialization

```ocaml
(* Initialize all parameters of a tensor *)
let ctx = Train.init_params ctx bindings loss

(* With options *)
let ctx = Train.init_params
  ~reinit_all:true      (* Reinitialize even if already initialized *)
  ~hosted:true          (* Keep values accessible on host *)
  ctx bindings loss
```

### Forward and Backward Passes

```ocaml
(* Forward pass only *)
let ctx = Train.forward_once ctx tensor

(* Forward + backward (gradient update) *)
let ctx = Train.update_once ctx loss

(* Build gradient update computation *)
let update = Train.grad_update loss
(* Returns: forward + zero_grads + set grad to 1 + backprop *)

(* Build SGD parameter update *)
let sgd = Train.sgd_update ~learning_rate loss
```

### Training Loops

For efficient training, compile routines once and execute repeatedly:

```ocaml
let ctx = Context.auto () in
let step_n, bindings = IDX.get_static_symbol IDX.empty in

(* Define model and loss *)
let%op loss = ... in

(* Build update computations *)
let%op learning_rate = 0.01 in
let update = Train.grad_update loss in
let sgd = Train.sgd_update ~learning_rate loss in

(* Initialize parameters *)
let ctx = Train.init_params ctx bindings loss in

(* Compile training routine *)
let ctx, routine = Train.to_routine ctx bindings
  (Asgns.sequence [update; sgd]) in

(* Training loop *)
let step_ref = IDX.find_exn routine.Context.bindings step_n in
for step = 1 to num_steps do
  step_ref := step;
  Train.run ctx routine;
  if step mod 100 = 0 then
    printf "Step %d, Loss: %.4f\n" step (ctx, loss).@[0]
done
```

See [migration_guide.md](migration_guide.md) for more training patterns.

### Stopping and Resuming Training

`Checkpoint` (gh-ocannl-96) saves what a training loop needs to resume and continue exactly as
the uninterrupted run would have. Two kinds of state are involved:

- **Device state**: `Checkpoint.training_state ~routines loss` is the loss's `params` plus every
  node the compiled routines carry from one run to the next — the optimizer's moments (an SGD
  momentum buffer), accumulators. It is read off the lowered routines (a node the routine both
  writes and reads before writing), so it needs no labels or registration: an update reading its
  own state without the clearing `=:` prefix declares that state persistent.
- **Host-side progress**: a `Checkpoint.progress` record of the step and epoch counters, the
  position in the data, an optional `Stdlib.Random.State.t` (say, the generator shuffling the
  data), and free-form `extra` key-value pairs.

`Checkpoint.save` writes both into one file, in one atomic write (a crash mid-save leaves the
previous checkpoint whole). A resumed process rebuilds the same model and routines in the same
order — the checkpoint matches nodes by id — initializes the parameters as usual, then restores:

```ocaml
let state = Checkpoint.training_state ~routines:[ routine ] loss in
(* Every so often: *)
Checkpoint.save ~ctx
  ~progress:{ step; epoch; position; host_rng = Some rng; extra = [] }
  state path;
(* In the resumed process, after rebuilding the program and Train.init_params: *)
let ctx, progress = Checkpoint.restore ~ctx state path in
(* ... continue the loop from progress.step, progress.epoch, progress.position,
   re-arming step-dependent inputs (e.g. a scheduled learning rate) from progress.step *)
```

`Checkpoint.restore` refuses a state set different from the one saved (a set missing the
optimizer moments would silently restart them) and a plain `Persistence` checkpoint, which
records no progress. Device-side randomness needs no generator state saved: random operations are
counter-based (see [Counter-Based PRNG](#counter-based-prng)), so the same seed with counters
derived from the restored step draws the same numbers. The checkpoint therefore records the seeds
(`Tensor.random_seeds`: every one the session created, including a graph's scoped seed), and
`Checkpoint.restore` refuses a program built under different ones, naming both; a checkpoint from
before the seed was recorded restores with a warning instead.
`test/training/resume_checkpoint.ml`
resumes a run in a fresh process and checks it matches the uninterrupted one bit for bit.

## Reusable Tensor Programs

A `%op` function is a tensor program defined once and applied many times. It gets a *shape
scheme* automatically: the shape constraints its body imposes, traced once when its closure is
built and checked at every application ([Automatic shape schemes](syntax_extensions.md#automatic-shape-schemes);
the mechanism is in [Shape schemes](shape_inference.md#shape-schemes)). The tutorial
[`test/operations/reusable_tensor_programs.ml`](../test/operations/reusable_tensor_programs.ml)
walks through what that buys, in three steps; its `%expect` blocks are its real output, checked by
`dune build @test/operations/runtest-reusable_tensor_programs`, and every excerpt below is checked
to be verbatim from it.

### One Function, Two Shapes

The layer's `()` application creates its parameters `w` and `b` and returns the function that
uses them; the input size of `w` is left to inference:

```ocaml doc-excerpt=test/operations/reusable_tensor_programs.ml
let%op mlp_layer ~hid () x = relu (({ w; o = [ hid ] } * x) + { b = 0.; o = [ hid ] })
```

`Tensor.Scheme.to_string` prints a scheme like a type, `batch|input->output` for the argument,
then `=>` and the result. A row variable `..'a..` stands for any number of axes and is
generalized: each application gets its own. `..'_a..` is fixed, shared by every application
(here, the input row of `w`), the way OCaml prints a weak type variable. The indented lines are
the broadcasting relations, `r ⊑ o` reading "the result row `r` refines the operand row `o`":

```text doc-excerpt=test/operations/reusable_tensor_programs.ml
layer : ..'a..|..'b..->..'c.. => ..'d..|..'e..->..'f..,4
  ..'_a.. ⊑ ..'c..
  ..'d.. ⊑ ..'a..
  ..'e.. ⊑ ..'_b..
  ..'e.. ⊑ ..'b..
```

The same `layer` then applies at batch shapes `2` and `5,7`, each application checked against
the scheme before its body builds, giving results of shapes `2|4` and `5,7|4`.

### A Shape Mistake, Named

A shape mistake inside a tensor function is reported naming the function, and as early as the
mistake allows. One that no argument can fix -- here a bias declared with the hidden size
instead of the output size -- fails when the closure is built, and
`Tensor.Scheme.check_definition` finds it at program start from a trial configuration:

```ocaml doc-excerpt=test/operations/reusable_tensor_programs.ml
let%op two_layer ~hid ~out () x =
  let h = relu (({ w1; o = [ hid ] } * x) + { b1 = 0.; o = [ hid ] }) in
  ({ w2; o = [ out ] } * h) + { b2 = 0.; o = [ hid ] }
```

```text doc-excerpt=test/operations/reusable_tensor_programs.ml
checking the definition two_layer: tracing the shape scheme two_layer: solved dimensions for axis: mismatch
  conflicting sizes: 2 vs 8
```

One that depends on the argument fails at the application, before the body runs: a residual
block whose layer is 4 wide, applied to 3 features, reads

```text doc-excerpt=test/operations/reusable_tensor_programs.ml
Shape_error: applying the shape scheme residual_block to x: dimension comparison for axis: mismatch
  conflicting sizes: 4 vs 3
```

With the scheme opted out (`[@@ocannl.no_scheme]`), the same mistakes surface only inside the
body, naming neither the function nor the argument, and leave the session tainted.

### Recovering a Session from a Failed Construction

At a toplevel or in a notebook, a construction that fails inside its body taints the session:
every later construction is refused, naming the first failure. Take a checkpoint before trying
things out, and rewind to it after a failure; `Tensor.with_checkpoint` rewinds only when its
body raises:

```ocaml doc-excerpt=test/operations/reusable_tensor_programs.ml
let before_model = Tensor.checkpoint ~label:"before the model" () in
```

```ocaml doc-excerpt=test/operations/reusable_tensor_programs.ml
Tensor.rewind before_model;
(* [Tensor.with_checkpoint] does both in one phrase: it rewinds only when its body raises. *)
attempt (fun () -> Tensor.with_checkpoint (fun () -> [%op x + { bias = 0.; o = [ 5 ] }]));
```

The rewind discards everything created since the checkpoint -- a failed construction, and the
parameters of an attempt the scheme refused -- and clears the taint; the session then builds,
compiles and runs the right model. `Tensor.rewind_to_last_closure` recovers without a checkpoint,
discarding everything opened since shape inference last closed (at the last compile, say).

## Randomness and Parameter Initialization

OCANNL uses a deterministic, counter-based approach to random number generation, which differs significantly from traditional imperative RNGs.

### Counter-Based PRNG

OCANNL implements the Threefry algorithm, a counter-based PRNG that:

- Produces deterministic sequences from a seed and counter
- Enables reproducible initialization across runs
- Supports parallel random generation without synchronization

The random seed is managed globally:
```ocaml
(* Set the random seed *)
Tensor.set_random_seed ~seed:42 ()

(* Or configure via settings before tensor creation *)
Utils.settings.fixed_state_for_init <- Some 42;
Tensor.unsafe_reinitialize ()
```

### Randomness Operations

OCANNL provides several random operations in the DSL modules:

| Function | Description | Notes |
|----------|-------------|-------|
| `uniform ()` | Uniform [0,1) using global seed | Efficient, works with any shape |
| `uniform1 ()` | Uniform [0,1), pointwise | Deprecated (gh-ocannl-509); one value per 128-bit block |
| `uniform_at counter` | Uniform using explicit counter | For training-time randomness |
| `uniform_at1 counter` | Pointwise uniform with counter | For training-time, any shape |
| `normal ()`, `normal1 ()` | Standard normal N(0,1) | Uses Box-Muller transform |
| `normal_at counter`, `normal_at1 counter` | Normal with explicit counter | For training-time randomness |

**Important**: The `counter` argument in `_at` variants is for **randomness bifurcation**, not shape determination. The counter should be scalar / dimension-1 (e.g., a training step number). Different counter values produce different random streams across the resulting tensor cells.

The output shape is determined by:
1. **Shape inference** from how the result is used (e.g., pointwise ops with shaped tensors)
2. **Explicit dimensions** via `TDSL.uniform_at ~output_dims:[...] counter ()`

```ocaml
(* Dropout with training-step-dependent randomness *)
let%op dropout ~rate () ~train_step x =
  match train_step with
  | Some train_step when Float.(rate > 0.0) ->
      (* !@train_step embeds the step counter as a scalar tensor *)
      (* Shape is inferred from pointwise comparison with x *)
      x *. (!.rate < uniform_at !@train_step) /. (1.0 - !.rate)
  | _ -> x
```

**Note on shapes**: The packed `uniform` function (without `1`) is total over shapes: element `i` of the result is lane `i mod k` of the 128-bit random block `i / k`, where `k = 16 / bytes-per-element` (4 for single precision). When the total element count is not a multiple of `k`, the final block is consumed partially, so growing a tensor keeps the existing prefix of its value stream unchanged. `uniform1` remains available as a pointwise variant (one value per 128-bit block).

**Virtual (inlined) uniforms**: a packed `uniform` result does not need to be materialized — when the optimizer virtualizes it (e.g. a dropout mask read once by an elementwise consumer), each read cell inlines as lane `i mod k` of the conversion of counter block `i / k`, which is bitwise-identical to the materialized value stream. The uniform's internal counter tensor then stays materialized (usually as routine-local scratch) because it is read at a runtime-computed block index; the conversion is recomputed per cell, but the threefry chain still runs once per 128-bit block.

### Kaiming and Xavier Scaling Operations

For weight matrices, OCANNL provides scaled initialization functions that use shape inference to determine the scaling factor:

```ocaml
(* Kaiming (He) initialization: sqrt(scale_sq / fan_in) scaling, default scale_sq=6 *)
let kaiming ?scale_sq init_f () = ...

(* Xavier (Glorot) initialization: sqrt(scale_sq / (fan_in + fan_out)) scaling *)
let xavier ?scale_sq init_f () = ...

(* Counter-based variants for deferred initialization *)
let kaiming_at ?scale_sq init_f counter = ...
let xavier_at ?scale_sq init_f counter = ...
```

These functions use einsum dimension capture to extract fan_in and fan_out:

```ocaml
(* kaiming_impl captures input dimension *)
let%op _ = w_raw ++ "...|..i.. -> ... => |->0" [ "i" ] in
... sqrt (!.scale_sq /. dim i)

(* xavier_impl captures both input and output dimensions *)
let%op _ = w_raw ++ "...|..i.. -> ..o.. => |->0" [ "i"; "o" ] in
... sqrt (!.scale_sq /. (dim i + dim o))
```

### Parameter Initialization

By default, `%op` parameters use a centered, scaled packed `uniform` initializer
with values in `[-0.25, 0.25)`. The packed `uniform` is total over shapes (see
the note above), so no pointwise fallback is needed; set
`TDSL.default_param_init := Operation.default_uniform1_param_init` to reproduce
pre-0.9 random streams.

Usage example:
```ocaml
(* Set kaiming initialization as default. Initializers are forward-only. *)
TDSL.default_param_init := NTDSL.kaiming TDSL.O.uniform;

(* Or use directly in parameter definition *)
let%op layer x = { w = kaiming uniform () } * x + { b = 0. }
```

When setting `default_param_init`, use `NTDSL` helpers: initialization is a one-shot forward computation. `TDSL.param` makes the final parameter tensor differentiable after the initializer has produced its value.

## Memory Management

OCANNL manages tensor memory through memory modes:

```ocaml
(* Make a tensor's value persisted on-device and inspectable on demand via the context.
   After gh-ocannl-333 there is no separate "hosted" mode; CPU access is a context-mediated
   device-to-host transfer (Context.to_host / Context.get_values). *)
Train.set_materialized tensor.value

(* Set as virtual (inlined during compilation) *)
Train.set_virtual tensor.value
```

The `Train` module functions often set appropriate memory modes automatically:
- `forward` and `forward_once` set the result as hosted
- `init_params` sets parameter values as hosted by default

For advanced memory control, see [lowering_and_inlining.md](lowering_and_inlining.md).

## Further Reading

- [Syntax Extensions](syntax_extensions.md) - `%op` and `%cd` syntax details
- [Shape Inference](shape_inference.md) - How shapes are inferred
- [Migration Guide](migration_guide.md) - Coming from PyTorch/TensorFlow
- [Anatomy of a Backend](anatomy_of_a_backend.md) - Backend implementation details
