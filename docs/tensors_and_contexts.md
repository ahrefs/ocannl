# Tensors and Contexts

This document describes how to work with tensors and execution contexts in OCANNL. It covers the core `Tensor` module, the `Context` API for backend management, and the `Train` module for training workflows.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Tensors](#tensors)
  - [Tensor Structure](#tensor-structure)
  - [Roots, Embedded Nodes, and Params](#roots-embedded-nodes-and-params)
  - [Creating Tensors](#creating-tensors)
  - [Operation Functions and Shape Parameters](#operation-functions-and-shape-parameters)
  - [Gradient Specifications](#gradient-specifications)
  - [Tensor Precision](#tensor-precision)
- [Contexts](#contexts)
  - [Context Creation](#context-creation)
  - [Compilation and Execution](#compilation-and-execution)
  - [Node Initialization Tracking](#node-initialization-tracking)
- [The Train Module](#the-train-module)
  - [Parameter Initialization](#parameter-initialization)
  - [Forward and Backward Passes](#forward-and-backward-passes)
  - [Training Loops](#training-loops)
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

```ocaml
type t = {
  params : (t, comparator_witness) Base.Set.t;  (* Learnable parameters *)
  forward : comp;                                (* Forward computation *)
  diff : diff option;                            (* Gradient info if differentiable *)
  id : int;                                      (* Unique identifier *)
  value : tn;                                    (* The underlying tensor node *)
  shape : Shape.t;                               (* Shape with inference state *)
  children : subtensor list;                     (* Sub-tensors in the computation *)
  (* ... *)
}
```

Key fields:
- **`value`**: The tensor node (`Ir.Tnode.t`) holding the actual data
- **`diff`**: Contains gradient node and backpropagation code (if differentiable)
- **`params`**: Set of tensors requiring separate initialization (see [Roots, Embedded Nodes, and Params](#roots-embedded-nodes-and-params))
- **`forward`**: Computation to produce this tensor's value, including embedded subtensor computations
- **`shape`**: Shape information, progressively refined during inference

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
let weights = kaiming uniform1 ~input_dims:[784] ~output_dims:[256] ()

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

- `TDSL` uses `If_needed` (automatic gradient propagation)
- `NTDSL` uses `Prohibit_grad` (non-differentiable)
- `PDSL` uses `Require_grad` (parameters)

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

```ocaml
(* Automatic backend selection (respects OCANNL_BACKEND env var) *)
let ctx = Context.auto ()

(* Explicit backend selection *)
let ctx = Context.cuda ~device_id:0 ()  (* NVIDIA GPU *)
let ctx = Context.metal ~device_id:0 () (* Apple Metal *)
let ctx = Context.cpu ~threads:4 ()     (* Multi-threaded CPU *)
```

### Compilation and Execution

```ocaml
(* Compile a computation *)
let ctx, routine = Context.compile ctx computation bindings

(* Execute the routine *)
let ctx = Context.run ctx routine

(* Access routine metadata *)
let bindings = Context.bindings routine
let ctx = Context.context routine
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
let routine = Train.to_routine ctx bindings
  (Asgns.sequence [update; sgd]) in

(* Training loop *)
let step_ref = IDX.find_exn (Context.bindings routine) step_n in
for step = 1 to num_steps do
  step_ref := step;
  Train.run ctx routine;
  if step mod 100 = 0 then
    printf "Step %d, Loss: %.4f\n" step loss.@[0]
done
```

See [migration_guide.md](migration_guide.md) for more training patterns.

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
| `uniform ()` | Uniform [0,1) using global seed | Efficient, requires shape divisibility |
| `uniform1 ()` | Uniform [0,1), pointwise | Works with any shape, less efficient |
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

**Note on shape constraints**: The `uniform` function (without `1`) requires the total number of elements to be appropriately divisible, e.g. by 4 for single precision (due to `uint4x32` efficiency). Use `uniform1` for arbitrary shapes at the cost of some efficiency.

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
let%op _ = w_raw ++ "...|..i.. -> ... => 0" [ "i" ] in
... sqrt (!.scale_sq /. dim i)

(* xavier_impl captures both input and output dimensions *)
let%op _ = w_raw ++ "...|..i.. -> ..o.. => 0" [ "i"; "o" ] in
... sqrt (!.scale_sq /. (dim i + dim o))
```

### Parameter Initialization

Usage example:
```ocaml
(* Set kaiming initialization as default: PDSL outside, TDSL inside *)
TDSL.default_param_init := PDSL.kaiming TDSL.O.uniform1;

(* Or use directly in parameter definition *)
let%op layer x = { w = kaiming uniform1 () } * x + { b = 0. }
```

When setting `default_param_init`, we call `PDSL.kaiming` so that the result is differentiable, but `TDSL.O.uniform1` or `NTDSL.O.uniform1` so the intermediate values are not differentiable.

## Memory Management

OCANNL manages tensor memory through memory modes:

```ocaml
(* Make a tensor's value accessible on host *)
Train.set_hosted tensor.value

(* Set as materialized (on-device only) *)
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
