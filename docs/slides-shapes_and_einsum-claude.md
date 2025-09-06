# Shape Inference and Einsum in OCANNL: From Beginner to Advanced

{pause}

My work on OCANNL is sponsored by **[a]{style="color:orange"}[hrefs]{style="color:blue"}**

{pause}

{#shape-intro .definition title="What are Shapes in OCANNL?"}

## Tensors = N-dimensional Arrays with Structure

* Every tensor has a **shape**: the structure and dimensions of its data
* OCANNL shapes have **three kinds of axes**:
  * **Batch axes**: for processing multiple data points together
  * **Input axes**: dimensions that get "consumed" in operations
  * **Output axes**: dimensions that get "produced" by operations
* Shape notation: `batch|input->output` (or simpler forms when some parts are empty)

{pause}

{#why-three-kinds .block title="Why Three Kinds of Axes?"}

> * **Batch axes** enable data parallelism - process many examples at once
> * **Input/Output distinction** makes matrix operations natural:
>   * Matrix multiply: `(m×n) × (n×k) → (m×k)`
>   * In OCANNL: `m->n` times `n->k` gives `m->k`
> * This structure enables powerful **shape inference**
> * **Expressivity gains**: `~logic:"@"` for tensor multiplication, multiple row variables for flexible patterns

{pause up=shape-intro}

---

{#shape-basics}

## Creating Tensors with Shapes

{pause up=shape-basics}

### Simple tensor creation with automatic shape inference

```ocaml
let%op x = { x }  (* Shape will be inferred from usage *)
let%op w = { w; o = [hidden_dim] }  (* Output dimension specified *)
let%op b = { bias; o = [hidden_dim]; i = [] }  (* Scalar bias per output *)
```

{pause}

### Dimension specification shortcuts

* `o` = `output_dims`: list for output axes `[dim1; dim2; ...]`
* `i` = `input_dims`: tuple for input axes `(dim1, dim2, ...)`  
* `b` = `batch_dims`: array for batch axes `[|dim1; dim2; ...|]`

{pause}

### Example: A simple linear layer

```ocaml
let%op linear_layer ~hidden_dim () x =
  { w } * x + { b; o = [hidden_dim] }
```

Shape inference determines:

* `w` has shape `hidden_dim->input_dim` (where `input_dim` comes from `x`)
* `b` has shape `hidden_dim` (broadcast to match)
* Result has shape matching `x` but with `hidden_dim` output

{pause down}

---

{#broadcasting}

## Broadcasting: Making Tensors Compatible

{pause up=broadcasting}

{.definition title="Broadcasting Rules"}

> * Dimension-1 axes can match any dimension (like NumPy)
> * Shorter shapes extend "to the left" with new axes as needed
> * OCANNL tracks this with **row variables** for precise control

{pause}

### Broadcasting Examples

```ocaml
(* Pointwise operations broadcast automatically *)
let%op scaled = tensor *. 2.0  (* scalar broadcasts to tensor shape, *. is pointwise *)
let%op biased = matrix + { bias; o = [output_dim] }  (* bias broadcasts across batch *)

(* Matrix multiply preserves batch dimensions *)  
let%op batched_result = batched_matrix * weight_matrix
(* If batched_matrix is batch|->input and weight_matrix is output->input,
   result is batch|->output *)
```

{pause}

{.remark}
Unlike NumPy, OCANNL can broadcast "in the middle" - preserving both leading and trailing axes!

{pause down}

---

{#einsum-intro}

## Einsum: The Universal Tensor Operation Language

{pause up=einsum-intro}

{.definition title="What is Einsum?"}

> Einstein summation notation describes tensor operations by labeling axes:
> * Same labels across tensors must match dimensions
> * Labels that disappear get summed over (reduced)
> * New label arrangements permute axes
> * **OCANNL syntax**: `=>` separates arguments from result (not `->` like NumPy)

{pause}

### OCANNL's Einsum Operators

* `+*` for binary einsum (multiply-accumulate): `tensor1 +* "spec" tensor2`
* `++` for unary einsum (permute/reduce): `tensor ++ "spec"`

{pause}

### Basic Einsum Syntax

```ocaml
(* Matrix multiply: sum over shared dimension *)
let%op result = a +* "ik;kj=>ij" b

(* Batched matrix multiply *)
let%op batched = a +* "bik;bkj=>bij" b  

(* Outer product: no shared dimensions *)
let%op outer = a +* "i;j=>ij" b

(* Reduction: sum all elements *)
let%op sum = tensor ++ "...=>0"
```

{pause down}

---

{#einsum-syntax}

## OCANNL's Extended Einsum Notation

{pause up=einsum-syntax}

### Two Modes: Single-char vs Multi-char

* **Single-char mode** (default): Each character is an axis label
  * `"ijk=>kji"` - reverse three axes
* **Multi-char mode** (has commas): Labels are separated
  * `"batch,height,width=>width,height,batch"`
  * Trigger with comma: `"input->output, => output->input"` 

{pause}

### The Three-Part Structure

Einsum specs use OCANNL's axis kinds: `batch|input->output`

```ocaml
(* Full specification *)
"batch|input->output => new_batch|new_input->new_output"

(* Often simpler *)
"input->output => output->input"  (* transpose *)
"i->rgb => rg->ib"  (* permute RGB channels, single-char mode *)
```

{pause}

### Special Syntax Elements

* `_` placeholder: aligns axes without creating variables
* Numbers for fixed indices: `"i=>i0"` (slice at index 0)
* `...` ellipsis: matches any number of axes (row variable)

{pause down}

---

{#ellipsis-magic}

## Row Variables and Ellipsis: Advanced Broadcasting

{pause up=ellipsis-magic}

{.definition title="Row Variables"}

> Row variables (`...` or `..name..`) match zero or more axes:
> * Enable operations on tensors with unknown number of dimensions
> * Preserve structure you don't need to specify
> * Different from NumPy: can appear in the middle!

{pause}

### Ellipsis Examples

```ocaml
(* Sum over all axes to scalar *)
let%op total = tensor ++ "...=>0"

(* Preserve all batch dimensions, transpose input/output *)  
let%op transposed = tensor ++ "...|i->o => ...|o->i"

(* Reduce batch dimensions into the result *)
let%op reduced = tensor ++ "...|...->... => ...->..."

(* Works with any number of batch axes! *)
let%op flexible = a +* "...|i->...; ...|...->i => ...|i" b
```

{pause}

### Context-Dependent Ellipsis

* `...` in batch position means `..batch..`
* `...` in input position means `..input..`  
* `...` in output position means `..output..`

{pause down}

---

{#convolution-syntax}

## Convolution Support: Strided and Dilated Operations

{pause up=convolution-syntax}

{.definition title="Convolution Expressions"}

> OCANNL supports convolution-style indexing with the `+` operator:
> * `stride*output + dilation*kernel`
> * Automatically handles padding when configured
> * Enables CNN operations naturally

{pause}

### Convolution Syntax

```ocaml
(* Basic 1D convolution *)
let%op conv1d = input +* "i+k; k->o => i->o" kernel
(* i = output_position + kernel_position *)

(* With stride and dilation *)
let stride = 2 and dilation = 3 in
let%op strided_conv = 
  input +* "stride*o+dilation*k; k->c => o->c" kernel

(* 2D convolution *)  
let%op conv2d = 
  image +* "h+kh,w+kw,c; kh,kw,c->f => h,w->f" kernel
```

{pause}

{.remark}
Convolution expressions automatically trigger multi-char mode due to the `+` operator.

{pause down}

---

{#practical-patterns}

## Practical Patterns: Building Neural Networks

{pause up=practical-patterns}

### Common Operations with Einsum

```ocaml
(* Attention mechanism: Q×K^T with explicit batch axis *)
let%op attention_scores = 
  queries +* "b|qd;b|kd=>b|qk" keys

(* Apply attention weights to values with batch axis *)
let%op attended = 
  attention_weights +* "b|qk;b|kd=>b|qd" values

(* Global average pooling *)
let%op global_avg = 
  features ++ "batch,height,width,channels=>batch->channels" 
    /. float(height * width)

(* Channel-wise operations - note: * is tensor mult, *. is pointwise *)
let%op normalized = 
  input *. { scale; o = [channels] } + { shift; o = [channels] }
```

{pause}

### Flexible Multi-Head Attention Example

```ocaml
(* Works with 1 or 2 batch axes, 1D or 2D attention! *)
let%op multihead_attention ~heads () q k v =
  (* Split heads: ..batch..|..spatial..->d => ..batch..|heads,..spatial..->d_per_head *)
  let q_heads = q ++ "...|...->hd => ...|h,...->d" in
  let k_heads = k ++ "...|...->hd => ...|h,...->d" in
  let v_heads = v ++ "...|...->hd => ...|h,...->d" in
  (* Attention: preserves batch AND spatial structure *)
  let scores = q_heads +* "...|h,q,...->d; ...|h,k,...->d => ...|h,q,...->k,..." k_heads in
  let attended = softmax(scores) +* "...|h,q,...->k,...; ...|h,k,...->d => ...|h,q,...->d" v_heads in
  (* Concatenate heads back *)
  attended ++ "...|h,q,...->d => ...|q,...->hd"
```

{pause}

### Avoiding Dimension Commitment

```ocaml
(* Works for any number of spatial dimensions! *)
let%op flexible_pool = features ++ "b|...,c=>b|c"

(* Batch-size agnostic *)
let%op process = model { input; b = [...] }
```

{pause down}

---

{#max-pooling-challenge}

## Challenge: Implementing Max Pooling

{pause up=max-pooling-challenge}

{.block title="Your Task"}

> Design a max pooling layer that:
> * Takes 2D input with channels
> * Pools over spatial dimensions
> * Preserves batch and channel dimensions
> * Uses einsum notation where beneficial

{pause}

### Solution Approach

```ocaml
let%op max_pool2d ~pool_size input =
  (* Reshape input to expose pooling windows *)
  let h, w, c = get_dims input in
  let ph, pw = pool_size, pool_size in
  
  (* Use einsum to rearrange for pooling *)
  let reshaped = input ++ 
    "b,h,w,c => b,h/ph,ph,w/pw,pw->c" in
  
  (* Max over pooling dimensions *)
  let pooled = max_over_axes reshaped ~axes:[2; 4] in
  
  (* Result shape: batch,h/ph,w/pw->channels *)
  pooled
```

{pause}

{.remark}
Note: You can define a custom max-reduction operation similar to `einsum1` in lib/operation.ml, using max as the accumulation operator. Operations are user-extensible!

{pause down}

---

{#projections-advanced}

## Under the Hood: Projections and Shape Inference

{pause up=projections-advanced}

{.definition title="How Shape Inference Works"}

> OCANNL uses constraint solving similar to type inference:
> * Collects equality and inequality constraints between dimensions
> * Row variables enable polymorphic shapes (like type variables)
> * Solves constraints to determine concrete dimensions
> * Happens automatically - you rarely need to think about it!

{pause}

### Projections: How Tensors Are Indexed

* **Projections** map high-level operations to loop indices
* Shape inference determines which loops to generate
* Broadcasting becomes fixed indices in projections

{pause}

### Example: Matrix Multiply Projections

```ocaml
(* For C = A × B where A is m×k, B is k×n *)
(* Generated loops (conceptually): *)
for i = 0 to m-1 do
  for j = 0 to n-1 do  
    for k = 0 to k-1 do
      C[i,j] += A[i,k] * B[k,j]
```

Einsum `"ik;kj=>ij"` generates exactly these projections!

{pause down}

---

{#shape-debugging}

## Debugging Shapes: Tips and Tools

{pause up=shape-debugging}

### Inspecting Tensor Shapes

```ocaml
(* Print shape of a tensor *)
let%op t = some_operation input in
Printf.printf "Shape: %s\n" (Tensor.shape_to_string t);

(* Use labels for clarity *)
let%op labeled = { weights = uniform(); o = [hidden] } in
(* Shape will show: "weights:hidden->?" until input dims are known *)
```

{pause}

### Common Shape Errors and Solutions

1. **Dimension mismatch**: Check einsum specs match actual shapes
2. **Missing broadcasts**: Use `...` to handle variable dimensions  
3. **Wrong axis kind**: Remember batch|input->output structure

{pause}

### Shape Inference Stages

1. **Construction time**: Initial constraints collected
2. **Propagation**: Constraints solved during tensor composition
3. **Finalization**: Remaining variables resolved to concrete dimensions

{pause down}

---

{#advanced-patterns}

## Advanced Patterns: Leveraging Shape Inference

{pause up=advanced-patterns}

### Writing Shape-Polymorphic Functions

```ocaml
(* Works with any input shape! *)
let%op normalize ?(epsilon=1e-5) tensor =
  let mean = tensor ++ "...=>0" /. size_of tensor in
  let centered = tensor - mean in
  let variance = (centered *. centered) ++ "...=>0" /. size_of tensor in
  centered / sqrt(variance + epsilon)
```

{pause}

### Conditional Shapes with Row Variables

```ocaml
(* Flexible reduction - preserves structure *)
let%op smart_reduce mode tensor = 
  match mode with
  | `KeepBatch -> tensor ++ "..batch..|...->... => ..batch.."
  | `ReduceAll -> tensor ++ "...=>0"
  | `LastAxis -> tensor ++ "...x=>..."
```

{pause}

### Building Reusable Components

```ocaml
(* Generic convolution block *)
let%op conv_block ~filters ~kernel_size ~stride () input =
  let conv = input +* make_conv_spec ~stride ~kernel_size in
  batch_norm (relu conv)
```

{pause down}

---

{#einsum-vs-operators}

## When to Use Einsum vs Standard Operators

{pause up=einsum-vs-operators}

{.block title="Use Standard Operators When:"}

> * Simple pointwise operations: `+`, `-`, `*.`, `/`
> * Standard matrix multiply: `*` (tensor multiply)
> * Clear, readable code for common operations

{pause}

{.block title="Use Einsum When:"}

> * Complex axis permutations needed
> * Custom reductions over specific axes
> * Batched operations with non-standard broadcasting
> * You need precise control over which axes are summed

{pause}

### Comparison Example

```ocaml
(* Standard operator - clear and simple *)
let%op y = w * x + b

(* Einsum equivalent - more verbose but explicit *)
let%op y = w +* "oi;i=>o" x + b

(* Einsum shines for complex operations *)
let%op attention = 
  softmax (q +* "bhqd;bhkd=>bhqk" k / sqrt(float d))
```

{pause down}

---

{#summary}

## Summary: Mastering Shapes and Einsum

{pause up=summary}

{.block title="Key Takeaways"}

> 1. **Three-axis structure** (batch|input->output) makes operations natural
> 2. **Shape inference** eliminates most manual dimension tracking
> 3. **Row variables** (`...`) enable dimension-agnostic code
> 4. **Einsum notation** provides precise control when needed
> 5. **Convolution syntax** handles strided/dilated operations naturally

{pause}

### Your Shape Inference Toolkit

* Start simple: let shapes be inferred
* Specify only what's necessary: `{ w; o = [hidden] }`
* Use `...` for flexible dimensions
* Apply einsum for complex patterns
* Debug with shape printing when needed

{pause}

### Next Steps

* Try building a CNN with convolution expressions
* Implement attention mechanisms using einsum
* Experiment with shape-polymorphic utility functions
* Explore how projections optimize your operations

{pause}

{.remark}
Remember: OCANNL's shape system is designed to help you focus on the algorithm, not the bookkeeping!
