# OCANNL: Mysteries of NN Training Unveiled

## Slide 1

OCANNL
Mysteries of NN training unveiled

---

## Slide 2

My work on OCANNL is sponsored by **ahrefs**

---

## Slide 3

### OCaml Compiles Algorithms for Neural Network Learning

* OCANNL is distributed as two opam packages:
  * **arrayjit**: a backend for compilers for numerical array programming,
  * **neural\_nets\_lib**: a neural networks (Deep Learning) framework.
* You express numerical computations at runtime, OCANNL will optimize them for the runtime-dependent shape of your data, compile and dynamically load them using one of its backends.

---

## Slide 4

### OCaml Compiles Algorithms for Neural Network Learning

* OCANNL is distributed as two opam packages:
  * **arrayjit**: a backend for compilers for numerical array programming languages,
  * **neural\_nets\_lib**: a neural networks (Deep Learning) framework.
* You express numerical computations at runtime, OCANNL will compile them.
* There are startups doing it in other languages, so it must be worth it!
  * In Python: **tinygrad** - democratizing DL - at home or on premise training of large models.
  * In Rust: **Luminal** - simplifies deployment of large models with on-device inference.
* There are many optimization / NN frameworks in Rust, but few in OCaml! (e.g., Luminal, Candle, Cubecl, Burn)

---

## Slide 5

### OCaml Compiles Algorithms for Neural Network Learning

**Value added**:

* OCaml is a good fit for writing optimizing compilers.
* OCANNL has concise notation thanks to better shape inference (i.e. type inference for the data and transformation matrices).
* OCANNL might be a good fit for heterogeneous computing (e.g. combining GPUs from different companies), it is explicit about the backends (and devices) used.

---

## Slide 6

OCANNL is still at a **proof-of-concept stage**, but it will grow and evolve.

---

## Slide 7

Let's train a feed-forward neural network with 2 hidden layers (aka. a 3-layer MLP) to classify points on a plane.

---

## Slide 8

> A screenshot from TensorFlow Playground is shown, illustrating a 3-layer Multi-Layer Perceptron (MLP) configured to classify the "moons" dataset.
>
> * **Data & Features**: The input data is the "moons" dataset, with a 50% training-to-test ratio, zero noise, and a batch size of 20. The input features are $x_1$, $x_2$, $x_1^2$, $x_2^2$, $x_1x_2$, $\sin(x_1)$, and $\sin(x_2)$.
>
> * **Network Architecture**:
>   * An input layer feeding into the first hidden layer.
>   * Two hidden layers, each with 6 neurons. The activation function is ReLU.
>   * The connections between layers are represented by weights **w1**, **w2**, and **w3**, and biases **b1**, **b2**, and **b3**. The thickness of the lines indicates the magnitude of the weights.
> * **Training Parameters**:
>   * Learning Rate: 0.03 (starts at 0.1 and linearly decays).
>   * Regularization: L2 with a rate of 0.003 [weight decay](cite: 54).
> * **Output**: The output plot shows the model has learned to separate the two half-moon-shaped clusters of data points (blue and orange). The test loss is 0.462 and the training loss is 0.358 after 1,025 epochs.

---

## Slide 9

### Training NNs is first-order (i.e. gradient based) optimization

* Collect examples to learn from.
  * *a dataset*
* Express a solution as a parameterized differentiable computation.
  * *a model*
* Figure out a formula for how bad the solution is on a datapoint.
  * *a loss function*
* For each datapoint in the dataset:
  * Compute the direction for parameters to make them better/worse.
        * *a gradient*
  * Update parameters to make them better.
        * *Stochastic Gradient Descent (SGD)*

---

## Slide 10

### Training NNs is first-order (i.e. gradient based) optimization

* Collect examples to learn from.
  * *a dataset*
* Express a solution as a parameterized differentiable computation.
  * *a model*
* Figure out a formula for how bad the solution is on a datapoint.
  * *a loss function*
* For each (mini-)batch of data points in the dataset:
  * For each datapoint in the batch, compute the direction for parameters to make them better/worse → **datapoint gradient**.
  * Add up the batch datapoint gradients.
  * Update parameters to make them better.
        * *Stochastic Gradient Descent (SGD)*

---

## Slide 11

(Mini-)Batches are equal-sized subsets of the dataset. Main options for defining a batch:

* An element of a (fixed but random) partition of the dataset. **We'll use this.**
* A random subset of the dataset.

---

## Slide 12

### Half-moons toy dataset

> The slide shows a scatterplot of the half-moons dataset, where two classes of points form two interleaving crescent shapes. One class is marked with '#' and the other with '+'.

```ocaml
  let config = Datasets.Half_moons.Config.{ noise_range = 0.1; seed = Some seed } in
  let moons_coordinates, moons_labels = Datasets.Half_moons.generate_single_prec ~config ~len () in
  let moons_flat_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_coordinates in
  let moons_classes_ndarray = Ir.Ndarray.as_array Ir.Ops.Single moons_labels in
```

## Slide 13

### Half-moons toy dataset

> This slide shows the same half-moons scatterplot as the previous one, along with the OCaml code used to generate the plot using the `PrintBox_utils` library.

```ocaml
  let points = Tn.points_2d ~xdim:0 ~ydim:1 moons_flat.value in
  let classes = Tn.points_1d ~xdim:0 moons_classes.value in
  let points1, points2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
  let%cd mlp_result = mlp { point } in
  Train.set_on_host mlp_result.value;
  let result_routine =
    Train.to_routine (module Backend) sgd_routine.context IDX.empty
      [%cd ~~("moons infer"; mlp_result.forward)]
  in
  let callback (x, y) =
    Tn.set_values point.value [| x; y |];
    Train.run result_routine;
    Float.(mlp_result.@[0] >= 0.)
  in
  let plot_moons =
    PrintBox_utils.plot ~as_canvas:true
      [
        Scatterplot { points = points1; content = PrintBox.line "#" };
        Scatterplot { points = points2; content = PrintBox.line "%" };
        Boundary_map
          { content_false = PrintBox.line "."; content_true = PrintBox.line "*"; callback };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_moons;
```

---

## Slide 14

### Tensors as "differentiable" multidimensional matrices

> A diagram illustrates the forward pass of a 3-layer MLP as a composition of tensor operations. It uses a visual notation where blue downward arrows represent **output axes** and orange rightward arrows represent **input axes**. The overall computation is:
> $$b_3 + w_3 \cdot f(b_2 + w_2 \cdot f(b_1 + w_1 \cdot x))$$
> where $f$ is an activation function.

---

## Slide 15

### Tensors as "differentiable" multidimensional matrices

> This diagram is identical to the one on the previous slide, but adds a third type of axis: a red diagonal arrow representing **batch axes**. This is shown on the input tensor $x$.

---

## Slide 16

### Multi Layer Perceptron in one line of code

> **ReLU function:** A graph shows the Rectified Linear Unit (ReLU) function, which is $f(x) = \max(0, x)$. The function is zero for all negative inputs and increases linearly for positive inputs.

```ocaml
  let%op mlp x =
    { w3 } * relu ({ b2; o = [ 16 ] } + ({ w2 } * relu ({ b1; o = [ 16 ] } + ({ w1 } * x))))
  in
```



---

## Slide 17

### Multi Layer Perceptron in one line of code

```ocaml
  let%op mlp x =
    { w3 } * relu ({ b2; o = [ 16 ] } + ({ w2 } * relu ({ b1; o = [ 16 ] } + ({ w1 } * x))))
  in
```



* `{ w1 } * x`: Tensor (e.g. matrix) multiplication.
* `{ w1 }`: Introduces identifier `w1` for a parameter tensor.
* `{ b1; o = [ 16 ] }`: Sets the output dimension of `b1` to `16`.
* `relu`: "Rectified Linear Unit" unary operation.
* `let%op mlp x = ...`: Declarative expressions for differentiable tensor operations. This is a tensor function that expands to: `let w1 = ... in let b1 = ... in let mlp in ...`.

---

## Slide 18

### Hinge loss function: maximum margin classification

> **Left Image:** Shows a Support Vector Machine (SVM) separating two classes of data points with a hyperplane. The goal is to find the plane that has the **maximum margin** between the two classes.

> **Right Image:** A graph of the Hinge Loss function. For correct classifications with a margin greater than 1, the loss is 0. For incorrect or insufficiently confident classifications, the loss increases linearly.

**Formula:** $C(y, f(x)) = \max(0, 1 - y \cdot f(x))$

**OCANNL Code:**

```ocaml
(* pointwise multiplication *)
let%op margin_loss = relu (1. - (moons_class *. mlp moons_input)) in

(* sum out all batch and output axes to dimension 0 of the result *)
let%op scalar_loss = (margin_loss ++ "...|... => 0") /. !..batch_size in
```



---

## Slide 19

### Regularization - weight decay

* **Regularization**: keep models simple to be less accidentally wrong and to stabilize training.
* You can add a regularizer to the loss function or modify the SGD update step—we'll do the latter.
* **Weight decay** is L2 regularization because the gradient of $p^2$ is $2p$.

<br/>
<table align="center">
  <tr>
    <th style="text-align:left;">L1 Regularization</th>
    <th style="text-align:left;">L2 Regularization</th>
  </tr>
  <tr>
    <td>1. L1 penalizes the sum of <strong>absolute</strong> values of weights.</td>
    <td>1. L2 penalizes the sum of <strong>square</strong> values of weights.</td>
  </tr>
  <tr>
    <td>2. L1 generates a model that is simple and interpretable.</td>
    <td>2. L2 regularization is able to learn complex data patterns.</td>
  </tr>
  <tr>
    <td>3. L1 is robust to outliers.</td>
    <td>3. L2 is not robust to outliers.</td>
  </tr>
</table>

---

## Slide 20

### Backprop: compositionally deriving gradient computations

* Backprop is a special case of **reverse mode automatic differentiation** and is limited to first-order derivatives [it cannot compute Hessians, for example](cite: 336).
* It generalizes the chain rule: $\frac{df}{dx} = \frac{df}{dy} \cdot \frac{dy}{dx}$.
* A **forward pass** involves computing a tensor's value from the tensors in its definition.
* A **backward pass** computes the gradient of a value (like loss) with respect to a tensor $x$ ($\frac{df}{dx} = x.\text{grad}$) by summing contributions from every place $x$ appears.

---

## Slide 21

### Backprop: compositionally deriving gradient computations

* **Forward pass**: computing the value of a tensor from the values of tensors in its definition.
* **Backward pass**: computing the gradient of the value of interest $f$ (e.g. loss) with respect to a tensor $x$: $\frac{df}{dx} = x.\text{grad}$, by adding up contribution from each place in which $x$ appears.
  * Once we have $\frac{df}{dy}$, we use $\frac{dy}{dx}$ and proceed backward to compute $\frac{df}{dx}$.
  * The order of computation is reversed: $x \rightarrow y(x) \rightarrow f(y(x))$ but $df \rightarrow \frac{df}{dy} \rightarrow \frac{df}{dx}$.
  * The composition order remains bottom-up; we prepend the $\frac{df}{dy}$ code to the backward code of $y$ to build the backward code for $f$.

---

## Slide 22

### Backprop: compositionally deriving gradient computations

* **Example**: For $f(t(t_1, t_2))$ where $t = t_1 \cdot t_2$, let $g = \frac{df}{dt}$ be the incoming gradient.
  * $\frac{dt}{dt_1} = t_2$, therefore $\frac{df}{dt_1} = \frac{df}{dt} \cdot \frac{dt}{dt_1} = g \cdot t_2$.
  * $\frac{dt}{dt_2} = t_1$, therefore $\frac{df}{dt_2} = \frac{df}{dt} \cdot \frac{dt}{dt_2} = g \cdot t_1$.
  * At the node $t = t_1 \cdot t_2$, we back-propagate $g \cdot t_2$ toward $t_1$ and $g \cdot t_1$ toward $t_2$.

---

## Slide 23

### Interlude: what is code / computation?

* In OCANNL, a tensor is associated with a **value** node and, optionally, a **gradient** node.
* High-level numeric code primarily consists of sequences of **accumulating assignments**.
* The most common accumulation operators are "don't accumulate" (i.e., overwrite) and addition.
* The assignment can optionally reset the left-hand-side tensor to the operator's neutral element.
* The `%cd` syntax extension manages all of this.

---

## Slide 24-27

### Backprop by example: Addition

The derivative of $t_1+t_2$ with respect to $t_1$ is 1, i.e., $\frac{d(t_1+t_2)}{dt_1} = 1$. Thus, both `t1.grad` and `t2.grad` increase by the incoming gradient `t.grad`.

```ocaml
let add ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  (* Forward pass: t.value =: t1.value + t2.value *)
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 + v2 in
  
  (* Backward pass: *)
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    (* t1.grad =+ t.grad *)
    g1 =+ g;
    (* t2.grad =+ t.grad *)
    g2 =+ g
  in
  Tensor.binop ~label:("+" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn
```



**Annotations:**

* `v =: v1 + v2` is shorthand for `t.value =: t1.value + t2.value`. The `=:` operator sets the LHS tensor.
* `g1 =+ g` is shorthand for `t1.grad =+ t.grad`. The `=+` operator adds to the LHS tensor without resetting it.
* `g1` is shorthand for `t1.grad`, `g` is shorthand for `t.grad`, `v2` is shorthand for `t2.value`, etc..

---

## Slide 28

### Backprop by example: Subtraction

The gradient of `t1.grad` increases and `t2.grad` decreases by `t.grad` because $\frac{d(t_1-t_2)}{dt_2} = -1$.

```ocaml
let sub ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 - v2 in
  
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    (* t1.grad =+ t.grad *)
    g1 =+ g;
    (* t2.grad =- t.grad *)
    g2 =- g
  in
  Tensor.binop ~label:("-" :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn
```



---

## Slide 29-30

### Backprop by example: Multiplication

For both pointwise and tensor multiplication, gradient propagation follows the chain rule: multiply the incoming gradient by the *other* term from the forward pass.

```ocaml
(* Generic multiplication gradient *)
let mul compose_op ~op_asn =
  let module NTDSL = Initial_NTDSL in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g * v2;
    g2 =+ v1 * g
  in
  Tensor.binop ~compose_op ~op_asn ~grad_asn

(* Pointwise multiplication *)
let pointmul ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 * v2 in
  mul Pointwise_bin ~op_asn ~label:("*." :: label)

(* Tensor (e.g. matrix) multiplication *)
let matmul ?(label = []) =
  let module NTDSL = Initial_NTDSL in
  (* =:+ means: first reset v, then add up the results of v1*v2 *)
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =:+ v1 * v2 in
  mul Compose ~op_asn ~label:("*" :: label)
```



---

## Slide 31

### Backprop by example: Pointwise Power

This code defines pointwise power, $t_1^p$. The gradient is derived from the power rule, $(x^n)' = nx^{n-1}$.

```ocaml
let rec pointpow ?(label : string list = []) ~grad_spec p (t1 : Tensor.t) =
  let module NTDSL = ... in
  let p_t = NTDSL.number p in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 ** v2 ~projections in
  let%cd grad_asn =
    if Tensor.is_prohibit_grad grad_spec then fun ~t:_ ~g:_ ~t1:_ ~t2:_ ~projections:_ -> Asgns.Noop
    else if Float.equal p 2.0 then fun ~t:_ ~g ~t1 ~t2:_ ~projections -> g1 =+ p_t * t1 * g
    else if Float.equal p 1.0 then fun ~t:_ ~g ~t1:_ ~t2:_ ~projections -> g1 =+ g
    else fun ~t:_ ~g ~t1 ~t2:_ ~projections ->
      g1 =+ p_t * (t1 **. (p -. 1.)) * g
  in
  Tensor.binop ~label:("**." :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn ~grad_spec t1 p_t
```



---

## Slide 32

### Backprop by example: Pointwise Division

This code defines pointwise division, $t_1/t_2$. The gradient is derived from the quotient rule: $\nabla\left(\frac{t_1}{t_2}\right) = \frac{\nabla(t_1)t_2 - t_1\nabla(t_2)}{t_2^2}$.

```ocaml
let rec pointdiv ?(label: string list = []) ~grad_spec t1 t2 =
  let module NTDSL = ... in
  let%cd op_asn ~v ~t1 ~t2 ~projections = v =: v1 / v2 in
  let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections =
    g1 =+ g / v2;
    g2 =+ g * (-1. *. v1 /. (v2 **. 2.))
  in
  Tensor.binop ~label:("/." :: label) ~compose_op:Pointwise_bin ~op_asn ~grad_asn ~grad_spec t1 t2
```



---

## Slide 33

### Putting the forward and backward passes together

This function constructs the full computation graph for a gradient update step.

```ocaml
let grad_update_nochecks loss =
  let params = get_params loss in
  let diff = diff_or_error loss "Train.grad_update_nochecks" in
  let fwd_bprop =
    [%cd
      (* Block comment syntax, used for debugging *)
      ~~(loss "gradient update";
        ~~(loss "fwd";
          loss.forward);
        ~~(loss "zero grads";
          (* Zero out gradients before starting to accumulate *)
          diff.zero_grads);
        loss.grad =: 1.;
        ~~(loss "bprop";
          diff.backprop))]
  in
  { loss; params; fwd_bprop }
```



---

## Slide 34

### Stochastic Gradient Descent with Momentum

* Vanilla SGD subtracts scaled gradients from parameters: `p =- learning_rate * p.grad`.
* This can be slow in regions where loss differences are small.
* SGD with momentum accelerates training by accumulating gradients using a form of exponential smoothing. It sums gradients, where the weight for a gradient from $i$ steps back is $\text{momentum}^i$.
    `"sgd_momentum" =: (!.momentum * sgd_momentum) + p.grad;`
    `p =- learning_rate * sgd_momentum`

---

## Slide 35

### Stochastic Gradient Descent with apx. Nesterov Momentum

> An image shows the path of an optimizer on a contour plot. The path (in green) oscillates back and forth across a narrow valley (or "canyon") instead of moving directly towards the minimum, which is a common issue for standard SGD with momentum.

* To avoid this oscillation, **Nesterov Accelerated Gradient** computes the gradient at a "lookahead" point, which is estimated using the current momentum.
* We can approximate this by adding a momentum-scaled gradient term twice:
    `"sgd_delta" =: p.grad;`
    `"sgd_momentum" =: (!.momentum * sgd_momentum) + sgd_delta;`
    `sgd_delta =+ !.momentum * sgd_momentum;`
    `p =- learning_rate * sgd_delta`

---

## Slide 36-37

### Stochastic Gradient Descent with apx. Nesterov Momentum

The OCANNL code for a single parameter update step, including options for weight decay, momentum, and Nesterov acceleration.

```ocaml
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  if not @@ is_param p then raise @@ Tensor.Session_error ("Train.sgd_one: not a parameter", Some p);
  [%cd
    ~~(p "param sgd step";
      (* Instead of adding a regularizer to the loss tensor, regularize here. *)
      "sgd_delta" =: p.grad + (!.weight_decay *. p);
      if Float.(momentum > 0.0) then (
        (* Inline declarations of (non-differentiable) tensors. *)
        "sgd_momentum" =: (!.momentum *. sgd_momentum) + sgd_delta;
        if nesterov then sgd_delta =+ !.momentum *. sgd_momentum
        else sgd_delta =: sgd_momentum
      );
      (* Specifies that computations should be pointwise. *)
      (* This is a unary accumulating assignment. *)
      p =- learning_rate * sgd_delta ~logic:"."
    )
  ]
```



---

## Slide 38

### Compilation illustrated by running `bin/moons_demo.ml`

This snippet from the demo script shows how the backpropagation and SGD update computations are combined and compiled into a single routine.

```ocaml
(* We don't use momentum; we use weight decay. *)
let update = Train.grad_update scalar_loss in

let%op learning_rate = 0.1 *. (!..steps -!@step_n) /. !..steps in
Train.set_hosted learning_rate.value;

let sgd = Train.sgd_update ~learning_rate ~weight_decay update in

(* Use the C compiler CPU backend. *)
let module Backend = (val Arrayjit.Backends.fresh_backend ~backend_name:"cc" ()) in
let device = Backend.(new_virtual_device @@ get_device ~ordinal:0) in
let ctx = Backend.init device in

(* We combine backpropagation and SGD update in a single routine *)
let routine = Backend.(link ctx @@ compile bindings (Seq (update.fwd_bprop, sgd))) in
```



---

## Slide 39

### Compilation assignments: `scalar_loss_gradient_then_sgd_update.cd`

> This slide displays a high-level, intermediate representation of the computation graph. It's a sequence of assignments for the forward pass (calculating `mlp_moons_input` and `scalar_loss`) followed by the SGD update step for each parameter (`b3`, `w1`, `w2`, `w3`). This file is for debugging and doesn't include indexing information.

---

## Slide 40

### Compilation low level: `scalar_loss_gradient_then_sgd_update-unoptimized.ll`

> This slide shows a lower-level representation of the computation before optimization. The high-level operations from the previous slide have been translated into explicit `for` loops that iterate over the dimensions of the tensors. For example, the matrix multiplication `w1 * moons_input` is now a set of three nested loops.

---

## Slide 41

### Compilation - optimized: `scalar_loss_gradient_then_sgd_update.ll`

> This slide presents the low-level code after optimization. The key improvement highlighted is that inlining has reduced multiple loops for the SGD update into a single loop, making the computation more efficient.

---

## Slide 42

### Compilation - C code: `scalar_loss_gradient_then_sgd_update.c`

> This slide shows the final output of the compilation process: a C code file. The tensor operations have been converted into C loops and array manipulations. Pointers are defined for each tensor (`w1`, `b3`, etc.), and local arrays are declared for intermediate values and gradients. An optimization is noted: the step for zeroing out gradients was removed by the compiler.

---

## Slide 43

### Running `bin/moons_demo.ml`

> This slide shows the terminal output from running the compiled demo.
>
> 1.  The program starts, indicating it's loading the OCANNL configuration.
> 2.  It prints the training progress, showing the loss decreasing over epochs (e.g., `Epoch 74, lr=0.000042, epoch loss=0.006639`).
> 3.  Finally, it displays a text-based scatterplot of the result. The plot shows the half-moons dataset points (`#` and `+`) along with the decision boundary learned by the model (represented by `*`). The clear separation of the `*` symbols between the two moons indicates successful classification.

---

## Slide 44

### `cuda-gdb` session, including CUDA source position

> This slide demonstrates debugging the generated code using `cuda-gdb`.
>
> 1.  The user starts `cuda-gdb` with the compiled executable.
> 2.  A breakpoint is set on a specific line (`252`) inside the generated CUDA source file (`scalar_loss_gradient_then_sgd_update.cu`).
> 3.  The program is run, and it hits the breakpoint within a CUDA thread.
> 4.  The user then inspects the values of variables on the GPU at that point in execution, printing the values of `b3_grad[0]` ($0.50000006$) and `learning_rate[0]` [$0.09899999995$](cite: 978, 979). This shows OCANNL's ability to generate debuggable code for different backends.

---

## Slide 45

### Debug Logs

> This slide shows a detailed, tree-structured log trace generated by OCANNL's debugging utilities.
> It visualizes the execution flow of the training loop, including:
> *The call stack (`train loop` -> `for epoch` -> `for batch`).
> * The sequence of high-level operations (`scalar_loss_fwd`, `scalar_loss_bprop`, `scalar_loss_sgd_update`).
> * A detailed view of a single parameter update (`b1 param sgd step`), showing the exact C code line being executed and the values of the variables (`b1[0]`, `learning_rate[0]`, `b1.grad[0]`) before and after the update. This provides deep insight into the training process.

---

## Slide 46

### Data parallel training

1. Subdivide a batch into mini-batches for each device.
2. Schedule copying the data to each device.
3. Schedule updating gradients (running `fwd_bprop`) on each device.
4. Pairwise merge the gradients by repeatedly adding gradients from the second half of the devices to the first half, until all gradients are accumulated on device 0.
5. Run the SGD update on device 0.
6. Schedule copying the updated parameters from device 0 back to all other devices.

---

## Slide 47

### Multi-stream computations in OCANNL

* OCANNL has a loose notion of a stream, which can represent entities like CPU cores or CUDA streams.
* Code can be compiled for a backend independently of a device, but it is linked with a device-specific context for execution.
* The hierarchy is Context → stream → device.
* Tensor nodes are represented as arrays on the device as needed for computation.

---

## Slide 48

### Multi-device computations in OCANNL (design might likely change)

* Each stream has a **merge buffer** to hold an array coming from another device.
* Data can arrive in this buffer via copying, direct pointing (for CPUs or devices on the same GPU), or potentially streaming in the future.
* Unlike a regular device-to-device transfer that writes to a tensor's destination array, a transfer into the merge buffer does not.

---

## Slide 49

### Data parallel training: merging gradients in OCANNL

```ocaml
(* Define the merge operation: p.grad =+ p.grad.merge *)
let grad_merges : Asgns.t array =
  Array.map all_params ~f:(fun p -> [%cd p.grad =+ p.grad.merge])
in

(* Compile the merge operation for all necessary device pairs *)
let grad_merges_to : Backend.routine option array array =
  Array.mapi ctxs ~f:(fun dst_n ctx ->
    if occupancy_dst ~dst_n then
      snd @@ Backend.link_batch ctx
      @@ Backend.compile_batch ~shared:true ~occupancy:Idx.Empty grad_merges
    else [||]
  )
in

let merge_grads ~(from: int) ~(to_: int) : unit =
  Array.iteri all_params ~f:(fun i p ->
    let grad_merge = Option.value_exn grad_merges_to.(to_).(i) in
    (* Fill the merge buffer before running merging. *)
    assert (
      Backend.device_to_device (Option.value_exn p.diff).grad ~into_merge_buffer:BT.Copy
        ~dst:grad_merge.context ~src:ctxs.(from));
    (* Synchronization now happens automatically. *)
    (Task.run grad_merge.schedule : unit))
in
```



---

## Slide 50

### OCANNL Features

* **Declarative** differentiable tensors.
* **Imperative** array manipulation language.
* Flexibly combines these two layers.
* Very **concise** notations.
* Powerful **shape inference** integrated with expressive "generalized einsum" indexing.
* **Backprop** is handled automatically.
* **Generates optimized code**.
* Very little abstraction fluff, **close to the metal**.
* **Debuggable**.

---

### Thank you! Questions?

* "Is it functional?"
  * tl;dr: no, not like JAX, and less so than OWL.
* "What are the future directions?"
