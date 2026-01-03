# Migration Guide: PyTorch/TensorFlow to OCANNL

This guide helps deep learning practitioners familiar with PyTorch or TensorFlow understand OCANNL's approach and idioms.

## Key Conceptual Differences

### Shape Inference vs Explicit Shapes
- **PyTorch/TF**: Shapes are usually explicit (e.g., `Conv2d(in_channels=3, out_channels=64)`)
- **OCANNL**: Shapes are inferred where possible, using row variables for flexibility
  ```ocaml
  (* Channels as row variables allow multi-channel architectures *)
  conv2d ~label () x  (* ..ic.. and ..oc.. are inferred *)
  ```

### Two-Phase Inference System
OCANNL separates shape inference from projection inference:
- **Shape inference**: Global, propagates constraints across operations
- **Projection inference**: Local per-assignment, derives loop structures from tensor shapes

This is why pooling needs a dummy constant kernel - to carry shape info between phases.

## Common Operations Mapping

| PyTorch/TensorFlow | OCANNL | Notes |
|-----|------|----|
| `x.view(-1, d)` or `x.reshape(-1, d)` | Only supported for wrapping non-tensor data | Use shape inference and let tensors have the shape they want |
| `x.flatten()` | Might get supported for explicit axes | Future syntax might be: `"x,y => x&y"` |
| `nn.Conv2d(in_c, out_c, kernel_size=k)` | `conv2d ~kernel_size:k () x` | Channels inferred or use row vars |
| `F.max_pool2d(x, kernel_size=k)` | `max_pool2d ~window_size:k () x` | Uses `(0.5 + 0.5)` trick internally |
| `F.avg_pool2d(x, kernel_size=k)` | `avg_pool2d ~window_size:k () x` | Normalized by window size |
| `nn.BatchNorm2d(channels)` | `batch_norm2d () ~train_step x` | Channels inferred |
| `F.dropout(x, p=0.5)` | `dropout ~rate:0.5 () ~train_step x` | Needs train_step for PRNG |
| `F.relu(x)` | `relu x` | Direct function application |
| `F.softmax(x, dim=-1)` | `softmax ~spec:"... \| ... -> ... d" () x` | Specify axes explicitly |
| `torch.matmul(a, b)` | `a * b` or `a +* b "..b.. -> ..a..; ..b.. => ..a.."` | Einsum for complex cases |
| `x.mean(dim=[1,2])` | `x ++ "... \| h, w, c => ... \| 0, 0, c" ["h"; "w"] /. (dim h *. dim w)` | Sum then divide |
| `x.sum(dim=-1, keepdim=True)` | `x ++ "... \| ... d => ... \| ... 0"` | Reduce by summing |
| `x.sum(dim=-1, keepdim=False)` | `x ++ "... \| ... d => ... \| ..."` | Reduce by summing |
| `torch.cat([a, b], dim=0)` | `(a, b) ++^ "x; y => x^y"` | Concatenate tensors |
| `torch.stack([a, b], dim=0)` | `[a; b]` | Stack with new axis (block tensor syntax) |
| `x[:n]` (prefix slice) | `x ++^ "a^b => a"` | Extract prefix (size inferred) |
| `x[n:]` (suffix slice) | `x ++^ "a^b => b"` | Extract suffix (size inferred) |
| `x[:-3]` (all but last 3) | `x ++^ "a^3 => a"` | Drop last 3 elements |

## Tensor Creation Patterns

### Parameters (Learnable Tensors)

| PyTorch | OCANNL | 
|---------|---------|
| `nn.Parameter(torch.rand(d))` | `{ w }` or `{ w = uniform () }` |
| `nn.Parameter(torch.randn(d))` | `{ w = normal () }` |
| `nn.Parameter(torch.zeros(d))` | `{ w = 0. }` |
| `nn.Parameter(torch.ones(d))` | `{ w = 1. }` |
| With explicit dims | `{ w; o = [out_dim]; i = [in_dim] }` |

### Non-learnable Constants

| PyTorch | OCANNL | Notes |
|---------|---------|--------|
| `torch.ones_like(x)` | `0.5 + 0.5` | Shape-inferred constant 1 |
| `torch.tensor(1.0)` | `!.value` or `1.0` | Scalar constant |
| `torch.full_like(x, value)` | `NTDSL.term ~fetch_op:(Constant value) ()` | Shape-inferred |

## Network Architecture Patterns

### Sequential Models

**PyTorch:**
```python
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*14*14, 10)
)
```

**OCANNL:**
```ocaml
let%op model () =
  let conv1 = conv2d ~kernel_size:3 () in
  let pool = max_pool2d () in
  fun x ->
    let x = conv1 x |> relu |> pool in
    (* No flattening needed - FC layer works with spatial dims *)
    { w_out } * x + { b_out = 0.; o = [10] }
```

### Residual Connections

**PyTorch:**
```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(out + identity)
```

**OCANNL:**
```ocaml
let%op resnet_block () =
  let conv1 = conv2d () in
  let bn1 = batch_norm2d () in
  let conv2 = conv2d () in
  let bn2 = batch_norm2d () in
  fun ~train_step x ->
    let identity = x in
    let out = conv1 x |> bn1 ~train_step |> relu |> conv2 |> bn2 ~train_step in
    relu (out + identity)
```

## Einsum Notation

OCANNL's einsum is more general than PyTorch's, supporting row variables and convolutions.

### Syntax Modes

OCANNL's einsum has two syntax modes:

1. **Single-character mode** (PyTorch-compatible):
   - Triggered when NO commas appear in the spec
   - Each alphanumeric character is an axis identifier
   - Spaces are optional and ignored: `"ij"` = `"i j"`
   
2. **Multi-character mode**:
   - Triggered by ANY comma in the spec
   - Trailing commas ignored (can be used to trigger multi-char mode)
   - Identifiers can be multi-character (e.g., `height`, `width`)
   - Must be separated by non-alphanumeric: `,` `|` `->` `;` `=>`
   - Makes convolution syntax less confusing: `stride*out+kernel`

| Operation | PyTorch einsum | OCANNL single-char | OCANNL multi-char |
|--------|------------------|-------------------|-------------------|
| Matrix multiply | `torch.einsum('ij,jk->ik', a, b)` | `a +* "i j; j k => i k" b` | `a +* "i, j; j, k => i, k" b` |
| Batch matmul | `torch.einsum('bij,bjk->bik', a, b)` | `a +* "b i j; b j k => b i k" b` | `a +* "batch, i -> j; batch, j -> k => batch, i -> k" b` |
| Attention scores | `torch.einsum('bqhd,bkhd->bhqk', q, k)` | `q +* k "bq\|hd; bk\|hd => b\|qk->h" k` | `q +* "b, q \| h, d; b, k \| h, d => b \| q, k -> h"` |
| Convolution | N/A | always multi-char | `x +* kernel "... \| stride*oh+kh, stride*ow+kw, ic; kh, kw, ic -> oc => ... \| oh, ow, oc"` |

### Row Variables
- `...` context-dependent ellipsis: expands to `..batch..` in batch position, `..input..` before `->`, `..output..` after `->`
- Single-char mode example: `..b..|` for batch axes (arbitrary number)
- Multi-char mode examples: `h, w, ..ic..`, `h, w, ..oc..` for input/output channels (can be multi-axis), `..spatial.., channel` for spatial dimensions

## Common Gotchas and Solutions

### Variable Capture with Einsum
❌ **Wrong:**
```ocaml
let spec = "... | h, w => ... | h0" in
x ++ spec [ "h"; "w" ]  (* Error: spec must be literal *)
```

✅ **Right:**
```ocaml
x ++ "... | h, w => ... | h0" [ "h"; "w" ]
```

### Creating Non-learnable Constants
❌ **Wrong:**
```ocaml
{ kernel = 1. }  (* Creates learnable parameter *)
1.0             (* Creates fixed scalar shape *)
```

✅ **Right:**
```ocaml
0.5 + 0.5       (* Both are shape-inferred constant 1 *)
NTDSL.term ~fetch_op:(Constant 1.) ()
```

### Parameter Scoping
❌ **Wrong:**
```ocaml
let%op network () x =
  (* Sub-module defined after input *)
  let layer1 = my_layer () x in  
  { global_param } + x 
```

✅ **Right:**
```ocaml
let%op network () =
    (* Sub-modules before input *)
  let layer1 = my_layer () in
  fun x ->
    (* Inline definitions are lifted:
       used here, but defined before layer1 *)
    { global_param } + layer1 x
```

### Flattening for Linear Layers

⚠️ **Important:** OCANNL doesn't support arbitrary flattening/reshaping operations.

```ocaml
(* This performs REDUCTION (sum), not flattening: *)
x ++ "... | ..spatial.. => ... | 0"

(* OCANNL's approach: Let FC layers work with multiple axes!
   Instead of flattening [batch, h, w, c] to [batch, h*w*c],
   just let your FC layer handle [batch, h, w, c] directly.
   The tensor multiplication will work across all the axes. *)

(* Example: FC layer after conv without flattening *)
let%op fc_after_conv () x =
  (* x might have shape [batch, height, width, channels] *)
  { w } * x + { b }  (* w will adapt to match x's shape *)
```

### Tensor Concatenation and Block Tensors

OCANNL supports concatenation of tensors along an axis using the `^` operator in einsum notation. This is different from flattening—concatenation preserves the axis structure.

```ocaml
(* Concatenate two vectors *)
let%op concat_vectors a b = (a, b) ++^ "x; y => x^y"

(* Concatenate along a specific axis (output axis here) *)
let%op concat_matrices a b =
  (a, b) ++^ "...|m, n; ...|p, n => ...|m^p, n"

(* Extract prefix/suffix of a tensor *)
let%op get_prefix x = x ++^ "a^b => a"  (* size of 'a' inferred from context *)
let%op get_suffix x = x ++^ "a^b => b"  (* size of 'b' inferred from context *)

(* Block tensor construction (upcoming syntax) *)
let%op block_matrix () =
  [[a; b]; [c; d]]  (* Creates 2x2 block matrix from components *)
```

The `^` operator is fundamentally an indexing-level operation—it creates an axis that iterates over its components in sequence. This enables:
- **Tensor concatenation**: `a; b => a^b`
- **Axis slicing**: `a^b => a` (prefix) or `a^b => b` (suffix)
- **Block tensor construction**: Compose structured tensors from smaller parts
- **Partial updates**: `a => a^b` assigns to prefix, leaving suffix unchanged

## Training Loop Patterns

### Basic Training Step

**PyTorch:**
```python
optimizer.zero_grad()
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

**OCANNL (v0.6.1+ with Context API):**
```ocaml
(* Compile once, run many times *)
let ctx = Context.auto () in
let%op learning_rate = 0.01 in
let update = Train.grad_update loss in
let sgd = Train.sgd_update ~learning_rate loss in
let ctx = Train.init_params ctx IDX.empty loss in
let routine = Train.to_routine ctx IDX.empty 
  (Asgns.sequence [update; sgd]) in

(* Training loop - reuse compiled routine *)
for epoch = 1 to 100 do
  Train.run ctx routine;
  if epoch mod 10 = 0 then Printf.printf "Loss: %.4f\n" loss.@[0]
done
```

## Training with the Context API (v0.6.1+)

The Context API introduced in v0.6.1 significantly simplifies backend management and training workflows.

### Context Creation

**PyTorch:**
```python
# Automatic device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explicit device selection
device = torch.device("cuda:0")  # CUDA device 0
device = torch.device("mps")     # Apple Metal Performance Shaders
device = torch.device("cpu")     # CPU

# Move model and data to device
model = model.to(device)
data = data.to(device)
```

**OCANNL:**
```ocaml
(* Automatic backend selection (respects OCANNL_BACKEND env var) *)
let ctx = Context.auto () in

(* Or explicit backend selection *)
let ctx = Context.cuda ~device_id:0 () in
let ctx = Context.metal () in
let ctx = Context.cpu ~threads:4 () in
```

### Basic Training Patterns

#### Dynamic Learning Rate with Per-Step Data

**PyTorch:**
```python
def train_with_schedule(model, data_loader, steps):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - (step / steps))
    
    data_iter = iter(data_loader)
    for step in range(steps):
        # Get next batch, cycling if necessary
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, y = next(data_iter)
        
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step}, LR: {lr:.4f}, Loss: {loss.item():.6f}")
```

**OCANNL:**
```ocaml
let train_with_schedule get_batch model input_tensor target_tensor steps =
  let ctx = Context.auto () in
  let step_n, bindings = IDX.get_static_symbol IDX.empty in
  
  (* Define model and loss with input placeholders *)
  let%op predictions = model input_tensor in
  let%op loss = mse predictions target_tensor in
  
  (* Dynamic learning rate that decreases over time *)
  let%op learning_rate = 0.1 *. (1.0 - (!@step_n /. !..steps)) in
  Train.set_hosted learning_rate.value;
  
  (* Compile with dynamic learning rate *)
  let update = Train.grad_update loss in
  let sgd = Train.sgd_update ~learning_rate loss in
  let ctx = Train.init_params ctx bindings loss in
  let routine = Train.to_routine ctx bindings 
    (Asgns.sequence [update; sgd]) in
  
  (* Get reference to step counter *)
  let step_ref = IDX.find_exn (Context.bindings routine) step_n in
  step_ref := 0;
  
  (* Training loop - update input data each step *)
  for step = 1 to steps do
    (* Load next batch into tensors *)
    let x_data, y_data = get_batch step in
    Tn.set_values input_tensor.value x_data;
    Tn.set_values target_tensor.value y_data;
    
    (* Run training step *)
    Train.run ctx routine;
    Int.incr step_ref;
    
    if step mod 100 = 0 then
      Printf.printf "Step %d, LR: %.4f, Loss: %.6f\n" 
        step learning_rate.@[0] loss.@[0]
  done;
  ctx
```

### Batched Training Example

**PyTorch:**
```python
def train_batched(model, data, labels, batch_size, epochs):
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            batch_loss = criterion(predictions, y_batch)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}")
```

**OCANNL:**
```ocaml
let train_batched data labels batch_size epochs =
  let ctx = Context.auto () in
  
  (* Define model with batch dimension *)
  let batch_n, bindings = IDX.get_static_symbol IDX.empty in
  let%op x_batch = slice_batch ~batch_n ~batch_size data in
  let%op y_batch = slice_batch ~batch_n ~batch_size labels in
  let%op predictions = model x_batch in
  let%op batch_loss = mse predictions y_batch in
  
  (* Initialize and compile *)
  let%op learning_rate = 0.01 in
  let update = Train.grad_update batch_loss in
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  let ctx = Train.init_params ctx bindings batch_loss in
  let routine = Train.to_routine ctx bindings
    (Asgns.sequence [update; sgd]) in
  
  (* Get batch counter reference *)
  let batch_ref = IDX.find_exn (Context.bindings routine) batch_n in
  
  (* Training epochs *)
  let num_batches = Array.length data / batch_size in
  for epoch = 1 to epochs do
    let epoch_loss = ref 0. in
    for batch = 0 to num_batches - 1 do
      batch_ref := batch;
      Train.run ctx routine;
      epoch_loss := !epoch_loss +. batch_loss.@[0]
    done;
    Printf.printf "Epoch %d, Avg Loss: %.6f\n" 
      epoch (!epoch_loss /. float num_batches)
  done;
  ctx
```

### Inference After Training

**PyTorch:**
```python
def inference(model, test_input):
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    return output.numpy()
```

**OCANNL:**
```ocaml
let inference ctx model =
  (* Define inference computation - use %cd to avoid initialization *)
  let%cd output = model { test_input } in
  Train.set_on_host output.value;
  
  (* Compile inference routine *)
  let ctx, routine = Context.compile ctx output IDX.empty in
  
  fun input_data ->
    (* Run inference *)
    Tn.set_values test_input.value input_data;
    Train.run ctx infer_routine;
  
    (* Before OCANNL v0.7, to get all cells flattened: *)
    Tn.get_values output.value
    (* Or starting with the future OCANNL v0.7 to get a bigarray: *)
    Context.get ctx output.value
```

### Key API Functions

- **`Context.auto ()`**: Automatically selects best available backend
- **`Train.init_params ctx bindings tensor`**: Initializes model parameters
- **`Train.grad_update loss`**: Returns forward + backprop computation
- **`Train.sgd_update ~learning_rate loss`**: Returns SGD parameter update computation
- **`Train.forward_once ctx tensor`**: Forward pass only (initializes params if needed)
- **`Train.update_once ctx loss`**: Forward + backward pass (initializes params if needed)
- **`Train.to_routine ctx bindings comp`**: Compiles computation for repeated execution
- **`Train.run ctx routine`**: Executes compiled routine
- **`Asgns.sequence comps`**: Combines multiple computations into one

### Migration from Old Backend API

```ocaml
(* Old API (pre-v0.6.1) *)
let module Backend = (val Backends.fresh_backend ()) in
let ctx = Backend.make_context @@ Backend.new_stream @@ 
          Backend.get_device ~ordinal:0 in
ignore (Train.forward_once (module Backend) ~ctx tensor);

(* New API (v0.6.1+) *)
let ctx = Context.auto () in
let ctx = Train.forward_once ctx tensor in
```

### Performance Tips

1. **Compile once, run many**: Use `Train.to_routine` for operations that will be executed repeatedly
2. **Track initialization**: The Context automatically tracks which tensors are initialized
3. **Reuse contexts**: Pass the same context through your training pipeline
4. **Set backend via environment**: Use `OCANNL_BACKEND=cuda` to control backend selection globally

## Demystifying `Train` - Under the Hood

The `Train` module isn't magic - it's just convenience functions that build computation graphs. Here's how OCANNL's implementations compare to PyTorch internals:

### Gradient Computation

**PyTorch** ([from autograd/__init__.py](https://github.com/pytorch/pytorch/blob/main/torch/autograd/__init__.py)):
```python
def backward(tensors, grad_tensors=None, retain_graph=None):
    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
    grad_tensors = _make_grads(tensors, grad_tensors)
    
    # The actual backward is in C++, but conceptually:
    # 1. Zero gradients if needed
    # 2. Set output gradient to grad_tensors (default 1.0)
    # 3. Walk computation graph backwards accumulating gradients
```

**OCANNL** (from train.ml):
```ocaml
let grad_update ?(setup_for_parallel = false) loss =
  set_hosted loss.Tensor.value;
  (* This just builds a computation graph, doesn't execute *)
  [%cd
    loss.forward;           (* Run forward pass *)
    loss.zero_grads;        (* Zero all parameter gradients *)
    loss.grad =: 1;         (* Set output gradient to 1 *)
    loss.backprop]          (* Run backpropagation *)
```

Key difference: PyTorch executes immediately, OCANNL returns a computation description to compile.

### SGD Optimizer

**PyTorch** ([from optim/sgd.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py)):
```python
@torch.no_grad()
def step(self):
    for group in self.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad
            
            # Add weight decay
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            
            # Apply momentum
            if momentum != 0:
                buf = self.state[p].get('momentum_buffer', None)
                if buf is None:
                    buf = torch.clone(d_p).detach()
                else:
                    buf.mul_(momentum).add_(d_p)
                
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
            
            # Update parameters
            p.add_(d_p, alpha=-lr)
```

**OCANNL** (from train.ml):
```ocaml
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  (* Again, just building computation graph *)
  [%cd
    { sgd_delta } =: p.grad + (!.weight_decay *. p);
    if Float.(momentum > 0.0) then (
      { sgd_momentum } =: (!.momentum *. sgd_momentum) + sgd_delta;
      if nesterov then 
        sgd_delta =+ !.momentum *. sgd_momentum 
      else 
        sgd_delta =: sgd_momentum);
    p =- learning_rate * sgd_delta]

let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov loss =
  (* Apply sgd_one to all parameters *)
  let f = sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov in
  Set.to_list loss.Tensor.params 
  |> List.map ~f 
  |> Asgns.sequence  (* Combine into single computation *)
```

The math is identical! The difference is compilation vs eager execution.

### Training Step Execution

**PyTorch** (typical training loop):
```python
# Everything executes immediately when called
optimizer.zero_grad()        # Modifies tensors in-place
output = model(input)        # Executes forward pass
loss = criterion(output, target)  # Computes loss
loss.backward()              # Executes backprop
optimizer.step()             # Updates parameters
```

**OCANNL** (equivalent pattern):
```ocaml
(* Build computation graph - nothing executes yet *)
let update = Train.grad_update loss in           (* forward + backward *)
let sgd = Train.sgd_update ~learning_rate loss in (* parameter updates *)

(* Compile once *)
let routine = Train.to_routine ctx bindings 
  (Asgns.sequence [update; sgd]) in

(* Execute many times *)
Train.run ctx routine  (* NOW it executes everything *)
```

### Writing Your Own Training Functions

Since `Train` is just convenience functions, you can easily write your own:

**Custom optimizer example:**
```ocaml
(* Adam optimizer - not in Train module but easy to add *)
let adam_update ~learning_rate ~beta1 ~beta2 ~epsilon loss =
  let adam_one p =
    [%cd
      (* First moment estimate *)
      { m } =: (!.beta1 *. m) + ((1.0 - !.beta1) *. p.grad);
      (* Second moment estimate *)  
      { v } =: (!.beta2 *. v) + ((1.0 - !.beta2) *. (p.grad *. p.grad));
      (* Bias correction and update *)
      p =- learning_rate * (m / (sqrt v + !.epsilon))]
  in
  Set.to_list loss.Tensor.params 
  |> List.map ~f:adam_one
  |> Asgns.sequence
```

### Key Insights

1. **No magic**: `Train` functions are just tensor operations packaged conveniently
2. **Same math**: The SGD/momentum/weight_decay math is identical to PyTorch
3. **Compilation advantage**: OCANNL can optimize the entire training step as one unit
4. **Hackable**: Easy to add custom optimizers or training patterns
5. **Transparent**: You can see exactly what operations will run on your device

The `Train` module is meant to be read, understood, and extended by users - it's a recipe book, not a black box!

## Debugging Tips

### Shape Errors
- Use `Shape.set_dim` (or `Shape.set_equal`) to add constraints when inference needs hints
- Remember that `..var..` row variables can match zero or more axes
- Check if you're unnecessarily capturing variables in einsum

### Type Errors with Inline Definitions
- `{ x }` creates learnable parameters, not constants
- Inline definitions are lifted to the unit parameter `()` scope
- Sub-modules don't auto-lift - bind them before use

### Performance
- Virtual tensors (like `0.5 + 0.5`) are inlined during optimization
- Row variables allow operations to work on grouped/multi-channel data
- Input axes (→) in kernels end up rightmost for better memory locality

## Random Number Generation Details

### Initialization Functions

OCANNL's random initialization has some important nuances:

1. **Default initialization is configurable** - There is a global reference that defaults to the `uniform1` operation but can be changed to any nullary operation.

2. **Divisibility requirements** - Functions like `uniform` require the total number of elements to be divisible by certain values (they work with `uint4x32` for efficiency):
   - `uniform()` - requires specific size divisibility for efficient bit usage
   - `uniform1()` - works pointwise on `uint4x32` arrays, allows any size but wastes random bits

3. **Deterministic PRNG** - OCANNL uses counter-based pseudo-random generation:
   - Each `uniform()` call combines global seed with a unique tensor identifier
   - Different calls generate different streams, but deterministically
   - For training randomness (e.g., dropout), use `uniform_at` with `~train_step` to split the randomness key

**Example:**
```ocaml
(* Parameter init - happens once, deterministic is fine *)
{ w = uniform () }

(* Training randomness - needs train_step for proper key splitting *)
dropout ~rate:0.5 () ~train_step x
(* internally uses: uniform_at !@train_step *)
```

## Further Resources

- [Shape Inference Documentation](../dev/neural_nets_lib/Ocannl/Shape/index.html) - Detailed einsum notation spec
- [Syntax Extensions Guide](syntax_extensions.html) - `%op` and `%cd` details  
- [Neural Network Blocks](https://github.com/ahrefs/ocannl/blob/master/lib/nn_blocks.ml) - Example implementations
- [GitHub Discussions](https://github.com/ahrefs/ocannl/discussions) - Community Q&A
