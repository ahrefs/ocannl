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
|-------------------|---------|--------|
| `x.view(-1, d)` or `x.reshape(-1, d)` | Not directly supported | Use manual dimension setting on constant tensor as workaround |
| `x.flatten()` | Not supported | Future syntax might be: `"x,y => x&y"` |
| `nn.Conv2d(in_c, out_c, kernel_size=k)` | `conv2d ~kernel_size:k () x` | Channels inferred or use row vars |
| `F.max_pool2d(x, kernel_size=k)` | `max_pool2d ~window_size:k () x` | Uses `(0.5 + 0.5)` trick internally |
| `F.avg_pool2d(x, kernel_size=k)` | `avg_pool2d ~window_size:k () x` | Normalized by window size |
| `nn.BatchNorm2d(channels)` | `batch_norm2d () ~train_step x` | Channels inferred |
| `F.dropout(x, p=0.5)` | `dropout ~rate:0.5 () ~train_step x` | Needs train_step for PRNG |
| `F.relu(x)` | `relu x` | Direct function application |
| `F.softmax(x, dim=-1)` | `softmax ~spec:"... \| ... -> ... d" () x` | Specify axes explicitly |
| `torch.matmul(a, b)` | `a * b` or `a +* "...; ... => ..." b` | Einsum for complex cases |
| `x.mean(dim=[1,2])` | `x ++ "... \| h, w, c => ... \| 0, 0, c" ["h"; "w"] /. (dim h *. dim w)` | Sum then divide |
| `x.sum(dim=-1)` | `x ++ "... \| ... d => ... \| 0"` | Reduce by summing |

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
    let x = conv1 x in
    let x = relu x in
    let x = pool x in
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
    let out = conv1 x in
    let out = bn1 ~train_step out in
    let out = relu out in
    let out = conv2 out in
    let out = bn2 ~train_step out in
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
   - Identifiers can be multi-character (e.g., `height`, `width`)
   - Must be separated by non-alphanumeric: `,` `|` `->` `;` `=>`
   - Enables convolution syntax: `stride*out+kernel`

| Operation | PyTorch einsum | OCANNL single-char | OCANNL multi-char |
|-----------|---------------|-------------------|-------------------|
| Matrix multiply | `torch.einsum('ij,jk->ik', a, b)` | `a +* "i j; j k => i k" b` | `a +* "i, j; j, k => i, k" b` |
| Batch matmul | `torch.einsum('bij,bjk->bik', a, b)` | `a +* "b i j; b j k => b i k" b` | `a +* "batch, i -> j; batch, j -> k => batch, i -> k" b` |
| Attention scores | `torch.einsum('bqhd,bkhd->bhqk', q, k)` | `q +* "b q \| h d; b k \| h d => b \| q k -> h" k` | `q +* "b, q \| h, d; b, k \| h, d => b \| q, k -> h" k` |
| Convolution | N/A | N/A (needs multi-char) | `x +* "... \| stride*oh+kh, stride*ow+kw, ic; kh, kw, ic -> oc => ... \| oh, ow, oc" kernel` |

### Row Variables
- `...` context-dependent ellipsis: expands to `..batch..` in batch position, `..input..` before `->`, `..output..` after `->`
- `..b..` for batch axes (arbitrary number)
- `..ic..`, `..oc..` for input/output channels (can be multi-dimensional)
- `..spatial..` for spatial dimensions

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

⚠️ **Important:** OCANNL doesn't currently support flattening/reshaping operations.

```ocaml
(* This performs REDUCTION (sum), not flattening: *)
x ++ "... | ..spatial.. => ... | 0"  

(* OCANNL's approach: Let FC layers work with multiple axes!
   Instead of flattening [batch, h, w, c] to [batch, h*w*c],
   just let your FC layer handle [batch, h, w, c] directly.
   The matrix multiplication will work across all the axes. *)

(* Example: FC layer after conv without flattening *)
let%op fc_after_conv () x =
  (* x might have shape [batch, height, width, channels] *)
  { w } * x + { b }  (* w will adapt to match x's shape *)
```

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

**OCANNL (conceptual):**
```ocaml
(* OCANNL handles training differently - see Train module *)
let sgd = Train.sgd_update ~learning_rate loss in
let train_step = Train.to_routine ~ctx [%cd update; sgd] in
(* Training happens via routines and contexts *)
```

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

1. **Default initialization is configurable** - There is a global reference that defaults to the `uniform` operation but can be changed to any nullary operation.

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

- [Shape Inference Documentation](../lib/shape.mli) - Detailed einsum notation spec
- [Syntax Extensions Guide](../lib/syntax_extensions.md) - `%op` and `%cd` details  
- [Neural Network Blocks](../lib/nn_blocks.ml) - Example implementations
- [GitHub Discussions](https://github.com/ahrefs/ocannl/discussions) - Community Q&A