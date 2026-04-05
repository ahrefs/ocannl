# OCANNL notation compared to einops

[einops](https://einops.rocks/) is a popular Python library providing readable tensor manipulation through three core operations -- `rearrange`, `reduce`, and `repeat` -- using a string-based notation. It works with numpy, PyTorch, TensorFlow, JAX, and many other backends.

OCANNL has independently developed a **generalized einsum notation** that covers the same territory and goes significantly beyond it, because it handles compilation, shape inference, and autodiff -- not just runtime structural manipulation.

This document helps two audiences:
1. **Python ML practitioners** familiar with einops who are evaluating OCANNL -- OCANNL's notation is not alien, just an extended version of familiar concepts.
2. **OCANNL users** who encounter einops discussions online -- translate einops examples into OCANNL.

See also: [syntax extensions documentation](syntax_extensions.md#using-ocannls-generalized-einsum-notation), [shapes and einsum slides](https://ahrefs.github.io/ocannl/docs/shapes_and_einsum.html), and [shape.mli](../tensor/shape.mli) for the formal grammar.

## Operation-by-operation comparison

### Transpose / permute

einops:
```python
rearrange(x, 'b c h w -> b h w c')
```

OCANNL (single-char mode):
```ocaml
x ++ "bchw => bhwc"
```

### Reduce (sum)

einops:
```python
reduce(x, 'b c h w -> b c', 'sum')
```

OCANNL:
```ocaml
x ++ "...|...hw => ...|0"
```

The `0` means "reduce to a scalar for this row". The `...` row variables handle the remaining axes via broadcasting. Axes present on the right-hand side but absent from the left-hand side are summed out (for `++`).

### Reduce (max)

einops:
```python
reduce(x, 'b c h w -> b c', 'max')
```

OCANNL:
```ocaml
x @^^ "...|...hw => ...|0"
```

The `@^^` operator uses maximum instead of addition for the accumulation.

### Reduce to scalar

einops:
```python
reduce(x, 'b c h w -> ', 'sum')
```

OCANNL:
```ocaml
x ++ "...|...->... => 0"
```

This reduces all axis kinds (batch, input, output) into a single number.

### Matrix multiplication / einsum

einops:
```python
einsum(A, B, 'b i k, b k j -> b i j')
```

OCANNL (multichar mode, triggered by commas):
```ocaml
A +* B "...|i, k; ...|k, j => ...|i, j"
```

Or in single-char mode:
```ocaml
A +* B "...|ik; ...|kj => ...|ij"
```

Note: OCANNL uses `|` and `->` to separate axis kinds (batch, input, output), which is more information than einops' flat axis list.

### Multi-head attention scores

einops (reshape then matmul):
```python
# Reshape: split heads
q = rearrange(q, 'b t (h d) -> b h t d', h=num_heads)
k = rearrange(k, 'b t (h d) -> b h t d', h=num_heads)
# Matmul: attention scores
scores = einsum(q, k, 'b h t1 d, b h t2 d -> b h t1 t2')
```

OCANNL (from [nn_blocks.ml](../lib/nn_blocks.ml), line 121):
```ocaml
let scores =
  (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
```

OCANNL captures dimension variables `h` and `d` directly in the einsum spec via `["h"; "d"]`, then constrains them with `Shape.set_dim h num_heads`. No separate reshape step is needed -- shape inference handles the axis decomposition.

### Repeat / broadcast

einops:
```python
repeat(x, 'h w -> h w c', c=3)
```

OCANNL handles this implicitly through broadcasting and shape constraints. When a tensor is used in a context requiring additional axes, OCANNL broadcasts automatically. To explicitly add a dimension:
```ocaml
x ++ "hw => hw_"
```
where `_` is a placeholder that introduces an axis without constraining it. The dimension can then be set via shape constraints.

### Average pooling

einops:
```python
reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)
```

OCANNL uses affine indexing for pooling (from [syntax_extensions.md](syntax_extensions.md#affine-indexing-for-convolutions-and-pooling)):
```ocaml
input +* "...|2*oh<+wh, 2*ow<+ww, ..c..; wh, ww => ...|oh, ow, ..c.." window
```

For max pooling, use `@^+` instead of `+*`:
```ocaml
input @^+ "...|2*oh<+wh, 2*ow<+ww, ..c..; wh, ww => ...|oh, ow, ..c.." window
```

The `2*oh<+wh` means: input index = `2 * output_position + kernel_offset`, i.e. stride-2 with no padding (`<`).

### Global average pooling

einops:
```python
reduce(x, 'b c h w -> b c', 'mean')
```

OCANNL:
```ocaml
x ++ "...|...hw => ...|0"
```

For a true average (not sum), divide by the dimension sizes.

### 2D convolution

einops has no built-in convolution support.

OCANNL ([syntax_extensions.md](syntax_extensions.md#affine-indexing-for-convolutions-and-pooling), line 499):
```ocaml
input +* "...|oh<+wh, ow<+ww, ..ic..; wh, ww, ic => ...|oh, ow, ..oc.." kernel
```

This sum-reduces over kernel height `wh`, kernel width `ww`, and input channels `ic`. Output channels come from the kernel's output shape, inferred automatically.

### Tensor concatenation

einops has no native axis concatenation.

OCANNL:
```ocaml
(a, b, c) ++^ "x, ...; y, ...; z, ... => x^y^z, ..."
```

The `^` operator creates a concatenated axis. This also enables slicing:
```ocaml
(* Extract prefix *)
x ++^ "a^b => a"
(* Extract suffix *)
x ++^ "a^b => b"
(* Skip first 3 elements *)
x ++^ "3^a => a"
```

### Layer normalization

einops (structural part only, no learned parameters):
```python
mean = reduce(x, 'b t d -> b t 1', 'mean')
```

OCANNL (from [nn_blocks.ml](../lib/nn_blocks.ml), line 136, complete with learned parameters):
```ocaml
let%op layer_norm ~label ?(epsilon = 1e-5) () x =
  let mean = x ++ " ... | ..d..  => ... | 0 " [ "d" ] in
  let centered = (x - mean) /. dim d in
  let variance = (centered *. centered) ++ " ... | ... => ... |  0 " in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  ({ gamma = 1. } *. normalized) + { beta = 0. }
```

The `{ gamma = 1. }` and `{ beta = 0. }` are learnable parameters created inline by the `%op` syntax extension.

## Summary comparison table

| Feature | einops | OCANNL |
|---------|--------|--------|
| Transpose/permute | `rearrange(x, 'a b c -> c a b')` | `x ++ "abc => cab"` |
| Flatten axes | `rearrange(x, 'b c h w -> b (c h w)')` | Implicit via row structure |
| Split axes (reshape) | `rearrange(x, 'b (h h2) w -> b h h2 w', h2=2)` | Shape constraints + dim capture |
| Reduce (sum) | `reduce(x, 'b c h w -> b c', 'sum')` | `x ++ "...\|...hw => ...\|0"` |
| Reduce (max) | `reduce(x, 'b c h w -> b c', 'max')` | `x @^^ "...\|...hw => ...\|0"` |
| Repeat/broadcast | `repeat(x, 'h w -> h w c', c=3)` | Implicit broadcasting + shape constraints |
| Einsum (matmul) | `einsum(A, B, 'b i k, b k j -> b i j')` | `A +* B "...\|ik; ...\|kj => ...\|ij"` |
| Convolution | Not supported | `input +* "oh<+wh, ow<+ww, ic; wh, ww, ic => oh, ow, oc" kernel` |
| Pooling | Reshape + reduce | `input @^+ "2*oh<+wh, 2*ow<+ww; wh, ww => oh, ow" window` |
| Axis concatenation | Not natively | `(a, b) ++^ "x; y => x^y"` |
| Shape inference | Runtime dimension check | Compile-time constraint solving |
| Autodiff integration | None (structural only) | Full: `%op` creates differentiable tensors |
| Named broadcasting | `...` ellipsis only | `..batch..`, `..input..`, named row variables |

## Key conceptual differences

### Three-row shape system vs flat axis list

einops treats all axes as equivalent -- `'b c h w'` is a flat list of four dimensions. OCANNL distinguishes three axis kinds separated by `|` and `->`:

```
batch | input -> output
```

This distinction enables semantically richer operations: matrix multiplication naturally contracts input axes, batch axes broadcast pointwise, and output axes define the result shape. The three-row system also drives OCANNL's shape inference engine.

### Compile-time shape inference vs runtime checking

einops checks dimensions at runtime: if you write `rearrange(x, 'b c h w -> b h w c')` on a 3D tensor, you get a runtime error. OCANNL's shape inference catches many such errors at graph construction time, before any computation runs. The inference engine solves constraints across the entire computation graph in 7 stages (see [shape_inference.md](shape_inference.md)).

### Integrated autodiff

einops operations are purely structural -- they have no gradient implications. OCANNL's `%op` syntax extension creates differentiable tensor operations with automatic backpropagation. The einsum notation is part of the autodiff graph, not a separate pre/post-processing step.

### Axis decomposition

This is one area where einops has a more concise notation. einops' parenthesized axes `(h h2)` directly express "this single dimension is the product of `h` and `h2`", enabling reshape-like axis splitting and merging inline. OCANNL does not have direct syntactic support for reshape -- instead, it uses shape constraints and dimension capture (the `["h"; "d"]` syntax) to achieve similar results. The OCANNL approach is more verbose for simple reshapes but integrates naturally with the constraint-based shape inference system.

### Framework scope

einops is a lightweight structural manipulation library that works across 10+ frameworks. OCANNL is a complete deep learning framework with its own compiler, backends (CPU, CUDA, Metal), and training infrastructure. The comparison here is notation-to-notation, showing that OCANNL's generalized einsum subsumes einops' expressive power while adding capabilities needed for compilation and shape inference.
