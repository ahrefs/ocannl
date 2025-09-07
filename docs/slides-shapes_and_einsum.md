# Shapes and Einsum in OCANNL: From Basics to Advanced

{pause}

{#intro .definition title="Shape Inference in OCANNL"}
> OCANNL provides powerful shape inference and generalized einsum notation for building neural networks:
> 
> * **Shape inference**: Automatic deduction of tensor dimensions
> * **Row variables**: Flexible handling of unknown axis counts  
> * **Einsum notation**: Concise syntax for complex tensor operations
> * **Three axis kinds**: `batch | input -> output` (matrix convention)

{pause}

{#why-shapes .block title="Why Care About Shapes?"}
> * **Expressivity gains**: The axis kind split doesn't enforce semantics but offers better expressivity
> * **Broadcasting magic**: Tensors with dimension-1 axes broadcast automatically
> * **Less commitment**: Use row variables `..d..` where axis count doesn't matter
> * **Type safety**: Shape mismatches caught at compile time, not runtime

{pause up=why-shapes #numpy-differences}
## From NumPy to OCANNL: Key Differences

{.remark}
> OCANNL's einsum differs syntactically from NumPy:
> 
> * `->` separates input/output axes (not arguments/result)
> * `=>` appears to the left of the result tensor
> * `;` separates argument tensors
> * `,` separates identifiers in multi-char mode

{pause up=numpy-differences #basic-example .example title="Your First Einsum"}
```ocaml
open Ocannl.Operation.DSL_modules

let%op matrix_multiply a b = 
  a +* "m n; n p => m p" b
```

Compare with NumPy: `np.einsum("mn,np->mp", a, b)`

{pause up}
## The Three Axis Kinds

{#axis-kinds .definition title="Batch | Input -> Output"}
> Every tensor shape has three rows of axes:
> 
> * **Batch axes**: Data samples, preserved in operations
> * **Output axes**: Results of computations (leftmost in matrix convention)
> * **Input axes**: Arguments to operations (rightmost in matrix convention)

{pause}

{#axis-example .example title="Shape Specifications"}
```ocaml
(* Different ways to specify the same shape *)
"batch | input -> output"    (* Full specification *)
"input -> output"             (* No batch axes *)
"batch | output"              (* No input axes *)  
"output"                      (* Just output axes *)
```

{pause up}
## Row Variables and Broadcasting

{#row-variables .definition title="The Power of ..."}
Row variables enable flexible axis handling:

* `...` - Context-dependent ellipsis
* `..var..` / `..v..` - Named row variable (multi-char / single-char mode)
* Broadcasting happens "in the middle"

{pause}

{#ellipsis-examples .example title="Using Ellipsis"}
```ocaml
(* Reduce all output axes to scalar,
   require no batch or input axes *)
let%op sum_all_output x = x ++ "... => 0"

(* Reduce all axes to scalar *)
let%op sum_all x = x ++ "...|...->... => 0"

(* Pointwise operation preserving all axes *)
let%op square x = x *. x
       (* Implicit: "...|...->... => ...|...->..." *)

(* Sum over last output axis only, require no input axes *)
let%op sum_last x = x ++ "...|...k => ...|..." 
```

{pause up}
## Multi-Character Mode

{#multichar-mode .block title="When Identifiers Get Long"}
> Add a comma anywhere to enable multi-character identifiers:
> 
> ```ocaml
> (* Single-char mode *)
> "b | hw -> c"  (* b, h, w, c are separate axes *)
> 
> (* Multi-char mode (note the comma) *)
> "batch | height, width -> channels,"  
> ```

{pause}

{#multichar-example .example title="Real-World Multi-Char"}
```ocaml
let%op attention ~num_heads () x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  (* Capture dimensions for scaling *)
  let scores = 
    q +* "... seq | heads, ..dims..; ... time | heads, ..dims..
          => ... seq | time -> heads"
      ["heads"; "dims"] k 
  in
  Shape.set_dim heads num_heads;
  scores /. sqrt (dim dims)
```

{pause up=multichar-example}

## Einsum Operators

{#einsum-ops .definition title="Built-in Einsum Operators"}
> OCANNL provides specialized einsum operators:
> 
> | Operator | Unary/Binary | Accumulation | Operation | Function |
> |----------|--------------|--------------|-----------|----------|
> | `++`     | Unary        | Add          | -         | `einsum1` |
> | `+*`     | Binary       | Add          | Multiply  | `einsum` |
> | `@^^`    | Unary        | Max          | -         | `einmax1` |
> | `@^+`    | Binary       | Max          | Add       | `tropical` |

{pause up=einsum-ops}

{#operator-examples .example title="Using Einsum Operators"}
```ocaml
(* Matrix multiplication on individual output axes *)
let%op matmul a b = a +* "ik; kj => ij" b

(* Batch matrix multiply on individual output axes
   with broadcasting *)  
let%op batch_matmul a b = 
  a +* "... | ik; ... | kj => ... | ij" b

(* Full tensor multiplication, equivalent to [a * b] *)  
let%op tensor_mul a b = 
  a +* "... | ..k..->..i..; ... | ..j..->..k..
        => ... | ..j..->..i.." b

(* Max pooling, requiring specifically 4 output axes *)
let%op max_reduce_ouput x = x @^^ "bhwc => b00c"

(* Max pooling, with arbitrary batch axes and no input axes *)
let%op max_reduce x = x @^^ "...|hwc => ...|00c"
```

{pause up=operator-examples}

{pause up}
## Convolutions with Einsum

{#convolution-syntax .definition title="Convolution Notation"}
> Use `+` for convolution axes with stride and dilation:
> 
> ```
> "stride*output + dilation*kernel"
> ```
>
> Within the syntax extensions, `stride` and `dilation` can be identifiers of `int` values, in addition to integer literals.

{pause}

{#conv-example .example title="2D Convolution"}
```ocaml
let%op conv2d ?(stride=1) ?(kernel_size=3) () x =
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  x +* "... | stride*oh+kh, stride*ow+kw, ic;
        kh, kw, ic -> oc => ... | oh, ow, oc"
       ["kh"; "kw"] { kernel }
  + { bias = 0. }
```

{pause up=conv-example}

## Capturing Dimensions

{#capture-dims .block title="Dimension Variables"}
> Capture dimension values for computation:
> 
> ```ocaml
> let%op normalize x =
>   let mean = x ++ "... | ..d.. => ... | 0" ["d"] in
>   let centered = (x - mean) /. dim d in
>   centered /. sqrt (variance + !.1e-5)
> ```

{pause up=capture-dims #capture-warning .remark}
**Warning**: Captured variables can shadow other identifiers. Only capture what you need!

{pause down #constants-trick .block title="Creating Shape-Inferred Constants"}
> Need a tensor of ones that matches another tensor's shape?
> 
> ```ocaml
> (* Used in pooling to propagate dimensions *)
> let%op avg_pool2d ~window () x =
>   let kernel = 0.5 + 0.5 in  (* Shape-inferred 1s *)
>   x +* "... | stride*oh+kh, stride*ow+kw, c; 
>         kh, kw => ... | oh, ow, c" kernel
>   /. !.(window * window)
> ```

{pause up}
## Advanced: Building Custom Operations

{#custom-ops .example title="Beyond Built-in Operators"}
> You can define operations with custom accumulation:
> 
> ```ocaml
> let einmax ?(label = []) ?(capture_dims = []) spec =
>   let%cd op_asn ~v ~t1 ~t2 ~projections = v =:@^ v1 * v2 in
>   let%cd grad_asn ~t ~g ~t1 ~t2 ~projections =
>     g1 =+ where (t = t1 + t2) (g *. v2) 0;
>     g2 =+ where (t = t1 + t2) (v1 *. g) 0
>   in
>   Tensor.binop
>     ~compose_op:(Shape.Einsum (spec, capture_dims))
>     ~op_asn ~grad_asn ~label:("@^=>*" :: label)
> ```

{pause center}

It looks complicated because we introduce a brand new differentiable tensor operation. It's easier to use custom accumulations in code. That uses `~logic:` which can be any spec, with special handling for `"."` (pointwise), `"@"` (tensor inner product / multiply, like `*`), `"T"` (transpose: swap input and output axes).

{pause down #custom-sgd .example title="Custom SGD step with per-parameter min/max values"}
```ocaml
let sgd_and_track ~learning_rate p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error
      ("sgd_and_track: not differentiable", Some p);
  let%cd update =
    ~~(p "sgd track min max";
       p =- learning_rate * sgd_delta ~logic:".";
       { p_min } =@- p ~logic:"...->... => 0";
       { p_max } =@^ p ~logic:"...->... => 0") in
  update, p_min, p_max
```

{pause up}
You can programmatically create the spec for use with the dedicated syntaxes, but for the sake of code clarity this does not support variable capture.

{#custom-example .example title="Reduce Last N Dimensions"}
```ocaml
(* Reduce last N output dimensions, PyTorch-style keepdim *)
let%op reduce_last_n ~n ?(keepdim = true) () =
  let vars =
    [%oc List.init n ~f:(fun i -> 
           Char.to_string (Char.of_int_exn (97 + i)))] in
  let result_dims = 
    [%oc if keepdim then String.make n '0' else ""] in
  let spec = [%oc "... | ..." ^ String.concat "" vars ^ 
                  " => ... | ..." ^ result_dims] in
  fun x -> x ++ spec
  
(* Example: reduce_last_n ~n:3 ~keepdim:true () 
   generates: "... | ...abc => ... | ...000" 
   
   With ~keepdim:false:
   generates: "... | ...abc => ... | ..." *)
```

{pause}

The `[%oc ...]` syntax allows embedding arbitrary OCaml code without `%op` attempting to interpret things as tensors.

{pause up}
## Practical Patterns

{#patterns .block title="Common Shape Patterns"}
> **Principle of least commitment**: Use row variables where possible
> 
> * `"...|...->..."` - Handle ANY shape (all three axis kinds)
> * `"...->..."` - Parameters (no batch axes expected)
> * `"...|..."` - Data tensors (no input axes expected)
> * `"... | ..d.. => ... | 0"` - Reduce unknown output axes
> * `"...|...->...; ...|...->... => ...|...->..."` - Pointwise binary op

{pause down #pattern-examples .example title="Real Examples from nn_blocks.ml"}
```ocaml
(* Layer norm - reduce over feature dimensions *)
let%op layer_norm () x =
  let mean = x ++ "... | ..d.. => ... | 0" ["d"] in
  let normalized = (x - mean) /. dim d in
  ({ gamma = 1. } *. normalized) + { beta = 0. }

(* Attention scores with flexible batching *)
let scores = 
  q +* "...batch.., seq | heads, ..dims..; 
        ...batch.., time | heads, ..dims.. => 
        ...batch.., seq | time -> heads" 
    ["heads"; "dims"] k
```

{pause up}
## Shape Inference Magic

{#inference .definition title="How Shape Inference Works"}
> OCANNL's shape inference is declarative:
> 
> 1. **Collect constraints** from tensor operations
> 2. **Propagate shapes** bottom-up and top-down  
> 3. **Solve inequalities** (dim-1 broadcasts to any size)
> 4. **Substitute variables** with least upper bounds

{pause}

{#inference-example .example title="Shape Inference in Action"}
```ocaml
let%op flexible_mlp () x =
  (* Fix hidden dim where it can't be inferred, 
     let other shapes propagate *)
  { w_out } * relu ({ w_hid } * x + { bias; o = [128] })
  
(* Shape propagation:
   - From x: determines w_hid's input dims
   - From usage (e.g., loss): determines w_out's dims
   - Only bias needs explicit [128] - no other context! *)
```

{pause up}
## Projections: Under the Hood

{#projections .block title="From Shapes to Loops"}
> Projections determine how tensors are indexed in generated code:
> 
> * Each axis gets a projection (iterator or fixed index)
> * Broadcasting axes → fixed index 0
> * Convolutions → strided iteration with padding
> * Projections are freshened per operation (no contamination)

{pause}

{#projection-example .example title="Projection Example"}
```ocaml
(* This einsum: *)
a +* "ij; jk => ik" b

(* Generates loops like: *)
(* for i = 0 to I-1 do
     for k = 0 to K-1 do  
       c[i,k] = 0;
       for j = 0 to J-1 do
         c[i,k] += a[i,j] * b[j,k] *)
```

{pause up}
## Tips and Tricks

{#tips .block title="Best Practices"}
> 1. **Use `|`, `->` for axis kinds** when there's a meaningful batch/input/output split
> 2. **Add trailing comma** for multi-char mode: `"input->output,"`
> 3. **Avoid over-capturing** dimensions in einsum specs
> 4. **Remember tensor operators**: `*` (matmul) vs `*.` (pointwise)
> 5. **Let inference work**: Don't over-specify dimensions

{pause}

{#debugging .remark title="Debugging Shapes"}
> When shapes don't match:
> 
> * Print tensor shapes: `Tensor.print ~force:true tensor`  
>   [but not before all relevant tensor expressions are constructed]{.unrevealed #premature-inference-finalize}
> * Check axis kinds are correctly specified
> * Verify broadcasting assumptions
> * Use explicit dimension constraints when needed

{pause reveal=premature-inference-finalize}

{pause focus=premature-inference-finalize}

{pause unfocus up}
## Common Pitfalls

**Tensor operators matter:**
* `*` - tensor multiplication (matrix multiply generalized)
* `*.` - pointwise multiplication
* `/.` - pointwise division (not using `/` with tensors, for consistency)

**Einsum spec with variable capture must be literal:**
```ocaml
(* Wrong: *)
let spec = "i j => j i" in (x ++ spec [ "j" ]) /. dim j

(* Right: *)
(x ++ "i j => j i" [ "j" ]) /. dim j
```

**Single vs multi-char mode:**
```ocaml
"abc"        (* 3 axes: a, b, c *)
"abc,"       (* 1 axis: abc (comma triggers multi-char) *)
"a, b, c"    (* 3 axes: a, b, c (multi-char mode) *)
```

{pause up}
## Building Your First Model

Try building a simplified attention mechanism:

{#exercise carousel .example title="Exercise: Custom Attention"}
> {.block title="Task"}
> ```ocaml
> let%op simple_attention () input =
>   (* Hints:
>      - Project input to query and key spaces
>      - Compute attention scores (QK^T)
>      - Apply softmax normalization
>      - Weight values by attention *)
>   let query = { w_q } * input in
>   let key = { w_k } * input in
>   let value = { w_v } * input in
>   (* Your einsum operations here... *)
>   ???
> ```
> 
> ---
> 
> {.block title="Solution"}
> ```ocaml
> let%op simple_attention () input =
>   let query = { w_q } * input in
>   let key = { w_k } * input in
>   let value = { w_v } * input in
>   let scores =
>     query +* "batch, seq, dim; batch, time, dim
>               => batch, seq, time" key in
>   let weights =
>     softmax ~spec:"... | ..., time" () scores in
>   weights +* "batch, seq, time; batch, time, dim
>               => batch, seq, dim" value
> ```

{pause change-page=exercise}

{pause}

Check [nn_blocks.ml](https://github.com/ahrefs/ocannl/blob/master/lib/nn_blocks.ml#L68), which uses an input axis in the scores.

{pause up}
## Summary

{#summary .definition title="Key Takeaways"}
> You've learned to:
> 
> * **Navigate** OCANNL's three-axis shape system
> * **Write** concise einsum specifications  
> * **Leverage** row variables for flexibility
> * **Build** complex operations from primitives
> * **Trust** shape inference to handle the details

{pause}

{#next-steps .block title="Next Steps"}
> * Read `docs/migration_guide.md` for PyTorch/TF comparisons
> * Study `lib/nn_blocks.ml` for production patterns
> * Experiment with custom einsum operations
> * Build your own neural network layers
> 
> Remember: **Lean on shape inference** - commit only to what matters!

{pause down #philosophy .remark title="The OCANNL Way"}
> Unlike PyTorch/TensorFlow where you specify exact shapes, OCANNL encourages:
> 
> * Using row variables (`..v..`) for flexible architectures
> * Letting shape inference propagate constraints
> * Working with multi-dimensional channels naturally
> * Avoiding premature shape commitments

{pause up=philosophy}
## References

{#references}
* [OCANNL Website](https://github.com/ahrefs/ocannl)
* [OCANNL Documentation](https://ahrefs.github.io/ocannl/docs)
* [doc/syntax_extensions.md](syntax_extensions.html) - Full `%op` and `%cd` syntax
* [lib/shape.mli](../dev/neural_nets_lib/Ocannl/Shape/index.html) - Shape inference internals
* [lib/nn_blocks.ml](https://github.com/ahrefs/ocannl/blob/master/lib/nn_blocks.ml#L68) - Production examples
* [test/einsum_trivia.ml](https://github.com/ahrefs/ocannl/blob/master/test/einsum/einsum_trivia.ml) - Einsum test cases