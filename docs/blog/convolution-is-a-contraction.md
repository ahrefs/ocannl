---
title: "Convolution Is a Contraction: Indexing and Initialization in OCANNL"
author: "Łukasz Stafiniak and Claude (Anthropic)"
---

# Convolution Is a Contraction

Here is the loop nest a one-dimensional convolution compiles to, one input channel and one output channel, so that only the slide is in play:

```
for o in 0 .. O-1:
    out[o] = 0
    for k in 0 .. K-1:
        out[o] += x[o + k] * kernel[k]
```

There is a loop for the output position and a loop for the kernel position. There is none for the input's axis: the input is the largest array in sight, yet it is never traversed, only read at `o + k`, a position computed from the two loops that do exist. Set that nest beside the one a matrix-vector product compiles to:

```
for i in 0 .. I-1:
    out[i] = 0
    for j in 0 .. J-1:
        out[i] += a[i, j] * v[j]
```

The two are the same nest. Both accumulate a sum of products over an axis the output does not have; both pre-clear the output cell because that axis is summed away. They differ in one place: `a[i, j]` reads at a pair of loop variables, `x[o + k]` at their sum. This post is about that one difference.

Two companion pieces built the machinery underneath. [*Broadcasting Is an Order*](https://lukstafi.github.io/notes/broadcast-aware-shape-inference.html) developed shape inference: what a shape is, and how a solver fills in the sizes. [*Contraction Is Emergent*](https://lukstafi.github.io/notes/contraction-is-emergent.html) developed projection inference: how the same constraints, re-read as an equivalence over axes rather than an order over sizes, become the loop nest of an operation, with reductions read off the result rather than declared. Both deferred two constructs: strided and convolutional indexing, and axis concatenation. This post takes the first. Concatenation changes the shape of the iteration domain and is left to a sequel; convolution leaves the domain alone and changes only what is read inside it.

## The third index form

The second post compiled an operation to a product space --- a Cartesian product of integer ranges, one loop variable per factor --- and, for each tensor, an index map sending a product point to a multi-index into that tensor's array. Each component of an index map took one of two forms: `Iter(s)`, the axis driven by loop variable `s`, or `Fix(c)`, the axis pinned to the constant `c`. The matrix-vector product uses only these: output `(Iter i)`, matrix `(Iter i, Iter j)`, vector `(Iter j)`, with `j` shared, so the product space is `{i, j}` and the output's omission of `j` is what makes `j` a reduction axis. The user wrote `a * v`; the shared axis, the contraction, and the pre-clear were inferred.

Convolution needs one more form:

- `Affine(Σ cᵢ·sᵢ + o)`: the axis is an affine combination of loop variables, with integer coefficients and an integer offset.

The convolution's input map is `(Affine(1·o + 1·k))` --- the sum of two loops that exist for reasons unrelated to the slide: `o` because the output is traversed, `k` because the kernel is summed.

The new form contains the old two: `Iter(s)` is `Affine(1·s + 0)`, `Fix(c)` is `Affine(0 + c)`. OCANNL keeps all three as separate variants and forbids an `Affine` carrying a single unit-coefficient symbol or no symbols at all --- such an index must be written as the `Iter` or `Fix` it is. Convolution adds no form that was not already the general shape of an index; it populates an interior the core left empty. The product space stays flat: one factor per iterated axis, each a single range. The only change is that one index-map component now names two factors instead of one.

The reduction characterization carries over verbatim. A product axis is a reduction axis exactly when the output map is independent of its loop variable. The convolution's output map `(Iter o)` omits `k`, so `k` is summed, by the predicate that summed `j` above. The affine index *uses* `k` to address the input, but addressing an operand with a loop variable is not mentioning it in the output, and only the latter governs reduction. The kernel is contracted because the output omits it; the arithmetic, the accumulation, and the reduction are the contraction's, untouched. Only the input's address is new.

## One term, two readings

The affine index comes from an enriched constraint, as every projection in the second post came from a constraint re-read. The dimension term was a size and a basis, with a per-axis identifier added in the second post for axis identity. We add a constructor:

```
Affine { stride; over; conv; stride_offset }
```

`over` is the transformed dimension --- the output axis; `stride > 0` is its coefficient; `stride_offset`, with `0 ≤ stride_offset < stride`, selects a position within a stride window; `conv` is an optional `{ dilation; kernel; use_padding }`, absent for pure striding and present to add the kernel. Two solvers read this one term, and reading it from one source is what keeps the input's size and address from disagreeing.

The size solver asks how big the input axis is, given the output. For pure striding, `input = stride · output`. For valid convolution (`use_padding = false`), `input = stride · (output − 1) + span`, where `span = 1 + (kernel − 1) · dilation` is the range of input positions a dilated kernel covers: the largest input index reached is `(output − 1)·stride + (kernel − 1)·dilation`, at the last output and kernel positions, and the axis is one longer than that. Solving for the output, `output = (input − span) / stride + 1`, requires `input − span` non-negative and divisible by the stride; an image and kernel that do not tile leave no integer output, and the constraint fails. For padded convolution (`use_padding = true`), `input = stride · output` again: the padding absorbs the kernel's overrun, so the kernel enters the address but not the size. The solver simplifies where it can --- a `stride = 1` term over a known dimension is that dimension, a valid-mode term with a known kernel evaluates the span --- the same solver the first post built for broadcasting, now with more to compute than an order.

The projection solver asks which input cell a product point reads. The same term evaluates to

```
stride · output_iterator + dilation · kernel_iterator + stride_offset − padding
```

the kernel term and padding present only in convolution mode: coefficient `stride` on the output loop, `dilation` on the kernel loop, a constant offset. `stride_offset` is where the two readings come apart most cleanly. It appears in no size relation; the same input is read at every offset from `0` to `stride − 1` with no size changing. A size concern would grow the tensor; an addressing concern moves the read.

## Solve, then evaluate

Writing `Affine(1·o + 1·k)` for the input's index assumes `o` and `k` already exist --- the actual loop variables minted for the output and kernel axes. But minting those is the solver's job, and the input's index is something the solver is meant to produce: the index is a function of results the solve has not yet computed.

So the projection solve runs in two strata. The first is the core solver of the second post, unchanged: union-find the base classes (here the `k`-class, shared by the input's affine expression and the kernel, and the `o`-class, shared by the expression and the output), pin any class forced to a constant, and mint one fresh loop variable per iterated class --- except for an axis whose index will come from an affine term, which gets no loop of its own, its index being computed rather than iterated. The second stratum walks the deferred affine terms and evaluates each against the loop variables the first produced.

The second post noted that its core projection language had no variable form: every index was an atom the union-find settled, leaving nothing to stand for an unknown. Here a variable form returns. An affine index names loop variables the first stratum has not yet minted, so the solver carries a projection variable and a small binding environment to hold the not-yet-known iterator until the second stratum resolves it. The variable was dispensable exactly as long as indices were atoms.

Two affine terms can be required to denote the same index, and whether they do cannot be checked until both are evaluated; the solver defers such checks to the second stratum. The core solver, the second post observed, never resolves an ambiguity --- it succeeds or reports a conflict --- which made it more principal than shape inference. The affine extension keeps that; there is still nothing to guess. What it gives up is flatness: a check can fire only after the stratum that evaluates its operands, so a conflict can surface there rather than where the constraint was written.

## Padding

A padded convolution reads outside its input: at output `0`, kernel `0`, the address is `0 − padding`, left of the array. The buffer is widened to hold the overrun, and the widening is never written down --- it is inferred, in the same sense the contraction was.

OCANNL separates the two notions of size this requires. Shapes, shape inference, and tensors carry sizes without padding; the underlying buffers carry it. A convolution's input *shape* is `stride · output`; the *buffer* is larger by the padding margins. The margin is computed during projection inference, keyed by the input axis's projection identifier, as the running maximum of the kernel extents of every operation addressing that axis: an axis read by a 3-wide kernel and elsewhere a 5-wide one is padded for the 5, and the maximum is what makes every reader's overrun fit. For a kernel of size `k` the margin splits `right = (k + 1) / 2`, `left = k − right`.

That maximum tightens monotonically as constraints are processed and is committed once, at finalization --- the boundary at which the first post substituted unsolved shape variables by their bounds, and the second required shapes closed before projections could be read. Inference in OCANNL is monotonic in its core rules and non-monotonic only in the large: something must decide when a shape is final, because a tensor cannot be allocated until it is. Padding is one more quantity accumulated under the monotonic rules and frozen at that seam; after the freeze the buffer can be allocated, before it the margin can still grow to fit another reader.

## Striding without a kernel

Drop the kernel and the same machinery describes plain strided access. The term `Affine { stride; over; conv = None; stride_offset }` has size reading `input = stride · output` and index reading `stride · output_iterator + stride_offset` --- one loop variable scaled by the stride, plus an offset, no second iterator and no padding. This is the smallest affine index: a single coefficient and a constant. It is enough to express operations that have nothing to do with convolution.

Interleaving is the clearest example. Given two tensors of the same length, produce one twice as long, alternating their elements: the first tensor's values land in the even positions, the second's in the odd. In OCANNL it is two strided copies:

```
t =:+ id t1 ~logic:"…, i => …, 2*i";
t =+  id t2 ~logic:"…, i => …, 2*i + 1"
```

Each line copies a source cell `i` to a result cell at an affine position. The first writes to `2*i` --- stride two, offset zero. The second writes to `2*i + 1` --- stride two, offset one. The stride spaces the writes out; the offset is what makes the two copies miss each other and tile the result exactly. There is no kernel, no contraction, no padding; the whole content of the operation is in the two affine output indices, and they differ only in the constant the first section called `stride_offset`.

Two things are worth drawing out. First, the affine index here sits on the side written to, not the side read --- convolution addressed an operand, interleaving addresses the result. The form is indifferent to which: an index is an index, whether it places a read or a write. Second, the offset, which in a strided convolution merely chose a phase within each window and never mattered to a result, is here load-bearing: it is the entire difference between the even copy and the odd one. The same constant that was a projection-time afterthought for convolution is, for interleaving, the operation.

The inverses are the same form read the other way. Extracting the even-indexed elements is a copy from `2*i` to `i`; the odd elements, from `2*i + 1` to `i`. And the gradients are the inverses again: the gradient of the even-extraction scatters `i` back to `2*i`, returning the strided write the forward pass undid. A whole small family of layouts --- interleave, its two projections, their gradients --- is generated by one coefficient and one offset, varied.

## What the notation carries

The surface syntax is the einsum notation of the second post with one addition: a label position may hold an affine expression. An expression like `stride*oh+kh` triggers multi-character mode, where axes are comma-separated identifiers and `stride`, `dilation` may be integer literals or, under the syntax extensions, spliced-in integer variables.

```
"... | stride*oh<+kh, stride*ow<+kw, ic -> oc; kh, kw, ic -> oc => ... | oh, ow, oc"
```

is a 2-D convolution: spatial input axes as affine expressions, `ic` the contracted input channel, `oc` the output channel, `oh, ow, oc` the result. The `<` after the output label marks valid mode; written `=`, or omitted where `use_padding` is in scope, it marks padded mode. The notation is the affine index written down: input position equals `stride` times the output iterator plus the kernel iterator.

The product space stayed flat throughout: one factor per axis, each a single range. The construct that breaks this is concatenation, where several axes share one factor and iterate within it in sequence; that is the sequel. Two smaller pieces of OCANNL's indexing fit here first, because each reuses machinery the affine index already introduced.

---

# Part 2: Vectorized Operations and Initialization

A vectorized operation writes several output cells per step, the converse of the contraction's several reads per write. Initialization, where a tensor's cells are first filled, is where shapes are seeded from data. Where convolution enriched the *dimension term* with an affine constructor, these two enrich the *row constraint* instead --- with `Total_elems` and `Exact`, the forms that say how many elements a row has, or which axes it has exactly. The two enrichments are siblings, growing different parts of the same constraint language; they touch at one point, a shared notion of a strided count, reached below.

## A vectorized operation writes several cells per step

The map-reduce of the second post read one value per right-hand cell, combined them with a scalar function, and wrote one value per output cell. A vectorized operation breaks the last symmetry: it reads one cell and writes several. OCANNL has one such operation at present, the conversion from a 128-bit random word to a block of typed numbers, and its shape is the cleanest illustration of the idea.

The operation is `uint4x32_to_prec_uniform`. Its input is a tensor of `uint4x32` cells --- each a 128-bit value, the output of a counter-based random generator --- and its output is a tensor of floats (or other typed numbers) at some target precision, drawn uniformly. One 128-bit word yields as many output numbers as fit: sixteen bytes, eight half-precision floats, four single-precision, two double, one if the target is itself `uint4x32`. The count is `16 / prec_in_bytes(target)`. The vectorized step reads one input cell and writes that many output cells.

### Shape: an element-count relation

The output has more elements than the input, by exactly the precision ratio, and this is a relation across *rows* rather than within an axis. The convolution related one input dimension to one output dimension, `input_dim = stride · output_dim`; here the relation is between the operation's two ends as wholes: the output's total elements are `coeff · (input's total elements)`, with `coeff` the precision ratio. That needs a different constraint form than a dimension equation. The shape logic emits two. The input's rows are pinned to a single fresh variable, `Exact [Var v]` --- the operation assumes a single input axis, so as not to force minimum sizes on the output. The output's rows carry `Total_elems { numerator = Strided_var { coeff; var = v; denom = 1 } }`: the output has `coeff · v` elements in total, where `v` is the input's element count.

`Total_elems` is the row constraint that says "these rows, inclusive of their row variable, have this many elements," and `Strided_var` is its numerator when the count is not yet a literal but a coefficient times an unknown --- a placeholder for "the element count is `coeff · v`, for a `v` to be solved." This is the one point of contact with convolution promised above: the `coeff` of a `Strided_var` is a stride, the same kind of multiplicative coefficient the affine dimension carried, but here it scales a whole row's element count rather than a single axis. The constraint solver resolves `v` from whichever side is known first --- an input element count fixes the output's, or a required output count fixes the input's --- and the coefficient is forced at the stage that resolves precision-derived coefficients, the second post's stage 2. On the shape side a vectorized operation is a tensor whose two ends are related by a multiplicative element-count constraint --- the same row-constraint form a reshape uses, which we meet again under initialization.

### Projection: one read, several writes

On the projection side the operation iterates a product space as any operation does, but its assignment is not the map-reduce's. It lowers to a dedicated form, `Set_from_vec`, carrying the output index, the input read, the vector operation, and a `length` --- the number of output cells the one read produces. Where the second post's accumulation read several cells and wrote one, this writes `length` contiguous cells from one read.

The `length` is computed at lowering, from the same precision ratio the shape constraint used. The index maps are otherwise ordinary --- the product iterators substitute into the output and input indices as usual --- and the construct deliberately does not admit the coupled indices of the next post: a concatenated index has no meaning for a vectorized write, and the lowering rejects it. The vectorized operation adds no index form, only a fan-out at the leaf of the loop body, sized by the ratio that sized its shape.

### Surface

The conversion is a unary primitive, written `uint4x32_to_prec_uniform` in the `%cd` and `%op` notations, classified "dedicated" rather than pointwise because of the element-count change. It is rarely written directly. Its role is the last step of random initialization: a counter-based generator, `threefry4x32` (the binary primitive `^^^^`), produces `uint4x32` words, and this conversion unpacks them into uniform numbers at the tensor's precision. How those words are produced, and why the result is deterministic, we return to once initialization is on the table.

## Initialization seeds shapes from data

A tensor's cells must start somewhere. In OCANNL the starting value is a `Fetch` --- a reset of an array by a computation or a data read --- or a `Data` init carrying an array literal. Both are *terminal* shape logic: a leaf of the expression graph, with no sub-tensor to take a shape from, so whatever shape information exists must come from the initializer itself. This is the from-usage inference of the first post seen from its other end. There, a leaf's shape was driven toward the specific by its uses; here, an initializer can pin a leaf's shape directly from its data, and the two meet in the closing policy.

### Shape: pin exactly, or constrain the count

The terminal initializers split by how much they say about the shape, and the split is exactly the `Exact`-versus-`Total_elems` distinction.

An initializer that knows its axes emits `Exact`. `Keep_shape_no_padding` and `Padded` carry an array whose dimensions are taken verbatim, `Exact [d₁; d₂; …]`, pinning each axis. `Slice` --- taking one batch index of an existing tensor --- emits the sliced tensor's dimensions with the leading axis dropped, again `Exact`. These leave nothing to inference: the leaf's shape is the data's shape.

An initializer that knows only its size emits `Total_elems`. `Reshape` carries an array but lets its elements be laid into whatever axes inference assigns, so it constrains only the product: `Total_elems { numerator = Num_elems n }`, where `n` is the element count. `Constant_fill` --- a small array unrolled into the cells, the rightmost axis contiguous --- does the same. These are shape-polymorphic data: the same numbers fill a vector, a matrix, or a batch of either, as the surrounding computation decides, and `Total_elems` is what lets the count be honored while the axes stay free. It is the row constraint of the vectorized operation again, with a bare count for a numerator instead of a coefficient times a variable.

The remaining fetch-ops --- `Constant`, `Constant_bits`, `Range_over_offsets`, `Embed_symbol`, `Embed_dim`, `Embed_self_id` --- assert no shape at all; they mark the shape terminal and let the closing policy of the first post finish it, downward to the uses' greatest lower bound, or upward to the unit shape if nothing pins it. A constant fills whatever shape it lands in, exactly as the scalar that opened the first post did.

### Projection: the index is the data's layout

Initializers do not take part in the operation-level projection inference of the second post --- there is no contraction to infer, no operand to align. They lower directly to a loop over the array's own dimensions, writing each cell: `Constant` zeroes or fills, `Constant_fill` unrolls its literal, `Slice` reads the source at the fixed batch index prepended to the running indices.

One is worth singling out, because it is where an index becomes data. `Range_over_offsets` fills each cell with its own linear offset --- how many cells from the start it lies, in the tensor's logical layout. Computing that offset is the inverse of indexing: given the per-axis indices and the dimensions, collapse them to a single number by the row-major strides. OCANNL computes it by reflecting the projection --- folding the index array against the dimensions into a strided sum --- and embedding the result as the cell's value. The same arithmetic that an index map runs forward to address a cell is run backward here to label it. (This reflection does not yet handle the coupled indices of the next post; it is defined for the iterator, fixed, and affine forms of the first part, which is all an offset range needs.)

### Surface: literals, inline declarations, and the unit boundary

Two surface mechanisms introduce initialized tensors. The first is the literal. An array literal uses OCaml's list, tuple, and array syntax to separate the three axis kinds: a list `[ … ; … ]` for output axes, a tuple `( … , … )` for input axes, an array `[| … |]` for batch axes, nesting for higher rank. `[ (1, 2, 3); (4, 5, 6) ]` is a matrix taking three-dimensional input to two-dimensional output. A bare number is a scalar. An OCaml type annotation on an axis container names that axis's basis --- `([ 2.0 ] : rgb)` tags the output axis `rgb` --- the same dimension basis the first post used to keep incompatible axes from fusing. A literal written this way gets its exact shape from parsing; there is no variable for inference to resolve.

The second is the inline declaration, the record-brace syntax. `{ w }` introduces a tensor named `w` whose shape is left to inference; `{ w = expr }` supplies an initialization expression; further labelled fields supply output dimensions or other parameters, as in `{ b; o = [hid_dim] }`. Under `%op` a declaration is a parameter, differentiable, randomly initialized by default when no initialization expression is given. Under `%cd` it is a non-differentiable intermediate, self-referential, with no separate initializer. The distinction the first post drew at the closing policy --- a parameter errors if nothing pins its shape, an ordinary leaf guesses the unit --- is set here, at the declaration.

Where these declarations take effect is governed by the unit parameter the first post described. In a function `let layer = make () in …`, the code before the `()` runs once when `make ()` is evaluated and mints the parameter tensors; the body after it builds a fresh expression on each application, reusing those same parameter values. This is an OCaml-evaluation boundary: it decides when `Tensor.t` values are constructed, and so whether two applications share a parameter or each get their own. It says nothing about when any compiled code runs. Execution in OCANNL is explicit and separate --- a tensor's forward code, its initialization, and its gradient updates are each compiled to routines the caller runs by hand --- and the unit parameter is invisible to all of it. What the unit controls is the OCaml-level sharing of tensor values; what runs, and when, is a later and independent matter.

### What the tensor carries

A `Tensor.t` holds, besides its shape and its forward code, a set of tensors called its `params` --- the descendants whose own forward code is *not* folded into this tensor's forward code, because it is meant for initialization rather than recomputation. An ordinary subexpression is embedded: its forward code becomes part of the enclosing forward code and runs whenever the forward routine runs. A parameter is held out: its forward code --- a literal, or the random initializer built below --- is collected separately, and what the enclosing computation embeds is only a read of the parameter's array. Gathering that held-out code, transitively through parameters of parameters, produces an initialization computation, which is compiled and run as its own routine; a skip-check on what is already initialized keeps a parameter from being seeded twice across routines that share it. The `params` set is thus a statement about *where code goes* --- held out of the forward routine, gathered into the initialization routine --- not about when anything runs; the staging is explicit and in the caller's hands.

The set carries a second reading, the one a user of a model sees. The descendants that need initializing before the forward code can run are, read another way, the trainable parameters of the expression when it is taken to be a model. There is no separate notion of "the parameters of a model": a model in OCANNL is just a tensor, and its parameters are the members of its `params` set. The two meanings coincide because they are one fact about a leaf --- that it is held out of the forward code, its value supplied separately and then only read, rather than recomputed inline. Held out for initialization and exposed for training are the same property under two names.

### Random initialization, assembled from the parts

The default initializer for a parameter --- uniform random values --- is not a primitive. It is an expression built from pieces already on the table, and seeing how it is built shows where the determinism comes from. Stripped to its skeleton, the uniform generator is

```
uint4x32_to_prec_uniform
  (threefry4x32
     (threefry4x32 (embed_self_id ()) (random_seed ()))
     range_over_offsets)
```

Read it inside out. `embed_self_id` is the `Embed_self_id` fetch-op of the initialization subject: a leaf that fills its cell with the tensor's own integer id. `random_seed` is a global seed tensor, shared across the program. The inner `threefry4x32` --- the counter-based hash, the binary primitive `^^^^` --- combines the two, hashing the global seed under the tensor's id to produce a sub-key unique to that tensor. The outer `threefry4x32` combines that sub-key with `range_over_offsets`, the fetch-op that fills each cell with its own linear position: so each cell is hashed under its own offset, giving every cell of the tensor a distinct value from one sub-key. Finally `uint4x32_to_prec_uniform`, the vectorized operation of the first subject, unpacks each 128-bit hash into typed uniform numbers.

Every part is something the article has already introduced --- two leaf fetch-ops, the threefry primitive, the vectorized unpack --- composed into an ordinary tensor expression; nothing is special-cased. The determinism falls out of the composition. The key is split by the tensor's id and the counter is the cell's offset, so the values a parameter receives are a pure function of its identity and the global seed: no mutable random state threaded through the program, no seed argument passed down into the layers that build parameters. A tensor's id is already in scope --- it *is* the tensor --- so the splitting is implicit, and re-running with the same seed reproduces every cell exactly. Counter-based generation keyed by identity is what makes pseudorandomness both deterministic and argument-free.

## Closing

Three extensions, three reuses. The affine index is a function of loop variables, read from one term by two solvers and resolved in a second stratum. The vectorized operation is a fan-out at the loop leaf, sized by the same kind of count constraint a reshape uses. Initialization is from-usage inference run from the data end, pinning a leaf with `Exact` or constraining its count with `Total_elems`, lowering an index backward into data where an offset range needs it, and held out of the forward code in the tensor's `params` set --- the same leaves that are a model's trainable parameters. None adds a mechanism the first two posts did not already contain; each populates a corner the core left empty. Random initialization adds nothing further at all: it is those corners composed --- two leaf fetch-ops, a counter-based hash, and the vectorized unpack --- into an expression whose determinism comes from keying the hash on a tensor's identity, no state and no arguments threaded through. The construct that genuinely enlarges the machinery --- coupling several axes into one iteration factor, and reversing a gather into a scatter --- is concatenation, and it is the sequel.

OCANNL is open source, at [`github.com/ahrefs/ocannl`](https://github.com/ahrefs/ocannl).
