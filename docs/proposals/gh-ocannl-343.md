# Proposal: Optimize One-Hot Encoding Pattern for Embeddings

## Goal

Eliminate the computational waste of materializing full one-hot matrices for embedding lookups. Currently, embedding lookup `C[X]` in OCANNL requires pre-computing one-hot vectors on the CPU, transferring the (much larger) one-hot matrix to the device, and performing a full matrix multiply -- O(vocab_size) work per token when O(1) suffices. This optimization adds a virtual one-hot mechanism via a new `Equality_with_index` fetch op that never materializes the sparse matrix, and adds a pattern-based optimization in `low_level.ml` that collapses the resulting for-loop into a direct row lookup.

GitHub issue: https://github.com/ahrefs/ocannl/issues/343

## Acceptance Criteria

- [ ] New `Equality_with_index` fetch op in `assignments.ml` that represents "1.0 if offset == embedded_index, else 0.0" without allocating a dense one-hot matrix
- [ ] `one_hot_virtual` function in `operation.ml` (or `nn_blocks.ml`) that creates a virtual one-hot tensor using `Equality_with_index`, taking only the index tensor and number of classes -- no Bigarray allocation
- [ ] Pattern detection in `low_level.ml` `simplify_llc`: when a for-loop accumulates `equality_result * table[k, ...]` over index `k`, replace with a direct lookup `table[index_value, ...]`, eliminating the loop entirely
- [ ] The `bigram_mlp.ml` test updated to use `one_hot_virtual` instead of pre-computed one-hot matrices, with identical training results (loss values match within float32 tolerance)
- [ ] Memory usage for the embedding input is proportional to batch_size (one int-as-float per token), not batch_size * vocab_size
- [ ] Backward pass works correctly: gradient through the virtual one-hot uses the same pattern in reverse (the one-hot transpose backward is acceptable for the first implementation, since that matches current behavior)
- [ ] No regression in existing tests
- [ ] The optimization fires and is observable: generated C code for the optimized path contains a direct array index (cast-to-int of a float value) rather than a reduction loop over vocab_size

## Context

### The one-hot inefficiency

For a vocabulary of size V and batch of size B with embedding dimension D:
- **Current one-hot approach**: Allocates B*V floats (e.g., 50K vocab * 512 batch = 25M floats = 100MB), most are zeros. Then does B*V*D multiplies.
- **After optimization**: Index tensor is B floats. The for-loop over V with conditional multiply is replaced by a direct lookup. In the generated code: `result[batch][dim] = table[(int)index[batch]][dim]`.

### Current embedding patterns in OCANNL

**Pre-computed one-hot** (`test/training/bigram_mlp.ml` lines 13-24):
```ocaml
let tensor_of_int_list lst =
  let genarray = Genarray.create Float32 c_layout [| len; dict_size |] in
  for i = 0 to len - 1 do
    Genarray.set genarray [| i; arr.(i) |] 1.
  done;
  TDSL.rebatch ~l:"tensor" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()
```

**FSM transformer** (`test/training/fsm_transformer.ml` lines 51-59): Same pattern, flat one-hot arrays.

**nn_blocks helper** (`lib/nn_blocks.ml` lines 61-76): `one_hot_of_int_list ~num_classes` -- pre-computes a dense Bigarray.

All of these create dense B*V float arrays on the CPU before any computation.

### Key code pointers

- **`fetch_op` type** (`arrayjit/lib/assignments.ml:24-41`): Where the new `Equality_with_index` variant goes. Current variants include `Range_over_offsets`, `Embed_symbol`, `Constant`, etc.
- **`to_low_level`** (`arrayjit/lib/assignments.ml:190+`): Where `fetch_op` variants are lowered to `Low_level.t` IR. The new variant's lowering generates `Cmpeq(Embed_index(offset), Get(index_tensor, batch_idcs))`.
- **`simplify_llc`** (`arrayjit/lib/low_level.ml:1006-1214`): The scalar simplification pass where the pattern detection goes. Already handles constant folding, FMA fusion, etc.
- **`axis_index` type** (`arrayjit/lib/indexing.ml:104-120`): Current index types: `Fixed_idx`, `Iterator`, `Affine`, `Sub_axis`, `Concat`. No changes needed for Phase 1.
- **`scalar_t` / `Embed_index`** (`arrayjit/lib/low_level.ml:52-63`): `Embed_index` converts an `axis_index` to a scalar float -- this is how `Range_over_offsets` puts index values into computations.
- **`c_syntax.ml` Embed_index** (`arrayjit/lib/c_syntax.ml:572-578`): How index-as-scalar renders in C code.

### How the IR looks today (one-hot * embedding matrix)

When the einsum `{ w_embed } * input` is lowered with a one-hot `input` tensor, the low-level IR is approximately:

```
For_loop { index = k; from_ = 0; to_ = vocab_size - 1;
  body = Set { tn = result; idcs = [batch; dim];
    llsc = Binop(Add,
      Get_local acc,
      Binop(Mul,
        Get(one_hot_input, [batch; k]),      (* 0.0 or 1.0 *)
        Get(embedding_table, [k; dim])))     (* table row *)
  }
}
```

### How the IR will look with virtual one-hot

With `Equality_with_index`, the one-hot tensor is instead initialized via:
```
Fetch { array = virtual_onehot; fetch_op = Equality_with_index { index_tensor }; dims = ... }
```

Which lowers to:
```
For_loop { index = k; body =
  Set { tn = virtual_onehot; idcs = [batch; k];
    llsc = Binop(Cmpeq,
      Embed_index(Iterator k),           (* the class index *)
      Get(index_tensor, [batch]))        (* the index value *)
  }
}
```

After einsum with the embedding table, the reduction loop body becomes:
```
For_loop { index = k;
  body = Set { tn = result; idcs = [batch; dim];
    llsc = Binop(Add, Get_local acc,
      Binop(Mul,
        Binop(Cmpeq, Embed_index(Iterator k), Get(index_tensor, [batch_idx])),
        Get(embedding_table, [k; dim])))
  }
}
```

The simplification pass detects this pattern and replaces it with:
```
Set { tn = result; idcs = [batch; dim];
  llsc = Get(embedding_table, [cast_to_int(Get(index_tensor, [batch_idx])); dim])
}
```

This eliminates the for-loop entirely.

### Design choice: why `Equality_with_index` fetch op

lukstafi considered three approaches in the GitHub discussion:

1. **Low-level pattern detection only** (Approach A): Fragile, requires dynamic indexing in `indexing.ml`.
2. **High-level `embedding` operation** (Approach B): Clean but requires new shape inference support.
3. **Virtual one-hot / `Equality_with_index`** (Approach C): The selected approach. Reasons:
   - No new `axis_index` variant needed (no `Dynamic_idx` in `indexing.ml`)
   - No new shape inference rules
   - Works with existing einsum -- the virtual one-hot participates in standard matrix multiply projections
   - The optimization is a local pattern in `simplify_llc`, not a global restructuring
   - Backward pass works automatically: the gradient of `w * one_hot` with respect to `w` uses the one-hot as-is (transposed), which is correct whether the one-hot is real or virtual

The key insight: instead of adding dynamic indexing to the index system (which complicates shape inference and optimization), we keep indices static and add a fetch op that generates equality comparisons. The simplifier then recognizes that "sum over k of (k == index) * table[k]" is just "table[index]".

## Approach

### Phase 1: `Equality_with_index` fetch op and lowering

**`arrayjit/lib/assignments.ml`**: Add to `fetch_op`:
```ocaml
| Equality_with_index of { index_tn : Tn.t }
    (** Virtual one-hot: cell value is 1.0 if offset along the last axis equals
        the value in [index_tn] at the corresponding batch position, else 0.0.
        Shape: [index_tn.dims...; num_classes]. *)
```

**`assignments.ml` `to_low_level`**: Add lowering case. The tensor has dims `[..batch_dims; num_classes]`. The index tensor has dims `[..batch_dims]`. For each cell, compare the last-axis offset against the index tensor value at the batch position:
```ocaml
| Fetch { array; fetch_op = Equality_with_index { index_tn }; dims = (lazy dims) } ->
    let num_classes = dims.(Array.length dims - 1) in
    let batch_dims = Array.sub dims ~pos:0 ~len:(Array.length dims - 1) in
    default_padding_before array
    @@ Low_level.loop_over_dims dims ~body:(fun idcs ->
        let batch_idcs = Array.sub idcs ~pos:0 ~len:(Array.length batch_dims) in
        let class_idx = idcs.(Array.length idcs - 1) in
        set array idcs @@
          mk_binop Cmpeq
            (Embed_index class_idx)
            (Get (index_tn, batch_idcs)))
```

### Phase 2: Pattern detection in `simplify_llc`

Add a new pass (or extend the existing `simplify_llc`) that recognizes the embedding lookup pattern in the low-level IR. The pattern to detect:

```
For_loop { index = k; from_ = 0; to_ = N-1;
  body = Set { tn = result; idcs = result_idcs;
    llsc = Binop(Add, Get_local acc,
      Binop(Mul,
        (Cmpeq(Embed_index(Iterator k), index_expr), _),
        (Get(table, table_idcs_containing_k), _)))
  }
}
```

Where `k` appears exactly once in `table_idcs` as `Iterator k`. Replace with:
```
Set { tn = result; idcs = result_idcs;
  llsc = Get(table, table_idcs_with_k_replaced_by_index_expr)
}
```

This is a structural pattern match on the `For_loop` node, not on scalars. It should be a separate function called from the top-level `loop_proc` in `simplify_llc`, matching on `For_loop` nodes specifically.

**Important**: the accumulation variable (`Get_local acc` / `Set_local`) wrapping must be handled. The pattern occurs inside a `Local_scope` that initializes the accumulator to 0.0 and accumulates with Add. The full pattern including the scope is:
```
Local_scope { id; body =
  Set_local(id, Constant 0.0);  (* init *)
  For_loop { index = k; ... body =
    Set_local(id, Binop(Add, Get_local id,
      Binop(Mul, cmpeq_pattern, table_get)))
  }
}
```

After optimization, this becomes:
```
Get(table, optimized_idcs)
```

(The entire Local_scope is replaced.)

### Phase 3: Dynamic index in generated code

The optimized IR contains `Get(table, [...; index_expr; ...])` where `index_expr` is something like `Get(index_tensor, batch_idcs)` -- a float value that needs to be cast to an integer index. This is a new pattern for `c_syntax.ml`: an axis index that is itself a scalar expression rather than a loop variable or constant.

Options:
- **Float-to-int cast in index position**: The simplifier can produce a `Get` where one index position is derived from a scalar expression. This requires either:
  - A new `axis_index` variant `Dynamic of Tn.t * axis_index array` in `indexing.ml`, or
  - Keeping the scalar expression as a `Get` nested inside the index computation.

The cleanest approach: add `Dynamic of scalar_t` to `axis_index` in `indexing.ml`, where `scalar_t` is `Low_level.scalar_t`. This variant renders in `c_syntax.ml` as `(int)(scalar_expr)`.

However, to minimize changes in Phase 1, an alternative: the simplifier can rewrite the pattern to use a fresh for-loop-free `Set` with the index tensor value embedded via `Embed_index` of a new `Dynamic_lookup` axis_index variant:
```ocaml
| Dynamic_lookup of { tn : Tn.t; idcs : axis_index array }
    (** Index value is the integer conversion of the float at tn[idcs]. *)
```

This is the minimal addition to `indexing.ml` needed. It renders in c_syntax as `(int)(buffer[offset])`.

### Phase 4: User-facing API

**`lib/nn_blocks.ml`** or **`tensor/operation.ml`**: Add `one_hot_virtual`:
```ocaml
let one_hot_virtual ~num_classes index_tensor =
  Tensor.term ~fetch_op:(Equality_with_index { index_tn = index_tensor.Tensor.value })
    ~grad_spec:Prohibit_grad
    ~batch_dims:(* inherit from index_tensor *)
    ~input_dims:[ num_classes ] ~output_dims:[] ()
```

The function creates a virtual one-hot tensor that can be used in einsum expressions identically to a real one-hot tensor, but never materializes the dense matrix. When multiplied by an embedding table via einsum, the low-level optimizer collapses it to a direct lookup.

Usage in `bigram_mlp.ml`:
```ocaml
(* Before: *)
let inputs = tensor_of_int_list int_input in  (* allocates batch*vocab_size floats *)
(* After: *)
let index_tensor = (* batch-sized tensor of integer-as-float values *) in
let inputs = one_hot_virtual ~num_classes:dict_size index_tensor in
```

### Phase 5: Backward pass

The gradient of `w * one_hot` with respect to `w` is `grad_output * one_hot^T`. With virtual one-hot, this is:
- `grad_w[k, d] += grad_output[batch, d] * cmpeq(k, index[batch])`

This is the standard dense backward pass -- O(B * V * D). It is no worse than the current approach (the one-hot transpose multiply has the same complexity). Sparse gradient (scatter-add) is a follow-up optimization.

The gradient with respect to the index tensor is zero (indices are not differentiable), which is handled by `grad_spec:Prohibit_grad` on the virtual one-hot tensor.

### Risks and edge cases

- **Float-to-int precision**: Index values stored as floats (e.g., 42.0) must cast cleanly to integers. For vocab sizes up to ~16M, float32 represents all integers exactly. GPT-2's 50257 is well within range.
- **Out-of-bounds indices**: If `index_tensor` contains a value >= num_classes or < 0, the direct lookup accesses out-of-bounds memory. The `Equality_with_index` lowering naturally produces 0.0 for all classes (safe but silent). The optimized direct-lookup path must add bounds checking or document that indices must be valid.
- **Pattern fragility**: The simplifier pattern match is structural. If the einsum lowering changes the IR shape (e.g., different accumulator pattern, different operation order), the pattern won't fire and the code falls back to the unoptimized but correct equality-comparison loop. This is a safe fallback.
- **Interaction with virtualization/inlining**: The `Equality_with_index` tensor may be virtualized (inlined) by `low_level.ml`'s tracing pass, which would change the IR structure before `simplify_llc` runs. The pattern detection must account for both inlined and non-inlined forms, or the optimization should run before virtualization.
