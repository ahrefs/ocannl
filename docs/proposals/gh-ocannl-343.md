# Proposal: Optimize One-Hot Encoding Pattern for Embeddings

## Status update (2026-06-12)

- Issue #343 is OPEN, GitHub milestone v0.8 (mid-June 2026 per ROADMAP.md; #343 is not explicitly listed in ROADMAP.md's milestone breakdowns).
- Not started: there is no `Equality_with_index` in `assignments.ml`, no pattern detection in `simplify_llc`, and no `one_hot_virtual` anywhere. `fetch_op` has since gained `Constant_bits`, `Constant_fill`, `Slice`, `Embed_self_id`, and `Embed_dim` variants (`assignments.ml:24-41`).
- The motivating test `test/training/bigram_mlp.ml` no longer exists: it was rewritten as `test/training/mlp_names.ml` (makemore Part 2, commit 1046bf37), which still feeds the learned embedding table via one-hot context encodings (`fill_ctx_one_hot`); `test/training/bigram.ml` uses `Nn_blocks.one_hot_of_int_list` (lines 36-37). The acceptance criterion should target these files instead.
- New since the proposal: cross-statement CSE hoisting in `low_level.ml` introduced a `Declare_local` statement variant and runs *after* `simplify_llc` in `optimize_proc` (`simplify_llc` → `eliminate_common_subexpressions` → `hoist_cross_statement_cse`), so a Phase-2 matcher placed in `simplify_llc` sees the IR before hoisting — the accumulator `Local_scope` pattern described below is unaffected.
- The #377 GPT-2 inference proposal sidesteps this issue via single-token one-hot at inference time; #343 remains the fix for training-time / full-sequence embedding efficiency.
- Cited line numbers drifted slightly (e.g., `simplify_llc` now at `low_level.ml:1014`; `Embed_index` rendering now around `c_syntax.ml:658`); refreshed in place below as of 2026-06-12.
- The design (Approach C, virtual one-hot fetch op + simplifier pattern) remains valid; nothing landed invalidates it.
- **Scope note (2026-06-12)**: #186 (re-introducing dynamic indexing) has been
  retired as subsumed by this issue (see the resolution in
  [gh-ocannl-186.md](gh-ocannl-186.md)). #343 therefore *owns* the
  dynamic-index lowering construct (the low-level-only `Get_dynamic` sketched
  in Phase 3 / the design review below); general dynamic indexing as an
  `axis_index`/shape-inference feature is not coming back unless a scatter /
  top-k / KV-cache need reopens it.

## Goal

Eliminate the computational waste of materializing full one-hot matrices for embedding lookups. Currently, embedding lookup `C[X]` in OCANNL requires pre-computing one-hot vectors on the CPU, transferring the (much larger) one-hot matrix to the device, and performing a full matrix multiply -- O(vocab_size) work per token when O(1) suffices. This optimization adds a virtual one-hot mechanism via a new `Equality_with_index` fetch op that never materializes the sparse matrix, and adds a pattern-based optimization in `low_level.ml` that collapses the resulting for-loop into a direct row lookup.

GitHub issue: https://github.com/ahrefs/ocannl/issues/343

## Acceptance Criteria

- [ ] New `Equality_with_index` fetch op in `assignments.ml` that represents "1.0 if offset == embedded_index, else 0.0" without allocating a dense one-hot matrix
- [ ] `one_hot_virtual` function in `operation.ml` (or `nn_blocks.ml`) that creates a virtual one-hot tensor using `Equality_with_index`, taking only the index tensor and number of classes -- no Bigarray allocation
- [ ] Pattern detection in `low_level.ml` `simplify_llc`: when a for-loop accumulates `equality_result * table[k, ...]` over index `k`, replace with a direct lookup `table[index_value, ...]`, eliminating the loop entirely
- [ ] The makemore tests (`test/training/mlp_names.ml`, formerly `bigram_mlp.ml`, and/or `test/training/bigram.ml`) updated to use `one_hot_virtual` instead of pre-computed one-hot matrices, with identical training results (loss values match within float32 tolerance) *(Update 2026-06-12: `bigram_mlp.ml` was rewritten as `mlp_names.ml`)*
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

**Pre-computed one-hot** (historical: `test/training/bigram_mlp.ml` lines 13-24; *(Update 2026-06-12: this file was rewritten as `mlp_names.ml`, which fills one-hot context buffers via `fill_ctx_one_hot`/`fill_tgt_one_hot`, and `bigram.ml`, which uses `Nn_blocks.one_hot_of_int_list`)*):
```ocaml
let tensor_of_int_list lst =
  let genarray = Genarray.create Float32 c_layout [| len; dict_size |] in
  for i = 0 to len - 1 do
    Genarray.set genarray [| i; arr.(i) |] 1.
  done;
  TDSL.rebatch ~l:"tensor" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()
```

**FSM transformer** (`test/training/fsm_transformer.ml`, `seqs_to_flat_one_hot` at line 43): Same pattern, flat one-hot arrays.

**nn_blocks helper** (`lib/nn_blocks.ml` lines 61-76): `one_hot_of_int_list ~num_classes` -- pre-computes a dense Bigarray.

All of these create dense B*V float arrays on the CPU before any computation.

### Key code pointers

- **`fetch_op` type** (`arrayjit/lib/assignments.ml:24-41`): Where the new `Equality_with_index` variant goes. Current variants include `Range_over_offsets`, `Embed_symbol`, `Constant`, etc.
- **`to_low_level`** (`arrayjit/lib/assignments.ml:190+`): Where `fetch_op` variants are lowered to `Low_level.t` IR. The new variant's lowering generates `Cmpeq(Embed_index(offset), Get(index_tensor, batch_idcs))`.
- **`simplify_llc`** (`arrayjit/lib/low_level.ml:1014+`): The scalar simplification pass where the pattern detection goes. Already handles constant folding, FMA fusion, etc.
- **`axis_index` type** (`arrayjit/lib/indexing.ml:105-119`): Current index types: `Fixed_idx`, `Iterator`, `Affine`, `Sub_axis`, `Concat`. No changes needed for Phase 1.
- **`scalar_t` / `Embed_index`** (`arrayjit/lib/low_level.ml:53-66`): `Embed_index` converts an `axis_index` to a scalar float -- this is how `Range_over_offsets` puts index values into computations.
- **`c_syntax.ml` Embed_index** (`arrayjit/lib/c_syntax.ml:658`): How index-as-scalar renders in C code.

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

This is a structural pattern match on the `For_loop` node, not on scalars. It should be a separate function called from the top-level `loop_proc` in `simplify_llc`, matching on `For_loop` nodes specifically. *(Update 2026-06-12: better as a separate pass run **after** `simplify_llc` (and before `eliminate_common_subexpressions`). `simplify_llc` rewrites top-down, so a statement-level matcher inside its `loop_proc` would see the one-hot factor still wrapped as `Local_scope { body = Set_local (id, Cmpeq ...) }` — the collapse to a bare `Cmpeq` happens at `low_level.ml:1053` only while the enclosing scalar is being processed. A separate pass sees the already-collapsed form and stays independent of `simplify_llc`'s internal ordering.)*

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

The cleanest approach: add `Dynamic of scalar_t` to `axis_index` in `indexing.ml`, where `scalar_t` is `Low_level.scalar_t`. This variant renders in `c_syntax.ml` as `(int)(scalar_expr)`. *(Update 2026-06-12: not the cleanest — recommended against. `axis_index` is pervasive static-index machinery used by shape inference, projections, and `visit_llc`'s `lookup` (which evaluates indices to concrete ints for tracing); a data-dependent variant would also create an `indexing.ml` → `low_level.ml` reference inversion (`scalar_t` is defined in `low_level.ml`, which depends on `indexing.ml`). Prefer a `Low_level`-only scalar variant; see Design review.)*

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

*(Update 2026-06-12: as sketched this does not work — a bare `Tensor.term ~fetch_op` cannot "inherit" batch dims from `index_tensor` (no shape-inference link to it) and does not register `index_tensor` as a subtensor, so its forward code is not sequenced before the fetch. Compare `slice` (`tensor/operation.ml:617-630`), which wraps its `Fetch` in a `Tensor.unop` `op_asn` precisely so the source tensor is a real graph dependency with a dedicated `transpose_op` for shape inference. See Design review for the two viable variants.)*

Usage in `mlp_names.ml` (formerly `bigram_mlp.ml`):
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

## Design review (2026-06-12)

**Verdict: sound-with-changes.** Approach C is the right shape: the virtual one-hot is safe-by-fallback (if the rewrite doesn't fire, you still get a correct device-side `Cmpeq` loop — already better than host-side dense fill + transfer), and inlining genuinely produces the claimed pattern: `virtual_llc` inlines the fetch body as `Local_scope { Set_local (id, Cmpeq (Embed_index k', Get (index_tn, ...))) }`, which `simplify_llc:1053` collapses to a bare `Cmpeq` inside the reduction. Verified: `Cmpeq` exists in `ops.ml:390`; the `Slice` fetch op is precedent for a fetch reading another tensor; with default `inline_simple/complex_computations = true` and single-visit accesses the one-hot is virtualizable. Three parts need rework before implementation:

**Recommendations:**

1. **Pass placement** (see Phase 2 update note): make the matcher its own pass between `simplify_llc` and `eliminate_common_subexpressions` in `optimize_proc` (`low_level.ml:1631-1635`), so it sees the collapsed `Cmpeq` form. Inside `simplify_llc`'s top-down traversal it would not fire.
2. **Phase 3 representation**: keep `indexing.ml` untouched (this was the selling point of Approach C, and matches the original issue instinct "no changes outside low_level.ml"). Add a `Low_level`-only scalar variant, e.g. `Get_dynamic of { tn : Tn.t; idcs : Indexing.axis_index array; dyn_axis : int; dyn_val : scalar_arg }`, rendered by splicing `(int)(dyn_val)` into `pp_array_offset` at position `dyn_axis`. It is introduced only post-tracing, so only the later passes need cases: `cse_equal_scalar`, the hoisting pass's read collection, the doc printer, and `c_syntax.ml`'s `pp_scalar`. Caution: `tn.dims` are *padded* dims and padding is encoded via `Affine` offsets in projections — restrict the first iteration to `k` occurring as a pure `Iterator` (bail on `Affine`/`Concat`), which also correctly excludes conv-style einsums where the substitution would be wrong.
3. **State the matcher's side conditions explicitly** (currently implicit): (a) `index_expr` must not mention `k` — including inside nested `Local_scope` bodies and `Affine` symbol lists; (b) `k` appears exactly once in the table indices, as `Iterator k`; (c) the loop body is exactly the single accumulating assignment; (d) accumulation is `Add` with `Mul` (match `Cmpeq`/`Mul` operands commutatively; skip tropical `@^+`/`einmax` reductions for now); (e) the loop range `[from_, to_]` spans the full class axis. Also handle **both accumulator forms**: the `Local_scope` form described here (virtual result) and the materialized form `Zero_out lhs; For k { Set (lhs, idcs, Add (Get (lhs, idcs), Mul (...))) }` produced by `to_low_level` when the result is not virtualized — check `build_files/*.ll` for `mlp_names.ml` to see which one actually occurs and prioritize it.
4. **Phase 4 API**: choose between (a) the `slice` pattern — `Tensor.unop` with the `Fetch` in `op_asn` plus a new `transpose_op`/shape rule introducing the class axis (honest cost: one new case in `shape.ml`, contradicting the "no new shape inference rules" claim, but required when the index tensor is itself computed, e.g. #377 autoregressive sampling); or (b) a first iteration that takes explicit dims at the call site and requires a hosted/materialized index tensor (sufficient for `mlp_names.ml`, `bigram.ml`, `fsm_transformer.ml`, whose index data is host-filled anyway). (b) first, (a) as follow-up is a reasonable sequencing.
5. **OOB and precision**: the rewrite changes out-of-range-index semantics from "contributes 0" to an out-of-bounds memory read — decide on a policy (documented precondition, clamp, or config-gated bounds check) as part of acceptance, not as a leftover risk. Pin the index tensor precision in `one_hot_virtual` (reject fp16: exact integers only up to 2048, below realistic vocab sizes; fp32's 2^24 is fine).

**Open decision points for Łukasz:**
- `Get_dynamic` low-level-only variant vs. extending `axis_index` (review strongly favors the former).
- Phase 4: new shape-inference rule now (unop, like `slice`) vs. explicit-dims first iteration.
- OOB policy for the optimized gather (document / clamp / debug-mode check).
- Keep the dense backward fallback (as proposed — agreed) or scope a scatter-add pattern into this issue.

**Alternative considered**: a dedicated `embedding`/gather primitive (Approach B) would be more robust than a structural rewrite, but requires the same dynamic-index codegen *plus* new shape inference and backprop definitions, and loses the free composability with arbitrary einsum specs. The fetch-op + rewrite design is the better first step; if the matcher proves fragile in practice, the `Get_dynamic` codegen built here is exactly the substrate a future gather primitive would lower to, so no work is wasted.
