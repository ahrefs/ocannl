# Compilation to Cross-Backend Low-Level Representation, and Backend-Independent Optimizations

Computation in OCANNL is imperative. At the high-level, we store tensor node assignments as `Assignments.t`, which provides high-level operations like `Accum_op` (with `Ternop`/`Binop`/`Unop` right-hand sides), `Set_vec_unop`, and `Fetch`. This is translated to a low-level representation `Low_level.t` which is a C-like mini-language operating on scalars.

## Low-Level Representation

The `Low_level.t` type represents a C-like imperative language with for loops and scalar operations:

```ocaml
type t =
  | Noop
  | Comment of string
  | Staged_compilation of (unit -> PPrint.document)
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of Tn.t
  | Set of { tn : Tn.t; idcs : Indexing.axis_index array; llsc : scalar_t; mutable debug : string }
  | Set_from_vec of { tn : Tn.t; idcs : Indexing.axis_index array; length : int;
                      vec_unop : Ops.vec_unop; arg : scalar_arg; mutable debug : string }
  | Set_local of scope_id * scalar_t

and scalar_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get of Tn.t * Indexing.axis_index array
  | Get_merge_buffer of Tn.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_arg * scalar_arg * scalar_arg
  | Binop of Ops.binop * scalar_arg * scalar_arg
  | Unop of Ops.unop * scalar_arg
  | Constant of float
  | Constant_bits of int64
  | Embed_index of Indexing.axis_index

and scalar_arg = scalar_t * Ops.prec
```

`t` represents code/statements while `scalar_t` represents scalar expressions. The `trace_it` flag in `For_loop` indicates whether the loop should be traced for optimization (its initial segment will be unrolled for analysis).

### Index Types

The `Indexing.axis_index` type is central to understanding how array accesses work:

```ocaml
type axis_index =
  | Fixed_idx of int           (* A constant index *)
  | Iterator of symbol         (* A simple loop variable: symbol *)
  | Affine of { symbols : (int * symbol) list; offset : int }
      (* An affine expression: Σ(coeff_i * symbol_i) + offset *)
  | Sub_axis                   (* Part of a multi-axis vectorized access *)
```

`Affine` indices are crucial for convolutions: `symbols = [(stride, i1); (dilation, i2)]` with `offset = -padding`.

## Translation from Assignments

The translation `Assignments.to_low_level` converts high-level operations to low-level code:

1. **Projections to Loops**: `projections.product_space` elements become nested for loops with fresh loop index symbols
2. **Index Translation**: Tensor indices are derived from `projections.project_lhs` and `projections.project_rhs` with symbol substitution
3. **Operations**: High-level operations like `Accum_op` become loops over scalar operations
4. **Initialization**: If `initialize_neutral` is true and the projection isn't surjective+injective, we initialize with the neutral element

### Symbol Freshening During Lowering

An important detail: `projections.product_iterators` may be shared across different operations, so lowering creates **fresh symbols** for each loop. The substitution map tracks how product iterators map to fresh loop iterators, including handling `Affine` indices by substituting each symbol in the affine combination.

## Backend-Independent Optimizations

The optimization pipeline in `optimize_proc` consists of three main phases:

### 1. Tracing Phase (`visit_llc`)

This phase symbolically executes the computation to build a `traced_store` mapping each tensor node to a `traced_array`:

```ocaml
type traced_array = {
  tn : Tn.t;
  assignments : int array Hash_set.t;     (* Positions written to *)
  accesses : (int array, visits) Hashtbl.t; (* Positions read, with visit counts *)
  mutable zero_initialized_by_code : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
  mutable is_accessing : bool;            (* Does computation access non-constant arrays? *)
  mutable is_complex : bool;              (* Does computation involve non-trivial ops? *)
}
```

Additionally, a `reverse_node_map : (Symbol, Tnode) Hashtbl.t` tracks which tensor node's computation "owns" each loop symbol. This is used to associate for-loops with the tensor computations they belong to.

Key analyses performed:

- **Access Pattern Analysis**: Tracks which positions are read/written and how many times (`visits`)
- **Dependency Analysis**: Determines read-before-write patterns (recurrence)
- **Scalar Constant Expression Detection**: Identifies tensor nodes that are constant scalars
- **Complexity Classification**: Determines `is_accessing` (reads non-constant arrays) and `is_complex` (performs non-trivial operations on array accesses)
- **Memory Mode Inference**: Decides whether tensors should be virtual, materialized, etc.

#### Index Position Computation

For tracing, the `lookup` function converts symbolic indices to concrete integer positions:

```ocaml
let lookup env indices =
  Array.map indices ~f:(function
    | Fixed_idx i -> i
    | Sub_axis -> 0
    | Iterator s -> Option.value ~default:0 @@ Map.find env s
    | Affine { symbols; offset } ->
        List.fold symbols ~init:offset ~f:(fun acc (coeff, s) ->
            acc + (coeff * Option.value ~default:0 @@ Map.find env s)))
```

For `Affine` indices, this correctly computes the linear combination of symbol values.

### 2. Virtualization Phase (`virtual_llc` + `check_and_store_virtual`)

This phase determines which tensor computations can be inlined ("virtualized").

#### `virtual_llc`: Identifying Computation Boundaries

The `virtual_llc` function traverses the code and identifies the code blocks that define each tensor node's computation. It tracks a `process_for` set of tensor nodes whose computations are currently being traversed.

When encountering a `For_loop`, it checks `reverse_node_map` to see if this loop "belongs to" a tensor that hasn't been processed yet. If so, it marks this as the top-level computation for that tensor.

For each identified computation block, it calls `check_and_store_virtual` to determine if it can be inlined.

#### `check_and_store_virtual`: Validating Inlinability

This function performs several validation checks:

1. **Index Consistency** (`check_idcs`): All accesses to the tensor being virtualized must use the **same index pattern**.

2. **Symbol Uniqueness**: Each index position should use a unique dynamic symbol. The key restriction is at line 471:
   ```ocaml
   | Affine { symbols; offset = _ } -> (
       List.filter_map symbols ~f:(fun (_, s) -> ...)
       |> function
       | [] -> None
       | [ s ] -> Some s
       | _ -> failwith "check_idcs: multiple non-static symbols in affine index")
   ```
   Currently, an `Affine` index with **multiple non-static symbols** causes virtualization to fail. This restriction exists because the inlining mechanism assumes a 1:1 mapping between index positions and loop symbols.

3. **No Escaping Variables**: Dynamic symbols used in nested computations must be bound within the computation's scope (or be static indices).

4. **Non-Traced Loops Forbidden**: Loops with `trace_it = false` prevent virtualization.

5. **Has Setter**: The computation must actually write to the tensor.

If all checks pass, the computation (along with its defining indices) is stored in `computations_table` for later inlining.

#### Non_virtual Exit Codes

When virtualization fails, a `Non_virtual i` exception is raised with diagnostic codes:
- 4: Inconsistent index patterns between accesses
- 5: Symbol uniqueness violation (same symbol used multiple times)
- 51: Multiple non-static symbols in affine index (e.g., convolutions) — not yet supported
- 6: Non-traced loop encountered
- 7: Escaping variable in Set indices
- 8: Staged_compilation node encountered
- 9: Escaping variable in Get indices
- 10: Escaping variable in Embed_index
- 11: Tensor already marked non-virtual
- 12: No setter found
- 13: Index mismatch during inlining
- 14: Empty computation list
- 140: Vector operations cannot be inlined as scalar operations

### 3. Inlining Phase (`inline_computation`)

When a `Get` references a virtual tensor, `inline_computation` produces the inlined code:

```ocaml
let inline_computation ~id computations_table traced static_indices call_args =
  ...
  let make_subst i lhs_ind =
    let rhs_ind = call_args.(i) in
    match lhs_ind with
    | Iterator lhs_s when not (Set.mem static_indices lhs_s) -> Some (lhs_s, rhs_ind)
    | _ when equal_axis_index lhs_ind rhs_ind -> None
    | _ -> raise @@ Non_virtual 13
  in
  ...
```

#### Substitution with Affine Indices

The `subst` function handles symbol substitution, including for `Affine` indices:

```ocaml
let subst env idx =
  match idx with
  | Iterator s when Map.mem env s -> Map.find_exn env s
  | Affine { symbols; offset } ->
      let expand_symbol (coeff, s) =
        match Map.find env s with
        | Some (Iterator new_s) -> [ (coeff, new_s) ]
        | Some (Fixed_idx _) -> []  (* Contributes to offset *)
        | Some (Affine { symbols = inner; offset = _ }) ->
            List.map inner ~f:(fun (ic, is) -> (coeff * ic, is))
        | None -> [ (coeff, s) ]
      in
      let all_terms = List.concat_map symbols ~f:expand_symbol in
      let offset_additions = ... in  (* Handle Fixed_idx contributions *)
      Affine { symbols = all_terms; offset = offset + offset_additions }
  | idx -> idx
```

This handles:
- Simple symbol replacement
- `Fixed_idx` substitutions adding to the offset
- Nested `Affine` expansion (multiplying coefficients)

#### Loop Freshening

When inlining through a `For_loop`, fresh symbols are created to avoid capture:

```ocaml
| For_loop { index; body; _ } when Map.mem env index -> loop env body
| For_loop { index; from_; to_; body; trace_it } ->
    let fresh = Indexing.get_symbol () in
    let env = Map.add_exn ~key:index ~data:(Iterator fresh) env in
    Option.map ~f:(fun body -> For_loop { index = fresh; from_; to_; body; trace_it })
    @@ loop env body
```

### 4. Cleanup Phase (`cleanup_virtual_llc`)

After inlining, this phase:

1. **Removes virtualized computations**: For-loops and assignments to virtual tensors are eliminated
2. **Validates symbol scoping**: Asserts all `Iterator` symbols are in scope
3. **Finalizes memory modes**: Marks nodes as `Virtual` or `Never_virtual`

### 5. Simplification Phase (`simplify_llc`)

A traditional optimizing compiler pass:

- **Constant Folding**: `Constant 2.0 + Constant 3.0` → `Constant 5.0`
- **Algebraic Simplification**: `x + 0` → `x`, `x * 1` → `x`, etc.
- **Local Scope Elimination**: `Local_scope { body = Set_local(id, v) }` → `v`
- **Sequential Local Scopes**: Two consecutive `Set_local` to the same scope get substituted
- **Integer Power Unrolling**: `x ** 3` → `x * x * x` for small integer powers
- **FMA Detection**: `a + b * c` → `FMA(b, c, a)`

## Optimization Settings

The optimization behavior is controlled by `virtualize_settings`:

- `max_visits`: Maximum number of times a tensor can be accessed before being materialized (default: 1)
- `max_tracing_dim`: Maximum dimension size for loop unrolling during analysis (default: 5)
- `enable_device_only`: Whether to prefer device-only storage when possible
- `inline_scalar_constexprs`: Whether to inline scalar constant expressions regardless of accesses
- `inline_simple_computations`: Whether to inline computations built from single getters, or index embeddings and scalar constant expressions
- `inline_complex_computations`: Whether to inline complex computations (default: false, pending CSE implementation)

## Memory Mode Management

The optimization process works closely with OCANNL's memory mode system:

- **Virtual**: Computations are inlined, no storage allocated
- **Never_virtual**: Tensor must be stored (provenance int indicates why)
- **Materialized**: Tensor is stored and reused
- **Device_only**: Stored only on device, not accessible from host
- **Hosted**: Stored on both host and device

The optimizer uses provenance tracking (the `int` in memory mode updates) to debug conflicts in memory mode decisions.

## Current Limitations and Future Work

### Affine Index Limitations (Issue #133)

The current restriction at line 471 (`check_idcs: multiple non-static symbols in affine index`) prevents virtualization of convolutions. This happens because convolution indices are `Affine { symbols = [(stride, i); (dilation, j)]; offset = -padding }` with two dynamic symbols.

To support convolution inlining, several changes would be needed:

1. **Symbol Tracking**: Instead of requiring exactly one symbol per index position, track the full set of symbols used across all `Affine` indices.

2. **Substitution Map Construction**: `make_subst` needs to handle mapping multiple definition-site symbols to call-site expressions. This requires either:
   - Solving a system of equations (if call-site also uses `Affine`)
   - Direct symbol-to-symbol mapping if the affine structure matches

3. **Consistency Checking**: `check_idcs` should verify that all accesses use structurally equivalent `Affine` patterns (same coefficients, potentially different offsets).

4. **Loop Structure Preservation**: When inlining, the nested loop structure (iterating over both kernel and spatial dimensions) must be preserved with proper symbol freshening.

### Common Subexpression Elimination (Issue #351)

Currently `inline_complex_computations` defaults to false because inlining the same computation at multiple sites duplicates work. CSE would allow:
- Detecting when the same subexpression appears multiple times
- Computing it once and reusing the result
- Enabling safe inlining of complex computations

## Code Generation Integration

The optimized `Low_level.t` can be:

1. **Printed** using `to_doc` (OCANNL %cd syntax) or `to_doc_cstyle` (C-like syntax)
2. **Backend Compilation**: Each backend pattern-matches on `Low_level.t` to generate device-specific code
3. **Staged Compilation**: `Staged_compilation` nodes allow backends to embed generated code during optimization

The `Staged_compilation` construct is particularly important for backends that need to emit complex code patterns that can't be easily represented in the simple `Low_level.t` grammar.

This optimization pipeline enables OCANNL to achieve high performance by eliminating intermediate tensor allocations and generating specialized code for each computation pattern.
