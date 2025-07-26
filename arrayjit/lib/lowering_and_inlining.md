# Compilation to Cross-Backend Low-Level Representation, and Backend-Independent Optimizations

Computation in OCANNL is imperative. At the high-level, we store tensor node assignments as `Assignments.t`, which provides high-level operations like `Accum_binop`, `Accum_unop`, and `Fetch`. This is translated to a low-level representation `Low_level.t` which is a C-like mini-language operating on scalars.

## Low-Level Representation

The `Low_level.t` type represents a C-like imperative language with for loops and scalar operations:

```ocaml
type t =
  | Noop
  | Comment of string
  | Staged_compilation of (unit -> PPrint.document)
  | Seq of t * t
  | For_loop of { index : Indexing.symbol; from_ : int; to_ : int; body : t; trace_it : bool }
  | Zero_out of Tnode.t
  | Set of { array : Tnode.t; idcs : Indexing.axis_index array; llsc : scalar_t; mutable debug : string }
  | Set_local of scope_id * scalar_t

and scalar_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Access of dedicated_access * Indexing.axis_index array option
  | Get of Tnode.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_t * scalar_t * scalar_t
  | Binop of Ops.binop * scalar_t * scalar_t
  | Unop of Ops.unop * scalar_t
  | Constant of float
  | Embed_index of Indexing.axis_index
```

`t` represents code/statements while `scalar_t` represents scalar expressions. The `trace_it` flag in `For_loop` indicates whether the loop should be traced for optimization (its initial segment will be unrolled for analysis).

## Translation from Assignments

The translation `Assignments.to_low_level` is straightforward:

1. **Projections to Loops**: `projections.product_space` elements become nested for loops
2. **Index Translation**: Tensor indices are derived from `projections.project_lhs` and `projections.project_rhs`
3. **Operations**: High-level operations like `Accum_binop` become loops over scalar operations
4. **Initialization**: If `initialize_neutral` is true and the operation isn't total, we initialize with the neutral element

## Backend-Independent Optimizations

The optimization pipeline in `optimize_proc` consists of three main phases:

### 1. Tracing Phase (`visit_llc`)

This phase symbolically executes the computation to build a `traced_store` mapping each tensor node to a `traced_array`:

```ocaml
type traced_array = {
  tn : Tn.t;
  mutable computations : (Indexing.axis_index array option * t) list;
  assignments : int array Hash_set.t;
  accesses : (int array, visits) Hashtbl.t;
  mutable zero_initialized : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
}
```

Key analyses performed:

- **Access Pattern Analysis**: Tracks which positions are read/written and how many times (`visits`)
- **Dependency Analysis**: Determines read-before-write patterns (recurrence)
- **Scalar Constant Expression Detection**: Identifies tensor nodes that are constant scalars
- **Memory Mode Inference**: Decides whether tensors should be virtual, materialized, etc.

### 2. Virtualization and Inlining Phase (`virtual_llc`)

This is the core optimization phase that implements **computation inlining**:

#### Virtualization Decision

- Tensors with too many accesses (`> max_visits`) are marked `Never_virtual`
- Read-only tensors are typically materialized
- Recurrent tensors (read-before-write) are materialized

#### Inlining Process (`inline_computation`)

When a tensor node is accessed via `Get`, if it's determined to be virtual:

1. **Retrieve Computations**: Get the stored computations for the tensor from `traced_array.computations`
2. **Symbol Freshening**: Create fresh symbols to avoid variable capture when inlining
3. **Substitution**: Replace the definition's indices with the call site's indices
4. **Code Generation**: Generate a `Local_scope` that computes the value inline

#### Critical Invariant: Symbol Freshening

When inlining, we must ensure that loop variables don't clash. The `subst` function handles index substitution, mapping old symbols to new ones. This is crucial for correctness.

### 3. Cleanup and Simplification Phase (`cleanup_virtual_llc` + `simplify_llc`)

#### Cleanup (`cleanup_virtual_llc`)

- **Environment Validation**: Ensures all symbols are properly bound in their scope
- **Virtual Tensor Removal**: Removes references to virtual tensors that were successfully inlined
- **Constraint Checking**: Validates that symbol substitution was correct

#### Simplification (`simplify_llc`)

A traditional optimizing compiler pass that performs:

- **Constant Folding**: `Constant 2.0 + Constant 3.0` → `Constant 5.0`
- **Algebraic Simplification**: `x + 0` → `x`, `x * 1` → `x`, etc.
- **Dead Code Elimination**: Removes `Local_scope` that just return values
- **Integer Power Unrolling**: `x ** 3` → `x * x * x` for small integer powers

## Optimization Settings

The optimization behavior is controlled by `virtualize_settings`:

- `max_visits`: Maximum number of times a tensor can be accessed before being materialized
- `max_tracing_dim`: Maximum dimension size for loop unrolling during analysis
- `enable_device_only`: Whether to prefer device-only storage when possible
- `inline_scalar_constexprs`: Whether to inline scalar constant expressions regardless of accesses
- `inline_simple_computations`: Currently, whether to inline computations built from index embeddings and scalar constant expressions, regardless of accesses

## Memory Mode Management

The optimization process works closely with OCANNL's memory mode system:

- **Virtual**: Computations are inlined, no storage allocated
- **Materialized**: Tensor is stored and reused
- **Device_only**: Stored only on device, not accessible from host
- **Hosted**: Stored on both host and device

The optimizer uses provenance tracking (the `int` in memory mode tuples) to debug conflicts in memory mode decisions.

## Code Generation Integration

The optimized `Low_level.t` can be:

1. **Printed** using `to_doc` (OCANNL %cd syntax) or `to_doc_cstyle` (C-like syntax)
2. **Backend Compilation**: Each backend pattern-matches on `Low_level.t` to generate device-specific code
3. **Staged Compilation**: `Staged_compilation` nodes allow backends to embed generated code during optimization

The `Staged_compilation` construct is particularly important for backends that need to emit complex code patterns that can't be easily represented in the simple `Low_level.t` grammar.

This optimization pipeline enables OCANNL to achieve high performance by eliminating intermediate tensor allocations and generating specialized code for each computation pattern.
