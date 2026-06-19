# Compilation to Cross-Backend Low-Level Representation, and Backend-Independent Optimizations

Computation in OCANNL is imperative. At the high level, we store tensor node assignments as
`Assignments.t`, which provides high-level operations like `Accum_op` (with `Ternop`/`Binop`/`Unop`/
`Block`/`Rev_sides` right-hand sides), `Set_vec_unop`, and `Fetch`. This is translated ("lowered")
to a low-level representation `Low_level.t`, a C-like mini-language operating on scalars, and then
run through a sequence of backend-independent optimization passes before backend code generation.

This document describes the lowering and the optimization pipeline at a conceptual, algorithmic
level. It deliberately avoids source line numbers (which drift); everything is referenced by
function, phase, constructor, and exception-code name, so the description stays valid as the file
evolves.

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
  | Declare_local of { id : scope_id; needs_init : bool }

and scalar_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get of Tn.t * Indexing.axis_index array
  | Get_dynamic of { tn : Tn.t; idcs : Indexing.axis_index array;
                     dyn_axis : int; dyn_value : scalar_arg }
  | Get_merge_buffer of Tn.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_arg * scalar_arg * scalar_arg
  | Binop of Ops.binop * scalar_arg * scalar_arg
  | Unop of Ops.unop * scalar_arg
  | Constant of float
  | Constant_bits of int64
  | Embed_index of Indexing.axis_index

and scalar_arg = scalar_t * Ops.prec
```

`t` represents code/statements while `scalar_t` represents scalar expressions. The `trace_it` flag in
`For_loop` indicates whether the loop should be traced for optimization (its initial segment is
unrolled for analysis).

Notable constructors:

- **`Set` / `Set_from_vec`**: scalar and vectorized stores. `Set_from_vec` applies a `vec_unop`
  (e.g. a `threefry`-style fill) producing `length` consecutive lanes from a single scalar `arg`;
  it cannot be inlined as a scalar computation (see the virtualization phase).
- **`Set_local` / `Get_local` / `Local_scope`**: the machinery of inlining. A `Local_scope` packages
  a virtual tensor's computation `body` together with the `orig_indices` at which it is accessed; its
  result is read back via `Get_local`. `Set_local` writes the scope's accumulator.
- **`Declare_local { id; needs_init }`**: declares a local accumulator hoisted out of individual
  statements. It is produced **only** by `hoist_cross_statement_cse` (the last pipeline phase);
  fresh lowering from `Assignments.to_low_level` never emits it. `needs_init` records whether the
  hoisted local needs a zero initializer before its first use.
- **`Get_dynamic`**: a guarded dynamic gather (reads `tn`'s row at a runtime-computed `dyn_value`
  along `dyn_axis`). It is produced **only** by `rewrite_one_hot_reductions` (gh-343), after
  virtualization; earlier passes handle it defensively.
- **`Constant_bits`**: a direct 64-bit pattern, primarily for `uint4x32` values where a `float`
  literal would be lossy.
- **`Get_merge_buffer`**: reads from the single per-routine "merge buffer" used for cross-stream
  reductions (see below).

### Scope identifiers

Local scopes are identified by a `scope_id` record `{ tn : Tn.t; scope_id : int }`. The `get_scope`
helper allocates a globally fresh integer id for each `Local_scope`/`Declare_local`, tied to the
tensor node `tn` whose computation the scope inlines. This keeps distinct inlinings of the same
tensor node distinguishable during substitution and CSE.

### Index Types

The `Indexing.axis_index` type is central to understanding how array accesses work:

```ocaml
type axis_index =
  | Fixed_idx of int           (* A constant index *)
  | Iterator of symbol         (* A simple loop variable: symbol *)
  | Affine of { symbols : (int * symbol) list; offset : int }
      (* An affine expression: Σ(coeff_i * symbol_i) + offset *)
  | Sub_axis                   (* Part of a multi-axis vectorized access *)
  | Concat of symbol list
      (** This axis is formed by concatenating multiple axes, each represented by an iterator
          symbol. [Concat] indices are eliminated during lowering. *)
```

`Affine` indices are crucial for convolutions: `symbols = [(stride, i1); (dilation, i2)]` with
`offset = -padding`.

`Concat` indices arise in tensor concatenation and block-tensor construction. They are only intended
for assignments of the `Block` or `Rev_sides` variety (variants of `Assignments.accum_rhs`), and are
eliminated during lowering — so they must never reach virtualization (see `Non_virtual 52`).

## Translation from Assignments

The translation `Assignments.to_low_level` converts high-level operations to low-level code:

1. **Projections to Loops**: `projections.product_space` elements become nested for loops with fresh
   loop index symbols; elements in the same list become loops sequenced after each other.
2. **Index Translation**: Tensor indices are derived from `projections.project_lhs` and
   `projections.project_rhs` with symbol substitution.
3. **Operations**: High-level operations like `Accum_op` become loops over scalar operations.
   `Set_vec_unop` lowers to `Set_from_vec`.
4. **Initialization**: If `initialize_neutral` is true and the projection isn't surjective+injective,
   we initialize with the neutral element.

### Symbol Freshening During Lowering

An important detail: `projections.product_iterators` may be shared across different operations, so
lowering creates **fresh symbols** for each loop. The substitution map tracks how product iterators
map to fresh loop iterators, including handling `Affine` indices by substituting each symbol in the
affine combination.

### Converting Concatenation to Sequencing

When the elements of `projections.product_space` and `projections.product_iterators` being processed
are lists of more than one element, we generate one (nested) loop for each as usual, and put the
loops in sequence (preserving the list order). We remember which of the components was picked for the
given loop in the recursive call. When we get down to the base case, we select the specific buffer
out of `Block`, so that the projection of this buffer (i.e. the `project_rhs` at the same position
as the buffer) has only symbols that were picked when descending this path of the nested loop
recursive calls. With the RHS projection and the RHS buffer thus selected, we lower the assignment as
if it were a unary assignment with identity as the unary operation. We emit `Noop` if either the LHS
is invalid, or no RHS can be selected. If multiple RHS buffers are valid, we raise a user error about
block tensor operation ambiguity.

## Backend-Independent Optimizations

`optimize_proc` runs the full pipeline. After tracing builds analysis state, the code is transformed
by a chain of passes:

```
visit_llc            (tracing: builds traced_store + reverse_node_map)
  ↓
virtual_llc          (identify + validate inlinable computations)
  ↓
cleanup_virtual_llc  (remove virtualized writes, finalize memory modes)
  ↓
simplify_llc         (constant folding, algebraic simplification, FMA, ...)
  ↓
rewrite_one_hot_reductions          (one-hot select-and-reduce → guarded Get_dynamic gather)
  ↓
eliminate_common_subexpressions     (within-statement scalar CSE)
  ↓
hoist_cross_statement_cse           (cross-statement CSE; introduces Declare_local)
```

`optimize` is a thin wrapper around `optimize_proc` that additionally invokes optional
pretty-printing callbacks (`unoptim_ll_source` before optimization, `ll_source` after) so that
backends and debug tooling can capture the `.ll` source at both stages.

The `optimize_ctx` record carries a `computations` table (a map from tensor node to its stored
inlinable computations) **across** compilation calls, so that a tensor virtualized while compiling
one routine can be inlined into a later routine that reads it. `optimize_proc` threads the incoming
`optimize_ctx` through `virtual_llc` and returns it unchanged in the `optimized` result.

### 1. Tracing Phase (`visit_llc`)

This phase symbolically executes the computation to build a `traced_store` mapping each tensor node
to a `traced_array`:

```ocaml
type traced_array = {
  tn : Tn.t;
  assignments : int array Hash_set.t;       (* Positions written to *)
  accesses : (int array, visits) Hashtbl.t; (* Positions read, with visit counts *)
  mutable zero_initialized_by_code : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
  mutable read_only : bool;
  mutable is_scalar_constexpr : bool;
  mutable is_accessing : bool;              (* Does computation access non-constant arrays? *)
  mutable is_complex : bool;                (* Does computation involve non-trivial ops? *)
}
```

Additionally, a `reverse_node_map : (Symbol, Tnode) Hashtbl.t` tracks which tensor node's
computation "owns" each loop symbol. This is used to associate for-loops with the tensor
computations they belong to. (A single for-loop can be shared by several tensors' computations; see
#134 and the cleanup phase.)

A single `merge_node_id` ref records the merge buffer's source node: every `Get_merge_buffer` in a
routine must reference the **same** merge node, and tracing asserts this single-merge-node
constraint.

Key analyses performed:

- **Access Pattern Analysis**: Tracks which positions are read/written and how many times (`visits`).
- **Dependency Analysis**: Determines read-before-write patterns (recurrence).
- **Scalar Constant Expression Detection**: Identifies tensor nodes that are constant scalars.
- **Complexity Classification**: Determines `is_accessing` (reads non-constant arrays) and
  `is_complex` (performs non-trivial operations on array accesses).
- **Memory Mode Inference**: Decides whether tensors should be virtual, materialized, etc.

#### Index Position Computation

For tracing, a `lookup` helper converts symbolic indices to concrete integer positions: `Fixed_idx`
yields its constant, `Sub_axis` yields 0, `Iterator` yields the symbol's bound value (default 0), and
`Affine { symbols; offset }` computes the linear combination `offset + Σ coeff·value`. For-loops are
only unrolled up to `max_tracing_dim` positions during this symbolic execution.

### 2. Virtualization Phase (`virtual_llc` + `check_and_store_virtual`)

This phase determines which tensor computations can be inlined ("virtualized").

#### `virtual_llc`: Identifying Computation Boundaries

The `virtual_llc` function traverses the code and identifies the code blocks that define each tensor
node's computation. It tracks a `process_for` set of tensor nodes whose computations are currently
being traversed. When encountering a `For_loop`, it consults `reverse_node_map` to see if this loop
"belongs to" a tensor that hasn't been processed yet, marking that as the top-level computation for
the tensor. For each identified computation block it calls `check_and_store_virtual` to decide
inlinability, storing the result in the `optimize_ctx.computations` table.

#### `check_and_store_virtual`: Validating Inlinability

This function validates that the computation can be safely inlined, via these checks:

1. **Index Consistency** (`check_idcs`): All accesses to the tensor being virtualized must use the
   **same** index pattern (`Non_virtual 4` otherwise).

2. **Symbol coverage / groundability**: Every non-static symbol used in the LHS index map must be
   *groundable* by `inline_computation` at each call site. A symbol that appears in a bare `Iterator`
   position is bound directly from the call args; a symbol that occurs only inside `Affine` positions
   must be pinned by the map being affine-injective. If neither holds (`syms` not covered by the bare
   iterator symbols and the map is not injective), virtualization fails with `Non_virtual 5`.

3. **Affine index handling (gh-133 Stage A / Stage B)**: Earlier OCANNL rejected any `Affine` index
   with more than one non-static symbol. That restriction is **gone**. `check_idcs` now accepts:
   - *Stage A*: repeated/diagonal symbols (`[i; i]`, partially diagonal `[i; j; i]`) and covered
     single-symbol affine positions;
   - *Stage B*: genuinely multi-symbol affine positions (`stride·oh + kh`, `K·i + k`, triangular
     `(s1, s1+s2)`), **but only when the whole LHS index map is proven affine-injective** over the
     producer loop widths (`Indexing.affine_injective`). Injectivity is the soundness condition:
     dropping the producer loops during inlining is only safe if no two producer-index tuples collide
     on the same LHS position (otherwise fold contributions would be lost). A multi-symbol affine
     position in a non-injective map fails with `Non_virtual 51`.

   `Concat` indices are not eliminated by this point only by mistake — they must have been lowered
   away — so encountering one here fails with `Non_virtual 52`.

4. **No Escaping Variables**: Dynamic symbols used in nested computations must be bound within the
   computation's scope (or be static indices). Escaping symbols are rejected with `Non_virtual 7`
   (in sibling `Set`/`Set_from_vec` indices), `Non_virtual 9` (in sibling `Get`/`Get_merge_buffer`
   indices), or `Non_virtual 10` (in `Embed_index`).

5. **Non-Traced Loops Forbidden**: Loops with `trace_it = false` prevent virtualization
   (`Non_virtual 6`).

6. **No vector stores / staged code / hoisted locals**: a `Staged_compilation` node fails with
   `Non_virtual 8`; a `Declare_local` fails with `Non_virtual 19` (defensive — see below).

7. **Has Setter**: the computation must actually write to the tensor (`Non_virtual 12`); and the
   tensor must not be already non-virtual (`Non_virtual 11`).

If all checks pass, the computation (with its defining indices) is stored in the `computations` table
for later inlining. Over-acceptance here is safe: `inline_computation` re-validates per call site and
falls back to materialization via `Non_virtual 13` if a particular site cannot be grounded.

#### Non_virtual Exit Codes

When validation fails, `check_and_store_virtual` (or `inline_computation`) raises `Non_virtual i`,
and the handler commits the tensor to `Never_virtual i` (the provenance `i` records *why*):

- **4** — Inconsistent index patterns between accesses.
- **5** — Symbol coverage/groundability failure (a non-static symbol is neither bound from a bare
  iterator position nor pinned by an injective affine map).
- **6** — Non-traced loop (`trace_it = false`) encountered.
- **7** — Escaping variable in a sibling `Set`/`Set_from_vec` index.
- **8** — `Staged_compilation` node encountered.
- **9** — Escaping variable in a sibling `Get`/`Get_merge_buffer` index.
- **10** — Escaping variable in `Embed_index`.
- **11** — Tensor already marked non-virtual.
- **12** — No setter found.
- **13** — Index mismatch at a particular inlining site (per-site fallback to materialization).
- **14** — Empty computation list at inline time.
- **19** — `Declare_local` encountered during virtualization (defensive; see dead-code note).
- **51** — Multi-symbol affine position in a non-injective LHS map (gh-133 soundness guard).
- **52** — `Concat` index reached virtualization (should have been eliminated during lowering).
- **140** — A `Set_from_vec` (vector op) cannot be inlined as a scalar computation.

`Non_virtual 19` is a defensive arm. `Declare_local` is produced only by the final
`hoist_cross_statement_cse` pass, whereas computations are stored during `virtual_llc` (well before
hoisting), so a stored computation never contains a `Declare_local`. The arm only guards the
not-currently-exercised case of a hoisted program being fed back through virtualization.

### 3. Inlining Phase (`inline_computation`)

When a `Get` references a virtual tensor, `inline_computation` produces the inlined code as a
`Local_scope`. It works in two conceptual passes over the call site's index arguments:

1. **Bind bare iterators**: each LHS index that is a bare non-static `Iterator` is mapped to the
   corresponding call-site index expression (`call_args.(i)`).
2. **Ground affine occurrences**: any remaining affine occurrence is grounded via `subst`, which
   rewrites symbols through the binding environment.

If an LHS index neither matches a bound iterator nor equals the call-site index, the site cannot be
inlined and falls back with `Non_virtual 13`. A `Set_from_vec` to the virtual tensor cannot be
inlined as a scalar and raises `Non_virtual 140` (forcing the tensor `Never_virtual 140`).

#### Substitution with Affine Indices

The `subst` helper handles symbol substitution for all index forms, including `Affine`:

- a bound `Iterator` is replaced directly;
- a `Fixed_idx` substitution folds into the affine `offset`;
- a nested `Affine` substitution multiplies through the coefficients and merges symbol lists.

#### Loop Freshening

When inlining through a `For_loop`, fresh symbols are created to avoid capture: the loop index is
rebound to a fresh `Iterator` in the environment, and the body is rewritten under that binding.

#### Guards and initialization

Repeated/diagonal LHS positions and partially-covered affine maps produce *equality guards*: the
inlined accumulation is wrapped in `Where (cond, acc, Get_local id)` so a contribution is applied
only when the guard holds. When guards are present and the scope is not already zero-initialized, the
scope is marked `needs_init` and a `Set_local (id, Constant 0.)` initializer is prepended.

### 4. Cleanup Phase (`cleanup_virtual_llc`)

After inlining, this phase removes the now-redundant materialized writes of virtualized tensors,
validates symbol scoping, and finalizes memory modes. Its policy for nodes whose virtuality was not
positively decided is the subject of the #296 audit (see "Audit notes").

- **Shared loops (`For_loop`)**: a loop may compute several tensors. Cleanup recurses into the body —
  the per-statement arms below drop the setters of still-virtual tensors and keep those of
  non-virtual tensors — and elides the whole loop only when its cleaned body becomes empty. (This is
  the #134 shared-loop behavior; cleanup does **not** drop a loop merely because its index has a
  virtual owner.)
- **Undecided writes default to `Virtual`**: for `Zero_out`, `Set`, and `Set_from_vec`, if the target
  is still not `known_non_virtual`, cleanup commits it to `Virtual` (provenance **151** for
  `Zero_out`, **152** for `Set`/`Set_from_vec`) and drops the statement. The rationale (audit
  outcome): a node not forced `Never_virtual` by tracing/virtualization has no materialized reader
  left after inlining — its only uses were inlined into `Local_scope` bodies — so the materialized
  write is dead. For `Set_from_vec` specifically, a vector op that genuinely could not be
  scalar-inlined was already forced `Never_virtual 140` during inlining, so reaching the default-Virtual
  arm means the node stayed virtual-eligible.
- **Surviving reads finalize to `Never_virtual`**: a `Get` (or `Get_dynamic`) surviving into cleaned
  code reads a materialized array; cleanup commits its target to `Never_virtual 17`. This is the
  commitment point, not a redundant re-assertion — a node read here but only written under a
  virtualized setter is decided right now (see "Audit notes" for why this is an `update`, not an
  `assert`).
- **Local scopes**: `Set_local`/`Get_local` confirm their scope's tensor `Virtual` (provenance 16); a
  `Local_scope` whose tensor turned out non-virtual collapses back to a plain `Get`, otherwise its
  body is cleaned and the tensor confirmed `Virtual 18`.
- **Scope validation**: every `Iterator` symbol appearing in a surviving index is asserted to be in
  scope (bound by an enclosing for-loop or a static index).

### 5. Simplification Phase (`simplify_llc`)

A traditional optimizing compiler pass over scalar expressions:

- **Constant Folding**: `Constant 2.0 + Constant 3.0` → `Constant 5.0`.
- **Algebraic Simplification**: `x + 0` → `x`, `x * 1` → `x`, etc.
- **Local Scope Elimination**: `Local_scope { body = Set_local (id, v) }` → `v`.
- **Sequential Local Scopes**: two consecutive `Set_local` to the same scope get substituted.
- **Integer Power Unrolling**: `x ** 3` → `x * x * x` for small integer powers.
- **FMA Detection**: `a + b * c` → `FMA(b, c, a)`.

Substitutions are performed by the `substitute_float` (scalar) and `substitute_proc` (statement)
helpers: given a `~var` scalar expression and its `~value`, they replace every occurrence of `var`
throughout a scalar/statement, which is how local-scope results get folded into their single use.

### 6. One-Hot Reduction Rewrite (`rewrite_one_hot_reductions`)

A targeted rewrite (gh-343) that recognizes a reduction over a loop variable `k` whose body is a
one-hot selector — either `Where (Cmpeq (Embed_index k, index_expr), table_get, 0.)` (in either
operand order) or the equivalent multiply form `Cmpeq(...) * table_get` — and replaces the whole
reduction with a single guarded `Get_dynamic` gather that reads the table row at `index_expr`
directly. `match_one_hot_contribution` performs the structural match (accepting the range index both
as a plain `Iterator k` and as the unit affine `1·k + 0` that shape inference / reflection can
produce). The resulting `Get_dynamic` carries a range guard so fractional/out-of-range ids stay safe.
This construct never escapes `Low_level`/backend codegen.

### 7. Common Subexpression Elimination

Two CSE passes run after the one-hot rewrite (#351 and follow-ups):

- **`eliminate_common_subexpressions`** — *within-statement* scalar CSE. It detects repeated scalar
  subexpressions inside a single statement and shares them through a local scope, so a value is
  computed once and reused.
- **`hoist_cross_statement_cse`** — *cross-statement* CSE. It finds a computation shared across
  sibling statements and hoists it to a common ancestor scope, inserting a `Declare_local` (plus its
  body) before the first user and replacing each occurrence with a read of that local. A hoisted
  local is marked `needs_init` when it is read before being set in some path
  (`reads_scope_before_set`), so a zero initializer is emitted. This is the **only** producer of
  `Declare_local`.

Both passes compare scalar expressions up to alpha-equivalence via `cse_equal_scalar`: two
expressions that differ only in their local-scope ids (or dynamic-gather scope ids) are treated as
equal, which is what makes inlined/hoisted computations matchable across sites. (`cse_equal_scalar`
is the soundness-critical comparator; an overly-loose comparison would merge distinct computations.)

## Optimization Settings

The optimization behavior is controlled by `virtualize_settings`:

- `max_visits`: maximum number of times a tensor can be accessed before being materialized
  (default: **1**).
- `max_tracing_dim`: maximum dimension size for loop unrolling during analysis (default: **5**).
- `enable_device_only`: whether to prefer device-only storage when possible (default: **true**).
- `inline_scalar_constexprs`: whether to inline scalar constant expressions regardless of access
  counts (default: **true**).
- `inline_simple_computations`: whether to inline computations built from single getters, index
  embeddings, and scalar constant expressions (default: **true**).
- `inline_complex_computations`: whether to inline complex computations (default: **true**). This was
  formerly `false` "pending CSE"; with the CSE passes in place, complex inlining is on by default and
  the resulting duplication is recovered by `eliminate_common_subexpressions` /
  `hoist_cross_statement_cse`.

## Memory Mode Management

The optimization process works closely with OCANNL's memory mode system:

- **Virtual**: computations are inlined, no storage allocated.
- **Never_virtual**: tensor must be stored (the provenance int indicates why).
- **Materialized**: an unresolved request for a persisted node; resolves to **On_device**.
- **Device_only**: stored only on device (one of **Local** / **On_device**).
- **On_device**: stored on the devices that compute with it; CPU access is on-demand via
  context-mediated device-to-host transfers (no host copy on the node, after gh-ocannl-333).

The optimizer uses provenance tracking (the `int` in memory mode updates) to debug conflicts in
memory mode decisions. The cleanup-phase provenances (151/152 for default-to-Virtual, 17 for
finalize-to-Never_virtual, 16/18 for local scopes) are the most commonly observed in practice.

## Loop-Generation Utilities

A few helpers generate loop nests for backends and for the optimizer itself. They are conceptual
building blocks, also documented in `low_level.mli`:

- **`loop_over_dims dims ~body`**: builds a nested for-loop over `dims`, calling `body` with the index
  array at the innermost point. Dimensions that are not "iterated" (size-1 / non-iterable) contribute
  a `Fixed_idx 0` instead of a loop. With empty `dims`, `body` is called once with an empty index
  array.
- **`unroll_dims dims ~body`**: fully unrolls the iteration — it enumerates every fixed-index tuple
  and its row-major `offset` (rightmost dimension fastest) and concatenates the resulting statements.
  The empty-`dims` case calls `body [||] ~offset:0` exactly once.
- **`loop_over_padding_region ~dims ~padding ~body`**: iterates **only** the padding margins of a
  tensor, never the data interior. For each padded dimension it generates a left strip
  (`[0, left)`), a recursing middle (`[left, dim-right)`), and a right strip (`[dim-right, dim)`);
  unpadded dimensions iterate their full range while recursing. `body` is invoked only when at least
  one dimension contributed a padding index, so the all-data interior is skipped.

## Input/Output Partitioning

`input_and_output_nodes` inspects an `optimized` result and partitions its tensor nodes into the set
that the routine reads as inputs and the set it writes as outputs (plus the optional merge node). It
drives buffer allocation and the host/device transfer decisions a backend must make around a routine.

## Code Generation Integration

The optimized `Low_level.t` can be:

1. **Printed** using `to_doc` (OCANNL `%cd` syntax) or `to_doc_cstyle` (C-like syntax).
2. **Backend Compilation**: each backend pattern-matches on `Low_level.t` to generate device-specific
   code.
3. **Staged Compilation**: `Staged_compilation` nodes allow backends to embed generated code during
   optimization — important for backends that need to emit complex patterns the simple `Low_level.t`
   grammar cannot represent directly.

This optimization pipeline enables OCANNL to achieve high performance by eliminating intermediate
tensor allocations and generating specialized code for each computation pattern.

## Current Limitations and Future Work

- **Affine virtualization (#133)**: multi-symbol affine inlining is now supported, but **only** for
  index maps proven affine-injective. Non-injective multi-symbol affine maps (`Non_virtual 51`) and
  `Concat`-based axes (`Non_virtual 52`) remain non-virtualizable and fall back to materialization.
- **Shared for-loops (#134)**: cleanup keeps shared loops alive and drops only the virtual setters
  within them; further sharing of loop structure across more computation shapes is ongoing.

## Audit notes (#296)

The cleanup phase's `FIXME(#296)` markers were reviewed and replaced by explanatory comments; the
behavior is **retained**, not changed:

- The default-to-`Virtual` policy for undecided `Zero_out`/`Set`/`Set_from_vec` targets (provenance
  151/152) is accepted: such a node has no surviving materialized reader after inlining, so dropping
  its write is correct. (The deeper question of whether a not-yet-decided node could ever have its
  *needed* computation discarded was explicitly out of scope for this audit.)
- The `Get` arm's `TODO(#296)` was resolved by keeping `update_memory_mode … Never_virtual 17` rather
  than converting it to an `assert`: a `Get` can reach cleanup before its target's mode is finalized,
  because cleanup is itself the phase that commits surviving reads. Converting to an assert is not
  guaranteed safe, so the conservative `update` is retained.
- The defensive `Declare_local` arm (`Non_virtual 19`) and the defensive `Get_dynamic` arms in
  pre-rewrite passes are **retained**: `Declare_local`/`Get_dynamic` are produced by the *last*
  pipeline phases (`hoist_cross_statement_cse` / `rewrite_one_hot_reductions`), so earlier passes
  cannot encounter them on the fresh-lowering path, but the arms guard against re-entry of an
  already-optimized program.
