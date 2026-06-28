# Shape Constraint Generation in OCANNL

Companion notes to `docs/blog/ocannl-formal-core.md` and
`docs/formal-core-appendix.md`. The formal core defines the constraint language,
the row and dimension orders, and the solver. This note records the missing
front-end layer: how `tensor/shape.ml` generates those constraints from tensor
operation shapes.

The goal is not to prove the whole implementation correct in one document. It is
to make the elaboration boundary precise enough that the core statements can be
read against the code. In particular, this note discharges most of the informal
"shape.ml audit" left in the core, and isolates the remaining proof obligations
for affine axes, concatenation, total-element constraints, and marker provenance.

Notation follows the formal core. Dimensions are ordered by `\sqsubseteq`, with
the claim-free broadcast unit `1_emptyset` as top. Rows are written
`l . <rho> . r` when open and `l . diamond . r` when closed. OCANNL's concrete
row record is:

```ocaml
{ beg_dims : dim list; dims : dim list; bcast : bcast; prov : provenance }
```

where `beg_dims` is the leading flank `l`, `dims` is the trailing flank `r`, and
`bcast = Row_var rho` or `Broadcastable` determines whether the row is open or
closed.

## 1. What Generation Produces

`shape.ml` produces constraints of type `Row.constraint_`:

```ocaml
Dim_eq      { d1; d2; origin }
Row_eq      { r1; r2; origin }
Dim_ineq    { res; opnd; origin; ... }
Row_ineq    { res; opnd; origin }
Dim_constr  { d; constr; origin }
Rows_constr { r; constr; origin }
Terminal_dim of bool * dim * origin
Terminal_row of bool * row * origin
Shape_row    of row * origin
```

The first four are the core equality and inequality forms. They are generated
with the convention:

```text
res \sqsubseteq opnd
```

where `res` is the shape demanded by the result position and `opnd` is an
operand shape that may broadcast into that demand. Thus pointwise operations say
"the result row refines each operand row"; composition says "the consuming input
row refines the produced output row".

The remaining forms are implementation extensions:

- `Dim_constr` carries direct size lower bounds such as `At_least_dim i`.
- `Rows_constr` carries row-level arithmetic constraints: `Exact dims` and
  `Total_elems`.
- `Terminal_row` and `Terminal_dim` mark leaf tensors whose unresolved axes may
  be closed downward by the staged policy.
- `Shape_row` is injected during `finish_inference` so updated non-terminal
  rows can be revisited without traversing the whole graph.

Constraint generation also returns projection-side metadata: a map from freshly
introduced dimension variables to fixed or iterator projection symbols. This is
not a shape constraint, but it must be generated from the same parse tree so that
projection inference is a second reading of the same operation.

We can summarize the implementation as a judgment:

```text
G |- update_step u ==> (P, Z, Phi)
```

where `P` is the projection-axis environment, `Z` is the set of discardable
dimension variables allowed to close to `0` under block/concat semantics, and
`Phi` is the generated shape-constraint list. `get_inequalities` implements this
judgment.

## 2. Rows Generated from Specs

The shared spec elaborator is `einsum_slot_spec_to_dims_bio`, built on
`axes_spec_to_dims_bio`. It translates one parsed slot of an einsum-like spec
into three rows:

```text
(batch row, input row, output row)
```

It is parameterized by two operation-local environments:

- `dim_var_env : label -> dim_var`
- `row_var_env : label -> row_var`

The environments are fresh per operation instance. Reusing a label inside one
spec therefore means using the same variable; calling the same OCaml operator
again re-emits the spec with fresh variables.

### 2.1 Axis Positions

The parser gives each axis an `AxisKey` containing its row kind and whether the
axis is counted from the outer-left or outer-right side. `axis_map_to_dims_bio`
then forms:

```text
beg_dims = axes anchored at the leading flank
dims     = axes anchored at the trailing flank
```

For the common no-ellipsis form, all axes are trailing. Axes written before an
ellipsis become leading axes. This is the concrete source of the formal marker:
the marker sits between `beg_dims` and `dims`.

### 2.2 Dimension Syntax

For each axis specification:

- A label such as `h` maps to `Var alpha_h`, allocating `alpha_h` if needed.
- A fixed index creates a fresh dimension variable `alpha`, records
  `P(alpha) = Fixed_idx i`, and emits `Dim_constr(alpha, At_least_dim (i + 1))`.
- An affine spec creates
  `Affine { stride; over; conv; stride_offset }`, with `over` and optional
  `kernel` labels resolved through `dim_var_env`.
- A concat spec such as `a^b` creates `Concat [Var alpha_a; Var alpha_b]`.

The fixed-index rule is one of the details not present in the core calculus: a
fixed projection is also a size lower bound.

### 2.3 Row Variables

For each row kind in a slot:

```text
no ellipsis / no row variable  ==> Broadcastable
..name.. or ...                ==> Row_var rho_name
```

The special ellipsis `...` is expanded contextually to the kind name
(`batch`, `input`, or `output`), so the ordinary ellipsis is kind-indexed.
Explicit row variables `..name..` are keyed only by `name`; if the user repeats
one across different kinds or slots, that is a deliberate equality of row
variables in the generated system.

Each row variable used in an einsum/compose-style spec is marked by
`Row.add_used_in_spec_or_compose`; row variables used in pointwise broadcasting
are marked by `Row.add_used_in_pointwise`. These marks are not constraints in
the formal core, but they participate in later "forgotten hidden dimension"
diagnostics and closing safety.

### 2.4 Captured Variables

The trailing OCaml capture list in `%op`/`%cd` is represented as
`delayed_var_ref list`. After parsing the spec, `bind_delayed_vars_to_envs`
resolves each captured name:

- If the name is a dimension variable, the reference is bound to `Dim alpha`.
- If the name is a row variable, the reference is bound to `Row rho`.
- If `Shape.set_dim` already supplied a concrete value and the pass is not the
  projection re-solve, generation emits:
  - `Dim_eq(alpha, n)` for dimension captures;
  - `Rows_constr([rho], Total_elems n)` for row captures.

This is the bridge between user-facing scalar dimensions and row products: a
captured row variable denotes the product of the axes later assigned to that row.

## 3. Operation Rules

Write a shape `S` as the triple:

```text
S = (S_B, S_I, S_O)
```

for batch, input, and output rows. The current update result is `C`; operands
are `A`, `B`, `D`, etc.

### 3.1 Terminals

All terminal shapes emit `Terminal_row` for each row:

```text
Terminal_row(is_param, C_B)
Terminal_row(is_param, C_I)
Terminal_row(is_param, C_O)
```

The flag records whether the terminal belongs to a parameter. The solver uses it
to reject unconstrained hidden parameter dimensions instead of silently guessing
them.

Some terminal initializers also seed row constraints over the flattened
`batch @ output @ input` rows:

- `Data (Reshape nd)` emits `Total_elems(product(dims nd))`.
- `Fetch (Constant_fill values)` emits `Total_elems(length values)`.
- `Data (Keep_shape_no_padding nd)` emits `Exact(dims nd)` outside projection
  re-solving.
- `Data (Padded { data; ... })` emits `Exact(dims data)` outside projection
  re-solving.
- `Fetch (Slice tn)` emits `Exact(tail(dims tn))` if the fetched tensor's dims
  are already known.

Fetches such as constants, ranges, embedded symbols, and embedded dimensions
otherwise only mark the row as terminal.

These rules explain why the formal core's pure row order is not the whole
implementation: data sources can constrain a whole row product or exact flattened
axis list without saying which operation row forced it.

### 3.2 Unary Structural Operations

Transpose emits inequalities:

```text
C_B \sqsubseteq A_B
C_I \sqsubseteq A_O
C_O \sqsubseteq A_I
```

Pointwise unary emits:

```text
C_B \sqsubseteq A_B
C_I \sqsubseteq A_I
C_O \sqsubseteq A_O
```

Batch slicing is exact rather than broadcast-style. It creates a fresh dimension
`s`, records its projection as the slice iterator symbol, optionally pins it to
the static range size, and expands the result batch row on the left:

```text
(s :: C_B.beg_dims) . <C_B.bcast> . C_B.dims  \approx  A_B
C_I \approx A_I
C_O \approx A_O
```

This is the canonical source of nonempty leading flanks outside explicit
ellipsis syntax. It is also the reason the row algebra needs a leading flank:
the sliced axis is outer-anchored.

Permute parses a unary einsum spec and emits exact equations between spec
templates and actual rows:

```text
C_k       \approx template_lhs_k
template_rhs_k \approx A_k
```

for `k in {B,I,O}`.

`Uint4x32_to_prec` assumes the input has one flattened axis and relates the
output's total elements by a precision-dependent coefficient. It emits:

```text
Rows_constr([A_B, A_O, A_I], Exact [v])
Rows_constr([C_B, C_O, C_I], Total_elems(Strided_var(coeff, v, denom=1)))
```

The coefficient may be forced only in a later stage, which is why
`Strided_var` stores a lazy coefficient.

### 3.3 Binary and Ternary Broadcast Operations

Pointwise binary is pure broadcast matching:

```text
C_B \sqsubseteq A_B    C_B \sqsubseteq B_B
C_I \sqsubseteq A_I    C_I \sqsubseteq B_I
C_O \sqsubseteq A_O    C_O \sqsubseteq B_O
```

Pointwise ternary is the same rule with three operands.

Composition, corresponding to `A * B` or function composition, emits one
contraction compatibility plus the row flows for the remaining parts:

```text
A_I \sqsubseteq B_O
C_B \sqsubseteq A_B
C_B \sqsubseteq B_B
C_I \sqsubseteq B_I
C_O \sqsubseteq A_O
```

`Compose_accumulate` adds a third operand `D` that is pointwise-compatible with
the result:

```text
D_B, D_I, D_O are each above C_B, C_I, C_O respectively
```

in the same `C_k \sqsubseteq D_k` direction.

These rules are the concrete source of the core's row inequalities. They also
show that non-einsum operations are already "checking semantics" operations:
they permit ordinary broadcasting.

### 3.4 Einsum, Permute, and Block Specs

The shared n-ary helper `einsum_n_constraints` handles binary einsum, ternary
einsum, and `Block`.

For an n-ary spec:

```text
rhs_1; ...; rhs_n => lhs
```

generation first elaborates every `rhs_i` and `lhs` slot to row triples using
the shared dimension and row environments. Then it emits, for every kind `k`:

```text
C_k            \approx template_lhs_k
template_rhs_i_k \approx A_i_k       for each operand i
```

That is all: einsum-family operations emit row equalities, not row inequalities.
This is an inference policy, not the weakest possible checking relation. The
formal core's one-shot inequality sandwich would be:

```text
flat(C_k)       \sqsubseteq template_lhs_k
flat(template_rhs_i_k) \sqsubseteq A_i_k
```

at the rigid/one-shot boundary. `shape.ml` deliberately strengthens this to
equality so that labels and row variables are inferred bidirectionally. This is
the implementation fact behind Remark 2.15 and Appendix A.8.

`Block` adds one more generation product: `discardable_vars`. For concat axes,
`compute_block_discardable_vars` identifies component variables that may close
to size `0` because the complementary components cover the other side. This is
outside the broadcast order: `0` is the additive unit for concatenation, not the
broadcast top and not the contradiction bottom. The formal core's Conjecture 8.3
is precisely the soundness obligation for this generation-side set.

### 3.5 Logic Defined Elsewhere

The `Defined_by_cd_logic` cases emit no shape constraints in `shape.ml`:

```text
G |- Defined_by_cd_logic ==> empty Phi
```

Their correctness obligation belongs to the `%cd`/assignment logic that created
the shapes.

## 4. Explicit User Constraints

The generator is not only `get_inequalities`. Several public `Shape` APIs append
constraints to the active global constraint list.

`infer_equal sh1 sh2` emits:

```text
sh1_B \approx sh2_B
sh1_I \approx sh2_I
sh1_O \approx sh2_O
```

`set_dim ref n` behaves according to how the delayed reference has resolved:

- not yet resolved: store `n` in the reference for later generation;
- dimension variable `alpha`: emit `alpha = n`;
- row variable `rho`: emit `Total_elems([rho], n)`.

`set_equal ref1 ref2` emits:

- dimension/dimension: `alpha = beta`;
- row/row: `rho_1 \approx rho_2`;
- dimension/row: `Total_elems([rho], Strided_var(coeff=1, var=alpha))`.

`set_scale ~factor large small` is restricted to dimension variables. For
`factor > 1`, unresolved dimension references generate:

```text
large = Affine { stride = factor; over = small; conv = None; stride_offset = 0 }
```

If either side is already solved, it checks or propagates the concrete arithmetic
immediately.

`set_terminal ~is_param sh` emits `Terminal_row` constraints for all three rows
and marks the shape as unused unless later consumed. `finish_inference` filters
terminal/shape-row constraints for still-unused shapes so dead tensor nodes do
not force inference.

## 5. Shape Construction Rules

`Shape.make` creates the initial rows for a tensor:

- provided dimensions become closed rows with `beg_dims = []`,
  `dims = provided axes`, and `bcast = Broadcastable`;
- omitted kinds become open rows with empty flanks and a fresh row variable.

The default user dimension constructor uses the `default` basis even for size
`1`. This is deliberate: an explicit user `1` is a concrete atom and does not
silently become the claim-free broadcast top. Only internal broadcast fillers
use `get_bcast_dim`.

`Shape.of_spec` is the textual version of the same construction. It parses
`batch | input -> output`, creates closed or open rows as requested by the spec,
and optionally applies the `Input_equals_output` deduction.

`Input_equals_output` immediately calls `Row.unify_row` on the input and output
rows. This is an online constraint, not a separate `Row_eq` stored for later.
Semantically it is the same row equality; operationally it gives earlier
feedback and simplification.

## 6. Loose Ends Closed by Generation

### 6.1 Why Equality Appears in Einsum

The formal core distinguishes checking from inference. Checking an einsum-like
operation could be expressed by one-shot inequalities against rigidified spec
rows. `shape.ml` instead emits `Row_eq` constraints for every spec row. This
stronger relation is what lets a label in the result determine a parameter axis,
or an operand axis determine a result axis.

Thus:

```text
einsum equality = inference policy
pointwise/compose inequality = broadcast checking
```

The implementation is not accidentally inconsistent here. It has two readings
of operation specs, and the equality reading is chosen exactly for the
operations where bidirectional shape recovery is valuable.

### 6.2 Marker Provenance

The core notes that flat row equality leaves marker placement underdetermined.
`shape.ml` supplies the missing provenance story. Marker placements are created
only by declared sources:

- external/literal shapes are left-marked: `beg_dims = []`, all axes in `dims`;
- spec ellipses put the marker exactly where the user wrote `...` or `..name..`;
- `Batch_slice` prepends an outer-left axis, requiring a leading flank;
- row equality inherits a closed side's split as a policy choice when binding an
  open row variable.

The expected invariant is:

```text
Every marker consumed by a later inequality descends from one of these declared
placements, through substitution and row-equality binding.
```

This is weaker than saying solved row values are never consumed as broadcast
operands; they are. It is the right invariant for the current implementation:
markers may be transported, but they are not fabricated without a source.

The pathological witness in the formal core requires a row equality and a marked
inequality that disagree about the same flat content's split. The generation
rules above explain why that witness is expected to be unreachable from surface
programs, but the full proof still needs an induction over every operation rule
and every row-binding policy step.

### 6.3 Projection Locality

`derive_projections` freshens projection ids, regenerates the operation's
constraints with `for_projections = true`, and re-solves locally. The important
generation facts are:

- each call to `einsum_n_constraints` creates fresh spec variables;
- the projection-axis environment mentions only variables introduced while
  elaborating that operation's slots;
- `for_projections = true` suppresses delayed-reference constraints coming from
  previous `set_dim` calls, because projection inference wants the operation's
  local axis identities, not another global size solve.

This supplies the bookkeeping premise for Lemma 7.7 in the core: after global
shape closing, per-operation projection re-solving sees the same local operation
shape relation, but with fresh projection identities, so size equalities learned
elsewhere cannot force co-iteration here.

### 6.4 Constraint Order

Most generated constraints are order-insensitive at the semantic level. However,
the solver has policy choices: flat equality commits marker placement, and later
closing commits least-material representatives. `einsum_n_constraints` has an
explicit `lhs_constraints_first` flag used by binary einsum. This flag does not
change the intended relation, but it can choose which policy representative is
reached when several declared placements race.

The right theorem is therefore not full order-independence of generated runs. It
is:

```text
Semantic satisfiability is independent of generation order, while marker and
closing policies are deterministic functions of the generated constraint order.
```

This is the implementation-level version of Appendix A.4.

## 7. Extended Constraint Obligations

The formal core intentionally excludes `Affine`, `Concat`, `Exact`,
`Total_elems`, terminals, and the staged close. `shape.ml` generates all of
them, so a full paper needs these extra obligations.

### 7.1 Affine Dimensions

Generation creates affine dimensions in two places:

- spec axes such as `stride*i + k`;
- `set_scale`, which creates a simple stride relation.

The required invariant is mode agreement:

```text
The size equation encoded in the generated Affine term must match the projection
mode later used to address the tensor.
```

For `conv = None`, the shape relation is a strided relation. For convolution,
the no-padding and padding modes have different size equations. The generator
stores the mode in `conv.use_padding`; the solver and projection derivation must
interpret that flag identically.

### 7.2 Total Elements

`Total_elems` relates the product of one or more rows to either a literal count
or a coefficient times a dimension variable. Generation uses it for reshapes,
constant fills, vectorized precision conversion, row captures, and mixed
dimension/row equality.

The solver does not represent arbitrary product equations principally. It keeps
one stored row constraint per row variable and uses staged heuristics to
eliminate enough variables for the generated fragment. Therefore the strongest
reasonable theorem is relative:

```text
For the fragment generated by shape.ml, successful staged solving satisfies all
generated Total_elems constraints.
```

Completeness cannot hold for an unrestricted extension: with concat sums inside
row products, `Rows_constr` can express Diophantine-style equations.

### 7.3 Exact Rows

`Exact dims` is generated by exact-data initializers, slices with known sizes,
`Uint4x32_to_prec`'s input-side assumption, and some row-constraint reductions.
Its semantics is flat: the concatenated rows `r_1 @ ... @ r_n` have exactly the
listed dimensions after all row variables are resolved.

The main loose end is interaction with open rows. The implementation sometimes
chooses an output row variable to absorb the single exact axis and closes other
rows to empty. That is a staged policy, not a principal consequence of the row
order.

### 7.4 Concat and Discardable Zero

`Concat` dimension terms are generated by block/reverse-side specs. The
discardability analysis is generation-side; it decides which variables may be
guessed to `0` in closing.

This introduces a second unit:

- `1_emptyset`: top of the broadcast order, a multiplicative/product neutral
  size in the broadcast sense;
- `0`: additive unit for concatenation components, outside the broadcast order.

Any formalization that treats `0` as a broadcast dimension is wrong. The proof
obligation is that every variable in `discardable_vars` occurs only where its
zero contribution is projected away or covered by the complement test.

### 7.5 Terminals and Shape Rows

`Terminal_row` and `Shape_row` are not logical connectives in the declarative
constraint language. They are stage triggers:

- terminals close leaf shapes downward to their recorded lower bounds, or guess
  minimal dimensions when safe;
- shape rows make non-terminal rows available to the later global closing stages.

The formal core already classifies closing as policy. Generation adds the source
of the policy boundary: terminals are exactly the leaves that cannot be forced
from an upstream producer, while non-terminals are revisited after operation
constraints have propagated their lower bounds.

## 8. Compact Rule Table

For quick reference, the core generation rules in `get_inequalities` are:

| Logic | Generated constraints |
|---|---|
| terminal fetch/data | `Terminal_row` for each row; optionally `Exact` or `Total_elems` |
| transpose | `C_B <= A_B`, `C_I <= A_O`, `C_O <= A_I` |
| pointwise unary | `C_k <= A_k` for all kinds |
| pointwise binary | `C_k <= A_k` and `C_k <= B_k` |
| pointwise ternary | `C_k <= A_k`, `C_k <= B_k`, `C_k <= D_k` |
| compose | `A_I <= B_O`, `C_B <= A_B`, `C_B <= B_B`, `C_I <= B_I`, `C_O <= A_O` |
| compose accumulate | compose constraints plus `C_k <= D_k` |
| batch slice | expanded result batch row equals operand batch; input/output equal |
| permute | result equals LHS template; RHS template equals operand |
| einsum | result equals LHS template; each RHS template equals its operand |
| block | n-ary einsum equalities plus `discardable_vars` from concat analysis |
| uint4x32 conversion | input `Exact [v]`; output `Total_elems(coeff * v)` |
| defined-by-cd | no constraints in `shape.ml` |

Here `<=` abbreviates `\sqsubseteq`, and every row rule is emitted separately
for batch, input, and output unless the rule states a cross-kind connection.

## 9. Suggested Formal Statements

The companion proof work can be organized around four statements.

**Elaboration Well-Formedness.** For every `update_step`, generation produces a
finite constraint set whose row terms contain at most one row variable, at the
marker; whose variable environments are operation-local except for actual tensor
shape variables; and whose projection-axis map mentions only generated
dimension variables.

**Core Compatibility.** Erasing `Dim_constr`, `Rows_constr`, `Terminal_*`,
`Shape_row`, `Affine`, and `Concat` leaves a constraint set in the formal core's
language. For pointwise, transpose, and compose operations this erased set is
exactly the broadcast checking relation. For einsum-family operations it is the
chosen equality-based inference strengthening of the one-shot checking relation.

**Marker Provenance.** Every marker placement in a generated or solved row is
traceable to an external left-marked row, a written spec ellipsis, `Batch_slice`,
or an inherit-the-split equality binding. Consequently, the formal core's
placement-discriminating witness is not generated by surface programs, assuming
the row-binding policy preserves provenance.

**Local Projection Re-Elaboration.** After global shape inference succeeds,
regenerating constraints for one operation with fresh projection ids and
`for_projections = true` has a solution obtained from the globally closed
shapes, and no projection id from another operation appears in the local system.
This is the missing implementation-facing premise of the core's projection
soundness theorem.

## 10. Reading the Code Against This Note

The main code correspondences are:

- `get_inequalities`: the generation judgment for one `update_step`.
- `einsum_slot_spec_to_dims_bio`: one spec slot to batch/input/output rows,
  plus fixed-index and affine side constraints.
- `einsum_n_constraints`: n-ary exact spec matching for binary/ternary einsum
  and block operations.
- `compute_block_discardable_vars`: generation-side additive-zero policy for
  concat specs.
- `propagate_shapes`: online Stage 1 generation and solving.
- `finish_inference`: global staged solving, `Shape_row` injection, and
  projection derivation.
- `derive_projections`: local re-generation with fresh projection ids.

The formal core explains what `Row.solve_inequalities` does with the generated
constraints. This document explains why those constraints, rather than some
other constraints, are the ones produced by the tensor operation language.
