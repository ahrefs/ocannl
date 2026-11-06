# gh-490: Symbolic extents — v1: symbolic dimensions in shape inference

Issue gh-490 proposes launch-time-bound extents: specific axes (batch, sequence) stay symbolic
through lowering and codegen, becoming runtime kernel parameters, so one compile + one tune-cache
entry serves every extent up to a maximum. This document covers the **first increment**: plugging
static-indexing symbols into shape representations and inference, and reconciling symbolic shapes
with Tnode dims. Lowering, codegen, launch binding, and autotune identity are follow-up tasks (see
the issue's task list).

## Design

### Representation: a new `Row.dim` constructor

```ocaml
| Sym of sym_dim
and sym_dim =
  { sym : Ir.Indexing.static_symbol; sym_basis : string; sym_proj_id : proj_id option }
```

A symbolic dimension reuses the **existing static-indexing binding mechanism**: the symbol is an
ordinary `Ir.Indexing.static_symbol` (as threaded through `bindings` and bound per launch). Its
declared `static_range` is mandatory (and must be positive): it is the **maximum extent**, used to
size allocations and, until launch-time binding lands, to materialize the axis everywhere
downstream of inference. `sym_basis` and `sym_proj_id` mirror `solved_dim`'s fields so the basis
discipline and projection inference extend mechanically.

Note the range convention: for slicing (`@|`) the bound value is an index in `[0, range)`; for an
extent the bound value will be a size in `[0, range]` (buffers are sized `range`). Bind-time
validation of extent parameters is part of the backend-plumbing follow-up.

### Inference semantics: a rigid constant (universal variable)

`Sym s` behaves as a minted constant:

- **Equality** (`unify_dim`): `Sym s = Sym s` (same symbol; bases must agree, mirroring `Dim`).
  `Sym` vs. a different `Sym`, or vs. any concrete `Dim` — even one equal to its range — is a
  `Shape_error` mismatch. `Var v = Sym s` solves `v := Sym s`.
- **Broadcasting** (`solve_dim_ineq`, row GLB merge): the broadcast top `1_(bcast_if_1)` broadcasts
  *to* a symbolic axis (scalars compose with symbolic-extent tensors); a `Sym` never broadcasts
  itself. Two incompatible rigid bounds flowing into the same variable demote the bound to the
  broadcast top, exactly as differing concrete `Dim`s do.
- **Arithmetic is off-limits (v1)**: any `Total_elems` constraint that would need the symbolic
  axis's size — reshape-style totals, `set_dim` on rows containing it, strided-var substitutions,
  divided-by factors — fails fast with "Total_elems constraints are not supported with symbolic
  dimensions" (`Row.fail_if_total_elems_over_sym` at the reduction/substitution choke points). We
  deliberately refrain from a `Sym_denom`-style solver extension in v1. Similarly, convolutions /
  affine transforms over a symbolic axis, and `Concat` axes equated with a symbolic extent, are
  rejected (structural cancellation of equal `Sym` components inside `Concat`-vs-`Concat` still
  works).
  - Consequence: **random-init parameters cannot have symbolic axes in v1** — `uniform ()` etc.
    route the threefry counter mapping through a `Total_elems` (uint4x32-blocks = numel/4)
    constraint. Constant-init and data terminals are fine. Lifting this (the RNG constraint only
    needs the *materialized* numel, which is well-defined) is a natural follow-up.

### Reconciliation with Tnode dims: materialize at the maximum extent

`Shape.to_dims` (→ `Tnode.dims`) maps `Sym s` to its range: allocation, host transfers, and numel
contracts all see the max extent. Projection inference (`Row.solved_dim_of_sym`) does the same, so
product spaces, iterators, and generated loops run over the full range. In this increment a
symbolic-dim program is therefore **semantically identical to writing the max extent**, with the
symbol tracked end-to-end through inference; the follow-up tasks swap the materialization seam for
a launch-time parameter without touching the solver again.

### API: `Shape.set_sym_dim`

Symbols plug in through the existing einsum capture mechanism, mirroring `Shape.set_dim`:

```ocaml
let seq, bindings = Idx.get_static_symbol ~static_range:512 bindings in
let%op y = x ++ "b s d => b s d" [ "s" ] in
Shape.set_sym_dim s seq
```

`delayed_var_ref` gains `mutable solved_sym : static_symbol option`; the einsum re-binding path
(`bind_delayed_vars_to_envs`) re-emits `Var v = Sym sym` equations for symbolic refs the same way
it re-emits concrete `set_dim` values, so a captured ref can be reused across specs. `set_equal`
propagates symbolic settings; `set_scale` and row-variable refs reject them. Conflicts (concrete
then symbolic, symbolic then different symbol) raise `Shape_error`.

Solutions are *not* captured back into `var_ref.solved_dim` (it is an `int option`); a var solved
to a symbolic dim by inference leaves the ref unresolved. Capturing into `solved_sym` is a possible
refinement.

## Testing

`test/einsum/test_symbolic_extents.ml`: happy paths (capture + solve + materialize at range,
whole-buffer forward and reduction executed on the default backend, same-symbol unification across
tensors, scalar broadcast) and error paths (missing range, set_dim/set_sym_dim conflicts,
symbolic-vs-concrete solver mismatch, distinct symbols, row-var rejection, Total_elems fail-fast).

## Stage 2: launch-time-bound extents via body guards (landed)

Instead of a symbolic `For_loop` bound (which would ripple through every bounds-reading analysis —
virtualization tracing, interval simplification, schedule transforms, digests, launch geometry),
stage 2 keeps every loop at its **maximum extent** and guards the loop *body*:

```c
for (int32_t i7 = 0; i7 <= 5; ++i7) { if (i7 < i1) { ... } }   // i1: const int32_t kernel param
```

- `Indexing.projections` gains `extent_syms : (symbol * static_symbol) list` (product iterator →
  extent symbol), populated by `Shape.derive_projections` from `Row.Sym` axes; `product_space`
  keeps the max. The `%cd` inline-projections records in `ppx_cd` carry the field through.
- `Assignments.to_low_level` wraps each loop body iterating a symbolic axis in
  `If (iterator < Embed_index (Iterator sym))` — **only when the extent symbol is among the
  routine's bindings** (`static_indices`). An unbound extent keeps stage-1 max semantics, so
  `forward_once`-style flows (no bindings) are unchanged. The kernel parameter, launch binding,
  and per-backend application reuse the existing static-index machinery unchanged.
- Soundness of existing analyses: every bounds-reading pass sees the max loop (a superset of the
  runtime range — sound for interval/guard discharge, allocation, launch geometry, digests). The
  `If` guard makes guarded computations non-virtual (`Non_virtual 142`), which conservatively
  blocks value-semantics folds (inlining/unrolled reductions) over symbolic loops. The interval
  env seeds an extent symbol's value as `[0, range]` (inclusive).
- Bind validation: `static_symbol` gains `used_as_extent` (set by `Row.get_sym_dim`); an extent
  binds a **size** in `[0, range]` (inclusive; buffers are sized `range`), while plain static
  indices keep the strict `[0, range)` check. `Train.sequential_loop` skips extent symbols (an
  extent is set once, not iterated).
- Semantics: reductions and consumers only touch the valid region (their loops are guarded by the
  same symbol); read the valid prefix on the host. What a run leaves in the cells beyond the bound
  extent was settled later (gh-ocannl-928): `docs/shape_inference.md`, "Symbolic extents: cells
  past a bound extent", owns that contract.
- Test `test/operations/symbolic_extent_launch.ml`: one compiled cc routine runs at extents
  6/4/1/0 with exact prefix results (the extent=4 total of 4.0, not 6.0, proves the guard), and
  extent=7 is rejected at bind validation.

## Stage 3: autotune at the upper bound + serial guard fusion (landed)

- **Autotune identity comes for free from stage 2**: the extent is a kernel parameter, not part of
  the lowered program, so the schedule digest (`Schedule_cache.canonicalize`, which does emit the
  `If` guard including the symbol) is identical for every extent value — one compile *and* one
  tune-cache entry per architecture, the gh-490 payoff. What remained was the tuning-size policy:
  `Autotune.set_test_bindings` (now exported) binds a symbolic extent at its **upper bound**
  `range` during timing (plain ranged indices keep `range / 2`), making the tuned schedule's cost
  model conservative for smaller runtime extents. Bucketing by representative sizes remains
  available as a future refinement if measurements diverge.
- **Serial-loop guard fusion peephole** (`C_syntax` `serial_loop`): a body-wrapping
  `if (i < s)` extent guard — with `s` a kernel parameter, not an enclosing loop index — hoists
  into the loop header as `for (i = 0; i <= max && i < s; ++i)`. The iteration variable is
  monotone, so once the guard fails it stays false: exiting equals skipping. This removes the
  per-iteration guard overhead on CPU serial loops (including GPU serial-fallback loops); grid
  loops keep the guard, which is the canonical GPU form. Pure codegen: `Low_level` and digests
  are unchanged.
- Test (`symbolic_extent_launch`, extended): the fused loops compute exact prefixes at extents
  6/4/1/0; timing measurement observably binds the extent at 6 (not `range/2 = 3`); a tuned
  routine serves extents 6 and 3; and a second `Autotune.tune` of the same program hits the
  extent-value-independent cache entry.

## Follow-ups (rest of gh-490)

1. Schedule transforms (Grid/Split/Tensorize) interacting with guarded loops — currently they see
   max-extent loops with an inner guard, which is correct but untuned; tuning-size bucketing.
2. RNG init over symbolic axes via materialized numel; capturing `solved_sym`; possibly `Sym`
   support in `Total_elems` (`Sym_denom`).
