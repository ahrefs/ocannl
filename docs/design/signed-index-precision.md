# Signed index precision

**Date**: 2026-07-03, core migration implemented 2026-07-04
**Status**: Core migration implemented — seeded by the interval-analysis discussion
([interval-analysis-scalar-t](interval-analysis-scalar-t.md), now landed) and the integer-ids
gather work. Decision: migrate and accept the churn — signedness simplifies correctness.

**Implemented (2026-07-04)**: `Ops.index_prec ()` is signed (int32; int64 under
`large_models`); signed loop counters and index kernel arguments in the cc, CUDA, and Metal
backends (`loop_index_type` / `arg_int_prefix`); the CUDA `int32/int64 → uint4x32` PRNG counter
conversions route through the bit-spreading builtins (the previously-unsigned index precision
now hits the signed arms); the per-node padded-element-count contract enforced in
`Tnode.create`/`create_from_padded`/`create_with_reshape` (hard `User_error` naming the node);
bind-time validation of launch parameters (`Indexing.validate_bound_value`, called per launch
from `Context.run` and the Metal argument marshalling; the CUDA backend already validated) —
non-negative, within the declared `static_range`, and within the int32 index width when
unbounded (turning what used to be silent truncation into a hard error directing to
`large_models` or a declared range). Golden churn: 8 `.expected` files (loop counter types) +
the index-precision unit test.

**Deferred** (in dependency order): the abstract storable `Index_prec` constructor (host
native-int arrays, `TDSL.range` as `Index_prec` by construction); tnode-granular
interval-driven width selection at codegen (the global int32/int64 switch stands in as the
fallback width; the interval infrastructure it needs is now in place); per-parameter FFI widths
(unbounded params as int64 requires ctypes-variadic and `set_bytes` surgery — the bind-time
width validation covers the soundness gap meanwhile); `large_models` retirement (explicitly
gated on a separate resolution for the Metal pool-slot width, which stays unsigned).

## Goal

Migrate index arithmetic from unsigned (`Ops.index_prec ()` = uint32/uint64) to signed
(int32, int64 under `large_models`). Unsigned precisions remain for *data* domains only:
uint4x32 RNG state, uint8/uint32 stored values (e.g. class IDs).

## Motivation

The unsigned choice has charged a recurring correctness tax; three incidents so far:

1. `Fixed_idx (-1)` wraps to MAX (see the unsigned-index-precision lesson).
2. Guards must be written `a < b + 1` because `a <= b` with a possibly `-1` operand is wrong.
3. The gh-343 gather guard could not run in index precision at all (`-1 < idx` vacuously
   false after wrap) and resorted to `double` — since split into integer/float flavors, but
   the signed branch still had to pick int64 rather than the index precision.

Each was survivable because physical padding keeps all emitted index arithmetic
non-negative today. Logical padding / masks (the Padto direction) end that:
`Affine { offset = -padding }` makes negative intermediates *routine* rather than
exceptional, and `0 <= i` is inexpressible in unsigned arithmetic.

tinygrad precedent: index computation is signed throughout (int32 default, upcast to
int64 for large buffers; recent versions use an abstract index dtype that lowers to a
signed concrete type). Unsigned dtypes appear only as data (uint8 images/quantization,
bit-manipulation, threefry counters) — the exact split this proposal adopts.

## Design points

- Signed int32/int64 everywhere index arithmetic is emitted (loop counters, `Embed_index`,
  `Get_dynamic` casts), with width chosen at codegen — see the tnode-granularity bullet;
  the `large_models` switch is retired rather than flipped to signed.
- Int32 overflow is excluded by contract, not by widening: per-node (padded) element count
  must fit int32 unless 64-bit indices are in use. This is sufficient — every index
  intermediate is bounded by some node's extent by projection construction (axis indices by
  their dims, conv affine forms by the padded input dim, flat offsets by numel); loop
  *counts* may exceed 2^31 without any index expression doing so. Enforce once per node by
  wrapping the dims lazy in `Tnode.create` (forcing dims validates padded numel; every
  consumer inherits the guarantee). Violation is a hard error naming the node — auto-setting
  the global flag would be inconsistent across already-compiled routines.
- Width selection at **tnode granularity**, interval-driven (this task now *blocks on*
  [interval-analysis-scalar-t](interval-analysis-scalar-t.md) — see Sequencing): an index
  expression's width is the join over the extents and settled `vmin`/`vmax` bounds of the
  tensor nodes it indexes or reads. Static offsets into a node are bounded by its (padded)
  numel from projections; dynamic index values (`Get_dynamic.dyn_value`) are bounded by
  the Tnode interprocedural bounds layer — the same summaries that discharge guards also
  serve as the width oracle. This is tinygrad's actual mechanism (verified against the
  2026-06 source): index expressions are built in an abstract signed dtype
  (`dtypes.weakint`), and `pm_lower_index_dtype` in `tinygrad/uop/ops.py` selects width
  per expression at codegen — `select_dtype u = int64 if u.overflows(int32) else int32`,
  where `overflows` is one vmin/vmax query; binary ops join widths via
  `least_upper_dtype` with casts at boundaries; GPU special dims pinned int32. All
  produced widths are signed — unsigned never appears in tinygrad indexing. Mixed widths
  within a routine are safe *because* the choices derive from proven bounds; without
  intervals this would be ad-hoc casting, which is why the migration waits for them.
  `Ops.index_prec ()` becomes a codegen-time resolution instead of a global read, and the
  `large_models` switch is retired *from index-width selection only*: the same setting
  currently also selects Metal pool-slot/offset width (`pool_slot_msl_typ`, `uint` vs
  `ulong`), which needs its own resolution — always-64-bit slots, or a link-time
  per-device choice from actual pooled buffer sizes — before the flag can be deleted.
  (A per-routine max-numel fold remains the fallback for *position* expressions the
  analysis cannot otherwise bound; value-embedded launch params instead default to
  int64 — see the launch-parameter bullet.)
- Launch-parameter (FFI) widths need no first-class symbols: tinygrad's `DEFINE_VAR` is a
  graph node only because everything there is a UOp; the load-bearing ingredients are
  declared bounds + bind-time validation + a width pin, and OCANNL's
  `Indexing.static_symbol` already carries the bounds slot (`static_range : int option`,
  min 0 implicit). The interval analysis consumes symbols through its environment
  (`symbol → interval`, as `visit_llc` already does concretely): iterators from
  `For_loop`'s `[from_, to_]`, bound symbols from `[0, static_range)`. To add: validate
  the bound value against `static_range` at the `lowered_bindings` assignment (mirroring
  tinygrad's `bind` assert; one host compare per launch); parameter width = int32 when
  the declared range fits, **int64 when `static_range` is `None`**. The per-routine
  max-numel fallback is NOT sound for launch params: value-embedded params (`!@step_n`,
  `uniform_at`-style counters) take runtime values unrelated to any touched node's
  extent, so a small routine could pick int32 and silently truncate a large counter.
  Narrowing an unbounded param requires declaring (and bind-validating) a range.
- The unsigned single-compare trick stays available *at guard sites* as a codegen choice:
  `(uint32_t)(i) < size` implements `0 <= i < size` in one comparison. Signed core
  arithmetic + unsigned-cast guards is strictly more expressive than either pure regime.
  Caveat carried over from the masking discussion: the trick is sound only when the
  possibly-negative value flows directly into the bounds compare — never through
  div/mod/multiply first — so guards must be per-axis, before flat-offset computation.
- Simplifies the planned interval analysis: machine integers and mathematical integers
  agree, so no wrap modeling and no "refuse when the lower bound could cross zero" rule.
- C compilers treat signed overflow as UB, hence assume no wrap — friendlier to loop
  strength-reduction and vectorization than defined-wrap unsigned.

## IR representation: abstract `Index_prec`

The OCANNL translation of tinygrad's `weakint`: a payload-less constructor in `Ops.prec`
(precedent: `Void_prec`), defined as *signed, integral, width unspecified*. Discipline:

- Storable, with canonical widths (revised from an earlier "annotation-only" draft —
  virtual nodes are retrievable by recomputation, so materialization concerns don't bar
  storage): host representation is OCaml native `int` (Bigarray `int` kind), device
  storage is int64; `prec_in_bytes` = 8. Width-abstractness applies to *arithmetic*
  (resolved per routine at codegen); storage width never depends on a routine's
  resolution, so cross-routine buffer sharing is consistent. 63-bit OCaml int reaches
  every index even for large models (byte offsets of any allocatable buffer are far
  below 2^62; the per-node numel contract keeps device values inside it).
- Payoff at the root: `TDSL.range` (and future argmax-style producers) become
  `Index_prec` by construction — virtual in the common case (inlined width-abstractly,
  as the `Embed_index` c_syntax arm already does), recomputed into native-int host
  arrays when observed, int64 when genuinely materialized. This retires the
  range-producer-in-model-precision trap without inference gymnastics, and
  `class_ids_of_int_list` can fill a plain `int` bigarray directly. Explicitly narrowed
  concrete storage (e.g. uint32 ids at half the footprint) remains an ordinary
  precision choice; both read into index arithmetic via the usual boundary casts.
- Lowering/optimization need only signedness + integrality, which the abstract prec
  carries by definition; width is exclusively a codegen concern. `Embed_index` is
  *already* width-abstract in the IR (`axis_index` has no precision; the c_syntax arm
  resolves `index_prec ()` at codegen) — the only early width commitments today are
  optimization-created guards (the gather's int64/uint64 branches become `Index_prec`)
  and the `Get_dynamic` cast target.
- Backends resolve it at emission, tnode-granularly via `interval_of` + Tnode bounds (see
  design points), with a per-routine max-numel fold as the fallback for unbounded
  expressions and as the launch-parameter FFI width; kernel-parameter and launch-site
  types agree automatically since both come from the same resolution. `Ops.index_prec ()`
  becomes this resolution rather than a settings read. (An earlier draft deferred
  per-expression narrowing as bug-prone; that held only without intervals — widths derived
  from proven bounds make the mixed-width joins mechanical, per tinygrad.)

## Migration inventory (churn accepted)

- `ops.ml`: `index_prec`.
- `c_syntax.ml`: loop counter emission, `Embed_index` / `Get_dynamic` cast targets.
- Backend builtins (`builtins.c`, `builtins_cuda.ml`, `builtins_metal.ml`): conversions
  involving the index precision.
- `low_level.ml`: guards using the `a < b + 1` idiom can revert to `a <= b` -like forms;
  the gather's signed branch can use index precision directly.
- Test goldens: `.expected` files asserting `((uint32_t)(` / `((uint64_t)(` casts
  (e.g. `test/operations/test_one_hot_embedding_lookup`), plus any `build_files` snapshots.
- Audit: pool slot types and buffer-offset arithmetic stay 64-bit-safe; the `large_models`
  retirement must not regress Metal pool-slot width (`pool_slot_msl_typ` and the
  large-model slot-width test) — make slots always 64-bit or add a separate allocator
  knob before deleting the flag.

## Sequencing

**Blocks on [interval-analysis-scalar-t](interval-analysis-scalar-t.md)** (revised
2026-07-03; previously "independent"). The two are independent at their cores — intervals
over the current IR need no wrap modeling because physical padding keeps all emitted index
arithmetic non-negative by construction — but implementing this migration before intervals
would mean an interim per-routine width scheme and paying the `.expected` golden churn
twice. Landing after intervals lets width selection ship in its final tnode-granular form
in one pass. Nothing correctness-urgent is delayed: the unsigned wrap traps only bite when
logical-padding masks introduce negative offsets, and masks are sequenced after intervals
regardless. Still before or together with masks-replacing-physical-padding (negative
offsets are the first real consumer of signedness).

## Relations

[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (blocks this; also simplified
by this once masks introduce negative offsets);
[schedule-ir-optops](schedule-ir-optops.md) (Padto masks are the first consumer of
negative index arithmetic); gh-343 integer-guard flavors in `build_guarded_gather`
(2026-07-03) as the transitional state.

## Acceptance criteria (for the elaborated proposal)

- [x] Decided: int32 default, overflow excluded by the per-node numel contract (see design
      points); interval-analysis dependency landed first as sequenced. Tnode-granular
      interval-driven width selection remains future work; until then the global switch is the
      fallback width and `large_models` is NOT yet deprecated (see Deferred above).
- [x] Enforcement site specified and implemented: dims-lazy wrapper in `Tnode.create` (and the
      eager/`create_with_reshape` variants) validating padded numel once per node.
- [x] Inventory verified by grep (2026-07-04): `ops.ml:index_prec`; `c_syntax.ml`
      `arg_int_prefix`/`loop_index_type` + `Get_dynamic`/`Embed_index` cast targets (both track
      `index_prec` and needed no per-site change); `cuda_backend.ml` and `metal_backend.ml`
      overrides of the same; CUDA `convert_precision` uint4x32 counter arms (needed new
      int32/int64 arms — cc and Metal use `prec_string`-generic naming and the
      `int32/int64_to_uint4x32` builtins already existed in all three backends);
      `low_level.ml` gh-133 `a < b + 1` guards (kept: there is no `Cmple` primitive, and the
      form is correct — merely no longer load-bearing against wrap); gh-343
      `build_guarded_gather` guard precisions (unchanged by design: uint64/int64/double are
      chosen by the ids storage precision, not the index precision); Metal
      `pool_slot_msl_typ` (kept unsigned).
- [x] Guard-form policy: signed two-compare everywhere today (no mask consumers exist yet);
      the unsigned-cast single-compare trick remains available at guard sites once masks land,
      restricted to per-axis guards before flat-offset computation (a possibly-negative value
      must flow directly into the bounds compare, never through div/mod/multiply first).
- [x] Golden churn measured: 8 `.expected` files (7 loop-counter-type goldens across
      `arrayjit/test` and `test/operations`, incl. the Metal variant) plus
      `test_index_prec.expected` and the cast assertions inside
      `test_one_hot_embedding_lookup.ml`.
