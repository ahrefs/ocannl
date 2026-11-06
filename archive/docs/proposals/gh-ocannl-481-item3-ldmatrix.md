# gh-ocannl-481 item 3: `ldmatrix` over swizzled shared tiles; Blackwell block-scaled `kind::mxf8f6f4`

> **Status (2026-08-04): implemented, D1–D3 and D4.1.** Steps 1–5 of the plan below landed; step 6
> (f16/tf32 `mma.sync` arms) was not attempted, and step 7 (the gh-476 sweep) has not run — every
> emission here is verified for correctness on hardware, none of it yet for speed. Two decisions
> deviate from this document on purpose, both recorded with their reasons in
> [gh-ocannl-481](gh-ocannl-481.md): `Swizzle_b128`'s extent rule is a SEPARATE validation from the
> element flavor's rather than an addition to it (neither implies the other), and
> `mma_staged_layouts` is keyed by format triple rather than being a per-backend flag (eligibility
> is per operand and per orientation, so a flat flag would seed fp8 twins that render the scalar
> fallback under a tensorized label). Read that file for what the tree actually does; read this one
> for why.

Design resolution + implementation handoff for the T4 ceiling chase of
[tensorize-mma](../../../docs/design/tensorize-mma.md). Written against the state of the tree after gh-ocannl-528
(batched matmul sites; `Tile_mma` carries explicit `ldd`/`lda`/`ldb` leading-dimension strides).
The decisions below are resolved; the coder should implement, not re-litigate — but each records
its rationale so a contradicting discovery has something concrete to argue with.

## What exists today (the parts this builds on)

- **Cooperative staging**: `Stage { shared = true; cooperative = Some w }` writes row-major shared
  tiles under a fresh lane loop; the operand tiles reaching `Tensorize` are Stage-minted rank-2
  contiguous nodes (this holds for batched sites too — staging normalizes the layout, gh-528).
- **Swizzle mark, element-XOR flavor**: `Stage { swizzle = true }` marks the tile in
  `optimized.swizzled`; `C_syntax.pp_tn_offset` remaps `P*C + col` to `P*C + (col ^ (P & (C-1)))`.
  Every intrinsic path (`mma_syntax`, the fragment scope, the register tiling) *declines* swizzled
  operands into the swizzle-aware scalar fallback. Autotune deliberately never seeds `swizzle`
  with `Tensorize` — it would trade the intrinsics for a bank-conflict fix.
- **CUDA emission, two arms**: the `nvcuda::wmma` C++ API (f16, f16→f32, bf16→f32, tf32) with
  `load_matrix_sync` from generic addresses; and inline-PTX `mma.sync` arms (uniform bf16
  m16n8k16, fp8 e5m2 m16n8k32) whose per-lane *byte gathers* build the `.b32` fragment registers.
  Arch floors are selected by source markers (`(wmma-bf16)`, `(mma-fp8)`, …); the fp8 arm pins
  `compute_89` so the driver forward-JITs its PTX on later GPUs.
- **Accumulator residency** (gh-480): `contract_tensorized_accumulator` + the
  `render_mma_fragment_scope` recognition; swizzled a/b operands currently decline the fragment
  scope.

`ldmatrix` is the warp-cooperative shared-memory load: one instruction loads 8×8 tiles of 16-bit
elements (`.x1/.x2/.x4`, optional `.trans`) into the exact per-lane fragment layouts `mma.sync`
consumes, with each thread supplying one 16-byte-aligned row address. Its throughput win over
per-lane gathers is real only when the shared tile is laid out so the 8 row addresses of each
phase hit distinct banks — the CUTLASS-style XOR swizzle at 16-byte granularity, not the
element-XOR flavor we have.

## Decisions

### D1. The layout contract is a typed node mark shared by `Stage` and the emission

Extend the swizzle mark from a set to a kind: `Low_level.optimized.swizzled` becomes a map
`Tn.t -> swizzle_kind` with

```ocaml
type swizzle_kind =
  | Swizzle_elem          (* today's element-granularity XOR: col ^ (P & (C-1)) *)
  | Swizzle_b128          (* 16-byte-unit XOR: the ldmatrix/vectorized layout *)
```

and `Stage.swizzle : swizzle_kind option` (replacing the bool; the three existing validation
errors keep firing for both kinds; `Swizzle_b128` additionally requires the minor dim to span a
multiple of 16 bytes, i.e. `C * prec_bytes % 16 = 0` — with item 4's `~pad_stride` as the future
knob for shapes that miss it).

`Swizzle_b128` remaps the linear offset `P*C + col` by XORing the *16-byte-unit index* of the
column with the low bits of `P`: with `u = 16 / prec_bytes` elements per unit and `U = C / u`
units per row, `col -> (((col / u) lxor (P % U)) * u) + (col % u)`. Same bijection-per-row
argument as the element flavor, so the IR stays oblivious; `pp_tn_offset` renders it with shifts
and masks (`U`, `u` are powers of two by the validation above). The Stage copy nest and every
scalar fallback read the same offsets, so correctness never depends on which rendering fired —
exactly the existing discipline.

Rationale for keeping this codegen-level rather than an IR term: unchanged from the swizzled
staging section of tensorize-mma.md — XOR is not in the affine index algebra and does not need to
be.

### D2. `ldmatrix` lands on the inline-PTX `mma.sync` arms, not under `nvcuda::wmma`

wmma fragments are opaque; there is no supported way to feed them from `ldmatrix` destination
registers. The inline-PTX arms already own per-lane fragment registers, and their gather loops
are precisely what `ldmatrix` replaces. So:

- The operand tuples the hooks receive grow the layout:
  `(ptr, ld, space)` becomes `(ptr, ld, space, layout)` with
  `layout : [ `Plain | `Swizzled_elem | `Swizzled_b128 ]` (derived from the mark; `C_syntax`
  computes it where it already computes `space`). This is the *entire* Stage→emission contract:
  the emission never re-derives the layout, it trusts the mark it is handed.
- In the bf16 (m16n8k16) and fp8 (m16n8k32) arms: when an operand has `space = `Shared` and
  `layout = `Swizzled_b128` (and extents/alignment fit), emit
  `ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16` sequences computing each lane's row address
  through the same XOR formula; otherwise keep the existing byte gathers (which remain correct
  for `` `Plain `` shared tiles and device pointers). Mixed operands are fine — per-operand
  choice, one statement.
- `` `Swizzled_elem `` stays a decline for all intrinsic arms (unchanged semantics for existing
  schedules and caches).
- f16 and tf32 currently render only through wmma. They inherit `ldmatrix` only by growing their
  own `mma.sync` arms (m16n8k8/m16n8k16) — mechanical clones of the bf16 arm. Do bf16 first
  (benchmark legs exist), fp8 second, f16/tf32 arms after, each measured.

The `mma_census` grows a rendering constructor (e.g. `Mma_intrinsics_ldmatrix`) so tests and the
gh-476 sweep can pin which path fired; `declinef` diagnostics name the reason when the layout or
alignment forces the gather path (the gh-479 discipline).

### D3. The decline inversion is capability-gated, seeding follows emission

`Backend_intf.mma_capability` grows `mma_staged_layouts : [ `Swizzled_b128 ] list` (empty
everywhere except CUDA once the emission lands; Metal banks but has no ldmatrix analogue — a
later `simdgroup`-era entry would reuse the field). Autotune's *staged* mma sketches
(`sk_bk > 0`) then seed a swizzled twin per staged seed — same tile sizes,
`Stage { swizzle = Some Swizzle_b128 }` on both operands — only when the capability advertises
it, labeled (e.g. suffix `swz`) so timings distinguish the twins. Unstaged seeds are unaffected
(no shared tile to swizzle). The tuner, not a heuristic, decides whether the swizzled twin wins;
this is the same "propose both, measure" pattern as hoisted packing.

`render_mma_fragment_scope`'s blanket `is_swizzled` decline on a/b relaxes to "decline only
`Swizzled_elem`" once the arms handle `Swizzled_b128`; the accumulator (`d`/fragment) is
register-resident there and never swizzled. This is the gh-480 interaction: whoever lands second
rebases, and the pin is the staged-half residency check of `schedule_mma_matmul.ml`.

### D4. Blackwell `kind::mxf8f6f4` is split: arch-marker mechanism now, block scaling blocked

Two separable halves:

1. **Per-arch family targeting** (mechanism, land with this item): a source marker (reserve
   `(mma-mxfp8)`) that makes `cuda_to_ptx` target the *device family* arch (`sm_120a`-style,
   `--gpu-architecture=compute_120a`) for kernels carrying it — unlike every existing marker,
   which selects a forward-JIT-able floor. Gate at seeding AND emission on the device family, so
   the family-specific PTX is never produced for a device that cannot run it; everything without
   the marker keeps floor targeting and loses nothing. This resolves the arch-flag tension
   recorded in the proposal and in `cuda_backend.ml`'s fp8 comment.
2. **Block scaling itself** (blocked, do not attempt): `kind::mxf8f6f4` computes
   `d = (A · 2^SFA)(B · 2^SFB)` with e8m0 scale factors per 32-element block — extra *operands*
   with their own layout, not a precision reinterpretation. `Tile_mma` has no slot for them, and
   OCANNL has no mx/microscaling storage story (this is the same blocker as item 2's e4m3: the
   emission delta is small, the precision/quantization type is the real work). A unit-scale arm
   would be numerically identical to the plain fp8 path while forfeiting forward-JIT, so it buys
   nothing measurable until scales exist. Record the dependency: revisit when a quantization
   policy (gh-492's codomain growing an mx format) or an e4m3/e8m0 precision lands.

Consequence: this item's deliverable on Blackwell is the *marker mechanism* plus `ldmatrix`
(which sm_120 benefits from immediately via the existing e5m2/bf16 arms); `kind::` forms wait.

### D5. Item 4 (`Stage ~pad_stride`) stays independent but shares one constraint

`~pad_stride` pads the tile's minor dimension; `Swizzle_b128` requires the (possibly padded) row
byte-length to be a multiple of 16 and a power of two in 16-byte units. Validation for both lives
in `Stage`; when both are requested, pad first, then check. Nothing else couples them — do not
sequence one behind the other.

## Implementation plan (suggested order, each step green on its own)

1. `low_level.ml`/`.mli`: `swizzle_kind`, `swizzled` set→map (`sexp_of`, digest in
   `schedule_cache.ml` — the kind must enter the canonical digest). `c_syntax.ml`:
   `pp_tn_offset` b128 formula; operand-tuple `layout`; census constructor. All existing
   behavior = `Swizzle_elem`.
2. `schedule.ml` `Stage`: field type change + b128 validation; update
   `schedule_swizzle_matmul.ml` and any other `swizzle = …` sites mechanically
   (`true` → `Some Swizzle_elem`).
3. `cuda_backend.ml`: `ldmatrix` in the bf16 arm behind `layout = `Swizzled_b128`; hardware
   parity on the staged+tensorized bf16 leg (extend `schedule_mma_matmul.ml` or a new
   `schedule_ldmatrix_matmul.ml` with structural `ldmatrix.sync` pins + bitwise parity — bf16
   inputs exact-by-construction like the existing legs). Then the fp8 arm (its byte gathers have
   the most to gain).
4. `backend_intf.ml` capability + `autotune.ml` swizzled staged twins + census/label; pin with a
   `tile_mma_declines.ml`-style synthetic-capability test (mechanism, not current arm set).
5. `(mma-mxfp8)` family-arch marker in `cuda_to_ptx` (no consumer yet; a unit test on the
   arch-flag selection suffices).
6. f16/tf32 `mma.sync` arms if the bf16/fp8 measurements justify them.
7. Measure via the gh-476 sweep (`bench_mlp` bf16, `bench_gpt` once gh-528's seeds put
   attention/ffn GEMMs on tensor cores); acceptance is a measured win on at least one leg and no
   regression elsewhere — T4 is "driven by benchmarks, not speculation".

Interaction warnings for the coder:

- gh-480 (`render_mma_fragment_scope`) and this item both restructure the emission's load path —
  rebase point is D3's decline relaxation.
- gh-485 pad masks: padded staged tiles are zero-fringe; the XOR remap must be applied to the
  *padded* dims (it already is — `pp_tn_offset` sees the tile's real dims), so masked edges stay
  exact zeros read through swizzled offsets. A test leg with a padded, swizzled, staged tile is
  cheap insurance.
- Schedule-cache digests change wherever the `swizzled` mark's representation enters them — bump
  or re-derive; a cached winner replayed across the change must not silently alias the other
  layout kind.
