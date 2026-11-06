# gh-ocannl-481: CUDA tensor-core profile completeness — what landed

Outcome record for [ahrefs/ocannl#481](https://github.com/ahrefs/ocannl/issues/481), the T4 ceiling
chase of [tensorize-mma](../../../docs/design/tensorize-mma.md) plus the fp8 profile gaps. The design resolution for
item 3 is [gh-ocannl-481-item3-ldmatrix](gh-ocannl-481-item3-ldmatrix.md); this file records what
was implemented against it, what deviated, and what is still blocked and why.

Everything below was verified on an RTX 5070 Ti Laptop (sm_120, driver 610, CUDA 13) — the parity
legs are bitwise, so they test element mappings rather than tolerances.

## Item 1 — fp8 transposed-storage operands (`ta`/`tb`): landed

The e5m2 `m16n8k32` inline-PTX arm declined `ta`/`tb` in v1 because its per-lane byte gathers
hardcoded the roles' own layouts. Unlike wmma, where transposition is a fragment-layout constraint,
that path builds every `.b32` register byte by byte from an explicit address, so the flags were
bookkeeping. Both gathers now route through an `elem_at ~ld ~transposed ~row ~col` helper — the
same shape the bf16 arm already used — leaving the fragment content unchanged and only its
addresses transposed.

Why it mattered: `dA = g·Bᵀ` and `dB = Aᵀ·g` are exactly the transposed layouts, so until this
landed fp8 *training* silently scalar-fell-back on two of its three GEMMs.

Pinned by `schedule_mma_matmul`'s `f8` / `f8_ta` / `f8_tb` legs (e5m2-exact inputs, f32
accumulation ⇒ bitwise parity in every orientation).

## Item 2 — e4m3: still blocked, unchanged

Not attempted, and this issue never proposed it. The emission delta is small (`mma.sync`'s
`m16n8k32` encoding takes the element kinds per operand, and `mma_format_tiles` /
`mma_staged_layouts` are keyed by operand-pair triples, so mixed e5m2×e4m3 needs no interface
change); the real work is the precision type itself — `ops.ml` plus conversions in `builtins.c` /
`builtins_cuda.ml` / `builtins_metal.ml` per the established checklist. `Backend_intf` needs only
a constructor next to `Mma_fp8_e5m2` when that happens.

## Item 3 — `ldmatrix` over swizzled staging: landed (D1–D3, D4.1)

### D1 — the layout contract is a typed node mark

`Low_level.optimized.swizzled` went from a set to a `Tn.t -> swizzle_kind` map, and
`Schedule.Stage.swizzle` from a bool to a `swizzle_kind option`. `Swizzle_b128` XORs the 16-byte-unit
index of the column instead of the element index, leaving the offset within a unit alone — the
layout `ldmatrix` wants, since its per-phase row addresses are 16-byte-aligned and only a remap that
keeps 16-byte units intact can both de-conflict them and stay loadable.

The IR stays oblivious for the same reason it already did: each flavor is a bijection per row, so
`pp_tn_offset` remains the single place that knows, and correctness never depends on which rendering
fired.

**Deviation from the doc, deliberate.** The doc stated `Swizzle_b128`'s extent rule as an addition to
the element flavor's ("power-of-two minor dim, *additionally* a 16-byte multiple"). It is not an
addition: the two count different things and neither implies the other. A 12-element f32 row is 3
units — rejected by b128, and by the element rule too, but a 24-element f16 row is also 3 units while
being a fine element extent; conversely a 4-element f32 row is a legal power of two but only one
16-byte unit, with nothing to permute. They are therefore separate validations, and `Swizzle_b128`
admits minor extents the element flavor rejects.

The kind enters the canonical schedule digest (the two flavors are different physical layouts
consumed by different renderings, so a cached winner must never alias across them), with the element
flavor keeping its bare rendering so pre-change digests stay valid.

### D2 — `ldmatrix` on the inline-PTX arms

The emission hooks' operand tuples grew a `layout` component next to `space`, and that component is
the entire `Stage` → emission contract: the emission never re-derives the layout.

`Swizzled_b128` reaches a hook only when the access is reconstructible from `(ptr, ld)` alone — a
rank-2 node addressed from its origin whose minor dim is `ld`, which is exactly the Stage-minted
operand tile (batched sites included, since `Stage` mints rank-2 tiles there too). Everything else —
the element flavor, a b128 tile addressed at an offset — declines centrally with a named reason.
Accepting a swizzled operand is a promise that it was read through a swizzle-aware load, so the
caller records `Mma_intrinsics_ldmatrix` on that basis and the gh-476 sweep can tell the two load
paths apart.

Coverage by arm:

| arm | A | B |
| --- | --- | --- |
| bf16 `m16n8k16` | `.x4` / `.x4.trans` | `.x2.trans` / `.x2` |
| fp8 `m16n8k32` | `.x4` when `ta = false` | `.x2` when `tb = true` |
| wmma (f16, bf16→f32, tf32) | declines | declines |

bf16 takes all four shapes because `.trans` transposes each 8×8 tile on distribution, which is
exactly the difference between an operand stored in its role's orientation and its transpose. fp8's
eligibility is one-sided per operand and the sides are opposite: `ldmatrix.b16` moves 16-bit units,
so it can build a register holding 4 fp8 bytes only when those bytes are contiguous — 4 consecutive
`k` at fixed `m` for A, 4 consecutive `k` at fixed `n` for B. The 8-bit `ldmatrix` forms that would
lift the other side are Blackwell-only. wmma declines because its fragments are opaque and cannot be
fed from `ldmatrix` destination registers, which is what the doc predicted.

b128 shared tiles get a 16-byte alignment attribute on their declaration: row starts are 16-byte
multiples by the `Stage` validation, so aligning the base is what makes every row address aligned.

`test/operations/schedule_ldmatrix_matmul.ml` pins seven legs including the fp8 wrong-side and the
f16/wmma declines. A nonzero guard accompanies each parity check — a fragment mapping that read
outside the staged block would plausibly produce all zeros, and zeros compare equal to zeros.

### D3 — capability gate and seeding

`mma_capability` grew `mma_staged_layouts`, and each staged GPU mma seed gains a twin marking both
operand tiles `Swizzle_b128` — same tile sizes, same everything else, labeled `swz-b128` so the two
are distinguishable in a report rather than looking like one candidate timed twice.

**Deviation from the doc, deliberate.** The doc specified `mma_staged_layouts : [`Swizzled_b128 ]
list` — a per-backend flag. It is keyed by format triple instead, like `mma_format_tiles`, because
eligibility is per operand *and* per orientation and the orientation the staged sketches mint is each
role's own. CUDA can feed fp8's A from `ldmatrix` at that orientation but not its B, so a flat flag
would seed fp8 twins that render the scalar fallback under a tensorized label — precisely the
gh-479 failure the accumulator key was introduced to stop. The b128 extent rule is checked at seeding
for the same reason: an inapplicable twin should not exist, not merely fail.

`render_mma_fragment_scope`'s blanket swizzle decline relaxed to "decline only what has no load
form". wmma-based fragment scopes still decline b128 per call, and that decline is exactly what
routes a swizzled staged bf16 leg through the caller's target aliasing to the inline-PTX arm. This is
the gh-480 rebase point.

### D4.1 — the family-arch marker mechanism

The nvrtc arch-flag policy moved out of `cuda_to_ptx` into a pure
`Cuda_backend.gpu_arch_options ~device_cc`, and gained the target kind it lacked. Every existing
marker selects a **floor** — the lowest arch whose PTX contains the instruction — because
floor-targeted PTX is forward-JIT-compiled on every later GPU, which is why a triggered floor is
never raised to the device arch. `(mma-mxfp8)` selects a **family** target (`compute_120a`-style)
instead, loadable only by the family it names, and is therefore the one marker gated on the attached
devices' own family: family PTX is never produced for a device that could not load it.

No arm emits it yet, so `arrayjit/test/test_cuda_arch_flags.ml` pins the mechanism — the marker
selects a family target on a family device, does not on any other, and changes nothing for the floor
markers. The policy is pure, so the test needs no GPU (same `select` stub pattern as
`test_cuda_classify_failure`).

### D4.2 — block scaling: still blocked

Not attempted. `kind::mxf8f6f4` computes `d = (A·2^SFA)(B·2^SFB)` with e8m0 scale factors per
32-element block: extra mma *operands* with their own layout, for which `Tile_mma` has no slot and
OCANNL has no microscaling storage story. A unit-scale arm would be numerically identical to the
plain fp8 path while forfeiting forward-JIT, so it buys nothing measurable. Same blocker class as
item 2; revisit when a quantization policy (gh-492's codomain growing an mx format) or an e4m3/e8m0
precision lands.

### f16 and tf32 `mma.sync` arms — not attempted

D2 sequenced these last ("if measurements justify them"), and nothing measured yet does. They would
be mechanical clones of the bf16 arm; both currently render only through wmma and therefore decline
swizzled operands.

## Item 4 — `Stage ~pad_stride`: landed

`Stage` grew `pad_stride : int option`, rounding the tile's minor dim up to a multiple. The tile's
leading-dimension stride *is* that dim, so this changes the stride while the iterated index space
stays the unpadded extents.

The padded slots hold nothing under a row-major layout — no loop reaches them, so they are neither
written nor read, and the `zero_fringe` contract (about the fringe of the staged *source* region
within the iterated space) is unaffected. Under a swizzle they do carry data, the XOR being a
bijection of the whole padded row, and reads go through the same map.

D5 is the one place items 3 and 4 touch, and the validations share its order: `Swizzle_b128`'s
16-byte-unit rule is checked on the **padded** dims, so a tile that misses it is lifted over it
rather than declined.

**Where the issue's coverage claim does not currently reach.** The issue expected `pad_stride` to
convert wmma's leading-dimension-stride declines into emissions. For a rank-2 staged tile that
cannot happen today: the tile's minor dim equals a tensorized extent, which `Tensorize`'s role rule
and the intrinsic tile already force to a multiple of 8 (16-bit) or 4 (f32). The reachable payoffs
are the bank conflicts (proposal §5's original motivation) and the b128 rule above. Autotune
therefore seeds no `pad_stride` variants yet; a stride-decline inventory from gh-479 would be the
evidence that changes this.

## Measured (2026-08-04): rendered, correct, neutral

The gh-476 sweep ran on rog-nv (RTX 5070 Ti, sm_120) over `mlp_small` and `mlp_wide` at f32 and
bf16 — full protocol and numbers in [benchmarks/report-gh481-cuda.md](../../../benchmarks/report-gh481-cuda.md).
Summary:

- **The mechanism works end to end.** Per `mlp_wide` bf16 search: 12 distinct swizzled twins
  seeded (staged seeds only — unstaged ones are correctly never twinned), 12 timed, 0 `Tile_mma`
  statements falling back to the lane-0 scalar rendering, 0 declines naming a swizzled layout.
  `ldmatrix` rendered.
- **The layout is worth ~nothing here.** Pairing each twin with the plain sibling it directly
  follows in the same search process — identical tile sizes, identical rest-of-pipeline, only the
  layout differs — gives n = 48, **median +0.20%, stdev 0.50%, range [−1.1%, +1.3%]**.
- **It cannot reach the shipping artifact on these shapes anyway.** The crowned candidate is always
  the *unstaged* tensorized family (`sk_bk = 0`, 0.93–0.95 ms); swizzled twins exist only in the
  staged family, which times 1.54–1.56 ms — and whose plain siblings time the same. The staged
  family loses by ~65%, so its tiles' layout is a question about a schedule that does not ship.
- **A whole-cell before/after could not have answered this.** The f32 tuned cell is a negative
  control (no twins are seeded at f32; both binaries run identical code there) and it varies by
  19.8% on `mlp_wide`, 40.6% on `mlp_small`, because the beam does not always crown the same
  family. Any headline A/B smaller than that is search noise wearing a result's clothes.

So T4's acceptance rule — a measured win on at least one leg — is **not met**, and the honest
status of this work is "the emission exists, is correct, and is inert on the schedules today's
search selects".

## Where the next person looks

- `ldmatrix` reaches the artifact only once *staged* beats *unstaged* on some shape. Candidates:
  larger `k`, or a reduction that must be blocked for residency; gh-480's accumulator residency
  making the staged form cheaper per k-block; a workload arithmetic-intense enough that shared-tile
  bandwidth rather than launch/epilogue overhead is the constraint. Re-run the sweep there before
  concluding anything about the layout.
- If the swizzled twin loses *within* the staged family on such a shape, the interesting question
  is *which* half: `ldmatrix` with a plain (bank-conflicted) tile is not expressible today, and
  separating the instruction's win from the layout's would need that arm.
- The instrument to reach for is the in-process twin pairing, not the cell p50. It resolves ~0.5%;
  the cell resolves ~20%. Anything reported from the latter needs the f32 control beside it.
- gh-480 and this item both restructure the emission's load path; the rebase point is D3's decline
  relaxation, pinned by the staged-half residency check of `schedule_mma_matmul.ml`.
- gh-485 pad masks compose: padded staged tiles are zero-fringe and the XOR remap applies to the
  padded dims (`pp_tn_offset` sees the tile's real dims), so masked edges stay exact zeros read
  through swizzled offsets.
