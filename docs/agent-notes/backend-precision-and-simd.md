# Precision, narrow formats and CPU SIMD

The storage-vs-compute seam, the software codecs, accumulator widths, and vector codegen quality.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- `check_half_prec_constants_cutoff` (`Ops.exceeds_fp16_cutoff`, enforced from
  `Low_level.simplify_llc.check_constant` during lowering, hence backend-independently) is a
  HEADROOM policy, not a representability check: its default 2^14 sits far below fp16's 65504 max
  finite, so a constant it rejects may be perfectly representable. Read "too big for FP16" twice
  before believing it names an overflow — the one message covered two opposite defects at once
  (gh-ocannl-547/548), and fixing either alone just moves the failure to the other. Reduction
  identities are out of scope by construction (`Ops.neutral_elem`: `Max` → `-inf`, `Min` → `+inf`),
  exempted via `Float.is_finite`: they are sentinels arithmetic consumes, exactly representable, and
  every backend converts them per IEEE (`__float2half` / `__double2half` / a `(half)` cast). Attention
  masks fill with `Nn_blocks.default_mask_fill` = `-inf` (per-call `?mask_fill` for the one case that
  needs finite: a mask that can cover a whole row, where `-inf - -inf` would give NaN) rather than a
  large finite magic number, so the fill needs no per-precision tuning.
- The `bf16_ops`/`half_ops` convention of picking inputs that are exactly representable in the
  reduced precision, so printed numbers stay backend-uniform, does NOT extend to transcendental
  results: no choice of inputs makes an `exp` output exactly representable, and backend libm
  implementations disagree in the last mantissa bit (HIP's `exp` gives `2.1215e-1` where cc, metal
  and cuda give `2.1228e-1` — one ulp at 2^-13, found only by running `half_softmax` on real ROCm
  hardware). A reduced-precision golden containing `exp`/`log`/`tanh` output must therefore print
  coarsely enough to sit above an ulp and carry its numeric content in a tolerance comparison against
  a double-precision reference computed in the test, not in the printed digits.

- Metal has no fp8 *type* either, but fp8 works: e5m2 stores as a byte and computes in f32 through
  the `Builtins_metal` software codec, i.e. Metal takes gh-ocannl-517's storage/compute seam for
  that one format (`compute_prec`), the way `cc` takes it for every narrow float. The codec is
  bit-identical to `builtins.c`'s for all 2^32 floats and all 256 codes — verified exhaustively
  off-tree, which is how a hand-written float codec should be checked, not by sampling — and it is
  written in integer/bitcast form on purpose: Metal compiles with fast math by default, under which
  the infinity and NaN branches of a float-arithmetic codec are not reliable. fp8 rounding is portable
  too, but only because ours was changed to theirs — see the next entry.
- fp8 (e5m2) NARROWING is one rounding everywhere, and getting there meant changing ours, not
  theirs (gh-ocannl-646). A float-to-e5m2 conversion decides four things, and the software codec
  (`builtins.c`, `Builtins_cc`, `Builtins_metal` — and, through `Ops.single_to_fp8`, the HOST side
  of every backend) decided all four differently from `__nv_fp8_e5m2` / `__hip_fp8_e5m2`: ties away
  from zero instead of to even, subnormals flushed instead of rounded into (so code `0x01` was
  unreachable by narrowing), the sign of a zero dropped, and finite overflow going to infinity
  instead of saturating to 57344. Since the host side is backend-independent, CUDA and HIP each had
  a host-vs-device split inside one tensor. The codec now does what both vendors do. They disagree
  with each other on exactly two inputs — an already-infinite input (CUDA saturates, HIP keeps it
  infinite) and the sign of a NaN (CUDA drops it) — so those are asserted only up to what all four
  agree on.
  Two reusable lessons from how this was settled. Verify a codec against the HARDWARE, not the
  vendor docs: a kernel sweeping all 2^32 float bit patterns against a candidate runs in seconds on
  either GPU box, and that is what turned "I believe both use RNE" into an exact difference set.
  And when both sides of a parity check run the same code — cc and Metal narrow on the host and on
  the device through this one codec — the check is vacuous by construction, so
  `test_fp8_codec_parity` also pins a GOLDEN of the narrowed values, which is what those two
  backends can actually fail.
- A test that means to exercise a NARROWING conversion has to pin the source precision, or it
  exercises nothing. Two separate mechanisms make the conversion disappear, and both were hit
  writing `test_fp8_codec_parity`. Precision inference flows the destination's precision BACKWARDS
  into the source, so `dst:fp8 =: src` lowers to a byte copy between two fp8 buffers with the
  narrowing having happened on the host when the source was filled — pass `~top_down_prec:false`
  AND `Tnode.update_prec src single`. And a virtualized source inlines its cells as literals, at
  which point the conversion is a constant expression the BACKEND COMPILER folds on the host, so
  a device-side defect cannot show — `Train.set_materialized` the source. Both failure modes look
  identical from the outside: the test passes on every backend, which is what one wants to see.
  The only reliable checks are to read the emitted kernel (`dst[i] = single_to_fp8(src[i])`, source
  declared `float *`) and to run the mutation — with HIP's guard forced off the leg must FAIL, and
  it did not until both mechanisms were closed.
- Narrowing f64 to fp8 must not go through f32 (gh-ocannl-648). Rounding twice moves a double
  that sits just off an f32 tie ONTO that tie, and the second rounding then breaks it by a rule the
  first has already made wrong — so `single_to_fp8((float)x)` disagreed with the GPU fp8 types,
  which convert straight from the double. Not a tie-rule question and not new: under the old
  tie-away rule the same seam disagreed on the mirror inputs (`x-eps` instead of `x+eps`).
  `Ops.double_to_fp8` is the one-step codec, used by the C backends' `Double_prec` arm and by the
  host side — where it matters most, since an OCaml float IS a double, so every host-side fp8 write
  was double-rounding. Verified against `__nv_fp8_e5m2` over 17.2 billion finite doubles (all 2^32
  top-halves crossed with four low-half patterns, ties included) and against the f32 codec over all
  2^32 f32-exact doubles. Metal needs none of it: its `double` is `float`.
- ROCm's fp8 narrowing is broken for tiny magnitudes and the HIP backend guards it
  UNCONDITIONALLY (gh-ocannl-647). `(__hip_fp8_e5m2)(float)` returns up to 2^-14 where the answer
  is a signed zero, for magnitudes around 4e-25 to 3.3e-24 — an out-of-range shift in
  `hip/amd_detail/amd_hip_fp8.h`'s `cast_to_f8` (`exponent_diff` reaches 85 for f32; both
  `mantissa >>= exponent_diff` and the `midpoint` mask shift a 64-bit value by ≥ 64, and the shift
  is taken mod 64). **The mod-64 is visible in the data**: the defect recurs with PERIOD 64 in the
  input's binary exponent, four adjacent exponents at a time. Exhaustive gfx1151 sweeps, ROCm
  7.14.60850, now reproducible in a minute with `tools/fp8_soak.exe --arm=hip --spelling=raw`:
  67108862 of 2^32 float patterns wrong, confined to f32 exponent fields 46–49 (2^-81..2^-77 —
  the only member of the family an f32 can represent); over 17.2e9 doubles, 503316450 wrong on 60
  exponent fields, which are fifteen groups of four spaced exactly 64 apart, from 2^-81..2^-77 down
  to 2^-977..2^-973. The "478 more from the double path, the f32-subnormal magnitudes ~4.5e-44 to
  1.8e-43" recorded here before gh-ocannl-757 was the SECOND member seen through an f32-exact-double
  sweep, not a separate phenomenon. The one-line invariant: a magnitude 2^m is affected iff
  `m <= -78 && (-78 - m) mod 64 < 4`. It reproduces on the HOST too (the header's
  software path is what any arch without `HIP_FP8_CVT_FAST_PATH` uses, i.e. everything but
  gfx942/950/1200/1201/1250 — the soak asks the compiled kernel for that macro and prints it, so a
  run on one of those five says which side it swept). CUDA is correct there, and so is our software
  codec. So HIP's ONLY
  float-to-fp8 spelling is `ocannl_single_to_fp8_uniform` / `ocannl_double_to_fp8_uniform`, which
  pre-round everything below half the smallest subnormal to a signed zero — exact, since those
  magnitudes round to zero anyway: the same sweeps report 0 disagreements guarded, on both the f32
  and the f64 entry point. Reported
  upstream at https://github.com/ROCm/rocm-systems/issues/10591 (with a verified two-line clamp),
  which is the guard's removal trigger; there is no ROCm-version predicate because no released
  version is known correct. **The removal check is in the repository**: `tools/fp8_soak.exe
  --arm=hip --spelling=raw` prints the disagreement count descriptively, and that count reaching 0
  is what says a ROCm release has fixed it and the two helpers plus the `fp8_from_prec_fn` funnel
  can go. Its two claims are localization, not agreement — every disagreement is one the guard
  closes (|x| < 2^-17), and all of them sit in the residue class above — so the tool passes on
  affected hardware instead of failing by design, and both claims stay true, vacuously, afterwards. The guard covers both narrowing sites — the conversions AND the
  operator bridges, which narrow an f32 result back to fp8 — through one funnel
  (`fp8_from_prec_fn`); guarding only the conversions was the first version, and a review caught
  it. `test_fp8_codec_parity`'s two underflow legs are therefore unconditional assertions, on HIP
  as everywhere else. It was opt-in under `prefer_backend_uniformity` first, and that flag no
  longer has an fp8 clause (nor a `/fp8-guard` component in HIP's `codegen_tag`): a
  twenty-order-of-magnitude silent error is not a uniformity preference.
- **Both those sweeps are now in the repository** (gh-ocannl-657), because every defect the three
  bullets above name lived in the tails — the negative-NaN sign, the NaN payload, the subnormal
  boundary, the tie-to-even carry, ROCm's four broken exponents — and sampling would have shipped
  all of them. Two programs, split by what they need:
  - `dune build @test/operations/slow-fp8_codec_exhaustive` — no GPU, ~8 s wall (8 domains, ~38 s
    of single-core work). `single_to_fp8` over all 2^32 f32 bit patterns and `double_to_fp8` over
    17.2e9 doubles (every top half crossed with four low halves, the mantissa's midpoint bit among
    them) against a rounding ORACLE; `double_to_fp8` against `single_to_fp8` over all 2^32 f32-exact
    doubles; `fp8_to_single` over all 256 codes. The oracle is not a second codec — it is the
    format's decode table plus "a code owns the interval between the midpoints to its neighbours,
    ties to the even code", which makes correct rounding a LOCAL property and therefore cheap enough
    to evaluate 21 billion times. Saturation is the one asymmetry: code 0x7B has no upper midpoint.
  - `dune exec tools/fp8_soak.exe` (`--arm=cuda|hip`, `--sweep=f32|f64|both`,
    `--spelling=default|raw|guarded|both`, `--arch=device|backend`) — needs the hardware,
    and answers the one question the CPU half cannot: whether the codec still agrees with the vendor
    type a kernel casts to. The host side is the shipped object code (`builtins.c`, reached from
    `fp8_soak_stubs.c` by `extern`, not transcribed); the device side is `(__nv_fp8_e5m2)x` exactly
    as `Cuda_backend.convert_precision` emits it. RTX 5070 Ti Laptop, CUDA 13.3, 2026-08-23: 6.1 s
    for the f32 sweep (2^32 inputs), 29.5 s for the f64 sweep (17.2e9 inputs), zero disagreements on
    every FINITE input of either — which is the claim, together with a non-vacuity one, that the
    sweep drove the vendor conversion onto all 248 signed finite codes. Radeon 8060S (gfx1151),
    ROCm 7.14.60850, 2026-08-23: 7.5 s and 32 s for the same two sweeps, zero disagreements on every
    finite input of either **with the default spelling**, which on HIP is the guarded one — see the
    next bullet.
- **`--arch` decides what the CUDA soak is measuring**, and the default is not the backend's
  setting. `cuda_fp8.hpp` guards its conversions with `#if __CUDA_ARCH__ >= 890`: at or above sm_89
  the cast is the hardware `cvt` instruction, below it the header's own software emulation. The
  repo's arch policy (`Cuda_backend.gpu_arch_options`) is MARKER-driven — a source with no
  tensor-core markers gets no `--gpu-architecture` at all, so nvrtc's default target applies:
  measured as `__CUDA_ARCH__ = 750` under CUDA 13.3, hence the software path. That is the honest
  answer to "does the codec agree with what the backend emits" (`--arch=backend`), but it is not the
  hardware check gh-ocannl-646's lesson asks for, so the default is `--arch=device`, this GPU's own
  capability. **The two agree bit-for-bit** on all 21.5e9 inputs on compute_120 / CUDA 13.3, NaN and
  infinity classes included — worth knowing, and not something to assume of the next toolkit.
  Which path a run swept is never inferred from the options passed: a `ocannl_report_arch` kernel
  hands back its own `__CUDA_ARCH__`, and that value is printed in the header AND carried in every
  claim's label ("… via the hardware cvt (__CUDA_ARCH__ = 1200)"). That matters below sm_89, where
  `--arch=device` asks honestly for the device's own architecture and still gets the software
  conversion: the run says so rather than reading as a hardware check (Codex P2, round 4). For the
  same reason **the device's capability is not the kernel's compile target on CUDA and must not be
  labelled as one**: under `--arch=backend` a compute-12.0 box compiles for 7.5, so the run header
  derives `target` from the kernel's own `__CUDA_ARCH__` and reports the device's capability
  separately, as `device capability`. HIP is the case where the two coincide — hiprtc given no
  `--offload-arch` compiles for the current default device — and that is stated where it is relied
  on rather than left as an assumption (gh-ocannl-758).
- The soak's NON-finite disagreements are permanent, and it prints them rather than hiding them.
  `±inf` narrows to 0x7B/0xFB under CUDA (saturating) where our codec keeps 0x7C/0xFC: 2 inputs in
  each sweep. A NaN f32 narrows to 0x7F whatever its sign, so 8388607 of the 16777214 NaN patterns
  — exactly the negative ones — disagree with our sign-keeping codec. A NaN f64 is the surprise:
  `(__nv_fp8_e5m2)` of a NaN DOUBLE answers 0x7E/0x7F/0xFE/0xFF, keeping the sign and letting two
  payload bits through, which the float path does not; 4194302 of the 8388606 swept disagree, the
  0x7E/0xFE half. All of them are NaNs on both sides, and the vendor's own two entry points
  disagreeing with each other is why `test_fp8_codec_parity` asserts only NaN-ness there.
- **WHICH SPELLING the soak sweeps is a question only ROCm makes interesting** (gh-ocannl-757).
  CUDA has one narrowing, the bare `(__nv_fp8_e5m2)x` the backend emits. HIP has two, because its
  bare cast is broken (above) and every OCANNL narrowing there goes through a guarded helper. So the
  DEFAULT is what the backend actually emits — guarded on HIP, the cast elsewhere — and that run is
  a pass/fail gate on every box, expecting 0. `--spelling=raw` is the opt-in probe, and it claims
  localization rather than agreement for the reason above. The guarded kernels do not transcribe the
  helpers: they take the source text from `Hip_backend.fp8_guard_source ()`, i.e. from
  `Builtins_hip`, which raises if either helper is renamed — the same discipline as the host side
  reaching `builtins.c` by `extern`. Nothing else in the suite sweeps the guard exhaustively.
- Adding a vendor to the soak is a module of its `ARM` signature, a `vendor` record beside it, and
  one `select` clause in `tools/dune` — not a second program, which is how the CUDA and HIP sweeps
  drifted apart the first time. The HIP arm (`tools/fp8_soak_hip.hipjit.ml`) was written on the CUDA
  box, where the `select` resolves to the stub, and first compiled on a ROCm box a wave later
  (gh-ocannl-757): **every name in it resolved unchanged**, which is the mechanical-mirror
  discipline paying off — what it needed was the things a mirror cannot know, the backend's own
  hiprtc include options (`Hip_backend.hip_include_options`, lifted out of `Impl` exactly as
  `cuda_include_options` was) and a kernel that reports `HIP_FP8_CVT_FAST_PATH` so a run says which
  side of the header's compile-time split it swept.
- **A file behind a dune `select` is compiled only on a box that has the library it selects for**,
  so every edit to one made elsewhere is blind — `@check` on a laptop compiles the `.missing` stub
  and says nothing about the arm. Two fp8-soak arms shipped through a green CI having only ever been
  parsed before this was addressed (gh-ocannl-758). The rule that keeps the exposure small is that
  **an arm holds its vendor kernel source, its compile/load/launch/copy calls, and data extractors,
  and nothing else**: the vendor's name and C type, its narrowing spellings and their labels, the
  probe and header wording, the thresholds that say what a reported macro MEANS, and every claim
  label live in `fp8_soak.ml`, which every box compiles — as the per-vendor `vendor` records beside
  the `ARM` signature. A change to how the soak behaves then never edits an arm, and the `.missing`
  stubs mirror no vendor knowledge at all (`built = false` is what the selection reads; nothing else
  in a stub is reachable, because an unbuilt arm is refused before it is asked anything). The same
  shape applies to any future `select`-gated file.
- **Each arm records where it last really compiled** — `last_compiled`: box, date, and the PR whose
  evidence says so, a PR rather than a sha because the commit a session verifies does not survive
  the rebase before merge — and the run header prints it beside the arm's source path, so an editor
  on another box knows whether they are editing blind and a sweep's output is also the record that
  the file still compiles somewhere.
  Update it when you compile the arm on a box that has its vendor library, which is also the moment
  to run the sweep: `opam exec -- dune build tools/fp8_soak.exe` then
  `./_build/default/tools/fp8_soak.exe --sweep=f32` (add `--spelling=both` on HIP) takes about ten
  seconds on either GPU box and exercises every path an arm has. `dune build @check` alone proves
  the arm compiles but never calls it.

- A tensor node's precision is its **storage** precision; the precision its arithmetic runs at is a
  separate thing, `C_syntax_config.compute_prec` (gh-ocannl-517). They coincide on the GPU backends
  for the formats those have as types (native `__nv_bfloat16` / MSL `bfloat`/`half`, and the 16-bit
  tensor-core shapes that consume them), and diverge on `cc`, where every narrow-float operator was
  a widen/op/narrow round-trip
  anyway: there the narrow floats compute in f32 (`Ir.Numerics.narrow_compute_f32`, on by default),
  so a load widens once and a store narrows once, and an assignment's intermediates keep f32
  mantissa. The rule when touching `c_syntax.ml`: a **declaration**, a kernel parameter or a buffer
  element type takes the storage precision; a **rendered expression** takes `comp_prec` of it. Two
  exceptions are load-bearing. The RNG lane conversions (`uint4x32_to_<prec>_uniform*`) pick both
  their result type and which random bits they consume from the precision they render at — the fp8
  generator is not a rounding of the f32 one — so they stay at storage precision, and the
  scope-locals they write are excluded by a whole-proc scan (`rng_scope_local_uids`), because a
  `Declare_local` carries no value to test. And a `Set` whose value contains no operator renders at
  storage precision, so a copy loop stays a copy instead of a round-trip through f32.
- Convert-on-load/store is what makes the `Vectorized` renderings reachable for 16-bit nodes: the
  lane count comes from the **compute** vector, so the narrow side is a half-width vector, and the
  conversion happens at the memory boundary rather than per lane inside the body (per-lane
  conversion would give the traffic win straight back). Bitwise parity with the serial remainder
  loop is by construction — every fallback arm calls the same scalar conversion the serial path
  does — and `test/operations/narrow_storage_compute.ml` asserts it with `=`, not a tolerance.
- **fp16 is the one narrow format a CPU can compute in natively** (gh-ocannl-516), and whether it
  can is a C-preprocessor fact the OCaml renderer cannot see. `cc` probes the configured compiler
  once per process and reports three states — no `_Float16`, `_Float16` with arithmetic *promoted*
  to float (correct, no lane-count win: x86 without AVX512-FP16), and genuinely native
  (ARMv8.2-FP16, AVX512-FP16) — surfacing the last as `hardware_limits.native_fp16_arithmetic`.
  `Ir.Numerics.fp16_arithmetic = Fp16_narrow` (config `true`; the policy is TERNARY since
  gh-ocannl-680 — see the dedicated bullet below — and the narrow request is not the default,
  because it trades mantissa for throughput, unlike `narrow_compute_f32`) then makes
  `compute_prec` leave `Half_prec` alone, so `vec_ext_typ` mints a `HALF_T` vector and the lane
  count doubles. The middle state is why the probe is not a boolean: seeding and the cost model
  must not expect a lane-count win where only the type exists.
- The fp16 FMA is where parity nearly breaks: `fmaf` on `_Float16` operands promotes to float and
  rounds **twice**, while `__builtin_elementwise_fma` on an fp16 vector rounds once. The scalar
  rendering and the vector rendering's per-lane fallback therefore both go through one builtin
  macro, `OCANNL_HALF_FMA`, defined by the same `#if` — so both configurations agree by
  construction rather than by inspection. Any new fp16 op admitted to the vector path needs the
  same treatment.
- **`-march=native` is the wrong flag on ARM and was silently downgrading every CPU kernel.** Apple
  clang accepts it on arm64 and targets a *lower* baseline than passing nothing: 22
  `__ARM_FEATURE_*` macros against 26 with no flag and 33 with `-mcpu=native`, losing
  `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` among them — so a machine with native 16-bit arithmetic
  probed as one without. `cc_backend_arch_flags` now defaults to `auto`, which asks the target
  which family it is in and probes that family's spelling (`-mcpu=native` on ARM, `-march=native`
  on x86 — where `-mcpu=` is merely an alias for `-mtune=` and would not select the ISA at all).
- The register-tiled `Tile_mma` rendering is on the same seam (gh-ocannl-575): gates, lane count
  and the C-tile registers follow `comp_prec`, operands bridge at the memory boundary
  (`vec_bridge` identity = the old memcpys, so f32 emission is unchanged), and the accumulator
  narrows ONCE per cell after the whole k extent. Since gh-ocannl-639 the serial fallback narrows
  once per nest too (below), so narrow-storage parity vs the fallback no longer needs narrow-exact
  inputs when the reduction scopes coincide. Packing `Stage`s
  take `tile_prec` (exact widenings only) to fold the widening into the pack; seeding resolves the
  same `Numerics.cpu_compute_prec` the emission uses — change either side only through that
  helper, or "timed is not tensorized" returns for narrow sites.
- **A reduction accumulator's WIDTH is policy and its RESIDENCY is unconditional; its narrowing
  POINTS are schedule** (gh-ocannl-639, gh-ocannl-693). The plain serial fallback of an
  accumulation nest holds the accumulator in a scope LOCAL at `acc_prec` and stores once after the
  nest, implemented not by new emission but by locally rewriting the nest at codegen into the
  `Local_scope` form virtualization gives virtual accumulators
  (`C_syntax.try_localize_serial_reduce`) and rendering that — `scope_prec_of` and the
  `Set_local`/`Get_local` arms already carry the widening. The rewrite is NOT precision-gated: it
  used to bail when `acc_prec` was the identity on the storage precision, which left every f32
  reduction no schedule op reached accumulating in the output node's global memory, one
  read-modify-write per step — and on Metal `volatile_serial_accumulation` pinned it there by construction,
  since its trigger predicate is verbatim the localization opportunity. At identity precision the
  widening half is vacuous and the rewrite is exactly value-neutral, so there is nothing for a gate
  to protect. Codegen is the ONLY localizer of a materialized accumulator (`optimize` rejects such
  a scope, gh-ocannl-681).
- **The rendered forms ONE reduction loop can take**, which is the space a property test over a
  single reduction has to cover (gh-ocannl-664). `pp_ll`'s `For_loop` arm dispatches on the loop's
  axis kind, and every arm that can serialize falls back through `try_localize_serial_reduce`:
  `Serial` -> localized scope, else plain serial loop; `Unrolled` -> localized scope (the repeated
  bodies update the scope local), else repeated bodies each doing a global RMW; `Vectorized` ->
  `try_vectorize_reduce`'s SIMD accumulator grid + epilogue, else (accumulating body) the localized
  scope or plain serial loop — never the pragma'd loop, whose independence assertion a loop-carried
  accumulation does not satisfy; `Workgroup_reduce` -> `try_warp_reduce`'s shuffle tree, else the
  hardware binding, else localized scope / serial; `Workgroup` -> hardware binding, else localized
  scope / serial; `Grid` -> the pool-rendered parallel loop or the hardware binding, whose fallback
  is the plain `serial_loop` and NOT the localizing one — the one arm that can serialize without
  localizing. Deliberately left so: an unbindable, non-parallel-eligible `Grid` level over a
  reduction axis would be a cross-thread race wherever it is reachable at all, and no schedule op in
  the tree produces one, so making the dispatch uniform there would be an untestable change.
  Orthogonally,
  the `Set` arm's `try_register_tile` C-tile and `Tile_mma` hold accumulators too. Every form except
  the two RMW ones (plain serial, unrolled repetition) keeps the accumulator in a register for the
  whole nest, so a property test's invariant is: all forms agree on the value, and only the two RMW
  forms touch the node more than twice.
  The set is now swept MECHANICALLY rather than by review, by `test/operations/reduction_forms.ml`:
  a table of (schedule composition x storage precision) over one hand-built row sum, each member
  naming the form it claims, with two claims apiece. The value claim compares a localizing member
  against the executed serial baseline and a declining one against a host reference that narrows at
  every step — both exact, because the operands are storage-exact multiples of 1/8 whose partial
  sums leave the format's exactness range (the test proves that rather than assuming it, and prints
  on stderr which residency regime the backend is in). The form claim reads the emitted kernel and
  classifies it, which is what keeps the value claim worth anything: agreement between two
  renderings says nothing if they are the same rendering, and a composition that stops reaching its
  form falls back to one that passes. Classification is phrased over the STORED node, not over any
  local's name — the localizing forms differ in where the accumulator lives but agree on doing at
  most one read and one write of the node, and the RMW forms are exactly the ones with a statement
  doing both. Two traps it had to absorb: the closing store is `node[i] =
  single_to_bfloat16(v5_node)` at narrow storage, so the right-hand side is searched for a
  scope-local TOKEN rather than compared to one (a bare-local test reports every narrow kernel as
  unlocalized, which is the reading the whole policy is about); and `partials_<parent>[` contains
  `<parent>[`, so node subscripts are counted as whole identifiers or `Split_reduce`'s partials read
  as the target being touched twice. Adding a form to codegen without a table entry is a golden
  diff, since the member list is printed. Hand-built IR carrying a runtime-extent guard additionally
  needs `Ll_test.optimize ~static_indices`: the virtualization walk asserts every
  `Embed_index (Iterator s)` is in scope, and a launch parameter is in scope only because the caller
  declared it (and its symbol needs `used_as_extent`, or bind-time validation rejects the extent
  covering the whole axis).
- **What forces one of the two RMW forms**, i.e. the declines a property test must be able to
  provoke: `debug_log_from_routines` (a `Local_scope` body renders with `log_set_locals:false`, so
  localizing would silence the per-iteration trace — the SIMD and tensorized renderings bail under
  logging for the same reason); an update mentioning an RNG conversion (gh-ocannl-517: the
  conversion picks which random bits it consumes from the precision it renders at); and
  `peel_accum_nest`'s structural refusals — a level holding more than one statement, a guard that is
  not `pure_index_guard`, a cell that is not invariant across the peeled levels, an update outside
  the `accum_update_parts` grammar (only `Add`/`Mul`/`Max`/`Min` and the `FMA` form, with the
  accumulator read a DIRECT operand of the top operator), and, for a scope-form base, updates under
  mixed operators or a write to another node inside the body.
- **A DEAD level (`to_ < from_`) is never peeled**, and that refusal is load-bearing rather than
  tidiness. A dead loop's body performs no accesses at all — the routine-interface walk propagates
  liveness as `live && to_ >= from_`, so a node reached only under one is absent from the parameters
  and need not be allocated — while every form the peel licenses reads and writes the accumulated
  cell OUTSIDE the levels, unconditionally. Peeling one would invent accesses the program does not
  make, possibly naming an identifier the interface never declared; it is the same convention
  `drop_dead_loop_accesses` keeps for the affine metrics and virtualization keeps by dropping dead
  loops outright ("mint phantom parameters for identifiers only dead code renders"). Because
  `optimize` drops them, ordinary lowering cannot deliver a dead loop to codegen — a post-optimize
  transform can, which is why the refusal lives in the shared `peel_accum_nest` (covering the
  schedule mints) plus the rendered level's own bounds in `try_localize_serial_reduce`, which the
  peel never sees. Pinned by `test/operations/peel_dead_level.ml`, live twin included.
  Refusing to peel is only half of it: the level then has to RENDER, and the `Unrolled` arm repeats
  its body `to_ - from_ + 1` times through `Base.List.init`, which RAISES on a negative length
  rather than answering the empty list — so the count is clamped at zero, zero repetitions being
  exactly a dead level's access-free meaning. That abort was reachable before gh-ocannl-693 for a
  dead `Unrolled` level whose body is not a recognized accumulation; refusing to peel the
  accumulating ones widened its reach, which is how it surfaced.
- **Peel-guard legality belongs to `Ir.Affine`, not to `peel_accum_nest`** (gh-ocannl-722). Five
  review rounds of gh-693 re-derived it as an ad hoc predicate before it moved; if you find yourself
  writing a sixth clause, add it to the engine. Two queries state the whole rule:
  `Affine.peel_guard ~loop_bound ~peeled ~guard_syms` answers `Confined_to_peel`,
  `Not_peelable why`, or `Lane_private_if_separated syms`; `Affine.separates ~range ~concurrent
  ~syms ~idcs` then decides the third against the accumulated cell. `peel_accum_nest` asks the first
  at each guard (what a guard mentions is known on the way down) and defers the second to the base,
  where the cell is finally in hand.
  WHY the rule is what it is: `rebuild` keeps a guard around the accumulating update only, so the
  localized form performs its opening load and closing store OUTSIDE it — right when the guard's
  truth is not fixed for the whole nest, wrong when it is. A guard mentioning no peeled symbol is
  fixed for the whole nest, so hoisting turns "this instance performs no access" into a load and a
  store. A guard mentioning an ENCLOSING loop symbol selects among that level's iterations: for
  `Workgroup w -> Serial k -> If (w < 1) (acc[0] += x[k])` every lane would load `acc[0]` and write
  its unchanged local back, so a lane reading before lane 0's store and writing after it silently
  discards the reduction — the same 1/N fingerprint as the Metal RMW miscompile, from a different
  cause. "Varies with a peeled symbol" is NOT the predicate: a MIXED guard like `w + k <= 0` varies
  with the peeled `k` and still selects among lanes.
- **What decides the enclosing case is a SHARED CELL, not a mixed guard** (gh-ocannl-721). Under
  `Workgroup w -> Serial k -> If (w + k < n) (acc[w] += x[w,k])` every lane owns a distinct cell, so
  the invented load/store pair is private to its lane and idempotent and the hoist is race-free —
  which is exactly `Affine.separates` asked of the cell, and why the cell reaches the decision.
  Three traps in using that query. Its `concurrent` set must cover EVERY symbol whose value may
  differ between the two instances, not only the ones being told apart: with `acc[w1 + w2]` and
  `syms = [w1]`, holding `w2` equal "proves" a separation that instances `(0,1)` and `(1,0)` refute.
  Mentioning a symbol is not separating it — the same `acc[w1 + w2]` mentions both and separates
  neither. And **separation is distinctness, not access validity**: the guard being hoisted past may
  be what keeps the cell in bounds, so a consumer must ALSO ask `Affine.within_box` of the cell over
  the enclosing symbols' full ranges, judged without the guard. With a one-element `acc`,
  `Workgroup w (0..3) -> Serial k -> If (w + k < 1) (acc[w] += ...)` separates `w` perfectly while
  lanes 1–3 address cells that do not exist (Codex P1 on staging#443). The confined case needs no such
  check — there the guard mentions only peeled symbols and symbols no loop binds, while the cell is
  invariant across the peeled levels, so the guard cannot bound anything the cell mentions.
  Uninterpretable components (`Sub_axis`, `Concat`, dynamic indices) contribute no information to
  either query, so they decline rather than admit.
- **The gh-490 runtime-extent guard is NOT constant-bounded** — worth knowing, because assuming it
  was cost a review round: `Assignments.extent_guard` (assignments.ml:225) emits
  `Cmplt (Embed_index (Iterator index), Embed_index (Iterator sym.static_symbol))`, whose bound is a
  STATIC symbol, a kernel parameter bound at launch. (`Schedule`'s Pad guards ARE constant-bounded;
  the two shapes are easy to conflate.) A static symbol cannot select among enclosing loop
  iterations, but the peel cannot tell it from a loop index on its own, so it reads the
  classification off the program: the REQUIRED `~loop_bounds` is `Low_level.loop_bounds` of the
  enclosing code — the same "box environment for Affine queries" the engine's other consumers pass,
  which also supplies the iteration ranges `separates` needs. A guard symbol in it that is not
  peeled is an enclosing level's index; one outside it is bound outside every loop and is harmless.
  Derived from the program, not certified by the caller, and required rather than defaulted, because
  **declining is not neutral for the schedule mints**: a refused mint makes `Unroll
  ~materialize:true` round-trip the accumulator per copy and `Partition` turn its segment seams into
  narrowing points, so on narrow storage the scheduled candidate stops agreeing with the serial
  baseline — the invariant those mints exist to hold ("candidates compete on speed, never
  numerics"). A defaulted certification is exactly the kind a call site forgets; three review rounds
  went into finding that out.
  `peel_dead_level.ml` carries the guard shapes at the peel (including the lane-private ones and the
  two-lanes-collapsing counterexample), `affine_legality.ml` pins both queries directly —
  `separates` against the same brute-force oracle `pair_conflict` is checked with — and
  `reduction_forms.ml` carries the executed consequences: `runtime-guard*` for gh-ocannl-715 (a bf16
  and f16 reduction under a runtime-extent guard, scheduled with a materializing `Unroll` and with a
  `Partition`, bitwise against its serial baseline) and `mixed-guard` / `mixed-guard-workgroup` for
  gh-ocannl-721. Note where the mixed shape is discriminating: the OUTPUT loop must not be peelable
  too, or the whole nest localizes at that level and the enclosing case never arises — in
  `reduction_forms` the cell `out[r]` mentions the output index, which is what stops the outer peel.
  Keeping an outer guard around the whole scope instead of declining would localize more still; it
  needs the peel to report its outer guards separately, which is wider than any of this.
- Metal's `volatile_serial_accumulation` has TWO localization interactions (gh-ocannl-731). The
  pointer shadow keys on an emitted `Set` reading its own node at a cell invariant across an
  enclosing SERIAL loop, and localization lifts that `Set` out of exactly those loops
  (`peel_accum_nest` runs outermost-first, since `pp_ll` recurses top-down), so at a fully localized
  site the shadow predicate is false. But the Metal compiler can also corrupt the replacement
  scope-local accumulator whenever its loop reads any device pointer; therefore `scope_decl_type`
  qualifies every reduction-shaped scope local as `volatile` on Metal — at a measured cost of up to
  4x on the shapes where the accumulator is the critical path, which is the price of correctness
  until the defect is fixed (gh-ocannl-782, and see the dialects note for what the matrix refuted).
  Both decisions are censused: `Context.routine.volatility` reports per routine how many
  accumulators were qualified, how many stayed register-resident, and whether the backend asked for
  the workaround at all, which is how a test states the expectation without re-deriving it from the
  backend's name. Where the
  peel was blocked at an outer level — a sibling statement, a data-dependent guard — the store
  stays inside an invariant-address loop and the POINTER shadow still fires, correctly: the
  per-iteration device-memory RMW genuinely remains. Localization joins virtual scopes,
  `try_vectorize_reduce`'s epilogue and `try_register_tile`'s C-tile. The peel accepts
  Serial/`Unrolled`/`Vectorized` levels (autotune proposes `Unroll` over any Serial loop of extent
  <= 8 and Retype-`Vectorized` over reductions; a `Vectorized` level rides into the scope and
  `try_vectorize_reduce` recognizes the `Set_local` update form, folding its chains into the scope
  local with no storage round-trip — its scalar TAIL also folds into the wide total before the
  single store), sees through the pure-index guard shape `If (i < bound)` (gh-490
  symbolic extents — data-dependent guards are NOT transparent), hoists through a scope-form base
  (a `Set` whose value is already the accumulation `Local_scope`, reusing its id — accepted ONLY
  when every update fits the mint grammar under ONE reduction operator,
  `Low_level.scope_updates_reduce_op`: hoisting is licensed by the reduction shape, and both a
  general recurrence like `local := local - x` and an individually-valid-but-MIXED sequence like
  `local += x; local *= y` must keep their per-iteration narrowing), and also serves
  hardware-annotated loops the backend serializes for lack of a hardware index (cc's
  `Workgroup_reduce`) — standalone via the dispatch fallback, and NESTED via the peel's
  `extra_level` predicate, which is codegen-only (a schedule mint wrapping a hardware-annotated
  loop in a scope would break backends that bind the dimension). A merge-buffer read
  (`Get_merge_buffer`) is NOT self-dependence of its node — it is a separate read-only staging
  buffer — so `p =+ p.merge`-style updates stay recognizable accumulations. `Schedule.Partition` of a recognized nest mints ONE scope spanning its
  segment loops (an index-set specialization is not a partial-reduction boundary), and
  `rewrite_loop` descends into `Local_scope` bodies so minted-scope interiors (partition segments,
  an outer materialized unroll's inner loops) stay addressable by later schedule ops. The walk that
  LOCATES a loop has to reach exactly as far as the one that REWRITES it, or a probe reports absent
  a loop the very next op happily rewrites (`partition_breakpoints` did, gh-ocannl-668): both come
  from one walker, `Schedule.find_loops_env` — `find_loop`/`find_loops` are its unit-environment
  instances and `partition_breakpoints` threads the enclosing loop ranges through it — and the
  reach has TWO dimensions, both of which cost a review round. As deep: `~in_scopes` (default true)
  descends into scopes. As WIDE: `rewrite_loop` rewrites every copy, so a first-match probe speaks
  for a fraction of the op — a materializing `Unroll` leaves one copy of the inner loop per step,
  each with the outer index substituted by a different constant, so copies of ONE source guard flip
  at different points and `partition_breakpoints` must return their union (a first-copy answer
  leaves the siblings' guards mixed after the `Partition` rewrites them all). Same rule for
  legality: `loops_independent` combines the verdicts of every binding, worst-of. The autotuner's
  own enumeration obeys the same law (gh-ocannl-666): `Autotune.collect_loops` descends scope
  bodies and treats binder-sharing mint copies as ONE decision (one proposal per binder, exactly
  as `rewrite_loop` rewrites every copy), so a materialized unroll or partition no longer hides
  the inner loops from the beam's later rounds; `collect_serial_triples` stays statement-level on
  purpose — `Tensorize`'s `Workgroup` lane loop is refused inside a scope by `validate_parallel`
  (which `op_legality` does not decide: races, not scope nesting), and mint interiors reduce into
  one loop-invariant cell, so no viable micro-kernel triple can sit there anyway.
  `apply_split_reduce` is the one caller taking the first statement-level match, and for a contract
  reason: it inserts its combine statement at statement level, which only sequences correctly if
  the reduction runs there too. A MATERIALIZED unroll never reaches codegen as bare copies:
  `Sched.Unroll ~materialize:true` itself rewrites a recognized accumulation nest into the scope
  form — that is where the provenance lives, since a codegen pass looking at adjacent same-cell
  `Set`s cannot tell unrolled copies of one assignment from two user-authored assignments, whose
  separate stores (and separate narrowings) are their semantics (`accum_width`'s 256+1+1 leg pins
  the boundary). The whole nest recognition — Serial/Unrolled levels, pure-index guards,
  raw-update or scope-form base — is ONE function, `Low_level.peel_accum_nest` (over
  `accum_update_parts`), shared by the transform and the emission precisely because three review
  rounds found them drifting one capability at a time (guards, scope-form bases, the logging
  decline); extend nest recognition only there. The
  widening is inert wherever `acc_prec` is the identity (f32/f64; GPU backends per their
  residency table above, e.g. f16 there under `Fp16_auto`; `narrow_compute_f32=false` except
  f16-under-`Fp16_wide`; native fp16 under `Fp16_narrow`), and it declines twice more: under
  `debug_log_from_routines` (a `Local_scope` body renders with `log_set_locals:false`, so the
  rewrite would silence the per-iteration trace — the per-step `Set` form is the traceable one,
  and every tensorized rendering already declines under logging), and on updates mentioning an
  RNG conversion (the conversion picks its result type AND which random bits it consumes from the
  precision it renders at, so an f32-precision scope would change the draw, not widen it —
  `narrow_rng_nesting`'s reduced-uniform leg pins this). Two traps for tests in this area: (1) cross-schedule BITWISE parity on
  non-storage-exact sums additionally needs the same narrowing points — a k-blocked schedule
  stores storage-precision partials at every `bk` boundary by construction, so give the packed
  leg a whole-k tile (`tile_mma_narrow`'s gh-639 leg uses `bk = n`); (2) discriminating inputs
  must DRIFT out of storage exactness — a zero-mean operand random-walks small enough that every
  bf16 partial sum stays exact and per-step narrowing is invisible (`accum_width.ml`'s policy-off
  negative control is the canary).
- **On GPU the accumulator residency follows the backend's tensor-unit formats, per backend**
  (gh-ocannl-663): `C_syntax_config.accum_prec` — the width a recognized reduction accumulator
  resides at given the storage precision — feeds the try_widen gate AND `scope_prec_of`, whose
  reduction-shaped scopes a codegen census (`C_syntax.accum_scope_ids`, per `scope_id`, verdicts
  from `Low_level.accum_local_update_parts`) resolves at accumulator width, so schedule-minted
  scopes (materialized `Unroll`, `Partition`) and virtual accumulators match the widened serial
  fallback on every backend; the codegen-minted scope registers itself there. The per-backend
  table: CUDA widens bf16→f32 (its mma legs hold f32 per-lane registers whole-k — NVIDIA has no
  bf16 accumulate) and fp8→f32; HIP widens only fp8 (RDNA WMMA has genuine bf16/f16 accumulator
  variants and the uniform triples are seeded, so bf16 serial legs deliberately stay narrow —
  width-uniform with the mma legs); Metal's `accum_prec = compute_prec` (fp8→f32); cc's likewise
  (the CPU accumulator IS a compute intermediate). Since gh-ocannl-680 every backend's `accum_prec`
  additionally widens f16→f32 under `Numerics.Fp16_wide` — see the dedicated bullet below.
  `narrow_compute_f32` (already in the
  schedule-cache key, gh-ocannl-568) reaches a GPU accumulator only where policy-off can restore
  per-step narrowing SCHEDULE-UNIFORMLY: fp8 on CUDA/HIP (nothing tensorizes fp8 destinations).
  CUDA's bf16 residency is structural like Metal's fp8 — the mma accumulate is hardware-f32, so a
  policy-narrowed serial leg would resurrect the schedule dependence. Scope INITS (a `Set_local`
  not reading its own local — the inlined image of a separate source assignment) render at
  `comp_prec` and convert once into the residency, preserving each source assignment's own
  narrowing (the adjacent-accumulations provenance boundary); virtualization's guarded-read
  updates `Where (index-only cond, update, Get_local id)` classify as reductions via
  `Low_level.accum_local_update_op` — a recognizer deliberately separate from both
  `accum_local_update_parts` (whose `(op, contrib)` licenses rebuilding an unguarded update) and
  `scope_updates_reduce_op` (the hoist license). Two traps: `compute_prec`/`accum_prec` bind at `include Pure_C_config` time, so
  overriding one without restating the other silently keeps the default pairing — a startup
  width assert in the `C_syntax` functor catches the narrow direction; and `Workgroup_reduce`'s
  warp-shuffle rendering used to hard-error on every narrow accumulator, which gh-ocannl-682 turned
  into the residency gate described in the next bullet.
- **f16 residency is the ternary `fp16_arithmetic` policy's question** (gh-ocannl-680, refining
  gh-ocannl-516's boolean; config `auto|true|false`, default `auto`, old boolean spellings intact).
  `Numerics.fp16_mode`: `Fp16_auto` keeps each backend's structural residency — CPU computes f16 in
  f32 under `narrow_compute_f32`, GPUs keep storage-width f16 accumulators mirroring the
  f16-accumulate triple every backend seeds — so `auto` changes nothing numerically vs. the old
  default, but the uniformity is now POLICY rather than a coincidence of seeding. `Fp16_narrow`
  (`true`) is unchanged gh-516 opt-in. `Fp16_wide` (`false` — note the repurposing: an explicit
  old-style `false` now REQUESTS strict wide) gives f16 accumulators f32 residency on every
  backend, `narrow_compute_f32=false` included: each backend's `accum_prec` widens `Half`, and the
  mma story is per-EMISSION-SCOPE all-or-nothing (gh-ocannl-545, gh-ocannl-836), through the
  `Numerics.fp16_accum_wide` policy predicate and the backend capability's scope list. CUDA sm_80+
  stays tensorized only in the per-statement scope:
  the uniform-f16 combination routes to the f32-accumulate inline-PTX m16n8k16 arm (the bf16
  uniform arm's body parameterized by `mma16_spellings` — the PTX fragment layouts are shared by
  .f16/.bf16; marker `(mma-f16)`, arch floor 80 in `gpu_arch_options`), and its f16-accumulate
  wmma combo is gated off. Its persistent-fragment scope has no corresponding wide arm —
  `wmma_combo` cannot load/store an f16 destination through an f32 accumulator fragment — so
  `mma_f16_wide_acc_scopes = [Mma_per_statement]` preserves only candidates whose emitted
  accumulator has no enclosing reduction loop with extent greater than one. That includes the
  unstaged rank-2 forms and a staged matmul whose padded k-block count is one; a multi-axis matmul's
  inherited contraction loops, a multi-block staged matmul, and a multi-window convolution require
  `Mma_fragment_scope`. A detected convolution with only extent-one kernel-window loops likewise
  resolves per-statement, though today's affine-fingerprint detector does not recognize a fully
  degenerate 1x1 convolution after lowering erases those loops. Before gh-ocannl-836 the scope-blind
  boolean admitted both scopes: a
  staged discriminator with a 144-element `k` extent, starting at 2048 with +1 per 16-wide tile,
  emitted the inline-PTX arm
  inside the nine-iteration outer loop and returned 2048, while a single wide scope reaches 2057
  and narrows once to 2056. HIP advertises both scopes since gh-ocannl-789 (see the rocWMMA
  d-boundary bullet below). Metal advertises both scopes since gh-ocannl-837: its per-statement and
  persistent-fragment hooks use an f32 accumulator with converted `thread_elements()` boundaries.
  `Sketch_families.fp16_wide_withholds` is keyed on the DESTINATION's storage precision, so
  `(f16,f16,f32-storage)` sites are untouched; `matmul_mma_scope` and `conv_mma_scope` derive the
  scope from the actual outer reduction extents rather than from whether operands happen to be
  staged. Placement enablement applies the same resolver to its unstaged eligibility gate, so a
  device with no applicable wide scope cannot spend its placement budget unlocking an empty
  family. Thus CUDA alone trades multi-statement legs under the wide policy. `auto`
  deliberately RETAINS LATITUDE to later resolve wide on hardware where wide f16 accumulate is
  free (datacenter NVIDIA runs f32-accumulate f16 mma at full rate; GeForce halves it) — do not
  write code or tests assuming `auto ≡ narrow` as a contract; `accum_width.ml`'s default-policy
  legs pin auto's CURRENT resolution and say so. Pinned by: `accum_width.ml`'s universal f16 legs
  (scalar 2048+1×8 discriminates 2056 wide vs 2048 per-step; matmul parity vs once-narrowed wide
  reference under `Fp16_wide` on every backend — inputs exact in f32 so schedule reassociation
  cannot break bitwise equality), `sketch_family_tree.ml`'s seeding-gate section (mma seeds present
  under default, absent with no wide scope, unstaged-only with the per-statement scope, restored in
  full with both), and
  `hardware_warp_shuffle.ml`'s `Fp16_wide` legs, which execute the path the policy newly makes
  reachable: the f16 warp-shuffle rendering the residency gate refuses under `auto` (next bullet). The reproducible
  profile pins `fp16_arithmetic=auto` (the default, so the profile still changes no math); the
  performance profile keeps `true`. The CUDA `(mma-f16)` per-statement arm and the scope-boundary
  discriminator were executed on sm_120 by gh-ocannl-836; HIP's two-scope arm was closed by
  gh-ocannl-789.
- **The `approximate` profile is the one word for the numerics-changing regime** (gh-ocannl-719):
  the `performance` payload plus `tf32_matmuls=true`, `cc_backend_fast_math=true`,
  `cc_backend_fp_contract=fast` and `tune_inline_flips=2`, contract "results differ from the exact
  profiles by the tolerance the benchmark parity envelope names, never in shape or in which
  operations run". A rewrite that changes numerics for throughput (online-softmax attention,
  gh-ocannl-483; Winograd, gh-ocannl-505) gets a key defaulting off, flips it in this payload, and
  adds it to `Schedule_cache.numerics_tag` in the same PR — every knob the profile flips today is
  already in the numerics or codegen digest, which is what keeps a default-flags run from
  replaying a winner tuned under the profile. Benchmark side: `orchestrate.py --profile
  approximate` dispatches the OCANNL cells under `--ocannl_profile=approximate`, the torch cells
  under torch's own defaults (`high` matmul precision, SDPA, `cudnn.benchmark`), gates at
  `PARITY_TOL_APPROX` (looser than every exact envelope), labels every row with its regime, and
  reports whether an approximate row ALSO passed the exact envelope — a rewrite that changes
  nothing measurable is a finding, not a pass. The exact torch CPU eager cell stays the parity
  reference in every regime; the runner reports the profile it resolved (`profile` in the result
  line) and where each approximate-payload key resolved from (`regime_knobs`, via
  `Utils.profile_payload_sources`, so the list is the payload's own), and the sweep refuses a row
  whose claimed regime the runner contradicts — the regime is the resolution of those keys, not
  the profile's name: exact owns only built-in defaults, approximate only the profile, and an
  ambient `OCANNL_TF32_MATMULS=true` or a `reproducible` profile in the environment is a
  REGIME MISMATCH row that names the key and its source.
  Keys OUTSIDE that payload are recorded instead of checked (gh-ocannl-720): the sweep reads its
  ambient `OCANNL_*` environment once, stamps it onto every OCANNL row as `ambient_ocannl_env` and
  prints it in the report header, so a `narrow_compute_f32` or `cc_vector_bytes` in the operator's
  shell — or any key added after the gate was written — is visible in the artifact rather than
  silently part of the measurement.
- **rocWMMA fragments are opaque for LAYOUT but not for ELEMENTS, and that is what wires HIP's
  wide-f16 d boundary** (gh-ocannl-789, `arrayjit/lib/hip_backend.ml`'s `mma_d_boundary`). Under
  `Fp16_wide` the uniform-f16 arm pairs a `float` accumulator fragment with the unchanged f16
  STORAGE destination; neither `load_matrix_sync` nor `store_matrix_sync` is type-correct across
  that mismatch, so the boundary stages through a destination-typed accumulator fragment rocWMMA
  does load and store, and copies element-for-element via `num_elements` / `x[]` — the surface the
  header documents as "compatibility with nvcuda::wmma". The copy never assumes WHICH matrix cell
  an element index names, only that two accumulator fragments of the same 16x16x16 shape name the
  same cell at the same index; that holds because rocWMMA derives the accumulator's IO layout from
  the fragment shape and wave size, not from `DataT`, and an emitted `static_assert` on the two
  `num_elements` fails the kernel compile if a release ever packs them differently. Measured on
  gfx1151: both fragments report 8 elements and dump identical per-`(lane, index)` values from the
  same 16x16 tile of distinct values. No LDS traffic — the warp-staged float tile that was the
  fallback design is not needed. The combination table and fragment-type spelling are now shared
  by `mma_syntax` and `mma_fragment_syntax` (they were duplicated verbatim, and the two hooks are
  required to accept together), so an arm added to one reaches both.
- **Metal's `simdgroup_matrix` surface also permits a wide-f16 accumulator despite the uniform
  storage-format table** (gh-ocannl-837, `arrayjit/lib/metal_backend.ml`'s
  `mma_d_boundary_lines`). A standalone `MTLDevice.makeLibrary(source:)` probe on an Apple M4 Max
  (40-core GPU, Metal 4, macOS 26.6.2 build 25G83, Swift 6.3.3) compiled and ran half A/B fragments
  with a `simdgroup_float8x8` accumulator. `thread_elements()` exposes the distributed fragment
  storage: the half destination stages through `simdgroup_half8x8`, and each lane converts indices
  0 and 1 to/from the float accumulator. An emitted `static_assert` compares the two
  `storage_type` element counts (64 logical elements each); with the fixed 32-thread capability,
  that is two elements per lane. A whole-`storage_type` converting constructor crashed the runtime
  compiler service on this toolchain, while the scalar element copies compiled and executed, so
  keep the scalar spelling. The k=144 probe/test makes the first 16-wide block contribute 2048 and
  each of the remaining eight contribute 1: residency across the reduction returns 2056, whereas
  narrowing at each block boundary returns 2048. The default policy still uses
  `simdgroup_half8x8`; only `Fp16_wide` selects the mixed accumulator. As on HIP, one shared
  combination resolver and boundary helper feed both `mma_syntax` and `mma_fragment_syntax`, so
  their accepted shapes cannot drift.
- **The warp-shuffle rendering stages at the residency, and gates on it** (gh-ocannl-682).
  `C_syntax.try_warp_reduce` holds `wred_v_*`, the `__shared__ wred_partials_*` slots and every
  `ocannl_shfl_xor` stage at `accum_prec` of the storage precision, and renders the contribution
  there; the one place the residency meets storage is `fold_total`'s read-modify-write of the
  narrow cell, which carries the single widen/narrow pair (a buffer element type is never
  `accum_prec`'d, so nothing else moves). A bf16 `Workgroup_reduce` on CUDA therefore shuffles
  `float` and computes exactly what its serial rendering does, and no backend needs a narrow
  `ocannl_shfl_xor` overload — the two float/double ones remain the whole ask. The gate is on the
  RESIDENCY, not on storage: f16 under `Fp16_auto`/`Fp16_narrow`, and bf16 on HIP/Metal, still
  raise (`accumulator residency` in the message) rather than gaining an untested narrow-shuffle
  path, since a plain hardware binding would race the accumulator; under `Fp16_wide` the f16
  residency is f32 (gh-ocannl-680), so f16 shuffles float exactly as bf16 on CUDA does. **RNG-bearing contributions are refused too**, but
  only where the residency is actually wider than storage — and the reason is worth holding onto,
  because the first cut of gh-ocannl-682 got it wrong (Codex P1 on staging#461). It rendered
  such a contribution at storage precision and widened it once, reasoning that this preserved
  gh-ocannl-517's draw. It does preserve the draw — but not the *reduction*: the serial rendering of
  an RNG-bearing update is one `try_localize_serial_reduce` explicitly DECLINES to localize, so it
  accumulates in the narrow cell and narrows on every iteration, while the shuffle would have
  accumulated the whole tree wide and narrowed once. That is a change in accumulation WIDTH, not
  association — precisely the property gh-ocannl-682 exists to preserve. The lesson generalizes past
  RNG: **the shuffle may widen only where the serial path widens** — which gh-ocannl-754 then made
  structural (next bullet): the shuffle, the SIMD grid and the localizing serial form all consume
  ONE width decision, and `C_syntax.accum_pinned_to_storage_prec` is one of that decision's
  declines rather than a predicate each rendering remembers to consult. Tests:
  `hardware_warp_shuffle.ml`'s bf16 legs — 32 lanes of `1 + (k mod 11)/128` separate the three
  candidate renderings as 33.25 / 33 / 32.75 (once-narrowed f32 tree, tree staged at bf16,
  per-step read-modify-write), and 128 lanes give 133 / 132 / 129 while also pinning the shared
  slots' element type; its `Fp16_wide` twin legs — the f16 analogue `1 + (k mod 11)/1024`, giving
  32.15625 / 32.125 / 32.09375 and 128.625 / 128.5 / 128.125 — execute the same rendering under
  the wide policy on every backend, and are the legs that actually run the two-phase staging on
  Metal (whose bf16 legs are skipped by design). The 11-cycle is load-bearing at both widths: at
  f16 a 7-cycle leaves the four-warp staging indistinguishable from the once-narrowed value, and
  deriving the actual XOR-tree association exposed the same collision in the old bf16 fixture.
  When transplanting a discrimination like this, derive it rather than assuming the constants or
  association carry over. Plus the default-policy f16 leg and the bf16 RNG leg for the two
  refusals, and
  `reduction_forms.ml`'s `retype-workgroup-reduce` member, whose availability now asks
  `expected_residency` instead of naming f32.
- **One accumulator-width decision per reduction level, consumed by every rendering**
  (gh-ocannl-754). Before it, the warp-shuffle tree and the SIMD grid recognized an accumulation
  through a single-statement recognizer of their own while the localizing serial form went through
  `Low_level.peel_accum_nest` plus the RNG carve-out, and the two analyses agreed by maintenance:
  gh-ocannl-639 (`Unrolled`) and gh-ocannl-682 (RNG bodies under the shuffle) were the same genre —
  two renderings of one body accumulating at different WIDTHS, so a retype changed the value in its
  low mantissa bits, invisibly at f32 storage. Now `pp_ll`'s `For_loop` arm asks once —
  `decide_accum_width` (type `C_syntax.accum_width`): the peel over the level's body, then the
  declines the serial form applies to the base it reaches (debug logging, dead level,
  `accum_pinned_to_storage_prec`, in that order) — and every rendering consumes the answer.
  `try_localize_serial_reduce` renders it and records it; `try_warp_reduce` and
  `try_vectorize_reduce` take the base only through `statement_accumulation` (a raw update at
  exactly this level, no level or guard between) and, where the decision PINNED it, hold the
  accumulator wide only if the residency is the storage width (`residency_is_storage`: f32 storage
  and every non-widening backend render RNG-bearing reductions exactly as before) — otherwise the
  shuffle refuses loudly and the grid bails to the serial fallback, which keeps the per-step
  narrowing. The census gained `Peel_ceded verdict` — the decision was wide and the tree or the
  grid took it — so `Context.routine.peel` attributes each rendering's width to the shared call;
  `reduction_forms` claims it as `Widened_elsewhere` (retype-vectorized on cc,
  retype-workgroup-reduce on the GPUs) and the pinned members as `Pinned_to_storage`. Two
  consequences to know: (1) an UNGUARDED accumulation nest the shuffle cannot render — inner
  levels, or a schedule-minted scope, under a `Workgroup_reduce` level whose lane index the backend
  binds — is refused loudly instead of falling through to the hardware binding, under which every
  lane read-modify-wrote the shared cell (a silent race before); a GUARDED one still takes the
  binding, since a guard can select the lane, which is exactly what the explicitly staged tree's
  `if (i == 0)` write is (`hardware_workgroup_reduce.ml`). (2) A schedule mint
  (`Unroll ~materialize`) DOES scope an RNG-bearing nest, and the rng census then pins that scope to
  storage precision, so its local narrows on every update: a localized FORM at storage width —
  `rng-unroll-mat` pins it, and it is why width is a property of the decision, never of the form.
  The corpus is `reduction_forms`' `rng*`, `sibling-*` and `data-guard-*` members — parity with
  each body's own serial rendering at bf16/f16, with the "RNG-scaled operands discriminate
  accumulator width" control beside the baselines, whose draw is read back from a one-cell kernel
  because no host model of it exists — plus `hardware_warp_shuffle`'s nested-nest leg. Before the
  decision was shared, `rng-vectorized` at bf16 on cc rendered the SIMD grid: the loop-invariant
  draw splatted at compute precision (a different generator from the storage-precision one) and the
  chains accumulated wide — the third sighting, found by the corpus rather than by review. Building
  its f16 legs on Apple Silicon also exposed `Builtins_cc.uint4x32_to_half_uniform` declaring a
  `uint16_t` return where `FLOAT_TO_HALF` yields a `_Float16` under native fp16: the value went
  through an integer (every draw became 0) and the half FMA builtin refused the operand; it returns
  `HALF_T` now. And running those legs on the Linux CI leg exposed a trap one level down: the OCaml
  executable carries a HOST copy of the builtins (`arrayjit/lib/builtins.c`, the `ir` library's
  foreign stubs) under the same names, and on ELF the dynamic linker resolves a kernel's `-fPIC`
  PLT calls against the executable first — so the kernel's `uint4x32_to_half_uniform` bound to the
  host copy, which returns its `uint16_t` in an integer register while the kernel read a
  `_Float16` from `xmm0`: the draw came back as `-0.0`, while every C-side call of the very same
  `.so` was correct. Kernels now link with `-Wl,-Bsymbolic` on ELF (`Cc_backend.kernel_link_flags`),
  binding a kernel's references to its own definitions as macOS's two-level namespace always did;
  the diagnosis recipe is `LD_DEBUG=bindings` on a driver that preloads the executable's stubs.
- **A "packmma" timing is not evidence that anything tensorized.** A `Tile_mma` whose register-tile
  preconditions fail renders the scalar fallback and the run still reports under whatever the
  variant was named — the column extent below the compute vector width is the easiest way in (at
  f32/16-byte vectors the crossover is n = 4; at f16 it is n = 8; on a wider machine the crossover
  follows the DEGRADED width, i.e. the 32-byte floor, not the configured one), but a narrow
  `vector_bytes`,
  mixed operand compute precisions, transposed-B storage, and `debug_log_from_routines` all decline
  too. Check the compiled routine's `Context.routine.mma.tensorization` (gh-ocannl-626 — the
  census travels with the routine; do not bracket `mma_census_enabled` by hand) rather than
  trusting the label; `bin/narrow_gebp_bench` and `bin/schedule_bench` print it on every timing
  line and warn when any statement declined, and `schedule_log_declines=true` gives the per-rule
  reason. This is the same
  "timed is not tensorized" hazard the seeding note above raises, seen from the bench side.
- `bin/narrow_gebp_bench` takes its blocking factors as arguments (`[bm] [bk]` positionally, or
  `--bm=`/`--bk=`), defaulting to 64/256. The packed variants need `n mod bm = 0` and `n mod bk = 0`
  — an n meeting neither still runs the unblocked naive variant, so an arbitrary extent (the sort a
  register-tiling review actually asks about) can be measured against something.
- **Negative zero is what breaks a "bitwise equal to the scalar twin" claim** (gh-ocannl-615). Two
  spellings normalized it, both fixed but both easy to reintroduce: a scalar-to-vector splat written
  `((vtyp){0} + x)` returns `+0.0` for `x = -0.0` (IEEE `(+0.0) + (-0.0) = +0.0`), so use
  `C_syntax.vec_splat` — `x - (vtyp){0}`, the exact identity on every input; and a float constant
  printed with `%.16g` alone comes out as the C *integer* literal `-0`, i.e. `+0.0` once cast, so
  print through `C_syntax.c_float_literal`. Neither shows up in ordinary parity tests: `Float.equal`
  reports `-0. = +0.`, and a divergence in the sign of a zero product only survives into a result
  whose accumulator is itself a signed zero. `test/operations/vec_signed_zero` is the regression, and
  its splat legs need an accumulator preloaded with `-0.0` (a zeroed one absorbs the sign) — which is
  why they are an explicit `=+` into an initialized tensor rather than a `schedule_mma_matmul` leg.
  The literal is the CROSS-BACKEND half — confirmed to have corrupted CUDA host data too, not just
  cc — so its leg deliberately runs on every backend while the splat legs are CPU-gated; a
  `Vec_extensions`-only test would have missed it. **A splat must be arithmetic-free, not merely
  value-preserving**: `vec_splat`'s first fix, `x - (vtyp){0}`, is the identity on both zeros but
  still quiets a signaling NaN (`0x7f800001 -> 0x7fc00001` under gcc `-fsignaling-nans`; that it
  usually survives is only the optimizer folding the subtraction away, which nothing requires — the
  same "the compiler will probably do the right thing" dependency `vec_fma_builtin` refuses for
  `a * b + c`). It renders an initializer of `lanes` copies of a bound scalar temp instead, which is
  why the callers bind the operand first. sNaN cannot be tested end to end — host values cross as
  OCaml doubles and the narrowing quiets it — so the structural pins keep both arithmetic spellings
  out by name.
- **gcc rounded fp16 FMAs twice, and that is visible, not theoretical** (gh-ocannl-621).
  `OCANNL_HALF_FMA` had two arms: clang's `__builtin_elementwise_fma`, which rounds once at fp16,
  and everyone else's `FLOAT_TO_HALF(fmaf(...))`, which rounds at float and again at fp16 — so on a
  target with genuine fp16 arithmetic, gcc disagreed with clang and with every GPU backend's
  single-rounding `__hfma` / `fma(half, …)`. It now takes `__builtin_fmaf16` where the ISA has the
  instruction (`__AVX512FP16__` or `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` — exactly what
  `cc_backend`'s fp16 probe calls `Native`), which is also 3-5x fewer instructions, because the
  promoting arm widens and narrows every lane inside the loop. The divergence is not a corner case:
  a single-rounded fp16 FMA and one promoted through float differ on about one triple in ten
  thousand (42039 of 4.1e8 searched), and on one in ~29000 even when all three operands are
  *normal* (3393 of 9.9e7). The tempting dismissal is wrong — `float`'s 24 bits are the `2p + 2`
  that makes double rounding innocuous for a single multiply or add, but an FMA's exact `a*b + c`
  can need far more than 24 bits and the guarantee does not extend to it. A search over well-scaled
  inputs near 1.0 finds zero divergences and reads as a false all-clear. Guard it on the ISA macro,
  never on `__has_builtin`, which always answers yes for `__builtin_fmaf16`: without the
  instruction gcc emits a call to `fmaf16()`, which a glibc need not export at all (verified here
  as a link error).
- **A vector accumulator update must reach the compiler as ONE vector operation** (gh-ocannl-614,
  gh-ocannl-621, fixed). gcc -O3 register-allocated the per-lane `fmaf` loop catastrophically: on
  the packed GEBP shape it unrolled the k-loop and spilled the whole C-tile — 18 insns / 8 vector
  FMAs / 0 stack refs per k step at `-O2` becoming 200 / 4 vector + 48 scalar / 8 at `-O3`, a ~9x
  throughput loss at every storage precision (`bin/narrow_gebp_bench` packmma: 112 → 12.6 GFLOP/s
  at f32, 94 → 10.4 at f16; the earlier claim that f32/bf16 were spared predates the extent-adapted
  tile width). The cause is the lane subscripting itself, not the unrolling: straight-line lane
  stores, a lane-indexed scratch array, a `(vtyp){...}` constructor of per-lane `fmaf` calls, and
  `#pragma GCC unroll 1` all spill the same way, while every form that arrives as one vector op
  costs the `-O2` count at `-O3`. `C_syntax.vec_fma_builtin` therefore adds middle arms to
  `vec_acc_fma` — target whole-vector FMA builtins mirroring clang's `__builtin_elementwise_fma`,
  with the per-lane loop still last. Not `dst = a * b + dst`, which allocates just as well but is
  only maybe-contracted: under `cc_backend_fp_contract=off` gcc emits a mul and an add, and
  `schedule_mma_matmul`'s full-mantissa leg (the only bitwise leg whose products are inexact, hence
  the only one that can see this) then fails.
- **The per-lane arm fails in TWO ways, and the second one is silent.** Besides spilling it can
  SCALARIZE: SLP declines to reassemble the lane loop, so one lanes-wide FMA becomes `lanes` scalar
  ones — a lane-count arithmetic loss with no stack traffic to give it away, which is why the
  census below counts vector and scalar FMAs separately rather than "FMAs". gh-ocannl-621 swept
  every reachable width this way (static census of the innermost FMA-carrying loop of the emitted
  micro-kernel, `-O2` vs `-O3`, insns / vector FMAs / scalar FMAs / stack refs), and the verdict is
  not uniform:
  - **gcc/aarch64 is the worst-affected target and it is every Linux ARM box's default** (Apple
    clang takes the `__builtin_elementwise_fma` arm and never sees any of this). f32 × 4 lanes:
    294 / 8 / 48 / 58 at `-O3` against the builtin's 26 / 16 / 0. f64 × 2 lanes scalarizes
    completely at *both* levels — 0 vector FMAs, 32 scalar.
  - x86 f64 × 2 lanes (16-byte vectors) scalarizes the same way at both levels; f64 × 8 (AVX-512)
    goes 28 / 16 / 0 → 417 / 8 / 96 at `-O3`.
  - f32 × 16 (AVX-512) is the one width where the per-lane arm is *fine*: 31 insns against the
    builtin's 28, no spill, no scalarization. Its row is robustness, not a measured win.
  - At fp16 the vector arm is not where the cost was; see the `OCANNL_HALF_FMA` note below. Once
    that macro rounds once, SLP reassembles the per-lane fp16 loop into exactly what the explicit
    builtin emits, so the fp16 rows are robustness rather than a measured win — but their guards
    have to be the *same* target question as the macro's, or the vector body would round once
    against a scalar peel rounding twice.
  - `vec_acc_combine`'s `Max`/`Min` loop USED to keep the per-lane form, on the reasoning that no
    builtin matches `fmaxf`'s NaN semantics. Both halves of that were wrong and gh-ocannl-649 fixed
    it — the per-lane spelling was not a slow vector loop but a libm CALL per lane, and a faithful
    whole-vector form exists (a mask blend everywhere, gcc's NEON `fmax`/`fmin` = `FMAXNM` on
    aarch64); see the gh-ocannl-649 bullet below for what to expect now. The diagnostic advice here
    is also superseded: `cc_backend_optimization_level=2` does not confirm this class, because the
    calls are there at `-O2` too.
- **`-U__FMA__` does NOT disable the builtin arm of a generated kernel** — `<immintrin.h>`, which
  the prelude includes under `__AVX2__`, re-defines `__FMA__` through `#pragma GCC target("fma")`
  in `fmaintrin.h`, and the definition survives the matching `pop_options`. An A/B run that way
  measures the same arm twice and reads as "the spill is gone"; it cost gh-ocannl-621 a false
  negative before the preprocessed output was compared. Force the per-lane arm by deleting the
  `#elif` lines from the emitted `.c` instead, and keep a positive control: building
  `bin/narrow_gebp_bench` at 97e7d286 (gh-614's parent) still reproduces 12.57 GFLOP/s against
  HEAD's ~128, and its kernel censuses 199 / 52 / 6 at `-O3`.
- **Widths that no local hardware can execute still get checked, three ways.** gh-ocannl-621's
  AVX-512, AVX512-FP16 and aarch64 rows were written on a machine with none of them (QEMU's TCG
  implements neither AVX-512 nor AVX512-FP16 — `query-cpu-model-expansion` on `-cpu max` reports
  every `avx512*` flag false — and no qemu-user was installable without root). What is checkable
  without the hardware: (1) the arm compiles under the target its guard names, and renders exactly
  one fused instruction at `-ffp-contract=off`, with the operand order readable off the asm;
  (2) the builtin is the same call gcc's own `<immintrin.h>` inlines for `_mm512_fmadd_ps` and
  friends, so its semantics are the ISA's; (3) the register-allocation census above, which is a
  compile-time property and needs no hardware at all. A cross toolchain for (1) and (3) on ARM is
  two commands without root: `apt-get download gcc-aarch64-linux-gnu …` then `dpkg-deb -x` into a
  prefix. What that leaves unverified is a wrong *signature*, which is why the new rows carry a
  second guard the shipped AVX/AVX2 rows do not — `__has_builtin`, which on gcc tracks enabled
  target features exactly, so a compiler that spells the builtin differently falls through to the
  per-lane arm instead of failing the kernel compile. (`__has_builtin` is useless for
  `__builtin_fmaf16`: it always answers yes, and without the ISA feature gcc emits a call to
  `fmaf16()`, a symbol glibc does not necessarily export — verified here as a link error. That one
  is guarded on the feature macro.)
- **That three-way check is now a test, not a shell loop** (gh-ocannl-650):
  `test/operations/cc_march_census` compiles the emitted kernel under seven `-march` targets at two
  optimization levels and three vector widths, and censuses the loop carrying each accumulator
  update. The generated `.c` includes only libc headers, so the whole matrix needs a
  toolchain and nothing else; the ARM columns need a cross gcc, pointed at by `AARCH64_CROSS_GCC`
  and reported through `Verdict.skipped` when absent, so the golden does not depend on the box
  having one. Two shapes are load-bearing and were each arrived at from their failure. The census
  ordinarily picks the SMALLEST-span loop carrying the construct, identified through the `.loc`
  line numbers `-g` leaves in the assembly rather than guessed from the instruction mix — an outer
  loop dilutes every ratio, and a surviving serial tail is a smaller loop mentioning the same array,
  which is why the fixture's extent is a multiple of `chains * lanes` at every width. fp8 uses the
  immediate outer carrier because its codec contributes an intentional nested loop. The census
  counts vector ops,
  scalar FP ops and libm calls SEPARATELY, never consulting them to select: scoring only what the
  good outcome produces made a fully scalarized loop read as "no FMA loop found", a pass arrived at
  from the failure (gh-ocannl-621). Answering "no loop" is therefore a failure — which is what
  caught `Asm_census` not knowing that gcc spells the aarch64 conditional branch `bne`, not `b.ne`:
  72 rows of silent absence across both ARM columns, reported at once.
- **The census reaches the narrow-storage arms and the GEBP grid too** (gh-ocannl-752). gh-650's
  fixture censused only loops whose storage precision equalled their compute precision, so
  `vec_bridge`'s widening and narrowing arms, `Ops.c_convert_precision`'s bf16/fp8/fp16 codecs and
  `try_register_tile`'s RMxRN grid were in no generated source at all — the compile matrix proved
  nothing about the paths real kernels take. Three additions, each with a trap worth keeping.
  (1) **fp16's compute precision is a per-process POLICY**, not a fact about the format
  (`Numerics.fp16_mode`), so one emission cannot carry both the native-fp16 builtin rows and the
  (fp16, f32) bridge pair: the driver emits under two `fp16_arithmetic` settings and each child
  asks `Numerics.cpu_compute_prec` which side it is on rather than being told a second time through
  the environment. (2) **A `Tile_mma` anchor must name something only the tiled rendering emits**
  (the A-splat binding `tmma_as_0__`, and its B-row load), which is what makes a silently DECLINED
  tiling — the
  gh-ocannl-479 failure mode — report "no loop carried the anchor" instead of censusing the scalar
  fallback under the tile's name. Its rows are also legitimately scalar in part (`rm` A splats per
  k step, measured at exactly 4 on every x86 column with FMA), so they carry a vector-majority
  inequality of their own rather than joining the scalarization claim. (3) **A coverage claim is
  needed where every other claim is an inequality**: a loop that stopped being narrow-storage — a
  flipped numerics default, a lost `vec_bridge` arm — is still a `Vectorized` fold over an array
  called `maxs_bf16` and passes all of them, so the fixture would report coverage it no longer has.
  What closes that is reading the EMITTED SOURCE: every `OCANNL_VEC_WIDEN_*`/`OCANNL_VEC_NARROW_*`
  key `Builtins_cc` defines must reach some kernel (derived from the table, so a bridge added for a
  fourth format fails until the fixture grows), and every loop whose storage and compute precisions
  differ must carry both directions of its `c_convert_precision` pair in its own kernel — the
  source-level attestation fp8 needs because `vec_bridge` has no vector arm for it and converts per lane.
  The emission is its own negative control: the native-fp16 kernel carries the `BFLOAT16` bridge
  pair and neither `HALF` one, the wide-fp16 kernel exactly the reverse. fp8 additionally joins the
  accumulator libm claims: `Asm_census.Smallest_outer_anchor_carrier` selects the smallest loop
  containing its nested per-lane codec, rather than letting that codec answer for the combine
  (gh-ocannl-845); a synthetic libm-bearing combine is the negative control.
- **A census reading is a fact about the emission AND about the compiler, and CI runs two of them**
  (gh-ocannl-752). The extended fixture passed on a gcc 15.2 box and was red on BOTH CI legs, in two
  unrelated ways, neither reachable from a gcc-only host. (a) **Line attribution.** A row is found
  by matching the DWARF `.loc` lines inside a loop against the source lines its anchor matches, and
  clang attributes an inlined function's instructions to the CALLEE's lines while gcc attributes
  them to the call site. The bf16 tile's anchor was `tmma_as_0__ = bfloat16_to_single(...)` — a line
  whose only content is a call to a function defined in the same kernel — so on macos-latest no
  `.loc` for it appeared anywhere in the k-loop and every bf16 tile row reported "no loop", failing
  three claims at once. clang folding a bare load into its use does the same to a plain
  `tmma_as_0__ = tmma_a__[...]` under some `-march`es. The fix is a LIST of anchors per row
  including a macro use or a `__builtin_memcpy`, whose instructions no inliner can move; a
  disjointness claim keeps the extra patterns from making two loops answer for each other. Reproduce
  without a Mac: `apt-get download clang-21 libclang-cpp21 libllvm21 libclang-common-21-dev` plus
  the arm64 cross libc, `dpkg-deb -x` into a prefix, and point `AARCH64_CROSS_GCC` at a wrapper
  running `clang --target=aarch64-linux-gnu --sysroot=...` — the two aarch64 columns then exercise
  clang's attribution on arm64. (b) **Whether a whole-vector builtin is lowered whole-vector.** On
  `-march=sapphirerapids`, where `_Float16` is a native arithmetic type, gcc 13.4 (ubuntu-latest)
  lowers the `__builtin_convertvector` that `OCANNL_VEC_WIDEN_HALF` is made of one lane at a time —
  `vcvtsh2ss` per lane, 4/8/16 of them at the three widths — while gcc 15.2 emits one `vcvtph2psx`,
  from identical emission and with the arithmetic packed either way. 18 rows failed "no accumulator
  loop is scalarized" on a claim that was really about the compiler's version. So the claim is now
  two: `vector_ops = 0` (wholly scalar — the gh-621 class, stable across both compilers) over every
  row, and `scalar_fp_ops = 0` over the rows whose bridge THIS toolchain lowers packed, asked by
  compiling that one conversion and censusing it rather than by testing a version. Reproduce with
  `apt-get download gcc-13-x86-64-linux-gnu cpp-13-x86-64-linux-gnu libgcc-13-dev` and
  `OCANNL_CC_BACKEND_COMPILER_COMMAND=<prefix>/usr/bin/gcc-13` on the census exe. Neither compiler
  substitutes for the other here. A row first asks for its exact tensor-derived source lines, then
  falls back to the smallest brace-delimited range containing them only when no loop carries an exact
  line. That keeps GCC's established narrow selection while clang's different attribution no longer
  makes x86 `dot` rows disappear; the modeled set is stated as GCC/GAS and Clang assembly on x86-64
  and aarch64, with an ELF/clang attribution probe alongside the existing dialect probes
  (gh-ocannl-844). The ORDER between the two readings needs a fixture of its own -- the clang probe
  pins the case where the exact anchor answers nothing, so the readings never disagree in it --
  which is `anchor_precedence_probe`: a construct whose brace range also names a staging loop GCC
  nested beside the accumulator, shorter than it and therefore what `Innermost` reports as soon as
  the widening fires unconditionally. (c) **How the
  assembler SPELLS a packed instruction.** Apple's arm64 assembler carries a NEON instruction's
  arrangement on the MNEMONIC (`fmla.4s v0, v1, v2`) where GAS carries it on the registers (`fmla
  v0.4s, v1.4s, v2.4s`), so `Asm_census`'s operand rules saw three plain `v` names and classified 34
  of the bf16 tile k-loop's 49 instructions as neither vector nor scalar. Unlike the three earlier
  Mach-O details — dot-less `LBB` labels, the checksummed `.file`, the `;` loop-header annotation —
  this one does not make rows go missing: the loop is found, its span and instruction total are
  right, and only the vector/scalar SPLIT reads zero. So it surfaced as ONE claim failing `0 > 0` on
  `mma/bf16`, the only tile row with no `ldr q0` or `fcvtl v2.4s, v0.4h` for the operand rules to
  catch by accident — `mma/f32` and `mma/f16w` PASSED the same claim on 3 and 10 accidental counts
  out of 28 and 78 real ones. Fixed by reading an arrangement suffix on the mnemonic too (`b.ne` is
  a dot that is not an arrangement, so the suffix is matched against the arrangement list rather
  than merely detected), and the dialect probes now claim the COUNTS of one loop in both spellings,
  not just that it is found — the earlier probe already carried `fmax.4s` and asserted only
  found-ness, which is why the blindness shipped. Reproduce without a Mac: the clang prefix above
  plus `-mllvm -aarch64-neon-syntax=apple`, which reproduces CI's row byte for byte (`insns=49
  vector=0 scalar_fp=0`). The lesson generalizes past this census: a fixture for a foreign dialect
  owes a claim about what was MEASURED, not only about what was found. Every profile also reports a
  residual count for instructions matched by no vector, scalar-FP, libm, or stack classifier; it is
  descriptive because loop control belongs there, but an unknown mnemonic can no longer vanish into
  passing-looking zeroes (gh-ocannl-844).
- **`Max`/`Min` SIMD reductions were a libm call per lane, on every x86 target** (gh-ocannl-649,
  fixed). The `Vectorized` accumulation loop rendered them as a fixed-trip per-lane loop calling the
  scalar `fmaxf`/`fminf`, on the reasoning that the packed-max builtins have the wrong NaN semantics
  and SLP would reassemble the lanes. SLP never gets the chance: gcc will not contract `fmax` into
  `maxsd`/`maxss` without `-ffinite-math-only`, so each lane compiled to a library call, at every
  `-march` from `x86-64` to `x86-64-v4` and both optimization levels — and at `-O3` the loop was
  fully scalarized besides (x86-64-v3, 64-byte width, f32: 259 instructions, 0 vector ops, 190
  scalar, 64 libm calls, 126 stack refs, against the FMA loop's 20/16/0/0/0 in the same kernel).
  The lesson generalizes past this operator: an opaque CALL cannot be vectorized at any grid size or
  register budget, so the register-pressure reasoning that the comment used to justify per-lane
  spelling was answering the wrong question. Check for calls before reasoning about allocation.
  The replacement is `fmax`'s own definition as a mask blend on GNU C vector comparisons (`a >= b`
  or `b` is NaN selects `a`), which agrees with glibc bitwise over every ordered pair of
  `{±0, ±1, ±inf, ±NaN, ±denormal}` at f32 and f64 except the two cases C leaves unspecified — which
  of `±0` a tie returns, which payload a both-NaN pair returns — and both of those a `Vectorized`
  retype reassociates away regardless. On aarch64 a builtin arm precedes it: gcc's internal NEON
  `fmax`/`fmin` render as one `FMAXNM`, which is IEEE `maxNum` and so exactly `fmax` (the
  NaN-propagating `FMAX` is a different builtin, `__builtin_aarch64_fmax_nan*`). That arm is what
  closes fp16, where gcc scalarizes a 16-bit vector COMPARISON whatever type it is spelled in
  — `_Float16` and `__fp16` alike, at `-O2` and `-O3` — widening every lane to float: 343
  instructions and 172 scalar ops become 9 and 0.
- **On a target with no FMA instruction the FMA is a libm call per element, on BOTH the vector and
  the scalar path** (gh-ocannl-753, documented rather than fixed). Three facts compose into it, none
  of them local to the SIMD emission. The simplifier rewrites every `a*b + c` into `Ternop (FMA, …)`
  (`Low_level`, no config gate); `Ops.ternop_c_syntax` spells that `fmaf(` / `fma(` for every C
  backend; and `C_syntax.vec_fma_builtin`'s x86 rows are guarded on `defined(__FMA__)`, so on a
  compiler WITHOUT `__builtin_elementwise_fma` (gcc) an FMA-less target matches no row and
  `vec_acc_fma` falls through to its per-lane `#else` loop. Under clang the chain never gets that
  far — `OCANNL_HAS_ELEMENTWISE_FMA` is 1, the first arm wins, and LLVM scalarizes the builtin into
  the same `fmaf` calls; same destination, different emitted path, and the distinction matters to
  anyone reading a kernel. Where the target has the instruction the compiler NORMALLY contracts the
  calls into it and none survives — normally, not always: that is the compiler's lowering decision
  and it is not guaranteed at every `-O` the backend can be set to (gcc is reported to keep the
  calls at `-O0` even on an FMA-capable target). Where the target does not have it, each
  stays a CALL — unless the arithmetic is relaxed, `cc_backend_fast_math` being the one setting that
  measurably expands them (under clang) — and by the gh-ocannl-649 lesson one bullet up, an
  opaque call cannot be vectorized, so the loop around it can degrade badly — gh-ocannl-753's census
  has it fully scalarized at `-O3` (64-byte width, f32, `-march=x86-64`: 324 instructions, 0 vector
  ops, 256 scalar, 64 libm calls, 128 stack refs). That census is GCC's, and the amplifier is not
  universal: clang measured here keeps the surrounding loop packed at both `-O2` and `-O3` (26
  packed ops, 0 scalar) while still emitting the per-lane calls. Three traps in reasoning about it, each of which cost this
  note a review round. **Do not name a feature macro as the condition. The condition is whether the
  compiler can lower `fmaf` to SOME fused instruction the target has** — two successive proxies were
  written here and both were wrong, each failing on a real part and each failing the same way, by
  prescribing instructions a CPU cannot execute. `x86-64-v3` was wrong because AMD Piledriver and
  Steamroller (`bdver2`/`bdver3`) carry FMA3 with AVX and no AVX2: already fast, and
  `-march=x86-64-v3` would hand them AVX2 they fault on. `__FMA__` was then wrong because Bulldozer
  (`bdver1`) has FMA4 ONLY — `__FMA4__` and `__XOP__` defined, `__FMA__` absent — and still compiles
  the loop to four-operand `vfmaddps`/`vfmaddss` with zero calls, so it has no cliff either, while
  the `-mfma` remedy that macro suggests selects FMA3 specifically, and **do not talk yourself into
  it being safe behind `-march=bdver1`** — that combination's outcome is COMPILER-DEPENDENT and is
  not settled: defining `__FMA__` is exactly what makes the emitted kernel take the
  `#elif defined(__FMA__)` arm and call `__builtin_ia32_vfmaddps`, for which clang was measured here
  to keep four-operand `vfmaddps` but gcc is reported to emit three-operand `vfmadd132ps`, which a
  Bulldozer cannot execute. (An earlier revision of this note called it safe on the strength of a
  clang probe of a *generic* `__builtin_elementwise_fma` loop — a different code path than the one
  the table emits.) On such a part the architecture flag alone already supplies FMA4, so `-mfma`
  buys nothing there anyway; over an architecture lacking FMA4 it emits FMA3 (`vfmadd132ps` /
  `vfmadd231ps`) outright. A
  consequence worth keeping DISTINCT hides in that last case, and it is COMPILER-CONDITIONAL:
  `vec_fma_builtin`'s x86 rows are keyed on `defined(__FMA__)`, so on a compiler WITHOUT
  `__builtin_elementwise_fma` — gcc — an FMA4-only target matches no builtin row and lands on the
  per-lane `#else`: no libm calls, but precisely the arm gh-ocannl-614/621 measured as gcc's
  spill-or-scalarize hazard. Under clang none of that applies, since `OCANNL_HAS_ELEMENTWISE_FMA` is
  1 and the first arm wins before any `__FMA__` guard is consulted. Escaping the libm cliff and
  taking the good arm are two different questions, and it is gcc + bdver1 specifically that answers
  them differently. (The bdver family is not uniform on the AVX2 half either: Excavator, `bdver4`,
  has AVX2 and FMA3 both — it is bdver1 through bdver3 that cannot execute AVX2.) Ask the
  compiler-and-target pair directly rather than a macro, and **do not hand-build the flags to ask
  it**: `compiler_flags` composes `-O` (`cc_backend_optimization_level`), `arch_flags`, the resolved
  `simd_flags`, `-ffast-math`, `-ffp-contract=` and `-fopenmp`, so a probe omitting any of them
  answers about a different compilation — an explicit `cc_backend_simd_flags=-mfma` makes a bare
  `-march=x86-64` probe report calls the real kernels do not have, and `cc_backend_fast_math` can
  erase calls the probe still shows (measured, f32 × 4 lanes at `-march=x86-64 -O3`: 4 calls, and 0
  once either `-mfma` or `-ffast-math` joins the line). Compile your own small `fmaf` loop under that
  flag set with `-S`, then `grep -E 'call.*fmaf?\b' kernel.s` — name the file LITERALLY: a bare
  `grep` waits on stdin and its empty output means nothing, while WITH the operand a no-match is a
  real negative (exit status 1) — that assembly is clear of the libm cliff, though NOT thereby on the
  good arm: a gcc/FMA4-only kernel takes the per-lane `#else` with no calls to find and keeps the
  spill-or-scalarize hazard, per the paragraph above. A placeholder
  like `<name>.s` is worse, since the shell reads `<` and `>` as redirection and would truncate the
  very file you are inspecting. Match both NAMES and both dialects.
  Double-precision kernels call `fma`, not `fmaf` (`Ops.ternop_c_syntax` emits `fma(` for
  `Double_prec`), so the symbols are `callq _fma` / `callq _fmaf` on clang+Mach-O (measured here)
  and `call fma@PLT` / `call fmaf@PLT` on gcc+ELF (review-reported, no GNU gcc here) — the
  double-precision one has no `f`; an `fmaf`-only pattern finds
  none of a double kernel's calls — measured 0 of 5,
  against 5 of 5 for `fmaf?\b`, on the same `-march=x86-64` object whose calls are `callq _fma`.
  That is the same dialect trap that cost `Asm_census` 72 rows of silent absence in gh-ocannl-650,
  one bullet up. gcc is also reported to keep the calls at `-O0` even on an FMA-capable target,
  where clang measurably does not — which cuts one way only: at the `-O` the build actually uses,
  finding the calls IS the answer about those kernels; what a low-`-O` result cannot support is
  extrapolation to a different level.
  (`-march=<t> -E -dM` prints which macros a target defines: an input to the question, not the
  answer to it.) **Every claim here carries its source class, and that is the discipline this bullet
  cost the most to learn.** Measured = executed on arm64 macOS with Apple clang cross-targeting
  x86_64. Code = read off the expression named, never off its comment — after
  `c_syntax.ml`'s "promises to equal bit for bit" was restated here as a whole-result guarantee the
  renderer does not make (see the rationale below), the rule is that **a comment is a claim by a
  past author, not a specification: cite the code or a transcript, never the comment**. Everything
  gcc-sourced — calls kept at `-O0`, `-ffast-math` not expanding them, the `@PLT` spellings, the
  three-operand `vfmadd132ps`, the fully scalarized `-O3` loop — is from review on gh-ocannl-753 or
  that issue's census, unreproducible here for want of a GNU gcc, and is flagged at each use. To skip the flag reconstruction entirely and read the backend's own logged compile
  command, see the bullet below — it works, at `OCANNL_LOG_LEVEL_CC_BACKEND=9` with a text backend.
  **It is a property
  of the TARGET, not of which arm of the `#if` chain is taken**:
  `OCANNL_HAS_ELEMENTWISE_FMA` carries no target guard, so clang always takes the first arm, and
  LLVM then scalarizes `llvm.fma` into the same `fmaf` calls — "clang, so the elementwise builtin,
  so fine" is wrong. And **it is not reached only by exotic hardware**, nor by anything as tidy as a
  date cutoff — Intel's low-power line (Silvermont, Goldmont, Tremont) and AMD Jaguar (`btver2`)
  define neither `__FMA__` nor `__FMA4__` and reach it under a successful `-march=native`, long
  after Haswell and Bulldozer. On the configuration side: `cc_backend_arch_flags=none`
  (the `reproducible` profile's pin — which also sets `cc_vector_bytes=0`, so only the serial calls
  remain there), an explicitly pinned FMA-less `-march`, or a toolchain that rejects the probed
  `-march=native` (`arch_flags` then falls back to no flag at all) each select such a target on a
  modern host. No x86 hardware is needed to check any of this from an Apple Silicon box: Apple clang
  cross-targets x86_64, so `clang -arch x86_64 -march=<target> -O2 -S` over a four-line
  `__builtin_elementwise_fma` loop counts `callq _fmaf` directly — 4 per iteration at 4 lanes and 8
  at 8 lanes under `x86-64` and `x86-64-v2`, zero under `x86-64-v3`, `bdver2` and `x86-64 -mfma`
  (`vfmadd*`) and under `bdver1` (`vfmaddps`/`vfmaddss`, the FMA4 forms), with the plain scalar
  `fmaf` loop going the same way. **Scope every claim here to the compiler that
  produced it.** All of the above is clang. The one claim that does not survive the change of
  compiler is `cc_backend_fast_math`: clang expands the calls to `mulps`/`addps` under `-ffast-math`
  (verified at `-march=x86-64`), recovering the vectorization at two roundings, but gcc was reported
  in review to keep the `fmaf` call there at `-O3 -ffast-math` — unverified here, this fleet's Macs
  have no GNU gcc, and gcc is the common default since `cc_backend_compiler_command` follows
  `ocamlc -config`. So fast-math is not a portable workaround — and neither is "raise the target",
  which only helps where the CPU HAS an FMA that a conservative target was hiding; on a genuinely
  FMA-less part there is nothing to raise to and changing the arithmetic is the only lever left.
  `cc_backend_fp_contract` is not a lever either way: the calls are fused by construction, and
  `-ffp-contract=fast` leaves every one of them.
  For FLOATING-POINT precisions the cliff is a cost of CORRECTNESS and the fix is not obvious — but
  **state the promise as per-UPDATE rounding, not as path-level bit equality**, or the rationale
  claims something the renderer does not deliver. A vectorized reduction reassociates by
  construction — read it off the emission, not off a comment: `vec_acc_grid_fold` folds the register
  grid into one accumulator and `vec_acc_lane_fold` then chains that vector's lanes into a scalar, so
  lanes accumulate independently and combine at the end, a different summation order from the serial
  fallback's. (Consistent with `cc_vector_bytes=0` being what the `reproducible` profile pins to keep
  the serial order, and with the width-ladder bullet below calling a newly-vectorized loop a numerics
  change.) What IS guaranteed is that every update is the same single-rounded operation in
  the vector body, its scalar peel and the serial fallback, so those differ by summation order
  alone; `dst = a * b + dst` on a machine without the instruction rounds each update twice, which is
  a second and independent numerics change stacked on the reassociation. (`c_syntax.ml`'s own
  comment says the paths "promise to equal bit for bit" — read that as the per-update promise, and
  do not repeat it as whole-result parity, which the reassociation defeats.) The
  2026-08-27 wave decision was to document that here and at `cc_backend_arch_flags` in
  `ocannl_config.reference`, and to leave the config-gated double-rounding arm to the
  approximate-numerics work (gh-ocannl-719) — where it would have to be taken by the peel and the
  serial fallback too, or the promise breaks worse than the cliff costs. If a fix does land,
  `test/operations/cc_march_census` is where it shows: its "no FMA accumulator loop calls libm where
  the ISA has a fused multiply-add" claim excludes the FMA-less rows by design, so widening that
  claim is the test-side move. **That census does not currently see FMA4 at all**, which matters
  before citing it as cover for anything written above: `has_fma` is
  `has "__FMA__" || has "__ARM_FEATURE_FMA"` with no `__FMA4__`, so `isa_has` drops the FMA row on
  an FMA4-only target, and `toolchains ()` has no `bdver` row to raise the question anyway. It can
  therefore stay green no matter what the FMA4 path does — adding `__FMA4__` to the predicate and a
  `bdver1` column is unfiled work, and until it exists the FMA4 claims in this note rest on the
  hand probes recorded here rather than on the suite.
- **Reading the backend's own compile command: use `OCANNL_LOG_LEVEL_CC_BACKEND=3` or higher and a
  text backend** (gh-ocannl-753, gh-ocannl-823). `cc_backend.ml` logs its exact invocation as
  `[%log3 "command", _cmdline]`, which is the direct answer to "what flags did my kernel actually
  compile with". Three things stand between you and it, all established by running them. The gate is
  a PREPROCESSING one (`[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_CC_BACKEND"]`,
  declared as a `preprocessor_deps` `env_var` in `arrayjit/lib/dune`), so a runtime `log_level`
  alone leaves the call stripped — the same runtime-vs-preprocessing confusion recorded at the top
  of that dune file from gh-ocannl-628. The tracked scopes can be stripped independently of locals
  they reference, so those locals use leading-underscore names: this keeps warning 27 clean at
  every compile-time level without changing their evaluation or real uses. The CI guard builds
  `@check` at `=3`, the minimum level that retains the command log. The default `debug_backend=db`
  writes
  `log_files/<exe>/debug.db` + `debug_meta.db` (that directory follows `build_files_prefix` like
  every artifact — `log_files/<prefix>/` when set, flat `log_files/` at `.`), ppx_minidebug's binary
  database, which needs its
  rendering client; `debug_backend=text` gives readable output instead. With
  `OCANNL_LOG_LEVEL_CC_BACKEND=3`, `log_level=3`, `debug_backend=text` the command appears verbatim
  under a `command` node:
  `cc '<...>.c' -O3 -mcpu=native -o '<...>.so' -bundle -undefined dynamic_lookup > '<...>' 2>&1`.
  **Where that lands is decided by `log_main_domain_to_stdout`, and both answers are live in this
  repo.** It defaults to FALSE, and `Utils.get_local_debug_runtime` then resolves `log_file_stem`
  (default `debug`) into a filename — so the ordinary answer is the EXTENSIONLESS file
  `log_files/<exe>/debug` — or `log_files/<prefix>/debug`, or a flat `log_files/debug`, as
  `build_files_prefix` dictates — and watching stdout finds nothing. But `test/config/ocannl_config`
  sets
  `log_main_domain_to_stdout=true`, so a run under `dune runtest` prints it inline instead and
  writes no file at all — which is what a golden diff then trips over, a perturbation of the test's
  stdout rather than a signal about the compile. That difference is worth stating because searching
  the wrong one of the two produces a confident wrong conclusion: globbing `log_files/**/*.log`
  under the test config found nothing (there was no file, and it would have had no `.log` suffix
  either way), and an earlier revision of this note concluded from that emptiness that the whole
  route was broken and deleted it. One more caution: the logged command carries `-o <...>.so`, so
  rerunning it with `-S` added but `-o` unchanged writes assembly over the shared library, possibly
  one a running process still has mapped — repoint `-o` at a fresh `.s`.
- **The mul-add → `Ternop (FMA, …)` rewrite is guarded to floating point, and that guard is
  load-bearing rather than a rounding preference** (gh-ocannl-824). Downstream,
  `Ops.ternop_c_syntax` renders `Byte`, `Uint16`, `Int32`, `Uint32`, `Int64`, `Uint64` and `Fp8`
  precisions through the DOUBLE-precision `fma(`, and `C_syntax.vec_acc_fma`'s per-lane arm sends
  anything that is neither `Double` nor `Half` to `fmaf` — single precision. So an integer mul-add
  reaching either loses bits past 2^53 and changes wraparound semantics, and there `a*b + c` is the
  RESTORATION of correctness, not a second rounding: the bit-parity rationale one bullet up is a
  floating-point argument only, and must not be read as covering the integer path. Reachability is
  established, not hypothetical — through hand-built low-level IR rather than ordinary
  `Assignments` lowering: an `int64` `Set` of `(x * 1 + 1) - x` put through `Low_level.simplify_llc`
  produced one integer `Ternop (FMA, …)`, and executing it with `x` at 2^53 returned 0 instead of
  the exact 1, the double `fma` having rounded 2^53 + 1 back down to 2^53. The `Low_level` arm now
  carries `Ops.is_float` like its neighbouring reassociations, and
  `test/operations/simplify_fma_precision.ml` pins both directions — integer unfused, single
  precision still fused — plus that executed witness. What the guard does NOT cover is the codegen
  boundary: a hand-built integer-precision `Ternop (FMA, …)` is still renderable through the double
  `fma(`, which gh-ocannl-873 tracks rejecting loudly before rendering.
- **"No such hardware" is a claim about a machine, not about the project — so name the machine.**
  The rows above were written from an Arrow Lake-HX box, where AVX-512 is fused off across the
  whole hybrid part, and the note recorded that as if it held everywhere; the machine that actually
  benchmarks CPU work here is Zen 5, which has AVX-512 F/VL/BW/DQ/BF16 and ran both AVX-512 rows
  correctly on the first attempt. Agents work in worktrees on several boxes, and this file
  deliberately carries no machine facts, so an unqualified "cannot be run here" reads as a
  project-wide limit and suppresses exactly the check it should have prompted. Check with
  `gcc -march=native -E -dM -x c /dev/null | grep AVX512` and say which host answered. What DOES
  hold fleet-wide: no AMD part implements AVX512-FP16, so those rows stay compile-checked until an
  Intel Sapphire-Rapids-or-newer P-core part joins the fleet — ARM reaches native fp16 through the
  NEON rows instead.
- **The auto SIMD width is 64 bytes on an AVX-512 target** (gh-ocannl-621 follow-up), probed the
  way `simd_flags`' first stage probes AVX2 — asserting what the configured target already has,
  never requesting more — and gated on that AVX2 stage having fired, so `cc_backend_simd_flags=none`
  still answers 16. Worth 1.7x on the packed f32 GEBP of `bin/narrow_gebp_bench` at n = 512 on
  Zen 5 (225.7 vs 130.5 GFLOP/s; f16 189.6 vs 108.1, bf16 140.6 vs 95.1), checksums identical.
- **A single width per machine would make a wider machine emit LESS vector code, and "the widest
  that fits" is still not enough.** Every explicit-SIMD rendering declines below one full vector,
  so at 64 bytes an f32 loop of extent 8..15 — and a `Tile_mma` column extent in that range — falls
  to the serial or scalar path that the same code vectorizes at 32. And at extent 40, 16 lanes
  cover 32 columns and peel 8 while 8 lanes divide it evenly: the wide machine running the narrow
  one's kernel plus a scalar tail. So the width is RANKED over a ladder
  (`Backend_intf.simd_lane_ladder`, halving to a floor of `min vector_bytes 32`): the loop
  renderings minimize trips (`extent / lanes` vector steps plus `extent mod lanes` scalar
  iterations — one instruction per body op either way, so no fitted constant is needed), and the
  register tiling folds the ladder into the peel-cost search it already ran over `rn`. The
  register-pressure cap stays keyed on the MACHINE's width: stepping down does not shrink the
  register file, which is why n = 40 still comes out ahead on the wider machine (118.0 GFLOP/s at
  a 4x5 tile of 8-lane vectors against the 32-byte machine's 103.3 at 4x1). The floor is not
  cosmetic — degrading below the machine's pre-widening width would newly vectorize loops that
  render serially today, and a vector accumulation reassociates, so that is a numerics change
  rather than a scheduling one. Autotune's seeding pre-filter calls the same function, or it would
  withhold candidates the renderer would in fact tile.
- Computing fp16 in fp16 on a *promoted* target is a ~18x loss against f32-compute-over-fp16
  (measured, same bench) — the reason `fp16_arithmetic` is ignored off-native and pure-f16 seeds
  gate on `hardware_limits.native_fp16_arithmetic`. The decisive pure-f16-vs-f32-GEBP measurement
  on genuinely native hardware (NEON) has NOT run yet; see the gh-575 proposal doc's pending note.
- The traffic win is real but it favors **fp16, not bf16**, the reverse of gh-ocannl-517's
  expectation. On an M-series at n = 2^22, a bandwidth-bound elementwise add measures 131 GB/s at
  f32, 1.97x that at half storage, and **0.91x** at bf16 (`bin/narrow_storage_bench.exe`); a
  compute-bound control stays below 1x for both, as it must. bf16's round-to-nearest-even narrowing
  is four vector ops against fp16's single NEON instruction, and at stream speed that costs more
  than halving the bytes saves. The route to competitive bf16 is a hardware convert (`BFCVT` on
  ARMv8.6-A, AVX512-BF16 on x86) — but only if it can be shown to agree with `single_to_bfloat16`
  bitwise, NaN payloads included, or the vectorized rendering stops matching its serial twin.
- Benchmark trap: `Context.get_values` walks the whole buffer into an OCaml `float array`, an O(n)
  host-side cost that does **not** depend on storage precision. Timing it inside the measured region
  (as `bin/cpu_vectorization_bench.ml` did until gh-ocannl-517) makes every kernel look equally slow
  — an order of magnitude below the machine's stream bandwidth — and exactly masks any traffic
  difference. Keep readbacks outside the timed region; the `cc` scheduler is synchronous, so no
  separate await is needed.

- **CUDA and HIP compile with fast-math umbrellas; Metal pins safe arithmetic while retaining fast
  math functions**: CUDA passes `--use_fast_math`, HIP `-ffast-math`, and Metal selects
  `MathMode.Safe` plus `MathFloatingPointFunctions.Fast` (measured below). `cc` is the exception
  (opt-in `cc_backend_fast_math`). HIP additionally
  passes `-fno-associative-math` **after** the umbrella flag: fast math had let hiprtc reorder
  ordinary scalar bf16/f16 recurrences differently across loop, unrolled and scope-local spellings,
  defeating `accum_prec`'s promised per-update storage rounding (gh-ocannl-735). This must be a
  compiler option, not a kernel-body pragma, because bf16 operators parsed in the HIP headers retain
  their floating-point flags after inlining. Schedule forms that license reassociation still spell
  it explicitly; the override only prevents the compiler inventing another one. HIP passes
  `-fhonor-infinities` after both: this codebase emits `(-INFINITY)` as a VALUE (the `Max` neutral
  element, `Nn_blocks.default_mask_fill`), and under bare `-ffast-math` that only survived by
  accident of which optimization hiprtc happened to pick — adding `-fno-associative-math` changed
  the pick and `half_softmax`'s causally masked rows started reading `exp(0)`/`exp(1)` where the
  mask demands exact zeros (gh-ocannl-735, found on the ROCm validation run). NaN stays unhonored,
  because it is only ever tested for, never emitted as a value. So a device-side
  non-finiteness test must be a RANGE COMPARE of a runtime value (`-3e38 < x && x < 3e38`); `x <> x`
  and `x - x = 0` fold to a constant, silently disabling an overflow gate (the shape
  `Mixed_prec.gated_scaled_update` needs). It is the same reason `Builtins_metal`'s fp8 codec is
  written in integer/bitcast form rather than float arithmetic.
- **First suspect for a `reduction_forms` red on a GPU backend: that backend's fast math
  reassociated the recurrence.** HIP is the only one guarded by a flag (above). The other two:
  - **CUDA is a measured boundary, not a flag** (gh-ocannl-784). nvrtc has no `-fno-associative-math`
    — every clang-shaped spelling is rejected as an unrecognized option — so there is nothing to
    pass. What was measured instead, on the project's CUDA box (nvrtc 13.3, driver 610.62, GeForce
    RTX 5070 Ti, sm_120), is that there is nothing to guard against: a 128-term float reduction
    compiled through nvrtc and **executed** returned the bit-exact strictly-sequential value under
    `--use_fast_math`, and also under `--use_fast_math` plus `--extra-device-vectorization`, plus
    `--dopt=on --ptxas-options=-O3`, and plus `--fmad=false` — in all three of the spellings that
    broke hiprtc (a counted loop, a runtime-bound loop, 128 repeated statements), plus an
    `(a+b)-a` cancellation probe. The detector is not blind: the same reduction built by host gcc at
    `-O3 -ffast-math` does reassociate, and `-fno-associative-math` restores it. nvrtc 13.3's parser
    *does* know `--fassociative-math` — a bare opt-in flag, with no negative spelling and no entry in
    NVIDIA's nvrtc option reference — which is why `Compiler_options.nvrtc`'s test claims MEMBERSHIP
    (that vector never contains it) where HIP's claims ordering. The probe is
    `tools/nvrtc_reassoc_probe.c` (standalone C, build line in its header) — re-run it after a
    toolkit upgrade before suspecting anything else, and read its EXIT STATUS: 0 only if every
    variant the verdict rests on compiled and returned the sequential value. The failing nvrtc
    compile's own effective vector is appended to its log by the backend, and a red GPU sweep unit
    carries an `rtc-context` block (`tools/sweep.sh`'s `rtc_context_cmd`) with the box's toolkit and
    driver versions, the include-slot discovery input, and the builder's option POLICY from the
    GPU-free test — whose include and arch slots are test sentinels, not that box's values.
  - **Metal is guarded by `MathMode.Safe`, with fast math functions retained** (gh-ocannl-848).
    Measured on M4 Max, macOS 26.6.2 / Metal 4, MSL 3.1: Default, Fast and Relaxed all changed the
    cancellation probe [(a+b)-a] from strict 0 to [b] = 1; Safe preserved 0. All modes kept three
    128-term reduction spellings bit-exactly sequential and a runtime `-INFINITY` mask at zero.
    The exact production setter sequence (Safe followed by Fast functions) measured 0.991x default
    for 262144 threads x 128 fixed-count terms and 1.006x for one thread x 1048576 runtime-bound
    terms (GPU-clock medians, 21 rotated interleaved repeats). The legacy
    `fastMathEnabled` property (set to false) also selects Precise functions, so the production state instead pins
    the two modern properties separately on macOS 15+. macOS 14 lacks those selectors; the backend
    detects the Objective-C methods before constructing the pure state and there falls back to
    setting `fastMathEnabled` to false as the safe, compatible choice. `Compiler_options.metal` owns
    both ordered property sequences; `test_metal_compile_options` pins both API/debug variants
    GPU-free and
    `benchmarks/runners/ocannl/metal_reassoc_probe.ml` is the rerunnable measurement.
- **`__FAST_MATH__` audit**: negating one component of a fast-math umbrella can *unset the macro*
  while leaving the remaining optimizations on — clang defines `__FAST_MATH__` only when the whole
  bundle is in force, so HIP's `-fno-associative-math` (and `-fhonor-infinities`) turn it off. No
  OCANNL-emitted code reads it, and today's ROCm narrow operators do not branch on it either, so
  nothing depends on this. It is a **vendor-header upgrade hazard**: a future `hip_fp16.h`/
  `hip_bf16.h` that selects a faster operator body under `#ifdef __FAST_MATH__` would silently take
  the slow branch on HIP alone, and the symptom would be a performance or precision difference
  between backends with no OCANNL-side change to point at. When bumping ROCm (or any RTC vendor
  headers), grep the SDK include tree for `__FAST_MATH__` and check whether any header OCANNL
  includes newly conditions on it.

- **An `external` typed `int array` carries no length, so a stub reading fixed fields must check
  `Wosize_val` first** (gh-ocannl-688). `builtins.c`'s uint4x32 helpers take an OCaml array and read
  lanes 0..3 out of it; nothing in the OCaml type says the caller passed four. An under-length array
  is then an out-of-bounds read that is *usually invisible* — it picks up adjacent heap words and the
  wrong random number is discarded — which is why such a mismatch can sit in the tree indefinitely.
  `ocaml_array_to_uint4x32` now raises `Invalid_argument` on any arity but 4; `arrayjit_copy_with_padding`
  is the older instance of the same check. When adding a stub that reads a fixed number of fields off
  an OCaml block, validate the arity at the boundary — the type system is not doing it for you.
- **Fingerprint: SIGBUS / `KERN_PROTECTION_FAILURE` inside `caml_c_call` under a `camlFoo$entry`
  frame is an FFI over-read at module-initialization time, not a JIT or W^X problem.** The macOS
  crash report names the faulting address; if it equals the top of a 2 MB `rw-` `VM_ALLOCATE` region
  followed by a `---` one (OCaml 5 commits domain 0's minor heap at the low end of a 256 MB
  PROT_NONE reservation), the stub read past a block that happened to be the topmost allocation in
  the minor heap. Allocation-layout dependence is what makes it present as flakiness correlated with
  nothing meaningful — load, launcher, concurrency — so do not chase the correlation. To reproduce
  deterministically, put the short block at the top of a fresh minor heap: `Gc.minor (); let a =
  Array.make 1 0 in <call>`. That turns a 3-in-5 flake into 5-in-5 (`test/operations/uint4x32_stub_bounds.ml`).
