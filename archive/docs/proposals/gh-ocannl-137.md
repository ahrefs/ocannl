# Quantization Support — Disposition Memo

## Goal

Address [gh-ocannl-137](https://github.com/ahrefs/ocannl/issues/137) —
"Implement support for quantization" — by producing a written disposition
document, not by implementing quantization. The issue is labelled `explore`,
milestoned `v1.1`, task effort `large` (8–12 days). Like other v1.1 explore
issues processed this Mag session (gh-ocannl-161, 200, 123, 278), the right
output is a memo that surfaces the disposition choice to the user rather than
a unilateral pick or a speculative design.

The issue has reframed itself across its lifetime, and the load-bearing
question today is no longer *"how should quantization be implemented?"* but
*"is the v0.7.1 transformer-inference demo (gh-ocannl-377) going to require
quantization, and if not, does anything else on the roadmap before v1.1
require it?"* This memo answers both questions (no in both cases, as best
the orchestrator and worker can verify), enumerates four dispositions, and
recommends one.

## Acceptance Criteria

This proposal is a memo recommending a disposition. It is satisfied when the
user picks one of:

- [ ] **(A — recommended)** Issue #137 is closed as `not-planned`, with a
      closing comment quoting the user's 2023-09-27 comment ("This isn't
      really worthwhile, we would rather need the significantly harder
      support for quantization") and linking to this memo. Task
      `gh-ocannl-137` is marked abandoned. If a specific quantization
      use-case becomes load-bearing later (e.g., a LLaMA/Gemma demo
      requiring GGUF weights), a narrow follow-up issue is filed at that
      time.
- [ ] **(B)** Issue #137 stays open as a v1.1 placeholder, this memo is
      committed as the standing context, and a thin "design memo" follow-up
      task enumerates the prerequisite chain (extended precision types,
      block-quantized storage, GGUF/safetensors parser, dynamic indexing
      reintroduction per gh-ocannl-186, mixed-precision compute paths,
      Adam-precision interaction per gh-ocannl-271).
- [ ] **(C)** Issue #137 is rescoped to "weight-only int8/int4 quantization
      for inference + GGUF loader," tied explicitly to gh-ocannl-377 if and
      only if a LLaMA/Gemma demo (rather than GPT-2 only) is in scope for
      v0.7.1. QAT, PTQ-with-calibration, mixed-precision training, and the
      original "quantized dynamic indexing" concern are deferred to separate
      issues. **Conditional on gh-ocannl-377's actual scope** — see
      *Entanglements* below.
- [ ] **(D)** Sibling-merge: gh-ocannl-137 is merged into gh-ocannl-163
      (ggml efficiency lessons) as a single quantization-feasibility track,
      since 163's proposal already explicitly defers quantization to 137.
      One tracking issue, one design memo, implementation deferred.

The user picks. The recommendation is **(A)** with a fallback to **(C)** if
gh-ocannl-377 confirms LLaMA/Gemma scope.

Out of scope (explicitly):

- Implementing quantization, scale/zero-point parameters, or any conversion
  kernels.
- Choosing among quantization schemes (int8 vs int4 vs fp8 vs mixed; symmetric
  vs asymmetric; per-tensor vs per-channel vs per-group).
- Designing a GGUF or GPTQ parser.
- Redesigning `Ops.precision` / `prec` GADTs to carry scale metadata.
- Re-litigating the original "quantized dynamic indexing" concern from the
  2023 issue body — that was explicitly retracted by the user in the
  2023-09-27 comment.

## Context

### History of the issue (and why "quantization" here is ambiguous)

The issue title has stayed stable since 2023, but its meaning has flipped:

- **2023-05-04 (issue body):** "The part that's a bit confusing is dynamic
  indexing. For consistency, the underlying integers still need to be scaled
  before becoming the indexing integers." This is *not* model quantization —
  it's small-integer indexing arithmetic.
- **2023-05-04 (comment):** "It's probably more interesting than it is
  practical..." and "Let's consider this together with figuring out how to
  support `Float16`."
- **2023-09-27 (comment):** "This isn't really worthwhile, we would rather
  need the significantly harder support for quantization."

The 2023-09-27 comment explicitly retracts the original concern (the
indexing-integer scaling problem) and replaces it with a much harder problem
(real model quantization: int8/int4 weights, scale/zero-point, calibration,
mixed precision). The issue title was not updated. The task body in
`tasks/gh-ocannl-137.md` has tracked the *new* meaning, which is what the
acceptance criteria there reflect ("Tensors can be quantized to lower
precision (int8, fp8, int4) with scale/zero-point parameters").

### Verified state of OCANNL precision support

The task elaboration claims `Half`, `Bfloat16`, `Fp8`, `Uint4x32`, `Single`,
`Double` are present. Verified in `arrayjit/lib/ops.ml`:

- The `precision` GADT (around the *Precision* section header) defines all
  twelve constructors: `Byte`, `Uint16`, `Int32`, `Uint32`, `Int64`, `Uint64`,
  `Uint4x32`, `Half`, `Bfloat16`, `Fp8`, `Single`, `Double`. The `prec`
  variant mirrors this with `*_prec` constructors.
- `promote_prec` (in the same file) implements the precision-promotion
  lattice: `Uint4x32` dominates everything, then `Double`, `Single`,
  `Bfloat16`, `Half`, `Fp8`, then the integer types. The doc-comment on
  `promote_prec` reads "uint4x32 always dominates, because operations that
  work on uint4x32 do not support other precisions" — i.e., `Uint4x32` is a
  bit-exact 128-bit transport type, not a numeric precision in the
  arithmetic sense.

The task elaboration claims FP8 conversion builtins exist in
`arrayjit/lib/builtins_cc.ml`. Verified: the file defines `fp8_to_single`
(E5M2 → float), `single_to_fp8` (float → E5M2), `fp8_to_uint4x32`,
`uint4x32_to_fp8_uniform`, and a vectorized `uint4x32_to_fp8_uniform_vec`.
These are real C functions emitted into the builtin header, not stubs.

What is *not* present, verified by `grep -ni "quantiz\|GGUF\|GPTQ\|calibration"`
across `arrayjit/lib/`:

- No quantization-specific code at all. No `quantize`, `dequantize`,
  `Quantized`, `ScaleZeroPoint`, `BlockQ4`, etc.
- No GGUF or GPTQ parser. The only `int8` mention in core source is
  `int8x16_t` in `cuda_backend.ml` and `ops.ml`, used as a vector type for
  byte/FP8 transport — not a quantization-aware type.
- No calibration infrastructure. No `Calibration`, no histogram-based
  scale-factor estimation.
- The `Fp8` constructor is wired through the type system and has conversion
  builtins, but there are no operators that consume FP8 inputs and produce
  meaningful floating-point arithmetic — `apply_prec` raises `invalid_arg`
  for several FP8 paths, and FP8 values are treated as `unsigned char` in
  the C emit path (`ops.ml` `prec_string`: `Fp8_prec _ -> "unsigned char"`).

So the precision-type machinery is the *transport layer* for quantized
formats, but the *arithmetic semantics* of quantization (scale, zero-point,
fused dequantize-matmul, calibration, QAT gradients) are entirely absent.

### Strategic context (why net-new quantization work is unaffordable now)

Per `ROADMAP.md` and harness memory:

- v0.7 / v0.7.1 milestones: streams (#68), hosted tensor (#70), persistence
  (#373), RoPE (#65/#398), transformer toy (#66), tokenizers (#73), GPT-2/
  LLaMA/Gemma inference (#377). None of these explicitly require
  quantization.
- v0.8 milestone: CPU tiling (#79), CUDA performance (#80), llm.c (#81),
  loop hoisting (#76), AVX/AVX2 intrinsics (#164). CUDA kernels currently
  run with `grid_dim=1, block_dim=1`. This is the dominant cycle sink
  through mid-2026.
- ICFP 2026 OCaml Workshop / FProPer paper deadlines: May–June 2026.
- Autonomous OCANNL work is paused pending the user's hands-on quality
  audit; proposals are artifact-only.
- v1.1 ("Post-1.0 Considerations") in `ROADMAP.md` lists "Quantization for
  optimizers (#271)" as a *separate* explore item — that's low-bit Adam
  state, not weight quantization. gh-ocannl-137 is not listed in
  `ROADMAP.md` at all.

A real quantization implementation of the scope the task body describes
(QAT + PTQ + mixed-precision + GGUF/GPTQ + quantized dynamic indexing) is
plausibly multi-month, well past the 8–12 day "large" estimate. Nothing on
the roadmap before v1.1 makes it load-bearing.

### Entanglements (the load-bearing scoping question)

Two adjacent tasks make the disposition non-trivial:

#### gh-ocannl-377 (transformer inference demo, milestone v0.7.1)

The proposal `docs/proposals/gh-ocannl-377.md` and the task elaboration list
"Weight format loader (safetensors/GGUF)" as a missing component, but the
demo target is GPT-2 first (124M params, fp16/fp32 weights, no quantization
required). The task body explicitly notes "GPT-2 requires the fewest missing
components (only GELU + weight loading + tokenizer)." LLaMA/Gemma are
presented as *more complex* alternatives, not as the chosen target.

If GPT-2 is the actual v0.7.1 deliverable, **quantization is not on the
critical path for any roadmap item before v1.1**, and disposition (A) is
clean. If LLaMA/Gemma is the chosen target *and* the demo loads pretrained
weights from GGUF rather than safetensors, then a *narrow* GGUF loader
(read-only, dequantize-on-load, no quantized arithmetic) becomes
load-bearing — that's disposition (C). The disambiguation is a question
for the user, not a question this memo can settle from code.

The cheapest framing: the GGUF-weight-loader question is *separable* from
the quantization-arithmetic question. A read-then-dequantize loader (load
Q4_0 blocks → produce fp16/fp32 Bigarray) does not require any of the
five missing pieces enumerated in the task body — it's a parser plus
arithmetic. That work, if needed, belongs in its own narrow issue, not in
gh-ocannl-137's current scope.

#### gh-ocannl-186 (re-introduction of dynamic indexing, milestone v0.7.1)

The original 2023 issue body for #137 was about "dynamic indexing with
quantized values." That concern was retracted by the user in 2023-09-27,
and dynamic indexing itself was removed from OCANNL and is now tracked by
gh-ocannl-186 (which was elaborated this Mag session as a v0.7.1
prerequisite). If dynamic indexing is reintroduced and at some future point
quantized tensors exist, the original concern reactivates. But that
intersection requires *both* abstractions — neither alone gives it weight.
Disposition (A) is consistent with this: re-file at the intersection point
if and when both abstractions exist, not pre-emptively now.

### gh-ocannl-163 sibling overlap

`docs/proposals/gh-ocannl-163.md` (ggml efficiency lessons) explicitly
defers quantization to gh-ocannl-137: its disposition template includes
"already covered by gh-ocannl-137 (quantization)" as a verdict for any
quantization-flavored ggml lesson. The 163 proposal also flags **block
quantization with shared scale factors** as something that "plausibly
falls under gh-ocannl-137, but only if that task explicitly adopts a
block-grouped storage layout — currently it does not."

In other words, 163 already treats 137 as the canonical
quantization-tracking issue, and notes that 137's current acceptance
criteria don't explicitly include block-quantized storage (the ggml-style
encoding most relevant to GGUF). If 137 is closed as `not-planned`
(disposition A), 163's "already covered by gh-ocannl-137" verdict needs to
be replaced — likely by a narrow `file follow-up issue` for
block-quantized storage, *if* 163 ever runs as a research-note task. (163
is currently `deferred`.) Disposition (D) — sibling-merge — is the
cheapest way to keep these two issues from drifting independently.

### Disposition options

- **(A) Close as "not planned," recommended.** The user's own 2023-09-27
  comment is the strongest possible signal ("This isn't really
  worthwhile..."). The original concern was retracted; the harder concern
  has no roadmap motivation before v1.1. No partial work would be lost
  (zero quantization-arithmetic code in the tree). If a specific use-case
  becomes load-bearing later, a *narrow* follow-up issue (e.g., "GGUF
  read-only loader for #377 LLaMA demo") is cheaper to scope and execute
  than maintaining a 5-item omnibus task. This memo preserves the
  archaeology so a future re-opener doesn't have to redo it.

- **(B) Defer with design memo.** Land this memo, run a 1–2 day
  design-memo session that enumerates the prerequisite chain (precision
  types extended with scale/zero-point metadata, block-quantized storage
  layout, GGUF parser, dynamic-indexing-with-quantized-values per #186,
  mixed-precision compute paths, Adam interaction per #271), close 137
  against that memo. Useful only if the user wants a record of *how*
  quantization would be built when prerequisites land, distinct from the
  *what-and-why* in this document. Cost: 1–2 days of worker time, no
  code.

- **(C) Promote and narrow, conditional on #377 scope.** Rescope 137 to
  "weight-only int8 quantization + GGUF read-only loader for the
  LLaMA/Gemma inference demo." Defer QAT, PTQ-with-calibration,
  mixed-precision *training*, and quantized dynamic indexing to separate
  issues. **Only valid if gh-ocannl-377 confirms LLaMA/Gemma is the
  v0.7.1 target.** If GPT-2 is the target, the narrow scope is
  unmotivated and (A) is preferred.

- **(D) Sibling-merge with gh-ocannl-163.** Bundle 137 and 163 into one
  "quantization-design exploration" tracking issue, defer all
  implementation. Closes one of the two stale issues, keeps the design
  archaeology in one place, and aligns with 163's existing treatment of
  137 as the canonical quantization-tracking issue. Cost: light; mostly
  GitHub housekeeping.

### Why (A) is recommended

- Strongest historical signal: the user's own 2023-09-27 retraction.
- Zero partial work in-tree (verified above).
- No roadmap item before v1.1 makes quantization load-bearing
  (gh-ocannl-377 is GPT-2-first; LLaMA/Gemma is contingent and could be
  served by a narrow GGUF loader filed separately).
- The 5-item omnibus scope (QAT + PTQ + mixed-precision + GGUF + quantized
  dynamic indexing) is far past the "large" 8–12 day estimate and is best
  decomposed at the point of need, not pre-emptively.
- This Mag session has already produced 4 close-as-not-planned
  recommendations on v1.1 explore tasks (gh-ocannl-161, 200, 123, plus
  gh-ocannl-278's defer-with-doc); (A) here is consistent with that
  pattern when the criteria are met (no partial work, retraction signal,
  no roadmap dependency).

### Code pointers (by symbol, not line number)

- `Ops.precision` / `Ops.prec` GADTs in `arrayjit/lib/ops.ml` — current
  precision-type machinery; transport layer only.
- `Ops.promote_prec` in the same file — precision-promotion lattice; would
  need scale-aware variants if quantization-arithmetic types were added.
- `Ops.apply_prec` and `Ops.prec_string` — pattern-match dispatch over
  `prec`; current FP8 paths use `unsigned char` storage and `invalid_arg`
  for unsupported conversions, illustrating the gap between transport and
  arithmetic.
- `arrayjit/lib/builtins_cc.ml` — FP8 conversion builtins
  (`fp8_to_single`, `single_to_fp8`, `fp8_to_uint4x32`,
  `uint4x32_to_fp8_uniform`, `uint4x32_to_fp8_uniform_vec`). The natural
  place GGUF block-dequantize kernels (Q4_0/Q4_1/Q8_0) would land if
  disposition (C) is taken.
- `arrayjit/lib/cuda_backend.ml` `int8x16_t` mention — vector transport
  type for byte/FP8; not a quantization-aware type.
- `arrayjit/lib/ndarray.ml` `byte_nd` / `fp8_nd` — Bigarray wrappers; the
  hosted tensor representation a GGUF loader would target.
- `docs/proposals/gh-ocannl-377.md` — the inference demo proposal that
  determines whether a GGUF loader is on-path.
- `docs/proposals/gh-ocannl-163.md` — the ggml-lessons proposal that
  treats 137 as the canonical quantization-tracking issue.
- `docs/proposals/gh-ocannl-161.md`, `gh-ocannl-123.md`, `gh-ocannl-200.md`,
  `distro-feasibility-study.md` — sibling v1.1 disposition memos. The
  template here matches theirs.
- `ROADMAP.md` Post-1.0 Considerations — lists "Quantization for
  optimizers (#271)" as a separate explore item, does not list #137.

## Approach

*Suggested approach — the assigned worker may deviate.*

This document is itself the deliverable for disposition (A). For (B), it
seeds a deeper design-memo session. For (C), it's the design appendix for a
narrowed tracking issue. For (D), it's the merge document. The doc-first
move is no-regret in all four branches.

Concrete steps (only step 4 varies by disposition):

1. Land this proposal at `docs/proposals/gh-ocannl-137.md`.
2. Comment on gh-ocannl-137 with a link to this doc and a summary of the
   recommended disposition (A/B/C/D), letting the user pick.
3. Update task frontmatter (`proposal: docs/proposals/gh-ocannl-137.md`).
4. Branch by user's pick:
   - **(A):** close gh-ocannl-137 with a link to this doc. Add a TODO to
     gh-ocannl-163's proposal noting that 163's "already covered by
     gh-ocannl-137" verdict needs to be reinterpreted as `file follow-up
     issue` if 163 ever runs.
   - **(B):** create a follow-up task for the 1–2 day design memo; leave
     137 open until that memo lands.
   - **(C):** clarify gh-ocannl-377 scope first (GPT-2 only vs.
     LLaMA/Gemma). If LLaMA/Gemma, rewrite #137's body and acceptance
     criteria around weight-only int8 + GGUF; file separate issues for
     QAT, PTQ, mixed-precision, quantized dynamic indexing. If GPT-2
     only, fall back to (A).
   - **(D):** rewrite #137 and #163 into a single
     "quantization-feasibility track" issue; close one as duplicate of
     the other.
5. *No code changes in any branch.*

## Scope

**In scope**:

- Writing this disposition memo.
- Updating the GH issue to reference it.
- Surfacing the disposition question (A/B/C/D) and the gh-ocannl-377
  scope question to the user.

**Out of scope**:

- Implementing any quantization scheme or arithmetic.
- Designing GGUF or GPTQ parsers.
- Extending `Ops.precision` / `Ops.prec` GADTs with scale/zero-point
  metadata.
- Touching `builtins_cc.ml` to add dequantize kernels.
- Re-litigating the original "quantized dynamic indexing" concern —
  retracted by the user in 2023-09-27.
- Settling the gh-ocannl-377 GPT-2-vs-LLaMA scope question —
  surfaced here, decided there.

**Dependencies**:

- gh-ocannl-377 (transformer inference demo, v0.7.1) — disposition (C) is
  conditional on its scope. If GPT-2 only, (C) collapses to (A). If
  LLaMA/Gemma, a narrow GGUF-loader follow-up may be filed regardless of
  the disposition chosen here.
- gh-ocannl-186 (dynamic indexing reintroduction, v0.7.1) — the original
  2023 framing of #137 was the intersection of dynamic indexing and
  quantization. That intersection is moot until both abstractions exist.
- gh-ocannl-163 (ggml efficiency lessons, v0.8) — currently treats #137
  as the canonical quantization-tracking issue. Disposition (D) explicitly
  merges them; dispositions (A)/(B)/(C) require a follow-up edit to 163's
  proposal to reinterpret its quantization verdict.
- gh-ocannl-271 (low-bit optimizer state, v1.1) — separate quantization-
  flavored issue that is *not* subsumed by #137 (it's optimizer-state
  quantization, not weight/activation quantization). Out of scope here in
  all dispositions.
