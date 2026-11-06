# Candle Study: Lessons for OCANNL from a Hand-Written-Kernel Framework

**Issue:** [ahrefs/ocannl#265](https://github.com/ahrefs/ocannl/issues/265)
**Status:** Draft proposal

## Status update (2026-06-12)

- Issue [#265](https://github.com/ahrefs/ocannl/issues/265) is **OPEN**, milestone **v0.9** (GH milestone due-date 2026-05-30 is stale; per ROADMAP.md, v0.9 targets Aug 24, 2026 — ICFP week).
- The study has **not** been done — no write-up or issue comment exists (harness task status: deferred).
- Sibling-study states: #242 (TVM) and #316 (DumPy/torchdim) CLOSED/completed; #301 (IREE) and #306 (Petalisp/Caten) CLOSED as **not planned**; #267 (Tiramisu) still OPEN as a draft proposal.
- **Assumption invalidated:** gh-ocannl-271 ("Support quantization for optimizers: low-bit optimizers") was CLOSED as **not planned** (milestone v1.1). The "quantization findings feed directly into #271" framing below no longer has a live target issue; quantization lessons from Candle would need a new home (e.g. a fresh issue) if the study surfaces them.
- OCANNL-side code pointers re-verified: all listed backend files exist unchanged; CUDA kernels still launch single-threaded (`grid_dim_x:1, block_dim_x:1` in `cuda_backend.ml`), so the v0.8 framing stands.
- Op-coverage drift: `lib/nn_blocks.ml` has since gained RoPE (rotary embeddings via the `pos_embed` strategy, #444), `batch_norm1d`, and decoder-only transformer blocks; a tensor stacking operation with block-literal `%op` syntax also landed. The op-gap catalogue in study area 3 should be re-derived against current `nn_blocks.ml`, not the list sketched below.
- Verdict: still actionable as a study task; the quantization-related acceptance criterion needs retargeting.

## Goal

Survey [huggingface/candle](https://github.com/huggingface/candle) and identify
what is transferable to OCANNL, while being explicit about what is not. The
issue author already noted upfront that Candle's goals are *less* aligned with
OCANNL than Luminal's: Candle uses **hand-written kernels per backend**, whereas
OCANNL **generates code** from its `Low_level.t` IR. The study should respect
that framing -- focus on what a code-generating framework can still learn from a
minimalist hand-written-kernel framework, rather than treating Candle as a
template to imitate.

This continues the exploration series alongside the TVM deep dive (#242,
completed), DumPy/torchdim (#316), Petalisp/Caten (#306), Tiramisu (#267), and
IREE (#301). Where those compare OCANNL to other compilers, Candle gives a
contrasting data point: a *non-compiler* framework that achieves competitive
inference performance through careful curation of operations and ergonomic
APIs.

## Acceptance Criteria

- [ ] Study Candle (architecture, backends, op coverage, model zoo,
      quantization story) and produce a written summary covering each area.
- [ ] Explicitly contrast Candle's hand-written-kernel approach with OCANNL's
      generated-code approach, calling out where the difference matters and
      where it does not.
- [ ] Identify concrete transferable lessons -- for example, in API design,
      operation coverage gaps, or quantization patterns -- with a brief
      assessment of value and effort for each.
- [ ] Note the relationship to gh-ocannl-271 (quantization, where Candle's
      GGUF support is most directly relevant) and any other tasks where
      Candle observations apply. *(Update 2026-06-12: #271 was closed as
      not planned — file a new issue if the study surfaces actionable
      quantization work.)*
- [ ] Post the findings as a comment on issue #265 and/or land them as the
      final write-up replacing this proposal.

## Context

This is a research/exploration task. The deliverable is a comparison document;
no OCANNL source-tree changes are required to satisfy the acceptance criteria.

**Candle in one paragraph (per the issue and Candle's own docs).**
Candle is HuggingFace's minimalist Rust ML framework, designed for serverless
and edge inference with small binary footprints. Backends include CPU (with
optional MKL/Accelerate), CUDA (with Flash Attention v2 and NCCL), Metal, and
WebAssembly. Like PyTorch, each operation is implemented per backend by hand;
unlike PyTorch, the surface area is deliberately curated. Candle emphasizes
shipping working, efficient implementations of the operations that real LLM and
vision models actually need.

**OCANNL's contrasting position.**
OCANNL compiles tensor expressions through a defined pipeline -- `%op`/`%cd`
syntax extensions, `Assignments.t`, lowering to `Low_level.t`, optimization
(virtualization, simplification), and shared C-syntax codegen via
`c_syntax.ml`. Backends (`cuda_backend.ml`, `metal_backend.ml`, `cc_backend.ml`)
plug into the shared codegen and currently emit single-threaded kernels
(`grid_dim=1, block_dim=1` is forced in CUDA -- the v0.8 milestone is about
relaxing this). The architectural axis where Candle and OCANNL differ most is
exactly the one the issue author flagged: code generation vs. hand-written
kernels.

### Suggested study areas

1. **Architecture comparison**
   - Tensor representation, dtype/device handling, autodiff approach
   - Backend dispatch: how Candle routes ops across CPU/CUDA/Metal/WASM, and
     what OCANNL's analogue is (the `Backend` interface in
     `arrayjit/lib/backends.ml` and per-backend modules).
   - Lazy vs. eager: does Candle build any kind of graph before execution? How
     does that compare to OCANNL's compile-then-run model?

2. **Metal backend, side by side**
   - Candle's `candle-metal-kernels` directory contains the actual Metal
     shaders. Compare a couple of representative kernels (e.g. matmul,
     softmax, layernorm) to what OCANNL's `metal_backend.ml` plus
     `c_syntax.ml` produce for the same operations.
   - This is the most concrete place to see "hand-written vs. generated" in
     practice and is most relevant to the v0.8 GPU milestone.

3. **Operation coverage**
   - Catalogue Candle's op set; identify operations Candle ships that OCANNL
     does not yet have, especially anything required by the standard model
     zoo (RoPE, RMSNorm, GQA, KV-cache primitives, top-k sampling helpers,
     etc. -- some of these are already tracked by other tasks).
   - For each gap, note whether OCANNL likely needs the same op, a generated
     equivalent built from existing primitives, or no change.

4. **Quantization**
   - Candle has substantial GGUF / quantized inference support. This is
     directly relevant to gh-ocannl-271 *(Update 2026-06-12: closed as not
     planned; see Status update)* and should be the area where
     Candle's lessons are most actionable. Capture: what dtypes Candle
     supports, where the dequantize-on-load vs. on-the-fly decisions live,
     how kernels are specialized per quantization scheme, and what Candle
     learned the hard way that OCANNL can avoid.

5. **Model zoo and ergonomics**
   - The set of models Candle ships ends-to-end (Llama, Mistral, Whisper,
     SD, Phi, etc.) is informative: it implicitly defines the operations and
     control flow a "useful" framework must support. Note which model
     families are well-covered and which ergonomics choices (config loading,
     tokenizer integration, streaming generation) feel transferable.

6. **What is *not* worth porting**
   - Be explicit about cases where Candle's hand-written kernel (or any other
     approach) is incompatible with OCANNL's generated-code direction --
     so that the write-up does not accidentally argue OCANNL should imitate
     Candle's lower layers.

### Code pointers (OCANNL side)

- Backends interface: `arrayjit/lib/backends.ml`, `backends.mli`,
  `backend_intf.ml`, `backend_impl.ml`
- Metal backend: `arrayjit/lib/metal_backend.ml`,
  `metal_backend_impl.metal.ml`, `builtins_metal.ml`
- CUDA backend: `arrayjit/lib/cuda_backend.ml`,
  `cuda_backend_impl.cudajit.ml`, `builtins_cuda.ml`
- Shared C-syntax codegen: `arrayjit/lib/c_syntax.ml`
- Low-level IR: `arrayjit/lib/low_level.ml`, `assignments.ml`
- NN building blocks (where op-coverage gaps surface): `lib/nn_blocks.ml`
- README context: `README.md` (lines 9, 85-86 -- the "inspirations" and
  "schedule statically by program search" notes)

### Code pointers (Candle side)

- Repository: <https://github.com/huggingface/candle>
- `candle-core/` -- tensor, ops, backend dispatch
- `candle-nn/` -- NN building blocks (analogue of OCANNL's `lib/nn_blocks.ml`)
- `candle-metal-kernels/`, `candle-kernels/` (CUDA) -- hand-written shaders,
  the most direct comparison point
- `candle-transformers/` -- model zoo
- `candle-core/src/quantized/` and the GGUF loader -- relevant to #271

### Related tasks

- gh-ocannl-242 -- TVM deep dive (more aligned with OCANNL; completed)
- gh-ocannl-267 -- Tiramisu deep dive (still open)
- gh-ocannl-271 -- Quantization (closed as not planned as of 2026-06)
- gh-ocannl-301 -- IREE deep dive (closed as not planned)
- gh-ocannl-306 -- Petalisp/Caten deep dive (closed as not planned)
- gh-ocannl-316 -- DumPy/torchdim study (completed)

## Approach (optional)

*Suggested approach -- agents may deviate if they find a better path.*

1. Skim the Candle repo top-down: `README.md`, the `candle-book/` if present,
   `candle-core` module layout, and the list of crates. Establish a mental
   map.
2. Read one end-to-end model in `candle-transformers` (Llama is a good
   choice -- well-known shape) to see how ops compose into a model, and
   compare to the analogous OCANNL example under `bin/` or `test/`.
3. For sections 1, 3, 5, 6 above, prose comparison is enough. For sections 2
   (Metal) and 4 (quantization), include short representative code excerpts
   from both projects -- that's where the contrast becomes concrete.
4. Land the final write-up either by replacing the body of this proposal
   with the findings (and committing) or as a comment on issue #265, per
   the convention used by the TVM, Petalisp/Caten, and DumPy studies.
5. If during the study you discover follow-up work that OCANNL should do
   (op gaps, ergonomic improvements, quantization patterns), file them as
   separate issues rather than expanding this task.

## Scope

**In scope:**
- A written comparison covering the six study areas above.
- Identification of transferable lessons with effort/value notes.
- Posting findings as a comment on issue #265 and/or replacing this
  proposal body with the final write-up.

**Out of scope:**
- Implementing any Candle-inspired changes in OCANNL -- those are tracked
  separately (quantization formerly via #271, now closed not-planned; GPU
  performance via the v0.8 milestone).
- Benchmarking Candle against OCANNL.
- Adding a Rust toolchain or Candle as any kind of dependency.

**Dependencies:**
- Reads cleanly alongside the other "study" tasks (#242, #267, #301, #306,
  #316). Most actionable findings will likely feed into quantization work
  (formerly #271, now closed as not planned — would need a fresh issue)
  and the v0.8 GPU performance milestone.
