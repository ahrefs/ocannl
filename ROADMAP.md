# OCANNL Roadmap to v1.0

**Headline target: ICFP 2026 week (August 24, 2026)**

This roadmap outlines the development plan for OCANNL from the current state to version 1.0, incorporating academic paper milestones for workshops collocated with ICFP 2026 (OCaml Workshop, FProPer). Dates indicate **end of period** targets.

> **Schedule note (June 2026):** the roadmap drifted from its original dating because of a slowdown between January and May 2026. We are now catching up. Three structural changes follow from that:
>
> - **v0.6.4 is skipped as a release.** Its scope — axis concatenation/block tensors (#49), RoPE and non-learned position embeddings (#398), the decoder-only transformer toy (#57) — is complete (the GitHub milestone is closed), but it ships inside **v0.7** rather than as a separate tagged release. The last tagged release is **0.6.3**.
> - **v0.7.2 is consolidated into v0.7.** The compiler-optimization and memory-management work that was scheduled separately (loop hoisting, CSE, the universal pool allocator) is part of the single **v0.7** milestone.
> - **v0.7.1 is dissolved.** Its two tracks were redistributed: the **AMD HIP backend (#411)** moves to **v0.8**, and the **real-world examples** (makemore #59, CNN/CIFAR #54, LSTM #60, transformer inference #377) and **tokenizer bindings** move to **v0.9**. The GitHub milestone has been deleted.
>
> The version sequence is now: `0.6.3 → 0.7 → 0.8 → 0.9 → 1.0 → 1.1`. Milestone *scope* below tracks the GitHub milestones, which are the source of truth.

---

## Released: Foundation (through 0.6.3)

The 0.6.x line stabilized the frontend: the Menhir einsum parser and "missing hidden dimensions" error detection (0.6.2), then padding inference for convolutions with a toy CNN (0.6.3). See [CHANGES.md](CHANGES.md) for details.

---

## v0.7 — Late June 2026
**Theme: Frontend finalization and compiler optimizations (paper-ready)**

This is the consolidated "paper-ready" release. It absorbs the frontend-finalization work originally split across v0.6.4/v0.6.5/v0.7.0 and the compiler-optimization work originally planned as v0.7.2. GitHub milestone scope: *"inlining- and simplification-related optimizations, memory management, session management."*

**Frontend finalization (done):**
- **Remove the hosted tensor mode** (#333) — got rid of the `array` field of `Tnode.t` and the "hosted" memory mode; value access and printing are now context-mediated.
- **Tensor persistence** (#373) — tensor saving, loading, and restoring.
- **Tensor-node ID namespaces** (#372).
- **Axis concatenation / block tensors** (#49) — `a^b` einsum syntax for stacking/concatenation, with shifting (`1^i=>i`) and padding (`i=>1^i`) as fixed-index special cases; n-ary block-tensor specs.
- **RoPE and non-learned position embeddings** (#398).
- **Decoder-only autoregressive transformer toy example** (#57).
- **Ternary einsum notation** (#305) and ternary projection inference.
- **Sasha Rush Tensor Puzzles** (#308) in extended einsum notation.
- **uint32/uint64 indexing precisions** (#349, #177) driven by the `big_models` setting.
- Identifier hygiene: blacklist primitive-operator/reserved names (#383); collapse repeated label components in `debug_name` (#281).
- Configuration: relax the required `ocannl_` CLI prefix and validate config keys (#409).
- `-march=native` C-compiler flag (#311); restore CUDA pre-loaded builtins via a cudajit helper (#353); remove remaining unnecessary buffer zeroing (#382); rename routine/kernel params to `kparam`/`kparams` (#356).

**Compiler optimizations (done):**
- **Loop-invariant code motion** (#350), prior to visit counting.
- **Common subexpression elimination** after inlining (#351).
- Extend virtual-node inlining to non-scalar constants and ranges (#142).

**Still open in v0.7:**
- **Universal Pool Allocator across backends** (#344) — in progress (buffer-addressing seam landed; full pooling scoped).
- **`Local_scope` initialization tracking** (#340).
- **MSVC support for the native-Windows C backend** (#313).
- **Sharding and slicing with minimal copying** (#293) — the data-parallel driver with merge-buffer all-reduce has landed; remaining work continues here.
- **Documentation:** flesh out `lowering_and_inlining.md` and audit `low_level.ml` (#296).
- Inlining stretch goals: share one `for` loop across virtual tensors (#134); inline virtual nodes with non-linear index symbols (#133).

This release is the basis for the workshop paper examples: a clean context-based API (no hosted tensors), shape concatenation, and a complete transformer with RoPE.

---

## v0.8 — Summer 2026
**Theme: GPU-style performance — low-hanging fruit; AMD HIP backend**

A substantial milestone (~2 months). GitHub milestone scope: *"GPU tiling and related optimizations in the polyhedral style, with heuristic syntactic metrics for now."*

- **Matmul tiling** (#412) — fast multidimensional matrix multiplication, first from Böhm's CPU article, then the CUDA worklog, then lessons from llm.c (#253).
- **Megakernel exploration** (#318, done as a study) — may require splitting routines into multiple kernels.
- **Metal private mode** (#320, done).
- **AMD HIP backend** (#411) — a major effort, comparable to the CUDA and Metal backends (redistributed here from the dissolved v0.7.1). Standalone HIP bindings ship as an independent GitHub project and opam package, following the same pattern as the CUDA bindings (`cudajit`) and the Metal bindings (`metal`), so the OCaml community can use them without taking on the weight of OCANNL; the `arrayjit` backend then **depends on** those bindings, with the usual code-generation, memory-management, and synchronization plumbing.
- Stretch / study: AVX/AVX2 intrinsics for the C backend (#164); `ggml` efficiency lessons (#163); restore CUDA `__constant__` arrays (#195); small-Transformer digit-addition reproduction (#427).

> **Date note:** the GitHub milestone still carries a stale 2026-02-28 due date from before the slip; treat the date above as authoritative.

---

## v0.9 — August 24, 2026 (ICFP week)
**Theme: Program search and optimization**

A research-heavy milestone (~2.5 months). GitHub milestone scope: *"Program search with execution-based per-backend or aggregate-of-backends cost functions; broadening code-graph rewriting rules."*

- **Static scheduling via program search** — an alternative to tinygrad's dynamic scheduling; Halide-inspired search.
- **Cost functions** — per-backend execution-based metrics and aggregate cost functions across backends.
- **Code-graph rewriting** — a broader range of rewriting rules, augmenting the v0.8 tiling/layout mechanisms.
- Study tracks: Tiramisu (#267), Candle (#265), superoptimizers for tensor programs (#261).

**Real-world examples and tokenization** (redistributed here from the dissolved v0.7.1):
- **makemore progression** (#59, done) — the character-level language-model series mirroring Karpathy's *Neural Networks: Zero to Hero* (see [docs/makemore_tutorial.md](docs/makemore_tutorial.md)); includes the Bengio-style MLP and BatchNorm variants.
- **CNN classifiers** (#54) — MNIST and CIFAR-10 training examples.
- **LSTM example** (#60).
- **Transformer inference demo** (#377) — inference for a small open-weights model (GPT-2, LLaMA, or Gemma).
- **Tokenizer bindings** — developed in the spin-off [ocaml-dataprep](https://github.com/ahrefs/ocaml-dataprep) project (opam package `dataprep`).

> **Date note:** the GitHub milestone carries a stale 2026-05-30 due date; the ICFP-week anchor above is authoritative.

---

## v1.0 — Q4 2026
**Theme: Documentation, completeness, ergonomics, safety**

GitHub milestone scope: *"Few documentation gaps, some degree of feature completeness, ergonomics, safety."* Already largely de-risked — key items below are done.

- **Safety (done):** static verification of merge-buffer nodes "in the right direction" (#288); rank-cycle detection for row variables (#247).
- **Determinism (done):** resolve `multicore_cc` non-determinism and restore it as the primary testing target (#341).
- **`%cd` ergonomics (done):** simplify translations from `%cd` (#348); accept `:=` for the `Fetch` constructor (#209).
- **Open — ergonomics:** concise syntax for merge-buffer transfers; execution dependency tracking (mirroring compilation); local let-bindings in `%cd` (#80).
- **Open — completeness:** demonstrate model surgery (#33); training checkpointing (#96); inference plugin/binary generation (#97); experiment tracking and plot improvements (#122, #103); `polars-ocaml` integration (#219).

---

## v1.1 and beyond
**Theme: Shape-inference and safety enhancements; advanced examples**

GitHub milestone scope: *"Consider introducing axis labels. Consider introducing shape schemes."*

- **Shape schemes for tensor functions** (#404).
- **Axis labels** (vs. the dimension basis) — design exploration.
- **Advanced examples:** BERT / ModernBERT (#297); DisTrO low-communication distributed data parallelism (#278).

---

## Key Milestones Summary

| Version | Target | Status | Key Deliverables |
|---------|--------|--------|------------------|
| 0.6.2  | Nov 2025 | released | Menhir parser, hidden-dimension errors |
| 0.6.3  | Dec 2025 | released | Padding inference, toy CNN |
| ~~0.6.4~~ | — | **skipped** (folds into 0.7) | Concatenation, RoPE, transformer toy |
| **0.7** | Late Jun 2026 | **in progress** | **Frontend finalization + compiler optimizations** (consolidates 0.7.2) |
| ~~0.7.1~~ | — | **dissolved** | AMD HIP backend → 0.8; examples + tokenizers → 0.9 |
| 0.8    | Summer 2026 | planned | GPU tiling, megakernels, matmul; AMD HIP backend (major) |
| 0.9    | Aug 24, 2026 | planned | Program search **(ICFP week)**; examples: makemore, MNIST/CIFAR, LSTM, transformer inference, tokenizers |
| 1.0    | Q4 2026 | mostly de-risked | Docs, completeness, ergonomics, safety |
| 1.1+   | post-1.0 | backlog | Shape schemes, axis labels, BERT, DisTrO |

---

## Workshop Paper Plan (OCaml Workshop / FProPer at ICFP 2026)

**Target deadline: per each workshop's CFP (typically May–June 2026).**

### Proposed Title
*"Generalized Einsum with Row Variables: Shape Inference for Deep Learning in OCaml"*

### Key Contributions
1. **Generalized einsum notation** with convolutions, strided iteration, and concatenation
2. **Row variables** for flexible axis handling ("principle of least commitment")
3. **Constraint-based shape inference** with provenance tracking for error messages
4. **Dimension basis** design rationale (vs. axis labels)
5. **Integration with OCaml's type system** via syntax extensions

### Related Work to Address
- einops (#413)
- torchdim / DumPy (#316)
- Named tensors in PyTorch/JAX
- Dependent types for tensor shapes

### Why v0.7 Before the Paper
The paper needs working examples on OCANNL's mature frontend, all delivered by v0.7:
- Clean context-based API (no hosted tensors)
- Shape concatenation syntax (`^`)
- Complete transformer example with RoPE
- Consistent, documented API surface

The deep semantic groundwork for the paper (the two-sorted ground algebra, the rank-fact graph and rank-cycle check, ≈-semantics for row equality) has been developing alongside v0.7 in the in-progress proposals.
