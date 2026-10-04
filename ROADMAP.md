# OCANNL Roadmap

**v1.1 released October 4, 2026: fused attention and better transformer scheduling. Next: consolidation (v1.1.1, October 16), performance beyond transformers (v1.1.2, October 24), consumers (v1.1.3, October 31) and v1.2 around November 15, 2026.**

This roadmap outlines the development plan for OCANNL through version 1.0 and beyond. Dates indicate **end of period** targets. Through v1.0 the schedule was pinned to conference deadlines; it is now project-internal, and the dates below are aspirational rather than external commitments.

> **Schedule note (July 2026):** the roadmap drifted from its original dating because of a slowdown between January and May 2026. v0.7 is the catch-up release. Three structural changes follow from that:
>
> - **v0.6.4 is skipped as a release.** Its scope — axis concatenation/block tensors (#49), RoPE and non-learned position embeddings (#398), the decoder-only transformer toy (#57) — is complete (the GitHub milestone is closed), but it ships inside **v0.7** rather than as a separate tagged release. The last tagged release before v0.7 was **0.6.3**.
> - **v0.7.2 is consolidated into v0.7.** The compiler-optimization and memory-management work that was scheduled separately (loop hoisting, CSE, the universal pool allocator) is part of the single **v0.7** milestone.
> - **v0.7.1 was dissolved.** Its two tracks were redistributed: the **AMD HIP backend (#411)** shipped in **v0.8**; completed examples and tokenizer work landed subsequently, while remaining examples now follow their current GitHub milestone assignments. The GitHub milestone has been deleted.
>
> **Update (August 2026):** the v0.9 milestone closed on schedule, and two rebalances came with that: CUDA/HIP graph capture (#488) moved from v0.9 to v1.0, and the training/deployment utilities plus the `lib/` design study moved out of v1.0, in favor of the compiler-tier and diagnostics work the v0.9 sweep exposed. (Their current homes are v1.0.1 and v1.1.2 — see the rebalance and renumbering notes below.)
>
> **Venue history (August 2026):** the OCaml Workshop submission was not accepted — the article was written as a research report rather than as an introductory demonstration, which put it outside that audience's scope. IFL 2026 was then considered as the next target and **decided against as a poor fit**. No conference submission is currently scheduled; the paper-facing artifacts below stay in the repository and the formal core technical report continues as live work. The workshop article and its PDF are kept unchanged, as a historical artifact capturing the state of the project at v0.8.
>
> **v1.0 shipped August 13, 2026**, with its milestone fully closed (49 issues). Three consequences for what follows: the release dates are no longer pinned to paper deadlines, **v1.1's soft target moves to August 24, 2026** (the OCaml Workshop date, used as an anchor rather than as a submission), and v1.1/v1.2 were rebalanced along a different seam than the original split — **v1.1 is the compiler work plus the training-loop mechanics it needs**, and **v1.2 is the consumers and explorations**: models, reproductions, demos, integrations, the training experience a user sees, and performance items gated on hardware or on evidence not yet in hand.
>
> **Update (late August 2026):** v1.2 was split along the performance seam. **v1.2 is now performance-chasing in the `approximate` profile, demonstrated on benchmarks** — the numerics-changing tier (fused attention, Winograd, tf32/fp16 arithmetic) behind a third preset, the exact-numerics performance residue, and the benchmark legs that expose where OCANNL wins and loses; the Winograd and zero-nest conv tiers (#505, #503) moved into it from v1.1. Everything else that was in v1.2 — the consumers, explorations, training experience, engineering hygiene, and hardware-gated items — is **v1.3**. With its numerics-changing carry-overs gone, **v1.1 reads as consolidation after v1.0**: the search follow-ups v1.0's evaluation filed, inlining and reduction soundness, the test and benchmark seams that cannot report a false pass, and the training-loop mechanics.
>
> **Renumbering (August 26, 2026):** the ladder was renumbered so that version-number depth tracks release *scope*, as it did through the 0.6.x line (0.6.1 shipped features; this project does not follow semver, and 0.x releases never did). Consolidation and robustness releases take a third component; feature releases take a second. Concretely: the release planned as v1.1 shipped as **v1.0.1**; a new **v1.0.2** pulls the robustness and engineering-hygiene backlog forward out of the feature milestones, so refactoring issues are worked while they still describe the code they were filed against; the performance milestone (formerly v1.2) is now **v1.1**; the consumers/demos milestone (formerly v1.3) is now **v1.1.1**; and a new **v1.2** holds the ambitious feature-grade work (shape schemes #404, CDNA MFMA #477, PoPE #444, mmap zero-copy #585, CUDA pinned host buffers #170 and `__constant__` arrays #195).
>
> **Dating (September 7, 2026):** the post-1.0.1 ladder is dated again, aspirationally. v1.2 targets **October 28, 2026**, and the milestones before it split the interval from September 7 by the scope their version depth signals (a feature release counts twice a third-component release, so 1 : 2 : 1 : 2 — issue counts were rejected as the weight, since v1.0.2's are review-filed follow-ups and the later milestones' are design spaces): **v1.0.2 September 16, v1.1 October 3, v1.1.1 October 11**. These are end-of-period targets rather than commitments; the GitHub milestone due dates then carried the same values.
>
> **Rebalance (September 14, 2026):** finish v1.0.2 with a bounded compiler-deduplication set, then start performance work promptly. The deferred consolidation becomes **v1.1.1**, and the former consumers milestone becomes **v1.1.2**. Working backward from **November 3** for v1.2 gives **October 18** for consumers, **October 10** for consolidation and **October 2** for v1.1, after **September 16** for v1.0.2. The 48 days after that cut split 16 : 8 : 8 : 16 (feature : consolidation : consumers : feature), preserving the scope weighting used in September. These are soft end-of-period targets, not requirements to empty each milestone; protect the early-November anchor by reducing scope when necessary.
>
> **Update (October 4, 2026):** the September 29 split keeps **v1.1 focused on transformer workloads**. Convolution and CPU kernel/sketch economy now have their own **v1.1.2** milestone; consumers moved to **v1.1.3**. **v1.1 released October 4** with fused attention and better transformer scheduling, ahead of its October 5 GitHub due date. Later GitHub targets remain October 16, 24 and 31, then November 15 for v1.2. Undated v1.2.1/v1.2.2 placeholders allow consolidation and consumers to continue if scope moves again. These dates supersede the September schedule above.
>
> The version sequence is: `0.7 → 0.8 → 0.9 → 1.0 → 1.0.1 → 1.0.2 → 1.1 → 1.1.1 → 1.1.2 → 1.1.3 → 1.2`. Milestone *scope* below tracks the GitHub milestones, which are the source of truth for issue assignments. Release outcomes distinguish delivered work from unfinished goals; they do not imply that every issue in a released milestone is closed.

---

## Released: Foundation (through 0.6.3)

The 0.6.x line stabilized the frontend: the Menhir einsum parser and "missing hidden dimensions" error detection (0.6.2), then padding inference for convolutions with a toy CNN (0.6.3). See [CHANGES.md](CHANGES.md) for details.

---

## v0.7 — July 3, 2026
**Theme: Frontend finalization and compiler optimizations (paper-ready)**

This is the consolidated "paper-ready" release. It absorbs the frontend-finalization work originally split across v0.6.4/v0.6.5/v0.7.0 and the compiler-optimization work originally planned as v0.7.2. GitHub milestone scope: *"inlining- and simplification-related optimizations, memory management, session management."*

**Frontend finalization (done):**
- **Remove the hosted tensor mode** (#333) — got rid of the `array` field of `Tnode.t` and the "hosted" memory mode; value access and printing are now context-mediated.
- **Tensor persistence** (#373) — tensor saving, loading, and restoring.
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
- **Universal pool allocator across backends** (#344) — tensors are addressed through pooled locations; working tensors are bump-packed per context delta, constants live in per-device pools, merge buffers stay reserved, and Metal uses pool slabs plus a slot table to avoid binding-limit pressure.
- **Sharding and minimal-copy slicing foundation** (#293) — `shard_along` / `gather`, data-parallel training with merge-buffer all-reduce, and zero-copy leading-axis slice views.

**Documentation and paper artifacts (done):**
- **`lowering_and_inlining.md` audit** (#296) — the lowering/optimization docs were fleshed out alongside a `low_level.ml` audit.
- **Workshop article:** `docs/ocannl_workshop_article_human.md`, LaTeX source, and rendered PDF.
- **Formal core technical report:** `docs/ocannl-formal-core-technical-report.latex` (the authoritative source) and its rendered PDF, covering the core shape/projection inference proof effort.
- **Shape constraint generation notes:** `docs/shape-constraint-generation.md`, documenting the front-end elaboration boundary from `shape.ml` into core constraints.

**Deferred after v0.7:**
- **Tensor-node ID namespaces** (#372).
- **`Local_scope` initialization tracking** (#340).
- Remaining sharding/slicing extensions beyond the v0.7 data-parallel and zero-copy leading-axis foundation (#293 follow-ups).
- Inlining stretch goals: share one `for` loop across virtual tensors (#134); inline virtual nodes with non-linear index symbols (#133).

This release is the basis for the workshop paper examples: a clean context-based API (no hosted tensors), shape concatenation, a complete transformer with RoPE, and a written formal account of the core shape/projection inference machinery.

---

## v0.8 — July 13, 2026
**Theme: Parallel schedules and autotuning; AMD HIP backend**

GitHub milestone scope: *"GPU tiling and related optimizations in the polyhedral style, with heuristic syntactic metrics for now. HIP backend (AMD hardware)."*

> **Outcome vs. the original plan:** measured schedule search was pulled in one release early, and the matmul track continued through generated tensor-core instructions (#412). Follow-up schedule-quality work is assigned according to the current GitHub milestones below.

**Parallel schedules (done):**
- **Automatic GPU schedules** — CUDA and Metal kernels now parallelize by default (`automatic_gpu_schedule`): hardware axis types render to grid/block/thread loops with launch dimensions, barriers, and shared-memory tiles; per-backend `hardware_limits` validate block sizes, thread counts, and shared-memory use of every kernel.
- **Kernel fission** — routines split into multiple kernels at materialized cross-nest edges, with aligned cross-nest parallelism merging equal-geometry nests losslessly; Metal encodes fissioned steps as fused command-buffer segments.
- **CPU kernel-level parallelism** — the `cc` backend renders parallel loops through a thread pool by default; backends renamed to `cc` / `multidev_cc` (`sync_cc` / `multicore_cc` remain as deprecated aliases).
- **Matmul tiling and tensor cores** (#412) — register-tiled `Tile_mma` microkernels, SIMD vector-extension codegen with reduction chains (#468, #469), shared/packed staging, warp shuffles, CUDA WMMA/inline PTX, Metal simdgroup matrices, and HIP rocWMMA. The first tranche shipped here; the issue itself continued into v0.9 and was closed there.

**Autotuning (done — pulled into v0.8):**
- **Measured schedule search** — `Autotune.tune` searches canonical schedule candidates with execution-based timing: a digest-guarded schedule cache, per-segment schedules for fissioned routines, sketch seeding (e.g. matmul), and placement A/B tuning.

**AMD HIP backend (done)** (#411) — implemented via the standalone `hipjit` bindings (independent GitHub project and opam package, following the `cudajit`/`metal` pattern), mirroring the CUDA backend's code generation, memory management, and synchronization.

**Benchmarks and platforms (done):**
- **Cross-framework benchmark suite** — `benchmarks/` compares OCANNL against PyTorch (including `torch.compile`) and tinygrad (including BEAM search), gated on loss-parity; checked-in example reports for Metal, CUDA, and Windows/HIP feed the workshop article's benchmark appendix.
- **Windows** — the full test suite is green on the `cc` and `hip` backends; the CUDA backend was restored on Windows (NVRTC arch floors for half/bf16 intrinsics).
- **Megakernel exploration** (#318, done as a study); **Metal private mode** (#320, done).

**Deferred after v0.8:**
- **MSVC on the native-Windows C backend** (#313, closed as not planned) — Windows remains supported through mingw-w64.
- **AVX/AVX2 intrinsics** (#164, done) — the delivered CPU bundle uses pool-backed `Grid` rendering, probed SIMD compiler flags, and portable compiler vector extensions rather than architecture-specific intrinsic calls.
- Stretch / study items not taken up: `ggml` efficiency lessons (#163); restore CUDA `__constant__` arrays (#195); small-Transformer digit-addition reproduction (#427).

---

## v0.9 — August 3, 2026
**Theme: Schedule quality, deterministic parallelism, and convolution performance**

A research-heavy milestone, closed with all 44 assigned issues resolved. GitHub milestone scope: *"Program search with execution-based per-backend or aggregate-of-backends cost functions; broadening code-graph rewriting rules."*

**Search quality and schedule legality (done):**
- **Constraint-based schedule legality** (#494) — `Ir.Affine` and `Low_level.affine_accesses` expose loop boxes and access relations; conflict, coverage, fiber-cardinality and read-before-write queries drive shared-memory safety, fission, scratch validation, and a `Schedule.op_legality` oracle that prunes proven-illegal proposals.
- **Analytic cost model** (#491) — footprint/FLOP extraction, roofline lower bounds, per-backend envelopes, model-picked untuned defaults (`model_default_schedule`) and a keep-fraction pre-filter over sketch seeds, with calibration logging. Advisory throughout: candidates without model coverage are never dropped.
- **Cross-machine benchmark and tuning sweep** (#476), re-measured under #538 after the search changes — full Metal, CUDA and HIP columns from a wiped autotune cache, plus paired A/B legs. The checked-in reports under `benchmarks/` are the record.
- **Pad-to-tile scheduling** (#485), **static partitioning** (#508), and the **tf32 numerics policy** (#478); CUDA 13 / cudajit fixes (#482).

**Parallel execution and numerics (done):**
- **Deterministic split reductions** (#484) — two-pass tree combines with autotune seeding, extended by `Swap`-hoist composition (#537) so conv-gradient accumulations became reachable.
- **Mixed-precision training recipe** (#492) — precision-assignment policy, master weights with cast twins, dynamic loss scaling with a fused on-device gate, and forward-only reduced precision by load-time conversion.
- **Packed-uniform retirement** (#509) — the packed `uniform` became total over shapes, took over default parameter initialization, and gained lane-extract virtualization.
- Correct overlapping-window gradients for tropical/einmax1 reductions (#512), with the non-overlapping fast path restored (#527).

**Convolution performance (done):**
- Implicit-GEMM sketch families (#493), blocked tile flavors (#500), epilogue twins (#501), compact strided-row staging (#502), and clamped-window lowering for padded max-family pooling (#504).

**Search survivability (done — scope that emerged from the sweep):**
- **Typed candidate-failure containment** (#536) with the Metal, CUDA and HIP arms; **HIP scratch pre-validation** (#533); unparallelized GPU candidates refused rather than dispatched (#532); the decline census made complete (#541, #543); advisory-fallback and model-ranking fixes (#519, #522); GPU mma candidates made reachable (#521); benchmark matrix and parity-gate corrections (#523, #529, #538, #539); HIP tensor-core and memory-accounting fixes (#540, #542).
- Frontend and backend defects surfaced along the way: unary einsum specs with convolution indices (#515), CUDA's clang-only `0.0h` half literal (#518), and Metal bf16 uniform builtins writing raw bit patterns (#520).

**Examples, research, and dispositions:**
- CNN classifiers (#54), GPT-2 inference (#377), and the matmul-to-tensor-cores track (#412) closed here.
- Research: TVM (#242), [Tiramisu/Telamon](docs/blog/tiramisu-telamon-optimization-space-pruning.md) (#267), [superoptimizers](docs/research/superoptimizers.md) (#261), and Lean Attention (#263).
- Candle (#265), Petalisp/Caten (#306), and MSVC (#313) were evaluated and closed as not planned.

**Deferred out of v0.9:**
- CUDA/HIP graph capture (#488) moved to v1.0.

---

## v1.0 — August 13, 2026 (released)
**Theme: Advanced compiler tiers and schedule-quality follow-through**

Closed with all 49 assigned issues resolved. GitHub milestone scope: *"Branch-and-bound on the analytic cost model. Better performance: better tensor cores, algebraic rewrites (non-numeric-preserving), better beam search."* The completeness, ergonomics and safety goals originally under v1.0 moved to v1.1 and v1.2: v1.0 marks the compilation side reaching the shape argued for in [the compilation manifesto](docs/compilation_manifesto.md).

**Advanced compiler tiers (done):**
- **Branch-and-bound schedule inference** (#514) — `Ir.Schedule_space` as a refinement tree over partial schedules, legality verdicts with witnesses deciding subtrees before any member is built, `Cost_model.completion_floor` as an admissible downward bound, the placement-space search with its enablement prior, and the staged tile lattice as corner-judged interval boxes. Delivered in phases 0–6, with a three-machine evaluation as its research output ([report](benchmarks/report-gh514-eval.md)).
- **Inlining as a first-class, searchable schedule decision** (#555), on top of retiring the concrete-index tracer in favor of the affine access relations (#554); the analysis is shared across sibling candidate compiles (#560).
- CUDA tensor-core profile completeness (#481), software-pipelined double-buffered staging (#487), CUDA/HIP graph capture of the fissioned step (#488), and budget-driven rematerialization on top of the liveness planner (#498).
- CPU reduced precision: 16-bit storage with f32 compute (#517) and native fp16 arithmetic (#516).

**Schedule quality — the `gpt2_mini` arc (done):**
- The v0.9 sweep left `gpt2_mini` 72x off torch CUDA and unmoved by tuning or materialization (#531). Attributing the step (#531, [report](benchmarks/report-gh531-profile.md)) put 70.2% of it in five kernels declined by one companion-coverage rule; judging that rule at the site's arity (#569) took the tuned step 107.4 → 52.4 ms on CUDA and 45.6 → 25.4 ms on HIP. Batched/rank-3 matmul sites are seeded (#528), so tensor cores are reachable on transformer workloads at all.
- CUDA bf16 mma timed scalar-fallback code under an mma label, because capability was keyed on the multiplicand pair rather than the accumulator format (#545); the tuner reports the untuned default's measured time as its honest reference point (#552); the search report names the winning candidate, and Metal's placement A/B was measured and found sound (#546, [report](benchmarks/report-gh546-metal.md)).
- Conv-sketch tuning wins did not port across CPUs; the cause was pool heterogeneity rather than the seeds, and `cc` now restricts its worker pool to one core class on hybrid machines (#530, [proposal](docs/proposals/gh-ocannl-530-pool-uniformity.md)). Device-memory accumulation during tuning and the placement-arm containment around it (#550), `bench_gpt` gate-cost legs (#551).
- Search-plumbing consolidation: intra-statement order in affine access paths (#561), a shared canonical-llc emission core (#563), pre-dispatch validation as its own contained phase (#564), interval narrowing from `If` conditions in `simplify_llc` (#566), barrier elision for same-anchor shared stages (#567), and the numerics policy entering the schedule cache key (#568).

**Frontend, configuration and diagnostics (done):**
- Shape inference: "close down when known" narrowed to the leaf-tensor rule it always was, with `stretch` requesting use-site resolution by name (#544).
- Config profiles `reproducible` / `performance` with picker-inherited precedence (#559); config startup chatter moved off stdout (#581).
- Routine name-clash policy established — status quo adequate, the correctness hazard being structurally solved and the residue debugging-quality only (#513).
- Site-targeted materialization: the capability exists twice over and ships nowhere measured (#558).

**Infrastructure:**
- CI moved to OCaml 5.5, Windows off the per-PR path onto a twice-weekly schedule, and the GPU backends onto a daily cross-machine sweep (`tools/sweep.sh`) — CI's runners have no GPU, so Metal, CUDA and HIP had never been covered there at all.

**Not taken up in v1.0, and where it went:**
- Fused attention via online softmax (#483) and the remaining convolution tiers — zero-nest workgroup geometry (#503) and Winograd (#505) — moved to v1.1, and from there to v1.2 in the late-August split. The `gpt2_mini` attribution retargeted #483 at 5.4% of the step at seq 128 (a plurality at native context), which is the trigger it now waits on.
- Roadmap-only ergonomics — concise merge-buffer transfer composition, execution-dependency tracking — remain unscheduled proposals.

Work that landed in this milestone before the v0.9 cutoff and shipped inside v0.9: safety, determinism and `%cd` simplification (#288, #247, #341, #348, #209), the tracing design (#160), `%op` inline-initializer scoping (#511), the mixed-precision cost diagnosis (#535), and the f16/bf16 defects — the reduction-identity cutoff (#547), the causal mask's sentinel (#548), and the GPU bfloat16 math builtins (#549).

---

## v1.0.1 — August 26, 2026 (released)
**Theme: Consolidation after v1.0 — search follow-through, inlining and reduction soundness, honest test and benchmark seams, and training-loop mechanics**

Planned and worked as "v1.1" until the August 26 renumbering. GitHub milestone scope: *"Consolidation after v1.0: the search follow-ups its evaluation filed, soundness of inlining and of reduction accumulators under every schedule, test and benchmark seams that cannot report a false pass, and the training-loop mechanics."* v1.0 brought the compiler's search to the shape [the manifesto](docs/compilation_manifesto.md) argued for; v1.0.1 is where its claims became trustworthy. The milestone was first scoped as "performance carry-overs and algebraic rewrites", but the carry-overs that change numerics or close on a benchmark cell moved to the performance milestone (now v1.1) in the late-August split, and what the milestone actually filled with — 135 issues, most filed by PR review cycles — is the work of making a green result mean what it says: a pass that cannot be promoted into a golden, an environment variable that cannot be mistyped silently, an inlined computation that cannot lose its guard, a reduction whose width cannot depend on which schedule won.

**Search follow-through from the v1.0 evaluation (done)** — the deliberate consequence of v1.0 recording its nulls rather than shipping them as wins:
- Statically-decidable builder preconditions lifted into tree verdicts (#577), with the epilogue-fusion level factored into the family tree (#613); the envelope's memory leg made fittable (#578); a profitability term weighing enablement promotion (#579); the sketch-family trees extracted from `autotune.ml` (#580); family-tree decision labels as a typed protocol (#591).
- `Tile_mma` register tiling for narrow 16-bit operands (#575), and the cc SIMD residue it exposed: gcc `-O3` spilling the register tile (#614), the `-0.0`-normalizing A-splat (#615), FMA builtins above AVX2 width (#621), `cc_vector_bytes` capped at 32 (#648).
- The `gpt2_mini` residue: the `lm_head` segment fissioned apart from its `max_logits` reduction (#574), the Virtual residual stream's quadratic re-summation (#573), rank-4 q/k/v projection sites getting geometry on two axes (#643), the attention out projection matched as a matmul site (#683) with precision-neutral accumulator localization behind it (#693, −74% on the Metal `gpt2_mini` forward step), the HIP leg measured on the Radeon 8060S (#612).
- The autotuner's loop enumeration sees through accumulation mints (#666, #687); `Autotune.report` as a typed five-state outcome (#677); a chosen placement arm can be shipped and output-verified (#638).

**Soundness of inlining and of reductions (done):**
- Inlining: a guarded setter is rejected for virtualization rather than replayed without its `If` (#651), and a looped one rather than replayed without its repetition loop (#674), with which candidate shapes reject at store time pinned by a characterization test (#658); cross-routine inlined virtuals declare their leaf reads (#610), and a routine that optimizes to nothing is a legal empty result (#611); recompute-at-read semantics of deferred computations documented and guarded (#617, #618); a `Local_scope` over a materialized node no longer collapses silently (#681) nor races sibling reads (#584).
- Reductions: accumulator width no longer depends on which schedule was chosen (#639), the last schedule-dependent width at narrow storage resolved (#663), f32 accumulators localized without a widening request (#693); the pre-driver launch gate checks per-dimension block caps (#679).

**Seams that cannot report a false pass (done):**
- Verdicts: a test that decides its own verdict reports it through `Verdict` and exits nonzero (#601), with a ratchet against bare `printf "<claim>: %b"` (#668); generated-kernel assertions establish the artifact's provenance (#655); executed legs for the hand-built-IR virtualization tests (#589) over a shared `ll_test` library (#600), `Context.get_values` honest on Local placements (#599), access sets exposed for assertion (#590), and a compile-and-link seam for hand-built `Low_level.optimized` (#562).
- Configuration: one uppercase environment spelling, tracked by every dune rule (#628, #652, #605), a mistyped variable warns (#629), backend-pinned single-test probes are sound (#622), a backend-sensitive stanza cannot go undeclared (#659), `ocannl_config` deps are complete and git-tracked (#586, #597, #602), the consistency scan globs rather than lists (#592), startup stdout pinned clean (#593, #595); digest completeness over every codegen-affecting knob (#572); operand-evaluation conditionality encoded once (#582).
- Benchmarks and CI: result JSON records which pass produced `step_ms` (#644), fixtures are digested (#645), a diverged cell reports its divergence rather than a runner failure (#676); Windows CI runs under the Git Bash it was verified on (#661, #662); a non-compiling master cannot survive a merge (#694); one argv convention across `bin/` (#634); the Metal ops suite baseline green (#632); an `Ir.Ops` startup SIGBUS (#688).

**Training-loop mechanics (done):**
- LR schedules, global-norm clipping and gradient accumulation (#465); mmap-backed checkpoint loading with aligned payloads and zero-copy hosted arrays (#467, #587), its Windows arm verified (#588); `trainable_params` derived as distinct from "needs initialization" (#673), and params-driven helpers refusing a paramless loss rather than compiling empty routines (#670). The user-facing training *experience* — resumable checkpoints, tracking, plots — is v1.1.2.

**Closed in the final stretch:** whether the tinygrad BEAM and torch.compile cells need the two-pass protocol — measured, they do not (#675); post-finalization placement seams in `ll_test` (#631); the fault-injection inventory for resource-owning seams (#571); and the accumulator-width and reduction-forms arc (#639, #663, #664, #682, #693, #721, #722, #735).

**Moved out in the late-August split:** fused attention via online softmax (#483), zero-nest workgroup geometry (#503) and the Winograd conv rewrite (#505) are numerics-changing or benchmark-demonstrated work, which defines the performance milestone (now v1.1).

Quantization (#137, #271), the WebGPU/WASM target (#123), the LLVM backend (#200), the fork-based backend (#161), `strict` axis naming (#190), local let-bindings in `%cd` (#80), the DumPy/torchdim deep dive (#316), the Imbue training-in-the-large study (#270), and the Fleuret lecture examples (#216) were completed or dispositioned in this milestone.

---

## v1.0.2 — September 16, 2026 (released)
**Theme: Robustness and compiler elegance through shared structure**

Created in the August 26 renumbering to pull the robustness and engineering-hygiene backlog out
of the feature milestones, so refactoring issues were worked while they still described the code
they were filed against. The milestone closed with **175 issues**, all of them closed after the
1.0.1 tag; it had 55 open before the September 14 rebalance, which bounded the remainder to eight
compiler-structure and coverage issues and moved the rest past v1.1. See
[CHANGES.md](CHANGES.md) for the release's entries.

**Robustness, landed through the milestone:**

- Reduction-width unification (#754), hardware-write cell separation (#950/#959), Metal's
  device-memory barrier fence (the fence half of #963), matching CPU/GPU half initialization
  (#951), and avoiding tiny repeated CPU pool dispatch (#933).
- Explicit C renderer state (#769), replayable tensor code (#955), wide-index routine logging
  (#953), and partial autotune reports that preserve admitted work (#962/#972).
- The earlier cache, merge-buffer dependency, optimizer, API and precision fixes are recorded
  in [CHANGES.md](CHANGES.md); they are no longer future-work checklists.
- Test-run launch consolidation and supported queries (#606/#671), verdict provenance
  (#908/#931/#968/#973), shared IR builders (#954), and restored Windows CI setup (#935).
  These improve release verification; they do not close the remaining analysis gaps.

**The eight compiler-structure and coverage issues shipped, and v1.1 starts.**

Compiler-side deduplication was kept in the milestone for its own sake: one expression of a
shared idea, with differences made explicit. All eight closed on September 14 (#604, #917, #630, #656, #875, #774, #770, #794): one
configuration resolver, one `Ops`-owned precision enumeration, one ordered `Low_level` access
traversal, one C builtins table compiled by host stubs and cc kernels alike, one warp-shuffle
stage description its own simulator consumes, one normalizing `Indexing.affine` construction with
a coverage proof behind initialization elision, and shared CUDA/HIP scalar semantics behind one
compilation driver for all four backends, with #794 supplying the vendor-arm compile coverage.

**Closed after that set, on the way to the tag:** emulated half narrowing rounding the interval
just above half of the smallest subnormal instead of flushing it (#981); the sweep's
machine-readable per-run record (#977); the dxg bridge's own kernel evidence as an
environment-red trigger (#979); and the `/dev/dxg` `-j` cap as a single source that manual GPU
suites get automatically (#983). The conflict-free einsum grammar landed alongside them
(`lukstafi/ocannl-staging` PR #712).

Testing-side refactorings and the wider consolidation backlog follow v1.1, in v1.1.1.

| Retain | Structural improvement |
|--------|------------------------|
| #770 + #794 | Express the common CUDA/HIP syntax and backend compilation structure once, with backend differences explicit; #794 supplies actual vendor-arm compile coverage for that refactoring. |
| #630 | Express evaluated-operand, guard and dead-code traversal semantics in a shared IR fold rather than reconstructing them in each walker. |
| #656 + #917 | Give shared C builtins and the renderer/test precision enumeration one source of truth. |
| #774 | Centralize affine normalization and enforce its invariant; put the issue's surjectivity reasoning on a principled footing. |
| #604 | Express configuration precedence through one resolver, with bootstrap differences represented as arguments. |
| #875 | Give the shuffle-stage sequence one renderer-owned description, also consumed by its simulator. Like #917, this spans implementation and tests rather than refactoring test infrastructure alone. |

The Assignments leaf-type split (#818) stays deferred until a new leaf constructor needs it;
its issue explicitly names that trigger. Scanner resolution, harness deduplication and report
plumbing remain after v1.1 unless an experiment exposes an immediate need.

**September 14 disposition of all 55 remaining issues:**

| Destination | Issues | Why / when to work them |
|-------------|--------|------------------------|
| v1.0.2, compiler elegance and coverage (8) | #770, #794, #630, #656, #917, #774, #604, #875 | The bounded deduplication and compile-coverage set above. |
| v1.1, alongside affected experiments (5) | #963, #975, #833, #834, #922 | Scheduling legality, candidate ownership, timing-objective evidence and calibration accounting support the work v1.1 will exercise. They are not a five-issue entrance exam: take each with the feature or measurement that needs it. |
| v1.1, lower-priority performance evidence (2) | #594, #819 | Per-device cache identity and cache-hit benefit measurement fit the performance theme. Neither blocks the single-GPU fleet's initial experiments. |
| v1.1.2 (2) | #793, #777 | Explicit persistent optimizer state and computed-value observability support the training/debugging experience. The current SGD materialization fix is already landed. |
| v1.1.1, trigger-gated (2) | #695, #966 | PPX migration depends on the upstream AST release; legacy-lock deletion follows its retirement trigger, with October 10 the sequencing plan's proposed date. |
| v1.1.1, post-performance consolidation (24) | #603, #607, #609, #625, #641, #642, #660, #672, #678, #705, #707, #778, #797, #798, #799, #818, #907, #910, #911, #913, #914, #915, #916, #920 | Scanner and harness refactoring, diagnostics and remaining IR/API work. Renderer/backend deduplication itself is retained in v1.0.2. |
| v1.1.1 (12 more) | #928, #929, #940, #596, #926, #919, #921, #932, #942, #946, #918, #934 | Symbolic-extent/API guard work, evidence meta-checks, historical report provenance and tooling/docs improvements. Pull a bounded fix forward only if a chosen v1.1 workload actually depends on it. |

The concrete boundary matters. #928 → #929 concerns symbolic-extent semantics and gradients;
start the performance comparisons at fixed extents unless an experiment needs dynamic ones.
#940 concerns forgotten launch assignments; explicitly initialize bindings in those experiments
while the stronger production API waits. #919 concerns historical Metal fixtures: use newly
recorded fixture provenance for v1.1 measurements, without presenting old numbers as matched
controls. #596 and #926 improve verification machinery, but no current defect requires making
their generalization a prerequisite to a measured compiler improvement.

Within v1.1, #963's Metal device-memory fence is already fixed; the remaining barrier-region
analysis matters when new schedules cross its boundary. Establish legality for the schedules a
feature proposes, using a conservative refusal where necessary, instead of front-loading a
general analysis. #975 is a suspected callback-cleanup gap, not a measured leak: investigate it
when exercising tuning callbacks. #794's real select-arm compile checks accompany #770 in
v1.0.2; hosted Metal CI evaluation (#942) need not precede them. #833/#834 should share
measurement sessions with the performance work, and #922 belongs with calibration consumers.

The redistribution leaves **eight open issues in v1.0.2, 37 in v1.1, 38 in the new v1.1.1,
17 in the renamed v1.1.2 and seven in v1.2**. The new consolidation milestone is a queue to
prioritize after performance work, not a promise to clear all 38 issues.
This rebalance does not cut a release or close the v1.0.2 milestone; tagging remains separate.

**At the September 16 tag** those counts read **none open in v1.0.2, 37 in v1.1, 46 in v1.1.1,
17 in v1.1.2 and seven in v1.2**: the eight retained issues closed, and v1.1.1 grew by the
follow-ups the v1.0.2 review cycles filed against it.

---

## v1.1 — October 4, 2026 (released)
**Theme: Fused attention and better transformer scheduling**

The release delivers online-softmax attention, fused backward, block-tiled matrix-unit rendering,
improved GPU scheduling and fission, and more reliable tuning and benchmark evidence.
Convolutional workloads and CPU kernel and sketch economy continue in v1.1.2.

The milestone's original goal — performance-chasing in the `approximate` profile on transformer
workloads — remains unfinished. The completed measurements support a release of the compiler
features with bounded speed and memory wins, rather than a universal approximate-profile speedup.
The profile remains an opt-in numerics regime with workload-dependent tradeoffs. Passing a parity
envelope establishes acceptance of a configuration; it does not establish that it is faster.
The benchmark's `exact` label is a comparison regime, not a bitwise floating-point guarantee:
#998's separation of rounding-changing simplifications remains open, as does #720's broader
benchmark coverage. Release scope does not close or reassign those issues.

**Delivered since the September 15 roadmap edit:**

- **Online-softmax forward, fused backward and block-tiled attention** (#483, #1002, #1003;
  staging PRs [#737](https://github.com/lukstafi/ocannl-staging/pull/737),
  [#885](https://github.com/lukstafi/ocannl-staging/pull/885),
  [#892](https://github.com/lukstafi/ocannl-staging/pull/892),
  [#905](https://github.com/lukstafi/ocannl-staging/pull/905)). The forward fold renders both
  contractions on supported GPU matrix units; fused backward removes quadratic gradient buffers.
  Its portable implementation can trade latency for memory. High-level cumulative/top-k operations
  (#952) remain consumer work; general scan licences and schedule-chosen block geometry
  (#1122/#1123) remain in v1.2.
- **GPU launch geometry and fission** (#995, #728, #1133, #1126, #1124): map longer parallel chains,
  preserve each statement's mapping when deciding whether to merge kernels, and reach channel
  lanes through the fused-backward reduction. The cooperative reduction policy is enabled by
  device capability on Metal/CUDA and conservatively refused on HIP. These improvements also
  benefit composed attention in the exact regime.
- **Placement and register-tile follow-through**: footprint-scoped scratch materialization
  (#616, [PR #750](https://github.com/lukstafi/ocannl-staging/pull/750)), persistent placement
  decisions (#786), vectorized register-tile remainders (#620), model-ranked register geometry
  and emitted-geometry census (#947/#948). The #1099 measurement rejected an unhelpful C-traffic
  ranking term; [PR #933](https://github.com/lukstafi/ocannl-staging/pull/933) changes research
  documentation, and the tail tie rule remains #1180.
- **Backend precision and execution**: Metal f16 inputs with f32 storage (#923), CUDA
  persistent-fragment wide-f16 destinations (#925), HIP wide bf16 accumulation (#838), native
  HIP matrix-unit support and pool addressing corrections (#1032, #344), plus scoped inline-PTX
  temporaries (#1073). These are supported rendering paths, not a claim that every benchmark
  schedule uses matrix units.
- **Tuning and benchmark evidence**: concrete per-device cache identity (#594), cache-hit benefit
  measurement (#819), completed timing-objective/session-cost studies (#833/#834), bounded
  calibration and failed-candidate cleanup, a census of the shipped training step (#1170), timed
  allocation peaks (#1006), regime/parity provenance and explicit skipped cells. The development
  `simplify_fp_algebra` selector landed in [PR #935](https://github.com/lukstafi/ocannl-staging/pull/935)
  for #998's ablation; its default and the production profile policies retain current behavior.

**What the measurements establish:**

The completed [cross-framework comparison with a same-night August anchor](benchmarks/report-tagline-gpt2.md)
(#1181) measured tuned f32 `gpt2_mini` inference at one geometry: batch 8, sequence 128,
four layers, d_model 256. The October 3/4 campaign rebuilt and re-measured August commit
`7014dc44` alongside master `a723dc5b` on each box; this is **not a v1.0.2-to-v1.1 comparison**.

| Backend and device | Exact latency | Approximate latency | Approximate vs exact | Exact speedup over same-night August anchor |
|---|---|---|---|---|
| CUDA, RTX 5070 Ti Laptop (discrete) | 3.504 ms | 4.493 ms | 28% slower | 2.02× |
| Metal, M4 Max (unified) | 8.334 ms | 7.457 ms | 11% lower latency | 1.31× |
| HIP, Radeon 8060S/gfx1151 (unified) | 4.757 ms | 4.692 ms | Effectively neutral | 1.66× |
| HIP, RX 7700S/gfx1102 (discrete) | Not quotable | Not quotable | Not established | Not established |

With OCANNL exact and PyTorch exact-pinned, CUDA remains 3.41× slower than `torch.compile`
and 2.18× slower than tinygrad BEAM=2 (which has no exact pin); Metal remains 4.40× and 4.04× slower respectively. Unified-memory HIP is
1.54× slower than tinygrad BEAM=2, with no PyTorch ratio because that box's ROCm wheel
segfaulted. Discrete-HIP master timing processes kept searching uncached arms, so none passed
the clean-replay provenance gate. The report retains those diagnostics without quoting a
master result. Training was smoke-only, with no controlled training comparison. These are
whole-profile measurements, not an isolated attribution to fused attention or TF32.

- **Approximate CUDA acceptance is complete** (#719). The September 27 native RTX 5070 Ti
  tuned row passed both the approximate and exact envelopes, but took **6.900 ms versus
  6.271 ms exact**. These whole-profile searches differ in program and search settings;
  the result does not isolate TF32. The [issue's acceptance report](https://github.com/ahrefs/ocannl/issues/719)
  records the search cost and deliberately skipped CPU tuning cells.
- **The attention features have useful bounded wins.** The
  [block-fold ablation](benchmarks/report-gh1003-block-fold.md), on a September 28 M4 Max build,
  measured seq1024 inference at **0.483× composed latency**, with requested memory **91–96 MiB
  across the sequence endpoints versus 124–353 MiB composed**. The
  [fused-backward report](benchmarks/report-gh1002-fused-backward.md) records seq1024 training
  memory falling from **689 to 305 MiB** with scores stored, alongside a Metal latency regression
  on that build. Later scheduling fixes and whole-profile sweeps have their own baselines.
- **The latest transformer sweep is mixed** ([#720 report](benchmarks/report-gh720-transformer.md),
  October 3). Default f32 approximate is slower on all eight HIP endpoints; base inference is
  **20.024 versus 17.695 ms exact**. Metal inference improves at the three sequence endpoints,
  including **22.572 versus 28.784 ms at seq1024**, while base training is **70.631 versus
  70.162 ms**. These are untuned whole-profile comparisons on separately stamped HIP/Metal
  revisions, not isolated rewrite effects or a controlled cross-backend ranking.
- **HIP block-fold policy now uses a measured fallback** (#1171,
  [PR #939](https://github.com/lukstafi/ocannl-staging/pull/939)). `online_softmax_block=auto`
  resolves to two-pass on HIP and unmeasured targets, keeping block 16 on CUDA, Metal and CPU.
  Cached gfx1151 confirmation measured **4.812 ms resolved approximate, 6.937 ms forced block 16,
  4.552 ms exact**. That evidence is from unified memory; discrete gfx1102 confirmation remains
  #1184. The fallback avoids the measured regression without promising a profile-wide win.

**Release boundary and remaining work:**

The October 3 sweep found exact HIP f16 failures at base training and seq1024 inference/training
(#1182), and falling per-token training throughput at batch256 on HIP and Metal (#1183).
Ten requested CPU f32 cells timed out and 28 reduced-format cells were deliberately skipped;
#720 remains partially reported. The completed #1181 report adds the controlled inference
comparison above, while CUDA approximate performance (#1194), discrete-HIP fallback confirmation
(#1184), HIP f16 failures (#1182), and large-batch scaling (#1183) remain follow-ups. A complete
cross-framework matrix or a universal approximate speedup is therefore not a release claim.

For #998, measurements support separating optional float algebra from integer-power lowering:
the latter is currently required for finite CUDA GPT training under NVRTC fast math. The compiler
split is still outstanding. Q/K/V zero-init folding (#1175, PR #934) was **reverted by
[PR #941](https://github.com/lukstafi/ocannl-staging/pull/941)** after doubling tuned CUDA forward
latency; it is excluded from the delivered feature list. The release includes the completed measurements
and carries the remaining limitations explicitly; the original approximate-performance goal
and the pending compiler split are not presented as complete.

---

## v1.1.1 — October 16, 2026
**Theme: Consolidation after the transformer work**

GitHub scope remains testing-side refactorings, scanner maintenance, diagnostics, stable goldens,
test tooling and remaining IR/API work. As of October 4, 88 assigned issues are closed and 109
remain open: much consolidation has already landed alongside the performance experiments, and
review continues to add follow-ups. The September redistribution tables above are historical
snapshots, not the current inventory. Use the measured v1.1 defects to prioritize a bounded
release; emptying the queue is not a prerequisite to subsequent work. The PPX migration (#695)
remains upstream-release-gated. Legacy-lock retirement (#966) remains tied to its documented
trigger rather than to the roadmap date alone.

---

## v1.1.2 — October 24, 2026
**Theme: Performance beyond transformers**

Created in the September 29 split, with 11 open issues as of October 4. Each compiler improvement
should close with a before/after benchmark cell; a measured null result is useful evidence too.

- **Reduced ResNet-style benchmark** (#1161): padded 3×3 convolution, residual adds, strided
  1×1 shortcuts and batch-1 latency. The full ResNet consumer, with batch norm and `resnet_block`,
  belongs to v1.1.3.
- **Convolution compiler work**: Winograd F(2×2, 3×3) (#505), zero-nest workgroup geometry (#503),
  flavor-indexed convolution family search (#697), second-reader staging (#1147), and batch-1
  strided 1×1 seeds (#1148).
- **CPU kernel and sketch economy**: non-dividing packed GEBP (#627), pad profitability (#740),
  blocktile/packed-scalar padding (#741), per-site flip-chain profitability (#717), and hermetic
  `bin/` benchmark drivers (#743).

---

## v1.1.3 — October 31, 2026
**Theme: Consumers and explorations**

GitHub scope: models, reproductions, demos, integrations and the training experience
(checkpointing, tracking, plots). Renamed from v1.1.2 in the September 29 split; 24 open issues
as of October 4. It can continue in the undated v1.2.2 descendant if scope is deferred.

**Training experience:**
- Explicit persistent optimizer state (#793) and computed-value observability (#777).
- Batch-norm running-stat momentum (#879), MobileNet width multiplier (#880), and compilable
  `resnet_block` channel constraints (#1146).
- Resumable checkpoints (#96), experiment tracking (#122), plot legends and axis ticks (#103).

**Models, reproductions and demos:**
- Model surgery (#33), LSTM (#60), Bonsai RNN (#182), digit addition (#427), BERT/ModernBERT (#297),
  DisTrO (#278), and Gemma 3 with real weights and long context (#570).

**Explorations, integrations and deployment:**
- Simply/NanoDO study (#435), inference plugins/binaries (#97), Polars (#219), krnl/autograph
  (#277), public `lib/` interfaces (#1010), cumulative/top-k operations (#952), mmap-backed
  checkpoint zero-copy (#585), async-copy staging (#576), and pressure-aware memory policy (#565).

---

## v1.2 — November 15, 2026
**Proposed theme: Reusable tensor programs on a standalone ArrayJIT compiler**

GitHub currently calls this *"Ambitious feature-grade work"*, with **11 open issues as of
October 4**: #404, #903, #852, #444, #477, #170, #195, #963, #1000, #1122 and #1123.
The three architectural anchors below remain a proposed core; the later additions concern
compiler phase order, cross-statement conflict analysis and general scan/scheduling licences.
#585 is now consumer work in v1.1.3. November 15 is the current soft target, not a promise
to finish all 11 design spaces.

**Recommended core, with concrete completion criteria:**

| Track | Existing issue | Proposed deliverable |
|-------|----------------|----------------------|
| Standalone compiler | #852 | ArrayJIT accepts a loop-nest program without depending on tensor/Assignments orchestration, with a reference driver, documented package boundary and an executed non-neural example. The public `arrayjit.ll_builders` from PR #686 is useful groundwork, not completion of the package split. |
| Recoverable inference sessions | #903 | Solver state belongs to a session; a failed tensor construction can rewind without contaminating a later valid one. Exercise shared operand state and two independent sessions. The maintainer still needs to choose the transaction representation. |
| Reusable tensor functions | #404 | Infer a function's constraint scheme once, freshen it at multiple applications and report contradictory shapes at the appropriate boundary. Demonstrate reuse at different shapes, error provenance and measured inference cost against retracing. |

Sequence the session ownership/rollback contract before committing to shape-scheme storage and
freshening. The package split can progress alongside that design, but agree the frontend/compiler
boundary first to avoid moving APIs twice. C renderer state isolation (PR #707) removes one
source of shared mutable state; it does **not** establish compiler-wide or solver reentrance.

Flesh out these three issues with acceptance work packages rather than adding unrelated features:
a package-dependency/install check and standalone example for #852; failure-recovery and session
isolation examples for #903; polymorphic reuse, diagnostics and an inference benchmark for #404.
A release demonstration should pair the standalone ArrayJIT example with one reusable tensor
block compiled at several shapes, including a failed application followed by a valid one.
These are proposed scope refinements, not newly filed issues.

**Optional companions, not core release gates:**

- PoPE (#444) can demonstrate a reusable position-embedding block; it should not determine the
  session architecture or hold the release if interest lies elsewhere.
- CUDA pinned host buffers (#170) and `__constant__` arrays (#195) need explicit ownership,
  capability and transfer/read benchmarks before promotion into the core scope. They may ship
  independently when measured.
- CDNA MFMA (#477) requires a CDNA machine; the current fleet's gfx1151 HIP results do not cover
  wave64 MFMA. Keep it hardware-gated and propose a later backend milestone only when hardware
  access exists, rather than inventing a v1.3 date now.

If the three architectural tracks cannot fit by November 15, finish a coherent subset and defer
the remainder explicitly, preserving the current landing target. A useful v1.2 advances reusable programs and clear ownership; clearing
11 heterogeneous issue numbers is not its acceptance criterion. The undated v1.2.1 and v1.2.2
milestones reserve room for later consolidation and consumers; they are not additional scheduled
feature releases.

---

## Key Milestones Summary

| Version | Target | Status | Key Deliverables |
|---------|--------|--------|------------------|
| 0.6.2  | Nov 2025 | released | Menhir parser, hidden-dimension errors |
| 0.6.3  | Dec 2025 | released | Padding inference, toy CNN |
| ~~0.6.4~~ | — | **skipped** (folds into 0.7) | Concatenation, RoPE, transformer toy |
| **0.7** | Jul 3, 2026 | **released** | **Frontend finalization + compiler optimizations** (consolidates 0.7.2) |
| ~~0.7.1~~ | — | **dissolved** | AMD HIP backend → 0.8; completed examples and tokenizers landed subsequently |
| **0.8** | Jul 13, 2026 | **released** | **Parallel schedules (GPU + CPU), autotuning, SIMD/`Tile_mma`, AMD HIP backend, benchmark suite** |
| **0.9** | Aug 3, 2026 | **released** | **Schedule quality, deterministic parallelism, mixed precision, convolution performance, and search survivability** |
| **1.0** | Aug 13, 2026 | **released** | **Branch-and-bound schedule inference, inlining as a searchable decision, graph capture, software pipelining, rematerialization, CPU reduced precision, and the 2x `gpt2_mini` step** |
| **1.0.1** | Aug 26, 2026 | **released** | **Consolidation after v1.0** (planned as "v1.1"): search follow-through, inlining and reduction soundness, test and benchmark seams that cannot report a false pass, and the training-loop mechanics |
| **1.0.2** | Sep 16, 2026 | **released** | **Robustness pulled forward, plus compiler elegance through shared structure**: the landed robustness fixes and eight compiler-structure/coverage issues |
| **1.1** | Oct 4, 2026 | **released** | **Fused attention and better transformer scheduling**; tuned f32 inference improves over the same-night August anchor, while approximate performance remains workload-dependent |
| 1.1.1  | Oct 16, 2026 | planned; work already landing | Consolidation informed by transformer measurements and review findings |
| 1.1.2  | Oct 24, 2026 | planned | Performance beyond transformers: reduced ResNet-style convs and CPU kernel/sketch economy |
| 1.1.3  | Oct 31, 2026 | planned | Consumers, models, integrations, checkpointing and observability |
| 1.2    | Nov 15, 2026 | planned; core refinement proposed | Standalone ArrayJIT, session transactions and shape schemes; broader compiler licences and hardware features remain design spaces |
| 1.2.1 / 1.2.2 | undated | placeholders | Later consolidation / consumers if scope moves beyond the current ladder |

---

## Paper Artifacts (no venue currently targeted)

**No conference submission is scheduled.** The OCaml Workshop / FProPer submission was not accepted, and IFL 2026 was considered and decided against as a poor fit. The written material below is therefore maintained for its own sake — as the project's technical account, and as the starting point should a venue be chosen later. Only the formal core technical report and the constraint-generation notes are live work; the rest is archival.

The paper-facing material, first assembled for the v0.7/v0.8 workshop submission:

- Workshop article: [docs/ocannl_workshop_article_human.md](docs/ocannl_workshop_article_human.md) — **historical artifact**, kept as-is; it describes the project as of the [0.8 release](https://github.com/ahrefs/ocannl/releases/tag/0.8) and was written for the OCaml Workshop / FProPer submission, which was not accepted.
- Workshop article PDF: [docs/html/pdfs/ocannl_workshop_article_human.pdf](docs/html/pdfs/ocannl_workshop_article_human.pdf) — likewise archival, rendered from the v0.8-era source.
- Formal core technical report: [rendered PDF](https://ahrefs.github.io/ocannl/docs/pdfs/ocannl-formal-core-technical-report.pdf), LaTeX source in [docs/](docs/ocannl-formal-core-technical-report.latex) — live, still developing.
- Shape constraint generation notes: [docs/shape-constraint-generation.md](docs/shape-constraint-generation.md) — live.

The intended shape of a paper, should one be written, is recorded below.

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

### Why v0.7 Was the Prerequisite
Any such paper needs working examples on OCANNL's mature frontend, all delivered by v0.7:
- Clean context-based API (no hosted tensors)
- Shape concatenation syntax (`^`)
- Complete transformer example with RoPE
- Consistent, documented API surface

The deep semantic groundwork (the two-sorted ground algebra, the rank-fact graph and rank-cycle check, ≈-semantics for row equality) now lives in the formal core technical report and its appendix.
