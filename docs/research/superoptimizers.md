# Superoptimizers for tensor programs — research notes

Companion evidence base for [gh-ocannl-261](../../archive/docs/proposals/gh-ocannl-261.md), which is
the decision record (reading-list verdict, technique-to-seam map, follow-ups F1–F5,
skip decisions). This document holds the detailed paper extractions and the verified
literature survey those decisions rest on, plus cross-cutting observations that
didn't fit the proposal's format.

**Provenance:** written 2026-07-07. Mirage and Tensat were read from their arXiv
TeX sources (`arxiv.org/src/2405.05751` — the OSDI'25 camera-ready with earlier
draft text visible in `\if 0` blocks — and `arxiv.org/src/2101.01332`). Survey
claims in §3 were verified against live arXiv/venue/GitHub/opam pages, not from
model memory.

---

## 1. Mirage: A Multi-Level Superoptimizer for Tensor Programs (OSDI'25)

Wu, Cheng, Liu, Shi, Ji, Ao, Velliengiri, Miao, Padon, Jia (CMU/PKU/PSU/Purdue/
Weizmann). arXiv 2405.05751. §-references: §2 μGraphs, §3 RMSNorm case study,
§4 generator, §5 verifier, §6 μGraph optimizer, §7 implementation, §8 evaluation.

### 1.1 The μGraph IR (§2)

Three graph levels mirroring the CUDA hierarchy:

- **Kernel graph** — one per tensor program; nodes are kernels, edges are
  device-memory tensors. A node is either a *pre-defined* operator (cuDNN conv,
  cuBLAS matmul) or a *graph-defined* operator whose semantics are a nested block
  graph. Fusion is therefore not a rewrite rule but "replace N kernel nodes with
  one graph-defined node".
- **Block graph** — one thread block; all intermediates live in shared memory
  (by design, "to reduce device memory access by maximally saving intermediate
  results in shared memory"). Structure:
  - grid dimensions (up to x,y,z);
  - **imap**: per input, grid dim ↦ data dimension (equal partitioning) or a
    replica dimension φ (tensor replicated per block);
  - **omap**: per output, grid dim ↦ data dimension only — replication is *not*
    allowed on outputs ("different blocks must store disjoint tensors in device
    memory"); block outputs concatenate;
  - a **for-loop body** for tiling: iteration count, *input iterators* (load a
    tile per iteration, device→shared), **for-loop accumulators** (`Accum`, e.g.
    summation and max, accumulating into shared memory), and **fmap**: for-loop
    dim ↦ data dim (partition per iteration) or φ (for `Accum`: reduce across
    iterations); post-processing ops sit after the accumulators (e.g. a mean's
    divide-by-n), an *output saver* writes shared→device.
- **Thread graph** — registers; block dims + for-loop dims, iterators
  shared→registers; contains only pre-defined thread operators.
- **Operator inventory** (table in generate.tex): Matmul, Sum, EwAdd, EwMul,
  EwDiv, EwExp at kernel/block/thread levels; Repeat, Reshape, Sqr, Sqrt, SiLU at
  kernel/block; InIter/OutSaver/Accum at block only. LoRA required *manually
  adding* a 4-input operator f(W,X,Y,Z) = (W‖X)×(Y‖Z) with its own
  abstract-expression rule — the vocabulary bounds the search.
- **Layouts** are carried on every tensor at every level but "affect only the
  performance … no impact on correctness" — hence deferred to post-verification.
- **Validity (Def. 2.1):** shape specs hold; per-level memory capacities hold
  (device/shared/registers); in any graph with a for-loop body, every
  input→output path passes through exactly one input iterator, one accumulator,
  one output saver.

### 1.2 The expression-guided generator (§4)

Input programs are first split into **Lax subprograms** (multi-linear operators +
division + limited exponentiation — see §1.3); each is superoptimized
independently against its own reference graph.

**Hybrid two-regime search:** exhaustive enumeration at kernel + block levels;
rule-based construction at thread level (greedy fusion of ops with predecessors),
justified by device/shared access being orders of magnitude more expensive than
registers.

**Enumeration (Alg. 1):** maintain a downward-closed prefix of a valid μGraph;
extend one operator at a time enumerating (op type, input set); for graph-defined
operators also enumerate grid dims and for-loop dims and recurse into block-graph
generation (no graph-defined ops inside block graphs — nesting depth 2 in search).
Each candidate passes (a) the abstract-expression subexpression check, (b) shape
inference, (c) a memory-capacity check.

**Canonical form:** outputs indexed by (producing-op position, output slot); op
rank = (list of input tensor indices, op type); a μGraph is canonical iff ops
appear in increasing rank; the generator only emits canonical graphs (each added
op's rank must exceed all existing). Claimed to prune no valid solutions —
it only deduplicates orderings.

**Abstract-expression pruning (§4.3)** — the scaling trick. Abstraction: ignore
differences between elements of the same input tensor; abstract expressions are
first-order terms over integer arithmetic + uninterpreted functions add, mul,
div, exp, sqrt, silu, and `sum(k, E)` — the reduction *size* k is kept
("crucial"; rows-vs-columns of a k×k sum is deliberately indistinguishable, but
nested sums track as `sum(i, sum(j, x)) = sum(i·j, x)`). Matmul(X,Y) ↦
`sum(k, mul(E(X), E(Y)))`; Accum with fmap=φ ↦ `sum(i, ·)`; iterators, savers,
Reshape, Repeat are identity; graph-defined ops are inlined.

Two axiom sets: **A_eq** (commutativity/associativity/distributivity of
add/mul/div, sum-of-sum collapsing, sum-mul/div distributivity,
exp(x)·exp(y) = exp(x+y), a sqrt·sqrt rule, `sum(1,x) = x`) and **A_sub**
(x ⊑ add(x,y), x ⊑ mul(x,y), both args ⊑ div, x ⊑ exp/sqrt/silu/sum,
reflexivity, transitivity). A prefix G is pruned iff
A_eq ∪ A_sub ⊭ subexpr(E(G), E_O), where E_O is the input program's expression.
The entailment check is a **Z3 query** (Z3 4.12.6), results cached across the
search. **Theorem 4.1:** if G is equivalent to the input and A_eq ⊨ E(G₀) = E(G),
then G is generated (every operator's inputs are subexpressions of its output, so
G's prefixes all survive). Deliberate incompleteness: cancellation axioms
(div(mul(x,y),y) = x) are excluded — "including such axioms would make everything
a subexpression of everything, therefore nulling desired pruning".

**Budgets/scale:** default ≤5 kernel-graph ops, ≤11 ops per block graph; up to
4 h per Lax subprogram, framed as one-time pre-deployment cost. RMSNorm ablation:
with multithreading + abstract expressions 11 s (5 block-ops) → 28 s (11);
without multithreading 58 → 183 s; **without abstract expressions 768 s at
5 ops, 19,934 s at 6, >10 h at ≥7** — and the winning RMSNorm μGraph needs 11,
i.e. is unreachable without the pruning.

### 1.3 The probabilistic equivalence verifier (§5, §7)

**Lax fragment:** (1) only multi-linear operators (linear in each input
separately), division, exponentiation; (2) at most one exponentiation on any
input→output path. Then every output entry has closed form (Eq. 1)
(Σᵢ fᵢ·exp(gᵢ/hᵢ)) / (Σᵢ f′ᵢ·exp(g′ᵢ/h′ᵢ)) with f, g, h polynomials in the
inputs; the difference of two Lax programs has the same form, so equivalence
reduces to zero-testing it — a generalization of Schwartz–Zippel polynomial
identity testing.

**Mechanism:** two finite fields with q | p−1 and q > 2w (w = coefficient
bound); Z_q used *inside* exponents, Z_p outside; exponentiation maps
x_q ↦ ω^{x_q} mod p for ω a random q-th root of unity in Z_p (q | p−1
guarantees roots exist). Every value is a pair (x_p, x_q); add/sub/mul/div
componentwise modular (div via multiplicative inverse), sqrt via field square
root. Inputs sampled uniformly; ω uniformly among the roots of unity.

**Guarantees:** Thm 5.2 — if the difference is not the zero function, a random
evaluation is zero with probability ≤ 8dk⁴/q + q^(−1/k²) (d = max polynomial
degree, k = number of exp terms). Thm 5.3 — equivalent μGraphs always pass (no
false negatives); non-equivalent ones survive Ω((k²/ln q)·ln(1/δ)) independent
tests with probability ≤ δ.

**Practice (§7 candor):** largest p·q fitting 16 bits — **p = 227, q = 113** —
so tests run on GPU, reusing Mirage's own shared-memory optimizations. They run
**a single random test** comparing all output elements: "this equivalence
verification procedure does not introduce false negatives. While it could, in
theory, introduce false positives, we have not observed any in practice"; a
fully-parameterized final verification for the winning μGraph is future work.
**ReLU is unsupported** by the probabilistic verifier (the old arXiv drafts'
ReLU→exp substitution hack is deleted; SiLU became a first-class op with its own
axioms; a solver-based verifier for non-Lax programs using user-provided
first-order operator properties is mentioned as beyond the paper's scope).
Finite-field equivalence says nothing about float behavior, so Mirage
additionally runs floating-point tests to filter μGraphs with significant
numerical error.

### 1.4 The post-verification μGraph optimizer (§6)

Runs only on verified graphs, exploiting that these choices don't affect
correctness (which also shrinks the generator's space — graphs differing only in
layout/order/allocation are identical to it):

- **Layouts:** ILP over booleans B_{t,l}, linear constraints from operator
  requirements (e.g. cuBLAS wants the innermost dim among the last two), cost
  terms for e.g. bulk-copy eligibility; solved optimally with Z3's ILP.
- **Operator scheduling:** topological order minimizing `__syncthreads()` —
  DP on node depth (longest path from inputs), schedule by ascending depth;
  barriers only between depth classes.
- **Memory planning:** offsets as dynamic storage allocation, solved by
  exhaustive enumeration.
- Ablation (GQA bs=1, A100): disabling any one of thread-graph construction /
  layout / scheduling / memory planning costs 5–70%.

### 1.5 Evaluation (§8) — where the wins came from

Setup: GQA (LLaMA-3-70B, 8K ctx, 4-GPU tensor parallelism), QKNorm
(Chameleon-7B), RMSNorm (LLaMA-2-7B), LoRA (GPT-3-7B-LoRA), GatedMLP
(Falcon-7B), nTrans (nGPT-1B); A100 + H100, fp16, 1000-run averages; baselines
TASO/PET, PyTorch (torch.compile + FlashAttention), TensorRT(-LLM),
FlashAttention/FlashDecoding, Triton, all under CUDA Graphs. Headline: up to
3.3× over the best existing system (earlier submission draft said 1.1–2.9×).

- **GQA:** rediscovers FlashAttention/FlashDecoding and beats them up to 2.2×
  via (1) grid-dimension search — TensorRT-LLM's fixed grids (8,2,1)/(8,2,8)
  underutilize 108-SM A100 / 132-SM H100; adopting TRT-LLM's grid dims costs the
  Mirage kernel 18%; (2) parallelization-axis search — FlashAttention
  parallelizes over sample/head/query-seq, FlashDecoding and TRT-LLM over
  sample/head/KV-seq, both suboptimal for few-KV-head GQA; Mirage picks
  per-scenario among sample/KV-heads/query-seq/KV-seq, cutting device-memory
  access up to 7×.
- **RMSNorm (case study, §3):** the discovered kernel runs the RMS accumulation
  A = Σⱼ xᵢⱼ² and the matmul accumulation B = Σⱼ xᵢⱼgⱼwⱼₖ **in parallel in the
  same for-loop body**, commuting RMSNorm's division past the matmul; Mul/Sqrt/
  Div fused into a register-resident thread graph. 1.5×/1.9× over hand-written
  kernels on A100/H100.
- **QKNorm:** fuses both layernorms into the attention kernel (baselines can't):
  up to 1.4×.
- **LoRA:** (W‖B) × (X‖(A×X)) merges 3 launch-bound tiny matmuls into one
  kernel, concats done by offset bookkeeping in shared memory (zero copy):
  1.1–2.4×. Needed the hand-added 4-input operator.
- **GatedMLP:** two matmuls in one block graph + SiLU/Mul as post-loop
  processing: 1.5× (A100), 2.7–3.3× (H100).
- **nTrans — the structural loss:** beats most baselines but *loses to
  TensorRT* because graph-defined kernels always round-trip every tensor through
  shared memory, which dominates light-compute kernels (draft text quantifies
  1.8–2.0× overhead); planned fix is a shared-memory bypass.
- **End-to-end** (PyTorch + Mirage kernels vs stock PyTorch on Chameleon, nGPT,
  LLaMA-3, LoRA): 0.9–1.9× — note the 0.9, an end-to-end slowdown case exists.

### 1.6 OSDI camera-ready vs earlier arXiv drafts

Visible via `\if 0` blocks in the source: benchmark set changed (was
MHA/GQA/MQA × 3 decoding modes + MLP + MoE + LoRA; now the six above plus
end-to-end runs); headline 1.1–2.9× → up to 3.3×; the ReLU→exp workaround was
replaced by SiLU-as-operator + an (undetailed) solver-based non-Lax verifier;
the verifier-practice paragraph (single 16-bit test, no observed false
positives) is new candor. This version has **no Triton transpiler**: 30K LoC
C++/CUDA/Python generating CUDA source JIT-compiled by nvcc, pre-defined ops on
cuDNN/cuBLAS, block/thread ops on CUTLASS + PTX, Z3 as both SMT engine and ILP
solver. (The Triton/CUDA dual-transpiler story belongs to the later open-source
project.)

---

## 2. Tensat: Equality Saturation for Tensor Graph Superoptimization (MLSys'21)

Yang, Phothilimthana, Wang, Willsey, Roy, Pienaar. arXiv 2101.01332. Rust on
`egg`, SCIP (OR-tools) as ILP solver; code at github.com/uwplse/tensat.

### 2.1 Representation (§3.1, Table 2)

TASO's operator language as e-graph terms: each operator is an e-node whose
value is its output tensor; inputs *and* scalar/string parameters are children.
Node types: tensor, string, integer, tensor-tuple. ~20 ops: ewadd, ewmul,
matmul (activation flag as child), grouped conv (stride/padding/activation as
integer children; normal + depthwise as special cases), relu/tanh/sigmoid,
poolmax/poolavg, transpose, enlarge, concat_n, split + split_0/split_1, merge,
reshape, input, weight, noop. Encoding decisions worth remembering:

- everything syntactic — parameters are child nodes (permutations/shapes as
  format strings), keeping rewriting purely term-level;
- multi-output split returns a tuple; projections extract components; the split
  position is implicit ("at the place of the most recent concat") with split
  locations tracked in e-class analysis data;
- `egg` needs fixed arity → one `concat_n` per input count;
- multiple graph outputs joined under a semantics-free `noop` root so extraction
  has a single root e-class;
- e-class analyses carry shape, layout, and split-location data; rewrites carry
  TASO-style shape preconditions checked before applying a match.

### 2.2 Rules and multi-pattern application (§3.2, §4)

Rules are **TASO's automatically synthesized and formally verified substitution
set, unchanged** (743 in TASO's publication) — Tensat contributes no rule
machinery. Single-pattern rules are native `egg` e-matching. **Multi-pattern
rules** (≥2 output patterns; e.g. Fig. 2's matmul batching:
`(matmul ?a ?b), (matmul ?a ?c) → (split_0 (split 1 (matmul ?a (concat_2 1 ?b
?c)))), (split_1 …)`) are applied by Algorithm 1: canonicalize all source
patterns up to variable renaming; e-match each canonical pattern once per
iteration; per rule, take the Cartesian product of component-match lists, keep
combinations binding shared variables to the same e-class, apply survivors.

**Blowup:** with N matmuls sharing an input, iteration 1 creates O(N²) new
matmuls sharing it, iteration 2 O(N⁴) — double-exponential. Mitigation: a
separate cap **k_multi** for multi-pattern iterations (default **1**), then
single-pattern rules only, overall caps 50,000 e-nodes / 15 iterations.

### 2.3 Cycle filtering (§5.2, Alg. 2)

Valid rewrites create e-graph cycles (after matmul-merge, each `split_i`
e-class transitively references the other), and extracting a cyclic term is not
a runnable DAG. Since acyclicity constraints cripple the ILP (below), cycles are
filtered during exploration:

- *vanilla:* whole-e-graph cycle check per candidate application — O(n_m·N)
  per iteration, dies as matches scale with e-graph size;
- *efficient:* (a) pre-filtering — one descendants-map pass per iteration, then
  O(1) per match (sound but misses cycles created earlier in the same
  iteration); (b) post-processing — DFS passes after each iteration put the
  last-added node of each cycle on a **filter list**, enforced at extraction as
  x_i = 0; repeat until cycle-free.

Ablation (Table 6, k_multi=2, exploration time): NasRNN 2932 s → 1.47 s
(~2000×); NasNet-A >3600 s → 8.62 s; BERT 32.9 s → 0.89 s.

### 2.4 Extraction (§5, §5.1)

**Cost model:** TASO's — each op's cost is its measured standalone runtime at
concrete shapes; graph cost = sum. Explicitly assumes serial kernel execution
("GPUs typically run one operator at a time"); a footnote concedes parallel-
kernel hardware would need e.g. a learned extractor.

**Greedy fails on exactly the interesting rules:** it evaluates each e-class
independently and never picks `split_i` (the merged kernel amortizes across
*both* consumers). Table 4: NasNet-A greedy 22.5 ms vs 17.8 original (a
pessimization; ILP 16.6); BERT greedy no-op (ILP 1.73 vs 1.88); NasRNN greedy
1.15 vs ILP 1.10.

**ILP:** binary x_i per e-node; minimize Σ cᵢxᵢ; exactly one pick in the root
e-class; demand propagation x_i ≤ Σ_{j∈e_m} x_j per child e-class m; filter-list
zeroing. Acyclicity, if encoded (topological-order reals t_m ∈ [0,1] with big-M
constraints), is the killer: NasRNN k_multi=1 extraction 1116 s with cycle
constraints vs **0.32 s** without; at k_multi=2 all benchmarks with cycle
constraints time out at 1 h while cycle-free ILP solves in 75–510 s. At
k_multi=3 the ILP times out anyway for BERT/NasRNN/NasNet-A/Inception-v3 —
**extraction, not exploration, is the scaling wall.**

### 2.5 Results (§6) and admitted limitations

One NVIDIA T4, TASO's cuDNN runtime, TASO baseline at default backtracking
(n=100; n=1000 gains <1% at 11× the time). Speedup over original graph
(TASO → Tensat, %): NasRNN 45.4 → 68.9, BERT 8.5 → 9.2, ResNeXt-50 5.5 → 8.8,
NasNet-A 1.9 → 7.3, SqueezeNet 6.7 → 24.5, VGG-19 8.9 → 8.9, Inception-v3
6.3 → 10.0. The "16%" headline is NasRNN's relative improvement over TASO's
graph; draft comments put the average at ~6.6%. Optimizer time 9.5–379× faster
than TASO (avg ~48×); at k_multi=1 total times are 0.2–11 s.

All showcased wins are sharing rewrites (Appendix A): k matmuls/convs sharing
an operand merged via concat+op+split; four convs → two by concatenating weight
kernels along output and input channels, with weight-only concats precomputable
at inference time. Admitted limits: beyond a few multi-pattern iterations "the
e-graph becomes too big for the extraction phase"; the cost model's linearity/
independence assumptions fail observably (SqueezeNet gets slower as k_multi
grows while the model claims wins); ResNet-50 gets zero speedup — the rule set
binds, not the search; inference graphs only, no training/backprop graphs;
layouts carried for shape checking but layout-conversion costs are not modeled.

---

## 3. Literature survey 2021–2026 (verified 2026-07-07)

### A. The TASO → PET → EinNet → Mirage lineage

- **TASO** (SOSP'19), **PET** (OSDI'21; partially-equivalent transformations +
  automated corrections), **EinNet** (OSDI'23, Zheng et al., code in
  InfiniTensor; derivation-based transformations over general tensor-algebra
  expressions — essentially einsums — creating new operators on demand; 1.52×/
  1.55× average over prior optimizers on A100/V100). EinNet is the closest
  ancestor to OCANNL's einsum IR — the one to skim closely. SECONDARY.
- **Mirage** (OSDI'25) — still SOTA for joint algebraic + schedule
  superoptimization in mid-2026. PRIMARY (kept).
- **MPK / Mirage Persistent Kernel** (arXiv 2512.22219, accepted OSDI'26;
  released June 2025, ~2.4k stars, active; the mirage-project repo's current
  headline). Compiles multi-GPU LLM inference into one persistent megakernel
  (1.2–6.7× latency reduction), reusing the Mirage superoptimizer at thread-block
  level. The opposite pole of OCANNL's kernel fission; an inference-runtime
  paper, not a search paper. SECONDARY.
- **Axon** (arXiv 2606.26344, June 24 2026 preprint; Kothari, Zhu, Kroening,
  Sung — Hydride's lead author). Synthesizes target instructions from semantics
  specs, discovers algebraic transformations by operator propagation, verifies
  with **SMT over unbounded tensors** — replacing exactly Mirage's probabilistic
  verifier. Two weeks old and unreviewed. SECONDARY / watch closely, likely
  future PRIMARY.

### B. Equality saturation ecosystem

- **egglog** (PLDI'23, arXiv 2304.04332) — the ecosystem center of gravity,
  superseding `egg`: Datalog-style analyses, scheduling, proofs, Python
  bindings, a PLDI'25 tutorial. SECONDARY (design ideas; Rust, so OCaml use
  means FFI).
- **Tensat superseded?** No — still canonical. Best successor: **Hartmann, He &
  Yoneki** (PACT'24, arXiv 2410.05534) — MCTS-guided rewrite selection + fast
  extraction with runtime estimates, up to 11% over prior eqsat approaches.
  Read paired with Tensat.
- MLIR infrastructure: **DialEgg** (CGO'25; MLIR↔egglog bridge), the **eqsat
  MLIR dialect** (EGRAPHS'25, arXiv 2505.09363), **Guided Equality Saturation**
  (POPL'24 — human/oracle waypoints against saturation blowup). Footnote-level.
- **OCaml options, verified:** [`ego`](https://opam.ocaml.org/packages/ego/)
  (github.com/verse-lab/ego, Kiran Gopinathan) is a real, explicit loose port
  of egg with `Ego.Basic`/`Ego.Generic` — but latest release 0.0.6 (Nov 2021),
  ~35 commits, ~83 stars, effectively unmaintained 4+ years, **GPL-3.0+**
  (vendoring concern). Nothing else on opam or awesome-egraphs. Practical
  paths if eqsat is ever wanted: revive/fork ego, write a ~1–2 kLOC core from
  scratch, or FFI/shell out to egglog.

### C. Verified tensor-graph rewriting

- **TensorRight** (POPL'25, Arora et al., ADAPT/UIUC — Charith Mendis's group;
  arXiv 2511.17838; github.com/ADAPT-uiuc/TensorRight). First system verifying
  tensor-graph rewrites for **arbitrary rank and size**, via a DSL with
  *aggregated axes* (bundles of axes quantified as a unit); proves 115/175 XLA
  rules in full generality where the best bounded-verification alternative
  expresses 18. Aggregated axes are conceptual cousins of OCANNL's row
  variables. PRIMARY (promoted into the anchor set).
- Narrower relatives: egglog-based verification of computational graphs in
  distributed-ML frameworks (arXiv 2509.10694); Axon's SMT (above).

### D. Schedule search / autotuning cost models

- **Halide learned autoscheduler** (Adams et al., SIGGRAPH 2019) — beam search +
  learned cost model; the archetype of OCANNL's setup, CPU-era. SECONDARY.
- **Ansor** (OSDI'20) — sketch generation + evolutionary search + learned
  model; its sketch/annotation split is the standard vocabulary. SECONDARY.
- **TVM MetaSchedule** (NeurIPS'22) — SKIP (engineering doc more than reading).
- **TLP** (ASPLOS'23) — cost model over schedule primitives as text. SKIP
  unless building a learned model.
- **ROLLER** (OSDI'22, Microsoft) — **constructive** generation: rTiles aligned
  with hardware features (transaction granularity, tensor-core shapes, bank
  widths) + a static throughput model walking the memory hierarchy; kernels in
  seconds, no learned model, competitive with hours-long tuners. The single
  most relevant idea for BEAM convergence. PRIMARY.
- **Heron** (ASPLOS'23) — auto-generated constraints + CSP/genetic exploration
  for tensor-core-like accelerators; 2.71× over auto-generation baselines.
  Directly relevant post-`Tensorize`. SECONDARY (high).
- **Felix** (ASPLOS'24) — gradient descent over a differentiable symbolic cost
  surrogate; needs a differentiable schedule parameterization OCANNL doesn't
  have. SKIP.
- **Pruner** (ASPLOS'25, arXiv 2402.02361) — "draft-then-verify": a cheap
  analytical hardware-fitness model drafts a small candidate set, the expensive
  model/measurement verifies; ~4× tuning-time speedup vs TenSet/TLP; MoA-Pruner
  adds cross-platform online adaptation. The best current answer to "when does
  a model beat timing": zero-training analytical filter first, timing as ground
  truth. PRIMARY (co-winner with ROLLER).
- **tinygrad BEAM** — implemented reference (hand-coded OptOps action space,
  beam over action sequences, on-device timing, kernel cache); documented
  mainly in community notes (mesozoic-egg/tinygrad-notes); no paper.

### E. LLM-guided kernel optimization

- **KernelBench** (arXiv 2502.10517, Stanford; ICML'25): 250 PyTorch workloads;
  the `fast_p` metric (fraction of tasks both correct and >p× faster) is
  independently useful for evaluating any search's output. Frontier reasoning
  models beat PyTorch baselines in <20% of cases. SECONDARY (methodology only).
- **AlphaEvolve** (DeepMind, May 2025): verified 23% speedup on a Gemini matmul
  kernel (~1% training-time reduction), up to 32.5% on a FlashAttention kernel,
  the 48-multiplication 4×4 complex matmul — but an industrial-scale
  evolutionary loop. SECONDARY/watch.
- **Sakana "AI CUDA Engineer"** (Feb 2025): claimed 10–100×; shown to be
  **reward-hacking the evaluation harness** (memory-reuse exploits bypassing
  correctness checks); retracted/corrected. Enduring lesson: any
  execution-based cost function gets exploited by its search — including a
  non-LLM BEAM. Cautionary footnote, not reading.
- NVIDIA/DeepSeek-R1 experiment (Feb 2025): verifier-in-the-loop reached 100%
  KernelBench L1 / 96% L2 *correctness*, not speed records. SKIP.
- The 2026 preprint wave (EvoEngineer, cuPilot, KForge, Kernel-Smith, AccelOpt,
  …) is dense but unsettled. Category verdict: watch-only.

### F. Classical / SIMD superoptimizers

- **Minotaur** (OOPSLA'24, arXiv 2306.00229; Liu, Mada, Regehr) — synthesizing
  superoptimizer for LLVM SIMD: cut extraction → synthesis → Alive2-style
  verification; 7.3% avg on GMP, 1.5% on SPEC; upstreamed to LLVM. Its "cut"
  granularity maps well onto OCANNL's vectorized loop bodies (gh-ocannl-164).
  SECONDARY (high); a possible post-v0.9 lead if SIMD codegen underperforms.
- **Hydride** (ASPLOS'24) — auto-generated retargetable vector IR from
  pseudocode specs; aimed at backend generation. SKIP (note the Axon
  connection via its lead author).
- **Diospyros** (ASPLOS'21) / **Isaria** (ASPLOS'24) — eqsat for DSP
  vectorization; the historical bridge between categories B and F. Footnote.

---

## 4. Cross-cutting observations (beyond the proposal's seam map)

Observations that emerged from holding all three inputs together; recorded here
because they inform *how* to implement F1–F4, not *whether*.

1. **OCANNL's einsum specs are already an abstract-expression language.**
   Mirage's abstraction (element identity erased, reduction structure and sizes
   kept) is approximately what an einsum spec plus dimension bindings encodes.
   If OCANNL ever needs derivability-style pruning for a generative rewrite
   search, the natural formulation is over einsum specs and `Assignments.t`
   accumulation structure — no Z3 required for the fragment OCANNL emits,
   because the algebra is fixed (add/max-reduce with mul/add) rather than
   free-form. This is why skipping Mirage's SMT machinery loses little.
2. **Mirage's Lax fragment does not cover tropical accumulation.** OCANNL's
   `@^+`/`@^^` (max-reduce) operators fall outside multi-linear + div + exp, so
   finite-field identity testing does not apply to them. For the F3 oracle this
   partitions cleanly: finite-field testing with Schwartz–Zippel-style error
   bounds for the linear fragment; for max-plus programs, integer sampling makes
   each trial *rounding-free* (max and add are exact on ints — any discrepancy
   is decisive, no tolerance needed), **but the check remains probabilistic**:
   inequivalent tropical expressions can agree on sampled inputs, and no
   Schwartz–Zippel analog is claimed here, so acceptance means "passed N
   randomized trials", not proven equivalence; only genuinely floating-point
   programs additionally need a tolerance policy. Worth encoding as three
   oracle modes rather than one tolerance knob — with the false-positive story
   documented per mode (bounded / unbounded-but-rounding-free / tolerance).
3. **imap/omap/fmap ↔ existing OCANNL invariants.** Mirage's "no replication on
   outputs" (omap) is the same fact as `validate_parallel`'s
   materialized-write-coverage check; fmap = φ on an accumulator is
   `Workgroup_reduce`/`Privatize` territory; imap replication is what OCANNL
   gets implicitly when a broadcast operand's loop isn't retyped. The μGraph
   formalism is a *naming* of decisions OCANNL's schedule layer already makes —
   useful as a checklist that the BEAM action space spans all three maps, which
   it currently does not (imap/omap choice is fixed by the annotator; only tile
   sizes vary).
4. **Convergent conclusion on cost models across all three sources:** Tensat's
   per-op-sum model fails where kernels interact (SqueezedNet regression);
   Mirage defers cost entirely to post-verification profiling; Pruner keeps
   measurement as ground truth and uses models only to *order* candidates.
   Nobody in the surveyed literature successfully replaces end-to-end
   measurement for final selection. OCANNL's timing-based BEAM is on the right
   side of this; the only model worth building is the cheap ordering filter.
5. **Both papers' wins concentrate in inference-shaped graphs.** Tensat's
   sharing rewrites pay on multi-branch inference graphs (NasNet, Inception);
   Mirage's fusions pay on small-batch decoder inference (GQA, LoRA). OCANNL's
   current benchmarks are training loops, where the backward pass reuses
   forward intermediates and the sharing patterns differ. F4's realistic first
   targets are therefore OCANNL's *inference* paths (the #377 transformer
   inference demo is the natural benchmark), not the training tests.
6. **Fission and megakernels are one axis, not two camps.** OCANNL's kernel
   fission (split at materialized cross-nest edges, event chains) and MPK's
   persistent megakernel are endpoints of a launch-granularity spectrum; the
   grid-sync cooperative-launch strategy already sketched in
   schedule-ir-optops §7 sits between them. When the BEAM matures, launch
   granularity is itself a searchable decision over the *same* segmentation —
   a cheap observation now, a possible v1.x follow-up later.
7. **A workshop-paper angle exists.** TensorRight's aggregated axes and
   OCANNL's row variables solve the same quantification problem from opposite
   ends (verification vs. inference). A short paper sketching row-variable
   einsum specs as a rewrite-rule language whose rules are verifiable
   TensorRight-style — with the concat-axis sharing rules (F4) as the worked
   example — would be a natural FProPer/OCaml-Workshop follow-up to the v0.7
   shape-inference paper material.
8. **Canonical-form discipline transfers cheaply.** Mirage's increasing-rank
   canonical enumeration is the same trick as keying the BEAM winner cache by a
   canonicalized kernel hash (schedule-ir-optops §8): both exist to stop the
   search from paying twice for permuted equivalents. Worth making the
   canonicalization shared infrastructure (one function used by both the cache
   key and any future dedup of schedule prefixes) rather than two ad-hoc ones.
