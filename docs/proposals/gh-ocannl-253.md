# Proposal: Study and Incorporate llm.c Lessons

**Task**: gh-ocannl-253
**Issue**: https://github.com/ahrefs/ocannl/issues/253
**Milestone**: v0.8

## Goal

Produce a structured analysis of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) project, identifying which design decisions and optimization techniques are applicable to OCANNL, which are already covered by existing tasks, and which represent unique lessons that should inform OCANNL's v0.8-v0.9 development. The deliverable is a documented gap analysis with concrete recommendations, not direct code porting -- llm.c is hand-written C/CUDA for GPT-2, while OCANNL generates kernels from a high-level einsum IR. The value is in understanding *why* llm.c's choices work and mapping those insights to OCANNL's architecture.

## Acceptance Criteria

- [ ] A written analysis document (`docs/llm-c-analysis.md`) covering all major llm.c subsystems listed in the Context section below, with each subsystem assessed for OCANNL applicability
- [ ] Each llm.c technique is classified as: (a) already addressed by an existing OCANNL task, (b) applicable and recommended for a specific OCANNL milestone, or (c) not applicable with rationale
- [ ] For techniques in category (b), the analysis includes concrete recommendations: which OCANNL file/module would change, what the change looks like at a high level, and which milestone it fits
- [ ] Identification of any llm.c patterns that challenge OCANNL's architectural assumptions (e.g., patterns that are hard to express in the einsum IR or that require manual kernel specialization)
- [ ] The analysis references specific llm.c source files and line ranges for traceability
- [ ] Cross-references to existing OCANNL proposals (gh-ocannl-412 tiling, gh-ocannl-164 AVX/AVX2, gh-ocannl-318 megakernels) are explicit, noting what each existing task covers and what gaps remain
- [ ] A summary table mapping llm.c components to OCANNL equivalents/gaps

## Context

### What llm.c Is

llm.c implements GPT-2 (124M) training in pure C and CUDA, achieving parity with PyTorch fp32 training performance. Key characteristics:
- Single-file C implementation (`train_gpt2.c`) and single-file CUDA implementation (`train_gpt2.cu` / `train_gpt2_fp32.cu`)
- All operations hand-written: matmul, layernorm, softmax, GELU, attention, residual connections, AdamW optimizer
- Multi-GPU training via NCCL
- Mixed precision (fp32 and bf16/fp16 paths)
- Achieves PyTorch-level throughput through careful memory management, kernel fusion, and GPU utilization

### llm.c Subsystems to Analyze

**1. Kernel implementations (CUDA)**
- Matrix multiplication: custom tiled CUDA kernels, cuBLAS integration, cooperative groups
- LayerNorm: fused mean+variance computation, online Welford algorithm, vectorized loads (`float4`)
- Softmax: online softmax algorithm (numerically stable single-pass), warp-level reductions
- GELU: fused activation with `tanhf` approximation
- Attention: fused attention score computation, causal masking, softmax, attention output
- Residual connections: fused add operations
- Cross-entropy loss: fused softmax + log-likelihood in single kernel

**Already covered by existing tasks:**
- Matrix multiplication tiling/parallelization -> gh-ocannl-412 (comprehensive 5-phase proposal covering IR tiling, multi-threaded CUDA, SMEM, register blocktiling, C backend tiling)
- AVX/AVX2 for CPU inner loops -> gh-ocannl-164 (aligned memory, compiler flags, auto-vectorization)
- Megakernel approach -> gh-ocannl-318 (deep dive document already exists at `docs/megakernel-deep-dive.md`)

**NOT covered by existing tasks (unique llm.c lessons):**
- Fused reduction kernels (layernorm, softmax) with warp shuffle primitives (`__shfl_xor_sync`, `__shfl_down_sync`)
- Online/streaming algorithms for numerically stable reductions (Welford for variance, online softmax)
- Vectorized memory access patterns (`float4` loads/stores for coalesced memory access)
- Fused activation functions (GELU as a single inlined computation)
- Cooperative groups for flexible thread synchronization

**2. Memory management**
- Pre-allocated contiguous parameter and gradient buffers (single `malloc` for all weights)
- Activation checkpointing / recomputation during backprop
- In-place operations where possible (residual adds)
- Memory-mapped weight loading from checkpoint files

**OCANNL relevance:**
- OCANNL allocates per-tensor buffers (via `Cu.Deviceptr.mem_alloc` per tensor in `cuda_backend.ml`). llm.c's contiguous allocation reduces allocation overhead and enables pointer arithmetic. The pool allocator planned for v0.7.2 (gh-ocannl-344) partially addresses this.
- Activation checkpointing is not currently in OCANNL's scope but becomes important for training larger models. The `Virtual` memory mode and virtualization pass in OCANNL serve a related purpose (eliminating intermediate allocations) but don't support recomputation.

**3. Training loop design**
- Gradient accumulation across micro-batches
- Learning rate warmup + cosine decay schedule
- Gradient clipping (global norm)
- Validation loss computation interleaved with training
- Checkpoint saving/restoring

**OCANNL relevance:**
- OCANNL's `Train` module provides `sgd_one` (SGD with momentum) but no AdamW, learning rate scheduling, or gradient clipping. These are higher-level training utilities.
- The `sequential_loop` and `round_robin` functions in `train.ml` handle batch iteration but not gradient accumulation across micro-batches.
- llm.c's training loop patterns are informative for OCANNL's training examples (v0.7.1 milestone: MNIST, CIFAR, transformer inference).

**4. CUDA optimization patterns**
- Warp-level primitives: `__shfl_xor_sync`, `__shfl_down_sync` for intra-warp reductions without shared memory
- Block-level reductions: two-phase reduction (intra-warp shuffle, then inter-warp via shared memory)
- Memory coalescing: careful index arithmetic to ensure adjacent threads access adjacent memory
- Occupancy tuning: block sizes chosen to maximize SM occupancy
- Stream-based overlapping of compute and data transfer

**OCANNL relevance:**
- Warp shuffles are absent from OCANNL's CUDA builtins (`builtins_cuda.ml`). Adding them would benefit any reduction operation (layernorm, softmax, loss computation) on GPU. This is a gap not covered by gh-ocannl-412 (which focuses on matmul tiling).
- Memory coalescing is implicitly addressed by the tiling proposal (gh-ocannl-412 Phase 2) but only for matmul. General coalesced access patterns for element-wise and reduction operations need attention.
- OCANNL currently launches one kernel per routine; there is no compute/transfer overlap.

**5. Numerical precision**
- Mixed precision training: bf16 forward/backward, fp32 master weights and optimizer state
- Stochastic rounding for gradient updates
- Kahan summation for accurate gradient accumulation

**OCANNL relevance:**
- OCANNL has `Half_prec`, `Bfloat16_prec`, and `Fp8_prec` type support in `ops.ml` and corresponding CUDA types in `cuda_backend.ml`. However, there is no mixed-precision training infrastructure (maintaining fp32 master weights alongside bf16 compute weights).
- This is a v0.9+ concern but worth documenting as a gap.

### OCANNL Architecture Constraints

Key constraints that affect which llm.c lessons transfer cleanly:

1. **Generated kernels, not hand-written.** OCANNL generates C/CUDA from the `Low_level.t` IR. Any llm.c technique must be expressible as an IR transformation or code generation pattern, not as a hand-written kernel.

2. **Einsum-based computation model.** Operations are specified as einsum contractions with batch/output/input dimensions. llm.c's operation-specific kernels (e.g., "fused layernorm kernel") would need to be recognized as patterns in the einsum IR and lowered to specialized code.

3. **Single-threaded kernels (current state).** All CUDA kernels run with `grid_dim=1, block_dim=1`. The tiling proposal (gh-ocannl-412) is the prerequisite for any llm.c GPU optimization technique.

4. **No GELU activation.** OCANNL's `nn_blocks.ml` provides `relu`, `softmax`, `layer_norm`, `dropout`, and attention blocks, but no GELU. Adding GELU is straightforward (it composes from existing operations: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`) but does not currently exist.

5. **SGD only.** The `Train` module has SGD with momentum/weight_decay/Nesterov, but no Adam/AdamW optimizer. AdamW is essential for transformer training.

### Related Proposals and Documents

| Document | Coverage |
|----------|----------|
| `docs/proposals/gh-ocannl-412.md` | Tiling, multi-threaded CUDA, SMEM, register blocktiling (5 phases) |
| `docs/proposals/gh-ocannl-164.md` | AVX/AVX2 intrinsics, aligned memory, auto-vectorization for C backend |
| `docs/megakernel-deep-dive.md` | Megakernel patterns from Hazy Research and Mirage MPK, mapped to OCANNL |
| `docs/proposals/gh-ocannl-377.md` | Transformer inference demo (GPT-2/LLaMA) |
| `ROADMAP.md` | v0.8 = GPU-style performance (tiling + megakernels + llm.c lessons) |

## Approach

### Phase 1: llm.c Source Study (1-2 days)

Read and annotate the following llm.c source files:
- `train_gpt2.c`: CPU reference implementation -- understand the complete training pipeline
- `train_gpt2_fp32.cu` (or `train_gpt2.cu`): CUDA implementation -- focus on kernel implementations
- `llmc/` directory: modular CUDA kernels (matmul, layernorm, attention, etc.)

For each kernel/subsystem, document:
- The algorithm and its performance characteristics
- Which CUDA primitives it uses (warp shuffles, shared memory, cooperative groups, etc.)
- Memory access patterns and how coalescing is achieved
- Whether the technique is general or GPT-2-specific

### Phase 2: Gap Analysis (1 day)

Cross-reference llm.c techniques against:
1. Existing OCANNL capabilities (what already works)
2. Existing proposals (gh-ocannl-412, gh-ocannl-164, gh-ocannl-318)
3. OCANNL's IR constraints (what can be expressed as IR transformations)

Produce a classification table for every identified technique.

### Phase 3: Recommendations Document (1 day)

Write `docs/llm-c-analysis.md` with:
- Executive summary of findings
- Per-subsystem analysis sections
- Recommendations table: technique, OCANNL applicability, target milestone, effort estimate
- Architectural insights: patterns that are hard for OCANNL's IR to express, with suggestions for IR extensions

### Prioritized Recommendations (Expected)

Based on preliminary analysis, the likely priority ordering for unique llm.c lessons (not already covered by other tasks) is:

1. **Warp shuffle reduction builtins** (v0.8, small effort): Add `__shfl_xor_sync` and `__shfl_down_sync` to `builtins_cuda.ml`. These enable efficient intra-warp reductions without shared memory, benefiting layernorm, softmax, and loss kernels. Prerequisite: multi-threaded kernels from gh-ocannl-412.

2. **GELU activation** (v0.7.1, trivial): Add to `nn_blocks.ml`. Needed for any GPT-2/transformer training example.

3. **AdamW optimizer** (v0.7.1, small effort): Add to `train.ml` alongside existing SGD. Essential for transformer training. Pattern is well-known: per-parameter first/second moment tracking with weight decay decoupled from gradient.

4. **Fused reduction code generation** (v0.8, medium effort): Teach the code generator to emit efficient parallel reductions for operations that reduce across dimensions (layernorm variance, softmax normalization). This extends gh-ocannl-412's parallel loop mapping to handle reduction patterns specifically.

5. **Vectorized memory access** (v0.8, medium effort): Generate `float4` loads/stores for coalesced memory access in generated CUDA kernels. Deferred in gh-ocannl-412 but identified as a follow-up. llm.c's usage provides concrete patterns.

6. **Learning rate scheduling and gradient clipping** (v0.7.1-v0.8, small effort): Higher-level training utilities for `train.ml`. Needed for realistic training examples.

7. **Mixed precision infrastructure** (v0.9+, large effort): fp32 master weights with bf16 compute. Architecturally significant -- requires tracking two copies of parameters with different precisions.
