# llm.c lessons for OCANNL — research notes

Deliverable of [gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253)
("Study and Incorporate llm.c Lessons"); companion evidence base for
[`docs/proposals/gh-ocannl-253.md`](../proposals/gh-ocannl-253.md). Every llm.c
claim below was extracted from fetched source, and every OCANNL claim was
verified against the repository — not taken from the proposal's (stale) 2026-06-12
status update.

**Provenance:** written 2026-07-07. llm.c was read at `master` commit
`f1e2ace651495b74ae22d45d1723443fd00ecd3a` (2025-05-10, "Merge pull request #801
… Fix gradient tests" — the repository's last commit; the project is in
maintenance mode, which makes this a stable, final artifact to study). Files
fetched from `raw.githubusercontent.com/karpathy/llm.c/master/`: `README.md`,
`train_gpt2.c`, `train_gpt2.cu`, `llmc/{adamw,attention,layernorm,gelu,
fused_classifier,matmul,encoder,global_norm,zero,cuda_utils}.cuh`,
`llmc/schedulers.h`, `llmc/cudnn_att.cpp`, and the `dev/cuda` directory page.
OCANNL was verified against `master` @ `e22a1d69` (2026-07-07); file citations
are by module/function name, with line numbers only where load-bearing.
Note: `e22a1d69` is ahead of the docs branch this note lands on (based on
`605d565a`) — items marked (†) below are on master but not yet in this branch's
tree; they become accurate here once the branch merges into current master.

---

## 1. What changed in OCANNL since the proposal was written

The proposal's Context section (and its 2026-06-12 status update) describes a
pre-v0.7/v0.8 OCANNL: single-threaded `grid_dim=1, block_dim=1` CUDA kernels,
per-tensor `mem_alloc`, no schedule search. **None of that holds anymore.**
Verified current state (all on `master` @ `e22a1d69`):

- **Schedule IR** (`arrayjit/lib/schedule.ml{,i}`): `Split`, `Swap`, `Retype`,
  `Unroll` (with materializing variant), `Stage` (shared-memory tiles),
  `Privatize`, `Expand_zero`, `Tensorize` (MMA/`simdgroup_matrix` intrinsics).
  Loops carry axis types `Serial | Grid | Workgroup | Workgroup_reduce |
  Unrolled | Vectorized` (`arrayjit/lib/low_level.mli`), with
  `Workgroup_barrier` statements and workgroup-shared staging nodes.
- **Default GPU annotator + kernel fission**: `Schedule.maybe_default_schedules`
  auto-parallelizes CUDA/Metal kernels (`automatic_gpu_schedule`, default true);
  `schedule_fission` splits routines at materialized cross-nest edges, with
  aligned cross-nest parallelism (PR #111, †) merging what can stay in one kernel.
  `Schedule.check_hardware_limits` validates every kernel against
  `Backend_intf.hardware_limits` (threads, shared memory, MMA capability).
- **Autotune** (`arrayjit/lib/autotune.ml{,i}`): beam search over schedule
  transforms timed on the real device, per-fission-segment schedules, matmul
  register-blocktiling/operand-packing **sketches**, disk cache keyed by
  canonical digest (`schedule_cache.ml`), scratch `timing_ctx` lineages.
- **SIMD on CPU**: `Vectorized` axes render as vector-extension code on the cc
  backend (`cc_backend.ml` `cc_vector_bytes`, `c_syntax.ml` `vector_bytes`
  functor hook; the default implementation is 0 = scalar, and **only cc
  overrides it** — GPU backends emit scalar accesses).
- **Universal pool allocator** (#344, `arrayjit/lib/backends.ml` planner +
  `Backend_intf.buffer_loc = { pool_id; offset }`): bump-packed pooled buffers
  on all backends, offsets padded to `Ops.buffer_alignment = 32` bytes
  (`arrayjit/lib/ops.ml`).
- **Interval analysis** (`arrayjit/lib/interval.ml`) folds bounds guards;
  index precision is signed int32/int64 selected by `Ops.index_prec` /
  `large_models` (both in `arrayjit/lib/ops.ml`).
- **Backends renamed** (†): `cc` (default), `multidev_cc` (one domain + FIFO per
  device ordinal), `cuda`, `metal`; old names are `get_backend` aliases
  (the `Backends.get_backend` name match on master — this branch's tree still has
  the pre-rename `sync_cc`/`multicore_cc` surface). Metal encodes fissioned
  segments of one routine into batched dispatches on one command buffer with
  event chains at segment boundaries (†).
- **Model-side**: tensor persistence with namespaced checkpoints
  (`lib/persistence.mli`), one-hot→gather rewrite
  (`Low_level.rewrite_one_hot_reductions` / `Get_dynamic`, gh-343), RoPE and
  sinusoidal position embeddings, decoder-only transformer blocks, softmax with
  `@^^` max-reduce (`lib/nn_blocks.ml`), BPE token-id bridging
  (`token_ids_of_batch`), data-parallel merge-buffer all-reduce
  (`lib/parallel.ml` `shard_along`/`gather`, exercised by
  `test/training/data_parallel.ml`).

Still true from the proposal: **no GELU anywhere in the tree** (`git grep gelu`
on master is empty outside docs), **no AdamW / LR scheduling / gradient
clipping** in `lib/train.ml` (only `sgd_one`/`sgd_update`), **no
warp-shuffle builtins** (`git grep shfl` matches nothing; `Workgroup_reduce`
communicates only through workgroup-shared memory + barriers per
`low_level.mli`), and mixed-precision *training* infrastructure (master
weights) does not exist — though the precision *types* (`half`, `bfloat16`,
`fp8`) and per-tensor precision plumbing do (the `precision` and `prec` types
in `arrayjit/lib/ops.ml`).

**Driver workload framing.** GPT-2 (and Gemma) end-to-end inference has been
promoted to a ready task (repo home: `docs/proposals/gh-ocannl-377.md`), and is
the designated driver for v0.8/v0.9 performance work. The optimizer chain —
**AdamW → mixed-precision optimizer states → quantized AdamW → stochastic
rounding** — is the other open arc these lessons feed. (Quantization verdicts
below deliberately do *not* route through gh-ocannl-137, which was closed
not-planned as an omnibus; the open optimizer task chain is the current home
for quantized-optimizer work.)

---

## 2. llm.c evidence base

### 2.1 Shape of the project

llm.c trains GPT-2/GPT-3-miniseries models in ~1,400 lines of C
(`train_gpt2.c`, CPU reference) and a single-file CUDA implementation
(`train_gpt2.cu`) plus ~23 headers in `llmc/`. README claims ~7% faster than
PyTorch Nightly at the time; mixed precision (BF16 default) and cuDNN flash
attention are supported; multi-GPU via NCCL (MPI/TCP/filesystem rendezvous).
Governance lesson stated outright: PRs that buy 2% for 500 lines of complex C
are rejected from the root folder; complexity lives in `dev/`.

The `dev/cuda/` directory is a *kernel progression* library: each file
(`softmax_forward.cu`, `layernorm_backward.cu`, `adamw.cu`, …) contains
numbered kernel versions of increasing sophistication, each verified against
the CPU reference and benchmarked across block sizes; the fastest is then
hand-copied into `train_gpt2.cu`. This is a manual, human-in-the-loop version
of exactly what OCANNL's autotune automates (candidate schedules timed on
device, winner cached).

### 2.2 Reductions: warp shuffles + two-phase block reduction

`llmc/cuda_utils.cuh` defines the reduction vocabulary used everywhere:

```c
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

`warpReduceMax` is identical with `fmaxf`. `blockReduce<warp_reduction>` is the
canonical two-phase pattern: intra-warp shuffle, one value per warp to
`__shared__ float shared_val[WARP_SIZE]`, `__syncthreads()`, final warp
reduction. Every normalization/loss kernel is built from these.

### 2.3 Packed128: 128-bit vectorized memory access

`llmc/cuda_utils.cuh` `Packed128<ElementType>` — `alignas(16)`, payload sized
`sizeof(int4)/sizeof(ElementType)` — "forces the compiler to use 128-bit
loads/stores in GPUs that support (the LDG.128 and STS.128 instructions)".
Typedefs `f128` (float) and `x128` (`floatX`, the compute precision). Load/store
helpers carry cache hints: `load128cs`/`store128cs` (`__ldcs`/`__stcs`,
streaming, bypass L1 for use-once data), `store128cg` (`__stcg`, keep in L2).
Kernels assert alignment/divisibility (e.g. `gelu_forward` asserts
`N % (block_size * x128::size) == 0`) — the pointers can be assumed 16-byte
aligned *because* all parameters/activations come from one contiguous
allocation.

### 2.4 Matmul: cuBLASLt with epilogue fusion, not hand-written kernels

`llmc/matmul.cuh`: forward and both backward matmuls go through
`matmul_cublaslt()` (heuristic algorithm selection via
`cublasLtMatmulAlgoGetHeuristic`, FP32 accumulation `CUDA_R_32F`, TF32 enabled
for the FP32 mode via `CUBLAS_COMPUTE_32F_FAST_TF32`). Fusion happens through
**cuBLASLt epilogues**: bias via `CUBLASLT_MATMUL_DESC_BIAS_POINTER`, GELU via
`CUBLASLT_EPILOGUE_GELU_AUX_BIAS` (forward, `gelu_fusion >= 1`) and
`CUBLASLT_EPILOGUE_DGELU` (backward, `gelu_fusion >= 2`); weight-gradient
matmul sets `beta=1.0f` (`accumulate=true`) to fuse gradient accumulation. The
only hand-written matmul-adjacent kernel is `matmul_backward_bias_kernel9`
(column reduction of `dout`, two-stage: per-block partials into a buffer, then
`reduce_add_sum_kernel` with `f128` loads — explicitly avoiding atomics).

So llm.c's answer to "fast matmul" is *outsource to the vendor library and fuse
epilogues into it*; the interesting hand-written work is everything around the
matmuls.

### 2.5 LayerNorm family

`llmc/layernorm.cuh`:

- Forward `layernorm_forward_kernel6`: **warp-per-row** (`dim3(WARP_SIZE,
  block_y)`, each warp owns one C-length row), `x128` loads, weights/biases
  staged in shared memory, plain mean/variance (`sum/C`, `rsqrtf(v + eps)`) —
  notably **no Welford** in the production kernel (the proposal assumed
  Welford; dev/cuda explores it, production settled on the naive two-moment
  form with `warpReduceSum`).
- `fused_residual_forward_kernel5`: residual add + layernorm in one kernel
  (`out[k] = (float)in1[k] + (float)in2[k]` feeding the normalization),
  avoiding one round-trip of the residual stream through HBM.
- Backward `layernorm_backward_kernel10`: warp-per-token for dinp; `dweight`/
  `dbias` cross-block reduction goes through an **fp32 scratch buffer + an
  `atomicInc`-flag "last block finishes" protocol** — one atomic on a *flag*,
  none on data, keeping the reduction deterministic. Comment: activations use
  `=` not `+=` except the residual stream, where gradients must add.
- Launch configs: forward block 256; backward block 512 with grid
  `blocks_per_sm * multiProcessorCount` (occupancy-driven).

### 2.6 Softmax and the fused classifier

Two distinct softmaxes:

- Attention softmax `softmax_forward_kernel5` (`llmc/attention.cuh`):
  warp-per-row **online softmax** — running `maxval` with rescale of the
  running sum (`sumval *= expf(old_maxval - maxval)`), causal masking by simply
  iterating `i <= own_pos`, 4-wide inner unrolling, `warpReduceMax`/
  `warpReduceSum`.
- `fused_classifier_kernel5` (`llmc/fused_classifier.cuh`): fuses final-layer
  **softmax + cross-entropy loss + dlogits** in one block-per-row kernel
  (`__launch_bounds__(1024)`, grid B·T). `prepare_softmax_blockwide3` computes
  online max/sum via `blockReduce`; then the kernel writes the loss and
  overwrites logits with `(p - indicator(target)) * dloss` in the same pass —
  "calculate the gradients directly, saves bandwidth from probs during
  training": the [B,T,V] probability tensor is **never materialized**, and
  `dlogits` aliases the logits buffer. A template flag `WriteDLogits` selects
  training vs inference behavior at compile time; stores use `store128cs`.
  For GPT-2, V = 50257: this single fusion removes reads+writes of the largest
  activation in the model.

### 2.7 Attention

Non-cuDNN path (`llmc/attention.cuh`): `permute_kernel` reorders the fused QKV
projection output (B,T,3,NH,HS) into contiguous Q,K,V (B,NH,T,HS); Q·Kᵀ and
att·V are **cuBLASLt strided-batched matmuls** (`batch_count = B*NH`); softmax
is the custom kernel above; `unpermute_kernel` restores (B,T,NH·HS). Backward
uses `softmax_autoregressive_backward_inplace_kernel` (in-place, blocks walk
`t` in reverse for cache reuse) and reuses forward buffers as scratch
(`floatX* preatt = inp` — deliberate aliasing to cap memory).

cuDNN path (`llmc/cudnn_att.cpp`): flash attention through the cuDNN frontend
graph API (`graph->sdpa()` / `sdpa_backward()`), BF16/FP16 only (FP32
asserts out). Graph *construction* is "the VERY SLOW PART", so built graphs are
cached in static maps keyed by (B,NH,T,HS,is_inference)
(`lookup_cache_or_build_graph_fwd`). Disabled by default because of compile
overhead — a build-time-vs-runtime tradeoff OCANNL's schedule cache also
navigates.

### 2.8 GELU

`llmc/gelu.cuh`: tanh approximation
`0.5f * x * (1 + tanhf(sqrtf(2/M_PI) * (x + 0.044715f*x*x*x)))`, elementwise
kernels with `x128` loads (`load128cs`), `gelu_backward_inplace_kernel` writes
the gradient over its input. Kept as standalone kernels only when cuBLASLt
epilogue fusion is off (§2.4) or recomputation is on (§2.10).

### 2.9 AdamW, master weights, stochastic rounding

`llmc/adamw.cuh` `adamw_kernel3` (block 512, 2-D grid: x over parameters, y
over `num_slices` with per-slice strides — one launch covers a whole parameter
group, including ZeRO shards):

- moments via lerp: `m = lerp(grad, m, beta1)`, `v = lerp(grad*grad, v, beta2)`
  (fewer rounding steps and FMA-friendly vs the textbook form), bias correction
  `m /= (1 - beta1^t)`, decoupled weight decay
  `param -= lr * (m/(sqrt(v)+eps) + weight_decay*param)`;
- **master weights**: `float old_param = master ? master[idx] :
  (float)params[idx]`; the update is computed in fp32, written back to the
  fp32 master, and *stochastically rounded* into the low-precision working
  copy: `stochastic_rounding(param, &params_memory[idx], seed)`;
- `init_from_master` regenerates the working copy after checkpoint load, with
  the saved RNG state (`rng_state_last_update` in `train_gpt2.cu`) so the
  low-precision weights are **bit-identical** to the pre-checkpoint state.

`stochastic_rounding` (`llmc/cuda_utils.cuh`): per-thread random from
SquirrelNoise5 (`Get2dNoiseUint(threadIdx.x, blockIdx.x*blockDim.x+blockIdx.y,
seed)`), compares the float's low 16 mantissa bits against the random
threshold, then rounds to BF16 — i.e. a counter-based, stateless RNG keyed on
(position, step-seed), chosen precisely so rounding is deterministic and
reproducible. FP16 variant is a `todo`; FP32 is passthrough.

Optimizer memory: `m`/`v` are fp32 (`gpt2_allocate_state`, with
`cudaMallocConditionallyManaged` falling back to managed memory when the device
is full); master weights are optional (a flag), acknowledging the memory cost.

### 2.10 Memory management and activation recomputation

`train_gpt2.cu` / `train_gpt2.c`: `malloc_and_point_parameters()` computes the
total size of all 16 parameter tensors and does **one** `cudaMalloc`/`malloc`,
then points per-tensor pointers into the slab; same for activations
(`NUM_ACTIVATION_TENSORS = 21` on GPU, 23 on CPU) and grads. Benefits llm.c
actually exploits: guaranteed 16-byte alignment for `x128`, single-pointer
optimizer/norm kernels sweeping *all* parameters in one launch (§2.9, §2.11),
trivial checkpoint I/O (one contiguous write), zero allocator overhead in the
step loop.

Activation recomputation is a two-level `recompute` flag:
`recompute >= 1` drops GELU outputs and re-runs `gelu_forward` inside the
backward pass ("saves ~25% activation memory" for GPT-2); `recompute >= 2`
also drops per-layer layernorm outputs and recomputes them on demand (ln
buffers allocated `0` and a scratch reused). Trades a cheap elementwise/
normalization recompute for the largest activation buffers.

### 2.11 Training loop

`train_gpt2.cu` main loop:

- **Gradient accumulation**: `for (micro_step = 0; micro_step <
  grad_accum_steps; micro_step++)`; the loss gradient is pre-scaled `dloss =
  1.0f / (B*T*grad_accum_steps)` inside `fused_classifier`; gradients zeroed
  only on the first micro-step ("we're about to += accumulate into them").
- **LR schedules** (`llmc/schedulers.h`): cosine / linear / constant / WSD
  ("warmup-stable-decay", final 20% decays as `1 - sqrt(ratio)`,
  arXiv:2405.18392), all with linear warmup `lr * (step+1)/warmup_iterations`.
- **Gradient clipping**: `gpt2_calculate_grad_norm()` →
  `llmc/global_norm.cuh`: grid-stride per-block partial sums of squares,
  written to per-block slots (`out[blockIdx.y*gridDim.x + blockIdx.x]`,
  "we want to avoid using atomic add here"), then
  `global_norm_aggregate_kernel` sums the ≤1024 partials in a single block.
  The result feeds `grad_scale = grad_clip / grad_norm` in the AdamW launch.
- **Loss aggregation**: `global_sum_deterministic()` — a single-block kernel
  (`assert(gridDim.x == 1)`) so the summation order is fixed.
- **Outlier defense** (`llmc/outlier_detector.h`): z-score of loss and grad
  norm against a sliding window; `-sl`/`-sg` thresholds make `gpt2_update`
  *skip the update* on outlier steps.
- Single `main_stream` for all kernels; no intra-model multi-stream
  parallelism (overlap exists only for NCCL, below).

### 2.12 Multi-GPU: NCCL + ZeRO-1

`llmc/zero.cuh`: `MultiGpuConfig` (rank, `nccl_comm`, dedicated `nccl_stream`,
`compute_nccl_sync` event); rendezvous via MPI, TCP sockets, or a shared
filesystem file. `multi_gpu_async_reduce_gradient`: ZeRO stage 0 =
`ncclAllReduce`; stage 1 = `ncclReduceScatter` so each rank holds only its
gradient shard; stages 2–3 explicitly unsupported. Compute/communication
overlap via `cudaEventRecord(compute_stream)` +
`cudaStreamWaitEvent(nccl_stream)`, and the reduce is kicked off **per
transformer block inside `gpt2_backward_and_reduce()`** — communication of
layer ℓ's gradients overlaps the backward of layer ℓ-1. After the sharded
`adamw_update`, parameters are re-gathered with `ncclAllGather`
(`train_gpt2.cu`). Optimizer state (`m`, `v`, master) is allocated
shard-sized — that is the actual ZeRO-1 memory win.

### 2.13 Determinism discipline

A design invariant, not an afterthought: no data atomics anywhere (bias
backward, layernorm backward, and global norm all use two-phase buffer
reductions; the only atomic is a completion *flag*); loss sums are single-block
(`global_sum_deterministic`); stochastic rounding seeds are "deterministic and
unique for each parameter" (`llmc/encoder.cuh`: `seed + bucket*WARP_SIZE +
threadIdx.x + k`). The crown jewel is the **encoder backward**
(`llmc/encoder.cuh`): instead of `atomicAdd` scatter into `d_wte`, work items
are bucketed on CPU by `(c_group, token_id)`, buckets sorted by size descending
for load balance, and `wte_backward_kernel` gives each bucket to one warp which
accumulates *all* occurrences of that token in fp32 and writes once (with
stochastic rounding). `wpe_backward_kernel` similarly owns each (t, c) slot.
Deterministic gradient scatter costs a sort — llm.c pays it.

### 2.14 The CPU reference

`train_gpt2.c`: every op is a plain nested loop (attention forward is four
passes under `#pragma omp parallel for collapse(3)`); matmul has a naive
version kept for readability plus a mildly unrolled OpenMP one; AdamW is a
15-line loop; backward is hand-derived per op in reverse order. Its purpose is
explicit: readable, portable, and the *correctness oracle* — `test_gpt2.c`
checks against PyTorch-exported reference tensors, and every dev/cuda kernel
checks against the CPU implementation before being timed.

---

## 3. Technique → OCANNL seam map

Verdict key: **(a)** already covered (mechanism landed or an open task owns
it); **(b)** follow-up-issue candidate (draft in §7); **(c)** not applicable
to OCANNL's compiler architecture; **(d)** future — real, but gated on a named
prerequisite/milestone.

| # | llm.c technique (evidence) | OCANNL seam | Verdict |
|---|---|---|---|
| 1 | Grid/block geometry + occupancy tuning (§2.5, launch configs; dev/cuda block-size sweeps) | `Schedule.default_gpu` annotator, autotune `seed_block_sizes` sweep, `check_hardware_limits` | **(a)** |
| 2 | Warp-shuffle reductions `__shfl_xor_sync` + two-phase `blockReduce` (§2.2) | `Workgroup_reduce` axes render via shared memory + barriers only; no shuffle builtins | **(b)** — issue B1 |
| 3 | Warp-per-row mapping for row normalizations (§2.5, §2.6) | Expressible as `Split`+`Retype Workgroup(_reduce)`; reachable by autotune menu, not by the default annotator's heuristic | **(a)**, with B1 making it *worth* reaching |
| 4 | Packed128 128-bit loads/stores + cache hints (§2.3) | `Vectorized` axis type renders only on cc (`vector_bytes = 0` elsewhere); pool allocator already guarantees 32-byte alignment | **(b)** — issue B2 |
| 5 | cuBLASLt matmul + heuristic algo selection (§2.4) | OCANNL generates kernels; `Tensorize` + autotune matmul sketches are the in-tree answer | **(c)** (escape-hatch option noted §5) |
| 6 | Epilogue fusion: bias/GELU into matmul (§2.4) | Virtual-node inlining fuses elementwise producers/consumers into the matmul loop nest by default; fission only cuts at materialized cross-nest edges | **(a)** |
| 7 | TF32 / tensor-core precision modes (§2.4) | `Tensorize` precision choices; Metal `simdgroup_matrix` is uniform-precision, CUDA wmma pending | **(d)** — CUDA Tensorize bring-up |
| 8 | Online softmax, single pass max+sum (§2.6) | Needs two accumulators in one loop; lowering gives each reduction its own loop (`Assignments` → `Low_level`); gh-134 shared-loop virtualization is the seam | **(d)** — gh-134 / schedule-level loop sharing |
| 9 | Fused classifier: softmax+CE+dlogits, probs never materialized (§2.6) | `docs/in-progress/cross-entropy-loss.md` (frontend helper, no GH issue yet); virtualization can already elide probs if the graph is written well; needs a fission/inlining guarantee + backward fusion | **(b)** — issue B3 |
| 10 | Fused residual+layernorm (§2.5) | Residual adds inline into consumers when virtual; mean/var remain two loops (same limit as #8) | **(a)** for the add; the reduction pair is #8 **(d)** |
| 11 | LayerNorm-backward scratch+flag cross-block reduction (§2.5) | OCANNL never emits data atomics; cross-workgroup reductions become separate fission segments (same effect, cleaner semantics) | **(c)** |
| 12 | GELU (tanh approx) (§2.8) | Absent from tree; explicitly scoped as GPT-2 prerequisite in `docs/proposals/gh-ocannl-377.md` Phase 1 | **(a)** — tracked by #377 |
| 13 | AdamW (lerp moments, decoupled decay) (§2.9) | `lib/train.ml` has only `sgd_one`; AdamW is the head of the open optimizer task chain; `%cd` inline-declaration style of `sgd_one` extends directly | **(a)** — open task; design notes §4 |
| 14 | FP32 master weights + `init_from_master` + saved RNG state (§2.9) | Mixed-precision-optimizer-states task (open chain); per-tensor precision plumbing exists (`Ops`, tensor-level prec args); persistence has the checkpoint side | **(a)** — open task; design notes §4 |
| 15 | Stochastic rounding via counter-based RNG (§2.9) | Stochastic-rounding task (open chain); OCANNL's Threefry4x32 (`Ops.Threefry4x32_light`, `uniform_at` counter-based API) is a *better* fit than SquirrelNoise5; needs a rounding primitive in `ops.ml` + backend builtins | **(a)** — open task; design notes §4 |
| 16 | Single contiguous param/grad/optimizer allocation (§2.10) | Pool allocator #344 landed: `buffer_loc {pool_id; offset}`, bump-packed context deltas, 32-byte aligned offsets | **(a)** |
| 17 | Activation recomputation (`recompute` levels) (§2.10) | `Virtual` mode recomputes-by-inlining only what was never materialized; no user-facing "drop and recompute in backward" for materialized activations | **(d)** — GPT-2 *training* arc |
| 18 | Gradient accumulation micro-batching (§2.11) | `%cd` accumulation (`=+`) and `grad_update` make it expressible; no recipe/utility; autotune caveat already documents non-idempotent routines | **(b)** — issue B4 |
| 19 | LR schedules (cosine/linear/WSD + warmup) (§2.11) | Nothing in `lib/train.ml`; pure-OCaml host-side functions + bindings | **(b)** — issue B4 |
| 20 | Global-norm gradient clipping, non-atomic two-phase (§2.11) | Expressible today as an einsum sum-of-squares per param + scalar scale in `%cd`; deterministic by construction (no atomics emitted) | **(b)** — issue B4 |
| 21 | Outlier detector + skip-update (§2.11) | Host-side utility; pairs with #122/#103 experiment tracking | **(b)** — folded into B4 (stretch) |
| 22 | NCCL all-reduce; per-layer comm/compute overlap (§2.12) | Merge-buffer all-reduce data parallel landed (#293, `lib/parallel.ml`, `test/training/data_parallel.ml`); streams+events exist per backend; per-layer overlap would need fission-segment-granularity transfers | **(a)** core; overlap **(d)** |
| 23 | ZeRO-1 sharded optimizer states (§2.12) | No sharded-optimizer concept; sensible only after AdamW + multi-device training matures (DisTrO #278 adjacency) | **(d)** |
| 24 | Determinism-by-design: no data atomics, single-block sums, per-param seeds (§2.13) | OCANNL codegen emits no atomics at all (verified: no atomic emission in `c_syntax.ml`/backends); #341 resolved scheduler nondeterminism; counter-based RNG is position-keyed | **(a)** — keep as an explicit invariant |
| 25 | Bucketed deterministic embedding backward (§2.13) | gh-343 rewrote the *forward* one-hot→gather; gh-466 landed phase 1 of the backward: the transposed one-hot reduction rewrites to a guarded `Set_dynamic` scatter-accumulate over positions — O(B·T·C), serial (no atomics; schedule analyses treat dynamic writes as never-parallelizable). llm.c-style host-side bucketing for GPU position-parallelism remains phase 2 | **(b)** — issue B5, phase 1 landed |
| 26 | cuDNN flash attention + built-graph caching (§2.7) | Library call: architecture says generate, not call (see #5); the *caching* lesson is already embodied (`schedule_cache.ml`, autotune disk cache); flash-attention-shaped kernels via schedule search: see `docs/research/lean-attention-feasibility.md` | **(c)** / caching **(a)** |
| 27 | CPU reference as oracle; kernel-progression methodology (§2.14) | cc backend serves as reference; backend-parity `.expected` tests; autotune automates the progression loop | **(a)** |
| 28 | Root-folder simplicity governance (§2.1) | Matches existing practice (complexity quarantined in `arrayjit/lib` internals, recipes stay small in `lib/`) | **(a)** — cultural, no action |

**Counts:** (a) already covered / owned by an open task: **13** · (b)
follow-up-issue candidates: **8 techniques → 5 draft issues** · (c) not
applicable: **3** · (d) future with named prerequisite: **5**. (#10 and #26
are split verdicts, counted by their primary half.)

---

## 4. Notes for the open optimizer task chain (verdict (a) items with design content)

These need no new issues, but the llm.c specifics should transfer into the
open tasks' proposals:

1. **AdamW** (`lib/train.ml`, alongside `sgd_one`): use the lerp formulation
   (`m = lerp(g, m, β1)`) — it is one FMA per moment and matches llm.c's
   numerics; apply bias correction as a scalar host-side factor per step
   (`1-β^t` via a static-index binding or a host-computed constant tensor)
   rather than `powf` per element; keep weight decay decoupled
   (`p =- lr *. (m_hat /. (sqrt v_hat + eps) + wd *. p)` in `%cd`). Moments are
   per-parameter tensors declared with `{ m } `/`{ v }` inline `%cd`
   declarations exactly like `sgd_momentum` today (`sgd_one` in `lib/train.ml`).
   llm.c keeps m/v in **fp32 even when params are BF16** — the chain's
   mixed-precision step should preserve that default and make lower-precision
   moments the *quantized-AdamW* step, not the baseline.
2. **Master weights**: llm.c's key subtlety is checkpoint semantics —
   `rng_state_last_update` is saved so `init_from_master` reproduces the
   low-precision working weights bit-identically after restore. OCANNL analog:
   persist the master (fp32) tensors plus the rounding-step counter in the
   checkpoint namespace (`lib/persistence.mli` already namespaces entries);
   regenerate working copies on load. Placement note: master↔working is a
   *pair of tnodes with a cast-copy routine*, which the context/pool machinery
   supports today; no IR extension needed.
3. **Stochastic rounding**: implement as an `Ops` binop/ternop
   (value, random-bits → rounded value) with per-backend builtins
   (`builtins.c`, `builtins_cuda.ml`, `builtins_metal.ml`), driven by
   Threefry4x32 counters keyed on (tensor uid, flat index, step) — this is
   *stronger* than llm.c's SquirrelNoise5 seed discipline and reuses the
   existing `uniform_at` counter plumbing (`lib/nn_blocks.ml` `normal_at`
   pattern). llm.c's BF16 mantissa-threshold trick (compare low 16 bits
   against 16 random bits) is the right primitive shape. Do **not** route this
   through gh-ocannl-137 (closed not-planned); the optimizer chain owns it.
4. **Gradient clipping interaction**: llm.c folds `grad_scale` into the AdamW
   launch rather than scaling gradients in place — one fewer full sweep of
   grads. The OCANNL AdamW `%cd` recipe should take an optional scalar
   `grad_scale` tensor for the same reason (B4's clipping composes with it).

---

## 5. Architectural observations

1. **Fusion-up vs fission-down converge on the same segmentation.** llm.c
   starts from per-op kernels and fuses adjacent cheap ops (residual+LN,
   matmul epilogues, classifier); OCANNL starts from a whole-program routine
   and fissions at materialized cross-nest edges. For a GPT-2 block both
   should converge to roughly: [QKV matmul] [attention] [proj matmul]
   [residual+LN] [FFN matmul+GELU] [FFN matmul] [residual+LN] — llm.c's
   final kernel list is therefore a **reference segmentation to validate
   OCANNL's fission output against** on the GPT-2 driver. Where OCANNL's
   fission produces strictly more segments than llm.c's kernel list, that gap
   is a concrete, measurable to-do (either an inlining miss or a missing
   aligned-cross-nest merge).
2. **The one pattern OCANNL's lowering cannot express today is
   multi-accumulator loops**: online softmax (max and rescaled sum together),
   layernorm's mean+var pair, and the fused classifier's loss+dlogits pass all
   want ≥2 reductions sharing one traversal. OCANNL lowers each assignment to
   its own loop nest (`docs/proposals/gh-ocannl-134.md` is the shared-loop
   seam; memory: high-level never shares for-loops). This is the same gap
   Mirage's for-loop `Accum`s highlight (`docs/research/superoptimizers.md`
   §4.3). It is *the* structural prerequisite for llm.c-class normalization
   and classifier kernels — worth stating as the headline "challenges our
   assumptions" finding of this study.
3. **Library escape hatch.** llm.c hand-writes *nothing* matmul-shaped and
   still matches PyTorch — all throughput-critical GEMMs are cuBLASLt. OCANNL
   deliberately generates its kernels; the risk this study confirms is that
   generated matmuls must reach a usable fraction of vendor-library throughput
   before the GPT-2 driver is credible. The mitigations already in flight are
   `Tensorize` (MMA units) and autotune matmul sketches; if those stall, a
   per-primitive library-call fallback (cuBLAS/MPSMatrix behind a
   `Backend_intf` hook, chosen when a fission segment is exactly a matmul) is
   the pragmatic fallback — noted here, not proposed.
4. **Determinism is an asset OCANNL gets architecturally.** llm.c spends
   real engineering (bucketed encoder backward, two-phase reductions,
   single-block sums, seed discipline) to avoid atomics; OCANNL's codegen
   simply has no atomics, its reductions are loop-carried or fission-staged,
   and its RNG is counter-based. This should be promoted from accident to
   documented invariant — cheap now, valuable when parallel scatter (B5)
   tempts an atomicAdd shortcut.
5. **BF16 differs from llm.c's assumptions on the OCANNL side in one spot**:
   `Ops.Bfloat16` is currently represented as uint16 ("Using uint16
   representation for now", the `Bfloat16` constructor of `Ops.precision`) — the
   optimizer-chain mixed-precision step lands on top of whatever conversion builtins exist per
   backend, so the BF16 story (and its `check_half_prec_constants_cutoff`
   analog) needs an audit before master-weights work starts.
6. **Occupancy-driven grid sizing** (llm.c backward LN: grid =
   `blocks_per_sm * multiProcessorCount`) is a schedule decision keyed on a
   hardware attribute OCANNL already exposes (`hardware_limits`); the autotune
   search space covers block sizes but not SM-count-derived grid caps. Cheap
   menu extension when a workload shows it matters; not filed as an issue.

---

## 6. Prioritized shortlist for the GPT-2 driver workload

1. **Fused classifier pattern (B3).** Largest single-buffer win in GPT-2
   training and the loss-side win for inference-with-evaluation: never
   materialize [B,T,V] probabilities. Frontend `cross_entropy_loss` helper +
   a test that fission keeps softmax/loss/dlogits in one segment.
2. **AdamW with fp32 master-weight seam (open task chain).** Blocks every
   training milestone; llm.c gives the exact numerics (lerp moments, decoupled
   decay, grad-scale folded into the update). GPT-2 *training* parity is
   unreachable without it; §4 transfers the design.
3. **Warp-shuffle `Workgroup_reduce` rendering (B1).** Every normalization,
   softmax, loss, and norm kernel in GPT-2 is reduction-bound; shuffle
   intrinsics halve the shared-memory traffic and barrier count of the current
   staging scheme. Small, backend-local, immediately visible in attention
   softmax timings.
4. **GPU vectorized loads (B2).** llm.c treats `x128` as table stakes for
   bandwidth-bound kernels (GELU, residual, LN, classifier are all
   bandwidth-bound). The pool allocator's 32-byte alignment already satisfies
   the precondition; cc's `Vectorized` rendering provides the IR pattern.
5. **Bucketed embedding backward (B5).** With V=50257, a dense one-hot
   reduction for `d_wte` dwarfs the rest of the backward pass; llm.c's
   CPU-bucketing shows a deterministic, atomics-free shape for the fix, and
   the gh-343 forward rewrite already built the analysis half (one-hot
   selector detection).

(Online softmax / multi-accumulator loops — observation §5.2 — outranks some
of these in eventual impact but is gated on gh-134-class IR work; it is the
recommended *next deep-dive*, not a v0.8 action.)

---

## 7. Draft follow-up issues

### B1 — Render `Workgroup_reduce` reductions with warp/simdgroup shuffles

Currently `Workgroup_reduce` axes communicate exclusively through
workgroup-shared staging nodes plus `Workgroup_barrier`s
(`arrayjit/lib/low_level.mli`). llm.c's universal reduction idiom
(`warpReduceSum` via `__shfl_xor_sync`, then one shared slot per warp —
`llmc/cuda_utils.cuh`) halves shared-memory traffic and barriers for the
row-sized reductions GPT-2 is full of (softmax, layernorm, losses, global
norm). Add shuffle builtins (`builtins_cuda.ml`: `__shfl_xor_sync`;
`builtins_metal.ml`: `simd_shuffle_xor`) and teach the `c_syntax.ml`
reduction rendering to emit the two-phase pattern when the reduce extent
covers whole warps; cc keeps the current rendering. Backend-local; no IR
change.

### B2 — 128-bit vectorized loads/stores on GPU backends

`Retype … Vectorized` only renders as vector code on cc
(`cc_backend.ml` `cc_vector_bytes`; `c_syntax.ml` defaults `vector_bytes = 0`).
llm.c's `Packed128` (`llmc/cuda_utils.cuh`) shows 128-bit loads are the
baseline for bandwidth-bound GPU kernels, with cache hints (`__ldcs`/`__stcs`)
for streaming data. The pool allocator already pads offsets to 32 bytes
(`Ops.buffer_alignment`, `backends.ml`), so alignment preconditions hold for
pool-resident tnodes. Override `vector_bytes` in the CUDA/Metal codegen
(CUDA `float4`/vector types, MSL `float4`), guard on extent divisibility the
same way cc does, and expose it to the autotune Retype-Vectorized menu action
(currently CPU-only, `autotune.mli`). Streaming cache hints are a stretch goal.

### B3 — Fused cross-entropy classifier: loss and dlogits without materializing probabilities

Adopt llm.c's `fused_classifier` contract (`llmc/fused_classifier.cuh`) at the
OCANNL level: a `Nn_blocks.cross_entropy_loss` helper (absorbing
`docs/in-progress/cross-entropy-loss.md`, which has no GH issue yet) written
so that (i) the softmax-probabilities intermediate stays `Virtual`, (ii) the
backward produces `p - onehot(target)` directly against logits, and (iii)
kernel fission keeps max-reduce, sum-reduce, loss, and dlogits in as few
segments as the multi-accumulator limitation allows — with a
`test/training/` check on the segment count and on numerical stability
(log-sum-exp, not `log(softmax)` as `transformer_with_loss` does today). For
GPT-2 (V=50257) this is the largest activation eliminated from the model.

### B4 — Training-loop utilities: LR schedules, global-norm clipping, gradient accumulation

`lib/train.ml` stops at `sgd_update`/`sequential_loop`. Port llm.c's loop
scaffolding (`train_gpt2.cu`, `llmc/schedulers.h`, `llmc/global_norm.cuh`):
(1) host-side LR schedulers — cosine, linear, constant, WSD with linear
warmup — feeding the existing learning-rate binding; (2) global-norm gradient
clipping as a `%cd` sum-of-squares reduction over `t.params` plus a scalar
`grad_scale` consumed by the optimizer step (fold the scale into the update,
llm.c-style, instead of rescaling gradient buffers); (3) a gradient-accumulation
recipe: zero grads on micro-step 0, `=+` accumulation, loss pre-scaled by
`1/(B*T*accum_steps)`; (4) stretch: a z-score loss/grad-norm outlier detector
with skip-update, pairing with #122/#103 experiment tracking. Everything is
expressible with current IR; this is recipes + tests, sized for the GPT-2
training example.

### B5 — Deterministic bucketed embedding-table gradients

gh-343 rewrote the *forward* one-hot contraction into a guarded gather
(`Low_level.rewrite_one_hot_reductions` / `Get_dynamic`), but the embedding
*backward* (`d_wte`) still lowers as a dense one-hot reduction — O(V·B·T)
work; at GPT-2 scale (V=50257) it dominates backward cost. llm.c solves the
scatter deterministically without atomics by bucketing work items per
(token, channel-group) on the host and giving each bucket to one warp
(`llmc/encoder.cuh`, `wte_backward_kernel`). OCANNL seam: recognize the
transposed one-hot pattern in `rewrite_one_hot_reductions` (the detection
metadata `prefers_virtual_one_hot`/`has_non_one_hot_setter` already exists in
`low_level.mli`) and lower to an index-driven accumulation loop over
positions, keeping the no-atomics invariant — a host-side bucket/sort step à
la llm.c is an acceptable phase-2 if the per-row loop is insufficient.

---

## 8. Draft GitHub comment for issue #253

> The llm.c study is done: evidence base + verdicts in
> `docs/research/llmc-lessons.md` (read at llm.c `master` `f1e2ace`, its final
> commit — the project is now effectively frozen, so this analysis won't rot).
>
> Headline: the proposal's premise aged well but its gap list didn't — most of
> the *infrastructure* gaps it listed have since landed in OCANNL (schedule IR
> with Grid/Workgroup axes and default GPU annotator, kernel fission,
> autotune with matmul sketches, universal pool allocator, hardware-limit
> checks), so the study's value shifted from "what to build" to "which llm.c
> techniques are still missing and where they attach".
>
> Verdicts over 28 extracted techniques: **13 already covered** (or owned by
> an open task: GELU → #377, AdamW/master-weights/stochastic-rounding → the
> open optimizer chain), **8 follow-up-worthy** (5 draft issues: warp-shuffle
> `Workgroup_reduce` rendering; 128-bit vectorized loads on GPU backends;
> fused cross-entropy classifier; training-loop utilities — LR schedules,
> global-norm clipping, grad accumulation; bucketed embedding backward),
> **3 not applicable** (cuBLASLt calls, cuDNN flash-attention integration,
> atomics-based reduction protocols — our codegen has no atomics at all),
> **5 future** (TF32/tensor-core precision after CUDA Tensorize, online
> softmax and other multi-accumulator loops after #134-class loop sharing,
> activation recomputation, ZeRO-1, per-layer comm/compute overlap).
>
> Structural finding: llm.c fuses upward from per-op kernels, we fission
> downward from a whole program — its GPT-2 kernel list is a reference
> segmentation to validate our fission output against. The one llm.c pattern
> our IR cannot express today is multi-accumulator loops (online softmax,
> mean+var, loss+dlogits in one pass); that's the recommended next deep dive.
>
> Top-5 for the GPT-2 driver: fused classifier, AdamW (+fp32 master weights),
> warp-shuffle reductions, GPU vectorized loads, bucketed embedding backward.
> Filing the five follow-ups next; then I'd close this issue.

---

## 9. Cross-references

- `docs/proposals/gh-ocannl-253.md` — the task proposal this note discharges.
- `docs/proposals/gh-ocannl-412.md` (tiling), `gh-ocannl-164.md` (SIMD/AVX),
  `docs/research/megakernel-deep-dive.md` (#318),
  `docs/proposals/schedule-ir-optops.md` — the landed/ongoing performance arcs
  the proposal predicted; verdicts above assume their current state, not their
  proposal-time state.
- `docs/proposals/gh-ocannl-377.md` — GPT-2 inference pipeline (driver
  workload; owns GELU and weight conversion).
- `docs/in-progress/cross-entropy-loss.md` — absorbed by draft issue B3.
- `docs/research/lean-attention-feasibility.md` — flash/lean attention as
  schedule search, the (c)-verdict counterpart of llm.c's cuDNN path.
- `docs/research/superoptimizers.md` — §4.3 multi-accumulator observation
  aligns with §5.2 here; `docs/research/tinygrad-deep-dive.md` — the autotune
  lineage that already automates llm.c's dev/cuda methodology.
- `docs/imbue-infrastructure-lessons.md` (#270) — cluster/ops-side lessons;
  that memo defers all kernel/training-loop material to this note.
