# Imbue 70B Infrastructure — Lessons Memo

GitHub issue: https://github.com/ahrefs/ocannl/issues/270
Proposal: [docs/proposals/gh-ocannl-270.md](proposals/gh-ocannl-270.md)
Companion memo (llm.c lessons): [docs/proposals/gh-ocannl-253.md](proposals/gh-ocannl-253.md) (future `docs/llm-c-analysis.md`)
Status: disposition memo, milestone v1.1, scope `explore`.

## §1. Scope and what this memo is not

The GH issue links two resources: [Imbue's 70B infrastructure
guide](https://imbue.com/research/70b-infrastructure/) and Karpathy's
[llm.c GPT-2 1.6B training discussion](https://github.com/karpathy/llm.c/discussions/677).
This memo covers **only the Imbue portion**. The llm.c portion — kernel-level
techniques (warp shuffles, online softmax, `float4` loads), memory layout
(contiguous parameter buffers, in-place ops), and training-loop patterns
(AdamW, cosine schedule, gradient clipping) — overlaps almost completely with
[gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253) and lives in
`docs/proposals/gh-ocannl-253.md` and (when written) `docs/llm-c-analysis.md`.
Re-deriving those lessons here would split the discussion across two issues
and duplicate work; cross-links go in both directions.

The Imbue guide is overwhelmingly about *external* infrastructure — bare-metal
cluster bring-up, host-OS provisioning, InfiniBand fabric health monitoring,
fleet-level fault tolerance, evaluation harnesses — i.e. the layer *around*
a trainer, not the trainer itself. Two architectural facts about OCANNL
sharply constrain what transfers cleanly today:

1. **Multi-streaming was deliberately removed** in
   [gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341) ("Not
   planned", 2026-02-07). OCANNL has no in-process notion of multiple GPUs
   as parallel devices, no AllReduce / NCCL / MPI integration, and no
   cross-process coordination. Imbue's distributed-training lessons live one
   architectural layer above OCANNL today; they belong to a future external
   infrastructure layer alongside [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278)
   (DisTrO) and [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293)
   (sharding). Re-introducing multi-streaming is *not* a goal of this memo.
2. **CUDA kernels run single-threaded today** (`grid_dim=1, block_dim=1`,
   enforced by the `kernel_prep_line` guard in
   `arrayjit/lib/cuda_backend.ml`). The whole v0.8 GPU performance milestone
   begins from this baseline; any kernel-level lessons would feed v0.8 via
   #253 / megakernel work, not the present memo.

### §1.1. Workshop-paper relevance (AC5)

Imbue's infrastructure lessons do **not** materially feed the OCaml Workshop /
FProPer 2026 paper. The paper's angle is generalized einsum, row variables,
and shape inference — a type-level / language-design contribution. Imbue's
guide is a fleet-management ops report. The two have no thematic overlap;
citing Imbue in the paper would be scope inflation, not support, and is
explicitly out of scope here. (Same applies to llm.c-derived ops/perf
material in the paper bibliography.)

## §2. Imbue's guide at a glance

The Imbue guide ([imbue.com/research/70b-infrastructure](https://imbue.com/research/70b-infrastructure/),
June 2024) is a public end-to-end account of training a 70B parameter model
on a rented bare-metal H100 cluster. Its sections (paraphrased):

- **Cluster bring-up & host-OS provisioning** — kickstarting a fresh
  multi-node H100 cluster, image building, configuration management.
- **InfiniBand fabric setup, monitoring, and link-failure recovery** —
  topology, subnet manager, link-flap detection, per-link counters.
- **NCCL configuration** — collective comms tuning for the fabric above.
- **Pre-flight diagnostics** — burn-in tests, NCCL all-reduce sanity,
  per-GPU memory checks before launching a training run.
- **Checkpointing strategy** — frequency, sharding across nodes, async
  upload to object storage, restart cost.
- **Evaluation harness** — running model evals during/between training
  steps without dominating throughput.
- **Fault tolerance & automated recovery** — supervisor that detects
  hangs / NCCL aborts / GPU faults and respawns the trainer from the most
  recent checkpoint.
- **Post-mortems on real failures** — concrete examples of what broke.

### §2.1. Headline empirical claims (AC2, verbatim, "as of June 2024")

> The following figures are reproduced as Imbue stated them in June 2024.
> They are operational facts a future infra layer must design around;
> hardware reliability has continued to improve, so re-validate before
> using these numbers as a design budget.

- **~3% of machines break per week on the newest hardware.** (Imbue, "70B
  infrastructure" guide, fault-tolerance section.) The implication is that
  in a 256-machine fleet, you should expect ≈8 host failures per week,
  i.e. multiple per day on average.
- **Most training failures trace to faulty InfiniBand links or GPU
  hardware, not software.** (Same source, post-mortem and fault-tolerance
  sections.) The implication is that the trainer must treat its substrate
  as fallible and the fleet-management layer must treat link-flap and
  GPU-fault detection as first-class concerns.
- **Checkpointing and evaluation are the primary training slowdowns.**
  (Same source, checkpointing strategy and evaluation harness sections.)
  The implication is that any "checkpoint format" or "eval cadence"
  decision is load-bearing for end-to-end wall-clock training time.

These three facts together are the load-bearing argument for treating
training-in-the-large as a fleet-management problem, not an algorithms
problem.

## §3. Bucket A — Compiler / runtime concerns close to OCANNL today

These are the Imbue lessons whose implementation surface lands inside
OCANNL's current architectural scope (single-process, single-device tensor
compiler with C / CUDA / Metal backends). All three already have homes.

### A1. Pre-allocated, contiguous parameter buffers

**Imbue's claim.** Allocating all parameter / gradient / optimizer-state
memory in one contiguous buffer up front (rather than per-tensor) is what
makes NCCL collectives, deterministic execution, and checkpoint I/O
tractable at scale. (Imbue guide, checkpointing and NCCL sections; same
pattern is described independently in llm.c.)

**OCANNL status.** Tensor-node memory modes (`Virtual`, `Local`,
`On_device`, `Materialized`) and the universal pool allocator design in
[gh-ocannl-344](https://github.com/ahrefs/ocannl/issues/344) cover this for
single-device execution. Contiguous parameter blocks (in the NCCL sense of
"one contiguous range across all parameters") are not a goal at v0.8 single
device; they would re-emerge if and only if a future external-infrastructure
layer wires AllReduce on top.

**Routing.** Already covered for single-device — see
[gh-ocannl-344](https://github.com/ahrefs/ocannl/issues/344). The
multi-device variant is bucket-B (see §4).

### A2. Bitwise-deterministic execution

**Imbue's claim.** Bitwise-deterministic training (same inputs, same seeds
⇒ same loss curve down to the last bit) is what made debugging at 70B
tractable; without it, "is this divergence a bug or numerical noise?" has no
answer. (Imbue guide, debugging discussions; llm.c makes the same claim.)

**OCANNL status.** OCANNL has the
[gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341) thread on
multicore_cc non-determinism (closed "Not planned" for the multi-streaming
side, but the determinism argument remains live for kernel-level work).
Determinism is a known design goal for v0.8 kernel work via #253.

**Routing.** v0.8 input — included by reference in
[gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253) (see its
"reproducibility / determinism" recommendations) and
`docs/megakernel-deep-dive.md`. No new task needed: Imbue's contribution
here is *empirical reinforcement* of an already-recognized requirement, not
a new technique.

### A3. Checkpoint format efficiency

**Imbue's claim.** Checkpointing dominated wall-clock training time
alongside evaluation; reducing checkpoint frequency was not viable (it
controls the restart-cost upper bound), so the format and write path had
to be made cheap. (Imbue guide, checkpointing strategy section.)

**OCANNL status.** Tensor save/load shipped in
[gh-ocannl-373](https://github.com/ahrefs/ocannl/issues/373)
(`docs/proposals/tensor-persistence.md`). The current format is correct
but has not been tuned against any "checkpointing dominates" budget,
because OCANNL has no training run today against which to measure.

**Routing.** Already covered structurally by
[gh-ocannl-373](https://github.com/ahrefs/ocannl/issues/373); Imbue's
finding is *tuning input* for that subsystem when the first end-to-end
training run on OCANNL exists (post v0.7.x transformer milestone). No new
task needed now — re-open as a #373 follow-up if/when checkpoint cost
becomes measurable.

## §4. Bucket B — Future external-infrastructure layer (post-v1.0)

These lessons are real and load-bearing at scale, but their
implementation surface is *outside* OCANNL's current process boundary.
They belong to a future external-infrastructure layer (orchestrator,
fleet manager, fabric monitor) that is at minimum v1.1+, and is
gated on the v0.7.x → v0.8 chain (RoPE → transformer → tokenizers →
GPT-2 inference → GPU performance baseline) landing first. None require
new OCANNL tasks today; cross-references go to existing v1.1+ tasks.

### B1. Cluster bring-up & host-OS provisioning

The image-building / Kubernetes-or-Slurm / configuration-management layer.
Entirely orthogonal to the trainer. v1.1+, no OCANNL task.

### B2. InfiniBand fabric health & link-failure recovery

Topology mgmt, subnet-manager config, per-link counters, link-flap
detection. Lives in the fabric-monitoring side of a hypothetical OCANNL
infrastructure layer. v1.1+; relevant to
[gh-ocannl-278 (DisTrO)](https://github.com/ahrefs/ocannl/issues/278) and
[gh-ocannl-293 (sharding)](https://github.com/ahrefs/ocannl/issues/293)
only insofar as those tasks would eventually need to *trust* the fabric;
neither task currently designs that trust.

### B3. NCCL / collective-communications configuration

Knobs for the chosen collective library (NCCL on NVIDIA, RCCL on AMD).
This presupposes a multi-process / multi-device OCANNL runtime, which was
explicitly removed in
[gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341). Routes to
[gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) and
[gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293) when those
land.

### B4. Pre-flight diagnostics

Burn-in tests, NCCL all-reduce sanity, per-GPU memory checks before
launching a training run. Same architectural prerequisite as B3. v1.1+,
attaches to whichever task introduces the orchestrator.

### B5. Fault-tolerant supervisor / automated recovery

Process supervisor that detects hangs / NCCL aborts / GPU faults and
respawns the trainer from the most recent checkpoint. Requires a
process-supervision layer that does not exist in OCANNL today; the current
trainer is a single OCaml process with no out-of-process restart story.
v1.1+, sits above the orchestrator; relevant to
[gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) /
[gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293) only in the
sense that *some* supervisor is needed before either of those tasks could
run a real training job.

### B6. Fleet-level evaluation harness

Running evals across the fleet without dominating throughput. Presupposes
the same multi-process runtime as B3. The single-device eval cadence is a
much smaller problem and lives inside whatever training-loop design is
chosen for the v0.8 GPT-2 milestone (#253). v1.1+ for the fleet variant.

## §5. Bucket C — Not applicable to OCANNL's scope

Items genuinely orthogonal to anything OCANNL builds, in any plausible
milestone. Listed for completeness so future readers do not re-debate.

- **Hardware procurement / vendor selection.** Not a software concern.
- **Host-OS image construction (kickstart, PXE, ipmitool).** Cluster ops,
  not trainer.
- **RDMA driver troubleshooting at the kernel-module level.** Below the
  fabric-monitor layer; vendor-specific.
- **Datacenter physical layout / power / cooling.** Not in software scope.
- **Object-storage credentials & networking for checkpoint upload.** Once
  the checkpoint format (A3) is decided, the upload path is generic infra
  outside OCANNL.
- **Recruiting / team-building observations from the Imbue post.** Not a
  software concern.

## §6. Backlog items filed during this memo

None. Every bucket-A lesson maps to an existing task or proposal; every
bucket-B lesson is captured by the existing v1.1+ infrastructure-layer
tasks ([gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278),
[gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293)) or is
explicitly deferred until that layer exists; bucket-C lessons need no
issue. If a future reader of this memo finds an Imbue lesson that does not
fit any existing task, the right move is to file a new GH issue and link
it back here.

## §7. Routing summary (for AC3 traceability)

| Imbue lesson | Bucket | Disposition / route |
| --- | --- | --- |
| Pre-allocated contiguous param buffers | A | [gh-ocannl-344](https://github.com/ahrefs/ocannl/issues/344) (Universal Pool Allocator); multi-device variant deferred to bucket B |
| Bitwise-deterministic execution | A | v0.8 input via [gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253) and `docs/megakernel-deep-dive.md` |
| Checkpoint format efficiency | A | Tuning input for [gh-ocannl-373](https://github.com/ahrefs/ocannl/issues/373) (`tensor-persistence.md`) |
| Cluster bring-up & host-OS provisioning | B | v1.1+, no OCANNL task |
| InfiniBand fabric health | B | v1.1+, gates [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) / [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293) |
| NCCL / collectives configuration | B | v1.1+, [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) / [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293) |
| Pre-flight diagnostics | B | v1.1+, attaches to future orchestrator |
| Fault-tolerant supervisor | B | v1.1+, prerequisite for [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) / [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293) |
| Fleet-level eval harness | B | v1.1+; single-device eval cadence handled inside #253 |
| Hardware procurement / OS imaging / RDMA drivers / datacenter ops | C | Not in software scope |

## §8. Related tasks

- [gh-ocannl-253](https://github.com/ahrefs/ocannl/issues/253) — Karpathy
  llm.c study; canonical home for llm.c lessons (per §1).
- [gh-ocannl-278](https://github.com/ahrefs/ocannl/issues/278) — DisTrO
  distributed training; bucket-B routing target.
- [gh-ocannl-293](https://github.com/ahrefs/ocannl/issues/293) — sharding
  and slicing with minimal copying; bucket-B routing target.
- [gh-ocannl-341](https://github.com/ahrefs/ocannl/issues/341) —
  multi-streaming removal ("Not planned"); the architectural boundary
  that forces Imbue's distributed lessons into bucket B.
- [gh-ocannl-344](https://github.com/ahrefs/ocannl/issues/344) — Universal
  Pool Allocator; bucket-A routing target for pre-allocated buffers.
- [gh-ocannl-373](https://github.com/ahrefs/ocannl/issues/373) — tensor
  persistence (done); bucket-A routing target for checkpoint cost tuning.
