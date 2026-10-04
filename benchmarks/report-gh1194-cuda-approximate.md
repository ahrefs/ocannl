# CUDA approximate-profile transformer regression (gh-ocannl-1194)

The approximate profile's automatic online-softmax block fold made tuned f32 `gpt2_mini`
slower on the RTX 5070 Ti Laptop GPU. Keeping the two-pass online-softmax rewrite preserves the
tf32 matmul gain: the block-off artifact replayed at **2.442 ms**, versus **3.505 ms exact** and
**4.476 ms approximate** in the same exclusive window. Both placement arms were timed under the
original approximate profile; arm B correctly beat arm A (4.426 versus 4.487 ms). The placement
comparison did not omit the faster exact arm: the two profiles lower different programs.

## Protocol and provenance

Exclusive ROG window `w1004-1194-rog-nv-linux-ablation-1`, source
`a723dc5b23a815c670bb4047f0b023c6cbd3b14f` (the original regression's source), native CUDA on an
RTX 5070 Ti Laptop GPU. The runner was rebuilt at the pinned revision. Its old untracked
`benchmarks/gh1181_cells.py` was excluded from the experiment's inputs; the driver imported the
tracked `orchestrate.py` and `gh675_cells.py` helpers.

Each treatment uses its own empty schedule cache, one search process and three fresh replay
processes. Reported p50 is the median of those three processes' step p50s, **one search per
row**, rather than three independent searches. All timing rows must replay, pass parity against
one exact PyTorch CPU reference, run on CUDA, and have finite timings. A searched replay is
rejected. `orchestrate.run_cell` bounds each process to 1,200 seconds and proves its child group
reaped; the driver has a 7,500-second cap. Processes use CPUs 0-15 and a treatment-free inherited
environment from `gh675_cells.base_env`.

Fixture content digest `c322a00b72df143612eafafedf7c468e9c05f3e4bc1e7eb6d77ada651ef98de8`
(the declared m4-max/tuf bytes), 13,871,384 bytes. The runner SHA256, raw fixture digest,
explicit flags and source SHA are recorded in `env.json`. Raw results, cell logs, caches and
summary records are at `/home/lukstafi/.local/state/gh1194/1194-ablation-1` on ROG; the bounded
batch log is `../1194-ablation-1-run/run-1/log`. The driver is preserved alongside that run as
`../1194-ablate.py`.

Raw fixture SHA256: `89702197c593596901865aceffa2c99cfd9640d046de04b9cb99c95c5dd5d49f`.
Rebuilt runner SHA256: `7af61c40c741238a69c26a089a91a60106ec5579ce17c1632edb017ba21e9026`.

## Completed ablations

| Treatment | Explicit flags beyond CUDA/progress | Replay median p50 ms | Ships | Emitted MMA statements | Parity/provenance |
| --- | --- | ---: | --- | ---: | --- |
| Exact | none | 3.505 | A | 0 | PASS / REPLAY |
| Approximate, block off | `profile=approximate`, `online_softmax_block=0` | 2.442 | A | 29 | PASS / REPLAY |
| Approximate, tf32 off | `profile=approximate`, `tf32_matmuls=false` | 5.583 | A | 0 | PASS / REPLAY |
| Approximate | `profile=approximate` | 4.476 | B | 25 | PASS / REPLAY |
| Exact plus tf32 only | `tf32_matmuls=true` | 2.388 | A | 33 | PASS / REPLAY |

The original approximate row's 4.493 ms is the six-replay result in
[the tagline report](report-tagline-gpt2.md#rog-nv-linux-cuda-rtx-5070-ti-laptop-12-gib-discrete),
from a different exclusive window on the same source and fixture. The fresh control reproduced
the regression: A measured 4.49207 ms and B 4.39608 ms, both searched and with zero timing
contention or unbatched refusals. The tf32-only control confirms that matmul tensorization
supplies the gain. The final online-softmax-off control and patched-head confirmation are
tracked in [gh-ocannl-1194 and its linked PR](https://github.com/ahrefs/ocannl/issues/1194).

The block-off search measured A at 2.41352 ms and B at 2.44704 ms, with no contention or unbatched
refusals. Its shipped artifact emitted 29 tensorized statements and zero scalar fallbacks;
removing tf32 instead left the fold in place and produced the slower scalar result. The original
approximate artifact emitted 25 tensorized statements. Tensorization here describes the actual
shipped artifact's census, not a candidate label.

Inference is f32: the fp16/bf16 arithmetic settings do not affect its storage or accumulators;
the backward gate has no backward to rewrite; the C compiler settings apply to the CPU backend.
`bench_gpt` explicitly uses `rounds:0`, so the profile's round count is not this run's search
budget. The original two inline refinements were abandoned and did not change what shipped.
The distinct numerics cache keys separate exact and approximate; gh-ocannl-1153 concerns later
changes to a mode's arm resolution, rather than these profiles aliasing today.

## Resolution

CUDA's device limits resolve `online_softmax_block=auto` to 0, selecting the two-pass rewrite.
The tf32 policy and explicit block sizes remain available. Metal and CPU retain their measured
block-16 automatic policy; HIP already selected the two-pass form. CUDA devices without a
profitable-fold measurement use the conservative default too.

The resolved program and the full hardware-limits record enter schedule-cache identity. This
policy change therefore invalidates existing CUDA schedule entries, including exact paths whose
lowered program is unchanged; their next tuned invocation searches again.

The policy is a default for CUDA devices, not a claim that every shape loses under the fold.
This experiment covers the f32 inference endpoint at this fixture's dimensions. Explicit
integer block sizes permit experiments with other shapes and devices. Executed attention
forward/gradient oracles and the automatic-policy lowering/compilation checks provide
correctness coverage; this report makes no training-throughput claim.

The measured fresh replay allocator high-water was 138,108,932 bytes exact and 121,327,620 bytes
for block-off approximate. The original approximate replays recorded 327,725,668 bytes. These
are allocator seam high-water readings for the full compiled artifact, not a measurement of the
attention's score buffer alone.
