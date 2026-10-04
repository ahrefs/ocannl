# gfx1102 half-load corruption in exact GPT attention

The October 4, 2026 bounded investigation used source
`ade83a84c384f4a056288c718e97e013cee3d3ce`, then diagnostic observer
`df7180b7eb4c360214d428490b32d3a88e1f8ee9`, on the TUF Radeon RX 7700S
(`gfx1102`, discrete memory). The runtime reported `70152801` (7.1.52801),
HIPRTC reported 9.0, and the installed disassembler was LLVM 21.

The original s1024 artifact failures were **finite parity drift**: they did not
contain nonfinite losses. The separate base-training artifact had a nonfinite
step-zero loss. Six default/wide/materialized correctness cells on the current
source did not reproduce that nonfinite loss. Wider arithmetic did not resolve
the finite instability. No first-overflow claim follows from these observations.

Two fresh s1024 inference processes emitted byte-identical HIP source and code
objects but returned different losses. Materializing intermediates repeatedly
returned the same losses. Buffer aliasing was disabled, all configuration was
explicitly isolated from ambient files, and routine logging stayed disabled.
Disabling graph capture alone did not remove the finite instability.

The opt-in `BENCH_SNAPSHOT=1` observer found matching first-layer scores, row
maxima and denominators, followed by different first-layer softmax outputs.
All default output snapshots were finite. The materialized graph's intentional
masked `-inf` intermediates are not overflow. Softmax sums were around 11,500
instead of the 8,192 normalized rows.

The first softmax binary had this sequence (VGPR halves shown explicitly):

```text
global_load_d16_b16    v0.l, maximum_address
global_load_d16_hi_b16 v0.h, denominator_address
s_waitcnt vmcnt(1)
v_sub_f16             v0.l, score.l, v0.l
v_cvt_f32_f16          v1, v0.l
s_waitcnt vmcnt(0)
```

An isolated replay of the exact emitted kernel used its original
`grid=(4,8,1024)`, `block=(256,1,1)` and completely initialized buffers:
scores 0, maxima 1, denominators 2, and a triangular mask. Correct unmasked
outputs are `exp(-1)/2`, while masked outputs are exactly zero. The default
compiler returned `1.359375`, consistent with `exp(+1)/2`, in corrupted cells.

| Compiler control | Wrong cells in three launches | Nonzero masked cells |
| --- | --- | --- |
| Default | 128032 / 145504 / 142144 | 53716 / 58520 / 56474 |
| `-mllvm -amdgpu-waitcnt-load-forcezero` | 0 / 0 / 0 | 0 / 0 / 0 |

Both arms returned finite values; this reproduces corruption, not overflow.
The successful control's worst error against the double reference was about
`2.025e-5`, ordinary half quantization. Its disassembly changed the critical
`vmcnt(1)` to `vmcnt(0)` before the subtraction. Disabling subregister liveness
was also tested and retained both the instruction hazard and the corruption.

The initial compiler control forces vector-load waits to zero; LLVM's
[wait insertion source](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp)
defines the options. The production workaround now forces all wait counters to
zero, for the backward masked path described below. It preserves storage and
arithmetic precision, mask semantics and parity envelopes. It conservatively
reduces memory overlap;
these correctness probes make no performance claim. The selector applies to
device sets containing `gfx1102` (including feature suffixes), and the cache
regime derives from that same set and selector.
Compilation is backend-wide and an artifact can link on any device, so mixed
sets conservatively apply the guard even when HIPRTC targets another ordinal.
Sets without `gfx1102` retain their compiler policy. Other architectures have
not been established as affected.

The flag was accepted by the measured HIPRTC version. An unsupported-option
error remains a compiler failure with the effective option vector; there is no
retry with the unsafe option omitted. Removing or narrowing the workaround
requires executed evidence with the replacement compiler, not just a version
number. `hip_half_load` replays the consumer through the shipped pipeline three
times with initialized inputs and NaN-poisoned output.

At implementation `f69f9cb60d6e4f73a9cc8477bd424c96347735f6`, the shipped
HIP half-load regression and existing half-softmax test passed. Two fresh,
untuned s1024 inference processes returned the same four losses:
`[7.1051445, 7.09284115, 7.0877161, 7.11898851]`, within the existing f16
`0.002` relative envelope against the recorded CPU-F32 reference. Base training
returned six finite losses but still missed that envelope; its final two losses
repeated its first two. The s1024 training parity phase returned six finite
losses, then exceeded its 90-second cap during subsequent dominant-kernel
instrumentation, before emitting its result. That endpoint is a timeout, not a
completed acceptance pass. The whole issue remains unresolved; this workaround
addresses the independently reproduced load corruption. Parameter snapshots
and host optimizer-gate state in `bench_gpt_diag` support the bounded follow-up
on training. Correctness probes disable dominant-kernel instrumentation.

At `1126d85abd286af5cd8522316606140cb888a157`, s1024 training completed
with six finite losses and maximum relative error `0.000137887`, within that
same envelope. Base-training default and graph-capture-disabled controls
launched no optimizer steps in six parity steps: their gradient checksums were
NaN and all 52 master parameter hashes stayed unchanged while loss scaling
backed off. The materialized control skipped its first step, then launched five
optimizer steps; 43 master parameter hashes changed. In the default's final
buffer observations, the earliest producer with a nonfinite final output was
segment 155's last-layer softmax denominator gradient (16 of 8192 rows).
Its stored scores, maxima, denominators and incoming softmax gradients were
finite. This identifies the next replay boundary; it does not yet establish
which arithmetic or instruction first produced a nonfinite value.
`BENCH_DUMP_DIR` with `BENCH_DUMP_NODES` optionally saves selected step-zero
buffers as little-endian float32 data for exact half/single-input replay.

An exact replay of segment 155 used those captured direct inputs. They were all
finite; half-rounded denominator squares were nonzero and at most 10728. The
reference's largest term and partial sum were about 12.758, far below half
overflow. The reference uses float FMA followed by half rounding, so its bit
differences are diagnostic rather than a promise of exact half-FMA ties.

The disassembly's masked path issued a D16 high-half gradient load, then could
modify the low half before waiting for vector memory. Its scalar mask wait was
only `lgkmcnt(0)`: strengthening existing vector waits did not drain that load.
Moving the denominator square to global memory retained both the chain and
nonfinite outputs. Changing only the wait option gave this result on the same
source and captured buffers:

| Segment 155 control | Nonfinite rows in three launches (of 8192) |
| --- | --- |
| `-amdgpu-waitcnt-load-forcezero` | 0 / 40 / 16 |
| `-amdgpu-waitcnt-forcezero` | 0 / 0 / 0 |

The all-counter control changed waits to
`vmcnt(0) expcnt(0) lgkmcnt(0)` and drained pending high-half loads before low-half
updates. The earlier forward replay also returned zero wrong or nonfinite cells
in all three launches under this control. A single passing old-policy launch
is not sufficient evidence: its failures were asynchronous and stochastic.
`hip_half_masked_gradient` uses synthetic inputs varying with every coordinate,
bounded denominator squares and six repeated launches through the shipped
pipeline; no captured fixture is required by the test. These controls establish
the broader guard, not the complete model's final acceptance, which must be
verified separately at the production revision.

At production revision `1c84af7eb6ff7e36017e1f40bce7d6948d25823e`, the TUF
HIP regressions and all four untuned full-parity endpoints completed. Against
unique same-fixture CPU-F32 exact reference rows, maximum relative errors were
`0.0000652923` for base training, `0.0000405403` for each of two s1024 inference
processes, and `0.0001573344` for s1024 training. Every loss was finite and each
endpoint met the original `0.002` envelope. Both training endpoints executed
all six parity optimizer steps at the original loss scale 65536. The two
inference processes returned identical four-loss sequences. These are bounded
correctness results; subsequent protocol timings are not a performance claim.
The retained reference rows omit a precision field, but each workload has one
CPU/eager/exact row with the float32 policy recorded; their fixture tensors are
F32 and the historical PyTorch runner does not cast them to reduced precision.

Durable raw scripts, hashes, code objects, disassembly and logs are retained
under `~/.local/state/issue-wave/wave2-20261004/1182-scratch` on TUF, with
`1182-forcezero-runs/run-1/{log,rc}` recording the successful control leg.
The isolated HIP source SHA256 is
`4f996f9865ad7e4ee5f9a8ecc6665e0056cc5097ed0179dd76f1b214256f889e`;
the host replay source SHA256 is
`f59d3ef898d6ded5363bc593d45f0d7b6ed9a10d2809128c2444d4aae85573d7`.
