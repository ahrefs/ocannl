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

The compiler option forces load waits to zero; LLVM's
[wait insertion source](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp)
defines the option. The workaround preserves storage and arithmetic precision,
mask semantics and parity envelopes. It conservatively reduces load overlap;
these correctness probes make no performance claim. The selector applies only
to `gfx1102` (including feature suffixes), and the cache regime derives from
that same selector. Other architectures have not been established as affected.

The flag was accepted by the measured HIPRTC version. An unsupported-option
error remains a compiler failure with the effective option vector; there is no
retry with the unsafe option omitted. Removing or narrowing the workaround
requires executed evidence with the replacement compiler, not just a version
number. `hip_half_load` replays the consumer through the shipped pipeline three
times with initialized inputs and NaN-poisoned output.

Durable raw scripts, hashes, code objects, disassembly and logs are retained
under `~/.local/state/issue-wave/wave2-20261004/1182-scratch` on TUF, with
`1182-forcezero-runs/run-1/{log,rc}` recording the successful control leg.
The isolated HIP source SHA256 is
`4f996f9865ad7e4ee5f9a8ecc6665e0056cc5097ed0179dd76f1b214256f889e`;
the host replay source SHA256 is
`f59d3ef898d6ded5363bc593d45f0d7b6ed9a10d2809128c2444d4aae85573d7`.
