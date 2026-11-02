#!/usr/bin/env python3
"""Generate a per-kernel timing harness for an emitted OCANNL HIP step (gh-ocannl-569).

Why this exists. The gh-ocannl-531 profile got its per-kernel timeline from Nsight Systems. On a
WSL2 guest there is no ``/dev/kfd`` -- the GPU arrives through ``/dev/dxg`` -- and ROCm's profiler
needs KFD, so ``rocprofv3 --kernel-trace`` collects nothing at all (verified against a ten-line HIP
program, not just against OCANNL). This reconstructs the same per-kernel numbers without a
profiler: it compiles the emitted batch source as-is and times every ``__segN`` kernel individually
with HIP events, at the launch geometry the compile actually used.

What it is and is not. Kernels are timed in ISOLATION on synthetic buffers, not in the step's
dispatch order, so this is a reconstruction rather than a timeline: it cannot see inter-kernel gaps,
and each kernel meets a cache the real step would have left in a different state. It is sound for
these kernels because every loop bound in the emitted source is a literal and the only
data-dependent construct is a ``select``-shaped ``Where``, so the work done does not depend on the
buffer contents. The validation that makes it usable is arithmetic: the sum of the per-kernel
medians is compared against the measured step time, and the report quotes that agreement.

Buffers are deliberately generous (32 MiB each against a 4 MiB largest tensor, 8x headroom) and
distinct per parameter, since the kernels declare ``__restrict__`` and aliasing them would be
undefined. Contents are a benign non-zero pattern: zeros would risk denormal slow paths on some
operators and bias the timings.

Usage:
    gpt2_kernel_harness.py --source armA.hip --launches source.err --out harness.hip
    hipcc --offload-arch=gfx1151 -O2 -o harness harness.hip && ./harness > kernels.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_output  # noqa: E402

SIG_RE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+__seg(\d+))\s*\(([^)]*)\)', re.S)

BUF_BYTES = 32 << 20


def parse_kernels(src: Path):
    out = []
    for name, idx, params in SIG_RE.findall(src.read_text()):
        kinds = []
        for p in params.split(","):
            p = " ".join(p.split())
            if not p:
                continue
            if "unsigned int" in p and "*" in p:
                kinds.append("u32*")
            elif "*" in p:
                kinds.append("f32*")           # every pointer param in an f32 step
            else:
                kinds.append("int")
        out.append((name, int(idx), kinds))
    if not out:
        sys.exit(f"{src}: no __global__ ...__segN kernels found")
    return sorted(out, key=lambda k: k[1])


def parse_launches(log: Path, routine: str, n_segs: int):
    """seg index -> (grid, block) for the compile that fissioned `routine` into `n_segs` kernels:
    the lines config [schedule_log_launches] prints, one per segment, read by bench_output's one
    reader of them (gh-ocannl-1767). The segment COUNT disambiguates the placement arms, which
    compile the same routine name at different fissions."""
    geo = {}
    for launch in bench_output.schedule_launches(log.read_text()):
        if launch["routine"] == routine and launch["segs"] == n_segs:
            geo[launch["seg"]] = (launch["grid"], launch["block"])
    return geo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="emitted batch source (arm A)")
    ap.add_argument("--launches", required=True, type=Path,
                    help="stderr of a run with --ocannl_schedule_log_launches=true")
    ap.add_argument("--out", required=True, type=Path, help="generated .hip harness")
    ap.add_argument("--reps", type=int, default=20, help="timed launches per kernel (median wins)")
    args = ap.parse_args()

    kernels = parse_kernels(args.source)
    routine = kernels[0][0].rsplit("__seg", 1)[0]
    geo = parse_launches(args.launches, routine, len(kernels))
    missing = [i for _, i, _ in kernels if i not in geo]
    if missing:
        sys.exit(f"no launch geometry for {routine} at {len(kernels)} segments: seg {missing[:5]}... "
                 f"-- is the launch log from the same compile?")

    # Two pools, and the split is load-bearing rather than tidiness. An unsigned-int parameter is an
    # INDEX buffer (the token-embedding gather reads wte[... + ids[...]]), so its contents are
    # addresses, not data: filling it with the float pool's benign 0x3f3f3f3f pattern asks for
    # element 1,061,109,567 and the kernel runs away off the end of the allocation. Index buffers
    # get zeros -- always a valid element -- and only the float pool gets the non-zero pattern.
    nf = max(sum(1 for k in ks if k == "f32*") for _, _, ks in kernels)
    nu = max(sum(1 for k in ks if k == "u32*") for _, _, ks in kernels)
    src_inc = args.source.resolve()

    L = []
    L.append('#include <hip/hip_runtime.h>')
    L.append(f'#include "{src_inc}"')
    L.append('#include <cstdio>')
    L.append('#include <vector>')
    L.append('#include <algorithm>')
    L.append('')
    # Underscore-prefixed macro local: the timing loop below has its own `e` (an event), and an
    # unhygienic `e` here shadows it into a compile error.
    L.append('#define HC(x) do { hipError_t _hc=(x); if(_hc!=hipSuccess){ \\')
    L.append('  fprintf(stderr,"HIP %s @%d: %s\\n",#x,__LINE__,hipGetErrorString(_hc)); exit(1);} } while(0)')
    L.append('')
    L.append(f'static const size_t BUF_BYTES = {BUF_BYTES}ull;')
    L.append(f'static void *fbufs[{nf}];')
    L.append(f'static void *ubufs[{max(nu, 1)}];')
    L.append(f'static const int REPS = {args.reps};')
    L.append('')
    L.append('''static double time_kernel(void (*launch)(), int reps) {
  for (int i = 0; i < 3; ++i) launch();          // warm up: first launch pays module/code-object costs
  HC(hipDeviceSynchronize());
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    hipEvent_t b, e; HC(hipEventCreate(&b)); HC(hipEventCreate(&e));
    HC(hipEventRecord(b)); launch(); HC(hipEventRecord(e));
    HC(hipEventSynchronize(e));
    float t; HC(hipEventElapsedTime(&t, b, e)); ms.push_back(t);
    HC(hipEventDestroy(b)); HC(hipEventDestroy(e));
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}
''')

    for name, idx, kinds in kernels:
        (gx, gy, gz), (bx, by, bz) = geo[idx]
        argv, fi, ui = [], 0, 0
        for k in kinds:
            if k == "int":
                argv.append("0")                      # the static batch index: 0 is always in range
            elif k == "u32*":
                argv.append(f"(unsigned int*)ubufs[{ui}]"); ui += 1
            else:
                argv.append(f"(float*)fbufs[{fi}]"); fi += 1
        L.append(f'static void launch_{idx}() {{ hipLaunchKernelGGL({name}, '
                 f'dim3({gx},{gy},{gz}), dim3({bx},{by},{bz}), 0, 0, {", ".join(argv)}); }}')

    L.append('')
    L.append('int main() {')
    L.append(f'  for (int i = 0; i < {nf}; ++i) {{')
    L.append('    HC(hipMalloc(&fbufs[i], BUF_BYTES));')
    # 0x3f3f3f3f is ~0.7477f: non-zero, normal, and finite, so no operator meets a denormal and
    # none of exp/log/sqrt sees an argument that would take a slow path.
    L.append('    HC(hipMemset(fbufs[i], 0x3f, BUF_BYTES));')
    L.append('  }')
    L.append(f'  for (int i = 0; i < {nu}; ++i) {{')
    L.append('    HC(hipMalloc(&ubufs[i], BUF_BYTES));')
    L.append('    HC(hipMemset(ubufs[i], 0x00, BUF_BYTES));')   # index 0: in range for every gather
    L.append('  }')
    # Progress goes to stderr and every CSV row is flushed as it is produced: a kernel that hangs
    # (a single-work-item segment can) must not take the rows already measured down with it.
    L.append(f'  fprintf(stderr, "setup done: {nf} float + {nu} index buffers x {BUF_BYTES >> 20} MiB\\n");')
    L.append('  printf("Name,Calls,TotalDurationNs\\n"); fflush(stdout);')
    L.append('  double total = 0;')
    for _, idx, _ in kernels:
        L.append(f'  {{ double ms = time_kernel(launch_{idx}, REPS); total += ms;')
        L.append(f'    printf("{routine}__seg{idx},1,%.0f\\n", ms * 1e6); fflush(stdout);')
        L.append(f'    fprintf(stderr, "seg{idx}: %.4f ms\\n", ms); }}')
    L.append('  fprintf(stderr, "sum of per-kernel medians: %.3f ms over %zu kernels\\n", total, '
             f'(size_t){len(kernels)});')
    L.append('  return 0;')
    L.append('}')

    args.out.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}: {len(kernels)} kernels, "
          f"{nf} float + {nu} index buffers x {BUF_BYTES >> 20} MiB")


if __name__ == "__main__":
    main()
