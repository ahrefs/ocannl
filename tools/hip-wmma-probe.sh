#!/usr/bin/env bash
# Build and run tools/hip_wmma_sign_probe.hip at -O3 and at -O0 on this box's first HIP device,
# and say in one line what it found (gh-ocannl-1658). The probe asks a question about the gfx11
# WMMA INSTRUCTION -- whether a same-sign sum comes back exact, and how each accumulator rounds
# an inexact one -- so its answer moves with the ROCm toolkit and the target, not with OCANNL:
# tools/sweep.sh runs this after each HIP unit (minix's gfx1151, tuf's gfx1102), and a worker
# runs it by hand after a ROCm upgrade or on a new target.
#
# Usage: tools/hip-wmma-probe.sh   (from any directory; needs hipcc and rocminfo, found on PATH
#                                   or under $ROCM_PATH, $HIP_PATH or /opt/rocm)
#
# The probe's own output follows a `=== hip-wmma-probe -O<n> ===` header per level. The last line
# on stdout is the verdict, and the exit status says the same:
#   hip-wmma-probe: verdict: holds                    0  every same-sign case exact, every
#                                                        rounding case in its recorded mode
#   hip-wmma-probe: verdict: holds, rounding news     0  same, but a rounding case found a mode
#                                                        other than the one the probe records:
#                                                        news for the agent note, not a failure
#   hip-wmma-probe: verdict: SAME-SIGN RULE BROKE     1  a same-sign case came back inexact: the
#                                                        bitwise HIP tensorized claims rest on it
#   hip-wmma-probe: verdict: unavailable: <reason>    2  nothing was judged (no toolkit, no gfx11
#                                                        device, a compile or launch failure)
# A level at which the rule broke outranks one with news; a level that built or ran nothing is
# named beside the other level's verdict, and the verdict is unavailable only when neither judged. Its boundary: it reads the probe's exit status
# and its `[rounding: news` marker, nothing else of its output.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
src=$root/tools/hip_wmma_sign_probe.hip

verdict() { # status text
  printf 'hip-wmma-probe: verdict: %s\n' "$2"
  exit "$1"
}

find_tool() { # name -- prints its path, or nothing
  local d
  if command -v "$1" >/dev/null 2>&1; then
    command -v "$1"
    return 0
  fi
  for d in "${ROCM_PATH:-}" "${HIP_PATH:-}" /opt/rocm; do
    if [ -n "$d" ] && [ -x "$d/bin/$1" ]; then
      printf '%s\n' "$d/bin/$1"
      return 0
    fi
  done
}

[ -r "$src" ] || verdict 2 "unavailable: no $src"
hipcc=$(find_tool hipcc)
[ -n "$hipcc" ] || verdict 2 "unavailable: no hipcc on PATH or under ROCM_PATH, HIP_PATH, /opt/rocm"
rocminfo=$(find_tool rocminfo)
[ -n "$rocminfo" ] || verdict 2 "unavailable: no rocminfo on PATH or under ROCM_PATH, HIP_PATH, /opt/rocm"
arch=$("$rocminfo" 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*')
[ -n "$arch" ] || verdict 2 "unavailable: rocminfo names no gfx agent"
# The probe calls gfx11's wave32 builtins; gfx12 has other ones (see the probe's header).
case $arch in
  gfx11*) ;;
  *) verdict 2 "unavailable: $arch is not gfx11 (the probe calls gfx11's WMMA builtins)" ;;
esac

work=$(mktemp -d "${TMPDIR:-/tmp}/hip-wmma-probe.XXXXXX") || verdict 2 "unavailable: no temporary directory"
trap 'rm -rf "$work"' EXIT
broke= news= judged= unjudged=
for level in 3 0; do
  echo "=== hip-wmma-probe -O$level ($arch, $hipcc) ==="
  if ! "$hipcc" -std=c++17 "-O$level" "--offload-arch=$arch" "$src" -o "$work/probe$level" 2>&1; then
    echo "hip-wmma-probe: -O$level: hipcc failed"
    unjudged="$unjudged -O$level"
    continue
  fi
  "$work/probe$level" >"$work/out$level" 2>&1
  rc=$?
  cat "$work/out$level"
  case $rc in
    0 | 1)
      judged=1
      [ "$rc" = 0 ] || broke="$broke -O$level"
      grep -qF '[rounding: news' "$work/out$level" && news="$news -O$level"
      ;;
    *)
      echo "hip-wmma-probe: -O$level: the probe exited $rc, judging nothing"
      unjudged="$unjudged -O$level"
      ;;
  esac
done
[ -n "$judged" ] || verdict 2 "unavailable: no level built and ran ($arch)"
# A level that judged nothing is named beside a verdict the other level reached.
partial=
[ -z "$unjudged" ] || partial="; not judged at$unjudged"
[ -z "$broke" ] || verdict 1 "SAME-SIGN RULE BROKE at$broke ($arch$partial)"
[ -z "$news" ] || verdict 0 "holds, rounding news at$news ($arch$partial)"
verdict 0 "holds ($arch$partial)"
