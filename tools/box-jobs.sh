#!/usr/bin/env bash
# The single source of the dune width caps for a GPU suite: the one behind a
# WSL2 `/dev/dxg` bridge, and the native one minix's copy-engine queues set --
# sourced by tools/sweep.sh (whose `unit_jobs` decides a sweep unit's width)
# and by tools/test-run.sh (which injects the bridge's into a manual run that
# expressed no width at all). Sourced, never executed.
#
# Why one file rather than a constant in each: the cap was sweep policy, written
# into `unit_jobs` and a bullet of docs/agent-notes/build-and-test.md, and every
# other way into a GPU suite on such a box -- a manual run, a wave worker's
# verification leg, a PR author validating a backend change -- ran at dune's
# default width (32 jobs on minix) and came back red in the same stanzas a real
# regression lands in. An hour of box time and a misleading bisect went into
# rediscovering it (gh-ocannl-983). Two copies of a number nobody reads until it
# is wrong is how that repeats.
#
# The hazard: the GPU is reached over a Hyper-V VM-bus ring, and every
# allocation and module load is a synchronous message on it. The ring overflows
# when the suite's test executables hold the device at once, and the runtime
# reports the lost messages as device/binary/stream-creation refusals -- see the
# dxg bullet in docs/agent-notes/build-and-test.md for the signature, the kernel
# evidence (`dmesg | grep 'misc dxg'`) and the recovery.

# Measured on minix's degraded bridge, 2026-09-05: dune's default width lost 67
# stanzas (356 kernel-side refusals), `-j 4` still lost 27 (120), and `-j 2` ran
# a forced full `@runtest @train` unit clean in 18.5 minutes. A single GPU
# serialises the kernels anyway, so the cap costs the test phase little; the
# compile phase is never capped.
BOX_JOBS_DXG_CAP=2

# A native boot has no bridge, but minix's iGPU has a pool of its own that the
# suite's width can drain: the amdgpu driver has 8 SDMA (copy-engine) queues
# for the whole device, not per process, and every HIP process that copies
# takes one. Measured on minix's native Ubuntu 26.04 boot, 2026-09-23
# (gh-ocannl-1029), over a forced full `@runtest @train` hip unit at each width:
# dune's default (32) lost `schedule_conv_gemm` to a ROCr assertion
# (`GpuAgent::ReleaseQueueMainScratch`), with the kernel logging `No more SDMA
# queue to allocate (8 total queues)` and `DQM create queue type 1 failed.
# ret -12` in that window, while `-j 8`, `-j 4` and `-j 2` logged no GPU kernel
# line at all. The width that cannot oversubscribe the pool is its size, and it
# costs nothing: the test phase ran 921 s at 32, 939 s at 8, 1059 s at 4 and
# 1138 s at 2. (All four widths share seven red stanzas that no width
# changes; the width is not what they are about.)
BOX_JOBS_SDMA_CAP=8

# The backends that hold the device, i.e. the ones the bridge carries. A CPU
# backend on the same box runs at full width.
box_jobs_gpu_backend() { # <backend>; 0 iff it holds a GPU
  case ${1:-} in cuda | hip) return 0 ;; *) return 1 ;; esac
}

# How the sweep reached a box, from the ssh destination it used: `dxg` for a
# WSL2 boot's endpoint, `native` for a native Linux boot's, nothing for a name
# that says neither (a local unit, or a destination overridden by hand). The
# two lab boxes dual-boot, and each boot answers on its own endpoint -- the
# `-wsl` alias is the WSL guest's sshd, the `-linux` alias the native install's
# (lukstafi/ludics-lite#313) -- so the name the sweep connected through IS the
# boot it tested, known before the unit starts, with no extra round trip and
# no probe that can itself fail (gh-ocannl-1029).
box_jobs_dest_transport() { # <ssh-destination>; prints dxg, native, or nothing
  case ${1:-} in
    *-wsl) printf 'dxg' ;;
    *-linux) printf 'native' ;;
    *) ;;
  esac
}

# The sweep's view. It runs units on boxes it never stats, so the cap is a
# property of the (machine, backend, transport) triple here rather than
# something probed on the box.
#
# Across the bridge, only minix's hip unit is capped. rog-nv reached its CUDA
# device through the same bridge under WSL, and one `vmbus_sendpacket` burst
# was recorded there (2026-09-13), but nothing measured what width its discrete
# GPU tolerated there, and halving the width of a daily unit that had been green
# at full width is not a change to make from an analogy. A MANUAL run on a dxg
# boot is capped all the same (box_jobs_local_cap below probes the device, not
# this table): that caller expressed no width, is about to read a red suite, and
# has no 5400-second budget riding on the answer.
#
# On a native boot, minix's hip unit takes the SDMA cap above instead of the
# bridge's, and rog-nv's cuda unit stays uncapped: its native ladder ran the
# forced full unit green at dune's default (24), 8, 4 and 2, with no NVRM/Xid
# line in any window (gh-ocannl-1029).
#
# An unrecognised destination keeps the WSL measurement: running a native unit
# at the bridge's width costs minutes, while running a dxg one at full width
# costs a red unit that reads like a backend regression.
box_jobs_sweep_cap() { # <machine> <backend> [<ssh-destination>]; prints the cap, or nothing
  case "${1:-}:${2:-}:$(box_jobs_dest_transport "${3:-}")" in
    minix:hip:native) printf '%s' "$BOX_JOBS_SDMA_CAP" ;;
    minix:hip:*) printf '%s' "$BOX_JOBS_DXG_CAP" ;;
    *) ;;
  esac
}

# The device a dxg host publishes. Overridable so the detection is testable off
# such a host: tools/test-test-run.sh points it at a file it creates (a faked
# bridge) and at a path that does not exist (a faked ordinary box). An empty
# value is not a way to disable the probe -- it falls back to the real device,
# so a caller cannot unset its way past the cap by accident.
box_jobs_dxg_device() { printf '%s' "${OCANNL_TOOL_DXG_DEVICE:-/dev/dxg}"; }

box_jobs_dxg_host() { # 0 iff this box reaches its GPU through the bridge
  [ -e "$(box_jobs_dxg_device)" ]
}

# The local view, for a run on THIS box: the cap applies when the bridge is
# present and the selected backend holds the device. The backend is read from
# the ENVIRONMENT only -- see the caller (tools/test-run.sh) for why a config
# file is not consulted.
box_jobs_local_cap() { # <backend>; prints the cap, or nothing
  box_jobs_dxg_host || return 0
  box_jobs_gpu_backend "${1:-}" || return 0
  printf '%s' "$BOX_JOBS_DXG_CAP"
}
