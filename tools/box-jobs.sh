#!/usr/bin/env bash
# The single source of the dune width caps for a GPU suite: the one behind a
# WSL2 `/dev/dxg` bridge, and the native ones minix's copy-engine queues and
# rog-nv's measured correctness slots set -- sourced by tools/sweep.sh (whose
# `unit_jobs` decides a sweep unit's width) and by tools/test-run.sh (which
# injects the local one into a manual run that expressed no width at all).
# Sourced, never executed.
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
# line in any window (gh-ocannl-1029). These are one batch's widths, measured
# with the box to itself. A manual or worker batch on a native boot may share
# the box with the fleet's other correctness slot, so it takes the per-slot
# widths of the local view below instead (gh-ocannl-1033).
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

# A native boot has neither table entry to key on nor a bridge to find, so the
# local view probes the two things that make a native GPU batch's width
# matter, each overridable the way the dxg device is (tools/test-test-run.sh
# fakes both, present and absent), and each falling back to the real path when
# empty.
#
# First, the SDMA pool, from the KFD topology: one directory per node, each
# with a `properties` file, a GPU node being one with SIMDs. The pool a node
# hands to processes is `num_sdma_engines` x `num_sdma_queues_per_engine` as
# topology reports them -- the queues KFD can allocate, which leaves out the
# ones the driver reserves (so minix's gfx1151 reports 1 x 6 while the kernel's
# refusal counts "8 total queues"). Read on the fleet's boxes, 2026-09-23:
# minix-amd-linux (gfx1151 iGPU) reports 1 x 6, tuf-amd-linux (gfx1102
# discrete) 2 x 6; rog-nv-linux has no KFD node at all.
box_jobs_kfd_topology() {
  printf '%s' "${OCANNL_TOOL_KFD_TOPOLOGY:-/sys/class/kfd/kfd/topology/nodes}"
}

box_jobs_sdma_pool() { # prints the smallest GPU node's allocatable SDMA queues, or nothing
  local topo
  topo=$(box_jobs_kfd_topology)
  [ -d "$topo" ] || return 0
  # One awk over every node: FNR resets per file, so each file's values are
  # judged at its own end. A node that does not report both counts (an older
  # kernel), or reports no SDMA engine at all, has no pool this can judge, so
  # it is skipped rather than read as a pool of zero.
  awk '
    function judge() {
      if (simd > 0 && eng > 0 && per > 0) {
        pool = eng * per
        if (best == "" || pool < best) best = pool
      }
    }
    FNR == 1 { if (NR > 1) judge(); simd = 0; eng = ""; per = "" }
    $1 == "simd_count" { simd = $2 + 0 }
    $1 == "num_sdma_engines" { eng = $2 + 0 }
    $1 == "num_sdma_queues_per_engine" { per = $2 + 0 }
    END { if (NR > 0) judge(); if (best != "") printf "%d", best }
  ' "$topo"/*/properties 2>/dev/null
}

# The pool the SDMA measurement was made on: minix's topology, 1 engine x 6.
# A device whose pool is no larger is capped; a larger one (tuf's 12) is not,
# because nothing has measured what width it tolerates, and halving a box's
# width on an analogy is not a change this file makes (see the sweep table).
BOX_JOBS_SDMA_MEASURED_POOL=6

box_jobs_small_sdma_pool() { # 0 iff a GPU node's SDMA pool is no larger than the measured one
  local pool
  pool=$(box_jobs_sdma_pool)
  [ -n "$pool" ] && [ "$pool" -le "$BOX_JOBS_SDMA_MEASURED_POOL" ]
}

# Second, a native NVIDIA driver: its control device exists on a native Linux
# boot and on no other (a WSL boot reaches the GPU through /dev/dxg instead,
# and Windows and macOS have no such node).
box_jobs_nvidia_device() { printf '%s' "${OCANNL_TOOL_NVIDIA_DEVICE:-/dev/nvidiactl}"; }

box_jobs_native_nvidia_host() { # 0 iff this is a native NVIDIA boot
  [ -e "$(box_jobs_nvidia_device)" ]
}

# How many correctness batches the fleet runs at once on a native GPU box:
# lukstafi/ludics-lite#316 gave rog-nv-linux and minix-amd-linux two
# `execution slot` slots each, measured with two concurrent batches at the
# widths below. Restated here rather than read from the fleet, because a run
# launched outside `execution slot` -- by hand, or by a worker that skipped
# the slot -- must get the same width, and the width must be the same number
# the slot count was measured at.
BOX_JOBS_NATIVE_GPU_SLOTS=2

# The native hip width: the concurrent-HIP-process budget the SDMA pool was
# measured to take (BOX_JOBS_SDMA_CAP, 8), split across the box's slots, so
# that every slot running a hip batch at once stays within it. Two slots of 4
# is what ludics-lite#316 measured on minix: two `-j 4` hip batches peaked at
# exactly 8 GPU-holding processes, with no SDMA or DQM line in any kernel
# window. The budget is those 8 GPU-holding processes rather than the pool's 6
# queues: the slot rounds sampled 8 at once, twice, with a clean kernel window,
# so not every GPU-holding process holds a queue at the same moment, and the
# measured count is the one to stay within. A single batch alone could run at
# 8; the cap cannot know whether it is alone, and 4 costs a full unit 12% of
# its wall time (1065 s against 946 s, gh-ocannl-1029).
BOX_JOBS_SDMA_SLOT_CAP=$((BOX_JOBS_SDMA_CAP / BOX_JOBS_NATIVE_GPU_SLOTS))

# The native cuda width: what ludics-lite#316 measured rog-nv-linux's two
# slots at. Two concurrent `-j 8` cuda batches (14 GPU-holding processes) were
# green in every rung; three (21) hit one `CUDA_ERROR_OUT_OF_MEMORY` in
# `fused_classifier` with an empty kernel window, which is why the count is
# two. Dune's default width there is 24, and two 24-wide batches -- up to 48
# GPU processes -- were never run. One batch alone was green at every width
# (gh-ocannl-1029), and `-j 8` was its fastest.
BOX_JOBS_NATIVE_CUDA_CAP=8

# The local view, for a run on THIS box: which hazard, if any, a batch of the
# selected backend meets here -- `dxg` (the bridge), `sdma` (a small copy-engine
# pool, hip only) or `nvidia` (a native CUDA box's slots, cuda only). The
# bridge comes first: a WSL boot keeps its own cap whatever else it reports.
# The backend is read from the ENVIRONMENT only -- see the caller
# (tools/test-run.sh) for why a config file is not consulted.
box_jobs_local_hazard() { # <backend>; prints dxg, sdma, nvidia, or nothing
  box_jobs_gpu_backend "${1:-}" || return 0
  if box_jobs_dxg_host; then
    printf 'dxg'
    return 0
  fi
  case $1 in
    hip) box_jobs_small_sdma_pool && printf 'sdma' ;;
    cuda) box_jobs_native_nvidia_host && printf 'nvidia' ;;
  esac
  return 0
}

box_jobs_hazard_cap() { # <hazard>; prints its cap, or nothing
  case ${1:-} in
    dxg) printf '%s' "$BOX_JOBS_DXG_CAP" ;;
    sdma) printf '%s' "$BOX_JOBS_SDMA_SLOT_CAP" ;;
    nvidia) printf '%s' "$BOX_JOBS_NATIVE_CUDA_CAP" ;;
    *) ;;
  esac
}

box_jobs_local_cap() { # <backend>; prints the cap, or nothing
  box_jobs_hazard_cap "$(box_jobs_local_hazard "${1:-}")"
}
