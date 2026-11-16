#!/usr/bin/env bash
# Shared GPU runtime-refusal signatures for the sweep floor and native-PTX leg (gh-ocannl-2219).
# Boundary: reads only the allowlisted, line-start uncaught binding exceptions (and exact launch
# memory statuses) and the named runtime assertion below. It does not read arbitrary mentions of
# OOM, retry chatter, kernel journals, typed schedule declines, or decide a test's final verdict.
# A matching log needs a serial retry before attributing its failures to the code or environment.

# The failure names that mean the ENVIRONMENT refused the run rather than a
# test judging it. On minix the WSL2 dxg bridge (tools/box-jobs.sh, and the dxg
# bullet in docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals)
# surfaces its lost messages as three HIP exceptions; rog-nv's CUDA reaches its
# GPU through the same WSL2 bridge, so the cudajit checks at the same three call sites are
# listed by analogy, plus the primary-context retain every CUDA process makes
# first -- where rog-nv's one lost bridge message was reported, as
# CUDA_ERROR_OUT_OF_MEMORY on a device with VRAM to spare. A unit whose log
# carries any of them is environment-red: its stanzas were refused a device,
# not judged, and whatever test-logic failures it also holds are hidden under
# that noise until the environment is repaired -- days, for a WSL box. The 2026-09-05 wide run
# hid a genuine hip-only regression that way (gh-ocannl-943; the executable
# segfaulted after its hip_init was refused, so no fingerprint could have shown
# it), and only rerunning the failing stanzas at `-j 1` on the box told the two
# apart: 4 of 27 stayed red on their own. So such a unit gets exactly that rerun
# (sweep.sh's serial_rerun, gh-ocannl-945): every failing stanza again under `-j 1`,
# and `serial rerun:` lines in its log and fingerprint saying which stayed red.
# Keyed as dune prints an uncaught binding error, `Fatal error: exception
# <name>:` with the status on the next line. An entry is a name, or a name and
# the one status that qualifies it: the kernel launch is where a concurrent
# suite's device-memory pressure lands (rog-nv's native cuda unit lost
# `schedule_strided_1x1` to `cu_launch_kernel` CUDA_ERROR_OUT_OF_MEMORY on
# 2026-10-02 with an empty kernel window, and passed alone), but it is also
# where a kernel's own resource bug lands deterministically
# (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, INVALID_VALUE for shared memory over the
# limit) and where an earlier kernel's sticky fault surfaces
# (CUDA_ERROR_ILLEGAL_ADDRESS) -- those are a test's own failures, so only the
# memory status buys the rerun. The statuses each name has been
# seen with, and how to read a rerun's verdict, are the signature table in
# docs/agent-notes/build-and-test.md#gpu-boxes-job-caps-and-runtime-refusals
# (the record half of gh-ocannl-927).
ENVIRONMENT_REFUSALS='hip_init
hip_module_load_data_ex
hip_stream_create_with_priority
hip_module_launch_kernel HIP_ERROR_OUT_OF_MEMORY
cu_init
cu_device_primary_ctx_retain
cu_module_load_data_ex
cu_stream_create_with_priority
cu_launch_kernel CUDA_ERROR_OUT_OF_MEMORY'

# The runtime ASSERTIONS that mean the same thing, for a refusal the runtime does not report as an
# error at all but dies of. On a native boot, a HIP process whose SDMA queue KFD refused goes on
# into ROCr's scratch teardown and aborts there -- the 2026-09-23 ladder lost `schedule_conv_gemm`
# at dune's default width to exactly this, beside the kernel's `No more SDMA queue to allocate`
# (gh-ocannl-1029), as one log line:
#   schedule_conv_gemm.exe: ./runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp:2003: virtual void
#   rocr::AMD::GpuAgent::ReleaseQueueMainScratch(rocr::AMD::ScratchInfo&): Assertion
#   `scratch.main_queue_base' failed.
# The kernel window counts the refusal itself; this is the half that lands in the unit's LOG, which
# keeps the rerun when the window could not be read. Keyed as glibc prints a failed assert --
# `<prog>: <file>:<line>: <function>: Assertion `<expr>' failed.` -- with the function as the
# entry, so an unrelated assertion in the same runtime is not caught by it.
ENVIRONMENT_ASSERTIONS='GpuAgent::ReleaseQueueMainScratch'

# Whether the log holds `Fatal error: exception <name>:` with exactly <status> after it: on the
# next line, where the bindings' printer breaks, or on the same line should it ever not break.
refusal_with_status() { # log name status
  awk -v head="Fatal error: exception $2:" -v status="$3" '
    pending { line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t\r]+$/, "", line)
              if (line == status) { found = 1; exit } }
    { pending = 0 }
    index($0, head) == 1 {
      rest = substr($0, length(head) + 1); sub(/^[ \t]+/, "", rest); sub(/[ \t\r]+$/, "", rest)
      if (rest == status) { found = 1; exit }
      pending = rest == ""
    }
    END { exit !found }' "$1"
}

environment_log_refusal() { # log; print the first matching signature, or exit nonzero
  local name status
  while read -r name status; do
    [ -n "$name" ] || continue
    if [ -z "$status" ]; then
      if grep -q "^Fatal error: exception $name:" "$1"; then
        printf '%s\n' "$name"
        return 0
      fi
    elif refusal_with_status "$1" "$name" "$status"; then
      printf '%s %s\n' "$name" "$status"
      return 0
    fi
  done <<<"$ENVIRONMENT_REFUSALS"
  while IFS= read -r name; do
    if [ -n "$name" ] &&
      grep -qE "^[^ ]+: [^ ]+:[0-9]+: .*${name}.*: Assertion .* failed\.$" "$1"; then
      printf '%s\n' "$name"
      return 0
    fi
  done <<<"$ENVIRONMENT_ASSERTIONS"
  return 1
}
