#!/usr/bin/env bash
# A remote GPU unit's kernel evidence: which lines of the unit's window count as
# the environment refusing the device, and how many. Sourced by tools/sweep.sh
# (whose `environment_red` gives a unit with a positive count its serial rerun,
# and whose run record carries the window, the count and its kind) and by
# test/operations/sweep_harness.sh (which feeds it the recorded evidence
# directly). Sourced, never executed; `window_kind_of` reads
# `box_jobs_dest_transport`, so tools/box-jobs.sh is sourced first.
#
# One file rather than functions inside the sweep for the reason box-jobs.sh
# gives: the judgement is in the FILTER, and a filter no test can call is one
# that gets its errno list wrong quietly. Everything here reads kernel lines and
# writes text -- no ssh, no box, no state -- so the harness drives it directly.

# ---------------------------------------------------------------- the kinds
# WHICH evidence a unit's window holds depends on how the sweep reached its box, and the sweep knows
# that before the unit starts: the ssh destination names the boot (box_jobs_dest_transport,
# gh-ocannl-1029). Two kinds, named as that function names the transports:
#
#   dxg     a WSL2 boot (`-wsl`). The GPU is behind the Hyper-V `/dev/dxg` bridge, and a lost
#           message on its ring is the refusal (gh-ocannl-979).
#   native  a native Linux boot (`-linux`). No bridge; the refusals are the GPU drivers' own --
#           amdgpu's device-wide SDMA queue pool running dry, an NVIDIA Xid event (gh-ocannl-1034).
#
# Each kind has its own filter, count and collection query, and the two signature sets are kept
# APART rather than unioned: a dxg line cannot appear on a native boot and an amdgpu queue refusal
# cannot appear through the bridge, so a union would only let a window of one kind count lines that
# describe the other. Before the split a native boot collected the dxg window, which read
# `unavailable` on every GPU unit (no `/dev/dxg`, and `dmesg` restricted by
# `kernel.dmesg_restrict=1` on both native boxes) while the SDMA refusals that did happen went
# unread and unrerun.
#
# A destination whose transport is neither has NO window: the unit writes no sidecar, and the record
# reads `-`, the value that means none was collected. Not `unavailable`: that says a collection was
# attempted and failed, a claim about a box nobody tried to read. None exists in the sweep's unit
# table today (a destination override must be one of the box's two canonical aliases), and a new box
# addressed as `<name>-linux` or `<name>-wsl` gets its kind from the suffix alone.
window_kind_of() { # ssh-destination -> dxg | native, or nothing for no window
  local kind
  kind=$(box_jobs_dest_transport "${1:-}")
  case $kind in dxg | native) printf '%s' "$kind" ;; *) ;; esac
}

# What each kind's count counts, as its verdict line spells it. The dxg noun is the one the block
# has carried since gh-ocannl-979, kept byte-for-byte.
window_noun() { # kind
  case $1 in
    dxg) printf 'vmbus_sendpacket failures' ;;
    *) printf 'GPU queue refusals' ;;
  esac
}

utc_of() { # epoch-seconds -> the stamp format, so a window reads like every other time here
  perl -MPOSIX -e 'print strftime("%Y%m%dT%H%M%SZ", gmtime($ARGV[0]))' "$1"
}

# ---------------------------------------------------------------- the dxg filter
# The WSL2 dxg bridge's own kernel evidence, filtered and counted (gh-ocannl-979).
#
# A GPU unit on a WSL box reaches its device through /dev/dxg, and a lost message
# on that Hyper-V ring surfaces in userspace as whatever exception the binding
# happened to be making at the time. `ENVIRONMENT_REFUSALS` lists the names seen
# so far, and a list of names can only ever grow AFTER a miss: rog-nv's
# CUDA_ERROR_OUT_OF_MEMORY at `cu_device_primary_ctx_retain` cost a remote session
# to attribute (gh-ocannl-979), and a SEGV -- which the 2026-09-15 minix runs also
# produced -- can never become a name entry at all. The kernel says it plainly in
# the unit's own time window, so the unit reads that window and the sweep stops
# depending on having already seen the call site.
#
# What counts, from the 2026-09-15 evidence on both boxes:
#   - `dxgkio_query_adapter_info` and `dxgkio_is_feature_enabled` failures are
#     dropped whatever their errno. Both are logged by every VM boot before any
#     test runs (minix: -22, -2, -11 and -1 across twelve lines) and nvidia-smi
#     emits the first constantly. An errno-keyed filter would have kept most of
#     them.
#   - the burst is counted on `vmbus_sendpacket failed` ALONE. A real lost message
#     arrives as a triple -- `dxgvmb_send_sync_msg: vmbus_sendpacket failed:
#     fffffff5`, `create_existing_sysmem: failed set existing pages: fffffff5`,
#     `dxgkio_create_allocation: Ioctl failed: -11` -- so keying on the status
#     `fffffff5` would count the same lost message twice.
# Everything else in the window is kept and shown, counted or not: a new signature
# is exactly what this exists to surface, and dropping the unrecognised is how the
# name list came to lag in the first place.
DXG_BENIGN='dxgkio_query_adapter_info|dxgkio_is_feature_enabled'

dxg_window_summary() { # start-utc end-utc [boot-verdict] -- kernel lines on stdin, block on stdout
  local kept bursts
  kept=$(grep -E 'misc dxg' | grep -Ev "$DXG_BENIGN") || true
  bursts=$(printf '%s\n' "$kept" | grep -c 'vmbus_sendpacket failed') || true
  # `grep -c` counts an empty line as no match, but say so explicitly rather than
  # relying on it: an empty `kept` must read as zero, never as one.
  [ -n "$kept" ] || bursts=0
  printf '=== dxg window %s..%s (utc) ===\n' "$1" "$2"
  # The signature is the line from the driver's own prefix on, which drops the timestamp whichever
  # tool printed it.
  printf '%s\n' "$kept" | window_lines dxg 's/^.*misc dxg: /misc dxg: /'
  window_verdict dxg "$bursts" "${3:-}"
}

# ---------------------------------------------------------------- the native filter
# A native boot's GPU drivers' own refusals (gh-ocannl-1034), read from the same kernel journal.
#
# Kept: every line from the GPU drivers -- `amdgpu`, KFD (`kfd`, `amdkfd`) and NVIDIA's resource
# manager (`NVRM:`) -- and nothing else. A native box's kernel log is mostly NOT about the GPU (NIC
# link changes, input devices, workqueue warnings, the nvme queue census at every resume), and none
# of it is evidence about a unit's device; the dxg filter's `misc dxg` match does the same job for
# the bridge. Within the drivers' lines nothing is dropped: they are rare outside a boot or a
# resume, and a resume INSIDE a unit's window -- which logs `ring sdma0 uses VM inv eng ...` -- is
# worth seeing beside a failure even though it counts for nothing.
#
# Counted, from the 2026-09-23 evidence (gh-ocannl-1029's width ladder on minix-amd-linux) and the
# NVIDIA driver's own error channel:
#   - `DQM create queue type <n> failed` -- KFD refusing a process a hardware queue, whatever the
#     reason. One line per refused queue: `amdgpu: process pid 85562 DQM create queue type 1
#     failed. ret -12` (-12 is -ENOMEM).
#   - `No more SDMA queue to allocate` is that refusal's REASON when the device-wide SDMA pool ran
#     dry (`amdgpu 0000:c5:00.0: No more SDMA queue to allocate (8 total queues)`), logged just
#     before the DQM line for the same queue. So it counts only in a window with no DQM line at
#     all -- the pair is one refusal, for the reason the dxg count keys on the vmbus line alone --
#     and a driver that logs only the reason still gets its unit rerun.
#   - `NVRM: Xid` -- the NVIDIA driver's GPU error event (`NVRM: Xid (PCI:0000:02:00): 79, pid=...,
#     GPU has fallen off the bus.`), one line per event. Keyed on the `NVRM: ` prefix, not on the
#     word: rog-nv's NIC driver logs `RTL8125B, ..., XID 641` at every boot.
# The userspace half of the SDMA refusal -- ROCr's `GpuAgent::ReleaseQueueMainScratch` assertion,
# which is how the process whose queue was refused dies -- is not a kernel line; the sweep reads it
# from the unit's log instead (ENVIRONMENT_ASSERTIONS in tools/sweep.sh).
#
# NOT counted, and shown: any other driver line. rog-nv's journal holds 57 `NVRM: VM: invalid mmap
# context` lines and an `NVRM: API mismatch` from the last minutes of a driver upgrade before its
# reboot (2026-09-22), and nothing has established what those mean for a unit; they surface as
# signatures, which move the fingerprint, and promoting one to a counted refusal is a line here once
# something has.
NATIVE_KEEP='amdgpu|kfd|NVRM:'
NATIVE_QUEUE_REFUSED='DQM create queue type [0-9]+ failed'
NATIVE_SDMA_EXHAUSTED='No more SDMA queue to allocate'
NATIVE_XID='NVRM: Xid '

native_window_summary() { # start-utc end-utc [boot-verdict] -- kernel lines on stdin, block on stdout
  local kept refusals xids
  kept=$(grep -E "$NATIVE_KEEP") || true
  refusals=$(printf '%s\n' "$kept" | grep -cE "$NATIVE_QUEUE_REFUSED") || true
  [ "$refusals" -gt 0 ] ||
    refusals=$(printf '%s\n' "$kept" | grep -c "$NATIVE_SDMA_EXHAUSTED") || true
  xids=$(printf '%s\n' "$kept" | grep -c "$NATIVE_XID") || true
  refusals=$((refusals + xids))
  [ -n "$kept" ] || refusals=0
  printf '=== native window %s..%s (utc) ===\n' "$1" "$2"
  # The signature is the line from the driver's name on, with the process ids that differ between
  # two equally refused runs replaced: `pid 85562` in KFD's line, `pid=1234` in an Xid event. A
  # fingerprint compares these bytewise, and a pid would make a standing SDMA red report
  # `fingerprint moved` on every run. The first two expressions strip the timestamp in the
  # journal's form and in `dmesg -T`'s.
  printf '%s\n' "$kept" | window_lines native \
    's/^.* kernel: //; s/^\[[^]]*\] //; s/pid [0-9][0-9]*/pid N/g; s/pid=[0-9][0-9]*/pid=N/g'
  window_verdict native "$refusals" "${3:-}"
}

# ---------------------------------------------------------------- the block, either kind
window_summary() { # kind start-utc end-utc [boot-verdict] -- kernel lines on stdin, block on stdout
  case $1 in
    dxg) dxg_window_summary "$2" "$3" "${4:-}" ;;
    native) native_window_summary "$2" "$3" "${4:-}" ;;
    *) return 1 ;;
  esac
}

# The raw lines, capped, then the distinct signatures, uncapped. stdin is the kept lines.
window_lines() { # kind signature-sed
  local kept shown
  kept=$(cat)
  [ -n "$kept" ] || return 0
  # The raw lines are capped: a bad window runs to hundreds of them (496 on
  # minix in one boot) and the log is for reading.
  printf '%s\n' "$kept" | head -40
  shown=$(printf '%s\n' "$kept" | wc -l | tr -d ' ')
  [ "$shown" -gt 40 ] && printf '(%s lines in the window; the first 40 are shown)\n' "$shown"
  # The distinct SIGNATURES, timestamps stripped, never capped -- there are a
  # handful however bad the window is, and they are what the fingerprint reads.
  # Capping these instead of the raw lines is the difference between a new
  # kernel signature at line 300 being surfaced and being invisible, which is
  # the whole purpose of keeping unrecognised lines at all.
  printf '%s\n' "$kept" | sed "$2" | sed "s/^/$1 signature: /" | sort -u
}

# What a window whose GUEST WAS REPLACED records instead of a count. Like WINDOW_UNAVAILABLE it must
# not be `0`, and for a sharper reason: a window that spans a VM death collects the journal of BOTH
# boots (the query is deliberately `_TRANSPORT=kernel` and not `-k`, so it spans them), and a new
# VM that came up clean contributes no dxg lines at all. The window therefore reads `0
# vmbus_sendpacket failures` -- "the bridge was fine" -- over a unit whose machine ceased to exist
# underneath it. A native box that rebooted mid-unit is the same finding about a different machine:
# the boot id is the kernel's, not the bridge's, so the check and the verdict are shared.
#
# That is exactly what happened on 2026-09-16. Sweep 20260916T074913Z lost rog-nv/cuda and
# minix/hip when a concurrent `wake-lab.sh --restart-wsl` destroyed both guests mid-unit, and both
# sidecars recorded a clean window -- which was then cited as evidence AGAINST the VM having been
# the problem, and the failure was attributed to the GPU autotune tests for two days. A replaced
# guest is not a weaker finding than a burst; it is a stronger one, and it has to say so where
# every reader of this block already looks.
WINDOW_VM_REPLACED=vm-replaced

# The block's closing half, either kind: the boot finding, then the verdict line.
window_verdict() { # kind count [boot-verdict]
  local kind=$1 count=$2
  # A replaced guest is reported with the count still visible above it -- the lines are real and
  # worth reading -- but it TAKES THE VERDICT, because the count no longer describes one machine
  # and `0` would read as a clean window over a machine that went away. `unknown` is the
  # pre-existing state under a new name: a box whose boot id could not be read is no worse off than
  # it was before the check existed, and saying so beats both silence and a false alarm.
  case ${3:-} in
    replaced)
      printf 'the guest was REPLACED during this window: the VM that ran the unit is gone, so\n'
      printf 'the count above spans two boots and certifies nothing about either.\n'
      count=$WINDOW_VM_REPLACED ;;
    unknown)
      printf 'the guest boot id could not be read: this window is not known to cover one VM.\n' ;;
  esac
  # The line environment_red keys on, and the field the run record carries. Last,
  # so that a truncated block still ends with its verdict.
  printf '=== %s window: %s %s ===\n' "$kind" "$count" "$(window_noun "$kind")"
}

# What a collection that did not happen writes instead of a count. It must not be
# `0`: a zero-count window is a POSITIVE finding -- the device was fine -- and an
# ssh that timed out establishes nothing, so reading one as the other would hide
# exactly the unlisted failure (a SEGV, say) this trigger exists to catch.
WINDOW_UNAVAILABLE=unavailable
# A failed collection over a guest KNOWN to have been replaced still reports the replacement. The
# two verdicts are not equal in strength and the weaker one must not win by arriving last: a
# replaced guest is a positive finding about the box, while `unavailable` says only that nobody
# read the kernel log. They also coincide often -- a VM that has just been destroyed and recreated
# is exactly the one whose journal query is most likely to fail -- so reporting `unavailable`
# there would drop the stronger evidence precisely in the case it was collected for.
window_unavailable() { # kind start-utc end-utc reason [boot-verdict]
  local verdict=$WINDOW_UNAVAILABLE
  [ "${5:-}" = replaced ] && verdict=$WINDOW_VM_REPLACED
  printf '=== %s window %s..%s (utc) ===\n' "$1" "$2" "$3"
  printf 'collection failed: %s\n' "$4"
  [ "$verdict" = "$WINDOW_VM_REPLACED" ] &&
    printf 'the guest was REPLACED during this window: the VM that ran the unit is gone.\n'
  printf '=== %s window: %s %s ===\n' "$1" "$verdict" "$(window_noun "$1")"
}

# ---------------------------------------------------------------- reading a unit's evidence
# Where a unit's collected evidence lives, beside its log rather than inside it.
# The log is appended by the unit's own test leg, and a test leg can print
# anything -- this repository's sweep harness dumps window FIXTURES on failure,
# and it runs as a test action inside a sweep unit, so a local cc unit's log can
# end up holding a complete synthetic burst block. Parsing the log for evidence
# therefore cannot distinguish what the collector wrote from what the tests
# printed, and the consequences are not cosmetic: a local unit that never touched
# a GPU would be marked environment-red and given the expensive serial rerun,
# and its record row and fingerprint would report a device failure. Provenance is
# the file: only collect_kernel_window writes this one, and only remote GPU units
# with a window kind get one at all. The block is ALSO appended to the log, for
# whoever reads it there.
window_sidecar() { # log -> the path of its collected-evidence file
  printf '%s.kernel-window' "${1%.log}"
}

# The verdict a unit's collected window recorded -- a count, `unavailable` or `vm-replaced` -- or
# nothing if it has no window (a local unit, one that never ran, or a transport with no kind).
# `[0-9a-z-]`: a `-` last in the bracket expression is a literal one in every dialect, and no
# dialect reads it as a range there. The kind is matched as `[a-z]*` rather than as an alternation
# of the two, because `\|` in a basic regex is a GNU extension the macOS controller's sed lacks.
window_count() { # log
  sed -n 's/^=== [a-z]* window: \([0-9a-z-][0-9a-z-]*\) [^=]* ===$/\1/p' \
    "$(window_sidecar "$1")" 2>/dev/null | tail -1
}

# WHICH kind of window the unit collected, or nothing. The count's meaning depends on it -- lost
# bridge messages or refused GPU queues -- so the record carries it beside the count.
window_kind() { # log
  sed -n 's/^=== \([a-z]*\) window [0-9TZ-]*\.\.[0-9TZ-]* (utc) ===$/\1/p' \
    "$(window_sidecar "$1")" 2>/dev/null | tail -1
}

# The block's stable half, for the fingerprint. The log and the run record keep
# the window, its lines and the exact count; a fingerprint must not, because it is
# compared BYTEWISE against the previous failure's and a standing environment red
# would otherwise report `fingerprint moved` on every single run -- the window
# instants differ, the kernel timestamps differ, and the count differs between two
# equally broken runs (161 and 123 on minix within an hour). What is stable, and
# is what a reader wants from a diff, is WHICH signatures appeared and whether the
# device was being refused at all: a window that starts or stops showing refusals,
# or shows a new signature, still moves the fingerprint; the same failure twice does not.
window_fingerprint_lines() { # log
  local block kind
  block=$(cat "$(window_sidecar "$1")" 2>/dev/null)
  [ -n "$block" ] || return 0
  kind=$(window_kind "$1")
  # The block's own signature lines, which the summary emits uncapped -- NOT a
  # re-derivation from the raw lines it shows, which are capped at 40 and would
  # silently drop a signature that first appeared late in a bad window.
  # `|| true` for the same reason `kept` above has one, and it matters more here: a window with a
  # VERDICT but no signature lines -- an unavailable collection, a replaced guest, a clean zero --
  # makes this grep exit 1, and under `set -e` with `pipefail` that aborts the function before the
  # case below, dropping the verdict line entirely. sweep.sh runs `set -uo pipefail` with no `-e`
  # today, so nothing in production loses it; a caller that turns errexit on would, silently, and
  # the line it would lose is the one that tells a standing red from a fingerprint that moved.
  printf '%s\n' "$block" | grep '^[a-z]* signature: ' | sort -u || true
  # The dxg words are the ones gh-ocannl-979 wrote, kept so an existing dxg fingerprint stays put.
  case $(window_count "$1"):$kind in
    "$WINDOW_UNAVAILABLE":*) printf '%s window: collection unavailable\n' "$kind" ;;
    "$WINDOW_VM_REPLACED":*) printf '%s window: guest replaced mid-window\n' "$kind" ;;
    0:dxg) printf 'dxg window: no burst\n' ;;
    0:*) printf '%s window: no refusal\n' "$kind" ;;
    :*) ;;
    *:dxg) printf 'dxg window: burst present\n' ;;
    *) printf '%s window: refusal present\n' "$kind" ;;
  esac
}

# The window a unit's collected evidence claims, space-separated, or nothing if it
# collected none. `-` for both bounds is what a collection that failed before the
# remote could report them writes, and it is a state of its own: the record must
# show that unit as `unavailable`, not as one where collection was never tried.
# SPACE-separated, not tab: a tab would have to come from a `\t` in a sed
# replacement, which is a GNU extension that this repository's macOS controller
# happens to honour but no standard requires. Neither bound can contain a space,
# so the separator costs nothing and depends on no dialect.
window_bounds() { # log -> "<start> <end>", or nothing
  sed -n 's/^=== [a-z]* window \([0-9TZ-]*\)\.\.\([0-9TZ-]*\) (utc) ===$/\1 \2/p' \
    "$(window_sidecar "$1")" 2>/dev/null | tail -1
}

# Whether a unit's guest was destroyed and recreated during its window. Its own predicate because
# the two readers want it for opposite reasons: window_red folds it into "this unit earned a
# rerun", and the `error` path needs it as a plain fact about the box, where no rerun exists.
window_guest_replaced() { # log
  [ "$(window_count "$1")" = "$WINDOW_VM_REPLACED" ]
}

# Whether a unit's collected evidence makes it environment-red. A POSITIVE count
# only: `0` is a positive finding the other way (the device was fine), and
# `unavailable` establishes nothing in either direction -- an unread box must not
# buy a rerun any more than it may certify a clean one. Absent evidence (a local
# unit, one that never ran, a transport with no window) is likewise not red.
window_red() { # log
  window_count "$1" | grep -qE "^([1-9][0-9]*|$WINDOW_VM_REPLACED)$"
}

# ---------------------------------------------------------------- the collection query
# Read the kernel's log for the window on the box that ran the unit. The journal
# rather than `dmesg`, because the VM can DIE inside the window: on 2026-09-15
# minix's went away twice mid-unit (a Windows Update restart, then an unheld VM
# powering off), and `dmesg` in the next session starts from the new boot and
# loses exactly the evidence being collected. Every sweep box keeps a persistent
# journal (`/var/log/journal`), so it spans those deaths; `dmesg -T` stays as the
# fallback for a box whose journald keeps no kernel log, where losing a dead
# boot's window beats collecting nothing.
#
# `_TRANSPORT=kernel` and NOT `-k`, which is the same match plus an implied `-b`:
# `-k` restricts the answer to the CURRENT boot, so a window spanning a VM death
# -- the one case this collection exists for -- would come back truncated at the
# boot boundary, with nothing to say it had been. Measured on minix, 2026-09-15,
# over a window covering three boots: `-k --since` returned 123 of the window's
# `vmbus_sendpacket` lines, `_TRANSPORT=kernel --since` all 365. The same implied
# `-b` reported ZERO dxg lines for rog-nv's 2026-09-13 window, which in fact holds
# 255 of them and that unit's three-message burst.
#
# BOTH bounds are instants of the remote's own clock -- the start read by the
# reachability probe when the unit began there, the end read here -- because the
# log's timestamps are in that clock and no other. The controller and a WSL VM do
# not share a wall clock (these boxes resynchronise after host resumes, which
# the note records as having preceded a burst), so a controller instant carried
# across would put a terminal burst after `--until` or before `--since`,
# reporting a false clean window and losing the rerun; and reconstructing the
# start from a locally measured duration assumes the remote clock advanced
# continuously through the unit, which is the same assumption in a thinner
# disguise. The remote echoes the bounds it used, so the block reports the
# window that was actually queried.
#
# `+ 1` on the end, as the start rounds up off the probe's second: both bounds
# round AWAY from the neighbouring units, in whole seconds, so a window can
# never inherit the burst of whatever ran beside it. Whole seconds because finer
# ones are not available here: measured on both sweep boxes (2026-09-15),
# `journalctl --since "@<epoch>.<frac>"` is refused and returns ZERO lines
# rather than an error -- silently blanking the evidence, which is the failure
# this exists to prevent -- while `dmesg --since` accepts the fractional form,
# on the branch neither box takes. The residual that leaves (a burst inside the
# probe's own second, before the unit's first GPU call) is gh-ocannl-984.
#
# Both ends bounded, on both branches of either kind: the block and the record row
# claim a CLOSED window, so an event arriving after the unit finished -- while this
# query is on its way, or from whatever ran next -- must not be attributed to it,
# inflating its count and buying it a rerun it did not earn. `--since @<epoch>` is
# accepted by util-linux dmesg (2.41.3 on both sweep boxes); a dmesg without it
# fails the collection rather than reporting an unbounded ring as this window.
window_cmd() { # kind remote-start-epoch
  printf 'window_end=$(date +%%s); window_end=$((window_end + 1)); '
  printf 'window_start=%s; ' "$2"
  printf 'echo "window-bounds $window_start $window_end"; '
  # The guest's identity as of the END of the window, against the one the start probe read. A WSL2
  # VM that is destroyed and recreated comes back at the same ssh alias, with the same hostname and
  # the same home directory, and answers this collection perfectly well -- nothing else in the
  # round trip can tell the operator they are talking to a different machine than the unit ran on.
  # /proc/sys/kernel/random/boot_id is regenerated per boot and needs no privilege to read.
  printf 'echo "window-boot $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"; '
  printf 'if command -v journalctl >/dev/null 2>&1 && '
  case $1 in
    dxg)
      # The probe must see an ENTRY, and one INSIDE THIS WINDOW. journalctl with no
      # readable kernel journal exits 0 and prints `-- No entries --` on stdout (the
      # explanation goes to stderr), so both a status test and a nonempty test select
      # journald there and query a second empty journal instead of falling back --
      # recording a burst still sitting in the kernel ring as zero. Keying on "a line
      # that is not a `--` marker" holds whatever journalctl decides to print, which
      # `--quiet` alone does not guarantee. Bounding the probe to the window covers
      # the other half of the same hazard: a journal holding historical entries but
      # not recording the CURRENT boot answers an unbounded probe from its history,
      # and the bounded query that follows then returns nothing while the burst sits
      # in the dmesg ring nobody consulted. A window genuinely quiet in the journal
      # falls through to dmesg, which -- bounded the same way -- simply agrees.
      printf 'journalctl -q _TRANSPORT=kernel --since @$window_start --until @$window_end '
      printf -- '-n 1 --no-pager 2>/dev/null | grep -qv "^--"; then '
      ;;
    *)
      # A native boot restricts `dmesg` (`kernel.dmesg_restrict=1` on both native boxes, read
      # 2026-09-23) while its user reads the journal (the `adm` group), so the dxg probe's
      # fall-through for a quiet window -- the NORMAL native window, since a native kernel logs
      # nothing about the GPU while nothing is refused -- lands on a dmesg that refuses, and every
      # clean native unit would read `unavailable`. So the journal is judged instead on whether it
      # records the CURRENT BOOT's kernel at all, which a readable one always does (the boot's own
      # messages are in it), and it then answers for the window whatever the window holds. The two
      # hazards the dxg probe guards are still closed: a journal with no readable kernel log has no
      # entry for `-b`, and neither has one that keeps only history.
      printf 'journalctl -q _TRANSPORT=kernel -b -n 1 --no-pager 2>/dev/null | grep -qv "^--"; then '
      ;;
  esac
  printf 'journalctl -q _TRANSPORT=kernel --since @$window_start --until @$window_end '
  printf -- '--no-pager 2>/dev/null; '
  printf 'else dmesg -T --since @$window_start --until @$window_end 2>/dev/null; fi'
}
