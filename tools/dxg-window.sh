#!/usr/bin/env bash
# The WSL2 `/dev/dxg` bridge's kernel evidence: which lines of a unit's window
# count as a lost-message burst, and how many. Sourced by tools/sweep.sh (whose
# `environment_red` gives a unit with a burst its serial rerun, and whose run
# record carries the window and the count) and by test/operations/sweep_harness.sh
# (which feeds it the recorded evidence directly). Sourced, never executed.
#
# One file rather than a function inside the sweep for the reason box-jobs.sh
# gives: the judgement is in the FILTER, and a filter no test can call is one
# that gets its errno list wrong quietly. Everything here reads kernel lines and
# writes text -- no ssh, no box, no state -- so the harness drives it directly.

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
utc_of() { # epoch-seconds -> the stamp format, so a window reads like every other time here
  perl -MPOSIX -e 'print strftime("%Y%m%dT%H%M%SZ", gmtime($ARGV[0]))' "$1"
}

DXG_BENIGN='dxgkio_query_adapter_info|dxgkio_is_feature_enabled'
dxg_window_summary() { # start-utc end-utc -- kernel lines on stdin, block on stdout
  local kept bursts shown
  kept=$(grep -E 'misc dxg' | grep -Ev "$DXG_BENIGN") || true
  bursts=$(printf '%s\n' "$kept" | grep -c 'vmbus_sendpacket failed') || true
  # `grep -c` counts an empty line as no match, but say so explicitly rather than
  # relying on it: an empty `kept` must read as zero, never as one.
  [ -n "$kept" ] || bursts=0
  printf '=== dxg window %s..%s (utc) ===\n' "$1" "$2"
  if [ -n "$kept" ]; then
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
    printf '%s\n' "$kept" | sed 's/^.*misc dxg: /dxg signature: misc dxg: /' | sort -u
  fi
  # The line environment_red keys on, and the field the run record carries. Last,
  # so that a truncated block still ends with its verdict.
  printf '=== dxg window: %s vmbus_sendpacket failures ===\n' "$bursts"
}

# The burst count a unit's collected window recorded, or nothing if it has no
# window (a local unit, or one that never ran).
# What a collection that did not happen writes instead of a count. It must not be
# `0`: a zero-burst window is a POSITIVE finding -- the bridge was fine -- and an
# ssh that timed out establishes nothing, so reading one as the other would hide
# exactly the unlisted failure (a SEGV, say) this trigger exists to catch.
DXG_UNAVAILABLE=unavailable
dxg_window_unavailable() { # start-utc end-utc reason
  printf '=== dxg window %s..%s (utc) ===\n' "$1" "$2"
  printf 'collection failed: %s\n' "$3"
  printf '=== dxg window: %s vmbus_sendpacket failures ===\n' "$DXG_UNAVAILABLE"
}

dxg_bursts() { # log
  sed -n 's/^=== dxg window: \([0-9a-z][0-9a-z]*\) vmbus_sendpacket failures ===$/\1/p' "$1" \
    2>/dev/null | tail -1
}

# Read the kernel's log for the window on the box that ran the unit. The journal
# rather than `dmesg`, because the VM can DIE inside the window: on 2026-09-15
# minix's went away twice mid-unit (a Windows Update restart, then an unheld VM
# powering off), and `dmesg` in the next session starts from the new boot and
# loses exactly the evidence being collected. Both sweep boxes keep a persistent
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
# The journal is chosen on whether it ANSWERS, not on whether the command exits 0:
# a host with journalctl installed and no kernel journal prints "No journal files
# were found" and exits 0, and selecting it there means querying a second empty
# journal instead of falling back -- a burst still in the kernel ring would be
# recorded as zero. And the fallback is bounded to the window like the journal
# query is: `dmesg -T` alone returns the whole current-boot ring, so an EARLIER
# unit's burst would be attributed to this one and buy it a rerun it did not earn.
# `--since @<epoch>` is accepted by util-linux dmesg (2.41.3 on both sweep boxes);
# a dmesg without it fails the collection rather than reporting an unbounded ring
# as this window.
dxg_window_cmd() { # elapsed-seconds
  # The probe must see an ENTRY, not merely output: journalctl with no readable
  # kernel journal exits 0 and prints `-- No entries --` on stdout (the
  # explanation goes to stderr), so both a status test and a nonempty test select
  # journald there and query a second empty journal instead of falling back --
  # recording a burst still sitting in the kernel ring as zero. Keying on "a line
  # that is not a `--` marker" holds whatever journalctl decides to print, which
  # `--quiet` alone does not guarantee.
  #
  # The bounds are computed ON THE BOX whose log is being read, from the unit's
  # ELAPSED seconds rather than from this machine's instants. The controller and a
  # WSL VM do not share a wall clock -- these boxes resynchronise after host
  # resumes, which is documented in the note as having preceded a burst -- and the
  # log's timestamps are the remote's, so a remote clock ahead of ours would put
  # the unit's terminal burst after `--until` and one behind ours would put it
  # before `--since`, either way reporting a false clean window and losing the
  # rerun. A duration survives that; an instant does not. The remote also prints
  # the bounds it used, so the block reports the window that was actually queried.
  #
  # `+ 1` on the end: a burst timestamped N.xxx in the same second the collection
  # begins is excluded by `--until @N`, and losing a terminal burst to a rounding
  # boundary is exactly the failure this exists to catch. The extra fraction of a
  # second can only contain the unit's own activity -- its lane runs one unit at a
  # time on that box -- so widening is the safe direction, and the reported bound
  # is widened with it so the claim stays the query.
  printf 'dxg_end=$(date +%%s); dxg_end=$((dxg_end + 1)); '
  printf 'dxg_start=$((dxg_end - %s - 1)); ' "$1"
  printf 'echo "dxg-window-bounds $dxg_start $dxg_end"; '
  printf 'if command -v journalctl >/dev/null 2>&1 && '
  printf 'journalctl -q _TRANSPORT=kernel -n 1 --no-pager 2>/dev/null | '
  printf 'grep -qv "^--"; then '
  # Both ends bounded, on both branches: the block and the record row claim a
  # CLOSED window, so an event arriving after the unit finished -- while this
  # query is on its way, or from whatever ran next -- must not be attributed to
  # it, inflating its count and buying it a rerun it did not earn.
  printf 'journalctl -q _TRANSPORT=kernel --since @$dxg_start --until @$dxg_end '
  printf -- '--no-pager 2>/dev/null; '
  printf 'else dmesg -T --since @$dxg_start --until @$dxg_end 2>/dev/null; fi'
}

# The block's stable half, for the fingerprint. The log and the run record keep
# the window, its lines and the exact count; a fingerprint must not, because it is
# compared BYTEWISE against the previous failure's and a standing environment red
# would otherwise report `fingerprint moved` on every single run -- the window
# instants differ, the kernel timestamps differ, and the count differs between two
# equally broken runs (161 and 123 on minix within an hour). What is stable, and
# is what a reader wants from a diff, is WHICH signatures appeared and whether the
# bridge was losing messages at all: a bridge that starts or stops failing, or
# fails in a new way, still moves the fingerprint; the same failure twice does not.
dxg_fingerprint_lines() { # log
  local block
  # Only the LAST such block, which is the collector's own: a unit's log is
  # whatever its test leg wrote plus what the post-unit phases append, and a test
  # leg can print marker-delimited text of its own -- this repository's sweep
  # harness dumps its dxg fixtures on failure, and it runs as a test action inside
  # a sweep unit. A range expression over the whole log would fold those fixture
  # signatures into the unit's fingerprint and report bridge failures that never
  # happened.
  block=$(awk '/^=== dxg window /{ b = "" } { if (b != "" || /^=== dxg window /) b = b $0 "\n" }
    /^=== dxg window: .* ===$/ { last = b; b = "" } END { printf "%s", last }' \
    "$1" 2>/dev/null)
  [ -n "$block" ] || return 0
  # The block's own signature lines, which dxg_window_summary emits uncapped --
  # NOT a re-derivation from the raw lines it shows, which are capped at 40 and
  # would silently drop a signature that first appeared late in a bad window.
  printf '%s\n' "$block" | grep '^dxg signature: ' | sort -u
  case $(printf '%s\n' "$block" | sed -n \
    's/^=== dxg window: \([0-9a-z][0-9a-z]*\) vmbus_sendpacket failures ===$/\1/p' | tail -1) in
    "$DXG_UNAVAILABLE") printf 'dxg window: collection unavailable\n' ;;
    0) printf 'dxg window: no burst\n' ;;
    "") ;;
    *) printf 'dxg window: burst present\n' ;;
  esac
}
