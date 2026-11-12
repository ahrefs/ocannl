#!/usr/bin/env bash
# Tests for tools/test-run.sh: the width caps, the fleet's run-time slot, the backends a batch
# resolves to, the device probe and the per-test cap.
#
#   tools/test-test-run-devices.sh          # run every leg of this member
#   tools/test-test-run-devices.sh --keep   # keep the temp dir for inspection
#   tools/test-test-run-devices.sh --only '^53:'   # only the legs whose names match
#   tools/test-test-run-devices.sh --subject FILE   # test FILE, not tools/test-run.sh
#
# One member of the harness of tools/test-run.sh (gh-ocannl-2002), whose
# others `tools/test-test-run.sh --members` lists; that script runs them all,
# or the ones holding the legs an --only names. The conventions every member
# shares -- --subject, how a leg is named, gated and abandoned, what is tested
# and the fixtures the legs run against -- are tools/test-test-run-common.sh's
# header. Leg numbers are the harness's, not this file's: they stayed as they
# were when it was one file, so a member's are not consecutive.
#
# Groups: #width (29-34, 48, 49, 50, 51, 64, 52), #slot (53, 54, 55, 56, 57, 58, 93, 78, 88), #device (75, 76, 77, 91, 106), #hermetic (70).
#
# Legs:
#  29. the dxg width cap is injected, announced on stderr, written into the run
#      log and recorded in the command, for a GPU run that named no width.
#  30. a width the caller named is honored in every spelling dune takes, and
#      says so; one past dune's own `--` is the program's, so it still caps.
#  31. a CPU backend, or a box with no bridge, is neither capped nor told
#      anything.
#  32. with OCANNL_BACKEND unset the backend is resolved (gh-ocannl-1066): a
#      batch the configurations pin to cc is silent; one whose configuration
#      names hip, one reaching a stanza naming cuda, and an unreadable one are
#      capped, saying why; a named width is only told.
#  33. hip caps too, and the width is spliced after dune's subcommand.
#  34. the sweep's `box_jobs_sweep_jobs` helper and the injected cap read one table, with
#      the sweep's own override still winning; the sweep's width follows the
#      unit's destination (a WSL boot's bridge cap, a native boot's own), and
#      every box_jobs_sweep_jobs call site passes that destination.
#  48-52 sit beside the dxg legs, after leg 34: the native caps
#      (gh-ocannl-1033), against faked KFD topologies and a faked NVIDIA
#      device.
#  48. hip on a small SDMA pool (minix's recorded topology) and on a larger one
#      (tuf's) is capped at that box's per-slot width, announced and recorded;
#      cuda on a native NVIDIA boot is capped at its own, and so is a CPU
#      backend on the fleet's rog-nv-linux (named by a faked FLEET_LOCAL_BOX,
#      or by the hostname through FLEET_HOSTNAME_MAP; gh-ocannl-1065), with
#      the `execution slot --cpu` note; the sweep's native tuf/hip and
#      rog-nv/cuda units run at exactly the width injected on that box.
#  49. negative controls: no KFD topology, a GPU node that reports no pool, a
#      CPU backend on each AMD hazard, on an NVIDIA box that is not the
#      fleet's rog and on a WSL boot of rog, and each GPU backend on the other
#      one's hazard -- no cap, no word.
#  50. a width the caller named is honored on every native hazard, and a dxg
#      boot that also reports a small pool keeps its dxg cap.
#  51. with OCANNL_BACKEND unset on a native hazard the width follows the
#      batch's resolved backends: cc is uncapped on the AMD boxes and an
#      unfleeted NVIDIA one; a configuration naming hip, a reached hip stanza
#      or an unread answer gets hip's width; cc on the fleet's rog gets the CPU
#      width; hip and cc beside a small pool on rog take the tighter.
#  52. the native widths derive from one place: each hip cap is its measured
#      budget over its slot count, and the numbers are the measured ones.
#  75-77 sit just before leg 70: a wedged device cannot hold a batch
#      (gh-ocannl-1211).
#  75. a test past --test-cap ends the run as TIMEOUT (142) naming it, the test
#      killed and the worktree idle, blaming a device only on a GPU batch; a
#      test inside the cap, a lifted cap, a slow dune and the slot's long-lived
#      holder do not (skipped where a $0-renamed fixture dune does not read
#      back as `dune` in ps); a ps serving a skewed etime (a resume from
#      suspend, gh-ocannl-1319) neither ends a run nor spares a hung test, and
#      a test that execs under new names is still capped.
#  76. a GPU batch whose device query does not answer is DEVICE UNHEALTHY (69),
#      the probe being the slot's command ahead of dune and dune never started;
#      an answer or a fast failure starts dune once; an unread batch asks every
#      GPU backend, a CPU batch none; the real probe is bin/device_props, built
#      inside the slot from DUNE_BUILD_DIR's tree.
# 106. slow aliases receive a four-hour run cap before launch, including files,
#      fresh and repeat; plan states the cap source, explicit caps win,
#      and an operand after dune's separator does not raise the default.
#  77. `plan` reports the probe and the per-test cap as `run` would apply them
#      (GPU batches only, not @slow, not where run resolves nothing), and asks
#      no device.
#  91 sits after leg 84: the per-test cap keys a test on its start token too
#      (gh-ocannl-1448) -- a stub ps and the watcher's /proc seam advance a 7 s
#      test's starttime at every listing, and it is never capped under 4 s;
#      held steady, the same test is capped.
#  64-65 sit between legs 51 and 52: `plan`, and the run's first phase, in
#      which the supervisor's child resolves the backends (gh-ocannl-1066,
#      gh-ocannl-1106).
#  64. `plan` prints the resolved backends, the width (a caller's too) and
#      the slot, runs no dune, leaves no run, and is refused under a held lock.
#  53-58 sit at the end: the fleet's run-time slot, taken by the runner
#      (gh-ocannl-1004), against a fake fleet-worker.sh and the fake readers
#      every leg runs with.
#  53. the kind: --cpu only when none of the batch's resolved backends holds a
#      GPU; a GPU, empty, unknown or unreadable configuration, `exec`, a
#      command-line GPU backend, a reachable GPU stanza, or a reachability
#      answer outside its grammar is --gpu; the cap bounds the wait.
#  54. a refused or unreachable slot is SLOT REFUSED, exit 75, and dune never runs.
#  55. a red suite under a held slot is still FAIL; `start` prints the caller's argv;
#      a batch admitted inside a measurement hold is admitted, its red still FAIL.
#  56. no probe, OCANNL_TOOL_FLEET_WORKER=none, and repeat all run dune directly;
#      a stale first skill tree does not hide a second that answers the probe.
#  57. every tracked ocannl_config naming a backend is one the backends are
#      read from.
#  58. the readers batch-backends.sh builds are test/config's: ocannl_read_config
#      answers --read=backend (through test/config/config_reads.ml), ocannl_slot_kind asks Test_scan.Slot_kind and
#      prints the grammar batch_resolve parses, and Slot_kind's config_dirs is
#      BATCH_CONFIG_DIRS.
#  93 sits after leg 58: a tree carrying a backend marker env_var_deps refuses
#      (gh-ocannl-1652's chip) is MARKER REFUSED, exit 2, before the slot and
#      dune -- naming the marker's dune file and line in the digest and the
#      log, cut short or beside a `names` line too; `plan` says run would
#      refuse it and `repeat` runs no iteration; the repaired marker runs --gpu.
#  78. a test configuration's backend counts only where ocannl_slot_kind says
#      it does (gh-ocannl-1227: a batch off its configuration-free list or a
#      plain build): beside a hip configuration on a small SDMA pool, a batch
#      it answers with no `reads config` line -- naming nothing, or naming cc --
#      stays uncapped and --cpu; one it answers with one is -j 4, --gpu.
#  88. a GPU box's batch resolving to cc alone (gh-ocannl-1433) runs and WARNs
#      in the announcements, the log and the digest, and `--expect-backend`
#      refuses it before dune (BACKEND REFUSED, exit 2); the GPU backend held,
#      cc expected, a Mac, a batch holding no backend say nothing; a WSL bridge
#      names cuda or hip and recommends neither; the option
#      still refuses on a Mac and refuses an unread batch; `plan` reports both.

. "$(cd "$(dirname "$0")" && pwd)/test-test-run-common.sh"


shared_through dxg

# The injection, in all four places it has to appear: dune's argv, the RECORDED
# command (so a later digest shows the width too), stderr, and the run's log --
# the artifact whoever triages the red suite reads.
leg_29_34() {
dxg_probe cap-inject "$dxg_present" cuda run build @cheap
dxg_detail=
[ "$argv_rc" = 0 ] || dxg_detail="exit $argv_rc"
[ -n "$dxg_detail" ] || [ "$argv_calls" = "build -j 2 @cheap" ] ||
  dxg_detail="calls: ${argv_calls:-<none>}"
[ -n "$dxg_detail" ] || grep -q -- 'capping dune at -j 2' <<<"$argv_err" ||
  dxg_detail="stderr did not announce the cap: $argv_err"
[ -n "$dxg_detail" ] || grep -q -- 'box-jobs.sh' <<<"$argv_err" ||
  dxg_detail="stderr did not name the cap's source: $argv_err"
[ -n "$dxg_detail" ] || grep -q -- 'agent-notes/build-and-test.md' <<<"$argv_err" ||
  dxg_detail="stderr did not name the agent note: $argv_err"
[ -n "$dxg_detail" ] || { [ -n "$argv_dir" ] && grep -q -- 'capping dune at -j 2' "$argv_dir/log"; } ||
  dxg_detail="the run log does not carry the cap: ${argv_dir:-<no run>}"
[ -n "$dxg_detail" ] || { [ -n "$argv_dir" ] && grep -q -- '-j 2' "$argv_dir/cmd"; } ||
  dxg_detail="the recorded command does not carry the cap: $(cat "$argv_dir/cmd" 2>/dev/null)"
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: a GPU run that named no width is capped, announced and recorded"
else
  report 1 "dxg: a GPU run that named no width is capped, announced and recorded" "$dxg_detail"
fi

# A width the caller named is honored, in every spelling dune takes -- and
# `-j` PAST dune's own separator is the program's argument, not a width, so
# that one is still capped. The negative control matters: a scan that stopped
# at the wrong place would either forward two widths or cap nothing.
dxg_detail=
for probe in "sp-space:-j 8:build -j 8 @cheap" \
             "sp-glued:-j8:build -j8 @cheap" \
             "sp-long:--jobs 8:build --jobs 8 @cheap" \
             "sp-eq:--jobs=8:build --jobs=8 @cheap" \
             "sp-auto:-jauto:build -jauto @cheap"; do
  tag=${probe%%:*}; rest=${probe#*:}; want=${rest#*:}
  # shellcheck disable=SC2086
  dxg_probe "cap-$tag" "$dxg_present" cuda run build ${rest%%:*} @cheap
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "$want" ]; then
    dxg_detail="${rest%%:*}: exit $argv_rc; calls: ${argv_calls:-<none>} (want $want)"; break
  fi
  case $argv_err in
    *"names its own dune width"*"NOT injected"*) ;;
    *) dxg_detail="${rest%%:*}: the honored width was not reported: ${argv_err:-<nothing>}"; break ;;
  esac
done
if [ -z "$dxg_detail" ]; then
  dxg_probe cap-past-sep "$dxg_present" cuda run exec ./prog.exe -- -j 8
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec -j 2 ./prog.exe -- -j 8" ] ||
    dxg_detail="past the separator: exit $argv_rc; calls: ${argv_calls:-<none>}"
fi
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: a width the caller named is honored; one past dune's -- is not a width"
else
  report 1 "dxg: a width the caller named is honored; one past dune's -- is not a width" "$dxg_detail"
fi

# The two halves of the condition, each denied on its own. Silence is part of
# the contract here: an announcement on an ordinary box or a CPU suite is noise
# that trains its reader to skip the one line that matters.
dxg_detail=
for probe in "cpu:$dxg_present:cc" "cpu-multidev:$dxg_present:multidev_cc" \
             "offbox-cuda:$dxg_absent:cuda" "offbox-hip:$dxg_absent:hip"; do
  tag=${probe%%:*}; rest=${probe#*:}
  dxg_probe "cap-$tag" "${rest%%:*}" "${rest#*:}" run build @cheap
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|dxg' <<<"$argv_err"; then
    dxg_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"; break
  fi
done
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: no cap and no announcement for a CPU backend, or off a dxg host"
else
  report 1 "dxg: no cap and no announcement for a CPU backend, or off a dxg host" "$dxg_detail"
fi

# The backend no longer has to be exported to be read (gh-ocannl-1066): it is
# resolved, from the test configurations, the stanzas the argv reaches and the
# argv. With OCANNL_BACKEND unset, a batch the configurations pin to cc and
# that reaches no stanza naming a GPU is a CPU batch -- no cap, and silence;
# one whose configuration names hip, one reaching a stanza that names cuda, and
# one whose backends cannot be read are capped, with the announcement saying
# which backend and why. A caller who named a width is only told.
dxg_detail=
dxg_probe cap-unset "$dxg_present" "" run build @cheap
{ [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] && ! grep -qi 'cap\|dxg' <<<"$argv_err"; } ||
  dxg_detail="a cc batch: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
[ -n "$dxg_detail" ] || { [ -n "$argv_dir" ] && grep -q '^test-run: batch: holds cc: test/config resolves backend=cc$' "$argv_dir/log"; } ||
  dxg_detail="a cc batch: the log does not record what it holds: $(cat "$argv_dir/log" 2>/dev/null)"
for probe in "config|hip|$FAKE_READS\nend|test/config resolves backend=hip" \
             "reach|cc|names cuda: it reaches t_cuda in x, which names cuda\nend|it reaches t_cuda in x, which names cuda" \
             "unread|fail|$FAKE_READS\nend|its backends are unread"; do
  [ -z "$dxg_detail" ] || break
  IFS='|' read -r tag bt reach what <<<"$probe"
  export FAKE_BACKEND_TEST=$bt FAKE_REACH=$reach
  dxg_probe "cap-unset-$tag" "$dxg_present" "" run build @cheap
  unset FAKE_BACKEND_TEST FAKE_REACH
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build -j 2 @cheap" ]; then
    dxg_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>} (want build -j 2 @cheap)"; break
  fi
  case $(tr '\n' ' ' <<<"$argv_err") in
    *"capping dune at -j 2"*"$what"*) ;;
    *) dxg_detail="$tag: the announcement does not say why: ${argv_err:-<nothing>}"; break ;;
  esac
done
if [ -z "$dxg_detail" ]; then
  export FAKE_BACKEND_TEST=hip
  dxg_probe cap-unset-explicit "$dxg_present" "" run build -j 8 @cheap
  unset FAKE_BACKEND_TEST
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build -j 8 @cheap" ] &&
    grep -q 'names its own dune width' <<<"$argv_err"; } ||
    dxg_detail="with a width named: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
fi
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: with OCANNL_BACKEND unset the backend is resolved: a cc batch is silent, a GPU or unread one capped"
else
  report 1 "dxg: with OCANNL_BACKEND unset the backend is resolved: a cc batch is silent, a GPU or unread one capped" "$dxg_detail"
fi

# The other GPU backend, and the position the width is spliced at: immediately
# after dune's subcommand, which is where dune accepts it for every target --
# and always before dune's own `--`, past which it would be an argument to the
# program dune runs.
dxg_probe cap-hip "$dxg_present" hip run exec ./prog.exe -- alpha
dxg_detail=
[ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec -j 2 ./prog.exe -- alpha" ] ||
  dxg_detail="exit $argv_rc; calls: ${argv_calls:-<none>}"
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: hip is capped too, and the width precedes dune's own separator"
else
  report 1 "dxg: hip is capped too, and the width precedes dune's own separator" "$dxg_detail"
fi

# The anti-drift half calls the shared shipping sweep helper directly. The
# sweep and this runner must source that table, and the sweep must pass the
# resolved destination to every width lookup.
dxg_detail=
grep -q '^\. "\$SWEEP_TOOLS/box-jobs.sh"$' "$SWEEP_SRC" ||
  dxg_detail="sweep.sh does not source tools/box-jobs.sh"
[ -n "$dxg_detail" ] || grep -q '^\. tools/box-jobs\.sh$' "$SRC" ||
  dxg_detail="test-run.sh does not source tools/box-jobs.sh"
# The width depends on how the unit reached its box (gh-ocannl-1029): the two
# lab boxes dual-boot, and only a WSL boot crosses the bridge. Every call site
# must hand box_jobs_sweep_jobs the unit's RESOLVED destination -- the `@<box>`
# placeholder is replaced in UNITS before any unit runs -- since a call that
# dropped it would silently give a native boot the WSL cap. And the
# destinations are resolved by the sourced lab-map.sh's lab_dest_of (with the lab_row it
# reads), over the lab's endpoint map as `wake-lab.sh endpoint-map` printed it
# on 2026-09-27 -- a FIXTURE, never the live map, which is site data outside the
# repository and the sweep's only table of aliases (gh-ocannl-1121). So what is
# pinned is the relationship: whatever alias the sweep picks for a boot,
# box-jobs.sh classifies it as that boot. (LAB_MAP_SRC and the LAB_MAP fixture
# are set beside dxg_probe, outside this leg's gate: leg 48 reads them too.)
if [ -z "$dxg_detail" ]; then
  unit_jobs_calls=$(grep -c 'box_jobs_sweep_jobs "' "$SWEEP_SRC")
  unit_jobs_dest_calls=$(grep -c 'box_jobs_sweep_jobs "$machine" "$backend" "$host"' "$SWEEP_SRC")
  if [ "$unit_jobs_calls" -eq 0 ] || [ "$unit_jobs_calls" != "$unit_jobs_dest_calls" ]; then
    dxg_detail="sweep.sh calls box_jobs_sweep_jobs $unit_jobs_calls times, $unit_jobs_dest_calls with the unit's destination"
  fi
fi
[ -n "$dxg_detail" ] || [ -r "$LAB_MAP_SRC" ] ||
  dxg_detail="cannot read shared endpoint lookups: $LAB_MAP_SRC"
if [ -z "$dxg_detail" ]; then
  # The values themselves, from the shipping table, in one shell: the sweep's
  # hip unit over each of minix's boots (and with no destination, or one the
  # table does not know), rog's cuda unit over each of its boots (uncapped over
  # the bridge, its slot width natively), a CPU unit on a native boot, tuf's
  # native hip unit at its slot width (leg 48 pins both slot widths to the ones
  # test-run.sh injects on such a box), the override that must still win over
  # both transports, the local probe, and the two constants.
  sweep_cap=$(
    . "$JOBS_SRC"
    . "$LAB_MAP_SRC"
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' \
      "$(box_jobs_sweep_jobs minix hip)" \
      "$(box_jobs_sweep_jobs minix hip "$(lab_dest_of minix wsl)")" \
      "$(box_jobs_sweep_jobs minix hip "$(lab_dest_of minix linux)")" \
      "$(box_jobs_sweep_jobs minix hip 10.0.0.7)" \
      "$(box_jobs_sweep_jobs rog-nv cuda "$(lab_dest_of rog wsl)")" \
      "$(box_jobs_sweep_jobs rog-nv cuda "$(lab_dest_of rog linux)")" \
      "$(box_jobs_sweep_jobs minix multidev_cc "$(lab_dest_of minix linux)")" \
      "$(box_jobs_sweep_jobs tuf hip "$(lab_dest_of tuf linux)")" \
      "$(OCANNL_TOOL_SWEEP_JOBS=7 box_jobs_sweep_jobs minix hip)" \
      "$(OCANNL_TOOL_SWEEP_JOBS=7 box_jobs_sweep_jobs minix hip "$(lab_dest_of minix linux)")" \
      "$(box_jobs_sweep_jobs m4-max metal)" \
      "$(OCANNL_TOOL_DXG_DEVICE=$dxg_present box_jobs_local_cap cuda)" \
      "$BOX_JOBS_DXG_CAP/$BOX_JOBS_SDMA_CAP"
  )
  case $sweep_cap in
    "2|2|8|2||8||8|7|7||2|2/8") ;;
    *) dxg_detail="shared table disagrees: $sweep_cap (want 2|2|8|2||8||8|7|7||2|2/8)" ;;
  esac
fi
if [ -z "$dxg_detail" ]; then
  # Every alias the sweep can resolve to is one box-jobs.sh classifies, and
  # classifies as the boot lab_dest_of says it is.
  for box in rog minix; do
    for kind in wsl linux; do
      got=$(. "$JOBS_SRC"; . "$LAB_MAP_SRC"; box_jobs_dest_transport "$(lab_dest_of "$box" "$kind")")
      case $kind:$got in
        wsl:dxg | linux:native) ;;
        *) dxg_detail="$box's $kind alias classifies as '${got}'" ;;
      esac
    done
  done
fi
if [ -z "$dxg_detail" ]; then
  report 0 "dxg: the sweep's box_jobs_sweep_jobs and the injected cap come from one table"
else
  report 1 "dxg: the sweep's box_jobs_sweep_jobs and the injected cap come from one table" "$dxg_detail"
fi
}
harness_leg_run '29-34: the dxg width cap #width' leg_29_34

shared_through native

# Leg 48: the injection on each native hazard, in the four places the dxg leg
# pins it -- argv, the recorded command, stderr (with its source and the note)
# and the run's log -- plus what the announcement says the hazard was. The
# fleet's state records rog's slots here, which the CPU rows need.
leg_48() {
native_detail=
native_fleet=$fleet_rog
# The last field is the sweep unit the same box runs, as `<machine>/<lab box>`:
# a sweep unit takes no correctness slot, so on tuf and rog it must run at the
# slot width injected here, and its width is compared with the injected one
# rather than with a number (gh-ocannl-1211's 2026-10-14 sweep ran tuf's unit 16
# wide). minix's unit is the deliberate exception: one unit's SDMA cap, not its
# slot width (tools/box-jobs.sh, box_jobs_sweep_cap).
for probe in "sdma:$kfd_small:$nv_absent:hip:4:SDMA:-" \
             "wide-sdma:$kfd_large:$nv_absent:hip:8:AMD GPU reports 12 allocatable SDMA:tuf/tuf" \
"nvidia:$kfd_absent:$nv_present:cuda:8:native NVIDIA boot:rog-nv/rog" \
             "nvidia-cc:$kfd_absent:$nv_present:cc:8:rog-nv-linux, natively booted:-" \
             "nvidia-multidev:$kfd_absent:$nv_present:multidev_cc:8:execution slot --cpu:-"; do
  IFS=: read -r tag kfd nv backend want what sweep_unit <<<"$probe"
  native_probe "native-$tag" "$dxg_absent" "$kfd" "$nv" "$backend" run build @cheap
  [ "$argv_rc" = 0 ] || { native_detail="$tag: exit $argv_rc"; break; }
  [ "$argv_calls" = "build -j $want @cheap" ] ||
    { native_detail="$tag: calls: ${argv_calls:-<none>} (want build -j $want @cheap)"; break; }
  case $argv_err in
    *"capping dune at -j $want"*"$what"*"box-jobs.sh"*"agent-notes/build-and-test.md"*) ;;
    *) native_detail="$tag: the announcement is not the native one: ${argv_err:-<nothing>}"; break ;;
  esac
  { [ -n "$argv_dir" ] && grep -q -- "capping dune at -j $want" "$argv_dir/log"; } ||
    { native_detail="$tag: the run log does not carry the cap: ${argv_dir:-<no run>}"; break; }
  { [ -n "$argv_dir" ] && grep -q -- "-j $want" "$argv_dir/cmd"; } ||
    { native_detail="$tag: the recorded command does not carry the cap: $(cat "$argv_dir/cmd" 2>/dev/null)"; break; }
  if [ "$sweep_unit" != - ]; then
    sweep_width=$(unset OCANNL_TOOL_SWEEP_JOBS; . "$JOBS_SRC"; . "$LAB_MAP_SRC"
      box_jobs_sweep_jobs "${sweep_unit%/*}" "$backend" "$(lab_dest_of "${sweep_unit#*/}" linux)")
    [ "build -j $sweep_width @cheap" = "$argv_calls" ] ||
      { native_detail="$tag: the sweep's native ${sweep_unit%/*}/$backend unit runs at '${sweep_width}', not the injected width ($argv_calls)"; break; }
  fi
done
# The hostname path of the fleet's identity rule: with FLEET_LOCAL_BOX unset, a
# map whose glob matches any host names it rog, and one matching none does not.
native_fleet=
for probe in "*=rog-nv-linux:build -j 8 @cheap" "no-such-host-*=rog-nv-linux:build @cheap"; do
  [ -z "$native_detail" ] || break
  native_map=${probe%%:*}
  native_probe native-hostname "$dxg_absent" "$kfd_absent" "$nv_present" cc run build @cheap
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "${probe#*:}" ] ||
    native_detail="hostname map '$native_map': exit $argv_rc; calls: ${argv_calls:-<none>} (want ${probe#*:})"
done
native_map=
native_fleet=$fleet_none
if [ -z "$native_detail" ]; then
  report 0 "native: hip on a small or a larger SDMA pool, and cuda and cc on a native NVIDIA boot, are capped, announced and recorded"
else
  report 1 "native: hip on a small or a larger SDMA pool, and cuda and cc on a native NVIDIA boot, are capped, announced and recorded" "$native_detail"
fi
}
harness_leg_run '48: native GPU width caps injected #width' leg_48

# Leg 49: the negative controls. Each denies one half of a condition leg 48
# met, so a probe that capped every Linux GPU batch -- or keyed on the backend
# alone -- fails here. Silence is part of the contract, as for the dxg legs.
leg_49() {
native_detail=
for probe in "cpu-on-wide-sdma:$kfd_large:$nv_absent:cc" \
             "cuda-on-wide-sdma:$kfd_large:$nv_absent:cuda" \
             "no-kfd:$kfd_absent:$nv_absent:hip" \
             "no-pool:$kfd_nopool:$nv_absent:hip" \
             "cpu-on-sdma:$kfd_small:$nv_absent:cc" \
             "cpu-on-unfleeted-nvidia:$kfd_absent:$nv_present:cc" \
             "cuda-on-sdma:$kfd_small:$nv_absent:cuda" \
             "hip-on-nvidia:$kfd_absent:$nv_present:hip" \
             "cuda-off-nvidia:$kfd_absent:$nv_absent:cuda"; do
  IFS=: read -r tag kfd nv backend <<<"$probe"
  native_probe "native-$tag" "$dxg_absent" "$kfd" "$nv" "$backend" run build @cheap
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|sdma\|nvidia\|dxg' <<<"$argv_err"; then
    native_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"; break
  fi
done
# A WSL boot of the fleet's rog: the bridge carries only the GPU, so a CPU
# batch there is not the native slots' and meets no cap.
if [ -z "$native_detail" ]; then
  native_fleet=$fleet_rog
  native_probe native-cpu-on-nvidia-dxg "$dxg_present" "$kfd_absent" "$nv_present" cc run build @cheap
  native_fleet=$fleet_none
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build @cheap" ] ||
     grep -qi 'cap\|sdma\|nvidia\|dxg' <<<"$argv_err"; then
    native_detail="cpu-on-nvidia-dxg: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
  fi
fi
if [ -z "$native_detail" ]; then
  report 0 "native: no cap and no announcement off the measured hazards, or for the other backend"
else
  report 1 "native: no cap and no announcement off the measured hazards, or for the other backend" "$native_detail"
fi
}
harness_leg_run '49: native width-cap negative controls #width' leg_49

# Leg 50: an explicit width still wins on a native hazard, and says so; and a
# WSL boot keeps the bridge's cap even where its topology would read as a
# small pool (the bridge is the tighter limit, and the one measured there).
leg_50() {
native_detail=
for probe in "sdma:$kfd_small:$nv_absent:hip:small-SDMA-pool host" \
             "wide-sdma:$kfd_large:$nv_absent:hip:native AMD GPU host" \
             "nvidia:$kfd_absent:$nv_present:cuda:native NVIDIA host" \
             "nvidia-cpu:$kfd_absent:$nv_present:cc:rog-nv-linux"; do
  IFS=: read -r tag kfd nv backend what <<<"$probe"
  native_fleet=$fleet_rog
  native_probe "native-explicit-$tag" "$dxg_absent" "$kfd" "$nv" "$backend" run build -j 16 @cheap
  native_fleet=$fleet_none
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build -j 16 @cheap" ]; then
    native_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>} (want build -j 16 @cheap)"; break
  fi
  case $argv_err in
    *"$what"*"names its own dune width"*"NOT injected"*) ;;
    *) native_detail="$tag: the honored width was not reported: ${argv_err:-<nothing>}"; break ;;
  esac
done
if [ -z "$native_detail" ]; then
  native_probe native-dxg-wins "$dxg_present" "$kfd_small" "$nv_present" hip run build @cheap
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build -j 2 @cheap" ] &&
    grep -q 'WSL2' <<<"$argv_err"; } ||
    native_detail="dxg with a small pool: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: ${argv_err:-<none>}"
fi
if [ -z "$native_detail" ]; then
  report 0 "native: a named width is honored on each native hazard; a dxg boot keeps its own cap"
else
  report 1 "native: a named width is honored on each native hazard; a dxg boot keeps its own cap" "$native_detail"
fi
}
harness_leg_run '50: a named width wins on a native hazard #width' leg_50

# Leg 51: the backend resolved on a native hazard (gh-ocannl-1066). With
# OCANNL_BACKEND unset the width follows what the batch can hold: a batch the
# configurations pin to cc is uncapped on the AMD boxes and on an NVIDIA box
# that is not the fleet's rog; one whose configuration names hip -- the minix
# shape the issue names, reached through the configuration rather than the
# environment -- gets hip's width on either pool, as does one reaching a
# stanza that names hip, and one whose backends cannot be read; on the fleet's
# rog a cc batch gets the CPU width; and beside a small AMD pool there, a batch
# that can hold both hip and cc takes the tighter of the two. The recorded
# command carries the width, and a named width is honored and only told.
leg_51() {
native_detail=
for probe in "cc-on-sdma|$kfd_small|$nv_absent|$fleet_none|cc|$FAKE_READS\nend|build @cheap|" \
             "cc-on-wide-sdma|$kfd_large|$nv_absent|$fleet_none|cc|$FAKE_READS\nend|build @cheap|" \
             "cc-on-unfleeted-nvidia|$kfd_absent|$nv_present|$fleet_none|cc|$FAKE_READS\nend|build @cheap|" \
             "hip-config-sdma|$kfd_small|$nv_absent|$fleet_none|hip|$FAKE_READS\nend|build -j 4 @cheap|SDMA*test/config resolves backend=hip" \
             "hip-config-wide-sdma|$kfd_large|$nv_absent|$fleet_none|hip|$FAKE_READS\nend|build -j 8 @cheap|12 allocatable SDMA*test/config resolves backend=hip" \
             "hip-reach-sdma|$kfd_small|$nv_absent|$fleet_none|cc|names hip: it reaches t_hip in x, which names hip\nend|build -j 4 @cheap|SDMA*it reaches t_hip in x" \
             "unread-sdma|$kfd_small|$nv_absent|$fleet_none|fail|$FAKE_READS\nend|build -j 4 @cheap|SDMA*its backends are unread" \
             "cc-on-rog|$kfd_absent|$nv_present|$fleet_rog|cc|$FAKE_READS\nend|build -j 8 @cheap|rog-nv-linux, natively booted*test/config resolves backend=cc*execution slot --cpu" \
             "tightest-on-rog|$kfd_small|$nv_present|$fleet_rog|cc|names hip: it reaches t_hip in x, which names hip\n$FAKE_READS\nend|build -j 4 @cheap|SDMA*it reaches t_hip in x"; do
  IFS='|' read -r tag kfd nv native_fleet bt reach want what <<<"$probe"
  export FAKE_BACKEND_TEST=$bt FAKE_REACH=$reach
  native_probe "native-unset-$tag" "$dxg_absent" "$kfd" "$nv" "" run build @cheap
  unset FAKE_BACKEND_TEST FAKE_REACH
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "$want" ]; then
    native_detail="$tag: exit $argv_rc; calls: ${argv_calls:-<none>} (want $want)"; break
  fi
  if [ -z "$what" ]; then
    ! grep -qi 'cap\|sdma\|nvidia' <<<"$argv_err" ||
      { native_detail="$tag: an uncapped batch was told something: $argv_err"; break; }
  else
    # shellcheck disable=SC2254  # the expected phrases are a glob by design
    case $(tr '\n' ' ' <<<"$argv_err") in
      *"capping dune at -j "*$what*) ;;
      *) native_detail="$tag: the announcement does not say why: ${argv_err:-<nothing>}"; break ;;
    esac
    { [ -n "$argv_dir" ] && grep -q -- "${want% @cheap}" "$argv_dir/cmd"; } ||
      { native_detail="$tag: the recorded command does not carry the cap: $(cat "$argv_dir/cmd" 2>/dev/null)"; break; }
  fi
done
for probe in "hip-config-sdma|hip|names its own dune width" "cc-on-sdma|cc|"; do
  [ -z "$native_detail" ] || break
  IFS='|' read -r tag bt what <<<"$probe"
  native_fleet=$fleet_none
  export FAKE_BACKEND_TEST=$bt
  native_probe "native-unset-explicit-$tag" "$dxg_absent" "$kfd_small" "$nv_absent" "" run build -j 16 @cheap
  unset FAKE_BACKEND_TEST
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "build -j 16 @cheap" ]; then
    native_detail="$tag with a width named: exit $argv_rc; calls: ${argv_calls:-<none>}"; break
  fi
  if [ -n "$what" ]; then
    grep -q "$what" <<<"$argv_err" || { native_detail="$tag with a width named: not told: ${argv_err:-<nothing>}"; break; }
  else
    ! grep -qi 'cap\|sdma\|nvidia' <<<"$argv_err" ||
      { native_detail="$tag with a width named: told something: $argv_err"; break; }
  fi
done
native_fleet=$fleet_none
if [ -z "$native_detail" ]; then
  report 0 "native: with OCANNL_BACKEND unset the width follows the batch's resolved backends, the tightest of them"
else
  report 1 "native: with OCANNL_BACKEND unset the width follows the batch's resolved backends, the tightest of them" "$native_detail"
fi
}
harness_leg_run '51: the native width follows the resolved backends #width' leg_51

# Leg 64: `plan` prints what `run` would do with an argv -- the resolved
# backends and why, the width, the slot -- and runs nothing: no dune call, no
# run recorded. Driven on the minix shape of leg 51 (a configuration naming
# hip on a small SDMA pool).
leg_64() {
native_detail=
export FAKE_BACKEND_TEST=hip
native_probe native-plan "$dxg_absent" "$kfd_small" "$nv_absent" "" plan build @cheap
unset FAKE_BACKEND_TEST
{ [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && [ -z "$argv_dir" ]; } ||
  native_detail="exit $argv_rc; dune calls: ${argv_calls:-<none>}; run: ${argv_dir:-<none>}; stderr: $argv_err"
for line in "command: dune build -j 4 @cheap" "backends: hip, cc" "holds hip: test/config resolves backend=hip" \
            "width: -j 4, injected (sdma hazard, for hip)" "slot: none"; do
  [ -z "$native_detail" ] || break
  grep -qF -- "$line" <<<"$argv_out" || native_detail="no '$line' in: $argv_out"
done
# A width the caller named is reported as the caller's, with or without a cap
# on the box (Codex review round 1 on PR #832).
for probe in "$kfd_small|hip|width: the caller's (a -j 4 cap applies here, for hip)" \
             "$kfd_absent|cc|width: the caller's (no backend of the batch meets a cap on this box)"; do
  [ -z "$native_detail" ] || break
  IFS='|' read -r kfd bt want <<<"$probe"
  export FAKE_BACKEND_TEST=$bt
  native_probe native-plan-explicit "$dxg_absent" "$kfd" "$nv_absent" "" plan build -j 16 @cheap
  unset FAKE_BACKEND_TEST
  { [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && grep -qF -- "$want" <<<"$argv_out" &&
    grep -qF -- "command: dune build -j 16 @cheap" <<<"$argv_out"; } ||
    native_detail="explicit width ($bt): exit $argv_rc; stdout: $argv_out"
done
# It holds the worktree lock like a launch, under a run directory it never
# publishes and removes: none is left behind, and a plan while a run holds
# the worktree is refused.
if [ -z "$native_detail" ]; then
  ls -d "$TMP/argv-runs-native-plan"/2*Z-* >/dev/null 2>&1 &&
    native_detail="plan left a run directory: $(ls -d "$TMP/argv-runs-native-plan"/2*Z-*)"
fi
if [ -z "$native_detail" ]; then
  plan_lock_runs=$TMP/argv-runs-plan-locked
  mkdir -p "$plan_lock_runs"
  plan_lock=$(OCANNL_TOOL_TEST_RUNS=$plan_lock_runs "$repeat_root/tools/test-run.sh" paths lock)
  perl -e 'use Fcntl ":flock"; open(my $fh, ">>", $ARGV[0]) or exit 1; flock($fh, LOCK_EX) or exit 1;
           open(my $r, ">", $ARGV[1]); close $r; sleep 20' "$plan_lock" "$TMP/plan-lock.ready" &
  plan_lock_pid=$!
  for _ in $(seq 1 50); do [ -e "$TMP/plan-lock.ready" ] && break; sleep 0.1; done
  argv_runs=$plan_lock_runs native_probe native-plan-locked "$dxg_absent" "$kfd_small" "$nv_absent" "" plan build @cheap
  argv_runs=
  kill "$plan_lock_pid" 2>/dev/null
  wait "$plan_lock_pid" 2>/dev/null
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q "another test-run is active" <<<"$argv_err"; } ||
    native_detail="plan under a held lock: exit $argv_rc; stderr: $argv_err; stdout: $argv_out"
fi
# A resolution that spends the plan's cap is forecast as the cap's verdict,
# the way run records it (Codex review round 4 on PR #832).
if [ -z "$native_detail" ]; then
  export FAKE_REACH_SLEEP=30
  plan_spent_start=$SECONDS
  native_probe native-plan-spent "$dxg_absent" "$kfd_small" "$nv_absent" "" plan --cap 2 build @cheap
  unset FAKE_REACH_SLEEP
  { [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && [ $((SECONDS - plan_spent_start)) -lt 15 ] &&
    grep -q "used the whole cap" <<<"$argv_out"; } ||
    native_detail="plan with its cap spent: exit $argv_rc; stdout: $argv_out"
fi
# A plan cancelled while a TERM-ignoring reader runs waits for that reader's
# grace and KILL before it exits, so the worktree is idle the moment it is
# gone (Codex review round 4 on PR #832).
if [ -z "$native_detail" ]; then
  plan_term_runs=$TMP/argv-runs-plan-term
  mkdir -p "$plan_term_runs"
  : >"$TMP/plan-term.reach"
  FAKE_REACH_CALLS=$TMP/plan-term.reach FAKE_REACH_SLEEP=30 FAKE_REACH_IGNORE_TERM=1 FAKE_BACKEND_TEST=hip \
  OCANNL_TOOL_TEST_RUNS=$plan_term_runs OCANNL_TOOL_DXG_DEVICE=$dxg_absent \
  OCANNL_TOOL_KFD_TOPOLOGY=$kfd_small OCANNL_TOOL_NVIDIA_DEVICE=$nv_absent PATH=$repeat_bin:$PATH \
    "$repeat_root/tools/test-run.sh" plan build @cheap >"$TMP/plan-term.out" 2>"$TMP/plan-term.err" &
  plan_term_pid=$!
  plan_term_marker=
  for _ in $(seq 1 100); do
    plan_term_marker=$(ls "$plan_term_runs"/2*/pid 2>/dev/null | head -n 1)
    [ -n "$plan_term_marker" ] && [ -s "$TMP/plan-term.reach" ] && break
    sleep 0.1
  done
  kill -TERM "$plan_term_pid" 2>/dev/null
  wait "$plan_term_pid"
  plan_term_rc=$?
  { [ -s "$TMP/plan-term.reach" ] && [ -n "$plan_term_marker" ] && [ "$plan_term_rc" = 143 ] && [ ! -e "$(dirname "$plan_term_marker")" ] &&
    OCANNL_TOOL_TEST_RUNS=$plan_term_runs "$repeat_root/tools/test-run.sh" idle 2>/dev/null; } ||
    native_detail="a cancelled plan: marker ${plan_term_marker:-<none>}; exit $plan_term_rc (want 143); or not idle once it exited: $(cat "$TMP/plan-term.err")"
fi
if [ -z "$native_detail" ]; then
  report 0 "plan: prints the resolved backends, the width and the slot, and runs nothing"
else
  report 1 "plan: prints the resolved backends, the width and the slot, and runs nothing" "$native_detail"
fi
}
harness_leg_run '64: plan prints what run would do #width' leg_64

# Leg 52: one place for the numbers. A hip cap is not a literal of its own
# but its measured budget over its box's slot count, so a change to either
# moves it; and the values are the measured ones (lukstafi/ludics-lite#316,
# #344, #391), pinned here so an edit to box-jobs.sh has to say what it measured.
leg_52() {
native_consts=$(
  . "$JOBS_SRC"
  printf '%s/%s/%s/%s/%s/%s/%s/%s/%s/%s/%s/%s' "$BOX_JOBS_SDMA_CAP" \
    "$BOX_JOBS_SDMA_BUDGET" "$BOX_JOBS_SDMA_SLOTS" "$BOX_JOBS_SDMA_SLOT_CAP" \
    "$BOX_JOBS_WIDE_SDMA_BUDGET" "$BOX_JOBS_WIDE_SDMA_SLOTS" "$BOX_JOBS_WIDE_SDMA_SLOT_CAP" \
    "$BOX_JOBS_NATIVE_NVIDIA_SLOTS" "$BOX_JOBS_NATIVE_CUDA_TOKENS" "$BOX_JOBS_NATIVE_CUDA_CAP" \
    "$BOX_JOBS_NATIVE_CPU_CAP" "$BOX_JOBS_SDMA_MEASURED_POOL"
)
native_detail=
[ "$native_consts" = "8/16/4/4/24/3/8/4/2/8/8/6" ] ||
  native_detail="box-jobs.sh's native numbers are $native_consts (want 8/16/4/4/24/3/8/4/2/8/8/6)"
[ -n "$native_detail" ] ||
  grep -q '^BOX_JOBS_SDMA_SLOT_CAP=\$((BOX_JOBS_SDMA_BUDGET / BOX_JOBS_SDMA_SLOTS))$' "$JOBS_SRC" ||
  native_detail="BOX_JOBS_SDMA_SLOT_CAP is not derived from the SDMA budget and the slot count"
[ -n "$native_detail" ] ||
  grep -q '^BOX_JOBS_WIDE_SDMA_SLOT_CAP=\$((BOX_JOBS_WIDE_SDMA_BUDGET / BOX_JOBS_WIDE_SDMA_SLOTS))$' "$JOBS_SRC" ||
  native_detail="BOX_JOBS_WIDE_SDMA_SLOT_CAP is not derived from the wide-pool budget and the slot count"
if [ -z "$native_detail" ]; then
  report 0 "native: the per-slot widths derive from box-jobs.sh's measured numbers"
else
  report 1 "native: the per-slot widths derive from box-jobs.sh's measured numbers" "$native_detail"
fi
}
harness_leg_run '52: the native widths come from one place #width' leg_52

shared_through slot

# Leg 53: the kind. --cpu only when none of the batch's resolved backends
# holds a GPU: a GPU backend in a configuration, none (Context.auto tries GPUs
# first), an unreadable or unknown one, a reached stanza naming a GPU, an
# answer the reachability tool cut short, and a dune argv that can pick a
# backend the configurations do not show are each --gpu -- the slot's
# fail-closed default. The fake's recorded call is the argv the runner handed
# the real slot, so it pins the wait (the cap bounds it) and that dune's own
# argv passes through intact.
leg_53() {
slot_detail=
for probe in "cpu:cc:sync_cc:--cpu" "unset:unset:cc:--gpu" "multidev:multidev_cc:cc:--cpu" \
             "gpu-test:cuda:cc:--gpu" "gpu-arrayjit:cc:hip:--gpu" "metal:cc:metal:--gpu" \
             "unknown:cc:vulkan:--gpu" "unreadable:cc:fail:--gpu"; do
  IFS=: read -r tag bt ba want <<<"$probe"
  slot_probe "slot-$tag" hold "$bt" "$ba" run build @cheap
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] ||
    { slot_detail="$tag: exit $argv_rc; dune calls: ${argv_calls:-<none>}"; break; }
  [ "$slot_calls" = "execution slot --wait 600 $want -- dune build @cheap" ] ||
    { slot_detail="$tag: slot call: ${slot_calls:-<none>} (want $want)"; break; }
  grep -q "fleet correctness slots.*as $want" <<<"$(tr '\n' ' ' <<<"$argv_err")" ||
    { slot_detail="$tag: the slot and its kind were not announced: $argv_err"; break; }
  { grep -q 'if no enclosing slot is reusable' <<<"$argv_err" &&
    grep -q 'reuses an enclosing slot without taking an additional slot or GPU token' <<<"$argv_err" &&
    grep -q 'a GPU batch inside a CPU-only slot is refused' <<<"$argv_err"; } ||
    { slot_detail="$tag: the announcement does not distinguish acquisition from validated reuse: $argv_err"; break; }
  { [ -n "$argv_dir" ] && grep -q -- "as $want" "$argv_dir/log" &&
      grep -q "^test-run: batch: " "$argv_dir/log" &&
      grep -q "^EXECUTION SLOT fakebox: slot 1 of 4" "$argv_dir/log"; } ||
    { slot_detail="$tag: the log does not record the backends, the decision and the slot: $(cat "$argv_dir/log" 2>/dev/null)"; break; }
done
for probe in "exec:exec ./prog.exe" "flag-gpu:exec ./prog.exe -- --ocannl_backend=cuda" \
             "flag-build:build @cheap --ocannl-backend=hip" "flag-bare:build @cheap --ocannl-backend"; do
  [ -z "$slot_detail" ] || break
  IFS=: read -r tag argv <<<"$probe"
  # shellcheck disable=SC2086  # the argv is word-split on purpose
  slot_probe "slot-argv-$tag" hold cc cc run $argv
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune $argv" ] ||
    slot_detail="argv $tag: slot call: ${slot_calls:-<none>} (want --gpu)"
done
# A run that reaches a stanza naming a GPU backend is --gpu however the
# configurations resolve, and the reachability question is asked with the
# caller's own dune argv; one reaching only a stanza that names a CPU backend
# stays --cpu.
if [ -z "$slot_detail" ]; then
  FAKE_REACH="names cuda: it reaches test_cuda_pool_offset in test/operations, which names cuda\nend" \
    slot_probe slot-reach-gpu hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune build @cheap" ] ||
    slot_detail="reach: slot call: ${slot_calls:-<none>} (want --gpu)"
  [ -n "$slot_detail" ] || [ "$(cat "$TMP/slot-reach-gpu.reach")" = "build @cheap" ] ||
    slot_detail="reach: asked about: $(cat "$TMP/slot-reach-gpu.reach") (want build @cheap)"
  [ -n "$slot_detail" ] || grep -q "batch: holds cuda: it reaches test_cuda_pool_offset" "$argv_dir/log" ||
    slot_detail="reach: the log does not say why: $(cat "$argv_dir/log")"
fi
if [ -z "$slot_detail" ]; then
  FAKE_REACH="names cc: it reaches cc_backend_trace_name in test/operations, which names cc\nend" \
    slot_probe slot-reach-cpu hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap" ] ||
    slot_detail="reach of a CPU-named stanza: slot call: ${slot_calls:-<none>} (want --cpu)"
fi
# Every answer the grammar does not accept is every backend: garbled, cut
# short of its `end`, running past it, naming what is not a backend, or the
# tool's own unknown.
for probe in "garbled:garbled" "cut-short:names cc: it reaches x in y, which names cc" \
             "past-end:end\nnames cuda: it reaches x in y, which names cuda" \
             "unknown-name:names vulkan: it reaches x in y, which names vulkan\nend" \
             "unmodelled:unknown: it carries a word this does not model"; do
  [ -z "$slot_detail" ] || break
  FAKE_REACH=${probe#*:} slot_probe "slot-reach-${probe%%:*}" hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune build @cheap" ] ||
    slot_detail="an unreadable reachability answer (${probe%%:*}): slot call: ${slot_calls:-<none>} (want --gpu)"
done
# A reachability read that fails is unread whatever it printed -- even a
# complete, CPU-safe `end` (Codex review round 5 on PR #832).
if [ -z "$slot_detail" ]; then
  FAKE_REACH_EXIT=1 slot_probe slot-reach-failed hold cc cc run build @cheap
  [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune build @cheap" ] ||
    slot_detail="a failed reachability read: slot call: ${slot_calls:-<none>} (want --gpu)"
fi
if [ -z "$slot_detail" ]; then
  slot_probe slot-argv-cpu-flag hold cc cc run build @cheap --ocannl_backend=cc
  [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap --ocannl_backend=cc" ] ||
    slot_detail="a command-line CPU backend: slot call: ${slot_calls:-<none>} (want --cpu)"
  # ...and the reachability tool is asked without the flag the argv scan
  # already read (Codex review round 6 on PR #832).
  [ -n "$slot_detail" ] || [ "$(cat "$TMP/slot-argv-cpu-flag.reach")" = "build @cheap" ] ||
    slot_detail="a command-line CPU backend: reachability asked about: $(cat "$TMP/slot-argv-cpu-flag.reach")"
fi
if [ -z "$slot_detail" ]; then
  # At most half the cap, so a refusal returns before the cap's alarm.
  # Half of what the resolution left of the cap: 45, or 44 when resolving
  # crossed a second (Codex review round 3 on PR #832).
  slot_probe slot-capped hold cc cc run --cap 90 build @cheap
  case $slot_calls in
    "execution slot --wait 45 --cpu -- dune build @cheap" | "execution slot --wait 44 --cpu -- dune build @cheap") ;;
    *) slot_detail="capped: the slot's wait is not half the cap: ${slot_calls:-<none>}" ;;
  esac
fi
if [ -z "$slot_detail" ]; then
  report 0 "slot: a fleet box's run takes its slot, --cpu only when none of the batch's resolved backends holds a GPU"
else
  report 1 "slot: a fleet box's run takes its slot, --cpu only when none of the batch's resolved backends holds a GPU" "$slot_detail"
fi
}
harness_leg_run '53: the slot kind #slot' leg_53

# Leg 54: a refused or unreachable slot is its own verdict, never a test one:
# exit 75, dune never invoked, and the fleet's own line quoted in the log.
leg_54() {
slot_detail=
for mode in refuse unreachable; do
  slot_probe "slot-$mode" "$mode" cc cc run build @cheap
  [ "$argv_rc" = 75 ] || { slot_detail="$mode: exit $argv_rc (want 75): $argv_out"; break; }
  [ -z "$argv_calls" ] || { slot_detail="$mode: dune ran: $argv_calls"; break; }
  case $argv_out in
    *"verdict: SLOT REFUSED"*) ;;
    *) slot_detail="$mode: verdict: $argv_out"; break ;;
  esac
done
if [ -z "$slot_detail" ]; then
  report 0 "slot: a refused or unreachable slot reports SLOT REFUSED (exit 75), and dune never runs"
else
  report 1 "slot: a refused or unreachable slot reports SLOT REFUSED (exit 75), and dune never runs" "$slot_detail"
fi
}
harness_leg_run '54: a refused or unreachable slot is SLOT REFUSED #slot' leg_54

# Leg 55a: `start` prints the caller's dune command, not the supervisor's.
leg_55a() {
slot_probe slot-start hold cc cc start build @cheap
case $argv_out in
  *"execution slot"* | *"fake-fleet-worker"*) report 1 "slot: start prints the caller's command" "$argv_out" ;;
  *"command: dune build @cheap"*) report 0 "slot: start prints the caller's command" ;;
  *) report 1 "slot: start prints the caller's command" "${argv_out:-<nothing>}" ;;
esac
OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-slot-start "$repeat_root/tools/test-run.sh" wait last >/dev/null 2>&1
}
harness_leg_run '55a: start prints the caller'\''s command #slot' leg_55a

# Leg 55: a red suite under a slot is still FAIL -- the slot's admission line
# is what tells the two apart, not the status.
leg_55() {
argv_mode=red slot_probe slot-red hold cc cc run build @cheap
case "$argv_rc:$argv_out" in
  1:*"verdict: FAIL"*) report 0 "slot: a red suite under a held slot is still FAIL" ;;
  *) report 1 "slot: a red suite under a held slot is still FAIL" "exit $argv_rc: $argv_out" ;;
esac
}
harness_leg_run '55: a red suite under a held slot is still FAIL #slot' leg_55

# Leg 55b: inside a live measurement hold the slot admits the batch under the
# hold (`inside measurement <id>`, lukstafi/ludics-lite#480) -- an admission
# like a held slot's, so a refusal line the batch itself logs leaves a red
# suite FAIL, never SLOT REFUSED. The probe's trailing `measurement <id>` still
# reads as a fleet box.
leg_55b() {
argv_mode=red slot_probe slot-measurement measurement cc cc run build @cheap
case "$argv_rc:$argv_out" in
  1:*"verdict: FAIL"*)
    if [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap" ] &&
       grep -q "^EXECUTION SLOT fakebox: inside measurement M-7" "$argv_dir/log"; then
      report 0 "slot: a batch admitted inside a measurement hold is admitted, not SLOT REFUSED"
    else
      report 1 "slot: a batch admitted inside a measurement hold is admitted, not SLOT REFUSED" \
        "slot call: ${slot_calls:-<none>}; log: $(cat "$argv_dir/log" 2>/dev/null)"
    fi ;;
  *) report 1 "slot: a batch admitted inside a measurement hold is admitted, not SLOT REFUSED" \
       "exit $argv_rc: $argv_out" ;;
esac
}
harness_leg_run '55b: a batch inside a measurement hold is admitted #slot' leg_55b

# Leg 56: no slot where the fleet cannot give one safely -- a fleet-worker.sh
# without the probe (a version from before nested slots, around which a
# worker's own wrapper would cost two slots), the slot turned off, and repeat
# -- each runs dune directly and says nothing about a slot.
leg_56() {
slot_detail=
slot_probe slot-noprobe noprobe cc cc run build @cheap
{ [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] && [ -z "$slot_calls" ] &&
    ! grep -qi 'slot' <<<"$argv_err"; } ||
  slot_detail="noprobe: exit $argv_rc; slot calls: ${slot_calls:-<none>}; stderr: $argv_err"
if [ -z "$slot_detail" ]; then
  : >"$TMP/slot-off.fw"
  FAKE_FW_CALLS=$TMP/slot-off.fw OCANNL_TOOL_FLEET_WORKER=none argv_probe slot-off run build @cheap
  { [ "$argv_rc" = 0 ] && [ ! -s "$TMP/slot-off.fw" ]; } ||
    slot_detail="none: exit $argv_rc; fleet-worker called: $(cat "$TMP/slot-off.fw")"
fi
# Two skill trees, the first without the probe: the second one answers, and
# its slot is taken -- a stale first candidate must not turn the slot off.
if [ -z "$slot_detail" ]; then
  slot_home=$TMP/slot-home
  mkdir -p "$slot_home/.claude/skills/issue-wave/scripts" "$slot_home/.codex/skills/issue-wave/scripts"
  printf '#!/usr/bin/env bash\necho "usage" >&2; exit 2\n' >"$slot_home/.claude/skills/issue-wave/scripts/fleet-worker.sh"
  cp "$slot_fake" "$slot_home/.codex/skills/issue-wave/scripts/fleet-worker.sh"
  chmod +x "$slot_home/.claude/skills/issue-wave/scripts/fleet-worker.sh" "$slot_home/.codex/skills/issue-wave/scripts/fleet-worker.sh"
  : >"$TMP/slot-second.fw"
  HOME=$slot_home FAKE_FW_CALLS=$TMP/slot-second.fw FAKE_FW_MODE=hold FAKE_BACKEND_TEST=cc FAKE_BACKEND_ARRAYJIT=cc \
  FAKE_REACH_CALLS=$TMP/slot-second.reach \
  OCANNL_TOOL_FLEET_WORKER= argv_probe slot-second run build @cheap
  grep -q -- "execution slot --wait 600 --cpu -- dune build @cheap" "$TMP/slot-second.fw" ||
    slot_detail="second candidate: its slot was not taken: $(cat "$TMP/slot-second.fw")"
fi
if [ -z "$slot_detail" ]; then
  slot_probe slot-repeat hold cc cc repeat 2 build @cheap
  { [ "$argv_rc" = 0 ] && [ -z "$slot_calls" ]; } ||
    slot_detail="repeat: exit $argv_rc; slot calls: ${slot_calls:-<none>}"
fi
if [ -z "$slot_detail" ]; then
  report 0 "slot: no probe, the slot turned off, and repeat all run dune directly"
else
  report 1 "slot: no probe, the slot turned off, and repeat all run dune directly" "$slot_detail"
fi
}
harness_leg_run '56: no slot where the fleet cannot give one safely #slot' leg_56

# Leg 57: the configurations the backends are read from are the ones that set
# a backend. Every tracked `ocannl_config` naming one must sit in one of
# batch-backends.sh's BATCH_CONFIG_DIRS (the test directories copy theirs from
# test/config), so a new directory with a backend of its own fails here until
# the runner reads it too.
# BATCH_CONFIG_DIRS as batch-backends.sh declares it, which legs 57 and 58 read.
slot_dirs=$(sed -n 's/^BATCH_CONFIG_DIRS="\(.*\)"$/\1/p' "$HERE/batch-backends.sh")
leg_57() {
slot_detail=
[ -n "$slot_dirs" ] || slot_detail="BATCH_CONFIG_DIRS not found in batch-backends.sh"
if [ -z "$slot_detail" ] && git -C "$HERE/.." rev-parse --git-dir >/dev/null 2>&1; then
  while IFS= read -r cfg; do
    grep -Eq '^[[:space:]]*backend[[:space:]]*=' "$HERE/../$cfg" || continue
    d=$(dirname "$cfg")
    case " $slot_dirs " in *" $d "*) continue ;; esac
    slot_detail="$cfg names a backend but $d is not in BATCH_CONFIG_DIRS ($slot_dirs)"
    break
  done < <(git -C "$HERE/.." ls-files -- '*ocannl_config' 'ocannl_config')
else
  [ -n "$slot_detail" ] || { skip "batch: the backend-bearing configurations are the ones the backends are read from" "not a git checkout"; slot_detail=skipped; }
fi
if [ -z "$slot_detail" ]; then
  report 0 "batch: the backend-bearing configurations are the ones the backends are read from"
elif [ "$slot_detail" != skipped ]; then
  report 1 "batch: the backend-bearing configurations are the ones the backends are read from" "$slot_detail"
fi
}
harness_leg_run '57: backend-bearing configurations are the ones read #slot' leg_57

# Leg 58: the two readers the backends come from are the ones test/config
# builds. The legs above drive fakes, so a renamed executable, a dropped
# `--read=backend`, or a reachability tool no longer asking Slot_kind would
# pass them while every real batch fell back to every backend -- a narrower
# width and a GPU token, silently. Pinned against the stanzas and the readers'
# own sources instead.
leg_58() {
slot_detail=
slot_src=$HERE/batch-backends.sh
slot_built=$(sed -n 's|.*"\$dune" build \(\./test/config/[A-Za-z0-9_.]* \./test/config/[A-Za-z0-9_.]*\) .*|\1|p' "$slot_src")
slot_reader_exe=$(sed -n 's|^ *reader=\$build_dir/default/\(test/config/[A-Za-z0-9_]*\)\.exe$|\1|p' "$slot_src")
slot_reach_exe=$(sed -n 's|^ *reach=\$build_dir/default/\(test/config/[A-Za-z0-9_]*\)\.exe$|\1|p' "$slot_src")
# ...and from the build tree dune was pointed at, not a fixed _build.
grep -q '^ *local build_dir=\${DUNE_BUILD_DIR:-_build}$' "$slot_src" ||
  slot_detail="batch-backends.sh does not run the readers from DUNE_BUILD_DIR's tree"
if [ -n "$slot_detail" ]; then :
elif [ -z "$slot_reader_exe" ] || [ -z "$slot_reach_exe" ]; then
  slot_detail="batch-backends.sh names no reader (${slot_reader_exe:-?}) or reachability tool (${slot_reach_exe:-?})"
else
  for exe in "$slot_reader_exe" "$slot_reach_exe"; do
    case " $slot_built " in
      *" ./$exe.exe "*) ;;
      *) slot_detail="batch-backends.sh runs $exe but builds only: ${slot_built:-nothing}"; break ;;
    esac
    grep -q "^ (name $(basename "$exe"))$" "$HERE/../test/config/dune" ||
      { slot_detail="test/config/dune has no executable named $(basename "$exe")"; break; }
  done
fi
# The answer itself lives in test/config/config_reads.ml (gh-ocannl-1581), the
# code codegen_text_inventory also runs to name what a written value means: the
# reader dispatches the keys it lists there, and "backend" must be one of them.
[ -n "$slot_detail" ] || grep -q 'List.mem Config_reads.backend_keys key' "$HERE/../$slot_reader_exe.ml" ||
  slot_detail="$slot_reader_exe.ml does not answer the keys Config_reads decides"
[ -n "$slot_detail" ] || grep -q '^let backend_keys = \[ "backend"; ' "$HERE/../test/config/config_reads.ml" ||
  slot_detail="test/config/config_reads.ml does not list backend among its backend_keys"
[ -n "$slot_detail" ] || grep -q '^  | "backend" -> Some (canonical_backend setting)$' "$HERE/../test/config/config_reads.ml" ||
  slot_detail="test/config/config_reads.ml does not answer --read=backend"
[ -n "$slot_detail" ] || grep -q 'Test_scan.Slot_kind.answer' "$HERE/../$slot_reach_exe.ml" ||
  slot_detail="$slot_reach_exe.ml does not ask Test_scan.Slot_kind"
# The reachability tool's grammar is the one batch_resolve parses: a `names
# <backend>: <why>` line per backend, a `reads config: <why>` line when a
# configuration counts, then `end`; or one `unknown: <why>`; or a `refused:
# <dune file>:<line>: <why>` line per refused backend marker, then `end`.
for form in 'printf "names %s: %s\n"' 'printf "reads config: %s\n"' 'printf "end\n"' \
            'printf "unknown: %s\n"' 'printf "refused: %s\n"'; do
  [ -z "$slot_detail" ] || break
  grep -qF "$form" "$HERE/../$slot_reach_exe.ml" ||
    slot_detail="$slot_reach_exe.ml has no $form"
done
# ...and Slot_kind's own list of the directories whose configuration the runner
# reads -- slot_kind_live refuses a reader reading any other's
# (gh-ocannl-1249) -- is BATCH_CONFIG_DIRS.
if [ -z "$slot_detail" ]; then
  slot_kind_dirs=$(sed -n 's/^let config_dirs = \[ \(.*\) \]$/\1/p' "$HERE/../test/support/slot_kind.ml" | tr -d '"' | sed 's/ *; */ /g')
  [ "$slot_kind_dirs" = "$slot_dirs" ] ||
    slot_detail="Slot_kind.config_dirs ($slot_kind_dirs) is not BATCH_CONFIG_DIRS ($slot_dirs)"
fi
if [ -z "$slot_detail" ]; then
  report 0 "batch: the readers batch-backends.sh builds are test/config's, answering --read=backend and Slot_kind in its grammar"
else
  report 1 "batch: the readers batch-backends.sh builds are test/config's, answering --read=backend and Slot_kind in its grammar" "$slot_detail"
fi
}
harness_leg_run '58: the backend readers are test/config'\''s #slot' leg_58

# Leg 93: a backend marker env_var_deps refuses is MARKER REFUSED, not every
# backend (the chip from gh-ocannl-1652). A dune file `d/dune` opening
# `(test ; ocannl-backend: cuda` -- no reason -- is answered by ocannl_slot_kind
# with one `refused: <dune file>:<line>: <why>` line, then `end`: the refusal is
# read out of test/operations/slot_kind_cases.expected, where Slot_kind's own
# answer for that very fixture stanza is recorded, not restated here. The runner used to read any line
# outside its grammar as every backend and run the batch under --gpu; now it
# refuses before the slot and dune (exit 2, no slot or dune call), the digest
# and the run's log naming the file and line. A refusal cut
# short of its `end`, or beside a `names` line, is refused all the same;
# `plan` says run would refuse it; `repeat` runs no iteration. The control:
# the same tree with the marker repaired answers `names cuda` and runs --gpu.
leg_93() {
local d= label="batch: a backend marker env_var_deps refuses is MARKER REFUSED before the slot and dune, not every backend"
local refused
refused=$(sed -n 's/^runtest d (malformed markers)  *\(refused: d\/dune:1: .*\); e\/dune:2: .*/\1/p' \
  "$HERE/../test/operations/slot_kind_cases.expected")
case $refused in
  'refused: d/dune:1: a backend marker that does not parse: '*) ;;
  *) harness_abandon "slot_kind_cases.expected records no refusal of the d/dune:1 fixture marker: '$refused'"; return ;;
esac
FAKE_REACH="$refused\nend" slot_probe marker-refused hold cc cc run build @cheap
{ [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && [ -z "$slot_calls" ]; } ||
  d="refused: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; slot calls: ${slot_calls:-<none>} (want none)"
if [ -z "$d" ]; then
  case $argv_out in
    *"verdict: MARKER REFUSED (a backend marker env_var_deps refuses; nothing ran) (exit 2)"*"marker refused: d/dune:1: a backend marker that does not parse"*) ;;
    *) d="refused: the digest: $argv_out" ;;
  esac
fi
{ [ -n "$d" ] || { [ -n "$argv_dir" ] && grep -qF 'test-run: batch: refused marker: d/dune:1:' "$argv_dir/log" &&
    grep -qF 'test-run: MARKER REFUSED: d/dune:1: a backend marker that does not parse' "$argv_dir/log"; }; } ||
  d="refused: the log does not record the marker and the refusal: $(cat "$argv_dir/log" 2>/dev/null)"
for probe in "cut-short:$refused" "beside-names:names cc: it reaches x in y, which names cc\n$refused\nend"; do
  [ -z "$d" ] || break
  FAKE_REACH=${probe#*:} slot_probe "marker-refused-${probe%%:*}" hold cc cc run build @cheap
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && [ -z "$slot_calls" ] &&
      grep -q 'verdict: MARKER REFUSED' <<<"$argv_out"; } ||
    d="${probe%%:*}: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; slot calls: ${slot_calls:-<none>}; digest: $argv_out"
done
if [ -z "$d" ]; then
  FAKE_REACH="$refused\nend" slot_probe marker-refused-plan hold cc cc plan build @cheap
  case $argv_out in
    *"markers: run would refuse this batch (MARKER REFUSED, exit 2)"*"  d/dune:1: a backend marker that does not parse"*) ;;
    *) d="plan: exit $argv_rc; output: $argv_out" ;;
  esac
  [ -n "$d" ] || [ -z "$argv_calls" ] || d="plan ran dune: $argv_calls"
fi
if [ -z "$d" ]; then
  FAKE_REACH="$refused\nend" OCANNL_TOOL_FLEET_WORKER=$slot_fake FAKE_FW_CALLS=$TMP/marker-repeat.fw \
    FAKE_FW_MODE=hold repeat_probe marker-repeat stable 2 build @cheap
  { [ "$repeat_rc" = 2 ] && [ -z "$(cat "$TMP/marker-repeat.calls" 2>/dev/null)" ] &&
      grep -qF "MARKER REFUSED: d/dune:1:" <<<"$repeat_out"; } ||
    d="repeat: exit $repeat_rc (want 2); dune calls: $(cat "$TMP/marker-repeat.calls" 2>/dev/null); output: $repeat_out"
fi
if [ -z "$d" ]; then
  FAKE_REACH="names cuda: it reaches t in d, which names cuda\nend" slot_probe marker-repaired hold cc cc run build @cheap
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] &&
      [ "$slot_calls" = "execution slot --wait 600 --gpu -- dune build @cheap" ] &&
      ! grep -q 'MARKER REFUSED' <<<"$argv_err$argv_out"; } ||
    d="the repaired marker (control): exit $argv_rc; dune calls: ${argv_calls:-<none>}; slot calls: ${slot_calls:-<none>}; digest: $argv_out"
fi
if [ -z "$d" ]; then report 0 "$label"; else report 1 "$label" "$d"; fi
}
harness_leg_run '93: a backend marker env_var_deps refuses is MARKER REFUSED #slot' leg_93

# Leg 78: a test configuration's backend counts only where ocannl_slot_kind
# says it does -- every batch but one of its configuration-free aliases or a
# plain build (gh-ocannl-1227). A test configuration naming hip -- AGENTS.md's
# one-off route -- under a fleet slot beside a small SDMA pool, minix's shape:
# a batch answered with no `reads config` line and nothing named (as
# @test/operations/scans is) holds no backend, so it runs at dune's own width
# and --cpu, its log saying why; one naming cc holds cc alone, the
# configuration unread. The control: the same batch answered with a `reads
# config` line is hip's -- -j 4 and --gpu.
leg_78() {
reads_label="batch: a configuration's backend counts only where the reachability tool says it does"
reads_detail=
native_fleet=$fleet_none
for probe in "none-only|end|build @cheap|--cpu|holds no backend: it reaches no stanza" \
             "named-cc|names cc: it reaches t_cc in x, which names cc\nend|build @cheap|--cpu|reads no configuration" \
             "reads|$FAKE_READS\nend|build -j 4 @cheap|--gpu|holds hip: test/config resolves backend=hip"; do
  IFS='|' read -r tag reach want kind logline <<<"$probe"
  : >"$TMP/reads-$tag.fw"
  export FAKE_BACKEND_TEST=hip FAKE_REACH=$reach FAKE_FW_CALLS=$TMP/reads-$tag.fw FAKE_FW_MODE=hold \
    OCANNL_TOOL_FLEET_WORKER=$slot_fake
  native_probe "reads-$tag" "$dxg_absent" "$kfd_small" "$nv_absent" "" run build @cheap
  # Back to the harness-wide `none`, never unset: unset, the next legs would
  # find this box's own fleet-worker.sh.
  unset FAKE_BACKEND_TEST FAKE_REACH FAKE_FW_CALLS FAKE_FW_MODE
  export OCANNL_TOOL_FLEET_WORKER=none
  slot_calls=$(grep -v -- '--probe' "$TMP/reads-$tag.fw")
  if [ "$argv_rc" != 0 ] || [ "$argv_calls" != "$want" ]; then
    reads_detail="$tag: exit $argv_rc; dune calls: ${argv_calls:-<none>} (want $want)"; break
  fi
  [ "$slot_calls" = "execution slot --wait 600 $kind -- dune $want" ] ||
    { reads_detail="$tag: slot call: ${slot_calls:-<none>} (want $kind)"; break; }
  { [ -n "$argv_dir" ] && grep -qF -- "test-run: batch: $logline" "$argv_dir/log"; } ||
    { reads_detail="$tag: the log does not say '$logline': $(cat "$argv_dir/log" 2>/dev/null)"; break; }
  if [ "$kind" = --cpu ]; then
    ! grep -qi 'capping\|sdma' <<<"$argv_err" ||
      { reads_detail="$tag: an uncapped batch was told about a cap: $argv_err"; break; }
  fi
done
if [ -z "$reads_detail" ]; then
  report 0 "$reads_label"
else
  report 1 "$reads_label" "$reads_detail"
fi
}
harness_leg_run '78: a configuration'\''s backend counts only where the reachability tool says it does #slot' leg_78

# Leg 88 (gh-ocannl-1433): what the batch's backends say about the GPU it was
# meant to test. On a GPU box (cuda on a native NVIDIA boot, hip beside a KFD
# pool), a batch resolving to cc alone runs, green, and WARNs in the
# announcements, the log and the digest; `--expect-backend cuda` refuses it
# before dune (BACKEND REFUSED, exit 2, no dune call, the digest naming what
# it held). The controls: the same batch on cuda, or expecting cc, says
# nothing; a box with no GPU hazard (every Mac) says nothing for cc, yet the
# option still resolves and refuses there; an unread batch (`exec`) proves no
# backend and is refused; a batch holding no backend is not warned about;
# `plan` reports both; a non-backend or a misplaced option is a usage error.
leg_88() {
local d= label="backend check: a GPU box's cc-only batch is warned about, and --expect-backend refuses it"
native_fleet=$fleet_none
for probe in "nv-cc|$nv_present|$kfd_absent|cuda|" "kfd-cc|$nv_absent|$kfd_small|hip|"; do
  IFS='|' read -r tag nv kfd gpu _ <<<"$probe"
  native_probe "be-$tag" "$dxg_absent" "$kfd" "$nv" "" run build @cheap
  [ "$argv_rc" = 0 ] && [ -n "$argv_calls" ] ||
    { d="$tag: the warned batch did not run: exit $argv_rc; calls ${argv_calls:-<none>}"; break; }
  want="WARNING: this box has a GPU backend ($gpu), but this batch holds only cc: it tests nothing on the GPU."
  grep -qF -- "$want" <<<"$argv_err" ||
    { d="$tag: stderr lacks the warning: $argv_err"; break; }
  grep -qF -- "$want" <<<"$argv_out" ||
    { d="$tag: the digest lacks the warning: $argv_out"; break; }
  { [ -n "$argv_dir" ] && grep -qF -- "test-run: $want" "$argv_dir/log"; } ||
    { d="$tag: the log lacks the warning"; break; }
  native_probe "be-$tag-expect" "$dxg_absent" "$kfd" "$nv" "" run --expect-backend "$gpu" build @cheap
  if [ "$argv_rc" != 2 ] || [ -n "$argv_calls" ]; then
    d="$tag --expect-backend $gpu: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>} (want none)"; break
  fi
  case $argv_out in
    *"verdict: BACKEND REFUSED"*"backend refused: --expect-backend $gpu: this batch holds cc, not $gpu."*) ;;
    *) d="$tag --expect-backend $gpu: the digest: $argv_out"; break ;;
  esac
  # Controls: the GPU backend selected, with and without the option; cc expected outright.
  for ctl in "selected|$gpu|" "selected-expected|$gpu|--expect-backend $gpu" "cc-expected||--expect-backend cc"; do
    IFS='|' read -r ctag backend opt <<<"$ctl"
    # shellcheck disable=SC2086  # the option is word-split on purpose
    native_probe "be-$tag-$ctag" "$dxg_absent" "$kfd" "$nv" "$backend" run $opt build @cheap
    if [ "$argv_rc" != 0 ] || [ -z "$argv_calls" ] || grep -q 'WARNING:\|BACKEND REFUSED\|backend refused' <<<"$argv_err$argv_out"; then
      d="$tag control $ctag: exit $argv_rc; calls ${argv_calls:-<none>}; stderr $argv_err; digest $argv_out"; break 2
    fi
  done
done
if [ -z "$d" ]; then
  # A WSL guest's bridge is a hazard for cuda and hip alike: the warning names both and
  # recommends neither, since minix-amd-wsl's GPU is not cuda (Codex review round 1).
  native_probe be-wsl "$dxg_present" "$kfd_absent" "$nv_absent" "" run build @cheap
  grep -qF "WARNING: this box has a GPU backend (cuda or hip), but this batch holds only cc" <<<"$argv_err" &&
    grep -qF "(OCANNL_BACKEND=<this box's GPU backend>)" <<<"$argv_err" ||
    d="a WSL bridge: stderr $argv_err"
fi
if [ -z "$d" ]; then
  # No GPU hazard: no warning for cc, but the option still resolves, and refuses.
  native_probe be-mac "$dxg_absent" "$kfd_absent" "$nv_absent" "" run build @cheap
  [ "$argv_rc" = 0 ] && ! grep -q 'WARNING' <<<"$argv_err$argv_out" ||
    d="no hazard: exit $argv_rc; stderr $argv_err; digest $argv_out"
fi
if [ -z "$d" ]; then
  native_probe be-mac-expect "$dxg_absent" "$kfd_absent" "$nv_absent" "" run --expect-backend metal build @cheap
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'verdict: BACKEND REFUSED' <<<"$argv_out" ||
    d="no hazard, --expect-backend metal: exit $argv_rc; calls ${argv_calls:-<none>}; digest $argv_out"
fi
if [ -z "$d" ]; then
  native_probe be-mac-held "$dxg_absent" "$kfd_absent" "$nv_absent" metal run --expect-backend metal build @cheap
  [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] ||
    d="no hazard, metal expected and held: exit $argv_rc; calls ${argv_calls:-<none>}; digest $argv_out"
fi
if [ -z "$d" ]; then
  native_probe be-unread "$dxg_absent" "$kfd_absent" "$nv_present" cuda run --expect-backend cuda exec ./prog.exe
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'backends are unread' <<<"$argv_out" ||
    d="an unread batch: exit $argv_rc; calls ${argv_calls:-<none>}; digest $argv_out"
fi
if [ -z "$d" ]; then
  FAKE_REACH='end' native_probe be-none "$dxg_absent" "$kfd_absent" "$nv_present" "" run build @cheap
  [ "$argv_rc" = 0 ] && ! grep -q 'WARNING' <<<"$argv_err$argv_out" ||
    d="a batch holding no backend: exit $argv_rc; stderr $argv_err"
fi
if [ -z "$d" ]; then
  native_probe be-plan "$dxg_absent" "$kfd_absent" "$nv_present" "" plan --expect-backend cuda build @cheap
  case $argv_out in
    *"expect-backend: run would refuse this batch (BACKEND REFUSED, exit 2)"*"this batch holds cc, not cuda"*) ;;
    *) d="plan --expect-backend cuda: exit $argv_rc; output $argv_out" ;;
  esac
  [ -n "$d" ] || [ -z "$argv_calls" ] || d="plan ran dune: $argv_calls"
fi
if [ -z "$d" ]; then
  native_probe be-plan-warn "$dxg_absent" "$kfd_absent" "$nv_present" "" plan build @cheap
  grep -qF 'WARNING: this box has a GPU backend (cuda), but this batch holds only cc' <<<"$argv_out" ||
    d="plan on a GPU box's cc batch: output $argv_out"
fi
if [ -z "$d" ]; then
  argv_probe be-bad-name run --expect-backend vulkan build @cheap
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'not a backend' <<<"$argv_err" ||
    d="--expect-backend vulkan: exit $argv_rc; stderr $argv_err"
fi
if [ -z "$d" ]; then
  argv_probe be-misplaced run build @cheap --expect-backend cuda
  [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q 'belongs before the dune arguments' <<<"$argv_err" ||
    d="a misplaced --expect-backend: exit $argv_rc; stderr $argv_err"
fi
if [ -z "$d" ]; then report 0 "$label"; else report 1 "$label" "$d"; fi
}
harness_leg_run '88: a GPU box'\''s cc-only batch is warned about, and --expect-backend refuses it #slot' leg_88

# ---------------------------------------------------------------------------
# Legs 75-77: a wedged device cannot hold a batch (gh-ocannl-1211)
# ---------------------------------------------------------------------------
# A GPU fault on tuf left every later test of a HIP batch spinning in device
# init, eight at once, for 32 minutes, under a run cap with half an hour to
# go. Two bounds answer it: the per-test cap ends the run once a test outlives
# it (75), and a GPU batch's device probe refuses to start dune when a fresh
# device query does not answer (76); `plan` reports both (77).
#
# Leg 75: the per-test cap. A test past it ends the run as the run's cap
# would -- TIMEOUT, 142, the test named -- long before the test would end, the
# test killed and the worktree idle. Controls: a test inside the cap passes;
# OCANNL_TOOL_PER_TEST_CAP is the default, and --test-cap 0 lifts it; and
# neither a slow dune nor the slot's own long-lived holder, which is no child
# of dune, is taken for a test (a watcher counting every process group of the
# run would end both runs). The message blames a device only on a GPU batch,
# and always names the remedy. Tests are timed on the watcher's own clock:
# a stub ps serving every etime as 1.2 million years, as minix's procps did
# after a resume (gh-ocannl-1319), neither ends a run nor spares a hung test,
# and the stub's log shows it served the watcher. A bad value is refused
# before dune runs. The fixture's dune is perl renamed through $0, which Linux
# (prctl) and macOS (the argv area BSD ps reads) both show as the process
# name; the leg skips only where a renamed process does not read back as
# `dune` (Git Bash, say).
leg_75() {
test_cap_label="test cap: a test past --test-cap ends the run as TIMEOUT naming it; a test inside it, a skewed ps etime, a slow dune and the slot's holder do not"
perl -e '$0 = "dune"; sleep 10' </dev/null >/dev/null 2>&1 &
test_cap_named=$!
sleep 0.3
test_cap_named_ok=
ps -A -o pid= -o comm= 2>/dev/null | grep -Eq "^ *$test_cap_named +(.*/)?dune$" && test_cap_named_ok=1
kill "$test_cap_named" 2>/dev/null
wait "$test_cap_named" 2>/dev/null
if [ -z "${test_cap_named_ok:-}" ]; then
  skip "$test_cap_label" "a process renamed through \$0 does not read back as dune in ps here"
else
  test_cap_detail=
  test_cap_n=0
  test_cap_start=$SECONDS
  test_cap_n=$((test_cap_n + 1))
  export REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
  argv_mode=action argv_probe test-cap-hung run --test-cap 2 build @cheap
  test_cap_took=$((SECONDS - test_cap_start))
  test_cap_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
  case $argv_out in
    *"verdict: TIMEOUT (a test outlived the per-test cap; run was killed, not judged) (exit 142)"*"hung test: sleep 60"*"past the 2s per-test cap"*) ;;
    *) test_cap_detail="hung: digest: $argv_out" ;;
  esac
  [ -n "$test_cap_detail" ] || [ "$argv_rc" = 142 ] || test_cap_detail="hung: exit $argv_rc (want 142)"
  [ -n "$test_cap_detail" ] || [ "$test_cap_took" -lt 20 ] ||
    test_cap_detail="hung: the run took ${test_cap_took}s under a 2s test cap"
  [ -n "$test_cap_detail" ] || [ -n "$test_cap_action" ] || test_cap_detail="hung: the test never started"
  if [ -z "$test_cap_detail" ]; then
    case $(pstate "$test_cap_action") in
      '' | Z) ;;
      *) test_cap_detail="hung: the test (pid $test_cap_action) survived the run"
         kill -KILL "$test_cap_action" 2>/dev/null ;;
    esac
  fi
  [ -n "$test_cap_detail" ] || grep -q "^test-run: per-test cap: pid $test_cap_action has run" "$argv_dir/log" ||
    test_cap_detail="hung: the log does not name the test: $(cat "$argv_dir/log")"
  # A CPU batch is told the remedy and never sent looking for a GPU fault.
  [ -n "$test_cap_detail" ] || grep -qF "rerun with --test-cap 0" <<<"$argv_out" ||
    test_cap_detail="hung: the digest names no remedy: $argv_out"
  if [ -z "$test_cap_detail" ] &&
     { cat "$argv_dir/log"; printf '%s\n' "$argv_out"; } | grep -qi "wedged device\|kernel journal"; then
    test_cap_detail="hung: a CPU batch's run blames a device: $argv_out $(cat "$argv_dir/log")"
  fi
  [ -n "$test_cap_detail" ] ||
    OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-test-cap-hung "$repeat_root/tools/test-run.sh" idle >/dev/null 2>&1 ||
    test_cap_detail="hung: the worktree is not idle after the run"
  for probe in "inside:3:1:0" "env-default:1:4:142" "lifted:1:3:0"; do
    [ -z "$test_cap_detail" ] || break
    IFS=: read -r tag cap_s secs want <<<"$probe"
    test_cap_n=$((test_cap_n + 1))
    REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
    case $tag in
      inside) REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
                argv_probe "test-cap-$tag" run --test-cap "$cap_s" build @cheap ;;
      lifted) OCANNL_TOOL_PER_TEST_CAP=$cap_s REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
                argv_probe "test-cap-$tag" run --test-cap 0 build @cheap ;;
      *) OCANNL_TOOL_PER_TEST_CAP=$cap_s REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
           argv_probe "test-cap-$tag" run build @cheap ;;
    esac
    [ "$argv_rc" = "$want" ] || { test_cap_detail="$tag: exit $argv_rc (want $want): $argv_out"; break; }
    if [ "$want" = 0 ] && [ -e "$argv_dir/test-cap" ]; then
      test_cap_detail="$tag: a passing run left a test-cap record"
    fi
  done
  # A skewed ps clock (gh-ocannl-1319): after a resume from suspend, minix's
  # procps reported a test a minute old as 38109073018720s old, and the cap
  # ended the run. A ps on PATH that serves every etime as that age: a test
  # inside the cap still passes, a hung one is still capped, and the stub's
  # log proves the watcher was served the skewed etime.
  if [ -z "$test_cap_detail" ]; then
    skew_bin=$TMP/test-cap-skew-bin
    mkdir -p "$skew_bin"
    : >"$TMP/test-cap-skew.served"
    cat >"$skew_bin/ps" <<EOF
#!/usr/bin/env perl
# The real ps, with every etime column replaced by 441077234-07:58:40.
my @cols;
for (my \$i = 0; \$i < \$#ARGV; \$i++) { push @cols, \$ARGV[\$i + 1] if \$ARGV[\$i] eq "-o" }
my (\$at) = grep { \$cols[\$_] =~ /^etime=?\$/ } 0 .. \$#cols;
open my \$ps, "-|", "$(command -v ps)", @ARGV or exit 1;
while (my \$line = <\$ps>) {
  if (defined \$at) {
    chomp \$line;
    my @f = split " ", \$line, scalar @cols;
    if (@f == @cols) {
      \$f[\$at] = "441077234-07:58:40";
      \$line = join(" ", @f);
      if (open my \$log, ">>", "$TMP/test-cap-skew.served") { print \$log "\$line\n"; close \$log }
    }
    \$line .= "\n";
  }
  print \$line;
}
close \$ps;
exit(\$? >> 8);
EOF
    chmod +x "$skew_bin/ps"
    for probe in "skew-inside:4:2:0" "skew-hung:2:60:142"; do
      [ -z "$test_cap_detail" ] || break
      IFS=: read -r tag cap_s secs want <<<"$probe"
      test_cap_n=$((test_cap_n + 1))
      REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
      : >"$TMP/test-cap-skew.served"
      PATH=$skew_bin:$PATH REPEAT_TEST_ACTION_SECS=$secs argv_mode=action \
        argv_probe "test-cap-$tag" run --test-cap "$cap_s" build @cheap
      test_cap_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
      [ "$argv_rc" = "$want" ] || { test_cap_detail="$tag: exit $argv_rc (want $want): $argv_out $(cat "$argv_dir/log")"; break; }
      grep -q "^ *$test_cap_action .*441077234-07:58:40" "$TMP/test-cap-skew.served" ||
        { test_cap_detail="$tag: the watcher was never served the test's skewed etime: $(head -c 2000 "$TMP/test-cap-skew.served")"; break; }
      if [ "$want" = 0 ]; then
        [ ! -e "$argv_dir/test-cap" ] || test_cap_detail="$tag: a passing run left a test-cap record"
      else
        grep -Eq "^test-run: per-test cap: pid $test_cap_action has run [0-9]{1,2}s, past the 2s cap" "$argv_dir/log" ||
          test_cap_detail="$tag: the log does not name the test at its own age: $(cat "$argv_dir/log")"
        case $(pstate "$test_cap_action") in
          '' | Z) ;;
          *) test_cap_detail="$tag: the test (pid $test_cap_action) survived the run"
             kill -KILL "$test_cap_action" 2>/dev/null ;;
        esac
      fi
    done
  fi
  # A test that execs keeps its pid and changes its name: a wrapper handing
  # over to the real program, here a chain of ten 1.2s hops, each renamed. Its
  # age runs on across the renames, so a 4s cap ends the run long before the
  # chain would; a watcher restarting the age at each rename never would.
  if [ -z "$test_cap_detail" ]; then
    cat >"$TMP/test-cap-hops.pl" <<'EOF'
my $n = shift;
$0 = "hop$n";
select undef, undef, undef, 1.2;
exit 0 if $n >= 10;
exec $^X, __FILE__, $n + 1;
EOF
    test_cap_n=$((test_cap_n + 1))
    REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
    REPEAT_TEST_ACTION_HOPS=$TMP/test-cap-hops.pl argv_mode=action \
      argv_probe test-cap-renamed run --test-cap 4 build @cheap
    test_cap_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
    { [ "$argv_rc" = 142 ] && [ -n "$test_cap_action" ] &&
      grep -q "^test-run: per-test cap: pid $test_cap_action has run" "$argv_dir/log"; } ||
      test_cap_detail="renamed: exit $argv_rc (want 142, the test pid ${test_cap_action:-unknown} named): $argv_out $(cat "$argv_dir/log")"
    case $(pstate "$test_cap_action") in
      '' | Z) ;;
      *) [ -n "$test_cap_detail" ] || test_cap_detail="renamed: the test (pid $test_cap_action) survived the run"
         kill -KILL "$test_cap_action" 2>/dev/null ;;
    esac
  fi
  if [ -z "$test_cap_detail" ]; then
    : >"$TMP/test-cap-holder.fw"
    FAKE_FW_CALLS=$TMP/test-cap-holder.fw FAKE_FW_MODE=holder OCANNL_TOOL_FLEET_WORKER=$slot_fake \
    REPEAT_TEST_ACTION_SECS=4 argv_mode=slow_dune argv_probe test-cap-holder run --test-cap 1 build @cheap
    { [ "$argv_rc" = 0 ] && [ "$(grep -c -- '--probe' "$TMP/test-cap-holder.fw")" = 1 ] &&
      [ "$(grep -vc -- '--probe' "$TMP/test-cap-holder.fw")" = 1 ]; } ||
      test_cap_detail="slow dune beside the slot's holder: exit $argv_rc (want 0; slot calls: $(cat "$TMP/test-cap-holder.fw")): $argv_out"
  fi
  # ...and a GPU batch's hung test is pointed at the kernel journal.
  if [ -z "$test_cap_detail" ]; then
    test_cap_n=$((test_cap_n + 1))
    REPEAT_TEST_ACTION_PID=$TMP/test-cap-action-$test_cap_n.pid
    argv_mode=action slot_probe test-cap-gpu hold hip cc run --test-cap 2 build @cheap
    { [ "$argv_rc" = 142 ] && grep -qF "read the kernel journal" <<<"$argv_out" &&
      grep -q "a wedged device hangs every test after it" "$argv_dir/log"; } ||
      test_cap_detail="a GPU batch's hung test: exit $argv_rc; digest: $argv_out"
  fi
  if [ -z "$test_cap_detail" ]; then
    argv_probe test-cap-bad run --test-cap 2x build @cheap
    { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -q -- "--test-cap must be a nonnegative integer" <<<"$argv_err"; } ||
      test_cap_detail="a bad value: exit $argv_rc; dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
  fi
  unset REPEAT_TEST_ACTION_PID
  if [ -z "$test_cap_detail" ]; then report 0 "$test_cap_label"; else report 1 "$test_cap_label" "$test_cap_detail"; fi
fi
}
harness_leg_run '75: the per-test cap #device' leg_75

shared_through device_probe
leg_76() {
probe_label="device probe: a GPU batch whose fresh device query does not answer is DEVICE UNHEALTHY, dune never started"
probe_detail=
probe_root=$(cd "$repeat_root" && pwd -P)
probe_hang_run pid probe-hang hip cc hip "" run build @cheap
case $argv_out in
  *"verdict: DEVICE UNHEALTHY (a fresh device query did not answer; nothing ran) (exit 69)"*"hip: NO ANSWER within ${probe_cap}s"*) ;;
  *) probe_detail="hang: digest: $argv_out" ;;
esac
[ -n "$probe_detail" ] || { [ "$argv_rc" = 69 ] && [ -z "$argv_calls" ]; } ||
  probe_detail="hang: exit $argv_rc (want 69); dune calls: ${argv_calls:-<none>}"
[ -n "$probe_detail" ] || [ "$probe_hang_took" -lt $((probe_cap + 14)) ] ||
  probe_detail="hang: the refusal took ${probe_hang_took}s under a ${probe_cap}s bound"
case $slot_calls in
  "execution slot --wait 600 --gpu -- "*" $probe_root/tools/test-run.sh _probe dune $probe_fake $probe_cap hip -- dune build @cheap") ;;
  *) [ -n "$probe_detail" ] || probe_detail="hang: the probe is not the slot's command, ahead of dune: $slot_calls" ;;
esac
[ -n "$probe_detail" ] || [ "$probe_calls" = "$probe_root/test/config --ocannl_backend=hip" ] ||
  probe_detail="hang: the query was not asked once, for hip, from test/config: $probe_calls"
[ -n "$probe_detail" ] || grep -q "did not answer within ${probe_cap}s" "$argv_dir/log" ||
  probe_detail="hang: the log does not say why dune never started: $(cat "$argv_dir/log")"
if [ -z "$probe_detail" ]; then
  probe_pid=$(cat "$TMP/probe-hang.probe.pid" 2>/dev/null)
  if [ -z "$probe_pid" ]; then
    probe_detail="hang: the stand-in recorded its call but no pid within ${probe_cap}s"
  else
    case $(pstate "$probe_pid") in
      '' | Z) ;;
      *) probe_detail="hang: the query (pid $probe_pid) survived its bound"; kill -KILL "$probe_pid" 2>/dev/null ;;
    esac
  fi
fi
# A stand-in KILLed before it recorded its start -- as one was on a loaded box (gh-ocannl-1366) --
# does not fail the leg: its run is repeated under a wider bound, and that run is judged. The
# first run's stand-in records nothing by construction (the marker), not by a race with the
# bound, so the judged run is exactly the second, bounded at 3s.
if [ -z "$probe_detail" ]; then
  : >"$TMP/probe-late.probe.unstarted"
  probe_hang_late=1 probe_hang_run pid probe-late hip cc hip "" run build @cheap
  probe_pid=$(cat "$TMP/probe-late.probe.pid" 2>/dev/null)
  { [ "$probe_cap" = 3 ] && [ ! -e "$TMP/probe-late.probe.unstarted" ] && [ "$argv_rc" = 69 ] &&
    [ -z "$argv_calls" ] && [ -n "$probe_pid" ] &&
    [ "$probe_calls" = "$probe_root/test/config --ocannl_backend=hip" ] &&
    grep -qF "hip: NO ANSWER within 3s" "$argv_dir/device-probe"; } ||
    probe_detail="unstarted: judged under ${probe_cap}s (want 3s); exit $argv_rc (want 69); dune calls: ${argv_calls:-<none>}; probes: ${probe_calls:-<none>}"
  case ${probe_pid:+$(pstate "$probe_pid")} in
    '' | Z) ;;
    *) kill -KILL "$probe_pid" 2>/dev/null
       [ -n "$probe_detail" ] || probe_detail="unstarted: the query (pid $probe_pid) survived its bound" ;;
  esac
fi
# A query a forking stand-in left behind is killed with it: the bound KILLs the query's whole
# process group (Codex review round 2 on PR #1172).
if [ -z "$probe_detail" ]; then
  FAKE_PROBE_FORK=1 probe_hang_run child probe-fork hip cc hip "" run build @cheap
  probe_child=$(cat "$TMP/probe-fork.probe.child" 2>/dev/null)
  if [ "$argv_rc" != 69 ] || [ -z "$probe_child" ]; then
    probe_detail="fork: exit $argv_rc (want 69); forked query: ${probe_child:-none recorded}"
  else
    case $(pstate "$probe_child") in
      '' | Z) ;;
      *) probe_detail="fork: the forked query (pid $probe_child) outlived the bound"
         kill -KILL "$probe_child" 2>/dev/null ;;
    esac
  fi
fi
# An answer, and a fast failure, start dune once under the same slot call.
for probe in "answer::hip: answered in" "absent:hip:hip: failed fast (exit 2"; do
  [ -z "$probe_detail" ] || break
  IFS=: read -r tag absent want <<<"$probe"
  probe_run "probe-$tag" hip cc "" "$absent" run build @cheap
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] && grep -qF -- "$want" <<<"$argv_out" &&
    [ "$probe_calls" = "$probe_root/test/config --ocannl_backend=hip" ]; } ||
    probe_detail="$tag: exit $argv_rc; dune calls: ${argv_calls:-<none>}; probes: $probe_calls; digest: $argv_out"
done
# An unread batch holds any backend: every GPU one is asked, in order.
if [ -z "$probe_detail" ]; then
  probe_run probe-unread cc fail "" "cuda metal" run build @cheap
  probe_asked=$(sed 's/.*--ocannl_backend=//' <<<"$probe_calls" | paste -s -d ' ' -)
  { [ "$argv_rc" = 0 ] && [ "$probe_asked" = "cuda hip metal" ]; } ||
    probe_detail="unread: exit $argv_rc; asked: $probe_asked (want cuda hip metal)"
fi
# A CPU batch asks nothing, and its slot call is the bare one.
if [ -z "$probe_detail" ]; then
  probe_run probe-cpu cc cc hip "" run build @cheap
  { [ "$argv_rc" = 0 ] && [ -z "$probe_calls" ] &&
    [ "$slot_calls" = "execution slot --wait 600 --cpu -- dune build @cheap" ]; } ||
    probe_detail="cpu: exit $argv_rc; probes: ${probe_calls:-<none>}; slot call: $slot_calls"
fi
# Without a stand-in the probe is bin/device_props, built by `_probe` inside
# the slot -- after the slot call, before the batch's own dune -- from the
# tree dune is pointed at. The fixture dune builds nothing, so the batch is
# recorded as not probed and still starts.
if [ -z "$probe_detail" ] && [ "$(uname -s)" != Darwin ]; then
  for build_dir in "" elsewhere; do
    : >"$TMP/probe-build.probe"
    FAKE_PROBE_CALLS=$TMP/probe-build.probe OCANNL_TOOL_DEVICE_PROBE= DUNE_BUILD_DIR=$build_dir \
      slot_probe "probe-build${build_dir:+-$build_dir}" hold hip cc run build @cheap
    want_exe=$probe_root/${build_dir:-_build}/default/bin/device_props.exe
    { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build ./bin/device_props.exe
build @cheap" ] && grep -qF "not probed: $want_exe did not build" "$argv_dir/device-probe" &&
      [ ! -s "$TMP/probe-build.probe" ]; } ||
      { probe_detail="the real probe (DUNE_BUILD_DIR=${build_dir:-unset}): exit $argv_rc; dune calls: $argv_calls; record: $(cat "$argv_dir/device-probe" 2>/dev/null)"; break; }
  done
fi
if [ -z "$probe_detail" ]; then report 0 "$probe_label"; else report 1 "$probe_label" "$probe_detail"; fi
}
harness_leg_run '76: the device probe #device' leg_76

# Leg 77: `plan` reports the probe and the per-test cap, and asks no device.
leg_77() {
plan_probe_detail=
probe_run probe-plan hip cc "" "" plan build @cheap
{ [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] && [ -z "$probe_calls" ] &&
  grep -qF "device probe: hip, bounded at ${probe_answer_cap}s" <<<"$argv_out" &&
  grep -qF "test cap: 1500s for each process dune starts, the default for a GPU batch" <<<"$argv_out"; } ||
  plan_probe_detail="gpu: exit $argv_rc; probes: ${probe_calls:-<none>}; plan: $argv_out"
# The default's scope: no CPU batch, no @slow target; an explicit cap is kept
# on either, and 0 lifts it.
for probe in "cpu|cc|build @cheap|test cap: none (no default: a CPU batch)" \
             "slow|hip|build @test/training/slow-cifar_conv|test cap: none (no default: the targets reach the slow tests (@test/training/slow-cifar_conv); a wedged device" \
             "slow-asked|hip|--test-cap 9000 build @slow|test cap: 9000s for each process dune starts, asked for" \
             "cpu-lifted|cc|--test-cap 0 build @cheap|test cap: none (--test-cap 0)"; do
  [ -z "$plan_probe_detail" ] || break
  IFS='|' read -r tag bt argv want <<<"$probe"
  # shellcheck disable=SC2086  # the argv is word-split on purpose
  probe_run "probe-plan-$tag" "$bt" cc "" "" plan $argv
  { [ "$argv_rc" = 0 ] && grep -qF -- "$want" <<<"$argv_out"; } ||
    plan_probe_detail="$tag: exit $argv_rc; want '$want'; plan: $argv_out"
done
# Where run resolves no backends (no fleet slot, no width hazard), plan
# reports neither bound, however its own resolution reads the batch.
if [ -z "$plan_probe_detail" ]; then
  FAKE_BACKEND_TEST=hip OCANNL_TOOL_DEVICE_PROBE=$probe_fake argv_probe probe-plan-unread plan build @cheap
  { [ "$argv_rc" = 0 ] && grep -qF "device probe: none" <<<"$argv_out" &&
    grep -qF "test cap: none (no default: run does not resolve" <<<"$argv_out"; } ||
    plan_probe_detail="no slot, no hazard: exit $argv_rc; plan: $argv_out"
fi
if [ -z "$plan_probe_detail" ]; then
  report 0 "plan: reports the device probe and the per-test cap, asking no device"
else
  report 1 "plan: reports the device probe and the per-test cap, asking no device" "$plan_probe_detail"
fi
}
harness_leg_run '77: plan reports the probe and the per-test cap #device' leg_77
# The cap must be selected before launching the supervisor: checking only the
# plan text would miss a run still cut at the old bound. Fixture actions return
# immediately; their recorded cap proves the budget without waiting four hours.
leg_106() {
  local OCANNL_TOOL_TEST_CAP= d= probe tag argv want source
  . "$HERE/per-test-cap.sh"
  for probe in "ordinary|plan build @cheap|3600" \
               "slow|plan build @slow|$SLOW_RUN_CAP_DEFAULT" \
               "member|plan build @@test/training/slow-cifar_conv|$SLOW_RUN_CAP_DEFAULT" \
               "short|plan --cap 9 build @slow|9" \
               "lifted|plan --cap 0 build @slow|0" \
               "separator|plan exec prog -- @slow|3600"; do
    IFS='|' read -r tag argv want <<<"$probe"
    # shellcheck disable=SC2086  # fixture argv deliberately split
    probe_run "slow-cap-$tag" hip cc "" "" $argv
    { [ "$argv_rc" = 0 ] && grep -qF "run cap: ${want}s (" <<<"$argv_out"; } ||
      d="$tag: exit $argv_rc; plan: $argv_out"
    [ -z "$d" ] || break
    case $tag in
      slow | member) source="the slow default" ;;
      short | lifted) source="asked for" ;;
      *) source="the default" ;;
    esac
    grep -qF "run cap: ${want}s ($source" <<<"$argv_out" ||
      d="the plan lacks its cap source: $argv_out"
  done
  printf '%s\n' '@test/training/slow-cifar_conv' >"$TMP/slow-cap-targets"
  for probe in "run|run build @slow|$SLOW_RUN_CAP_DEFAULT" \
               "file|run --targets-file $TMP/slow-cap-targets build|$SLOW_RUN_CAP_DEFAULT" \
               "fresh|fresh @test/training/slow-cifar_conv|$SLOW_RUN_CAP_DEFAULT" \
               "repeat|repeat 2 build @slow|$SLOW_RUN_CAP_DEFAULT" \
               "override|run --cap 0 build @slow|0"; do
    [ -z "$d" ] || break
    IFS='|' read -r tag argv want <<<"$probe"
    # shellcheck disable=SC2086  # fixture argv deliberately split
    if [ "$tag" = fresh ]; then
      if ! command -v python3 >/dev/null 2>&1; then
        skip "fresh slow cap" "no python3 for the fixture action trace"
        continue
      fi
      cp "$HERE/action-durations.sh" "$repeat_root/tools/action-durations.sh"
      synthetic_trace "$TMP/slow-cap.trace" /fake/build 'cifar_conv.exe|0|test/training/cifar_conv.exe.output'
      # shellcheck disable=SC2086  # fixture argv deliberately split
      REPEAT_TEST_TRACE=$TMP/slow-cap.trace REPEAT_TEST_MKBUILD=1 probe_run "slow-cap-$tag" hip cc "" "" $argv
    else
      # shellcheck disable=SC2086  # fixture argv deliberately split
      probe_run "slow-cap-$tag" hip cc "" "" $argv
    fi
    { [ "$argv_rc" = 0 ] && [ -n "$argv_calls" ] &&
      [ "$(cat "$argv_dir/cap" 2>/dev/null)" = "$want" ]; } ||
      d="$tag: exit $argv_rc; calls: $argv_calls; recorded cap: $(cat "$argv_dir/cap" 2>/dev/null)"
  done
  if [ -z "$d" ]; then
    OCANNL_TOOL_TEST_CAP=17 probe_run slow-cap-env hip cc "" "" run build @slow
    { [ "$argv_rc" = 0 ] && [ "$(cat "$argv_dir/cap" 2>/dev/null)" = 17 ]; } ||
      d="the environment cap was not kept: exit $argv_rc; cap: $(cat "$argv_dir/cap" 2>/dev/null)"
  fi
  if [ -z "$d" ]; then
    report 0 "slow targets get four hours before launch; plan states the cap source; explicit caps win"
  else
    report 1 "slow targets get four hours before launch; plan states the cap source; explicit caps win" "$d"
  fi
}
harness_leg_run '106: slow run caps and their sources #device' leg_106
shared_through renamed_dune

# ---------------------------------------------------------------------------
# Leg 91: the per-test cap tells a test from a process that took its pid (gh-ocannl-1448)
# ---------------------------------------------------------------------------
# The watcher keyed a test on its pid and its parent, so a process the system
# gave the same pid under the same dune within one poll inherited the first
# one's age and could be capped early (the residual staging#1254 left). It now
# also keys on the start token: /proc/<pid>/stat's starttime on Linux, ps's
# lstart elsewhere. A recycled pid cannot be arranged on demand, so a stub ps
# serves the watcher's listing unchanged and, through the watcher's seam
# (OCANNL_TOOL_TEST_CAP_PROC, a directory read ahead of /proc), a stat for the
# test whose starttime it advances on every listing: a 7 s test under a 4 s cap
# is then a new process at every poll and is never capped (on the base, which
# reads no token, it is). The control serves one starttime throughout, and the
# same test is capped: the seam's token is read, not ignored, and a token that
# stays the same keeps the test's age -- as the real readers' must, or leg 75's
# hung test would never be capped.
leg_91() {
tok_label="per-test cap: a test whose start token changes is a new process (never capped); one whose token stays is capped"
renamed_dune_readable
if [ -z "${test_cap_named_ok:-}" ]; then
  skip "$tok_label" "a process renamed through \$0 does not read back as dune in ps here"
  return
fi
tok_bin=$TMP/test-cap-token-bin
tok_proc=$TMP/test-cap-token-proc
mkdir -p "$tok_bin"
cat >"$tok_bin/ps" <<EOF
#!/usr/bin/env bash
# The real ps; on the watcher's listing (an etime column), a stat for the test
# under TOK_PROC first, its starttime the listing's number when TOK_ADVANCE is
# set and 4242 otherwise. Each one served is logged to TOK_SERVED.
case " \$* " in
  *" etime= "*)
    pid=\$(cat "\$REPEAT_TEST_ACTION_PID" 2>/dev/null)
    if [ -n "\$pid" ]; then
      n=\$(( \$(wc -l <"\$TOK_SERVED" 2>/dev/null || echo 0) + 1 ))
      start=4242
      [ -z "\${TOK_ADVANCE:-}" ] || start=\$((100000 + n))
      mkdir -p "\$TOK_PROC/\$pid"
      # The name holds ") " and numbers, as a native name may: only the LAST
      # ") " ends it, and a reader stopping at the first one reads a
      # constant field as the token.
      printf '%s (sl) 9 9) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %s 0 0\n' "\$pid" "\$start" >"\$TOK_PROC/\$pid/stat.tmp"
      mv -f "\$TOK_PROC/\$pid/stat.tmp" "\$TOK_PROC/\$pid/stat"
      printf '%s %s\n' "\$pid" "\$start" >>"\$TOK_SERVED"
    fi ;;
esac
exec "$(command -v ps)" "\$@"
EOF
chmod +x "$tok_bin/ps"
tok_detail=
for probe in "advance:1:0" "stable::142"; do
  IFS=: read -r tag advance want <<<"$probe"
  rm -rf "$tok_proc"
  : >"$TMP/test-cap-token-$tag.served"
  REPEAT_TEST_ACTION_PID=$TMP/test-cap-token-$tag.pid
  rm -f "$REPEAT_TEST_ACTION_PID"
  PATH=$tok_bin:$PATH TOK_PROC=$tok_proc TOK_ADVANCE=$advance TOK_SERVED=$TMP/test-cap-token-$tag.served \
    OCANNL_TOOL_TEST_CAP_PROC=$tok_proc REPEAT_TEST_ACTION_PID=$REPEAT_TEST_ACTION_PID \
    REPEAT_TEST_ACTION_SECS=7 argv_mode=action argv_probe "test-cap-token-$tag" run --test-cap 4 build @cheap
  tok_action=$(cat "$REPEAT_TEST_ACTION_PID" 2>/dev/null)
  tok_served=$(grep -c "^$tok_action " "$TMP/test-cap-token-$tag.served" 2>/dev/null)
  if [ "$argv_rc" != "$want" ]; then
    tok_detail="$tag: exit $argv_rc (want $want): $argv_out $(cat "$argv_dir/log")"
  elif [ -z "$tok_action" ] || [ "${tok_served:-0}" -lt 3 ]; then
    tok_detail="$tag: the stub served the test's stat ${tok_served:-0} time(s) (want 3 or more; test pid ${tok_action:-unknown})"
  elif [ "$want" = 0 ] && [ -e "$argv_dir/test-cap" ]; then
    tok_detail="$tag: a passing run left a test-cap record"
  elif [ "$want" = 142 ] && ! grep -q "^test-run: per-test cap: pid $tok_action has run" "$argv_dir/log"; then
    tok_detail="$tag: the log does not name the test: $(cat "$argv_dir/log")"
  fi
  case ${tok_action:+$(pstate "$tok_action")} in
    '' | Z) ;;
    *) kill -KILL "$tok_action" 2>/dev/null ;;
  esac
  [ -z "$tok_detail" ] || break
done
unset REPEAT_TEST_ACTION_PID
if [ -z "$tok_detail" ]; then report 0 "$tok_label"; else report 1 "$tok_label" "$tok_detail"; fi
}
harness_leg_run '91: the per-test cap tells a test from a process that took its pid #device' leg_91

test_run_close
