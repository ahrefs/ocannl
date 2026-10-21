#!/usr/bin/env bash
# Check, never provision, what a measurement box needs before it measures (gh-ocannl-1062).
#
#   benchmarks/check-box.sh --box BOX [--fixture NAME]... [--fixture-dir DIR] [--venv DIR]
#                           [--digests FILE] [--box-only]
#   benchmarks/check-box.sh --fleet-names    # the fleet names --box takes, `<name> <box>` lines
#   benchmarks/check-box.sh --canonical-box NAME  # NAME's measurement box if NAME is a fleet
#                                                 # name, else NAME itself (box-gate's reading)
#
# Every measurement session runs it first, from the checkout it will measure in; tools/sweep.sh
# runs it once per box with --box-only against the box's fixture store. Each item prints one
# `check-box: <item>: ok|GAP|skip -- <detail>` line, a GAP adds a `repair:` line naming the fix
# (ludics-lite's `scripts/install-linux.sh --bench-venv`, a fixture copy or the generator), and the
# last line is the verdict: `check-box: <box>: READY` or `check-box: <box>: GAP (<n> of <m> items)`.
# Exit 0 READY, 1 any gap, 2 a usage or harness error (nothing was judged).
# The device item runs one tensor through the GPU: on a fleet box take a GPU slot for it
# (`fleet-worker.sh execution slot` without --cpu); there is deliberately no flag to skip it,
# since the round trip is the item that caught minix's segfaulting torch.
#
# What it accepts -- a fail-closed allowlist; anything else exits 2 rather than being guessed at:
#   - BOX: a measurement box declared in the DIGESTS file's `# measurement-boxes:` header AND
#     listed in box_profile below, which names the device torch must reach and the toolchain
#     the box's drivers compile with. A declared box missing here is refused, not defaulted.
#     Or the fleet's name for one (`rog-nv-linux` for rog-nv; fleet_box below, gh-ocannl-1433),
#     which a line before the items resolves to its box; the verdict names the box.
#   - NAME: the stem of a spec in benchmarks/workloads/*.json (`gpt2_mini_train`, not the file
#     name). Without --fixture, every fixture DIGESTS records under BOX's origin is required.
#   - DIR, FILE: paths; --venv defaults to ~/.venvs/ocannl-bench, the box's shared venv,
#     --fixture-dir to this checkout's benchmarks/fixtures (what orchestrate.py reads), --digests
#     to this checkout's fixtures/DIGESTS.txt -- so a box's stale main checkout never judges
#     fixtures recorded since (the issue's 2026-10-12 evidence).
#
# The items, in order: venv (its interpreter runs), packages (numpy, safetensors, tinygrad, torch
# import), requirements (`pip freeze` equals `<venv>.requirements.txt`, the freeze provisioning
# records), torch-device (the box's torch build round-trips one tensor through its device -- the
# copy that segfaulted minix's rocm7.1 wheel while is_available() said yes), python-headers
# (Python.h, which torch.compile's launcher builds against), driver-path (this checkout's drivers,
# resolving their interpreter through bench_venv.py as they do, reach that same venv, and the ok
# line names the route that resolved: BENCH_VENV_PY, benchmarks/.venv, or the shared venv a fresh
# worktree falls back to; --box-only
# skips it, since a sweep worktree is no measurement checkout), toolchain (nvcc, hipcc, or the
# macOS SDK the Metal bindings build against), and one `fixture <name>` item per required fixture
# (present, and fixture_digest.py --check MATCHes it against DIGESTS).
#
# What it does not judge: whether a fixture is THIS box's recorded bytes (the ok line names whose
# they are; reports carry the same origin), the speed of anything, or a driver's own arms
# (torch.compile, tinygrad's GPU device) beyond their imports.

set -u

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
usage() { sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "check-box: $*" >&2; exit 2; }

# <box> -> "<device> <toolchain>". The device is what torch must reach (cuda, hip, mps); the
# toolchain item checks nvcc, hipcc, or metal-sdk (`xcrun --show-sdk-path`). OCANNL's Metal
# backend and tinygrad compile MSL at run time, so the `metal` command-line compiler is no
# prerequisite and is not checked.
box_profile() {
  case $1 in
    m4-max) echo "mps metal-sdk" ;;
    rog-nv) echo "cuda nvcc" ;;
    minix | tuf) echo "hip hipcc" ;;
    *) return 1 ;;
  esac
}

# The fleet's name for a measurement box -> that box (gh-ocannl-1433). The fleet names its hosts by
# endpoint (FLEET_BOXES and FLEET_LOCAL_BOX, lukstafi/ludics-lite's issue-wave registry: the native
# Linux boot's ssh alias in wake-lab.sh's endpoint map, and the Mac's own name), while DIGESTS.txt
# and the reports name the box, so a worker given a fleet box passed the fleet's name and was
# refused. The table is here, not derived at run time from wake-lab.sh's map: the far boxes this
# check runs on have no wake-lab.sh, the map has no row for the Mac, and its box names (`rog`) are
# not the measurement IDs (`rog-nv`) either. benchmarks/test-check-box.sh pins it against the map,
# and against FLEET_BOXES, wherever those can be read. Only the native boots: no box measures from
# a WSL guest or from Windows.
fleet_box() {
  case $1 in
    mac-studio) echo m4-max ;;
    rog-nv-linux) echo rog-nv ;;
    minix-amd-linux) echo minix ;;
    tuf-amd-linux) echo tuf ;;
    *) return 1 ;;
  esac
}
fleet_names="mac-studio rog-nv-linux minix-amd-linux tuf-amd-linux" # fleet_box's domain

box= venv=$HOME/.venvs/ocannl-bench fixture_dir=$here/fixtures digests= box_only=0
fixtures=()
while [ $# -gt 0 ]; do
  case $1 in
    --box | --fixture | --fixture-dir | --venv | --digests)
      [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value"
      case $1 in
        --box) box=$2 ;;
        --fixture) fixtures+=("$2") ;;
        --fixture-dir) fixture_dir=$2 ;;
        --venv) venv=${2%/} ;;
        --digests) digests=$2 ;;
      esac
      shift 2
      ;;
    --box-only) box_only=1; shift ;;
    --fleet-names)
      [ $# -eq 1 ] || die "--fleet-names takes nothing else"
      for name in $fleet_names; do printf '%s %s\n' "$name" "$(fleet_box "$name")"; done
      exit 0
      ;;
    --canonical-box)
      # The name a driver records for its run (box-gate.sh, box_gate.py): the same resolution
      # as --box below, which judges it -- no fleet name is a declared box (test-check-box.sh).
      [ $# -eq 2 ] && [ -n "$2" ] || die "--canonical-box takes one name and nothing else"
      fleet_box "$2" || printf '%s\n' "$2"
      exit 0
      ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
done
[ -n "$box" ] || die "--box BOX is required: the measurement box this checks (see --help)"
# This checkout's DIGESTS by default, even when --fixture-dir names another tree's store.
[ -n "$digests" ] || digests=$here/fixtures/DIGESTS.txt
case $digests in /*) ;; *) digests=$PWD/$digests ;; esac # read again after a cd below
command -v python3 >/dev/null 2>&1 || die "no python3 on PATH (fixture_digest.py needs it)"
command -v perl >/dev/null 2>&1 || die "no perl on PATH (the device smoke is bounded through it)"
[ -r "$digests" ] || die "cannot read $digests"
[ -d "$fixture_dir" ] || die "no fixture directory $fixture_dir"

declared=$(python3 "$here/fixture_digest.py" --list-declared-measurement-boxes --digests "$digests") ||
  die "cannot read the declared measurement boxes of $digests"
if ! printf '%s\n' "$declared" | grep -qxF -- "$box"; then
  fleet_name=$box
  box=$(fleet_box "$fleet_name") ||
    die "'$fleet_name' is not a measurement box declared by $digests (declared: $(printf '%s' "$declared" | tr '\n' ' ')), nor the fleet's name for one ($fleet_names)"
  printf '%s\n' "$declared" | grep -qxF -- "$box" ||
    die "'$fleet_name' is the fleet's name for $box, which $digests does not declare (declared: $(printf '%s' "$declared" | tr '\n' ' '))"
  echo "check-box: --box $fleet_name is the fleet's name for measurement box $box"
fi
profile=$(box_profile "$box") ||
  die "'$box' is declared but has no box_profile entry in $0: add its device and toolchain there"
device=${profile% *} toolchain=${profile#* }

workloads=()
for spec in "$here"/workloads/*.json; do
  [ -f "$spec" ] || continue
  spec=${spec##*/}
  workloads+=("${spec%.json}")
done
[ ${#workloads[@]} -gt 0 ] || die "no workload specs under $here/workloads"
# Without --fixture: every fixture DIGESTS records bytes for under this box's origin -- the ones
# the box has measured on -- read through fixture_digest.py, the one parser of that file.
if [ ${#fixtures[@]} -eq 0 ]; then
  recorded=$(cd "$here" && python3 -c '
import sys, fixture_digest
for name, rows in sorted(fixture_digest.read_digests(sys.argv[1]).items()):
    if any(e.origin == sys.argv[2] for e in rows):
        print(name[: -len(".safetensors")] if name.endswith(".safetensors") else name)' \
    "$digests" "$box") || die "cannot read the entries of $digests"
  while IFS= read -r name; do [ -z "$name" ] || fixtures+=("$name"); done <<<"$recorded"
  [ ${#fixtures[@]} -gt 0 ] || die "$digests records no fixture for $box: name the ones to check with --fixture"
else
  for name in "${fixtures[@]}"; do
    found=0
    for w in "${workloads[@]}"; do [ "$w" = "$name" ] && found=1; done
    [ "$found" = 1 ] || die "--fixture '$name' names no spec in $here/workloads (give the stem: gpt2_mini_train)"
  done
fi

items=0 gaps=0
item() { # item status detail [repair]
  items=$((items + 1))
  printf 'check-box: %s: %s -- %s\n' "$1" "$2" "$3"
  if [ "$2" = GAP ]; then
    gaps=$((gaps + 1))
    [ $# -lt 4 ] || printf '  repair: %s\n' "$4"
  fi
}
indent() { sed 's/^/    /'; }
# Bounded by SIGALRM (exit 142): macOS has no timeout(1), and a wedged GPU must not hang the check.
# OCANNL_TOOL_CHECK_BOX_DEVICE_CAP moves the device smoke's bound (seconds; its harness's hang leg).
device_cap=${OCANNL_TOOL_CHECK_BOX_DEVICE_CAP:-300}
case $device_cap in "" | *[!0-9]* | 0) die "OCANNL_TOOL_CHECK_BOX_DEVICE_CAP must be a positive number of seconds" ;; esac
bounded() { perl -e 'alarm shift; exec @ARGV or exit 127' "$@"; }

case $device in
  mps) provision="build $venv per benchmarks/README.md#setup (install-linux.sh provisions Linux boxes only)" ;;
  cuda) provision="run lukstafi/ludics-lite's scripts/install-linux.sh --bench-venv nvidia on this box" ;;
  hip) provision="run lukstafi/ludics-lite's scripts/install-linux.sh --bench-venv amd on this box" ;;
esac

py=$venv/bin/python
if ! out=$("$py" -c 'import sys; print(sys.version.split()[0])' 2>&1); then
  item venv GAP "no working interpreter at $py" "$provision"
  for i in packages requirements torch-device python-headers driver-path; do
    item "$i" skip "no venv to check"
  done
else
  item venv ok "$venv (python $out)"

  if out=$("$py" -c '
import importlib.metadata as m
import numpy, safetensors, tinygrad, torch
print(" ".join("%s %s" % (d, m.version(d)) for d in ("numpy", "safetensors", "tinygrad", "torch")))' 2>&1); then
    item packages ok "$out"
  else
    item packages GAP "an import failed: $(printf '%s\n' "$out" | tail -n 1)" "$provision"
  fi

  # pip's freeze, the format provisioning records; uv's for a venv uv built without pip. The one
  # spelling the two differ in, an editable's `file://` prefix, is normalized on both sides.
  record=$venv.requirements.txt
  normalize() { sed 's|^-e file://|-e |'; }
  if [ ! -r "$record" ]; then
    item requirements GAP "no $record: nothing records what this venv should hold" "$provision (it records the freeze)"
  elif ! { freeze=$("$py" -m pip freeze --disable-pip-version-check 2>&1) ||
    { command -v uv >/dev/null 2>&1 && freeze=$(uv pip freeze --quiet --python "$py" 2>&1); }; }; then
    item requirements GAP "cannot list the venv's packages (no pip in it, no uv on PATH): $(printf '%s\n' "$freeze" | tail -n 1)" "$provision"
  elif drift=$(printf '%s\n' "$freeze" | normalize | diff <(normalize <"$record") - 2>&1); then
    item requirements ok "the installed packages match $record"
  else
    item requirements GAP "the installed packages differ from $record (< recorded, > installed):" \
      "$provision -- it rebuilds to the pins and re-records; or find who changed the venv"
    printf '%s\n' "$drift" | grep '^[<>]' | head -n 12 | indent
  fi

  if out=$(bounded "$device_cap" "$py" -X faulthandler -c '
import sys, torch
kind = sys.argv[1]
if kind == "mps":
    if not torch.backends.mps.is_available():
        sys.exit("torch.backends.mps.is_available() is False (torch %s)" % torch.__version__)
    x = torch.ones(1, device="mps")
    torch.mps.synchronize()
    where = "mps"
else:
    build = torch.version.cuda if kind == "cuda" else torch.version.hip
    if not build:
        sys.exit("torch %s is not a %s build" % (torch.__version__, kind))
    if not torch.cuda.is_available():
        sys.exit("torch.cuda.is_available() is False (torch %s)" % torch.__version__)
    x = torch.ones(1).cuda()
    torch.cuda.synchronize()
    where = "%s %s, %s" % (kind, build, torch.cuda.get_device_name(0))
if x.cpu().item() != 1.0:
    sys.exit("torch.ones(1) did not round-trip through " + kind)
print("torch %s: one tensor round-tripped through %s" % (torch.__version__, where))' "$device" 2>&1); then
    item torch-device ok "$out"
  else
    rc=$?
    [ "$rc" != 142 ] || out="no answer within ${device_cap}s (a wedged device?)"
    item torch-device GAP "$device: $(printf '%s\n' "$out" | tail -n 1)" "$provision"
  fi

  if out=$("$py" -c '
import os, sys, sysconfig
h = os.path.join(sysconfig.get_config_var("INCLUDEPY") or "", "Python.h")
print(h)
sys.exit(0 if os.path.isfile(h) else 1)' 2>&1); then
    item python-headers ok "$out"
  else
    item python-headers GAP "no $out: torch.compile cannot build its launcher" \
      "install the interpreter's development headers (Ubuntu: sudo apt-get install -y python3-dev)"
  fi

  if [ "$box_only" = 1 ]; then
    item driver-path skip "--box-only: no measurement checkout to resolve from"
  else
    prefix='import os, sys; print(os.path.realpath(sys.prefix))'
    # `<interpreter><TAB><route>`, bench_venv.py's own report of what it resolved and how.
    # A failed run, or anything but one line with one tab, resolved nothing.
    resolved_ok=
    if resolved=$(cd "$here" && python3 bench_venv.py 2>&1); then
      case $resolved in *$'\n'* | *$'\t'*$'\t'* | $'\t'* | *$'\t') ;; *$'\t'*) resolved_ok=1 ;; esac
    fi
    if [ -z "$resolved_ok" ]; then
      item driver-path GAP "bench_venv.py did not resolve: $(printf '%s\n' "$resolved" | tail -n 1)"
    else
      route=${resolved#*$'\t'} resolved=${resolved%%$'\t'*}
      if theirs=$("$resolved" -c "$prefix" 2>/dev/null) && [ "$theirs" = "$("$py" -c "$prefix")" ]; then
        item driver-path ok "drivers here run $resolved (route: $route)"
      else
        item driver-path GAP "drivers in $(dirname "$here") would run $resolved (route: $route), which is not $venv" \
          "ln -s $venv $here/.venv, or export BENCH_VENV_PY=$py"
      fi
    fi
  fi
fi

case $toolchain in
  nvcc | hipcc)
    if found=$(command -v "$toolchain"); then
      if version=$(bounded 60 "$toolchain" --version 2>&1); then
        item toolchain ok "$found ($(printf '%s\n' "$version" | grep -i -m1 -E 'release|version'))"
      else
        item toolchain GAP "$found --version failed: $(printf '%s\n' "$version" | tail -n 1)" \
          "repair the $([ "$toolchain" = nvcc ] && echo CUDA toolkit || echo ROCm HIP SDK) install"
      fi
    else
      item toolchain GAP "no $toolchain on PATH" \
        "install the $([ "$toolchain" = nvcc ] && echo CUDA toolkit || echo ROCm HIP SDK) or put its bin/ on PATH"
    fi
    ;;
  metal-sdk)
    if found=$(xcrun --sdk macosx --show-sdk-path 2>&1); then
      item toolchain ok "macOS SDK $found"
    else
      item toolchain GAP "xcrun cannot find the macOS SDK: $found" "xcode-select --install"
    fi
    ;;
esac

for name in "${fixtures[@]}"; do
  path=$fixture_dir/$name.safetensors
  if [ ! -e "$path" ]; then
    item "fixture $name" GAP "absent from $fixture_dir" \
      "copy the box's recorded bytes in and check them (benchmarks/README.md, 'Rebuilding a fixture that vanished'); install-linux.sh --bench-venv copies the anchor's into ~/ocannl-staging -- never regenerate to pass"
  elif out=$(python3 "$here/fixture_digest.py" --check --digests "$digests" "$path" 2>&1); then
    # Some box's recorded bytes, so measurable; whether they are THIS box's is the reports'
    # origin field, said here too because a box whose own row no longer describes its bytes
    # (rog-nv's lenet after the 2026-10-12 copy) is a record to fix, not a measurement to refuse.
    whose=
    python3 "$here/fixture_digest.py" --check --expect-origin "$box" --digests "$digests" "$path" \
      >/dev/null 2>&1 || whose=" (not $box's recorded bytes)"
    item "fixture $name" ok "$(printf '%s\n' "$out" | head -n 1 | sed 's/^[^ ]* //')$whose"
  else
    item "fixture $name" GAP "its bytes match no row of $digests:" \
      "see benchmarks/README.md on digest mismatches -- regeneration is a cross-box event, not a repair"
    printf '%s\n' "$out" | indent
  fi
done

if [ "$gaps" = 0 ]; then
  echo "check-box: $box: READY ($items items)"
  exit 0
fi
echo "check-box: $box: GAP ($gaps of $items items)"
exit 1
