#!/usr/bin/env bash
# ocannl-harness: standalone
# Exercise gh1253_cells.sh's gate-pair preflight, cell order and census/row join.
# The scratch driver uses the real cells.py and a synthetic runner. A dune
# command recorder and the runner replace compilation, hardware and fixtures.
# Usage: benchmarks/test-gh1253-cells.sh [--keep|--only REGEX|--help]
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3
harness_require git
harness_scratch test-gh1253-cells
SRC="$here/gh1253_cells.sh"
tree="$TMP/tree"
exe="$tree/_build/default/benchmarks/runners/ocannl/bench_rank_site.exe"
mkdir -p "$tree/benchmarks" "$(dirname "$exe")" "$TMP/shims" "$TMP/runs"
cp "$here/cells.py" "$here/cell_group.py" "$here/config_records.py" "$here/display_safe.py" \
  "$here/strict_json.py" "$here/gh1253_gate_compare.py" "$tree/benchmarks/" || exit 2
git -C "$tree" init -q &&
  git -C "$tree" -c user.name=Fixture -c user.email=fixture@example.invalid \
    -c commit.gpgsign=false commit -q --allow-empty -m "Initialize scratch tree" || exit 2
cat >"$TMP/shims/dune" <<'SH'
#!/bin/bash
printf '%s\n' "$*" >>"$CALLS"
SH
cat >"$exe" <<'PY'
#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def record(key, value, origin=None):
    source = "commandline" if origin is not None else "default"
    print(json.dumps(dict(ocannl_config=1, key=key, value=value, source=source,
                          layer=source, profile=None, origin=origin),
                     separators=(",", ":")), file=sys.stderr)


opts = {}
for arg in sys.argv[1:]:
    key, value = arg.removeprefix("--ocannl_").split("=", 1)
    opts[key] = value
    record(key, value, arg)
arm = opts["gpu_default_sketch_min_reduction"]
assert opts["autotune_keep_fraction"] == "1"
assert os.environ["BENCH_TUNE"] == "1"
with open(os.environ["CALLS"], "a") as calls:
    print(os.environ["BENCH_SITE"], os.environ["BENCH_SITE_DIMS"],
          Path.cwd().name, arm, file=calls)
label = "F_seeded[2 segs, 1 seeded, 0 sunk, 0 split, plan aaaaaaaa]"
legacy = "F_seeded[2 segs, 1 seeded, 0 sunk, 0 split, plan bbbbbbbb]"
scenario = os.environ.get("STUB_MUTATION")
mutation = scenario if arm == "auto-legacy" else None
preset = "F_preset[bs=cfg cfg-thresh]"
preset_arm = mutation in ("preset", "preset-dedup", "preset-wrong-digest")
selected = label if arm == "auto" or scenario == "equivalent" else legacy
if preset_arm:
    selected = preset
record("model_staged_copy_rate", "none" if mutation == "leaf" else "")
capacity = 43 if mutation == "gate-drift" else 42
gate = f"ranked capacity={capacity} cache_rate=10"
if (arm == "auto" and scenario != "equivalent") or mutation == "legacy-delivery":
    gate += " copy_rate=20"
print("autotune-seeding: gate", gate, file=sys.stderr)
if mutation != "census":
    verdict = "refused: score tie" if selected == preset else "seeded: geometry"
    print('autotune-seeding: where="seeded_default segment u0"', verdict, file=sys.stderr)
print("autotune-seeding: plan", selected, file=sys.stderr)
preset_digest = "33333333/cccccccc"
if preset_arm:
    attribution = "44444444/dddddddd" if mutation == "preset-wrong-digest" else preset_digest
    print("autotune-seeding: preset-digest", attribution, file=sys.stderr)
with open(opts["autotune_calibration_file"], "w") as rows:
    auto_ms, legacy_ms = ("1.25", "8.00") if arm == "auto" else ("9.00", "2.50")
    preset_row = "W_preset[bs=256]" if mutation in ("preset-dedup", "preset-wrong-digest") else preset
    for name, digest, ms in ((label, "11111111/aaaaaaaa", auto_ms),
                             (legacy, "22222222/bbbbbbbb", legacy_ms),
                             (preset_row, preset_digest, "0.25")):
        if mutation == "row" and name == selected:
            continue
        print(opts["backend"], digest, "rs_stub", name, ms, "", 1, 4, 8, "false", "false",
              "false", sep="\t", file=rows)
PY
chmod +x "$TMP/shims/dune" "$exe"

sites='wgrad:1,32,32,32 outproj:1,16,32,2,4'
drive() { # SUBJECT RUN [ENV=VALUE ...]
  local subject=$1 run=$2
  shift 2
  mkdir -p "$TMP/runs/$run"
  cp "$subject" "$tree/benchmarks/gh1253_cells.sh" || return 2
  got=0
  env -i PATH="$TMP/shims:$PATH" HOME="$HOME" LC_ALL=C \
    CALLS="$TMP/runs/$run/calls" SITES="$sites" TIME_REPEATS=2 COMPARE_AUTO_GATE=1 "$@" \
    bash "$tree/benchmarks/gh1253_cells.sh" cuda "$TMP/runs/$run/out" 5 \
    >"$TMP/runs/$run/stdout" 2>&1 || got=$?
}

refuses() { # SUBJECT RUN
  drive "$1" "$2" PICK_PROFILES=performance &&
    [[ $got == 2 && ! -e "$TMP/runs/$2/calls" ]] &&
    grep -qF 'COMPARE_AUTO_GATE: PICK_PROFILES conflicts with this mode' "$TMP/runs/$2/stdout"
}
ordered() { # SUBJECT RUN
  drive "$1" "$2" && [[ $got == 0 ]] || return 1
  diff -u "$TMP/order" "$TMP/runs/$2/calls" >>"$TMP/runs/$2/stdout" 2>&1
}
joins() { # SUBJECT RUN
  drive "$1" "$2" && [[ $got == 0 ]] || return 1
  local report="$TMP/runs/$2/out/rank_wgrad_1x32x32x32_gate_r1.txt"
  grep -qE "^auto: joined=1 .*plan aaaaaaaa.*measured_ms=1\.250000$" "$report" &&
    grep -qE "^auto-legacy: joined=1 .*plan bbbbbbbb.*measured_ms=2\.500000$" "$report" &&
    grep -qxF 'ranked gates: operand delivery priced by auto only' "$report" &&
    [[ $(grep -c '^autotune-seeding: where=' "$report") == 2 ]]
}
early_exit() { # RUN
  [[ $got == 2 && ! -e "$TMP/runs/$1/calls" ]]
}
check() { # LABEL COMMAND...
  local label=$1
  shift
  if "$@"; then report 0 "$label"; else report 1 "$label" "see $TMP/runs (--keep)"; fi
}

cat >"$TMP/order" <<'ORDER'
build -j 4 benchmarks/runners/ocannl/bench_rank_site.exe
wgrad 1,32,32,32 rank_wgrad_1x32x32x32_gate_r1_auto auto
wgrad 1,32,32,32 rank_wgrad_1x32x32x32_gate_r1_auto-legacy auto-legacy
wgrad 1,32,32,32 rank_wgrad_1x32x32x32_gate_r2_auto-legacy auto-legacy
wgrad 1,32,32,32 rank_wgrad_1x32x32x32_gate_r2_auto auto
outproj 1,16,32,2,4 rank_outproj_1x16x32x2x4_gate_r1_auto auto
outproj 1,16,32,2,4 rank_outproj_1x16x32x2x4_gate_r1_auto-legacy auto-legacy
outproj 1,16,32,2,4 rank_outproj_1x16x32x2x4_gate_r2_auto-legacy auto-legacy
outproj 1,16,32,2,4 rank_outproj_1x16x32x2x4_gate_r2_auto auto
ORDER

if harness_leg '1: refuses incompatible options before building'; then
  check 'the compare mode rejects PICK_PROFILES before building' refuses "$SRC" refuse
  for spec in 'TIME_REPEATS=0|TIME_REPEATS must be a positive integer' \
    'SITES=outproj:1,16,0,2,4|invalid dimensions' \
    'SITES=wgrad:1,32,32,32 wgrad:1,32,32,32|duplicate site'; do
    setting=${spec%%|*}
    run="refuse-${spec#*|}"
    drive "$SRC" "$run" "$setting"
    check "$setting is refused before building" early_exit "$run"
    check "$setting names the conflict" grep -qF "COMPARE_AUTO_GATE: ${spec#*|}" "$TMP/runs/$run/stdout"
  done
fi
if harness_leg '2: both auto gates run in alternating order at every site'; then
  check 'the planned cells run with the two actual gate arguments' ordered "$SRC" order
fi
if harness_leg '3: the two censuses join to their own exact timing rows'; then
  check 'the two census plans join the matching rows despite both labels being present' joins "$SRC" join
  drive "$SRC" preset STUB_MUTATION=preset
  check 'a pair with a preset arm succeeds' test "$got" = 0
  check 'the preset census joins the configured preset even when seeded rows are present' \
    grep -qE "^auto-legacy: joined=1 label='F_preset.*measured_ms=0\.250000$" \
      "$TMP/runs/preset/out/rank_wgrad_1x32x32x32_gate_r1.txt"
  drive "$SRC" dedup STUB_MUTATION=preset-dedup
  check 'a preset joins its deduplicated schedule by digest' test "$got" = 0
  check 'the joined row names the candidate that was timed' \
    grep -qE "^auto-legacy: joined=1 label='F_preset.*timed_as='W_preset.*measured_ms=0\.250000$" \
      "$TMP/runs/dedup/out/rank_wgrad_1x32x32x32_gate_r1.txt"
  drive "$SRC" equivalent STUB_MUTATION=equivalent
  check 'a backend with no copy rate can join both arms' test "$got" = 0
  check 'the report identifies equivalent gates when neither has a copy rate' \
    grep -qxF 'ranked gates: identical (auto resolved no operand delivery rate)' \
      "$TMP/runs/equivalent/out/rank_wgrad_1x32x32x32_gate_r1.txt"
fi
if harness_leg '4: mutated twins are rejected'; then
  no_refusal=$(mutant no-refusal '/^  for knob in EXPLAIN_ONLY / { sub(/ PICK_PROFILES /, " ") } { print }') || exit 2
  expect_rejected 'conflicting pick option slips through' "$no_refusal" refuses '^gh1253_cells: 0 failed'
  fixed_order=$(mutant fixed-order '/if \(\(rep % 2/ { next } { print }') || exit 2
  expect_rejected 'same arm order on every repeat' "$fixed_order" ordered '^--- '
  for mutation in row census leaf preset-wrong-digest legacy-delivery gate-drift; do
    run="mutant-$mutation"
    drive "$SRC" "$run" "STUB_MUTATION=$mutation"
    case "$mutation" in
      row) diagnostic="expected one timing row" ;;
      census) diagnostic="missing seeded-default segment census" ;;
      leaf) diagnostic="resolved configuration differs: model_staged_copy_rate" ;;
      preset-wrong-digest) diagnostic="at digest 44444444/dddddddd, found 0" ;;
      legacy-delivery | gate-drift) diagnostic="ranked gate tags must match" ;;
    esac
    check "$mutation mutation fails the pair" test "$got" = 1
    check "$mutation mutation names the failed join" grep -qF "$diagnostic" "$TMP/runs/$run/stdout"
  done
fi
finish
