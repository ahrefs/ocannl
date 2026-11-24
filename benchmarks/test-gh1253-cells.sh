#!/usr/bin/env bash
# ocannl-harness: standalone
# Exercise gh1253_cells.sh's gate-pair preflight, cell order and census/row join.
# The stock ranking report and its replay also join the incumbent by attribution.
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
  "$here/strict_json.py" "$here/bench_output.py" "$here/gh1253_gate_compare.py" \
  "$here/gh1442_agreement.py" "$here/gh1600_txn_fit.py" "$here/gh1821_geometry_fit.py" \
  "$tree/benchmarks/" || exit 2
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
arm = opts.get("gpu_default_sketch_min_reduction", "32")
scenario = os.environ.get("STUB_MUTATION")
preset_digest = "33333333/cccccccc"
preset_timed_as = "W_preset[bs=256]"
preset = "F_preset[bs=cfg cfg-thresh]"
if arm == "32":
    stock_seed = "F_seeded[stock]"
    profile_seed = "F_seeded[profile]"
    serial = scenario in (None, "stock-serial-ambiguous", "stock-serial-profile")
    preset_untimed = scenario in ("stock-preset-unused", "stock-preset-needed")
    incumbent = "seeded" if scenario in ("stock-seeded-preset", "stock-preset-unused") else "default"
    if os.environ.get("BENCH_TUNE") != "1":
        def emit(label, rank_ms):
            print(f'rank-site: explain label="{label}" rank_ms={rank_ms} admitted=true')
            print(f'rank-site: kernels label="{label}" kernels="c={rank_ms},m=0.0,txn=0,txn64=0,txn128=0,unc=0,wg=0,thr=32"')
        emit(incumbent, "0.5")
        if serial:
            emit("seeded", "0.75")
        emit("W_sketch[control]", "1.25")
        if preset_untimed:
            emit("W_sketch[second]", "1.50")
        if os.environ.get("BENCH_PICK") == "1":
            seeded_label = profile_seed if "profile" in opts else "none"
            print(f'rank-site: pick label="seeded" seeded_label="{seeded_label}"')
        sys.exit()
    if "profile" in opts:
        if scenario == "stock-serial-profile":
            rows = [("baseline", "22222222/bbbbbbbb", "0.05"),
                    ("W_sketch[control]", "11111111/aaaaaaaa", "0.05"),
                    (profile_seed, "55555555/eeeeeeee", "0.125")]
        else:
            rows = [(preset_timed_as, preset_digest, "0.05"),
                    (profile_seed, "55555555/eeeeeeee", "0.125"),
                    ("W_sketch[control]", "11111111/aaaaaaaa", "1.25")]
    elif scenario in ("stock-preset", "stock-seeded-preset", "stock-preset-conflict") or preset_untimed:
        if opts.get("autotune_log_seeding") == "true":
            plan = stock_seed if scenario == "stock-preset-unused" else preset
            print("autotune-seeding: plan", plan, file=sys.stderr)
            if os.environ.get("BENCH_PICK") == "1":
                pick_plan = stock_seed if scenario == "stock-preset-conflict" else plan
                print("autotune-seeding: plan", pick_plan, file=sys.stderr)
            print("autotune-seeding: preset-digest", preset_digest, file=sys.stderr)
        if preset_untimed:
            rows = [(stock_seed, "66666666/ffffffff", "0.50"),
                    ("W_sketch[second]", "77777777/aaaaaaaa", "1.50"),
                    ("W_sketch[control]", "11111111/aaaaaaaa", "1.25")]
        else:
            ms = "0.25" if Path.cwd().name.endswith("_r2") else "0.50"
            rows = [(preset_timed_as, preset_digest, ms),
                    (preset_timed_as, "44444444/dddddddd", "0.10"),
                    ("W_sketch[control]", "11111111/aaaaaaaa", "1.25")]
    elif scenario == "stock-serial-profile":
        if opts.get("autotune_log_seeding") == "true":
            print("autotune-seeding: plan", stock_seed, file=sys.stderr)
        r2 = Path.cwd().name.endswith("_r2")
        rows = [("baseline", "22222222/bbbbbbbb", "0.40" if r2 else "0.50"),
                (stock_seed, "66666666/ffffffff", "0.75"),
                ("W_sketch[control]", "11111111/aaaaaaaa", "1.00" if r2 else "1.25")]
    else:
        if opts.get("autotune_log_seeding") == "true":
            print("autotune-seeding: plan", stock_seed, file=sys.stderr)
        rows = [("baseline", "22222222/bbbbbbbb", "0.50"),
                (stock_seed, "66666666/ffffffff", "0.75"),
                ("W_sketch[control]", "11111111/aaaaaaaa", "1.25")]
        if scenario == "stock-serial-ambiguous":
            rows.append((profile_seed, "55555555/eeeeeeee", "0.25"))
    with open(opts["autotune_calibration_file"], "a") as tsv:
        for name, digest, ms in rows:
            print(opts["backend"], digest, "rs_stub", name, ms, "", 1, 4, 8, "false", "false",
                  "false", sep="\t", file=tsv)
    sys.exit()

assert opts["autotune_keep_fraction"] == "1"
assert os.environ["BENCH_TUNE"] == "1"
with open(os.environ["CALLS"], "a") as calls:
    print(os.environ["BENCH_SITE"], os.environ["BENCH_SITE_DIMS"],
          Path.cwd().name, arm, file=calls)
label = "F_seeded[2 segs, 1 seeded, 0 sunk, 0 split, plan aaaaaaaa]"
legacy = "F_seeded[2 segs, 1 seeded, 0 sunk, 0 split, plan bbbbbbbb]"
mutation = scenario if arm == "auto-legacy" else None
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
if preset_arm:
    attribution = "44444444/dddddddd" if mutation == "preset-wrong-digest" else preset_digest
    print("autotune-seeding: preset-digest", attribution, file=sys.stderr)
with open(opts["autotune_calibration_file"], "w") as rows:
    auto_ms, legacy_ms = ("1.25", "8.00") if arm == "auto" else ("9.00", "2.50")
    preset_row = preset_timed_as if mutation in ("preset-dedup", "preset-wrong-digest") else preset
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
if harness_leg '5: stock agreement uses preset attribution and retains the serial lookup'; then
  site=rank_wgrad_1x32x32x32
  stock_sites='SITES=wgrad:1,32,32,32'
  preset_row="  shipped: 'default' measured_ms=0.250000 rank_ms=0.500000 admitted=true timed_as='W_preset[bs=256]'"
  drive "$SRC" stock COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-preset BENCH_PICK=1
  check 'the stock run accepts agreeing plan records from pick and tune' test "$got" = 0
  stock_out="$TMP/runs/stock/out"
  check 'the stock timing log contains the repeated plan' test "$(
    grep -c '^autotune-seeding: plan ' "$stock_out/${site}_time.err")" = 2
  check 'stock default uses the fastest row of the attributed digest' \
    grep -qxF "$preset_row" "$stock_out/${site}_agreement.txt"
  drive "$SRC" replay COMPARE_AUTO_GATE=0 "$stock_sites" TIME_REPEATS=1 EXPLAIN_ONLY="$stock_out"
  check 'the preset join also works with archived timing rows and censuses' test "$got" = 0
  check 'the replay attributes the same measured row' \
    grep -qxF "$preset_row" "$TMP/runs/replay/out/${site}_agreement.txt"
  drive "$SRC" conflict COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-preset-conflict BENCH_PICK=1
  check 'conflicting plans within a stock timing log fail agreement' test "$got" = 1
  check 'the agreement reader diagnoses the conflicting plan values' \
    grep -qF 'conflicting autotune-seeding: plan values' "$TMP/runs/conflict/stdout"
  # The serial report below is the base reader's output for the single-seeded-row fixture.
  # A second seeded row defeats the old uniqueness heuristic.
  serial_arms=(ceilings nofill nomma mmaonly noocc notxn nostg noilp noshl nolane shipped)
  {
    for arm in "${serial_arms[@]}"; do
      cat <<SERIAL
$arm: joined=3 decided_pairs=3 ordered_right=3 (100.0%) argmin='default' measured_ms=0.500000 fastest='default' fastest_ms=0.500000 argmin_slowdown=1.000
  $arm: 'default' measured_ms=0.500000 rank_ms=0.500000 admitted=true
  $arm: 'seeded' measured_ms=0.750000 rank_ms=0.750000 admitted=true
  $arm: 'W_sketch[control]' measured_ms=1.250000 rank_ms=1.250000 admitted=true
$arm: pick_set=admitted members=3 argmin='default' measured_ms=0.500000 rank_ms=0.500000 fastest='default' fastest_ms=0.500000 rank_ms_of_fastest=0.500000 argmin_slowdown=1.000
$arm: pick_set=admitted+coalesced members=3 argmin='default' measured_ms=0.500000 rank_ms=0.500000 fastest='default' fastest_ms=0.500000 rank_ms_of_fastest=0.500000 argmin_slowdown=1.000
SERIAL
      printf '%s: pick none (no `rank-site: pick` line)\n' "$arm"
    done
    for arm in "${serial_arms[@]:1}"; do
      printf '%s against ceilings: common_decided=3 regressions=0 improvements=0\n' "$arm"
    done
  } > "$TMP/serial-agreement.expected"
  drive "$SRC" serial COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites"
  check 'an unconfigured stock run succeeds' test "$got" = 0
  check 'the complete unconfigured agreement matches the base' \
    diff -u "$TMP/serial-agreement.expected" "$TMP/runs/serial/out/${site}_agreement.txt"
  drive "$SRC" serial-ambiguous COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-serial-ambiguous
  check 'the logged seeded plan disambiguates a second seeded label' test "$got" = 0
  check 'the extra seeded row leaves the agreement tied to the census plan' \
    diff -u "$TMP/serial-agreement.expected" "$TMP/runs/serial-ambiguous/out/${site}_agreement.txt"

  drive "$SRC" profile COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-seeded-preset PICK_PROFILES=reproducible
  check 'the stock preset and explicit profile pick both join' test "$got" = 0
  profile_out="$TMP/runs/profile/out"
  seeded_row="${preset_row/default/seeded}"
  check 'the seeded incumbent uses the stock preset despite the faster profile row' \
    grep -qxF "$seeded_row" "$profile_out/${site}_agreement.txt"
  check 'the extra timing cell supplied its explicit profile pick row' \
    grep -qF "reproducible: pick label='seeded' seeded_label='F_seeded[profile]' row='F_seeded[profile]' measured_ms=0.125000" \
      "$profile_out/${site}_agreement.txt"
  drive "$SRC" profile-replay COMPARE_AUTO_GATE=0 "$stock_sites" TIME_REPEATS=1 \
    EXPLAIN_ONLY="$profile_out" STUB_MUTATION=stock-seeded-preset
  check 'the archived stock preset and profile rows can be replayed' test "$got" = 0
  check 'the replay keeps the stock timing for the seeded incumbent' \
    grep -qxF "$seeded_row" "$TMP/runs/profile-replay/out/${site}_agreement.txt"

  mixed="$profile_out/$site.tsv"
  stock="${mixed}.stock"
  check 'the mixed TSV marks that a stock snapshot is required' grep -qxF \
    '# gh1253_cells: stock snapshot required' "$mixed"
  replay_args=()
  for arm in "${serial_arms[@]}"; do
    replay_args+=("$arm=$profile_out/${site}_$arm.out")
  done
  replay_args+=(--pick "reproducible=$profile_out/${site}_pick_reproducible.out"
    --census "$profile_out/${site}_time.err" --census "$profile_out/${site}_time_r2.err")
  reader="$tree/benchmarks/gh1442_agreement.py"
  got=0
  python3 "$reader" "$mixed" "${replay_args[@]}" \
    >"$TMP/runs/profile/hand-replay.out" 2>"$TMP/runs/profile/hand-replay.err" || got=$?
  check 'hand replay without the stock snapshot refuses' test "$got" = 2
  check 'the refusal names the missing stock input' grep -qF \
    'stock_snapshot_missing:' "$TMP/runs/profile/hand-replay.err"
  check 'the refusal emits no agreement report' test ! -s "$TMP/runs/profile/hand-replay.out"
  got=0
  python3 "$reader" "$mixed" "${replay_args[@]}" --stock-tsv "$stock" \
    >"$TMP/runs/profile/hand-complete.out" 2>"$TMP/runs/profile/hand-complete.err" || got=$?
  check 'hand replay with the stock snapshot succeeds' test "$got" = 0
  check 'hand replay reproduces the complete agreement byte for byte' cmp \
    "$profile_out/${site}_agreement.txt" "$TMP/runs/profile/hand-complete.out"

  partial="$(cd "$TMP" && pwd)/runs/profile-partial"
  mkdir -p "$partial"
  cp "$mixed" "$profile_out/${site}_time.err" "$profile_out/${site}_time_r2.err" "$partial/"
  drive "$SRC" partial-replay COMPARE_AUTO_GATE=0 "$stock_sites" TIME_REPEATS=1 \
    EXPLAIN_ONLY="$partial" STUB_MUTATION=stock-seeded-preset
  check 'EXPLAIN_ONLY refuses a partial archive before building' early_exit partial-replay
  check 'EXPLAIN_ONLY names the missing snapshot path' grep -qF \
    "stock_snapshot_missing: $partial/$site.tsv.stock" "$TMP/runs/partial-replay/stdout"

  drive "$SRC" unused COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" STUB_MUTATION=stock-preset-unused
  check 'an untimed preset leaves a seeded agreement report usable' test "$got" = 0
  check 'the separately timed seeded plan still joins' \
    grep -qxF "  shipped: 'seeded' measured_ms=0.500000 rank_ms=0.500000 admitted=true" \
      "$TMP/runs/unused/out/${site}_agreement.txt"
  drive "$SRC" needed COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" STUB_MUTATION=stock-preset-needed
  check 'an untimed preset fails an agreement that needs it' test "$got" = 1
  check 'the missing preset names its digest' grep -qxF \
    "shipped: preset_unmatched label='default' digest=33333333/cccccccc" \
    "$TMP/runs/needed/out/${site}_agreement.txt"
  check 'the timed leaves still produce a score comparison' \
    grep -qF 'shipped: joined=2 ' "$TMP/runs/needed/out/${site}_agreement.txt"
fi
if harness_leg '6: stock fits use .tsv.stock and agreement refuses truncated stock snapshots'; then
  site=rank_wgrad_1x32x32x32
  stock_sites='SITES=wgrad:1,32,32,32'
  drive "$SRC" mixed-fits COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-serial-profile PICK_PROFILES=reproducible TXN_RATE=1e12
  check 'mixed stock run with TXN_RATE and profile timing cell succeeds' test "$got" = 0
  mixed_out="$TMP/runs/mixed-fits/out"
  check 'stock agreement reads baseline from .tsv.stock rather than the faster profile row' \
    grep -qxF "  shipped: 'default' measured_ms=0.400000 rank_ms=0.500000 admitted=true" \
      "$mixed_out/${site}_agreement.txt"
  check 'stock agreement reads W_sketch from .tsv.stock rather than the faster profile row' \
    grep -qxF "  shipped: 'W_sketch[control]' measured_ms=1.000000 rank_ms=1.250000 admitted=true" \
      "$mixed_out/${site}_agreement.txt"
  check 'explicit profile pick still joins its profile timing row' \
    grep -qF "reproducible: pick label='seeded' seeded_label='F_seeded[profile]' row='F_seeded[profile]' measured_ms=0.125000" \
      "$mixed_out/${site}_agreement.txt"
  check 'gh1253_cells passes .tsv.stock to gh1600_txn_fit (all 3 stock candidates join)' \
    grep -qF 'wgrad_1x32x32x32: joined=3 spread=1.000 ' "$mixed_out/${site}_txn.txt"
  check 'gh1600_txn_fit driver output keeps stock median for default' \
    grep -qF "  wgrad_1x32x32x32: 'default' median_ms=0.450000 " "$mixed_out/${site}_txn.txt"
  check 'gh1600_txn_fit driver output keeps stock median for W_sketch[control]' \
    grep -qF "  wgrad_1x32x32x32: 'W_sketch[control]' median_ms=1.125000 " "$mixed_out/${site}_txn.txt"

  txn_fit="$tree/benchmarks/gh1600_txn_fit.py"
  geo_fit="$tree/benchmarks/gh1821_geometry_fit.py"
  agreement="$tree/benchmarks/gh1442_agreement.py"

  got=0
  python3 "$txn_fit" --rate 1e12 --show \
    "wgrad_1x32x32x32=$mixed_out/$site.tsv,$mixed_out/${site}_notxn.out" \
    >"$TMP/runs/mixed-fits/txn-mixed.out" 2>"$TMP/runs/mixed-fits/txn-mixed.err" || got=$?
  check 'gh1600_txn_fit refuses a marked mixed TSV directly' test "$got" = 2
  check 'gh1600_txn_fit refusal names stock_snapshot_missing' \
    grep -qF 'stock_snapshot_missing:' "$TMP/runs/mixed-fits/txn-mixed.err"
  check 'gh1600_txn_fit refusal emits no stdout' test ! -s "$TMP/runs/mixed-fits/txn-mixed.out"

  got=0
  python3 "$txn_fit" --rate 1e12 --show \
    "wgrad_1x32x32x32=$mixed_out/$site.tsv.stock,$mixed_out/${site}_notxn.out" \
    >"$TMP/runs/mixed-fits/txn-stock.out" 2>"$TMP/runs/mixed-fits/txn-stock.err" || got=$?
  check 'gh1600_txn_fit succeeds on .tsv.stock' test "$got" = 0
  check 'gh1600_txn_fit on .tsv.stock matches the driver output byte for byte' \
    cmp "$mixed_out/${site}_txn.txt" "$TMP/runs/mixed-fits/txn-stock.out"

  got=0
  python3 "$geo_fit" "$mixed_out" \
    >"$TMP/runs/mixed-fits/geo-stock.out" 2>"$TMP/runs/mixed-fits/geo-stock.err" || got=$?
  check 'gh1821_geometry_fit succeeds on a mixed archive using .tsv.stock' test "$got" = 0
  check 'gh1821_geometry_fit reports stock agreement and spread' \
    grep -qxF "F ${site}: agreement=100.0% spread=1.000 argmin='default'@1.000 admitted_argmin@1.000 pass=true" \
      "$TMP/runs/mixed-fits/geo-stock.out"

  check 'gh1600_txn_fit and gh1821_geometry_fit read the exact stock replicate sets' \
    python3 - "$tree/benchmarks" "$mixed_out" "$site" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import gh1600_txn_fit
import gh1821_geometry_fit

out, site = sys.argv[2], sys.argv[3]
want = {"default": [0.5, 0.4], "seeded": [0.75, 0.75], "W_sketch[control]": [1.25, 1.0]}
_, cands, bad, _ = gh1600_txn_fit.load_cell(f"{site}={out}/{site}.tsv.stock,{out}/{site}_notxn.out")
assert not bad, bad
got_txn = {k: v["times"] for k, v in cands.items()}
assert got_txn == want, f"gh1600 replicates: {got_txn!r} != {want!r}"
cells = gh1821_geometry_fit.load(out, "shipped")
assert len(cells) == 1, cells
got_geo = {r["label"]: r["times"] for r in cells[0][2]}
assert got_geo == want, f"gh1821 replicates: {got_geo!r} != {want!r}"
PY

  geo_partial="$TMP/runs/mixed-fits/geo-partial"
  mkdir -p "$geo_partial"
  cp "$mixed_out/$site.tsv" "$mixed_out/${site}_shipped.out" "$geo_partial/"
  got=0
  python3 "$geo_fit" "$geo_partial" \
    >"$TMP/runs/mixed-fits/geo-partial.out" 2>"$TMP/runs/mixed-fits/geo-partial.err" || got=$?
  check 'gh1821_geometry_fit refuses a partial archive missing .tsv.stock' test "$got" = 2
  check 'gh1821_geometry_fit refusal names stock_snapshot_missing' \
    grep -qF 'stock_snapshot_missing:' "$TMP/runs/mixed-fits/geo-partial.err"
  check 'gh1821_geometry_fit refusal emits no stdout' test ! -s "$TMP/runs/mixed-fits/geo-partial.out"

  trunc_dir="$(cd "$TMP" && pwd)/runs/mixed-fits/trunc-archive"
  mkdir -p "$trunc_dir"
  cp "$mixed_out/$site.tsv" "$mixed_out/${site}_time.err" "$mixed_out/${site}_time_r2.err" "$trunc_dir/"
  grep -vF 'F_seeded[stock]' "$mixed_out/$site.tsv.stock" >"$trunc_dir/$site.tsv.stock"
  got=0
  python3 "$agreement" "$mixed_out/$site.tsv" "shipped=$mixed_out/${site}_shipped.out" \
    --stock-tsv "$trunc_dir/$site.tsv.stock" \
    --census "$mixed_out/${site}_time.err" --census "$mixed_out/${site}_time_r2.err" \
    >"$TMP/runs/mixed-fits/trunc-direct.out" 2>"$TMP/runs/mixed-fits/trunc-direct.err" || got=$?
  check 'direct agreement fails on a truncated stock snapshot missing F_seeded[stock]' test "$got" = 1
  check 'direct agreement diagnoses the missing seeded incumbent' \
    grep -qxF "shipped: seeded_unmatched label='seeded' plan='F_seeded[stock]'" \
      "$TMP/runs/mixed-fits/trunc-direct.out"
  check 'truncated snapshot still keeps remaining candidates on stock timings' \
    grep -qxF "  shipped: 'W_sketch[control]' measured_ms=1.000000 rank_ms=1.250000 admitted=true" \
      "$TMP/runs/mixed-fits/trunc-direct.out"

  drive "$SRC" trunc-replay COMPARE_AUTO_GATE=0 "$stock_sites" TIME_REPEATS=1 \
    EXPLAIN_ONLY="$trunc_dir" STUB_MUTATION=stock-serial-profile
  check 'EXPLAIN_ONLY fails on a truncated stock snapshot missing F_seeded[stock]' test "$got" = 1
  check 'EXPLAIN_ONLY diagnoses the missing seeded incumbent' \
    grep -qxF "shipped: seeded_unmatched label='seeded' plan='F_seeded[stock]'" \
      "$TMP/runs/trunc-replay/out/${site}_agreement.txt"

  : >"$TMP/runs/mixed-fits/empty.tsv.stock"
  got=0
  python3 "$agreement" "$mixed_out/$site.tsv" "shipped=$mixed_out/${site}_shipped.out" \
    --stock-tsv "$TMP/runs/mixed-fits/empty.tsv.stock" \
    --census "$mixed_out/${site}_time.err" --census "$mixed_out/${site}_time_r2.err" \
    >"$TMP/runs/mixed-fits/empty-stock.out" 2>"$TMP/runs/mixed-fits/empty-stock.err" || got=$?
  check 'direct agreement refuses an empty stock snapshot' test "$got" = 2
  check 'empty stock snapshot names no calibration rows on stderr' \
    grep -qF 'no calibration rows in ' "$TMP/runs/mixed-fits/empty-stock.err"
  check 'empty stock snapshot emits no stdout' test ! -s "$TMP/runs/mixed-fits/empty-stock.out"
fi
finish
