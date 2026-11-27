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
    'STAGED_COPY_RATE=5e11|STAGED_COPY_RATE conflicts with this mode' \
    'STAGED_IDLE_RATE=5e11|STAGED_IDLE_RATE conflicts with this mode' \
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
  serial_arms=(ceilings nofill nomma mmaonly noocc notxn nostg noidl noilp noshl nolane shipped)
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
    grep -qxF "shipped: seeded_plan_unmatched label='seeded' plan='F_seeded[stock]'" \
      "$TMP/runs/mixed-fits/trunc-direct.out"
  check 'truncated snapshot still keeps remaining candidates on stock timings' \
    grep -qxF "  shipped: 'W_sketch[control]' measured_ms=1.000000 rank_ms=1.250000 admitted=true" \
      "$TMP/runs/mixed-fits/trunc-direct.out"

  drive "$SRC" trunc-replay COMPARE_AUTO_GATE=0 "$stock_sites" TIME_REPEATS=1 \
    EXPLAIN_ONLY="$trunc_dir" STUB_MUTATION=stock-serial-profile
  check 'EXPLAIN_ONLY fails on a truncated stock snapshot missing F_seeded[stock]' test "$got" = 1
  check 'EXPLAIN_ONLY diagnoses the missing seeded incumbent' \
    grep -qxF "shipped: seeded_plan_unmatched label='seeded' plan='F_seeded[stock]'" \
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
if harness_leg '7: stock fits attribute F_preset and F_seeded incumbents via timing censuses'; then
  site=rank_wgrad_1x32x32x32
  stock_sites='SITES=wgrad:1,32,32,32'
  txn_fit="$tree/benchmarks/gh1600_txn_fit.py"
  geo_fit="$tree/benchmarks/gh1821_geometry_fit.py"

  drive "$SRC" preset-fits COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" TIME_REPEATS=3 \
    STUB_MUTATION=stock-seeded-preset PICK_PROFILES=reproducible TXN_RATE=1e12
  check 'stock-seeded-preset run with TXN_RATE and profile timing cell succeeds' test "$got" = 0
  pf_out="$TMP/runs/preset-fits/out"
  check 'gh1253_cells joins seeded preset incumbent in gh1600_txn_fit with preset spread' \
    grep -qF "wgrad_1x32x32x32: joined=2 spread=1.667 uncovered_accesses=0 off_agreement=100.0% decided=1 off_admitted='seeded'@1.000 off_admitted+coalesced='seeded'@1.000" \
      "$pf_out/${site}_txn.txt"
  check 'gh1600_txn_fit driver output attributes seeded median and timed_as to preset digest' \
    grep -qxF "  wgrad_1x32x32x32: 'seeded' median_ms=0.500000 rank_ms=0.500000 admitted=true uncovered=0 timed_as='W_preset[bs=256]'" \
      "$pf_out/${site}_txn.txt"

  got=0
  python3 "$txn_fit" --rate 1e12 --show \
    --census "$pf_out/${site}_time.err" --census "$pf_out/${site}_time_r2.err" \
    --census "$pf_out/${site}_time_r3.err" \
    "wgrad_1x32x32x32=$pf_out/$site.tsv.stock,$pf_out/${site}_notxn.out" \
    >"$TMP/runs/preset-fits/txn-bare.out" 2>"$TMP/runs/preset-fits/txn-bare.err" || got=$?
  check 'direct gh1600_txn_fit with bare --census succeeds on single cell' test "$got" = 0
  check 'direct gh1600_txn_fit matches driver output byte for byte' \
    cmp "$pf_out/${site}_txn.txt" "$TMP/runs/preset-fits/txn-bare.out"

  eq_dir="$TMP/runs/preset-fits/device=hip"
  mkdir -p "$eq_dir"
  cp "$pf_out/${site}_time.err" "$eq_dir/"
  got=0
  python3 "$txn_fit" --rate 1e12 --show \
    --census "$eq_dir/${site}_time.err" \
    "wgrad_1x32x32x32=$pf_out/$site.tsv.stock,$pf_out/${site}_notxn.out" \
    >"$TMP/runs/preset-fits/txn-eqpath.out" 2>"$TMP/runs/preset-fits/txn-eqpath.err" || got=$?
  check 'single-cell bare --census accepts a path containing =' test "$got" = 0
  check 'single-cell bare --census with = matches driver output byte for byte' \
    cmp "$pf_out/${site}_txn.txt" "$TMP/runs/preset-fits/txn-eqpath.out"

  got=0
  python3 "$txn_fit" --rate 1e12 --census "$eq_dir/missing.err" \
    "wgrad_1x32x32x32=$pf_out/$site.tsv.stock,$pf_out/${site}_notxn.out" \
    >"$TMP/runs/preset-fits/txn-eqmissing.out" 2>"$TMP/runs/preset-fits/txn-eqmissing.err" || got=$?
  check 'missing single-cell bare --census path with = fails with OSError' test "$got" = 2
  check 'missing single-cell bare --census path with = names the missing file' \
    grep -qF 'No such file or directory' "$TMP/runs/preset-fits/txn-eqmissing.err"
  check 'missing single-cell bare --census path emits no stdout' \
    test ! -s "$TMP/runs/preset-fits/txn-eqmissing.out"

  got=0
  python3 "$txn_fit" --staged-rate 1e12 --listed-staged-rate 1e12 \
    --census "wgrad_1x32x32x32=$pf_out/${site}_time.err" \
    --census "wgrad_1x32x32x32=$pf_out/${site}_time_r2.err" \
    --census "wgrad_1x32x32x32=$pf_out/${site}_time_r3.err" \
    "wgrad_1x32x32x32=$pf_out/$site.tsv.stock,$pf_out/${site}_notxn.out,$pf_out/${site}_nostg.out" \
    >"$TMP/runs/preset-fits/txn-staged.out" 2>"$TMP/runs/preset-fits/txn-staged.err" || got=$?
  check 'gh1600_txn_fit --staged-rate threads --census to the nostg reference too' test "$got" = 0
  check 'gh1600_txn_fit --staged-rate joins the preset incumbent' \
    grep -qF "wgrad_1x32x32x32: joined=2 spread=1.667 uncovered_accesses=0 off_agreement=100.0% decided=1 off_admitted='seeded'@1.000 off_admitted+coalesced='seeded'@1.000" \
      "$TMP/runs/preset-fits/txn-staged.out"

  # Plant a conflicting plan in the profile timing log to verify gh1821_geometry_fit ignores it.
  printf 'autotune-seeding: plan F_seeded[profile]\n' >>"$pf_out/${site}_time_reproducible.err"
  got=0
  python3 "$geo_fit" "$pf_out" \
    >"$TMP/runs/preset-fits/geo.out" 2>"$TMP/runs/preset-fits/geo.err" || got=$?
  check 'gh1821_geometry_fit succeeds on stock-seeded-preset and ignores profile stderr' test "$got" = 0
  check 'gh1821_geometry_fit joins seeded preset incumbent with preset spread' \
    grep -qxF "F ${site}: agreement=100.0% spread=1.667 argmin='seeded'@1.000 admitted_argmin@1.000 pass=true" \
      "$TMP/runs/preset-fits/geo.out"

  check 'gh1600_txn_fit and gh1821_geometry_fit keep only the attributed digest replicates' \
    python3 - "$tree/benchmarks" "$pf_out" "$site" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import gh1600_txn_fit
import gh1821_geometry_fit

out, site = sys.argv[2], sys.argv[3]
want = {"seeded": [0.5, 0.25, 0.5], "W_sketch[control]": [1.25, 1.25, 1.25]}
census = [f"{out}/{site}_time.err", f"{out}/{site}_time_r2.err", f"{out}/{site}_time_r3.err"]
_, cands, bad, _ = gh1600_txn_fit.load_cell(
    f"{site}={out}/{site}.tsv.stock,{out}/{site}_notxn.out", census=census)
assert not bad, bad
got_txn = {k: v["times"] for k, v in cands.items()}
assert got_txn == want, f"gh1600 replicates: {got_txn!r} != {want!r}"
assert cands["seeded"]["timed_as"] == "W_preset[bs=256]", cands["seeded"]
cells = gh1821_geometry_fit.load(out, "shipped")
assert len(cells) == 1, cells
got_geo = {r["label"]: r["times"] for r in cells[0][2]}
assert got_geo == want, f"gh1821 replicates: {got_geo!r} != {want!r}"
PY

  drive "$SRC" preset-default COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-preset TXN_RATE=1e12
  check 'stock-preset default incumbent run with TXN_RATE succeeds' test "$got" = 0
  pd_out="$TMP/runs/preset-default/out"
  check 'gh1600_txn_fit attributes default incumbent by preset-digest (median 0.375)' \
    grep -qxF "  wgrad_1x32x32x32: 'default' median_ms=0.375000 rank_ms=0.500000 admitted=true uncovered=0 timed_as='W_preset[bs=256]'" \
      "$pd_out/${site}_txn.txt"
  got=0
  python3 "$geo_fit" "$pd_out" \
    >"$TMP/runs/preset-default/geo.out" 2>"$TMP/runs/preset-default/geo.err" || got=$?
  check 'gh1821_geometry_fit joins default preset incumbent' test "$got" = 0
  check 'gh1821_geometry_fit reports default preset argmin' \
    grep -qxF "F ${site}: agreement=100.0% spread=1.000 argmin='default'@1.000 admitted_argmin@1.000 pass=true" \
      "$TMP/runs/preset-default/geo.out"

  drive "$SRC" preset-multi-serial COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-serial-ambiguous
  check 'serial-ambiguous fixture succeeds for multi-cell census check' test "$got" = 0
  sa_out="$TMP/runs/preset-multi-serial/out"
  got=0
  python3 "$txn_fit" --rate 1e12 --show \
    --census "preset=$pf_out/${site}_time.err" \
    --census "preset=$pf_out/${site}_time_r2.err" \
    --census "preset=$pf_out/${site}_time_r3.err" \
    --census "serial=$sa_out/${site}_time.err" \
    --census "serial=$sa_out/${site}_time_r2.err" \
    "preset=$pf_out/$site.tsv.stock,$pf_out/${site}_notxn.out" \
    "serial=$sa_out/$site.tsv,$sa_out/${site}_notxn.out" \
    >"$TMP/runs/preset-fits/multi.out" 2>"$TMP/runs/preset-fits/multi.err" || got=$?
  check 'multi-cell gh1600_txn_fit binds per-cell censuses' test "$got" = 0
  check 'multi-cell gh1600_txn_fit joins preset incumbent in preset cell' \
    grep -qF "preset: joined=2 spread=1.667 " "$TMP/runs/preset-fits/multi.out"
  check 'multi-cell gh1600_txn_fit disambiguates F_seeded[stock] in serial cell' \
    grep -qF "serial: joined=3 spread=1.000 " "$TMP/runs/preset-fits/multi.out"

  got=0
  python3 "$txn_fit" --rate 1e12 --census "$pf_out/${site}_time.err" \
    "preset=$pf_out/$site.tsv.stock,$pf_out/${site}_notxn.out" \
    "serial=$sa_out/$site.tsv,$sa_out/${site}_notxn.out" \
    >"$TMP/runs/preset-fits/multi-bare.out" 2>"$TMP/runs/preset-fits/multi-bare.err" || got=$?
  check 'multi-cell gh1600_txn_fit refuses bare --census' test "$got" = 2
  check 'bare multi-cell refusal names CELL=TIMING_STDERR' \
    grep -qF 'CELL=TIMING_STDERR' "$TMP/runs/preset-fits/multi-bare.err"
  check 'bare multi-cell refusal emits no stdout' test ! -s "$TMP/runs/preset-fits/multi-bare.out"

  got=0
  python3 "$txn_fit" --rate 1e12 --census "bogus=$pf_out/${site}_time.err" \
    "preset=$pf_out/$site.tsv.stock,$pf_out/${site}_notxn.out" \
    "serial=$sa_out/$site.tsv,$sa_out/${site}_notxn.out" \
    >"$TMP/runs/preset-fits/unknown-cell.out" 2>"$TMP/runs/preset-fits/unknown-cell.err" || got=$?
  check 'multi-cell gh1600_txn_fit refuses --census naming an unknown cell' test "$got" = 2
  check 'unknown-cell refusal names the unknown cell' \
    grep -qF "unknown cell 'bogus'" "$TMP/runs/preset-fits/unknown-cell.err"
  check 'unknown-cell refusal emits no stdout' test ! -s "$TMP/runs/preset-fits/unknown-cell.out"

  drive "$SRC" preset-unused-fits COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-preset-unused TXN_RATE=1e12
  check 'untimed unused preset still succeeds in gh1600_txn_fit via driver' test "$got" = 0
  pu_out="$TMP/runs/preset-unused-fits/out"
  check 'unused preset joins F_seeded[stock] without timed_as' \
    grep -qxF "  wgrad_1x32x32x32: 'seeded' median_ms=0.500000 rank_ms=0.500000 admitted=true uncovered=0" \
      "$pu_out/${site}_txn.txt"
  got=0
  python3 "$geo_fit" "$pu_out" \
    >"$TMP/runs/preset-unused-fits/geo.out" 2>"$TMP/runs/preset-unused-fits/geo.err" || got=$?
  check 'gh1821_geometry_fit succeeds when preset is unused and F_seeded[stock] is timed' test "$got" = 0

  drive "$SRC" preset-needed-fits COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-preset-needed
  pn_out="$TMP/runs/preset-needed-fits/out"
  got=0
  python3 "$txn_fit" --rate 1e12 \
    --census "$pn_out/${site}_time.err" --census "$pn_out/${site}_time_r2.err" \
    "wgrad_1x32x32x32=$pn_out/$site.tsv,$pn_out/${site}_notxn.out" \
    >"$TMP/runs/preset-needed-fits/txn.out" 2>"$TMP/runs/preset-needed-fits/txn.err" || got=$?
  check 'gh1600_txn_fit refuses a missing attributed preset' test "$got" = 2
  check 'gh1600_txn_fit names preset_unmatched and digest' \
    grep -qF "preset_unmatched label='default' digest=33333333/cccccccc" \
      "$TMP/runs/preset-needed-fits/txn.err"
  check 'gh1600_txn_fit preset_unmatched emits no stdout' test ! -s "$TMP/runs/preset-needed-fits/txn.out"

  got=0
  python3 "$geo_fit" "$pn_out" \
    >"$TMP/runs/preset-needed-fits/geo.out" 2>"$TMP/runs/preset-needed-fits/geo.err" || got=$?
  check 'gh1821_geometry_fit refuses a missing attributed preset' test "$got" = 2
  check 'gh1821_geometry_fit names preset_unmatched and digest' \
    grep -qF "preset_unmatched label='default' digest=33333333/cccccccc" \
      "$TMP/runs/preset-needed-fits/geo.err"
  check 'gh1821_geometry_fit preset_unmatched emits no stdout' test ! -s "$TMP/runs/preset-needed-fits/geo.out"

  # Legacy (<12-column) rows cannot satisfy a preset-digest join.
  cut -f1-5 "$pd_out/$site.tsv" >"$TMP/runs/preset-default/legacy.tsv"
  got=0
  python3 "$txn_fit" --rate 1e12 \
    --census "$pd_out/${site}_time.err" --census "$pd_out/${site}_time_r2.err" \
    "wgrad_1x32x32x32=$TMP/runs/preset-default/legacy.tsv,$pd_out/${site}_notxn.out" \
    >"$TMP/runs/preset-default/legacy.out" 2>"$TMP/runs/preset-default/legacy.err" || got=$?
  check 'gh1600_txn_fit refuses a legacy 5-column row for preset-digest join' test "$got" = 2
  check 'legacy-row refusal names preset_unmatched' \
    grep -qF "preset_unmatched label='default' digest=33333333/cccccccc" \
      "$TMP/runs/preset-default/legacy.err"
  check 'legacy-row refusal emits no stdout' test ! -s "$TMP/runs/preset-default/legacy.out"

  # Truncated F_seeded[stock] with census fails closed in both fit scripts.
  sa_trunc="$TMP/runs/preset-multi-serial/trunc"
  mkdir -p "$sa_trunc"
  cp "$sa_out/${site}_shipped.out" "$sa_out/${site}_notxn.out" \
    "$sa_out/${site}_time.err" "$sa_out/${site}_time_r2.err" "$sa_trunc/"
  grep -vF 'F_seeded[stock]' "$sa_out/$site.tsv" >"$sa_trunc/$site.tsv"
  got=0
  python3 "$txn_fit" --rate 1e12 \
    --census "$sa_trunc/${site}_time.err" --census "$sa_trunc/${site}_time_r2.err" \
    "wgrad_1x32x32x32=$sa_trunc/$site.tsv,$sa_trunc/${site}_notxn.out" \
    >"$TMP/runs/preset-multi-serial/trunc-txn.out" 2>"$TMP/runs/preset-multi-serial/trunc-txn.err" || got=$?
  check 'gh1600_txn_fit refuses a missing census-selected F_seeded[stock]' test "$got" = 2
  check 'gh1600_txn_fit names seeded_plan_unmatched' \
    grep -qF "seeded_plan_unmatched label='seeded' plan='F_seeded[stock]'" \
      "$TMP/runs/preset-multi-serial/trunc-txn.err"
  check 'gh1600_txn_fit seeded_plan_unmatched emits no stdout' \
    test ! -s "$TMP/runs/preset-multi-serial/trunc-txn.out"

  got=0
  python3 "$geo_fit" "$sa_trunc" \
    >"$TMP/runs/preset-multi-serial/trunc-geo.out" 2>"$TMP/runs/preset-multi-serial/trunc-geo.err" || got=$?
  check 'gh1821_geometry_fit refuses a missing census-selected F_seeded[stock]' test "$got" = 2
  check 'gh1821_geometry_fit names seeded_plan_unmatched' \
    grep -qF "seeded_plan_unmatched label='seeded' plan='F_seeded[stock]'" \
      "$TMP/runs/preset-multi-serial/trunc-geo.err"
  check 'gh1821_geometry_fit seeded_plan_unmatched emits no stdout' \
    test ! -s "$TMP/runs/preset-multi-serial/trunc-geo.out"

  drive "$SRC" preset-conflict-fits COMPARE_AUTO_GATE=0 RANK_ONLY=1 "$stock_sites" \
    STUB_MUTATION=stock-preset-conflict BENCH_PICK=1
  pc_out="$TMP/runs/preset-conflict-fits/out"
  got=0
  python3 "$txn_fit" --rate 1e12 \
    --census "$pc_out/${site}_time.err" --census "$pc_out/${site}_time_r2.err" \
    "wgrad_1x32x32x32=$pc_out/$site.tsv,$pc_out/${site}_notxn.out" \
    >"$TMP/runs/preset-conflict-fits/txn.out" 2>"$TMP/runs/preset-conflict-fits/txn.err" || got=$?
  check 'gh1600_txn_fit refuses conflicting plans in a census' test "$got" = 2
  check 'gh1600_txn_fit diagnoses conflicting plans' \
    grep -qF 'conflicting autotune-seeding: plan values' "$TMP/runs/preset-conflict-fits/txn.err"
  check 'gh1600_txn_fit conflict emits no stdout' test ! -s "$TMP/runs/preset-conflict-fits/txn.out"

  got=0
  python3 "$geo_fit" "$pc_out" \
    >"$TMP/runs/preset-conflict-fits/geo.out" 2>"$TMP/runs/preset-conflict-fits/geo.err" || got=$?
  check 'gh1821_geometry_fit refuses conflicting plans in a census' test "$got" = 2
  check 'gh1821_geometry_fit diagnoses conflicting plans' \
    grep -qF 'conflicting autotune-seeding: plan values' "$TMP/runs/preset-conflict-fits/geo.err"
  check 'gh1821_geometry_fit conflict emits no stdout' test ! -s "$TMP/runs/preset-conflict-fits/geo.out"
fi
if harness_leg '8: gh1821_geometry_fit refuses <2 candidates and handles empty pick sets'; then
  site=rank_wgrad_1x32x32x32
  geo_fit="$tree/benchmarks/gh1821_geometry_fit.py"
  geo_base="$TMP/runs/geo-guards"

  write_geo_cell() { # DIR ROW_SPECS EXPLAIN_SPECS
    local dir=$1 rows=$2 explains=$3
    mkdir -p "$dir"
    : >"$dir/$site.tsv"
    : >"$dir/${site}_shipped.out"
    while IFS='|' read -r label digest ms; do
      [ -n "$label" ] || continue
      printf 'cuda\t%s\trs_stub\t%s\t%s\t\t1\t4\t8\tfalse\tfalse\tfalse\n' \
        "$digest" "$label" "$ms" >>"$dir/$site.tsv"
    done <<<"$rows"
    while IFS='|' read -r label rank_ms admitted; do
      [ -n "$label" ] || continue
      printf 'rank-site: explain label="%s" rank_ms=%s admitted=%s\n' \
        "$label" "$rank_ms" "$admitted" >>"$dir/${site}_shipped.out"
      printf 'rank-site: kernels label="%s" kernels="c=%s,m=0.0,txn=0,txn64=0,txn128=0,unc=0,wg=0,thr=32"\n' \
        "$label" "$rank_ms" >>"$dir/${site}_shipped.out"
    done <<<"$explains"
  }

  # 1-candidate cell (2 listed, only 1 joins): must refuse with exit 2 and no stdout.
  one_dir="$geo_base/one-cand"
  write_geo_cell "$one_dir" \
    $'W_sketch[control]|11111111/aaaaaaaa|1.25' \
    $'default|0.5|true\nW_sketch[control]|1.25|true'
  got=0
  python3 "$geo_fit" "$one_dir" >"$geo_base/one-cand.out" 2>"$geo_base/one-cand.err" || got=$?
  check 'gh1821_geometry_fit refuses a 1-candidate cell with exit 2' test "$got" = 2
  check '1-candidate refusal names fewer than two candidates join' \
    grep -qF "${site}: fewer than two candidates join" "$geo_base/one-cand.err"
  check '1-candidate refusal emits no stdout' test ! -s "$geo_base/one-cand.out"

  # Repeated explain label for a single timed candidate still counts as 1 candidate.
  dup_dir="$geo_base/dup-cand"
  write_geo_cell "$dup_dir" \
    $'W_sketch[control]|11111111/aaaaaaaa|1.25' \
    $'W_sketch[control]|1.25|true\nW_sketch[control]|1.25|true'
  got=0
  python3 "$geo_fit" "$dup_dir" >"$geo_base/dup-cand.out" 2>"$geo_base/dup-cand.err" || got=$?
  check 'gh1821_geometry_fit refuses a repeated single explain label with exit 2' test "$got" = 2
  check 'repeated-label refusal names fewer than two candidates join' \
    grep -qF "${site}: fewer than two candidates join" "$geo_base/dup-cand.err"
  check 'repeated-label refusal emits no stdout' test ! -s "$geo_base/dup-cand.out"

  # 0-candidate cell (calibration rows present, none join): same refusal.
  zero_dir="$geo_base/zero-cand"
  write_geo_cell "$zero_dir" \
    $'W_sketch[other]|11111111/aaaaaaaa|1.25' \
    $'W_sketch[control]|1.25|true'
  got=0
  python3 "$geo_fit" "$zero_dir" >"$geo_base/zero-cand.out" 2>"$geo_base/zero-cand.err" || got=$?
  check 'gh1821_geometry_fit refuses a 0-candidate cell with exit 2' test "$got" = 2
  check '0-candidate refusal names fewer than two candidates join' \
    grep -qF "${site}: fewer than two candidates join" "$geo_base/zero-cand.err"
  check '0-candidate refusal emits no stdout' test ! -s "$geo_base/zero-cand.out"

  # All-coalesced cell: admitted set is empty, admitted+coalesced still judges slowdown vs spread.
  coal_dir="$geo_base/all-coal"
  write_geo_cell "$coal_dir" \
    $'W_sketch[gpu 64x64x16/4x4 coal-k]|11111111/aaaaaaaa|0.50\nW_sketch[gpu 64x64x16/4x4 coal-k]|11111111/aaaaaaaa|0.25\nW_sketch[gpu 64x64x16/4x4 coal-k]|11111111/aaaaaaaa|0.50\nW_sketch[gpu 32x32x16/4x4 coal-n]|22222222/bbbbbbbb|1.25\nW_sketch[gpu 32x32x16/4x4 coal-n]|22222222/bbbbbbbb|1.25\nW_sketch[gpu 32x32x16/4x4 coal-n]|22222222/bbbbbbbb|1.25' \
    $'W_sketch[gpu 64x64x16/4x4 coal-k]|0.5|true\nW_sketch[gpu 32x32x16/4x4 coal-n]|1.25|true'
  got=0
  python3 "$geo_fit" "$coal_dir" --copy-rate 1e12 \
    >"$geo_base/all-coal.out" 2>"$geo_base/all-coal.err" || got=$?
  check 'gh1821_geometry_fit handles an all-coalesced cell without crashing' test "$got" = 0
  check 'all-coalesced cell reports admitted+coalesced argmin with admitted_argmin=none' \
    grep -qxF "F ${site}: agreement=100.0% spread=1.667 argmin='W_sketch[gpu 64x64x16/4x4 coal-k]'@1.000 admitted_argmin=none pass=true" \
      "$geo_base/all-coal.out"
  check 'all-coalesced cell rescoring with --copy-rate passes' \
    grep -qxF 'cells/s=1.000e+12 (bytes/s=4.000e+12): passing=1/1' "$geo_base/all-coal.out"
  check 'all-coalesced cell emits no stderr' test ! -s "$geo_base/all-coal.err"

  # Inverted all-coalesced cell: coalesced argmin slower than spread fails pass=false.
  coal_bad_dir="$geo_base/all-coal-bad"
  write_geo_cell "$coal_bad_dir" \
    $'W_sketch[gpu 64x64x16/4x4 coal-k]|11111111/aaaaaaaa|1.25\nW_sketch[gpu 32x32x16/4x4 coal-n]|22222222/bbbbbbbb|0.50' \
    $'W_sketch[gpu 64x64x16/4x4 coal-k]|0.5|true\nW_sketch[gpu 32x32x16/4x4 coal-n]|1.25|true'
  got=0
  python3 "$geo_fit" "$coal_bad_dir" \
    >"$geo_base/all-coal-bad.out" 2>"$geo_base/all-coal-bad.err" || got=$?
  check 'inverted all-coalesced cell exits 0 without crashing' test "$got" = 0
  check 'inverted all-coalesced cell fails the coalesced slowdown criterion' \
    grep -qxF "F ${site}: agreement=0.0% spread=1.000 argmin='W_sketch[gpu 64x64x16/4x4 coal-k]'@2.500 admitted_argmin=none pass=false" \
      "$geo_base/all-coal-bad.out"

  # All-unadmitted cell: both pick sets empty, spread defaults to 1.0 like gh1600_txn_fit.spread_of.
  unadm_dir="$geo_base/all-unadm"
  write_geo_cell "$unadm_dir" \
    $'W_sketch[gpu 64x64x16/4x4]|11111111/aaaaaaaa|0.50\nW_sketch[gpu 64x64x16/4x4]|11111111/aaaaaaaa|0.25\nW_sketch[gpu 64x64x16/4x4]|11111111/aaaaaaaa|0.50\nW_sketch[gpu 32x32x16/4x4]|22222222/bbbbbbbb|1.25' \
    $'W_sketch[gpu 64x64x16/4x4]|0.5|false\nW_sketch[gpu 32x32x16/4x4]|1.25|false'
  got=0
  python3 "$geo_fit" "$unadm_dir" --copy-rate 1e12 \
    >"$geo_base/all-unadm.out" 2>"$geo_base/all-unadm.err" || got=$?
  check 'gh1821_geometry_fit handles an all-unadmitted cell without crashing' test "$got" = 0
  check 'all-unadmitted cell reports empty pick sets and spread=1.000' \
    grep -qxF "F ${site}: agreement=100.0% spread=1.000 argmin=none admitted_argmin=none pass=true" \
      "$geo_base/all-unadm.out"
  check 'all-unadmitted cell rescoring with --copy-rate passes' \
    grep -qxF 'cells/s=1.000e+12 (bytes/s=4.000e+12): passing=1/1' "$geo_base/all-unadm.out"
  check 'all-unadmitted cell emits no stderr' test ! -s "$geo_base/all-unadm.err"

  check 'gh1821_geometry_fit.evaluate omits empty pick sets like gh1600_txn_fit.evaluate' \
    python3 - "$tree/benchmarks" "$coal_dir" "$unadm_dir" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import gh1821_geometry_fit

coal_res = gh1821_geometry_fit.evaluate(gh1821_geometry_fit.load(sys.argv[2], "shipped"), None, 0, 0.10)
assert len(coal_res) == 1 and sorted(coal_res[0][2]) == ["admitted+coalesced"], coal_res
unadm_res = gh1821_geometry_fit.evaluate(gh1821_geometry_fit.load(sys.argv[3], "shipped"), None, 0, 0.10)
assert len(unadm_res) == 1 and unadm_res[0][2] == {} and unadm_res[0][3] == 1.0, unadm_res
PY
fi
if harness_leg '9: gh1600_txn_fit --idle-rate refuses listings without idl= and rescores ones with it'; then
  site=rank_wgrad_1x32x32x32
  txn_fit="$tree/benchmarks/gh1600_txn_fit.py"
  idl_dir="$TMP/runs/idle-fit"
  mkdir -p "$idl_dir"
  cat >"$idl_dir/rows.tsv" <<'TSV'
hip	11111111/aaaaaaaa	rs_stub	W_sketch[guarded]	2.000		1	4	8	false	false	false
hip	22222222/bbbbbbbb	rs_stub	W_sketch[unguarded]	1.000		1	4	8	false	false	false
TSV
  cat >"$idl_dir/no_idl.out" <<'OUT'
rank-site: explain site=wgrad label="W_sketch[guarded]" rank_ms=0.900000 admitted=true
rank-site: kernels site=wgrad label="W_sketch[guarded]" kernels=c=0.900000,m=0.000000,txn=0.000000e+00,txn64=0.000000e+00,txn128=0.000000e+00,unc=0,wg=4096,thr=256,occ=1.000000,stg=1.000000e+08,shl=0.000000e+00
rank-site: explain site=wgrad label="W_sketch[unguarded]" rank_ms=1.000000 admitted=true
rank-site: kernels site=wgrad label="W_sketch[unguarded]" kernels=c=1.000000,m=0.000000,txn=0.000000e+00,txn64=0.000000e+00,txn128=0.000000e+00,unc=0,wg=4096,thr=256,occ=1.000000,stg=2.000000e+08,shl=0.000000e+00
OUT
  cat >"$idl_dir/with_idl.out" <<'OUT'
rank-site: explain site=wgrad label="W_sketch[guarded]" rank_ms=0.900000 admitted=true
rank-site: kernels site=wgrad label="W_sketch[guarded]" kernels=c=0.900000,m=0.000000,txn=0.000000e+00,txn64=0.000000e+00,txn128=0.000000e+00,unc=0,wg=4096,thr=256,occ=1.000000,stg=1.000000e+08,shl=0.000000e+00,idl=1.000000e+08
rank-site: explain site=wgrad label="W_sketch[unguarded]" rank_ms=1.000000 admitted=true
rank-site: kernels site=wgrad label="W_sketch[unguarded]" kernels=c=1.000000,m=0.000000,txn=0.000000e+00,txn64=0.000000e+00,txn128=0.000000e+00,unc=0,wg=4096,thr=256,occ=1.000000,stg=2.000000e+08,shl=0.000000e+00,idl=0.000000e+00
OUT
  got=0
  python3 "$txn_fit" --idle-rate 5e11 "w=$idl_dir/rows.tsv,$idl_dir/no_idl.out" \
    >"$idl_dir/no_idl.stdout" 2>"$idl_dir/no_idl.stderr" || got=$?
  check 'gh1600_txn_fit --idle-rate refuses listings without idl=' test "$got" = 2
  check 'missing idl= refusal names idl=' grep -qF 'needs kernel lines carrying idl=' "$idl_dir/no_idl.stderr"
  check 'missing idl= refusal emits no stdout' test ! -s "$idl_dir/no_idl.stdout"
  got=0
  python3 "$txn_fit" --idle-rate 5e11 "w=$idl_dir/rows.tsv,$idl_dir/with_idl.out" \
    >"$idl_dir/with_idl.stdout" 2>"$idl_dir/with_idl.stderr" || got=$?
  check 'gh1600_txn_fit --idle-rate succeeds and flips order on listings with idl=' test "$got" = 0
  check 'gh1600_txn_fit --idle-rate picks the unguarded leaf at 100% agreement' \
    grep -qF "w=100.0%@1.000:'W_sketch[unguarded]'" "$idl_dir/with_idl.stdout"
  for bad_spec in \
    '--idle-rate=-1' '--idle-rate=0' '--idle-rate=nan' '--idle-rate=inf' \
    '--listed-idle-rate=-1' '--listed-idle-rate=sweep' '--listed-idle-rate=nan' \
    '--listed-idle-rate=inf'; do
    flag="${bad_spec%%=*}"
    val="${bad_spec#*=}"
    tag="$(printf '%s_%s' "${flag#--}" "$val" | tr -c 'A-Za-z0-9_' '_')"
    got=0
    python3 "$txn_fit" "$flag" "$val" "w=$idl_dir/nonexistent.tsv,$idl_dir/nonexistent.out" \
      >"$idl_dir/bad_${tag}.out" 2>"$idl_dir/bad_${tag}.err" || got=$?
    check "gh1600_txn_fit refuses $flag $val before loading listings" test "$got" = 2
    check "refusal of $flag $val names $flag" grep -qF -- "$flag:" "$idl_dir/bad_${tag}.err"
    check "refusal of $flag $val emits no stdout" test ! -s "$idl_dir/bad_${tag}.out"
  done
  got=0
  python3 "$txn_fit" --idle-rate none --listed-idle-rate none "w=$idl_dir/rows.tsv,$idl_dir/with_idl.out" \
    >"$idl_dir/none_idl.out" 2>"$idl_dir/none_idl.err" || got=$?
  check 'gh1600_txn_fit accepts none for --idle-rate and --listed-idle-rate' test "$got" = 0
fi
finish
