#!/usr/bin/env python3
"""Join one gh1253_cells.sh gate pair to its calibration rows.

Reads only the pair's successful cells.py cell.json records, autotune-seeding:
stderr lines, current 12-column calibration TSVs, and (for timing-admission
diagnostics on missing rows) the arm's `rank-site: tune` stdout line via
bench_output.rank_site_lines. A join needs one ranked gate, one exact plan
label, segment census evidence, and one positive timing row in the same arm.
Seeded plans join by label. The configured preset joins by the tuner's resolved
digest, which can also identify a deduplicated row. The ranked gate tags must
match after removing auto's optional copy_rate.

Timing-admission reader boundary (closed allowlist): reads only the
`timing_decisions` field of `rank-site: tune` lines (`none` or comma-separated
`<settle>:<admission>=<positive int>` where `<settle>` is in KNOWN_SETTLES and
`<admission>` is in KNOWN_ADMISSIONS). Malformed `rank-site:` lines, multiple
`rank-site: tune` lines, or unrecognized settle/admission/count tokens fail
closed with ValueError. Other stdout/stderr lines, old TSV schemas, and rows
from other cells are outside the join and ignored.
"""

import math
import sys
from pathlib import Path

import bench_output
import config_records
import strict_json


GATE_KEY = "gpu_default_sketch_min_reduction"
TSV_KEY = "autotune_calibration_file"
CENSUS_PREFIX = "autotune-seeding: "
CALIBRATION_COLUMNS = 12

KNOWN_SETTLES = frozenset({
    "isolated",
    "single_estimate",
    "singles_reach_target",
    "singles_far_over_target",
    "prior_depth",
    "measured_batch",
    "affine_projection",
    "wall_scaled",
    "cap_measured",
    "cap_unresolved",
    "budget_exhausted",
})
ADMISSION_ORDER = (
    "refused_unbatched",
    "refused_contended",
    "refused_unreadable",
    "admitted",
)
KNOWN_ADMISSIONS = frozenset(ADMISSION_ORDER)


class MissingTimingRowError(ValueError):
    """Raised when the calibration TSV holds zero rows for the arm's selected plan."""


def parse_timing_decisions(raw):
    """Parse a `rank-site: tune` timing_decisions value under the closed allowlist."""
    if raw == "none":
        return []
    if not raw:
        raise ValueError("empty timing_decisions field")
    entries = []
    seen = set()
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"malformed timing_decisions entry {item!r}")
        lhs, count_str = item.split("=", 1)
        if ":" not in lhs:
            raise ValueError(f"malformed timing_decisions key {lhs!r}")
        settle, admission = lhs.split(":", 1)
        if settle not in KNOWN_SETTLES:
            raise ValueError(f"unrecognized timing_decisions settle {settle!r}")
        if admission not in KNOWN_ADMISSIONS:
            raise ValueError(f"unrecognized timing_decisions admission {admission!r}")
        if not count_str.isdigit() or int(count_str) <= 0:
            raise ValueError(f"timing_decisions count must be a positive integer, got {count_str!r}")
        if (settle, admission) in seen:
            raise ValueError(f"duplicate timing_decisions entry {lhs!r}")
        seen.add((settle, admission))
        entries.append((settle, admission, int(count_str)))
    return entries


def format_timing_admissions(entries):
    """Format parsed timing_decisions as admission totals plus the settle:admission breakdown."""
    if entries is None:
        return "unavailable (no rank-site: tune timing_decisions)"
    if not entries:
        return "none"
    totals = {}
    for _, admission, count in entries:
        totals[admission] = totals.get(admission, 0) + count
    refusals = [f"{adm}={totals[adm]}" for adm in ADMISSION_ORDER if adm != "admitted" and adm in totals]
    admitted = f"admitted={totals['admitted']}" if "admitted" in totals else None
    summary_parts = list(refusals)
    if not refusals:
        summary_parts.append("no refusals")
    if admitted is not None:
        summary_parts.append(admitted)
    detail = ", ".join(f"{settle}:{adm}={count}" for settle, adm, count in entries)
    return f"{', '.join(summary_parts)} ({detail})"


def read_timing_admissions(stem):
    """Read and validate the arm's `rank-site: tune` timing_decisions from `<stem>.out`."""
    out_path = Path(f"{stem}.out")
    if not out_path.is_file():
        return format_timing_admissions(None)
    text = out_path.read_text(encoding="utf-8", errors="replace")
    tune_lines = bench_output.rank_site_lines(text, "tune")
    if not tune_lines:
        return format_timing_admissions(None)
    if len(tune_lines) > 1:
        raise ValueError(f"{out_path}: expected at most one rank-site: tune line, found {len(tune_lines)}")
    _, fields, _ = tune_lines[0]
    if "timing_decisions" not in fields:
        return format_timing_admissions(None)
    return format_timing_admissions(parse_timing_decisions(fields["timing_decisions"]))


def one_field(census, prefix, *, optional=False, allow_repeated=False):
    fields = [line[len(prefix):] for line in census if line.startswith(prefix)]
    if allow_repeated:
        fields = list(dict.fromkeys(fields))
        if len(fields) > 1:
            raise ValueError(f"conflicting {prefix.strip()} values in the seeding census")
    if optional and not fields:
        return None
    if len(fields) != 1 or not fields[0]:
        raise ValueError(f"expected exactly one {prefix.strip()} line in the seeding census")
    return fields[0]


def preset_digest(census, *, optional=False):
    return one_field(census, CENSUS_PREFIX + "preset-digest ", optional=optional)


def seeding_plan(census, *, optional=False, allow_repeated=False):
    return one_field(census, CENSUS_PREFIX + "plan ", optional=optional,
                     allow_repeated=allow_repeated)


def joined_row(path, backend, label, preset_digest, timing_admissions=None):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != CALIBRATION_COLUMNS or fields[0] != backend:
            raise ValueError(f"{path}: expected a current calibration row for {backend}")
        if (fields[1] == preset_digest if preset_digest is not None else fields[3] == label):
            rows.append(fields)
    if len(rows) != 1:
        key = f"{label!r} at digest {preset_digest}" if preset_digest is not None else repr(label)
        if len(rows) == 0:
            detail = (f" (timing_admissions: {timing_admissions})"
                      if timing_admissions is not None else "")
            raise MissingTimingRowError(
                f"{path}: expected one timing row for {key}, found 0{detail}")
        raise ValueError(f"{path}: expected one timing row for {key}, found {len(rows)}")
    _, digest, routine, row_label, time, *_ = rows[0]
    ms = float(time)
    if not digest or not routine or not math.isfinite(ms) or ms <= 0:
        raise ValueError(f"{path}: incomplete timing row for {label!r}")
    return digest, routine, row_label, ms


def read_arm(stem, backend, arm):
    tsv = Path(f"{stem}.tsv")
    record = strict_json.loads((stem / "cell.json").read_text(encoding="utf-8"))
    if not record["ok"]:
        raise ValueError("cell did not pass its configuration check")
    config = record["config"]
    expects = {GATE_KEY: arm, TSV_KEY: str(tsv), "autotune_log_seeding": "true"}
    problems = config_records.expect_problems(config, expects)
    if problems:
        raise ValueError("; ".join(problems))
    config = {key: entry for key, entry in config.items() if key not in (GATE_KEY, TSV_KEY)}
    census = [line for line in Path(f"{stem}.err").read_text(encoding="utf-8", errors="replace")
              .split("\n") if line.startswith(CENSUS_PREFIX)]
    gate = one_field(census, CENSUS_PREFIX + "gate ")
    if not gate.startswith("ranked "):
        raise ValueError(f"expected a ranked gate, got {gate!r}")
    if not any(line.startswith(CENSUS_PREFIX + 'where="seeded_default segment u') for line in census):
        raise ValueError("missing seeded-default segment census")
    label = seeding_plan(census)
    if not label.startswith(("F_preset[", "F_seeded[")):
        raise ValueError(f"unrecognized seeded plan label {label!r}")
    attribution = preset_digest(census) if label.startswith("F_preset[") else None
    timing_admissions = read_timing_admissions(stem)
    digest, routine, row_label, ms = joined_row(tsv, backend, label, attribution, timing_admissions)
    timed_as = f" timed_as={row_label!r}" if row_label != label else ""
    return (config, gate.split(), (f"{arm}: joined=1 label={label!r}{timed_as} routine={routine!r} "
                    f"digest={digest} measured_ms={ms:.6f}", *census), routine, timing_admissions)


def compare(backend, pair):
    records = {}
    errors = []
    missing_row = False
    for arm in ("auto", "auto-legacy"):
        try:
            records[arm] = read_arm(Path(f"{pair}_{arm}"), backend, arm)
        except MissingTimingRowError as exc:
            missing_row = True
            errors.append(f"{arm}: {exc}")
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"{arm}: {exc}")
    if errors:
        if missing_row:
            for arm in ("auto", "auto-legacy"):
                if arm in records:
                    errors.append(f"{arm}: timing_admissions: {records[arm][4]}")
        raise ValueError("\n".join(errors))
    (auto_config, auto_gate, auto_lines, auto_routine, _), (
        legacy_config, legacy_gate, legacy_lines, legacy_routine, _) = (
            records["auto"], records["auto-legacy"])
    changed = sorted(key for key in auto_config.keys() | legacy_config.keys()
                     if auto_config.get(key) != legacy_config.get(key))
    if changed:
        raise ValueError("resolved configuration differs: " + ", ".join(changed))
    if auto_routine != legacy_routine:
        raise ValueError(f"timing rows name different routines: {auto_routine!r}, {legacy_routine!r}")
    if [field for field in auto_gate if not field.startswith("copy_rate=")] != legacy_gate:
        raise ValueError("ranked gate tags must match after removing auto's optional copy_rate")
    print("resolved configuration matches apart from the gate and calibration path")
    if auto_gate == legacy_gate:
        print("ranked gates: identical (auto resolved no operand delivery rate)")
    else:
        print("ranked gates: operand delivery priced by auto only")
    print(*auto_lines, *legacy_lines, sep="\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: gh1253_gate_compare.py BACKEND PAIR_PATH_STEM")
    try:
        compare(*sys.argv[1:])
    except ValueError as exc:
        for line in str(exc).splitlines():
            print(f"gh1253_gate_compare: {line}", file=sys.stderr)
        sys.exit(1)
