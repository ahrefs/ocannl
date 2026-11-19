#!/usr/bin/env python3
"""Join one gh1253_cells.sh gate pair to its calibration rows.

Reads only the pair's successful cells.py cell.json records, autotune-seeding:
stderr lines, and current 12-column calibration TSVs. A join needs one ranked
gate, one exact plan label, segment census evidence, and one positive timing
row in the same arm. Seeded plans join by label. The configured preset joins
by the tuner's resolved digest, which can also identify a deduplicated row.
The ranked gate tags must match after removing auto's optional copy_rate.
Other stderr is ignored. Old TSV schemas and rows from other cells are outside
the join.
"""

import math
import sys
from pathlib import Path

import config_records
import strict_json


GATE_KEY = "gpu_default_sketch_min_reduction"
TSV_KEY = "autotune_calibration_file"
CENSUS_PREFIX = "autotune-seeding: "
CALIBRATION_COLUMNS = 12


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


def joined_row(path, backend, label, preset_digest):
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
    digest, routine, row_label, ms = joined_row(tsv, backend, label, attribution)
    timed_as = f" timed_as={row_label!r}" if row_label != label else ""
    return config, gate.split(), (f"{arm}: joined=1 label={label!r}{timed_as} routine={routine!r} "
                    f"digest={digest} measured_ms={ms:.6f}", *census), routine


def compare(backend, pair):
    records = []
    for arm in ("auto", "auto-legacy"):
        try:
            records.append(read_arm(Path(f"{pair}_{arm}"), backend, arm))
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(f"{arm}: {exc}") from exc
    (auto_config, auto_gate, auto_lines, auto_routine), (
        legacy_config, legacy_gate, legacy_lines, legacy_routine) = records
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
        print(f"gh1253_gate_compare: {exc}", file=sys.stderr)
        sys.exit(1)
