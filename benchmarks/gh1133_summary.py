"""Step matrix and validated training tables for gh1133_cells.sh (stdlib only).

Its cell reader (`read_cells`) and the statistics both drivers' matrices quote (`median_p50`,
`p90_p10_spread`, `loss_vs`) are also gh1003_summary.py's: one owner for how a cell's result line is
read and refused."""
import os
from pathlib import Path
import re
import statistics
import sys

import bench_output
from display_safe import display_value, log_line, table_lines


def training_output(out, name):
    """Accept a training table only after its cell exited successfully in training mode."""
    content = (Path(out) / name).read_text()
    mode = next((line for line in content.splitlines() if line.startswith("mode:")), "")
    if not mode.startswith("mode: train backend:"):
        return None, "is not a training diagnostic"
    status = Path(out) / (name[:-4] + ".exit")
    if not status.is_file() or status.read_text().strip() != "0":
        return None, "did not complete successfully"
    return content, None


def read_cells(out, treatments):
    """The result lines of OUT's numbered cells of `treatments` (a list of names), as
    {(backend, fixture, treatment): [(repeat, record), ...]}, and whether they are incomplete (a
    cell without a result line, one that is not strict JSON, or no cell at all); each such defect
    is printed by name ahead of the matrix."""
    # The cells of exactly the treatments the driver ran: its own treatment list, not a second copy
    # of the names its flags know. Only the numbered passes enter (a dry or seg cell is no repeat).
    cell_name = re.compile(r"(\w+)-(gpt2_mini\w*)-(%s)-(r\d+)\.out" % "|".join(map(re.escape, treatments)))
    cells, missing, unreadable = {}, [], []
    for name in sorted(os.listdir(out)):
        m = cell_name.fullmatch(name)
        if not m:
            continue
        rec, error = None, None
        try:
            with open(os.path.join(out, name)) as f:
                rec = bench_output.result_record(f.read())
        except ValueError as e:
            # Truncated, or a non-finite number JSON cannot spell (strict_json).
            error = e
        if error is not None:
            unreadable.append((name, error))
            continue
        if rec is None:
            missing.append(name)
            continue
        cells.setdefault(m.group(1, 2, 3), []).append((m.group(4), rec))
    # A summary is only as complete as its cells (the table is still printed for what exists).
    incomplete = bool(missing) or bool(unreadable) or not cells
    for name in missing:
        print(log_line("MISSING RESULT: %s has no result line" % display_value(name)))
    for name, error in unreadable:
        print(log_line("UNREADABLE RESULT: %s: its result line is not strict JSON (%s)"
                       % (display_value(name), error)))
    if not cells:
        print("NO MEASUREMENT: no numbered cell produced a result line")
    return cells, incomplete


def median_p50(cells, key):
    """The median p50 over the repeats of the cell `key`, or None when it has none."""
    reps = cells.get(key, [])
    return statistics.median(r["step_ms"]["p50"] for _, r in reps) if reps else None


def p90_p10_spread(reps):
    """The widest p90/p10 ratio among a cell's repeats."""
    return max(r["step_ms"]["p90"] / r["step_ms"]["p10"] for _, r in reps)


def loss_vs(reps, ref_reps):
    """The largest relative loss difference of a cell's first repeat from its reference's first
    repeat, formatted; empty when either has no losses or their lengths differ."""
    if not ref_reps:
        return ""
    a, b = reps[0][1].get("losses") or [], ref_reps[0][1].get("losses") or []
    if a and b and len(a) == len(b):
        return "%.2g" % max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(a, b))
    return ""


def step_matrix(out, treatments, ref_treatment, ref_all):
    """Print the step-time matrix; whether it is incomplete (a missing result or reference)."""
    treatments = treatments.split()
    cells, incomplete = read_cells(out, treatments)
    rows, missing_refs = [], []
    for (backend, fixture, treatment), reps in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], treatments.index(kv[0][2]) if kv[0][2] in treatments else 99)):
        p50s = [r["step_ms"]["p50"] for _, r in reps]
        m = statistics.median(p50s)
        spread = p90_p10_spread(reps)
        queued = statistics.median(r.get("queued_step_ms") or 0.0 for _, r in reps)
        # A d1 treatment is compared with the d1 form of the reference (the same attention form).
        rt = ref_all or (("d1" + ref_treatment) if treatment.startswith("d1") else ref_treatment)
        ref = median_p50(cells, (backend, fixture, rt))
        if ref is None:
            # The comparison the summary advertises did not happen: an incomplete matrix.
            missing_refs.append("%s %s %s (reference %s)" % tuple(
                map(display_value, (backend, fixture, treatment, rt))))
        loss = loss_vs(reps, cells.get((backend, fixture, rt)))
        rows.append((
            display_value(backend), display_value(fixture), display_value(treatment),
            ", ".join("%.2f" % p for p in p50s), "%.2f" % m, "%.3fx" % (m / ref) if ref else "",
            "%.3fx" % spread, "%.2f" % queued, loss))
    shown_ref = display_value(ref_treatment)
    print("\n".join(table_lines(
        ["backend", "fixture", "treatment", "p50 per repeat (ms)", "median p50", "vs " + shown_ref,
         "p10..p90 spread", "queued (median)", "loss vs " + shown_ref], rows)))
    for row in missing_refs:
        print(log_line("MISSING REFERENCE: %s has no reference cell to compare with" % row))
    return incomplete or bool(missing_refs)


def summary(out, treatments, ref_treatment, ref_all="", step_stage=True):
    """The report of one driver invocation. Without a step stage there is no matrix to owe: a prep
    or dry run passes on its training tables alone, and only a step stage's empty matrix fails."""
    if step_stage:
        incomplete = step_matrix(out, treatments, ref_treatment, ref_all)
    else:
        print("No step stage requested: no step-time matrix.")
        incomplete = False
    for name in sorted(os.listdir(out)):
        if name.endswith("-trainseg.out"):
            content, reason = training_output(out, name)
            if reason:
                print(log_line("REFUSED: %s %s" % (display_value(name), reason)))
                incomplete = True
                continue
            print("\n" + log_line("### Training segments: %s" % display_value(name[:-4])) + "\n")
            print("Isolated min-of-20 launch + sync times; their sum is not a step latency, and a row"
                  " is not that kernel's share of the step: the kprof step's in-step column is"
                  " (benchmarks/kernel_attribution.py).\n")
            print("```text")
            print(content.rstrip())
            print("```")
    return 1 if incomplete else 0


if __name__ == "__main__":
    if sys.argv[1] == "--check-training":
        _, reason = training_output(sys.argv[2], sys.argv[3])
        if reason:
            print(log_line("REFUSED: %s %s" % (display_value(sys.argv[3]), reason)))
        sys.exit(1 if reason else 0)
    if sys.argv[1] == "--no-step-stage":
        sys.exit(summary(*sys.argv[2:], step_stage=False))
    sys.exit(summary(*sys.argv[1:]))
