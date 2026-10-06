"""Step matrix and validated training tables for gh1133_cells.sh (stdlib only)."""
import json
import os
from pathlib import Path
import re
import statistics
import sys


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


def step_matrix(out, treatments, ref_treatment, ref_all):
    """Print the step-time matrix; whether it is incomplete (a missing result or reference)."""
    treatments = treatments.split()
    # The cells of exactly the treatments the driver ran: its TREATMENTS list, not a second copy of
    # the names its flags_of knows.
    cell_name = re.compile(r"(\w+)-(gpt2_mini\w*)-(%s)-(r\d+)\.out" % "|".join(map(re.escape, treatments)))
    cells, missing = {}, []
    for name in sorted(os.listdir(out)):
        m = cell_name.fullmatch(name)
        if not m:
            continue
        rec = None
        for line in open(os.path.join(out, name)):
            line = line.strip()
            if line.startswith("{") and '"step_ms"' in line:
                rec = json.loads(line)
        if rec is None:
            missing.append(name)
            continue
        cells.setdefault(m.group(1, 2, 3), []).append((m.group(4), rec))
    incomplete = bool(missing) or not cells
    for name in missing:
        print("MISSING RESULT: %s has no result line" % name)
    if not cells:
        print("NO MEASUREMENT: no numbered cell produced a result line")
    print("| backend | fixture | treatment | p50 per repeat (ms) | median p50 | vs %s | p10..p90 spread | queued (median) | loss vs %s |" % (ref_treatment, ref_treatment))
    print("|---|---|---|---|---|---|---|---|---|")
    missing_refs = []
    def med(key):
        reps = cells.get(key, [])
        return statistics.median(r["step_ms"]["p50"] for _, r in reps) if reps else None
    for (backend, fixture, treatment), reps in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], treatments.index(kv[0][2]) if kv[0][2] in treatments else 99)):
        p50s = [r["step_ms"]["p50"] for _, r in reps]
        m = statistics.median(p50s)
        spread = max(r["step_ms"]["p90"] / r["step_ms"]["p10"] for _, r in reps)
        queued = statistics.median(r.get("queued_step_ms") or 0.0 for _, r in reps)
        # A d1 treatment is compared with the d1 form of the reference (the same attention form).
        rt = ref_all or (("d1" + ref_treatment) if treatment.startswith("d1") else ref_treatment)
        ref = med((backend, fixture, rt))
        if ref is None:
            # The comparison the summary advertises did not happen: an incomplete matrix.
            missing_refs.append("%s %s %s (reference %s)" % (backend, fixture, treatment, rt))
        loss = ""
        bref = cells.get((backend, fixture, rt))
        if bref:
            a, b = reps[0][1].get("losses") or [], bref[0][1].get("losses") or []
            if a and b and len(a) == len(b):
                loss = "%.2g" % max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(a, b))
        print("| %s | %s | %s | %s | %.2f | %s | %.3fx | %.2f | %s |" % (
            backend, fixture, treatment, ", ".join("%.2f" % p for p in p50s), m,
            "%.3fx" % (m / ref) if ref else "", spread, queued, loss))
    for row in missing_refs:
        print("MISSING REFERENCE: %s has no reference cell to compare with" % row)
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
                print("REFUSED: %s %s" % (name, reason))
                incomplete = True
                continue
            print("\n### Training segments: %s\n" % name[:-4])
            print("Isolated min-of-20 launch + sync times; their sum is not a step latency.\n")
            print("```text")
            print(content.rstrip())
            print("```")
    return 1 if incomplete else 0


if __name__ == "__main__":
    if sys.argv[1] == "--check-training":
        _, reason = training_output(sys.argv[2], sys.argv[3])
        if reason:
            print("REFUSED: %s %s" % (sys.argv[3], reason))
        sys.exit(1 if reason else 0)
    if sys.argv[1] == "--no-step-stage":
        sys.exit(summary(*sys.argv[2:], step_stage=False))
    sys.exit(summary(*sys.argv[1:]))
