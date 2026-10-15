"""Step-time matrix for gh1003_fold_cells.sh's `summary` step (stdlib only).

Usage: python3 gh1003_summary.py OUT TREATMENTS
  OUT         the driver's results directory; only its numbered-pass cells
              (<backend>-<fixture>-<treatment>-r<N>.out) enter, never the dry or seg cells.
  TREATMENTS  the driver's treatments, space-separated (inference then training), in the order the
              rows list them within a fixture.
Prints the Markdown matrix and exits 1 when it is incomplete (a cell without a result line, one
that is not strict JSON, or no cell at all), 0 otherwise. The cell reader and the shared statistics
are gh1133_summary.py's."""
import json
import statistics
import sys

from gh1133_summary import loss_vs, median_p50, p90_p10_spread, read_cells


def fold_matrix(out, treatments):
    """Print the matrix: per cell the p50 of each repeat, their median, the ratio to the same
    fixture's composed cell and to its two-pass cell (two-pass-bwd for the training fixtures), the
    widest p90/p10 spread, the shipped mma census of the first repeat (`shipped_mma`, else an older
    tuned record's `tune.shipped_mma`) and its losses' agreement with the composed cell. Returns
    whether the matrix is incomplete."""
    treatments = treatments.split()
    cells, incomplete = read_cells(out, treatments)
    print("| backend | fixture | treatment | p50 per repeat (ms) | median p50 | vs composed | vs two-pass | p10..p90 spread | shipped mma | loss vs composed |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (backend, fixture, treatment), reps in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], treatments.index(kv[0][2]))):
        p50s = [r["step_ms"]["p50"] for _, r in reps]
        m = statistics.median(p50s)
        comp = median_p50(cells, (backend, fixture, "composed"))
        two = median_p50(cells, (backend, fixture, "two-pass")) or median_p50(cells, (backend, fixture, "two-pass-bwd"))
        record = reps[0][1]
        mma = record["shipped_mma"] if "shipped_mma" in record else (record.get("tune") or {}).get("shipped_mma")
        loss = loss_vs(reps, cells.get((backend, fixture, "composed")))
        print("| %s | %s | %s | %s | %.1f | %s | %s | %.3fx | %s | %s |" % (
            backend, fixture, treatment, ", ".join("%.1f" % p for p in p50s), m,
            "%.3fx" % (m / comp) if comp else "", "%.3fx" % (m / two) if two else "",
            p90_p10_spread(reps), json.dumps(mma) if mma is not None else "", loss))
    return incomplete


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(1 if fold_matrix(sys.argv[1], sys.argv[2]) else 0)
