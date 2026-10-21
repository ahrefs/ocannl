#!/usr/bin/env python3
"""gh-ocannl-1442: how well a ranking score orders what the autotuner measured at one site.

Usage: gh1442_agreement.py TSV NAME=EXPLAIN [NAME=EXPLAIN ...] [--band B]

  TSV      the calibration rows the site's tuning cell appended (config autotune_calibration_file:
           backend, digest, routine, label, measured ms, ...), one per timed candidate.
  EXPLAIN  a bench_rank_site.exe stdout: its `rank-site: explain label="..." rank_ms=...` lines,
           the ranking score of each candidate model_explain lists, under one configuration; NAME
           labels it in the report (say `ceilings` and `fitted`).
  --band   relative measured difference below which a pair counts as tied, noise (default 0.10).

The explain lines and the rows join by label: model_explain labels its sketch leaves as the tuner
does; its incumbent `default` is the tuner's `baseline` row and `seeded` its `F_seeded[...]` row.
A candidate measured more than once keeps its fastest time.

Per score: the joined candidates; the DECIDED pairs (measured apart by more than the band); how
many of them the score orders as measured (a score tie orders none); and the score's argmin, with
its measured time against the fastest joined candidate. With two or more scores, also each later
score against the first: decided pairs the first ordered right and the later one orders wrong
(regressions), and the reverse (improvements). Stdout only; exit 2 on unusable input.
"""

import argparse
import re
import sys

EXPLAIN = re.compile(r'rank-site: explain .*?label="((?:[^"\\]|\\.)*)" rank_ms=(\S+)')


def measured(path):
    times = {}
    with open(path) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5 or line.startswith("#"):
                continue
            try:
                ms = float(cols[4])
            except ValueError:
                continue
            label = cols[3]
            times[label] = min(ms, times.get(label, float("inf")))
    return times


def scores(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = EXPLAIN.search(line)
            if m and m.group(2) != "n/a":
                out[m.group(1).encode().decode("unicode_escape")] = float(m.group(2))
    return out


def row_label(label, times):
    if label == "default":
        return "baseline" if "baseline" in times else None
    if label == "seeded":
        seeded = [t for t in times if t.startswith("F_seeded[")]
        return seeded[0] if len(seeded) == 1 else None
    return label if label in times else None


def decided_pairs(joined, band):
    labels = sorted(joined)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ta, tb = joined[a][0], joined[b][0]
            if max(ta, tb) > (1.0 + band) * min(ta, tb):
                yield (a, b) if ta < tb else (b, a)  # (faster, slower)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("explain", nargs="+")
    ap.add_argument("--band", type=float, default=0.10)
    args = ap.parse_args()
    times = measured(args.tsv)
    if not times:
        print(f"gh1442_agreement: no calibration rows in {args.tsv}", file=sys.stderr)
        return 2
    results = []
    for spec in args.explain:
        if "=" not in spec:
            print(f"gh1442_agreement: expected NAME=EXPLAIN, got {spec}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        joined = {}
        for label, rank in scores(path).items():
            row = row_label(label, times)
            if row is not None:
                joined[label] = (times[row], rank)
        if len(joined) < 2:
            print(f"gh1442_agreement: {name}: fewer than two candidates join the rows",
                  file=sys.stderr)
            return 2
        pairs = list(decided_pairs(joined, args.band))
        right = {p for p in pairs if joined[p[0]][1] < joined[p[1]][1]}
        best = min(joined, key=lambda l: joined[l][0])
        pick = min(joined, key=lambda l: (joined[l][1], l))
        print(f"{name}: joined={len(joined)} decided_pairs={len(pairs)} ordered_right={len(right)}"
              f" ({100.0 * len(right) / max(1, len(pairs)):.1f}%) argmin={pick!r}"
              f" measured_ms={joined[pick][0]:.6f} fastest={best!r}"
              f" fastest_ms={joined[best][0]:.6f}"
              f" argmin_slowdown={joined[pick][0] / joined[best][0]:.3f}")
        for label in sorted(joined, key=lambda l: joined[l][0]):
            t, r = joined[label]
            print(f"  {name}: {label!r} measured_ms={t:.6f} rank_ms={r:.6f}")
        results.append((name, set(pairs), right))
    if len(results) > 1:
        first, first_pairs, first_right = results[0]
        for name, pairs, right in results[1:]:
            common = first_pairs & pairs
            regressions = sorted(p for p in common if p in first_right and p not in right)
            improvements = sorted(p for p in common if p not in first_right and p in right)
            print(f"{name} against {first}: common_decided={len(common)}"
                  f" regressions={len(regressions)} improvements={len(improvements)}")
            for faster, slower in regressions:
                print(f"  regression: {faster!r} measured faster than {slower!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
