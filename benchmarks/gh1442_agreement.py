#!/usr/bin/env python3
"""gh-ocannl-1442: how well a ranking score orders what the autotuner measured at one site.

Usage: gh1442_agreement.py TSV NAME=EXPLAIN [NAME=EXPLAIN ...] [--pick NAME=OUT ...] [--band B]

  TSV      the calibration rows the site's tuning cell appended (config autotune_calibration_file:
           backend, digest, routine, label, measured ms, ...), one per timed candidate.
  EXPLAIN  a bench_rank_site.exe stdout: its `rank-site: explain label="..." rank_ms=...` lines,
           the ranking score of each candidate model_explain lists, under one configuration; NAME
           labels it in the report (say `ceilings` and `fitted`).
  --pick   a bench_rank_site.exe stdout holding only a `rank-site: pick` line (BENCH_PICK=1), say
           under a profile: its pick is reported as below, without the ranking analysis.
  --band   relative measured difference below which a pair counts as tied, noise (default 0.10).

The explain lines and the rows join by label: model_explain labels its sketch leaves as the tuner
does; its incumbent `default` is the tuner's `baseline` row and `seeded` its `F_seeded[...]` row.
A candidate measured more than once keeps its fastest time.

Per score: the joined candidates; the DECIDED pairs (measured apart by more than the band); how
many of them the score orders as measured (a score tie orders none); and the score's argmin, with
its measured time against the fastest joined candidate. The argmin breaks score ties by listing
order, as model_default does: its incumbent comes first and wins every tie, and among sketch leaves
the first one strictly below the threshold is kept. With two or more scores, also each later
score against the first: decided pairs the first ordered right and the later one orders wrong
(regressions), and the reverse (improvements).

gh-ocannl-1278 adds, per score, the PICK SETS model_default chooses from: the candidates the
explain line marks `admitted=true` (the numerics restriction's set), without and with the
coalesced branch's leaves (labels carrying `coal-`; bench_rank_site marks one admitted when its
uncoalesced form is). For each set: its argmin under the tie rule above, the fastest member, and
the argmin's slowdown against it -- the ranking check of covering the coalesced seeds in
model_default. These argmins are the SCORE's (incumbent first on ties), not model_default's rule:
since gh-ocannl-1477 its walks are anchored at the seeded plan's own sketch (`anchor=true`), which
this script does not replay.

What model_default actually picks is read off the compile instead (gh-ocannl-1477): an EXPLAIN or
--pick output carrying a `rank-site: pick label="..."` line (BENCH_PICK=1) gets a `pick` line --
the label joined to its measured row as above, its time, the seeded plan's, and their ratio
(`pick_over_seeded`). A `seeded` pick joins only the F_seeded row whose label is the line's
`seeded_label`, the plan its own configuration seeds (content digest included), or, where that plan
seeds nothing (`none`), the `baseline` row; and a `seeded` pick that names its plan passes as that
very plan (ratio 1 by construction) whether or not the timing cell timed it -- an approximate
profile's plan is not the timing cell's. A leaf pick needs its own row and its configuration's
seeded plan's row both timed. An explicit --pick output without a pick line fails too. A pick that does not join a measured row is reported
`pick_unmatched` and makes the exit status 1: an acceptance that cannot time the pick has not
passed. Stdout only; exit 2 on unusable input.
"""

import argparse
import re
import sys

EXPLAIN = re.compile(r'rank-site: explain .*?label="((?:[^"\\]|\\.)*)" rank_ms=(\S+)'
                     r'(?:.* admitted=(true|false))?')


PICK = re.compile(r'rank-site: pick .*?label="((?:[^"\\]|\\.)*)"'
                  r'(?:.* seeded_label="((?:[^"\\]|\\.)*)")?')


def unescape(text):
    return None if text is None else text.encode().decode("unicode_escape")


def pick_of(path):
    """(pick label, the seeded plan's label under the pick's configuration, or None)."""
    with open(path) as f:
        for line in f:
            m = PICK.search(line)
            if m:
                return unescape(m.group(1)), unescape(m.group(2))
    return None, None


def report_pick(name, path, times, required):
    """Prints the pick line; False when there is a pick that does not join a measured row, or when
    a [required] output (an explicit --pick) carries no pick line at all."""
    label, seeded_label = pick_of(path)
    if label is None:
        print(f"{name}: pick none (no `rank-site: pick` line)")
        return not required
    # The seeded plan of the pick's own configuration: the very plan its seeding derives (a profile's
    # auto gate can seed a different plan than the timing cell's pinned threshold), the `baseline`
    # row where that plan seeds, splits and sinks nothing (`none`); the timed F_seeded row only for
    # an older pick line that does not name it.
    if seeded_label is None:
        seeded = row_label("seeded", times)
    elif seeded_label == "none":
        seeded = row_label("default", times)
    else:
        seeded = seeded_label if seeded_label in times else None
    if label == "seeded" and seeded_label is not None:
        # The pick IS the configuration's seeded plan: no slower by construction, timed or not (an
        # approximate profile's plan differs from the timing cell's, its digest with it).
        timed = (f" row={seeded!r} measured_ms={times[seeded]:.6f}" if seeded is not None
                 else " (that plan untimed here)")
        print(f"{name}: pick label='seeded' seeded_label={seeded_label!r}{timed}"
              f" pick_over_seeded=1.000 (the same plan)")
        return True
    row = seeded if label == "seeded" else row_label(label, times)
    if row is None:
        print(f"{name}: pick_unmatched label={label!r} seeded_label={seeded_label!r}")
        return False
    if seeded is None and seeded_label is not None:
        # The pick is timed, but not the plan it is to be compared with.
        print(f"{name}: pick label={label!r} row={row!r} measured_ms={times[row]:.6f}"
              f" seeded_unmatched seeded_label={seeded_label!r}")
        return False
    line = f"{name}: pick label={label!r} row={row!r} measured_ms={times[row]:.6f}"
    if seeded is not None:
        line += (f" seeded_ms={times[seeded]:.6f}"
                 f" pick_over_seeded={times[row] / times[seeded]:.3f}")
    print(line)
    return True


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
    """Label -> (rank, listing position, admitted), in listing order."""
    out = {}
    with open(path) as f:
        for line in f:
            m = EXPLAIN.search(line)
            if m and m.group(2) != "n/a":
                label = m.group(1).encode().decode("unicode_escape")
                out[label] = (float(m.group(2)), len(out), m.group(3) != "false")
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
    ap.add_argument("--pick", action="append", default=[])
    ap.add_argument("--band", type=float, default=0.10)
    args = ap.parse_args()
    times = measured(args.tsv)
    if not times:
        print(f"gh1442_agreement: no calibration rows in {args.tsv}", file=sys.stderr)
        return 2
    results = []
    picks_ok = True
    for spec in args.pick:
        if "=" not in spec:
            print(f"gh1442_agreement: expected NAME=OUT after --pick, got {spec}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        picks_ok = report_pick(name, path, times, required=True) and picks_ok
    for spec in args.explain:
        if "=" not in spec:
            print(f"gh1442_agreement: expected NAME=EXPLAIN, got {spec}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        joined = {}
        order = {}
        admitted = set()
        for label, (rank, pos, adm) in scores(path).items():
            row = row_label(label, times)
            if row is not None:
                joined[label] = (times[row], rank)
                order[label] = pos
                if adm:
                    admitted.add(label)
        if len(joined) < 2:
            print(f"gh1442_agreement: {name}: fewer than two candidates join the rows",
                  file=sys.stderr)
            return 2
        pairs = list(decided_pairs(joined, args.band))
        right = {p for p in pairs if joined[p[0]][1] < joined[p[1]][1]}
        best = min(joined, key=lambda l: joined[l][0])
        pick = min(joined, key=lambda l: (joined[l][1], order[l]))
        print(f"{name}: joined={len(joined)} decided_pairs={len(pairs)} ordered_right={len(right)}"
              f" ({100.0 * len(right) / max(1, len(pairs)):.1f}%) argmin={pick!r}"
              f" measured_ms={joined[pick][0]:.6f} fastest={best!r}"
              f" fastest_ms={joined[best][0]:.6f}"
              f" argmin_slowdown={joined[pick][0] / joined[best][0]:.3f}")
        for label in sorted(joined, key=lambda l: joined[l][0]):
            t, r = joined[label]
            print(f"  {name}: {label!r} measured_ms={t:.6f} rank_ms={r:.6f}"
                  f" admitted={str(label in admitted).lower()}")
        for set_name, members in (
            ("admitted", [l for l in admitted if "coal-" not in l]),
            ("admitted+coalesced", list(admitted)),
        ):
            if not members:
                print(f"{name}: pick_set={set_name} empty")
                continue
            set_pick = min(members, key=lambda l: (joined[l][1], order[l]))
            set_best = min(members, key=lambda l: joined[l][0])
            print(f"{name}: pick_set={set_name} members={len(members)} argmin={set_pick!r}"
                  f" measured_ms={joined[set_pick][0]:.6f} rank_ms={joined[set_pick][1]:.6f}"
                  f" fastest={set_best!r} fastest_ms={joined[set_best][0]:.6f}"
                  f" rank_ms_of_fastest={joined[set_best][1]:.6f}"
                  f" argmin_slowdown={joined[set_pick][0] / joined[set_best][0]:.3f}")
        picks_ok = report_pick(name, path, times, required=False) and picks_ok
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
    return 0 if picks_ok else 1


if __name__ == "__main__":
    sys.exit(main())
