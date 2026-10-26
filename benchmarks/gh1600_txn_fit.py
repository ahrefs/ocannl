#!/usr/bin/env python3
"""gh-ocannl-1600: fit and evaluate the ranking score's memory-transaction rate offline.

Usage: gh1600_txn_fit.py CELL [CELL ...] [--band B] [--sector 32|64|128] [--rate R] [--show]
                         [--occupancy CAP,TPM]

  CELL      NAME=TSV,EXPLAIN: one site's timing rows (config autotune_calibration_file, every
            candidate timed by gh1253_cells.sh's `time` cell) and a bench_rank_site.exe stdout
            listed WITHOUT a transaction rate (model_transaction_rate unset or none): its
            `rank-site: explain` lines (rank, admitted) and `rank-site: kernels` lines (per kernel:
            compute leg c, memory leg m, the fill-weighted excess transactions txn / txn64 /
            txn128, uncovered accesses, workgroup bytes, block threads).
  --band    relative measured difference below which a pair counts as tied (default 0.10).
  --sector  the counting unit whose excess is priced (default 32: what the score counts).
  --rate    evaluate this one rate (sectors/s; `none`: no transaction term) instead of sweeping:
            the held-out check, on cells timed apart from the ones the rate was fitted on.
  --show    with --rate, list every candidate's measured median and re-scored rank.
  --no-coalesced  for a backend that does not admit the coalesced layouts (no occupancy capacity):
            judge the admitted set alone -- the `coal-` rows leave the population before anything
            is scored, so neither the agreement nor the spread reads an ordering the backend never
            ranks.
  --occupancy  re-score under the occupancy factor of a workgroup capacity CAP (bytes per
            multiprocessor) and TPM thread slots per multiprocessor (cost_model.ml's occupancy,
            from each kernel's wg= and thr=), replacing the one the listing priced (its occ=,
            1 where it priced none): what the factor does before an advisory value ships.

What it re-scores: per candidate, the sum over its kernels of max(c + txn / rate, m) -- the
runtime score with the term priced (cost_model.ml's ranking_legs_of adds the filled excess over the
rate to the compute leg). Before anything is fitted it checks that, with no rate, the sum of
max(c, m) reproduces the explain line's rank_ms for every candidate: a listing scored with a rate,
or legs the kernel lines do not carry, fail here rather than fit noise (exit 2).

Times are per-label MEDIANS over every replicate row, not the fastest-of: the acceptance compares
medians against a spread fixed before fitting -- the fastest admitted+coalesced candidate's p90/p10
over its own replicates (1.0 with fewer than 3). Per cell and rate: the agreement (decided pairs,
measured more than the band apart, ordered as measured; a score tie orders none), and per pick set
-- `admitted` (the explain line's admitted=true, labels without `coal-`) and `admitted+coalesced`
-- the score's argmin (ties to the listing order, model_default's rule) and its median slowdown
against the set's fastest member. A cell PASSES at a rate when its admitted+coalesced argmin's
slowdown is at most the spread or is not a coalesced leaf (the issue's criterion), its agreement is
no lower than the listing's own, and its admitted-set argmin is no slower than the listing's.

The sweep (no --rate) prints one row per rate on a log grid and the rate the cells agree on: the
largest number of passing cells, then the smallest worst slowdown, then the highest least
agreement. These are the SCORE's argmins, not model_default's anchored walk; BENCH_PICK=1 pick
cells read what model_default actually ships.
"""

import argparse
import math
import re
import statistics
import sys

EXPLAIN = re.compile(r'rank-site: explain .*?label="((?:[^"\\]|\\.)*)" rank_ms=(\S+)'
                     r'(?:.* admitted=(true|false))?')
KERNELS = re.compile(r'rank-site: kernels .*?label="((?:[^"\\]|\\.)*)" kernels=(\S*)')


def unescape(text):
    return text.encode().decode("unicode_escape")


def replicates(path):
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
            times.setdefault(cols[3], []).append(ms)
    return times


def row_label(label, times):
    if label == "default":
        return "baseline" if "baseline" in times else None
    if label == "seeded":
        seeded = [t for t in times if t.startswith("F_seeded[")]
        return seeded[0] if len(seeded) == 1 else None
    return label if label in times else None


def parse_kernel(spec):
    fields = dict(kv.split("=", 1) for kv in spec.split(","))
    num = lambda k: None if fields[k] == "n/a" else float(fields[k])
    return {"c": num("c"), "m": num("m"), "txn32": num("txn"), "txn64": num("txn64"),
            "txn128": num("txn128"), "unc": int(fields["unc"]), "wg": int(fields["wg"]),
            "thr": int(fields["thr"]), "occ": float(fields.get("occ", "1"))}


def occupancy(wg, thr, cap, tpm):
    t = max(1, thr)
    by_threads = tpm // t
    by_bytes = by_threads if wg <= 0 else max(0, cap) // wg
    return min(1.0, max(1, min(by_threads, by_bytes)) * t / tpm)


def listing(path):
    """Label -> (rank_ms, position, admitted, kernels)."""
    ranks, kernels = {}, {}
    with open(path) as f:
        for line in f:
            m = EXPLAIN.search(line)
            if m and m.group(2) != "n/a":
                ranks[unescape(m.group(1))] = (float(m.group(2)), len(ranks), m.group(3) != "false")
                continue
            m = KERNELS.search(line)
            if m:
                kernels[unescape(m.group(1))] = [parse_kernel(k) for k in m.group(2).split(";") if k]
    return {l: (r, pos, adm, kernels.get(l)) for l, (r, pos, adm) in ranks.items()}


def rescore(kernels, rate, sector, occ=None):
    total = 0.0
    for k in kernels:
        c = k["c"]
        txn = k[f"txn{sector}"] or 0.0
        # The listing's compute leg is over its own occupancy: back to the unoccupied leg first.
        if c is not None:
            c = c * k["occ"]
        if c is not None and rate is not None:
            c = c + txn / rate * 1e3
        if c is not None:
            o = k["occ"] if occ is None else occupancy(k["wg"], k["thr"], *occ)
            c = c / o
        legs = [x for x in (c, k["m"]) if x is not None]
        total += max(legs) if legs else 0.0
    return total


def load_cell(spec):
    name, rest = spec.split("=", 1)
    tsv, explain = rest.split(",", 1)
    reps = replicates(tsv)
    cands = {}
    bad = []
    for label, (rank, pos, adm, kernels) in listing(explain).items():
        row = row_label(label, reps)
        if row is None or kernels is None:
            continue
        offline = rescore(kernels, None, 32)
        # The kernel legs print at 1e-6 ms; their sum may round apart by a few units.
        if abs(offline - rank) > 1e-5 * max(1, len(kernels)) + 1e-9 * rank:
            bad.append((label, rank, offline))
        cands[label] = {"times": reps[row], "median": statistics.median(reps[row]), "pos": pos,
                        "admitted": adm, "kernels": kernels,
                        "uncovered": sum(k["unc"] for k in kernels)}
    return name, cands, bad


def spread_of(cands):
    members = [l for l, c in cands.items() if c["admitted"]]
    if not members:
        return 1.0
    best = min(members, key=lambda l: cands[l]["median"])
    ts = sorted(cands[best]["times"])
    if len(ts) < 3:
        return 1.0
    q = statistics.quantiles(ts, n=10, method="inclusive")
    return q[8] / q[0]


def evaluate(cands, rate, sector, band, occ=None):
    ranks = {l: rescore(c["kernels"], rate, sector, occ) for l, c in cands.items()}
    labels = sorted(cands)
    decided = right = 0
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ta, tb = cands[a]["median"], cands[b]["median"]
            if max(ta, tb) > (1.0 + band) * min(ta, tb):
                decided += 1
                fast, slow = (a, b) if ta < tb else (b, a)
                right += ranks[fast] < ranks[slow]
    sets = {}
    for set_name, members in (
        ("admitted", [l for l in cands if cands[l]["admitted"] and "coal-" not in l]),
        ("admitted+coalesced", [l for l in cands if cands[l]["admitted"]]),
    ):
        if not members:
            continue
        pick = min(members, key=lambda l: (ranks[l], cands[l]["pos"]))
        best = min(members, key=lambda l: cands[l]["median"])
        sets[set_name] = (pick, cands[pick]["median"] / cands[best]["median"], best)
    return 100.0 * right / max(1, decided), decided, sets, ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--sector", type=int, default=32, choices=(32, 64, 128))
    ap.add_argument("--rate", type=lambda v: 0.0 if v == "none" else float(v))
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--occupancy")
    ap.add_argument("--no-coalesced", dest="no_coalesced", action="store_true")
    args = ap.parse_args()
    occ = None
    if args.occupancy:
        cap, tpm = args.occupancy.split(",")
        occ = (int(cap), int(tpm))
    cells = []
    for spec in args.cells:
        name, cands, bad = load_cell(spec)
        if args.no_coalesced:
            cands = {l: c for l, c in cands.items() if "coal-" not in l}
        if bad:
            for label, rank, offline in bad[:5]:
                print(f"gh1600_txn_fit: {name}: {label!r} rank_ms={rank:.6f} but the kernel legs"
                      f" sum to {offline:.6f}: a listing scored with a transaction rate?",
                      file=sys.stderr)
            return 2
        if len(cands) < 2:
            print(f"gh1600_txn_fit: {name}: fewer than two candidates join", file=sys.stderr)
            return 2
        spread = spread_of(cands)
        unc = sum(c["uncovered"] for c in cands.values())
        base = evaluate(cands, None, args.sector, args.band)  # the listing as it was scored
        print(f"{name}: joined={len(cands)} spread={spread:.3f} uncovered_accesses={unc}"
              f" off_agreement={base[0]:.1f}% decided={base[1]}"
              + "".join(f" off_{s}={p!r}@{sl:.3f}" for s, (p, sl, _) in base[2].items()))
        cells.append((name, cands, spread, base))

    def judge(rate):
        rows = []
        for name, cands, spread, base in cells:
            agree, _, sets, ranks = evaluate(cands, rate, args.sector, args.band, occ)
            pick, slow, best = sets.get("admitted+coalesced", (None, 1.0, None))
            _, adm_slow, _ = sets.get("admitted", (None, 1.0, None))
            _, base_adm_slow, _ = base[2].get("admitted", (None, 1.0, None))
            # The issue's criterion: the argmin is never a COALESCED leaf slower than the spread;
            # beside it, no agreement and no admitted-set argmin worse than the listing's own.
            coalesced_ok = "coal-" not in (pick or "") or slow <= spread
            ok = (coalesced_ok and agree >= base[0] - 1e-9
                  and adm_slow <= base_adm_slow + 1e-9)
            rows.append((name, agree, pick, slow, best, ok, sets, ranks, cands))
        return rows

    if args.rate is not None:
        rows = judge(args.rate or None)
        for name, agree, pick, slow, best, ok, sets, ranks, cands in rows:
            print(f"{name}: rate={args.rate or 0:.3e} agreement={agree:.1f}% pass={str(ok).lower()}"
                  + "".join(f" {s}={p!r}@{sl:.3f} fastest={b!r}" for s, (p, sl, b) in sets.items()))
            if args.show:
                for l in sorted(cands, key=lambda l: cands[l]["median"]):
                    c = cands[l]
                    print(f"  {name}: {l!r} median_ms={c['median']:.6f} rank_ms={ranks[l]:.6f}"
                          f" admitted={str(c['admitted']).lower()} uncovered={c['uncovered']}")
        return 0 if all(r[5] for r in rows) else 1
    grid = [10 ** (e / 10) for e in range(80, 141)]
    scored = []
    for rate in grid:
        rows = judge(rate)
        passing = sum(r[5] for r in rows)
        worst = max(r[3] for r in rows)
        least = min(r[1] for r in rows)
        scored.append(((-passing, worst, -least), rate))
        print(f"rate={rate:.3e} passing={passing}/{len(rows)} worst_slowdown={worst:.3f}"
              f" least_agreement={least:.1f}%"
              + "".join(f" {r[0]}={r[1]:.1f}%@{r[3]:.3f}/adm@{r[6].get('admitted', (0, 1.0))[1]:.3f}"
                        for r in rows))
    (key, rate) = min(scored)
    print(f"fit: model_transaction_rate={rate:.6e} passing={-key[0]}/{len(cells)}"
          f" worst_slowdown={key[1]:.3f} least_agreement={-key[2]:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
