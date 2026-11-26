#!/usr/bin/env python3
"""gh-ocannl-1600: fit and evaluate the ranking score's memory-transaction rate offline.

Usage: gh1600_txn_fit.py CELL [CELL ...] [--band B] [--sector 32|64|128] [--rate R] [--show]
                         [--occupancy CAP,TPM] [--staged-rate R|sweep --listed-staged-rate L]
                         [--shared-rate R|sweep [--listed-shared-rate L]] [--pairs]
                         [--census [CELL=]TIMING_STDERR ...]

  CELL      NAME=TSV,EXPLAIN: one site's stock timing rows (config autotune_calibration_file,
            every candidate timed by gh1253_cells.sh's `time` cell; pass the `.tsv.stock` snapshot
            when profile timing cells appended to the main TSV) and a bench_rank_site.exe stdout
            listed WITHOUT a transaction rate (model_transaction_rate unset or none): its
            `rank-site: explain` lines (rank, admitted) and `rank-site: kernels` lines (per kernel:
            compute leg c, memory leg m, the fill-weighted excess transactions txn / txn64 /
            txn128, uncovered accesses, workgroup bytes, block threads). Archive provenance comes
            solely from the exact `# gh1253_cells: stock snapshot required` comment line: a marked
            TSV is refused (exit 2, `stock_snapshot_missing` on stderr, no stdout) so profile rows
            cannot enter a stock fit; pass `.tsv.stock` explicitly for older unmarked mixed TSVs.
  --census  stderr of a stock timing cell (repeatable): `CELL=TIMING_STDERR`, or bare
            `TIMING_STDERR` when a single CELL is given (refused with multiple cells; in single-
            cell mode an argument without the `CELL=` prefix is treated as a bare path). Reads only
            `autotune-seeding: plan` and `autotune-seeding: preset-digest` records via
            `gh1442_agreement.stock_attribution`, requiring agreement within and across logs per
            cell; other stderr is ignored. Attributes `default` and a preset-plan `seeded`
            (`F_preset[...]`) by `preset-digest` against positive 12-column stock rows, keeping
            attributed replicates independent of same-label rows at other digests, and attributes
            a census-selected `F_seeded[...]` by exact label. Missing attributed incumbents fail
            closed (`preset_unmatched` / `seeded_plan_unmatched`, exit 2, no stdout). With
            `--staged-rate`, the same cell's census applies to its `NOSTG_LISTING` reference too.
            Without `--census`, retains the legacy `baseline` / unique `F_seeded[...]` lookup.
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
  --staged-rate  gh-ocannl-1821 (each CELL then NAME=TSV,LISTING,NOSTG_LISTING, the third the
            cell's `nostg` arm, the fill-only reference every agreement and admitted-set argmin is
            judged against): re-score the staged-copy term at this rate (bytes/s), or `sweep`
            a log grid of rates, on a listing scored at --listed-staged-rate L (the backend's
            advisory or the model_staged_copy_rate it ran under): each kernel's stg= (its staged
            bytes over their fill) moves its compute leg by stg * (1/R - 1/L), before its
            occupancy. The launch tail the rate switches on stays as listed, so R is a rate of the
            same calibration. The listing passes the rebuild check below as is: its
            c= already carries the term. Kernel lines without stg= (pre-1821 listings) carry none;
            benchmarks/gh1821_geometry_fit.py fits those from the leaves' block tiles instead.
  --shared-rate  gh-ocannl-2262: re-score the register-reuse term at this rate (cells/s), or
            `sweep` a log grid, on a listing scored at --listed-shared-rate L (default none): each
            kernel's shl= (its shared loads beyond the calibration tile's reuse, over their fill,
            listed whether or not a rate priced them) moves its compute leg by shl * (1/R - 1/L),
            before its occupancy. With --staged-rate too, the two sweep jointly (a grid of pairs):
            the check that the copy rate fitted before this term still holds beside it. The base
            each agreement is judged against is the listing re-scored without the term (or, with
            --staged-rate, the nostg reference as there). The lane group is not re-scored: it moves
            the fill, so a listing per lane width (gh1253_cells.sh FILL_LANE_WIDTH=W) is swept on
            its own.
  --pairs   gh-ocannl-2262: list the saturated register-tile pairs -- two leaves of one block tile,
            flavour and site, 2x2 against 4x4 -- with their measured median ratio and the score's
            ratio as listed and (with --shared-rate R) re-scored.

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
import statistics
import sys
from typing import NamedTuple

import bench_output
from gh1442_agreement import (calibration_rows, stock_attribution, uses_preset,
                              uses_seeded_plan)


class PresetReplicates(NamedTuple):
    label: str
    times: list[float]


class StockReplicates(NamedTuple):
    plan: str | None
    digest: str | None
    times: dict[str, list[float]]
    preset: PresetReplicates | None

    def uses_preset(self, label):
        return uses_preset(self.plan, self.digest, label)

    def uses_seeded_plan(self, label):
        return uses_seeded_plan(self.plan, label)


def stock_replicates(path, census=()):
    plan, digest = stock_attribution(census)
    times = {}
    preset_times = []
    preset_label = None
    preset_min = float("inf")
    hint = f"pass the separate {path}.stock snapshot instead of a mixed TSV"
    for label, ms, attributed in calibration_rows(path, digest, missing_snapshot_hint=hint):
        times.setdefault(label, []).append(ms)
        # A label can cover several digests. Keep the attributed replicates independently
        # so a faster namesake cannot enter the preset's median or spread.
        if attributed:
            preset_times.append(ms)
            if ms < preset_min:
                preset_min = ms
                preset_label = label
    preset = PresetReplicates(preset_label, preset_times) if preset_times else None
    return StockReplicates(plan, digest, times, preset)


def row_replicates(label, stock, cell_name=None):
    prefix = f"{cell_name}: " if cell_name else ""
    if stock.uses_preset(label):
        if stock.preset is None:
            raise ValueError(
                f"{prefix}preset_unmatched label={label!r}"
                f" digest={stock.digest or 'unavailable'}")
        return stock.preset.times, stock.preset.label
    if label == "default":
        key = "baseline"
    elif label == "seeded":
        if stock.uses_seeded_plan(label):
            if stock.plan not in stock.times:
                raise ValueError(
                    f"{prefix}seeded_plan_unmatched label={label!r} plan={stock.plan!r}")
            key = stock.plan
        else:
            seeded = [t for t in stock.times if t.startswith("F_seeded[")]
            if len(seeded) != 1:
                return None
            key = seeded[0]
    else:
        key = label
    return (stock.times[key], None) if key in stock.times else None


def parse_census_args(cell_specs, census_specs):
    cell_names = []
    for spec in cell_specs:
        if "=" not in spec:
            raise ValueError(f"expected NAME=TSV,EXPLAIN[,NOSTG], got {spec!r}")
        name = spec.split("=", 1)[0]
        if not name:
            raise ValueError(f"expected NAME=TSV,EXPLAIN[,NOSTG], got {spec!r}")
        cell_names.append(name)
    by_cell = {name: [] for name in cell_names}
    if not census_specs:
        return by_cell
    if len(by_cell) != len(cell_names):
        raise ValueError("--census requires distinct cell names")
    for spec in census_specs:
        if len(cell_names) == 1:
            cell = cell_names[0]
            prefix = f"{cell}="
            if spec.startswith(prefix):
                cpath = spec[len(prefix):]
                if not cpath:
                    raise ValueError(f"--census for cell {cell!r} has an empty path")
                by_cell[cell].append(cpath)
            else:
                if not spec:
                    raise ValueError("--census path must not be empty")
                by_cell[cell].append(spec)
        else:
            if "=" not in spec:
                raise ValueError(
                    "--census requires CELL=TIMING_STDERR when multiple cells are given")
            cname, cpath = spec.split("=", 1)
            if cname not in by_cell:
                raise ValueError(f"--census names unknown cell {cname!r}")
            if not cpath:
                raise ValueError(f"--census for cell {cname!r} has an empty path")
            by_cell[cname].append(cpath)
    return by_cell


def parse_kernel(spec):
    fields = dict(kv.split("=", 1) for kv in spec.split(","))
    num = lambda k: None if fields[k] == "n/a" else float(fields[k])
    return {"c": num("c"), "m": num("m"), "txn32": num("txn"), "txn64": num("txn64"),
            "txn128": num("txn128"), "unc": int(fields["unc"]), "wg": int(fields["wg"]),
            "thr": int(fields["thr"]), "occ": float(fields.get("occ", "1")),
            "stg": float(fields.get("stg", "0")),
            # gh-ocannl-2262: None on a listing from before the term, which --shared-rate refuses.
            "shl": float(fields["shl"]) if "shl" in fields else None}


def occupancy(wg, thr, cap, tpm):
    t = max(1, thr)
    by_threads = tpm // t
    by_bytes = by_threads if wg <= 0 else max(0, cap) // wg
    return min(1.0, max(1, min(by_threads, by_bytes)) * t / tpm)


def listing(path):
    """Label -> (rank_ms, position, admitted, kernels)."""
    ranks, kernels = {}, {}
    with open(path) as f:
        # bench_output's one reader of the rank-site lines (gh-ocannl-1767).
        lines = bench_output.rank_site_lines(f.read())
    for kind, fields, _ in lines:
        if kind == "explain" and fields.get("rank_ms", "n/a") != "n/a":
            ranks[fields["label"]] = (float(fields["rank_ms"]), len(ranks),
                                      fields.get("admitted") != "false")
        elif kind == "kernels" and "label" in fields:
            kernels[fields["label"]] = [parse_kernel(k)
                                        for k in fields.get("kernels", "").split(";") if k]
    return {l: (r, pos, adm, kernels.get(l)) for l, (r, pos, adm) in ranks.items()}


def rescore(kernels, rate, sector, occ=None, staged=None, shared=None):
    total = 0.0
    for k in kernels:
        c = k["c"]
        txn = k[f"txn{sector}"] or 0.0
        # The listing's compute leg is over its own occupancy: back to the unoccupied leg first.
        if c is not None:
            c = c * k["occ"]
        if c is not None and rate is not None:
            c = c + txn / rate * 1e3
        if c is not None and staged is not None:
            # gh-ocannl-1821: (rate, listed rate) -- the staged bytes' move between the two.
            r, listed = staged
            c = c + k["stg"] * (1.0 / r - 1.0 / listed) * 1e3
        if c is not None and shared is not None:
            # gh-ocannl-2262: (rate, listed rate), a rate of 0 meaning none.
            r, listed = shared
            inv = lambda x: 0.0 if not x else 1.0 / x
            c = c + (k["shl"] or 0.0) * (inv(r) - inv(listed)) * 1e3
        if c is not None:
            o = k["occ"] if occ is None else occupancy(k["wg"], k["thr"], *occ)
            c = c / o
        legs = [x for x in (c, k["m"]) if x is not None]
        total += max(legs) if legs else 0.0
    return total


def load_cell(spec, census=()):
    name, rest = spec.split("=", 1)
    tsv, explain = rest.split(",", 1)
    # gh-ocannl-1821: an optional third listing, the cell's `nostg` reference (--staged-rate).
    explain, _, reference = explain.partition(",")
    stock = stock_replicates(tsv, census)
    if not stock.times:
        raise ValueError(f"{tsv}: no calibration rows")
    cands = {}
    bad = []
    for label, (rank, pos, adm, kernels) in listing(explain).items():
        resolved = row_replicates(label, stock, name)
        if resolved is None or kernels is None:
            continue
        rep_times, timed_as = resolved
        offline = rescore(kernels, None, 32)
        # The kernel legs print at 1e-6 ms; their sum may round apart by a few units.
        if abs(offline - rank) > 1e-5 * max(1, len(kernels)) + 1e-9 * rank:
            bad.append((label, rank, offline))
        cands[label] = {"times": rep_times, "median": statistics.median(rep_times), "pos": pos,
                        "admitted": adm, "kernels": kernels,
                        "uncovered": sum(k["unc"] for k in kernels), "timed_as": timed_as}
    return name, cands, bad, (tsv, reference) if reference else None


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


def evaluate(cands, rate, sector, band, occ=None, staged=None, shared=None):
    ranks = {l: rescore(c["kernels"], rate, sector, occ, staged, shared) for l, c in cands.items()}
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


def print_pairs(cells, sector, scoring):
    """gh-ocannl-2262: the register-tile pairs at one block tile, flavour and site."""
    import re
    lab = re.compile(r"gpu (\d+)x(\d+)x(\d+)/(\d)x(\d)((?: [a-z-]+)*)\]")
    for name, cands, _, _ in cells:
        by = {}
        for label, c in cands.items():
            mm = lab.search(label)
            if not mm:
                continue
            bm, bn, _, tm, tn, flavour = mm.groups()
            by.setdefault((int(bm), int(bn), flavour.strip()), {}).setdefault(
                (int(tm), int(tn)), []).append(label)
        for key, g in sorted(by.items()):
            for a in g.get((2, 2), []):
                for b in g.get((4, 4), []):
                    ca, cb = cands[a], cands[b]
                    thr = lambda c: ",".join(str(k["thr"]) for k in c["kernels"])
                    listed = rescore(ca["kernels"], None, sector) / rescore(cb["kernels"], None, sector)
                    line = (f"pair {name} {key[0]}x{key[1]} {key[2] or 'plain'}: {a!r} / {b!r}"
                            f" threads={thr(ca)}/{thr(cb)}"
                            f" measured={ca['median'] / cb['median']:.3f} listed={listed:.3f}")
                    if any(v is not None for v in scoring.values()):
                        ra = rescore(ca["kernels"], scoring["rate"], sector, scoring["occ"],
                                     scoring["staged"], scoring["shared"])
                        rb = rescore(cb["kernels"], scoring["rate"], sector, scoring["occ"],
                                     scoring["staged"], scoring["shared"])
                        line += f" rescored={ra / rb:.3f}"
                    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--sector", type=int, default=32, choices=(32, 64, 128))
    ap.add_argument("--rate", type=lambda v: 0.0 if v == "none" else float(v))
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--occupancy")
    ap.add_argument("--no-coalesced", dest="no_coalesced", action="store_true")
    ap.add_argument("--staged-rate", dest="staged_rate")
    ap.add_argument("--listed-staged-rate", dest="listed_staged_rate", type=float)
    ap.add_argument("--shared-rate", dest="shared_rate")
    ap.add_argument("--listed-shared-rate", dest="listed_shared_rate", type=float, default=0.0)
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--census", action="append", default=[], metavar="[CELL=]TIMING_STDERR",
                    help="stock timing cell's stderr for preset attribution (repeatable)")
    args = ap.parse_args()
    if args.staged_rate is not None and not args.listed_staged_rate:
        print("gh1600_txn_fit: --staged-rate needs --listed-staged-rate (the rate the listing was"
              " scored at)", file=sys.stderr)
        return 2
    try:
        census_by_cell = parse_census_args(args.cells, args.census)
    except ValueError as e:
        print(f"gh1600_txn_fit: {e}", file=sys.stderr)
        return 2
    occ = None
    if args.occupancy:
        cap, tpm = args.occupancy.split(",")
        occ = (int(cap), int(tpm))
    cells = []
    summaries = []
    for spec in args.cells:
        cell_census = census_by_cell[spec.split("=", 1)[0]]
        try:
            name, cands, bad, reference = load_cell(spec, census=cell_census)
        except (OSError, ValueError) as e:
            print(f"gh1600_txn_fit: {e}", file=sys.stderr)
            return 2
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
        if args.shared_rate is not None and any(
                k["shl"] is None for c in cands.values() for k in c["kernels"]):
            # A listing without shl= would re-score to itself at every rate and pass vacuously.
            print(f"gh1600_txn_fit: {name}: --shared-rate needs kernel lines carrying shl="
                  " (a listing from before gh-ocannl-2262 has none)", file=sys.stderr)
            return 2
        spread = spread_of(cands)
        unc = sum(c["uncovered"] for c in cands.values())
        base = evaluate(cands, None, args.sector, args.band)  # the listing as it was scored
        if args.shared_rate is not None and args.staged_rate is None:
            # gh-ocannl-2262: judged against the listing without the register-reuse term.
            base = evaluate(cands, None, args.sector, args.band,
                            shared=(0.0, args.listed_shared_rate))
        if args.staged_rate is not None:
            # gh-ocannl-1821: the comparison is against the cell's fill-only reference, its `nostg`
            # listing, scored as listed -- the staged term cannot be subtracted back out of this
            # one, since the rate also switched the launch tail on.
            if reference is None:
                print(f"gh1600_txn_fit: {name}: --staged-rate needs the cell's nostg listing as"
                      " NAME=TSV,LISTING,NOSTG_LISTING", file=sys.stderr)
                return 2
            try:
                _, ref_cands, ref_bad, _ = load_cell(
                    f"{name}={reference[0]},{reference[1]}", census=cell_census)
            except (OSError, ValueError) as e:
                print(f"gh1600_txn_fit: {e}", file=sys.stderr)
                return 2
            # The same population as the re-scored listing's.
            if args.no_coalesced:
                ref_cands = {l: c for l, c in ref_cands.items() if "coal-" not in l}
            if ref_bad:
                print(f"gh1600_txn_fit: {name}: the nostg listing does not rebuild from its kernel"
                      " legs", file=sys.stderr)
                return 2
            base = evaluate(ref_cands, None, args.sector, args.band)
        summaries.append(
            f"{name}: joined={len(cands)} spread={spread:.3f} uncovered_accesses={unc}"
            f" off_agreement={base[0]:.1f}% decided={base[1]}"
            + "".join(f" off_{s}={p!r}@{sl:.3f}" for s, (p, sl, _) in base[2].items()))
        cells.append((name, cands, spread, base))
    for line in summaries:
        print(line)

    def judge(rate, staged=None, shared=None):
        rows = []
        for name, cands, spread, base in cells:
            agree, _, sets, ranks = evaluate(cands, rate, args.sector, args.band, occ, staged,
                                             shared)
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

    if args.pairs:
        # The same re-scoring the fit evaluates: every option stated as one value.
        one = lambda v: v is not None and v != "sweep"
        print_pairs(cells, args.sector, dict(
            rate=args.rate or None, occ=occ,
            staged=((float(args.staged_rate), args.listed_staged_rate)
                    if one(args.staged_rate) else None),
            shared=((float(args.shared_rate), args.listed_shared_rate)
                    if one(args.shared_rate) else None)))

    if args.shared_rate is not None:
        listed = args.listed_shared_rate
        rates = ([10 ** (e / 10) for e in range(110, 151)] if args.shared_rate == "sweep"
                 else [float(args.shared_rate)])
        if args.staged_rate is None:
            stageds = [None]
        else:
            stageds = ([10 ** (e / 10) for e in range(110, 131)] if args.staged_rate == "sweep"
                       else [float(args.staged_rate)])
        scored = []
        for st in stageds:
            for r in rates:
                rows = judge(args.rate or None,
                             None if st is None else (st, args.listed_staged_rate), (r, listed))
                passing = sum(row[5] for row in rows)
                worst = max(row[3] for row in rows)
                least = min(row[1] for row in rows)
                scored.append(((-passing, worst, -least), r, st))
                print(f"shared_rate={r:.3e}"
                      + ("" if st is None else f" staged_rate={st:.3e}")
                      + f" passing={passing}/{len(rows)} worst_slowdown={worst:.3f}"
                      f" least_agreement={least:.1f}%"
                      + "".join(f" {row[0]}={row[1]:.1f}%@{row[3]:.3f}:{row[2]!r}" for row in rows))
        (key, r, st) = min(scored)
        print(f"fit: model_shared_load_rate={r:.6e}"
              + ("" if st is None else f" model_staged_copy_rate={st:.6e}")
              + f" passing={-key[0]}/{len(cells)} worst_slowdown={key[1]:.3f}"
              f" least_agreement={-key[2]:.1f}%")
        return 0 if len(rates) * len(stageds) > 1 or key[0] == -len(cells) else 1
    if args.staged_rate is not None:
        listed = args.listed_staged_rate
        grid = ([10 ** (e / 10) for e in range(100, 141)] if args.staged_rate == "sweep"
                else [float(args.staged_rate)])
        scored = []
        for r in grid:
            rows = judge(args.rate or None, (r, listed))
            passing = sum(row[5] for row in rows)
            worst = max(row[3] for row in rows)
            least = min(row[1] for row in rows)
            scored.append(((-passing, worst, -least), r))
            print(f"staged_rate={r:.3e} passing={passing}/{len(rows)} worst_slowdown={worst:.3f}"
                  f" least_agreement={least:.1f}%"
                  + "".join(f" {row[0]}={row[1]:.1f}%@{row[3]:.3f}:{row[2]!r}" for row in rows))
        (key, r) = min(scored)
        print(f"fit: model_staged_copy_rate={r:.6e} passing={-key[0]}/{len(cells)}"
              f" worst_slowdown={key[1]:.3f} least_agreement={-key[2]:.1f}%")
        return 0 if len(grid) > 1 or key[0] == -len(cells) else 1
    if args.rate is not None:
        rows = judge(args.rate or None)
        for name, agree, pick, slow, best, ok, sets, ranks, cands in rows:
            print(f"{name}: rate={args.rate or 0:.3e} agreement={agree:.1f}% pass={str(ok).lower()}"
                  + "".join(f" {s}={p!r}@{sl:.3f} fastest={b!r}" for s, (p, sl, b) in sets.items()))
            if args.show:
                for l in sorted(cands, key=lambda l: cands[l]["median"]):
                    c = cands[l]
                    timed_as = f" timed_as={c['timed_as']!r}" if c["timed_as"] is not None else ""
                    print(f"  {name}: {l!r} median_ms={c['median']:.6f} rank_ms={ranks[l]:.6f}"
                          f" admitted={str(c['admitted']).lower()} uncovered={c['uncovered']}{timed_as}")
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
