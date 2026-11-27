#!/usr/bin/env python3
"""gh-ocannl-1821: the residual analysis behind the staged-copy rate, on listings whose kernel lines
carry no stg= (gh1253_cells.sh outputs from before the term existed).

Usage: gh1821_geometry_fit.py OUT [--arm ARM] [--multiprocessors P] [--warp W] [--band B]
                              [--copy-rate R|sweep] [--regress]

  OUT   a gh1253_cells.sh output directory: per cell, rank_<site>_<dims>.tsv (and
        rank_<site>_<dims>.tsv.stock when profile timing cells appended to the main TSV),
        rank_<site>_<dims>_<ARM>.out (bench_rank_site.exe's rank-site lines), and any stock timing
        censuses (`<cell>_time.err` and `<cell>_time_r<N>.err`; profile logs such as
        `<cell>_time_<profile>.err` are never read). Reads `.tsv.stock` when present so profile
        timings never enter the stock fit; refuses a marked mixed TSV
        (`# gh1253_cells: stock snapshot required`) whose `.tsv.stock` is missing (exit 2,
        `stock_snapshot_missing` on stderr, no stdout). Archive provenance comes solely from this
        marker. From the stock timing censuses, reads only `autotune-seeding: plan` and
        `autotune-seeding: preset-digest` records via `gh1442_agreement.stock_attribution`, joins
        configured preset incumbents (`default` and `F_preset[...]` `seeded`) by `preset-digest`
        against positive 12-column stock rows (replicates kept independent of same-label rows at
        other digests) and census-selected `F_seeded[...]` by exact label, and refuses conflicting
        censuses or missing attributed incumbents (`preset_unmatched` / `seeded_plan_unmatched`,
        exit 2, no stdout) as well as any cell where fewer than two candidates join (exit 2, no
        stdout). Archives without census logs keep the legacy `baseline` / unique `F_seeded[...]`
        lookup.
  --arm the listing to re-score (default `shipped`): the fill-only reference F, i.e. scored with no
        staged-copy term and a continuous fill (`shipped` before gh-ocannl-1821 on CUDA; `noocc`
        on HIP, whose shipped listing carries an occupancy factor).
  --multiprocessors  P of the launch tail the copy rate switches on (rog-nv's RTX 5070 Ti: 46).
  --copy-rate  re-score at this staged-copy rate in f32 cells/s (bytes/s over 4), or `sweep` a
        log grid; the launch tail is applied with it.
  --regress  the within-cell regression of log median time on the staged leaves' geometry.

What it reads and does not: a leaf's geometry from its label, `gpu BMxBNxBK/RMxRN [bgrid]
[coal-k|coal-n]`, and the cell's M, N, K from its name (wgrad b,s,o,i: M=o N=i K=b*s; qkv
b,s,h,e,k: M=b*s N=h*e K=k; outproj b,s,j,h,e: M=b*s N=j K=h*e). The copy volume is the block-tile
arithmetic M*N*K*(BM+BN)/(BM*BN) cells, which test/operations/cost_model_staged_copy pins equal to
Ir.Cost_model.staged_copies on the lowered out projection's every staged leaf. The seeded plan
borrows the geometry of the leaf whose main kernel matches its workgroup bytes, threads and
compute leg; a candidate without a geometry (a preset) is scored as listed. Only each listing's
main kernel (the largest compute leg) takes the term, over that kernel's listed fill (the explain
line's fills=), and the launch tail only at fill 1 where the leaf's blocks are its tiles: bgrid
leaves, the seeded plan through the leaf it borrows from, and every leaf of the batch-free weight
gradient (a plain qkv/outproj leaf's batch loops are serial inside its blocks). It
does not model the occupancy factor, the transaction term, or tile-MMA leaves (none of these
cells has them priced).

Per cell it prints the agreement (decided pairs, measured more than the band apart, ordered as
measured; a tie orders none), and the admitted+coalesced and admitted-only argmins (ties to the
listing order, model_default's rule; `none` when a pick set is empty) with their median slowdown
against the fastest member, and the measured spread (the fastest admitted candidate's p90/p10 over
its replicates, 1.0 under 3 or when no candidate is admitted): #1600's criterion passes where the
admitted+coalesced argmin is absent, not coalesced, or within the spread, at an agreement no lower
than F's.
"""

import argparse
import glob
import itertools
import math
import os
import re
import statistics
import sys

import bench_output
from gh1600_txn_fit import row_replicates, stock_replicates

GEO = re.compile(r"gpu (\d+)x(\d+)x(\d+)/(\d+)x(\d+)( bgrid)?( coal-[kn])?")


def stock_tsv_path(tsv):
    """Prefer the archived .tsv.stock snapshot when present; otherwise read tsv, which
    stock_replicates() refuses if marked as mixed."""
    stock = f"{tsv}.stock"
    return stock if os.path.isfile(stock) else tsv


def stock_census_paths(out, cell):
    """Stock timing stderr logs in out (`<cell>_time.err` and `<cell>_time_r<N>.err`),
    excluding profile timing logs such as `<cell>_time_<profile>.err`."""
    pat = re.compile(rf"^{re.escape(cell)}_time(?:_r([1-9][0-9]*))?\.err$")
    found = []
    for entry in os.listdir(out):
        m = pat.match(entry)
        if m:
            path = os.path.join(out, entry)
            if os.path.isfile(path):
                found.append((0 if m.group(1) is None else int(m.group(1)), path))
    return [path for _, path in sorted(found)]


def dims(cell):
    _, site, d = cell.split("_", 2)
    v = [int(x) for x in d.split("x")]
    if site == "wgrad":
        b, s, o, i = v
        return o, i, b * s
    if site == "qkv":
        b, s, h, e, k = v
        return b * s, h * e, k
    if site == "outproj":
        b, s, j, h, e = v
        return b * s, j, h * e
    raise ValueError(cell)


def load(out, arm):
    cells = []
    for tsv in sorted(glob.glob(os.path.join(out, "rank_*.tsv"))):
        cell = os.path.basename(tsv)[:-4]
        listing = os.path.join(out, f"{cell}_{arm}.out")
        if not os.path.exists(listing):
            continue
        m, n, k = dims(cell)
        src = stock_tsv_path(tsv)
        stock = stock_replicates(src, stock_census_paths(out, cell))
        if not stock.times:
            raise ValueError(f"{src}: no calibration rows")
        explain, kernels = {}, {}
        with open(listing) as f:
            # bench_output's one reader of the rank-site lines (gh-ocannl-1767).
            lines = bench_output.rank_site_lines(f.read())
        for kind, fields, _ in lines:
            if kind == "explain" and fields.get("rank_ms", "n/a") != "n/a":
                fills = [float(x) for x in fields.get("fills", "").split(",") if x not in ("", "n/a")]
                explain[fields["label"]] = (float(fields["rank_ms"]),
                                            fields.get("admitted") != "false", fills)
            elif kind == "kernels" and "label" in fields:
                kernels[fields["label"]] = [dict(kv.split("=", 1) for kv in spec.split(","))
                                            for spec in fields.get("kernels", "").split(";") if spec]
        rows = []
        for pos, label in enumerate(explain):
            resolved = row_replicates(label, stock, cell)
            if resolved is None or label not in kernels:
                continue
            rep_times, _ = resolved
            fills = explain[label][2]
            ks = [{"c": float(x["c"]), "m": float(x["m"]), "wg": int(x["wg"]), "thr": int(x["thr"]),
                   "fill": fills[i] if i < len(fills) else 1.0}
                  for i, x in enumerate(kernels[label])]
            g = GEO.search(label)
            rows.append({"label": label, "pos": pos, "times": rep_times,
                         "med": statistics.median(rep_times), "rank": explain[label][0],
                         "adm": explain[label][1], "kernels": ks,
                         "geo": tuple(map(int, g.groups()[:5])) if g else None,
                         "bgrid": bool(g and g.group(6)), "coal": bool(g and g.group(7))})
        if len(rows) < 2:
            raise ValueError(f"{cell}: fewer than two candidates join")
        # The seeded plan borrows the geometry of a leaf whose main kernel matches its own.
        for r in rows:
            if r["geo"] is None and r["kernels"]:
                main = max(r["kernels"], key=lambda q: q["c"])
                for q in rows:
                    if q["geo"] and q["kernels"]:
                        qm = max(q["kernels"], key=lambda x: x["c"])
                        if (qm["wg"], qm["thr"]) == (main["wg"], main["thr"]) and abs(
                                qm["c"] - main["c"]) < 1e-6:
                            r["geo"], r["bgrid"] = q["geo"], q["bgrid"]
                            break
        # The weight gradient has no batch loops: a plain leaf's blocks are its tiles too.
        for r in rows:
            r["tiled"] = r["bgrid"] or cell.split("_")[1] == "wgrad"
        cells.append((cell, (m, n, k), rows))
    return cells


def score(r, mnk, copy_rate, p):
    """The candidate's rank: the sum over kernels of max(c, m), the main kernel's c taking the copy
    term (ms, over the kernel's listed fill, as the runtime divides each copy context's bytes) and,
    for a saturated leaf whose blocks are its tiles (a bgrid one, the seeded plan's borrowed
    geometry, or any leaf of the batch-free weight gradient), the launch tail where a rate is given.
    A plain leaf's batch loops are serial inside its blocks, so its block count is not the cell's
    M*N/(BM*BN); it takes no tail."""
    m, n, k = mnk
    total = 0.0
    main = max(range(len(r["kernels"])), key=lambda i: r["kernels"][i]["c"]) if r["kernels"] else -1
    for i, kern in enumerate(r["kernels"]):
        c = kern["c"]
        if i == main and r["geo"] and copy_rate:
            bm, bn = r["geo"][0], r["geo"][1]
            cells = m * n * k * (bm + bn) / (bm * bn)
            c += cells / copy_rate * 1e3 / max(kern["fill"], 1e-9)
            blocks = m * n / (bm * bn)
            if p and r["tiled"] and kern["fill"] >= 1.0 and blocks > p:
                c *= math.ceil(blocks / p) / (blocks / p)
        total += max(c, kern["m"])
    return total


def spread(ts):
    if len(ts) < 3:
        return 1.0
    q = statistics.quantiles(sorted(ts), n=10, method="inclusive")
    return q[8] / q[0]


def evaluate(cells, copy_rate, p, band):
    out = []
    for cell, mnk, rows in cells:
        sc = {r["label"]: score(r, mnk, copy_rate, p) for r in rows}
        decided = right = 0
        for a, b in itertools.combinations(rows, 2):
            if max(a["med"], b["med"]) <= (1 + band) * min(a["med"], b["med"]):
                continue
            decided += 1
            if sc[a["label"]] != sc[b["label"]] and (sc[a["label"]] < sc[b["label"]]) == (
                    a["med"] < b["med"]):
                right += 1
        sets = {}
        for name, pred in (("admitted", lambda r: r["adm"] and not r["coal"]),
                           ("admitted+coalesced", lambda r: r["adm"])):
            members = [r for r in rows if pred(r)]
            if not members:
                continue
            pick = min(members, key=lambda r: (sc[r["label"]], r["pos"]))
            best = min(members, key=lambda r: r["med"])
            sets[name] = (pick, pick["med"] / best["med"], best)
        admitted_best = sets["admitted+coalesced"][2] if "admitted+coalesced" in sets else None
        spr = spread(admitted_best["times"]) if admitted_best is not None else 1.0
        out.append((cell, 100.0 * right / max(1, decided), sets, spr))
    return out


def report(tag, res, base=None):
    passing = 0
    for i, (cell, agr, sets, spr) in enumerate(res):
        pick, slow, _ = sets.get("admitted+coalesced", (None, 1.0, None))
        ok = ((pick is None or not pick["coal"] or slow <= spr)
              and (base is None or agr >= base[i][1] - 1e-9))
        passing += ok
        argmin = f"{pick['label']!r}@{slow:.3f}" if pick is not None else "none"
        admitted_argmin = (f"admitted_argmin@{sets['admitted'][1]:.3f}"
                           if "admitted" in sets else "admitted_argmin=none")
        print(f"{tag} {cell}: agreement={agr:.1f}% spread={spr:.3f}"
              f" argmin={argmin} {admitted_argmin} pass={str(ok).lower()}")
    return passing


def solve(a, b):
    """Least squares by the normal equations (small, well-conditioned systems)."""
    n = len(a[0])
    ata = [[sum(r[i] * r[j] for r in a) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * y for r, y in zip(a, b)) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(ata[r][col]))
        ata[col], ata[piv] = ata[piv], ata[col]
        atb[col], atb[piv] = atb[piv], atb[col]
        if abs(ata[col][col]) < 1e-15:
            continue
        for r in range(n):
            if r != col:
                f = ata[r][col] / ata[col][col]
                ata[r] = [x - f * y for x, y in zip(ata[r], ata[col])]
                atb[r] -= f * atb[col]
    return [atb[i] / ata[i][i] if abs(ata[i][i]) > 1e-15 else 0.0 for i in range(n)]


def regress(cells, warp):
    pop = [(cell, mnk, r) for cell, mnk, rows in cells for r in rows
           if r["geo"] and r["bgrid"] and "bgrid-in" not in r["label"]]
    names = sorted({c for c, _, _ in pop})

    def feats(mnk, r):
        bm, bn, bk, rm, rn = r["geo"]
        threads = (bm // rm) * (bn // rn)
        return {"log_ksteps": math.log2(mnk[2] / bk),
                "copy_cells_per_fma": (bm + bn) / (bm * bn),
                "register_tile": (rm + rn) / (rm * rn),
                "subwarp": math.log2(max(1, warp / threads))}

    y = [math.log(r["med"]) for _, _, r in pop]

    def fit(fs):
        a = [[1.0 if c == name else 0.0 for name in names] + [feats(mnk, r)[f] for f in fs]
             for c, mnk, r in pop]
        beta = solve(a, y)
        pred = [sum(x * b for x, b in zip(row, beta)) for row in a]
        return sum((u - v) ** 2 for u, v in zip(y, pred)), beta[len(names):]

    ss0, _ = fit([])
    print(f"regression: n={len(pop)} staged bgrid leaves over {len(names)} cells")
    for fs in (["log_ksteps"], ["copy_cells_per_fma"], ["subwarp"], ["register_tile"],
               ["copy_cells_per_fma", "subwarp", "register_tile"],
               ["copy_cells_per_fma", "subwarp", "register_tile", "log_ksteps"]):
        ss, beta = fit(fs)
        print(f"  {'+'.join(fs)}: R2(within-cell)={1 - ss / ss0:.3f} coefficients="
              + ",".join(f"{f}={b:.3f}" for f, b in zip(fs, beta)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arm", default="shipped")
    ap.add_argument("--multiprocessors", type=int, default=0)
    ap.add_argument("--warp", type=int, default=32)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--copy-rate", dest="copy_rate")
    ap.add_argument("--regress", action="store_true")
    args = ap.parse_args()
    try:
        cells = load(args.out, args.arm)
    except (OSError, ValueError) as e:
        print(f"gh1821_geometry_fit: {e}", file=sys.stderr)
        return 2
    if not cells:
        print(f"gh1821_geometry_fit: no cell with a {args.arm} listing in {args.out}",
              file=sys.stderr)
        return 2
    if args.regress:
        regress(cells, args.warp)
    base = evaluate(cells, None, 0, args.band)
    report("F", base)
    if args.copy_rate:
        grid = ([10 ** (e / 10) for e in range(100, 131)] if args.copy_rate == "sweep"
                else [float(args.copy_rate)])
        for rate in grid:
            res = evaluate(cells, rate, args.multiprocessors, args.band)
            n = report(f"cells/s={rate:.3e}", res, base)
            print(f"cells/s={rate:.3e} (bytes/s={4 * rate:.3e}): passing={n}/{len(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
