#!/usr/bin/env python3
"""Per-site attribution of the cifar_resnet cells (gh-ocannl-1161), from a sweep's cell logs.

`orchestrate.py` with `BENCH_KERNEL_TABLE=1` and `BENCH_CELL_LOG_DIR=<dir>` leaves, per OCANNL
cell, the kernel table of the step as shipped on stderr (`bench: kernel i/n <ms> ms ... w: <written
nodes>`) and the cell's result line on stdout. This reads every such log and prints, per cell,
the step p50 beside the sum of the kernel minima attributed to each site of the reduced ResNet.

A kernel belongs to every site whose names it writes (`Bench_conv_model.resnet`'s names):
  conv site X (stem, s<i>c1, s<i>c2, s<i>sc)   kernel_X, bias_X (their .grad), relu_Xout (.grad)
  residual site s<i>res                         relu_s<i>out (.grad), s<i>res (.grad)
so a fused kernel counts toward each site it serves: the columns attribute, they do not
partition, and they do not sum to the step. A virtual node writes nothing and names nothing, and
a conv output is an unnamed node: a kernel writing only those is `unattributed`. The families are
the cell names the dependent issues cite:
  pad3x3    the padded 3x3 stride-1 convs (stem, s<i>c1, s<i>c2)
  resadd    the residual adds (s<i>res)
  sc1x1s2   the stride-2 1x1 shortcut convs (s<i>sc)
Each kernel minimum includes a launch and device sync, and a kernel that could not be compiled on
its own (DECLINED) has no time: the `timed` column says how many were.

Usage: gh1161_sites.py <cell-log-dir> [--per-site]
"""

import argparse
import re
import sys
from pathlib import Path

from strict_json import loads

KERNEL = re.compile(r"^bench: kernel (\d+)/(\d+) (?:([0-9.]+) ms .*? w: (.*)|.*DECLINED.*)$")
PARAM = re.compile(r"^(?:kernel|bias)_(stem|s\d+(?:c[12]|sc))(?:\.grad)?$")
RELU = re.compile(r"^relu_(stem|s\d+(?:c1)?)out(?:\.grad)?$")
RES = re.compile(r"^(s\d+res)(?:\.grad)?$")
FAMILIES = (
    ("pad3x3", re.compile(r"^(stem|s\d+c[12])$")),
    ("resadd", re.compile(r"^s\d+res$")),
    ("sc1x1s2", re.compile(r"^s\d+sc$")),
)


def site_of(name):
    """The site a written node names, or None."""
    for pattern in (PARAM, RES):
        m = pattern.match(name)
        if m:
            return m.group(1)
    m = RELU.match(name)
    if m:
        site = m.group(1)
        # relu_s<i>out follows the residual add; relu_stemout / relu_s<i>c1out follow a conv.
        return f"{site}res" if re.fullmatch(r"s\d+", site) else site
    return None


def family_of(site):
    return next(name for name, pattern in FAMILIES if pattern.match(site))


def read_cell(path):
    """(result line, kernel rows) of one cell log, or None if it is not an OCANNL resnet cell."""
    result, kernels = None, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = KERNEL.match(line)
        if m:
            # A DECLINED row's detail runs over several lines before its `w:`: no time, no names.
            ms = float(m.group(3)) if m.group(3) else None
            kernels.append((int(m.group(1)), ms, (m.group(4) or "").split()))
        elif line.startswith('{"framework":"ocannl"'):
            try:
                result = loads(line)
            except ValueError:
                pass
    if result is None or not str(result.get("workload", "")).startswith("cifar_resnet"):
        return None
    return result, kernels


def attribute(kernels):
    """Per site and per family, the summed minima of the kernels naming it: a kernel counts once
    toward each site it names and once toward each family those sites belong to."""
    sites, families, unattributed = {}, {}, 0.0
    for _, ms, written in kernels:
        if ms is None:
            continue
        named = {s for s in map(site_of, written) if s}
        if not named:
            unattributed += ms
        for site in named:
            sites[site] = sites.get(site, 0.0) + ms
        for family in {family_of(s) for s in named}:
            families[family] = families.get(family, 0.0) + ms
    return sites, families, unattributed


def label(result):
    regime = result.get("profile") or "exact"
    return f"{result['workload']}/{result['backend']}/{result['variant']}/{regime}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", type=Path, help="the sweep's BENCH_CELL_LOG_DIR")
    ap.add_argument("--per-site", action="store_true", help="add one column per site")
    args = ap.parse_args(argv)
    cells = [c for c in map(read_cell, sorted(args.logs.glob("*.log"))) if c]
    if not cells:
        print(f"gh1161_sites: no OCANNL cifar_resnet cell log in {args.logs}", file=sys.stderr)
        return 1
    rows, all_sites = [], set()
    for result, kernels in cells:
        sites, families, unattributed = attribute(kernels)
        all_sites |= set(sites)
        rows.append((label(result), result, kernels, sites, families, unattributed))
    per_site = sorted(all_sites, key=lambda s: (s != "stem", s)) if args.per_site else []
    head = ["cell", "step p50 ms", "kernels timed", "sum of minima ms",
            *(name for name, _ in FAMILIES), "unattributed", *per_site]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, result, kernels, sites, families, unattributed in sorted(rows):
        timed = [ms for _, ms, _ in kernels if ms is not None]
        p50 = (result.get("step_ms") or {}).get("p50")
        cols = [
            name,
            f"{p50:.3f}" if isinstance(p50, (int, float)) else "n/a",
            f"{len(timed)}/{len(kernels)}" if kernels else "no table",
            f"{sum(timed):.3f}" if timed else "n/a",
            *(f"{families.get(f, 0.0):.3f}" for f, _ in FAMILIES),
            f"{unattributed:.3f}",
            *(f"{sites.get(s, 0.0):.3f}" for s in per_site),
        ]
        print("| " + " | ".join(cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
