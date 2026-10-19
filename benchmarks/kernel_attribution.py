#!/usr/bin/env python3
"""Per-kernel attribution of a step's time from the step itself (gh-ocannl-1347), stdlib only.

A `kprof` cell of benchmarks/gh1133_cells.sh is one bench_gpt process run under
`nsys profile --trace=cuda --cuda-graph-trace=node`, with the per-kernel table on
(`BENCH_KERNEL_TABLE=1`). Its trace holds both instruments, from the same compile:

  in-step   every shipped segment's device duration inside the timed steps, as the step ran it:
            `<routine>__seg<j>` nodes of the replayed CUDA graph (one instance per step);
  isolated  the same segment recompiled as a routine of its own and run 21 times back to back
            after the timed steps: `<routine>__kernel<i>`, i being the kernel table's index (the
            table's min-of-20 is a host wall time with a launch + sync in it; the trace gives the
            device time of the same runs).

Only the in-step durations attribute step time: their per-step sum is the step's device time.
The isolated minimum is a statement about the kernel alone (gh-ocannl-1347: within 1% between
processes at a fixed revision, yet 0.6x-2.3x off its in-step duration for a sixth of rog's b256
step, in every repeat, with identical code and launch geometry). Each row says which it is:

  ISO      the isolated device minimum is more than ISO_TOLERANCE away from the in-step median (and
           by at least ABS_FLOOR_MS): an isolated-time comparison of this kernel says nothing about
           the step.
  SPREAD   the in-step p90/p10 over the timed steps exceeds SPREAD_TOLERANCE: a difference of this
           kernel smaller than its own spread is not a difference.
  OWN      the isolated minimum times the cell's common factor is still more than ISO_TOLERANCE
           (and ABS_FLOOR_MS) away from the in-step median: the kernel's own deviation, not the
           clock's (gh-ocannl-1391).

The common factor F is the median in-step/isolated ratio over the kernels of at least
FACTOR_FLOOR_MS in-step. It is a property of the step, not of a kernel: on rog's power-capped laptop
GPU a b256 step runs every sizeable kernel ~6.5% slower than its isolated minimum, and the per-step
slowdowns of kernels adjacent in time move together (gh-ocannl-1391), while a 21-run isolated burst
draws too little for long enough to be capped. ISO therefore flags most of a capped step; OWN flags
what remains once the step's clock is accounted for.

Subcommands:
  check OUT NAME   validate one cell (OUT/NAME.out, .err, _cuda_gpu_trace.csv); exit 1 with a reason
  report OUT       the attribution table of every kprof cell in OUT, then the cross-cell
                   repeatability of cells that shipped the same kernel list (an A/A when the arms
                   are one binary); exit 1 when a cell is refused
"""
import csv
import os
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strict_json  # noqa: E402
from gh1002_cells import KERNEL  # noqa: E402  (the one reader of the kernel-table row)

ISO_TOLERANCE = 0.10
SPREAD_TOLERANCE = 1.10
ABS_FLOOR_MS = 0.02
# The common factor's population: kernels of at least this in-step median, at least this many.
FACTOR_FLOOR_MS = 0.5
FACTOR_MIN = 3
# Rows printed per cell: the largest kernels until they cover this share of the in-step total.
SHOWN_SHARE = 0.95
SHOWN_MAX = 60
# Cross-cell spread worth a row of its own in the repeatability table.
REPEAT_TOLERANCE = 1.05
# Written nodes a row names before eliding the rest (Bench_harness.label_writes).
LABEL_WRITES = 3

KPROF_CELL = re.compile(r"(\w+)-(gpt2_mini\w*)-(\w+)-(kprof\d+)\.out")
SEG = re.compile(r"^(.*)__seg(\d+)$")
ISOLATED = re.compile(r"^(.*)__kernel(\d+)$")


class Refused(Exception):
    """A cell whose attribution cannot be trusted; the message says why."""


def result_line(path):
    rec = None
    for line in open(path):
        line = line.strip()
        if line.startswith("{") and '"step_ms"' in line:
            rec = strict_json.loads(line)
    if rec is None:
        raise Refused("%s has no result line" % path.name)
    return rec


def table(path):
    """The kernel table's rows {i: (host_min_ms, geometry, writes)}, refusing an incomplete one:
    a declined kernel has no isolated run to compare and no row to label its in-step time."""
    rows, totals = {}, set()
    for line in open(path):
        m = KERNEL.match(line.rstrip("\n"))
        if m:
            totals.add(int(m.group(2)))
            rows[int(m.group(1))] = (float(m.group(3)), "grid=[%s] block=[%s]" % m.group(4, 5),
                                     m.group(7))
    if len(totals) != 1 or sorted(rows) != list(range(next(iter(totals)))):
        raise Refused("%s: the kernel table is missing or incomplete" % path.name)
    return rows


def trace(path):
    """{routine: {j: [in-step durations, in start order]}} and {routine: {i: [isolated durations]}},
    in ms, from an `nsys stats --report cuda_gpu_trace --format csv` export."""
    seg, iso = {}, {}
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["Start (ns)"]), r["Name"], float(r["Duration (ns)"]) / 1e6))
    for _, name, ms in sorted(rows):
        m = SEG.match(name)
        if m:
            seg.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append(ms)
            continue
        m = ISOLATED.match(name)
        if m:
            iso.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append(ms)
    return seg, iso


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]


def attribute(out, name):
    """One cell's rows, in kernel-table order, and its totals. A routine's in-step segment j is the
    table's kernel off+j, off being the first table index its isolated kernels carry (the table
    concatenates the step's routines in order); the routine of a parameter init, which no isolated
    kernel names, maps to nothing and is left out."""
    out = Path(out)
    rec = result_line(out / name)
    stem = name[:-4]
    rows = table(out / (stem + ".err"))
    csv_path = out / (stem + "_cuda_gpu_trace.csv")
    if not csv_path.is_file():
        raise Refused("%s has no trace export" % csv_path.name)
    seg, iso = trace(csv_path)
    window = 2 * int(rec["timed_steps"])  # the per-step-synced steps, then the queued ones
    mapped = {}
    for routine, kernels in iso.items():
        ids = sorted(kernels)
        off = ids[0]
        if ids != list(range(off, off + len(ids))) or sorted(seg.get(routine, {})) != list(range(len(ids))):
            raise Refused("%s: routine %s's in-step segments do not match its isolated kernels %d..%d"
                          % (name, routine, off, ids[-1]))
        for j in range(len(ids)):
            mapped[off + j] = (seg[routine][j], kernels[off + j])
    if sorted(mapped) != sorted(rows):
        raise Refused("%s: the trace maps %d of the table's %d kernels" % (name, len(mapped), len(rows)))
    attributed = []
    for i in sorted(rows):
        steps, runs = mapped[i]
        if len(steps) < window:
            raise Refused("%s: kernel %d ran %d times in the step, fewer than the %d timed steps"
                          % (name, i, len(steps), window))
        if len(runs) < 2:
            raise Refused("%s: kernel %d has no timed isolated run" % (name, i))
        steps, runs = steps[-window:], runs[1:]  # drop the isolated warmup run
        med = statistics.median(steps)
        p10, p90 = percentile(steps, 10), percentile(steps, 90)
        iso_min = min(runs)
        host_min, geometry, writes = rows[i]
        flags = []
        if abs(iso_min - med) >= ABS_FLOOR_MS and abs(iso_min / med - 1) > ISO_TOLERANCE:
            flags.append("ISO")
        if p10 > 0 and p90 / p10 > SPREAD_TOLERANCE:
            flags.append("SPREAD")
        attributed.append({"i": i, "med": med, "p10": p10, "p90": p90, "iso_min": iso_min,
                           "host_min": host_min, "geometry": geometry, "writes": writes,
                           "flags": flags, "steps": steps})
    factor = common_factor(attributed)
    if factor is not None:
        for r in attributed:
            scaled = r["iso_min"] * factor["f"]
            if abs(scaled - r["med"]) >= ABS_FLOOR_MS and abs(scaled / r["med"] - 1) > ISO_TOLERANCE:
                r["flags"].append("OWN")
    return rec, attributed


def common_factor(rows):
    """The cell's common in-step/isolated factor: {"f": the median ratio over the kernels of at
    least FACTOR_FLOOR_MS in-step, "n": how many, "p10"/"p90": the spread over the timed steps of
    each step's own median ratio}; None with fewer than FACTOR_MIN such kernels."""
    big = [r for r in rows if r["med"] >= FACTOR_FLOOR_MS and r["iso_min"] > 0]
    if len(big) < FACTOR_MIN:
        return None
    per_step = [statistics.median(r["steps"][s] / r["iso_min"] for r in big)
                for s in range(len(big[0]["steps"]))]
    return {"f": statistics.median(r["med"] / r["iso_min"] for r in big), "n": len(big),
            "p10": percentile(per_step, 10), "p90": percentile(per_step, 90)}


def clocks(out, stem):
    """The SM clock over the cell's nvidia-smi samples taken in P0, or None without a sampler."""
    path = Path(out) / (stem + ".smi.csv")
    if not path.is_file():
        return None
    mhz = []
    for r in list(csv.reader(open(path)))[1:]:
        if len(r) > 3 and r[3].strip() == "P0":
            try:
                mhz.append(int(r[1].split()[0]))
            except (ValueError, IndexError):
                pass
    return (min(mhz), statistics.median(mhz), max(mhz), len(mhz)) if mhz else None


def label(writes):
    names = writes.split()
    return " ".join(names[:LABEL_WRITES]) + (" +%d" % (len(names) - LABEL_WRITES)
                                             if len(names) > LABEL_WRITES else "")


def render_cell(out, name, rec, rows):
    stem = name[:-4]
    total = sum(r["med"] for r in rows)
    print("\n### Kernel attribution: %s\n" % stem)
    print("In-step: device time of each shipped segment over the %d timed steps (median, p10..p90)."
          % (2 * int(rec["timed_steps"])))
    print("Isolated: the same segment recompiled alone, device minimum of 20 runs (the kernel"
          " table's host minimum beside it). Only the in-step column attributes step time.\n")
    print("- step p50 %.2f ms; sum of in-step medians %.2f ms; sum of isolated minima %.2f ms"
          " (host table %.2f ms)" % (rec["step_ms"]["p50"], total, sum(r["iso_min"] for r in rows),
                                     sum(r["host_min"] for r in rows)))
    iso = [r for r in rows if "ISO" in r["flags"]]
    spread = [r for r in rows if "SPREAD" in r["flags"]]
    print("- ISO (isolated off in-step by >%d%%): %d kernels, %.2f ms in-step (%.1f%%)"
          % (round(ISO_TOLERANCE * 100), len(iso), sum(r["med"] for r in iso),
             100 * sum(r["med"] for r in iso) / total))
    print("- SPREAD (in-step p90/p10 > %.2f): %d kernels, %.2f ms in-step (%.1f%%)"
          % (SPREAD_TOLERANCE, len(spread), sum(r["med"] for r in spread),
             100 * sum(r["med"] for r in spread) / total))
    factor = common_factor(rows)
    if factor is None:
        print("- common in-step/isolated factor: none (fewer than %d kernels of >= %.1f ms in-step);"
              " OWN flags nothing" % (FACTOR_MIN, FACTOR_FLOOR_MS))
    else:
        own = [r for r in rows if "OWN" in r["flags"]]
        print("- common in-step/isolated factor F = %.3f (median over the %d kernels of >= %.1f ms"
              " in-step; per-step factor p10..p90 %.3f..%.3f)"
              % (factor["f"], factor["n"], FACTOR_FLOOR_MS, factor["p10"], factor["p90"]))
        print("- OWN (isolated x F off in-step by >%d%%): %d kernels, %.2f ms in-step (%.1f%%)"
              % (round(ISO_TOLERANCE * 100), len(own), sum(r["med"] for r in own),
                 100 * sum(r["med"] for r in own) / total))
    clk = clocks(out, stem)
    if clk:
        print("- SM clock in P0: %d..%d MHz, median %d (%d samples)" % (clk[0], clk[2], clk[1], clk[3]))
    print("\n| kernel | in-step median | p10..p90 | isolated min | iso / in-step | table min | flags | writes |")
    print("|---|---|---|---|---|---|---|---|")
    shown, covered = 0, 0.0
    for r in sorted(rows, key=lambda r: -r["med"]):
        if covered >= SHOWN_SHARE * total or shown >= SHOWN_MAX:
            break
        shown += 1
        covered += r["med"]
        print("| %d | %.3f | %.3f..%.3f | %.3f | %.2f | %.3f | %s | %s |" % (
            r["i"], r["med"], r["p10"], r["p90"], r["iso_min"], r["iso_min"] / r["med"],
            r["host_min"], " ".join(r["flags"]), label(r["writes"])))
    rest = sorted(rows, key=lambda r: -r["med"])[shown:]
    if rest:
        print("\n%d further kernels: %.2f ms in-step (%.1f%%), %d flagged ISO, %d SPREAD, %d OWN." % (
            len(rest), sum(r["med"] for r in rest), 100 * sum(r["med"] for r in rest) / total,
            sum("ISO" in r["flags"] for r in rest), sum("SPREAD" in r["flags"] for r in rest),
            sum("OWN" in r["flags"] for r in rest)))


def render_repeatability(cells):
    """Cells of one fixture that shipped the same kernels (writes and geometry, in order): how far
    apart each kernel's in-step median and isolated minimum land between them."""
    groups = {}
    for (backend, fixture, stem), rows in cells:
        signature = tuple((r["writes"], r["geometry"]) for r in rows)
        groups.setdefault((backend, fixture, signature), []).append((stem, rows))
    for (backend, fixture, _), members in sorted(groups.items(), key=lambda kv: kv[0][:2]):
        if len(members) < 2:
            continue
        stems = [s for s, _ in members]
        n = len(members[0][1])
        total = statistics.median(sum(r["med"] for r in rows) for _, rows in members)
        print("\n### Kernel repeatability: %s %s, %d cells of one kernel list (%s)\n"
              % (backend, fixture, len(members), ", ".join(stems)))

        def spread(key, i):
            xs = [rows[i][key] for _, rows in members]
            return max(xs) / min(xs) if min(xs) > 0 else float("inf"), xs
        large = [i for i in range(n) if members[0][1][i]["med"] >= 0.01 * total]
        for key, what in (("med", "in-step median"), ("iso_min", "isolated minimum")):
            worst = max(large, key=lambda i: spread(key, i)[0]) if large else None
            if worst is not None:
                print("- %s: largest max/min across cells %.3f (kernel %d, %s) among the %d kernels"
                      " of >= 1%% of the step" % (what, spread(key, worst)[0], worst,
                                                  label(members[0][1][worst]["writes"]), len(large)))
        sums = [sum(r["med"] for r in rows) for _, rows in members]
        print("- sum of in-step medians per cell: %s ms" % ", ".join("%.2f" % s for s in sums))
        rows_out = [i for i in range(n)
                    if members[0][1][i]["med"] >= 0.1
                    and max(spread("med", i)[0], spread("iso_min", i)[0]) > REPEAT_TOLERANCE]
        if rows_out:
            print("\n| kernel | in-step medians | isolated minima | writes |")
            print("|---|---|---|---|")
            for i in rows_out:
                print("| %d | %s | %s | %s |" % (
                    i, ", ".join("%.3f" % x for x in spread("med", i)[1]),
                    ", ".join("%.3f" % x for x in spread("iso_min", i)[1]),
                    label(members[0][1][i]["writes"])))
        else:
            print("- no kernel of >= 0.1 ms moved by more than %.0f%% between cells"
                  % ((REPEAT_TOLERANCE - 1) * 100))


def report(out):
    refused, cells = False, []
    for name in sorted(os.listdir(out)):
        m = KPROF_CELL.fullmatch(name)
        if not m:
            continue
        try:
            rec, rows = attribute(out, name)
        except (Refused, OSError, ValueError, KeyError) as e:
            print("REFUSED: %s: %s" % (name, e))
            refused = True
            continue
        render_cell(out, name, rec, rows)
        cells.append(((m.group(1), m.group(2), name[:-4]), rows))
    render_repeatability(cells)
    return 1 if refused or not cells else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "check":
        try:
            attribute(sys.argv[2], sys.argv[3])
        except (Refused, OSError, ValueError, KeyError) as e:
            print("REFUSED: %s: %s" % (sys.argv[3], e))
            sys.exit(1)
        sys.exit(0)
    if len(sys.argv) == 3 and sys.argv[1] == "report":
        sys.exit(report(sys.argv[2]))
    print(__doc__, file=sys.stderr)
    sys.exit(2)
