#!/usr/bin/env python3
"""Per-kernel attribution of a step's time from the step itself (gh-ocannl-1347), stdlib only.

A `kprof` cell of benchmarks/gh1133_cells.sh is one bench_gpt process run under a device tracer,
with the per-kernel table on (`BENCH_KERNEL_TABLE=1`): on CUDA `nsys profile --trace=cuda
--cuda-graph-trace=node`, exported as CELL_cuda_gpu_trace.csv; on HIP `rocprofv3 --kernel-trace`
(gh-ocannl-1392), whose CELL_kernel_trace.csv records every dispatch at the HSA queue, graph nodes
included. Its trace holds both instruments, from the same compile:

  in-step   every shipped segment's device duration inside the timed steps, as the step ran it:
            `<routine>__seg<j>` nodes of the replayed graph (one instance per step);
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

A HIP cell also states the memory model of the device it ran on -- unified (an integrated part
sharing host DRAM, minix's gfx1151) or discrete (tuf's gfx1102) -- on a `kprof: memory model` line
of its .err, which the driver writes from the backend's own device query; a HIP cell without one,
or whose kernels ran on more than one agent, is refused, since its rows would not say what they
attribute. A CUDA cell states it too since gh-ocannl-1444, printed when present.

How the in-step durations add up (gh-ocannl-1444): per timed step, the durations summed, the union
of their stamped intervals (the device's busy time) and the span, against the host's step p50 and
queued step time. A step's segments run one after another on one stream, so stamped intervals
that overlap do not place the kernels in time (flagged STAMPS: what rocprofv3 records on minix's
unified gfx1151, where only each dispatch's own duration holds), and a device span longer than the
host wall time enclosing it is flagged CLOCK. Each row carries the cost model's bytes read/written from the kernel
table (`io:`) and the in-step bandwidth they imply.

Co-tenants (gh-ocannl-1521): each cell prints the GPU-using processes the driver recorded just
before it started and just after it ended (OUT/CELL.cotenants.jsonl, benchmarks/gpu_cotenants.py),
UNREAD where no probe ran and `not recorded` for a cell from before the record existed -- the
evidence the 108% reading lacked. A co-tenant explains nothing by itself; a reader weighs it
against the composition lines.

Joining cells (gh-ocannl-1444): cells of one kernel list -- repeats of one build, whose trace names
are identical -- are compared kernel by kernel in table order; cells of different kernel lists (a
treatment, a revision) are joined by stable_key, the written nodes without schedule-minted staging
nodes, since the segment index in `<routine>__seg<j>` shifts whenever a change adds or merges an
earlier segment.

Subcommands:
  check OUT NAME   validate one cell (OUT/NAME.out, .err, and one trace export); exit 1 with a reason
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
from display_safe import display_value, log_line, table_lines  # noqa: E402
from gh1002_cells import KERNEL  # noqa: E402  (the one reader of the kernel-table row)
import gpu_cotenants  # noqa: E402  (the co-tenant record's writer and reader)

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
# A device span this much longer than the host wall time enclosing it is a time-base error.
CLOCK_TOLERANCE = 0.01
# A step whose stamped intervals overlap by more than this share of its summed durations is one the
# stamps cannot place in time (a step's segments run one after another).
OVERLAP_TOLERANCE = 0.001
# An amdgpu clock sample counts as the step's when the card reports itself at least this busy.
BUSY_FLOOR = 50

KPROF_CELL = re.compile(r"(\w+)-(gpt2_mini\w*)-(\w+)-(kprof\d+)\.out")
# The trace exports a cell may carry, by suffix: (what traced it, its reader).
EXPORTS = {"_cuda_gpu_trace.csv": "nsys", "_kernel_trace.csv": "rocprofv3"}
MEMORY_MODEL = re.compile(r"^kprof: memory model (unified|discrete)\b(.*)$")
SEG = re.compile(r"^(.*)__seg(\d+)$")
ISOLATED = re.compile(r"^(.*)__kernel(\d+)$")
# A node minted under a namespace of its own: Tnode.ident_prefix renders it `<namespace>__n<id>`.
MINTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*__n\d+")


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
    """The kernel table's rows {i: (host_min_ms, geometry, writes, io)}, refusing an incomplete one:
    a declined kernel has no isolated run to compare and no row to label its in-step time. io is
    (bytes read, bytes written, flags) by the cost model, or None on a row from before
    gh-ocannl-1444."""
    rows, totals = {}, set()
    for line in open(path):
        m = KERNEL.match(line.rstrip("\n"))
        if m:
            totals.add(int(m["n"]))
            io = None if m["read"] is None else (int(m["read"]), int(m["written"]), m["io_flags"])
            rows[int(m["i"])] = (float(m["ms"]), "grid=[%s] block=[%s]" % (m["grid"], m["block"]),
                                 m["writes"], io)
    if len(totals) != 1 or sorted(rows) != list(range(next(iter(totals)))):
        raise Refused("%s: the kernel table is missing or incomplete" % path.name)
    return rows


def memory_model(path):
    """The cell's `kprof: memory model` statement, (model, detail), or None."""
    for line in open(path):
        m = MEMORY_MODEL.match(line.rstrip("\n"))
        if m:
            return m.group(1), m.group(2).strip()
    return None


def export(out, stem):
    """The cell's one trace export: (path, tracer). Refuses none, and two (which would be read as
    one cell's in-step durations twice over)."""
    found = [(Path(out) / (stem + suffix), tracer) for suffix, tracer in EXPORTS.items()
             if (Path(out) / (stem + suffix)).is_file()]
    if not found:
        raise Refused("%s has no trace export (%s)" % (stem, " or ".join(stem + s for s in EXPORTS)))
    if len(found) > 1:
        raise Refused("%s has two trace exports: %s" % (stem, ", ".join(p.name for p, _ in found)))
    return found[0]


def trace(path, tracer="nsys"):
    """{routine: {j: [in-step instances (start, end, queue), in start order]}} and {routine: {i:
    [isolated durations]}}, from an `nsys stats --report cuda_gpu_trace --format csv` export or a
    `rocprofv3 --kernel-trace --output-format csv` one: stamps in ns, durations in ms, queue the
    stream (nsys) or HSA queue (rocprofv3) a dispatch went to, None where the export has no such
    column. A rocprofv3 trace whose segments ran on more than one agent is refused: the cell's
    memory model is one device's."""
    seg, iso = {}, {}
    rows, agents = [], set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if tracer == "nsys":
                start = float(r["Start (ns)"])
                rows.append((start, r["Name"], start + float(r["Duration (ns)"]), r.get("Strm")))
            else:
                start, end = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
                rows.append((start, r["Kernel_Name"], end, r.get("Queue_Id")))
                if SEG.match(r["Kernel_Name"]) or ISOLATED.match(r["Kernel_Name"]):
                    agents.add(r["Agent_Id"])
    if len(agents) > 1:
        raise Refused("%s: the step's kernels ran on %d agents (%s); the cell states one device's"
                      " memory model" % (Path(path).name, len(agents),
                                         ", ".join(map(display_value, sorted(agents)))))
    for start, name, end, queue in sorted(rows, key=lambda r: r[0]):
        m = SEG.match(name)
        if m:
            seg.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append((start, end, queue))
            continue
        m = ISOLATED.match(name)
        if m:
            iso.setdefault(m.group(1), {}).setdefault(int(m.group(2)), []).append((end - start) / 1e6)
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
    csv_path, tracer = export(out, stem)
    model = memory_model(out / (stem + ".err"))
    if tracer == "rocprofv3" and model is None:
        raise Refused("%s: a HIP cell states the memory model it ran on, and this one does not"
                      % name)
    seg, iso = trace(csv_path, tracer)
    window = 2 * int(rec["timed_steps"])  # the per-step-synced steps, then the queued ones
    mapped = {}
    for routine, kernels in iso.items():
        ids = sorted(kernels)
        off = ids[0]
        if ids != list(range(off, off + len(ids))) or sorted(seg.get(routine, {})) != list(range(len(ids))):
            raise Refused("%s: routine %s's in-step segments do not match its isolated kernels %d..%d"
                          % (name, display_value(routine), off, ids[-1]))
        for j in range(len(ids)):
            mapped[off + j] = (seg[routine][j], kernels[off + j])
    if sorted(mapped) != sorted(rows):
        raise Refused("%s: the trace maps %d of the table's %d kernels" % (name, len(mapped), len(rows)))
    attributed, instances = [], {}
    for i in sorted(rows):
        stamps, runs = mapped[i]
        if len(stamps) < window:
            raise Refused("%s: kernel %d ran %d times in the step, fewer than the %d timed steps"
                          % (name, i, len(stamps), window))
        if len(runs) < 2:
            raise Refused("%s: kernel %d has no timed isolated run" % (name, i))
        instances[i], runs = stamps[-window:], runs[1:]  # drop the isolated warmup run
        steps = [(end - start) / 1e6 for start, end, _ in instances[i]]
        med = statistics.median(steps)
        p10, p90 = percentile(steps, 10), percentile(steps, 90)
        iso_min = min(runs)
        host_min, geometry, writes, io = rows[i]
        flags = []
        if abs(iso_min - med) >= ABS_FLOOR_MS and abs(iso_min / med - 1) > ISO_TOLERANCE:
            flags.append("ISO")
        if p10 > 0 and p90 / p10 > SPREAD_TOLERANCE:
            flags.append("SPREAD")
        attributed.append({"i": i, "med": med, "p10": p10, "p90": p90, "iso_min": iso_min,
                           "host_min": host_min, "geometry": geometry, "writes": writes, "io": io,
                           "flags": flags, "steps": steps})
    factor = common_factor(attributed)
    if factor is not None:
        for r in attributed:
            scaled = r["iso_min"] * factor["f"]
            if abs(scaled - r["med"]) >= ABS_FLOOR_MS and abs(scaled / r["med"] - 1) > ISO_TOLERANCE:
                r["flags"].append("OWN")
    rec = dict(rec, tracer=tracer, memory_model=model,
               composition=composition(attributed, instances, int(rec["timed_steps"])))
    return rec, attributed


def composition(rows, instances, timed):
    """How the step's in-step kernel durations add up (gh-ocannl-1444), from each timed step's own
    instances: instance s of every segment is step s of the window, the first `timed` steps synced
    one by one and the last `timed` queued back to back. Per step: `sum`, the kernels' durations
    added; `busy`, the union of their [start, end) intervals, which is what the device was occupied
    for; `span`, first start to last end. sum > busy only when stamped intervals overlap, which a
    step's segments -- dependent, on one stream, a linear graph when captured -- cannot do while they
    run: overlapping stamps say where the tracer stamped, not when the kernels ran. The device spans are set beside the host's clocks: the synced half's per-step span against the
    step p50, the queued half's span over `timed` against queued_step_ms. A device span longer than
    the host wall time that encloses it is a clock-domain error, not an attribution. None when the
    window's steps interleave (a segment that did not run once per step)."""
    window = 2 * timed
    per_step = []
    for s in range(window):
        iv = sorted((instances[i][s][0], instances[i][s][1], i) for i in instances)
        covered, busy = None, 0.0
        for start, end, _ in iv:
            busy += max(0.0, end - (start if covered is None else max(start, covered)))
            covered = end if covered is None else max(covered, end)
        per_step.append({"start": iv[0][0], "end": max(e for _, e, _ in iv),
                         "last_start": iv[-1][0],
                         "sum": sum(e - b for b, e, _ in iv) / 1e6, "busy": busy / 1e6})
    for a, b in zip(per_step, per_step[1:]):
        if b["start"] < a["last_start"]:
            return None
    queues = {q for i in instances for _, _, q in instances[i] if q is not None}
    span = [(x["end"] - x["start"]) / 1e6 for x in per_step]
    queued = per_step[timed:]
    return {"sum": statistics.median(x["sum"] for x in per_step),
            "busy": statistics.median(x["busy"] for x in per_step),
            "span": statistics.median(span),
            "overlap_steps": sum(1 for x in per_step
                                 if x["sum"] - x["busy"] > OVERLAP_TOLERANCE * x["sum"]),
            "synced_span": statistics.median(span[:timed]),
            "queued_span": (max(x["end"] for x in queued) - queued[0]["start"]) / 1e6 / timed,
            "queues": len(queues) if queues else None, "steps": window}


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


def amdgpu_clocks(out, stem):
    """The shader clock and power over the cell's amdgpu sysfs samples taken while the GPU reported
    itself at least BUSY_FLOOR % busy with its clock running (gh-ocannl-1444), or None without a
    sampler or such samples: (min, median, max MHz, samples, median W or None). The sampler reads
    every 250 ms, so it sees a few samples of a short window: a coarse reading of the clock the
    step ran at, not a per-kernel one."""
    path = Path(out) / (stem + ".clk.csv")
    if not path.is_file():
        return None
    mhz, watts = [], []
    for r in csv.DictReader(open(path)):
        try:
            # A zero clock is a gated core (gfxoff), not a speed: tuf's gfx1102 reads 0 MHz at 99%
            # busy between a step's bursts.
            if (r.get("busy_pct") and int(r["busy_pct"]) >= BUSY_FLOOR and r.get("sclk_mhz")
                    and int(r["sclk_mhz"]) > 0):
                mhz.append(int(r["sclk_mhz"]))
                if r.get("power_w"):
                    watts.append(float(r["power_w"]))
        except (ValueError, TypeError):
            pass
    if not mhz:
        return None
    return (min(mhz), statistics.median(mhz), max(mhz), len(mhz),
            statistics.median(watts) if watts else None)


def io_cells(r):
    """The kernel's cost-model traffic (MB read/written, its flags) and the in-step bandwidth it
    implies; dashes for a row from before gh-ocannl-1444."""
    if r["io"] is None:
        return "-", "-"
    read, written, flags = r["io"]
    return ("%.2f/%.2f%s" % (read / 1e6, written / 1e6, flags),
            "%.1f%s" % ((read + written) / (r["med"] * 1e6), flags) if r["med"] > 0 else "-")


def render_composition(rec):
    c = rec["composition"]
    if c is None:
        print("- per-step composition: none (the window's steps interleave: a segment did not run"
              " once per step)")
        return
    overlap = c["sum"] - c["busy"]
    print("- per step, median over the %d timed steps: durations summed %.2f ms, device busy (their"
          " union) %.2f ms, span %.2f ms; intervals overlap in %d of the %d steps (sum - busy"
          " %.2f ms, %.1f%% of the sum)" % (c["steps"], c["sum"], c["busy"], c["span"],
                                            c["overlap_steps"], c["steps"], overlap,
                                            100 * overlap / c["sum"] if c["sum"] > 0 else 0.0))
    queued = rec.get("queued_step_ms")
    host = ("step p50 %.2f ms" % rec["step_ms"]["p50"]) + (
        "" if queued is None else "; queued %.2f ms/step, device %.2f" % (queued, c["queued_span"]))
    print("- device span against the host clock: synced steps %.2f ms against the %s" % (
        c["synced_span"], host))
    if c["overlap_steps"]:
        print("- STAMPS: the stamped intervals of one step's segments overlap, which segments run one"
              " after another cannot: the trace's start/end stamps do not place the kernels in time."
              " Each row's own duration may still hold (on minix's gfx1151 they agree with and"
              " without graph capture); the union and span above, and anything read off positions,"
              " do not")
    if c["synced_span"] > rec["step_ms"]["p50"] * (1 + CLOCK_TOLERANCE) or (
            queued is not None and c["queued_span"] > queued * (1 + CLOCK_TOLERANCE)):
        print("- CLOCK: the device stamps span more than the host wall time enclosing them (by"
              " >%d%%): the trace's time base, not the kernels, sets these durations"
              % round(CLOCK_TOLERANCE * 100))
    if c["queues"] is not None:
        print("- in-step dispatches went to %d %s" % (
            c["queues"], "queue" if c["queues"] == 1 else "queues or streams"))


def label(writes):
    names = writes.split()
    return " ".join(names[:LABEL_WRITES]) + (" +%d" % (len(names) - LABEL_WRITES)
                                             if len(names) > LABEL_WRITES else "")


def render_cell(out, name, rec, rows):
    stem = name[:-4]
    total = sum(r["med"] for r in rows)
    print("\n### Kernel attribution: %s\n" % stem)
    print("In-step: device time of each shipped segment over the %d timed steps (median, p10..p90;"
          " traced by %s)." % (2 * int(rec["timed_steps"]), rec["tracer"]))
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
    render_composition(rec)
    if rec["memory_model"] is not None:
        print(log_line("- memory model: %s %s" % rec["memory_model"]))
    print(log_line("- GPU co-tenants: %s"
                   % gpu_cotenants.summary(Path(out) / (stem + ".cotenants.jsonl"))))
    clk = clocks(out, stem)
    if clk:
        print("- SM clock in P0: %d..%d MHz, median %d (%d samples)" % (clk[0], clk[2], clk[1], clk[3]))
    clk = amdgpu_clocks(out, stem)
    if clk:
        print("- shader clock while >= %d%% busy and ungated (amdgpu sysfs): %d..%d MHz, median %d"
              " (%d samples)%s"
              % (BUSY_FLOOR, clk[0], clk[2], clk[1], clk[3],
                 "" if clk[4] is None else "; power median %.1f W" % clk[4]))
    listed, shown, covered = [], 0, 0.0
    for r in sorted(rows, key=lambda r: -r["med"]):
        if covered >= SHOWN_SHARE * total or shown >= SHOWN_MAX:
            break
        shown += 1
        covered += r["med"]
        io, bandwidth = io_cells(r)
        listed.append(("%d" % r["i"], "%.3f" % r["med"], "%.3f..%.3f" % (r["p10"], r["p90"]),
                       "%.3f" % r["iso_min"], "%.2f" % (r["iso_min"] / r["med"]),
                       "%.3f" % r["host_min"], io, bandwidth, " ".join(r["flags"]),
                       label(r["writes"])))
    print()
    print("\n".join(table_lines(
        ["kernel", "in-step median", "p10..p90", "isolated min", "iso / in-step", "table min",
         "MB read/written", "in-step GB/s", "flags", "writes"], listed)))
    rest = sorted(rows, key=lambda r: -r["med"])[shown:]
    if rest:
        print("\n%d further kernels: %.2f ms in-step (%.1f%%), %d flagged ISO, %d SPREAD, %d OWN." % (
            len(rest), sum(r["med"] for r in rest), 100 * sum(r["med"] for r in rest) / total,
            sum("ISO" in r["flags"] for r in rest), sum("SPREAD" in r["flags"] for r in rest),
            sum("OWN" in r["flags"] for r in rest)))


def stable_key(writes):
    """A kernel's join key across builds (gh-ocannl-1444): its written nodes without the ones a
    schedule or rewrite minted under a namespace of its own (`tile__n412_...`, `footprint__n...`,
    `rewrite__n...`: Tnode.ident_prefix), whose ids come from a counter that moves whenever an
    earlier site's staging changes. The trace's own names, `<routine>__seg<j>`, are stable between
    runs of one build -- every repeat of a cell names the same kernels -- but j shifts when a change
    adds or merges a segment anywhere before it, so they join runs of one build and nothing else.
    The names are sorted: the table lists them in creation-uid order (Tn.compare), which two builds
    need not share. A kernel that writes only minted nodes keys on them, sorted, as they are."""
    names = writes.split()
    kept = [w for w in names if not MINTED.match(w)]
    return " ".join(sorted(kept or names))


def join(a, b):
    """Kernels of two cells matched by stable_key, a repeated key by its occurrence in table order:
    ([(row_a, row_b)], [unmatched rows of a], [unmatched rows of b])."""
    def keyed(rows):
        seen, out = {}, {}
        for r in rows:
            k = stable_key(r["writes"])
            seen[k] = seen.get(k, 0) + 1
            out[(k, seen[k])] = r
        return out
    ka, kb = keyed(a), keyed(b)
    pairs = [(ka[k], kb[k]) for k in ka if k in kb]
    return (pairs, [ka[k] for k in ka if k not in kb], [kb[k] for k in kb if k not in ka])


def render_join(cells):
    """Cells of one fixture whose kernel lists differ -- another treatment or revision: each kernel
    list's per-kernel in-step median (the median over its cells) joined, by stable_key, with the
    reference list's (the one holding the `base` treatment, else the first)."""
    groups = {}
    for (backend, fixture, stem), rows in cells:
        signature = tuple((r["writes"], r["geometry"]) for r in rows)
        groups.setdefault((backend, fixture), {}).setdefault(signature, []).append((stem, rows))
    for (backend, fixture), lists in sorted(groups.items()):
        if len(lists) < 2:
            continue
        def treatments(members):
            return sorted({KPROF_CELL.fullmatch(stem + ".out").group(3) for stem, _ in members})
        def medians(members):
            return [dict(r, med=statistics.median(m[1][k]["med"] for m in members))
                    for k, r in enumerate(members[0][1])]
        ordered = sorted(lists.values(), key=lambda ms: ("base" not in treatments(ms), ms[0][0]))
        ref = ordered[0]
        for other in ordered[1:]:
            pairs, only_a, only_b = join(medians(ref), medians(other))
            ta, tb = ", ".join(treatments(ref)), ", ".join(treatments(other))
            print("\n### Kernel join: %s %s, %s against %s\n" % (backend, fixture, tb, ta))
            print("- %d kernels joined by the nodes they write (of %d and %d), %d and %d left over;"
                  " joined in-step medians sum to %.2f ms (%s) and %.2f ms (%s)"
                  % (len(pairs), len(ref[0][1]), len(other[0][1]), len(only_a), len(only_b),
                     sum(a["med"] for a, _ in pairs), ta, sum(b["med"] for _, b in pairs), tb))
            for what, rows in (("only in " + ta, only_a), ("only in " + tb, only_b)):
                if rows:
                    print("- %s: %d kernels, %.2f ms in-step" % (what, len(rows),
                                                              sum(r["med"] for r in rows)))
            moved = [(a, b) for a, b in pairs
                     if abs(b["med"] - a["med"]) >= ABS_FLOOR_MS
                     and max(a["med"], b["med"]) > REPEAT_TOLERANCE * min(a["med"], b["med"])]
            listed = [("%d" % a["i"], "%d" % b["i"], "%.3f" % a["med"], "%.3f" % b["med"],
                       "%+.3f" % (b["med"] - a["med"]), label(a["writes"]))
                      for a, b in sorted(moved, key=lambda ab: -abs(ab[1]["med"] - ab[0]["med"]))]
            listed += [("%d" % r["i"], "-", "%.3f" % r["med"], "-", "%+.3f" % -r["med"],
                        label(r["writes"])) for r in only_a]
            listed += [("-", "%d" % r["i"], "-", "%.3f" % r["med"], "%+.3f" % r["med"],
                        label(r["writes"])) for r in only_b]
            if listed:
                print()
                print("\n".join(table_lines(
                    ["kernel (%s)" % ta, "kernel (%s)" % tb, "in-step (%s)" % ta,
                     "in-step (%s)" % tb, "change ms", "writes"], listed[:SHOWN_MAX])))
                if len(listed) > SHOWN_MAX:
                    print("\n%d further rows not shown." % (len(listed) - SHOWN_MAX))
            else:
                print("- no joined kernel moved by more than %.0f%%, and none is left over"
                      % ((REPEAT_TOLERANCE - 1) * 100))


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
            print()
            print("\n".join(table_lines(
                ["kernel", "in-step medians", "isolated minima", "writes"],
                [("%d" % i, ", ".join("%.3f" % x for x in spread("med", i)[1]),
                  ", ".join("%.3f" % x for x in spread("iso_min", i)[1]),
                  label(members[0][1][i]["writes"])) for i in rows_out])))
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
            print(log_line("REFUSED: %s: %s" % (name, e)))
            refused = True
            continue
        render_cell(out, name, rec, rows)
        cells.append(((m.group(1), m.group(2), name[:-4]), rows))
    render_repeatability(cells)
    render_join(cells)
    return 1 if refused or not cells else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "check":
        try:
            attribute(sys.argv[2], sys.argv[3])
        except (Refused, OSError, ValueError, KeyError) as e:
            print(log_line("REFUSED: %s: %s" % (display_value(sys.argv[3]), e)))
            sys.exit(1)
        sys.exit(0)
    if len(sys.argv) == 3 and sys.argv[1] == "report":
        sys.exit(report(sys.argv[2]))
    print(__doc__, file=sys.stderr)
    sys.exit(2)
