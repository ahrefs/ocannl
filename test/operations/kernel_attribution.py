"""Host-only controls for benchmarks/kernel_attribution.py: the in-step/isolated join of a kprof
cell, its flags, its refusals and the cross-cell repeatability table (gh-ocannl-1347), from an
nsys export or a rocprofv3 one with the memory model the HIP cell states (gh-ocannl-1392)."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

owner = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(owner.parent))
spec = importlib.util.spec_from_file_location("kernel_attribution", owner)
ka = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ka)


def check(condition, detail):
    # A failing control must remain fatal under python -O.
    if not condition:
        raise SystemExit("FAIL: " + detail)


TIMED = 2  # timed_steps: the window is the last 2 * TIMED instances of each segment
# Kernel 0: in-step 1.0 = isolated 1.0. Kernel 1: in-step 2.0, isolated 3.0 (ISO). Kernel 2: an
# in-step spread of 1.0..1.5 (SPREAD). Kernels 3-4 belong to a second routine. Every segment's
# first two instances (parity, warmup) are 100 ms: a reader that keeps them reports them.
IN_STEP = {0: [1.0] * 4, 1: [2.0] * 4, 2: [1.0, 1.0, 1.5, 1.5], 3: [0.5] * 4, 4: [0.25] * 4}
# The first isolated run is the untimed warmup, tiny on purpose: a reader that keeps it under-reports.
ISOLATED = {0: [1.0] * 20, 1: [3.0] * 20, 2: [1.2] * 20, 3: [0.5] * 20, 4: [0.25] * 20}
ROUTINES = [("step", [0, 1, 2]), ("gate", [3, 4])]


UNIFIED = "unified (Radeon 8060S Graphics, gfx1151: integrated=true)"


def node_writes(i):
    return "node%d extra%d" % (i, i)


def write_cell(out, stem, in_step=IN_STEP, isolated=ISOLATED, table_rows=None, trace=True,
               result=True, early=2, tracer="nsys", model=None, agent_of=lambda name: "Agent 1",
               routines=ROUTINES, writes=node_writes, io=None, step_ms=6.0, back_ns=0,
               queue_of=None):
    """A kprof cell: the result line, the kernel table (with an init routine's segments in the
    trace that no isolated kernel names), and the trace export -- nsys's cuda_gpu_trace, or
    rocprofv3's kernel_trace (with the HIP runtime's own fill kernels in it, and each dispatch's
    agent from agent_of), whose .err states the memory model when one is given. Dispatches run back
    to back 1 us apart, except that an in-step segment after a step's first starts back_ns before
    its predecessor ends; queue_of names a dispatch's queue/stream column, absent when None. io
    gives a table row its `io:` field."""
    n = len(isolated)
    if result:
        (out / (stem + ".out")).write_text(
            json.dumps({"step_ms": {"p10": step_ms, "p50": step_ms, "p90": step_ms},
                        "queued_step_ms": step_ms, "timed_steps": TIMED}) + "\n")
    header = "backend=hip fixture=gpt2_mini\nkprof: tracer rocprofv3 --kernel-trace (1.0.0)\n"
    (out / (stem + ".err")).write_text(
        (header + "kprof: memory model %s\n" % model if model is not None else "") + "".join(
            "bench: kernel %d/%d %.4f ms grid=[1;1;1] block=[32;1;1] %smma:none w: %s\n"
            % (i, n, isolated[i][0] + 0.01, "" if io is None else "io:%d/%d " % io(i), writes(i))
            for i in (table_rows if table_rows is not None else range(n))))
    if not trace:
        return
    rows, t, last_end = [], 0, 0
    def launch(name, ms, back=0):
        nonlocal t, last_end
        start = last_end - back if back else t
        end = start + int(ms * 1e6)
        last_end = end
        t = max(t, end + 1000)
        queue = () if queue_of is None else (queue_of(name),)
        if tracer == "nsys":
            rows.append(",".join(map(str, (start, end - start) + queue + (name,))))
        else:
            # rocprofv3 quotes its strings and stamps start and end, not a duration.
            rows.append(",".join(('"KERNEL_DISPATCH"', '"%s"' % agent_of(name)) + queue
                                 + ('"%s"' % name, str(10**14 + start), str(10**14 + end))))
    launch("__amd_rocclr_fillBufferAligned", 50.0)
    launch("init__seg0", 9.0)
    launch("init__seg1", 9.0)
    for k in range(early + len(next(iter(in_step.values())))):
        first = True
        for routine, ids in routines:
            for j, i in enumerate(ids):
                if i in in_step:
                    launch("%s__seg%d" % (routine, j), 100.0 if k < early else in_step[i][k - early],
                           back=0 if first or k < early else back_ns)
                    first = False
    for routine, ids in routines:
        for i in ids:
            launch("%s__kernel%d" % (routine, i), 0.01)
            for ms in isolated[i]:
                launch("%s__kernel%d" % (routine, i), ms)
    queue = () if queue_of is None else (("Strm",) if tracer == "nsys" else ('"Queue_Id"',))
    if tracer == "nsys":
        (out / (stem + "_cuda_gpu_trace.csv")).write_text(
            ",".join(("Start (ns)", "Duration (ns)") + queue + ("Name",)) + "\n"
            + "\n".join(rows) + "\n")
    else:
        (out / (stem + "_kernel_trace.csv")).write_text(
            ",".join(('"Kind"', '"Agent_Id"') + queue
                     + ('"Kernel_Name"', '"Start_Timestamp"', '"End_Timestamp"')) + "\n"
            + "\n".join(rows) + "\n")


def refused(out, name, reason):
    try:
        ka.attribute(out, name)
    except ka.Refused as e:
        check(reason in str(e), "%s: %s" % (reason, e))
        return
    check(False, "accepted a cell that should be refused for: " + reason)


with tempfile.TemporaryDirectory() as scratch:
    out = Path(scratch)
    stem = "cuda-gpt2_mini_train-base-kprof1"
    write_cell(out, stem)
    rec, rows = ka.attribute(out, stem + ".out")
    by = {r["i"]: r for r in rows}
    check([r["i"] for r in rows] == [0, 1, 2, 3, 4], str(rows))
    check(by[0]["med"] == 1.0 and by[1]["med"] == 2.0 and by[3]["med"] == 0.5, str(rows))
    print("in-step durations come from the timed window only: true")
    check(by[3]["med"] == 0.5 and by[4]["med"] == 0.25 and by[4]["writes"] == "node4 extra4", str(rows))
    print("a later routine's segments map to the table's indices after its predecessors': true")
    check(by[0]["iso_min"] == 1.0 and by[1]["iso_min"] == 3.0, str(rows))
    print("the isolated warmup run is not an isolated timing: true")
    check(by[0]["flags"] == [] and by[3]["flags"] == [] and by[4]["flags"] == [], str(rows))
    check(by[1]["flags"] == ["ISO", "OWN"], str(by[1]))
    check(by[2]["flags"] == ["SPREAD"], str(by[2]))
    print("ISO flags an isolated time off its in-step one, SPREAD a wide in-step spread: true")
    print("with a common factor of 1, OWN flags exactly what ISO does: true")

    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(str(out))
    text = text.getvalue()
    check(code == 0 and "### Kernel attribution: " + stem in text, text)
    check("sum of in-step medians 5.00 ms" in text and "| 1 | 2.000 | 2.000..2.000 | 3.000 | 1.50 |" in text, text)
    check("Kernel repeatability" not in text, text)
    print("one cell renders its table and owes no repeatability section: true")
    check("- GPU co-tenants: not recorded (no %s.cotenants.jsonl: a cell from before gh-ocannl-1521)"
          % stem in text, text)
    # The co-tenant record the driver writes around the cell (gh-ocannl-1521), printed under it.
    snap = {"cell": stem, "box": "rog-nv", "host": "h", "at": "2026-10-25T00:00:00Z",
            "probes": [{"probe": "nvidia-smi", "status": "ran", "detail": ""}],
            "gpu_processes": [{"pid": 77, "user": "bob", "command": "python3 train.py",
                               "probe": "nvidia-smi", "gpu_memory_mib": 640}]}
    (out / (stem + ".cotenants.jsonl")).write_text(
        json.dumps(dict(snap, event="opened")) + "\n"
        + json.dumps(dict(snap, event="closed", gpu_processes=[])) + "\n")
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(str(out))
    text = text.getvalue()
    check(code == 0 and "- GPU co-tenants: opened 1 [pid 77 bob python3 train.py (nvidia-smi, 640 MiB)]"
          " (nvidia-smi ran); closed 0 (nvidia-smi ran)" in text, text)
    print("a cell prints the GPU co-tenants recorded at its start and end, or that none were: true")

    # An A/A pair: a second cell of the same kernel list, one kernel moved in-step by 20%.
    moved = {**IN_STEP, 0: [1.2] * 4}
    write_cell(out, "cuda-gpt2_mini_train-keep-kprof1", in_step=moved)
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(str(out))
    text = text.getvalue()
    check(code == 0 and "### Kernel repeatability: cuda gpt2_mini_train, 2 cells" in text, text)
    check("in-step median: largest max/min across cells 1.200 (kernel 0" in text, text)
    check("| 0 | 1.000, 1.200 | 1.000, 1.000 | node0 extra0 |" in text, text)
    check("| 1 |" not in text.split("Kernel repeatability")[1], text)
    print("cells of one kernel list are compared kernel by kernel, the moved one listed: true")

    # A cell of another kernel list is attributed but not compared with them.
    other = dict(ISOLATED)
    write_cell(out, "cuda-gpt2_mini_train-fill1-kprof1", isolated=other)
    (out / "cuda-gpt2_mini_train-fill1-kprof1.err").write_text(
        (out / "cuda-gpt2_mini_train-fill1-kprof1.err").read_text().replace("node2 ", "nodeX "))
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(str(out))
    text = text.getvalue()
    check(code == 0 and "2 cells of one kernel list" in text and "fill1-kprof1" in text, text)
    check("3 cells" not in text, text)
    print("a cell that shipped other kernels is attributed but compared with none: true")

with tempfile.TemporaryDirectory() as scratch:
    # A capped step (gh-ocannl-1391): every kernel of >= 0.5 ms runs 1.2x its isolated minimum
    # in-step, and kernel 4, below the factor's population floor, 1.6x. One step of the window runs
    # 1.4x throughout, so the per-step factor has a spread to report.
    out = Path(scratch)
    iso = {0: [1.0] * 20, 1: [2.0] * 20, 2: [1.0] * 20, 3: [0.5] * 20, 4: [0.25] * 20}
    capped = {i: [v[0] * 1.2] * 3 + [v[0] * 1.4] for i, v in iso.items()}
    capped[4] = [0.4] * 4
    stem = "cuda-gpt2_mini_train_b256-base-kprof1"
    write_cell(out, stem, in_step=capped, isolated=iso)
    _, rows = ka.attribute(out, stem + ".out")
    by = {r["i"]: r for r in rows}
    check(all(by[i]["flags"] == ["ISO", "SPREAD"] for i in range(4)), str(rows))
    check(by[4]["flags"] == ["ISO", "OWN"], str(by[4]))
    print("a step-wide clock factor is ISO but not OWN; a kernel's own deviation beyond it is: true")
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        ka.report(str(out))
    text = text.getvalue()
    check("common in-step/isolated factor F = 1.200 (median over the 4 kernels of >= 0.5 ms"
          " in-step; per-step factor p10..p90 1.200..1.400)" in text, text)
    check("- OWN (isolated x F off in-step by >10%): 1 kernels, 0.40 ms in-step" in text, text)
    print("the report states the factor, its per-step spread and the OWN share: true")
    small = {i: [0.1] * 4 for i in iso}
    write_cell(out, stem, in_step=small, isolated={i: [0.05] * 20 for i in iso})
    _, rows = ka.attribute(out, stem + ".out")
    check(all("OWN" not in r["flags"] and "ISO" in r["flags"] for r in rows), str(rows))
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        ka.report(str(out))
    check("common in-step/isolated factor: none (fewer than 3 kernels" in text.getvalue(),
          text.getvalue())
    print("without enough sizeable kernels there is no factor, and OWN flags nothing: true")

with tempfile.TemporaryDirectory() as scratch:
    # The same cell traced by rocprofv3 on HIP (gh-ocannl-1392): the join reads start/end stamps.
    out = Path(scratch)
    write_cell(out, "cuda-gpt2_mini-base-kprof1")
    stem = "hip-gpt2_mini-base-kprof1"
    write_cell(out, stem, tracer="rocprofv3", model=UNIFIED)
    cuda_rec, cuda_rows = ka.attribute(out, "cuda-gpt2_mini-base-kprof1.out")
    rec, rows = ka.attribute(out, stem + ".out")
    check(rows == cuda_rows and rec["tracer"] == "rocprofv3" and cuda_rec["tracer"] == "nsys",
          "%s\n%s" % (rows, cuda_rows))
    check(rec["memory_model"] == ("unified", UNIFIED[len("unified "):]) and
          cuda_rec["memory_model"] is None, str((rec, cuda_rec)))
    print("a rocprofv3 kernel trace attributes as the nsys export does: true")
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(str(out))
    text = text.getvalue()
    hip = text.split("### Kernel attribution: " + stem)[1].split("###")[0]
    check(code == 0 and "traced by rocprofv3" in hip and "- memory model: " + UNIFIED in hip, text)
    check("memory model" not in text.split("### Kernel attribution: cuda-")[1].split("###")[0], text)
    print("a HIP cell's attribution states its memory model and its tracer: true")

    write_cell(out, "hip-gpt2_mini-keep-kprof1", tracer="rocprofv3")
    refused(out, "hip-gpt2_mini-keep-kprof1.out", "states the memory model")
    print("a HIP cell that states no memory model: refused")
    write_cell(out, "hip-gpt2_mini-fill1-kprof1", tracer="rocprofv3", model=UNIFIED,
               agent_of=lambda name: "Agent 2" if name.startswith("gate__") else "Agent 1")
    refused(out, "hip-gpt2_mini-fill1-kprof1.out", "ran on 2 agents")
    print("a HIP cell whose kernels ran on two agents: refused")
    write_cell(out, "hip-gpt2_mini-fill256-kprof1", tracer="rocprofv3", model=UNIFIED)
    write_cell(out, "hip-gpt2_mini-fill256-kprof1", model=UNIFIED)
    refused(out, "hip-gpt2_mini-fill256-kprof1.out", "two trace exports")
    print("a cell with two trace exports: refused")

with tempfile.TemporaryDirectory() as scratch:
    out = Path(scratch)
    write_cell(out, "a-kprof1", table_rows=[0, 1, 3, 4])
    refused(out, "a-kprof1.out", "incomplete")
    write_cell(out, "b-kprof1", trace=False)
    refused(out, "b-kprof1.out", "no trace export")
    write_cell(out, "c-kprof1", result=False)
    (out / "c-kprof1.out").write_text("nsys: Generating report\n")
    refused(out, "c-kprof1.out", "no result line")
    short = {i: v[:3] for i, v in IN_STEP.items()}
    write_cell(out, "d-kprof1", in_step=short, early=0)
    refused(out, "d-kprof1.out", "fewer than the 4 timed steps")
    gap = dict(IN_STEP)
    del gap[2]
    write_cell(out, "e-kprof1", in_step=gap)
    refused(out, "e-kprof1.out", "do not match its isolated kernels")
    print("an incomplete table, a missing trace or result, a short window, an unmapped segment: refused")

    # report: a refused cell fails the report, and no cell at all is no attribution.
    name = "cuda-gpt2_mini_train-base-kprof1"
    write_cell(out, name, trace=False)
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(str(out))
    check(code == 1 and "REFUSED: %s.out" % name in text.getvalue(), text.getvalue())
with tempfile.TemporaryDirectory() as scratch:
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(scratch)
    check(code == 1, text.getvalue())
    print("a report with a refused cell, or with no cell, fails: true")


def report_text(out):
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        code = ka.report(str(out))
    return code, text.getvalue()


with tempfile.TemporaryDirectory() as scratch:
    # gh-ocannl-1444: how a cell's in-step durations add up against its device time and the step.
    out = Path(scratch)
    stem = "hip-gpt2_mini_train-base-kprof1"
    write_cell(out, stem, tracer="rocprofv3", model=UNIFIED, queue_of=lambda name: "1")
    rec, rows = ka.attribute(out, stem + ".out")
    c = rec["composition"]
    # Per-step sums 4.75, 4.75, 5.25, 5.25; each step's 5 kernels sit 1 us apart.
    check(abs(c["sum"] - 5.0) < 1e-9 and abs(c["busy"] - c["sum"]) < 1e-9
          and c["overlap_steps"] == 0 and abs(c["span"] - 5.004) < 1e-9, str(c))
    check(c["queues"] == 1, str(c))
    code, text = report_text(out)
    check(code == 0 and "intervals overlap in 0 of the 4 steps (sum - busy 0.00 ms" in text
          and "STAMPS" not in text and "CLOCK" not in text
          and "in-step dispatches went to 1 queue" in text, text)
    print("back-to-back kernels: the summed durations are the device's busy time, nothing overlaps: true")

    # Each in-step segment after a step's first is stamped as opening 0.2 ms before its predecessor
    # ends (minix's gfx1151 stamps whole batches of dispatches with one start): the sum counts those
    # 0.2 ms twice, the union once, and segments that run one after another cannot overlap.
    stem = "hip-gpt2_mini_train-keep-kprof1"
    write_cell(out, stem, tracer="rocprofv3", model=UNIFIED, back_ns=200000,
               queue_of=lambda name: "2" if name.startswith("gate__") else "1")
    rec, rows = ka.attribute(out, stem + ".out")
    c = rec["composition"]
    by = {r["i"]: r for r in rows}
    check(abs(c["sum"] - 5.0) < 1e-9 and abs(c["busy"] - 4.2) < 1e-9 and c["overlap_steps"] == 4,
          str(c))
    check(by[1]["med"] == 2.0, str(rows))
    print("overlapping stamps: the union counts the overlap once, the durations stay each dispatch's"
          " own: true")
    code, text = report_text(out)
    cell = text.split("### Kernel attribution: " + stem)[1].split("###")[0]
    check("intervals overlap in 4 of the 4 steps (sum - busy 0.80 ms, 16.0% of the sum)" in cell
          and "- STAMPS: the stamped intervals of one step's segments overlap" in cell
          and "| 1 | 2.000 | 2.000..2.000 |" in cell and "went to 2 queues or streams" in cell, cell)
    print("the report calls overlapping stamps of a serial step a stamping artifact: true")

with tempfile.TemporaryDirectory() as scratch:
    out = Path(scratch)
    # A step p50 shorter than the device span of a synced step: the stamps' time base is off.
    write_cell(out, "cuda-gpt2_mini-base-kprof1", step_ms=4.0)
    code, text = report_text(out)
    check("device span against the host clock: synced steps 4.75 ms against the step p50 4.00 ms;"
          " queued 4.00 ms/step, device 5.25" in text and "- CLOCK: the device stamps span more" in text,
          text)
    print("a device span longer than the host wall time enclosing it is called a clock error: true")

with tempfile.TemporaryDirectory() as scratch:
    out = Path(scratch)
    stem = "cuda-gpt2_mini-base-kprof1"
    write_cell(out, stem, io=lambda i: (4000000 * (i + 1), 1000000))
    _, rows = ka.attribute(out, stem + ".out")
    check(rows[0]["io"] == (4000000, 1000000, ""), str(rows[0]))
    code, text = report_text(out)
    # Kernel 0 moves 5 MB in its 1 ms in-step median: 5 GB/s.
    check("| MB read/written | in-step GB/s |" in text
          and "| 0 | 1.000 | 1.000..1.000 | 1.000 | 1.00 | 1.010 | 4.00/1.00 | 5.0 |" in text, text)
    write_cell(out, "cuda-gpt2_mini-keep-kprof1")
    code, text = report_text(out)
    check("| 0 | 1.000 | 1.000..1.000 | 1.000 | 1.00 | 1.010 | - | - |" in text, text)
    print("the table carries each kernel's cost-model bytes and the in-step bandwidth they imply,"
          " a row without them dashes: true")

with tempfile.TemporaryDirectory() as scratch:
    # Joining kernels across runs and builds (gh-ocannl-1444). Build A ships 5 kernels; build B
    # splits A's kernel 1 into a zero, a staged product and a max (so every later segment index
    # shifts) and mints its staging tiles under other ids.
    out = Path(scratch)
    a_writes = {0: "n10 tile__n5_tile_q", 1: "n11 n12_max", 2: "layer.grad tile__n6_tile",
                3: "w.grad", 4: "w.grad"}
    a_in = {0: [1.0] * 4, 1: [0.5] * 4, 2: [2.0] * 4, 3: [0.25] * 4, 4: [0.25] * 4}
    b_writes = {0: "n10 tile__n8_tile_q", 1: "n11", 2: "n11 tile__n9_tile_k", 3: "n12_max",
                4: "layer.grad tile__n11_tile", 5: "w.grad", 6: "w.grad"}
    b_in = {0: [1.0] * 4, 1: [0.1] * 4, 2: [0.3] * 4, 3: [0.2] * 4, 4: [1.5] * 4, 5: [0.25] * 4,
            6: [0.25] * 4}
    def cell_of(stem, writes, in_step, routines):
        write_cell(out, stem, in_step=in_step, isolated={i: v[:1] * 20 for i, v in in_step.items()},
                   writes=writes.__getitem__, routines=routines)
        return ka.attribute(out, stem + ".out")[1]
    a_routines = [("step", [0, 1, 2]), ("gate", [3, 4])]
    run1 = cell_of("cuda-gpt2_mini_train-base-kprof1", a_writes, a_in, a_routines)
    run2 = cell_of("cuda-gpt2_mini_train-base-kprof2", a_writes, a_in, a_routines)
    names = [set(r["Name"] for r in csv.DictReader(open(out / (s + "_cuda_gpu_trace.csv"))))
             for s in ("cuda-gpt2_mini_train-base-kprof1", "cuda-gpt2_mini_train-base-kprof2")]
    pairs, left_a, left_b = ka.join(run1, run2)
    check(names[0] == names[1] and [(a["i"], b["i"]) for a, b in pairs] == [(i, i) for i in range(5)]
          and not left_a and not left_b, str(pairs))
    print("two runs of one build name the same kernels and join kernel for kernel: true")
    b = cell_of("cuda-gpt2_mini_train-keep-kprof1", b_writes, b_in,
                [("step", [0, 1, 2, 3, 4]), ("gate", [5, 6])])
    # The table lists written nodes in creation-uid order, which two builds need not share.
    check(ka.stable_key("n12_max n11 tile__n5_tile_q") == ka.stable_key("n11 tile__n9_tile_k n12_max")
          == "n11 n12_max", ka.stable_key("n12_max n11 tile__n5_tile_q"))
    print("a written-node set keys the same in any order, minted staging aside: true")
    # By trace name A's step__seg2 is B's step__seg2, which is another kernel.
    check(ka.stable_key(run1[2]["writes"]) != ka.stable_key(b[2]["writes"]), str(b))
    pairs, left_a, left_b = ka.join(run1, b)
    check(sorted((x["i"], y["i"]) for x, y in pairs) == [(0, 0), (2, 4), (3, 5), (4, 6)]
          and [r["i"] for r in left_a] == [1] and [r["i"] for r in left_b] == [1, 2, 3],
          str((pairs, left_a, left_b)))
    print("across builds the written nodes, minted staging set aside, join what the segment index"
          " no longer does, and name what is new or gone: true")
    code, text = report_text(out)
    joined = text.split("### Kernel join: cuda gpt2_mini_train, keep against base")[1]
    check(code == 0 and "- 4 kernels joined by the nodes they write (of 5 and 7), 1 and 3 left over;"
          " joined in-step medians sum to 3.50 ms (base) and 3.00 ms (keep)" in joined
          and "| 2 | 4 | 2.000 | 1.500 | -0.500 | layer.grad tile__n6_tile |" in joined
          and "| 1 | - | 0.500 | - | -0.500 | n11 n12_max |" in joined
          and "| - | 3 | - | 0.200 | +0.200 | n12_max |" in joined
          and "| 0 | 0 |" not in joined, joined)
    check("### Kernel repeatability: cuda gpt2_mini_train, 2 cells of one kernel list" in text, text)
    print("the report joins each other kernel list against base's, listing what moved, appeared or"
          " went: true")

with tempfile.TemporaryDirectory() as scratch:
    # gh-ocannl-1444: a HIP cell's amdgpu sysfs samples; idle ones (the compile, the teardown),
    # gated ones (0 MHz however busy) and ones the card left blank are not the step's.
    out = Path(scratch)
    stem = "hip-gpt2_mini_train-base-kprof1"
    write_cell(out, stem, tracer="rocprofv3", model=UNIFIED)
    (out / (stem + ".clk.csv")).write_text(
        "timestamp_ns,sclk_mhz,power_w,temp_c,busy_pct\n1,600,10.0,40.0,0\n2,2800,80.0,60.0,99\n"
        "3,2600,70.0,61.0,97\n4,2900,,62.0,100\n5,,,,\n6,0,5.0,40.0,3\n7,0,1.0,51.0,99\n")
    code, text = report_text(out)
    check("- shader clock while >= 50% busy and ungated (amdgpu sysfs): 2600..2900 MHz, median 2800"
          " (3 samples); power median 75.0 W" in text, text)
    print("a HIP cell reports its busy shader clock and power from the sysfs samples: true")
