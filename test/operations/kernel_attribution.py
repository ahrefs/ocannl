"""Host-only controls for benchmarks/kernel_attribution.py: the in-step/isolated join of a kprof
cell, its flags, its refusals and the cross-cell repeatability table (gh-ocannl-1347), from an
nsys export or a rocprofv3 one with the memory model the HIP cell states (gh-ocannl-1392)."""
import contextlib
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


def write_cell(out, stem, in_step=IN_STEP, isolated=ISOLATED, table_rows=None, trace=True,
               result=True, early=2, tracer="nsys", model=None, agent_of=lambda name: "Agent 1"):
    """A kprof cell: the result line, the kernel table (with an init routine's segments in the
    trace that no isolated kernel names), and the trace export -- nsys's cuda_gpu_trace, or
    rocprofv3's kernel_trace (with the HIP runtime's own fill kernels in it, and each dispatch's
    agent from agent_of), whose .err states the memory model when one is given."""
    n = len(isolated)
    if result:
        (out / (stem + ".out")).write_text(
            json.dumps({"step_ms": {"p10": 5, "p50": 5, "p90": 5}, "timed_steps": TIMED}) + "\n")
    header = "backend=hip fixture=gpt2_mini\nkprof: tracer rocprofv3 --kernel-trace (1.0.0)\n"
    (out / (stem + ".err")).write_text(
        (header + "kprof: memory model %s\n" % model if model is not None else "") + "".join(
            "bench: kernel %d/%d %.4f ms grid=[1;1;1] block=[32;1;1] mma:none w: node%d extra%d\n"
            % (i, n, isolated[i][0] + 0.01, i, i)
            for i in (table_rows if table_rows is not None else range(n))))
    if not trace:
        return
    rows, t = [], 0
    def launch(name, ms):
        nonlocal t
        t += 1000
        if tracer == "nsys":
            rows.append("%d,%d,%s" % (t, int(ms * 1e6), name))
        else:
            # rocprofv3 quotes its strings and stamps start and end, not a duration.
            rows.append('"KERNEL_DISPATCH","%s","%s",%d,%d'
                        % (agent_of(name), name, 10**14 + t, 10**14 + t + int(ms * 1e6)))
    launch("__amd_rocclr_fillBufferAligned", 50.0)
    launch("init__seg0", 9.0)
    launch("init__seg1", 9.0)
    for k in range(early + len(next(iter(in_step.values())))):
        for routine, ids in ROUTINES:
            for j, i in enumerate(ids):
                if i in in_step:
                    launch("%s__seg%d" % (routine, j), 100.0 if k < early else in_step[i][k - early])
    for routine, ids in ROUTINES:
        for i in ids:
            launch("%s__kernel%d" % (routine, i), 0.01)
            for ms in isolated[i]:
                launch("%s__kernel%d" % (routine, i), ms)
    if tracer == "nsys":
        (out / (stem + "_cuda_gpu_trace.csv")).write_text(
            "Start (ns),Duration (ns),Name\n" + "\n".join(rows) + "\n")
    else:
        (out / (stem + "_kernel_trace.csv")).write_text(
            '"Kind","Agent_Id","Kernel_Name","Start_Timestamp","End_Timestamp"\n'
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
