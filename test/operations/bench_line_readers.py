"""gh-ocannl-1767: every reader of a bench text line, held to lines its producer printed.

Since gh-ocannl-1668 appended a `[grid: ...]` census to the GEBP benches' timing line, a reader that
took the LAST bracket for the mma census read the grid census in its place, and every cell it judged
failed -- unnoticed, because each reader was tested on a line written by hand. Here the lines come
from the producers themselves, run cheaply on cc by the rules beside this test (bin/schedule_bench,
bin/narrow_gebp_bench, bench_rank_site, and bench_line_samples.exe driving Bench_harness's own
emitters), and every reader in benchmarks/bench_output.py runs over them -- with the drivers'
own functions that consume them -- its fields checked against what the same run states another way:
a timing line against its `bench-result:` record, a kernel row against the table's own count, the
trace's events against its summary's totals, a pick against the explain line of the label it picked.
benchmarks/test/test_bench_line_scan.py keeps the readers in bench_output.

Usage: bench_line_readers.py ROOT, ROOT holding benchmarks/ and the captured
`test/operations/bench_lines_<run>.{out,log}` (stdout, stderr).
"""
import importlib
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
bench = root / "benchmarks"
sys.path.insert(0, str(bench))
import bench_output  # noqa: E402


def check(condition, detail):
    # A failed claim stays fatal under python -O, and never reaches the golden.
    if not condition:
        raise SystemExit("FAIL: " + detail)


def claim(text, condition, detail=""):
    check(condition, text + (": " + detail if detail else ""))
    print(text + ": true")


def p_all(text, items, pred, describe=repr):
    """Every one of a NON-EMPTY collection satisfies `pred` (an empty one fails: vacuous)."""
    items = list(items)
    bad = [describe(x) for x in items if not pred(x)]
    claim(text, items and not bad, "empty collection" if not items else f"fails for {bad[:3]}")


def captured(run, stream):
    """A run's capture: its stdout (`out`) or its stderr (`log`)."""
    return (root / "test" / "operations" / f"bench_lines_{run}.{stream}").read_text()


def cli(*args):
    proc = subprocess.run([sys.executable, str(bench / "bench_output.py"), *map(str, args)],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout


# ---------------------------------------------------------------------------------------------
print("== the registry")
formats = bench_output.FORMATS
p_all("every format's readers are bench_output's", formats.items(),
      lambda kv: all(callable(getattr(bench_output, r, None)) for r in kv[1]["readers"]),
      describe=lambda kv: kv[0])
for name, f in formats.items():
    if f["pinned"] is not True:
        print(f"{name}: not pinned here -- {f['pinned']}")
pinned = {name for name, f in formats.items() if f["pinned"] is True}
exercised = set()

# ---------------------------------------------------------------------------------------------
print("== the GEBP timing line (bin/schedule_bench, bin/narrow_gebp_bench)")
gh1051 = importlib.import_module("gh1051_cells")
for run in ("schedule", "gebp"):
    text = captured(run, "out")
    lines = bench_output.timing_lines(text)
    results = {v: s for v, s, _, _ in bench_output.bench_results(text)}
    claim(f"{run}: the timing lines name the variants the bench-result records do",
          lines and [t["variant"] for t in lines] == list(results),
          f"{[t['variant'] for t in lines]} vs {list(results)}")
    # The printed time is the record's at %8.3f.
    p_all(f"{run}: each line's ms is its record's time", lines,
          lambda t: abs(t["ms"] - 1e3 * results[t["variant"]]) <= 5e-4 + 1e-9,
          describe=lambda t: t["variant"])
    p_all(f"{run}: the mma census is read as the mma census, the grid census as the grid census",
          lines, lambda t: not t["mma"].startswith("grid:") and t["grid"].startswith("grid:"),
          describe=lambda t: (t["variant"], t["mma"], t["grid"]))
    p_all(f"{run}: the packed variants' census reads Mma_register_tiled",
          [t for t in lines if t["variant"] in ("packmma", "packmma_par")],
          lambda t: "Mma_register_tiled" in t["mma"], describe=lambda t: (t["variant"], t["mma"]))
    p_all(f"{run}: each line states its checksum", lines,
          lambda t: t["chk"] is not None and all(math.isfinite(float(x)) for x in t["chk"].split("/")),
          describe=lambda t: t["variant"])
    # gh1051_cells' own parse (its variants are GPU ones; here, every variant this run printed).
    gh1051.VARIANTS = list(results)
    parsed = gh1051.parse_text(text)
    claim(f"{run}: gh1051_cells reads each variant's mma census as bench_output does",
          parsed and all(parsed[t["variant"]][2] == t["mma"] for t in lines), repr(parsed))
header = bench_output.gebp_header(captured("gebp", "out"))
claim("gebp: the header names the geometry --rn=2 requested, as gh1180_tie_ab.sh asks",
      header is not None and " rn2 " in header and "(requested)" in header, repr(header))
exercised |= {"gebp_timing", "gebp_header"}

# ---------------------------------------------------------------------------------------------
print("== the per-kernel table (Bench_harness.time_shipped_kernels)")
err = captured("harness", "log")
rows = bench_output.kernel_rows(err)
counts = [tuple(map(int, m.groups())) for m in
          re.finditer(r"^bench: per-kernel table: (\d+) of (\d+) kernels timed", err, re.M)]
claim("the table states one count", len(counts) == 1, repr(counts))
timed_count, n = counts[0]
claim("the rows are the table's kernels, each once, in order",
      [r["i"] for r in rows] == list(range(n)) and {r["n"] for r in rows} == {n},
      repr([(r["i"], r["n"]) for r in rows]))
claim("the injected refusals read as a DECLINED and an UNTIMED row with no time",
      [(r["failure"], r["ms"]) for r in rows[:2]] == [("DECLINED", None), ("UNTIMED", None)])
timed = [r for r in rows if r["failure"] is None]
claim("the timed rows are the ones the table counts as timed", len(timed) == timed_count > 0)
p_all("a timed row reads its range, geometry, bytes and writes", timed,
      lambda r: r["lo"] is not None and r["hi"] is not None and r["grid"] and r["block"]
      and r["io"] is not None and r["mma"] is not None, describe=lambda r: r["i"])
gh1002 = importlib.import_module("gh1002_cells")
kattr = importlib.import_module("kernel_attribution")
gh1161 = importlib.import_module("gh1161_sites")
with tempfile.TemporaryDirectory() as scratch:
    base = Path(scratch) / "cell"
    Path(str(base) + ".err").write_text(err)
    claim("gh1002_cells reads the table with an untimed kernel as incomplete",
          gh1002.kernels(base) is None)
    try:
        kattr.table(Path(str(base) + ".err"))
        refused = None
    except kattr.Refused as e:
        refused = str(e)
    claim("kernel_attribution refuses it as incomplete",
          refused is not None and "incomplete" in refused, repr(refused))
listed = [(r["i"], r["ms"], r["writes"]) for r in rows]
claim("gh1161_sites reads every kernel as listed, the refused two untimed",
      gh1161.complete(listed, n) and sum(ms is None for _, ms, _ in listed) == 2)
exercised.add("kernel_row")

# ---------------------------------------------------------------------------------------------
print("== the launch-geometry trace (schedule_log_launches)")
launches = bench_output.schedule_launches(err)
claim("the harness run traced its launches", bool(launches))
p_all("each launch reads a 3-D grid and block", launches,
      lambda s: len(s["grid"]) == 3 == len(s["block"]) and s["seg"] < s["segs"],
      describe=lambda s: (s["routine"], s["seg"]))
p_all("each compile's launches are its segments, in order", launches,
      lambda s: [x["seg"] for x in launches if x["routine"] == s["routine"]][:s["segs"]]
      == list(range(s["segs"])), describe=lambda s: s["routine"])
# A CPU backend ships a routine as one kernel: nothing here fissions, which is the answer
# gh612_cells.sh's `schedule-segs` gives as its own refusal (exit 3).
log = root / "test/operations/bench_lines_harness.log"
routine = launches[0]["routine"]
code, out = cli("schedule-segs", log, routine)
claim("schedule-segs finds no fissioned compile of a one-kernel routine", code == 3 and not out,
      repr((code, out)))
code, out = cli("schedule-geometry", log, routine, 1)
want = [s for s in launches if s["routine"] == routine and s["segs"] == 1]
claim("schedule-geometry prints each of its launches",
      code == 0 and out.splitlines() == [f"seg {s['seg']} grid=[1;1;1] block=[1;1;1]"
                                         for s in want] and want, repr((code, out[:200])))
exercised.add("schedule_launch")

# ---------------------------------------------------------------------------------------------
print("== the timing trace (Bench_harness.install_timing_trace)")
err = captured("trace", "log")
path = root / "test/operations/bench_lines_trace.log"
summary = bench_output.trace_summary(err)
attempts = bench_output.trace_events(err, "attempt")
calls = bench_output.trace_events(err, "call")
compiles = bench_output.trace_events(err, "compile")
claim("the search printed a summary", summary is not None)
claim("the summary's attempts and timing calls are the trace's attempt and call lines",
      summary["attempts"] == len(attempts) > 0 and summary["calls"] == len(calls) > 0,
      f"{summary} vs {len(attempts)} attempts, {len(calls)} calls")
traces = bench_output.compile_traces(err)
claim("every compile line reads as a compile record", len(traces) == len(compiles) > 0,
      f"{len(traces)} of {len(compiles)}")
p_all("a compile record's kernel counts read in order", traces,
      lambda t: 0 <= t["kernels_compiled"] <= t["kernels"] and t["wall"] >= 0)
want = {"summary": 0, "timing-calls": 0, "attempt": 0, "search-done": 0, "incomplete": 3}
got = {p: cli("trace-has", path, p)[0] for p in want}
claim("trace-has answers what the trace holds", got == want, repr(got))
code, out = cli("trace-digest", path)
claim("trace-digest counts the attempts", code == 0
      and out.splitlines()[0] == f"candidate attempts {len(attempts)}", repr(out[:200]))
exercised.add("timing_trace")

# ---------------------------------------------------------------------------------------------
print("== the rank-site lines (bench_rank_site)")
text = captured("rank_site", "out")
path = root / "test/operations/bench_lines_rank_site.out"
explains = {f["label"]: f for _, f, _ in bench_output.rank_site_lines(text, "explain")}
kernels = {f["label"]: f for _, f, _ in bench_output.rank_site_lines(text, "kernels")}
picks = [f for _, f, _ in bench_output.rank_site_lines(text, "pick")]
claim("several candidates are explained, each with its kernels line",
      len(explains) > 1 and set(explains) == set(kernels), f"{list(explains)} / {list(kernels)}")
claim("one pick, of an explained candidate, at the rank its explain line states",
      len(picks) == 1 and picks[0]["label"] in explains
      and picks[0]["rank_ms"] == explains[picks[0]["label"]]["rank_ms"], repr(picks))
gh1442 = importlib.import_module("gh1442_agreement")
gh1600 = importlib.import_module("gh1600_txn_fit")
scores = gh1442.scores(path)
claim("gh1442_agreement scores each explained candidate, admitted as its line says",
      set(scores) == set(explains)
      and all(scores[l][2] == (explains[l]["admitted"] == "true") for l in scores))
claim("gh1442_agreement reads the pick and the seeded plan it names",
      gh1442.pick_of(path) == (picks[0]["label"], picks[0]["seeded_label"]))
own = gh1442.pick_timing(path)
claim("gh1442_agreement reads the pick's own timing and its paired ratio",
      own is not None and own[2] is not None and len(own[2]) == 4, repr(own))
listing = gh1600.listing(path)
claim("gh1600_txn_fit lists each candidate with its kernels",
      set(listing) == set(explains) and all(v[3] for v in listing.values()))
code, out = cli("rank-site-field", path, "pick", "seeded_label")
claim("rank-site-field prints the pick's seeded_label, as gh1253_cells.sh asks",
      code == 0 and out == picks[0]["seeded_label"] + "\n", repr((code, out)))
# OCaml's %S writes non-ASCII and control bytes as DECIMAL \ddd, which Python's unicode_escape
# reads as octal: the labels are decoded as the literals they are.
claim("an OCaml literal decodes as OCaml writes it",
      bench_output.ocaml_string(r'caf\195\169 \"q\" a\\b\n') == 'café "q" a\\b\n')
exercised.add("rank_site")

claim("every pinned format was read off a producer's line here", exercised == pinned,
      f"pinned {sorted(pinned)}, exercised {sorted(exercised)}")
