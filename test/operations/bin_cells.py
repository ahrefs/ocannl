"""Controls for benchmarks/bin_cells.py, the hermetic bin/ bench driver (gh-ocannl-743).

Arguments: the driver, the stand-in bench `bin_cells_probe.exe`, and the golden of the
`bench_design` test (whose ORDERS table the driver's design must reproduce).

The driver runs end to end against the probe, which is a real OCANNL-linked executable: its
configuration trace and result lines come from the library code a bench uses. The ambient sources
the driver exists to clear are planted first -- OCANNL_*, BENCH_* and OMP_* variables, and an
`ocannl_config` above the output directory -- and the cells must not see them.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

driver_path, probe, design_golden = (Path(a).resolve() for a in sys.argv[1:4])
sys.path.insert(0, str(driver_path.parent))  # the driver's own imports live beside it
spec = importlib.util.spec_from_file_location("bin_cells", driver_path)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def claim(label, holds):
    # Printed AND enforced: python -O must not erase a verdict while leaving a passing line.
    print(f"{label}: {str(bool(holds)).lower()}")
    if not holds:
        raise SystemExit("FAIL: " + label)


def refused(f, exc=ValueError):
    try:
        f()
    except exc:
        return True
    return False


# --- The design is Bench_design's, row for row. -------------------------------------------------
rows, inside = [], False
for line in design_golden.read_text().splitlines():
    if line == "ORDERS":
        inside = True
    elif line == "END ORDERS":
        inside = False
    elif inside:
        head, order = line.split(": ")
        arms, batches, b = map(int, head.split())
        rows.append((arms, batches, b, [int(x) for x in order.split()]))
claim("the design golden carries an ORDERS table", len(rows) >= 40)
claim("design_orders reproduces every row of Bench_design's ORDERS table",
      all(bc.design_orders(arms, batches)[b] == order for arms, batches, b, order in rows))
claim("an odd round count is refused", refused(lambda: bc.design_orders(3, 5)))
claim("no arms is refused", refused(lambda: bc.design_orders(0, 2)))

# --- The trace parser, on the shapes Utils.describe_config_source prints. -----------------------
R = "Retrieving commandline, environment, or config file variable ocannl_"
trace = "\n".join([
    R + "backend", "Found cc, commandline --ocannl_backend=cc",
    "a line the bench printed in between",
    R + "autotune_rounds", "Not found, using default 2",
    R + "tf32_matmuls", 'Found true, in the profile "approximate" picked via the commandline',
    R + "fp_contract", "Found true, environment OCANNL_FP_CONTRACT",
    R + "log_level", "Found 0, in the config file",
    R + "debug_highlight_pcre", "Found a, b, commandline --ocannl_debug_highlight_pcre=a, b",
])
parsed = bc.parse_config_trace(trace)
claim("every source shape is read with its value", [(k, parsed[k]["value"], parsed[k]["source"])
      for k in sorted(parsed)] == [
    ("autotune_rounds", "2", "default"), ("backend", "cc", "commandline"),
    ("debug_highlight_pcre", "a, b", "commandline"), ("fp_contract", "true", "environment"),
    ("log_level", "0", "config file"), ("tf32_matmuls", "true", "profile")])
claim("a request never answered is refused",
      refused(lambda: bc.parse_config_trace(R + "backend\nnothing\n"), bc.TraceError))
claim("two requests in a row are refused",
      refused(lambda: bc.parse_config_trace(R + "a\n" + R + "b\nNot found, using default 1"),
              bc.TraceError))
claim("one key resolved to two values is refused", refused(lambda: bc.parse_config_trace(
    "\n".join([R + "k", "Not found, using default 1", R + "k", "Found 2, commandline --k=2"])),
    bc.TraceError))
problems = bc.treatment_problems({**parsed, "no_config_file": {
    "value": "true", "source": "commandline", "detail": ""}}, "cc")
claim("the environment and config-file sources each fail the treatment",
      problems == ["ocannl_fp_contract came from the environment",
                   "ocannl_log_level came from the config file"])
claim("a backend other than the pinned one fails the treatment",
      any("not to the pinned 'cuda'" in p for p in bc.treatment_problems(parsed, "cuda")))
claim("a profile picked outside the command line fails the treatment",
      bc.treatment_problems({"backend": parsed["backend"], "no_config_file": {
          "value": "true", "source": "commandline", "detail": ""}, "x": {
          "value": "1", "source": "profile", "detail": "approximate via the environment"}}, "cc")
      == ["ocannl_x came from a profile picked outside the command line"])
claim("an empty trace fails the treatment", bc.treatment_problems({}, "cc") != [])
claim("two cells of one arm resolving a setting differently are reported",
      bc.consistency_problems([
          {"arm": "a", "cell": 0, "config": {"k": {"value": "1"}}},
          {"arm": "b", "cell": 1, "config": {"k": {"value": "2"}}},
          {"arm": "a", "cell": 2, "config": {"k": {"value": "3"}}}])
      == ["arm a: ocannl_k resolved to '1' in cell 0 and to '3' in cell 2"])
CHK = ',"check":"passed","check_note":""}'
claim("a well-formed result line is read with its check",
      bc.parse_results('bench-result: {"variant":"v","seconds":0.5' + CHK)
      == [("v", 0.5, "passed", "")])
claim("a non-finite or non-positive result is refused",
      refused(lambda: bc.parse_results('bench-result: {"variant":"v","seconds":null' + CHK))
      and refused(lambda: bc.parse_results('bench-result: {"variant":"v","seconds":0' + CHK))
      and refused(lambda: bc.parse_results('bench-result: {"variant":"v","seconds":NaN' + CHK)))
claim("a result that does not state its output check is refused",
      refused(lambda: bc.parse_results('bench-result: {"variant":"v","seconds":0.5}'))
      and refused(lambda: bc.parse_results(
          'bench-result: {"variant":"v","seconds":0.5,"check":"maybe","check_note":""}')))
claim("the sign test is exact", (bc.sign_test_p(12, 12), bc.sign_test_p(6, 12),
                                  round(bc.sign_test_p(10, 12), 6)) == (2 / 4096, 1.0, 0.038574))

# --- End to end, against the probe, with every ambient source planted. --------------------------
with tempfile.TemporaryDirectory() as scratch:
    scratch = Path(scratch)
    (scratch / "ocannl_config").write_text("autotune_bound_pruning=true\nbackend=metal\n")
    env = {**os.environ, "OCANNL_AUTOTUNE_BOUND_PRUNING": "true", "OCANNL_AUTOTUNE_TIMING": "x",
           "BENCH_TUNE": "1", "OMP_NUM_THREADS": "3", "CUDA_LAUNCH_BLOCKING": "1"}

    def drive(out, *argv):
        return subprocess.run([sys.executable, str(driver_path), "--exe", str(probe), "--backend",
                               "cc", "--out", str(scratch / out), *argv], env=env,
                              capture_output=True, text=True)

    run = drive("ab", "--rounds", "4", "--arm", "slow=0.004",
                "--arm", "fast=0.002 --ocannl_autotune_timing=isolated")
    claim("a clean run exits 0", run.returncode == 0)
    records = [json.loads(l) for l in (scratch / "ab/cells.jsonl").read_text().splitlines()]
    manifest = json.loads((scratch / "ab/manifest.json").read_text())
    claim("every round runs every arm, in the design's order",
          [(r["round"], r["arm"]) for r in records]
          == [(rnd, ["slow", "fast"][a]) for rnd, order in enumerate(bc.design_orders(2, 4))
              for a in order])
    claim("the planted variables are cleared, by name",
          {"OCANNL_AUTOTUNE_BOUND_PRUNING", "OCANNL_AUTOTUNE_TIMING", "BENCH_TUNE",
           "OMP_NUM_THREADS"} <= set(manifest["cleared"]))
    claim("a device runtime control passes through and is recorded with its value",
          manifest["runtime_controls"].get("CUDA_LAUNCH_BLOCKING") == "1"
          and "CUDA_LAUNCH_BLOCKING=1" in run.stdout)
    claim("no cell resolved a setting from the environment or a config file",
          all(e["source"] in ("commandline", "default")
              for r in records for e in r["config"].values()))
    claim("the planted settings resolved to their defaults",
          all(r["config"]["autotune_bound_pruning"]["value"] == "false"
              and r["config"]["backend"]["value"] == "cc" for r in records))
    claim("each cell kept its streams in a directory of its own under the output directory",
          all((scratch / "ab/cells" / f"{r['cell']:03d}-{r['arm']}" / "stderr").read_text()
              for r in records))
    claim("each cell's command line is the driver's pins, then the arm's arguments",
          all(r["argv"][:4] == [str(probe), "--ocannl_backend=cc", "--ocannl_no_config_file=true",
                                "--ocannl_log_config_sourcing=true"] for r in records)
          and {tuple(r["argv"][4:]) for r in records}
          == {("0.004",), ("0.002", "--ocannl_autotune_timing=isolated")})
    report = run.stdout
    claim("report.md is what was printed", (scratch / "ab/report.md").read_text() == report)
    # The report from its arms on: the identity block names this host and binary.
    print(report[report.index("## Arms"):report.index("## Resolved configuration")], end="")
    section = report[report.index("What the arms differ by"):]
    print(section[:section.index("Every setting")], end="")
    print(report[report.index("## Cells"):], end="")

    bad = drive("bad", "--rounds", "2", "--arm", "ok=0.001", "--arm", "fails=fail",
                "--arm", "nonfinite=nonfinite")
    claim("a failing cell fails the run", bad.returncode == 1)
    print(bad.stdout[bad.stdout.index("## Verdict"):], end="")

    silent = drive("silent", "--rounds", "2", "--arm", "quiet=silent")
    claim("a cell reporting no result fails the run",
          silent.returncode == 1 and "reported no bench-result line" in silent.stdout)
    wrong = drive("wrong", "--rounds", "2", "--arm", "w=wrong")
    claim("a variant that failed its output check fails the run",
          wrong.returncode == 1
          and "probe failed its output check: differs from the oracle" in wrong.stdout)
    unchecked = drive("unchecked", "--rounds", "2", "--arm", "u=unchecked")
    claim("an unchecked variant is measured, and reported as unchecked with its reason",
          unchecked.returncode == 0
          and "- probe (u): no oracle for this one" in unchecked.stdout)
    twice = drive("twice", "--rounds", "2", "--arm", "dup=twice")
    claim("a cell reporting one variant twice fails the run",
          twice.returncode == 1 and "a variant reported twice: probe" in twice.stdout)

    hung = drive("hung", "--rounds", "2", "--cell-timeout", "2", "--arm", "sleeps=sleep")
    hung_records = [json.loads(l) for l in (scratch / "hung/cells.jsonl").read_text().splitlines()]
    claim("a cell past its cap is recorded as timed out and fails the run",
          hung.returncode == 1 and [r["status"] for r in hung_records] == ["timeout", "timeout"])

    # Refusals before anything runs.
    (scratch / "full").mkdir()
    (scratch / "full/leftover").write_text("")
    claim("a non-empty output directory is refused",
          drive("full").returncode != 0 and not (scratch / "full/manifest.json").exists())
    claim("an output directory inside the checkout is refused",
          subprocess.run([sys.executable, str(driver_path), "--exe", str(probe), "--backend", "cc",
                          "--out", str(driver_path.parent / "inside")], env=env,
                         capture_output=True).returncode != 0
          and not (driver_path.parent / "inside").exists())
    claim("duplicate arm labels are refused",
          drive("dup", "--arm", "a=1", "--arm", "a=2").returncode != 0)
    claim("a malformed arm is refused", drive("malformed", "--arm", "no label").returncode != 0)
    claim("an odd round count is refused before any cell runs",
          drive("odd", "--rounds", "3").returncode != 0 and not (scratch / "odd").exists())
