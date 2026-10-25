"""Controls for benchmarks/bin_cells.py, the hermetic bin/ bench driver (gh-ocannl-743).

Arguments: the driver, the stand-in bench `bin_cells_probe.exe`, and the golden of the
`bench_design` test (whose ORDERS table the driver's design must reproduce).

The driver runs end to end against the probe, which is a real OCANNL-linked executable: its
configuration records and result lines come from the library code a bench uses. The ambient sources
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

# --- The record reader, on the shapes Utils.config_record_to_json prints. ------------------------
cr = bc.config_records


def rec(key, value, source, layer=None, profile=None, origin=None):
    return json.dumps({"ocannl_config": 1, "key": key, "value": value, "source": source,
                       "layer": source if layer is None else layer, "profile": profile,
                       "origin": origin}, separators=(",", ":"), ensure_ascii=False)


stream = "\n".join([
    rec("backend", "cc", "commandline", origin="--ocannl_backend=cc"),
    "a line the bench printed in between",
    rec("no_config_file", "true", "commandline", origin="--ocannl_no_config_file=true"),
    rec("autotune_rounds", "2", "default"),
    rec("tf32_matmuls", "true", "profile", "commandline", "approximate",
        "--ocannl_profile=approximate"),
    rec("fp_contract", "true", "environment", origin="OCANNL_FP_CONTRACT"),
    rec("log_level", "0", "config_file", origin="/x/ocannl_config"),
    rec("debug_highlight_pcre", "x, commandline y",
        "commandline", origin="--ocannl_debug_highlight_pcre=x, commandline y"),
    rec("debug_highlight_pcre", "x, commandline y",
        "commandline", origin="--ocannl_debug_highlight_pcre=x, commandline y") + "\r",
])
ARGV = ["bench.exe", "--ocannl_backend=cc", "--ocannl_no_config_file=true",
        "--ocannl_profile=approximate", "--ocannl_debug_highlight_pcre=x, commandline y"]
parsed = cr.parse_config_records(stream)
claim("every source shape is read with its value and layer, a repeated identical record once",
      [(k, parsed[k]["value"], parsed[k]["source"], parsed[k]["layer"]) for k in sorted(parsed)]
      == [("autotune_rounds", "2", "default", "default"),
          ("backend", "cc", "commandline", "commandline"),
          ("debug_highlight_pcre", "x, commandline y", "commandline", "commandline"),
          ("fp_contract", "true", "environment", "environment"),
          ("log_level", "0", "config_file", "config_file"),
          ("no_config_file", "true", "commandline", "commandline"),
          ("tf32_matmuls", "true", "profile", "commandline")])
claim("a value holding a Unicode line separator stays one record",
      cr.parse_config_records(rec("k", "a\u2028b\u0085c", "default") + "\n")["k"]["value"]
      == "a\u2028b\u0085c")
GOOD = rec("k", "1", "default")
claim("a tagged line that is not JSON is refused",
      refused(lambda: cr.parse_config_records(GOOD[:-5]), cr.ConfigRecordError))
claim("a schema version this reader does not know is refused",
      refused(lambda: cr.parse_config_records(GOOD.replace(':1,', ':2,', 1)), cr.ConfigRecordError))
claim("a record missing a field, or with a field of the wrong type, is refused",
      refused(lambda: cr.parse_config_records(GOOD.replace(',"origin":null', '')),
              cr.ConfigRecordError)
      and refused(lambda: cr.parse_config_records(GOOD.replace('"value":"1"', '"value":1')),
                  cr.ConfigRecordError))
claim("a source that does not fit its layer, profile and origin is refused",
      all(refused(lambda line=line: cr.parse_config_records(line), cr.ConfigRecordError) for line in [
          rec("fp_contract", "1", "environment", "commandline", origin="OCANNL_FP_CONTRACT"),
          rec("k", "1", "commandline"),
          rec("k", "1", "profile", "commandline", None, "--ocannl_profile=p"),
          rec("k", "1", "default", origin="x"),
          rec("k", "1", "profile", "default", "p", "x")]))
claim("one key resolved two ways is refused",
      refused(lambda: cr.parse_config_records(GOOD + "\n" + rec("k", "2", "environment",
                                                                  origin="OCANNL_FP_CONTRACT")),
              cr.ConfigRecordError))
claim("the prose trace of log_config_sourcing carries no records",
      cr.parse_config_records("Retrieving commandline, environment, or config file variable "
                              "ocannl_backend\nFound cc, commandline --ocannl_backend=cc") == {})
claim("the environment and config-file sources each fail the treatment",
      bc.treatment_problems(parsed, "cc", ARGV)
      == ["ocannl_fp_contract came from the environment",
          "ocannl_log_level came from the config_file"])
claim("a backend other than the pinned one fails the treatment",
      any("not to the pinned 'cuda'" in p for p in bc.treatment_problems(parsed, "cuda", ARGV)))
clean = {k: parsed[k] for k in ("backend", "no_config_file")}
claim("a profile picked outside the command line fails the treatment, whatever its value",
      bc.treatment_problems({**clean, "x": cr.parse_config_records(rec(
          "x", "1", "profile", "environment", "approximate", "OCANNL_PROFILE"))["x"]}, "cc", ARGV)
      == ["ocannl_x came from the environment through a profile picked there"])
claim("a commandline source naming no argument of the cell fails the treatment",
      bc.treatment_problems(clean, "cc", ["bench.exe", "--ocannl_no_config_file=true"])
      == ["ocannl_backend came from a commandline argument the cell was not given: "
          "'--ocannl_backend=cc'"])
claim("no records at all fail the treatment", bc.treatment_problems({}, "cc", ARGV) != [])
claim("two cells of one arm resolving a setting differently are reported",
      bc.consistency_problems([
          {"arm": "a", "cell": 0, "config": {"k": {"value": "1"}}},
          {"arm": "b", "cell": 1, "config": {"k": {"value": "2"}}},
          {"arm": "a", "cell": 2, "config": {"k": {"value": "3"}}}])
      == ["arm a: ocannl_k resolved differently -- '1' in cell(s) 0; '3' in cell(s) 2"])
claim("a setting read in some cells of an arm and not in others is reported",
      bc.consistency_problems([
          {"arm": "a", "cell": 0, "config": {"k": {"value": "1"}}},
          {"arm": "a", "cell": 1, "config": {"k": {"value": "1"}, "late": {"value": "x"}}}])
      == ["arm a: ocannl_late resolved differently -- not read in cell(s) 0; 'x' in cell(s) 1"])
CHK = ',"check":"passed","check_note":""}'
claim("a well-formed result line is read with its check",
      bc.parse_results('bench-result: {"variant":"v","seconds":0.5' + CHK)
      == [("v", 0.5, "passed", "")])
claim("a non-finite or non-positive result is refused",
      refused(lambda: bc.parse_results('bench-result: {"variant":"v","seconds":null' + CHK))
      and refused(lambda: bc.parse_results('bench-result: {"variant":"v","seconds":0' + CHK))
      and refused(lambda: bc.parse_results('bench-result: {"variant":"v","seconds":NaN' + CHK)))
claim("a result line that is JSON but not an object is refused as unusable",
      refused(lambda: bc.parse_results("bench-result: []")))
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
           "BENCH_TUNE": "1", "OMP_NUM_THREADS": "3", "CUDA_LAUNCH_BLOCKING": "1",
           "DATABASE_URL": "postgres://u:s3cret@h/db"}

    def drive(out, *argv):
        return subprocess.run([sys.executable, str(driver_path), "--exe", str(probe), "--backend",
                               "cc", "--out", str(scratch / out), *argv], env=env,
                              capture_output=True, text=True)

    # The stream and the function are one list: the probe's [records] mode prints
    # Utils.resolved_config on stdout, after the same run streamed it on stderr.
    (scratch / "direct").mkdir()
    direct = subprocess.run([str(probe), "--ocannl_no_config_file=true", "--ocannl_backend=cc",
                             "--ocannl_log_config_json=true", "records"], cwd=scratch / "direct",
                            env=env, capture_output=True, text=True)
    streamed = [l for l in direct.stderr.splitlines() if l.startswith(cr.TAG)]
    claim("the stream on stderr is, line for line, the records the function returns",
          direct.returncode == 0 and streamed and streamed == direct.stdout.splitlines())
    seen = cr.parse_config_records(direct.stderr)
    claim("the stream opens with the bootstrap settings and carries a read made late in main",
          [k for k in seen][:5] == ["log_config_sourcing", "log_config_json", "no_config_file",
                                    "suppress_welcome_message", "profile"]
          and seen.get("debug_highlight_pcre", {}).get("source") == "default")
    claim("the stream names each source: an argv element, a planted variable",
          seen.get("backend", {}).get("origin") == "--ocannl_backend=cc"
          and seen.get("log_config_json", {}).get("origin") == "--ocannl_log_config_json=true"
          and [seen.get("autotune_timing", {}).get(f) for f in ("value", "origin")]
          == ["x", "OCANNL_AUTOTUNE_TIMING"])

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
    claim("toolchain variables are recorded by value, every other variable by name only",
          manifest["environment_values"].get("PATH") == os.environ.get("PATH")
          and "DATABASE_URL" in manifest["environment_names"]
          and "DATABASE_URL" not in manifest["environment_values"]
          and "s3cret" not in (scratch / "ab/manifest.json").read_text())
    claim("each arm's kernel compiler is resolved from ocamlc -config and fingerprinted",
          set(manifest["kernel_compilers"]) == {"slow", "fast"}
          and all(len(c.get("sha256", "")) == 64 and os.path.isfile(c["resolved"])
                  for c in manifest["kernel_compilers"].values()))
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
                                "--ocannl_log_config_json=true"] for r in records)
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

    # --- The shared runner's front-end features (gh-ocannl-1565). ------------------------------
    refused_run = drive("refused", "--rounds", "2", "--arm", "r=refused")
    claim("a run whose only faults are cells refusing their measurement exits 3, though they "
          "reported no result",
          refused_run.returncode == 3 and "refused its measurement (exit 3)" in refused_run.stdout
          and "reported no bench-result line" not in refused_run.stdout)
    mixed = drive("mixed", "--rounds", "2", "--arm", "r=refused", "--arm", "f=fail")
    claim("a refusal beside a failure is a failed run", mixed.returncode == 1)

    stated = drive("stated", "--rounds", "2", "--env", "BENCH_PROBE_UNUSED=x",
                   "--arm", "e3=BENCH_PROBE_SECONDS=0.003 env",
                   "--arm", "e6=BENCH_PROBE_SECONDS=0.006 env")
    stated_records = [json.loads(l)
                      for l in (scratch / "stated/cells.jsonl").read_text().splitlines()]
    stated_manifest = json.loads((scratch / "stated/manifest.json").read_text())
    claim("an arm's stated variable reaches its cells, the ambient BENCH_TUNE does not",
          stated.returncode == 0
          and {(r["arm"], r["results"][0][1]) for r in stated_records}
          == {("e3", 0.003), ("e6", 0.006)}
          and "BENCH_TUNE" in stated_manifest["cleared"])
    claim("the stated environment is recorded per cell and in the manifest",
          all(r["env"] == {"BENCH_PROBE_UNUSED": "x",
                           "BENCH_PROBE_SECONDS": "0.003" if r["arm"] == "e3" else "0.006"}
              for r in stated_records)
          and stated_manifest["common_env"] == {"BENCH_PROBE_UNUSED": "x"}
          and "`BENCH_PROBE_SECONDS=0.006`" in stated.stdout)
    claim("a stated library setting is refused before any cell runs",
          drive("ocannl_env", "--env", "OCANNL_BACKEND=cc").returncode == 2
          and drive("ocannl_arm", "--arm", "a=OCANNL_BACKEND=cc 1").returncode == 2
          and not (scratch / "ocannl_env").exists() and not (scratch / "ocannl_arm").exists())

    def drive_exes(out, *argv):
        return subprocess.run([sys.executable, str(driver_path), "--backend", "cc", "--out",
                               str(scratch / out), *argv], env=env, capture_output=True, text=True)

    paired = drive_exes("arm_exe", "--rounds", "2", "--arm-exe", f"one={probe}",
                        "--arm-exe", f"two={probe}", "--arm", "two=0.002")
    paired_manifest = json.loads((scratch / "arm_exe/manifest.json").read_text())
    claim("arms on executables of their own need no --bench or --exe, and carry their sha256",
          paired.returncode == 0 and paired_manifest["executable"] is None
          and all(a["executable"] == str(probe) and len(a["executable_sha256"]) == 64
                  for a in paired_manifest["arms"])
          and [a["args"] for a in paired_manifest["arms"]] == [["0.002"], []])
    claim("an arm without an executable is refused when neither --bench nor --exe is given",
          drive_exes("arm_exe_missing", "--arm-exe", f"one={probe}", "--arm", "two=1").returncode
          == 2)

    record = drive("record", "--rounds", "2", "--reader", "record", "--metric", "step_ms.p50",
                   "--metric", "compile_s", "--arm", "slow=record:4", "--arm", "fast=record:2")
    record_records = [json.loads(l)
                      for l in (scratch / "record/cells.jsonl").read_text().splitlines()]
    claim("a runner's result line is read as one unchecked variant per metric, in seconds",
          record.returncode == 0
          and all(sorted((v, s, c) for v, s, c, _ in r["results"])
                  == [("compile_s", 0.5, "unchecked"),
                      ("step_ms.p50", 0.004 if r["arm"] == "slow" else 0.002, "unchecked")]
                  for r in record_records))
    section = record.stdout[record.stdout.index("## Parity losses"):]
    print(section[:section.index("## Resolved configuration")], end="")
    claim("a metric that is no time field is refused",
          drive("record_bad", "--reader", "record", "--metric", "timed_steps").returncode == 2
          and drive("metric_alone", "--metric", "step_ms.p50").returncode == 2)
    missing = drive("record_missing", "--rounds", "2", "--reader", "record", "--arm", "q=0.001")
    claim("a cell printing no result line fails a record-reading run",
          missing.returncode == 1 and "reported no result line" in missing.stdout)

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
