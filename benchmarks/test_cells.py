"""End-to-end controls of the shared cell runner's front ends and the drivers converted onto them
(gh-ocannl-1565, gh-ocannl-1443), on stub runs: `ab.py` over two revisions of a scratch
repository, `gh1343_ab.sh` (which writes its "before" commit itself), and `gh1180_tie_ab.sh`.

A stub bench is a small Python program standing in for an OCANNL executable: it echoes the
`log_config_json` records a real one prints for its `--ocannl_*` arguments (the record reader
itself is held to the library's real output by test/operations/bin_cells.py) and prints the lines
the driver reads. A stub `dune` first on PATH builds a revision by writing such a stub, whose
reported time comes from that revision's own `arrayjit/lib/schedule.ml` -- so a number in a report
proves which tree it was built from. Run through test_orchestrate.py, which imports these classes.
"""

import contextlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock
from pathlib import Path

import box_gate

HERE = Path(__file__).resolve().parent

# The record `Utils.config_record_to_json` prints (config_records.py's schema) for every
# `--ocannl_<key>=<value>` the stub was given, and then `body` (Python source, with `argv` and
# `positionals` bound).
STUB = '''#!{python}
import json as _json, sys
argv = sys.argv[1:]
for a in argv:
    if a.startswith("--ocannl_") and "=" in a:
        key, value = a[len("--ocannl_"):].split("=", 1)
        print(_json.dumps({{"ocannl_config": 1, "key": key, "value": value,
                           "source": "commandline", "layer": "commandline", "profile": None,
                           "origin": a}}, separators=(",", ":")), file=sys.stderr)
positionals = [a for a in argv if not a.startswith("--")]
{body}
'''


def write_stub(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STUB.format(python=sys.executable, body=textwrap.dedent(body)))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def result_line(variant, seconds, check="passed", note=""):
    return ('print("bench-result: " + json.dumps({"variant": %r, "seconds": %s, "check": %r, '
            '"check_note": %r}))' % (variant, seconds, check, note))


# A stub `dune build --root R --build-dir B [-j N] TARGET`: the "executable" it builds reports the
# seconds R/arrayjit/lib/schedule.ml names, and the positional it was given.
FAKE_DUNE = '''#!{python}
import pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(args[args.index("--root") + 1])
build = pathlib.Path(args[args.index("--build-dir") + 1])
target = args[-1]
seconds = (root / "arrayjit/lib/schedule.ml").read_text().split("speed=")[1].split()[0]
exe = build / "default" / target
exe.parent.mkdir(parents=True, exist_ok=True)
exe.write_text({stub!r}.replace("SECONDS", seconds))
exe.chmod(0o755)
'''

STEP_BODY = '''
import json
print("steps " + " ".join(positionals), file=sys.stderr)
''' + result_line("step", "SECONDS", "unchecked", "a stub")


# The stand-ins are extensionless Python scripts run by their shebang (and a stub `dune` found on
# PATH ahead of the real one), which native Windows' CreateProcess cannot launch.
POSIX_ONLY = unittest.skipIf(os.name == "nt", "stub executables run by shebang: POSIX only")


@POSIX_ONLY
class StubRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        # A cc cell's kernel compiler is fingerprinted from `ocamlc -config`'s c_compiler; the
        # CPU-torch CI job has no OCaml at all, so a stub ocamlc names this interpreter, which
        # exists everywhere the stubs do.
        self.fakebin = self.dir / "fakebin"
        self.fakebin.mkdir()
        ocamlc = self.fakebin / "ocamlc"
        ocamlc.write_text(f"#!/bin/sh\necho 'c_compiler: {sys.executable}'\n")
        ocamlc.chmod(0o755)
        self.path = f"{self.fakebin}{os.pathsep}{os.environ.get('PATH', '')}"

    def run_tool(self, argv, env=None, timeout=300):
        return subprocess.run(argv, text=True, capture_output=True, cwd=self.dir, timeout=timeout,
                              env=dict(os.environ, PATH=self.path, **(env or {})))

    def records(self, run_dir):
        return [json.loads(l) for l in (run_dir / "cells.jsonl").read_text().splitlines()]


class AbDriverTest(StubRunTest):
    """ab.py and gh1343_ab.sh: two revisions, each built once, measured as two interleaved arms."""

    def setUp(self):
        super().setUp()
        fakebin = self.fakebin
        step_stub = STUB.format(python=sys.executable, body=textwrap.dedent(STEP_BODY))
        dune = fakebin / "dune"
        dune.write_text(FAKE_DUNE.format(python=sys.executable, stub=step_stub))
        dune.chmod(0o755)
        self.env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                    "GIT_COMMITTER_EMAIL": "t@t", "OCANNL_BACKEND": "metal",
                    "BENCH_TUNE": "1"}
        self.repo = self.dir / "repo"
        (self.repo / "arrayjit/lib").mkdir(parents=True)
        self.git("init", "-q")

    def git(self, *argv):
        proc = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *argv], cwd=self.repo,
                              text=True, capture_output=True, env=dict(os.environ, **self.env),
                              timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip()

    def commit(self, speed, other):
        (self.repo / "arrayjit/lib/schedule.ml").write_text(f"(* speed={speed} *)\n")
        (self.repo / "arrayjit/lib/schedule.mli").write_text(f"(* mli {speed} *)\n")
        (self.repo / "other.ml").write_text(f"(* {other} *)\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", f"speed {speed}")
        return self.git("rev-parse", "HEAD")

    def by_arm(self, run_dir):
        out = {}
        for r in self.records(run_dir):
            self.assertTrue(r["ok"], r["problems"])
            out.setdefault(r["arm"], set()).add(r["results"][0][1])
        return out

    def test_ab_builds_each_revision_once_and_pairs_them(self):
        slow = self.commit("0.004", "one")
        fast = self.commit("0.002", "two")
        out = self.dir / "out"
        proc = self.run_tool([sys.executable, str(HERE / "ab.py"), slow, "HEAD", "--repo",
                              str(self.repo), "--target", "test/training/bench_mlp_bn_step.exe",
                              "--backend", "cc", "--out", str(out), "--rounds", "4", "--", "7"],
                             self.env)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        record = json.loads((out / "ab.json").read_text())
        self.assertEqual(record["commits"], {"a": slow, "b": fast})
        # The numbers are each tree's own: arm a reports the slow commit's schedule.ml.
        self.assertEqual(self.by_arm(out / "cells"), {"a": {0.004}, "b": {0.002}})
        records = self.records(out / "cells")
        self.assertEqual([r["arm"] for r in records], ["a", "b", "b", "a", "a", "b", "b", "a"])
        self.assertTrue(all(r["argv"][-1] == "7" for r in records))
        # The ambient OCANNL_BACKEND and BENCH_TUNE reached no cell.
        self.assertTrue(all(r["config"]["backend"]["value"] == "cc" for r in records))
        report = (out / "cells/report.md").read_text()
        self.assertIn(f"- executable (a): {slow}, sha256", report)
        self.assertIn(f"- executable (b): {fast}, sha256", report)
        self.assertIn("| step | b | 4 | 0.500 | 0.500-0.500 | 4 of 4 | 0.125 |", report)
        # Each worktree went once its executable was copied out.
        self.assertFalse((out / "src-a").exists() or (out / "src-b").exists())
        self.assertNotIn(str(out), self.git("worktree", "list"))

    def test_ab_refuses_a_revision_that_names_no_commit(self):
        self.commit("0.004", "one")
        out = self.dir / "out"
        proc = self.run_tool([sys.executable, str(HERE / "ab.py"), "HEAD", "no-such-rev", "--repo",
                              str(self.repo), "--target", "x.exe", "--backend", "cc", "--out",
                              str(out)], self.env)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("names no commit", proc.stderr)
        self.assertFalse(out.exists())

    def test_ab_refuses_a_stated_library_setting_before_building(self):
        self.commit("0.004", "one")
        out = self.dir / "out"
        proc = self.run_tool([sys.executable, str(HERE / "ab.py"), "HEAD", "HEAD", "--repo",
                              str(self.repo), "--target", "x.exe", "--backend", "cc", "--out",
                              str(out), "--args-b", "OCANNL_FP_CONTRACT=true"], self.env)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("a library setting is stated on the command line", proc.stderr)
        self.assertFalse(out.exists())

    def test_gh1343_driver_measures_head_against_head_with_the_base_scheduler(self):
        base = self.commit("0.004", "base")
        head = self.commit("0.002", "head")
        out = self.dir / "out"
        proc = self.run_tool(["bash", str(HERE / "gh1343_ab.sh"), str(self.repo), base, str(out),
                              "2", "9"], self.env)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        record = json.loads((out / "ab.json").read_text())
        before = record["commits"]["a"]
        self.assertEqual(record["commits"]["b"], head)
        # "before" is HEAD's tree with BASE's scheduler sources, a child of HEAD, and no branch
        # moved to make it.
        self.assertEqual(self.git("rev-parse", f"{before}^"), head)
        self.assertEqual(self.git("show", f"{before}:other.ml"), "(* head *)")
        self.assertEqual(self.git("show", f"{before}:arrayjit/lib/schedule.mli"), "(* mli 0.004 *)")
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual(self.by_arm(out / "cells"), {"a": {0.004}, "b": {0.002}})
        self.assertTrue(all(r["argv"][-1] == "9" for r in self.records(out / "cells")))

    def test_gh1343_driver_refuses_uncommitted_changes(self):
        base = self.commit("0.004", "base")
        (self.repo / "other.ml").write_text("dirty\n")
        proc = self.run_tool(["bash", str(HERE / "gh1343_ab.sh"), str(self.repo), base,
                              str(self.dir / "out"), "2", "9"], self.env)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("uncommitted tracked changes", proc.stderr)


# narrow_gebp_bench's lines that gh1180_tie_ab.sh judges: the header naming the requested geometry,
# the packed rows' census, and the bench-result lines it pairs. CENSUS and EXIT vary per test. The
# timing lines are in the bench's own format, the mma census followed by gh-ocannl-1668's grid
# census (test/operations/bench_line_readers holds the reader to the lines the bench prints): a
# reader taking the last bracket for the mma census reads the grid census and fails every cell.
GEBP_BODY = '''
import json
rn = next(a.split("=")[1] for a in argv if a.startswith("--rn="))
print(f"GEBP n=64, 3 repeats, blocking bm=8 bk=16, register tile rm4 rn{{rn}} lanes4 (requested), "
      "storage single, compute single, packed panels (storage)")
secs = 0.001 * int(rn)
census = "CENSUS".replace("RN", rn)
print(f"packmma         1.000 ms      5.00 GFLOP/s  (spot check [65] 0.50, chk -1.5/2.25, = naive)  "
      f"[{{census}}] [grid: no Grid loop]")
print(f"packmma_par     1.000 ms      9.00 GFLOP/s  (spot check [65] 0.50, chk -1.5/2.25, = packmma)  "
      f"[{{census}}] [grid: 1 of 2 on the pool; i45 serial: below cc_grid_min_updates = 16384]")
print("bench-result: " + json.dumps({{"variant": "packmma", "seconds": secs, "check": "passed",
                                      "check_note": ""}}))
print("bench-result: " + json.dumps({{"variant": "packmma_par", "seconds": secs / 2,
                                      "check": "passed", "check_note": ""}}))
sys.exit(3 if positionals[1] == "REFUSE_N" else EXIT)
'''


class Gh1180DriverTest(StubRunTest):
    """gh1180_tie_ab.sh on bin_cells: the census and header checks over each case's cells."""

    def measure(self, census="Mma_register_tiled", exit_code=0, refuse_n="none",
                cases=("f16:64:8:16:3:4:6",)):
        root = self.dir / "root"
        write_stub(root / "_build/default/bin/narrow_gebp_bench.exe",
                   GEBP_BODY.replace("CENSUS", census).replace("REFUSE_N", refuse_n)
                   .replace("EXIT", str(exit_code)).replace("{{", "{").replace("}}", "}"))
        out = self.dir / "out"
        proc = self.run_tool(
            [box_gate.BASH, str(HERE / "gh1180_tie_ab.sh"), "measure", str(root), str(out), "2",
             *cases], {"OCANNL_FP16_ARITHMETIC": "true", "OMP_NUM_THREADS": "1"})
        return proc, out

    def test_a_clean_run_pairs_the_two_geometries(self):
        proc, out = self.measure()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        summary = (out / "summary.md").read_text()
        self.assertIn("| f16 n=64 bm=8 bk=16 | serial | rn4 | rn6 | 2 | 1.500 | 1.500-1.500 | 0 of 2 |",
                      summary)
        self.assertNotIn("INVALID", summary)
        run = out / "f16-64-8-16-3-rn4-rn6"
        records = self.records(run)
        self.assertEqual([r["arm"] for r in records], ["rn4", "rn6", "rn6", "rn4"])
        # The case's precision flag is on argv, the ambient one cleared.
        self.assertTrue(all(r["config"]["fp16_arithmetic"] == {
            "value": "false", "source": "commandline", "layer": "commandline", "profile": None,
            "origin": "--ocannl_fp16_arithmetic=false"} for r in records))

    def test_the_standalone_summary_reapplies_the_runs_own_checks(self):
        proc, out = self.measure()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # Rewrite one clean rn4 cell as having resolved a setting its siblings did not: every
        # cell still reads ok, and only the run-level consistency rule can fail it.
        jsonl = out / "f16-64-8-16-3-rn4-rn6/cells.jsonl"
        records = self.records(jsonl.parent)
        records[0]["config"]["late_setting"] = {"value": "x", "source": "default",
                                                "layer": "default", "profile": None, "origin": None}
        jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
        proc = self.run_tool([box_gate.BASH, str(HERE / "gh1180_tie_ab.sh"), "summary", str(out)])
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("ocannl_late_setting resolved differently", proc.stdout)

    def test_a_scalar_fallback_census_invalidates_the_run(self):
        proc, out = self.measure(census="scalar fallback RN")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn('packmma census ["scalar fallback 4"]', (out / "summary.md").read_text())

    def test_a_refused_measurement_exits_3(self):
        proc, out = self.measure(exit_code=3)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        summary = (out / "summary.md").read_text()
        self.assertIn("REFUSED CELLS", summary)
        self.assertNotIn("INVALID CELLS", summary)

    def test_a_failed_check_outranks_a_refusal_in_another_case(self):
        # The refusing case runs first: its 3 must not stand over the second case's census
        # failure, whichever order they come in.
        proc, out = self.measure(census="scalar fallback RN", refuse_n="32",
                                 cases=("f16:32:8:16:3:4:6", "f16:64:8:16:3:4:6"))
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        summary = (out / "summary.md").read_text()
        self.assertIn("REFUSED CELLS", summary)
        self.assertIn('f16-64-8-16-3-rn4-rn6 cell 0 (rn4): packmma census', summary)



CELLS = HERE / "cells.py"
CONFIG_RECORDS = HERE / "config_records.py"


def record_line(key, value, source, origin):
    """One `log_config_json` record, as a stub prints it: `source` and `origin` as the library
    reports them (a non-default source names its origin)."""
    return json.dumps({"ocannl_config": 1, "key": key, "value": value, "source": source,
                       "layer": source, "profile": None, "origin": origin},
                      separators=(",", ":"))


@POSIX_ONLY
class CellCommandTest(StubRunTest):
    """`cells.py cell`: one hermetic, supervised, configuration-checked cell, and its exit status."""

    BODY = """
import os
print("env BENCH_TUNE=%s seed=%s OMP=%s" % (os.environ.get("BENCH_TUNE"),
      os.environ.get("OCANNL_FIXED_STATE_FOR_INIT"), os.environ.get("OMP_NUM_THREADS")))
print("cwd " + os.getcwd())
if "env-record" in positionals:
    print(ENV_RECORD, file=sys.stderr)
if "sleep" in positionals:
    import time; time.sleep(0.9)
if "kill" in positionals:
    import signal; os.kill(os.getpid(), signal.SIGKILL)
for p in positionals:
    if p.startswith("exit"):
        sys.exit(int(p[4:]))
"""

    def setUp(self):
        super().setUp()
        self.exe = self.dir / "bench"
        write_stub(self.exe, self.BODY.replace("ENV_RECORD", repr(record_line(
            "log_level", "1", "environment", "OCANNL_LOG_LEVEL=1"))))

    def cell(self, *positionals, options=(), assignments=(), name="cell", env=None):
        d = self.dir / name
        proc = self.run_tool([sys.executable, str(CELLS), "cell", "--backend", "cc", "--dir",
                              str(d), *options, "--", *assignments, str(self.exe),
                              "--ocannl_fixed_state_for_init=1", *positionals], env)
        rec = json.loads((d / "cell.json").read_text()) if (d / "cell.json").exists() else None
        return proc, d, rec

    def test_a_clean_cell_runs_hermetic_from_its_own_directory(self):
        proc, d, rec = self.cell(assignments=["BENCH_TUNE=1"], env={
            "OCANNL_FIXED_STATE_FOR_INIT": "9", "OMP_NUM_THREADS": "3",
            "CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = (d / "stdout").read_text()
        # The stated runner mode reached the cell, the ambient setting and OpenMP control did not.
        self.assertIn("env BENCH_TUNE=1 seed=None OMP=None", out)
        self.assertIn(f"cwd {d}", out)
        self.assertTrue(rec["ok"], rec["problems"])
        self.assertEqual(rec["argv"][:4], [str(self.exe), "--ocannl_backend=cc",
                                           "--ocannl_no_config_file=true",
                                           "--ocannl_log_config_json=true"])
        self.assertEqual(rec["env"], {"BENCH_TUNE": "1"})
        self.assertEqual(rec["config"]["fixed_state_for_init"]["origin"],
                         "--ocannl_fixed_state_for_init=1")
        self.assertIn("OCANNL_FIXED_STATE_FOR_INIT", rec["cleared"])
        self.assertIn("OMP_NUM_THREADS", rec["cleared"])
        self.assertEqual(rec["runtime_controls"].get("CUDA_VISIBLE_DEVICES"), "0")

    def test_the_streams_go_where_the_driver_names(self):
        proc, d, rec = self.cell(options=["--stdout", str(self.dir / "c.out"), "--stderr",
                                          str(self.dir / "c.err")])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("cwd ", (self.dir / "c.out").read_text())
        self.assertIn('"key":"fixed_state_for_init"', (self.dir / "c.err").read_text())
        self.assertFalse((d / "stdout").exists())
        self.assertEqual(rec["stdout"], str(self.dir / "c.out"))

    def test_an_expectation_the_records_do_not_state_fails_the_cell(self):
        for expect, why in (("fixed_state_for_init=2", "ocannl_fixed_state_for_init resolved to '1'"),
                            ("debug_log_from_routines=true",
                             "ocannl_debug_log_from_routines was never resolved")):
            with self.subTest(expect=expect):
                proc, _, rec = self.cell(options=["--expect", expect], name=expect)
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertFalse(rec["ok"])
                self.assertTrue(any(why in p for p in rec["problems"]), rec["problems"])

    def test_a_setting_from_the_environment_fails_a_cell_that_exited_0(self):
        proc, _, rec = self.cell("env-record")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("ocannl_log_level came from the environment", rec["problems"])

    def test_the_exit_status_says_what_happened(self):
        for args, options, want in ((["exit7"], [], 7), (["exit3"], [], 3),
                                    (["exit3", "env-record"], [], 1),
                                    (["kill"], [], 137),
                                    (["sleep"], ["--timeout", "0.2"], 124),
                                    (["sleep"], ["--timeout", "0.2", "--expect",
                                                 "debug_log_from_routines=true"], 1)):
            with self.subTest(args=args, options=options):
                proc, _, rec = self.cell(*args, options=options, name="-".join(args + options))
                self.assertEqual(proc.returncode, want, proc.stdout + proc.stderr)
                self.assertFalse(rec["ok"])
        _, _, rec = self.cell("exit3", name="refused")
        self.assertTrue(rec["refused"])

    def test_a_survivor_stops_with_125(self):
        # A group not proven gone is the one fault a driver must stop on, so both front ends that
        # a shell driver calls name it apart (a real survivor is test_cell_group's to stage).
        import bin_cells
        import cell_group
        import cells

        def survive(**_):
            raise cell_group.CleanupFailed("cells: child group 1 cleanup alive")

        with unittest.mock.patch.object(cells, "run_one", survive), \
                unittest.mock.patch.object(cells, "install_cancellation"), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cells.cell_main(["--backend", "cc", "--dir", str(self.dir / "s1"),
                                              "--", str(self.exe)]), 125)
        with unittest.mock.patch.object(cells, "run_cells", survive), \
                unittest.mock.patch.object(cells, "install_cancellation"), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bin_cells.main(["--exe", str(self.exe), "--backend", "cc", "--out",
                                             str(self.dir / "s2"), "--rounds", "2"]), 125)

    def test_usage_errors_run_nothing(self):
        proc, d, _ = self.cell(assignments=["OCANNL_LOG_LEVEL=1"], name="stated")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("a library setting is stated on the command line", proc.stderr)
        self.assertFalse(d.exists())
        (self.dir / "busy").mkdir()
        (self.dir / "busy" / "left").write_text("x")
        proc, _, _ = self.cell(name="busy")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("is not an empty directory", proc.stderr)


@POSIX_ONLY
class ConfigRecordsExpectTest(StubRunTest):
    """`config_records.py expect`: the records check over a stderr a driver kept itself."""

    def expect(self, lines, *words):
        err = self.dir / "err"
        err.write_text("".join(line + "\n" for line in lines))
        return self.run_tool([sys.executable, "-I", str(CONFIG_RECORDS), "expect", str(err),
                              *words])

    def test_it_holds_only_for_command_line_values(self):
        base = ["chatter",
                record_line("backend", "hip", "commandline", "--ocannl_backend=hip"),
                record_line("autotune_timing", "queued", "commandline",
                            "--ocannl_autotune_timing=queued"),
                record_line("log_level", "0", "config_file", "/x/ocannl_config")]
        proc = self.expect(base, "backend=hip", "autotune_timing=queued")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        for words, why in ((["backend=cuda"], "ocannl_backend resolved to 'hip'"),
                           (["autotune_timing=isolated"], "not to 'isolated'"),
                           (["log_level=0"], "from config_file"),
                           (["profile=x"], "ocannl_profile was never resolved")):
            with self.subTest(words=words):
                proc = self.expect(base, *words)
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertIn(why, proc.stdout)
        proc = self.expect(base + [record_line("debug_log_from_routines", "true", "environment",
                                               "OCANNL_DEBUG_LOG_FROM_ROUTINES=true")],
                           "backend=hip")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("ocannl_debug_log_from_routines came from the environment", proc.stdout)
        proc = self.expect(base + ['{"ocannl_config":1,"key":'], "backend=hip")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("unreadable configuration records", proc.stdout)
        proc = self.expect(base, "backend")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

    def test_show_prints_each_keys_provenance(self):
        err = self.dir / "err"
        err.write_text(record_line("backend", "hip", "commandline", "--ocannl_backend=hip") + "\n")
        proc = self.run_tool([sys.executable, "-I", str(CONFIG_RECORDS), "show", str(err),
                              "backend", "autotune_timing"])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(proc.stdout.splitlines(), [
            "ocannl_backend: 'hip' (commandline --ocannl_backend=hip)",
            "ocannl_autotune_timing: (not resolved)"])


# The prose configuration trace (`Found <value>, <source> <origin>` lines) and the flag that turns it
# on. A driver reads the `log_config_json` records instead (gh-ocannl-1607).
PROSE_TRACE = re.compile(r"log_config_sourcing|Found\b[^\n]*\bcommandline")


def prose_trace_readers(root):
    """`file:line` of every driver source under `root` (its scripts and runners/, tests aside) that
    names the prose configuration trace."""
    paths = sorted({*root.glob("*.py"), *root.glob("*.sh"), *root.glob("runners/**/*.py"),
                    *root.glob("runners/**/*.sh")})
    found = []
    for path in paths:
        if path.name.startswith(("test_", "test-")):
            continue
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if PROSE_TRACE.search(line):
                found.append(f"{path.relative_to(root).as_posix()}:{i}")
    return found


class ProseTraceScanTest(unittest.TestCase):
    def test_no_driver_reads_the_prose_configuration_trace(self):
        self.assertEqual(prose_trace_readers(HERE), [],
                         "read the log_config_json records (config_records.py) instead")
        # The two drivers that read the trace before gh-ocannl-1607 are in the population.
        for name in ("gh834_cells.sh", "gh1002_cells.py"):
            self.assertTrue((HERE / name).is_file(), name)

    def test_the_scan_finds_a_planted_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runners" / "x").mkdir(parents=True)
            (root / "a.sh").write_text('run --ocannl_log_config_sourcing=true\n')
            (root / "b.py").write_text('ok = "Found %s, commandline --ocannl_backend" % b\n')
            (root / "runners" / "x" / "c.sh").write_text('grep "^Found $b, commandline"\n')
            (root / "clean.sh").write_text('run --ocannl_log_config_json=true\n')
            (root / "test_d.py").write_text('"log_config_sourcing"\n')
            self.assertEqual(prose_trace_readers(root), ["a.sh:1", "b.py:1", "runners/x/c.sh:1"])


@POSIX_ONLY
class DriverTreeTest(StubRunTest):
    """A driver that finds its checkout from its own location, run from a scratch copy of benchmarks/
    (a git repository of its own) whose `_build` holds stub benches and whose `dune` builds nothing."""

    def setUp(self):
        super().setUp()
        dune = self.fakebin / "dune"
        dune.write_text("#!/bin/sh\nexit 0\n")
        dune.chmod(0o755)
        self.root = self.dir / "root"
        bench = self.root / "benchmarks"
        bench.mkdir(parents=True)
        (self.root / "scripts").mkdir()
        shutil.copy2(HERE.parent / "scripts/process-group.sh", self.root / "scripts/process-group.sh")
        for f in [*HERE.glob("*.py"), *HERE.glob("*.sh")]:
            shutil.copy2(f, bench / f.name)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for argv in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "tree"]):
            subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *argv], cwd=self.root,
                           env=env, check=True, capture_output=True, timeout=60)
        self.out = self.dir / "out"

    def stub(self, target, body):
        write_stub(self.root / "_build" / "default" / target, body)

    def driver(self, name, *args, env=None):
        return self.run_tool([box_gate.BASH, str(self.root / "benchmarks" / name), *args], env)


class Gh740DriverTest(DriverTreeTest):
    def test_each_leg_is_a_checked_cell(self):
        self.stub("benchmarks/runners/ocannl/bench_pad_waste.exe", """
import os
thr = next((a.split("=", 1)[1] for a in argv if "pad_waste_max" in a), "default")
print("padwaste: site=" + positionals[0] + " threshold=" + thr
      + " ambient=" + str(os.environ.get("OCANNL_AUTOTUNE_PAD_WASTE_MAX")))
""")
        proc = self.driver("gh740_cells.sh", str(self.out), "cc", "AB", "t=1,8,8,8",
                           env={"OCANNL_AUTOTUNE_PAD_WASTE_MAX": "3"})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("padwaste: site=t=1,8,8,8 threshold=inf ambient=None", proc.stdout)
        self.assertIn("padwaste: site=t=1,8,8,8 threshold=default ambient=None", proc.stdout)
        a = json.loads((self.out / "cc-1-A" / "cell.json").read_text())
        b = json.loads((self.out / "cc-2-B" / "cell.json").read_text())
        self.assertTrue(a["ok"] and b["ok"], (a["problems"], b["problems"]))
        self.assertEqual(a["config"]["autotune_pad_waste_max"]["value"], "inf")
        self.assertNotIn("autotune_pad_waste_max", b["config"])


class Gh1279DriverTest(DriverTreeTest):
    def measure(self, rev_exit):
        self.stub("bin/projection_shape_bench.exe", f"""
print("gh1279: " + " ".join(positionals))
print("crowned: x" if positionals[3] == "fwd" else "seeds only")
sys.exit(0 if positionals[3] == "fwd" else {rev_exit})
""")
        return self.driver("gh1279_cells.sh", str(self.out), "metal", "2", "2")

    def test_two_legs_and_their_verdict(self):
        proc = self.measure(0)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("gh1279: 2 2 outproj fwd both", proc.stdout)
        self.assertIn("crowned: x", proc.stdout)
        self.assertIn("gh1279: 2 2 outproj rev seeds", proc.stdout)
        for leg in ("psb_fwd", "psb_rev"):
            rec = json.loads((self.out / leg / "cell.json").read_text())
            self.assertTrue(rec["ok"], rec["problems"])
            self.assertEqual(rec["config"]["backend"]["value"], "metal")
            self.assertTrue((self.out / leg / "stdout").read_text().endswith("exit: 0\n"))

    def test_a_refused_leg_exits_3(self):
        proc = self.measure(3)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("leg psb_rev (rev seeds): exit: 3", proc.stdout)


# bench_conv as gh1557_cells.sh reads it: a process whose cache directory holds no crown searches
# and stores one; one whose cache holds it replays the forced arm.
CONV_BODY = """
import json, os
cache = next(a.split("=", 1)[1] for a in argv if a.startswith("--ocannl_autotune_cache_dir="))
arm = next(a.split("=", 1)[1] for a in argv if a.startswith("--ocannl_tune_ship_arm=")).upper()
replay = os.path.exists(os.path.join(cache, "crown"))
os.makedirs(cache, exist_ok=True)
open(os.path.join(cache, "crown"), "w").close()
arms = [{"arm": a, "state": "cache-replay" if replay else "searched", "best_ms": 2.0,
         "timings_contended": 0, "best_label": "grid"} for a in "AB"]
print(json.dumps({"framework": "ocannl", "searched": not replay,
                  "tune": {"shipped": arm, "arms": arms, "ship_substituted": False},
                  "step_ms": {"p10": 2.2, "p50": 2.4, "p90": 2.6},
                  "weights": os.environ.get("BENCH_TUNE_SCRATCH_WEIGHTS")}))
"""


class Gh1557DriverTest(DriverTreeTest):
    def test_a_search_then_two_replays_per_weight_set(self):
        self.stub("benchmarks/runners/ocannl/bench_conv.exe", CONV_BODY)
        fixture = self.dir / "cifar_resnet.safetensors"
        fixture.write_text("weights")
        proc = self.driver("gh1557_cells.sh", "--skip-box-check", str(fixture), str(self.out), "5",
                           "init")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        summary = (self.out / "summary.txt").read_text()
        self.assertIn("weights=init shipped=A state=cache-replay crown=2.00 p10=2.20", summary)
        self.assertIn("weights=init shipped=B state=cache-replay", summary)
        for p in ("search", "a", "b"):
            rec = json.loads((self.out / f"init-{p}" / "cell.json").read_text())
            self.assertTrue(rec["ok"], rec["problems"])
            self.assertEqual(rec["env"]["BENCH_TUNE_SCRATCH_WEIGHTS"], "init")
            self.assertEqual(rec["config"]["autotune_cache_dir"]["value"],
                             str(self.out / "cache-init"))

    def test_a_replay_that_searched_fails_the_run(self):
        # Every process searches: a cache that never holds the crown leaves no replay to pair.
        self.stub("benchmarks/runners/ocannl/bench_conv.exe",
                  CONV_BODY.replace('open(os.path.join(cache, "crown"), "w").close()', ""))
        fixture = self.dir / "cifar_resnet.safetensors"
        fixture.write_text("weights")
        proc = self.driver("gh1557_cells.sh", "--skip-box-check", str(fixture), str(self.out), "5",
                           "init")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("no crown/step pair", (self.out / "summary.txt").read_text())


# bench_gpt's untuned step as gh1253_flip_cells.sh reads it: the result line on stdout, the seeding
# census and plan derivation on stderr, both keyed on the gate.
FLIP_BODY = """
import json
gate = next(a.split("=", 1)[1] for a in argv if "sketch_min_reduction" in a)
print("seeded_default: plan derived in 0.25 s", file=sys.stderr)
print(f"seeded_default: {3 if gate == 'auto' else 2} of 9 segments seeded [gate {gate}]",
      file=sys.stderr)
print(json.dumps({"framework": "ocannl", "step_ms": {"p10": 4, "p50": 5 if gate == "auto" else 6,
                  "p90": 7}, "losses": [1.0, 0.5]}))
"""


class Gh1253FlipDriverTest(DriverTreeTest):
    def test_the_two_gates_in_abba_order_with_their_census(self):
        self.stub("benchmarks/runners/ocannl/bench_gpt.exe", FLIP_BODY)
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        (fixtures / "fx.safetensors").write_text("weights")
        proc = self.driver("gh1253_flip_cells.sh", "--skip-box-check", "hip", str(self.out),
                           env={"FIXTURE_DIR": str(fixtures), "FIXTURES": "fx",
                                "OCANNL_GPU_DEFAULT_SKETCH_MIN_REDUCTION": "0"})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines = (self.out / "summary.txt").read_text().splitlines()
        self.assertEqual(lines, [
            "fx gate=32 pass=1 step_p50_ms=6 plan_derivation=[0.25 s] "
            "census=[seeded_default: 2 of 9 segments seeded [gate 32]]",
            "fx gate=auto pass=1 step_p50_ms=5 plan_derivation=[0.25 s] "
            "census=[seeded_default: 3 of 9 segments seeded [gate auto]]",
            "fx gate=auto pass=2 step_p50_ms=5 plan_derivation=[0.25 s] "
            "census=[seeded_default: 3 of 9 segments seeded [gate auto]]",
            "fx gate=32 pass=2 step_p50_ms=6 plan_derivation=[0.25 s] "
            "census=[seeded_default: 2 of 9 segments seeded [gate 32]]"])
        records = self.records(self.out / "fx")
        self.assertTrue(all(r["config"]["backend"]["value"] == "hip" for r in records))
        self.assertEqual({r["env"]["BENCH_TUNE"] for r in records}, {"0"})

    def test_a_failed_cell_fails_the_driver(self):
        self.stub("benchmarks/runners/ocannl/bench_gpt.exe",
                  FLIP_BODY + "sys.exit(1 if gate == 'auto' else 0)\n")
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        (fixtures / "fx.safetensors").write_text("weights")
        proc = self.driver("gh1253_flip_cells.sh", "--skip-box-check", "hip", str(self.out),
                           env={"FIXTURE_DIR": str(fixtures), "FIXTURES": "fx"})
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn('fx gate=auto pass=1 FAILED: "exited 1"', (self.out / "summary.txt").read_text())


# bin/schedule_bench as gh1051_bf16_ab.sh reads it: five timing lines whose checksums are the f32
# control's exact answer, near it for the wide and the narrow serial legs, and a gross but shared
# value for the narrow tensorized pair; with debug files on, one generated .hip under the cell's
# build_files/<prefix>/ that the bf16 policy does not reach at f32.
SCHEDULE_BODY = """
import json, os
opt = {a.split("=", 1)[0][len("--ocannl_"):]: a.split("=", 1)[1] for a in argv if "=" in a}
narrow, prec = opt["bf16_arithmetic"] == "true", opt["default_prec"]
if opt.get("output_debug_files_in_build_directory") == "true":
    d = os.path.join("build_files", opt["build_files_prefix"])
    os.makedirs(d)
    open(os.path.join(d, "mm.hip"), "w").write("kernel " + prec + "\\n")
for v in ("parallel", "smem", "regtile", "mma_pd1", "mma_pd2"):
    mma = v.startswith("mma")
    if prec == "single":
        chk = "100/200"
    elif narrow and mma:
        chk = "60/120"
    else:
        chk = "102/204" if narrow else "101/202"
    census = "MMA_CENSUS" if mma and prec == "bfloat16" else "no tile MMA"
    print(f"{v:<10} 1.000 ms  2.00 GFLOP/s  (spot [65] 1.0, chk {chk}, agrees (0 of 4))  [{census}] [grid: no Grid loop]")
    print("bench-result: " + json.dumps({"variant": v, "seconds": 0.001, "check": "unchecked",
                                          "check_note": "rounding expected"}))
"""


@POSIX_ONLY
class Gh1051AbDriverTest(StubRunTest):
    def measure(self, census="Mma_intrinsics 2"):
        root = self.dir / "root"
        write_stub(root / "_build/default/bin/schedule_bench.exe",
                   SCHEDULE_BODY.replace("MMA_CENSUS", census))
        dune = self.fakebin / "dune"
        dune.write_text("#!/bin/sh\nexit 0\n")
        dune.chmod(0o755)
        (self.dir / "tmp").mkdir()
        proc = self.run_tool([box_gate.BASH, str(HERE / "gh1051_bf16_ab.sh"), str(root), "stub",
                              "4", "64"], {"TMPDIR": str(self.dir / "tmp"),
                                           "OCANNL_BF16_ARITHMETIC": "auto"})
        raw = Path(re.search(r"raw=(\S+)", proc.stdout).group(1))
        return proc, raw

    def test_four_interleaved_arms_validated_and_tabled(self):
        proc, raw = self.measure()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # One space exactly: the driver strips the padding macOS's wc -l puts before its count.
        self.assertRegex(proc.stdout,
                         r"drift control: 1 generated f32 kernels byte-identical across arms")
        self.assertIn("| stub | 64 | mma_pd1 |", proc.stdout)
        records = self.records(raw / "n64")
        self.assertEqual(len(records), 16)
        self.assertEqual({r["arm"] for r in records},
                         {"false-bfloat16", "true-bfloat16", "false-single", "true-single"})
        # The ambient `auto` reached no cell: each ran the arm its label names.
        self.assertTrue(all(r["config"]["bf16_arithmetic"]["value"] == r["arm"].split("-")[0]
                            for r in records))

    def test_a_tensorized_cell_that_fell_back_invalidates_the_run(self):
        proc, _ = self.measure(census="scalar fallback")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("mma_pd1 did not tensorize [scalar fallback]", proc.stdout)


if __name__ == "__main__":
    unittest.main()
