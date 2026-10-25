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

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
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

    def run_tool(self, argv, env=None, timeout=300):
        return subprocess.run(argv, text=True, capture_output=True, cwd=self.dir, timeout=timeout,
                              env=dict(os.environ, **(env or {})))

    def records(self, run_dir):
        return [json.loads(l) for l in (run_dir / "cells.jsonl").read_text().splitlines()]


class AbDriverTest(StubRunTest):
    """ab.py and gh1343_ab.sh: two revisions, each built once, measured as two interleaved arms."""

    def setUp(self):
        super().setUp()
        fakebin = self.dir / "fakebin"
        fakebin.mkdir()
        step_stub = STUB.format(python=sys.executable, body=textwrap.dedent(STEP_BODY))
        dune = fakebin / "dune"
        dune.write_text(FAKE_DUNE.format(python=sys.executable, stub=step_stub))
        dune.chmod(0o755)
        self.env = {"PATH": f"{fakebin}{os.pathsep}{os.environ.get('PATH', '')}",
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
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
# the packed rows' census, and the bench-result lines it pairs. CENSUS and EXIT vary per test.
GEBP_BODY = '''
import json
rn = next(a.split("=")[1] for a in argv if a.startswith("--rn="))
print(f"GEBP n=64, 3 repeats, blocking bm=8 bk=16, register tile 4x{{rn}} rn{{rn}} (requested)")
secs = 0.001 * int(rn)
census = "CENSUS".replace("RN", rn)
print(f"packmma       1.0 ms     5.00 GFLOP/s  [{{census}}]")
print(f"packmma_par   1.0 ms     9.00 GFLOP/s  [{{census}}]")
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


if __name__ == "__main__":
    unittest.main()
