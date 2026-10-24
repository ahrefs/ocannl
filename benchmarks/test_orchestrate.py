import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat as stat_module
import subprocess
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
from pathlib import Path

import bench_venv
import box_gate
import fixture_digest
import cell_group
import display_safe
import gh1161_sites
import orchestrate
import strict_json
from runners import bench_common
from test.test_cell_group import CellGroupTest  # imported so unittest.main includes shared tests
from test.test_cell_group import kill_the_group_on_cleanup, publish_pid

HERE = Path(__file__).resolve().parent


def result(framework, backend, variant, losses, precision="f32"):
    return {
        "workload": "mlp_wide",
        "framework": framework,
        "backend": backend,
        "variant": variant,
        "precision": precision,
        "losses": losses,
    }


def cell(framework, backend, variant, losses, precision="f32", p50=1.0):
    """A result with the fields `report` reads, not just the parity ones."""
    r = result(framework, backend, variant, losses, precision=precision)
    r.update(
        {
            "step_ms": {"p10": p50, "p50": p50, "p90": p50},
            "queued_step_ms": p50,
            "compile_s": 1.5,
        }
    )
    return r


# JSON as every other reader has it: the loader the result-line readers use, not a second copy.
strict_loads = strict_json.loads


class ParityCheckTest(unittest.TestCase):
    def test_precision_tolerances_match_measured_headroom(self):
        self.assertEqual(orchestrate.PARITY_TOL_PRECISION, {"bf16": 4e-3, "f16": 2e-3})

    def test_flat_candidate_fails_even_when_within_bf16_tolerance(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        flat = result(
            "ocannl", "metal", "default", [2.3026, 2.3026, 2.3026], precision="bf16"
        )

        orchestrate.parity_check([ref, flat])

        self.assertLess(flat["parity_max_rel"], orchestrate.PARITY_TOL_PRECISION["bf16"])
        self.assertFalse(flat["parity_loss_moved"])
        self.assertEqual(flat["parity"], "FAIL")

    def test_moving_candidate_within_tolerance_passes(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        moving = result(
            "ocannl", "metal", "tuned", [2.3025, 2.3012, 2.3004], precision="bf16"
        )

        orchestrate.parity_check([ref, moving])

        self.assertTrue(moving["parity_loss_moved"])
        self.assertEqual(moving["parity"], "PASS")


class DivergenceTest(unittest.TestCase):
    """gh-ocannl-676: a cell whose training diverged ran, and its trajectory is the finding.

    The three runners now all emit `null` for a number JSON cannot express, and the OCANNL one
    used to emit OCaml's `nan`, whose line `json.loads` refuses -- so the cell was reported as a
    broken runner and the loss vector that shows the divergence was thrown away. Here the verdict
    is DIVERGED: a parity failure that names its cause.
    """

    def test_null_loss_is_a_diverged_cell_not_a_missing_one(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = result("ocannl", "metal", "default", [2.3026, None, None], precision="f16")

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)
        self.assertEqual(ref["parity"], "REF")

    def test_nan_from_a_python_runner_reads_the_same(self):
        # Python's json.loads accepts its own `NaN` extension, so an older result file can still
        # deliver one; it is the same fact as a null.
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = result("tinygrad", "metal", "jit", [2.3026, float("nan"), float("inf")])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)

    def test_a_diverged_cell_is_not_reported_as_stationary(self):
        # Its finite prefix moved; and with fewer than two finite steps there is nothing to say
        # about movement, which is reported as divergence rather than as a flat trajectory.
        self.assertTrue(orchestrate.loss_moved([2.3026, 1.5, None]))
        self.assertFalse(orchestrate.loss_moved([2.3026, None, None]))

    def test_the_finite_prefix_is_still_compared(self):
        ref = result("pytorch", "cpu", "eager", [2.0, 2.0, 2.0])
        blown = result("ocannl", "cc", "default", [2.0, None, None])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity_max_rel"], 0.0)

    def test_a_finite_value_after_the_divergence_is_not_evidence(self):
        # A loss that comes back finite after a NaN is whatever the arithmetic settled on, not
        # drift: comparing it would put a number on the DIVERGED row that its own contract calls
        # meaningless, and reading it as movement would say the trajectory moved when what it did
        # was blow up.
        ref = result("pytorch", "cpu", "eager", [2.0, 2.0, 2.0])
        blown = result("ocannl", "cc", "default", [2.0, None, 9.0])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity_max_rel"], 0.0)
        self.assertFalse(blown["parity_loss_moved"])
        self.assertEqual(orchestrate.finite_prefix([2.0, None, 9.0]), [2.0])

    def test_the_reference_prefix_bounds_the_comparison_too(self):
        ref = result("pytorch", "cpu", "eager", [2.0, float("nan"), 2.0])
        other = result("ocannl", "cc", "default", [2.0, 5.0, 5.0])

        orchestrate.parity_check([ref, other])

        self.assertEqual(other["parity_max_rel"], 0.0)

    def test_a_diverged_reference_is_no_reference(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, None, None])
        other = result("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])

        orchestrate.parity_check([ref, other])

        self.assertEqual(ref["parity"], "DIVERGED")
        self.assertEqual(other["parity"], "NO-REF")

    def test_report_names_the_divergence_and_writes_parseable_json(self):
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = cell(
            "ocannl", "metal", "default", [2.3026, None, None], precision="f16", p50=None
        )
        orchestrate.parity_check([ref, blown])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, blown], out)
            text = (out / "report.md").read_text()
            rows = [
                strict_loads(line) for line in (out / "results.jsonl").read_text().splitlines()
            ]

        self.assertIn("DIVERGED", text)
        self.assertIn("loss non-finite from step 1", text)
        # A time the runner never measured prints as n/a rather than crashing the report.
        self.assertIn("n/a", text)
        self.assertEqual([r["parity"] for r in rows], ["REF", "DIVERGED"])

    def test_emit_writes_null_for_a_non_finite_number(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bench_common.emit({"losses": [1.0, float("nan")], "step_ms": {"p50": float("inf")}})

        parsed = strict_loads(buf.getvalue().strip())
        self.assertEqual(parsed, {"losses": [1.0, None], "step_ms": {"p50": None}})


# A result line in each spelling of a non-finite number that `json.loads` accepts: its two
# constants (and the negated one) and an overflowing literal. Each spelling sits in its own line, so
# a loader that refuses only some of them fails on the others.
NON_FINITE_LINES = {
    "NaN": '{"losses":[2.5,NaN],"step_ms":{"p50":1.0}}',
    "Infinity": '{"losses":[2.5,1.75],"step_ms":{"p50":Infinity}}',
    "-Infinity": '{"losses":[-Infinity,1.75],"step_ms":{"p50":1.0}}',
    "1e999": '{"losses":[2.5,1.75],"step_ms":{"p50":1e999}}',
}


def refuses_every_non_finite_line(loads):
    """Whether `loads` raises `ValueError` on every line of NON_FINITE_LINES."""
    for line in NON_FINITE_LINES.values():
        try:
            loads(line)
        except ValueError:
            continue
        return False
    return True


class StrictJsonTest(unittest.TestCase):
    """The one loader every reader of a runner's result line goes through (benchmarks/strict_json.py,
    from lukstafi/ocannl-staging#1138): Python's `json.loads` reads NaN, Infinity and 1e999, which
    no other consumer of results.jsonl does."""

    def mutant(self, name, old, new):
        """strict_json with `old` replaced by `new` in its source, loaded as a module of its own."""
        source = (HERE / "strict_json.py").read_text()
        self.assertIn(old, source, f"mutant {name} no longer applies")
        module = types.ModuleType(f"strict_json_{name}")
        exec(compile(source.replace(old, new, 1), f"<strict_json {name}>", "exec"), module.__dict__)
        return module

    def test_the_lines_are_the_trap(self):
        # Precondition: the stdlib reads every line, and each carries a non-finite number.
        for spelled, line in NON_FINITE_LINES.items():
            with self.subTest(spelled=spelled):
                parsed = json.loads(line)
                numbers = parsed["losses"] + [parsed["step_ms"]["p50"]]
                self.assertFalse(all(orchestrate.finite(x) for x in numbers))

    def test_every_non_finite_spelling_is_refused(self):
        for spelled, line in NON_FINITE_LINES.items():
            with self.subTest(spelled=spelled):
                with self.assertRaises(strict_json.NonFinite) as caught:
                    strict_json.loads(line)
                self.assertIn(spelled.lstrip("-"), str(caught.exception))
                # A ValueError, as JSONDecodeError is, so a reader's existing handler covers it.
                self.assertIsInstance(caught.exception, ValueError)
        self.assertTrue(refuses_every_non_finite_line(strict_json.loads))

    def test_finite_json_reads_as_the_stdlib_reads_it(self):
        line = '{"losses":[2.5,null,-0.0,1e308,3],"step_ms":{"p50":12},"note":"NaN Infinity"}'
        self.assertEqual(strict_json.loads(line), json.loads(line))

    def test_mutants_that_drop_either_hook_accept_a_line(self):
        # Mutation control: the refusal above holds only while both hooks are in place.
        hooks = "parse_constant=_reject_constant, parse_float=_finite_float"
        for name, new in [
            ("no_hooks", ""),
            ("no_constant_hook", "parse_float=_finite_float"),
            ("no_float_hook", "parse_constant=_reject_constant"),
        ]:
            with self.subTest(mutant=name):
                mutant = self.mutant(name, hooks, new)
                self.assertFalse(refuses_every_non_finite_line(mutant.loads))
        self.assertFalse(refuses_every_non_finite_line(json.loads))


class ResultFieldReaderTest(unittest.TestCase):
    """strict_json's `last_result`/`field` and the `field` command a shell driver asks instead of
    grepping a result line's text (gh-ocannl-1355; gh-ocannl-1253's driver read `step_ms` as a
    number where the runner prints a `{p10,p50,p90}` object)."""

    LINES = ('noise\n{"step_ms":{"p10":9,"p50":9,"p90":9}}\n'
             '{"step_ms":{"p10":1,"p50":2.5,"p90":3},"tune":{"shipped":"A","searches":1}}\n'
             'trailing noise\n')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def ask(self, text, key):
        path = self.dir / "cell.out"
        path.write_text(text)
        return subprocess.run([sys.executable, str(HERE / "strict_json.py"), "field", str(path), key],
                              text=True, capture_output=True, timeout=60)

    def test_the_last_result_line_and_its_nested_fields(self):
        record = strict_json.last_result(self.LINES)
        self.assertEqual(strict_json.field(record, "step_ms.p50"), 2.5)
        self.assertEqual(strict_json.field(record, "tune.shipped"), "A")
        self.assertIsNone(strict_json.last_result("no record here\n"))
        for key in ("nope", "step_ms.p50.x", "tune.arms"):
            with self.subTest(key=key):
                with self.assertRaises(KeyError) as caught:
                    strict_json.field(record, key)
                self.assertIn("the result line has no", caught.exception.args[0])

    def test_the_command_prints_a_field_or_names_why_not(self):
        for key, shown in (("step_ms.p50", "2.5"), ("tune.shipped", "A"),
                           ("step_ms", '{"p10": 1, "p50": 2.5, "p90": 3}')):
            with self.subTest(key=key):
                proc = self.ask(self.LINES, key)
                self.assertEqual((proc.returncode, proc.stdout), (0, shown + "\n"), proc.stderr)
        for text, key, why in ((self.LINES, "step_ms.p99", "has no step_ms.p99"),
                               ("noise\n", "step_ms.p50", "no result line"),
                               ('{"step_ms":{"p50":NaN}}\n', "step_ms.p50", "NaN"),
                               ('{"step_ms":{"p50":1e999}}\n', "step_ms.p50", "overflows")):
            with self.subTest(why=why):
                proc = self.ask(text, key)
                self.assertEqual((proc.returncode, proc.stdout), (1, ""), proc.stderr)
                self.assertIn(why, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)


# A shell driver line that greps a quoted JSON key out of a file: a hand parse of a result line.
SHELL_JSON_KEY_GREP = re.compile(r'\bgrep\b.*"[A-Za-z_][A-Za-z0-9_]*":')


class ShellFieldGrepInventoryTest(unittest.TestCase):
    """No shell driver reads a result-line field with a text pattern (gh-ocannl-1355): it asks
    `strict_json.py field` for it, which parses the line."""

    def test_no_driver_greps_a_json_field(self):
        sources = ResultLineReaderInventoryTest.shipped_sources()
        self.assertIn("gh1253_flip_cells.sh", sources)
        found = [(name, line.strip()) for name, text in sorted(sources.items())
                 if name.endswith(".sh") for line in text.splitlines()
                 if SHELL_JSON_KEY_GREP.search(line)]
        self.assertEqual(found, [], "read the field with `python3 strict_json.py field FILE KEY`")

    def test_the_scan_finds_the_shapes_it_replaced(self):
        for planted in (
            """ms=$({ grep -o '"step_ms":{[^}]*}' "$out/$name.out" || true; } | tail -1)""",
            """shipped=$(grep -ho '"shipped":"[AB]"' "$out/search.out" | head -1)""",
            'grep -ho \'"p50":[0-9.]*\' "$out/replay2.out"',
        ):
            with self.subTest(planted=planted):
                self.assertTrue(SHELL_JSON_KEY_GREP.search(planted))
        # A presence check of a key, and a key in a source file's text, are not field reads.
        for clean in ("""elif ! grep -q '"step_ms"' "$f"; then""",
                      'grep -q \'"tune_ship_arm"\' "$t/arrayjit/lib/utils.ml"'):
            with self.subTest(clean=clean):
                self.assertFalse(SHELL_JSON_KEY_GREP.search(clean))


class NonFiniteResultLineReadersTest(unittest.TestCase):
    """Each reader of a runner's result line refuses one that spells a non-finite number, as
    `strict_json` does, instead of taking it as a measurement."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_run_cell_fails_the_cell_and_names_the_line(self):
        for spelled, line in NON_FINITE_LINES.items():
            with self.subTest(spelled=spelled):
                cell = [sys.executable, "-c", "import sys; print(sys.argv[1])", line]
                with contextlib.redirect_stdout(io.StringIO()):
                    result, note = orchestrate.run_cell(f"non-finite {spelled}", cell, timeout=60)
                self.assertIsNone(result)
                self.assertIn("its result line is not strict JSON", note)
        # Negative control: the same cell with a finite line is a result.
        finite = '{"losses":[2.5,null],"step_ms":{"p50":1.0},"compile_s":0.5}'
        cell = [sys.executable, "-c", "import sys; print(sys.argv[1])", finite]
        with contextlib.redirect_stdout(io.StringIO()):
            result, note = orchestrate.run_cell("finite", cell, timeout=60)
        self.assertIsNone(note)
        self.assertEqual(result["losses"], [2.5, None])

    def test_a_checkpoint_spelling_a_non_finite_loss_is_skipped(self):
        prefix = orchestrate.CHECKPOINT_PREFIX
        good = prefix + '{"record":"checkpoint","losses":[2.5]}'
        bad = prefix + '{"record":"checkpoint","losses":[2.5,NaN]}'
        self.assertEqual(orchestrate.last_checkpoint(good + "\n" + bad)["losses"], [2.5])

    def test_gh998_result_refuses_and_skips_only_noise(self):
        import gh998_cells

        for spelled, line in NON_FINITE_LINES.items():
            with self.subTest(spelled=spelled):
                path = self.dir / f"gh998-{spelled}.stdout"
                path.write_text("noise\n" + line + "\n")
                with self.assertRaisesRegex(RuntimeError, "not strict JSON"):
                    gh998_cells.result(path)
        # A non-finite JSON line that is no result line is noise, as any other non-result is.
        path = self.dir / "gh998-noise.stdout"
        path.write_text('{"probe":NaN}\n{"losses":[2.5],"step_ms":{"p50":1.0}}\n')
        self.assertEqual(gh998_cells.result(path)["losses"], [2.5])

    def test_gh1002_result_line_is_no_result(self):
        import gh1002_cells

        fixture = "gpt2_mini_train"
        steps = gh1002_cells.parity_steps(fixture)
        losses = ",".join(["2.5"] * steps)
        base = self.dir / f"cc__{fixture}__A__r0"
        Path(str(base) + ".status").write_text("0\n")
        # The losses are finite and complete, so only the non-finite timing can refuse the line.
        for spelled in ("NaN", "Infinity", "1e999"):
            with self.subTest(spelled=spelled):
                Path(str(base) + ".out").write_text(
                    '{"losses":[%s],"step_ms":{"p50":%s}}\n' % (losses, spelled))
                self.assertIsNone(gh1002_cells.result_line(base))
        Path(str(base) + ".out").write_text('{"losses":[%s],"step_ms":{"p50":1.0}}\n' % losses)
        self.assertEqual(gh1002_cells.result_line(base)["step_ms"]["p50"], 1.0)


class DriverResultReaderModulesTest(unittest.TestCase):
    """The shell drivers' result-line readers, run by path the way their drivers run them -- from a
    scratch cwd holding no copy of their imports -- and their wiring into the drivers
    (gh-ocannl-1365: they were heredocs these tests extracted from the shell text). Each refuses
    NaN, Infinity and 1e999 through strict_json with a finding rather than a traceback."""

    SPELLINGS = ("NaN", "Infinity", "1e999")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def module(self, name, *argv):
        return subprocess.run(
            [sys.executable, str(HERE / name), *map(str, argv)],
            text=True, capture_output=True, cwd=self.dir, timeout=60)

    def gh612_cell(self, root, label, rep, losses, kind="search.out"):
        cell = root / label / f"r{rep}"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / kind).write_text("noise\n" + '{"losses":[%s]}\n' % ",".join(losses))

    def test_gh612_parity_fails_on_the_line_not_with_a_traceback(self):
        for spelled in self.SPELLINGS:
            with self.subTest(spelled=spelled):
                root = self.dir / spelled
                self.gh612_cell(root, "cell", 1, ["2.5", spelled])
                proc = self.module("gh612_parity.py", root, 64, 0, "", 2)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("the result line is not strict JSON", proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)

    def test_gh612_parity_passes_agreeing_cells_and_fails_a_divergent_one(self):
        root = self.dir / "agree"
        losses = ["2.5", "2.25"]
        for label in ("base", "treat"):
            self.gh612_cell(root, label, 1, losses)
            self.gh612_cell(root, label, 1, losses, kind="replay2.out")
        proc = self.module("gh612_parity.py", root, 64, 2, "base/r1 treat/r1", 2)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("4 records (2 search + 2 pass-2 replay), 1 distinct loss sequences", proc.stdout)
        self.assertIn("parity gate PASSED", proc.stdout)
        # Two-sided: the same tree with one cell a whole unit off, and one required cell absent.
        self.gh612_cell(root, "treat", 1, ["3.5", "2.25"])
        proc = self.module("gh612_parity.py", root, 64, 2, "base/r1 treat/r1 extra/r1", 2)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("PARITY GATE FAILED: loss divergence", proc.stderr)
        self.assertIn("PARITY GATE FAILED: missing required cells: extra/r1", proc.stderr)
        self.assertNotIn("parity gate PASSED", proc.stdout)

    def test_gh612_parity_subcommand_runs_the_module(self):
        # The driver end to end: `parity` touches no tree, fixture or device, so it runs here.
        root = self.dir / "driven"
        for label in ("a", "b"):
            self.gh612_cell(root, label, 1, ["2.5", "2.25"])
        proc = subprocess.run(
            [box_gate.BASH, str(HERE / "gh612_cells.sh"), "parity"], text=True, capture_output=True,
            cwd=self.dir, timeout=120,
            env=dict(os.environ, OUT_ROOT=str(root), EXPECT_STEPS="2"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("parity gate PASSED", proc.stdout)
        # Negative control: the step count the driver passes through is the gate's.
        proc = subprocess.run(
            [box_gate.BASH, str(HERE / "gh612_cells.sh"), "parity"], text=True, capture_output=True,
            cwd=self.dir, timeout=120, env=dict(os.environ, OUT_ROOT=str(root)))
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("expected EXPECT_STEPS=8", proc.stderr)

    GH728_LINE = ('{"framework":"ocannl","searched":%s,"step_ms":{"p50":%s},'
                  '"tune":{"searches":%d,"replays":%d,"no_searches":0,"arms":[]}}\n')

    def test_gh728_protocol_check_refuses_the_line(self):
        for spelled in self.SPELLINGS:
            with self.subTest(spelled=spelled):
                path = self.dir / f"gh728-{spelled}.out"
                path.write_text(self.GH728_LINE % ("true", spelled, 1, 0) + "exit: 0\n")
                proc = self.module("gh728_protocol.py", path, "search")
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertIn("a result line is not strict JSON", proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
        # Negative control: the same line with a finite timing is a search pass.
        path = self.dir / "gh728-finite.out"
        path.write_text(self.GH728_LINE % ("true", "1.5", 1, 0) + "exit: 0\n")
        proc = self.module("gh728_protocol.py", path, "search")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("protocol: search pass", proc.stdout)

    def test_gh728_a_replay_that_searched_is_no_replay(self):
        replay = self.dir / "replay.out"
        replay.write_text(self.GH728_LINE % ("false", "1.5", 0, 2))
        proc = self.module("gh728_protocol.py", replay, "replay")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("NOT A", proc.stdout)
        searched = self.dir / "searched.out"
        searched.write_text(self.GH728_LINE % ("true", "1.5", 1, 1))
        proc = self.module("gh728_protocol.py", searched, "replay")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("-- NOT A REPLAY PASS", proc.stdout)

    def test_gh728_driver_checks_each_tuned_leg_through_the_module(self):
        # The leg loop runs only after a build and on a device, so its wiring is read, not driven:
        # the check is the module, handed each tuned leg's output and its pass, with its verdict
        # failing the run.
        text = (HERE / "gh728_cells.sh").read_text()
        self.assertIn(
            'python3 "$TREE/benchmarks/gh728_protocol.py" "$OUT/gpt_${profile}_$pass.out" "$pass"',
            text)
        self.assertIn(") || FAILED=1", text)
        self.assertNotIn("python3 -", text)


class Gh1003SummaryTest(unittest.TestCase):
    """gh1003_fold_cells.sh's `summary` step, rendered by gh1003_summary.py: run as the driver runs
    it (by path, from a scratch cwd holding no copy of its imports), and the step's wiring."""

    TREATMENTS = "composed two-pass fold-8 fold-16 composed two-pass-bwd fold-8-bwd fold-16-bwd"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.out = self.dir / "out"
        self.out.mkdir()

    def summary(self, treatments=TREATMENTS):
        return subprocess.run(
            [sys.executable, str(HERE / "gh1003_summary.py"), str(self.out), treatments],
            text=True, capture_output=True, cwd=self.dir, timeout=60)

    def cell(self, name, p50, p10=None, p90=None, losses=(2.5, 2.25), **fields):
        rec = {"step_ms": {"p10": p10 or p50, "p50": p50, "p90": p90 or p50}, "losses": list(losses)}
        rec.update(fields)
        (self.out / name).write_text("noise\n" + json.dumps(rec) + "\n")

    def test_the_matrix_as_the_driver_rendered_it(self):
        # The expected rows are what the driver's inline heredoc printed for this directory before
        # the summary moved into gh1003_summary.py, except that an empty cell is now the shared
        # builder's single blank (`| |`, gh-ocannl-1355) rather than two spaces. They cover the
        # two-pass-bwd fallback of the training fixtures, both mma census fields (and a null one),
        # a loss disagreement, a missing composed reference, and a dry cell that never enters.
        self.cell("metal-gpt2_mini-composed-r1.out", 10.0, 9.5, 10.5)
        self.cell("metal-gpt2_mini-composed-r2.out", 12.0)
        self.cell("metal-gpt2_mini-two-pass-r1.out", 8.0, losses=(2.5, 2.0), shipped_mma={"mma": 2})
        self.cell("metal-gpt2_mini-fold-16-r1.out", 5.0, 4.0, 6.0, tune={"shipped_mma": 3})
        self.cell("metal-gpt2_mini-fold-16-dry.out", 1.0)
        self.cell("metal-gpt2_mini_train_s512-two-pass-bwd-r1.out", 20.0, losses=(1.0,))
        self.cell("metal-gpt2_mini_train_s512-fold-8-bwd-r1.out", 10.0, shipped_mma=None)
        self.cell("cc-gpt2_mini-fold-8-r1.out", 100.0)
        proc = self.summary()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(proc.stdout.splitlines(), [
            "| backend | fixture | treatment | p50 per repeat (ms) | median p50 | vs composed | vs two-pass | p10..p90 spread | shipped mma | loss vs composed |",
            "|---|---|---|---|---|---|---|---|---|---|",
            "| cc | gpt2_mini | fold-8 | 100.0 | 100.0 | | | 1.000x | | |",
            "| metal | gpt2_mini | composed | 10.0, 12.0 | 11.0 | 1.000x | 1.375x | 1.105x | | 0 |",
            '| metal | gpt2_mini | two-pass | 8.0 | 8.0 | 0.727x | 1.000x | 1.000x | {"mma": 2} | 0.11 |',
            "| metal | gpt2_mini | fold-16 | 5.0 | 5.0 | 0.455x | 0.625x | 1.500x | 3 | 0 |",
            "| metal | gpt2_mini_train_s512 | two-pass-bwd | 20.0 | 20.0 | | 1.000x | 1.000x | | |",
            "| metal | gpt2_mini_train_s512 | fold-8-bwd | 10.0 | 10.0 | | 0.500x | 1.000x | | |",
        ])

    def test_names_each_unreadable_cell(self):
        spellings = ("NaN", "Infinity", "1e999")
        for i, spelled in enumerate(spellings, 1):
            (self.out / f"metal-gpt2_mini_train-composed-r{i}.out").write_text(
                '{"step_ms":{"p10":1,"p50":%s,"p90":3},"losses":[7]}\n' % spelled)
        self.cell("metal-gpt2_mini_train-composed-r9.out", 2, 1, 3, losses=(7,))
        proc = self.summary("composed")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        for i in range(1, len(spellings) + 1):
            self.assertIn(f"UNREADABLE RESULT: metal-gpt2_mini_train-composed-r{i}.out:", proc.stdout)
        rows = [line for line in proc.stdout.splitlines() if line.startswith("| metal |")]
        self.assertEqual(len(rows), 1, proc.stdout)
        self.assertIn("| composed | 2.0 | 2.0 |", rows[0])

    def test_a_cell_without_a_result_line_fails_the_matrix(self):
        self.cell("metal-gpt2_mini-composed-r1.out", 2)
        (self.out / "metal-gpt2_mini-composed-r2.out").write_text("noise\n")
        proc = self.summary()
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("MISSING RESULT: metal-gpt2_mini-composed-r2.out has no result line", proc.stdout)
        # Negative control: the same matrix without the empty cell passes.
        (self.out / "metal-gpt2_mini-composed-r2.out").unlink()
        proc = self.summary()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("MISSING RESULT", proc.stdout)

    def test_the_driver_summary_step_writes_the_module_matrix(self):
        # A fresh OUT: the step reaches the module (summary.md holds its header and verdict) and
        # fails on the empty matrix.
        out = self.dir / "driven"
        proc = subprocess.run(
            [box_gate.BASH, str(HERE / "gh1003_fold_cells.sh"), str(out), "60", "summary"],
            text=True, capture_output=True, cwd=self.dir, timeout=120)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        summary = (out / "summary.md").read_text()
        self.assertIn("NO MEASUREMENT: no numbered cell produced a result line", summary)
        self.assertIn("| vs composed | vs two-pass |", summary)
        self.assertNotIn("Traceback", proc.stdout + proc.stderr)

    def driver(self, *steps, env=None):
        return subprocess.run(
            [box_gate.BASH, str(HERE / "gh1003_fold_cells.sh"), str(self.out), "60", *steps],
            text=True, capture_output=True, cwd=self.dir, timeout=120,
            env=dict(os.environ, **(env or {})))

    def recorded_run(self, blocks):
        """self.out as an earlier invocation left it: its driver.log header and a filled matrix."""
        (self.out / "driver.log").write_text(
            "gh1003: 2026-10-18T00:00:00Z root=/r out=/o cap=60 repeats=1\n"
            f"gh1003: fold blocks: {blocks}; train fixtures: t; kernel table: 1; fixture dir: d; "
            "steps: build provenance metal\n")
        self.cell("metal-gpt2_mini-composed-r1.out", 10.0)
        self.cell("metal-gpt2_mini-two-pass-r1.out", 8.0)
        self.cell("metal-gpt2_mini-fold-64-r1.out", 5.0)

    def test_a_lone_summary_step_re_renders_an_earlier_out(self):
        # The filled matrix driven end to end (gh-ocannl-1365), under the fold blocks the earlier
        # run recorded: fold-64 is in no default list, so a re-summary under the defaults would
        # drop its row without a word.
        self.recorded_run("8 64")
        (self.out / "summary.md").write_text("the earlier rendering\n")
        proc = self.driver("summary")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        expected = self.summary("composed two-pass fold-8 fold-64 composed two-pass-bwd fold-8-bwd fold-64-bwd")
        self.assertEqual(expected.returncode, 0, expected.stdout + expected.stderr)
        self.assertEqual((self.out / "summary.md").read_text(), expected.stdout)
        self.assertIn("| metal | gpt2_mini | fold-64 | 5.0 | 5.0 | 0.500x | 0.625x |", expected.stdout)
        log = (self.out / "driver.log").read_text()
        self.assertIn("re-summarizing an existing OUT under its recorded fold blocks", log)
        # The earlier header still leads, so a second re-summary reads the same blocks.
        self.assertIn("gh1003: fold blocks: 8 64;", log.splitlines()[1])
        self.assertEqual(self.driver("summary").returncode, 0)

    def test_a_re_summary_refuses_what_it_cannot_render_faithfully(self):
        self.recorded_run("8 64")
        cases = [
            # A measuring step would mix a new run's cells into the old one's.
            (("metal",), {}, "is not empty"),
            (("summary", "summary"), {"FOLD_BLOCKS": "8 16 32"}, "differs from the '8 64'"),
        ]
        for steps, env, why in cases:
            with self.subTest(steps=steps, env=env):
                proc = self.driver(*steps, env=env)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(why, proc.stderr)
        # Agreeing FOLD_BLOCKS is no refusal (the control for the disagreeing one).
        self.assertEqual(self.driver("summary", env={"FOLD_BLOCKS": "8 64"}).returncode, 0)
        # A non-empty directory this driver did not write.
        (self.out / "driver.log").unlink()
        proc = self.driver("summary")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("holds no driver.log", proc.stderr)


# Every `json.load`/`json.loads` call in the benchmark drivers, runners and their inline shell
# Python that does NOT read a runner's result or checkpoint line, with why: a judgment list, so a
# new reader has to choose between strict_json and a stated reason here. Keyed by file and a
# fragment of the call's line; an entry that matches nothing fails as stale.
JSON_LOADS_NOT_RESULT_LINES = {
    ("fixture_digest.py", "json.loads(raw_header"): "safetensors fixture header",
    ("runners/bench_common.py", "json.loads(f.read(n))"): "safetensors fixture header",
    ("gen_fixtures.py", "spec = json.loads(spec_path"): "workload spec",
    ("gen_fixtures.py", 'name = json.loads(spec_path'): "workload spec",
    ("gh1002_cells.py", "recorded = json.loads(pf.read_text())"): "the driver's own preflight.json",
    ("gh1002_cells.py", '"workloads"'): "workload spec",
    ("gh1002_cells.py", "pre = json.loads((out"): "the driver's own preflight.json",
    ("gh1002_cells.py", "ident = json.loads(pf.read_text())"): "the driver's own preflight.json",
    ("gh1181_cells.py", "return [json.loads(line) for line in path"):
        "the driver's own timings.jsonl/failures.jsonl (json_safe, allow_nan=False)",
    ("gh1181_cells.py", "versions = json.loads(supporting_output"): "the venv's version probe",
    ("gh1181_cells.py", 'env = json.loads((out / "env.json")'): "the driver's own env.json",
    ("gh1181_cells.py", 'old = json.loads((out / "summary.json")'): "the driver's own summary.json",
    ("gh675_cells.py", "r = json.loads(ln)"): "the driver's own records file",
    ("gh998_cells.py", "lenient = json.loads(line)"):
        "re-reads a line strict_json refused, only to tell a result line from noise",
    ("gh998_cells.py", "if json.loads(paths[0]"): "the driver's own source manifests",
    ("gh998_cells.py", "matrix = json.loads("): "the driver's own matrix.json",
    ("gh998_cells.py", "preflight = json.loads("): "the driver's own preflight.json",
    ("gh998_cells.py", "controls.append(json.loads("): "the driver's own result.json (allow_nan=False)",
    ("gh998_cells.py", "json.loads((baseline / 'result.json')"):
        "the driver's own result.json (allow_nan=False)",
    ("gh998_cells.py", "row = json.loads((out"): "the driver's own result.json (allow_nan=False)",
    ("gh998_cells.py", "on = json.loads((out"): "the driver's own result.json (allow_nan=False)",
    ("gh998_cells.py", "if json.loads(preflight.read_text())"): "the driver's own preflight.json",
    ("gh998_cells.py", "json.loads((out / 'dry-ok.json')"): "the driver's own dry-ok.json",
}


JSON_LOAD_CALL = re.compile(r"\bjson\.loads?\(")


def json_load_calls(texts):
    """`(file, line)` for every json.load(s) call line in `texts` ({relative path: source}) that
    JSON_LOADS_NOT_RESULT_LINES does not exempt, and the exemptions that matched nothing."""
    used, bare = set(), []
    for path, text in sorted(texts.items()):
        for line in text.splitlines():
            if not JSON_LOAD_CALL.search(line):
                continue
            keys = [k for k in JSON_LOADS_NOT_RESULT_LINES if k[0] == path and k[1] in line]
            used.update(keys)
            if not keys:
                bare.append((path, line.strip()))
    return bare, sorted(set(JSON_LOADS_NOT_RESULT_LINES) - used)


class ResultLineReaderInventoryTest(unittest.TestCase):
    """No reader of a runner's result line can bypass strict_json unnoticed: a json.load(s) call in
    a driver, a runner or a driver's inline Python is either exempt with a reason or a failure."""

    @staticmethod
    def shipped_sources():
        files = [
            p for p in sorted(HERE.rglob("*"))
            if p.suffix in (".py", ".sh") and p.is_file()
            and ".venv" not in p.relative_to(HERE).parts
            and p.relative_to(HERE).parts[0] != "test"  # the tests' own fixtures and probes
            and not p.name.startswith(("test_", "test-"))
            and p.name != "strict_json.py"  # the loader itself
        ]
        return {p.relative_to(HERE).as_posix(): p.read_text() for p in files}

    def test_every_json_load_is_strict_or_exempt(self):
        sources = self.shipped_sources()
        self.assertIn("orchestrate.py", sources)
        self.assertIn("gh728_cells.sh", sources)
        bare, stale = json_load_calls(sources)
        self.assertEqual(bare, [], "read these through strict_json.loads, or exempt them with a reason")
        self.assertEqual(stale, [], "exemptions that match no call any more")

    def test_a_bare_result_line_read_is_caught(self):
        # Negative control: the shape gh728_cells.sh shipped before this scan.
        sources = self.shipped_sources()
        sources["gh728_cells.sh"] += "rows = [json.loads(l) for l in open(path)]\n"
        bare, _ = json_load_calls(sources)
        self.assertEqual(bare, [("gh728_cells.sh", "rows = [json.loads(l) for l in open(path)]")])


class OcannlResultLineTest(unittest.TestCase):
    """The OCANNL runner's own result line, as this orchestrator reads it (gh-ocannl-676).

    The golden of test/operations/bench_result_line holds the line built from fabricated values --
    a diverged trajectory, times that were never measured, a diagnostic full of control characters
    -- and this is the parser that has to accept it. Pinning it in OCaml alone would leave the
    claim "this parses" resting on a second implementation of JSON.
    """

    GOLDEN = HERE.parent / "test/operations/bench_result_line.expected"

    def lines(self):
        return [
            line
            for line in self.GOLDEN.read_text().splitlines()
            if line.startswith("{")  # what run_cell picks out of a cell's output
        ]

    def test_every_emitted_line_parses_strictly(self):
        lines = self.lines()
        self.assertGreaterEqual(len(lines), 2, self.GOLDEN)
        for line in lines:
            with self.subTest(line=line[:60]):
                self.assertIn("losses", strict_loads(line))

    def test_the_diverged_line_is_read_as_a_diverged_cell(self):
        parsed = [strict_loads(line) for line in self.lines()]
        blown = next(r for r in parsed if None in r["losses"])
        ref = result("pytorch", "cpu", "eager", [2.5, 1.75, 1.25])
        ref["workload"] = blown["workload"]

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)

    def test_a_label_the_storage_read_back_contradicts_fails_the_precision_gate(self):
        # gh-ocannl-1262, on lines the OCaml emitter built: a row's storage_precision is what its
        # model ran at. Stamped with what it ran at, it passes; stamped as a precision the runner
        # did not apply -- the sweep's label over an ignored BENCH_PRECISION -- it fails.
        parsed = [strict_loads(line) for line in self.lines()]
        read_back = [r for r in parsed if r.get("storage_precision") is not None]
        self.assertTrue(read_back, "no golden line carries a storage read-back")
        unread = [r for r in parsed if r.get("storage_precision") is None]
        self.assertTrue(unread, "no golden line spells a missing read-back")
        honest = [dict(r) for r in read_back]
        ignored = [dict(r, precision="bf16") for r in read_back]
        self.assertEqual(orchestrate.precision_check(honest + unread), [])
        self.assertEqual(orchestrate.precision_check(ignored), ignored)
        for r in ignored:
            self.assertEqual(r["precision_mismatch"], f"{r['storage_precision']} storage")


def golden_checkpoint_lines():
    """The checkpoint lines `Bench_json` built, with the prefix a runner writes them behind, as the
    golden of test/operations/bench_result_line holds them (gh-ocannl-1209)."""
    return [
        line
        for line in OcannlResultLineTest.GOLDEN.read_text().splitlines()
        if line.startswith(orchestrate.CHECKPOINT_PREFIX)
    ]


def golden_checkpoint_line(workload, **stages):
    """The one golden checkpoint line of `workload` whose stages are as given, e.g.
    `golden_checkpoint_line("gpt2_mini_train_s1024", dominant_kernel="running")`.

    Looked up by identity, never by position: a checkpoint added to the golden mid-list must not
    silently change what a test reads. None or several matching is a failure, not a pick."""
    found = [
        line
        for line in golden_checkpoint_lines()
        if (c := orchestrate.last_checkpoint(line))["workload"] == workload
        and all(c["stages"].get(stage) == status for stage, status in stages.items())
    ]
    if len(found) != 1:
        raise AssertionError(
            f"{len(found)} golden checkpoints of {workload} with stages {stages}, expected one"
        )
    return found[0]


class CheckpointLineTest(unittest.TestCase):
    """gh-ocannl-1209: the checkpoints an OCANNL runner writes before its later stages, as the
    driver reads them."""

    def test_every_emitted_checkpoint_is_read_back_and_is_no_result_line(self):
        lines = golden_checkpoint_lines()
        self.assertGreaterEqual(len(lines), 2, OcannlResultLineTest.GOLDEN)
        for line in lines:
            with self.subTest(line=line[:60]):
                strict_loads(line[len(orchestrate.CHECKPOINT_PREFIX):])
                checkpoint = orchestrate.last_checkpoint(line)
                self.assertEqual(checkpoint["record"], "checkpoint")
                self.assertIs(checkpoint["accepted"], False)
                self.assertNotIn("step_ms", checkpoint)
                # What run_cell's result parser looks for, which a checkpoint must never be.
                self.assertFalse(line.startswith("{"))

    def test_the_last_whole_checkpoint_wins_over_a_line_the_kill_cut(self):
        # Cut mid-parity, then the one written before the dominant-kernel instrument.
        first = golden_checkpoint_line("gpt2_mini_train_s1024", parity="running")
        second = golden_checkpoint_line("gpt2_mini_train_s1024", dominant_kernel="running")
        text = "\n".join(["bench: parity step 0 loss 10.4", first, second[: len(second) // 2]])

        self.assertEqual(orchestrate.last_checkpoint(text), strict_loads(
            first[len(orchestrate.CHECKPOINT_PREFIX):]))
        self.assertIsNone(orchestrate.last_checkpoint("bench: parity step 0 loss 10.4\n"))


class SavedLossParityTest(unittest.TestCase):
    """The saved losses of a failed cell, against the reference: a diagnostic, never acceptance.

    Fed the checkpoints `Bench_json` built (the golden of test/operations/bench_result_line), so
    the reader is pinned to what a runner writes rather than to a fixture of this file's making.
    """

    def checkpoint(self, complete):
        return orchestrate.last_checkpoint(
            golden_checkpoint_line(
                "gpt2_mini_train_s1024", parity="complete" if complete else "running"
            )
        )

    def reference(self, checkpoint, scale):
        ref = result("pytorch", "cpu", "eager", [x * scale for x in checkpoint["losses"]])
        ref["workload"] = checkpoint["workload"]
        return ref

    def test_drift_is_read_against_the_envelope_the_cell_would_have_been_gated_at(self):
        saved = self.checkpoint(complete=True)
        self.assertEqual(saved["precision"], "f16")  # gated at PARITY_TOL_PRECISION["f16"]
        self.assertIsNone(saved["profile"])  # its runner resolved no profile: the exact regime
        close = [self.reference(saved, 1 + 1e-4)]
        apart = [self.reference(saved, 1.005)]

        inside = orchestrate.saved_loss_parity(saved, "exact", close)
        outside = orchestrate.saved_loss_parity(saved, "exact", apart)
        # The same drift fits the approximate regime's wider envelope. A checkpoint predating the
        # `profile` field says nothing of its regime, so the dispatched one stands in.
        legacy = {k: v for k, v in saved.items() if k != "profile"}
        approx = orchestrate.saved_loss_parity(legacy, "approximate", apart)

        self.assertIn("6 of 6 saved losses", inside)
        self.assertIn("inside the envelope 0.002", inside)
        self.assertIn("outside the envelope 0.002", outside)
        self.assertIn("inside the envelope 0.01 [approximate]", approx)
        for note in (inside, outside, approx):
            # Never in the parity column's vocabulary: a reader must not take it for a verdict.
            for verdict in ("PASS", "FAIL", "REF", "DIVERGED"):
                self.assertNotIn(verdict, note)

    def test_the_envelope_regime_is_read_off_the_checkpoint_not_the_dispatch(self):
        cut = self.checkpoint(complete=False)
        saved = self.checkpoint(complete=True)
        self.assertEqual(cut["profile"], "approximate")  # as the OCaml emitter wrote it
        # Step 0, the one finite step both share, is 0.5% off: outside f16's exact envelope,
        # inside the approximate one.
        ref = [self.reference(saved, 1.005)]

        # Whatever the call site passes -- nothing, the dispatched regime, or a forgotten
        # argument -- the losses are read in the regime their runner resolved.
        for dispatched in (None, "approximate"):
            with self.subTest(dispatched=dispatched):
                note = orchestrate.saved_loss_parity(cut, dispatched, ref)
                self.assertIn("inside the envelope 0.01 [approximate]", note)
                self.assertNotIn("dispatched", note)
        # A runner that resolved another regime than the sweep dispatched keeps its own envelope,
        # and the disagreement is named rather than silently resolved either way.
        leaked = orchestrate.saved_loss_parity(saved, "approximate", ref)
        self.assertIn("outside the envelope 0.002", leaked)
        self.assertIn("the runner resolved the exact regime, the sweep dispatched approximate",
                      leaked)
        # A non-approximate profile is still the exact regime, as `runner_regime` reads it.
        pinned = orchestrate.saved_loss_parity(dict(saved, profile="reproducible"), "exact", ref)
        self.assertIn("outside the envelope 0.002", pinned)
        self.assertNotIn("dispatched", pinned)

    def leaked(self):
        """The golden's exact cell an ambient knob reached with no profile behind it."""
        return orchestrate.last_checkpoint(golden_checkpoint_line("mlp_wide", parity="running"))

    def test_an_ambient_knob_no_profile_shows_is_named_from_the_checkpoint(self):
        """gh-ocannl-1297: the checkpoint's `regime_knobs`, read as the row gate reads a row's."""
        leaked = self.leaked()
        # The profile alone says exact: only the per-setting record shows the knob.
        self.assertIsNone(leaked["profile"])
        self.assertEqual(
            leaked["regime_knobs"]["tf32_matmuls"], {"value": "true", "source": "environment"}
        )
        ref = self.reference(leaked, 1 + 1e-4)

        note = orchestrate.saved_loss_parity(leaked, "exact", [ref])
        # Named in the row gate's own words, beside a drift that alone would read as on course.
        named = orchestrate.regime_knob_leaks(leaked, regime="exact")
        self.assertEqual(list(named), ["tf32_matmuls"])
        self.assertIn(f"outside the exact regime: {named['tf32_matmuls']}", note)
        self.assertIn("tf32_matmuls=true (environment)", note)
        self.assertIn("inside the envelope 0.002", note)
        # Whatever the call site dispatched, the regime is the one the runner resolved.
        self.assertIn("outside the exact regime: tf32_matmuls=true", orchestrate.saved_loss_parity(
            leaked, None, [ref]))
        for verdict in ("PASS", "FAIL", "REF", "DIVERGED"):
            self.assertNotIn(verdict, note)

    def test_checkpoints_inside_their_regime_name_no_leak(self):
        saved = self.checkpoint(complete=True)  # every knob at its default, no profile
        cut = self.checkpoint(complete=False)  # every knob from the approximate profile
        legacy = {k: v for k, v in self.leaked().items() if k != "regime_knobs"}
        ref = self.reference(saved, 1.0)
        for checkpoint, dispatched in ((saved, "exact"), (cut, "approximate"), (legacy, "exact")):
            with self.subTest(profile=checkpoint["profile"], knobs="regime_knobs" in checkpoint):
                note = orchestrate.saved_loss_parity(checkpoint, dispatched, [ref])
                self.assertNotIn("regime:", note)
        # A runner predating the field reports nothing to judge, as `regime_knob_leaks` says.
        self.assertIsNone(orchestrate.regime_knob_leaks(legacy, regime="exact"))

    def test_a_declared_cell_flag_is_no_leak_on_a_checkpoint_either(self):
        """The `--cell-flag` exemption (gh-ocannl-1273) holds on the checkpoint, given the flags."""
        flag = orchestrate.ocannl_arg("tf32_matmuls", "true")
        overridden = dict(
            self.leaked(),
            regime_knobs={
                "tf32_matmuls": {"value": "true", "source": "commandline"},
                "cc_backend_fast_math": {"source": "default"},
            },
        )
        ref = self.reference(overridden, 1.0)

        declared = orchestrate.saved_loss_parity(overridden, "exact", [ref], [flag])
        undeclared = orchestrate.saved_loss_parity(overridden, "exact", [ref])
        other = orchestrate.saved_loss_parity(
            overridden, "exact", [ref], [orchestrate.ocannl_arg("tf32_matmuls", "false")]
        )

        self.assertNotIn("regime:", declared)
        self.assertIn("outside the exact regime: tf32_matmuls=true (commandline)", undeclared)
        self.assertIn("outside the exact regime: tf32_matmuls=true (commandline)", other)

    def test_the_leak_survives_partial_failures_jsonl_and_reaches_the_report(self):
        """An offline reader of the failure record, and the report, both see it named."""
        leaked = self.leaked()
        ref = cell("pytorch", "cpu", "eager", [x * (1 + 1e-4) for x in leaked["losses"]])
        orchestrate.parity_check([ref])
        label = "mlp_wide ocannl/cuda/default"
        stamp = {"ocannl_flags": [], "ambient_ocannl_shadowed": {}}

        record = orchestrate.failure_record(
            label, "TIMED OUT after 90s", {"OCANNL_TF32_MATMULS": "true"}, dict(leaked),
            regime="exact", cell_stamp=stamp,
        )
        back = json.loads(json.dumps(orchestrate.json_safe(record), allow_nan=False))
        offline = orchestrate.saved_loss_parity(
            back["checkpoint"], back["regime"], [ref], back["ocannl_flags"]
        )
        self.assertIn("outside the exact regime: tf32_matmuls=true (environment)", offline)

        # Beside it, a cell whose knob came from its own declared `--cell-flag`: the report hands
        # the failure's stamp to the diagnostic, so that one is not named.
        flagged = "mlp_wide ocannl/cuda/default +tf32_matmuls=true"
        declared = dict(
            leaked,
            regime_knobs=dict(
                leaked["regime_knobs"], tf32_matmuls={"value": "true", "source": "commandline"}
            ),
        )
        flags = {"ocannl_flags": [orchestrate.ocannl_arg("tf32_matmuls", "true")]}
        failures = [
            orchestrate.Failure(label, "TIMED OUT after 90s", leaked, "exact", stamp),
            orchestrate.Failure(flagged, "TIMED OUT after 90s", declared, "exact", flags),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref], out, (), failures)
            text = (out / "report.md").read_text()
        section = text[text.index("## Runner failures") :]
        rows = {
            name: next(line for line in section.splitlines() if line.startswith(f"| {name} |"))
            for name in (label, flagged)
        }
        self.assertIn("outside the exact regime: tf32_matmuls=true (environment)", rows[label])
        self.assertIn("inside the envelope 0.002", rows[flagged])
        self.assertNotIn("regime:", rows[flagged])

    def test_a_trajectory_that_went_non_finite_reports_where_and_its_drift_before(self):
        cut = self.checkpoint(complete=False)
        self.assertEqual(cut["losses"][1], None)
        ref = self.reference(self.checkpoint(complete=True), 1.0)

        note = orchestrate.saved_loss_parity(cut, "exact", [ref])

        self.assertIn("2 of 6 saved losses", note)
        self.assertIn("drift 0.0e+00", note)  # step 0 is the reference's own value
        self.assertIn("loss non-finite from step 1", note)

    def test_without_a_reference_or_losses_it_says_so_rather_than_inventing_a_drift(self):
        saved = self.checkpoint(complete=True)
        other = result("pytorch", "cpu", "eager", saved["losses"])  # workload mlp_wide
        diverged = self.reference(saved, 1.0)
        diverged["losses"] = [float("nan")] * 6

        self.assertEqual(orchestrate.saved_loss_parity(None, "exact", [other]), "no checkpoint")
        self.assertIn(
            "no usable reference row", orchestrate.saved_loss_parity(saved, "exact", [other])
        )
        self.assertIn(
            "no usable reference row", orchestrate.saved_loss_parity(saved, "exact", [diverged])
        )
        empty = dict(saved, losses=[])
        self.assertIn("nothing to compare", orchestrate.saved_loss_parity(empty, "exact", [other]))

    def test_without_a_usable_reference_the_trajectory_still_speaks_for_itself(self):
        cut = self.checkpoint(complete=False)  # [10.375, null]
        other = result("pytorch", "cpu", "eager", cut["losses"][:1] * 6)  # workload mlp_wide
        diverged = self.reference(self.checkpoint(complete=True), 1.0)
        diverged["losses"] = [float("nan")] * 6

        for refs in ([], [other], [diverged]):
            with self.subTest(refs=refs):
                note = orchestrate.saved_loss_parity(cut, "exact", refs)
                self.assertIn("no usable reference row", note)
                self.assertNotIn("drift", note)
                self.assertIn("loss non-finite from step 1", note)

    def test_one_saved_step_is_not_called_stationary(self):
        saved = self.checkpoint(complete=True)
        one = dict(saved, losses=saved["losses"][:1])
        flat = dict(saved, losses=[saved["losses"][0]] * 3)
        ref = self.reference(saved, 1.0)

        note = orchestrate.saved_loss_parity(one, "exact", [ref])
        self.assertIn("1 of 6 saved losses", note)
        self.assertIn("drift 0.0e+00", note)
        self.assertNotIn("stationary", note)
        self.assertIn("movement not assessable from one step", note)
        # Two or more finite steps that do not move are still reported as such.
        self.assertIn("loss stationary", orchestrate.saved_loss_parity(flat, "exact", [ref]))

    def test_the_report_shows_it_beside_the_failure_labelled_as_never_acceptance(self):
        saved = self.checkpoint(complete=True)
        ref = cell("pytorch", "cpu", "eager", [x * (1 + 1e-4) for x in saved["losses"]])
        ref["workload"] = saved["workload"]
        orchestrate.parity_check([ref])
        label = "gpt2_mini_train_s1024 ocannl/hip/default/f16"
        failures = [
            orchestrate.Failure(label, "TIMED OUT after 90s", saved, "exact"),
            ("mlp_wide ocannl/cc/default", "exit 1"),  # a plain pair still reads
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref], out, (), failures)
            text = (out / "report.md").read_text()

        section = text[text.index("## Runner failures") :]
        self.assertNotIn(label, text[: text.index("## Runner failures")])
        self.assertIn("diagnostic, never acceptance", section)
        row = next(line for line in section.splitlines() if line.startswith(f"| {label} |"))
        self.assertIn("TIMED OUT after 90s", row)
        self.assertIn("inside the envelope 0.002", row)
        self.assertIn("| mlp_wide ocannl/cc/default | exit 1 | no checkpoint |", section)


class FailureRecordTest(unittest.TestCase):
    """gh-ocannl-1209: a failed cell's checkpoint is persisted with the identity it lacks."""

    def test_a_checkpoint_is_kept_whole_beside_the_fixture_stamp_and_source(self):
        checkpoint = strict_loads(
            golden_checkpoint_line("gpt2_mini_train_s1024", dominant_kernel="running")[
                len(orchestrate.CHECKPOINT_PREFIX):
            ]
        )
        stamp = {"fixture": "gpt2_mini_train_s1024.safetensors", "fixture_sha256": "ab" * 32,
                 "fixture_origin": "tuf"}
        source = {"source_revision": "cd" * 20, "source_tracked_changes": False}

        runner = Path(tempfile.mkdtemp()) / "bench_gpt.exe"
        runner.write_bytes(b"the runner that ran")
        checkpoint["executable"] = str(runner)

        record = orchestrate.failure_record(
            "cell", "TIMED OUT", {}, dict(checkpoint), provenance=dict(stamp, **source),
            regime="approximate",
        )
        line = json.dumps(orchestrate.json_safe(record), allow_nan=False)

        back = strict_loads(line)
        self.assertEqual(back["checkpoint"], checkpoint)  # its own `fixture` object untouched
        # Both sides of the saved-loss diagnostic, for a reader with only this file: the regime the
        # sweep dispatched, and the profile the runner resolved, inside the checkpoint.
        self.assertEqual(back["regime"], "approximate")
        self.assertIn("profile", back["checkpoint"])
        # The binary's own bytes name it, whatever HEAD or --skip-build said.
        digest = hashlib.sha256(b"the runner that ran").hexdigest()
        self.assertEqual(back["provenance"], dict(stamp, **source, executable_sha256=digest))
        runner.unlink()
        gone = orchestrate.failure_record("cell", "TIMED OUT", {}, dict(checkpoint), provenance={})
        self.assertIsNone(gone["provenance"]["executable_sha256"])

    def test_a_failure_without_a_checkpoint_carries_neither(self):
        record = orchestrate.failure_record("cell", "exit 1", {}, None, provenance={"x": 1})

        self.assertNotIn("checkpoint", record)
        self.assertNotIn("provenance", record)
        # The dispatched regime is the record's own field, stated as null when the caller gave none.
        self.assertIn("regime", record)
        self.assertIsNone(record["regime"])

    def test_the_source_identity_names_the_checkout_it_is_handed(self):
        # A checkout of its own rather than this one: the dune rule runs sandboxed, and dune plants
        # an empty `.git` above every sandbox so that nothing inside one can find the enclosing
        # repository -- an identity read off HERE would say nothing there. The repository is built
        # here, so its HEAD and its dirtiness are known to the test, not merely well-formed.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "checkout"
            repo.mkdir()
            env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                       GIT_CEILING_DIRECTORIES=tmp)

            def git(*args):
                return subprocess.run(["git", *args], cwd=repo, env=env, check=True,
                                      capture_output=True, text=True).stdout.strip()

            with unittest.mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": tmp}):
                # Not yet a repository: no revision to name, stated as null rather than a crash.
                self.assertEqual(orchestrate.source_identity(repo),
                                 {"source_revision": None, "source_tracked_changes": False})

                git("init", "-q")
                (repo / "tracked.txt").write_text("one\n")
                git("add", "tracked.txt")
                git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q",
                    "-m", "c")
                head = git("rev-parse", "HEAD")
                self.assertRegex(head, r"^[0-9a-f]{40}$")
                self.assertEqual(orchestrate.source_identity(repo),
                                 {"source_revision": head, "source_tracked_changes": False})

                # An untracked file is not a tracked change; an edit to a tracked one is.
                (repo / "untracked.txt").write_text("x\n")
                self.assertFalse(orchestrate.source_identity(repo)["source_tracked_changes"])
                (repo / "tracked.txt").write_text("two\n")
                self.assertEqual(orchestrate.source_identity(repo),
                                 {"source_revision": head, "source_tracked_changes": True})

    def test_the_sweep_records_failures_through_it_with_its_stamp_and_source(self):
        # main's closure is not callable from here; pin that it composes the two, read before
        # dispatch, rather than writing a record of its own.
        source = Path(orchestrate.__file__).read_text()
        main = source[source.index("\ndef main():") :]
        self.assertIn("source = source_identity(ROOT)", main)
        self.assertIn("provenance=dict(stamp, **source), regime=regime", main)
        # The report's saved-loss diagnostic reads the checkpoint and regime off the failure.
        self.assertIn("failures.append(Failure(label, note, checkpoint, regime, cell_stamp))", main)
        # Both the cell and its search pass go through the helper that records an interrupted
        # cell's checkpoint before the cancellation propagates -- the shared `run_cell_kept`, every
        # sweep driver's runner (gh-ocannl-1234), and never `run_cell` bare.
        self.assertEqual(main.count("run_cell_kept("), 1)
        self.assertEqual(main.count("run_cell("), 0)
        self.assertEqual(main.count("run_kept("), 3)
        self.assertLess(main.index("source = source_identity(ROOT)"), main.index("def collect("))


class CellIdentityTest(unittest.TestCase):
    """gh-ocannl-539: scheduling variant and storage precision are independent axes."""

    def test_f32_cells_keep_their_bare_variant_name(self):
        self.assertEqual(orchestrate.cell_name("tuned", "f32"), "tuned")
        self.assertEqual(orchestrate.cell_name("default", "f32"), "default")

    def test_reduced_precision_cells_are_named_by_the_product(self):
        self.assertEqual(orchestrate.cell_name("tuned", "bf16"), "tuned/bf16")
        self.assertEqual(orchestrate.cell_name("default", "f16"), "default/f16")

    def test_skip_entries_are_precision_wildcards(self):
        # A None precision means "at every precision": a scheduling pathology must not be let back
        # in through the bf16/f16 columns. Tested against an injected entry rather than a live one,
        # so emptying SKIP_CELLS does not silently drop coverage of the wildcard itself.
        entry = ("some_workload", "metal", "tuned", None)
        orchestrate.SKIP_CELLS.add(entry)
        try:
            self.assertTrue(orchestrate.cell_skipped("some_workload", "metal", "tuned", "f32"))
            self.assertTrue(orchestrate.cell_skipped("some_workload", "metal", "tuned", "bf16"))
            self.assertFalse(orchestrate.cell_skipped("some_workload", "metal", "default", "f32"))
            self.assertFalse(orchestrate.cell_skipped("some_workload", "cc", "tuned", "bf16"))
        finally:
            orchestrate.SKIP_CELLS.discard(entry)

    def test_cifar_conv_metal_tuned_is_no_longer_skipped(self):
        # gh-ocannl-538: the post-tune re-init hang the entry was added for did not reproduce —
        # the cell completed in ~4 min with no hang, and the sweep reproduced its p50 to 0.005%.
        for precision in ("f32", "bf16"):
            self.assertFalse(
                orchestrate.cell_skipped("cifar_conv", "metal", "tuned", precision), precision
            )

    def test_hip_tuned_cells_are_no_longer_skipped(self):
        # gh-ocannl-532's unparallelized-baseline dispatch is fixed in the tuner and confirmed on
        # the gfx1151 machine that produced the symptom; all three cells complete.
        for workload in ("mlp_wide", "cifar_conv", "cifar_stride"):
            for precision in ("f32", "bf16"):
                self.assertFalse(
                    orchestrate.cell_skipped(workload, "hip", "tuned", precision),
                    (workload, precision),
                )

    def test_every_skip_entry_is_a_four_tuple(self):
        for entry in orchestrate.SKIP_CELLS:
            self.assertEqual(len(entry), 4, entry)

    def test_tuned_f32_and_tuned_bf16_are_distinct_cells(self):
        rows = [
            result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1]),
            result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1], precision="bf16"),
        ]
        names = {orchestrate.cell_name(r["variant"], r["precision"]) for r in rows}

        self.assertEqual(names, {"tuned", "tuned/bf16"})


class ReportGroupingTest(unittest.TestCase):
    def test_rows_group_by_precision_f32_first(self):
        self.assertEqual(orchestrate.precision_rank("f32"), 0)
        self.assertLess(
            orchestrate.precision_rank("bf16"), orchestrate.precision_rank("f16")
        )
        # A gate leg sorts directly after the storage precision it varies, not last: it is a
        # variant of how f16's optimizer step is gated, and reads next to plain f16.
        self.assertGreater(
            orchestrate.precision_rank("f16-static"), orchestrate.precision_rank("f16")
        )
        self.assertLess(
            orchestrate.precision_rank("f16-gated8"),
            orchestrate.precision_rank("unknown-precision"),
        )

    def test_gate_legs_are_parity_gated_at_their_base_precision(self):
        self.assertEqual(
            orchestrate.parity_tol("f16-gated8"), orchestrate.PARITY_TOL_PRECISION["f16"]
        )
        self.assertEqual(
            orchestrate.parity_tol("f16-static"), orchestrate.PARITY_TOL_PRECISION["f16"]
        )


class PrecisionLegTest(unittest.TestCase):
    """gh-ocannl-551: the gate-cost legs are orchestrated cells, and an inexpressible cell is
    reported rather than silently absent."""

    def test_gate_legs_dispatch_their_own_flags(self):
        self.assertEqual(
            orchestrate.precision_env("f16-static"),
            {"BENCH_PRECISION": "f16", "BENCH_STATIC_SCALE": "1"},
        )
        self.assertEqual(
            orchestrate.precision_env("f16-gated16"),
            {"BENCH_PRECISION": "f16", "BENCH_GATE_INTERVAL": "16"},
        )
        self.assertEqual(orchestrate.precision_env("bf16"), {"BENCH_PRECISION": "bf16"})

    def test_cell_env_carries_the_leg_and_clears_the_others(self):
        # The gate flags are cleared per cell and then set by the leg — the two collided as
        # duplicate dict() keywords, which crashed the run only once a gate cell was dispatched.
        stray = {"BENCH_STATIC_SCALE": "1", "BENCH_GATE_INTERVAL": "7"}
        plain = orchestrate.cell_env(stray, "fx.safetensors", "default", "bf16")
        self.assertEqual(plain["BENCH_PRECISION"], "bf16")
        self.assertEqual(plain["BENCH_STATIC_SCALE"], "0")
        self.assertEqual(plain["BENCH_GATE_INTERVAL"], "0")
        gated = orchestrate.cell_env(stray, "fx.safetensors", "tuned", "f16-gated16")
        self.assertEqual(gated["BENCH_PRECISION"], "f16")
        self.assertEqual(gated["BENCH_GATE_INTERVAL"], "16")
        self.assertEqual(gated["BENCH_STATIC_SCALE"], "0")
        self.assertEqual(gated["BENCH_TUNE"], "1")
        static = orchestrate.cell_env(stray, "fx.safetensors", "default", "f16-static")
        self.assertEqual(static["BENCH_STATIC_SCALE"], "1")
        self.assertEqual(static["BENCH_GATE_INTERVAL"], "0")

    def test_precision_spec_rejects_nonsense(self):
        for spec in ("f16-gated0", "f16-gated", "f8", "f16-dynamic"):
            with self.assertRaises(Exception, msg=spec):
                orchestrate.precision_spec(spec)

    def test_gate_legs_need_a_training_workload(self):
        # The forward-only gpt2_mini has no optimizer, hence no loss scale to gate.
        self.assertIsNotNone(
            orchestrate.precision_unavailable("gpt", "infer", "f16-static")
        )
        self.assertIsNotNone(
            orchestrate.precision_unavailable("gpt", "infer", "f16-gated8")
        )
        # ... but gpt2_mini_train does, and plain reduced precisions work in both modes.
        self.assertIsNone(
            orchestrate.precision_unavailable("gpt", "train", "f16-static")
        )
        self.assertIsNone(orchestrate.precision_unavailable("gpt", "infer", "bf16"))
        self.assertIsNone(orchestrate.precision_unavailable("mlp", "train", "f16-gated8"))

    def test_conv_workloads_report_the_missing_runner_support(self):
        reason = orchestrate.precision_unavailable("conv", "train", "bf16")
        self.assertIsNotNone(reason)
        self.assertIn("conv", reason)


class ProvenanceTest(unittest.TestCase):
    """gh-ocannl-644: a tuned cell's result says which pass timed it."""

    def tuned(self, searched, searches=0, replays=0, no_searches=0):
        r = result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1])
        if searched is not None:
            r["searched"] = searched
            r["tune"] = {
                "shipped": "A", "searches": searches, "replays": replays,
                "no_searches": no_searches, "arms": []
            }
        return r

    def test_a_replay_is_the_compliant_case(self):
        replay = self.tuned(False, replays=2)

        self.assertEqual(orchestrate.provenance_check([replay]), [])
        self.assertEqual(replay["provenance"], "REPLAY")

    def test_search_pass_timings_are_a_violation(self):
        # The failure this exists to catch: both passes emit the same framework/backend/variant/
        # precision, so before `searched` a report could quote pass-1 numbers indefinitely.
        searching = self.tuned(True, searches=2)

        self.assertEqual(orchestrate.provenance_check([searching]), [searching])
        self.assertEqual(searching["provenance"], "SEARCH-PASS")

    def test_one_classifier_reads_searched_for_every_consumer(self):
        # The round-1 and round-2 findings were the same shape — a three-valued fact read through
        # a boolean, wrong in a different place each time. There is now one reading of it, and the
        # gate, the `pass` column and the carried-over compile_s label all go through it.
        tune = lambda searches, replays: {  # noqa: E731
            "shipped": "A", "searches": searches, "replays": replays, "arms": []
        }
        searched = {"searched": True, "tune": tune(2, 0)}
        replay = {"searched": False, "tune": tune(0, 2)}
        no_search = {"searched": False, "tune": tune(0, 0)}

        self.assertEqual(orchestrate.search_provenance(searched), "SEARCHED")
        self.assertEqual(orchestrate.search_provenance(replay), "REPLAY")
        self.assertEqual(orchestrate.search_provenance(no_search), "NO-SEARCH")
        self.assertEqual(orchestrate.search_provenance({"searched": None}), "UNKNOWN")
        # No tune object: `searched: false` alone cannot separate a replay from a cell that
        # searched nothing, and guessing either way has already been a bug.
        self.assertEqual(orchestrate.search_provenance({"searched": False}), "UNKNOWN")

    def test_the_runner_states_the_no_search_case_instead_of_it_being_derived(self):
        # gh-ocannl-677: the OCaml call's outcome is one state, and the runner now names it per
        # arm and counts it (`no_searches`) rather than leaving the reader to recover "neither
        # searched nor replayed" from two zeroed counters in a JSON artifact.
        stated = lambda replays, no_searches: {  # noqa: E731
            "searched": False,
            "tune": {"shipped": "A", "searches": 0, "replays": replays,
                     "no_searches": no_searches, "arms": []},
        }

        self.assertEqual(orchestrate.search_provenance(stated(0, 2)), "NO-SEARCH")
        self.assertEqual(orchestrate.search_provenance(stated(2, 0)), "REPLAY")
        # A mixed cell — one arm replayed, the other had nothing to replay — still carries a tuned
        # artifact, so it is a replay.
        self.assertEqual(orchestrate.search_provenance(stated(1, 1)), "REPLAY")

    def test_only_a_real_replay_labels_the_carried_over_compile_cost_cached(self):
        # A search pass under autotune_search=false hands over the cost of compiling the untuned
        # default; calling that "(cached)" would claim a schedule cache it never touched.
        self.assertEqual(orchestrate.COMPILE_S_NOTE.get("REPLAY"), " (cached)")
        self.assertEqual(orchestrate.COMPILE_S_NOTE.get("NO-SEARCH"), " (no search)")
        self.assertIsNone(orchestrate.COMPILE_S_NOTE.get("SEARCHED"))
        self.assertIsNone(orchestrate.COMPILE_S_NOTE.get("UNKNOWN"))

    def test_a_disabled_search_is_neither_a_violation_nor_a_replay(self):
        # gh-ocannl-559's reproducible profile turns the search off: the cell ships the untuned
        # default, having neither searched nor replayed. Gating it would fail BOTH passes of every
        # tuned cell, and calling it a replay would credit the row with a tuned artifact it does
        # not have.
        cell = self.tuned(False, searches=0, replays=0, no_searches=2)

        self.assertEqual(orchestrate.provenance_check([cell]), [])
        self.assertEqual(cell["provenance"], "NO-SEARCH")

        # And the same verdict from an artifact written before `no_searches` existed.
        legacy = self.tuned(False, searches=0, replays=0)
        del legacy["tune"]["no_searches"]

        self.assertEqual(orchestrate.provenance_check([legacy]), [])
        self.assertEqual(legacy["provenance"], "NO-SEARCH")

    def test_a_runner_without_the_field_is_unknown_not_a_violation(self):
        old = self.tuned(None)

        self.assertEqual(orchestrate.provenance_check([old]), [])
        self.assertEqual(old["provenance"], "UNKNOWN")

    def test_cells_that_neither_search_nor_compile_are_not_annotated(self):
        # They have no process distinction to be on the wrong side of; a `pass` entry for them
        # would be a claim about a protocol they do not run.
        default = result("ocannl", "hip", "default", [2.3, 2.2, 2.1])
        default["searched"] = False
        eager = result("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])
        jit = result("tinygrad", "CPU", "jit", [2.3, 2.2, 2.1])

        self.assertEqual(orchestrate.provenance_check([default, eager, jit]), [])
        for r in (default, eager, jit):
            self.assertNotIn("provenance", r)

    def test_a_single_pass_framework_states_its_search_without_being_gated(self):
        # gh-ocannl-675: tinygrad's beam search and torch.compile run in the timing process by
        # protocol. Nothing yet says they pay for it, so the report states it and the gate keeps
        # its hands off — the alternative, saying nothing, reads as though only OCANNL searches.
        beam = result("tinygrad", "AMD", "beam", [2.3, 2.2, 2.1])
        beam["searched"] = True
        compiled = result("pytorch", "cuda", "compiled", [2.3, 2.2, 2.1])
        compiled["searched"] = False

        self.assertEqual(orchestrate.provenance_check([beam, compiled]), [])
        self.assertEqual(beam["provenance"], "SAME-PROCESS")
        self.assertEqual(compiled["provenance"], "CACHED")

    def test_a_single_pass_cell_that_cannot_tell_is_unknown(self):
        beam = result("tinygrad", "AMD", "beam", [2.3, 2.2, 2.1])
        beam["searched"] = None

        self.assertEqual(orchestrate.provenance_check([beam]), [])
        self.assertEqual(beam["provenance"], "UNKNOWN")

    def test_every_verdict_renders(self):
        for verdict in (
            "REPLAY",
            "SEARCH-PASS",
            "NO-SEARCH",
            "SAME-PROCESS",
            "CACHED",
            "UNKNOWN",
        ):
            self.assertIn(verdict, orchestrate.PROVENANCE_MARK)

    def test_the_gated_and_stated_cell_sets_do_not_overlap(self):
        self.assertFalse(orchestrate.TWO_PASS_CELLS & orchestrate.SAME_PROCESS_CELLS)


class TensorizationTest(unittest.TestCase):
    """gh-ocannl-626: a "tensorized" timing must not be able to be a scalar-fallback timing."""

    def cell(self, arms, shipped="A", shipped_mma=...):
        r = result("ocannl", "metal", "tuned", [2.3, 2.2, 2.1])
        r["searched"] = False
        r["step_ms"] = {"p10": 1.0, "p50": 1.1, "p90": 1.2}
        r["queued_step_ms"] = 0.1
        r["compile_s"] = 3.0
        r["parity"] = "REF"
        r["parity_loss_moved"] = True
        r["diverged_at"] = None
        r["tune"] = {
            "shipped": shipped, "searches": 0, "replays": 1, "no_searches": 0, "arms": arms
        }
        if shipped_mma is not ...:
            r["tune"]["shipped_mma"] = shipped_mma
        return r

    def mma(self, tensorization, statements=0, scalar_fallbacks=0):
        return {
            "tensorization": tensorization, "statements": statements,
            "scalar_fallbacks": scalar_fallbacks,
        }


    def arm(self, name, tensorized, tensorization, statements=0, fallbacks=0):
        return {
            "arm": name, "state": "cache-replay", "searched": False, "cache_hit": True,
            "best_ms": 1.0, "best_label": "L", "tensorized": tensorized,
            "tensorization": tensorization, "mma_statements": statements,
            "mma_scalar_fallbacks": fallbacks, "mma_seeded": 0, "mma_timed": 0,
            "mma_best_ms": 1.0, "terminal_failure": None,
        }

    def test_untuned_compiled_census_reaches_the_verdict(self):
        for label, verdict in [("tensorized", "TENSORIZED"),
                               ("scalar-fallback", "SCALAR-FALLBACK"),
                               ("not-requested", "NOT-REQUESTED")]:
            with self.subTest(label=label):
                cell = result("ocannl", "metal", "default", [1.0])
                cell["shipped_mma"] = self.mma(label, 4, 2)
                self.assertEqual(orchestrate.tensorization_verdict(cell), verdict)
                orchestrate.tensorization_check([cell])
                self.assertEqual(cell["tensorization"], verdict)

    def test_mismatch_notice_names_tuned_and_untuned_variants(self):
        untuned = result("ocannl", "metal", "default", [1.0])
        untuned["shipped_mma"] = self.mma("scalar-fallback", 2, 2)
        tuned = self.cell([self.arm("A", True, "scalar-fallback", 2, 2)])
        mismatches = orchestrate.tensorization_check([untuned, tuned])
        notice = orchestrate.tensorization_notice(mismatches)
        self.assertIn("2 cell(s)", notice)
        self.assertIn("metal/default", notice)
        self.assertIn("metal/tuned", notice)
        self.assertNotIn("tuned cell(s)", notice)

    def test_top_level_census_overrides_legacy_tuned_census(self):
        cell = self.cell([self.arm("A", True, "tensorized")],
                         shipped_mma=self.mma("tensorized", 4))
        cell["shipped_mma"] = self.mma("scalar-fallback", 4, 4)
        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")
        cell["shipped_mma"] = None
        self.assertEqual(orchestrate.tensorization_verdict(cell), "UNKNOWN")

    def test_a_declined_tensorize_is_not_reported_as_a_tensorized_timing(self):
        # The defect: the schedule asked for tensor cores, every Tile_mma rendered the lane-0
        # scalar loop, and the row still carried a tensorized variant name.
        cell = self.cell([self.arm("A", True, "scalar-fallback", statements=3, fallbacks=3)])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_a_tensorize_that_emitted_no_statement_is_its_own_verdict(self):
        # Distinct from the fallback: nothing declined, because nothing was emitted to decline.
        cell = self.cell([self.arm("A", True, "not-requested")])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-EMITTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_an_honestly_tensorized_cell_is_not_flagged(self):
        cell = self.cell([self.arm("A", True, "tensorized", statements=4)])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "TENSORIZED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_an_artifact_that_never_asked_is_not_a_mismatch(self):
        cell = self.cell([self.arm("A", False, "not-requested")])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-REQUESTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_a_missing_census_never_reads_as_tensorized(self):
        # The negative control. A runner predating the field, or an arm with no crowned candidate,
        # carries a null label — which must answer UNKNOWN, never the passing reading.
        older = self.cell([{"arm": "A", "best_ms": 1.0}])
        no_winner = self.cell([self.arm("A", False, None)])

        self.assertEqual(orchestrate.tensorization_verdict(older), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_verdict(no_winner), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_check([older, no_winner]), [])
        # And a cell that tuned nothing at all has no arm to consult, so it gets no column entry
        # rather than a fabricated one.
        self.assertIsNone(orchestrate.tensorization_verdict(result("torch", "cuda", "eager", [1.0])))

    def test_the_shipped_routines_census_outranks_the_arm_reports(self):
        # A gh-555 flip refinement that beats the A/B winner ships under `shipped: "flip"` and is
        # not an arm at all; on the timing_ctx path the tuner can also fall back to the untuned
        # default after the arm was crowned. Either way the arm describes a discarded schedule, so
        # the routine that was actually timed is what the column must report.
        flip = self.cell(
            [self.arm("A", True, "tensorized", statements=4)],
            shipped="flip",
            shipped_mma=self.mma("scalar-fallback", statements=3, scalar_fallbacks=3),
        )

        self.assertIsNone(orchestrate.shipped_arm(flip))
        self.assertEqual(orchestrate.tensorization_verdict(flip), "SCALAR-FALLBACK")
        self.assertEqual(orchestrate.tensorization_check([flip]), [flip])

    def test_a_fallback_to_the_untuned_default_is_not_reported_as_the_crowned_arm(self):
        # The arm was crowned in the scratch context and its replay was rejected in the production
        # one, so the timed routine has no mma at all while the arm still says tensorized.
        cell = self.cell(
            [self.arm("A", True, "tensorized", statements=4)],
            shipped_mma=self.mma("not-requested"),
        )

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-EMITTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_a_harness_that_recorded_no_shipped_census_reads_unknown(self):
        # Present-but-empty is a runner that reported arms and forgot the routine: not a finding,
        # and not the passing reading either.
        cell = self.cell([self.arm("A", True, "tensorized", statements=4)], shipped_mma=None)

        self.assertEqual(orchestrate.tensorization_verdict(cell), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_an_artifact_predating_shipped_mma_still_reads_its_arm(self):
        # The key absent (not null) is an older result line; the arm is right whenever the crowned
        # candidate WAS the shipped artifact, which is the common case.
        cell = self.cell([self.arm("A", True, "scalar-fallback", statements=2, fallbacks=2)])

        self.assertNotIn("shipped_mma", cell["tune"])
        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")

    def test_the_verdict_describes_the_arm_that_shipped_not_the_fastest_one(self):
        # tune_placements searches several arms and keeps one artifact; reading any other arm
        # describes a schedule that was discarded (gh-ocannl-638).
        cell = self.cell(
            [
                self.arm("A", True, "tensorized", statements=4),
                self.arm("B", True, "scalar-fallback", statements=2, fallbacks=2),
            ],
            shipped="B",
        )

        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")

    def test_every_verdict_renders_in_the_table(self):
        # The same completeness check the provenance column carries: a verdict with no mark would
        # print as an em dash, which is the "never asked" reading.
        for verdict in ("TENSORIZED", "SCALAR-FALLBACK", "NOT-EMITTED", "NOT-REQUESTED", "UNKNOWN"):
            self.assertIn(verdict, orchestrate.TENSORIZATION_MARK)
        for verdict in orchestrate.TENSORIZATION_MISMATCH:
            self.assertIn(verdict, orchestrate.TENSORIZATION_MARK)
            self.assertIn("**", orchestrate.TENSORIZATION_MARK[verdict])

    def test_the_mismatch_is_visible_in_the_rendered_table(self):
        cells = [
            self.cell([self.arm("A", True, "scalar-fallback", statements=3, fallbacks=3)]),
            self.cell([self.arm("A", True, "tensorized", statements=4)]),
        ]
        orchestrate.tensorization_check(cells)
        out = Path(tempfile.mkdtemp())
        orchestrate.report(cells, out)
        table = (out / "report.md").read_text()

        self.assertIn(" mma |", table)
        self.assertIn("**SCALAR FALLBACK**", table)
        self.assertIn("tensorized", table)


class SubstitutionTest(unittest.TestCase):
    """gh-ocannl-1559: a tuned cell whose timed process shipped the untuned default is flagged.

    Its arms, `shipped` and `shipped_digest` still name the search whose crown was discarded, so
    `tune.ship_substituted` is the only record that the row's step times are untuned.
    """

    def cell(self, ship_substituted=..., substituted_from=None):
        r = TensorizationTest.cell(self, [TensorizationTest.arm(self, "A", False, "not-requested")])
        if ship_substituted is not ...:
            r["tune"]["ship_substituted"] = ship_substituted
            r["tune"]["substituted_from"] = substituted_from
        return r

    def test_only_a_true_record_is_a_substitution(self):
        # false is a measured ship; null and an absent key are unknown, never a substitution.
        unflagged = [self.cell(False), self.cell(None), self.cell()]
        flagged = self.cell(True, "grid 128")
        self.assertEqual(orchestrate.substitution_check(unflagged + [flagged]), [flagged])
        self.assertTrue(flagged["ship_substituted"])
        for r in unflagged:
            self.assertNotIn("ship_substituted", r)

    def test_the_notice_names_the_discarded_crown_or_the_last_resort(self):
        crowned, last_resort = self.cell(True, "grid 128"), self.cell(True, None)
        notice = orchestrate.substitution_notice(
            orchestrate.substitution_check([crowned, last_resort])
        )
        self.assertIn("2 tuned cell(s)", notice)
        self.assertIn("(discarded grid 128)", notice)
        self.assertIn("(last resort)", notice)

    def test_the_substitution_is_visible_in_the_rendered_table(self):
        cells = [self.cell(True, "grid 128"), self.cell(False)]
        orchestrate.substitution_check(cells)
        out = Path(tempfile.mkdtemp())
        orchestrate.report(cells, out)
        table = (out / "report.md").read_text()
        self.assertEqual(table.count("tuned" + orchestrate.SUBSTITUTED_MARK + " |"), 1, table)
        self.assertIn("`substituted_from`", table)

    def test_the_runners_own_spelling_is_what_is_read(self):
        # On the line the OCaml emitter built (the bench_result_line golden): its tuned line carries
        # a measured ship, read as unflagged; the same line with the emitter's key set to true is a
        # substitution. A renamed key on either side fails here.
        golden = [strict_loads(line) for line in OcannlResultLineTest.lines(OcannlResultLineTest)]
        tuned = [r for r in golden if r.get("tune")]
        self.assertTrue(tuned, "no golden line carries a tune object")
        for r in tuned:
            self.assertIs(r["tune"]["ship_substituted"], False)
            self.assertIn("substituted_from", r["tune"])
        self.assertEqual(orchestrate.substitution_check(tuned), [])
        flipped = [dict(r, tune=dict(r["tune"], ship_substituted=True)) for r in tuned]
        self.assertEqual(orchestrate.substitution_check(flipped), flipped)


class PeakMemoryTest(unittest.TestCase):
    """gh-ocannl-1006: the report's peak-device-memory column, and the probes behind it.

    The column exists so that a footprint-for-time trade (gh-ocannl-616) is visible as something
    other than pure cost. Two things make it misleadable, and both are pinned here: a cell whose
    framework has no device counter must read as "not measured" and never as "no footprint", and a
    row must say WHICH counter it carries, since an allocator's high-water mark, a gauge sampled at
    step boundaries and a driver-level figure are three different quantities.
    """

    SEAM_TAG = "ocannl-seam"
    SEAM_SOURCE = "OCANNL allocator seam high-water (requested bytes, all backends)"
    SAMPLED_TAG = "mps-driver"
    SAMPLED_SOURCE = "torch.mps.driver_allocated_memory (driver bytes, sampled at step boundaries)"

    def cell(self, framework, backend, peak=None, tag=None, source=None, p50=1.0):
        r = cell(framework, backend, "default", [2.3, 2.2, 2.1], p50=p50)
        r["peak_memory_bytes"] = peak
        r["peak_memory_counter"] = tag
        r["peak_memory_source"] = source
        return r

    def rendered(self, rows):
        orchestrate.parity_check(rows)
        out = Path(tempfile.mkdtemp())
        orchestrate.report(rows, out)
        return (out / "report.md").read_text()

    def test_the_column_appears_once_a_row_measured_a_footprint(self):
        text = self.rendered(
            [self.cell("ocannl", "metal", 3 * orchestrate.MIB, self.SEAM_TAG, self.SEAM_SOURCE)]
        )

        self.assertIn(" peak MiB |", text)
        self.assertIn("| 3.0 ocannl-seam |", text)

    def test_a_section_where_nothing_measured_one_carries_no_column(self):
        # The same rule every other conditional column follows: an empty column is a column of
        # dashes claiming a measurement was possible and skipped.
        text = self.rendered([self.cell("pytorch", "cpu")])

        self.assertNotIn(" peak MiB |", text)
        self.assertNotIn("gh-ocannl-1006", text)

    def test_a_cell_with_no_counter_renders_a_dash_beside_one_that_has_it(self):
        # The case the column is most easily misread in: a pytorch/cpu row next to a measured one.
        # A zero there would say the workload has no footprint, which is a claim nobody made.
        text = self.rendered(
            [
                self.cell(
                    "ocannl", "cc", 5 * orchestrate.MIB, self.SEAM_TAG, self.SEAM_SOURCE, p50=1.0
                ),
                self.cell("pytorch", "cpu", p50=2.0),
            ]
        )

        self.assertIn(" peak MiB |", text)
        self.assertIn("| 5.0 ocannl-seam |", text)
        self.assertEqual(orchestrate.peak_memory_mib({"peak_memory_bytes": None}), "\u2014")
        self.assertIn("\u2014", text)
        self.assertNotIn("| 0.0 |", text)

    def test_a_measured_zero_is_not_a_missing_measurement(self):
        # `is not None`, not truthiness: a counter that legitimately read zero has measured the
        # cell, and rendering it as a dash would lose that the cell was measured at all.
        self.assertEqual(orchestrate.peak_memory_mib({"peak_memory_bytes": 0}), "0.0")
        text = self.rendered([self.cell("ocannl", "cc", 0, self.SEAM_TAG, self.SEAM_SOURCE)])

        self.assertIn(" peak MiB |", text)
        self.assertIn("| 0.0 ocannl-seam |", text)

    def test_the_legend_names_every_counter_the_section_carries(self):
        rows = [
            self.cell(
                "ocannl", "metal", 3 * orchestrate.MIB, self.SEAM_TAG, self.SEAM_SOURCE, p50=1.0
            ),
            self.cell(
                "pytorch", "mps", 7 * orchestrate.MIB, self.SAMPLED_TAG, self.SAMPLED_SOURCE,
                p50=2.0,
            ),
            self.cell("pytorch", "cpu", p50=3.0),
        ]

        self.assertEqual(
            orchestrate.peak_memory_counters(rows),
            [(self.SEAM_TAG, self.SEAM_SOURCE), (self.SAMPLED_TAG, self.SAMPLED_SOURCE)],
        )
        text = self.rendered(rows)
        self.assertIn(self.SEAM_SOURCE, text)
        self.assertIn(self.SAMPLED_SOURCE, text)
        # And what the reader has to do with the difference, said where the numbers are read.
        self.assertIn("lower bound", text)

    def test_each_row_names_its_own_counter_not_just_the_section(self):
        # The finding of review round 1: with two counters in one section, a list of which
        # counters occur SOMEWHERE leaves every individual number unattributed -- and these two
        # numbers differ by an order of magnitude for reasons of counter kind, not footprint.
        rows = [
            self.cell(
                "ocannl", "metal", 3 * orchestrate.MIB, self.SEAM_TAG, self.SEAM_SOURCE, p50=1.0
            ),
            self.cell(
                "pytorch", "mps", 70 * orchestrate.MIB, self.SAMPLED_TAG, self.SAMPLED_SOURCE,
                p50=2.0,
            ),
        ]
        text = self.rendered(rows)

        self.assertIn("| 3.0 ocannl-seam |", text)
        self.assertIn("| 70.0 mps-driver |", text)

    def test_a_row_whose_runner_named_no_counter_still_shows_its_bytes(self):
        # An artifact predating the tag: the bytes stand, without a tag it does not have. The
        # alternative -- dropping the row's number -- would lose a measurement over a missing label.
        self.assertEqual(
            orchestrate.peak_memory_mib({"peak_memory_bytes": 3 * orchestrate.MIB}), "3.0"
        )

    def test_an_unmeasured_row_contributes_no_counter_name(self):
        # A source string on a row with no bytes is a counter that reported nothing; naming it in
        # the legend would offer the reader a column entry that does not exist.
        rows = [self.cell("pytorch", "mps", None, self.SAMPLED_TAG, self.SAMPLED_SOURCE)]

        self.assertEqual(orchestrate.peak_memory_counters(rows), [])

    def test_the_result_line_carries_the_bytes_and_the_counter_together(self):
        probe = bench_common.PeakMemoryProbe(
            tag="fake", source="fake counter", read=lambda: 4096
        )

        self.assertEqual(
            bench_common.peak_memory_fields(probe),
            {
                "peak_memory_bytes": 4096,
                "peak_memory_counter": "fake",
                "peak_memory_source": "fake counter",
            },
        )
        # All None, never some without the others: bytes whose counter is unnamed would be compared
        # with a different quantity in the next row.
        self.assertEqual(
            bench_common.peak_memory_fields(None),
            {
                "peak_memory_bytes": None,
                "peak_memory_counter": None,
                "peak_memory_source": None,
            },
        )

    def test_reading_the_fields_closes_the_window_where_it_is_called(self):
        # Review round 1: `--retime` times a second block after the reported one. A high-water
        # counter keeps accumulating through it while a sampled probe, taking no samples there,
        # does not -- so a window closed at result-assembly time would mean different steps
        # depending on the counter's kind. `peak_memory_fields` reads, so calling it before the
        # retime block is what makes the window the reported steps' for both kinds.
        readings = iter([100, 900])
        probe = bench_common.PeakMemoryProbe(
            tag="fake-hw", source="fake high-water", read=lambda: next(readings)
        )
        probe.start()

        self.assertEqual(bench_common.peak_memory_fields(probe)["peak_memory_bytes"], 100)
        # A later read would have seen the retimed block's higher mark.
        self.assertEqual(probe.read(), 900)

    def test_a_high_water_counter_is_read_not_sampled(self):
        # The allocator keeps the maximum itself, so the value at the end of the window is the
        # window's peak even though the gauge has since fallen.
        readings = iter([100, 900, 900])
        resets = []
        probe = bench_common.PeakMemoryProbe(
            tag="fake-hw",
            source="fake high-water",
            read=lambda: next(readings),
            reset=lambda: resets.append(1),
        )
        probe.start()
        probe.sample()

        self.assertEqual(resets, [1])
        self.assertEqual(probe.read(), 100)

    def test_a_sampled_gauge_keeps_the_maximum_over_the_window(self):
        # Including the reading taken when the window opened: the workload's weights are already
        # resident, and a peak starting from zero would report only what the steps added to them.
        readings = iter([500, 1500, 700])
        probe = bench_common.PeakMemoryProbe(
            tag="fake-gauge", source="fake gauge", read=lambda: next(readings), sampled=True
        )
        probe.start()
        probe.sample()
        probe.sample()

        self.assertEqual(probe.read(), 1500)

    def test_a_sampled_window_starting_after_a_bigger_one_does_not_inherit_it(self):
        # `start()` drops the previous window's samples, which is what makes the column the TIMED
        # STEPS' footprint rather than the process's.
        probe = bench_common.PeakMemoryProbe(
            tag="fake-gauge", source="fake gauge", read=iter([9000, 200, 300]).__next__,
            sampled=True,
        )
        probe.start()
        probe.start()
        probe.sample()

        self.assertEqual(probe.read(), 300)

    def test_torch_offers_no_counter_on_cpu_and_a_high_water_one_on_cuda(self):
        # cpu is the None case on purpose: torch allocates host memory through the system
        # allocator, and an RSS figure is not the device footprint this column reports.
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                max_memory_allocated=lambda: 2048,
                reset_peak_memory_stats=lambda: None,
            ),
            mps=types.SimpleNamespace(driver_allocated_memory=lambda: 4096),
        )

        self.assertIsNone(bench_common.torch_peak_memory(torch, "cpu"))
        cuda = bench_common.torch_peak_memory(torch, "cuda")
        cuda.start()
        self.assertEqual(cuda.read(), 2048)
        # Requested bytes off the caching allocator -- the same quantity as OCANNL's seam, which is
        # what makes those two rows comparable. Not `max_memory_reserved`.
        self.assertIn("max_memory_allocated", cuda.source)
        self.assertEqual(cuda.tag, "cuda-hw")
        mps = bench_common.torch_peak_memory(torch, "mps")
        mps.start()
        self.assertEqual(mps.read(), 4096)
        self.assertIn("sampled", mps.source)

    def test_a_torch_without_the_mps_counter_reports_none_rather_than_guessing(self):
        torch = types.SimpleNamespace(cuda=None, mps=types.SimpleNamespace())

        self.assertIsNone(bench_common.torch_peak_memory(torch, "mps"))


class DominantKernelTest(unittest.TestCase):
    """gh-ocannl-1006: the report's %-of-peak column for the dominant kernel.

    What makes the column misleadable is pinned here, against the OCaml emitter's own objects
    where it matters: the number is printed only on an exact count, every no-value verdict says
    which one it is in the cell, the ceiling a number is a percentage OF is named in the section
    header with its constants and whose they are, and the header says the constants are class
    constants that certify nothing about the device.
    """

    GOLDEN = HERE.parent / "test/operations/bench_result_line.expected"
    PREFIX = "dominant_kernel "

    def emitted(self):
        """The OCaml emitter's fabricated objects, one per verdict, by name."""
        out = {}
        for line in self.GOLDEN.read_text().splitlines():
            if line.startswith(self.PREFIX):
                name, _, obj = line[len(self.PREFIX):].partition(": ")
                out[name] = strict_loads(obj)
        return out

    def row(self, framework, backend, kernel, p50=1.0):
        r = cell(framework, backend, "default", [2.3, 2.2, 2.1], p50=p50)
        r["dominant_kernel"] = kernel
        return r

    def rendered(self, rows):
        orchestrate.parity_check(rows)
        out = Path(tempfile.mkdtemp())
        orchestrate.report(rows, out)
        return (out / "report.md").read_text()

    def test_the_emitter_offers_every_verdict(self):
        # The fabricated set is what the rendering tests below stand on; a verdict missing from it
        # would be a verdict whose cell nobody has looked at.
        verdicts = {k["verdict"] for k in self.emitted().values()}
        self.assertEqual(
            verdicts, {"exact", "approximate", "opaque", "no-ceiling", "no-kernel"}
        )

    def test_every_emitted_verdict_renders_as_a_named_cell(self):
        for name, k in self.emitted().items():
            with self.subTest(name=name):
                text = orchestrate.dominant_kernel_cell({"dominant_kernel": k})
                self.assertNotIn("UNKNOWN", text)
                self.assertNotEqual(text, "\u2014")
                if k["verdict"] == "exact":
                    self.assertIn("%", text)
                else:
                    # No number on an inexact count, a missing ceiling or no kernel at all.
                    self.assertNotIn("%", text)
                    self.assertIn(orchestrate.DOMINANT_KERNEL_MARK[k["verdict"]], text)

    def test_an_exact_cell_names_its_ceiling_leg_and_kernel(self):
        k = self.emitted()["exact"]

        self.assertEqual(
            orchestrate.dominant_kernel_cell({"dominant_kernel": k}),
            "25.0% f32 memory \u00b7 k3/12 4.000 ms w1.grad 'b1'",
        )

    def test_an_inexact_cell_keeps_the_kernels_identity_but_prints_no_number(self):
        k = self.emitted()["approximate"]

        self.assertEqual(
            orchestrate.dominant_kernel_cell({"dominant_kernel": k}),
            "approx \u00b7 k3/12 4.000 ms w1.grad 'b1'",
        )

    def test_untimed_kernels_are_counted_in_the_cell(self):
        # The dominant kernel is dominant among the kernels that could be timed alone; the cell
        # says how many could not.
        k = dict(self.emitted()["exact"], declined=2)

        self.assertIn("(+2 untimed)", orchestrate.dominant_kernel_cell({"dominant_kernel": k}))

    def test_a_row_without_the_instrument_is_a_dash_beside_one_with_it(self):
        text = self.rendered(
            [
                self.row("ocannl", "metal", self.emitted()["exact"], p50=1.0),
                self.row("pytorch", "mps", None, p50=2.0),
            ]
        )

        self.assertIn(" kernel %peak |", text)
        self.assertIn("| 25.0% f32 memory \u00b7 k3/12 4.000 ms w1.grad 'b1' |", text)
        pytorch_row = next(l for l in text.splitlines() if l.startswith("| pytorch"))
        self.assertIn("| \u2014 |", pytorch_row)

    def test_a_section_where_no_row_ran_the_instrument_carries_no_column(self):
        text = self.rendered([self.row("pytorch", "cpu", None)])

        self.assertNotIn("kernel %peak", text)

    def test_the_header_names_each_ceiling_with_its_constants_and_whose_they_are(self):
        emitted = self.emitted()
        rows = [
            self.row("ocannl", "metal", emitted["exact"], p50=1.0),
            self.row("ocannl", "cc", emitted["exact f16-native"], p50=2.0),
        ]

        self.assertEqual(
            orchestrate.dominant_kernel_ceilings(rows),
            [
                ("f32", "fab class constant", 5e12, 1e12),
                ("f16-native", "fab class constant", 1e13, 1e12),
            ],
        )
        text = self.rendered(rows)
        self.assertIn("`f32` = fab class constant: 5e+12 FLOP/s, 1e+12 B/s", text)
        self.assertIn("`f16-native` = fab class constant: 1e+13 FLOP/s, 1e+12 B/s", text)
        # The recorded decision (gh-ocannl-1006): what the constants are, said where they are read.
        self.assertIn("certifies nothing about the device", text)
        self.assertIn("ranks before/after on one box", text)

    def test_a_section_with_no_scored_row_says_so_rather_than_listing_nothing(self):
        text = self.rendered([self.row("ocannl", "cc", self.emitted()["no-ceiling (no constants)"])])

        self.assertIn("none in this section", text)
        self.assertIn("| no ceiling \u00b7 k3/12", text)

    def test_the_result_line_carries_the_object_through_the_report_unchanged(self):
        # results.jsonl is what a later re-render reads: the object must survive serialization.
        k = self.emitted()["exact"]
        out = Path(tempfile.mkdtemp())
        rows = [self.row("ocannl", "metal", k)]
        orchestrate.parity_check(rows)
        orchestrate.report(rows, out)
        (saved,) = [strict_loads(l) for l in (out / "results.jsonl").read_text().splitlines()]

        self.assertEqual(saved["dominant_kernel"], k)


class MeasurementProtocolTest(unittest.TestCase):
    """gh-ocannl-1008: the one measurement loop both Python runners call, pinned as a trace.

    The trace records every clock read, step, loss read, sync and probe call in order, so a change
    to where a timing boundary, a sync or the memory window sits fails here once instead of having
    to be made identically in two runners.
    """

    class Loss:
        def __init__(self, log, k):
            self.log, self.k = log, k

        def item(self):
            self.log.append("item")
            return float(self.k)

    class Probe:
        tag, source = "fake", "fake counter"

        def __init__(self, log):
            self.log = log

        def start(self):
            self.log.append("open")

        def sample(self):
            self.log.append("sample")

        def read(self):
            self.log.append("close")
            return 7

    def measure(self, meta, probe=False, retime=False):
        log = []
        ticks = iter(range(1000))

        def clock():
            log.append("clock")
            return next(ticks)  # one second per read, so every interval below is exact

        def step(k):
            log.append(f"step {k}")
            return self.Loss(log, k)

        clock_only = types.SimpleNamespace(perf_counter=clock)
        with unittest.mock.patch.object(bench_common, "time", clock_only):
            result = bench_common.run_protocol(
                step,
                lambda: log.append("sync"),
                meta,
                peak_memory=self.Probe(log) if probe else None,
                retime=retime,
            )
        return log, result

    META = {"parity_steps": "2", "warmup_steps": "1", "timed_steps": "2"}

    def test_the_phases_run_in_order_and_the_window_closes_before_retiming(self):
        log, result = self.measure(self.META, probe=True, retime=True)

        synced_step = lambda k, probe: ["clock", f"step {k}", "sync", "clock"] + probe
        self.assertEqual(
            log,
            # step 0 is the compile probe, timed up to its loss on the host; parity step 1
            ["clock", "step 0", "item", "clock", "step 1", "item"]
            # warmup, then a sync before anything is timed
            + ["step 2", "sync"]
            # the window opens, each synced step is sampled AFTER its elapsed time is taken
            + ["open"] + synced_step(3, ["sample"]) + synced_step(4, ["sample"])
            # the queued block: one sync for all of it, a sample, and the window closes
            + ["clock", "step 5", "step 6", "sync", "clock", "sample", "close"]
            # only then the retime block, synced and unsampled
            + ["sync"] + synced_step(7, []) + synced_step(8, []),
        )
        self.assertEqual(result["compile_s"], 1.0)
        self.assertEqual(result["step_ms"], {"p10": 1000.0, "p50": 1000.0, "p90": 1000.0})
        self.assertEqual(result["queued_step_ms"], 500.0)
        self.assertEqual(result["retime_step_ms"]["p50"], 1000.0)
        self.assertEqual(result["losses"], [0.0, 1.0])
        self.assertEqual(result["timed_steps"], 2)
        self.assertEqual(result["peak_memory_bytes"], 7)
        self.assertEqual(result["peak_memory_counter"], "fake")

    def test_without_a_counter_or_retiming_the_line_says_so(self):
        log, result = self.measure(self.META)

        self.assertNotIn("open", log)
        self.assertEqual(log[-5:], ["clock", "step 5", "step 6", "sync", "clock"])
        self.assertNotIn("retime_step_ms", result)
        # null for all three memory keys, never a zero (see `peak_memory_fields`)
        self.assertEqual(
            {k: v for k, v in result.items() if k.startswith("peak_memory")},
            bench_common.peak_memory_fields(None),
        )
        # gh-ocannl-1006: no per-kernel counts to score, so the %-of-peak column's key is null --
        # present, so the report can tell "not measured" from a runner that predates the column.
        self.assertIsNone(result["dominant_kernel"])
        self.assertEqual(
            set(result),
            {"compile_s", "step_ms", "queued_step_ms", "timed_steps", "losses", "dominant_kernel"}
            | set(bench_common.peak_memory_fields(None)),
        )


class RunnerProvenanceProbeTest(unittest.TestCase):
    """gh-ocannl-644: what the Python runners report, and what they refuse to claim."""

    def test_a_cell_that_runs_no_search_says_so(self):
        self.assertIs(bench_common.tinygrad_searched(None, beam=0), False)
        self.assertIs(bench_common.torch_searched(object(), compiled=False), False)

    def test_beam_counts_distinguish_a_search_from_a_kernel_cache_replay(self):
        self.assertIs(
            bench_common.tinygrad_searched({"call": 4, "hit": 3, "put": 1}, beam=4), True
        )
        self.assertIs(
            bench_common.tinygrad_searched({"call": 3, "hit": 3, "put": 0}, beam=4), False
        )

    def test_a_probe_that_saw_nothing_says_unknown_rather_than_no(self):
        # The internals moved under the probe. A wrong False is exactly the silent claim the
        # field exists to prevent, so this must not read as "replayed a cache".
        self.assertIsNone(bench_common.tinygrad_searched(None, beam=4))
        self.assertIsNone(
            bench_common.tinygrad_searched({"call": 0, "hit": 0, "put": 0}, beam=4)
        )

    def test_torch_reads_the_fx_graph_cache_counters(self):
        class Torch:
            def __init__(self, **counters):
                self._dynamo = type(
                    "d", (), {"utils": type("u", (), {"counters": {"inductor": counters}})}
                )

        self.assertIs(bench_common.torch_searched(Torch(fxgraph_cache_miss=2), True), True)
        self.assertIs(
            bench_common.torch_searched(Torch(fxgraph_cache_hit=2), compiled=True), False
        )
        self.assertIsNone(bench_common.torch_searched(Torch(), compiled=True))
        self.assertIsNone(bench_common.torch_searched(object(), compiled=True))

    def test_a_beam_search_that_writes_no_cache_entry_still_counts_as_a_search(self):
        # CACHELEVEL=0 / IGNORE_BEAM_CACHE: the search runs and touches no cache. Counting cache
        # traffic alone cannot see it, which is why the probe counts calls too.
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 0, "put": 0}, beam=4), True)
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 1, "put": 0}, beam=4), True)
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 4, "put": 0}, beam=4), False)

    def test_a_bypassed_torch_graph_is_codegen_not_a_replay(self):
        # A run with more than one graph is mixed: one served from the cache, one the cache
        # refused, is still a graph compiled in the timing process.
        class Torch:
            def __init__(self, **counters):
                self._dynamo = type(
                    "d", (), {"utils": type("u", (), {"counters": {"inductor": counters}})}
                )

        mixed = Torch(fxgraph_cache_hit=3, fxgraph_cache_bypass=1)
        self.assertIs(bench_common.torch_searched(mixed, compiled=True), True)
        self.assertIs(
            bench_common.torch_searched(Torch(fxgraph_cache_bypass=2), compiled=True), True
        )

    def test_the_beam_instrument_counts_only_beam_entries_and_passes_values_through(self):
        stored = {("beam_search", "k1"): "winner"}
        calls = []

        class Search:
            @staticmethod
            def diskcache_get(table, key):
                calls.append(("get", table, key))
                return stored.get((table, key))

            @staticmethod
            def diskcache_put(table, key, val):
                stored[(table, key)] = val
                return val

        module = types.ModuleType("tinygrad.engine.search")
        module.diskcache_get, module.diskcache_put = Search.diskcache_get, Search.diskcache_put
        packages = {
            "tinygrad": types.ModuleType("tinygrad"),
            "tinygrad.engine": types.ModuleType("tinygrad.engine"),
            "tinygrad.engine.search": module,
        }
        packages["tinygrad.engine"].search = module
        with unittest.mock.patch.dict(sys.modules, packages):
            counts = bench_common.instrument_tinygrad_beam()
            self.assertEqual(module.diskcache_get("beam_search", "k1"), "winner")
            self.assertIsNone(module.diskcache_get("beam_search", "k2"))
            module.diskcache_put("beam_search", "k2", "found")
            module.diskcache_get("compile", "unrelated")
            module.diskcache_put("compile", "unrelated", "x")

        self.assertEqual(counts, {"call": 0, "hit": 1, "put": 1})
        self.assertIn(("get", "compile", "unrelated"), calls)

    def test_an_uninstrumentable_tinygrad_is_not_an_error(self):
        module = types.ModuleType("tinygrad.engine.search")  # no diskcache helpers
        packages = {
            "tinygrad": types.ModuleType("tinygrad"),
            "tinygrad.engine": types.ModuleType("tinygrad.engine"),
            "tinygrad.engine.search": module,
        }
        with unittest.mock.patch.dict(sys.modules, packages):
            self.assertIsNone(bench_common.instrument_tinygrad_beam())


class FixtureDigestTest(unittest.TestCase):
    """gh-ocannl-645: what a report's numbers were measured on is recorded, not assumed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def write_fixture(path, content=b"weights", metadata=None, tensors=None, extra_padding=0):
        """Write a tiny safetensors fixture for digest tests.

        This is test construction, not a second production writer: the records deliberately keep
        insertion order so tests can vary the non-canonical serialization independently of the
        content identity.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"name": path.stem, **(metadata or {})}
        tensors = tensors or [("weights", "U8", [len(content)], content)]
        header = {"__metadata__": metadata}
        payload = bytearray()
        for tensor_name, dtype, shape, tensor_content in tensors:
            start = len(payload)
            payload.extend(tensor_content)
            header[tensor_name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, len(payload)],
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        padding = (-len(encoded)) % 8 + extra_padding
        encoded += b" " * padding
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
        return path

    def fixture(self, name, content=b"weights", **kwargs):
        return self.write_fixture(self.dir / name, content, **kwargs)

    def test_digest_canonicalizes_metadata_and_tensor_serialization_order(self):
        a = self.fixture(
            "a.safetensors",
            metadata={"name": "same", "zeta": "last", "alpha": "first"},
            tensors=[("z", "U8", [2], b"zz"), ("a", "U8", [2], b"aa")],
        )
        b = self.fixture(
            "b.safetensors",
            metadata={"name": "same", "alpha": "first", "zeta": "last"},
            tensors=[("a", "U8", [2], b"aa"), ("z", "U8", [2], b"zz")],
        )

        self.assertNotEqual(a.read_bytes(), b.read_bytes(), "control: serialization really differs")
        self.assertEqual(fixture_digest.sha256_file(a), fixture_digest.sha256_file(b))

    def test_content_identity_ignores_physical_header_size(self):
        fixture = self.fixture("padded.safetensors", b"same content")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fixture], "m4-max")
        recorded = fixture_digest.read_digests(digests)
        size = fixture.stat().st_size

        self.write_fixture(fixture, b"same content", extra_padding=16)

        self.assertNotEqual(fixture.stat().st_size, size, "control: physical size changed")
        self.assertEqual(fixture_digest.status(fixture, recorded)[0], "MATCH")
        self.assertEqual(fixture_digest.record(digests, [fixture], "m4-max"), [])

    def test_digest_changes_for_every_piece_of_runner_visible_content(self):
        baseline = self.fixture(
            "baseline.safetensors",
            metadata={"name": "same", "seed": "7"},
            tensors=[("weights", "U8", [2], b"ab")],
        )
        expected = fixture_digest.sha256_file(baseline)
        variants = {
            "metadata": dict(metadata={"name": "same", "seed": "8"}),
            "tensor name": dict(
                metadata={"name": "same", "seed": "7"},
                tensors=[("bias", "U8", [2], b"ab")],
            ),
            "dtype": dict(
                metadata={"name": "same", "seed": "7"},
                tensors=[("weights", "I8", [2], b"ab")],
            ),
            "shape": dict(
                metadata={"name": "same", "seed": "7"},
                tensors=[("weights", "U8", [1, 2], b"ab")],
            ),
            "payload": dict(
                metadata={"name": "same", "seed": "7"},
                tensors=[("weights", "U8", [2], b"ac")],
            ),
        }

        for index, (fact, kwargs) in enumerate(variants.items()):
            with self.subTest(fact=fact):
                changed = self.fixture(f"changed-{index}.safetensors", **kwargs)
                self.assertNotEqual(fixture_digest.sha256_file(changed), expected)

    def test_a_legacy_row_stays_raw_until_original_content_is_recorded(self):
        original = self.fixture(
            "original.safetensors",
            metadata={"name": "same", "zeta": "last", "alpha": "first"},
            tensors=[("z", "U8", [2], b"zz"), ("a", "U8", [2], b"aa")],
        )
        regenerated = self.fixture(
            "regenerated/original.safetensors",
            metadata={"name": "same", "alpha": "first", "zeta": "last"},
            tensors=[("a", "U8", [2], b"aa"), ("z", "U8", [2], b"zz")],
        )
        self.assertEqual(fixture_digest.sha256_file(original), fixture_digest.sha256_file(regenerated))
        self.assertNotEqual(original.read_bytes(), regenerated.read_bytes())
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.header(["m4-max"])
            + f"{fixture_digest._raw_sha256_file(original)}  {original.stat().st_size}  "
            "original.safetensors  m4-max\n"
        )

        entries = fixture_digest.read_digests(digests)
        self.assertEqual(fixture_digest.status(original, entries)[0], "MATCH")
        self.assertEqual(fixture_digest.status(regenerated, entries)[0], "MISMATCH")
        fixture_digest.record(digests, [original], "m4-max")
        migrated = fixture_digest.read_digests(digests)
        self.assertEqual(migrated[original.name][0].kind, "content-v1")
        self.assertEqual(fixture_digest.status(regenerated, migrated)[0], "MATCH")

    def test_malformed_or_truncated_fixtures_are_refused(self):
        malformed = self.dir / "malformed.safetensors"
        malformed.write_bytes((8).to_bytes(8, "little") + b"not json")
        complete = self.fixture("truncated.safetensors", b"payload")
        complete.write_bytes(complete.read_bytes()[:-1])

        wrong_width = self.fixture(
            "wrong-width.safetensors",
            tensors=[("weights", "F64", [1], b"x")],
        )

        for path in (malformed, complete, wrong_width):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                fixture_digest.sha256_file(path)

    #: The origin `gen_fixtures.main` records under when none is given, pinned so the
    #: generator tests do not depend on what this box happens to be called.
    this_box = "test-box"

    def gen_fixtures_module(self):
        """`gen_fixtures`, imported without the bench venv, with `build` never reached.

        It imports numpy and safetensors at module scope, but the property the tests below pin --
        the ORDER of `main()`'s steps -- patches `build` out, so nothing here draws a random
        number or writes a safetensors file. Stubbing the two imports is what keeps the check
        running under the bare `python3` the dune rule invokes: skipping there would mean the
        one place it runs unattended never evaluates it.
        """
        stub = types.ModuleType("safetensors.numpy")
        stub.save_file = None
        stubs = {
            "numpy": types.ModuleType("numpy"),
            "safetensors": types.ModuleType("safetensors"),
            "safetensors.numpy": stub,
        }
        with unittest.mock.patch.dict(sys.modules, stubs):
            sys.modules.pop("gen_fixtures", None)
            import gen_fixtures
        patch = unittest.mock.patch.object(
            fixture_digest.platform, "node", return_value=self.this_box
        )
        patch.start()
        self.addCleanup(patch.stop)
        return gen_fixtures

    def check(self, *args, **kwargs):
        """check_fixture_digests with its per-fixture log kept out of the test output."""
        with contextlib.redirect_stdout(io.StringIO()):
            return orchestrate.check_fixture_digests(*args, **kwargs)

    def test_recording_round_trips(self):
        fx = self.fixture("mlp_small.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        changes = fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(
            [(name, origin, was) for name, origin, was, _ in changes],
            [("mlp_small.safetensors", "rog-nv", None)],
        )
        entries = fixture_digest.read_digests(digests)
        self.assertEqual(
            entries[fx.name],
            [fixture_digest.Entry(fixture_digest.sha256_file(fx), fx.stat().st_size, "rog-nv")],
        )
        self.assertEqual(fixture_digest.status(fx, entries)[0], "MATCH")

    def test_a_match_names_the_box_whose_bytes_they_are(self):
        # gh-ocannl-759: "which bytes" was never the whole question. The reports compare boxes,
        # so a number has to carry whose workload it is on, not only that it is on a pinned one.
        fx = self.fixture("gpt2_mini.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")

        verdict, _, _, origins = fixture_digest.status(fx, fixture_digest.read_digests(digests))

        self.assertEqual((verdict, origins), ("MATCH", "minix"))

    def test_regenerating_different_bytes_is_announced_as_a_change(self):
        fx = self.fixture("gpt2_mini.safetensors", b"generated at spec revision A")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        (was,) = fixture_digest.read_digests(digests)[fx.name]

        self.write_fixture(fx, b"generated at spec revision B")
        changes = fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(len(changes), 1)
        name, origin, previous, now = changes[0]
        self.assertEqual((name, origin, previous), (fx.name, "rog-nv", was))
        self.assertEqual([now], fixture_digest.read_digests(digests)[fx.name])

    def test_regenerating_one_fixture_keeps_the_others_recorded(self):
        # gen_fixtures.py takes a spec list; regenerating one workload must not drop the
        # identities of the fixtures already on disk, or the pin silently narrows to one.
        a, b = self.fixture("a.safetensors", b"aaa"), self.fixture("b.safetensors", b"bbb")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [a, b], "rog-nv")

        fixture_digest.record(digests, [a], "rog-nv")

        self.assertEqual(set(fixture_digest.read_digests(digests)), {a.name, b.name})

    def test_two_boxes_bytes_are_both_recorded_and_both_match(self):
        # The state gh-ocannl-759 found and this format exists for: mlp_small and gpt2_mini hash
        # differently on minix and rog-nv at identical sizes (different venvs, different numpy
        # streams). Neither box's published numbers may be evicted to make the other's fit.
        fx = self.fixture("mlp_small.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")

        self.write_fixture(fx, b"rog-nv bytes")
        fixture_digest.record(digests, [fx], "rog-nv")

        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries[fx.name]], ["minix", "rog-nv"])
        self.assertEqual(fixture_digest.status(fx, entries)[3], "rog-nv")
        self.write_fixture(fx, b"minix bytes")
        self.assertEqual(fixture_digest.status(fx, entries)[3], "minix")

    def test_one_box_regenerating_does_not_unpin_the_other(self):
        # The trap the issue was filed about: with one entry per name, whichever box regenerated
        # first pinned its numpy stream and the other box's untouched fixtures then read as a
        # CHANGED workload -- a false alarm that would have been "fixed" by re-recording, i.e. by
        # evicting the first box in turn, forever.
        fx = self.fixture("gpt2_mini.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")
        minix_entries = fixture_digest.read_digests(digests)

        self.write_fixture(fx, b"rog-nv regenerates its own copy")
        fixture_digest.record(digests, [fx], "rog-nv")

        entries = fixture_digest.read_digests(digests)
        self.assertEqual(entries[fx.name][0], minix_entries[fx.name][0])
        self.write_fixture(fx, b"minix bytes")
        self.assertEqual(fixture_digest.status(fx, entries)[0], "MATCH")

    def test_agreeing_boxes_are_reported_as_agreeing(self):
        # The happy outcome of a coordinated regeneration, and it should be visible: one set of
        # bytes recorded by both boxes reads as both, not as an arbitrary one of them.
        fx = self.fixture("lenet.safetensors", b"the coordinated bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        fixture_digest.record(digests, [fx], "minix")

        verdict, _, _, origins = fixture_digest.status(fx, fixture_digest.read_digests(digests))

        self.assertEqual((verdict, origins), ("MATCH", "minix,rog-nv"))

    def test_a_differently_generated_fixture_does_not_match(self):
        # The hazard: the difference is applied uniformly to every cell, so the cross-cell parity
        # gate certifies it. Only the digest contradicts the report's workload name.
        fx = self.fixture("lenet.safetensors", b"the bytes the report was measured on")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        entries = fixture_digest.read_digests(digests)

        self.write_fixture(fx, b"the bytes someone regenerated later")

        verdict, _, _, origins = fixture_digest.status(fx, entries)
        self.assertEqual((verdict, origins), ("MISMATCH", None))

    def test_bytes_no_recorded_box_has_still_mismatch(self):
        # Multi-origin widens what MATCHes to "some recorded box's bytes" -- it must not widen it
        # to "recorded under this name", or the gate becomes a filename check.
        fx = self.fixture("mlp_wide.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")
        self.write_fixture(fx, b"rog-nv bytes")
        fixture_digest.record(digests, [fx], "rog-nv")
        entries = fixture_digest.read_digests(digests)

        self.write_fixture(fx, b"a third box nobody recorded")

        self.assertEqual(fixture_digest.status(fx, entries)[0], "MISMATCH")

    def test_an_unrecorded_fixture_is_not_silently_accepted(self):
        fx = self.fixture("mystery.safetensors")

        self.assertEqual(fixture_digest.status(fx, {})[0], "UNRECORDED")

    def test_a_malformed_line_is_an_error_not_a_dropped_pin(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  lenet.safetensors\n")

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_an_unattributed_legacy_line_is_refused_not_guessed(self):
        # A three-field line is the pre-759 format. Adopting it under a guessed origin would
        # record a coin flip as a fact, and the boxes' bytes really do differ.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  17  lenet.safetensors\n")

        with self.assertRaises(ValueError) as refused:
            fixture_digest.read_digests(digests)

        self.assertIn("--record", str(refused.exception))

    def test_one_origin_cannot_be_recorded_twice_for_one_fixture(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.HEADER
            + "aa  1  lenet.safetensors  rog-nv\nbb  2  lenet.safetensors  rog-nv\n"
        )

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_two_files_cannot_be_recorded_under_one_fixture_name(self):
        # `read_digests` refuses one (name, origin) twice; recording must not manufacture the same
        # ambiguity from the other side. Two boxes' copies of one workload passed to one command
        # (`box-a/mlp_small.safetensors`, `box-b/mlp_small.safetensors`) had the later replace the
        # earlier, and the replacement was ANNOUNCED as a changed workload -- a regeneration that
        # never happened, with the first box's bytes gone from the file.
        (self.dir / "box-a").mkdir()
        (self.dir / "box-b").mkdir()
        a = self.fixture("box-a/mlp_small.safetensors", b"box-a bytes")
        b = self.fixture("box-b/mlp_small.safetensors", b"box-b bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [a, b], "rog-nv")

        self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_repeating_one_fixture_path_is_not_an_ambiguity(self):
        # The control: the guard refuses two files claiming one name, not one file named twice --
        # which says nothing contradictory and records exactly what it always did.
        fx = self.fixture("mlp_small.safetensors", b"one box's bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        fixture_digest.record(digests, [fx, fx], "rog-nv")

        self.assertEqual(
            [(e.origin, e.sha256) for e in fixture_digest.read_digests(digests)[fx.name]],
            [("rog-nv", fixture_digest.sha256_file(fx))],
        )

    def test_an_origin_with_whitespace_is_refused(self):
        # The format is whitespace-split, so an origin containing a space would write a line that
        # reads back as a different (or malformed) entry.
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "rog nv")

        self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_an_origin_unsafe_for_a_log_filename_is_refused(self):
        # The same identifiers key sweep log paths. A slash creates an unintended subdirectory;
        # backslash and colon are not portable to the Windows measurement boxes.
        fx = self.fixture("lenet.safetensors")
        for origin in ("rog/nv", r"rog\nv", "rog:nv", ".hidden", "nul"):
            digests = self.dir / f"{origin.encode().hex()}.txt"
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    fixture_digest.record(digests, [fx], origin)
                self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_a_fixture_name_with_whitespace_is_refused(self):
        # The name-side twin of test_an_origin_with_whitespace_is_refused: `write_digests` emits
        # names unescaped into the whitespace-split format, so `a b.safetensors` would write a
        # line every later read refuses -- and recording rewrites the whole file, so one such
        # recording breaks the checked-in record. Refused before the file is touched.
        fx = self.fixture("a b.safetensors", b"whatever bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "rog-nv")

        self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_a_fixture_name_gen_fixtures_would_refuse_is_refused(self):
        # Recording existing bytes must not accept a name generation refuses: CON.safetensors is
        # whitespace-free, but on the Windows measurement boxes it names the console device, not
        # a file. Refused before the file is touched.
        fx = self.fixture("CON.safetensors", b"whatever bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "rog-nv")

        self.assertFalse(digests.exists(), "refused before the file is touched")

    def test_an_origin_read_from_the_file_is_held_to_the_same_identity_rules(self):
        # check_origin guards every writer CLI, but a checked-in or hand-merged row bypasses
        # them: a recorded `minix,rocm` parses into a fixture_origin field byte-identical to two
        # agreeing boxes named minix and rocm. Refused at the one insertion point, line named.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.HEADER + "deadbeef  17  lenet.safetensors  minix,rocm\n"
        )

        with self.assertRaises(ValueError) as refused:
            fixture_digest.read_digests(digests)

        self.assertIn("minix,rocm", str(refused.exception))
        self.assertIn(str(digests), str(refused.exception))

    def test_a_name_read_from_the_file_is_held_to_the_same_identity_rules(self):
        # The name-side twin of the origin check above: check_fixture_name guards every
        # recording, but a checked-in or hand-merged `CON.safetensors` row (no file on Windows)
        # would parse fine and be refused only by the next record(), which rewrites the whole
        # file. Refused at the one insertion point, line named.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  17  CON.safetensors  minix\n")

        with self.assertRaises(ValueError) as refused:
            fixture_digest.read_digests(digests)

        self.assertIn("CON.safetensors", str(refused.exception))
        self.assertIn(str(digests), str(refused.exception))

    def test_check_mode_refuses_an_origin_it_would_ignore(self):
        # --check reports against EVERY recorded origin; accepting --origin and doing nothing
        # with it would let `--check --origin "$BOX"` read as having verified that box's bytes.
        fx = self.fixture("lenet.safetensors")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as refused:
                fixture_digest._main(["--check", str(fx), "--origin", "rog-nv"])

        self.assertEqual(refused.exception.code, 2)

    def test_an_origin_that_reads_as_two_origins_is_refused(self):
        # Agreeing boxes are serialized into ONE comma-joined `fixture_origin` field, carried by
        # every result row and report section. A box named `minix,rocm` would publish a field
        # byte-identical to the one two boxes named `minix` and `rocm` produce by agreeing, and no
        # consumer of a published row could tell one box from two. The control is that exact
        # string, arrived at honestly.
        fx = self.fixture("lenet.safetensors", b"same bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "minix,rocm")

        self.assertFalse(digests.exists(), "refused before the file is touched")
        agreed = self.dir / "agreed.txt"
        fixture_digest.record(agreed, [fx], "minix")
        fixture_digest.record(agreed, [fx], "rocm")
        self.assertEqual(
            fixture_digest.status(fx, fixture_digest.read_digests(agreed))[3], "minix,rocm"
        )

    def test_the_checked_in_file_parses_and_names_live_workloads(self):
        # A digest for a workload spec that no longer exists is stale: it pins bytes no current
        # run can produce, and reads as coverage.
        entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
        specs = {p.stem for p in (HERE / "workloads").glob("*.json")}
        self.assertTrue(specs, "no workload specs found next to the test")
        for name in entries:
            self.assertTrue(name.endswith(".safetensors"), name)
            self.assertIn(name[: -len(".safetensors")], specs, name)
            # Re-recording the checked-in fixtures must stay possible under the name rule.
            fixture_digest.check_fixture_name(name)

    def test_the_checked_in_file_attributes_every_entry(self):
        # The whole point of gh-ocannl-759: no entry may be anonymous, because the published
        # numbers are on more than one box's bytes and the reports have to say which.
        entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
        self.assertTrue(entries, "the checked-in digest file records nothing (gh-ocannl-759)")
        for name, recorded in entries.items():
            for e in recorded:
                self.assertTrue(e.origin and e.origin.strip(), name)

    def test_the_checked_in_header_declares_the_measurement_boxes(self):
        digests = HERE / "fixtures" / fixture_digest.DIGEST_FILE

        declared = fixture_digest.declared_measurement_boxes(digests)
        self.assertEqual(declared, ["m4-max", "minix", "rog-nv", "tuf"])

        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = fixture_digest._main(
                ["--list-declared-measurement-boxes", "--digests", str(digests)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), "".join(f"{box}\n" for box in declared))

    def test_listing_boxes_does_not_infer_a_matrix_for_a_legacy_file(self):
        # A pre-gh-ocannl-850 file can still be swept for backend coverage, but its row origins
        # are not a declaration that every measuring host is present. Environment aggregation
        # must therefore receive no matrix rather than quietly treating the observed rows as one.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.HEADER + "deadbeef  17  lenet.safetensors  rog-nv\n"
        )

        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = fixture_digest._main(
                ["--list-declared-measurement-boxes", "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), "")

    def test_checked_in_divergence_accounts_for_copies_of_m4_max(self):
        # gh-ocannl-720 copies the transformer fixtures to other boxes instead of regenerating
        # them, and gh-ocannl-1161 the cifar_resnet ones to rog-nv and minix. Pin that recorded
        # policy and keep missing/different origins visible for every entry.
        digests = HERE / "fixtures" / fixture_digest.DIGEST_FILE
        entries, declared = fixture_digest._read_document(digests)
        specs = [json.loads(spec.read_text()) for spec in (HERE / "workloads").glob("*.json")]
        expected = {f"{spec['name']}.safetensors" for spec in specs if spec.get("model") == "gpt"}
        resnet = {f"{spec['name']}.safetensors" for spec in specs if spec.get("arch") == "resnet"}
        self.assertTrue(expected, "no GPT workload specs found")
        self.assertTrue(resnet, "no cifar_resnet workload specs found")
        on_tuf = {name for name, recorded in entries.items() if any(e.origin == "tuf" for e in recorded)}
        self.assertEqual(on_tuf, expected, "TUF must record every GPT fixture copy")
        copied = expected | resnet
        # Which boxes hold each family's copies: TUF's GPT copies, the resnet leg's three boxes.
        copy_boxes = {name: {"m4-max", "tuf"} for name in expected}
        copy_boxes.update({name: {"m4-max", "minix", "rog-nv"} for name in resnet})

        for name, recorded in entries.items():
            self.assertTrue(recorded, name)
            by_origin = {e.origin: e for e in recorded}
            # The copy rule: every content-v1 row of a GPT fixture is m4-max's bytes, whichever box
            # recorded it (TUF's copies, and each further box's row recorded on the copy it holds),
            # so a box that regenerated its own transformer fixture fails here. Legacy raw-v1 rows
            # predate the copies and stay their boxes' own bytes, divergent from m4-max's.
            copies = {e.origin for e in recorded if e.kind == "content-v1"} if name in copied else set()
            if name in copied:
                self.assertIn("m4-max", by_origin, name)
                self.assertLessEqual(copy_boxes[name], copies, name)
                for origin in copies:
                    self.assertEqual(
                        by_origin[origin].sha256, by_origin["m4-max"].sha256, f"{name} on {origin}"
                    )

            for entry in recorded:
                same = {entry.origin}
                if name in copied and entry.origin in copies:
                    same = copies
                self.assertEqual(
                    set(fixture_digest.divergent_origins(digests, [name], entry.origin)),
                    set(declared) - same,
                    f"{name} from {entry.origin}",
                )

    def test_a_duplicate_box_declaration_is_refused(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.header(["minix", "rog-nv"])
            + f"{fixture_digest.MEASUREMENT_BOXES_FIELD} third-box\n"
        )

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_an_entry_for_an_undeclared_box_is_refused(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(
            fixture_digest.header(["minix"])
            + "deadbeef  17  lenet.safetensors  rog-nv\n"
        )

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_a_regenerating_box_is_told_which_others_it_has_left_behind(self):
        # What gen_fixtures.py prints after regenerating: the other boxes' entries survive, so
        # nothing fails, and that silence is exactly what would let the boxes drift apart again.
        fx = self.fixture("mlp_small.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")
        self.write_fixture(fx, b"rog-nv regenerates")
        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), ["minix"])
        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "minix"), ["rog-nv"])

    def test_a_box_that_is_the_only_one_recorded_leaves_nobody_behind(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), [])

    def test_a_declared_box_with_no_fixture_entry_is_reported_as_left_behind(self):
        # gh-ocannl-850's missing fact: without an independent box declaration, an absent minix
        # row is indistinguishable from minix never measuring this workload and stays silent.
        fx = self.fixture("lenet.safetensors", b"rog-nv regenerates")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["minix", "rog-nv"]))

        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.measurement_boxes(digests), ["minix", "rog-nv"])
        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), ["minix"])

    def test_a_report_names_a_declared_box_with_no_fixture_entry(self):
        fx = self.fixture("mlp_wide.safetensors", b"rog-nv bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["minix", "rog-nv"]))
        fixture_digest.record(digests, [fx], "rog-nv")
        row = cell("ocannl", "cuda", "default", [2.0, 1.9])
        row.update(
            fixture=fx.name,
            fixture_sha256=fixture_digest.sha256_file(fx),
            fixture_origin="rog-nv",
            parity="PASS",
            parity_loss_moved=True,
        )
        out = self.dir / "report"

        orchestrate.report([row], out, digests_path=digests)
        text = (out / "report.md").read_text()

        self.assertIn("measurement boxes declared by `fixtures/DIGESTS.txt`: minix, rog-nv", text)
        self.assertIn("declared measurement box(es) with no entry", text)
        self.assertIn("minix", text)

    def test_raw_result_rows_preserve_the_box_set_and_missing_entries(self):
        # The JSONL outlives its DIGESTS.txt checkout (and partial.jsonl can outlive the run), so
        # the fleet and missing-entry reading belong in the row, not only in report prose.
        fx = self.fixture("mlp_wide.safetensors", b"rog-nv bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["m4-max", "minix", "rog-nv"]))
        fixture_digest.record(digests, [fx], "rog-nv")
        entries = fixture_digest.read_digests(digests)
        boxes = fixture_digest.measurement_boxes(digests)
        stamp = orchestrate.fixture_result_stamp(
            fx, fixture_digest.sha256_file(fx), "rog-nv", entries, boxes
        )
        # This is the stamp main applies before collect writes partial.jsonl. Asserting it before
        # report() matters: report enriches legacy in-memory rows for final JSONL compatibility,
        # but it cannot repair a partial checkpoint left by an interrupted run.
        self.assertEqual(stamp["measurement_boxes"], ["m4-max", "minix", "rog-nv"])
        self.assertEqual(stamp["fixture_missing_origins"], ["m4-max", "minix"])
        row = cell("ocannl", "cuda", "default", [2.0, 1.9])
        row.update(stamp)
        row.update(parity="PASS", parity_loss_moved=True)
        out = self.dir / "report-with-raw-provenance"

        orchestrate.report([row], out, digests_path=digests)
        raw = json.loads((out / "results.jsonl").read_text())

        self.assertEqual(raw["measurement_boxes"], ["m4-max", "minix", "rog-nv"])
        self.assertEqual(raw["fixture_missing_origins"], ["m4-max", "minix"])

    def test_a_coordinated_regeneration_is_not_reported_as_divergence(self):
        # The success case of the cross-box event: both boxes land the SAME bytes. Naming the
        # other box as "still on a different workload" would tell the operator to redo the thing
        # they just succeeded at. Divergence is judged on bytes, never on the name differing.
        fx = self.fixture("gpt2_mini.safetensors", b"the coordinated bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "minix")

        fixture_digest.record(digests, [fx], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [fx.name], "rog-nv"), [])

    def test_divergence_is_reported_per_fixture_not_per_box(self):
        # Boxes can agree on one workload and differ on another; the warning must fire on the
        # second without being suppressed by the first.
        agree = self.fixture("lenet.safetensors", b"same everywhere")
        differ = self.fixture("mlp_wide.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [agree, differ], "minix")
        self.write_fixture(differ, b"rog-nv bytes")
        fixture_digest.record(digests, [agree, differ], "rog-nv")

        self.assertEqual(fixture_digest.divergent_origins(digests, [agree.name], "rog-nv"), [])
        self.assertEqual(
            fixture_digest.divergent_origins(digests, [agree.name, differ.name], "rog-nv"),
            ["minix"],
        )

    def test_an_explicitly_empty_origin_is_refused_not_defaulted(self):
        # How automation fails: `--origin "$BOX"` with $BOX unset. Silently substituting this
        # hostname would attribute the fixture to the wrong box and persist it -- the exact error
        # the file exists to prevent.
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with self.assertRaises(ValueError):
            fixture_digest.record(digests, [fx], "")

    def test_the_advertised_legacy_migration_actually_runs(self):
        # The refusal names a command; that command has to be able to reach a valid file. Without
        # --adopt-legacy, record() would raise from its own read before rewriting anything.
        fx = self.fixture("lenet.safetensors", b"the legacy bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        sha, size = fixture_digest.sha256_file(fx), fx.stat().st_size
        digests.write_text(fixture_digest.HEADER + f"{sha}  {size}  {fx.name}\n")
        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

        with contextlib.redirect_stdout(io.StringIO()):
            code = fixture_digest._main(
                ["--record", str(fx), "--origin", "minix", "--adopt-legacy", "minix",
                 "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries[fx.name]], ["minix"])
        self.assertEqual(fixture_digest.status(fx, entries)[0], "MATCH")

    def test_adopting_legacy_lines_attributes_them_without_changing_bytes(self):
        # The migration states whose the bytes were; it must not restate WHAT they were. A legacy
        # line for a fixture not being recorded now keeps its digest exactly.
        other = "cifar_conv.safetensors"
        fx = self.fixture("lenet.safetensors", b"recorded now")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + f"deadbeef  17  {other}\n")

        fixture_digest.record(digests, [fx], "minix", legacy_origin="minix")

        entries = fixture_digest.read_digests(digests)
        self.assertEqual(
            entries[other], [fixture_digest.Entry("deadbeef", 17, "minix", "raw-v1")]
        )

    def test_a_migration_does_not_split_one_box_across_two_origin_names(self):
        # The migration writes two facts -- whose the old rows were, and whose the bytes on disk
        # are -- and resolved them from two independent sources: `--adopt-legacy rog-nv` on a host
        # that calls itself `rog-nv-wsl` attributed the rows to rog-nv and recorded the fixtures
        # as rog-nv-wsl. One box then wears two origin names, which reads downstream as two boxes
        # agreeing -- or, for a fixture whose legacy entry was just adopted, as the box having
        # diverged from itself, with `divergent_origins` telling it to coordinate with itself.
        fx = self.fixture("mlp_small.safetensors", b"this box's bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + f"deadbeef  17  {fx.name}\n")
        before = digests.read_text()

        with unittest.mock.patch.object(
            fixture_digest.platform, "node", return_value="rog-nv-wsl"
        ):
            with self.assertRaises(ValueError) as refused:
                fixture_digest._main(
                    ["--record", str(fx), "--adopt-legacy", "rog-nv", "--digests", str(digests)]
                )

        self.assertIn("--origin", str(refused.exception))
        self.assertEqual(digests.read_text(), before, "refused before the file is rewritten")

    def test_the_migration_still_defaults_where_the_two_names_agree(self):
        # First control: on the box the rows are being adopted as, the advertised command still
        # needs no --origin. The refusal above must not have made the ordinary migration harder.
        fx = self.fixture("mlp_small.safetensors", b"this box's bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + f"deadbeef  17  {fx.name}\n")

        with unittest.mock.patch.object(fixture_digest.platform, "node", return_value="rog-nv"):
            with contextlib.redirect_stdout(io.StringIO()):
                code = fixture_digest._main(
                    ["--record", str(fx), "--adopt-legacy", "rog-nv", "--digests", str(digests)]
                )

        self.assertEqual(code, 0)
        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries[fx.name]], ["rog-nv"])

    def test_one_box_may_still_migrate_another_boxs_legacy_rows(self):
        # Second control: two origins in one file ARE right when both were stated -- rog-nv
        # attributing rows it knows were minix's while recording its own bytes as its own. The
        # rule is about an origin nobody typed, not about a file naming two boxes.
        fx = self.fixture("lenet.safetensors", b"rog-nv bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  17  cifar_conv.safetensors\n")

        with contextlib.redirect_stdout(io.StringIO()):
            code = fixture_digest._main(
                ["--record", str(fx), "--origin", "rog-nv", "--adopt-legacy", "minix",
                 "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries["cifar_conv.safetensors"]], ["minix"])
        self.assertEqual([e.origin for e in entries[fx.name]], ["rog-nv"])

    def test_a_malformed_adopted_name_is_refused_even_with_nothing_to_adopt(self):
        # With --origin stated and no three-field row in the file, nothing downstream reads the
        # --adopt-legacy name, so the CLI is the only place that can hold it to the origin rule;
        # accepting it would report a migration claim as checked when nothing checked it.
        fx = self.fixture("lenet.safetensors", b"rog-nv bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER)
        before = digests.read_text()

        with self.assertRaises(ValueError) as refused:
            fixture_digest._main(
                ["--record", str(fx), "--origin", "rog-nv", "--adopt-legacy", "minix,rocm",
                 "--digests", str(digests)]
            )

        self.assertIn("minix,rocm", str(refused.exception))
        self.assertEqual(digests.read_text(), before, "refused before the file is rewritten")

    def test_the_legacy_refusal_names_a_command_that_runs_from_the_repo_root(self):
        # The canonical sweep command runs from the repository root, where a bare
        # `fixture_digest.py` names nothing -- the operator cannot paste the remediation.
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  17  lenet.safetensors\n")

        with self.assertRaises(ValueError) as refused:
            fixture_digest.read_digests(digests)

        self.assertIn("--adopt-legacy", str(refused.exception))
        self.assertIn(fixture_digest.cli_command(), str(refused.exception))

    def test_the_advertised_path_resolves_against_the_callers_cwd(self):
        # From the repository root it must spell benchmarks/fixture_digest.py, not
        # fixture_digest.py; from an unrelated cwd it falls back to an absolute path.
        root = HERE.parent
        with unittest.mock.patch.object(Path, "cwd", staticmethod(lambda: root)):
            self.assertEqual(fixture_digest.cli_command(), "benchmarks/fixture_digest.py")
        with unittest.mock.patch.object(Path, "cwd", staticmethod(lambda: Path(self.dir))):
            self.assertTrue(Path(fixture_digest.cli_command()).is_absolute())

    def test_the_checked_in_file_keeps_every_box_that_published_numbers(self):
        # gh-ocannl-759 recorded minix's and rog-nv's bytes for the two fixtures both boxes have
        # published numbers on. Neither may be evicted -- by a regeneration, or by a re-record
        # that forgets the other box -- or a standing report is retroactively on a workload
        # nothing pins. (Their digests are deliberately NOT pinned here: a coordinated
        # regeneration is allowed to change them, it is only allowed to change them for BOTH.)
        # gh-ocannl-483 added m4-max's bytes for gpt2_mini (its report is on them) and the
        # long-context legs it introduced, which only m4-max has recorded so far. gh-ocannl-1002's
        # training report is on m4-max's bytes of the training workload, its long-context legs and
        # the batch-1 sequence sweep.
        entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
        published = {
            "mlp_small.safetensors": {"minix", "rog-nv"},
            "gpt2_mini.safetensors": {"minix", "rog-nv", "m4-max"},
            "gpt2_mini_s512.safetensors": {"m4-max"},
            "gpt2_mini_s1024.safetensors": {"m4-max"},
            "gpt2_mini_train.safetensors": {"m4-max"},
            "gpt2_mini_train_s512.safetensors": {"m4-max"},
            "gpt2_mini_train_s1024.safetensors": {"m4-max"},
            "gpt2_mini_train_b1_s128.safetensors": {"m4-max"},
            "gpt2_mini_train_b1_s256.safetensors": {"m4-max"},
            "gpt2_mini_train_b1_s512.safetensors": {"m4-max"},
        }
        for name, boxes in published.items():
            self.assertTrue(boxes <= {e.origin for e in entries[name]}, name)

    def test_the_sweep_refuses_bytes_nothing_records(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.write_digests(digests, {})

        with self.assertRaises(SystemExit) as refused:
            self.check([fx], digests_path=digests)

        self.assertIn("--no-fixture-digest-check", str(refused.exception))

    def test_the_sweep_runs_and_stamps_a_recorded_fixture(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")

        ids = self.check([fx], digests_path=digests)

        self.assertEqual(ids, {fx: (fixture_digest.sha256_file(fx), "rog-nv")})

    def test_the_opt_out_measures_them_and_still_reports_the_digest(self):
        # A deliberate regeneration is a legitimate reason to run unpinned; it must not also cost
        # the run its record of what it ran on.
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")
        self.write_fixture(fx, b"regenerated")

        ids = self.check([fx], digests_path=digests, allow_unpinned=True)

        self.assertEqual(ids, {fx: (fixture_digest.sha256_file(fx), None)})

    def test_a_report_section_says_whose_bytes_its_numbers_are_on(self):
        # The published half of gh-ocannl-759: a reader comparing this report against the other
        # box's has to be able to see, without leaving the report, that the two are on different
        # bytes -- which for mlp_small and gpt2_mini they are.
        rows = [
            dict(
                cell("ocannl", "cuda", "default", [2.3026, 2.3010]),
                fixture="gpt2_mini.safetensors",
                fixture_sha256="043c1ea8",
                fixture_origin="rog-nv",
            )
        ]
        orchestrate.parity_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            text = (out / "report.md").read_text()

        self.assertIn("sha256 `043c1ea8`, rog-nv's bytes", text)

    def test_the_record_cli_pins_existing_bytes_without_regenerating(self):
        # The gh-ocannl-759 command: fixtures that predate the digest file get pinned as they are.
        fx = self.fixture("lenet.safetensors", b"bytes the published numbers are on")
        before = fx.read_bytes()
        digests = self.dir / fixture_digest.DIGEST_FILE

        with contextlib.redirect_stdout(io.StringIO()):
            code = fixture_digest._main(
                ["--record", str(fx), "--origin", "rog-nv", "--digests", str(digests)]
            )

        self.assertEqual(code, 0)
        self.assertEqual(fx.read_bytes(), before, "recording must not touch the fixture")
        self.assertEqual(fixture_digest.read_digests(digests)[fx.name][0].origin, "rog-nv")

    def test_the_record_cli_defaults_to_this_host(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        with contextlib.redirect_stdout(io.StringIO()):
            fixture_digest._main(["--record", str(fx), "--digests", str(digests)])

        self.assertEqual(
            fixture_digest.read_digests(digests)[fx.name][0].origin, fixture_digest.this_origin()
        )

    def test_the_check_cli_passes_a_recorded_fixture_and_fails_a_perturbed_one(self):
        fx = self.fixture("lenet.safetensors", b"recorded bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "rog-nv")

        with contextlib.redirect_stdout(io.StringIO()) as good:
            passed = fixture_digest._main(["--check", str(fx), "--digests", str(digests)])
        self.write_fixture(fx, b"perturbed bytes")
        with contextlib.redirect_stdout(io.StringIO()) as bad:
            failed = fixture_digest._main(["--check", str(fx), "--digests", str(digests)])

        self.assertEqual((passed, failed), (0, 1))
        self.assertIn("MATCH — rog-nv's bytes", good.getvalue())
        self.assertIn("MISMATCH", bad.getvalue())

    def expect_origin(self, digests, box, *fixtures):
        """`--check --expect-origin box` over `fixtures`: (exit status, report)."""
        argv = ["--check", *map(str, fixtures), "--digests", str(digests), "--expect-origin", box]
        with contextlib.redirect_stdout(io.StringIO()) as report:
            code = fixture_digest._main(argv)
        return code, report.getvalue()

    def test_expect_origin_decides_on_the_named_box_alone(self):
        # The measurement drivers pin their cells to ONE box's records. They used to count report
        # lines with a regex, which a second agreeing origin (`m4-max,tuf's bytes`) broke; the
        # exit status asks the question directly. Single- and multi-origin files both, and a
        # fixture that MATCHes only some OTHER box's bytes fails, as a MATCH alone would not.
        mine = self.fixture("gpt2_mini.safetensors", b"m4-max bytes")
        theirs = self.write_fixture(self.dir / "other" / "gpt2_mini.safetensors", b"minix bytes")
        single = self.dir / "single.txt"
        fixture_digest.record(single, [mine], "m4-max")
        multi = self.dir / "multi.txt"
        for box in ("tuf", "m4-max"):
            fixture_digest.record(multi, [mine], box)
        fixture_digest.record(multi, [theirs], "minix")

        self.assertEqual(self.expect_origin(single, "m4-max", mine)[0], 0)
        code, report = self.expect_origin(multi, "m4-max", mine)
        self.assertEqual(code, 0)
        self.assertIn("MATCH — m4-max,tuf's bytes", report, "control: the joined field is there")
        self.assertEqual(self.expect_origin(multi, "tuf", mine)[0], 0)
        code, report = self.expect_origin(multi, "m4-max", theirs)
        self.assertEqual(code, 1)
        self.assertIn("MATCH — minix's bytes", report, "control: it MATCHes, just not m4-max's")
        self.assertIn("not m4-max's bytes", report)
        self.assertEqual(self.expect_origin(multi, "m4-max", mine, theirs)[0], 1, "every fixture")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                fixture_digest._main(["--check", str(theirs), "--digests", str(multi)]),
                0,
                "control: without --expect-origin, some box's bytes pass",
            )
        self.write_fixture(mine, b"perturbed bytes")
        self.assertEqual(self.expect_origin(multi, "m4-max", mine)[0], 1, "a MISMATCH fails too")

    def test_expect_origin_refuses_a_box_the_file_does_not_declare(self):
        # No entry can name an undeclared box, so `--expect-origin m4max` would fail every
        # fixture as if its bytes were wrong; the misspelling is refused as one instead.
        fx = self.fixture("gpt2_mini.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "m4-max")

        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as refused:
                self.expect_origin(digests, "m4max", fx)

        self.assertEqual(refused.exception.code, 2)
        self.assertIn("m4-max", err.getvalue(), "it names the boxes the file does declare")
        for mode in (["--record", str(fx)], ["--list-declared-measurement-boxes"]):
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit, msg=mode[0]) as refused:
                    fixture_digest._main(
                        [*mode, "--digests", str(digests), "--expect-origin", "m4-max"]
                    )
            self.assertEqual(refused.exception.code, 2, f"--expect-origin with {mode[0]}")

    def check_cli(self, digests, *fixtures, expect_recorded=()):
        """`--check` over `fixtures`, each of `expect_recorded` as `--expect-recorded`:
        (exit status, report)."""
        argv = ["--check", *map(str, fixtures), "--digests", str(digests)]
        for box in expect_recorded:
            argv += ["--expect-recorded", box]
        with contextlib.redirect_stdout(io.StringIO()) as report:
            code = fixture_digest._main(argv)
        return code, report.getvalue()

    def test_check_names_the_declared_boxes_with_no_entry(self):
        # The MISSING DIGEST RECORD reading used to reach only orchestrate reports, so confirming
        # a newly recorded row (lukstafi/ocannl-staging#1136) took a hand-written
        # `python -c "missing_origins(...)"`. `--check` prints it per fixture, from the same
        # function the reports use; on its own it is information, not a failed verdict.
        fx = self.fixture("gpt2_mini.safetensors", b"m4-max bytes")
        other = self.write_fixture(self.dir / "other" / "lenet.safetensors", b"nobody's bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["m4-max", "minix", "rog-nv"]))
        fixture_digest.record(digests, [fx], "m4-max")
        fixture_digest.record(digests, [fx], "rog-nv")
        entries = fixture_digest.read_digests(digests)
        boxes = fixture_digest.measurement_boxes(digests)
        missing = fixture_digest.missing_origins(fx.name, entries, boxes)
        self.assertEqual(missing, ["minix"], "control: the reports' reading of this file")

        code, report = self.check_cli(digests, fx)
        self.assertEqual(code, 0, "a missing entry alone does not fail the bytes' verdict")
        self.assertIn(
            f"    no entry for declared measurement box(es): {', '.join(missing)}\n", report
        )
        code, report = self.check_cli(digests, other)
        self.assertEqual(code, 1, "control: UNRECORDED still fails")
        self.assertIn("box(es): m4-max, minix, rog-nv\n", report, "an unrecorded name lacks all")

        fixture_digest.record(digests, [fx], "minix")
        code, report = self.check_cli(digests, fx)
        self.assertEqual(code, 0)
        self.assertNotIn("no entry", report, "a complete record prints no missing line")

    def test_expect_recorded_decides_on_the_named_box_having_an_entry(self):
        # "Is the record complete for rog-nv?" as an exit status. It asks whether the box HAS an
        # entry, not whether that entry is these bytes (--expect-origin's question): a reviewer
        # on one box confirms another box's rows without that box's bytes.
        mine = self.fixture("gpt2_mini.safetensors", b"m4-max bytes")
        theirs = self.write_fixture(self.dir / "other" / "gpt2_mini.safetensors", b"minix bytes")
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.header(["m4-max", "minix", "rog-nv"]))
        fixture_digest.record(digests, [mine], "m4-max")
        fixture_digest.record(digests, [theirs], "minix")

        self.assertEqual(self.check_cli(digests, mine, expect_recorded=["m4-max"])[0], 0)
        self.assertEqual(
            self.check_cli(digests, mine, expect_recorded=["minix"])[0],
            0,
            "minix's entry is other bytes, and it is still an entry",
        )
        code, report = self.check_cli(digests, mine, expect_recorded=["rog-nv"])
        self.assertEqual(code, 1)
        self.assertIn("MATCH — m4-max's bytes", report, "control: the bytes themselves pass")
        self.assertIn("not recorded for rog-nv (--expect-recorded)", report)
        self.assertEqual(
            self.check_cli(digests, mine, expect_recorded=["m4-max", "rog-nv"])[0],
            1,
            "every named box must have an entry",
        )
        fixture_digest.record(digests, [mine], "rog-nv")
        self.assertEqual(self.check_cli(digests, mine, expect_recorded=["m4-max", "rog-nv"])[0], 0)
        self.write_fixture(mine, b"perturbed bytes")
        self.assertEqual(
            self.check_cli(digests, mine, expect_recorded=["rog-nv"])[0], 1, "a MISMATCH fails too"
        )

    def test_expect_recorded_refuses_a_box_the_file_does_not_declare(self):
        # The --expect-origin refusal's twin: an undeclared box can have no entry, so a
        # misspelling would read as an incomplete record instead of as the misspelling it is.
        fx = self.fixture("gpt2_mini.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx], "m4-max")

        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as refused:
                self.check_cli(digests, fx, expect_recorded=["m4max"])

        self.assertEqual(refused.exception.code, 2)
        self.assertIn("--expect-recorded 'm4max'", err.getvalue())
        self.assertIn("m4-max", err.getvalue(), "it names the boxes the file does declare")
        for mode in (["--record", str(fx)], ["--list-declared-measurement-boxes"]):
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit, msg=mode[0]) as refused:
                    fixture_digest._main(
                        [*mode, "--digests", str(digests), "--expect-recorded", "m4-max"]
                    )
            self.assertEqual(refused.exception.code, 2, f"--expect-recorded with {mode[0]}")

    def test_a_host_that_cannot_name_itself_is_refused_not_given_a_placeholder(self):
        # A literal `unknown-host` is not an origin, it is every nameless box sharing one name:
        # the second such box to record different bytes for a fixture would replace the first's
        # entry under it, which is the whichever-box-records-last provenance loss keying by
        # origin exists to prevent -- only now wearing a name that reads like an answer.
        with unittest.mock.patch.object(fixture_digest.platform, "node", return_value=""):
            self.assertIsNone(fixture_digest.this_origin())
            with self.assertRaises(ValueError) as refused:
                fixture_digest.resolve_origin(None)

        self.assertIn("--origin", str(refused.exception))

    def test_a_named_host_still_supplies_the_default(self):
        # The negative control for the refusal above: absence still defaults where there is a
        # name to default to, which is the whole convenience of the recording CLI.
        with unittest.mock.patch.object(fixture_digest.platform, "node", return_value="rog-nv"):
            self.assertEqual(fixture_digest.resolve_origin(None), "rog-nv")

    def test_the_digest_file_is_validated_before_any_fixture_is_overwritten(self):
        # Generating OVERWRITES the fixture bytes, so anything record() would refuse -- a legacy
        # three-field line, a duplicate origin, a malformed row -- has to be found while the
        # previous bytes still exist. Refusing afterwards loses the bytes the published numbers
        # were measured on AND leaves the new ones unrecorded, which is worse than either alone.
        gen_fixtures = self.gen_fixtures_module()
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        stale = fixtures / "lenet.safetensors"
        stale.write_bytes(b"the bytes the published numbers are on")
        before = stale.read_bytes()
        (fixtures / fixture_digest.DIGEST_FILE).write_text(
            fixture_digest.HEADER + "deadbeef  17  lenet.safetensors\n"
        )
        spec = self.dir / "workloads" / "lenet.json"
        spec.parent.mkdir()
        spec.write_text("{}")
        built = []

        with unittest.mock.patch.object(gen_fixtures, "build", lambda s, d: built.append(s)):
            with self.assertRaises(ValueError):
                gen_fixtures.main(["--origin", "rog-nv"], here=self.dir)

        # And the spec really was one this run would have built, so [] is a refusal and not an
        # empty work list: the same call with a readable digest file builds it (below).
        self.assertEqual(built, [])
        self.assertEqual(stale.read_bytes(), before)

    def test_a_spec_name_the_format_cannot_carry_is_refused_before_building(self):
        # The name-side twin of the digest-file validation above: build() writes
        # fixtures/<spec name>.safetensors, and record() would refuse a whitespace name only
        # AFTER the previous bytes are overwritten. The name check runs first, while the bytes
        # the published numbers rest on still exist.
        gen_fixtures = self.gen_fixtures_module()
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        stale = fixtures / "a b.safetensors"
        stale.write_bytes(b"the bytes the published numbers are on")
        (fixtures / fixture_digest.DIGEST_FILE).write_text(fixture_digest.HEADER)
        spec = self.dir / "workloads" / "spaced.json"
        spec.parent.mkdir()
        spec.write_text('{"name": "a b"}')
        built = []

        with unittest.mock.patch.object(gen_fixtures, "build", lambda s, d: built.append(s)):
            with self.assertRaises(ValueError):
                gen_fixtures.main(["--origin", "rog-nv"], here=self.dir)

        self.assertEqual(built, [], "refused before any build")
        self.assertEqual(stale.read_bytes(), b"the bytes the published numbers are on")

    def test_a_valid_digest_file_lets_the_generator_build(self):
        # The negative control: the guard above must refuse the unreadable file, not every file.
        gen_fixtures = self.gen_fixtures_module()
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        digests = fixtures / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER)

        spec = self.dir / "workloads" / "lenet.json"
        spec.parent.mkdir()
        spec.write_text("{}")
        built = []

        def build(spec_path, out_dir):
            built.append(spec_path)
            path = out_dir / "lenet.safetensors"
            self.write_fixture(path, b"regenerated")
            return path

        with unittest.mock.patch.object(gen_fixtures, "build", build):
            with contextlib.redirect_stdout(io.StringIO()):
                gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        self.assertEqual(built, [spec])

        entries = fixture_digest.read_digests(digests)
        self.assertEqual([e.origin for e in entries["lenet.safetensors"]], [self.this_box])

    def smoke_layout(self):
        """A benchmarks dir whose recorded fixture and digest file a smoke run must not touch."""
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        recorded = fixtures / "lenet.safetensors"
        self.write_fixture(recorded, b"the bytes the published numbers are on")
        digests = fixtures / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [recorded], "rog-nv")
        spec = self.dir / "workloads" / "lenet.json"
        spec.parent.mkdir()
        spec.write_text("{}")
        return spec, recorded.read_bytes(), digests.read_bytes()

    def test_out_dir_writes_elsewhere_and_records_nothing(self):
        # A smoke run (gh-ocannl-1020's warm-pass A/B) needs a fixture but publishes no number,
        # so its bytes must not become an origin's entry in the tracked DIGESTS.txt -- and must
        # not replace the recorded fixture either.
        gen_fixtures = self.gen_fixtures_module()
        spec, recorded_before, digests_before = self.smoke_layout()
        out = self.dir / "smoke"
        built = []

        def build(spec_path, out_dir):
            built.append((spec_path, out_dir))
            return self.write_fixture(out_dir / "lenet.safetensors", b"smoke bytes")

        # A box with no name: the recording path refuses to invent an origin, and this one must
        # not need one.
        with unittest.mock.patch.object(gen_fixtures, "build", build), \
                unittest.mock.patch.object(fixture_digest.platform, "node", return_value=""):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                written = gen_fixtures.main(["--out-dir", str(out)], here=self.dir)

        # Built into a staging directory inside DIR, which is gone afterwards.
        self.assertEqual([(s, d.parent) for s, d in built], [(spec, out)])
        self.assertEqual(written, [out / "lenet.safetensors"])
        self.assertEqual((self.dir / "fixtures" / "lenet.safetensors").read_bytes(),
                         recorded_before)
        self.assertEqual((self.dir / "fixtures" / fixture_digest.DIGEST_FILE).read_bytes(),
                         digests_before)
        self.assertEqual(list(out.iterdir()), [out / "lenet.safetensors"])
        # The path a caller hands to a runner is the one printed -- not the staging copy, which
        # is gone by the time anyone reads the line.
        self.assertIn(f"wrote {out / 'lenet.safetensors'} (", printed.getvalue())
        self.assertNotIn(".gen_fixtures-", printed.getvalue())

    def test_out_dir_refuses_the_fixtures_dir_and_an_origin_before_building(self):
        # Writing into fixtures/ without recording is the loss the pre-build validation guards
        # against: the published bytes overwritten, the new ones matching no entry.
        gen_fixtures = self.gen_fixtures_module()
        _, recorded_before, digests_before = self.smoke_layout()
        # Spelled differently from `here / "fixtures"`, so the refusal compares the directories'
        # identity: a `..` detour, one through a directory that does not exist yet (so it becomes
        # an alias only once DIR is created), a symlink, and -- where the filesystem folds case, as
        # a default macOS volume does -- a case alias, which path resolution keeps as spelled.
        aliases = [self.dir / "workloads" / ".." / "fixtures",
                   self.dir / "fixtures" / "new" / ".."]
        try:
            (self.dir / "fixtures-link").symlink_to(self.dir / "fixtures",
                                                    target_is_directory=True)
            aliases.append(self.dir / "fixtures-link")
        except OSError:
            pass  # Windows without Developer Mode or SeCreateSymbolicLinkPrivilege
        if (self.dir / "FIXTURES").exists():
            aliases.append(self.dir / "FIXTURES")
        built = []

        for argv in ([["--out-dir", str(alias)] for alias in aliases]
                     + [["--out-dir", str(self.dir / "smoke"), "--origin", "rog-nv"]]):
            with self.subTest(argv=argv):
                with unittest.mock.patch.object(gen_fixtures, "build",
                                                lambda s, d: built.append(s)), \
                        contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    gen_fixtures.main(argv, here=self.dir)

        self.assertEqual(built, [], "refused before any build")
        self.assertFalse((self.dir / "smoke").exists())
        self.assertEqual((self.dir / "fixtures" / "lenet.safetensors").read_bytes(),
                         recorded_before)
        self.assertEqual((self.dir / "fixtures" / fixture_digest.DIGEST_FILE).read_bytes(),
                         digests_before)

    def test_a_spec_name_that_leaves_the_output_dir_is_refused_before_building(self):
        # build() writes <out_dir>/<spec name>.safetensors, so checking DIR alone is not enough:
        # `--out-dir smoke` with name "../fixtures/lenet" lands on the recorded fixture without
        # touching its digest. The recording path has the same hole, one directory up.
        gen_fixtures = self.gen_fixtures_module()
        _, recorded_before, digests_before = self.smoke_layout()
        escaping = self.dir / "workloads" / "escaping.json"
        escaping.write_text('{"name": "../fixtures/lenet"}')
        built = []

        for argv in (["--out-dir", str(self.dir / "smoke"), str(escaping)],
                     ["--origin", "rog-nv", str(escaping)]):
            with self.subTest(argv=argv):
                with unittest.mock.patch.object(gen_fixtures, "build",
                                                lambda s, d: built.append(s)), \
                        self.assertRaises(ValueError):
                    gen_fixtures.main(argv, here=self.dir)

        self.assertEqual(built, [], "refused before any build")
        self.assertEqual((self.dir / "fixtures" / "lenet.safetensors").read_bytes(),
                         recorded_before)
        self.assertEqual((self.dir / "fixtures" / fixture_digest.DIGEST_FILE).read_bytes(),
                         digests_before)

    def test_out_dir_never_writes_through_a_link_in_dir(self):
        # save_file writes THROUGH an existing entry, so a symlink or hard link in DIR to the
        # recorded fixture would have the smoke bytes overwrite it with its digest untouched.
        # The stub writes the way save_file does (opening the path, following links).
        gen_fixtures = self.gen_fixtures_module()
        spec, recorded_before, digests_before = self.smoke_layout()
        recorded = self.dir / "fixtures" / "lenet.safetensors"

        def build(spec_path, out_dir):
            return self.write_fixture(out_dir / "lenet.safetensors", b"smoke bytes")

        for kind, link in (("symlink", os.symlink), ("hard link", os.link)):
            with self.subTest(kind=kind):
                out = self.dir / f"smoke-{kind.replace(' ', '-')}"
                out.mkdir()
                try:
                    link(recorded, out / "lenet.safetensors")
                except OSError as e:
                    # Windows grants symlink creation only with Developer Mode or
                    # SeCreateSymbolicLinkPrivilege; the hard-link leg still runs there.
                    self.skipTest(f"cannot create a {kind} here: {e}")
                with unittest.mock.patch.object(gen_fixtures, "build", build), \
                        contextlib.redirect_stdout(io.StringIO()):
                    gen_fixtures.main(["--out-dir", str(out), str(spec)], here=self.dir)

                self.assertEqual(recorded.read_bytes(), recorded_before)
                smoke = out / "lenet.safetensors"
                self.assertFalse(smoke.is_symlink())
                self.assertEqual(smoke.stat().st_nlink, 1)
                self.assertNotEqual(smoke.read_bytes(), recorded_before)
        self.assertEqual((self.dir / "fixtures" / fixture_digest.DIGEST_FILE).read_bytes(),
                         digests_before)

    def test_out_dir_keeps_previous_outputs_when_a_build_fails(self):
        # Each destination is replaced only once its own build succeeded: a spec that fails
        # (a missing seed, an unknown model) costs its own run, not the smoke fixtures already in
        # DIR -- neither its own previous output nor those of specs never reached.
        gen_fixtures = self.gen_fixtures_module()
        spec, _, _ = self.smoke_layout()
        other = spec.parent / "mlp.json"
        other.write_text("{}")
        out = self.dir / "smoke"
        for name in ("lenet", "mlp"):
            self.write_fixture(out / f"{name}.safetensors", f"previous {name}".encode())
        before = {p.name: p.read_bytes() for p in out.iterdir()}

        def build(spec_path, out_dir):
            raise ValueError("unknown model")

        with unittest.mock.patch.object(gen_fixtures, "build", build), \
                self.assertRaises(ValueError):
            gen_fixtures.main(["--out-dir", str(out), str(spec), str(other)], here=self.dir)

        self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, before,
                         "previous outputs intact and no staging directory left behind")

    def test_two_specs_with_one_destination_are_refused_before_building(self):
        # On both paths: the later spec would silently replace the earlier's fixture, and on the
        # case-insensitive measuring hosts a case-only difference is the same file.
        gen_fixtures = self.gen_fixtures_module()
        _, recorded_before, digests_before = self.smoke_layout()
        workloads = self.dir / "workloads"
        (workloads / "a.json").write_text('{"name": "mlp"}')
        (workloads / "b.json").write_text('{"name": "mlp"}')
        (workloads / "c.json").write_text('{"name": "MLP"}')
        built = []

        for pair in (("a", "b"), ("a", "c")):
            specs = [str(workloads / f"{x}.json") for x in pair]
            for argv in (["--out-dir", str(self.dir / "smoke"), *specs],
                         ["--origin", "rog-nv", *specs]):
                with self.subTest(argv=argv):
                    with unittest.mock.patch.object(gen_fixtures, "build",
                                                    lambda s, d: built.append(s)), \
                            self.assertRaises(ValueError):
                        gen_fixtures.main(argv, here=self.dir)

        self.assertEqual(built, [], "refused before any build")
        self.assertEqual((self.dir / "fixtures" / "lenet.safetensors").read_bytes(),
                         recorded_before)
        self.assertEqual((self.dir / "fixtures" / fixture_digest.DIGEST_FILE).read_bytes(),
                         digests_before)

    def recorded_layout(self, *names):
        """A benchmarks dir with one recorded fixture and one spec per name, in `names` order."""
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        digests = fixtures / fixture_digest.DIGEST_FILE
        recorded = [self.write_fixture(fixtures / f"{name}.safetensors",
                                       f"published {name}".encode()) for name in names]
        fixture_digest.record(digests, recorded, self.this_box)
        workloads = self.dir / "workloads"
        workloads.mkdir()
        specs = []
        for name in names:
            spec = workloads / f"{name}.json"
            spec.write_text(json.dumps({"name": name}))
            specs.append(spec)
        return fixtures, digests, specs

    def regenerating_build(self, fail_on=None):
        """A `build` stub writing the way save_file does, raising on the spec named `fail_on`."""

        def build(spec_path, out_dir):
            name = json.loads(spec_path.read_text())["name"]
            if name == fail_on:
                raise AssertionError("conv1: (image_size - kernel_size) mod stride1 != 0")
            return self.write_fixture(out_dir / f"{name}.safetensors", f"new {name}".encode())

        return build

    def snapshot(self, fixtures):
        """Every entry under `fixtures`, staging leftovers included, with the bytes it reads."""
        return {str(p.relative_to(fixtures)): (p.read_bytes() if p.is_file() else None)
                for p in fixtures.rglob("*")}

    def test_a_later_spec_failing_to_build_replaces_and_records_nothing(self):
        # gh-ocannl-1059: the build-time twin of the pre-build refusals. A spec that raises inside
        # build() (here the second of three) must not cost the first its published bytes: every
        # spec builds into a staging directory, and nothing is replaced until all of them built.
        gen_fixtures = self.gen_fixtures_module()
        fixtures, digests, specs = self.recorded_layout("conv", "gpt", "mlp")
        before = self.snapshot(fixtures)
        built = []
        build = self.regenerating_build(fail_on="gpt")

        def counting_build(spec_path, out_dir):
            built.append(spec_path)
            return build(spec_path, out_dir)

        with unittest.mock.patch.object(gen_fixtures, "build", counting_build), \
                contextlib.redirect_stdout(io.StringIO()) as printed, \
                contextlib.redirect_stderr(io.StringIO()) as errors, \
                self.assertRaises(AssertionError):
            gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        # The first spec really was regenerated (into staging, specs build in sorted order), so
        # the unchanged tree is the staging's doing, not a run that never got that far.
        self.assertEqual(built, specs[:2])
        self.assertEqual(self.snapshot(fixtures), before,
                         "fixtures and DIGESTS.txt byte-identical, no staging directory left")
        self.assertEqual(printed.getvalue(), "")
        self.assertIn(f"FAILED to build {specs[1]}", errors.getvalue())
        self.assertIn("no digest was recorded", errors.getvalue())

    def test_a_failing_build_exits_nonzero(self):
        # What the operator sees: the uncaught error is the process's exit status, not a warning
        # scrolled past. Run as a program, with the stubs and the failing build installed first.
        fixtures, _, _ = self.recorded_layout("lenet", "cifar")
        before = self.snapshot(fixtures)
        driver = self.dir / "driver.py"
        driver.write_text(
            "import json, sys, types\n"
            "numpy = types.ModuleType('numpy'); st = types.ModuleType('safetensors')\n"
            "stn = types.ModuleType('safetensors.numpy'); stn.save_file = None\n"
            "sys.modules.update({'numpy': numpy, 'safetensors': st, 'safetensors.numpy': stn})\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import gen_fixtures\n"
            "def build(spec_path, out_dir):\n"
            "    name = json.loads(spec_path.read_text())['name']\n"
            "    if name == 'lenet': raise ValueError('unknown model')\n"
            "    path = out_dir / f'{name}.safetensors'; path.write_bytes(b'new'); return path\n"
            "gen_fixtures.build = build\n"
            f"gen_fixtures.main(['--origin', 'test-box'], here={str(self.dir)!r})\n")
        run = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True,
                             timeout=60)

        self.assertNotEqual(run.returncode, 0, run.stdout)
        self.assertIn("FAILED to build", run.stderr)
        self.assertIn("lenet.json", run.stderr)
        self.assertEqual(self.snapshot(fixtures), before)

    def test_all_specs_building_replaces_and_records_every_one(self):
        # The negative control for the refusal above: staging defers the replacement, it does not
        # lose it. Every destination holds the new bytes and every digest names them.
        gen_fixtures = self.gen_fixtures_module()
        fixtures, digests, _ = self.recorded_layout("lenet", "cifar")

        with unittest.mock.patch.object(gen_fixtures, "build", self.regenerating_build()), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            written = gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        self.assertEqual(written, [fixtures / "cifar.safetensors", fixtures / "lenet.safetensors"])
        entries = fixture_digest.read_digests(digests)
        for fixture in written:
            self.assertEqual(fixture.read_bytes(), self.write_fixture(
                self.dir / "expected" / fixture.name, f"new {fixture.stem}".encode()).read_bytes())
            self.assertEqual(fixture_digest.status(fixture, entries)[0], "MATCH")
            self.assertIn(f"wrote {fixture} (", printed.getvalue())
        self.assertEqual(sorted(p.name for p in fixtures.iterdir()),
                         [fixture_digest.DIGEST_FILE, "cifar.safetensors", "lenet.safetensors"])
        self.assertNotIn(".gen_fixtures-", printed.getvalue())

    def test_a_first_generation_creates_the_fixtures_directory(self):
        # The resolved-target checks run before anything exists: with no fixtures/ yet (and so
        # no DIGESTS.txt, which records nothing), a plain destination is still accepted.
        gen_fixtures = self.gen_fixtures_module()
        spec = self.dir / "workloads" / "lenet.json"
        spec.parent.mkdir()
        spec.write_text('{"name": "lenet"}')

        with unittest.mock.patch.object(gen_fixtures, "build", self.regenerating_build()), \
                contextlib.redirect_stdout(io.StringIO()):
            written = gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        entries = fixture_digest.read_digests(self.dir / "fixtures" / fixture_digest.DIGEST_FILE)
        self.assertEqual(written, [self.dir / "fixtures" / "lenet.safetensors"])
        self.assertEqual(fixture_digest.status(written[0], entries)[0], "MATCH")

    def symlinked_fixture(self, fixtures, name):
        """Replace fixtures/<name>.safetensors by a symlink to a file in a sibling tree, the way
        gh612_cells.sh shares one fixture between measurement trees; returns the shared file."""
        shared = self.dir / "other-tree" / "fixtures" / f"{name}.safetensors"
        shared.parent.mkdir(parents=True, exist_ok=True)
        entry = fixtures / f"{name}.safetensors"
        os.replace(entry, shared)
        try:
            entry.symlink_to(shared)
        except OSError as e:
            self.skipTest(f"cannot create a symlink here: {e}")
        return shared

    def test_a_symlinked_recorded_fixture_stays_shared(self):
        # The recording path replaces the link's TARGET: replacing the entry would silently
        # un-share every tree linked to it, leaving them on the previous bytes. Also across a
        # filesystem boundary, where os.replace cannot rename a staged file onto the target and
        # a copy beside the target is renamed instead -- simulated by reporting the shared tree
        # on another device.
        for cross_device in (False, True):
            with self.subTest(cross_device=cross_device):
                self.tmp.cleanup()
                self.dir.mkdir()
                gen_fixtures = self.gen_fixtures_module()
                fixtures, digests, _ = self.recorded_layout("gpt2_mini")
                shared = self.symlinked_fixture(fixtures, "gpt2_mini")
                real_stat, other_device = os.stat, os.path.realpath(shared.parent)

                def stat(path, *args, **kwargs):
                    result = real_stat(path, *args, **kwargs)
                    if cross_device and os.path.realpath(path) == other_device:
                        fields = list(result)
                        fields[stat_module.ST_DEV] += 1
                        return os.stat_result(fields)
                    return result

                with unittest.mock.patch.object(gen_fixtures, "build",
                                                self.regenerating_build()), \
                        unittest.mock.patch.object(gen_fixtures.os, "stat", stat), \
                        contextlib.redirect_stdout(io.StringIO()) as printed:
                    gen_fixtures.main(["--origin", self.this_box], here=self.dir)

                entry = fixtures / "gpt2_mini.safetensors"
                self.assertTrue(entry.is_symlink())
                # Windows' readlink returns the link's substitute name, `\\?\C:\...`, where
                # the link was created from `C:\...`: compare the files the two spellings name.
                self.assertTrue(os.path.samefile(os.readlink(entry), shared),
                                f"{entry} links to {os.readlink(entry)}, not to {shared}")
                self.assertIn(b"new gpt2_mini", shared.read_bytes())
                entries = fixture_digest.read_digests(digests)
                self.assertEqual(fixture_digest.status(entry, entries)[0], "MATCH")
                self.assertIn(f"through the symlink to {os.path.realpath(shared)}",
                              printed.getvalue())
                self.assertEqual(sorted(p.name for p in shared.parent.iterdir()),
                                 ["gpt2_mini.safetensors"], "no copy left beside the target")
                # The mode a generated fixture gets, not the owner-only one of the copy's
                # temporary file: other users and services read the shared fixture too.
                fresh = self.write_fixture(self.dir / "mode-probe.safetensors")
                self.assertEqual(stat_module.S_IMODE(shared.stat().st_mode),
                                 stat_module.S_IMODE(fresh.stat().st_mode))

    def test_a_hard_linked_recorded_fixture_is_announced_as_unshared(self):
        # A rename replaces one NAME, so the other hard links keep the previous bytes. That is
        # the price of never writing through a published file; it is printed, not silent.
        gen_fixtures = self.gen_fixtures_module()
        fixtures, _, _ = self.recorded_layout("lenet")
        other = self.dir / "other-tree" / "lenet.safetensors"
        other.parent.mkdir()
        try:
            os.link(fixtures / "lenet.safetensors", other)
        except OSError as e:
            self.skipTest(f"cannot create a hard link here: {e}")
        before = other.read_bytes()

        with unittest.mock.patch.object(gen_fixtures, "build", self.regenerating_build()), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        self.assertEqual(other.read_bytes(), before)
        self.assertIn("had 1 other hard link(s): they keep the previous bytes", printed.getvalue())

    def test_a_failed_replacement_records_the_fixtures_already_replaced(self):
        # Past the staging, only the renames are left, and a rename can still fail (a locked file
        # on Windows, a read-only target directory). The fixtures replaced before it are new
        # bytes, so they are recorded before the error propagates, never left matching no entry.
        gen_fixtures = self.gen_fixtures_module()
        fixtures, digests, _ = self.recorded_layout("cifar", "lenet")
        real_replace = os.replace
        calls = []

        def replace(src, dst):
            calls.append(dst)
            if len(calls) == 2:
                raise PermissionError(13, "locked", str(dst))
            return real_replace(src, dst)

        with unittest.mock.patch.object(gen_fixtures, "build", self.regenerating_build()), \
                unittest.mock.patch.object(gen_fixtures.os, "replace", replace), \
                contextlib.redirect_stdout(io.StringIO()) as printed, \
                contextlib.redirect_stderr(io.StringIO()) as errors, \
                self.assertRaises(PermissionError):
            gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        entries = fixture_digest.read_digests(digests)
        self.assertIn(b"new cifar", (fixtures / "cifar.safetensors").read_bytes())
        self.assertIn(b"published lenet", (fixtures / "lenet.safetensors").read_bytes())
        for name in ("cifar", "lenet"):
            self.assertEqual(
                fixture_digest.status(fixtures / f"{name}.safetensors", entries)[0], "MATCH", name)
        self.assertIn("FAILED to replace", errors.getvalue())
        self.assertEqual(sorted(p.name for p in fixtures.iterdir()),
                         [fixture_digest.DIGEST_FILE, "cifar.safetensors", "lenet.safetensors"])
        # And announced through the same report as a complete run: a retry regenerates the same
        # bytes, whose digest then already matches, so this is the only notice the published
        # numbers on the previous cifar bytes ever get.
        self.assertIn(f"CHANGED for {self.this_box}: cifar.safetensors", printed.getvalue())
        self.assertNotIn("lenet", printed.getvalue())

    def test_an_output_failure_after_the_renames_still_records(self):
        # The report is printed only after record() ran: a closed stdout pipe
        # (`gen_fixtures.py | head -1` with PYTHONUNBUFFERED) raising on the first `wrote` line
        # must not leave the replaced fixtures matching no entry.
        gen_fixtures = self.gen_fixtures_module()
        fixtures, digests, _ = self.recorded_layout("cifar", "lenet")

        def report_written(path):
            raise BrokenPipeError(32, "Broken pipe")

        with unittest.mock.patch.object(gen_fixtures, "build", self.regenerating_build()), \
                unittest.mock.patch.object(gen_fixtures, "report_written", report_written), \
                self.assertRaises(BrokenPipeError):
            gen_fixtures.main(["--origin", self.this_box], here=self.dir)

        entries = fixture_digest.read_digests(digests)
        for name in ("cifar", "lenet"):
            fixture = fixtures / f"{name}.safetensors"
            self.assertIn(f"new {name}".encode(), fixture.read_bytes())
            self.assertEqual(fixture_digest.status(fixture, entries)[0], "MATCH", name)

    def test_an_unreplaceable_resolved_target_is_refused_before_building(self):
        # The resolved targets are checked with the names, before any build: two fixtures linked
        # to one file (the later spec's bytes would be recorded under both names), a link into a
        # directory that does not exist, and a link to a directory.
        gen_fixtures = self.gen_fixtures_module()
        fixtures, digests, specs = self.recorded_layout("cifar", "lenet")
        shared = self.symlinked_fixture(fixtures, "cifar")
        lenet = fixtures / "lenet.safetensors"
        lenet.unlink()
        built = []
        for label, target in (("one shared file", shared),
                              ("a missing directory", self.dir / "gone" / "lenet.safetensors"),
                              ("a directory", self.dir / "workloads")):
            with self.subTest(target=label):
                if lenet.is_symlink():
                    lenet.unlink()
                lenet.symlink_to(target)
                before = self.snapshot(fixtures)
                with unittest.mock.patch.object(gen_fixtures, "build",
                                                lambda s, d: built.append(s)), \
                        self.assertRaises(ValueError):
                    gen_fixtures.main(["--origin", self.this_box, *map(str, specs)],
                                      here=self.dir)
                self.assertEqual(self.snapshot(fixtures), before)
        self.assertEqual(built, [], "refused before any build")

    #: Every spelling that reaches outside a directory, or names no file in it, on either
    #: platform.
    NON_PORTABLE_NAMES = ("", ".", "..", "../x", "a/b", "/abs", "a\\b", "..\\x", "C:x", "a b",
                          "-x", "a\x00b", "a,b", "CON", "nul", "Com1", "LPT9.x", "aux.tar", None)

    def workload_names(self):
        names = [json.loads(spec.read_text())["name"]
                 for spec in (HERE / "workloads").glob("*.json")]
        self.assertTrue(names, "no workload specs found next to the test")
        return names

    def test_fixture_path_takes_portable_names_only(self):
        # The check build() itself makes before writing, so a direct build() caller is covered
        # too: every non-portable spelling -- and the negative control, the names the workloads
        # really use, and a device name that is only a prefix.
        gen_fixtures = self.gen_fixtures_module()
        out = self.dir / "smoke"
        for name in self.workload_names():
            self.assertEqual(gen_fixtures.fixture_path(out, name), out / f"{name}.safetensors")
        self.assertEqual(gen_fixtures.fixture_path(out, "CONSOLE"), out / "CONSOLE.safetensors")
        for name in self.NON_PORTABLE_NAMES:
            with self.subTest(name=name), self.assertRaises(ValueError):
                gen_fixtures.fixture_path(out, name)

    def test_generating_recording_and_origins_share_one_name_rule(self):
        # A name gen_fixtures refuses to generate must not be recordable from bytes that already
        # exist (`--record` of a hand-copied CON.safetensors), and an origin is held to the same
        # rule because it keys file names too. So the three checks agree with the one predicate
        # on every probe, accepted or refused -- derived from the predicate, not restated per check.
        gen_fixtures = self.gen_fixtures_module()
        out = self.dir / "smoke"

        def accepts(check, arg):
            try:
                check(arg)
            except ValueError:
                return False
            return True

        probes = [*self.workload_names(), "CONSOLE", *self.NON_PORTABLE_NAMES]
        for name in probes:
            portable = fixture_digest.is_portable_name(name)
            with self.subTest(name=name):
                self.assertEqual(accepts(lambda n: gen_fixtures.fixture_path(out, n), name),
                                 portable)
                self.assertEqual(accepts(fixture_digest.check_origin, name), portable)
                if isinstance(name, str):
                    self.assertEqual(
                        accepts(fixture_digest.check_fixture_name, f"{name}.safetensors"),
                        portable,
                    )
        self.assertTrue(any(map(fixture_digest.is_portable_name, probes)), "no accepted probe")
        self.assertFalse(all(map(fixture_digest.is_portable_name, probes)), "no refused probe")

    def test_generated_fixtures_are_named_after_their_spec(self):
        # What the recorded-name check above relies on: gen_fixtures.py writes
        # fixtures/<spec name>.safetensors, and every spec's `name` is its file stem.
        for spec in (HERE / "workloads").glob("*.json"):
            self.assertEqual(json.loads(spec.read_text())["name"], spec.stem, spec.name)


class Gh1161SitesTest(unittest.TestCase):
    """gh-ocannl-1161: the cifar_resnet site cells, read off a sweep's cell logs.

    The logs are built here in the producers' formats: `Bench_harness`'s kernel table on stderr
    (`bench: kernel i/n <ms> ms ... w: <nodes>`, a DECLINED row's detail running over lines) and the
    runner's result line, under the file names `orchestrate.cell_log_name` gives each label --
    a tuned cell's search pass included, which logs beside its replay under the same cell.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def log(self, label, kernels, variant="default", p50=10.0, workload="cifar_resnet_b1"):
        result = {"framework": "ocannl", "backend": "cuda", "variant": variant, "profile": None,
                  "workload": workload, "step_ms": {"p50": p50, "p10": p50, "p90": p50}}
        lines = []
        for row in kernels:
            if row[1] is None:
                lines += [f"bench: kernel {row[0]} grid=[1;1;1] block=[32;1;1] DECLINED (codegen",
                          "failed: duplicate parameter) w: n1"]
            else:
                lines.append(f"bench: kernel {row[0]} {row[1]:.4f} ms grid=[1;1;1] block=[1;1;1] "
                             f"mma:not-requested vol:0 w: {' '.join(row[2])}")
        lines.append(json.dumps(result, separators=(",", ":")))
        (self.dir / orchestrate.cell_log_name(label)).write_text("\n".join(lines) + "\n")

    def run_sites(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = gh1161_sites.main([str(self.dir), *args])
        return code, out.getvalue(), err.getvalue()

    def rows(self, out):
        cells = [[c.strip() for c in line.strip().strip("|").split("|")]
                 for line in out.splitlines()[2:]]
        return {row[0]: row for row in cells}

    def test_a_tuned_cell_reports_its_replay_not_its_search_pass(self):
        label = "cifar_resnet_b1 ocannl/cuda/tuned"
        replay = [("0/2", 1.0, ["kernel_s2sc.grad"]), ("1/2", 2.0, ["relu_s1out"])]
        search = [("0/2", 9.0, ["kernel_s2sc.grad"]), ("1/2", 9.0, ["relu_s1out"])]
        self.log(label, replay, variant="tuned", p50=3.0)
        self.log(f"{label} (search pass)", search, variant="tuned", p50=30.0)

        code, out, _ = self.run_sites("--per-site")

        self.assertEqual(code, 0)
        rows = self.rows(out)
        self.assertEqual(list(rows), ["cifar_resnet_b1/cuda/tuned/exact"])
        row = rows["cifar_resnet_b1/cuda/tuned/exact"]
        self.assertEqual(row[1:4], ["3.000", "2/2", "3.000"])

    def test_a_fused_kernel_counts_once_per_family_and_unnamed_nodes_are_unattributed(self):
        self.log("cifar_resnet_b1 ocannl/cuda/default", [
            ("0/3", 1.0, ["kernel_s2c1.grad", "relu_s2c1out.grad", "relu_s1out.grad"]),
            ("1/3", 2.0, ["kernel_s3sc.grad", "bias_s3sc.grad"]),
            ("2/3", 4.0, ["n186", "batch_x"]),
        ])

        code, out, _ = self.run_sites()

        self.assertEqual(code, 0)
        # cell, p50, timed, sum, pad3x3, resadd, sc1x1s2, unattributed
        self.assertEqual(self.rows(out)["cifar_resnet_b1/cuda/default/exact"][1:8],
                         ["10.000", "3/3", "7.000", "1.000", "1.000", "2.000", "4.000"])

    def test_declined_kernels_are_untimed_but_keep_the_table_complete(self):
        self.log("cifar_resnet_b1 ocannl/cuda/default", [
            ("0/2", None, None), ("1/2", 2.0, ["kernel_s2sc.grad"])])

        _, out, _ = self.run_sites()

        self.assertEqual(self.rows(out)["cifar_resnet_b1/cuda/default/exact"][2:4], ["1/2", "2.000"])

    def test_a_truncated_table_attributes_nothing(self):
        # The instrument catches its own failure after the result line is paid for, so a table can
        # stop short of its declared count (a backend whose compile failure is fatal).
        self.log("cifar_resnet_b1 ocannl/cuda/default", [("0/3", 1.0, ["kernel_s2sc.grad"])])

        _, out, _ = self.run_sites()

        row = self.rows(out)["cifar_resnet_b1/cuda/default/exact"]
        self.assertEqual(row[2], "1/3 INCOMPLETE (1 listed)")
        self.assertEqual(set(row[3:8]), {"n/a"})

    def test_two_logs_for_one_cell_are_refused(self):
        kernels = [("0/1", 1.0, ["kernel_s2sc.grad"])]
        self.log("cifar_resnet_b1 ocannl/cuda/default", kernels)
        self.log("cifar_resnet_b1 ocannl/cuda/default again", kernels)

        code, _, err = self.run_sites()

        self.assertEqual(code, 1)
        self.assertIn("two logs", err)


class CellTimeoutTest(unittest.TestCase):
    """gh-ocannl-760: a cell that stops making progress costs the cell, not the sweep."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_cell(self, *args, **kwargs):
        """run_cell with the cell's own output kept out of the test's."""
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result, note = orchestrate.run_cell(*args, **kwargs)
        return result, note, out.getvalue()

    def python(self, source, *argv):
        return [sys.executable, "-c", source, *map(str, argv)]

    def assert_tinygrad_cache_in_scratch(self, cachedb):
        scratch = self.dir.resolve()
        resolved = cachedb.resolve()
        try:
            resolved.relative_to(scratch)
        except ValueError:
            self.fail(
                f"tinygrad cache probe resolved outside the test scratch directory: "
                f"{resolved} (scratch: {scratch})"
            )

    def alive(self, pid):
        """Whether `pid` is still a running process — a zombie is not one.

        The distinction is load-bearing, not pedantry: the grandchild these tests kill is an
        ORPHAN by then (its parent, the cell, was killed too), so whether it disappears or lingers
        as an unreaped zombie is decided by whoever inherits it. Under a normal init it is reaped
        at once; in a container whose PID 1 does not reap, it stays a zombie indefinitely — and
        `kill(pid, 0)` succeeds for a zombie, so a bare signal-0 probe would report the process
        the kill did remove as alive and fail the test in exactly the environments CI runs in.

        The liveness question itself goes through `cell_group.process_is_alive`, which is where the
        answer stops being `os.kill(pid, 0)`: on Windows that call terminates its subject and
        raises `WinError 87` for a pid that already exited. Only the zombie refinement is left
        here, because only POSIX has zombies.
        """
        if not cell_group.process_is_alive(pid):
            return False
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists():
            return True  # not Linux, or no procfs: the signal-0 answer is all there is
        try:
            # `comm` can contain spaces and parentheses; the state letter is the field after the
            # LAST ')'.
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False  # it went away between the probe and the read

    def wait_gone(self, pid, deadline_s=15.0):
        end = time.monotonic() + deadline_s
        while time.monotonic() < end and self.alive(pid):
            time.sleep(0.05)
        return not self.alive(pid)

    def test_a_held_termination_is_delivered_during_exceptional_unwinding(self):
        # Code after a contextmanager's finally is skipped when the body raises.  The old
        # deferral put delivery there, so a SIGTERM held while spawn/cleanup also raised stayed
        # pending until a later child -- or forever when this was the last one.
        cancellation = orchestrate._cancellation
        self.assertEqual(cancellation.depth, 0)
        self.assertIsNone(cancellation.held_signal)
        try:
            with self.assertRaises(SystemExit) as raised:
                with cancellation.deferring():
                    cancellation.held_signal = signal.SIGTERM
                    raise RuntimeError("the concurrent child failure")
            self.assertIn("terminated by signal", str(raised.exception))
            self.assertIn("subprocess it was running", str(raised.exception))
            self.assertIsInstance(raised.exception.__context__, RuntimeError)
        finally:
            cancellation.depth = 0
            cancellation.held_signal = None

    def test_a_held_termination_does_not_replace_an_orchestrator_cleanup_failure(self):
        cancellation = orchestrate._cancellation
        self.assertEqual(cancellation.depth, 0)
        self.assertIsNone(cancellation.held_signal)
        try:
            with self.assertRaises(cell_group.CleanupFailed) as raised:
                with cancellation.deferring():
                    cancellation.held_signal = signal.SIGTERM
                    raise cell_group.CleanupFailed("SURVIVORS still hold the device")
            self.assertIn("SURVIVORS", str(raised.exception))
            self.assertNotIn("killed first", str(raised.exception))
        finally:
            cancellation.depth = 0
            cancellation.held_signal = None

    def test_the_tinygrad_cache_probe_collects_helpers_it_spawned(self):
        # This probe was the final raw subprocess.run site in the matrix sweep.  Importing a
        # framework can itself create helpers, so a direct-child timeout is not enough.
        package = self.dir / "tinygrad"
        package.mkdir()
        (package / "__init__.py").write_text("")
        pidfile = self.dir / "probe-helper.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cachedb = self.dir / "cache.db"
        (package / "helpers.py").write_text(
            "import os, subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            + publish_pid("os.environ['PROBE_HELPER_PID']", "kid.pid")
            + "CACHEDB = os.environ['CACHEDB']\n"
        )
        env = os.environ.copy()
        env.update(
            {
                "CACHEDB": str(cachedb),
                "PYTHONPATH": str(self.dir),
                "PROBE_HELPER_PID": str(pidfile),
            }
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "VENV_PY", Path(sys.executable)))
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            found = orchestrate.tinygrad_cachedb(env)

        self.assert_tinygrad_cache_in_scratch(found)
        self.assertEqual(found, cachedb)
        self.assertTrue(pidfile.exists(), "the probe fixture did not spawn its helper")
        helper = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(helper), f"pid {helper} outlived the cache probe")

    def test_the_tinygrad_cache_probe_refuses_an_ambient_cache(self):
        # Negative control for the containment assertion above: the old fallback selected the
        # user's cache after an import timeout. Reproduce the same resolved-path shape even with
        # the explicit cache setting present, and prove the test refuses before inspecting it.
        configured = self.dir / "cache.db"
        ambient = self.dir.parent / "ambient-tinygrad-cache" / "cache.db"
        probe = subprocess.CompletedProcess([], 0, f"{ambient}\n", "")

        with unittest.mock.patch.object(orchestrate, "run_supporting", return_value=probe):
            found = orchestrate.tinygrad_cachedb({"CACHEDB": str(configured)})
            with self.assertRaisesRegex(
                self.failureException, "resolved outside the test scratch directory"
            ):
                self.assert_tinygrad_cache_in_scratch(found)

    def test_a_finished_cell_is_unaffected_by_the_cap(self):
        # The cap must be invisible to every cell that runs: same result line, no note.
        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )

        result, note, _ = self.run_cell("finished", cell, timeout=60)

        self.assertIsNone(note)
        self.assertEqual(result["workload"], "w")

    def test_the_cap_can_be_turned_off(self):
        # `--cell-timeout 0` is the escape hatch for a box whose legitimate cells outrun any cap
        # worth setting; it must mean "no cap", not "a cap of zero seconds".
        cell = self.python(
            "import json, time; time.sleep(0.2); print(json.dumps("
            "{'workload': 'uncapped', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )

        result, note, _ = self.run_cell("uncapped", cell, timeout=0)

        self.assertIsNone(note)
        self.assertEqual(result["workload"], "uncapped")

    def test_a_cell_over_the_cap_is_a_runner_failure_naming_the_cap(self):
        pidfile = self.dir / "wedged.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )
        result, note, log = self.run_cell("wedged", cell, timeout=1.0)

        self.assertIsNone(result)
        self.assertIn("TIMED OUT after 1s", note)
        self.assertIn("--cell-timeout", note)
        self.assertIn("wedged", log)
        # It checkpointed nothing, so the note names no checkpoint (gh-ocannl-1209).
        self.assertNotIn("checkpoint", note)

    def golden_checkpoint(self, name):
        """A checkpoint line as the OCaml emitter built it (see CheckpointLineTest)."""
        return golden_checkpoint_line("gpt2_mini_train_s1024", dominant_kernel=name)

    # What a runner writes before its dominant-kernel instrument, as the first line of a child's
    # source: the golden checkpoint arrives as the child's last argument.
    WRITE_CHECKPOINT = "import sys; sys.stderr.write(sys.argv[-1] + '\\n'); sys.stderr.flush()\n"

    def test_a_late_timeout_keeps_the_losses_checkpointed_before_it(self):
        # The TUF s1024 case: every parity loss observed, then the cap expires inside the
        # dominant-kernel instrument. The cell is a failure; its losses are kept beside it.
        pidfile = self.dir / "late.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            self.WRITE_CHECKPOINT
            + publish_pid("sys.argv[1]", "os.getpid()")
            + "import time; time.sleep(300)\n",
            pidfile,
            self.golden_checkpoint("running"),
        )
        kept = []

        result, note, _ = self.run_cell(
            "killed in diagnostics", cell, timeout=1.0, on_checkpoint=kept.append
        )

        self.assertIsNone(result)
        self.assertIn("TIMED OUT after 1s", note)
        self.assertIn("kept 6 of 6 parity losses", note)
        self.assertIn("running: dominant_kernel", note)
        self.assertIn("not an accepted result", note)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["losses"], [10.375, 10.25, 10.125, 10.0, 9.875, 9.75])
        self.assertEqual(kept[0]["stages"]["timing"], "complete")

    def test_a_failed_exit_keeps_its_checkpoint_too(self):
        cell = self.python(self.WRITE_CHECKPOINT + "sys.exit(2)\n", self.golden_checkpoint("running"))
        kept = []

        result, note, _ = self.run_cell(
            "crashed in diagnostics", cell, timeout=60, on_checkpoint=kept.append
        )

        self.assertIsNone(result)
        self.assertIn("exit 2", note)
        self.assertIn("kept 6 of 6 parity losses", note)
        self.assertEqual(len(kept), 1)

    def test_a_cell_that_finishes_reports_its_result_not_its_checkpoints(self):
        # Negative control: the checkpoints of a cell that went on to emit its result line are
        # superseded by it. (A timeout with no checkpoint names none: the cap test above.)
        cell = self.python(
            self.WRITE_CHECKPOINT
            + "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))\n",
            self.golden_checkpoint("running"),
        )
        kept = []

        result, note, _ = self.run_cell(
            "finished after checkpoints", cell, timeout=60, on_checkpoint=kept.append
        )

        self.assertIsNone(note)
        self.assertEqual(result["workload"], "w")
        self.assertEqual(kept, [])

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_the_kill_takes_the_whole_process_group(self):
        # The failure mode this exists for: tinygrad's beam search wedges with a pool of workers
        # alive, and those workers hold the cell's stdout pipe open. Kill the direct child alone
        # and the sweep then blocks reading a pipe nobody will ever close -- the hang moves, it
        # does not go away. So the assertion is both halves: the grandchild dies, and the call
        # returns promptly rather than waiting on the inherited pipe.
        pidfile = self.dir / "grandchild.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "time.sleep(300)\n",
            pidfile,
        )

        t0 = time.monotonic()
        result, note, _ = self.run_cell("wedged with workers", cell, timeout=2.0)
        elapsed = time.monotonic() - t0

        self.assertIsNone(result)
        self.assertIn("process group", note)
        self.assertLess(elapsed, 2.0 + orchestrate.CELL_KILL_GRACE_S + 10)
        grandchild = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(grandchild), f"pid {grandchild} survived the cap")

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_the_escalation_is_decided_by_the_group_and_not_by_the_pipe(self):
        # The survivor case the pipe cannot see: a descendant that ignores SIGTERM and does NOT
        # hold the cell's stdout. The cell itself dies on SIGTERM, the pipe closes, and a kill path
        # that escalates only when its read blocks would return here reporting a killed group --
        # while the survivor keeps the GPU and every later cell of the sweep is measured against
        # it. So the SIGKILL has to be owed to the GROUP still having members (gh-ocannl-760
        # review).
        pidfile = self.dir / "stubborn.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import signal, subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c',\n"
            "  'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            " time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "sys.stdout.flush()\n"
            "time.sleep(300)\n",
            pidfile,
        )

        # The grace is shortened because this test spends ALL of it: the leader dies on the first
        # SIGTERM, so nothing can end the first pass early and the loop runs to its deadline
        # before the SIGKILL that is the point of the test. At the production 10 s that is ten
        # seconds of sleeping on every run of this suite on both CI platforms, and none of it is
        # under test here -- what is under test is which pass the escalation is owed to, and that
        # is the same at any grace length.
        with unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.5):
            result, note, _ = self.run_cell("stubborn descendant", cell, timeout=2.0)

        self.assertIsNone(result)
        stubborn = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(stubborn), f"pid {stubborn} ignored SIGTERM and survived")
        self.assertNotIn("SURVIVED SIGKILL", note)

    def test_a_member_outliving_sigkill_is_said_so_in_the_failure(self):
        # Nothing survives SIGKILL except a process stuck in the kernel (an uninterruptible driver
        # ioctl), which is not synthesizable -- so the group probe is stood in for. What is pinned
        # is the consequence: the sweep goes on, and the failure says every later cell in the run
        # was measured against whatever is still holding the device. With the probe stood in for,
        # nothing here observes whether the kill landed, so the cleanup is what makes sure of it.
        pidfile = self.dir / "stuck.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            _, note, _ = self.run_cell("stuck in the driver", cell, timeout=0.5)

        self.assertIn("SURVIVED SIGKILL", note)
        self.assertIn("measured against it", note)

    @unittest.skipUnless(os.name == "posix", "SIGALRM and process groups are POSIX here")
    def test_an_interrupted_cell_hands_over_its_checkpoint_before_propagating(self):
        # gh-ocannl-1209 review: the cancellation path deletes a temporary log, so the losses must
        # leave through on_checkpoint before the interrupt does.
        pidfile = self.dir / "interrupted-late.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            self.WRITE_CHECKPOINT
            + publish_pid("sys.argv[1]", "os.getpid()")
            + "import time; time.sleep(300)\n",
            pidfile,
            self.golden_checkpoint("running"),
        )
        kept = []

        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)

        with self.assertRaises(KeyboardInterrupt):
            self.run_cell("interrupted in diagnostics", cell, on_checkpoint=kept.append)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["losses"], [10.375, 10.25, 10.125, 10.0, 9.875, 9.75])

    def test_an_interrupted_cell_is_handed_over_with_or_without_a_checkpoint(self):
        # Review round 2 (gh-ocannl-1234): an interrupt before the first checkpoint is a failed
        # cell too, so `on_interrupted` hears of it with None rather than not at all.
        checkpoint = orchestrate.last_checkpoint(self.golden_checkpoint("running"))

        def interrupted(after_checkpoint):
            def run_cell(label, cmd, on_checkpoint=None, **kwargs):
                if after_checkpoint:
                    on_checkpoint(checkpoint)
                raise KeyboardInterrupt
            return run_cell

        for after_checkpoint, expected in ((False, [None]), (True, [checkpoint])):
            with self.subTest(after_checkpoint=after_checkpoint):
                heard = []
                with unittest.mock.patch.object(
                    orchestrate, "run_cell", side_effect=interrupted(after_checkpoint)
                ), self.assertRaises(KeyboardInterrupt):
                    orchestrate.run_cell_kept("cell", ["x"], on_interrupted=heard.append)
                self.assertEqual(heard, expected)

    @unittest.skipUnless(os.name == "posix", "SIGALRM and process groups are POSIX here")
    def test_an_interrupted_cell_gets_the_same_cache_treatment_as_a_capped_one(self):
        # Ctrl-C on a wedged beam cell is the likeliest way anyone meets this bug by hand, and the
        # search it interrupts leaves exactly the partial cache the cap's kill path quarantines.
        called = []
        pidfile = self.dir / "interrupted.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )

        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)

        with self.assertRaises(KeyboardInterrupt):
            self.run_cell(
                "interrupted",
                cell,
                on_incomplete=lambda killed: called.append(("killed", killed))
                or "quarantined the cache",
            )

        # ... and told that this cell was KILLED, which is what decides whether the cache may
        # be moved aside at all.
        self.assertEqual(called, [("killed", True)])

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_sigterm_to_the_sweep_takes_the_running_cell_with_it(self):
        # A job cancellation or a scheduler's time limit is a SIGTERM to the sweep, and Python's
        # default action for it exits without unwinding -- so without a handler the cell and its
        # whole worker pool are orphaned, holding the GPU, with nobody to reap them. The cell is
        # in its own session precisely so the sweep's signals do NOT reach it (gh-ocannl-760
        # review).
        pidfile = self.dir / "orphan.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell_source = (
            "import subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "time.sleep(300)\n"
        )
        driver = self.python(
            "import sys, orchestrate\n"
            "orchestrate.install_termination_handler()\n"
            f"cell = [sys.executable, '-c', {cell_source!r}, sys.argv[1]]\n"
            "orchestrate.run_cell('cell under a cancelled sweep', cell)\n",
            pidfile,
        )
        sweep = subprocess.Popen(
            driver,
            cwd=str(Path(orchestrate.__file__).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(lambda: sweep.poll() is None and sweep.kill())
        end = time.monotonic() + 30
        while time.monotonic() < end and not pidfile.exists():
            time.sleep(0.05)
        self.assertTrue(pidfile.exists(), "the cell never started")
        grandchild = int(pidfile.read_text())

        sweep.terminate()
        sweep.wait(timeout=30)

        self.assertTrue(
            self.wait_gone(grandchild), f"pid {grandchild} outlived the cancelled sweep"
        )

    @unittest.skipUnless(os.name == "posix", "SIGHUP is POSIX-only")
    def test_a_hangup_of_the_sweep_takes_its_cell_and_its_build_with_it(self):
        # gh-ocannl-1234: the hangup of the terminal or ssh session the sweep runs in, delivered to
        # orchestrate.py's own process while it waits on a cell and while it waits on a supporting
        # command (a build). Both children lead sessions of their own, so the sweep is the only one
        # to hear it, and each forks a SIGTERM-ignoring worker that only the group's SIGKILL takes.
        for entry in ("run_cell('cell under a hung-up sweep', child)", "run_supporting(child)"):
            with self.subTest(entry=entry):
                tag = entry.split("(")[0]
                pidfile = self.dir / f"hangup-{tag}-worker.pid"
                kill_the_group_on_cleanup(self, pidfile)
                leader = self.dir / f"hangup-{tag}-leader.pid"
                kill_the_group_on_cleanup(self, leader)
                child_source = (
                    "import subprocess, sys, time\n"
                    "kid = subprocess.Popen([sys.executable, '-c', 'import signal, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])\n"
                    + publish_pid("sys.argv[1]", "kid.pid")
                    + publish_pid("sys.argv[2]", "os.getpid()")
                    + "time.sleep(300)\n"
                )
                driver = self.python(
                    "import signal, sys, orchestrate\n"
                    # A sweep started from a terminal: the hangup is a termination (one started
                    # under nohup keeps ignoring it, test_a_nohup_driver_keeps_ignoring_hangups).
                    "signal.signal(signal.SIGHUP, signal.SIG_DFL)\n"
                    "orchestrate.CELL_KILL_GRACE_S = 0.5\n"
                    "orchestrate.install_termination_handler()\n"
                    f"child = [sys.executable, '-c', {child_source!r}, sys.argv[1], sys.argv[2]]\n"
                    f"orchestrate.{entry}\n"
                    "print('returned')\n",
                    pidfile,
                    leader,
                )
                sweep = subprocess.Popen(
                    driver,
                    cwd=str(Path(orchestrate.__file__).parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.addCleanup(lambda sweep=sweep: sweep.poll() is None and sweep.kill())
                end = time.monotonic() + 30
                while time.monotonic() < end and not (pidfile.exists() and leader.exists()):
                    time.sleep(0.05)
                self.assertTrue(pidfile.exists() and leader.exists(), "the child never started")
                worker, child = int(pidfile.read_text()), int(leader.read_text())

                sweep.send_signal(signal.SIGHUP)
                out, err = sweep.communicate(timeout=30)

                self.assertNotEqual(sweep.returncode, 0, (out, err))
                self.assertNotIn("returned", out)
                self.assertIn("terminated by signal", err)
                self.assertTrue(self.wait_gone(child), f"the {tag} child outlived the hangup")
                self.assertTrue(self.wait_gone(worker), f"the {tag} worker outlived the hangup")

    # gh-ocannl-1234: a cell killed at its cap after checkpointing its parity losses, in each sweep
    # driver -- orchestrate's shared runner and the records gh1181, gh675 and gh1002 make of what it
    # left. Each record carries the checkpoint whole, beside its provenance.
    def test_the_shared_runner_and_gh1181_keep_a_killed_cells_checkpoint(self):
        import gh1181_cells

        # orchestrate.py's runner, and gh1181_cells.py's record of what it returned.
        pidfile = self.dir / "kept.pid"
        kill_the_group_on_cleanup(self, pidfile)
        line = self.golden_checkpoint("running")
        expected = orchestrate.last_checkpoint(line)
        cell = self.python(
            self.WRITE_CHECKPOINT
            + publish_pid("sys.argv[1]", "os.getpid()")
            + "import time; time.sleep(300)\n",
            pidfile,
            line,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result, note, checkpoint = orchestrate.run_cell_kept("killed", cell, timeout=1.0)
        self.assertIsNone(result)
        self.assertIn("TIMED OUT after 1s", note)
        self.assertEqual(checkpoint, expected)
        record = gh1181_cells.cell_failure(
            "gpt2_mini_train_s1024", "ocannl-master-exact", 1, "search", note,
            {"OCANNL_BACKEND": "cuda", "PATH": "/bin"}, "exact",
            {"gpt2_mini_train_s1024": {"fixture_sha256": "f" * 64}}, {"master": "a" * 40},
            checkpoint,
        )
        self.assertEqual(record["checkpoint"], expected)
        self.assertEqual(record["why"], note)
        self.assertEqual(record["regime"], "exact")
        self.assertEqual(record["ambient_ocannl_env"], {"OCANNL_BACKEND": "cuda"})
        self.assertEqual(record["provenance"]["fixture_sha256"], "f" * 64)
        self.assertEqual(record["provenance"]["source_revision"], "a" * 40)
        self.assertEqual((record["arm"], record["repeat"], record["stage"]),
                         ("ocannl-master-exact", 1, "search"))

    def test_gh675_keeps_a_killed_passs_checkpoint(self):
        import gh675_cells

        # A pass that wedged is a failed record, not an exception.
        pidfile = self.dir / "gh675.pid"
        kill_the_group_on_cleanup(self, pidfile)
        line = self.golden_checkpoint("running")
        expected = orchestrate.last_checkpoint(line)
        cell = self.python(
            self.WRITE_CHECKPOINT
            + publish_pid("sys.argv[1]", "os.getpid()")
            + "import time; time.sleep(300)\n",
            pidfile,
            line,
        )
        with unittest.mock.patch.object(gh675_cells, "gate", return_value=True), \
                unittest.mock.patch.object(gh675_cells, "TASKSET", []), \
                unittest.mock.patch.dict(gh675_cells.DIGESTS, {"gpt2_mini": "d" * 64}), \
                unittest.mock.patch.dict(gh675_cells.SOURCE, {"source_revision": "b" * 40}), \
                contextlib.redirect_stdout(io.StringIO()):
            record = gh675_cells.run(
                "D_ocannl gpt2_mini rep1 pass1", cell, timeout=1.0, workload="gpt2_mini"
            )
        self.assertTrue(record["__failed__"])
        self.assertIn("TIMED OUT after 1s", record["why"])
        self.assertEqual(record["checkpoint"], expected)
        self.assertEqual(record["provenance"]["fixture_sha256"], "d" * 64)
        self.assertEqual(record["provenance"]["source_revision"], "b" * 40)
        self.assertIs(record["gate_ok"], True)

    def test_gh675_records_a_pass_whose_group_survived_before_stopping(self):
        # Review round 1: a pass whose group was not proven gone stops the sweep, but only after
        # the pair's record -- with the pass's checkpoint -- is in records.jsonl, and pass 2 is
        # never launched beside the survivor.
        import gh675_cells

        note = "TIMED OUT after 1s (cap); A MEMBER SURVIVED SIGKILL (survivors)"
        checkpoint = orchestrate.last_checkpoint(self.golden_checkpoint("running"))
        with unittest.mock.patch.object(gh675_cells, "gate", return_value=True), \
                unittest.mock.patch.object(
                    orchestrate, "run_cell_kept", return_value=(None, note, checkpoint)), \
                contextlib.redirect_stdout(io.StringIO()):
            failed = gh675_cells.run("A_beam2 mlp_small rep1 pass1", ["x"], workload="mlp_small")
        self.assertIs(failed["cleanup_unproven"], True)
        self.assertEqual(failed["checkpoint"], checkpoint)
        # Negative control: an ordinary failure carries no such mark.
        with unittest.mock.patch.object(gh675_cells, "gate", return_value=True), \
                unittest.mock.patch.object(
                    orchestrate, "run_cell_kept", return_value=(None, "exit 2", None)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertNotIn("cleanup_unproven", gh675_cells.run("x", ["x"]))

        records = self.dir / "records.jsonl"
        launched = []

        def pass_run(label, cmd, env, timeout, workload):
            launched.append(label)
            return dict(failed)

        with unittest.mock.patch.object(gh675_cells, "RECORDS", records), \
                unittest.mock.patch.object(gh675_cells, "timed_run", side_effect=pass_run), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(cell_group.CleanupFailed, "SURVIVED SIGKILL"):
                gh675_cells.do_pair(
                    "A_beam2", "mlp_small", 1, lambda: (["a"], {}), lambda: (["b"], {}))
        self.assertEqual(len(launched), 1)
        rows = [json.loads(line) for line in records.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertIs(rows[0]["valid"], False)
        self.assertEqual(rows[0]["pass1"]["checkpoint"], checkpoint)

    def test_gh675_warm_control_prewarms_through_the_cell_runner(self):
        # Review round 1: the warm control's prewarm is a pass like any other, with its own label.
        import gh675_cells

        calls = []

        def prewarm(label, cmd, env, timeout, workload):
            calls.append((label, timeout, workload))
            (self.dir / "tiny_jitwarm_mlp_small" / "cache.db").write_text("warm")
            return {"step_ms": {"p50": 1.0}, "gate_ok": True}

        with unittest.mock.patch.object(gh675_cells, "CACHES", self.dir), \
                unittest.mock.patch.object(gh675_cells, "PREWARMED", set()), \
                unittest.mock.patch.object(gh675_cells, "timed_run", side_effect=prewarm), \
                unittest.mock.patch.object(gh675_cells, "do_pair") as pair, \
                contextlib.redirect_stdout(io.StringIO()):
            gh675_cells.ARMS["C_jit_warm"]("mlp_small", 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("C_jit_warm mlp_small rep1 prewarm", calls[0][0])
        self.assertEqual(calls[0][1:], (1200, "mlp_small"))
        pair.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "the stand-in runner is a shebang script")
    def test_gh1002_keeps_a_killed_cells_checkpoint(self):
        import gh1002_cells

        # Its own runner and .out/.err layout, orchestrate's record.
        pidfile = self.dir / "gh1002.pid"
        kill_the_group_on_cleanup(self, pidfile)
        line = self.golden_checkpoint("running")
        expected = orchestrate.last_checkpoint(line)
        runner = self.dir / "bench_gpt"
        runner.write_text(
            f"#!{sys.executable}\nimport os, sys\n"
            f"sys.stderr.write({line!r} + '\\n'); sys.stderr.flush()\n"
            + publish_pid(repr(str(pidfile)), "os.getpid()")
            + "import time; time.sleep(300)\n"
        )
        runner.chmod(0o755)
        out = self.dir / "gh1002-out"
        out.mkdir()
        with unittest.mock.patch.object(gh1002_cells, "BENCH_GPT", runner), \
                unittest.mock.patch.object(gh1002_cells, "CELL_TIMEOUT_S", 1), \
                unittest.mock.patch.object(gh1002_cells, "CELL_TERMINATE_GRACE_S", 0.1), \
                contextlib.redirect_stdout(io.StringIO()):
            gh1002_cells.run_cell(out, "cc", "gpt2_mini_train", "A", 0)
        records = [
            strict_loads(text) for text in (out / "failures.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(records), 1)
        self.assertIn("TIMED OUT after 1s", records[0]["why"])
        self.assertEqual(records[0]["checkpoint"], expected)
        self.assertEqual(records[0]["regime"], "exact")
        self.assertEqual(
            (records[0]["backend"], records[0]["fixture"], records[0]["treatment"]),
            ("cc", "gpt2_mini_train", "A"),
        )

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "the zombie distinction needs procfs")
    def test_a_group_of_nothing_but_zombies_is_not_alive(self):
        # What the escalation is asking is "does anything still hold the device", and a zombie
        # holds nothing. But `killpg(pgid, 0)` succeeds for a group whose every member is one, and
        # after the kill that is the normal state of the descendants -- orphans, reaped or not at
        # the whim of whoever inherits them. Under a PID 1 that does not reap (a container) a bare
        # signal-0 probe would announce a SIGKILL survivor that does not exist, and sit through
        # both grace periods to do it (gh-ocannl-760 review).
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, start_new_session=True
        )
        self.addCleanup(proc.wait)
        # Bounded, not `while alive`: against a probe that cannot see the distinction this must
        # FAIL rather than hang. Deliberately unreaped throughout -- the process becomes a zombie
        # in its own group (pgid == pid), and signal 0 still finds it.
        end = time.monotonic() + 10
        alive = True
        while time.monotonic() < end and alive:
            alive = (
                cell_group._observe_posix_group(proc.pid, allow_zombie_gone=True)
                is not cell_group.GONE
            )
            if alive:
                time.sleep(0.02)
        os.kill(proc.pid, 0)

        self.assertFalse(alive, "a group of nothing but zombies read as alive")

    def test_two_cells_killed_in_the_same_second_do_not_overwrite_each_other(self):
        # The stamp is second-resolution, so a low cap or a GPU column that wedges at once puts
        # two kills inside one second -- and `os.replace` onto the same name would destroy the
        # first cell's quarantined database, which is the evidence the rename exists to keep.
        db = self.dir / "cache.db"
        db.write_bytes(b"first wedged search")
        first = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})
        db.write_bytes(b"second wedged search")

        second = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})

        kept = sorted(p for p in self.dir.glob("cache.db.wedged-*"))
        self.assertEqual(len(kept), 2, kept)
        self.assertEqual(
            sorted(p.read_bytes() for p in kept),
            [b"first wedged search", b"second wedged search"],
        )
        self.assertNotEqual(first, second)

    def test_the_fleet_default_disables_the_spawn_pool(self):
        # The deadlock exists in tinygrad 0.13.0 and 0.14.0, on both GPU boxes. Zero is the only
        # value that removes that pool; an ambient positive value must not override the pin.
        ambient = {"PATH": "/usr/bin", "PARALLEL": "32"}

        self.assertEqual(orchestrate.DEFAULT_BEAM_PARALLEL, 0)
        self.assertEqual(
            orchestrate.beam_cell_env(ambient, orchestrate.DEFAULT_BEAM_PARALLEL)["PARALLEL"],
            "0",
        )
        self.assertEqual(orchestrate.beam_cell_env(ambient, 4)["PARALLEL"], "4")
        # Zero is a value, not an absence: it is what disables tinygrad's pool outright.
        self.assertEqual(orchestrate.beam_cell_env(ambient, 0)["PARALLEL"], "0")

    def test_an_explicit_upstream_default_drops_ambient_parallel(self):
        # None remains the programmatic spelling of tinygrad's own default. It must not inherit
        # a shell experiment under a result whose absent beam_parallel field says "upstream".
        ambient = {"PATH": "/usr/bin", "PARALLEL": "0"}

        self.assertNotIn("PARALLEL", orchestrate.beam_cell_env(ambient, None))

    @unittest.skipUnless(os.name == "posix", "the deferral is about POSIX signal delivery")
    def test_a_cancellation_inside_the_spawn_window_still_kills_the_cell(self):
        # Between `_execute_child` starting the cell and `Popen` returning it, no name refers to
        # the new process -- so an exception raised there unwinds past a cleanup with nothing to
        # clean, and the isolated runner is orphaned on the GPU with the sweep gone. The signal is
        # delivered here at exactly that point, by a Popen wrapper.
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_int = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGTERM, previous_term)
        self.addCleanup(signal.signal, signal.SIGINT, previous_int)
        orchestrate.install_termination_handler()
        # The pidfile is the cleanup's alone -- the assertion below reads the pid from Popen: a
        # cancellation that fails leaves the cell to run, publish and sleep to its 60 s cap.
        pidfile = self.dir / "mid-spawn.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )
        real_popen = cell_group.subprocess.Popen
        spawned = []

        def popen_then_cancel(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            # The child's pid is taken HERE rather than from a file it writes: at this point it
            # may not have run a line of its own yet, and a cancellation that works kills it
            # before it does.
            spawned.append(proc.pid)
            os.kill(os.getpid(), signal.SIGTERM)  # lands inside the window
            return proc

        started = time.monotonic()
        with unittest.mock.patch.object(
            cell_group.subprocess, "Popen", side_effect=popen_then_cancel
        ):
            with self.assertRaises(SystemExit):
                self.run_cell("cancelled mid-spawn", cell, timeout=60)
        elapsed = time.monotonic() - started

        self.assertEqual(len(spawned), 1)
        self.assertTrue(self.wait_gone(spawned[0]), "the cell outlived the cancellation")
        # And it was killed on the way out of the spawn window, not left running until the cell's
        # own cap noticed: a held signal is delivered at the first point where raising is safe.
        self.assertLess(elapsed, orchestrate.CELL_KILL_GRACE_S + 5)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_finished_cell_that_left_workers_behind_has_them_collected(self):
        # `communicate` returned because the LEADER exited and the pipe closed, which says nothing
        # about a worker that redirected its own output -- and that worker still holds the GPU, so
        # every later cell of the sweep would be measured against it (gh-ocannl-760 review).
        pidfile = self.dir / "leftover.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "sys.exit(1)\n",
            pidfile,
        )

        result, note, _ = self.run_cell("leaky cell", cell, timeout=60)

        self.assertIsNone(result)
        self.assertIn("exit 1", note)
        self.assertIn("process group behind", note)
        leftover = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(leftover), f"pid {leftover} outlived its cell")

    def test_a_survivor_makes_a_successful_cell_a_failure(self):
        # The cell ran, printed a result line and exited zero -- and something it spawned outlived
        # SIGKILL and still holds the device. That makes this row's own timing suspect and every
        # later row of the run too, so recording it as a success (with a warning on a console
        # nobody keeps) publishes exactly what the failure section exists to stop. The survivor is
        # stood in for: nothing outlives SIGKILL but a process stuck in the kernel.
        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            result, note, _ = self.run_cell("successful with a survivor", cell, timeout=60)

        self.assertIsNone(result)
        self.assertIn("SURVIVED SIGKILL", note)
        self.assertIn("every later cell", note)

    @unittest.skipUnless(os.name == "posix", "the deferral is about POSIX signal delivery")
    def test_a_cancellation_during_the_kill_does_not_abandon_the_group(self):
        # A signal arriving while `kill_cell_group` is in its grace loop raises out of the `except
        # TimeoutExpired` clause that is doing the killing -- and a sibling `except BaseException`
        # does not catch what another `except` clause raises, so the escalation stops halfway and
        # the group outlives the sweep (gh-ocannl-760 review). The cell here holds the grace open
        # by ignoring SIGTERM, and the sweep is cancelled while it does.
        pidfile = self.dir / "mid_kill.pid"
        kill_the_group_on_cleanup(self, pidfile)
        termfile = self.dir / "mid_kill.termed"
        # The cell RECORDS the SIGTERM instead of `SIG_IGN`-ing it -- it still does not die of one,
        # which is all the test needs of it, and the marker is the moment the grace loop opened.
        # Sleeping a fixed 4 s to land inside a 10 s window was the old way to hit that window,
        # and it cost the suite the wait on every run; waiting for the marker hits it exactly, so
        # the grace can be shortened (below) without the cancellation racing the SIGKILL. (The
        # marker is read for its existence alone, so an empty file is the whole of it.)
        cell_source = (
            "import os, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: open(sys.argv[2], 'w').close())\n"
            + publish_pid("sys.argv[1]", "os.getpid()")
            + "sys.stdout.flush()\n"
            "time.sleep(300)\n"
        )
        driver = self.python(
            "import sys, orchestrate\n"
            "orchestrate.install_termination_handler()\n"
            # The grace this cancellation lands in has to be long enough for the parent to see the
            # marker and signal, not long enough to be realistic: what is under test is that the
            # escalation FINISHES across a signal, and the deferral that makes it do so is the
            # same at any grace length.
            "orchestrate.CELL_KILL_GRACE_S = 2.0\n"
            f"cell = [sys.executable, '-c', {cell_source!r}, sys.argv[1], sys.argv[2]]\n"
            "orchestrate.run_cell('cell cancelled mid-kill', cell, timeout=2)\n",
            pidfile,
            termfile,
        )
        sweep = subprocess.Popen(
            driver,
            cwd=str(Path(orchestrate.__file__).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(lambda: sweep.poll() is None and sweep.kill())
        end = time.monotonic() + 30
        while time.monotonic() < end and not pidfile.exists():
            time.sleep(0.05)
        self.assertTrue(pidfile.exists(), "the cell never started")
        stubborn = int(pidfile.read_text())
        # Through the cap (2 s) and into the grace, where the kill is waiting on a cell that will
        # not answer SIGTERM: the marker is written by the cell's handler, from inside that wait.
        end = time.monotonic() + 30
        while time.monotonic() < end and not termfile.exists():
            time.sleep(0.02)
        self.assertTrue(termfile.exists(), "the kill never reached its grace loop")
        self.assertTrue(self.alive(stubborn), "the cell was gone before the kill was interrupted")

        sweep.terminate()
        sweep.wait(timeout=40)

        self.assertTrue(
            self.wait_gone(stubborn, deadline_s=30),
            f"pid {stubborn} outlived a sweep cancelled mid-kill",
        )

    def test_a_search_that_failed_partway_names_its_cache_too(self):
        # A search that exits nonzero partway leaves the same partial cache a killed one does --
        # some arms committed, the rest never run -- and the next attempt over it reports a
        # provenance nobody wrote. So the cell is asked on this path too, with killed=False: the
        # risk is the same, what may be DONE about it is not (the cache is shared with every later
        # cell of the sweep, and the failure may have been a pre-search one).
        seen = []
        real_spawn = cell_group.spawn

        def spawn_without_console(*args, **kwargs):
            # A detached Windows supervisor otherwise gives this trivial cell a conhost.exe.
            # That helper can still be alive after Python exits, so collecting it correctly
            # reports killed=True. This fixture needs an ordinary exit with NO remaining member;
            # suppress the console helper while retaining the real Job/observation/cleanup path.
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS
            return real_spawn(*args, **kwargs)

        with unittest.mock.patch.object(cell_group, "spawn", side_effect=spawn_without_console):
            result, note, output = self.run_cell(
                "failed search",
                self.python("import sys; sys.exit(3)"),
                timeout=60,
                on_incomplete=lambda killed: seen.append(killed) or "CACHE AT RISK: partial search",
            )

        self.assertIsNone(result)
        self.assertEqual(seen, [False], output)
        self.assertIn("exit 3", note)
        self.assertIn("CACHE AT RISK", note)

    def test_the_failed_path_leaves_the_shared_cache_alone(self):
        # ... and "asked" must not mean "moved": a cell that exited on its own may never have
        # searched at all, while the cache is shared with every later cell of the sweep.
        db = self.dir / "cache.db"
        db.write_bytes(b"the sweep's warm kernels")

        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)}, killed=False)

        self.assertTrue(db.exists())
        self.assertEqual(db.read_bytes(), b"the sweep's warm kernels")
        self.assertIn("CACHE AT RISK", note)
        self.assertIn("exited rather than being killed", note)

    def test_the_report_names_the_pool_a_beam_row_searched_with(self):
        # A beam row's compile cost IS a search cost, and the pool changes it threefold -- so
        # `beam` alone gives the default, `PARALLEL=0` and an explicit N one identity in the table
        # people read numbers out of. The default keeps the bare name (every report so far
        # recorded it that way, and rows that are the same should read the same).
        out = Path(tempfile.mkdtemp())
        rows = [
            cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1]),
            cell("tinygrad", "HIP", "beam", [2.3, 2.2, 2.1]),
            cell("tinygrad", "HIP", "beam", [2.3, 2.2, 2.1]),
        ]
        rows[1]["beam_parallel"] = 0
        rows[2]["beam_parallel"] = None
        orchestrate.parity_check(rows)

        with contextlib.redirect_stdout(io.StringIO()):
            orchestrate.report(rows, out)

        text = (out / "report.md").read_text()
        self.assertIn("| beam P=0 |", text)
        self.assertIn("| beam |", text)

    def test_a_cell_whose_leftovers_were_killed_reports_a_killed_search(self):
        # The leftover sweep interrupted a member -- possibly mid-write -- so the cache is in the
        # state the kill path quarantines, whatever the leader's own exit said.
        seen = []
        pidfile = self.dir / "leftover_killed.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "sys.exit(1)\n",
            pidfile,
        )

        _, note, _ = self.run_cell(
            "failed leaving workers",
            cell,
            timeout=60,
            on_incomplete=lambda killed: seen.append(killed) or "cache handled",
        )

        self.assertEqual(seen, [True], "a cell whose members we killed reported killed=False")
        self.assertIn("process group behind", note)
        self.assertTrue(self.wait_gone(int(pidfile.read_text())))

    def test_a_survivor_failure_still_handles_the_cache(self):
        # The survivor branch returns early, and part of the search was forcibly interrupted --
        # which is what `stuck` means -- so the cache is in the state the cap's path quarantines.
        # Returning without asking would leave it for the next beam cell to read.
        seen = []
        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            _, note, _ = self.run_cell(
                "successful over a survivor",
                cell,
                timeout=60,
                on_incomplete=lambda killed: seen.append(killed) or "cache quarantined",
            )

        self.assertEqual(seen, [True])
        self.assertIn("SURVIVED SIGKILL", note)
        self.assertIn("cache quarantined", note)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_supporting_build_is_killed_with_its_own_children(self):
        # The sweep's own subprocesses -- the build, the device probes -- get the same discipline
        # as a cell: `subprocess.run` kills its direct child on an exception but knows nothing of
        # that child's children, and `dune build` forks compilers (gh-ocannl-760 review).
        pidfile = self.dir / "build_worker.pid"
        kill_the_group_on_cleanup(self, pidfile)
        build = self.python(
            "import subprocess, sys, time\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "time.sleep(300)\n",
            pidfile,
        )

        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 1.5)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)

        with self.assertRaises(KeyboardInterrupt):
            orchestrate.run_supporting(build, capture_output=True)

        worker = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(worker), f"pid {worker} outlived the cancelled build")

    def test_a_cap_that_cannot_mean_anything_is_refused_before_the_first_cell(self):
        # A negative cap expires every `communicate` at once -- killing every cell of the sweep
        # and quarantining their caches on the way -- and nan/inf raise from inside it. Both are
        # worth refusing at the argument rather than cell by cell.
        for bad in ("-1", "nan", "inf", "-0.5"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                orchestrate.cell_timeout_arg(bad)
        self.assertEqual(orchestrate.cell_timeout_arg("0"), 0.0)  # 0 is "no cap", not "cap of 0"
        self.assertEqual(orchestrate.cell_timeout_arg("1800"), 1800.0)

    def test_a_negative_pool_size_is_refused_too(self):
        # `-1` is not "unset" and not "default": tinygrad builds a pool for any truthy PARALLEL
        # and hands the value to `multiprocessing.Pool`, which refuses a negative count -- so
        # every beam cell of the sweep would die inside the runner.
        for bad in ("-1", "-8"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                orchestrate.beam_parallel_arg(bad)
        self.assertEqual(orchestrate.beam_parallel_arg("0"), 0)  # 0 is no pool at all
        self.assertEqual(orchestrate.beam_parallel_arg("4"), 4)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_survivor_holding_the_pipe_does_not_swallow_the_cells_output(self):
        # A member that outlives SIGKILL still owns the captured stdout, so every `communicate`
        # in the kill loop times out and the output would be lost -- exactly where the partial log
        # is the only evidence about a cell nobody will run again. It lives on the exception.
        # Nothing outlives SIGKILL on demand, so the survivor is stood in for by a grandchild
        # that ESCAPES the group (its own session) while still holding the stdout it inherited --
        # which reproduces the property that matters here: every `communicate` in the kill loop
        # times out, so the output can only come off the exception.
        pidfile = self.dir / "pipe_holder.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import subprocess, sys, time\n"
            "print('the evidence', flush=True)\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  start_new_session=True)\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "time.sleep(300)\n",
            pidfile,
        )

        with unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 1.0):
            _, note, log = self.run_cell("wedged behind a held pipe", cell, timeout=1.5)

        self.assertIn("TIMED OUT", note)
        self.assertIn("the evidence", log)

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_kill_that_could_not_reap_still_lets_go_of_the_pipes(self):
        # The one way out of the kill loop without reaping is a member that outlived SIGKILL
        # holding the write end, so every `communicate` timed out and `Popen` was left owning an
        # open read end for the rest of the sweep -- a descriptor per unreapable cell, leaked on
        # the path least able to spare them (it announced itself as a ResourceWarning under
        # `dune build @benchmarks/runtest`). The survivor is stood in for by a grandchild that
        # ESCAPES the group into its own session while still holding the stdout it inherited,
        # which is what makes every reap in the loop time out.
        pidfile = self.dir / "unreapable.pid"
        kill_the_group_on_cleanup(self, pidfile)
        holder = cell_group.spawn(
            self.python(
                "import subprocess, sys, time\n"
                "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
                "  start_new_session=True)\n"
                + publish_pid("sys.argv[1]", "kid.pid")
                + "time.sleep(300)\n",
                pidfile,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # The kill loop cannot reap this leader (that is the point), so the test reaps it -- killing
        # it first when the test failed before the kill loop did, or the wait would sit out the
        # leader's whole 300 s sleep. The escaped survivor's cleanup cannot stand in for that: it
        # kills another session. The kill is asked of an UNREAPED leader only (`poll()` is None),
        # which is when its group id cannot yet belong to anything else.
        def reap_the_holder():
            if holder.poll() is None:
                holder.signal(force=True)
            holder.wait()

        self.addCleanup(reap_the_holder)

        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        with unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 1.0):
            orchestrate.kill_cell_group(holder)

        self.assertTrue(holder.stdout.closed, "the kill left the cell's read end open")
        # And the leader is reaped: what the loop could not finish is the READ, blocked on a pipe
        # the survivor holds, not the wait -- the leader took the SIGKILL.
        self.assertIsNotNone(holder.returncode, "the kill left the cell's leader unreaped")

    @unittest.skipUnless(os.name == "posix", "the cancellation is delivered by a signal here")
    def test_a_cancelled_supporting_command_preserves_its_cleanup_failure(self):
        # The other exit from `_run_supporting`, and the same survivor its ordinary path stops
        # the sweep over. Here the sweep is already leaving, so there is nothing to stop -- but a
        # cancellation that reports a clean exit over a process still holding the device is how
        # the NEXT run gets measured against it (gh-ocannl-760 review). The survivor is stood in
        # for as elsewhere: nothing outlives SIGKILL but a process stuck in the kernel.
        pidfile = self.dir / "supporting.pid"
        kill_the_group_on_cleanup(self, pidfile)
        command = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )

        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            signal.setitimer(signal.ITIMER_REAL, 1.0)
            with self.assertRaises(cell_group.CleanupFailed) as raised:
                orchestrate.run_supporting(command)

        self.assertIn("SURVIVED SIGKILL", str(raised.exception))

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_successful_cell_whose_leftovers_were_killed_still_handles_the_cache(self):
        # The row stands -- the cell ran and printed its result -- but a member of its group was
        # killed on the way out, possibly mid-write, and the cache it shares with every later beam
        # cell is in the state a kill leaves. It is the cache that is at issue, not the row.
        seen = []
        pidfile = self.dir / "successful_leftover.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import json, subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "print(json.dumps({'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))\n",
            pidfile,
        )

        result, note, _ = self.run_cell(
            "successful with leftovers",
            cell,
            timeout=60,
            on_incomplete=lambda killed: seen.append(killed) or "cache quarantined",
        )

        self.assertIsNotNone(result)  # the row is good; the cache is what needed handling
        self.assertIsNone(note)
        self.assertEqual(seen, [True])
        self.assertTrue(self.wait_gone(int(pidfile.read_text())))

    def test_a_quarantine_that_cannot_finish_puts_back_what_it_moved(self):
        # A half-moved family is worse than an unmoved one: a database separated from its WAL
        # opens without the writes it holds, and the cache left behind gains a stale sidecar the
        # next tinygrad process reads against a database it never belonged to.
        db = self.dir / "cache.db"
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
            path.write_bytes(b"live cache " + path.name.encode())
        real_replace = orchestrate.os.replace
        calls = []

        def replace_then_fail(src, dst):
            calls.append(src)
            if len(calls) == 2:  # the -wal move, after the database has already moved
                raise OSError(13, "Permission denied")
            return real_replace(src, dst)

        with unittest.mock.patch.object(orchestrate.os, "replace", replace_then_fail):
            note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})

        self.assertIn("CACHE AT RISK", note)
        self.assertIn("could not quarantine", note)
        # Everything is back where the next cell expects it, and nothing is left half-quarantined.
        self.assertEqual(db.read_bytes(), b"live cache cache.db")
        self.assertEqual(Path(f"{db}-wal").read_bytes(), b"live cache cache.db-wal")
        self.assertEqual(sorted(p.name for p in self.dir.glob("*.wedged-*")), [])

    @unittest.skipUnless(os.name == "posix", "process groups are a POSIX notion here")
    def test_a_supporting_command_leaves_no_descendants_behind(self):
        # `communicate` returned because the LEADER exited, which says nothing about a build's
        # compiler worker or a probe's framework helper -- and those hold the GPU while the sweep
        # measures against them (gh-ocannl-760 review).
        pidfile = self.dir / "supporting_leftover.pid"
        kill_the_group_on_cleanup(self, pidfile)
        build = self.python(
            "import subprocess, sys\n"
            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            + publish_pid("sys.argv[1]", "kid.pid")
            + "sys.exit(0)\n",
            pidfile,
        )

        done = orchestrate.run_supporting(build, capture_output=True)

        self.assertEqual(done.returncode, 0)
        worker = int(pidfile.read_text())
        self.assertTrue(self.wait_gone(worker), f"pid {worker} outlived its supporting command")

    def test_a_supporting_survivor_stops_the_sweep_before_it_dispatches(self):
        # Where a cell's survivor fails that cell, a survivor from the BUILD or a device probe is
        # shared by every row the sweep is about to produce -- there is no subset of the results
        # to disbelieve, and a cap cannot bound damage that is already in all of them. So this one
        # stops the sweep. (The survivor is stood in for, as elsewhere: nothing outlives SIGKILL
        # except a process stuck in the kernel.)
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            with self.assertRaises(SystemExit) as raised:
                orchestrate.run_supporting(self.python("pass"), capture_output=True)

        self.assertIn("SURVIVED SIGKILL", str(raised.exception))
        self.assertIn("nothing is dispatched", str(raised.exception))

    def test_a_fractional_cap_is_reported_as_the_cap_it_was(self):
        # `{:.0f}` rounded a sub-second cap to `TIMED OUT after 0s`, one clause before the text
        # saying that zero disables the cap.
        pidfile = self.dir / "briefly.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )
        _, note, _ = self.run_cell("briefly capped", cell, timeout=0.25)

        self.assertIn("TIMED OUT after 0.25s", note)

    def test_the_leftover_probe_itself_runs_deferred(self):
        # A cancellation landing on the probe -- or between it reading the group as alive and the
        # kill starting -- would leave exactly the worker the probe just found, in a session
        # nothing else will reach (gh-ocannl-760 review). So the property is that the probe and
        # its cleanup are inside ONE deferral window, which is what this reads: the depth seen by
        # the probe itself.
        depth_at_probe = []

        def probe(_proc):
            depth_at_probe.append(orchestrate._cancellation.depth)
            return cell_group.GONE

        cell = self.python(
            "import json; print(json.dumps("
            "{'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )
        with unittest.mock.patch.object(orchestrate, "_group_observation", probe):
            result, note, _ = self.run_cell("probed", cell, timeout=60)

        self.assertIsNone(note)
        self.assertIsNotNone(result)
        self.assertTrue(depth_at_probe, "the ordinary path never probed for leftovers")
        self.assertTrue(
            all(depth > 0 for depth in depth_at_probe),
            f"the leftover probe ran with cancellations undeferred: {depth_at_probe}",
        )

    @unittest.skipUnless(os.name == "posix", "the interrupt path needs POSIX signals")
    def test_an_interrupted_cell_preserves_its_cleanup_failure(self):
        # The interrupt branch exits rather than records, so its print is the operator's only
        # chance to hear that something still holds the device -- while the cancellation's own
        # message says the cell was killed, and the retry they are about to start would be
        # measured against the survivor. The survivor is stood in for, as elsewhere.
        pidfile = self.dir / "interrupted-survivor.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )

        def handler(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGALRM, handler)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)

        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_KILL_GRACE_S", 0.2))
            stack.enter_context(
                unittest.mock.patch.object(
                    orchestrate,
                    "_group_observation",
                    lambda _proc, _allow_zombie_gone=False: cell_group.SURVIVORS,
                )
            )
            with self.assertRaises(cell_group.CleanupFailed) as raised:
                orchestrate.run_cell("interrupted over a survivor", cell)

        self.assertIn("SURVIVED SIGKILL", str(raised.exception))

    def test_the_cache_is_quarantined_even_if_the_cell_log_cannot_be_written(self):
        # The kill is what tore the cache, so undoing it must not sit behind fallible code: the
        # optional cell log writes to an operator-supplied directory, which can be unwritable or
        # full, and losing the sweep to that would leave the partial cache.db in place AND lose
        # the failure record.
        called = []
        pidfile = self.dir / "broken-log.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n" + publish_pid("sys.argv[1]", "os.getpid()") + "time.sleep(300)\n",
            pidfile,
        )
        with unittest.mock.patch.object(
            orchestrate, "CELL_LOG_DIR", Path("/dev/null/not-a-directory")
        ):
            result, note, _ = self.run_cell(
                "wedged with a broken log dir",
                cell,
                timeout=1.0,
                on_incomplete=lambda killed: called.append(("killed", killed))
                or "quarantined the cache",
            )

        self.assertIsNone(result)
        # ... and told that this cell was KILLED, which is what decides whether the cache may
        # be moved aside at all.
        self.assertEqual(called, [("killed", True)])
        self.assertIn("quarantined the cache", note)

    def test_a_killed_cell_leaves_a_log_of_what_it_printed(self):
        # gh-ocannl-1061: three tuned search passes over a 7200 s cap left 0-byte cell logs, so a
        # search's progress up to the kill was unreadable. The log is written as the cell runs;
        # what the cell printed before the cap is on disk after the kill.
        logs = self.dir / "cells"
        pidfile = self.dir / "logged.pid"
        kill_the_group_on_cleanup(self, pidfile)
        cell = self.python(
            "import sys, time\n"
            "print('autotune-progress: event=arm_start arm=\"A\"', flush=True)\n"
            "sys.stderr.write('stderr evidence\\n'); sys.stderr.flush()\n"
            + publish_pid("sys.argv[1]", "os.getpid()")
            + "time.sleep(300)\n",
            pidfile,
        )
        with unittest.mock.patch.object(orchestrate, "CELL_LOG_DIR", logs):
            result, note, _ = self.run_cell("gpt2 ocannl/cc/tuned (search pass)", cell, timeout=2.0)

        self.assertIsNone(result)
        self.assertIn("TIMED OUT", note)
        log = (logs / orchestrate.cell_log_name("gpt2 ocannl/cc/tuned (search pass)")).read_text()
        self.assertIn("event=arm_start", log)
        self.assertIn("stderr evidence", log)

    def test_a_cell_log_name_stays_within_the_file_system_limit(self):
        # Review round 1: a non-ASCII fixture name reaches the label as `\uXXXX` escapes, six bytes
        # a character, and a long one past 255 bytes would lose the log to a temporary file.
        short = orchestrate.cell_log_name("gpt2 ocannl/cc/tuned (search pass)")
        self.assertTrue(short.startswith("gpt2_ocannl_cc_tuned__search_pass_-"), short)
        # Review round 2: sanitizing is lossy and a file system may fold case, so labels the
        # sanitizer or the file system would merge still get files of their own.
        merged = [
            orchestrate.cell_log_name(f"{orchestrate.display_value(stem)} ocannl/cc/default")
            for stem in ("_a_b_", "a b", "A b")
        ]
        self.assertEqual(len({name.lower() for name in merged}), 3, merged)
        stem = orchestrate.display_value("\u5b57" * 40)
        names = {
            orchestrate.cell_log_name(f"{stem} ocannl/cc/{variant}")
            for variant in ("default", "tuned")
        }
        self.assertEqual(len(names), 2, names)
        for name in names | {orchestrate.cell_log_name("\u5b57" * 200)}:
            with self.subTest(name=name[:40]):
                self.assertLessEqual(len(name.encode()), orchestrate.CELL_LOG_NAME_MAX)
                self.assertTrue(name.endswith(".log"))
        # And the bounded name is one the file system takes (POSIX: a Windows temp directory plus
        # 200 bytes can pass MAX_PATH, which is the path's limit, not the component's).
        if os.name == "posix":
            path = self.dir / orchestrate.cell_log_name(f"{stem} ocannl/cc/default")
            path.write_text("log")
            self.assertEqual(path.read_text(), "log")

    def test_the_cell_log_grows_while_the_cell_runs(self):
        # Streaming, not written at the end: the cell reads its own log back mid-run and reports
        # whether its first line was already there.
        logs = self.dir / "cells"
        path = logs / orchestrate.cell_log_name("streaming")
        cell = self.python(
            "import json, sys, time\n"
            "print('first line', flush=True)\n"
            "time.sleep(0.2)\n"
            "seen = 'first line' in open(sys.argv[1]).read()\n"
            "print(json.dumps({'workload': 'w', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5,"
            " 'seen_mid_run': seen}))\n",
            path,
        )
        with unittest.mock.patch.object(orchestrate, "CELL_LOG_DIR", logs):
            result, note, _ = self.run_cell("streaming", cell, timeout=60)

        self.assertIsNone(note)
        self.assertTrue(result["seen_mid_run"], "the log was not on disk while the cell ran")
        self.assertIn("first line", path.read_text())

    def test_without_a_log_dir_the_output_is_still_read_and_then_removed(self):
        made = []
        real_mkstemp = tempfile.mkstemp

        def recording_mkstemp(*args, **kwargs):
            fd, name = real_mkstemp(*args, dir=self.dir, **kwargs)
            made.append(Path(name))
            return fd, name

        cell = self.python(
            "import json; print('chatter'); print(json.dumps("
            "{'workload': 'tmp', 'step_ms': {'p50': 1.0}, 'compile_s': 0.5}))"
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(orchestrate, "CELL_LOG_DIR", None))
            stack.enter_context(
                unittest.mock.patch.object(orchestrate.tempfile, "mkstemp", recording_mkstemp)
            )
            result, note, _ = self.run_cell("no log dir", cell, timeout=60)

        self.assertIsNone(note)
        self.assertEqual(result["workload"], "tmp")
        self.assertEqual(len(made), 1)
        self.assertFalse(made[0].exists(), "the temporary cell output outlived the cell")

    def test_a_killed_beam_cell_quarantines_the_cache_it_was_writing(self):
        # The contaminated-cache consequence recorded on the issue's HIP leg: the search writes
        # its winners into one sqlite file as it goes, so a kill leaves a partial cache that the
        # next run neither replays nor searches past -- while `searched` reports one of the two.
        db = self.dir / "cache.db"
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
            path.write_bytes(b"half a beam search")

        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)})

        self.assertFalse(db.exists())
        moved = sorted(p.name for p in self.dir.glob("cache.db.wedged-*"))
        self.assertEqual(len(moved), 3, moved)
        # The sidecars move UNDER the quarantined database's name, because that is the only place
        # sqlite looks for them: `cache.db-wal.wedged-<stamp>` beside `cache.db.wedged-<stamp>` is
        # a database that opens without the writes the killed search never checkpointed -- the
        # evidence the move exists to keep, dropped silently (gh-ocannl-760 review).
        base = next(name for name in moved if not name.endswith(("-wal", "-shm")))
        self.assertEqual(moved, sorted([base, f"{base}-wal", f"{base}-shm"]))
        self.assertIn("quarantined", note)
        self.assertIn("searched", note)

    def test_declining_the_quarantine_still_names_the_risk(self):
        db = self.dir / "cache.db"
        db.write_bytes(b"half a beam search")

        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(db)}, enabled=False)

        self.assertTrue(db.exists())
        self.assertIn("CACHE AT RISK", note)
        self.assertIn("--no-cache-quarantine", note)

    def test_a_cache_that_was_never_written_is_not_invented(self):
        note = orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(self.dir / "absent.db")})

        self.assertIn("nothing to quarantine", note)

    def test_the_ocannl_note_describes_that_cache_rather_than_tinygrads(self):
        # The two caches differ where it matters: OCANNL commits entries atomically, so the
        # honest statement is about the retry's provenance, not about a torn file.
        note = orchestrate.ocannl_cache_note()

        self.assertIn("autotune_cache", note)
        # And what it must NOT say is that the retry replays: a kill lands mid-search, so the
        # retry replays the finished arms and searches the rest, and `search_provenance` reads
        # `searched` -- true whenever any arm searched. The mixed pass reports SEARCHED, and a
        # note promising REPLAY would misdescribe exactly the case it exists for.
        self.assertIn("SEARCHED", note)
        self.assertEqual(orchestrate.search_provenance({"searched": True}), "SEARCHED")

    def test_a_failed_cell_is_named_in_the_report(self):
        # A failure is invisible in every table above it -- an unrun cell and a wedged one read
        # identically once the report outlives the run log.
        out = Path(tempfile.mkdtemp())
        failures = [("mlp_small tinygrad/CUDA/beam", "TIMED OUT after 1800s; quarantined ...")]
        cells = [cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])]
        orchestrate.parity_check(cells)

        orchestrate.report(cells, out, (), failures)

        text = (out / "report.md").read_text()
        self.assertIn("Runner failures", text)
        self.assertIn("mlp_small tinygrad/CUDA/beam", text)
        self.assertIn("TIMED OUT", text)

    def test_a_run_with_no_failures_says_nothing_about_them(self):
        out = Path(tempfile.mkdtemp())
        cells = [cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])]
        orchestrate.parity_check(cells)

        orchestrate.report(cells, out)

        self.assertNotIn("Runner failures", (out / "report.md").read_text())


class RegimeTest(unittest.TestCase):
    """gh-ocannl-719: the numerics regime is a third axis over the cells, with its own envelope."""

    def test_the_approximate_envelope_is_looser_than_every_exact_one(self):
        # The literal is the pin. Looser than PARITY_TOL and every precision envelope, so an
        # approximate cell of any precision is gated at it; tighter than the loss-movement floor.
        self.assertEqual(orchestrate.PARITY_TOL_APPROX, 1e-2)
        for precision in ("f32", "bf16", "f16", "f16-gated8"):
            self.assertEqual(orchestrate.parity_tol(precision, "approximate"), 1e-2)
            self.assertLess(orchestrate.parity_tol(precision), 1e-2)
        self.assertGreater(orchestrate.PARITY_TOL_APPROX, orchestrate.LOSS_MOVE_MIN_REL)
        # The exact envelopes are untouched by the regime's existence.
        self.assertEqual(orchestrate.parity_tol("f32"), orchestrate.PARITY_TOL)
        self.assertEqual(orchestrate.parity_tol("f32", "exact"), orchestrate.PARITY_TOL)

    def test_the_reference_is_the_exact_torch_cell_even_when_an_approximate_one_ran(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        counterpart = result("pytorch", "cpu", "eager", [2.3126, 2.3110, 2.3100])
        counterpart["regime"] = "approximate"
        candidate = result("ocannl", "cuda", "tuned", [2.3030, 2.3014, 2.3004])
        candidate["regime"] = "approximate"

        orchestrate.parity_check([counterpart, ref, candidate])

        self.assertEqual(ref["parity"], "REF")
        # The approximate torch cell is a counterpart gated like any other row, not a reference:
        # its 4e-3 drift passes the approximate envelope and not the exact one -- and that second
        # verdict is reported rather than assumed.
        self.assertEqual(counterpart["parity"], "PASS")
        self.assertFalse(counterpart["parity_exact_envelope"])
        self.assertEqual(candidate["parity"], "PASS")
        self.assertTrue(candidate["parity_exact_envelope"])

    def test_an_approximate_row_beyond_its_envelope_fails(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        drifted = result("ocannl", "cuda", "tuned", [2.3300, 2.3280, 2.3260])
        drifted["regime"] = "approximate"

        orchestrate.parity_check([ref, drifted])

        self.assertEqual(drifted["parity"], "FAIL")
        self.assertFalse(drifted["parity_exact_envelope"])

    def test_exact_rows_carry_no_exact_envelope_verdict(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        exact = result("ocannl", "cc", "default", [2.3026, 2.3010, 2.3001])

        orchestrate.parity_check([ref, exact])

        self.assertEqual(exact["parity"], "PASS")
        self.assertNotIn("parity_exact_envelope", exact)

    def test_a_sweep_with_no_arguments_runs_the_exact_regime_only(self):
        args = orchestrate.build_arg_parser().parse_args([])

        self.assertEqual(args.profile, ["exact"])

    def test_a_bare_profile_flag_is_refused_rather_than_an_empty_matrix(self):
        # `--profile` with no value would otherwise parse to [] and skip every OCANNL cell,
        # publishing an empty report with a clean exit (Codex P2 on PR #661).
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                orchestrate.build_arg_parser().parse_args(["--profile"])

    def test_the_exact_envelope_verdict_is_about_drift_alone(self):
        # A stationary approximate row within the exact envelope's drift: FAIL for not moving,
        # and "within exact envelope" -- the two facts are reported separately, so the report
        # never attributes a stationary loss to excessive drift (Codex P2 on PR #661).
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        flat = result("ocannl", "cc", "default", [2.3026, 2.3026, 2.3026])
        flat["regime"] = "approximate"

        orchestrate.parity_check([ref, flat])

        self.assertEqual(flat["parity"], "FAIL")
        self.assertFalse(flat["parity_loss_moved"])
        self.assertTrue(flat["parity_exact_envelope"])

    def test_a_mismatched_row_is_shouted_in_the_report_not_published_under_its_label(self):
        # The exact cell inherited OCANNL_PROFILE=approximate: the sweep fails on it, but
        # report.md is written before the failure exits, so the row must say so where its
        # number is read (Codex P1 on PR #661).
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        leaked = cell("ocannl", "cc", "default", [2.3026, 2.3011, 2.3001])
        leaked["profile"] = "approximate"
        orchestrate.parity_check([ref, leaked])
        self.assertEqual(orchestrate.regime_check([ref, leaked]), [leaked])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, leaked], out)
            text = (out / "report.md").read_text()
            rows = [
                strict_loads(line) for line in (out / "results.jsonl").read_text().splitlines()
            ]

        self.assertIn("| regime |", text)
        row = [line for line in text.splitlines() if line.startswith("| ocannl")][0]
        self.assertIn("**REGIME MISMATCH** (dispatched exact; ran approximate)", row)
        self.assertNotIn("| exact |", row)
        self.assertEqual(rows[1]["regime_mismatch"], "approximate")

    def test_both_regimes_can_share_one_sweep_and_an_unknown_one_is_refused(self):
        args = orchestrate.build_arg_parser().parse_args(["--profile", "exact", "approximate"])

        self.assertEqual(args.profile, ["exact", "approximate"])
        # `reproducible` is an OCANNL profile but not a benchmark regime: it has no envelope.
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                orchestrate.build_arg_parser().parse_args(["--profile", "reproducible"])

    def test_the_regime_reaches_each_runner_as_its_own_flag(self):
        # OCANNL: the profile IS the regime, and the exact regime passes no profile at all.
        self.assertEqual(orchestrate.ocannl_regime_args("exact"), [])
        self.assertEqual(
            orchestrate.ocannl_regime_args("approximate"), ["--ocannl_profile=approximate"]
        )
        # gh-ocannl-1061: a tuned cell writes the tuner's progress lines; no other variant tunes.
        self.assertEqual(
            orchestrate.ocannl_variant_args("tuned"), ["--ocannl_autotune_progress=true"]
        )
        for variant in ("default", "materialized"):
            self.assertEqual(orchestrate.ocannl_variant_args(variant), [])
        self.assertEqual(orchestrate.torch_regime_args("exact"), [])
        self.assertEqual(orchestrate.torch_regime_args("approximate"), ["--regime", "approximate"])
        # Exact labels are unchanged; a non-exact cell says so in its label.
        self.assertEqual(orchestrate.regime_label("exact"), "")
        self.assertEqual(orchestrate.regime_label("approximate"), " [approximate]")
        # tinygrad's one cell stands in the approximate regime once the sweep has one.
        self.assertEqual(orchestrate.tinygrad_regime(["exact"]), "exact")
        self.assertEqual(orchestrate.tinygrad_regime(["exact", "approximate"]), "approximate")

    def test_a_row_whose_runner_ran_the_other_regime_is_a_gate_failure(self):
        # An ambient OCANNL_PROFILE=approximate reaching the exact cells, or the reverse: the
        # parity gate cannot see it (an exact trajectory passes the approximate envelope too).
        mislabelled = result("ocannl", "cc", "default", [1.0])
        mislabelled["profile"] = "approximate"
        honest = result("ocannl", "cc", "default", [1.0])
        honest.update(regime="approximate", profile="approximate")
        # A reproducible or performance profile is still the exact regime.
        reproducible = result("ocannl", "cc", "tuned", [1.0])
        reproducible["profile"] = "reproducible"
        torch_wrong = result("pytorch", "cuda", "eager", [1.0])
        torch_wrong.update(regime="approximate", runner_regime="exact")
        # A runner predating the field, and tinygrad (whose regime is the sweep's decision), are
        # not checked.
        older = result("ocannl", "cc", "default", [1.0])
        older["regime"] = "approximate"
        tiny = result("tinygrad", "CPU", "jit", [1.0])
        tiny.update(regime="approximate", regime_settings="tinygrad defaults")

        mismatched = orchestrate.regime_check(
            [mislabelled, honest, reproducible, torch_wrong, older, tiny]
        )

        self.assertEqual(mismatched, [mislabelled, torch_wrong])
        self.assertEqual(mislabelled["regime_mismatch"], "approximate")
        self.assertEqual(torch_wrong["regime_mismatch"], "exact")

    def test_the_regime_is_the_resolution_of_the_profiles_keys_not_its_name(self):
        # Codex P1 on PR #661: OCANNL_TF32_MATMULS=true in the sweep's environment reaches an
        # exact cell with no profile picked, so the profile name says `exact` while the
        # arithmetic is approximate. The runner reports where each approximate-payload key
        # resolved from; an exact row owns only defaults, an approximate row only the profile.
        def knobs(**sources):
            return {
                key: ({"source": "default"} if src is None else {"value": val, "source": src})
                for key, (val, src) in sources.items()
            }

        clean = result("ocannl", "cc", "default", [1.0])
        clean.update(profile=None, regime_knobs=knobs(tf32_matmuls=(None, None)))
        leaked = result("ocannl", "cuda", "default", [1.0])
        leaked.update(
            profile=None,
            regime_knobs=knobs(
                tf32_matmuls=("true", "environment"), cc_backend_fast_math=(None, None)
            ),
        )
        honest = result("ocannl", "cuda", "tuned", [1.0])
        honest.update(
            regime="approximate",
            profile="approximate",
            regime_knobs=knobs(tf32_matmuls=("true", "profile 'approximate' via the commandline")),
        )
        overridden = result("ocannl", "cuda", "tuned", [1.0])
        overridden.update(
            regime="approximate",
            profile="approximate",
            regime_knobs=knobs(
                tf32_matmuls=("false", "commandline"),
                cc_backend_fast_math=("true", "profile 'approximate' via the commandline"),
            ),
        )
        # Another profile pins these keys to exact VALUES, but from the profile: not an
        # exact-regime sweep, and the row says which key came from where.
        reproducible = result("ocannl", "cc", "tuned", [1.0])
        reproducible.update(
            profile="reproducible",
            regime_knobs=knobs(tf32_matmuls=("false", "profile 'reproducible' via the environment")),
        )
        older = result("ocannl", "cc", "default", [1.0])
        older["profile"] = None

        mismatched = orchestrate.regime_check(
            [clean, leaked, honest, overridden, reproducible, older]
        )

        self.assertEqual(mismatched, [leaked, overridden, reproducible])
        self.assertEqual(leaked["regime_mismatch"], "exact but tf32_matmuls=true (environment)")
        self.assertEqual(
            overridden["regime_mismatch"], "approximate but tf32_matmuls=false (commandline)"
        )
        self.assertIn("profile 'reproducible'", reproducible["regime_mismatch"])
        self.assertIsNone(orchestrate.regime_knob_leaks(older))

    def test_the_report_names_the_regime_and_the_exact_envelope_verdict(self):
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        exact = cell("ocannl", "cuda", "tuned", [2.3026, 2.3011, 2.3001], p50=2.0)
        approx = cell("ocannl", "cuda", "tuned", [2.3126, 2.3110, 2.3100], p50=1.0)
        approx["regime"] = "approximate"
        orchestrate.parity_check([ref, exact, approx])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, exact, approx], out)
            text = (out / "report.md").read_text()

        self.assertIn("| regime |", text)
        self.assertIn("the approximate regime 0.01", text)
        table = [line for line in text.splitlines() if line.startswith("| ocannl")]
        # The exact row comes first within the precision though the approximate one is faster:
        # an approximate number is read against the exact one it relaxes.
        self.assertIn("| exact |", table[0])
        self.assertIn("| approximate |", table[1])
        self.assertIn("PASS (4.3e-03, beyond exact envelope)", table[1])
        self.assertNotIn("exact envelope", table[0])
        # The exact rows name their regime too once the column exists.
        self.assertIn("| exact |", [line for line in text.splitlines() if line.startswith("| pytorch")][0])

    def test_an_exact_only_report_has_no_regime_column(self):
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        exact = cell("ocannl", "cc", "default", [2.3026, 2.3011, 2.3001])
        orchestrate.parity_check([ref, exact])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, exact], out)
            text = (out / "report.md").read_text()

        self.assertNotIn("| regime |", text)
        self.assertNotIn("exact envelope", text)
        self.assertIn("| PASS (", text)


# The fixture the real-runner leg of TensorPrecisionTest measures: a two-layer mlp small enough
# that each runner process is torch's import and nothing else. Every tensor is stored at the dtype
# named on the commandline, so a row that reports float32 for a bfloat16 fixture echoed something
# instead of reading its tensors back.
PRECISION_PROBE_FIXTURE = """
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

path, dtype = sys.argv[1], getattr(torch, sys.argv[2])
g = torch.Generator().manual_seed(0)
n, d_in, hidden, d_out = 8, 4, 6, 3
tensors = {
    "x": torch.randn(n, d_in, generator=g),
    "y": F.one_hot(torch.arange(n) % d_out, d_out).float(),
    "w1": torch.randn(hidden, d_in, generator=g) * 0.5,
    "b1": torch.zeros(hidden),
    "w2": torch.randn(d_out, hidden, generator=g) * 0.5,
    "b2": torch.zeros(d_out),
}
meta = {
    "model": "mlp", "name": "precision_probe", "n_layers": "2", "batch_size": "4", "lr": "0.1",
    "parity_steps": "3", "warmup_steps": "1", "timed_steps": "2",
}
save_file({k: v.to(dtype).contiguous() for k, v in tensors.items()}, path, metadata=meta)
"""

# The torch probe fixture's twin for the tinygrad runner, which loads through safetensors.numpy
# (so no bfloat16: numpy has no such dtype).
TINYGRAD_PROBE_FIXTURE = """
import sys
import numpy as np
from safetensors.numpy import save_file

path, dtype = sys.argv[1], np.dtype(sys.argv[2])
rng = np.random.default_rng(0)
n, d_in, hidden, d_out = 8, 4, 6, 3
tensors = {
    "x": rng.standard_normal((n, d_in)),
    "y": np.eye(d_out)[np.arange(n) % d_out],
    "w1": rng.standard_normal((hidden, d_in)) * 0.5,
    "b1": np.zeros(hidden),
    "w2": rng.standard_normal((d_out, hidden)) * 0.5,
    "b2": np.zeros(d_out),
}
meta = {
    "model": "mlp", "name": "precision_probe", "n_layers": "2", "batch_size": "4", "lr": "0.1",
    "parity_steps": "3", "warmup_steps": "1", "timed_steps": "2",
}
save_file({k: np.ascontiguousarray(v.astype(dtype)) for k, v in tensors.items()}, path, metadata=meta)
"""

# Runs the torch runner inside an autocast context the runner knows nothing about: the case a
# matmul-policy getter cannot see and the step's own autocast read must.
UNDER_AUTOCAST = """
import runpy
import sys
import torch

run_py, fixture = sys.argv[1], sys.argv[2]
sys.argv = [run_py, "--fixture", fixture]
with torch.autocast("cpu", dtype=torch.bfloat16):
    runpy.run_path(run_py, run_name="__main__")
"""


def checkout_venv_python():
    """The bench venv the torch and tinygrad legs run: `BENCH_VENV_PY` or this checkout's own
    `benchmarks/.venv`, never the box's shared venv (gh-ocannl-1433). The dune rule running this
    file is sandboxed to what it declares, and a venv in the home directory is no declared input:
    found there, the legs would run torch on whichever box happens to have one."""
    return bench_venv.venv_python(HERE, shared=None)


def torch_venv_python():
    """`(interpreter, None)` if the bench venv can run the torch runner here, else `(None, why)`."""
    venv = checkout_venv_python()
    if not venv.exists():
        return None, f"no bench venv at {venv}"
    # numpy too: torch does not depend on it, but safetensors' torch save and load convert through
    # it, so a venv without it imports cleanly and then fails inside the runner.
    probe = subprocess.run(
        [str(venv), "-c", "import torch, safetensors.torch, numpy"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if probe.returncode != 0:
        return None, f"the bench venv at {venv} cannot import torch, safetensors and numpy: {probe.stderr}"
    return venv, None


def torch_venv_or_skip(test):
    """The torch venv's interpreter, else a skip -- or, under `BENCH_TORCH_REQUIRED=1`, a failure.

    Hosts without torch (the dune rule on CI's main legs, most dev checkouts) skip the torch legs.
    ci.yml's `torch-runner` job provisions a CPU torch venv and sets the variable, because there a
    missing torch is a broken provisioning step, and a skip would pass it off as green coverage.
    """
    venv, why = torch_venv_python()
    if venv is not None:
        return venv
    if os.environ.get("BENCH_TORCH_REQUIRED") == "1":
        test.fail(f"BENCH_TORCH_REQUIRED=1, but {why}")
    test.skipTest(f"{why} (set BENCH_VENV_PY to run this leg)")


class TorchVenvGateTest(unittest.TestCase):
    """The torch legs skip where torch is absent, and fail there under `BENCH_TORCH_REQUIRED=1`.

    The CI job that provisions torch reads its own coverage from that failure: if the variable
    stopped turning the skip into a red, a broken venv would pass as a green torch leg.
    """

    def gate(self, environ):
        missing = unittest.mock.patch(
            f"{__name__}.torch_venv_python", return_value=(None, "no bench venv at /nowhere")
        )
        with missing, unittest.mock.patch.dict(os.environ, environ, clear=True):
            torch_venv_or_skip(self)

    def test_a_host_without_torch_skips(self):
        with self.assertRaises(unittest.SkipTest) as raised:
            self.gate({})
        self.assertIn("/nowhere", str(raised.exception))

    def test_a_host_that_requires_torch_fails_instead(self):
        with self.assertRaises(self.failureException) as raised:
            self.gate({"BENCH_TORCH_REQUIRED": "1"})
        self.assertIn("/nowhere", str(raised.exception))

    def test_a_present_venv_is_returned(self):
        found = Path("/venv/bin/python")
        with unittest.mock.patch(f"{__name__}.torch_venv_python", return_value=(found, None)):
            self.assertEqual(torch_venv_or_skip(self), found)


class TensorPrecisionTest(unittest.TestCase):
    """gh-ocannl-1208: a torch row says what its tensors and autocast context WERE.

    `regime_settings` is the matmul policy; it cannot tell a bf16 weight or an autocast context
    from an f32 run. The runner reads the parameter and batch dtypes and the autocast state back
    from its own run, and the sweep fails a row whose label they contradict.
    """

    def test_the_effective_precision_is_the_floating_tensors_shared_label(self):
        f32 = bench_common.effective_precision(["float32"], {"input": "float32", "target": "float32"})
        self.assertEqual(f32, "f32")
        # Token ids carry no precision: the gpt fixture's int64 ids leave an f32 run f32.
        ids = bench_common.effective_precision(["float32"], {"input": "int64", "target": "float32"})
        self.assertEqual(ids, "f32")
        self.assertEqual(bench_common.effective_precision(["bfloat16"], {"input": "bfloat16"}), "bf16")
        self.assertEqual(
            bench_common.effective_precision(["bfloat16", "float32"], {"input": "float32"}),
            "mixed(bfloat16,float32)",
        )
        # A format with no report label is named, not mapped to the nearest one.
        self.assertEqual(
            bench_common.effective_precision(["float8_e4m3fn"], {}), "float8_e4m3fn"
        )
        self.assertIsNone(bench_common.effective_precision([], {"input": "int64"}))

    def test_the_fields_are_read_from_the_tensors_handed_in(self):
        def tensor(dtype):
            return types.SimpleNamespace(dtype=f"torch.{dtype}")

        fields = bench_common.tensor_precision_fields(
            [tensor("float32"), tensor("bfloat16"), tensor("float32")],
            {"input": tensor("int64"), "target": tensor("float32")},
            {"enabled": False, "dtype": None},
        )

        self.assertEqual(
            fields,
            {
                "param_dtypes": ["bfloat16", "float32"],
                "input_dtypes": {"input": "int64", "target": "float32"},
                "autocast": {"enabled": False, "dtype": None},
                "effective_precision": "mixed(bfloat16,float32)",
            },
        )

    def test_the_autocast_state_is_read_per_device_and_unknown_is_not_off(self):
        def torch_with(enabled, raises=None):
            def is_enabled(device_type):
                if raises:
                    raise raises
                return enabled.get(device_type, False)

            return types.SimpleNamespace(
                is_autocast_enabled=is_enabled,
                get_autocast_dtype=lambda device_type: "torch.bfloat16",
            )

        on_cpu = torch_with({"cpu": True})
        self.assertEqual(
            bench_common.torch_autocast_state(on_cpu, "cpu"),
            {"enabled": True, "dtype": "bfloat16"},
        )
        # Off reports no dtype: torch's getter answers its default dtype even when autocast is off.
        self.assertEqual(
            bench_common.torch_autocast_state(on_cpu, "mps"), {"enabled": False, "dtype": None}
        )
        # A torch predating the per-device getters, or a device autocast does not know: None.
        self.assertIsNone(bench_common.torch_autocast_state(types.SimpleNamespace(), "cpu"))
        self.assertIsNone(
            bench_common.torch_autocast_state(torch_with({}, raises=TypeError("no arg")), "cpu")
        )
        self.assertIsNone(
            bench_common.torch_autocast_state(torch_with({}, raises=RuntimeError("xla")), "xla")
        )

    def test_tinygrad_dtypes_are_named_as_torch_names_them(self):
        # tinygrad spells its dtypes in C (`float`, `half`, `__bf16`); the row names them as the
        # torch rows do, so one `effective_precision` and one label map read both.
        def dtype(name):
            return types.SimpleNamespace(name=name)

        dtypes = types.SimpleNamespace(
            float32=dtype("float"), float16=dtype("half"), bfloat16=dtype("__bf16"),
            float64=dtype("double"), int32=dtype("int"),
        )
        dtypes.default_float = dtypes.float16
        names = [
            bench_common.tinygrad_dtype_name(dtypes, getattr(dtypes, attr))
            for attr in ("float32", "float16", "bfloat16", "float64", "int32")
        ]
        self.assertEqual(names, ["float32", "float16", "bfloat16", "float64", "int"])

        def tensor(attr):
            return types.SimpleNamespace(dtype=getattr(dtypes, attr))

        fields = bench_common.tinygrad_precision_fields(
            dtypes, [tensor("float32"), tensor("float32")],
            {"input": tensor("int32"), "target": tensor("float32")},
        )
        self.assertEqual(
            fields,
            {
                "param_dtypes": ["float32"],
                "input_dtypes": {"input": "int", "target": "float32"},
                "default_float": "float16",
                "effective_precision": "f32",
            },
        )

    def test_a_tinygrad_or_ocannl_label_the_run_contradicts_is_a_mismatch(self):
        # gh-ocannl-1262: the two frameworks precision_check did not cover. A tinygrad row's
        # tensors and default float, and an OCANNL row's storage read-back, against the label.
        def tiny(effective="f32", default_float="float32", precision="f32"):
            row = cell("tinygrad", "CPU", "jit", [2.3, 2.2, 2.1], precision=precision)
            row.update(
                param_dtypes=["float32"], input_dtypes={"input": "float32", "target": "float32"},
                default_float=default_float, effective_precision=effective,
            )
            return row

        def ocannl(storage, precision="f32"):
            row = cell("ocannl", "cc", "default", [2.3, 2.2, 2.1], precision=precision)
            row["storage_precision"] = storage
            return row

        honest_tiny = tiny()
        ignored_tiny = tiny(precision="bf16")
        half_default = tiny(default_float="float16")
        honest_ocannl = ocannl("bf16", precision="bf16")
        gate_leg = ocannl("f16", precision="f16-gated4")
        ignored_ocannl = ocannl("f32", precision="bf16")
        older_ocannl = cell("ocannl", "cc", "default", [2.3, 2.2, 2.1], precision="bf16")

        mismatched = orchestrate.precision_check(
            [honest_tiny, ignored_tiny, half_default, honest_ocannl, gate_leg, ignored_ocannl,
             older_ocannl]
        )

        self.assertEqual(mismatched, [ignored_tiny, half_default, ignored_ocannl])
        # Labelled bf16, it ran f32 tensors under an f32 default: both say so.
        self.assertEqual(ignored_tiny["precision_mismatch"], "f32 tensors, default float float32")
        self.assertEqual(half_default["precision_mismatch"], "default float float16")
        self.assertEqual(ignored_ocannl["precision_mismatch"], "f32 storage")
        for row in (honest_tiny, honest_ocannl, gate_leg, older_ocannl):
            self.assertNotIn("precision_mismatch", row)

    def test_the_real_tinygrad_runner_reads_its_precision_back_from_a_cpu_run(self):
        """The tinygrad runner end to end on a tiny CPU fixture, where the bench venv has tinygrad:
        an f32 fixture, an f16 one (read back, not echoed), and the f32 fixture under a half
        DEFAULT_FLOAT. A requested-but-ignored precision is the first run stamped bf16."""
        venv = checkout_venv_python()
        probe = subprocess.run(
            [str(venv), "-c", "import tinygrad, safetensors.numpy, numpy"],
            capture_output=True, text=True, timeout=120,
        ) if venv.exists() else None
        if probe is None or probe.returncode != 0:
            self.skipTest(f"the bench venv at {venv} cannot import tinygrad, safetensors, numpy")
        run_py = HERE / "runners/tinygrad/run.py"

        def row(fixture, **env):
            done = subprocess.run(
                [str(venv), str(run_py), "--fixture", fixture, "--device", "CPU", "--jit", "0"],
                capture_output=True, text=True, timeout=600, env=dict(os.environ, **env),
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            return strict_loads(done.stdout.strip().splitlines()[-1])

        with tempfile.TemporaryDirectory() as tmp:
            fixtures = {}
            for dtype in ("float32", "float16"):
                fixtures[dtype] = str(Path(tmp) / f"probe_{dtype}.safetensors")
                subprocess.run(
                    [str(venv), "-c", TINYGRAD_PROBE_FIXTURE, fixtures[dtype], dtype],
                    check=True, timeout=300,
                )
            f32 = row(fixtures["float32"])
            f16 = row(fixtures["float16"])
            half_default = row(fixtures["float32"], DEFAULT_FLOAT="HALF")

        self.assertEqual(f32["param_dtypes"], ["float32"])
        self.assertEqual(f32["input_dtypes"], {"input": "float32", "target": "float32"})
        self.assertEqual(f32["default_float"], "float32")
        self.assertEqual(f32["effective_precision"], "f32")
        self.assertEqual(f16["param_dtypes"], ["float16"])
        self.assertEqual(f16["effective_precision"], "f16")
        self.assertEqual(half_default["default_float"], "float16")

        for r in (f32, f16, half_default):
            r["precision"] = "f32"  # as the sweep stamps every tinygrad cell
        ignored = dict(f32, precision="bf16")
        mismatched = orchestrate.precision_check([f32, ignored, f16, half_default])

        self.assertEqual(mismatched, [ignored, f16, half_default])
        self.assertEqual(ignored["precision_mismatch"], "f32 tensors, default float float32")
        self.assertEqual(f16["precision_mismatch"], "f16 tensors")
        self.assertIn("default float float16", half_default["precision_mismatch"])

    def torch_row(self, effective, precision="f32", autocast=None):
        row = cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1], precision=precision)
        row.update(
            param_dtypes=["float32"],
            input_dtypes={"input": "float32", "target": "float32"},
            autocast={"enabled": False, "dtype": None} if autocast is None else autocast,
            effective_precision=effective,
        )
        return row

    def test_a_label_the_run_contradicts_is_a_mismatch(self):
        honest = self.torch_row("f32")
        # A requested precision the runner did not apply: labelled bf16, tensors read back f32.
        ignored = self.torch_row("f32", precision="bf16")
        cast_on_load = self.torch_row("bf16")
        autocast = self.torch_row("f32", autocast={"enabled": True, "dtype": "float16"})
        # A gate leg computes in its base precision.
        gate_leg = self.torch_row("f16", precision="f16-static")
        # Unknown autocast is reported in the row, not a mismatch.
        unknown = self.torch_row("f32")
        unknown["autocast"] = None
        # Rows whose runner reports no effective precision -- OCANNL's, tinygrad's, a torch
        # runner predating the field -- are not checked.
        ocannl = cell("ocannl", "cc", "default", [2.3, 2.2, 2.1])
        older = cell("pytorch", "cuda", "eager", [2.3, 2.2, 2.1], precision="bf16")

        mismatched = orchestrate.precision_check(
            [honest, ignored, cast_on_load, autocast, gate_leg, unknown, ocannl, older]
        )

        self.assertEqual(mismatched, [ignored, cast_on_load, autocast])
        self.assertEqual(ignored["precision_mismatch"], "f32 tensors")
        self.assertEqual(cast_on_load["precision_mismatch"], "bf16 tensors")
        self.assertEqual(autocast["precision_mismatch"], "autocast to float16")
        for row in (honest, gate_leg, unknown, ocannl, older):
            self.assertNotIn("precision_mismatch", row)

    def test_the_report_shouts_a_mismatch_and_names_what_the_reference_computed_in(self):
        ref = self.torch_row("f32")
        cast = self.torch_row("bf16")
        cast["backend"] = "mps"
        rows = [ref, cast]
        orchestrate.parity_check(rows)
        orchestrate.precision_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            text = (out / "report.md").read_text()
            written = [
                strict_loads(line) for line in (out / "results.jsonl").read_text().splitlines()
            ]

        self.assertIn(
            "parity reference `pytorch/cpu/eager`: params float32; inputs input float32, "
            "target float32; autocast off",
            text,
        )
        mps = [line for line in text.splitlines() if line.startswith("| pytorch | mps")][0]
        self.assertIn("**PRECISION MISMATCH** (dispatched f32; ran bf16 tensors)", mps)
        self.assertEqual(written[1]["precision_mismatch"], "bf16 tensors")
        self.assertEqual(written[0]["param_dtypes"], ["float32"])
        self.assertEqual(written[0]["autocast"], {"enabled": False, "dtype": None})

    def test_a_reference_predating_the_fields_says_so(self):
        ref = cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])
        orchestrate.parity_check([ref])

        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref], Path(tmp))
            text = (Path(tmp) / "report.md").read_text()

        self.assertIn(
            "parity reference `pytorch/cpu/eager`: tensor precision not reported by its runner",
            text,
        )

    def test_the_real_runner_reads_its_precision_back_from_a_cpu_run(self):
        """The runner end to end on a tiny CPU fixture, where the bench venv has torch.

        Three runs: an f32 fixture (the reference's case), a bf16 fixture (read back, not echoed:
        nothing on the commandline differs), and the f32 fixture under an autocast context the
        runner never entered itself. A requested-but-ignored precision is the first run stamped
        bf16, as the sweep's override would stamp it.
        """
        venv = torch_venv_or_skip(self)
        run_py = HERE / "runners/pytorch/run.py"

        def row(*argv):
            done = subprocess.run(
                [str(venv), *argv], capture_output=True, text=True, timeout=300
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            return strict_loads(done.stdout.strip().splitlines()[-1])

        with tempfile.TemporaryDirectory() as tmp:
            fixtures = {}
            for dtype in ("float32", "bfloat16"):
                fixtures[dtype] = str(Path(tmp) / f"probe_{dtype}.safetensors")
                subprocess.run(
                    [str(venv), "-c", PRECISION_PROBE_FIXTURE, fixtures[dtype], dtype],
                    check=True,
                    timeout=300,
                )
            f32 = row(str(run_py), "--fixture", fixtures["float32"])
            bf16 = row(str(run_py), "--fixture", fixtures["bfloat16"])
            autocast = row("-c", UNDER_AUTOCAST, str(run_py), fixtures["float32"])

        self.assertEqual(f32["param_dtypes"], ["float32"])
        self.assertEqual(f32["input_dtypes"], {"input": "float32", "target": "float32"})
        self.assertEqual(f32["autocast"], {"enabled": False, "dtype": None})
        self.assertEqual(f32["effective_precision"], "f32")
        self.assertEqual(bf16["param_dtypes"], ["bfloat16"])
        self.assertEqual(bf16["effective_precision"], "bf16")
        self.assertEqual(autocast["autocast"], {"enabled": True, "dtype": "bfloat16"})
        self.assertEqual(autocast["effective_precision"], "f32")

        for r in (f32, bf16, autocast):
            r["precision"] = "f32"  # as the sweep stamps every torch cell
        ignored = dict(f32, precision="bf16")
        mismatched = orchestrate.precision_check([f32, ignored, bf16, autocast])

        self.assertEqual(mismatched, [ignored, bf16, autocast])
        self.assertEqual(ignored["precision_mismatch"], "f32 tensors")
        self.assertEqual(bf16["precision_mismatch"], "bf16 tensors")
        self.assertEqual(autocast["precision_mismatch"], "autocast to bfloat16")


class AmbientEnvTest(unittest.TestCase):
    """gh-ocannl-720: what the shell was already carrying is recorded, not guessed at.

    The regime gate (gh-ocannl-719) cross-checks the keys the `approximate` payload owns. Every
    other configuration key reaches the OCANNL cells through `cell_env`'s pass-through of
    `os.environ`, so the artifact has to say what that environment was -- for the keys nobody has
    enumerated as much as for the ones anybody has.
    """

    def test_the_sweep_records_every_ocannl_variable_it_inherited(self):
        # Values as well as names: `OCANNL_CC_VECTOR_BYTES=16` and `=64` are different
        # measurements, and a bare list of names cannot tell the two apart.
        ambient = orchestrate.ambient_ocannl_env(
            {
                "OCANNL_NARROW_COMPUTE_F32": "true",
                "PATH": "/usr/bin",
                "OCANNL_CC_VECTOR_BYTES": "64",
                "HOME": "/home/nobody",
            }
        )

        self.assertEqual(
            ambient,
            {"OCANNL_CC_VECTOR_BYTES": "64", "OCANNL_NARROW_COMPUTE_F32": "true"},
        )
        # Sorted, so two sweeps that inherited the same environment stamp the same mapping
        # whatever order the shell happened to export it in.
        self.assertEqual(
            list(ambient), ["OCANNL_CC_VECTOR_BYTES", "OCANNL_NARROW_COMPUTE_F32"]
        )

    def test_a_key_the_regime_gate_does_not_own_is_recorded_like_any_other(self):
        # The point of recording over enumerating: the rule is the `OCANNL_` prefix, not a list.
        # `regime_check` cross-checks the approximate payload's keys, and a key outside it --
        # `cc_backend_arch_flags` changes what the C backend compiles to, and the key added after
        # this test was written is by definition on nobody's list -- lands in the row all the same.
        self.assertEqual(
            orchestrate.ambient_ocannl_env({"OCANNL_CC_BACKEND_ARCH_FLAGS": "-mcpu=native"}),
            {"OCANNL_CC_BACKEND_ARCH_FLAGS": "-mcpu=native"},
        )

    def test_the_legitimate_variables_are_recorded_rather_than_stripped(self):
        # The sweep dispatches OCANNL_BACKEND itself and benchmarks/README.md documents running
        # under OCANNL_AUTOTUNE_LOG / OCANNL_AUTOTUNE_CACHE_DIR. None of them is refused or
        # filtered out -- a record that quietly omits what it considers normal is not a record.
        env = {
            "OCANNL_BACKEND": "metal",
            "OCANNL_AUTOTUNE_LOG": "1",
            "OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune",
        }

        self.assertEqual(orchestrate.ambient_ocannl_env(env), env)

    def test_only_the_ocannl_rows_carry_the_environment(self):
        ambient = {"OCANNL_NARROW_COMPUTE_F32": "true"}
        ours = result("ocannl", "cc", "default", [2.3026, 2.3010])
        theirs = result("pytorch", "cpu", "eager", [2.3026, 2.3010])

        orchestrate.stamp_ambient_env(ours, ambient)
        orchestrate.stamp_ambient_env(theirs, ambient)

        self.assertEqual(ours["ambient_ocannl_env"], ambient)
        # The torch runner reads none of these; a stamp on its row would invite reading its
        # number as though they had reached it.
        self.assertNotIn("ambient_ocannl_env", theirs)

    def test_each_row_records_its_own_copy(self):
        # One mapping shared by every row of a sweep is a row that changes after it was written.
        ambient = {"OCANNL_BACKEND": "cc"}
        row = orchestrate.stamp_ambient_env(
            result("ocannl", "cc", "default", [2.3026, 2.3010]), ambient
        )

        ambient["OCANNL_BACKEND"] = "metal"

        self.assertEqual(row["ambient_ocannl_env"], {"OCANNL_BACKEND": "cc"})

    def test_the_report_names_the_environment_its_numbers_were_taken_in(self):
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]),
                {"OCANNL_CC_VECTOR_BYTES": "64", "OCANNL_NARROW_COMPUTE_F32": "true"},
            ),
            cell("pytorch", "cpu", "eager", [2.3026, 2.3010]),
        ]
        orchestrate.parity_check(rows)

        text = self.report(rows)

        lines = text.splitlines()
        [index] = [
            i for i, line in enumerate(lines) if line.startswith("ambient OCANNL_* environment:")
        ]
        self.assertEqual(
            lines[index],
            'ambient OCANNL_* environment: {"OCANNL_CC_VECTOR_BYTES": "64", '
            '"OCANNL_NARROW_COMPUTE_F32": "true"}',
        )
        # Beside the measurement boxes, in the header: the configuration a number was taken under
        # is read with the box it was taken on, not hunted for in results.jsonl.
        before = [line for line in lines[:index] if line.strip()]
        self.assertTrue(before[-1].startswith("measurement boxes declared by"))

    def test_a_clean_shell_is_recorded_as_such(self):
        rows = [
            orchestrate.stamp_ambient_env(cell("ocannl", "cc", "default", [2.3026, 2.3010]), {})
        ]
        orchestrate.parity_check(rows)

        self.assertIn("ambient OCANNL_* environment: none\n", self.report(rows))

    def test_a_value_no_text_artifact_could_hold_is_kept_faithful_and_escaped_to_render(self):
        # An environment value is bytes: on POSIX `os.environ` hands a non-UTF-8 byte over as a
        # lone surrogate. The row keeps it verbatim -- `json.dumps` spells it `\udcff` under
        # `ensure_ascii`, so results.jsonl carries it losslessly -- while report.md's UTF-8 write
        # cannot hold it, which is why the rendering escapes and the record does not.
        ambient = orchestrate.ambient_ocannl_env(
            {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"}
        )

        self.assertEqual(ambient, {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"})
        self.assertIn("/tmp/tune-\\udcff", orchestrate.ambient_env_line([], ambient))

    def test_both_artifacts_of_a_sweep_under_such_a_value_are_written(self):
        # End to end, because the failure this prevents lands at the END of a sweep, after every
        # cell has been paid for: report.md's UTF-8 write is what cannot hold a lone surrogate.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]),
                orchestrate.ambient_ocannl_env(
                    {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"}
                ),
            )
        ]
        orchestrate.parity_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            text = (out / "report.md").read_text()
            [row] = [
                strict_loads(line)
                for line in (out / "results.jsonl").read_text().splitlines()
                if line
            ]

        self.assertIn("/tmp/tune-\\udcff", text)
        # The row is the record and keeps the value verbatim; only the rendering escapes.
        self.assertEqual(
            row["ambient_ocannl_env"], {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/tune-\udcff"}
        )

    def test_a_byte_and_the_characters_that_spell_it_stay_distinguishable(self):
        # Any escaping that leaves ordinary values untouched maps the 0xff byte and the four
        # literal characters `\xff` onto the same text, so two different cache directories would
        # dedupe into one environment and the record would say something false about both.
        byte = orchestrate.ambient_env_line(
            [], orchestrate.ambient_ocannl_env({"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/\udcff"})
        )
        spelling = orchestrate.ambient_env_line(
            [], orchestrate.ambient_ocannl_env({"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/\\xff"})
        )

        self.assertNotEqual(byte, spelling)

    def test_a_value_cannot_forge_a_second_header_line(self):
        # Environment values may contain newlines, and an interpolated one would open what reads
        # as a second `ambient OCANNL_* environment:` line -- the header contradicting itself
        # about the configuration its numbers were taken under.
        line = orchestrate.ambient_env_line(
            [],
            orchestrate.ambient_ocannl_env(
                {"OCANNL_AUTOTUNE_CACHE_DIR": "/tmp/a\nambient OCANNL_* environment: none"}
            ),
        )

        self.assertEqual(len(line.splitlines()), 1)
        self.assertIn("\\n", line)

    def test_the_report_says_the_environment_even_when_every_cell_failed(self):
        # The environment is a fact about the SWEEP, not about a row: an ambient setting the
        # cells cannot run under is exactly what makes every OCANNL cell fail, and a report that
        # then said `not recorded` would withhold the record at the moment it explains the run.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(
                    [],
                    out,
                    failures=[("mlp_wide ocannl/cc/default", "no result line")],
                    ambient={"OCANNL_NARROW_COMPUTE_F32": "true"},
                )
            text = (out / "report.md").read_text()

        self.assertIn(
            'ambient OCANNL_* environment: {"OCANNL_NARROW_COMPUTE_F32": "true"}\n', text
        )

    def test_rows_that_disagree_about_the_environment_are_refused(self):
        # Two header lines could only say that two environments exist somewhere in the tables,
        # never which row was measured under which -- swapping the two stamps would leave the
        # report byte-identical. Refused like a mixture of measurement-box declarations, with
        # rendering them separately as the remedy.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            orchestrate.stamp_ambient_env(
                cell("ocannl", "metal", "default", [2.3026, 2.3010]),
                {"OCANNL_BACKEND": "metal"},
            ),
        ]

        with self.assertRaises(ValueError) as refused:
            orchestrate.recorded_ambient_env(rows)

        self.assertIn("separate reports", str(refused.exception))

    def test_a_recorded_environment_beside_an_unrecorded_one_is_refused_too(self):
        # `not recorded` is a reading like any other: pairing it with a recorded environment
        # leaves the same ambiguity, and lets the recorded one stand at the head of measurements
        # whose own environment is unrecoverable.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            cell("ocannl", "metal", "default", [2.3026, 2.3010]),
        ]

        with self.assertRaises(ValueError) as refused:
            orchestrate.recorded_ambient_env(rows)

        self.assertIn("not recorded", str(refused.exception))

    def test_the_sweeps_own_value_does_not_excuse_the_rows_from_agreeing(self):
        # The refusal is about what the artifacts SAY, so it cannot be gated on how the header
        # got its value: a supplied environment printed over rows stamped with another would
        # misattribute the measurements results.jsonl records correctly.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            orchestrate.stamp_ambient_env(
                cell("ocannl", "metal", "default", [2.3026, 2.3010]),
                {"OCANNL_BACKEND": "metal"},
            ),
        ]

        with self.assertRaises(ValueError):
            orchestrate.ambient_env_line(rows, {"OCANNL_BACKEND": "cc"})

    def test_a_supplied_environment_its_rows_contradict_is_refused(self):
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            )
        ]

        with self.assertRaises(ValueError) as refused:
            orchestrate.ambient_env_line(rows, {"OCANNL_BACKEND": "metal"})

        self.assertIn("disagrees with what its rows recorded", str(refused.exception))
        # The agreeing case is the sweep's own, and is not disturbed by the check.
        self.assertIn(
            '{"OCANNL_BACKEND": "cc"}',
            orchestrate.ambient_env_line(rows, {"OCANNL_BACKEND": "cc"}),
        )

    def test_a_refused_report_leaves_the_output_directory_as_it_found_it(self):
        # The refusal happens before anything is written, like the measurement-box one: an output
        # directory is persistent, so overwriting results.jsonl and then raising would pair fresh
        # raw rows with the report.md a previous sweep left there, describing other measurements.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            orchestrate.stamp_ambient_env(
                cell("ocannl", "metal", "default", [2.3026, 2.3010]),
                {"OCANNL_BACKEND": "metal"},
            ),
        ]
        orchestrate.parity_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "results.jsonl").write_text('{"the previous sweep": true}\n')
            (out / "report.md").write_text("# the previous sweep\n")
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):
                    orchestrate.report(rows, out)

            self.assertEqual(
                (out / "results.jsonl").read_text(), '{"the previous sweep": true}\n'
            )
            self.assertEqual((out / "report.md").read_text(), "# the previous sweep\n")

    def test_the_unstamped_frameworks_are_not_a_lost_record(self):
        # Only OCANNL rows are in that agreement: torch and tinygrad rows carry no stamp by
        # design, and counting them would refuse every ordinary mixed-framework sweep.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]), {"OCANNL_BACKEND": "cc"}
            ),
            cell("pytorch", "cpu", "eager", [2.3026, 2.3010]),
        ]
        orchestrate.parity_check(rows)

        text = self.report(rows)

        self.assertIn(
            'ambient OCANNL_* environment: {"OCANNL_BACKEND": "cc"}\n', text
        )
        self.assertNotIn("not recorded", text)

    def test_rows_predating_the_stamp_do_not_claim_a_clean_shell(self):
        # Nothing at report time can recover what the shell held while the numbers were taken, so
        # an unstamped row's silence stays silence rather than becoming evidence of `none`.
        rows = [cell("ocannl", "cc", "default", [2.3026, 2.3010])]
        orchestrate.parity_check(rows)

        text = self.report(rows)

        self.assertIn("ambient OCANNL_* environment: not recorded\n", text)
        self.assertNotIn("environment: none", text)

    def test_the_recorded_environment_survives_into_the_raw_rows(self):
        # The report line is a summary; the per-row stamp is what a later script reads, and it has
        # to be JSON like everything else results.jsonl carries.
        rows = [
            orchestrate.stamp_ambient_env(
                cell("ocannl", "cc", "default", [2.3026, 2.3010]),
                {"OCANNL_NARROW_COMPUTE_F32": "true"},
            )
        ]
        orchestrate.parity_check(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            [row] = [
                strict_loads(line)
                for line in (out / "results.jsonl").read_text().splitlines()
                if line
            ]

        self.assertEqual(row["ambient_ocannl_env"], {"OCANNL_NARROW_COMPUTE_F32": "true"})

    def report(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report(rows, out)
            return (out / "report.md").read_text()


class VenvInterpreterTest(unittest.TestCase):
    """One rule finds the benchmark venv's interpreter, and every driver takes it from there.

    Each measurement box shares one venv outside the checkout, reached through `BENCH_VENV_PY`,
    a `benchmarks/.venv` symlink, or -- in a fresh worktree, which has neither -- the shared venv
    itself (gh-ocannl-1433). A driver spelling its own `.venv/bin/python` honours only the
    symlink, and only on POSIX -- which `gh675_cells.py` did until it took `bench_venv`'s rule.
    Every leg names its own shared venv, so none reads the box's real `~/.venvs/ocannl-bench`.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Resolved: the script leg compares with the path its copy prints, whose HERE is resolved
        # (macOS's temporary directory is behind a /var -> /private/var link).
        root = Path(self.tmp.name).resolve()
        self.dir = root / "benchmarks"
        self.dir.mkdir()
        self.shared = root / "shared-venv"

    def make(self, venv, rel="bin/python"):
        (venv / rel).parent.mkdir(parents=True, exist_ok=True)
        (venv / rel).touch()
        return venv / rel

    def resolve(self, environ=None):
        return bench_venv.resolve(self.dir, {} if environ is None else environ, self.shared)

    def test_the_override_wins_over_both_venv_layouts(self):
        self.make(self.dir / ".venv", "Scripts/python.exe")
        self.make(self.shared)
        found = self.resolve({"BENCH_VENV_PY": "/shared/venv/bin/python"})
        self.assertEqual(found, (Path("/shared/venv/bin/python"), "BENCH_VENV_PY"))

    def test_an_empty_override_is_no_override(self):
        local = self.make(self.dir / ".venv")
        self.assertEqual(self.resolve({"BENCH_VENV_PY": ""}), (local, "benchmarks/.venv"))

    def test_the_windows_layout_is_taken_when_it_exists(self):
        windows = self.make(self.dir / ".venv", "Scripts/python.exe")
        self.assertEqual(self.resolve(), (windows, "benchmarks/.venv"))

    def test_the_checkout_venv_wins_over_the_shared_one(self):
        local = self.make(self.dir / ".venv")
        self.make(self.shared)
        self.assertEqual(self.resolve(), (local, "benchmarks/.venv"))

    def test_a_fresh_worktree_finds_the_shared_venv(self):
        # The issue's case: no BENCH_VENV_PY and no `.venv` link, the box's venv right there.
        shared = self.make(self.shared)
        self.assertEqual(self.resolve(), (shared, f"shared venv {self.shared}"))
        windows = self.make(self.shared, "Scripts/python.exe")
        self.assertEqual(self.resolve()[0], windows)

    def test_a_dangling_checkout_link_falls_through_to_the_shared_venv(self):
        try:
            (self.dir / ".venv").symlink_to(self.dir.parent / "no-such-venv")
        except OSError as e:  # Windows without the symlink privilege
            self.skipTest(f"cannot create a symlink here: {e}")
        shared = self.make(self.shared)
        self.assertEqual(self.resolve(), (shared, f"shared venv {self.shared}"))

    def test_the_posix_layout_is_the_fallback(self):
        # Not conditional on existence: an unprovisioned checkout names the interpreter it is
        # missing, so the spawn error says where the venv was expected.
        self.assertEqual(self.resolve(), (self.dir / ".venv/bin/python", "none"))
        self.assertEqual(bench_venv.venv_python(self.dir, {}, self.shared), self.dir / ".venv/bin/python")

    def test_no_shared_step_when_the_caller_leaves_it_out(self):
        self.make(self.shared)
        self.assertEqual(bench_venv.resolve(self.dir, {}, None), (self.dir / ".venv/bin/python", "none"))

    def test_the_default_shared_venv_is_the_box_one(self):
        # The path check-box.sh's --venv defaults to, and install-linux.sh --bench-venv provisions.
        self.assertEqual(bench_venv.SHARED_VENV, Path.home() / ".venvs/ocannl-bench")

    def test_the_default_environment_is_the_process_environment(self):
        with unittest.mock.patch.dict(os.environ, {"BENCH_VENV_PY": "/from/the/process/python"}):
            self.assertEqual(bench_venv.venv_python(self.dir), Path("/from/the/process/python"))

    def test_the_script_prints_the_interpreter_and_its_route(self):
        # The line check-box.sh's driver-path item reads; run from a copy beside a `.venv` of its
        # own, so the route is this leg's and not the box's.
        shutil.copy(Path(bench_venv.__file__), self.dir / "bench_venv.py")
        local = self.make(self.dir / ".venv")
        env = {k: v for k, v in os.environ.items() if k != "BENCH_VENV_PY"}
        done = subprocess.run(
            [sys.executable, str(self.dir / "bench_venv.py")],
            capture_output=True, text=True, timeout=60, env=env,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout, f"{local}\tbenchmarks/.venv\n")

    def test_every_driver_takes_the_shared_rule(self):
        import gh675_cells

        expected = bench_venv.venv_python(orchestrate.HERE)
        self.assertEqual(orchestrate.VENV_PY, expected)
        self.assertEqual(gh675_cells.HERE, orchestrate.HERE)
        self.assertEqual(gh675_cells.VENV, expected)

    def test_no_driver_spells_the_venv_interpreter_itself(self):
        # The relationship test above covers the drivers that exist; this covers the next one,
        # which would otherwise copy `HERE / ".venv/bin/python"` from wherever it started.
        here = Path(orchestrate.HERE)
        sources = sorted(here.glob("*.py")) + sorted(here.glob("runners/**/*.py"))
        spelled = re.compile(r"""["']\.venv[/\\]""")
        offenders = [
            f"{src.relative_to(here)}:{n}"
            for src in sources
            if src.name not in ("bench_venv.py", "test_orchestrate.py")
            for n, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1)
            if spelled.search(line)
        ]
        self.assertTrue(len(sources) > 2, f"the scan found too few sources under {here}")
        self.assertEqual(offenders, [], "use bench_venv.venv_python instead")


class CommandLineTest(unittest.TestCase):
    """What the flags parse to — the wiring between a module constant and the sweep.

    `CellTimeoutTest` pins the values themselves: that `DEFAULT_BEAM_PARALLEL` is zero, that
    `beam_parallel_arg` refuses a negative pool, that `beam_cell_env` turns the number into
    `PARALLEL`, that `cell_timeout_arg` refuses a cap that cannot mean anything. None of that
    reaches the sweep unless the flag is declared with that default, and a `default=` edit is
    invisible to every one of those tests.
    """

    def test_a_sweep_with_no_arguments_runs_the_beam_cells_without_a_pool(self):
        # gh-ocannl-843: the pool deadlocks tinygrad 0.13/0.14 on both GPU boxes, so no pool is
        # what an operator gets by not asking -- on a newly provisioned box as much as here.
        args = orchestrate.build_arg_parser().parse_args([])

        self.assertEqual(args.beam_parallel, 0)

    def test_an_explicit_pool_size_reaches_the_beam_cells(self):
        # The opt-in half: the pin is a default, not a ceiling, and an operator who asks for a
        # parallel search gets the size they asked for rather than the pinned zero.
        args = orchestrate.build_arg_parser().parse_args(["--beam-parallel", "4"])

        self.assertEqual(args.beam_parallel, 4)

    def test_a_sweep_with_no_arguments_caps_each_cell_at_half_an_hour(self):
        # gh-ocannl-760: the cap is what makes an unattended sweep lose a wedged cell rather than
        # the sweep, and it only does that from a bare `orchestrate.py`. The literal is the pin --
        # comparing against DEFAULT_CELL_TIMEOUT_S would still pass if the constant and the flag
        # were edited together, which is the drift this claim exists to catch.
        args = orchestrate.build_arg_parser().parse_args([])

        self.assertEqual(args.cell_timeout, 1800)

    def test_an_explicit_cap_reaches_the_cells(self):
        # The opt-in half: an operator who knows a workload outlives the pinned default gets the
        # cap they asked for, through the same `cell_timeout_arg` that refuses a meaningless one.
        args = orchestrate.build_arg_parser().parse_args(["--cell-timeout", "900"])

        self.assertEqual(args.cell_timeout, 900.0)


class BoxGateStandIn:
    """A stand-in `check-box.sh` in a scratch directory, patched in as `box_gate.CHECK_BOX`: it
    appends its arguments to `calls` and exits with the status `answer` set (0 READY, 1 GAP)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.calls = self.dir / "calls"
        stub = self.dir / "check-box.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            # The parser's name reading answers as the real script does for a declared box;
            # BoxNameTest below asks the real script itself.
            '[ "$1" != --canonical-box ] || { printf \'%s\\n\' "$2"; exit 0; }\n'
            'here=$(dirname "$0")\n'
            'printf \'%s\\n\' "$*" >>"$here/calls"\n'
            'exit "$(cat "$here/rc")"\n'
        )
        patcher = unittest.mock.patch.object(box_gate, "CHECK_BOX", stub)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.answer(0)

    def answer(self, rc):
        (self.dir / "rc").write_text(str(rc))

    def called(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []


class BoxGateTest(BoxGateStandIn, unittest.TestCase):
    """`box_gate.require`, the Python drivers' half of the box gate (contract: box-gate.sh's
    header): the check runs on the box and the fixtures named, READY returns, a GAP exits 1 with a
    refusal, a check that judged nothing exits 2, and a run that names no box is refused before
    any check unless it skips the check on purpose. The drivers' own refusals and starts are
    benchmarks/test-box-gate.sh's legs."""

    @staticmethod
    def check_runner(cmd):
        return subprocess.run(cmd).returncode

    def require(self, box=None, skip=False, names=("gpt2_mini", "lenet")):
        args = argparse.Namespace(box=box, skip_box_check=skip)
        out, err = io.StringIO(), io.StringIO()
        code = None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                box_gate.require("driver.py", args, Path("/store"), names, run=self.check_runner)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_ready_returns_after_checking_the_box_and_every_fixture(self):
        code, _, err = self.require(box="m4-max")

        self.assertIsNone(code, err)
        self.assertEqual(
            self.called(),
            [f"--box m4-max --fixture-dir {Path('/store')} --fixture gpt2_mini --fixture lenet"],
        )

    def test_a_gap_refuses_to_start(self):
        self.answer(1)

        code, _, err = self.require(box="m4-max")

        self.assertEqual(code, 1)
        self.assertIn("driver.py: refusing to start: m4-max is not ready for this run", err)
        self.assertEqual(len(self.called()), 1)

    def test_a_check_that_judged_nothing_exits_2(self):
        self.answer(2)

        code, _, err = self.require(box="m4-max")

        self.assertEqual(code, 2)
        self.assertIn("benchmarks/check-box.sh judged nothing (exit 2)", err)

    def test_a_runner_verdict_that_is_no_status_is_a_check_that_judged_nothing(self):
        # The drivers' managed runners answer "timeout" for a check that outlived their cap.
        self.check_runner = lambda cmd: "timeout"

        code, _, err = self.require(box="m4-max")

        self.assertEqual(code, 2)
        self.assertIn("judged nothing (exit timeout)", err)

    def test_the_runner_is_the_drivers_own(self):
        # No default: a bare subprocess.run would leave the check's GPU probe running past a
        # SIGTERM to the driver, so each driver hands over its managed, group-killing runner.
        args = argparse.Namespace(box="m4-max", skip_box_check=False)
        with self.assertRaises(TypeError):
            box_gate.require("driver.py", args, Path("/store"), ["gpt2_mini"])
        self.assertEqual(self.called(), [])

    def test_no_box_is_refused_before_any_check(self):
        code, _, err = self.require()

        self.assertEqual(code, 2)
        self.assertIn("--box BOX is required to measure", err)
        self.assertEqual(self.called(), [])

    def test_skipping_runs_no_check_and_says_so(self):
        self.answer(1)
        for box in (None, "m4-max"):
            with self.subTest(box=box):
                code, out, _ = self.require(box=box, skip=True)

                self.assertIsNone(code)
                self.assertIn("driver.py: --skip-box-check: measuring without", out)
        self.assertEqual(self.called(), [])

    def test_no_fixture_named_is_refused(self):
        code, _, err = self.require(box="m4-max", names=())

        self.assertIsNotNone(code)
        self.assertEqual(self.called(), [])

    def test_the_options_parse_onto_a_driver_parser(self):
        ap = argparse.ArgumentParser()
        box_gate.add_arguments(ap)

        self.assertEqual(vars(ap.parse_args([])), {"box": None, "skip_box_check": False})
        self.assertEqual(
            vars(ap.parse_args(["--skip-box-check", "--box", "tuf"])),
            {"box": "tuf", "skip_box_check": True},
        )


class BoxNameTest(unittest.TestCase):
    """A driver's `--box` holds the measurement box's ID, also when given the fleet's name for it
    (gh-ocannl-1433): `rog-nv-linux` is recorded as `rog-nv`, in env.json and every row. The table
    is the real check-box.sh's, asked through `--canonical-box`."""

    def test_a_fleet_name_parses_to_its_box(self):
        ap = argparse.ArgumentParser()
        box_gate.add_arguments(ap)
        self.assertEqual(ap.parse_args(["--box", "rog-nv-linux"]).box, "rog-nv")
        self.assertEqual(ap.parse_args(["--box", "minix-amd-linux"]).box, "minix")
        self.assertEqual(ap.parse_args(["--box", "mac-studio"]).box, "m4-max")

    def test_a_box_id_and_an_unknown_name_stay_as_given(self):
        # The unknown one is check-box.sh's to refuse, when the gate runs it.
        self.assertEqual(box_gate.canonical_box("tuf"), "tuf")
        self.assertEqual(box_gate.canonical_box("rog-nv-wsl"), "rog-nv-wsl")

    def test_a_check_that_cannot_answer_leaves_the_name(self):
        with unittest.mock.patch.object(box_gate, "CHECK_BOX", Path("/no/such/check-box.sh")):
            self.assertEqual(box_gate.canonical_box("rog-nv-linux"), "rog-nv-linux")


class OrchestrateBoxGateTest(BoxGateStandIn, unittest.TestCase):
    """`orchestrate.py` runs the box gate before anything else touches the machine: on a GAP it
    exits before reading a fixture's digest, on READY it goes on to read them, and it checks the
    workloads asked for (an absent one is the check's GAP) or else every fixture it found."""

    class Started(Exception):
        """Raised by the sweep's first action past the gate: reading fixtures/DIGESTS.txt."""

    def setUp(self):
        super().setUp()
        (self.dir / "fixtures").mkdir()
        (self.dir / "fixtures" / "mlp_small.safetensors").touch()
        for target, value in (
            ("HERE", self.dir),
            ("install_termination_handler", lambda: None),
        ):
            patcher = unittest.mock.patch.object(orchestrate, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = unittest.mock.patch.object(
            orchestrate.fixture_digest, "read_digests", side_effect=self.Started
        )
        self.read_digests = patcher.start()
        self.addCleanup(patcher.stop)

    def sweep(self, *argv):
        err = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", ["orchestrate.py", "--only", "ocannl", *argv]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                try:
                    orchestrate.main()
                except self.Started:
                    return "started", err.getvalue()
                except SystemExit as e:
                    return e.code, err.getvalue()
        return "returned", err.getvalue()

    def test_a_gap_refuses_the_sweep_before_it_reads_a_fixture(self):
        self.answer(1)

        outcome, err = self.sweep("--box", "m4-max")

        self.assertEqual(outcome, 1, err)
        self.assertIn("orchestrate.py: refusing to start", err)
        self.read_digests.assert_not_called()
        self.assertEqual(
            self.called(),
            [f"--box m4-max --fixture-dir {self.dir / 'fixtures'} --fixture mlp_small"],
        )

    def test_ready_starts_the_sweep(self):
        outcome, err = self.sweep("--box", "m4-max")

        self.assertEqual(outcome, "started", err)
        self.assertEqual(len(self.called()), 1)

    def test_the_workloads_asked_for_are_checked_even_when_absent(self):
        self.answer(1)

        outcome, _ = self.sweep("--box", "rog-nv", "--workloads", "mlp_small", "gpt2_mini")

        self.assertEqual(outcome, 1)
        self.assertEqual(
            self.called(),
            [
                f"--box rog-nv --fixture-dir {self.dir / 'fixtures'} --fixture mlp_small "
                "--fixture gpt2_mini"
            ],
        )

    def test_a_sweep_naming_no_box_is_refused(self):
        outcome, err = self.sweep()

        self.assertEqual(outcome, 2)
        self.assertIn("--box BOX is required", err)
        self.assertEqual(self.called(), [])
        self.read_digests.assert_not_called()

    def test_a_deliberate_partial_run_skips_the_check(self):
        self.answer(1)

        outcome, _ = self.sweep("--skip-box-check")

        self.assertEqual(outcome, "started")
        self.assertEqual(self.called(), [])


class FixtureDriverGateInventoryTest(unittest.TestCase):
    """Every benchmark driver that loads a fixture calls the box gate, or is exempt with a reason.

    The scan's boundary, a fail-closed allowlist. A source is every `*.py` and `*.sh` under
    benchmarks/ at any depth except test/, `.venv` and the `test_*`/`test-*` files. It LOADS A
    FIXTURE when its text holds `safetensors`, `BENCH_FIXTURE` or `--fixture` anywhere, comments
    included -- over-inclusion is the failure this side can have, and an exemption with its
    reason resolves it. It CALLS THE GATE in exactly one of two forms, read on lines that are not
    whole-line comments: a shell source that sources box-gate.sh (`. "…/box-gate.sh"`) and calls
    `box_gate <driver> …`, or a Python source with `import box_gate` and `box_gate.require(`.
    Anything else -- a gate reached through another driver, a call spelled differently -- is a
    finding until an exemption names why it is not.
    """

    LOADS = re.compile(r"safetensors|BENCH_FIXTURE|--fixture")
    EXEMPT = {
        "box-gate.sh": "the gate itself",
        "box_gate.py": "the gate itself",
        "check-box.sh": "the check the gate runs",
        "fixture_digest.py": "records and checks fixture digests; measures nothing",
        "gen_fixtures.py": "writes fixtures; measures nothing",
        "gh612v_session.sh": "hands --box/--skip-box-check to every gh612_cells.sh cell, whose "
        "fixture check runs the gate",
        "runners/bench_common.py": "a runner's helper: reads the fixture its driver hands it",
        "runners/pytorch/run.py": "a runner: reads the fixture its driver hands it",
        "runners/tinygrad/run.py": "a runner: reads the fixture its driver hands it",
    }

    @staticmethod
    def sources():
        found = {}
        for p in sorted(HERE.rglob("*")):
            rel = p.relative_to(HERE)
            if p.suffix not in (".py", ".sh") or not p.is_file():
                continue
            if rel.parts[0] in ("test", ".venv") or p.name.startswith(("test_", "test-")):
                continue
            found[rel.as_posix()] = p.read_text(encoding="utf-8")
        return found

    @staticmethod
    def gated(path, text):
        code = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
        if path.endswith(".sh"):
            return any(re.match(r'\s*(\.|source)\s+.*/box-gate\.sh"?\s*$', l) for l in code) and any(
                re.search(r"(^|[\s;|&(])box_gate\s+[A-Za-z0-9_.-]+\s", l) for l in code
            )
        return any(l.strip() == "import box_gate" for l in code) and any(
            "box_gate.require(" in l for l in code
        )

    def findings(self, sources):
        loaders = {path for path, text in sources.items() if self.LOADS.search(text)}
        ungated = sorted(
            path for path in loaders if path not in self.EXEMPT and not self.gated(path, sources[path])
        )
        stale = sorted(
            path
            for path in self.EXEMPT
            if path not in loaders
            or (path not in ("box-gate.sh", "box_gate.py") and self.gated(path, sources[path]))
        )
        return ungated, stale

    def test_every_fixture_loading_driver_calls_the_gate(self):
        sources = self.sources()
        for driver in ("orchestrate.py", "gh514_cells.sh", "gh1181_cells.py"):
            self.assertIn(driver, sources)
        ungated, stale = self.findings(sources)
        self.assertEqual(ungated, [], "call the box gate (box-gate.sh / box_gate.py), or exempt with a reason")
        self.assertEqual(stale, [], "exemptions that no longer describe a fixture-loading, ungated source")

    def test_a_driver_that_loads_a_fixture_without_the_gate_is_caught(self):
        sources = self.sources()
        sources["gh9999_cells.sh"] = 'BENCH_FIXTURE=fixtures/gpt2_mini.safetensors "$GPT"\n'
        sources["gh9998_cells.py"] = (
            "import box_gate\n# box_gate.require(...) is called elsewhere\n"
            'env = {"BENCH_FIXTURE": "x"}\n'
        )
        sources["gh9997_cells.sh"] = '. "$here/box-gate.sh"\n# box_gate gh9997 "$fx" a\n--fixture a\n'

        ungated, _ = self.findings(sources)

        self.assertEqual(ungated, ["gh9997_cells.sh", "gh9998_cells.py", "gh9999_cells.sh"])

    def test_a_gated_exemption_is_stale(self):
        sources = self.sources()
        sources["gh612v_session.sh"] += '. "$HERE/box-gate.sh"\nbox_gate gh612v "$fx" gpt2_mini\n'

        _, stale = self.findings(sources)

        self.assertEqual(stale, ["gh612v_session.sh"])


class ResultsDirTest(unittest.TestCase):
    """`BENCH_RESULTS_DIR`: where a sweep's results/ goes (gh-ocannl-719).

    A sweep measured in a throwaway checkout -- machine-verify's detached worktree, removed on
    exit -- loses results/ with the checkout unless it is written somewhere that outlives it.
    """

    def test_unset_or_empty_is_results_beside_the_orchestrator(self):
        for env in ({}, {"BENCH_RESULTS_DIR": ""}):
            with self.subTest(env=env):
                self.assertEqual(orchestrate.results_dir(env), HERE / "results")

    def test_a_relative_directory_is_taken_against_the_launch_directory(self):
        launch = Path(tempfile.mkdtemp()).resolve()
        cwd = os.getcwd()
        os.chdir(launch)
        try:
            got = orchestrate.results_dir({"BENCH_RESULTS_DIR": "out/sweep"})
        finally:
            os.chdir(cwd)

        self.assertEqual(got, launch / "out" / "sweep")

    def test_the_module_reads_the_variable_at_import(self):
        # The wiring, observed in a fresh interpreter rather than by calling the helper again: the
        # directory main writes into is the module constant, fixed from the launch environment.
        outside = Path(tempfile.mkdtemp()).resolve() / "kept"
        probe = "import orchestrate; print(orchestrate.RESULTS_DIR)"
        for value, want in ((str(outside), outside), (None, HERE / "results")):
            env = {k: v for k, v in os.environ.items() if k != "BENCH_RESULTS_DIR"}
            if value is not None:
                env["BENCH_RESULTS_DIR"] = value
            with self.subTest(value=value):
                out = subprocess.run(
                    [sys.executable, "-c", probe],
                    cwd=HERE, env=env, capture_output=True, text=True, check=True,
                )
                self.assertEqual(Path(out.stdout.strip()), want)

    def test_every_sweep_output_is_written_under_the_results_directory(self):
        # main names its three outputs from RESULTS_DIR; a path spelled from HERE instead would
        # still land in the checkout the override exists to leave.
        source = Path(orchestrate.__file__).read_text()
        main = source[source.index("\ndef main():") :]
        self.assertNotIn('HERE / "results"', main)
        for name in ("partial.jsonl", "partial-failures.jsonl"):
            self.assertIn(f'RESULTS_DIR / "{name}"', main)
        self.assertIn("report(results, RESULTS_DIR,", main)


class SkipCellTest(unittest.TestCase):
    """`--skip-cell`: an operator leaving one OCANNL cell out of a sweep (gh-ocannl-719).

    The case it exists for is a GPU tuned acceptance row behind an uncapped CPU search that runs
    first and for hours; what matters is that a misspelt entry is refused rather than skipping
    nothing, and that a skipped cell's absence is reported as a choice.
    """

    def parse(self, *argv):
        return orchestrate.build_arg_parser().parse_args(list(argv))

    def test_no_entries_by_default(self):
        self.assertEqual(self.parse().skip_cell, [])

    def test_an_entry_without_precision_is_a_precision_wildcard(self):
        args = self.parse("--skip-cell", "gpt2_mini/cc/tuned", "--skip-cell", "mlp/cuda/default/bf16")

        self.assertEqual(
            args.skip_cell,
            [("gpt2_mini", "cc", "tuned", None), ("mlp", "cuda", "default", "bf16")],
        )
        skips = set(args.skip_cell)
        for precision in ("f32", "bf16", "f16"):
            self.assertTrue(
                orchestrate.cell_skipped("gpt2_mini", "cc", "tuned", precision, skips), precision
            )
        self.assertFalse(orchestrate.cell_skipped("gpt2_mini", "cuda", "tuned", "f32", skips))
        self.assertFalse(orchestrate.cell_skipped("mlp", "cuda", "default", "f32", skips))

    def test_a_misspelt_entry_is_refused_at_parse_time(self):
        for bad in (
            "gpt2_mini/cc",
            "gpt2_mini/cc/tuned/f32/x",
            "gpt2_mini//tuned",
            "gpt2_mini/cpu/tuned",
            "gpt2_mini/cc/tune",
            "gpt2_mini/cc/tuned/f64",
        ):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse("--skip-cell", bad)

    def test_the_accepted_names_are_the_ones_the_sweep_dispatches(self):
        # Derived, not restated: every variant the flags can select and every backend a --gpu
        # choice dispatches on is a name an entry may use, and nothing else is.
        args = self.parse("--materialized", "--tuned")
        self.assertEqual(orchestrate.selected_variants(args), list(orchestrate.SWEEP_VARIANTS))
        backends = {
            b for gpu in orchestrate.GPU_DEVICES.values() for b in orchestrate.swept_backends(gpu[0])
        }
        self.assertEqual(backends, set(orchestrate.SWEEP_BACKENDS))

    def test_an_entry_naming_no_cell_of_the_sweep_is_unmatched(self):
        args = self.parse("--tuned", "--precision", "bf16")
        cells = orchestrate.dispatchable_cells(
            {"gpt2_mini": ["f32", "bf16"]},
            orchestrate.swept_backends("cuda"),
            orchestrate.selected_variants(args),
        )
        good = ("gpt2_mini", "cc", "tuned", None)
        self.assertEqual(orchestrate.unmatched_skips([good], cells), [])
        self.assertEqual(
            orchestrate.unmatched_skips([("gpt2_mini", "cuda", "tuned", "bf16")], cells), []
        )
        for bad in (
            ("gpt2_small", "cc", "tuned", None),  # workload not selected
            ("gpt2_mini", "hip", "tuned", None),  # backend not swept under --gpu cuda
            ("gpt2_mini", "cc", "materialized", None),  # variant not requested
            ("gpt2_mini", "cc", "tuned", "f16"),  # precision not requested
        ):
            with self.subTest(bad=bad):
                self.assertEqual(orchestrate.unmatched_skips([good, bad], cells), [bad])

    def test_a_precision_the_workload_cannot_express_is_no_cell(self):
        # Review round 1: `--precision bf16` is requested sweep-wide, but a conv workload cannot
        # express it, so a skip naming its bf16 tuned cell skips nothing -- and must be refused
        # rather than let the f32 cell beside it run in the operator's belief it was skipped.
        # The precisions come from `available_precisions`, the dispatch loop's own filter.
        precisions, unavailable = orchestrate.available_precisions("conv", "train", ["bf16"])
        self.assertEqual(precisions, ["f32"])
        self.assertEqual([p for p, _ in unavailable], ["bf16"])
        cells = orchestrate.dispatchable_cells(
            {"lenet": precisions}, orchestrate.swept_backends("cuda"), ["default", "tuned"]
        )
        bad = ("lenet", "cc", "tuned", "bf16")

        self.assertEqual(orchestrate.unmatched_skips([bad], cells), [bad])
        self.assertEqual(
            orchestrate.unmatched_skips([("lenet", "cc", "tuned", None)], cells), []
        )

    def test_an_operator_skip_beats_no_skip_cells(self):
        skips = {("gpt2_mini", "cc", "tuned", None)}
        why = orchestrate.skip_reason("gpt2_mini", "cc", "tuned", "f32", skips, no_skip_cells=True)

        self.assertIn("--skip-cell", why)
        self.assertIsNone(
            orchestrate.skip_reason("gpt2_mini", "cuda", "tuned", "f32", skips, no_skip_cells=True)
        )

    def test_a_checked_in_skip_still_yields_to_no_skip_cells(self):
        entry = ("some_workload", "metal", "tuned", None)
        orchestrate.SKIP_CELLS.add(entry)
        try:
            why = orchestrate.skip_reason("some_workload", "metal", "tuned", "f32", set(), False)
            self.assertIn("SKIP_CELLS", why)
            self.assertIsNone(
                orchestrate.skip_reason("some_workload", "metal", "tuned", "f32", set(), True)
            )
        finally:
            orchestrate.SKIP_CELLS.discard(entry)

    def test_a_skipped_cell_is_named_in_the_report(self):
        out = Path(tempfile.mkdtemp())
        cells = [cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])]
        orchestrate.parity_check(cells)
        skipped = [("gpt2_mini ocannl/cc/tuned [approximate]", "--skip-cell (operator)")]

        orchestrate.report(cells, out, skipped=skipped)

        text = (out / "report.md").read_text()
        self.assertIn("Cells skipped", text)
        self.assertIn("| gpt2_mini ocannl/cc/tuned [approximate] | --skip-cell (operator) |", text)
        self.assertNotIn("Runner failures", text)

    def test_a_run_with_no_skips_says_nothing_about_them(self):
        out = Path(tempfile.mkdtemp())
        cells = [cell("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])]
        orchestrate.parity_check(cells)

        orchestrate.report(cells, out)

        self.assertNotIn("Cells skipped", (out / "report.md").read_text())


class OcannlFlagTest(unittest.TestCase):
    """Per-cell `--cell-flag` settings, and the ambient variables a cell's own command line
    outranks (gh-ocannl-1273).

    The case behind both: a flip-budget sweep under the approximate regime. The regime's
    `--ocannl_profile=approximate` outranks the environment, so an ambient
    OCANNL_TUNE_INLINE_FLIPS=16 lost to the payload's 2 while the row recorded the variable as
    though it applied; and there was no way to give the cells the budget at all.
    """

    def parse(self, *argv):
        return orchestrate.build_arg_parser().parse_args(list(argv))

    def refused(self, *argv):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse(*argv)

    def test_the_reference_is_read_for_the_keys_and_the_payloads(self):
        keys, payloads = orchestrate.config_reference()
        self.assertEqual(set(payloads), {"reproducible", "performance", "approximate"})
        self.assertIn("tune_inline_flips", orchestrate.profile_payload_keys("approximate"))
        self.assertIn("tf32_matmuls", orchestrate.profile_payload_keys("approximate"))
        self.assertNotIn("tf32_matmuls", orchestrate.profile_payload_keys("performance"))
        # Every key a payload sets is a key the reference documents (test_config_consistency
        # holds the library to the same), so the two readings agree with each other.
        self.assertTrue(keys)
        for name, payload in payloads.items():
            with self.subTest(profile=name):
                self.assertTrue(payload)
                self.assertLessEqual(payload, keys)
        with self.assertRaises(ValueError):
            orchestrate.profile_payload_keys("exact")

    def test_a_flag_parses_with_and_without_a_selector(self):
        args = self.parse(
            "--cell-flag", "tune_inline_flips=16",
            "--cell-flag", "gpt2_mini/*/tuned:autotune_cache_dir=/tmp/a=b:c",
            "--cell-flag", "*/cuda/default/bf16:tf32_matmuls=false",
        )
        F = orchestrate.CellFlag
        self.assertEqual(
            args.cell_flag,
            [
                F(None, "tune_inline_flips", "16"),
                # The value is everything after the first `=`: `=` and `:` inside it are its own.
                F(("gpt2_mini", None, "tuned", None), "autotune_cache_dir", "/tmp/a=b:c"),
                F((None, "cuda", "default", "bf16"), "tf32_matmuls", "false"),
            ],
        )
        self.assertEqual(self.parse().cell_flag, [])

    def test_a_flag_the_sweep_cannot_honour_is_refused_at_parse_time(self):
        for bad in (
            "tune_inline_flips",  # no value
            "tune_inline_flip=16",  # not a documented key
            "ocannl_tune_inline_flips=16",  # the prefix is the sweep's to add
            "tune_inline_flips=",  # empty is unset to OCANNL
            "gpt2_mini/cc:tune_inline_flips=16",  # selector too short
            "gpt2_mini/cpu/tuned:tune_inline_flips=16",  # not a swept backend
            "gpt2_mini/cc/tune:tune_inline_flips=16",  # not a variant
            "gpt2_mini/cc/tuned/f64:tune_inline_flips=16",  # not a precision
        ):
            with self.subTest(bad=bad):
                self.refused("--cell-flag", bad)

    def test_the_sweeps_own_keys_are_refused_and_derived_from_its_argument_builder(self):
        owned = orchestrate.sweep_owned_keys()
        self.assertEqual(owned, {"backend", "profile", "autotune_progress"})
        for key in sorted(owned):
            with self.subTest(key=key):
                self.refused("--cell-flag", f"{key}=x")

    def test_a_per_cell_flag_reaches_the_cells_argv_after_the_profile(self):
        flags = self.parse(
            "--cell-flag", "tune_inline_flips=16",
            "--cell-flag", "*/*/tuned:tf32_matmuls=false",
        ).cell_flag
        tuned = orchestrate.cell_flags(flags, ("gpt2_mini", "cuda", "tuned", "f32"))
        default = orchestrate.cell_flags(flags, ("gpt2_mini", "cuda", "default", "f32"))
        self.assertEqual(tuned, [("tune_inline_flips", "16"), ("tf32_matmuls", "false")])
        self.assertEqual(default, [("tune_inline_flips", "16")])

        cmd, stamp = orchestrate.ocannl_cell("gpt2", "cuda", "approximate", "tuned", tuned, {})

        self.assertEqual(cmd[0], str(orchestrate.ocannl_exe("gpt2")))
        self.assertEqual(
            cmd[1:],
            [
                "--ocannl_backend=cuda",
                "--ocannl_profile=approximate",
                "--ocannl_autotune_progress=true",
                "--ocannl_tune_inline_flips=16",
                "--ocannl_tf32_matmuls=false",
            ],
        )
        self.assertEqual(
            stamp,
            {
                "ocannl_flags": ["--ocannl_tune_inline_flips=16", "--ocannl_tf32_matmuls=false"],
                "ambient_ocannl_shadowed": {},
            },
        )

    def test_an_ambient_key_the_profile_sets_is_stamped_shadowed_not_applied(self):
        ambient = {
            "OCANNL_TUNE_INLINE_FLIPS": "16",  # in the approximate payload
            "OCANNL_NARROW_COMPUTE_F32": "true",  # in no payload: applies in every regime
            "OCANNL_PROFILE": "reproducible",  # outranked by the commandline profile only
            "OCANNL_BACKEND": "metal",  # outranked by the sweep's own --ocannl_backend
            "OCANNL_tf32_matmuls": "true",  # never read: OCANNL reads the upper-cased name
            "OCANNL_CC_BACKEND_FAST_MATH": "",  # empty is unset to OCANNL
        }
        _, approx = orchestrate.ocannl_cell("gpt2", "cc", "approximate", "default", [], ambient)
        _, exact = orchestrate.ocannl_cell("gpt2", "cc", "exact", "default", [], ambient)

        self.assertEqual(
            approx["ambient_ocannl_shadowed"],
            {
                "OCANNL_BACKEND": "--ocannl_backend=cc",
                "OCANNL_PROFILE": "--ocannl_profile=approximate",
                "OCANNL_TUNE_INLINE_FLIPS": "--ocannl_profile=approximate",
            },
        )
        # No profile on the exact cell's command line, so the environment's keys apply there.
        self.assertEqual(
            exact["ambient_ocannl_shadowed"], {"OCANNL_BACKEND": "--ocannl_backend=cc"}
        )
        # The flag outranks the ambient variable too, and the row names the flag as the reason.
        _, flagged = orchestrate.ocannl_cell(
            "gpt2", "cc", "exact", "default", [("narrow_compute_f32", "false")], ambient
        )
        self.assertEqual(
            flagged["ambient_ocannl_shadowed"]["OCANNL_NARROW_COMPUTE_F32"],
            "--ocannl_narrow_compute_f32=false",
        )

    def test_the_warning_is_said_once_for_the_whole_sweep(self):
        cells = {("gpt2_mini", "cc", v, "f32") for v in ("default", "tuned")}
        ambient = {"OCANNL_TUNE_INLINE_FLIPS": "16", "OCANNL_NARROW_COMPUTE_F32": "true"}

        warning = orchestrate.shadowing_warning(ambient, cells, ["exact", "approximate"], [])

        self.assertEqual(warning.count("OCANNL_TUNE_INLINE_FLIPS"), 1)
        self.assertIn("--ocannl_profile=approximate", warning)
        self.assertNotIn("OCANNL_NARROW_COMPUTE_F32", warning)
        self.assertIsNone(orchestrate.shadowing_warning(ambient, cells, ["exact"], []))

    def test_an_unmatched_or_conflicting_flag_is_refused_before_the_sweep(self):
        cells = orchestrate.dispatchable_cells({"gpt2_mini": ["f32"]}, ["cc"], ["default", "tuned"])
        F = orchestrate.CellFlag
        everywhere = F(None, "tune_inline_flips", "16")
        tuned = F((None, None, "tuned", None), "tune_inline_flips", "4")
        nowhere = F(("gpt2_mini", "metal", None, None), "tf32_matmuls", "false")

        self.assertEqual(orchestrate.flag_problems([everywhere], cells), [])
        problems = orchestrate.flag_problems([everywhere, tuned, nowhere], cells)
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("gpt2_mini/metal/*/*:tf32_matmuls=false names no OCANNL cell", problems[0])
        self.assertIn("sets tune_inline_flips 2 times on gpt2_mini/cc/tuned/f32", problems[1])
        # The cell is named as a label names it: a workload carrying a newline cannot forge a
        # second line of the refusal.
        odd = orchestrate.dispatchable_cells({"a\nb|c": ["f32"]}, ["cc"], ["default"])
        [dup] = orchestrate.flag_problems([everywhere, everywhere], odd)
        self.assertIn(f"on {orchestrate.display_value('a' + chr(10) + 'b|c')}/cc/default/f32", dup)
        self.assertNotIn("\n", dup)
        # A sweep with no OCANNL cells at all (`--only pytorch`) has nothing for a flag to reach.
        self.assertEqual(len(orchestrate.flag_problems([everywhere], set())), 1)

    def test_the_regime_gate_accepts_exactly_the_declared_override(self):
        def row(value, source, flags):
            r = result("ocannl", "cuda", "tuned", [1.0])
            r.update(
                regime="approximate",
                profile="approximate",
                ocannl_flags=flags,
                regime_knobs={"tune_inline_flips": {"value": value, "source": source}},
            )
            return r

        declared = ["--ocannl_tune_inline_flips=16"]
        accepted = row("16", "commandline", declared)
        # The runner resolved a different value than the sweep passed: not what was declared.
        other_value = row("8", "commandline", declared)
        # The profile won after all (the flag never reached the runner): the row's flag is false.
        profile_won = row("2", "profile 'approximate' via the commandline", declared)
        # The same commandline source with nothing declared stays a leak (gh-ocannl-719).
        undeclared = row("16", "commandline", [])

        mismatched = orchestrate.regime_check([accepted, other_value, profile_won, undeclared])

        self.assertEqual(mismatched, [other_value, undeclared])
        self.assertEqual(orchestrate.regime_knob_leaks(accepted), {})
        self.assertEqual(orchestrate.regime_knob_leaks(profile_won), {})

    def test_a_failure_record_carries_the_cells_stamp(self):
        stamp = {
            "ocannl_flags": ["--ocannl_tune_inline_flips=16"],
            "ambient_ocannl_shadowed": {"OCANNL_BACKEND": "--ocannl_backend=cc"},
        }
        record = orchestrate.failure_record(
            "w ocannl/cc/tuned [approximate] +tune_inline_flips=16",
            "timed out",
            {"OCANNL_BACKEND": "metal"},
            regime="approximate",
            cell_stamp=stamp,
        )
        self.assertEqual(record["ocannl_flags"], stamp["ocannl_flags"])
        self.assertEqual(record["ambient_ocannl_shadowed"], stamp["ambient_ocannl_shadowed"])
        self.assertEqual(record["ambient_ocannl_env"], {"OCANNL_BACKEND": "metal"})
        self.assertNotIn(
            "ocannl_flags", orchestrate.failure_record("w pytorch/cpu/eager", "x", {})
        )

    def test_a_flag_value_cannot_break_a_label_or_a_report_table(self):
        # Review round 1: `debug_highlight_pcre=a|b` is a legitimate setting, and a raw `|` in
        # the variant column adds a table column; a newline adds a report line.
        for value in ("a|b", "x\ny", "a`b", "two words"):
            with self.subTest(value=value):
                shown = orchestrate.display_value(value)
                self.assertNotIn("|", shown)
                self.assertNotIn("\n", shown)
                self.assertNotIn("`", shown)
                self.assertEqual(json.loads(shown), value)
        self.assertEqual(orchestrate.display_value("/tmp/cache-1"), "/tmp/cache-1")
        self.assertEqual(
            orchestrate.ocannl_flags_note(["--ocannl_debug_highlight_pcre=a|b"]),
            ' +debug_highlight_pcre="a\\u007cb"',
        )

        out = Path(tempfile.mkdtemp())
        row = cell("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])
        row["ocannl_flags"] = ["--ocannl_debug_highlight_pcre=a|b"]
        rows = [cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000]), row]
        orchestrate.parity_check(rows)
        orchestrate.report(rows, out)

        table = [l for l in (out / "report.md").read_text().splitlines() if l.startswith("| ")]
        self.assertEqual(len({l.count("|") for l in table}), 1, table)

        # Review round 2: the messages too -- the once-per-sweep warning and the refusals name
        # the flag, and the class is closed by routing every rendering through `display_value`.
        flag = orchestrate.CellFlag(None, "debug_highlight_pcre", "x\nWARNING: forged")
        cells = {("w", "cc", "default", "f32")}
        warning = orchestrate.shadowing_warning(
            {"OCANNL_DEBUG_HIGHLIGHT_PCRE": "y"}, cells, ["exact"], [flag]
        )
        self.assertNotIn("\n", warning)
        self.assertIn('--ocannl_debug_highlight_pcre="x\\nWARNING: forged"', warning)
        [problem] = orchestrate.flag_problems(
            [orchestrate.CellFlag(("a|b", None, None, None), "debug_highlight_pcre", "a|b")], cells
        )
        self.assertNotIn("|", problem)

    def test_the_cache_guidance_names_the_cells_own_cache(self):
        # Review round 1: a tuned cell given another autotune_cache_dir leaves its finished arms
        # there, so guidance naming the default would leave the real cache populated.
        argv = orchestrate.ocannl_cell_args(
            "cc", "exact", "tuned", [("autotune_cache_dir", "/c/x")]
        )
        env = {"OCANNL_AUTOTUNE_CACHE_DIR": "/env/cache"}
        self.assertEqual(orchestrate.ocannl_cache_dir(argv, env), "/c/x")
        plain = orchestrate.ocannl_cell_args("cc", "exact", "tuned")
        self.assertEqual(orchestrate.ocannl_cache_dir(plain, env), "/env/cache")
        self.assertEqual(orchestrate.ocannl_cache_dir(plain, {}), "autotune_cache/")
        self.assertEqual(
            orchestrate.ocannl_cache_dir(plain, {"OCANNL_AUTOTUNE_CACHE_DIR": "a|b"}),
            '"a\\u007cb"',
        )
        note = orchestrate.ocannl_cache_note(True, cache_dir="/c/x")
        self.assertIn("wipe /c/x for a comparable", note)
        self.assertNotIn("autotune_cache/", note)
        # What makes those the only sources: no profile payload sets the key, and the benchmarks'
        # own config file does not either.
        for name, payload in orchestrate.config_reference()[1].items():
            with self.subTest(profile=name):
                self.assertNotIn("autotune_cache_dir", payload)
        config = (HERE / "ocannl_config").read_text().splitlines()
        self.assertFalse(
            [l for l in config if l.strip().split("=", 1)[0].strip() == "autotune_cache_dir"]
        )

    def test_a_failed_cell_still_says_what_its_command_line_outranked(self):
        # Review round 1: when only the cells that shadow a variable fail, the rows alone would
        # leave the header saying nothing, and the variable reading as applied in those cells.
        stamp = {
            "ocannl_flags": [],
            "ambient_ocannl_shadowed": {
                "OCANNL_TUNE_INLINE_FLIPS": "--ocannl_profile=approximate"
            },
        }
        exact = cell("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])
        exact["ambient_ocannl_shadowed"] = {}
        failed = orchestrate.Failure(
            "w ocannl/cc/default [approximate]", "TIMED OUT", None, "approximate", stamp
        )
        other = ("w pytorch/cpu/eager", "no result line")

        line = orchestrate.shadowed_env_line([exact], [failed, other])

        self.assertIn('"OCANNL_TUNE_INLINE_FLIPS"', line)
        self.assertIn("in 1 of 2 OCANNL cells", line)
        self.assertIsNone(orchestrate.shadowed_env_line([exact], [other]))


class ShadowingObservationTest(unittest.TestCase):
    """gh-ocannl-1337: the sweep's `ambient_ocannl_shadowed` stamp, checked against where the
    runner reports each inherited key resolved (`ambient_ocannl_resolved`). Fed the checkpoints
    `Bench_json` built (the golden of test/operations/bench_result_line), so the comparison is
    pinned to what a runner writes; the prediction side is `ocannl_cell`'s own."""

    def observed(self, workload, **stages):
        return orchestrate.last_checkpoint(golden_checkpoint_line(workload, **stages))[
            "ambient_ocannl_resolved"
        ]

    # The golden's mid-parity checkpoint: a hip cell under the approximate profile, which
    # inherited OCANNL_BACKEND and OCANNL_TUNE_INLINE_FLIPS.
    AMBIENT = {"OCANNL_BACKEND": "cuda", "OCANNL_TUNE_INLINE_FLIPS": "16"}

    def stamp(self, regime, flags=()):
        _, stamp = orchestrate.ocannl_cell(
            "gpt2", "hip", regime, "default", list(flags), self.AMBIENT
        )
        return stamp["ambient_ocannl_shadowed"]

    def test_a_correct_prediction_agrees_with_the_runner(self):
        observed = self.observed("gpt2_mini_train_s1024", parity="running")
        self.assertEqual(orchestrate.shadowing_mismatches(self.stamp("approximate"), observed), {})

    def test_a_mispredicted_stamp_is_flagged_in_each_direction(self):
        observed = self.observed("gpt2_mini_train_s1024", parity="running")
        # Predicted applied, resolved from the profile.
        self.assertEqual(
            orchestrate.shadowing_mismatches(self.stamp("exact"), observed),
            {
                "OCANNL_TUNE_INLINE_FLIPS": "stamped applied, but the runner resolved it from "
                "profile 'approximate' via the commandline"
            },
        )
        # Predicted outranked by a per-cell flag, resolved from the profile: the right verdict
        # (not applied) for the wrong reason is still a wrong account.
        self.assertEqual(
            orchestrate.shadowing_mismatches(
                self.stamp("approximate", [("tune_inline_flips", "4")]), observed
            ),
            {
                "OCANNL_TUNE_INLINE_FLIPS": "stamped outranked by --ocannl_tune_inline_flips=4, "
                "but the runner resolved it from profile 'approximate' via the commandline"
            },
        )
        # Predicted outranked, resolved from the environment: the golden's leaked exact cell.
        leaked = self.observed("mlp_wide")
        self.assertEqual(
            orchestrate.shadowing_mismatches(
                {"OCANNL_TF32_MATMULS": "--ocannl_profile=approximate"}, leaked
            ),
            {
                "OCANNL_TF32_MATMULS": "stamped outranked by --ocannl_profile=approximate, but "
                "the runner resolved it from environment"
            },
        )
        # Predicted outranked, never inherited as a setting at all.
        self.assertEqual(
            orchestrate.shadowing_mismatches(
                {"OCANNL_NARROW_COMPUTE_F32": "--ocannl_profile=approximate"}, {}
            ),
            {
                "OCANNL_NARROW_COMPUTE_F32": "stamped outranked by --ocannl_profile=approximate, "
                "but the runner inherited no such setting"
            },
        )

    def test_the_variables_own_argument_is_a_commandline_source(self):
        # OCANNL_PROFILE is outranked by the profile argument as the key's own argument, which the
        # runner reports as `commandline`, not as the profile's payload.
        observed = {"OCANNL_PROFILE": {"key": "profile", "value": "approximate",
                                       "source": "commandline"}}
        stamp = {"OCANNL_PROFILE": "--ocannl_profile=approximate"}
        self.assertEqual(orchestrate.shadowing_mismatches(stamp, observed), {})

    def test_a_row_or_record_without_both_sides_is_not_checked(self):
        old = cell("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])
        old["ambient_ocannl_shadowed"] = {"OCANNL_BACKEND": "--ocannl_backend=cc"}
        torch = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        self.assertEqual(orchestrate.shadowing_check([old, torch]), [])
        self.assertIsNone(orchestrate.shadowing_mismatches(None, {}))

    def test_a_failure_record_checks_its_stamp_against_its_checkpoint(self):
        checkpoint = orchestrate.last_checkpoint(
            golden_checkpoint_line("gpt2_mini_train_s1024", parity="running")
        )
        wrong = {"ocannl_flags": [], "ambient_ocannl_shadowed": self.stamp("exact")}
        right = {"ocannl_flags": [], "ambient_ocannl_shadowed": self.stamp("approximate")}

        flagged = orchestrate.failure_record("w", "TIMED OUT", self.AMBIENT, checkpoint,
                                             regime="exact", cell_stamp=wrong)
        clean = orchestrate.failure_record("w", "TIMED OUT", self.AMBIENT, checkpoint,
                                           regime="approximate", cell_stamp=right)

        self.assertIn("OCANNL_TUNE_INLINE_FLIPS stamped applied", flagged["shadowing_mismatch"])
        self.assertNotIn("shadowing_mismatch", clean)
        line = orchestrate.shadowing_mismatch_line(
            [], [orchestrate.Failure("w ocannl/hip/default", "TIMED OUT", checkpoint, "exact",
                                     wrong)]
        )
        self.assertIn("w ocannl/hip/default (failed): OCANNL_TUNE_INLINE_FLIPS stamped", line)


class OcannlFlagSweepTest(unittest.TestCase):
    """The issue's acceptance, through `main` itself (gh-ocannl-1273): a sweep under the
    approximate regime with OCANNL_TUNE_INLINE_FLIPS=16 in the environment warns and stamps the
    key as shadowed, and the same sweep with `--cell-flag tune_inline_flips=16` runs its cells
    with the flag on their command lines, recorded on their rows, past the regime gate.

    Only the cell itself is faked: `run_cell` records the command it was handed and answers the
    way the runner resolves its configuration -- the commandline, then the commandline profile's
    payload, then the environment -- so the dispatch loop, the stamps, the gates and the report
    are the sweep's own.
    """

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        (fixtures / fixture_digest.DIGEST_FILE).write_bytes(
            (HERE / "fixtures" / fixture_digest.DIGEST_FILE).read_bytes()
        )
        header = json.dumps({"__metadata__": {"model": "mlp", "mode": "train"}}).encode()
        (fixtures / "flagged.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)

    def fake_cell(self, label, cmd, env=None, **_):
        self.dispatched.append((label, list(cmd)))
        settings = orchestrate.cmdline_settings(cmd[1:])
        if "tune_inline_flips" in settings:
            value = settings["tune_inline_flips"].split("=", 1)[1]
            knob = {"value": value, "source": "commandline"}
        elif settings.get("profile") == "--ocannl_profile=approximate":
            knob = {"value": "2", "source": "profile 'approximate' via the commandline"}
        elif (env or {}).get("OCANNL_TUNE_INLINE_FLIPS"):
            knob = {"value": env["OCANNL_TUNE_INLINE_FLIPS"], "source": "environment"}
        else:
            knob = {"source": "default"}
        row = cell("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])
        row.update(
            workload="flagged",
            profile="approximate" if "profile" in settings else None,
            regime_knobs={"tune_inline_flips": knob},
            # Where the inherited variable resolved, as the runner reports it (gh-ocannl-1337):
            # the same resolution as the knob's, since it is the same key.
            ambient_ocannl_resolved=(
                {"OCANNL_TUNE_INLINE_FLIPS": {"key": "tune_inline_flips", **knob}}
                if (env or {}).get("OCANNL_TUNE_INLINE_FLIPS")
                else {}
            ),
            searched=False,
        )
        return row, None

    def sweep(self, *argv, env=None, gate=None):
        """`main` over the fake cells; `gate`, when given, is the gate the sweep is expected to
        fail on, whose exit is caught so the artifacts can be read."""
        self.dispatched = []
        environ = {k: v for k, v in os.environ.items() if not k.startswith("OCANNL_")}
        environ.update(env or {})
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.dict(os.environ, environ, clear=True))
            stack.enter_context(unittest.mock.patch.object(orchestrate, "HERE", self.dir))
            stack.enter_context(
                unittest.mock.patch.object(orchestrate, "RESULTS_DIR", self.dir / "results")
            )
            stack.enter_context(unittest.mock.patch.object(orchestrate, "run_cell", self.fake_cell))
            stack.enter_context(
                unittest.mock.patch.object(orchestrate, "install_termination_handler", lambda: None)
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    sys,
                    "argv",
                    ["orchestrate.py", "--only", "ocannl", "--gpu", "none", "--skip-build",
                     "--no-fixture-digest-check", "--skip-box-check", "--profile", "approximate",
                     *argv],
                )
            )
            stack.enter_context(contextlib.redirect_stdout(out))
            if gate is None:
                # Returns rather than exiting: every gate passes, the regime gate included.
                orchestrate.main()
            else:
                with self.assertRaises(SystemExit) as failed:
                    orchestrate.main()
                self.assertEqual(failed.exception.code, 1)
                self.assertIn(gate, out.getvalue())
        rows = [
            json.loads(line)
            for line in (self.dir / "results" / "results.jsonl").read_text().splitlines()
        ]
        return out.getvalue(), rows, (self.dir / "results" / "report.md").read_text()

    def test_an_ambient_key_the_profile_outranks_is_warned_and_stamped_shadowed(self):
        stdout, rows, report_text = self.sweep(env={"OCANNL_TUNE_INLINE_FLIPS": "16"})

        warnings = [l for l in stdout.splitlines() if l.startswith("WARNING: ambient OCANNL_*")]
        self.assertEqual(len(warnings), 1, stdout)
        self.assertIn("OCANNL_TUNE_INLINE_FLIPS (by --ocannl_profile=approximate)", warnings[0])
        [(_, cmd)] = self.dispatched
        self.assertNotIn("tune_inline_flips", " ".join(cmd))
        [row] = rows
        # Inherited, and recorded as such -- and named as not having applied.
        self.assertEqual(row["ambient_ocannl_env"], {"OCANNL_TUNE_INLINE_FLIPS": "16"})
        self.assertEqual(
            row["ambient_ocannl_shadowed"],
            {"OCANNL_TUNE_INLINE_FLIPS": "--ocannl_profile=approximate"},
        )
        self.assertEqual(row["ocannl_flags"], [])
        self.assertIn(
            'ambient OCANNL_* variables outranked by a cell\'s own command line (not applied '
            'there): "OCANNL_TUNE_INLINE_FLIPS" by "--ocannl_profile=approximate" in 1 of 1 '
            "OCANNL cells",
            report_text,
        )
        # gh-ocannl-1337: and the row records what the runner observed, which agrees.
        self.assertEqual(
            row["ambient_ocannl_resolved"]["OCANNL_TUNE_INLINE_FLIPS"]["source"],
            "profile 'approximate' via the commandline",
        )
        self.assertNotIn("shadowing_mismatch", row)
        self.assertNotIn("SHADOWING GATE", stdout)
        self.assertNotIn("stamps their runner contradicted", report_text)

    def test_a_mispredicted_stamp_fails_the_sweep_and_is_named_in_the_report(self):
        # gh-ocannl-1337's acceptance, its second half: the stamp is deliberately mispredicted --
        # the sweep is made to believe the profile outranks nothing -- and the runner's own record
        # of where the variable resolved contradicts it.
        with unittest.mock.patch.object(orchestrate, "shadowed_ambient", lambda ambient, argv: {}):
            stdout, rows, report_text = self.sweep(
                env={"OCANNL_TUNE_INLINE_FLIPS": "16"}, gate="SHADOWING GATE"
            )

        [row] = rows
        self.assertEqual(row["ambient_ocannl_shadowed"], {})
        expected = (
            "OCANNL_TUNE_INLINE_FLIPS stamped applied, but the runner resolved it from "
            "profile 'approximate' via the commandline"
        )
        self.assertEqual(row["shadowing_mismatch"], expected)
        [gate] = [l for l in stdout.splitlines() if "SHADOWING GATE" in l]
        self.assertIn(expected, gate)
        self.assertIn("ambient OCANNL_* stamps their runner contradicted", report_text)
        self.assertIn(expected, report_text)

    def test_a_flag_reaching_only_skipped_cells_is_refused(self):
        # Review round 3: validated against the cells the sweep RUNS, so a selector naming only a
        # `--skip-cell`'d cell is refused instead of applying to nothing, while the sweep still
        # has a runnable cell for it to have meant.
        with self.assertRaises(SystemExit) as refused:
            self.sweep(
                "--materialized",
                "--skip-cell", "flagged/cc/materialized",
                "--cell-flag", "flagged/cc/materialized:tune_inline_flips=16",
            )
        self.assertIn("names no OCANNL cell this sweep runs", str(refused.exception.code))
        self.assertEqual(self.dispatched, [])

    def test_a_per_cell_flag_runs_the_cell_with_the_budget_applied(self):
        stdout, rows, report_text = self.sweep("--cell-flag", "tune_inline_flips=16")

        self.assertNotIn("WARNING: ambient OCANNL_*", stdout)
        [(label, cmd)] = self.dispatched
        self.assertEqual(cmd[-1], "--ocannl_tune_inline_flips=16")
        self.assertIn("--ocannl_profile=approximate", cmd)
        self.assertTrue(label.endswith("[approximate] +tune_inline_flips=16"), label)
        [row] = rows
        self.assertEqual(row["ocannl_flags"], ["--ocannl_tune_inline_flips=16"])
        self.assertEqual(row["regime_knobs"]["tune_inline_flips"]["value"], "16")
        self.assertNotIn("regime_mismatch", row)
        self.assertIn("| default +tune_inline_flips=16 |", report_text)
        self.assertNotIn("ambient OCANNL_* variables outranked", report_text)


def markdown_tables(text):
    """Every Markdown table in `text`, as `(header, rows)`: a `|` line followed by its `|---` rule,
    then the `|` lines after it up to the first line that is not one."""
    lines = text.splitlines()
    tables = []
    i = 0
    while i + 1 < len(lines):
        if lines[i].startswith("|") and lines[i + 1].startswith("|---"):
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(lines[j])
                j += 1
            tables.append((lines[i], rows))
            i = j
        else:
            i += 1
    return tables


@unittest.skipUnless(
    os.name == "posix", "the hostile fixture name is a file name Windows cannot hold"
)
class DisplaySafeSweepTest(unittest.TestCase):
    """Operator- and environment-supplied strings, through `main` to the report and the run log
    (post-wave follow-up of gh-ocannl-1273): a fixture's file name, the workload a runner names
    after it, a cache path from the environment and a knob value an ambient variable set. Each
    carries a `|`, a backtick and a newline; every one must reach the report's tables without
    adding a column or a line, and the run log without adding a line or opening a code span.

    Only the cell is faked, as in `OcannlFlagSweepTest`; the failure note is the real cache note.
    """

    NAME = "odd|work`load\nTAIL1"
    KNOB = "x|y`z\nTAIL2"
    RAW = ("|work", "`load", "\nTAIL", "x|y", "y`z", "c|a", "a`che")

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        fixtures = self.dir / "fixtures"
        fixtures.mkdir()
        (fixtures / fixture_digest.DIGEST_FILE).write_bytes(
            (HERE / "fixtures" / fixture_digest.DIGEST_FILE).read_bytes()
        )
        # Forward-only, so the requested f16-static gate leg is NOT APPLICABLE and fills that
        # table too.
        header = json.dumps({"__metadata__": {"model": "gpt", "mode": "infer"}}).encode()
        (fixtures / f"{self.NAME}.safetensors").write_bytes(
            len(header).to_bytes(8, "little") + header
        )
        self.cachedb = self.dir / "c|a`che\nTAIL3" / "cache.db"

    def leaking(self):
        return {"tf32_matmuls": {"value": self.KNOB, "source": "environment"}}

    def fake_cell(self, label, cmd, env=None, on_checkpoint=None, **_):
        if "ocannl/cc/materialized" in label:
            checkpoint = orchestrate.last_checkpoint(
                golden_checkpoint_line("mlp_wide", parity="running")
            )
            checkpoint.update(workload=self.NAME, regime_knobs=self.leaking())
            on_checkpoint(checkpoint)
            return None, orchestrate.quarantine_tinygrad_cache({"CACHEDB": str(self.cachedb)})
        row = cell("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])
        row.update(workload=self.NAME, profile=None, regime_knobs=self.leaking(), searched=False)
        return row, None

    def sweep(self, refusal=SystemExit):
        environ = {k: v for k, v in os.environ.items() if not k.startswith("OCANNL_")}
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.dict(os.environ, environ, clear=True))
            stack.enter_context(unittest.mock.patch.object(orchestrate, "HERE", self.dir))
            stack.enter_context(
                unittest.mock.patch.object(orchestrate, "RESULTS_DIR", self.dir / "results")
            )
            stack.enter_context(unittest.mock.patch.object(orchestrate, "run_cell", self.fake_cell))
            stack.enter_context(
                unittest.mock.patch.object(orchestrate, "install_termination_handler", lambda: None)
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    sys,
                    "argv",
                    ["orchestrate.py", "--only", "ocannl", "--gpu", "none", "--skip-build",
                     "--no-fixture-digest-check", "--skip-box-check", "--materialized", "--tuned",
                     "--precision", "f16-static",
                     "--skip-cell", f"{self.NAME}/cc/tuned"],
                )
            )
            stack.enter_context(contextlib.redirect_stdout(out))
            # A regime mismatch and a runner failure: the sweep fails, after writing the report.
            with self.assertRaises(refusal) as failed:
                orchestrate.main()
        if refusal is SystemExit:
            self.assertEqual(failed.exception.code, 1)
        return out.getvalue(), (self.dir / "results" / "report.md").read_text()

    def test_a_gate_line_built_from_a_raw_value_is_refused(self):
        # The planted raw interpolation (gh-ocannl-1355): a gate label that skipped `display_value`
        # for its workload. Gate labels reach the run log only, so the report's table refusal
        # cannot catch it; the line refusal does, and the report is already written.
        def raw_label(result):
            return f"{result['workload']} {result['framework']}/{result['backend']}"

        with unittest.mock.patch.object(orchestrate, "gate_label", raw_label):
            stdout, report_text = self.sweep(refusal=display_safe.UnsafeLogLine)
        self.assertIn("**REGIME MISMATCH**", report_text)
        self.assertNotIn("REGIME GATE", stdout)

    def test_report_tables_keep_their_columns_and_the_log_its_lines(self):
        stdout, report_text = self.sweep()

        tables = {header: rows for header, rows in markdown_tables(report_text)}
        # Every table the sweep can fill is filled: the results, NOT APPLICABLE, skipped, failures.
        self.assertEqual(len(tables), 4, report_text)
        for header, rows in tables.items():
            with self.subTest(header=header):
                self.assertEqual(len(rows), 1, rows)
                for row in rows:
                    self.assertEqual(row.count("|"), header.count("|"), row)
        shown = orchestrate.display_value(self.NAME)
        failure = tables["| cell | why | saved losses vs reference (diagnostic, never acceptance) |"]
        self.assertTrue(failure[0].startswith(f"| {shown} ocannl/cc/materialized |"), failure)
        self.assertIn("does not exist; nothing to quarantine", failure[0])
        self.assertIn(f"no usable reference row for `{shown}`", failure[0])
        self.assertIn(f"tf32_matmuls={orchestrate.display_value(self.KNOB)} (environment)",
                      failure[0])
        self.assertIn(f"\n## {shown}\n", report_text)
        [measured] = [rows for header, rows in tables.items() if header.startswith("| framework")]
        self.assertIn("**REGIME MISMATCH**", measured[0])
        for text in (report_text, stdout):
            for raw in self.RAW:
                self.assertNotIn(raw, text)
        # The run log names the fixture, the failure and the regime gate, each on its own line.
        [line] = [l for l in stdout.splitlines() if l.startswith("fixture ")]
        self.assertTrue(
            line.startswith(f"fixture {orchestrate.display_value(self.NAME + '.safetensors')}: "),
            line,
        )
        for prefix in ("  - ", "REGIME GATE", "NOT APPLICABLE", "SKIPPED"):
            with self.subTest(prefix=prefix):
                [line] = [l for l in stdout.splitlines() if l.startswith(prefix)]
                self.assertIn(shown, line)



class TableRowTest(unittest.TestCase):
    """Every report table row is built by `display_safe.table_row`, which refuses a raw `|` or line
    break at render time (post-wave follow-up of gh-ocannl-1273), in every report-writing script
    (gh-ocannl-1355): display safety checked where the row is rendered, not remembered at each
    interpolation."""

    # The builders themselves, and `display_value`, which spells the `|` it escapes.
    BUILDERS = ("table_row", "table_lines", "display_value")

    @staticmethod
    def breaks_a_cell(char):
        return char == "|" or len(f"a{char}b".splitlines()) != 1

    def test_a_row_renders_its_cells_and_an_empty_one_as_a_blank(self):
        self.assertEqual(display_safe.table_row("a", "", "b c"), "| a | | b c |")
        self.assertEqual(
            display_safe.table_lines(["x", "y"], [("1", "2")]),
            ["| x | y |", "|---|---|", "| 1 | 2 |"],
        )
        self.assertEqual(display_safe.table_lines(["x", "y"], [], align="lr")[1], "|---|---:|")
        for align in ("l", "lx", "lrl"):
            with self.subTest(align=align):
                with self.assertRaises(display_safe.UnsafeTableRow):
                    display_safe.table_lines(["x", "y"], [], align=align)

    def test_it_refuses_exactly_the_characters_that_break_a_cell(self):
        # Derived, not restated: the refusal set is `|` plus every character `str.splitlines` (and
        # so `markdown_tables`, and any line-based reader of the report) breaks a line at.
        refused = [
            chr(c) for c in range(0x110000) if display_safe._TABLE_CELL_BREAKERS_RE.match(chr(c))
        ]
        breaking = [chr(c) for c in range(0x110000) if self.breaks_a_cell(chr(c))]
        self.assertEqual(refused, breaking)
        for char in breaking:
            with self.subTest(char=char):
                with self.assertRaises(display_safe.UnsafeTableRow) as refusal:
                    display_safe.table_row("fine", f"odd{char}value")
                self.assertIn("table cell 1", str(refusal.exception))
                self.assertIn("display_value", str(refusal.exception))
                # The remedy the refusal names is sufficient: the shown value renders in one cell.
                row = display_safe.table_row("fine", display_safe.display_value(f"odd{char}value"))
                self.assertEqual(row.count("|"), 3, row)
                self.assertEqual(len(row.splitlines()), 1, row)

    def test_a_code_span_cannot_be_closed_early(self):
        self.assertEqual(display_safe.code_span("gpt2_mini_train"), "`gpt2_mini_train`")
        shown = display_safe.code_span("a`b|c")
        self.assertEqual(shown.count("`"), 2, shown)
        self.assertEqual(json.loads(shown[1:-1]), "a`b|c")

    def test_it_refuses_a_non_string_cell_and_a_row_of_the_wrong_width(self):
        with self.assertRaises(TypeError):
            display_safe.table_row("a", 1.5)
        for row in (("1",), ("1", "2", "3")):
            with self.subTest(row=row):
                with self.assertRaises(display_safe.UnsafeTableRow):
                    display_safe.table_lines(["x", "y"], [row])

    def test_a_raw_value_in_a_row_fails_the_report_and_writes_nothing(self):
        # The planted raw interpolation: a failure label that skipped `display_value`.
        failures = [orchestrate.Failure("odd|work ocannl/cc/default", "exit 1")]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(display_safe.UnsafeTableRow):
                    orchestrate.report([], out, (), failures)
            # Refused before results.jsonl or report.md was written, as the report's other
            # refusals are.
            self.assertEqual(list(out.iterdir()), [])

    def test_no_table_row_is_written_outside_the_builders(self):
        sources = report_sources()
        # The population is every shipped script: the drivers named here are the ones that wrote
        # rows by hand before gh-ocannl-1355, so a scan that lost them would pass vacuously.
        for name in ("orchestrate.py", "gh1002_cells.py", "gh1051_cells.py", "gh1133_summary.py",
                     "gh1181_cells.py", "gpt2_bucket.py", "gh1180_tie_ab.sh:EOF"):
            self.assertIn(name, sources)
        found = {
            name: rows
            for name, text in sources.items()
            for rows in [table_row_literals(text, self.BUILDERS)]
            if rows
        }
        self.assertEqual(found, {}, "build report table rows with display_safe.table_row")

    def test_the_scan_finds_a_planted_row(self):
        for planted in (
            'def report(lines, x):\n    lines.append(f"| {x} | y |")\n',
            'def report(lines, x):\n    lines.append("\\n| a | " + x + " |")\n',
            'def report(lines):\n    lines.append("|---|---|")\n',
            'def report(lines):\n    lines.append("| --- | --- |")\n',
            'print("| %s | %d |" % (name, n))\n',
            'print("|" + "|".join(cells) + "|")\n',
            'print("\\n| ".join(rows))\n',
        ):
            with self.subTest(planted=planted):
                self.assertTrue(table_row_literals(planted, self.BUILDERS))
        for clean in (
            'def table_row(*cells):\n    return "| " + " | ".join(cells) + " |"\n',
            'def f():\n    """| a | b |"""\n',
            'def f(a, b):\n    return f"{a} | {b}"\n',
            'pattern = "(%s)" % "|".join(names)\n',
        ):
            with self.subTest(clean=clean):
                self.assertEqual(table_row_literals(clean, self.BUILDERS), [])

    def test_the_scan_reaches_a_drivers_inline_python(self):
        # Negative control at the population: a row planted in a shell driver's heredoc is found.
        script = "summary() {\n  python3 - \"$1\" <<'EOF'\nprint('| a | b |')\nEOF\n}\n"
        sources = shell_heredocs("planted.sh", script)
        self.assertEqual(list(sources), ["planted.sh:EOF"])
        self.assertTrue(table_row_literals(sources["planted.sh:EOF"], self.BUILDERS))


class LogLineTest(unittest.TestCase):
    """A run-log line naming an operator- or runner-supplied value is built by
    `display_safe.log_line`, which refuses a raw line break (gh-ocannl-1355): a fixture's name
    cannot forge a line a reader takes as the driver's own."""

    def test_it_refuses_exactly_the_characters_that_break_a_line(self):
        breaking = [chr(c) for c in range(0x110000) if len(f"a{chr(c)}b".splitlines()) != 1]
        refused = [c for c in map(chr, range(0x110000)) if display_safe._LINE_BREAKERS_RE.match(c)]
        self.assertEqual(refused, breaking)
        for char in breaking:
            with self.subTest(char=char):
                with self.assertRaises(display_safe.UnsafeLogLine) as refusal:
                    display_safe.log_line(f"fixture odd{char}name")
                self.assertIn("display_value", str(refusal.exception))
                line = display_safe.log_line(
                    "fixture " + display_safe.display_value(f"odd{char}name"))
                self.assertEqual(len(line.splitlines()), 1, line)
        # `|` and the backtick are a table's and a code span's concern, not a line's.
        self.assertEqual(display_safe.log_line("a | b `c`"), "a | b `c`")
        self.assertEqual(display_safe.log_lines(["a", "b"]), "a\nb")
        with self.assertRaises(display_safe.UnsafeLogLine):
            display_safe.log_lines(["a", "b\nPARITY GATE: all cells passed"])
        with self.assertRaises(TypeError):
            display_safe.log_line(3)


class ReportScriptsDisplayTest(unittest.TestCase):
    """The drivers that wrote rows by hand before gh-ocannl-1355, rendering an operator-supplied
    string with a `|`, a backtick and a newline in it: every table keeps its columns, the text adds
    no line, and the value is still there, shown."""

    HOSTILE = "odd|lab`el\nTAIL"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def assert_display_safe(self, text):
        tables = markdown_tables(text)
        self.assertTrue(tables, text)
        for header, rows in tables:
            for row in rows:
                self.assertEqual(row.count("|"), header.count("|"), row)
        # No raw piece survives: no forged line, no raw `|` or backtick inside the value.
        self.assertFalse([l for l in text.splitlines() if l.startswith("TAIL")], text)
        for raw in ("odd|lab", "lab`el"):
            self.assertNotIn(raw, text)
        self.assertIn(display_safe.display_value(self.HOSTILE), text)

    def test_gh1051_tables_show_the_memory_label(self):
        # The driver's `memory` label is its operator's third argument.
        for arm in ("true", "false"):
            for prec in ("bfloat16", "single"):
                log = self.dir / f"n64_r1_{arm}_{prec}.log"
                log.write_text("".join(
                    f"{v}   1.5 ms   10.0 GFLOP/s   (x chk 1.0/2.0, y) [Mma_intrinsics 2]\n"
                    for v in ("parallel", "smem", "regtile", "mma_pd1", "mma_pd2")))
                Path(str(log) + ".status").write_text("0\n")
        proc = subprocess.run(
            [sys.executable, str(HERE / "gh1051_cells.py"), "ab", str(self.dir), self.HOSTILE],
            text=True, capture_output=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assert_display_safe(proc.stdout)

    def test_gh1181_summary_shows_the_box_and_workload(self):
        import gh1181_cells

        arms = ["ocannl-master-exact", "torch-exact-eager"]
        (self.dir / "env.json").write_text(json.dumps(
            {"box": self.HOSTILE, "backend": "cuda", "fixtures": [self.HOSTILE], "arms": arms}))
        rows = [{"workload": self.HOSTILE, "arm": arm, "repeat": rep, "quote_ok": True,
                 "step_ms": {"p10": p50, "p50": p50, "p90": p50}, "compile_s": 1.0,
                 "parity_max_rel": 0.0}
                for rep in (0, 1) for arm, p50 in zip(arms, (2.0, 4.0))]
        (self.dir / "timings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.dir / "failures.jsonl").write_text("")
        gh1181_cells.summarize(self.dir)
        text = (self.dir / "summary.md").read_text()
        self.assert_display_safe(text)
        self.assertIn("| ocannl-master-exact / torch-exact-eager | 0.50 |", text)


HEREDOC_START = re.compile(r"python3\b[^\n]*?<<'?(\w+)'?")


def shell_heredocs(name, text):
    """The inline Python of a shell driver: every heredoc fed to `python3`, as
    {"<script>:<tag>": source} (a repeated tag numbers its later bodies `<tag>#2`, ...)."""
    lines = text.splitlines()
    found = {}
    for i, line in enumerate(lines):
        m = HEREDOC_START.search(line)
        if not m:
            continue
        tag = m.group(1)
        end = next(j for j in range(i + 1, len(lines)) if lines[j] == tag)
        key, n = f"{name}:{tag}", 1
        while key in found:
            n += 1
            key = f"{name}:{tag}#{n}"
        found[key] = "\n".join(lines[i + 1:end]) + "\n"
    return found


def report_sources():
    """Every shipped script that could write a report, as {name: Python source}: each `*.py` here
    and under runners/ (the population of ResultLineReaderInventoryTest, minus the builders' own
    module) and the inline Python of each `*.sh`."""
    sources = {}
    for name, text in ResultLineReaderInventoryTest.shipped_sources().items():
        if name.endswith(".sh"):
            sources.update(shell_heredocs(name, text))
        elif name != "display_safe.py":
            sources[name] = text
    return sources


def table_row_literals(source, builders):
    """The string literals in `source` that open a Markdown table line (`| ...`, or a `|---` /
    `| ---` rule) outside the functions named `builders` and outside docstrings: each is a table
    row written around `table_row`, so its cells are never checked."""
    import ast

    tree = ast.parse(source)
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in builders:
            skip.update(id(n) for n in ast.walk(node))
        # Docstrings document rows; they do not write them.
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            skip.add(id(node.body[0].value))
        # A `.join` separator with no line break in it sits between two pieces of one line, so it
        # opens none (`"|".join(names)`, a regex alternation); what precedes the joined text is
        # still scanned (`"| " + " | ".join(cells)` is caught by its `"| "`).
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
            and len(f"a{node.func.value.value}b".splitlines()) == 1
        ):
            skip.add(id(node.func.value))
    found = []
    # Breadth-first, so an f-string is met before its literal pieces: only its FIRST piece can open
    # a line (`f"platform: {p} | commit"` does not).
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.JoinedStr):
            skip.update(id(v) for v in node.values)
            first = node.values[0] if node.values else None
            text = first.value if isinstance(first, ast.Constant) else ""
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        else:
            continue
        if text.lstrip().startswith("|") or re.search(r"\|\s*---", text):
            found.append(f"line {node.lineno}: {text!r}")
    return found


if __name__ == "__main__":
    unittest.main()
