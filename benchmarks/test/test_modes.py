"""gh-ocannl-2308: the cell runner's mode reading (`cells.MODE_RULE`) and its report.

An arm whose cells fall into two modes (a device or process state) makes the pooled paired ratios
mix them: on gfx1151 an identical control read [0.913, 1.090] around a clean 5-6% effect within
each mode (lukstafi/ocannl-staging#2103). The fixture here is a saved run of three arms --
a unimodal reference, a TWO-MODE identical control, and a unimodal control 5% slower -- written as
`manifest.json` plus `cells.jsonl` and re-rendered through `bin_cells.py report`, the path an older
run's directory takes. Its numbers are synthetic, so every ratio the report prints is exact by
construction: the two-mode arm's fast cells (three rounds) are 0.9x the reference's and its slow cells equal
it, and every printed median and range has at most four significant digits (no decimal tie).
The unit cases hold the rule's three refusals (one stray cell, a step within the arm's own effect,
ordinary spread) and its labeling of a reference whose fast mode has one cell.
"""
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cells

HERE = Path(__file__).resolve().parent.parent
ROUNDS = 12
FAST_ROUNDS = (1, 5, 9)


def base_ms(r):
    return 10.0 + 0.01 * r  # evenly spaced: no gap stands out


def fixture_times():
    """{arm: {round: ms}} of the fixture run."""
    base = {r: base_ms(r) for r in range(ROUNDS)}
    two_mode = {r: (0.9 * t if r in FAST_ROUNDS else t) for r, t in base.items()}
    steady = {r: 10.5 + 0.01 * ((5 * r) % ROUNDS) for r in range(ROUNDS)}
    return {"base": base, "twomode": two_mode, "steady": steady}


def write_run(run_dir, times, keep=None):
    arms = list(times)
    manifest = {
        "bench": "synthetic", "backend": "cc", "rounds": ROUNDS, "commit": "n/a (fixture)",
        "executable": None, "executable_sha256": None, "host": "fixture", "system": "fixture",
        "cpu": "fixture", "common_args": [], "common_env": {},
        "arms": [{"label": a, "args": [], "env": {}} for a in arms], "reader": "bench-result",
        "stated_files": {}, "pins": [], "cleared": [], "runtime_controls": {},
        "environment_values": {}, "environment_names": [], "cell_timeout_s": 60.0,
        "orders": cells.design_orders(len(arms), ROUNDS)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    lines = []
    for r, order in enumerate(manifest["orders"]):
        for position, i in enumerate(order):
            arm = arms[i]
            lines.append(json.dumps({
                "cell": len(lines), "round": r, "position": position, "arm": arm, "argv": [],
                "env": {}, "status": 0, "wall_s": 1.0,
                "results": [["step", times[arm][r] * 1e-3, "passed", ""]], "config": {},
                "problems": [], "refused": False, "ok": True}))
    (run_dir / "cells.jsonl").write_text("\n".join(lines[:keep]) + "\n")


def report(times, keep=None, edit=None):
    """`bin_cells.py report` over the fixture run of `times` (its first `keep` cells; `edit`
    rewrites the saved records' text first)."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_run(run_dir, times, keep)
        if edit:
            path = run_dir / "cells.jsonl"
            path.write_text(edit(path.read_text()))
        proc = subprocess.run([sys.executable, str(HERE / "bin_cells.py"), "report", str(run_dir)],
                              text=True, capture_output=True, timeout=120)
    return proc


def section(text, heading):
    """The report's `## heading...` section, up to the next one."""
    start = text.index("\n## " + heading)
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def table_row(text, *lead):
    """The cells of the report's table row starting with `lead`, or None."""
    for line in text.splitlines():
        cells_ = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and tuple(cells_[:len(lead)]) == lead:
            return cells_
    return None


class ModeReportTest(unittest.TestCase):
    def test_the_two_mode_arm_is_named_and_paired_within_its_modes(self):
        proc = report(fixture_times())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # The pooled table mixes the modes: the identical control reads 0.900-1.000.
        self.assertEqual(table_row(section(proc.stdout, "Paired against"), "step", "twomode")[2:5],
                         ["12", "1.000", "0.900-1.000"])
        text = section(proc.stdout, "Modes")
        self.assertIn(cells.MODE_RULE, text)
        # cells, fast n/median/range, slow n/median/range, gap, step, reading.
        self.assertEqual(table_row(text, "step", "twomode", "12", "3")[2:],
                         ["12", "3", "9.045", "9.009-9.081", "9", "10.06", "10-10.11", "0.919",
                          "1.101", "BIMODAL"])
        self.assertEqual(table_row(text, "step", "base", "12")[-1], "one mode")
        self.assertEqual(table_row(text, "step", "steady", "12")[-1], "one mode")
        # Within modes: every fast round is the 0.9x, every slow round the identical cell.
        self.assertEqual(table_row(text, "step", "twomode", "fast/-")[3:],
                         ["3", "0.900", "0.900-0.900", "3 of 3"])
        self.assertEqual(table_row(text, "step", "twomode", "slow/-")[3:],
                         ["9", "1.000", "1.000-1.000", "0 of 0"])
        self.assertIsNone(table_row(text, "step", "steady", "fast/-"))

    def test_a_run_without_modes_says_so(self):
        times = fixture_times()
        del times["twomode"]
        proc = report(times)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("No arm is bimodal.", section(proc.stdout, "Modes"))
        self.assertNotIn("BIMODAL |", proc.stdout)

    def test_an_interrupted_run_is_not_clean(self):
        proc = report(fixture_times(), keep=-1)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("the saved run is incomplete: 1 of 36 planned cells have no record",
                      proc.stdout)
        self.assertNotIn("every cell clean", proc.stdout)

    def test_records_before_the_configuration_schema_are_refused(self):
        legacy = '"config": {"backend": {"value": "cc", "source": "commandline", "detail": "-"}}'
        proc = report(fixture_times(), edit=lambda t: t.replace('"config": {}', legacy))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("predate the source/layer/origin schema", proc.stderr)

    def test_a_single_arm_run_is_read_for_modes(self):
        proc = report({"twomode": fixture_times()["twomode"]})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        text = section(proc.stdout, "Modes")
        self.assertEqual(table_row(text, "step", "twomode", "12")[-1], "BIMODAL")
        self.assertNotIn("within modes", text)

    def test_a_missing_run_is_a_usage_error(self):
        proc = subprocess.run([sys.executable, str(HERE / "bin_cells.py"), "report",
                               str(HERE / "no-such-run")], text=True, capture_output=True,
                              timeout=120)
        self.assertEqual(proc.returncode, 2)


class ModeRuleTest(unittest.TestCase):
    def reading(self, times, ref="base"):
        return cells.mode_reading(times, ref)

    def test_one_stray_cell_splits_but_is_no_mode(self):
        base = {r: base_ms(r) for r in range(ROUNDS)}
        stray = {**base, 3: 13.0}
        m = self.reading({"base": base, "stray": stray})["stray"]
        self.assertTrue(m["split"]["splits"])
        self.assertEqual(len(m["split"]["slow"]), 1)
        self.assertFalse(m["bimodal"])
        self.assertIsNone(m["labels"])

    def test_a_step_within_the_arms_own_effect_is_not_flagged(self):
        # Two separated groups 1.047x apart in an arm 2x slower than the reference: the step is
        # smaller than the difference the pooled ratio already reports.
        base = {r: base_ms(r) for r in range(ROUNDS)}
        slow = {r: (20.0 if r % 2 else 21.0) + 0.01 * (r // 2) for r in range(ROUNDS)}
        m = self.reading({"base": base, "slow": slow})["slow"]
        self.assertTrue(m["split"]["splits"])
        self.assertGreater(m["split"]["gap"], cells.MODE_SEPARATION * m["split"]["widths"])
        self.assertFalse(m["bimodal"])

    def test_the_reference_joins_the_modes_with_a_single_fast_cell(self):
        times = fixture_times()
        times["base"][FAST_ROUNDS[0]] = 0.9 * times["base"][FAST_ROUNDS[0]]
        reading = self.reading(times)
        self.assertFalse(reading["base"]["bimodal"])
        self.assertEqual(reading["base"]["labels"][FAST_ROUNDS[0]], "fast")
        self.assertEqual(sum(v == "fast" for v in reading["base"]["labels"].values()), 1)
        rows = dict(cells.paired_by_mode(times["base"], times["twomode"],
                                         reading["base"]["labels"], reading["twomode"]["labels"]))
        self.assertEqual([rows[k]["pairs"] for k in ("fast/fast", "slow/slow", "mixed")], [1, 9, 2])
        # Round 1's two cells are the same 0.9x value: the fast/fast pair is the identity.
        self.assertEqual(rows["fast/fast"]["median"], 1.0)
        self.assertEqual(rows["slow/slow"]["median"], 1.0)

    def test_ordinary_spread_does_not_fire(self):
        # Seeded unimodal draws of twelve rounds per arm, the shapes timing noise takes: none is
        # bimodal (MODE_RULE's claim), while two well separated modes are found.
        rng = random.Random(2308)
        draws = {"normal": lambda: rng.gauss(10.0, 0.2),
                 "lognormal": lambda: 10.0 * math.exp(rng.gauss(0.0, 0.3))}
        for name, draw in draws.items():
            flagged = 0
            for _ in range(300):
                times = {a: {r: draw() for r in range(ROUNDS)} for a in ("base", "a", "b")}
                flagged += sum(m["bimodal"] for m in self.reading(times).values())
            self.assertEqual(flagged, 0, name)
        found = 0
        for _ in range(100):
            times = {a: {r: rng.gauss(10.0, 0.05) + (1.0 if rng.random() < 0.6 else 0.0)
                         for r in range(ROUNDS)} for a in ("base", "a")}
            found += self.reading(times)["a"]["bimodal"]
        self.assertGreater(found, 80)

    def test_an_arm_without_the_variant_sets_no_threshold(self):
        # The reference's threshold is the smallest MEASURED difference: an arm that never timed
        # this variant pairs with nothing and must not lower it to zero. The reference's 1.1x step
        # is below the 2x arm's difference, so it is not bimodal with or without the empty arm.
        ref = {r: (10.0 if r % 3 else 11.0) + 0.001 * r for r in range(ROUNDS)}
        far = {r: 2 * t for r, t in ref.items()}
        for times in ({"base": ref, "far": far}, {"base": ref, "far": far, "none": {}}):
            self.assertFalse(self.reading(times)["base"]["bimodal"], sorted(times))
        self.assertTrue(self.reading({"base": ref})["base"]["bimodal"])

    def test_the_split_takes_the_lowest_of_tied_gaps(self):
        s = cells.mode_split([1.0, 2.0, 3.0])
        self.assertEqual((s["fast"], s["slow"]), ([1.0], [2.0, 3.0]))
        self.assertIsNone(cells.mode_split([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
