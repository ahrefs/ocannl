"""gh-ocannl-1767: a bench text line has one reader, in bench_output.py.

bench_output.FORMATS names each format's producers, with a token each must still print, and an
anchor: a pattern naming the format in code that reads it. Every other script here whose reading
code -- a regex, a string split or prefix test, grep/sed/awk/cut -- names an anchor is refused,
since that line's reader belongs in bench_output, where test/operations/bench_line_readers holds it
to lines the producers print. A script that WRITES such lines for a test (test_*.py, test-*.sh,
test/) is not a reader; comments are prose. tools/ held no reader when this was written, and a scan
reaching past benchmarks/ is a repository-wide scan, which test/operations/env_var_deps requires to
be an OCaml scanner on `@scans`.
"""
import re
import unittest
from pathlib import Path

import bench_output

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent

CONSTRUCT = re.compile(r"\bre\.\w+\(|\.(?:startswith|split|rsplit|partition|find|index|removeprefix)\("
                       r"|\b(?:grep|sed|awk|cut)\b|=~")
ANCHORS = [(name, re.compile(f["anchor"])) for name, f in bench_output.FORMATS.items()]


def readers_in(lines):
    """(line number, format) of each code line naming an anchor where it, or the code line before
    it (a pattern continued from `re.compile(`), reads text."""
    found, prev = [], ""
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        for name, anchor in ANCHORS:
            if anchor.search(line) and (CONSTRUCT.search(line) or CONSTRUCT.search(prev)):
                found.append((i, name))
        if s:
            prev = line
    return found


def writes_lines(path):
    rel = path.relative_to(HERE)
    return path.name.startswith(("test_", "test-")) or rel.parts[0] == "test"


class BenchLineReaderScanTest(unittest.TestCase):
    def test_every_format_names_what_its_producer_prints(self):
        rows = [(name, path, token) for name, f in bench_output.FORMATS.items()
                for path, token in f["producers"].items()]
        self.assertTrue(rows)
        for name, path, token in rows:
            self.assertIn(token, (ROOT / path).read_text(), f"{name}: {path}")

    def test_every_anchor_names_bench_outputs_own_reader(self):
        # An anchor that matches nothing in bench_output itself would refuse nothing anywhere.
        own = [line for line in (HERE / "bench_output.py").read_text().splitlines()
               if not line.strip().startswith("#")]
        for name, anchor in ANCHORS:
            self.assertTrue(any(anchor.search(line) for line in own), name)

    def test_the_scan_finds_a_reader_and_passes_prose(self):
        self.assertEqual(readers_in(['X = re.compile(', '    r"^(\\w+) ([0-9.]+) ms GFLOP/s")']),
                         [(2, "gebp_timing")])
        self.assertEqual(readers_in(['n=$(grep -o "$R seg 0/[0-9]*" f)']), [(1, "schedule_launch")])
        self.assertEqual(readers_in(['n = line.startswith("rank-site: pick")']), [(1, "rank_site")])
        self.assertEqual(readers_in(['# re.compile(r"GFLOP/s")', 'print("no `rank-site: pick`")']),
                         [])

    def test_no_bench_line_is_read_outside_bench_output(self):
        sources = sorted(p for ext in ("*.py", "*.sh") for p in HERE.rglob(ext)
                         if p.name != "bench_output.py" and not writes_lines(p))
        self.assertTrue(any(p.name == "gh1180_tie_ab.sh" for p in sources))
        stray = [f"{p.relative_to(HERE)}:{i} ({name})" for p in sources
                 for i, name in readers_in(p.read_text(errors="replace").splitlines())]
        self.assertEqual(stray, [])


if __name__ == "__main__":
    unittest.main()
