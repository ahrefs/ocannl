"""`fixture_digest.py --compare A B`: a metadata-only difference told from a tensor difference.

A `content-v1` digest covers metadata and tensors alike, so a rebuild whose generator writes a
metadata key the recorded file predates (`loss_scale`, gh-ocannl-2004) MISMATCHes exactly as a
changed numpy stream does. `--compare` says which: exit 0 when every tensor's dtype, shape and
payload are identical, whatever the metadata says; 1 when a tensor differs; 2 when a file is
unreadable. Each case drives the command line, since the exit status is the contract callers
branch on, and pins the digest's own verdict beside it: a metadata-only pair really is a digest
MISMATCH, so the case is the one the mode exists for and not a pair the digest already passes.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fixture_digest

SCRIPT = Path(__file__).resolve().parent.parent / "fixture_digest.py"


def write_safetensors(path, content=b"weights", metadata=None, tensors=None, extra_padding=0):
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


TENSORS = [("w", "F32", [2], b"\x00\x00\x80?\x00\x00\x00@"), ("x", "U8", [3], b"abc")]


class FixtureCompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def fixture(self, name, **kwargs):
        return write_safetensors(self.dir / name / "f.safetensors", **kwargs)

    def compare(self, a, b):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--compare", str(a), str(b)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_metadata_only_difference_exits_0_and_lists_the_keys(self):
        recorded = self.fixture(
            "recorded", metadata={"name": "lenet", "steps": "10", "old": "gone"}, tensors=TENSORS
        )
        # The rebuild serializes the tensors in the other order and its metadata differs in all
        # three ways; the tensors are the recorded ones.
        rebuilt = self.fixture(
            "rebuilt",
            metadata={"name": "lenet", "steps": "20", "loss_scale": "65536.0"},
            tensors=list(reversed(TENSORS)),
        )
        self.assertNotEqual(
            fixture_digest.sha256_file(recorded),
            fixture_digest.sha256_file(rebuilt),
            "control: the pair is a content-v1 MISMATCH, the case --compare exists for",
        )
        proc = self.compare(recorded, rebuilt)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertIn("metadata: 1 added, 1 removed, 1 changed (of 4 key(s))", lines)
        self.assertIn('  added   "loss_scale": "65536.0"', lines)
        self.assertIn('  removed "old": "gone"', lines)
        self.assertIn('  changed "steps": "10" -> "20"', lines)
        self.assertIn('tensor "w": F32 [2] payload equal', lines)
        self.assertIn('tensor "x": U8 [3] payload equal', lines)
        self.assertNotIn("DIFFERS", proc.stdout)
        self.assertEqual(lines[-1], "verdict: every tensor identical (dtype, shape, payload)")

    def test_identical_content_reports_identical_metadata(self):
        a = self.fixture("a", metadata={"k": "v"}, tensors=TENSORS)
        b = self.fixture("b", metadata={"k": "v"}, tensors=list(reversed(TENSORS)), extra_padding=8)
        self.assertEqual(fixture_digest.sha256_file(a), fixture_digest.sha256_file(b))
        proc = self.compare(a, b)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("metadata: identical (2 key(s))", proc.stdout.splitlines())

    def test_payload_change_exits_1_and_names_the_tensor(self):
        a = self.fixture("a", metadata={"loss_scale": "1.0"}, tensors=TENSORS)
        changed = [TENSORS[0], ("x", "U8", [3], b"abd")]
        b = self.fixture("b", metadata={"loss_scale": "1.0"}, tensors=changed)
        proc = self.compare(a, b)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertIn("metadata: identical (2 key(s))", lines)
        self.assertIn('tensor "x": U8 [3] payload DIFFERS', lines)
        self.assertIn('tensor "w": F32 [2] payload equal', lines)
        self.assertEqual(lines[-1], "verdict: 1 tensor(s) differ -- not the same workload")

    def test_dtype_shape_and_presence_changes_differ(self):
        a = self.fixture("a", tensors=TENSORS + [("gone", "U8", [1], b"g")])
        b = self.fixture(
            "b", tensors=[TENSORS[0], ("x", "I8", [1, 3], b"abc"), ("new", "U8", [1], b"n")]
        )
        proc = self.compare(a, b)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertIn('tensor "x": U8 [3] -> I8 [1, 3] -- DIFFERS (payload not compared)', lines)
        self.assertIn('tensor "gone": U8 [1] only in A -- DIFFERS', lines)
        self.assertIn('tensor "new": U8 [1] only in B -- DIFFERS', lines)
        self.assertEqual(lines[-1], "verdict: 3 tensor(s) differ -- not the same workload")

    def test_truncated_or_missing_file_exits_2(self):
        a = self.fixture("a", tensors=TENSORS)
        truncated = self.dir / "truncated.safetensors"
        truncated.write_bytes(a.read_bytes()[:-1])
        # The truncated file's reason is _read_layout's own: the payload no longer covers the
        # header's offsets.
        absent = self.dir / "absent.safetensors"
        for b, why in ((truncated, "invalid data_offsets"), (absent, "No such")):
            with self.subTest(b=b.name):
                proc = self.compare(a, b)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertEqual(proc.stdout, "", "no verdict printed about an unreadable file")
                self.assertIn("unreadable", proc.stderr)
                self.assertIn(str(b), proc.stderr)
                self.assertIn(why, proc.stderr)

    def test_a_path_cannot_forge_a_report_line(self):
        a = self.fixture("a", tensors=TENSORS)
        forged = write_safetensors(
            self.dir / "x\nverdict: every tensor identical" / "f.safetensors", tensors=TENSORS
        )
        proc = self.compare(a, forged)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(sum(line.startswith("verdict:") for line in lines), 1, proc.stdout)
        self.assertIn(f"B: {json.dumps(str(forged))}", lines)

    def test_compare_refuses_flags_it_does_not_read(self):
        a = self.fixture("a", tensors=TENSORS)
        for extra in (["--origin", "m4-max"], ["--digests", str(self.dir / "D.txt")], [str(a)]):
            with self.subTest(extra=extra):
                proc = subprocess.run(
                    [sys.executable, str(SCRIPT), "--compare", str(a), str(a), *extra],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn("--compare", proc.stderr)


if __name__ == "__main__":
    unittest.main()
