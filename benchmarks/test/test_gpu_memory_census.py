"""Fake compute-apps polls joined to Dune-produced shapes without a GPU."""

from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CANARY = "no-census-command-secret"


def csexp(value):
    if isinstance(value, list):
        return b"(" + b"".join(map(csexp, value)) + b")"
    raw = str(value).encode()
    return str(len(raw)).encode() + b":" + raw


def process(pid, program, start, duration, directory="test/operations", target=None):
    target = target or program + ".output"
    fields = [["pid", pid], ["prog", program], ["dir", "_build/default/" + directory],
              ["target_files", [f"_build/default/{directory}/{target}"]],
              ["process_args", [CANARY]]]
    return csexp(["process", "start", start, *fields]) + csexp(
        ["process", "finish", [start, duration], ["exit", "0"], *fields])


@unittest.skipUnless(sys.platform == "linux", "the census requires Linux /proc")
class GpuMemoryCensusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "gpu_memory_census", ROOT / "tools/gpu-memory-census.py")
        cls.census = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.census)

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.run_dir = Path(scratch.name)
        self.binary = self.run_dir / "nvidia-smi"
        self.trace_file = self.run_dir / "fixture.csexp"
        self.polls = Counter()
        self.samples = defaultdict(list)

    def trace(self, *processes, complete=True):
        config = csexp(["config", "init", 0, ["build_dir", "_build"], ["env", [CANARY]]])
        end = csexp(["config", "exit", 100]) if complete else b""
        self.trace_file.write_bytes(config + b"".join(processes) + end)

    def poll(self, output, start=20, end=21, exit_code=0):
        self.binary.write_text(
            "#!/bin/sh\n"
            '[ "$#" = 2 ] && [ "$1" = --query-compute-apps=pid,process_name,used_memory ] '
            '&& [ "$2" = --format=csv,noheader,nounits ] || exit 3\n'
            f"cat <<'EOF'\n{output}\nEOF\nexit {exit_code}\n",
            encoding="utf-8",
        )
        self.binary.chmod(0o700)
        with patch.object(self.census.time, "time_ns", side_effect=[start, end]):
            self.census.sample(str(self.binary), self.polls, self.samples)

    def record(self):
        trace_report = self.run_dir / "processes.json"
        if self.trace_file.exists():
            result = subprocess.check_output(
                [ROOT / "tools/action-durations.sh", "--processes-json", self.trace_file])
            trace_report.write_bytes(result)
        self.census.write_sidecar(self.run_dir, trace_report, 0.25, self.polls, self.samples)
        text = (self.run_dir / self.census.SIDECAR).read_text(encoding="utf-8")
        self.assertNotIn(CANARY, text)
        return json.loads(text)

    def test_covers_every_traced_action_and_only_attributes_owned_samples_in_its_lifetime(self):
        pid = os.getpid()
        foreign = pid + 10_000_000
        self.trace(process(pid, "long.exe", 10, 30),
                   process(foreign, "foreign.exe", 10, 30),
                   process(pid, "short.exe", 40, 1),
                   process(pid + 1, "ocamlopt.opt", 50, 2, directory="arrayjit/lib"))
        real_stat = Path.stat

        def with_foreign_owner(path, *args, **kwargs):
            if path == Path("/proc") / str(foreign):
                return SimpleNamespace(st_uid=os.geteuid() + 1)
            return real_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", with_foreign_owner):
            self.poll(f"{pid}, {CANARY}, 16\n{foreign}, {CANARY}, 900")
            self.poll(f"{pid}, {CANARY}, 32\n{pid}, {CANARY}, 12", start=24, end=25)
            self.poll(f"{pid}, {CANARY}, 700", start=39, end=41)
        self.assertNotIn(foreign, self.samples)
        record = self.record()
        self.assertEqual(record["sampling_period_seconds"], 0.25)
        self.assertEqual(record["probe_status"], "read")
        self.assertEqual(record["read_polls"], 3)
        self.assertEqual(record["tests"], "direct test processes traced")
        rows = {row["program"]: row for row in record["coverage"]}
        self.assertEqual(rows.pop("long.exe"), {
            "program": "long.exe", "pid": pid, "directory": "test/operations",
            "action": "test/operations/long.exe.output", "status": "sampled",
            "targets": ["test/operations/long.exe.output"],
            "process_peak_mib": 44, "sample_count": 2,
        })
        self.assertEqual({name: row["status"] for name, row in rows.items()},
                         {"short.exe": "unknown", "foreign.exe": "unknown", "ocamlopt.opt": "unknown"})
        for row in rows.values():
            self.assertNotIn("process_peak_mib", row)

    def test_poll_excludes_an_owned_pid_reused_during_the_query(self):
        pid = os.getpid()
        stat_file = Path("/proc") / str(pid) / "stat"
        boot_ns = 1_000_000_000
        start_ticks = 2 * os.sysconf("SC_CLK_TCK")
        real_read_bytes = Path.read_bytes

        def stat_after_poll_started(path):
            if path == stat_file:
                return f"{pid} (reused) S ".encode() + b"0 " * 18 + str(start_ticks).encode()
            return real_read_bytes(path)

        with patch.object(Path, "read_bytes", stat_after_poll_started):
            with patch.object(self.census.time, "clock_gettime_ns", return_value=boot_ns):
                self.poll(f"{pid}, {CANARY}, 512")
        self.assertEqual(self.polls, {"ran": 1})
        self.assertNotIn(pid, self.samples)

    def test_unread_is_neither_empty_nor_zero(self):
        pid = os.getpid()
        self.trace(process(pid, "test.exe", 10, 30))
        self.poll(CANARY, exit_code=1)
        self.poll(f"{pid}, {CANARY}, 32\n{pid}, {CANARY}, [N/A]")
        record = self.record()
        self.assertEqual(record["probe_status"], "UNREAD")
        self.assertEqual(record["poll_outcomes"], {"failed": 1, "unreadable": 1})
        self.assertEqual(record["coverage"][0]["status"], "unknown")
        self.assertNotIn("process_peak_mib", record["coverage"][0])

    def test_cached_trace_with_a_read_empty_probe_says_no_test_ran(self):
        self.trace()
        self.poll("")
        record = self.record()
        self.assertEqual(record["tests"], "no recognized direct test executable ran")
        self.assertEqual(record["trace_status"], "complete")
        self.assertEqual(record["coverage"], [])
        self.assertEqual(record["probe_status"], "read")

    def test_a_config_probe_and_header_generator_do_not_count_as_tests(self):
        self.trace(process(os.getpid(), "host_caps.exe", 10, 10, directory="test/config"),
                   process(os.getpid(), "gen_codec_header.exe", 30, 10, target="bf16_codec.h"))
        record = self.record()
        self.assertEqual(record["tests"], "no recognized direct test executable ran")
        self.assertEqual([row["program"] for row in record["coverage"]],
                         ["host_caps.exe", "gen_codec_header.exe"])

    def test_trace_stopping_between_events_cannot_claim_tests_were_cached(self):
        self.trace(process(os.getpid(), "host_caps.exe", 10, 10, directory="test/config"),
                   complete=False)
        record = self.record()
        self.assertEqual(record["trace_status"], "incomplete")
        self.assertEqual(record["tests"], "unknown from incomplete or unread trace")

    def test_missing_trace_and_no_polls_leave_coverage_unknown(self):
        record = self.record()
        self.assertEqual(record["trace_status"], "UNREAD")
        self.assertEqual(record["tests"], "unknown from incomplete or unread trace")
        self.assertEqual(record["probe_status"], "UNREAD")

    def test_foreground_launcher_uses_the_runner_trace_reader_and_keeps_its_table(self):
        self.trace(process(os.getpid(), "traced.exe", 10, 30))
        fake_runner = self.run_dir / "test-run.sh"
        trace_argument = self.run_dir / "trace-argument"
        timing_table = self.run_dir / "timing-table"
        fake_runner.write_text(
            '#!/bin/sh\n'
            'test "$#" = 3 || exit 2\n'
            f'printf "%s" "$OCANNL_TOOL_CENSUS_JSON" > {shlex.quote(str(trace_argument))}\n'
            f'bash "$OCANNL_TOOL_SLOWEST_READER" -n 5 {shlex.quote(str(self.trace_file))} '
            f'> {shlex.quote(str(timing_table))} || exit 3\n'
            f'unlink {shlex.quote(str(self.trace_file))} || exit 4\n'
            "printf '%s\\n' 'command: dune build @runtest' 'verdict: pass (exit 0)' "
            f"{shlex.quote(f'log:     {self.run_dir}/log')}\n",
            encoding="utf-8",
        )
        fake_runner.chmod(0o700)
        args = SimpleNamespace(period=0.25, nvidia_smi=str(self.binary),
                               runner=["build", "@runtest"])
        stdout = SimpleNamespace(buffer=io.BytesIO())
        with patch.object(self.census, "RUNNER", fake_runner), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            self.assertEqual(self.census.run_batch(args), 0)
        ephemeral = Path(trace_argument.read_text(encoding="utf-8"))
        self.assertFalse(ephemeral.exists())
        self.assertFalse(self.trace_file.exists())
        self.assertIn("traced.exe", timing_table.read_text(encoding="utf-8"))
        report = json.loads((self.run_dir / self.census.SIDECAR).read_text(encoding="utf-8"))
        self.assertEqual(report["trace_status"], "complete")
        self.assertEqual([row["program"] for row in report["coverage"]], ["traced.exe"])
