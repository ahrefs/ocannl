"""Fake CUDA and HIP polls joined to Dune-produced shapes without a GPU."""

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
            self.census.sample(lambda: self.census.probe_nvidia(str(self.binary), require_memory=True),
                               self.polls, self.samples)

    def record(self, **kwargs):
        trace_report = self.run_dir / "processes.json"
        if self.trace_file.exists():
            result = subprocess.check_output(
                [ROOT / "tools/action-durations.sh", "--processes-json", self.trace_file])
            trace_report.write_bytes(result)
        self.census.write_sidecar(self.run_dir, trace_report, 0.25, self.polls, self.samples, **kwargs)
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

    def test_launcher_requires_an_explicit_backend_pin(self):
        with patch.dict(os.environ, clear=True):
            with self.assertRaisesRegex(ValueError, "set OCANNL_BACKEND"):
                self.census.run_batch(SimpleNamespace())

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
        args = SimpleNamespace(period=0.25, nvidia_smi=str(self.binary), hip_memory_model="discrete",
                               runner=["build", "@runtest"])
        stdout = SimpleNamespace(buffer=io.BytesIO())
        with (patch.dict(os.environ, OCANNL_BACKEND="hip"),
              patch.object(self.census, "RUNNER", fake_runner),
              patch.object(self.census, "probe_drm", return_value=({"status": "empty"}, {})),
              redirect_stdout(stdout), redirect_stderr(io.StringIO())):
            self.assertEqual(self.census.run_batch(args), 0)
        ephemeral = Path(trace_argument.read_text(encoding="utf-8"))
        self.assertFalse(ephemeral.exists())
        self.assertFalse(self.trace_file.exists())
        self.assertIn("traced.exe", timing_table.read_text(encoding="utf-8"))
        report = json.loads((self.run_dir / self.census.SIDECAR).read_text(encoding="utf-8"))
        self.assertEqual(report["trace_status"], "complete")
        self.assertEqual(report["probe"], "drm-fdinfo")
        self.assertEqual(report["memory_model"], "discrete")
        self.assertEqual([row["program"] for row in report["coverage"]], ["traced.exe"])

    def drm_fd(self, pid, fd, target, fields=None):
        proc = self.run_dir / str(pid)
        (proc / "fd").mkdir(parents=True, exist_ok=True)
        (proc / "fd" / str(fd)).symlink_to(target)
        if fields is not None:
            (proc / "fdinfo").mkdir(exist_ok=True)
            (proc / "fdinfo" / str(fd)).write_text(fields, encoding="utf-8")

    def hip_client(self, pid, fd, counters, *, device="0000:01:00.0", client=7):
        self.drm_fd(pid, fd, "/dev/dri/renderD128",
                    f"drm-driver:\tamdgpu\ndrm-pdev:\t{device}\ndrm-client-id:\t{client}\n"
                    + counters + f"\nignored:\t{CANARY}\n")

    def test_discrete_hip_counts_only_resident_vram_once_per_client_and_device(self):
        pid = os.getpid()
        self.drm_fd(pid, 3, "/dev/kfd")
        for fd in (4, 5):
            self.hip_client(pid, fd, "drm-resident-vram: 512 KiB\ndrm-memory-vram: 512 KiB\n"
                            "drm-total-vram: 80 MiB\ndrm-resident-gtt: 21 MiB")
        self.hip_client(pid, 6, "drm-memory-vram: 1048576", device="0000:02:00.0")
        self.hip_client(pid + 1, 4, "drm-resident-vram: 16 MiB")  # graphics-only client
        probe, rows = self.census.probe_drm(self.run_dir, memory_regions=("vram",))
        self.assertEqual(probe["status"], "ran")
        self.assertEqual(set(rows), {pid})
        self.assertEqual(rows[pid]["gpu_memory_mib"], 1.5)
        self.assertEqual(set(rows[pid]["memory_fields"]), {"drm-resident-vram", "drm-memory-vram"})

    def test_unified_hip_accounts_for_gtt_and_leaves_unread_clients_unknown(self):
        pid = os.getpid()
        unread, malformed = pid + 1, pid + 2
        for i in (pid, unread, malformed):
            self.drm_fd(i, 3, "/dev/kfd")
        self.hip_client(pid, 4, "drm-resident-vram: 0\ndrm-memory-gtt: 16384 KiB\n"
                        "drm-resident-cpu: 32 MiB\ndrm-total-gtt: 64 MiB")
        self.hip_client(unread, 4, "drm-resident-vram: 0\ndrm-resident-gtt: 7 MiB")
        self.drm_fd(unread, 5, "/dev/dri/renderD128")  # fdinfo missing during the poll
        self.hip_client(malformed, 4, "drm-resident-vram: 0\ndrm-resident-gtt: N/A\n"
                        "drm-memory-gtt: 64 MiB")

        self.trace(process(pid, "unified.exe", 10, 30), process(unread, "unread.exe", 10, 30))
        with patch.object(self.census.time, "time_ns", side_effect=[20, 21]):
            self.census.sample(
                lambda: self.census.probe_drm(self.run_dir, memory_regions=("vram", "gtt")),
                self.polls, self.samples)
        self.assertEqual(self.polls, {"partial": 1})
        self.assertEqual(set(self.samples), {pid})
        # Neither a poll outside this lifetime nor a co-tenant's poll contributes fields.
        self.samples[pid].append((5, 6, 99, ("drm-memory-vram",)))
        self.samples[pid + 4].append((20, 21, 99, ("drm-resident-gtt",)))
        record = self.record(probe="drm-fdinfo", memory_model="unified")
        self.assertEqual(record["probe"], "drm-fdinfo")
        self.assertEqual(record["probe_status"], "partial")
        self.assertEqual(record["probe_fields"], ["drm-memory-gtt", "drm-resident-vram"])
        sampled, unknown = record["coverage"]
        self.assertEqual(sampled["memory_model"], "unified")
        self.assertEqual(sampled["process_peak_mib"], 16)
        self.assertEqual(unknown["memory_model"], "unified")
        self.assertEqual(unknown["status"], "unknown")
        self.assertNotIn("process_peak_mib", unknown)

    def test_hip_process_without_resident_or_legacy_fields_is_unread(self):
        pid = os.getpid()
        self.drm_fd(pid, 3, "/dev/kfd")
        self.hip_client(pid, 4, "drm-total-vram: 64 MiB\ndrm-total-gtt: 0")
        self.trace(process(pid, "total_only.exe", 10, 30))
        with patch.object(self.census.time, "time_ns", side_effect=[20, 21]):
            self.census.sample(
                lambda: self.census.probe_drm(self.run_dir, memory_regions=("vram", "gtt")),
                self.polls, self.samples)
        record = self.record(probe="drm-fdinfo", memory_model="unknown")
        self.assertEqual(record["probe_status"], "UNREAD")
        self.assertEqual(record["poll_outcomes"], {"unreadable": 1})
        self.assertEqual(record["probe_fields"], [])
        self.assertEqual(record["coverage"][0]["status"], "unknown")

    def test_empty_hip_poll_reads_without_sampling_a_process(self):
        self.trace(process(os.getpid(), "idle.exe", 10, 30))
        self.census.sample(
            lambda: self.census.probe_drm(self.run_dir, memory_regions=("vram",)),
            self.polls, self.samples)
        record = self.record(probe="drm-fdinfo", memory_model="discrete")
        self.assertEqual(record["probe_status"], "read")
        self.assertEqual(record["read_polls"], 1)
        self.assertEqual(record["poll_outcomes"], {"empty": 1})
        self.assertEqual(record["probe_fields"], [])
        self.assertEqual(record["coverage"][0]["status"], "unknown")
