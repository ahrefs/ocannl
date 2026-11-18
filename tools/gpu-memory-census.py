#!/usr/bin/env python3
"""Run a foreground test-run.sh batch and write gpu-memory-census.json in its record.

Linux only. Run from the worktree with a backend pin, for example:
    env OCANNL_BACKEND=cuda python3 tools/gpu-memory-census.py -- \
        --expect-backend cuda build @test/operations/runtest-online_softmax -j 4
    env OCANNL_BACKEND=hip python3 tools/gpu-memory-census.py --hip-memory-model unified -- \
        --expect-backend hip build @test/operations/runtest-online_softmax

HIP reads resident VRAM on discrete GPUs and resident VRAM plus GTT on
unified GPUs. Supply the memory model of the batch's GPU. If omitted, the
model is unknown and the probe reads both regions.

Input boundary: the foreground runner's digest header locates its own run.
Only action-durations.sh's allowlisted trace fields, probe status, numeric
PID/memory rows and memory field names, and Linux /proc ownership and start
ticks enter the census. CUDA reads compute-apps used_memory. HIP reads
/dev/kfd and /dev/dri/ fd links and amdgpu fdinfo's drm-driver, drm-pdev,
drm-client-id, and drm-resident-vram/gtt or their drm-memory-vram/gtt aliases.
Requested totals, CPU memory, device-wide counters, command arguments,
process environments and probe diagnostics are outside the boundary. The
runner's bounded trace reader exports only process JSON to the launcher.
The runner owns and deletes its Dune-produced raw trace.

Coverage includes every direct process in the trace. The test summary counts
direct .exe processes in test or arrayjit/test with a named test alias,
an inline-test runner's anonymous alias, or an output named for the program.
The shared test/config setup tree contributes coverage rows. Attribution
requires a complete traced lifetime.
"""

import argparse
from collections import Counter, defaultdict
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from gpu_cotenants import probe_drm, probe_nvidia  # noqa: E402

RUNNER = ROOT / "tools/test-run.sh"
TRACE_READER = ROOT / "tools/gpu-memory-census-trace.sh"
SIDECAR = "gpu-memory-census.json"


def owned_before(pid, boot_start_ns):
    """Conservatively exclude processes that vanished or whose PID was reused during the poll."""
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid != os.geteuid():
            return False
        # comm can contain spaces and parentheses. Field 22 is 19 fields after it.
        start_ticks = int((proc / "stat").read_bytes().rsplit(b")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return False
    return (start_ticks + 1) * 1_000_000_000 <= boot_start_ns * os.sysconf("SC_CLK_TCK")


def sample(probe_memory, polls, samples):
    boot_start = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    start = time.time_ns()
    probe, found = probe_memory()
    end = time.time_ns()
    polls[probe["status"]] += 1
    if end < start:
        return
    for pid, fields in found.items():
        if owned_before(pid, boot_start):
            samples[pid].append((start, end, fields["gpu_memory_mib"], fields.get("memory_fields", ())))


def coverage(trace_report, samples, memory_model):
    try:
        data = json.loads(trace_report.read_bytes())
    except (OSError, ValueError):
        return "UNREAD", []
    processes = []
    for row in data["processes"]:
        start, end = row.pop("start_ns"), row.pop("end_ns")
        # A query that straddles a lifetime boundary cannot identify the PID.
        readings = [mib for before, after, mib, _ in samples.get(row["pid"], ())
                    if end is not None and start <= before <= after <= end]
        if memory_model is not None:
            row["memory_model"] = memory_model
        if readings:
            row.update(status="sampled", process_peak_mib=max(readings), sample_count=len(readings))
        else:
            row.update(status="unknown")
        processes.append(row)
    return data["trace_status"], processes


def is_test_process(row):
    program = row["program"]
    directory = row["directory"] + "/"
    if (not program.endswith(".exe") or not directory.startswith(("test/", "arrayjit/test/"))
            or directory.startswith("test/config/")):
        return False
    stem = program[:-len(".exe")]
    for target in row["targets"]:
        if target.startswith("@"):
            alias = target[1:].rsplit("/", 1)[-1]
            if (alias in ("runtest", "train", "slow") or alias.startswith(("runtest-", "train-", "slow-"))
                    or (program == "inline-test-runner.exe" and alias.startswith("(anonymous-"))):
                return True
        elif target.rsplit("/", 1)[-1] in (program + ".output", stem + ".actual", stem + ".output"):
            return True
    return False


def write_sidecar(run_dir, trace_report, period, polls, samples, *, probe="nvidia-smi", memory_model=None):
    trace_status, processes = coverage(trace_report, samples, memory_model)
    has_tests = any(map(is_test_process, processes))
    if has_tests:
        tests = "direct test processes traced"
    elif trace_status == "complete":
        tests = "no recognized direct test executable ran"
    else:
        tests = "unknown from incomplete or unread trace"
    partial_polls = polls.get("partial", 0)
    read = polls.get("ran", 0) + partial_polls
    unread = sum(polls.values()) - read - polls.get("empty", 0)
    record = {
        "sampling_period_seconds": period,
        "probe": probe,
        "probe_status": ("partial" if unread or partial_polls else "read") if read else "UNREAD",
        "read_polls": read,
        "unread_polls": unread,
        "poll_outcomes": polls,
        "trace_status": trace_status,
        "tests": tests,
        "memory": "peak across observed polls for each process (MiB)",
        "coverage": processes,
    }
    if memory_model is not None:
        record["memory_model"] = memory_model
        record["probe_fields"] = sorted({field for readings in samples.values()
                                         for _, _, _, fields in readings for field in fields})
    with tempfile.TemporaryDirectory(dir=run_dir) as staging:
        finished = Path(staging) / SIDECAR
        finished.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        finished.replace(run_dir / SIDECAR)


def run_batch(args):
    polls, samples = Counter(), defaultdict(list)
    backend = os.environ.get("OCANNL_BACKEND") or "cuda"
    if backend == "hip":
        probe, memory_model = "drm-fdinfo", args.hip_memory_model or "unknown"
        regions = ("vram",) if memory_model == "discrete" else ("vram", "gtt")
        probe_memory = partial(probe_drm, "/proc", memory_regions=regions)
    elif backend == "cuda":
        probe, memory_model = "nvidia-smi", None
        probe_memory = partial(probe_nvidia, args.nvidia_smi, require_memory=True)
    else:
        raise ValueError("set OCANNL_BACKEND to cuda or hip for the census")
    sep = args.runner.index("--") if "--" in args.runner else len(args.runner)
    if any(arg.startswith("--tr") for arg in args.runner[:sep]):
        raise ValueError("the census uses test-run.sh's default trace")
    with tempfile.TemporaryFile() as process_json, tempfile.TemporaryFile() as digest:
        # The normal slowest reader runs before test-run.sh removes its raw trace.
        trace_report = Path(f"/proc/{os.getpid()}/fd/{process_json.fileno()}")
        environment = dict(os.environ, OCANNL_TOOL_SLOWEST_READER=str(TRACE_READER),
                           OCANNL_TOOL_CENSUS_JSON=str(trace_report))
        with subprocess.Popen([RUNNER, "run", *args.runner], env=environment,
                              stdout=digest) as runner:
            while runner.poll() is None:
                next_poll = time.monotonic() + args.period
                sample(probe_memory, polls, samples)
                try:
                    runner.wait(timeout=max(0, next_poll - time.monotonic()))
                except subprocess.TimeoutExpired:
                    continue
        digest.seek(0)
        header = [digest.readline() for _ in range(3)]
        digest.seek(0)
        shutil.copyfileobj(digest, sys.stdout.buffer)
        rc = runner.returncode if runner.returncode >= 0 else 128 - runner.returncode
        # The run's own stdout identifies it. A shared "last" pointer could have
        # advanced to another batch by the time the runner's digest is published.
        run_dir = None
        if (header[0].startswith(b"command: dune ") and header[1].startswith(b"verdict: ")
                and header[2].startswith(b"log:     ") and header[2].endswith(b"/log\n")):
            run_dir = Path(os.fsdecode(header[2][len(b"log:     "):-len(b"/log\n")]))
        if run_dir is None or not run_dir.is_absolute():
            print("gpu-memory-census: runner did not identify a run directory", file=sys.stderr)
            return rc or 1
        try:
            write_sidecar(run_dir, trace_report, args.period, polls, samples,
                          probe=probe, memory_model=memory_model)
        except (OSError, ValueError) as error:
            print(f"gpu-memory-census: sidecar failed: {type(error).__name__}", file=sys.stderr)
            return rc or 1
        print(f"GPU memory census: {run_dir / SIDECAR}", file=sys.stderr)
        return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", type=float, default=1.0, help="requested poll interval in seconds")
    parser.add_argument("--nvidia-smi", default="nvidia-smi", metavar="BIN")
    parser.add_argument("--hip-memory-model", choices=("discrete", "unified"),
                        help="HIP GPU memory model (default: unknown)")
    parser.add_argument("runner", nargs=argparse.REMAINDER, help="-- followed by test-run.sh run arguments")
    args = parser.parse_args()
    if not math.isfinite(args.period) or args.period <= 0:
        parser.error("--period must be a finite positive number")
    if args.runner[:1] != ["--"] or len(args.runner) < 2:
        parser.error("give test-run.sh run arguments after --")
    if sys.platform != "linux":
        parser.error("the census requires Linux /proc")
    args.runner.pop(0)
    try:
        return run_batch(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
