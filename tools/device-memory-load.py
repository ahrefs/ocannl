"""Supervisor for device-memory-load.sh. Only the worker imports torch."""

import argparse
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

# Use the benchmark interpreter and process-group policies for this GPU probe too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_venv
import cell_group


NAME = "device-memory-load.sh"
READY = b"ready\n"
HOG = """
import random
import sys
import time
from collections import deque
from contextlib import redirect_stdout

minimum, maximum, keep = map(int, sys.argv[1:])
buffers = deque()
ready = sys.stdout.buffer
with redirect_stdout(sys.stderr):
    try:
        import torch
    except ImportError as exc:
        sys.exit(f"device-memory-load.sh: torch unavailable: {exc}")
    if not torch.cuda.is_available():
        sys.exit("device-memory-load.sh: torch found no CUDA or HIP device")

    def allocate():
        size = random.randint(minimum, maximum) * 1024 * 1024
        return torch.empty(size, dtype=torch.uint8, device="cuda")

    for _ in range(keep):
        buffers.append(allocate())

    ready.write(%s)
    ready.flush()
    while True:
        buffers.popleft()
        buffers.append(allocate())
        torch.cuda.empty_cache()
        time.sleep(0.05)
""" % repr(READY)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=NAME, usage="%(prog)s [options] -- CMD [ARGS...]"
    )
    parser.add_argument("--min-mib", type=positive_int, default=16)
    parser.add_argument("--max-mib", type=positive_int, default=256)
    parser.add_argument("--keep", type=positive_int, default=4, help="live buffers (default: 4)")
    parser.add_argument(
        "--ready-timeout", type=positive_int, default=30, metavar="SECONDS",
        help="startup deadline including torch import and initial allocations (default: 30)",
    )
    if "--" in argv:
        separator = argv.index("--")
        options = parser.parse_args(argv[:separator])
        command = argv[separator + 1:]
    else:
        options = parser.parse_args(argv)
        command = []
    if not command:
        parser.error("expected -- CMD [ARGS...]")
    if options.min_mib > options.max_mib:
        parser.error("--min-mib must be <= --max-mib")
    return options, command


def refuse(reason):
    raise SystemExit(f"{NAME}: refused: {reason}")


def wait_ready(hog, timeout):
    deadline = time.monotonic() + timeout
    received = b""
    with selectors.DefaultSelector() as selector:
        selector.register(hog.stdout, selectors.EVENT_READ)
        while received != READY:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                refuse(f"hog did not become ready within {timeout}s")
            chunk = os.read(hog.stdout.fileno(), len(READY) - len(received))
            if not chunk:
                refuse("hog exited before reporting readiness")
            received += chunk
            if not READY.startswith(received):
                refuse("hog sent an invalid readiness line")
    if hog.poll() is not None:
        refuse("hog exited at startup")


def reap(group, label, grace):
    result = cell_group.terminate(group, grace=grace)
    if not result.reaped or result.observation is not cell_group.GONE:
        raise cell_group.CleanupFailed(f"{NAME}: could not reap {label} group {group.pgid}")


def run_command(argv, hog, cancellation):
    try:
        command = cell_group.spawn(argv)
    except OSError as exc:
        print(f"{NAME}: {exc}", file=sys.stderr)
        return 127 if isinstance(exc, FileNotFoundError) else 126

    try:
        with cancellation.cancellable():
            while True:
                try:
                    status = command.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    status = None
                if hog.poll() is not None:
                    refuse(f"hog exited during CMD (status {hog.returncode})")
                if status is not None:
                    return 128 - status if status < 0 else status
    finally:
        # Give a signal-aware command time to unwind before escalating its group.
        reap(command, "CMD", grace=2)


def main(argv):
    options, command = parse_args(argv)
    if os.name != "posix":
        refuse("this helper requires POSIX process groups")
    slot = os.environ.get("FLEET_SLOT_HELD", "").split()
    if slot and slot[-1] != "gpu":
        refuse("the enclosing fleet slot must be a GPU slot")

    python, route = bench_venv.resolve()
    cancellation = cell_group.CancellationDeferral(
        NAME, cleanup_message="the managed process groups were cleaned first"
    )
    cancellation.install()
    with cancellation.deferring():
        try:
            hog = cell_group.spawn(
                [str(python), "-c", HOG, str(options.min_mib), str(options.max_mib), str(options.keep)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            )
        except OSError as exc:
            refuse(f"cannot start benchmark Python ({route}): {exc}")
        try:
            with cancellation.cancellable():
                wait_ready(hog, options.ready_timeout)
            return run_command(command, hog, cancellation)
        finally:
            reap(hog, "hog", grace=0.5)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
