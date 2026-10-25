#!/usr/bin/env python3
"""The GPU co-tenants of a kprof cell (gh-ocannl-1521), stdlib only: which processes were on the
box's GPU just before the cell started and just after it ended.

gh-ocannl-1392's HIP cell whose in-step durations summed to 108% of its step stayed unexplained
partly because nothing recorded what else was on the GPU (gh-ocannl-1444): a shared-box co-tenant
was the likeliest reading and could not be a finding. benchmarks/gh1133_cells.sh now runs `record`
around every kprof cell -- `opened` before the cell's tracer and sampler start, `closed` after the
cell and its sampler have ended -- appending one JSON line per snapshot to OUT/CELL.cotenants.jsonl,
and benchmarks/kernel_attribution.py prints `summary` of that file under the cell's attribution.

A line's fields follow the fleet's co-tenant records (lukstafi/ludics-lite#566, the
`execution hold --request` snapshots), so a reader can line the two up:

- `cell`: the cell's name (OUT/CELL.out's stem); `box`: the run's `--box`, or null; `host`: this
  machine's hostname;
- `event`: `opened` or `closed`; `at`: UTC, ISO 8601;
- `probes`: one `{probe, status, detail}` per source below, `status` `ran`, `absent` (no such
  source on this box) or `failed`;
- `gpu_processes`: one entry per (probe, pid): `pid`, `user` and `command` (`ps`'s `args`, cut to
  COMMAND_WIDTH characters; both empty for a pid `ps` no longer finds), `probe`, and what that probe
  reports, as it reports it: `gpu_memory_mib` (nvidia-smi), `gpu_time_ns` (the client's cumulative
  busy time, drm-fdinfo), `driver` and `pdev` (drm-fdinfo).

WHAT IT RECORDS, AND WHAT IT DOES NOT (the boundary). Presence at two instants per cell, nothing in
between: a co-tenant that starts and ends inside the cell is not seen, and no usage is attributed to
the cell's interval (a reader compares the two snapshots, e.g. a client's gpu_time_ns). Nothing is
left out as the cell's own: at `opened` the cell has not started, and at `closed` its process group
has been waited for, so every listed process ran outside the cell -- a tracer's leftover daemon
included, listed by its command line. Another user's processes are seen only through nvidia-smi;
drm-fdinfo counts them as unreadable in its `detail`. drm-fdinfo lists every DRM client of every
card (its `pdev` says which), graphics clients included, which on a box with an integrated GPU
beside the measured one includes the other card's clients.

THE PROBES, a fail-closed allowlist: each reads exactly this, and nothing else counts as a GPU
process.

- `nvidia-smi` (`--nvidia-smi`, default `nvidia-smi` on PATH): `--query-compute-apps=
  pid,process_name,used_memory --format=csv,noheader,nounits`. A line counts only when its first
  field is a decimal pid; the memory is its last field when that is a decimal number; any other
  line is counted in `detail` and skipped. CUDA contexts only.
- `drm-fdinfo` (`--drm-proc`, default `/proc`; Linux): a process is a DRM client when one of
  `<proc>/<pid>/fd/<n>` links under `/dev/dri/`; its `<proc>/<pid>/fdinfo/<n>` gives `drm-driver`,
  `drm-pdev`, and per `drm-pdev` and `drm-client-id` (counted once however many descriptors share
  it) the `drm-engine-<name>: <n> ns` lines, summed into `gpu_time_ns`. The AMD path (minix, tuf).

Every probe is bounded by PROBE_TIMEOUT, and one that cannot run is a `failed` entry: the record is
evidence, and its absence must not fail the cell. A snapshot in which no probe ran is UNREAD, never
"no co-tenants" (`summary` says so).

Subcommands:
  record [--box BOX] [--nvidia-smi BIN] [--drm-proc DIR] FILE CELL EVENT
                 append one snapshot to FILE; always exits 0 unless the arguments are malformed
  summary FILE   print the record's one-line summary (what kernel_attribution.py prints)
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_safe import display_value  # noqa: E402
import strict_json  # noqa: E402

PROBE_TIMEOUT = 15  # seconds per probe; nvidia-smi answers in about one
COMMAND_WIDTH = 300  # characters of a process's args kept in the record
SHOWN = 8  # processes a summary names per snapshot before "+<n> more"
EVENTS = ("opened", "closed")


def run_bounded(argv):
    """(status, stdout) of argv, 127 when it cannot be started and 124 when it times out."""
    try:
        done = subprocess.run(list(argv), capture_output=True, timeout=PROBE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return 124, ""
    except OSError:
        return 127, ""
    return done.returncode, done.stdout.decode("utf-8", "replace")


def probe_nvidia(binary):
    """(probe entry, {pid: fields}) from nvidia-smi's compute-apps list."""
    path = shutil.which(binary)
    if path is None:
        return {"probe": "nvidia-smi", "status": "absent", "detail": "no %s" % binary}, {}
    rc, out = run_bounded([path, "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"])
    if rc != 0:
        first = next((line.strip() for line in out.splitlines() if line.strip()), "")
        return {"probe": "nvidia-smi", "status": "failed",
                "detail": "exit %d" % rc + (": " + first[:200] if first else "")}, {}
    found, skipped = {}, 0
    for line in out.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if not (fields[0].isdecimal() and fields[0].isascii()):
            skipped += 1
            continue
        # One row per process AND GPU: a process on two GPUs is one entry, its memory summed.
        entry = found.setdefault(int(fields[0]), {})
        if len(fields) >= 3 and fields[-1].isdecimal() and fields[-1].isascii():
            entry["gpu_memory_mib"] = entry.get("gpu_memory_mib", 0) + int(fields[-1])
    detail = "skipped %d unreadable line(s)" % skipped if skipped else ""
    return {"probe": "nvidia-smi", "status": "ran", "detail": detail}, found


ENGINE = re.compile(r"drm-engine-[A-Za-z0-9_-]+:\s*([0-9]+)\s*ns\s*$")


def probe_drm(root):
    """(probe entry, {pid: fields}) from the DRM clients' fdinfo under root."""
    try:
        pids = [e for e in os.listdir(root) if e.isdecimal() and e.isascii()]
    except OSError:
        return {"probe": "drm-fdinfo", "status": "absent", "detail": "no %s" % root}, {}
    found, unreadable = {}, 0
    for pid in pids:
        directory = os.path.join(root, pid, "fd")
        try:
            descriptors = os.listdir(directory)
        except PermissionError:
            unreadable += 1
            continue
        except OSError:  # ended meanwhile, or a kernel thread
            continue
        client, drivers, devices, clients = False, set(), set(), {}
        for fd in descriptors:
            try:
                target = os.readlink(os.path.join(directory, fd))
            except OSError:
                continue
            if not target.startswith("/dev/dri/"):
                continue
            client = True
            try:
                with open(os.path.join(root, pid, "fdinfo", fd), encoding="utf-8",
                          errors="replace") as f:
                    lines = f.read().splitlines()
            except OSError:
                continue
            client_id, device, busy = "fd " + fd, "", 0
            for line in lines:
                key, _, value = line.partition(":")
                if key == "drm-driver" and value.strip():
                    drivers.add(value.strip())
                elif key == "drm-client-id" and value.strip():
                    client_id = value.strip()
                elif key == "drm-pdev" and value.strip():
                    device = value.strip()
                matched = ENGINE.match(line)
                if matched:
                    busy += int(matched.group(1))
            if device:
                devices.add(device)
            # A client id is unique only within its device; descriptors sharing it count once.
            clients[(device, client_id)] = max(busy, clients.get((device, client_id), 0))
        if not client:
            continue
        entry = {"driver": ",".join(sorted(drivers)), "pdev": ",".join(sorted(devices))}
        if any(clients.values()):
            entry["gpu_time_ns"] = sum(clients.values())
        found[int(pid)] = entry
    detail = "%d process(es) unreadable" % unreadable if unreadable else ""
    return {"probe": "drm-fdinfo", "status": "ran", "detail": detail}, found


def who(pid):
    """(user, command) of pid by ps, both empty when ps no longer finds it."""
    rc, out = run_bounded(["ps", "-o", "user=", "-o", "args=", "-p", str(pid)])
    line = out.strip().splitlines()[0] if rc == 0 and out.strip() else ""
    user, _, command = line.strip().partition(" ")
    return user, command.strip()[:COMMAND_WIDTH]


def snapshot(cell, event, box=None, nvidia_smi="nvidia-smi", drm_proc="/proc"):
    """One record line: the probes' outcomes and the GPU processes they list."""
    probes, processes = [], []
    for entry, found in (probe_nvidia(nvidia_smi), probe_drm(drm_proc)):
        probes.append(entry)
        for pid, fields in sorted(found.items()):
            user, command = who(pid)
            processes.append(dict({"pid": pid, "user": user, "command": command,
                                   "probe": entry["probe"]}, **fields))
    return {"cell": cell, "box": box or None, "host": socket.gethostname(), "event": event,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "probes": probes, "gpu_processes": processes}


def record(path, cell, event, **kw):
    line = json.dumps(snapshot(cell, event, **kw), sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read(path):
    """{event: snapshot} of a record file, the last line of each event winning; None when the file
    is absent. Lines that are not snapshots are skipped and counted under the key None."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    snaps, bad = {}, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = strict_json.loads(line)
        except ValueError:  # strict_json.NonFinite included
            bad += 1
            continue
        if (isinstance(obj, dict) and obj.get("event") in EVENTS
                and isinstance(obj.get("probes"), list)
                and isinstance(obj.get("gpu_processes"), list)):
            snaps[obj["event"]] = obj
        else:
            bad += 1
    if bad:
        snaps[None] = bad
    return snaps


# What a report line must not carry raw: a line break (every character str.splitlines breaks at) ends
# the line, `|` adds a Markdown column, a backtick opens a code span.
UNSAFE = re.compile("[|`\x00-\x1f\x7f-\x9f\u2028\u2029]")


def shown(value):
    """A string the record carries (a process's user or command line, a probe's detail) as the
    summary shows it: as it is, or display_safe's quoting when it holds an unsafe character."""
    text = str(value)
    return display_value(text) if UNSAFE.search(text) else text


def describe(proc):
    """`pid 123 alice python train.py (nvidia-smi, 640 MiB)`."""
    what = [shown(proc.get("probe", "?"))]
    if isinstance(proc.get("gpu_memory_mib"), int):
        what.append("%d MiB" % proc["gpu_memory_mib"])
    if proc.get("driver"):
        what.append(shown(proc["driver"]))
    if isinstance(proc.get("gpu_time_ns"), int):
        what.append("%.3f s GPU" % (proc["gpu_time_ns"] / 1e9))
    who_ = " ".join(shown(x) for x in (proc.get("user"), proc.get("command")) if x)
    return "pid %s %s(%s)" % (shown(proc.get("pid")), who_ + " " if who_ else "", ", ".join(what))


def describe_snapshot(snap):
    """`1 [pid ...] (nvidia-smi ran, drm-fdinfo absent: ...)`, or UNREAD when no probe ran."""
    probes = [p for p in snap["probes"] if isinstance(p, dict)]
    outcome = ", ".join("%s %s" % (shown(p.get("probe")), shown(p.get("status")))
                        + (": " + shown(p["detail"]) if p.get("detail") else "") for p in probes)
    if not any(p.get("status") == "ran" for p in probes):
        return "UNREAD (%s)" % (outcome or "no probe")
    procs = [p for p in snap["gpu_processes"] if isinstance(p, dict)]
    listed = "; ".join(describe(p) for p in procs[:SHOWN])
    more = " +%d more" % (len(procs) - SHOWN) if len(procs) > SHOWN else ""
    return "%d%s (%s)" % (len(procs), " [%s%s]" % (listed, more) if procs else "", outcome)


def summary(path):
    """The record's one line: each event's snapshot, or why there is none."""
    snaps = read(path)
    if snaps is None:
        return "not recorded (no %s: a cell from before gh-ocannl-1521)" % Path(path).name
    parts = ["%s %s" % (e, describe_snapshot(snaps[e]) if e in snaps else "not recorded")
             for e in EVENTS]
    if None in snaps:
        parts.append("%d unreadable line(s)" % snaps[None])
    return "; ".join(parts)


def main(argv):
    if len(argv) == 2 and argv[0] == "summary":
        print(summary(argv[1]))
        return 0
    if argv and argv[0] == "record":
        parser = argparse.ArgumentParser(prog="gpu_cotenants.py record")
        parser.add_argument("--box", default=None)
        parser.add_argument("--nvidia-smi", default="nvidia-smi")
        parser.add_argument("--drm-proc", default="/proc")
        parser.add_argument("file")
        parser.add_argument("cell")
        parser.add_argument("event", choices=EVENTS)
        args = parser.parse_args(argv[1:])
        try:
            record(args.file, args.cell, args.event, box=args.box, nvidia_smi=args.nvidia_smi,
                   drm_proc=args.drm_proc)
        except OSError as e:  # evidence, never a failed cell: say so and go on
            print("gpu_cotenants: could not write %s: %s" % (args.file, e), file=sys.stderr)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
