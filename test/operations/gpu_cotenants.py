"""Host-only controls for benchmarks/gpu_cotenants.py, the GPU co-tenant record of a kprof cell
(gh-ocannl-1521): its two probes over a stubbed nvidia-smi and a stubbed /proc, the record's
snapshots, its summary -- an unread snapshot never reads as an empty GPU -- and its command line."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

owner = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(owner.parent))
spec = importlib.util.spec_from_file_location("gpu_cotenants", owner)
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)


def check(condition, detail):
    # A failing control must remain fatal under python -O.
    if not condition:
        raise SystemExit("FAIL: " + detail)


ME = os.getpid()
GONE = 4194303  # above any pid_max a box ships with: ps finds no such process


def stub(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def fake_proc(root):
    """A /proc with one DRM client (two descriptors sharing one client id), one process whose only
    descriptor is not a DRM device, and an entry that is not a pid."""
    def fd(pid, n, target, info=None):
        (root / str(pid) / "fd").mkdir(parents=True, exist_ok=True)
        (root / str(pid) / "fdinfo").mkdir(parents=True, exist_ok=True)
        os.symlink(target, root / str(pid) / "fd" / str(n))
        if info is not None:
            (root / str(pid) / "fdinfo" / str(n)).write_text(info)
    info = ("pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-pdev:\t0000:c3:00.0\n"
            "drm-client-id:\t7\ndrm-engine-gfx:\t1500000000 ns\ndrm-engine-compute:\t500000000 ns\n")
    fd(ME, 3, "/dev/dri/renderD128", info)
    fd(ME, 4, "/dev/dri/renderD128", info)
    fd(GONE, 0, "/dev/null", "pos:\t0\n")
    (root / "self").mkdir()
    return str(root)


with tempfile.TemporaryDirectory() as scratch:
    tmp = Path(scratch)
    # nvidia-smi lists this process on two GPUs, a pid that has ended with no memory reading, and a
    # line that is no process at all.
    smi = stub(tmp / "nvidia-smi", "echo 'No running processes found'\n"
               "echo '%d, python3, 512'\necho '%d, python3, 128'\necho '%d, [Not Found], [N/A]'\n"
               % (ME, ME, GONE))
    proc = fake_proc(tmp / "proc")
    snap = gc.snapshot("cuda-gpt2_mini-keep-kprof1", "opened", box="rog-nv", nvidia_smi=smi,
                       drm_proc=proc)
    by = {(p["probe"], p["pid"]): p for p in snap["gpu_processes"]}
    probes = {p["probe"]: p for p in snap["probes"]}
    check(set(by) == {("nvidia-smi", ME), ("nvidia-smi", GONE), ("drm-fdinfo", ME)}, str(by))
    print("each probe lists exactly its GPU processes, the non-DRM descriptor and non-pid entry left out: true")
    check(by[("nvidia-smi", ME)].get("gpu_memory_mib") == 640, str(by[("nvidia-smi", ME)]))
    check("gpu_memory_mib" not in by[("nvidia-smi", GONE)], str(by[("nvidia-smi", GONE)]))
    check(probes["nvidia-smi"] == {"probe": "nvidia-smi", "status": "ran",
                                   "detail": "skipped 1 unreadable line(s)"}, str(probes))
    print("nvidia-smi: a process on two GPUs is one entry, its memory summed; [N/A] gives none; a"
          " non-process line is counted, not listed: true")
    drm = by[("drm-fdinfo", ME)]
    check(drm.get("gpu_time_ns") == 2000000000 and drm.get("driver") == "amdgpu"
          and drm.get("pdev") == "0000:c3:00.0", str(drm))
    print("drm-fdinfo: a client's engine times are summed once however many descriptors share it: true")
    check(by[("nvidia-smi", ME)]["user"] != "" and "gpu_cotenants" in by[("nvidia-smi", ME)]["command"],
          str(by[("nvidia-smi", ME)]))
    check(by[("nvidia-smi", GONE)]["user"] == "" and by[("nvidia-smi", GONE)]["command"] == "",
          str(by[("nvidia-smi", GONE)]))
    print("a live process carries its user and command line, an ended one neither: true")
    check(snap["cell"] == "cuda-gpt2_mini-keep-kprof1" and snap["box"] == "rog-nv"
          and snap["event"] == "opened" and snap["at"].endswith("Z") and snap["host"], str(snap))
    print("the snapshot names its cell, box, host, event and UTC time: true")

    # The record file: two snapshots, read back by event, summarized on one line.
    record = tmp / "cell.cotenants.jsonl"
    gc.record(record, "c", "opened", nvidia_smi=smi, drm_proc=proc)
    gc.record(record, "c", "closed", nvidia_smi=stub(tmp / "idle-smi", "exit 0\n"),
              drm_proc=str(tmp / "empty"))
    snaps = gc.read(record)
    check(set(snaps) == {"opened", "closed"} and len(record.read_text().splitlines()) == 2, str(snaps))
    line = gc.summary(record)
    check(line.startswith("opened 3 [pid %d " % ME), line)
    check("640 MiB" in line and "amdgpu" in line and "2.000 s GPU" in line, line)
    check("; closed 0 (nvidia-smi ran, drm-fdinfo absent: no %s)" % (tmp / "empty") in line, line)
    print("the summary lists each snapshot's processes, and an idle GPU as 0 with the probes that ran: true")

    # A snapshot no probe could take is UNREAD, never an idle GPU.
    unread = tmp / "unread.cotenants.jsonl"
    gc.record(unread, "c", "opened", nvidia_smi=stub(tmp / "broken-smi", "echo 'NVML: driver gone'; exit 9\n"),
              drm_proc=str(tmp / "nowhere"))
    gc.record(unread, "c", "closed", nvidia_smi=str(tmp / "no-such-smi"), drm_proc=str(tmp / "nowhere"))
    line = gc.summary(unread)
    check(line.startswith("opened UNREAD (nvidia-smi failed: exit 9: NVML: driver gone, drm-fdinfo absent")
          and "closed UNREAD (nvidia-smi absent" in line and " 0 " not in line, line)
    print("a snapshot in which no probe ran is UNREAD, with each probe's reason: true")

    # A record cut short, a malformed line, and a cell from before the record.
    partial = tmp / "partial.cotenants.jsonl"
    gc.record(partial, "c", "opened", nvidia_smi=smi, drm_proc=proc)
    with open(partial, "a") as f:
        f.write('{"event": "closed"}\nnot json\n')
    line = gc.summary(partial)
    check(line.startswith("opened 3 [") and "; closed not recorded; 2 unreadable line(s)" in line, line)
    check(gc.summary(tmp / "absent.cotenants.jsonl")
          == "not recorded (no absent.cotenants.jsonl: a cell from before gh-ocannl-1521)",
          gc.summary(tmp / "absent.cotenants.jsonl"))
    print("a missing event, a malformed line and a missing record each say so: true")

    # The command line the driver runs: a record it cannot write is reported, never a failed cell.
    cli = [sys.executable, str(owner), "record", "--box", "tuf", "--nvidia-smi", smi,
           "--drm-proc", proc]
    done = subprocess.run(cli + [str(tmp / "cli.jsonl"), "c", "opened"], capture_output=True, text=True)
    written = json.loads((tmp / "cli.jsonl").read_text())
    check(done.returncode == 0 and written["box"] == "tuf" and written["event"] == "opened",
          done.stderr + str(written))
    done = subprocess.run(cli + [str(tmp / "no-dir" / "x.jsonl"), "c", "closed"],
                          capture_output=True, text=True)
    check(done.returncode == 0 and "could not write" in done.stderr, done.stderr)
    done = subprocess.run(cli + [str(tmp / "cli.jsonl"), "c", "midway"], capture_output=True, text=True)
    check(done.returncode == 2, "an unknown event: exit %d" % done.returncode)
    done = subprocess.run([sys.executable, str(owner), "summary", str(tmp / "cli.jsonl")],
                          capture_output=True, text=True)
    check(done.returncode == 0 and done.stdout.startswith("opened 3 [") and "closed not recorded"
          in done.stdout, done.stdout)
    print("record exits 0 even when it cannot write, refuses an unknown event; summary prints the line: true")
