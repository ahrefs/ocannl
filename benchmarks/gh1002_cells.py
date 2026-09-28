#!/usr/bin/env python3
"""The gh-ocannl-1002 training measurement: treatments A-D of the fused attention backward.

Reproduces every table of benchmarks/report-gh1002-fused-backward.md. Two subcommands:

  run        --out DIR [--backends metal,cc] [--phases ...]   run the cells (resumable)
  summarize  --out DIR                                        print the report's tables

Treatments (docs/proposals/gh-ocannl-1002-1003.md, "Measurement plan and completion gates"), as
flags on ONE tree, untuned default pipeline, f32:

  A   composed forward, composed backward       online_softmax=false, online_softmax_backward=false
  B   online forward, scores stored             online_softmax=true (head width 32 > the recompute
      (default cap), composed backward          cap 16: the scores stay a stored [seq, seq] buffer)
  C   online forward, scores recomputed         B + virtualize_max_inline_reduction=32
      (raised cap), composed backward
  D1  online forward (default cap), fused       online_softmax=true, online_softmax_backward=true
  D2  online forward (raised cap), fused        D1 + virtualize_max_inline_reduction=32

E and F (the block-tiled forward) are gh-ocannl-1003's and are not run here.

Protocol. Every cell is one bench_gpt process on a training fixture (fixture metadata: 6 parity
steps, 3 warmup, 10 per-step-synced timed steps, 10 queued), run from benchmarks/ with every
OCANNL_*, BENCH_* and OMP_* variable cleared and the schedule cache disabled
(--ocannl_autotune_cache_dir= : the untuned default consults none anyway; this pins it). Repeats are
order-balanced: repeat r runs the treatments forward for even r and reversed for odd r, so an order
effect splits the repeats of one cell -- three repeats on Metal, two on cc. Peak memory is the
result line's peak_memory_bytes: Alloc_census.reset_peak after the warmup, peak_pool_bytes after the
timed steps (the harness's allocation window; requested bytes at the allocator seam). On Metal every
cell also prints the per-kernel table (BENCH_KERNEL_TABLE=1): every kernel the step shipped, timed
alone min-of-20 after the timed steps -- the per-segment attribution. cc runs no kernel table (one
C compile per kernel would dominate the cell); its attribution is Metal's.

Phases, in order: preflight (revision, fixture identities against fixtures/DIGESTS.txt -- a
fixture that matches no recorded m4-max entry aborts the run -- and a torch CPU parity reference
per fixture), metal (the main matrix: gpt2_mini_train, _s512, _s1024), metal-sweep (batch 1 at seq
128, 256, 512 -- seq 1024 at batch 1 is the main matrix's _s1024 -- two repeats), cc, cc-sweep (one
repeat), artifacts (one untimed Metal cell per fixture x treatment with the generated sources kept
under DIR/artifacts/). A host snapshot (top CPU consumers) is logged at every phase boundary.

Resumable: a cell whose output already holds a result line is not run again. A run directory is
bound to one revision and one set of fixture bytes (preflight.json): resuming it from a different
checkout or with different fixtures refuses rather than mixing the two. Each cell runs in its own
process group, killed whole at its deadline. `summarize` fails on a matrix with a missing cell, a
missing torch reference or an incomplete kernel table (`--partial` lists missing cells instead).
"""

import argparse
import datetime
import hashlib
import math
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH_GPT = ROOT / "_build" / "default" / "benchmarks" / "runners" / "ocannl" / "bench_gpt.exe"

TREATMENTS = [
    ("A", "composed / composed",
     ["--ocannl_online_softmax=false", "--ocannl_online_softmax_backward=false"]),
    ("B", "online, scores stored / composed",
     ["--ocannl_online_softmax=true", "--ocannl_online_softmax_backward=false"]),
    ("C", "online, scores recomputed / composed",
     ["--ocannl_online_softmax=true", "--ocannl_online_softmax_backward=false",
      "--ocannl_virtualize_max_inline_reduction=32"]),
    ("D1", "online, scores stored / fused",
     ["--ocannl_online_softmax=true", "--ocannl_online_softmax_backward=true"]),
    ("D2", "online, scores recomputed / fused",
     ["--ocannl_online_softmax=true", "--ocannl_online_softmax_backward=true",
      "--ocannl_virtualize_max_inline_reduction=32"]),
]
NAMES = [t[0] for t in TREATMENTS]
MAIN = ["gpt2_mini_train", "gpt2_mini_train_s512", "gpt2_mini_train_s1024"]
SWEEP = ["gpt2_mini_train_b1_s128", "gpt2_mini_train_b1_s256", "gpt2_mini_train_b1_s512"]
# The sequence sweep at fixed batch 1: the three sweep fixtures and the main matrix's seq-1024 one.
SWEEP_SEQS = [("gpt2_mini_train_b1_s128", 128), ("gpt2_mini_train_b1_s256", 256),
              ("gpt2_mini_train_b1_s512", 512), ("gpt2_mini_train_s1024", 1024)]
ORIGIN = "m4-max"
PHASES = ["preflight", "metal", "metal-sweep", "cc", "cc-sweep", "artifacts"]
CELL_TIMEOUT_S = 1800
# Treatment A against the torch CPU runner, six SGD steps in f32 (measured 2e-7 to 1.3e-6): the
# cross-framework envelope, generous by an order of magnitude over what the run shows.
TORCH_PARITY = 1e-5
# Treatments against A on the same backend and repeat (measured at most 2.1e-7).
TREATMENT_PARITY = 1e-4
# [--limit N]: stop after running N cells (a dry run of the mechanics; never for a published run).
LIMIT = [None]


def log(msg):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")
    print(f"[{stamp}] {msg}", flush=True)


def host_snapshot(out, tag):
    snap = subprocess.run(["ps", "-Ao", "pcpu,pid,comm", "-r"], capture_output=True, text=True).stdout
    top = "\n".join(snap.splitlines()[:12])
    with open(out / "host_snapshots.txt", "a") as f:
        f.write(f"=== {tag} {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n{top}\n")
    log(f"host snapshot: {tag}")


def clean_env(extra):
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("OCANNL_") or k.startswith("BENCH_") or k.startswith("OMP_"))}
    env.update(extra)
    return env


def order(r):
    return NAMES if r % 2 == 0 else list(reversed(NAMES))


def cell_path(out, backend, fixture, treatment, r):
    return out / "cells" / f"{backend}__{fixture}__{treatment}__r{r}"


def result_line(path):
    """The cell's result, only if its process exited 0 and its loss trajectory is complete and
    finite: a result line followed by a failed teardown or a signal is not a measurement."""
    try:
        if open(str(path) + ".status").read().strip() != "0":
            return None
        for line in open(str(path) + ".out"):
            line = line.strip()
            if line.startswith("{"):
                res = json.loads(line)
                fixture = Path(str(path)).name.split("__")[1]
                return res if valid_losses(res.get("losses"), fixture) else None
    except (OSError, ValueError, IndexError):
        pass
    return None


def run_cell(out, backend, fixture, treatment, r, artifacts=False):
    base = cell_path(out, backend, fixture, treatment, r)
    artifact_dir = out / "artifacts" / f"{backend}__{fixture}__{treatment}"
    # Done means the deliverable exists: an artifact cell also needs its sources moved into DIR.
    if result_line(base) is not None and (not artifacts or artifact_dir.is_dir()):
        log(f"skip (done) {base.name}")
        return
    if LIMIT[0] is not None:
        if LIMIT[0] <= 0:
            return
        LIMIT[0] -= 1
    flags = dict((t[0], t[2]) for t in TREATMENTS)[treatment]
    argv = [str(BENCH_GPT), f"--ocannl_backend={backend}", "--ocannl_autotune_cache_dir=",
            "--ocannl_log_config_sourcing=true"] + flags
    extra = {"BENCH_FIXTURE": str(HERE / "fixtures" / f"{fixture}.safetensors")}
    # build_files_prefix names a subdirectory of ./build_files (a path is mangled into one name),
    # so the artifacts are written under benchmarks/build_files/ and moved into DIR afterwards.
    prefix = f"gh1002__{backend}__{fixture}__{treatment}"
    if artifacts:
        argv += ["--ocannl_output_debug_files_in_build_directory=true",
                 f"--ocannl_build_files_prefix={prefix}"]
        extra["BENCH_DOMINANT_KERNEL"] = "0"
    elif backend == "metal":
        extra["BENCH_KERNEL_TABLE"] = "1"
    else:
        extra["BENCH_DOMINANT_KERNEL"] = "0"
    base.parent.mkdir(parents=True, exist_ok=True)
    with open(str(base) + ".cmd", "w") as f:
        f.write(" ".join(f"{k}={v}" for k, v in extra.items()) + " " + " ".join(argv) + "\n")
    log(f"run {base.name}")
    with open(str(base) + ".out", "w") as o, open(str(base) + ".err", "w") as e:
        status = run_in_own_group(argv, env=clean_env(extra), stdout=o, stderr=e)
    with open(str(base) + ".status", "w") as f:
        f.write(f"{status}\n")
    if artifacts and (HERE / "build_files" / prefix).is_dir():
        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(artifact_dir, ignore_errors=True)
        shutil.move(str(HERE / "build_files" / prefix), str(artifact_dir))
    res = result_line(base)
    if res is None:
        log(f"FAILED {base.name}: status {status}, no result line")
    else:
        log(f"done {base.name}: p50 {res['step_ms']['p50']:.1f} ms, "
            f"peak {res['peak_memory_bytes'] / 2**20:.1f} MiB")


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run_in_own_group(argv, **kw):
    """Run [argv] as the leader of its own process group; on timeout (and after any exit) kill the
    whole group, so a C compiler or other descendant a cell spawned cannot outlive it and load the
    cells that follow. Refuses to continue when the group cannot be reaped."""
    # Cancellation (SIGTERM, SIGINT) is held while the group is spawned and while it is reaped, and
    # let through only during the wait: a signal between the fork and Popen's return would leave a
    # detached group with no handle, and a second one mid-reap would abandon the reaping.
    held = {signal.SIGTERM, signal.SIGINT}
    signal.pthread_sigmask(signal.SIG_BLOCK, held)
    try:
        proc = subprocess.Popen(argv, cwd=HERE, start_new_session=True, **kw)
    except BaseException:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, held)
        raise
    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, held)
        status = proc.wait(timeout=CELL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        status = "timeout"
    finally:
        # Also on SIGTERM/KeyboardInterrupt: the group is not the driver's, so nothing else would
        # reap it. A pending signal is delivered once the reaping is done.
        signal.pthread_sigmask(signal.SIG_BLOCK, held)
        try:
            reap_group(proc, argv)
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, held)
    return status


def reap_group(proc, argv):
    for _ in range(50):
        if not group_alive(proc.pid):
            break
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
    if group_alive(proc.pid):
        sys.exit(f"process group {proc.pid} of {argv[0]} survived SIGKILL; stopping the matrix "
                 "rather than timing the next cells beside it")


def identity():
    """The run's identity: the revision, whether tracked files are dirty, and the fixtures' content
    digests, each of which must MATCH a recorded m4-max entry of fixtures/DIGESTS.txt."""
    sys.path.insert(0, str(HERE))
    import fixture_digest  # stdlib-only

    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    # A dirty tree has no identity a later resume could compare (re-editing a modified file leaves
    # the status unchanged), so it is refused outright rather than recorded.
    if dirty:
        sys.exit(f"tracked files are modified; commit them first -- a measurement names a "
                 f"revision:\n{dirty}")
    # The executable measured is the one built from that revision: build it here, and bind its
    # digest into the identity, so a stale _build cannot run under the recorded revision.
    if subprocess.run(["dune", "build", "benchmarks/runners/ocannl/bench_gpt.exe"],
                      cwd=ROOT).returncode != 0:
        sys.exit("dune build benchmarks/runners/ocannl/bench_gpt.exe failed")
    exe = hashlib.sha256(BENCH_GPT.read_bytes()).hexdigest()
    entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
    ids = {}
    for fx in MAIN + SWEEP:
        verdict, sha, _, origins = fixture_digest.status(HERE / "fixtures" / f"{fx}.safetensors",
                                                         entries)
        if verdict != "MATCH" or ORIGIN not in origins.split(","):
            # Never regenerate to get past this: that draws a new workload from this box's numpy
            # and retires the published numbers (benchmarks/README.md, gh-ocannl-759).
            sys.exit(f"fixture {fx}: {verdict} against DIGESTS.txt (need a {ORIGIN} match). Obtain "
                     f"the recorded {ORIGIN} bytes (`python3 {fixture_digest.cli_command()} --check` "
                     "reports disk against record); regenerating is a coordinated cross-box event, "
                     "not a fix for this refusal")
        ids[fx] = sha
    # The host too: a run directory copied to another machine must not collect its cells.
    return {"revision": rev, "bench_gpt_sha256": exe, "fixtures": ids, "host": os.uname().nodename}


def check_identity(out, ident):
    """A run directory holds one revision's cells on one set of fixture bytes: resuming it from a
    different checkout or different fixtures would mix the two under one preflight record."""
    pf = out / "preflight.json"
    if not pf.exists():
        return False
    recorded = json.loads(pf.read_text())
    for key in ("revision", "bench_gpt_sha256", "fixtures", "host"):
        if recorded.get(key) != ident[key]:
            sys.exit(f"{out}: its cells were measured with a different {key} "
                     f"({recorded.get(key)!r} vs now {ident[key]!r}); use a fresh --out")
    return True


def preflight(out):
    ident = identity()
    recorded = check_identity(out, ident)
    log(f"revision {ident['revision']}, bench_gpt {ident['bench_gpt_sha256'][:16]}; "
        f"fixtures match {ORIGIN}")
    import bench_venv

    venv = bench_venv.venv_python(HERE)  # BENCH_VENV_PY is read here, before clean_env drops it
    for fx in MAIN + SWEEP:
        dst = out / "torch" / f"{fx}.out"
        if torch_losses(out, fx) is not None:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"torch cpu parity reference {fx}")
        with open(dst, "w") as o, open(out / "torch" / f"{fx}.err", "w") as e:
            status = run_in_own_group(
                [str(venv), str(HERE / "runners" / "pytorch" / "run.py"), "--fixture",
                 str(HERE / "fixtures" / f"{fx}.safetensors"), "--device", "cpu"],
                env=clean_env({}), stdout=o, stderr=e)
        # The parity reference is part of what the run claims: a failed or unparseable oracle
        # stops the run instead of leaving the A-vs-torch column silently empty.
        if status != 0 or torch_losses(out, fx) is None:
            dst.unlink(missing_ok=True)
            sys.exit(f"torch parity reference for {fx} failed (status {status}); see "
                     f"{out / 'torch' / (fx + '.err')}")
    # preflight.json is the completion marker the phase guard reads, so it is written last, once
    # every reference has validated.
    if not recorded:
        meta = dict(ident, started=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (out / "preflight.json").write_text(json.dumps(meta, indent=2) + "\n")


def parity_steps(fx):
    return json.loads((HERE / "workloads" / f"{fx}.json").read_text())["parity_steps"]


def valid_losses(losses, fx):
    """A loss trajectory is the fixture's full parity window of finite numbers, or nothing."""
    return (isinstance(losses, list) and len(losses) == parity_steps(fx)
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses))


def torch_losses(out, fx):
    tf = out / "torch" / f"{fx}.out"
    try:
        lines = tf.read_text().strip().splitlines()
        losses = json.loads(lines[-1])["losses"] if lines else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return losses if valid_losses(losses, fx) else None


def run(args):
    # SIGTERM unwinds like Ctrl-C, so the running cell's group is reaped by its `finally`.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    backends = args.backends.split(",")
    requested = args.phases.split(",") if args.phases else PHASES
    for phase in requested:
        if phase not in PHASES:
            sys.exit(f"unknown phase {phase}")
    # Canonical order whatever the argument's: preflight always precedes every cell.
    phases = [p for p in PHASES if p in requested]
    if "preflight" not in phases and not (
        check_identity(out, identity())
        and all(torch_losses(out, fx) is not None for fx in MAIN + SWEEP)
    ):
        sys.exit(f"{out}: preflight incomplete (no preflight.json, or a torch reference missing) "
                 "-- run the preflight phase first")
    for phase in phases:
        if phase not in PHASES:
            sys.exit(f"unknown phase {phase}")
        backend = "cc" if phase.startswith("cc") else "metal"
        if phase != "preflight" and backend not in backends:
            continue
        host_snapshot(out, f"before {phase}")
        log(f"phase {phase}")
        if phase == "preflight":
            preflight(out)
        elif phase in ("metal", "cc"):
            for r in range(3 if backend == "metal" else 2):
                for fx in MAIN:
                    for t in order(r):
                        run_cell(out, backend, fx, t, r)
        elif phase in ("metal-sweep", "cc-sweep"):
            for r in range(2 if backend == "metal" else 1):
                for fx in SWEEP:
                    for t in order(r):
                        run_cell(out, backend, fx, t, r)
        elif phase == "artifacts":
            for fx in MAIN:
                for t in NAMES:
                    run_cell(out, "metal", fx, t, "art", artifacts=True)
    host_snapshot(out, "end")
    log("all phases done")


# ---------------------------------------------------------------------------------------------
# Summaries

KERNEL = re.compile(r"^bench: kernel (\d+)/(\d+) ([0-9.]+) ms grid=\[([0-9;]+)\] block=\[([0-9;]+)\] "
                    r"mma:(\S+) w: (.*)$")


def kernels(base):
    """The cell's per-kernel table, or None when it is incomplete: a kernel that declined to compile
    on its own has no row, and summing the rest would under-report the step."""
    ks, totals = [], set()
    try:
        for line in open(str(base) + ".err"):
            m = KERNEL.match(line.rstrip("\n"))
            if m:
                totals.add(int(m.group(2)))
                ks.append({"i": int(m.group(1)), "ms": float(m.group(3)), "grid": m.group(4),
                           "block": m.group(5), "w": m.group(7).split()})
    except OSError:
        return None
    if len(totals) != 1 or [k["i"] for k in ks] != list(range(next(iter(totals)))):
        return None
    return ks


def attention_blocks(ks):
    """Per-layer attention kernels, anchored on labels (the step's kernel order is the program's).

    Forward: from the kernel writing a query projection (a node labelled `q`, i.e. `..._q`) up to,
    excluding, the next one writing the block's output projection (`..._multi_head_attention`):
    the q/k/v projections, the scores, the softmax or the scan, and the value pass. Backward: from
    the kernel writing `w_o_lL.grad` through the one writing `w_q_lL.grad`: w_o's gradient (which
    shares a kernel with dP or dQ), the attention core's backward, and the q/k/v projection
    gradients. The projections are the same work in every treatment; what differs lies between.
    """
    fwd, bwd = [], []
    i = 0
    while i < len(ks):
        if any(re.search(r"_q$", w) for w in ks[i]["w"]):
            j = i
            while j < len(ks) and not any(w.endswith("_multi_head_attention") for w in ks[j]["w"]):
                j += 1
            fwd.append(ks[i:j])
            i = j
        i += 1
    # The LAST writers: the gradient zeroing kernel ahead of the backward writes every gradient.
    def last(pred, upto):
        return next((k for k in range(upto, -1, -1) if pred(ks[k])), None)

    for layer in range(16):
        end = last(lambda x: f"w_q_l{layer}.grad" in x["w"], len(ks) - 1)
        if end is None:
            continue
        start = last(lambda x: f"w_o_l{layer}.grad" in x["w"], end)
        if start is not None:
            bwd.append(ks[start:end + 1])
    return fwd, bwd


def load_cells(out):
    cells = {}
    for f in sorted((out / "cells").glob("*.out")):
        base = Path(str(f)[:-4])
        backend, fixture, treatment, r = base.name.split("__")
        res = result_line(base)
        cells.setdefault((backend, fixture, treatment), {})[r] = (res, base)
    return cells


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rel_losses(a, b):
    return max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip(a, b))


def summarize(args):
    out = Path(args.out).resolve()
    cells = load_cells(out)
    problems, notes = [], []
    pre = json.loads((out / "preflight.json").read_text()) if (out / "preflight.json").exists() else {}
    print(f"revision: {pre.get('revision', '?')}  host: {pre.get('host', '?')}  "
          f"dirty: {bool(pre.get('dirty_tracked_files'))}")
    for fx, sha in pre.get("fixtures", {}).items():
        print(f"  {fx}: {ORIGIN} content-v1 {sha[:16]}")

    def timed(backend, fx, t):
        reps = cells.get((backend, fx, t), {})
        return {r: v for r, v in reps.items() if r != "rart" and v[0] is not None}

    for (backend, fx, t), reps in sorted(cells.items()):
        for r, (res, base) in reps.items():
            if res is None:
                problems.append(f"{base.name}: no result line")
    # Completeness: a matrix any cell of which exists must be whole -- every fixture, treatment and
    # repeat -- or its ratios are over a partial set (and a missing A or B makes them nan).
    for backend, fixtures, n_reps in (("metal", MAIN, 3), ("metal", SWEEP, 2), ("cc", MAIN, 2),
                                      ("cc", SWEEP, 1)):
        # Without --partial every matrix must be there: the command reproduces the whole report,
        # and a directory with no cells at all is no reproduction of it.
        if args.partial and not any(timed(backend, fx, t) for fx in fixtures for t in NAMES):
            continue
        for fx in fixtures:
            for t in NAMES:
                have = timed(backend, fx, t)
                for r in range(n_reps):
                    if f"r{r}" not in have:
                        (notes if args.partial else problems).append(
                            f"{backend}__{fx}__{t}__r{r}: missing (the matrix is incomplete)")

    for backend in ("metal", "cc"):
        for fixtures, title in ((MAIN, "main matrix"), (SWEEP, "sequence sweep, batch 1")):
            if not any(timed(backend, fx, t) for fx in fixtures for t in NAMES):
                continue
            print(f"\n### Step times, {backend}, {title}\n")
            print("| fixture | treatment | p50 per repeat (ms) | median p50 | vs A | vs B | "
                  "tokens/s | p10..p90 spread | compile s |")
            print("|---|---|---|---|---|---|---|---|---|")
            for fx in fixtures:
                meds = {t: med([v[0]["step_ms"]["p50"] for v in timed(backend, fx, t).values()])
                        for t in NAMES}
                for t in NAMES:
                    reps = timed(backend, fx, t)
                    if not reps:
                        continue
                    p50s = [reps[r][0]["step_ms"]["p50"] for r in sorted(reps)]
                    spread = max(v[0]["step_ms"]["p90"] / v[0]["step_ms"]["p10"]
                                 for v in reps.values())
                    tok = next(iter(reps.values()))[0].get("tokens_per_step", 0)
                    comp = med([v[0]["compile_s"] for v in reps.values()])
                    print(f"| `{fx}` | {t} | {', '.join(f'{x:.1f}' for x in p50s)} | "
                          f"{meds[t]:.1f} | {meds[t] / meds['A']:.3f}x | {meds[t] / meds['B']:.3f}x | "
                          f"{tok / (meds[t] / 1000):.0f} | up to {spread:.3f}x | {comp:.1f} |")

            print(f"\n### Peak requested memory, {backend}, {title}\n")
            print("| fixture | treatment | peak (MiB) | vs A | minus A (MiB) | across repeats |")
            print("|---|---|---|---|---|---|")
            for fx in fixtures:
                peaks = {t: [v[0]["peak_memory_bytes"] for v in timed(backend, fx, t).values()]
                         for t in NAMES}
                for t in NAMES:
                    if not peaks[t]:
                        continue
                    p, a = med(peaks[t]), med(peaks["A"])
                    same = "identical" if len(set(peaks[t])) == 1 else \
                        f"{min(peaks[t]) / 2**20:.2f}..{max(peaks[t]) / 2**20:.2f}"
                    print(f"| `{fx}` | {t} | {p / 2**20:.1f} | {p / a:.3f}x | "
                          f"{(p - a) / 2**20:+.1f} | {same} |")

            print(f"\n### Loss-trajectory parity, {backend}, {title}\n")
            print("| fixture | treatment | worst relative difference vs A, same repeat "
                  "(6 parity steps) | A vs torch cpu |")
            print("|---|---|---|---|")
            for fx in fixtures:
                ref_losses = torch_losses(out, fx)
                if ref_losses is None:
                    problems.append(f"{fx}: no torch parity reference")
                a_reps = timed(backend, fx, "A")
                for t in NAMES:
                    reps = timed(backend, fx, t)
                    diffs = [rel_losses(reps[r][0]["losses"], a_reps[r][0]["losses"])
                             for r in reps if r in a_reps]
                    vs_torch = ""
                    if t == "A" and ref_losses and a_reps:
                        worst_ref = max(rel_losses(v[0]["losses"], ref_losses)
                                        for v in a_reps.values())
                        if worst_ref > TORCH_PARITY:
                            problems.append(f"{backend} {fx} A: losses differ from torch cpu by "
                                            f"{worst_ref:.2e} (> {TORCH_PARITY:g})")
                        vs_torch = f"{worst_ref:.2e}"
                    if diffs:
                        worst = max(diffs)
                        if worst > TREATMENT_PARITY:
                            problems.append(f"{backend} {fx} {t}: losses differ from A by {worst:.2e}")
                        print(f"| `{fx}` | {t} | {worst:.2e} | {vs_torch} |")

            if backend != "metal":
                continue
            print(f"\n### Attribution, {backend}, {title} (per-kernel min-of-20, summed over the four "
                  "layers; median over repeats)\n")
            print("| fixture | treatment | kernels | all kernels (ms) | attention forward (ms) | "
                  "attention backward (ms) | backward kernels per layer |")
            print("|---|---|---|---|---|---|---|")
            for fx in fixtures:
                for t in NAMES:
                    reps = timed(backend, fx, t)
                    rows = []
                    for r, (res, base) in reps.items():
                        ks = kernels(base)
                        if ks is None:
                            problems.append(f"{base.name}: kernel table missing or incomplete")
                            continue
                        fwd, bwd = attention_blocks(ks)
                        rows.append((len(ks), sum(k["ms"] for k in ks),
                                     sum(k["ms"] for b in fwd for k in b),
                                     sum(k["ms"] for b in bwd for k in b),
                                     len(bwd[0]) if bwd else 0, len(fwd), len(bwd)))
                    if not rows:
                        continue
                    if any(r[5] != 4 or r[6] != 4 for r in rows):
                        problems.append(f"{fx} {t}: attention blocks not found for all 4 layers")
                    print(f"| `{fx}` | {t} | {rows[0][0]} | {med([r[1] for r in rows]):.1f} | "
                          f"{med([r[2] for r in rows]):.1f} | {med([r[3] for r in rows]):.1f} | "
                          f"{rows[0][4]} |")
            print(f"\n### Layer-0 attention backward kernels, {backend}, {title} (repeat 0)\n")
            for fx in fixtures:
                for t in NAMES:
                    rep = timed(backend, fx, t).get("r0")
                    if not rep:
                        continue
                    _, bwd = attention_blocks(kernels(rep[1]) or [])
                    if not bwd:
                        continue
                    print(f"- `{fx}` {t}:")
                    for k in bwd[0]:
                        print(f"  - {k['ms']:.3f} ms grid [{k['grid']}] block [{k['block']}]: "
                              f"{' '.join(k['w'])}")

    # The sweep: per treatment, peak memory at fixed batch 1 over seq, and a least-squares
    # a + b s + c s^2 fit -- c in units of one [seq, seq] f32 buffer per head per layer.
    for backend in ("metal", "cc"):
        rows = {}
        for t in NAMES:
            pts = []
            for fx, s in SWEEP_SEQS:
                reps = timed(backend, fx, t)
                if reps:
                    pts.append((s, med([v[0]["peak_memory_bytes"] for v in reps.values()]),
                                med([v[0]["step_ms"]["p50"] for v in reps.values()])))
            if len(pts) == len(SWEEP_SEQS):
                rows[t] = pts
        if not rows:
            continue
        print(f"\n### Sequence sweep at batch 1, {backend}: peak memory (MiB) and p50 (ms)\n")
        print("| treatment | " + " | ".join(f"seq {s}" for _, s in SWEEP_SEQS) +
              " | quadratic term c (bytes/seq^2) | c / (4 B x 8 heads x 4 layers) |")
        print("|---|" + "---|" * (len(SWEEP_SEQS) + 2))
        for t, pts in rows.items():
            c = quad_fit([(s, m) for s, m, _ in pts])[2]
            cells_txt = " | ".join(f"{m / 2**20:.1f} / {ms:.0f}" for _, m, ms in pts)
            print(f"| {t} | {cells_txt} | {c:.1f} | {c / (4 * 8 * 4):.2f} |")

    if notes:
        print("\nPARTIAL (--partial):")
        for n in notes:
            print(f"- {n}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"- {p}")
        return 1
    return 0


def quad_fit(pts):
    """Least-squares a + b s + c s^2 through (s, y) points (normal equations, 3x3)."""
    n = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for s, y in pts:
        v = [1.0, float(s), float(s) ** 2]
        for i in range(3):
            rhs[i] += v[i] * y
            for j in range(3):
                n[i][j] += v[i] * v[j]
    # Gaussian elimination.
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(n[r][i]))
        n[i], n[p], rhs[i], rhs[p] = n[p], n[i], rhs[p], rhs[i]
        for r in range(i + 1, 3):
            f = n[r][i] / n[i][i]
            for c in range(i, 3):
                n[r][c] -= f * n[i][c]
            rhs[r] -= f * rhs[i]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (rhs[i] - sum(n[i][c] * x[c] for c in range(i + 1, 3))) / n[i][i]
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--backends", default="metal,cc")
    r.add_argument("--phases", default=None, help=f"comma-separated subset of {','.join(PHASES)}")
    r.add_argument("--limit", type=int, default=None, help="run at most N cells (dry runs only)")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--partial", action="store_true",
                   help="summarize an incomplete matrix (missing cells are listed, not fatal)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        LIMIT[0] = args.limit
        run(args)
        return 0
    return summarize(args)


if __name__ == "__main__":
    sys.exit(main())
