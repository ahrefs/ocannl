#!/usr/bin/env python3
"""The gh-ocannl-1002 training measurement: treatments A-D of the fused attention backward.

Reproduces every table of benchmarks/report-gh1002-fused-backward.md. Two subcommands:

  run        --out DIR --box BOX [--backends metal,cc] [--phases ...]   run the cells (resumable;
             benchmarks/check-box.sh checks BOX and every fixture first, box_gate.py)
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
repeat), artifacts (one untimed Metal cell per fixture x treatment, sweep included, with the
generated sources kept under DIR/artifacts/). A host snapshot (top CPU consumers) is logged at
every phase boundary.

Resumable: a cell whose output already holds a result line is not run again. A run directory is
bound to one revision and one set of fixture bytes (preflight.json): resuming it from a different
checkout or with different fixtures refuses rather than mixing the two. Each cell -- and the
preflight's Dune build -- runs in its own process group, killed whole at its deadline. A cell that
ends without a result appends orchestrate's failure record to DIR/failures.jsonl: the reason, and
the checkpoint of the loss work it completed, with the run's identity (gh-ocannl-1234). `summarize`
fails on a matrix with a missing cell, a missing torch reference or an incomplete kernel table
(`--partial` lists missing cells instead).
"""

import argparse
import datetime
import hashlib
import math
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import box_gate
import cell_group
import strict_json
from display_safe import code_span, display_value, log_line, table_lines

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
CELL_TERMINATE_GRACE_S = 5
# The preflight build of the measured executable, run from ROOT. Its deadline covers a cold build of
# bench_gpt and everything it links -- minutes, not the hour -- so only a wedged build reaches it.
BUILD_ARGV = ["dune", "build", "benchmarks/runners/ocannl/bench_gpt.exe"]
BUILD_TIMEOUT_S = 3600
_cancellation = cell_group.CancellationDeferral("gh1002_cells")
# Treatment A against the torch CPU runner, six SGD steps in f32 (measured 2e-7 to 1.3e-6): the
# cross-framework envelope, generous by an order of magnitude over what the run shows.
TORCH_PARITY = 1e-5
# Treatments against A on the same backend and repeat (measured at most 2.1e-7).
TREATMENT_PARITY = 1e-4
# [--limit N]: stop after running N cells (a dry run of the mechanics; never for a published run).
LIMIT = [None]


def log(msg):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")
    print(log_line(f"[{stamp}] {msg}"), flush=True)


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
        if Path(str(path) + ".status").read_text().strip() != "0":
            return None
        for line in Path(str(path) + ".out").read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                res = strict_json.loads(line)  # NonFinite is a ValueError: no result
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
    env = clean_env(extra)
    try:
        with open(str(base) + ".out", "w") as o, open(str(base) + ".err", "w") as e:
            status = run_in_own_group(argv, env=env, stdout=o, stderr=e)
    except BaseException as exc:
        # Interrupted, or a cleanup not proven complete: the cell, and what it completed, is
        # recorded before the cancellation propagates, as orchestrate's sweep records it.
        why = str(exc) if isinstance(exc, cell_group.CleanupFailed) else "interrupted mid-cell"
        record_failure(out, base, why, env, cell_checkpoint(base))
        raise
    with open(str(base) + ".status", "w") as f:
        f.write(f"{status}\n")
    if artifacts:
        staged = HERE / "build_files" / prefix
        if staged.is_dir() and any(staged.iterdir()):
            artifact_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(artifact_dir, ignore_errors=True)
            shutil.move(str(staged), str(artifact_dir))
        else:
            # The sources ARE this cell's deliverable: without them it failed, whatever it timed.
            with open(str(base) + ".status", "w") as f:
                f.write("no-artifacts\n")
            log(f"FAILED {base.name}: no generated sources under {display_value(str(staged))}")
            record_failure(out, base, f"no generated sources (status {status})", env,
                           cell_checkpoint(base))
            return
    res = result_line(base)
    if res is None:
        log(f"FAILED {base.name}: status {status}, no result line")
        if status == "timeout":
            why = (f"TIMED OUT after {CELL_TIMEOUT_S:g}s (cap) -- killed the cell's whole "
                   "process group")
        elif status != 0:
            why = f"exit {status}"
        else:
            why = "no result line with the fixture's full finite loss window"
        record_failure(out, base, why, env, cell_checkpoint(base))
    else:
        log(f"done {base.name}: p50 {res['step_ms']['p50']:.1f} ms, "
            f"peak {res['peak_memory_bytes'] / 2**20:.1f} MiB")


def cell_checkpoint(base):
    """The last checkpoint the cell wrote to its .err (`orchestrate.last_checkpoint`), or None."""
    # Imported here, as gh1181_cells.py does: the summaries' readers (kernel_attribution.py's
    # KERNEL, test/operations' gh1002_kernels) load this module with its own imports only.
    import orchestrate

    try:
        return orchestrate.last_checkpoint(
            Path(str(base) + ".err").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def record_failure(out, base, why, env, checkpoint):
    """Append the cell's failure to DIR/failures.jsonl as orchestrate records one (gh-ocannl-1234):
    `orchestrate.failure_record` -- the reason, the regime (every cell here is untuned under no
    profile: exact), the inherited OCANNL_* environment, and the checkpoint a killed or failed cell
    left (gh-ocannl-1209) with its provenance, which is the run's preflight identity -- plus the
    cell's coordinates. The cell's .out/.err/.status stay what `summarize` reads; this is the record
    of what a failure kept."""
    import orchestrate  # see cell_checkpoint

    backend, fixture, treatment, r = base.name.split("__")
    pf = out / "preflight.json"
    ident = json.loads(pf.read_text()) if pf.exists() else {}
    record = orchestrate.failure_record(
        base.name, why, orchestrate.ambient_ocannl_env(env), checkpoint,
        provenance={"source_revision": ident.get("revision"),
                    # A dirty tree is refused before the preflight records its identity.
                    "source_tracked_changes": False if ident else None,
                    "bench_gpt_sha256": ident.get("bench_gpt_sha256"),
                    "fixture_content_sha256": (ident.get("fixtures") or {}).get(fixture),
                    "host": ident.get("host")},
        regime="exact")
    record.update(backend=backend, fixture=fixture, treatment=treatment, repeat=r)
    with open(out / "failures.jsonl", "a") as f:
        f.write(json.dumps(orchestrate.json_safe(record), allow_nan=False) + "\n")


def run_in_own_group(argv, cwd=HERE, timeout=None, **kw):
    """Run one child -- a cell, a parity reference, the preflight build -- through the shared
    supervisor, from `cwd` and killed whole at `timeout` seconds (default `CELL_TIMEOUT_S`); refuse
    to time beside an unproven cleanup."""
    timeout = CELL_TIMEOUT_S if timeout is None else timeout
    # Hold cancellation until the spawn has a cleanup owner, and throughout cleanup. The wait
    # opens the intentional cancellation window without giving fork/exec an inherited mask.
    with _cancellation.deferring():
        proc = cell_group.spawn(argv, cwd=cwd, **kw)
        try:
            with _cancellation.cancellable():
                status = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
        finally:
            result = cell_group.terminate(proc, grace=CELL_TERMINATE_GRACE_S)
            if result.observation is not cell_group.GONE or not result.reaped:
                raise cell_group.CleanupFailed(
                    f"process group {proc.pid} of {argv[0]} was not proven gone and reaped "
                    f"after cleanup ({result.observation.value}, reaped={result.reaped}); "
                    "stopping the matrix rather than timing the next cells beside it"
                )
        return status


def build_bench_gpt():
    """Build the measured executable under the cells' own supervisor (gh-ocannl-1200).

    Dune forks compilers and linkers. A bare `subprocess.run` kills only its direct child when the
    driver is cancelled -- the descendants live on beside the cells about to be timed -- and it had
    no deadline at all. In a group of its own, the build is collected whole on SIGTERM, SIGINT or
    its deadline, and a cleanup not proven complete stops the run (`cell_group.CleanupFailed`).
    Its output still goes straight to the driver's terminal."""
    status = run_in_own_group(BUILD_ARGV, cwd=ROOT, timeout=BUILD_TIMEOUT_S)
    if status != 0:
        sys.exit(f"{' '.join(BUILD_ARGV)} failed (status {status})")


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
    build_bench_gpt()
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
    if not recorded:
        # No completion marker: references left by an interrupted preflight may be from another
        # identity, so none is reused.
        shutil.rmtree(out / "torch", ignore_errors=True)
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
        losses = strict_json.loads(lines[-1])["losses"] if lines else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return losses if valid_losses(losses, fx) else None


def run(args):
    _cancellation.install()
    # Every fixture, whatever the phases: preflight digests them all, and a run without it checks
    # that same identity before its first cell.
    box_gate.require("gh1002_cells.py", args, HERE / "fixtures", MAIN + SWEEP, run=run_in_own_group)
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
            for fx in MAIN + SWEEP:
                for t in NAMES:
                    run_cell(out, "metal", fx, t, "art", artifacts=True)
    host_snapshot(out, "end")
    log("all phases done")


# ---------------------------------------------------------------------------------------------
# Summaries

# The kernel table's row (Bench_harness.time_shipped_kernels). `io:<read>/<written>` (global-memory
# bytes by the cost model; a `~` marks an upper bound, a `?` code the model cannot see) is
# gh-ocannl-1444's and absent from older rows; the groups are named so that a field added to the row
# renumbers nothing.
KERNEL = re.compile(r"^bench: kernel (?P<i>\d+)/(?P<n>\d+) (?P<ms>[0-9.]+) ms "
                    r"grid=\[(?P<grid>[0-9;]+)\] block=\[(?P<block>[0-9;]+)\] "
                    r"(?:io:(?P<read>\d+)/(?P<written>\d+)(?P<io_flags>[~?]*) )?"
                    r"mma:(?P<mma>.*?) (?:vol:.*?)?w: (?P<writes>.*)$")


def kernels(base):
    """The cell's per-kernel table, or None when it is incomplete: a kernel that declined to compile
    on its own has no row, and summing the rest would under-report the step."""
    ks, totals = [], set()
    try:
        for line in open(str(base) + ".err"):
            m = KERNEL.match(line.rstrip("\n"))
            if m:
                totals.add(int(m["n"]))
                ks.append({"i": int(m["i"]), "ms": float(m["ms"]), "grid": m["grid"],
                           "block": m["block"], "w": m["writes"].split()})
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
            rows = []
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
                    rows.append((code_span(fx), t, ", ".join(f"{x:.1f}" for x in p50s),
                                 f"{meds[t]:.1f}", f"{meds[t] / meds['A']:.3f}x",
                                 f"{meds[t] / meds['B']:.3f}x", f"{tok / (meds[t] / 1000):.0f}",
                                 f"up to {spread:.3f}x", f"{comp:.1f}"))
            print("\n".join(table_lines(
                ["fixture", "treatment", "p50 per repeat (ms)", "median p50", "vs A", "vs B",
                 "tokens/s", "p10..p90 spread", "compile s"], rows)))

            print(f"\n### Peak requested memory, {backend}, {title}\n")
            rows = []
            for fx in fixtures:
                peaks = {t: [v[0]["peak_memory_bytes"] for v in timed(backend, fx, t).values()]
                         for t in NAMES}
                for t in NAMES:
                    if not peaks[t]:
                        continue
                    p, a = med(peaks[t]), med(peaks["A"])
                    same = "identical" if len(set(peaks[t])) == 1 else \
                        f"{min(peaks[t]) / 2**20:.2f}..{max(peaks[t]) / 2**20:.2f}"
                    rows.append((code_span(fx), t, f"{p / 2**20:.1f}", f"{p / a:.3f}x",
                                 f"{(p - a) / 2**20:+.1f}", same))
            print("\n".join(table_lines(
                ["fixture", "treatment", "peak (MiB)", "vs A", "minus A (MiB)", "across repeats"],
                rows)))

            print(f"\n### Loss-trajectory parity, {backend}, {title}\n")
            rows = []
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
                        rows.append((code_span(fx), t, f"{worst:.2e}", vs_torch))
            print("\n".join(table_lines(
                ["fixture", "treatment",
                 "worst relative difference vs A, same repeat (6 parity steps)", "A vs torch cpu"],
                rows)))

            if backend != "metal":
                continue
            print(f"\n### Attribution, {backend}, {title} (per-kernel min-of-20, summed over the four "
                  "layers; median over repeats)\n")
            table = []
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
                    table.append((code_span(fx), t, str(rows[0][0]),
                                  f"{med([r[1] for r in rows]):.1f}",
                                  f"{med([r[2] for r in rows]):.1f}",
                                  f"{med([r[3] for r in rows]):.1f}", str(rows[0][4])))
            print("\n".join(table_lines(
                ["fixture", "treatment", "kernels", "all kernels (ms)", "attention forward (ms)",
                 "attention backward (ms)", "backward kernels per layer"], table)))
            print(f"\n### Layer-0 attention backward kernels, {backend}, {title} (repeat 0)\n")
            for fx in fixtures:
                for t in NAMES:
                    rep = timed(backend, fx, t).get("r0")
                    if not rep:
                        continue
                    _, bwd = attention_blocks(kernels(rep[1]) or [])
                    if not bwd:
                        continue
                    print(log_line(f"- {code_span(fx)} {t}:"))
                    for k in bwd[0]:
                        print(log_line(f"  - {k['ms']:.3f} ms grid [{k['grid']}] block [{k['block']}]: "
                                       f"{' '.join(map(display_value, k['w']))}"))

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
        table = []
        for t, pts in rows.items():
            c = quad_fit([(s, m) for s, m, _ in pts])[2]
            table.append((t, *(f"{m / 2**20:.1f} / {ms:.0f}" for _, m, ms in pts), f"{c:.1f}",
                          f"{c / (4 * 8 * 4):.2f}"))
        print("\n".join(table_lines(
            ["treatment", *(f"seq {s}" for _, s in SWEEP_SEQS), "quadratic term c (bytes/seq^2)",
             "c / (4 B x 8 heads x 4 layers)"], table)))

    if notes:
        print("\nPARTIAL (--partial):")
        for n in notes:
            print(log_line(f"- {n}"))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(log_line(f"- {p}"))
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
    box_gate.add_arguments(r)
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
