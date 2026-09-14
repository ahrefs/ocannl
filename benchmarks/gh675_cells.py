#!/usr/bin/env python3
"""gh-ocannl-675, CUDA leg: pass-1 (searched here) vs pass-2 (fresh process, replay) step times.

One arm = one pair of processes: pass 1 searches (cold cache) and times, pass 2 replays that
cache in a fresh process and times again. Arms are run one at a time, alternating across repeats
so a drifting box moves both halves of a pair together, behind an idle gate and pinned with
taskset. Every record goes to $GH675_OUT/records.jsonl verbatim; the analysis pairs pass 1 with
pass 2 WITHIN a repeat and reports the median of the paired ratios.

    python3 gh675_cells.py --arms A_beam2 C_jit_cold C_eager D_ocannl --workloads mlp_small \
        --repeats 3 --start-repeat 1

Arms: A_beam* tinygrad BEAM=N, B_compiled/B_maxauto torch.compile, C_* the non-searching
controls (their X must be ~0 or the spread is process noise, not search residue), D_ocannl the
tuned OCANNL cell the two-pass protocol exists for, REF_cpu the parity reference.

Four things a ratio needs in order to mean anything, which this driver establishes rather than
assumes -- each of them a way a pair can look valid and not be:
  * the TREATMENT is the arm's, not the shell's: every framework knob is stripped from what a
    child inherits and set explicitly per arm (an exported `BEAM` otherwise makes the controls
    search while they still report `searched: false`);
  * the PROVENANCE is checked per pair against what the arm claims, and a pair that searched in
    pass 2 (or failed to in pass 1) is stamped `provenance_ok: false` and announced;
  * the FIXTURE is identified by sha256 on every record, and `--expect-digest` refuses a run whose
    bytes are not the ones being resumed;
  * a WEDGE is recorded against the pass that wedged, so a searching-process hang and a replaying
    -process hang stay distinguishable;
  * and every PRECONDITION those rest on is established rather than attempted: a wipe that left
    residue, a prewarm that failed, a child that exited nonzero after printing a result, and an
    idle gate that gave up all mark the pair `valid: false` (loudly) instead of yielding a record
    that looks like any other. Read `valid` before using a pair; `analysis` should filter on it.
Nothing this driver writes touches the checkout: caches, records and the tuned cell's
`autotune_cache_dir` all live under $GH675_OUT.

See benchmarks/report-gh675-cuda.md for what it measured.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path

import bench_venv
import cell_group

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "benchmarks"
VENV = bench_venv.venv_python(HERE)  # honours BENCH_VENV_PY and the Windows venv layout
# Everything this driver writes -- the per-run records and the per-arm caches whose warmth IS the
# experiment -- goes outside the checkout, under $GH675_OUT (default: a tmp dir).
# Resolved, not as given: the children run with `cwd=benchmarks/`, so a RELATIVE GH675_OUT would
# name one directory to the driver and a different one to every runner -- the driver verifying an
# empty cache here while a child quietly reuses a stale one under benchmarks/.
SP = Path(os.environ.get("GH675_OUT", "/tmp/gh675")).resolve()
RECORDS = SP / "records.jsonl"
CACHES = SP / "caches"
NVSMI = "/usr/lib/wsl/lib/nvidia-smi"
TASKSET = ["taskset", "-c", "0-15"]

FIXTURE = {w: str(HERE / f"fixtures/{w}.safetensors") for w in ("mlp_small", "gpt2_mini")}
DIGESTS = {}  # filled at startup; stamped on every record
PREWARMED = set()  # workloads whose warm-cache control has been prewarmed IN THIS INVOCATION
ATTEMPTS = {}  # (arm, workload, repeat) -> prior rows, so a --rerun row says which attempt it is
_cancellation = cell_group.CancellationDeferral("gh675_cells")

# Every variable that is a TREATMENT in this experiment rather than part of the box. A child
# inherits the driver's environment, so one of these exported in the operator's shell silently
# rewrites an arm: a stray `BEAM=2` makes the non-beam CONTROLS beam-search, and the runner --
# which reports `searched` from its own command line, not from the env -- then labels them
# `searched: false`, which is the one reading that cannot be caught downstream. Each arm states
# its treatment explicitly below and everything here is stripped from what it inherits, so an arm
# is what the driver says it is rather than what the shell happened to hold (gh-ocannl-675).
TREATMENT_VARS = (
    "BEAM", "CACHEDB", "CACHELEVEL", "IGNORE_BEAM_CACHE", "JIT", "DEV", "PARALLEL",
    "TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_MAX_AUTOTUNE",
    "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
)
TREATMENT_PREFIXES = ("BENCH_", "OCANNL_")


def base_env():
    """The inherited environment with every treatment variable removed.

    The OCANNL namespaces are matched case-INSENSITIVELY. gh-ocannl-652 dropped the lowercase
    `ocannl_<key>` spelling and made setting it a fatal startup error, so a shell still carrying a
    legacy `ocannl_backend=cuda` would kill the tuned arm -- hours into a sweep, and only that arm.
    """
    lower = tuple(pre.lower() for pre in TREATMENT_PREFIXES)
    return {
        k: v
        for k, v in os.environ.items()
        if k not in TREATMENT_VARS and not k.lower().startswith(lower)
    }


def digests(workloads=None):
    """sha256 of each fixture, stamped onto every record.

    The fixtures are gitignored and regenerable, and `gen_fixtures.py` does not promise stable
    bytes across numpy releases, so a filename does not identify a workload. Without this a run
    resumed after a regeneration mixes two workloads under one `(workload, repeat)` label
    (gh-ocannl-645 is the same rule for published reports).
    """
    out = {}
    for w, path in FIXTURE.items():
        if workloads is not None and w not in workloads:
            # Hashing every known fixture would make an unrelated, un-generated one
            # (gen_fixtures.py can produce a single workload) a hard error for a run not using it.
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[w] = h.hexdigest()
    return out


def gate():
    """Wait until the box is quiet enough to time on; False if it never got there.

    Returning the verdict rather than printing it is the point: a run that started on a busy box
    is exactly the contended measurement this protocol claims to exclude, and a record of it must
    not be indistinguishable from an idle-gated one.
    """
    warned = False
    for _ in range(24):
        load = float(open("/proc/loadavg").read().split()[0])
        try:
            _probe, out, _ = run_managed(
                [NVSMI, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                timeout=30,
                context="GPU-utilization probe",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            util = out.strip().splitlines()[0]
            util = int(util)
        except Exception as ex:
            # An unreadable GPU is not an idle GPU. Absent `nvidia-smi`, a timeout or malformed
            # output used to become `util = 0`, which a quiet load average then turned into a
            # PASSING gate -- the measurement's central precondition established by a failure.
            if not warned:  # once, not once per poll: a repeated line hides the others
                print(f"gate: GPU utilization unreadable ({type(ex).__name__}: {ex}) -- "
                      "not treating that as idle", file=sys.stderr, flush=True)
                warned = True
            util = None
        if util is None:
            time.sleep(5)
            continue
        # WSL's load average decays slowly and counts uninterruptible tasks, so it is a poor
        # instantaneous gate; the GPU is the resource under measurement. Wait for both, but do
        # not stall the sweep on a stale load figure.
        if load < 8.0 and util < 10:
            return True
        time.sleep(5)
    print("gate: giving up waiting for idle -- measurement will be stamped gate_ok: false",
          file=sys.stderr, flush=True)
    return False


def communicate_clean(proc, timeout, context):
    """Wait for one managed child and abort this experiment if its group is not proven gone."""
    try:
        with _cancellation.cancellable():
            out, err = proc.communicate(timeout=timeout)
    except BaseException:
        result = cell_group.terminate(proc, grace=5)
        if result.observation is not cell_group.GONE:
            raise cell_group.CleanupFailed(
                f"{context} process group was not observed gone after SIGKILL "
                f"({result.observation.value}) and may still hold the pinned CPUs or GPU. "
                "Stopping the sweep: clear it, then resume past what is recorded."
            )
        raise
    initial = proc.observe()
    if initial is not cell_group.GONE:
        result = cell_group.terminate(proc, grace=5)
        if result.observation is not cell_group.GONE:
            raise cell_group.CleanupFailed(
                f"{context} left a process group that was not observed gone after SIGKILL "
                f"({result.observation.value}). Every later pair would be contaminated; "
                "clear it before resuming."
            )
    else:
        proc.close()
    return out, err


def run_managed(args, timeout, context, **kwargs):
    """Spawn and wait with cancellation deferred until the group has a cleanup owner."""
    with _cancellation.deferring():
        proc = cell_group.spawn(args, **kwargs)
        out, err = communicate_clean(proc, timeout=timeout, context=context)
        return proc, out, err


def run(cmd, env=None, cwd=None, timeout=None):
    e = base_env()
    if env:
        e.update(env)
    gate_ok = gate()
    t0 = time.monotonic()
    # Its own session, so a timeout can reach the DESCENDANTS. `subprocess.run(timeout=...)` kills
    # the runner only, and tinygrad's beam search farms its candidate compiles out to a spawn pool:
    # the orphaned workers survive on the same pinned cores and GPU, and the sweep moves straight
    # on to the next pair with nothing in its record to show it was measured against them. That is
    # not hypothetical -- it happened while taking these numbers, and a CPU reference taken over
    # the survivors read 340 ms against 0.13 ms once the box was clean.
    proc, out, err = run_managed(
        TASKSET + cmd,
        timeout=timeout,
        context="benchmark cell",
        env=e,
        cwd=cwd or str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wall = time.monotonic() - t0
    p = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    line = None
    for ln in reversed(p.stdout.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                line = json.loads(ln)
                break
            except Exception:
                pass
    if line is None or p.returncode != 0:
        # A nonzero exit invalidates the run even when a well-formed result line was printed
        # first: whatever raised after `emit` (framework teardown, an interpreter-shutdown error)
        # ran with the measurement's own resources still live, and this driver cannot show that it
        # did not touch the numbers. Cheaper to drop the pair than to publish an unexplained one.
        rec = {"__failed__": True, "rc": p.returncode, "wall_s": round(wall, 2),
               "gate_ok": gate_ok, "stdout_tail": p.stdout[-3000:],
               "stderr_tail": p.stderr[-3000:]}
        if line is not None:
            rec["result_despite_nonzero_exit"] = line
        return rec
    line["wall_s"] = round(wall, 2)
    line["rc"] = p.returncode
    line["gate_ok"] = gate_ok
    return line


def tiny(workload, beam=0, cachedb=None, retime=True):
    cmd = [str(VENV), str(HERE / "runners/tinygrad/run.py"), "--fixture", FIXTURE[workload],
           "--device", "CUDA", "--jit", "1"]
    if beam:
        cmd += ["--beam", str(beam)]
    if retime:
        cmd += ["--retime"]
    # BEAM is stated for every tinygrad arm, `0` included: the controls' whole claim is that they
    # do not search, and inheriting an exported BEAM would make that claim false while the runner
    # kept reporting `searched: false` from its own argv.
    return cmd, {"CACHEDB": str(cachedb), "BEAM": str(beam)}


def torchcmd(workload, compiled, mode=None, cachedir=None, retime=True):
    cmd = [str(VENV), str(HERE / "runners/pytorch/run.py"), "--fixture", FIXTURE[workload],
           "--device", "cuda"]
    if compiled:
        cmd += ["--compile"]
        if mode:
            cmd += ["--compile-mode", mode]
    if retime:
        cmd += ["--retime"]
    env = {}
    if cachedir:
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cachedir)
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        env["TRITON_CACHE_DIR"] = str(Path(cachedir) / "triton")
    return cmd, env


def ocannl_cache(workload):
    """The tuned cell's autotune cache -- under GH675_OUT like every other arm's.

    The default is `./autotune_cache` relative to the runner's cwd, i.e. inside the checkout: a
    driver that wiped THAT before each pass 1 would destroy whatever tuning results the operator
    had from other benchmark work, and would couple the experiment to checkout state.
    """
    return CACHES / f"ocannl_{workload}"


def ocannl(workload):
    exe = {"mlp_small": "bench_mlp", "gpt2_mini": "bench_gpt"}[workload]
    cmd = [str(ROOT / f"_build/default/benchmarks/runners/ocannl/{exe}.exe"),
           "--ocannl_backend=cuda",
           f"--ocannl_autotune_cache_dir={ocannl_cache(workload)}"]
    env = {"BENCH_FIXTURE": FIXTURE[workload], "BENCH_TUNE": "1", "BENCH_MATERIALIZE": "0",
           "BENCH_PRECISION": "f32", "BENCH_STATIC_SCALE": "0", "BENCH_GATE_INTERVAL": "0"}
    return cmd, env


class PreconditionFailed(Exception):
    """A condition the pair's meaning depends on did not hold, so no pair is measured."""


def wipe(p):
    """Empty a cache directory, and establish that it IS empty.

    `ignore_errors=True` alone can leave an old or half-deleted cache in place, and a searching
    arm over a mixed cache still reports `searched: true` in pass 1 (something was generated) and
    `false` in pass 2 -- which passes the provenance check while pass 1 was never cold. So the
    failure has to be raised here, where it is still about a precondition rather than about a
    number.
    """
    p = Path(p)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    residue = sorted(x.name for x in p.iterdir())
    if residue:
        raise PreconditionFailed(f"cache {p} still holds {residue[:5]} after wipe")


# What each arm's two passes MUST report for `searched`. A searching arm whose pass 1 comes back
# `false`/`null` did not search (a cache wipe that failed, a probe that could not bind), and one
# whose pass 2 comes back `true` did not replay (a partially written cache is the observed case) --
# in both the pair's ratio is not the quantity this experiment is about, and nothing downstream can
# tell. So the expectation is stated per arm and checked per pair, and a record that violates it is
# stamped `provenance_ok: false` with the reason (gh-ocannl-644 is the field this reads).
EXPECTED_SEARCHED = {
    "A_beam2": (True, False),
    "A_beam8": (True, False),
    "B_compiled": (True, False),
    "B_maxauto": (True, False),
    "D_ocannl": (True, False),
    "C_jit_warm": (False, False),
    "C_jit_cold": (False, False),
    "C_eager": (False, False),
}


def check_provenance(arm, p1, p2):
    """[] if the pair's `searched` fields are what this arm's design says, else the complaints."""
    want = EXPECTED_SEARCHED.get(arm)
    if want is None:
        return []
    bad = []
    for want_v, got, which in ((want[0], p1.get("searched"), "pass1"),
                               (want[1], p2.get("searched"), "pass2")):
        if got is not want_v:
            bad.append(f"{which} searched={got!r}, expected {want_v!r}")
    return bad


def emit(rec):
    with open(RECORDS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    r1, r2 = rec.get("pass1", {}), rec.get("pass2", {})
    def p50(r):
        return (r.get("step_ms") or {}).get("p50")
    print(f"  {rec['arm']:28s} {rec['workload']:10s} rep{rec['repeat']}  "
          f"p1={p50(r1)} p2={p50(r2)}  searched={r1.get('searched')}/{r2.get('searched')} "
          f"compile_s={r1.get('compile_s')}/{r2.get('compile_s')}", flush=True)


def timed_run(cmd, env, timeout):
    """`run`, with a timeout recorded as a failure of THIS invocation rather than of the pair.

    Letting `TimeoutExpired` escape to the caller loses which half wedged -- and a pass-1 wedge
    (the searching process) and a pass-2 wedge (the replaying one) are different findings, which
    is exactly the distinction gh-ocannl-760 rests on.
    """
    try:
        return run(cmd, env, timeout=timeout)
    except subprocess.TimeoutExpired as ex:
        return {"__failed__": True, "timeout_s": timeout, "timeout": str(ex)}


def existing_keys():
    """`(arm, workload, repeat)` already in the records file."""
    keys = {}
    if not RECORDS.exists():
        return keys
    with open(RECORDS) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            k = (r.get("arm"), r.get("workload"), r.get("repeat"))
            keys[k] = keys.get(k, 0) + 1
    return keys


def record(arm, workload, repeat, **fields):
    """The one record shape every arm emits -- REF_cpu included, so nothing lands unstamped."""
    rec = {"arm": arm, "workload": workload, "repeat": repeat,
           "fixture_sha256": DIGESTS.get(workload),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    key = (arm, workload, repeat)
    prior = ATTEMPTS.get(key, 0)
    if prior:
        # Only under --rerun; the identity is otherwise refused before the sweep starts.
        rec["attempt"] = prior + 1
    # Counted as it runs, so a second row for one identity WITHIN this sweep is numbered too --
    # seeding this from the records file alone would leave in-sweep repeats indistinguishable.
    ATTEMPTS[key] = prior + 1
    return rec


def do_pair(arm, workload, repeat, mk1, mk2, pre1=None, pre2=None, timeout=None):
    try:
        if pre1:
            pre1()
        c, e = mk1()
        p1 = timed_run(c, e, timeout)
        if pre2:
            pre2()
        c, e = mk2()
        p2 = timed_run(c, e, timeout)
    except PreconditionFailed as ex:
        # Not a measurement that came out badly -- a measurement that was never set up. Recorded
        # so the gap is visible in the raw data, and loud so a sweep is not left to discover it.
        print(f"  !! {arm} {workload} rep{repeat}: PRECONDITION {ex} -- no pair measured",
              file=sys.stderr, flush=True)
        emit(record(arm, workload, repeat, pass1={"__failed__": True, "precondition": str(ex)},
                    pass2={}, valid=False))
        return
    rec = record(arm, workload, repeat, pass1=p1, pass2=p2)
    failed = p1.get("__failed__") or p2.get("__failed__")
    if not failed:
        bad = check_provenance(arm, p1, p2)
        rec["provenance_ok"] = not bad
        if bad:
            rec["provenance"] = bad
            print(f"  !! {arm} {workload} rep{repeat}: PROVENANCE {'; '.join(bad)} "
                  f"-- pair recorded as invalid, exclude it", file=sys.stderr, flush=True)
        gated = p1.get("gate_ok") is not False and p2.get("gate_ok") is not False
        if not gated:
            print(f"  !! {arm} {workload} rep{repeat}: ran on a busy box (gate gave up) "
                  f"-- pair recorded as invalid, exclude it", file=sys.stderr, flush=True)
        rec["valid"] = bool(not bad and gated)
    else:
        rec["valid"] = False
    emit(rec)


ARMS = {}


def arm(name):
    def deco(fn):
        ARMS[name] = fn
        return fn
    return deco


@arm("A_beam2")
def a_beam2(w, rep):
    db = CACHES / f"tiny_beam2_{w}"
    do_pair("A_beam2", w, rep,
            lambda: tiny(w, beam=2, cachedb=db / "cache.db"),
            lambda: tiny(w, beam=2, cachedb=db / "cache.db"),
            pre1=lambda: wipe(db), timeout=2400)


@arm("A_beam8")
def a_beam8(w, rep):
    db = CACHES / f"tiny_beam8_{w}"
    do_pair("A_beam8", w, rep,
            lambda: tiny(w, beam=8, cachedb=db / "cache.db"),
            lambda: tiny(w, beam=8, cachedb=db / "cache.db"),
            pre1=lambda: wipe(db), timeout=3600)


@arm("C_jit_cold")
def c_jit_cold(w, rep):
    db = CACHES / f"tiny_jitcold_{w}"
    do_pair("C_jit_cold", w, rep,
            lambda: tiny(w, cachedb=db / "cache.db"),
            lambda: tiny(w, cachedb=db / "cache.db"),
            pre1=lambda: wipe(db), timeout=1200)


@arm("C_jit_warm")
def c_jit_warm(w, rep):
    db = CACHES / f"tiny_jitwarm_{w}"
    db.mkdir(parents=True, exist_ok=True)
    # Once per DRIVER INVOCATION rather than once per output directory. A marker that persists
    # across invocations cannot know that the tinygrad version, the CUDA toolchain or the workload
    # geometry changed under it, and a cache with no entries for the current kernels makes pass 1
    # compile what pass 2 replays -- a cold-vs-warm pair wearing the warm-vs-warm label, since
    # both passes still report `searched: false`. One extra ~20 s run per invocation buys the
    # whole question away; fingerprinting the environment would only approximate it.
    if w not in PREWARMED:
        # This arm's claim is warm-vs-warm: it is the control that says a pair of processes doing
        # no compilation and no search differ by ~nothing. Against an empty cache its first pair
        # would be cold-vs-warm instead -- a compile control, and one whose X would be read as
        # process spread. Prewarm once, unrecorded, rather than depending on the operator having
        # run (and discarded) a repeat 0.
        print(f"  prewarming {db} (unrecorded)", flush=True)
        c, e = tiny(w, cachedb=db / "cache.db", retime=False)
        pre = timed_run(c, e, 1200)
        if pre.get("__failed__") or not (db / "cache.db").exists():
            # A discarded prewarm failure is the subtlest way to get a cold-vs-warm pair labelled
            # as the warm-vs-warm control: both passes still report `searched: false`. And a
            # crashed prewarm can leave a PARTIAL `cache.db` behind, so its existence proves
            # nothing -- the next invocation would skip prewarming and pass 1 would compile the
            # kernels pass 2 then replays. Hence: drop whatever it left, and let only a completed
            # prewarm write the marker this test actually reads.
            shutil.rmtree(db, ignore_errors=True)
            raise PreconditionFailed(
                f"prewarm of {db} failed (rc={pre.get('rc')}, timeout={pre.get('timeout_s')}); "
                "its partial cache was removed")
        PREWARMED.add(w)
    do_pair("C_jit_warm", w, rep,
            lambda: tiny(w, cachedb=db / "cache.db"),
            lambda: tiny(w, cachedb=db / "cache.db"), timeout=1200)


@arm("B_compiled")
def b_compiled(w, rep):
    cd = CACHES / f"inductor_compiled_{w}_{rep}"
    do_pair("B_compiled", w, rep,
            lambda: torchcmd(w, True, cachedir=cd),
            lambda: torchcmd(w, True, cachedir=cd),
            pre1=lambda: wipe(cd), timeout=2400)


@arm("B_maxauto")
def b_maxauto(w, rep):
    cd = CACHES / f"inductor_maxauto_{w}_{rep}"
    do_pair("B_maxauto", w, rep,
            lambda: torchcmd(w, True, mode="max-autotune", cachedir=cd),
            lambda: torchcmd(w, True, mode="max-autotune", cachedir=cd),
            pre1=lambda: wipe(cd), timeout=3600)


@arm("C_eager")
def c_eager(w, rep):
    do_pair("C_eager", w, rep,
            lambda: torchcmd(w, False),
            lambda: torchcmd(w, False), timeout=1200)


@arm("D_ocannl")
def d_ocannl(w, rep):
    ac = ocannl_cache(w)
    do_pair("D_ocannl", w, rep,
            lambda: ocannl(w), lambda: ocannl(w),
            pre1=lambda: wipe(ac), timeout=3600)


@arm("REF_cpu")
def ref_cpu(w, rep):
    cmd = [str(VENV), str(HERE / "runners/pytorch/run.py"), "--fixture", FIXTURE[w], "--device", "cpu"]
    r = timed_run(cmd, {}, 3600)
    emit(record("REF_cpu", w, rep, pass1=r, pass2={},
                valid=not r.get("__failed__") and r.get("gate_ok") is not False))


if __name__ == "__main__":
    _cancellation.install()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--workloads", nargs="+", default=["mlp_small"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--start-repeat", type=int, default=1)
    ap.add_argument("--rerun", action="store_true",
                    help="record another attempt at an already-recorded (arm, workload, repeat) "
                         "instead of refusing; each extra row is stamped with `attempt`")
    ap.add_argument("--expect-digest", nargs="*", metavar="WORKLOAD=SHA256",
                    help="refuse to run unless the fixture's bytes hash to this -- what makes a "
                         "resumed run the same experiment as the one it resumes")
    args = ap.parse_args()
    # Before any directory is made or any child is launched: a typo AFTER a valid entry would
    # otherwise be discovered hours in, by a KeyError, with the preceding cells already recorded
    # and colliding on the restart.
    bad_arms = [a for a in args.arms if a not in ARMS]
    bad_workloads = [w for w in args.workloads if w not in FIXTURE]
    if bad_arms or bad_workloads:
        sys.exit(f"unknown arm(s) {bad_arms} / workload(s) {bad_workloads}; "
                 f"arms are {sorted(ARMS)}, workloads are {sorted(FIXTURE)}")
    CACHES.mkdir(parents=True, exist_ok=True)
    DIGESTS.update(digests(args.workloads))
    print(f"records: {RECORDS}", flush=True)
    for w, d in sorted(DIGESTS.items()):
        print(f"fixture {w}: sha256 {d}", flush=True)
    if args.expect_digest:
        pinned = dict(x.split("=", 1) for x in args.expect_digest)
        # A pin set is meant to be reusable across runs that select different workloads, so a pin
        # for a workload this run does not touch is skipped -- but said out loud, because silently
        # ignoring a pin is the same class of silence the rest of this driver exists to remove.
        skipped = sorted(w for w in pinned if w not in DIGESTS)
        if skipped:
            print(f"digest pins not checked (workload not selected): {', '.join(skipped)}",
                  flush=True)
        wrong = {w: DIGESTS[w] for w, d in pinned.items() if w in DIGESTS and DIGESTS[w] != d}
        if wrong:
            sys.exit(f"fixture digest mismatch (pinned vs actual): {pinned} vs {wrong} -- these "
                     "are different workload bytes under the same names; regenerate or re-pin")
    # A resumed sweep whose --start-repeat overlaps what is already recorded would append a second
    # row under the same identity, and the analysis pairs and weights BY that identity: one repeat
    # silently counted twice, or its two passes taken from different attempts.
    seen = existing_keys()
    ATTEMPTS.update(seen)
    planned = [(a, w, rep)
               for rep in range(args.start_repeat, args.start_repeat + args.repeats)
               for w in args.workloads for a in args.arms]
    within = [k for k, c in Counter(planned).items() if c > 1]
    if within and not args.rerun:
        # `--arms A_beam2 A_beam2`, or a repeated workload: two rows under one identity, neither
        # distinguishable from the other, without ever touching the records file.
        sys.exit(f"{len(within)} (arm, workload, repeat) requested more than once in this "
                 f"invocation: {within[:5]}{' ...' if len(within) > 5 else ''} -- drop the "
                 "duplicates, or pass --rerun to record them as numbered attempts")
    clash = [k for k in planned if k in seen]
    if clash and not args.rerun:
        sys.exit(f"{len(clash)} (arm, workload, repeat) already in {RECORDS}: "
                 f"{clash[:5]}{' ...' if len(clash) > 5 else ''} -- pick a --start-repeat past "
                 "them, or pass --rerun to record another attempt (each stamped with `attempt`)")

    for rep in range(args.start_repeat, args.start_repeat + args.repeats):
        for w in args.workloads:
            # Rotated by repeat: traversing the same order every time pins each arm to a fixed
            # position in the session, and the process-wide CUDA JIT cache (`~/.nv/ComputeCache`,
            # which no per-arm wipe touches) is warmer at the end of a repeat than at its start --
            # measurably so: it is what takes OCANNL's search from 16 s to 4 s over this sweep. A
            # fixed order confounds arm identity with accumulated compiler state; rotation spreads
            # each arm over all positions instead.
            for a in args.arms[rep % len(args.arms):] + args.arms[:rep % len(args.arms)]:
                print(f"== rep{rep} {w} {a}", flush=True)
                try:
                    ARMS[a](w, rep)
                except subprocess.TimeoutExpired as ex:
                    emit(record(a, w, rep, pass1={"__failed__": True, "timeout": str(ex)},
                                pass2={}, valid=False))
                except PreconditionFailed as ex:
                    # An arm can fail a precondition in its own setup, outside `do_pair` -- the
                    # warm control's prewarm is the case. That invalidates ONE pair; letting it
                    # reach the top level would abort a sweep that has hours of good arms left.
                    print(f"  !! {a} {w} rep{rep}: PRECONDITION {ex} -- no pair measured",
                          file=sys.stderr, flush=True)
                    emit(record(a, w, rep, pass1={"__failed__": True, "precondition": str(ex)},
                                pass2={}, valid=False))
