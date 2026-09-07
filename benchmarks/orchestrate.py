#!/usr/bin/env python3
"""Run the cross-framework benchmark matrix (OCANNL / PyTorch / tinygrad) and report.

For each fixture in fixtures/, runs every (framework, backend, variant, precision) cell,
collects the JSON result lines, enforces the loss-trajectory parity gate against the PyTorch
CPU reference, and writes results/results.jsonl plus a markdown report.

Scheduling (`default` / `materialized` / `tuned`) and storage precision (`f32` / `bf16` /
`f16`) are INDEPENDENT axes of an OCANNL cell, and the matrix is their product
(gh-ocannl-539). Overloading one variant string made tuned reduced-precision cells
inexpressible, which is where tensor cores would show at all on backends whose only mma route
is a reduced input format (RDNA3/3.5 WMMA has no f32-input shape).

Timing results are only comparable when the parity gate passes: a FAIL means that cell was
not computing the same training trajectory, so its step times are flagged, not compared.

A requested cell a workload cannot express — a reduced precision on the conv runner, an f16
gate-cost leg on a forward-only workload — is reported as NOT APPLICABLE with its reason, in
the run log and in a report section, rather than quietly not being run (gh-ocannl-551).
"""

import argparse
import contextlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import fixture_digest
import cell_group

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# BENCH_VENV_PY overrides the venv interpreter — for environments where benchmarks/.venv
# is unusable (e.g. deep worktree paths hitting Windows MAX_PATH during torch install).
VENV_PY = Path(
    os.environ.get(
        "BENCH_VENV_PY",
        HERE / ".venv/Scripts/python.exe"
        if (HERE / ".venv/Scripts/python.exe").exists()
        else HERE / ".venv/bin/python",
    )
)
# BENCH_CELL_LOG_DIR: keep every cell's raw combined output under this directory, one file per
# cell label. Unset (the default) discards a successful cell's output as before.
CELL_LOG_DIR = (
    Path(os.environ["BENCH_CELL_LOG_DIR"]) if os.environ.get("BENCH_CELL_LOG_DIR") else None
)
PARITY_TOL = 2e-3
# Per-cell wall-clock cap (gh-ocannl-760). tinygrad's parallel beam search deadlocks
# intermittently — a candidate-compile worker dying between `imap_unordered` chunks leaves the
# parent blocked in `futex_do_wait` forever — and an unattended sweep that meets one loses the
# whole sweep rather than one cell. The cap is generous against what has ever been MEASURED here:
# the beam searches that wedged take 53-115 s in their other repeats, the slowest tuned cell of
# any sweep on record is cifar_conv/metal at ~4 min wall, and the only run that ever came near
# half an hour is a pre-gh-538 standalone Metal search (2069 s) whose successor takes 230 s. A
# cell over the cap is reported as a runner failure with the cap named in the message, and
# --cell-timeout raises it (0 disables), so a legitimately slower box is a flag away rather than
# a silently truncated measurement.
DEFAULT_CELL_TIMEOUT_S = 1800
# tinygrad 0.13.0's GPU beam-search spawn pool wedged intermittently on both fleet GPU boxes,
# and 0.14.0 still reproduced it on minix (1/5 cold gpt2_mini searches exceeded a 300 s cap,
# against 37-51 s for three healthy repeats).  Zero is the only value that removes the pool
# rather than merely making the same failure less likely.  Every declared measurement box uses
# this pin; keeping one value here rather than dispatching on hostname also protects a renamed or
# newly provisioned box until its operator deliberately opts into a pool with --beam-parallel.
DEFAULT_BEAM_PARALLEL = 0
# Grace between the process group's SIGTERM and its SIGKILL. A wedged tinygrad parent is blocked
# in a futex and its pool workers in a socket read, so the handler usually never runs; the grace
# is for the cells that CAN unwind (a Python runner's atexit, a backend's device teardown).
CELL_KILL_GRACE_S = 10
# Wall-clock budget for the `tinygrad_cachedb` probe subprocess (spawn a fresh interpreter, import
# the framework). Deliberately its own constant rather than a reuse of `CELL_KILL_GRACE_S`: a test
# that patches the kill grace down to speed up an unrelated SIGTERM/SIGKILL assertion must not also
# starve this probe of the time a real interpreter start + import needs, which is exactly what
# made `test_the_tinygrad_cache_probe_collects_helpers_it_spawned` flake under CI load.
TINYGRAD_PROBE_TIMEOUT_S = 10
# Accuracy-parity gates for the OCANNL mixed-precision legs (gh-ocannl-492 task 4), with roughly
# 10x headroom over the largest drift measured by the macOS cc/Metal sweep.
PARITY_TOL_PRECISION = {"bf16": 4e-3, "f16": 2e-3}
# gh-ocannl-719: the parity envelope of the `approximate` regime. An approximate OCANNL cell runs
# under `--ocannl_profile=approximate` -- tf32 matmuls, fp16 arithmetic, the C compiler's
# reassociation and contraction licence, the inlining refinement, and the algebraic-rewrite gates
# as they land -- and its torch counterpart under torch's own defaults (`high` matmul precision,
# scaled_dot_product_attention, cudnn.benchmark). Looser than PARITY_TOL and the reduced-precision
# envelopes, tighter than "did the loss move": tf32 keeps two more mantissa bits than bf16, so a
# trajectory drifts on the order of the bf16 envelope, and 1e-2 gives it the same ~10x headroom
# the precision envelopes carry. The first approximate sweep on a tf32-capable box calibrates it;
# an approximate row also reports whether it passed the EXACT envelope, so a tightening is
# evidence-based and a rewrite that changes nothing measurable is visible as such.
PARITY_TOL_APPROX = 1e-2
# The numerics regimes a sweep can run its matrix in, in report order (exact rows first).
REGIMES = ("exact", "approximate")
# A parity tolerance cannot reject an input-independent forward when the reference itself moves
# slowly. Require at least one part per million of relative loss variation over the parity window.
LOSS_MOVE_MIN_REL = 1e-6
REFERENCE = ("pytorch", "cpu", "eager")
# Cells whose protocol splits the search and the timing into two processes (gh-ocannl-644): the
# search leaves an OCANNL process measurably slower per launch, so step times from it are not the
# artifact's. These are the only cells a searching process is a violation for.
TWO_PASS_CELLS = {("ocannl", "tuned")}
# Cells that search or compile in the process that then times steps — by protocol, not by
# mistake. What that costs them is MEASURED (gh-ocannl-675, benchmarks/report-gh675-cuda.md and
# the ROCm table in the issue): per box, tinygrad's beam +6.4%/+14.9% on mlp_small (CUDA/ROCm) and
# `torch.compile` -12.0%/+7.2% — which does not even keep its sign across the two. No cell clears
# a ~10% line on both boxes, so they stay single pass and are stated rather than gated; saying
# nothing would read as though the question applied to OCANNL alone.
SAME_PROCESS_CELLS = {("tinygrad", "beam"), ("pytorch", "compiled")}
# How a cell's measurement pass is rendered in the report's `pass` column (gh-ocannl-644). A
# runner predating the `searched` field, or one whose probe could not read its framework's
# internals, cannot answer — which is not the same as a violation.
PROVENANCE_MARK = {
    "REPLAY": "replay",
    "SEARCH-PASS": "**SEARCH PASS**",
    "NO-SEARCH": "no search",
    "SAME-PROCESS": "same-process",
    "CACHED": "cached",
    "UNKNOWN": "?",
}
# Report row order within a workload: precision-major, f32 first (it is the reference's precision
# and every non-OCANNL cell's), then the reduced precisions; a gate-cost leg sorts directly after
# the storage precision it varies; p50-ascending within each group.
PRECISION_ORDER = ["f32", "bf16", "f16"]
# The precision axis is a property of the runner, not of OCANNL: only bench_mlp and bench_gpt
# implement BENCH_PRECISION, so reduced-precision cells are generated for those models only.
PRECISION_MODELS = ("mlp", "gpt")
# The gh-ocannl-492 task-5 gate-cost legs: f16 with the dynamic loss-scaling gate replaced by a
# fixed scale (no gate, no host read) or by the fused on-device gate sampled every N steps. They
# vary how the optimizer's inf/nan gate is paid for, so they need a workload that HAS an optimizer
# — on a forward-only fixture they are reported as not applicable rather than silently dropped
# (gh-ocannl-551).
GATE_LEG_ENV = {"f16-static": {"BENCH_STATIC_SCALE": "1"}}
GATED_RE = re.compile(r"^f16-gated([1-9][0-9]*)$")

# The GPU column of the matrix, per --gpu choice: OCANNL backend, PyTorch device,
# tinygrad device. The CPU column (cc / cpu / CPU) is always run.
GPU_DEVICES = {
    "metal": ("metal", "mps", "METAL"),
    "cuda": ("cuda", "cuda", "CUDA"),
    # OCANNL's hip backend (AMD ROCm/HIP). PyTorch exposes HIP as its "cuda" device — on
    # Linux ROCm and, since ROCm 7.2, on Windows via AMD's official wheels (main() probes
    # torch.version.hip and drops the column when the venv's torch is not a HIP build).
    # tinygrad uses AMD on Linux ROCm and its OpenCL device (CL) on Windows.
    "hip": ("hip", "cuda", "AMD" if platform.system() == "Linux" else "CL"),
    "none": (None, None, None),
}

# Known-pathological cells excluded from the default matrix, as
# (workload, backend, variant, precision), where precision None means "at every precision" — a
# scheduling pathology has no dependence on the storage precision, so adding a precision axis must
# not let one back in through the bf16/f16 columns. Written as set(), not {}, so that emptying it
# does not silently turn it into a dict.
# Currently empty: the metal-default-schedule pathologies (gpt2_mini 81 s/step -> ~0.3 s,
# lenet 3.2 s/step + parity FAIL -> 0.22 s exact, mlp_wide >10 s/step -> 6 ms) were fixed by
# lowering the default GPU schedule's serial-fallback threshold, promoting statement-crossing
# Local scratch at fission, and working around a Metal compiler miscompilation of scalar
# read-modify-write accumulation (see arrayjit/lib/c_syntax.ml volatile_scalar_rmw and
# benchmarks/runners/ocannl/bench_metal_bug.ml).
# cifar_conv metal/tuned USED to belong here: the search completed but the post-tune re-init hung
# the process (Metal reinit-after-tune race, PR #109/#174). Retested at gh-ocannl-538 with the
# byte-for-byte command orchestrate issues for that cell — completed in ~4 min wall, no hang,
# search 230.3 s against the 2069 s the same standalone search took in the gh-476 sweep, and the
# full sweep reproduced its p50 to 0.005% (277.915 vs 277.928 ms). The searches got cheap enough to
# stop provoking it somewhere between those sweeps; gh-ocannl-532 (an unparallelized candidate is no
# longer dispatched on a GPU backend) is the likeliest reason.
SKIP_CELLS = set(
    # mlp_wide/cifar_conv/cifar_stride hip/tuned USED to belong here: the search appeared to wedge,
    # but perf showed 100% of the time in libhsa-runtime64 busy-waiting on a dispatch — the
    # autotuner was timing the unparallelized serial baseline, i.e. the whole training step in one
    # work-item, four times (warmup plus autotune_repeats). Hours per run on gfx1151, with
    # Windows-side driver timeouts and a lost display (gh-ocannl-532). Fixed in the tuner: an
    # unparallelized candidate is no longer dispatched on a GPU backend. Confirmed on the machine
    # that produced the symptom (gfx1151 / WSL, ROCm), all three cells completing with the display
    # intact: mlp_wide 34 s wall (search 32.1 s, 1.70 ms/step), cifar_stride 132 s (125.3 s,
    # 18.06 ms), cifar_conv 181 s (164.6 s, 61.09 ms).
)

sys.path.insert(0, str(HERE / "runners"))
from bench_common import read_st_metadata  # noqa: E402


def cell_skipped(workload, backend, variant, precision):
    """Whether a cell is in SKIP_CELLS, honouring the None-precision wildcard."""
    return (workload, backend, variant, precision) in SKIP_CELLS or (
        workload,
        backend,
        variant,
        None,
    ) in SKIP_CELLS


def cell_name(variant, precision):
    """Display name of an OCANNL cell's (scheduling, precision) pair.

    f32 cells keep their bare variant name, so the labels and reports of an f32-only matrix are
    unchanged; a reduced-precision cell is named by the product, e.g. `tuned/bf16`.
    """
    return variant if precision == "f32" else f"{variant}/{precision}"


def rendered_variant(result):
    """The variant as the report names it — with the search's pool size when one was chosen.

    A beam row's compile cost is a search cost, and the search's candidate pool changes it by a
    factor of three or four, so `beam` alone gives a historical row measured with tinygrad's
    upstream default, while `beam P=0` is the fleet default's no-pool serial search and `beam P=4`
    a four-worker one.  Keeping the pool in the row distinguishes old results whose
    `beam_parallel` is absent from results produced under the pinned default (gh-ocannl-760,
    gh-ocannl-843).
    """
    variant = result["variant"]
    parallel = result.get("beam_parallel")
    return variant if parallel is None else f"{variant} P={parallel}"


def precision_base(precision):
    """The storage precision a cell computes in, without a gate-leg suffix."""
    return precision.split("-", 1)[0]


def precision_rank(precision):
    base = precision_base(precision)
    rank = (
        PRECISION_ORDER.index(base) if base in PRECISION_ORDER else len(PRECISION_ORDER)
    )
    # A gate leg sorts right after the plain cell of the same storage precision: it is a variant
    # of how that precision's optimizer step is gated, not a precision of its own.
    return rank * 10 + (0 if precision == base else 1)


def parity_tol(precision, regime="exact"):
    """The parity envelope a cell is gated at: its storage precision's, or the approximate regime's.

    The approximate envelope is the loosest, so a reduced-precision approximate cell is gated at
    it too -- the max is the contract, should a precision envelope ever be the looser one.
    """
    tol = PARITY_TOL_PRECISION.get(precision_base(precision), PARITY_TOL)
    return max(tol, PARITY_TOL_APPROX) if regime == "approximate" else tol


def regime_of(result):
    """The numerics regime a row was measured and gated in; rows predating the column are exact."""
    return result.get("regime", "exact")


def regime_label(regime):
    """The cell-label suffix naming a non-exact regime.

    Exact labels are unchanged, so an exact-only matrix's labels and reports read as they always
    did; an approximate cell is `default [approximate]`, in the label as in the failure list.
    """
    return "" if regime == "exact" else f" [{regime}]"


def ocannl_regime_args(regime):
    """The flags an OCANNL cell is dispatched with for its regime: the profile IS the regime.

    The exact regime passes no profile rather than `--ocannl_profile=performance`, so the exact
    cells keep measuring what they always measured -- the `tuned` variant's search is the runner's
    (BENCH_TUNE), not a profile's. A commandline-picked profile outranks the benchmark config file
    and the environment, which is what lets the same process tree run both regimes.
    """
    return [] if regime == "exact" else [f"--ocannl_profile={regime}"]


def torch_regime_args(regime):
    """The flags the torch runner takes for its regime; its exact arm is its default."""
    return [] if regime == "exact" else ["--regime", regime]


def tinygrad_regime(regimes):
    """The regime tinygrad's one cell stands in.

    tinygrad has no exact pin -- its default reassociates freely -- so it is the counterpart of the
    approximate regime whenever a sweep has one, and the exact-gated row it always was otherwise.
    """
    return "approximate" if "approximate" in regimes else "exact"


def runner_regime(result):
    """The regime a runner's own process reports having run in, or None for a runner predating it.

    OCANNL names the profile it resolved (`profile`, null for none -- and a `reproducible` or
    `performance` profile is still the exact regime); the torch runner names the regime it was
    asked for (`runner_regime`, a key distinct from the sweep's `regime` stamp so the stamp
    cannot mask it).
    """
    if result.get("framework") == "ocannl":
        if "profile" not in result:
            return None
        return "approximate" if result["profile"] == "approximate" else "exact"
    return result.get("runner_regime")


def regime_knob_leaks(result):
    """The approximate-payload keys an OCANNL row resolved from a source its regime does not own.

    None when the runner did not report them. The regime is the resolution of those keys, not the
    profile's name (Codex P1 on PR #661): `exact` means every one of them at its built-in default
    -- the sweep passed no profile, and nothing else may have set one, an ambient
    OCANNL_TF32_MATMULS=true included -- and `approximate` means every one of them from the
    approximate profile, so an explicit override beside the profile is neither regime. The runner
    derives the key list from the payload itself, so a rewrite gate joining the profile is gated
    here without a second list. A sweep under another profile (`reproducible` pins these keys to
    exact values, from the profile) is thereby not an exact-regime sweep: the row says which key
    came from where, and the remedy is to run without it.
    """
    knobs = result.get("regime_knobs")
    if knobs is None:
        return None
    regime = regime_of(result)
    leaks = {}
    for key, knob in sorted(knobs.items()):
        source = knob.get("source", "?")
        owned = (
            source == "default"
            if regime == "exact"
            else source.startswith("profile 'approximate'")
        )
        if not owned:
            leaks[key] = f"{key}={knob.get('value', '?')} ({source})"
    return leaks


def regime_check(results):
    """Rows whose runner reports a regime other than the one the sweep dispatched (gh-ocannl-719).

    The row's `regime` is what the sweep asked for; the runner says what its process resolved,
    as the profile it picked and as where each of the profile's keys resolved from
    (`regime_knob_leaks`). A mismatch is a cell that ran under flags its regime does not own --
    an ambient OCANNL_PROFILE or OCANNL_TF32_MATMULS the process read, a config file picking a
    profile the commandline was expected to override -- and is labelled as a regime it did not
    run in, which no parity gate catches (an exact cell passes the approximate envelope too).
    `regime_mismatch` says what ran, in words the report prints. Rows from a runner predating
    the fields are not checked.
    """
    mismatched = []
    for r in results:
        actual = runner_regime(r)
        if actual is None:
            continue
        leaks = regime_knob_leaks(r)
        if actual != regime_of(r):
            r["regime_mismatch"] = actual
        elif leaks:
            r["regime_mismatch"] = f"{actual} but " + ", ".join(leaks.values())
        else:
            continue
        mismatched.append(r)
    return mismatched


def precision_spec(spec):
    """--precision argument: a storage precision, or one of the f16 gate-cost legs."""
    if spec in ("bf16", "f16") or spec in GATE_LEG_ENV or GATED_RE.match(spec):
        return spec
    raise argparse.ArgumentTypeError(
        f"unknown precision {spec!r}; expected bf16, f16, f16-static, or f16-gatedN"
    )


def precision_env(precision):
    """The BENCH_* environment a precision cell is dispatched with."""
    env = {"BENCH_PRECISION": precision_base(precision)}
    env.update(GATE_LEG_ENV.get(precision, {}))
    gated = GATED_RE.match(precision)
    if gated:
        env["BENCH_GATE_INTERVAL"] = gated.group(1)
    return env


def cell_env(base, fixture, variant, precision):
    """The environment an OCANNL cell is dispatched with."""
    env = dict(
        base,
        BENCH_FIXTURE=str(fixture),
        BENCH_TUNE="1" if variant == "tuned" else "0",
        BENCH_MATERIALIZE="1" if variant == "materialized" else "0",
        # A gate leg carries its own BENCH_* flags; clear the others so a stray value from the
        # caller's environment cannot leak into a cell.
        BENCH_STATIC_SCALE="0",
        BENCH_GATE_INTERVAL="0",
    )
    # Applied after, not as keywords: a gate leg's flags collide with the cleared defaults above
    # and dict() rejects duplicate keywords.
    env.update(precision_env(precision))
    return env


def precision_unavailable(model, mode, precision):
    """Why this workload cannot express this precision cell, or None if it can.

    An unavailable cell is reported (in the run log and the report) instead of quietly not being
    run: an empty row and an unrun row are otherwise indistinguishable (gh-ocannl-551).
    """
    if model not in PRECISION_MODELS:
        return f"the {model} runner has no BENCH_PRECISION support"
    if (precision in GATE_LEG_ENV or GATED_RE.match(precision)) and mode != "train":
        return (
            "gate-cost legs measure the optimizer's loss-scaling gate; this workload is "
            "forward-only (mode: infer)"
        )
    return None


def ocannl_exe(model):
    return ROOT / f"_build/default/benchmarks/runners/ocannl/bench_{model}.exe"


def _group_observation(proc, allow_zombie_gone=False):
    return proc.observe(allow_zombie_gone=allow_zombie_gone)


def _remaining_note(observation):
    if observation is cell_group.SURVIVORS:
        return "A MEMBER SURVIVED SIGKILL"
    return "GROUP LIVENESS IS UNKNOWN AFTER SIGKILL"


def kill_cell_group(proc):
    """Apply the shared TERM/KILL/reap mechanism and return its output and observation."""
    result = cell_group.terminate(
        proc,
        CELL_KILL_GRACE_S,
        observe=lambda allow_zombie_gone: _group_observation(proc, allow_zombie_gone),
    )
    out = result.stdout
    if isinstance(out, bytes):
        out = out.decode(errors="replace")
    return out or "", result.observation


_cancellation = cell_group.CancellationDeferral(
    "orchestrate",
    cleanup_message="the subprocess it was running, and its process group, were killed first",
)


def install_termination_handler():
    """Make a SIGTERM to the sweep reach the cell the sweep is running (gh-ocannl-760 review).

    The same `start_new_session` that lets the cap reach a cell's descendants detaches them from
    the sweep's own signals, and Python's default SIGTERM action exits the interpreter without
    unwinding — so a job cancellation, a scheduler's time limit or a plain `kill` on the sweep
    would leave the runner and its whole worker pool orphaned, holding the GPU, with nobody left
    to reap them. The handler turns the signal into an ordinary exception, which is all `run_cell`
    needs: its `except BaseException` already kills the group and re-raises.

    Installed from `main` rather than at import, since a process that merely imports this module
    (the unit tests, a notebook) has its own idea of what SIGTERM should mean.
    """

    _cancellation.install()


# Where a cancellation must not land, and the signal that tried to. Two such stretches, and they
# fail the same way — a cell left alive on the GPU with the sweep gone (gh-ocannl-760 review):
#
#   - between `_execute_child` starting a cell and `Popen` returning it, no name refers to the new
#     process, so an exception raised there unwinds past a cleanup that has nothing to clean;
#   - inside the kill itself, a signal raises out of the `except` clause that was doing the
#     killing — and a sibling `except BaseException` does not catch what another `except` clause
#     raises — so the escalation stops halfway and the group survives the sweep.
#
# The handlers therefore DEFER inside these stretches rather than raise, and the outermost one
# re-raises on the way out: by then the cell is bound and its group is dead, which is the state a
# cancellation wanted in the first place. What this must NOT do is block the signals with
# `pthread_sigmask`: the mask is inherited across fork/exec, so every cell would start with
# SIGTERM blocked — the graceful phase of its own kill would do nothing, every cap would cost the
# full grace before SIGKILL (measured: 1.0 s to 11.5 s per killed cell), and no runner would ever
# get to flush. `CancellationDeferral` owns this state machine for both benchmark drivers; the
# constructor argument above keeps the orchestrator's more specific successful-cleanup message.


def run_supporting(cmd, cwd=None, env=None, capture_output=False, check=False, timeout=None):
    """A subprocess the sweep runs for ITSELF — a build, a device probe — in its own group.

    Same discipline as a cell, for the same reason: `dune build` forks compilers, the probes
    import frameworks that spawn helpers, and a cancellation arriving here would otherwise leave
    them running while the sweep exits (gh-ocannl-760 review). `subprocess.run` kills its direct
    child on an exception but knows nothing of that child's own children; the group does.

    Returns the `CompletedProcess` that `subprocess.run` would have.
    """
    with _cancellation.deferring():
        return _run_supporting(cmd, cwd, env, capture_output, check, timeout)


def _run_supporting(cmd, cwd, env, capture_output, check, timeout):
    proc = None
    try:
        # The same window a cell gets, for the same reason: between `_execute_child` and `Popen`
        # returning there is no name for the new process, so a cancellation there orphans a
        # freshly isolated group (gh-ocannl-760 review).
        proc = cell_group.spawn(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=capture_output,
        )
        with _cancellation.cancellable():
            out, err = proc.communicate(timeout=timeout)
    except BaseException:
        if proc is not None:
            _, remaining = kill_cell_group(proc)
            if remaining is not cell_group.GONE:
                # The other exit from this function, and the same survivor. Here the sweep is
                # already leaving, so there is nothing to stop — but the operator is about to be
                # told the group was killed, and it was not: a cancellation that reports a clean
                # exit over a process still holding the device is how the next run gets measured
                # against it (gh-ocannl-760 review).
                raise cell_group.CleanupFailed(
                    "orchestrate: a member of the process group of "
                    f"{cmd[0]}: {_remaining_note(remaining)} ({remaining.value}) — clear the "
                    "survivors before re-running anything on this box"
                )
        raise
    initial = _group_observation(proc)
    if initial is not cell_group.GONE:
        # Same reason as a cell's leftover sweep: `communicate` returned because the LEADER
        # exited, and a build's compiler worker or a probe's framework helper can outlive it
        # holding the GPU (gh-ocannl-760 review).
        _, remaining = kill_cell_group(proc)
        if remaining is not cell_group.GONE:
            # And here the sweep DOES stop, where a cell's survivor only fails that cell. This
            # runs before any cell is dispatched, so a member still holding the device would be
            # shared by every row the sweep is about to produce — there is no subset of the
            # results to disbelieve, and a cap cannot bound damage that is already in all of
            # them (gh-ocannl-760 review).
            raise cell_group.CleanupFailed(
                f"orchestrate: {cmd[0]}: {_remaining_note(remaining)} ({remaining.value}). "
                "Every cell of this sweep could be measured against it, so "
                "nothing is dispatched: clear the survivors and re-run."
            )
    else:
        proc.close()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out, err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run_cell(label, cmd, env=None, cwd=None, timeout=None, on_incomplete=None):
    """Run one cell; return `(result, failure_note)`.

    `failure_note` is None when the cell produced a result line, and otherwise says what went
    wrong in a form the run's failure list can carry: a plain non-zero exit, or a cell that
    outlived `timeout` and was killed (gh-ocannl-760). `on_incomplete(killed)` is called on every
    path that ends without a result — the cap's kill, an interrupt, and an ordinary failure — and
    is the cell's chance to say, and where it can undo, what a search that stopped midway does to
    the cache it was writing; its sentence is appended to the failure note. `killed` separates the
    two cases, because what is safe to DO about them differs: see `quarantine_tinygrad_cache`.

    The whole body runs inside one cancellation-deferral window whose only hole is the
    `communicate` wait. Chasing that protection stretch by stretch is how several review rounds
    went — the spawn, then the kill, then the leftover probe, then the assignments between them —
    and the invariant behind all of them is single: from the moment a cell exists until its group
    is dead, a cancellation must not unwind anything, because what it unwinds is left running on
    the GPU with the sweep gone.
    """
    with _cancellation.deferring():
        return _run_cell(label, cmd, env, cwd, timeout, on_incomplete)


def _run_cell(label, cmd, env, cwd, timeout, on_incomplete):
    print(f"--- {label}", flush=True)
    timed_out = False
    remaining = cell_group.GONE
    cache_note = ""
    proc = None
    try:
        proc = cell_group.spawn(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with _cancellation.cancellable():
            # The wait a cancellation is FOR: here, and only here, a signal should raise at once.
            stdout, _ = proc.communicate(timeout=timeout or None)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, remaining = kill_cell_group(proc)
        # Before anything fallible: the kill is what tore the cache, so undoing it must not be
        # reachable only through code that can raise first (the optional cell log below writes
        # to an operator-supplied directory, which can be unwritable or full). Losing the sweep to
        # that would leave the partial cache.db in place AND lose the failure record.
        cache_note = (on_incomplete(True) if on_incomplete else "") or ""
    except BaseException:
        # An operator's Ctrl-C reaches the sweep alone once the cell is in its own group, so the
        # cell would keep running (and keep the GPU) after the sweep is gone. Take it with us —
        # and a cell killed here was killed midway just as surely as one over the cap, so the
        # cache it was writing gets the same treatment. Ctrl-C on a wedged beam cell is the
        # likeliest way anyone meets this bug by hand, and a retry over the cache that kill left
        # behind is not the pass it claims to be (gh-ocannl-760 review).
        cleanup_failure = None
        if proc is not None:
            _, outlived = kill_cell_group(proc)
            if outlived is not cell_group.GONE:
                # The one branch that exits rather than records, so this is the operator's
                # only chance to hear it: the cancellation's own message says the cell was
                # killed, and a retry started over a survivor that still holds the device
                # would be measured against it (gh-ocannl-760 review).
                cleanup_failure = cell_group.CleanupFailed(
                    f"{label}: {_remaining_note(outlived)} ({outlived.value}) — clear the "
                    "survivors before re-running anything on this box"
                )
            if on_incomplete:
                print(f"!!! {label} interrupted; {on_incomplete(True)}", flush=True)
        if cleanup_failure is not None:
            raise cleanup_failure
        raise
    stdout = stdout or ""
    leftovers = ""
    stuck = cell_group.GONE
    initial = _group_observation(proc)
    if not timed_out and initial is not cell_group.GONE:
        # The cell is done and something it spawned is not. `communicate` returned because the
        # LEADER exited and the pipe closed, which says nothing about a worker that redirected
        # its own output — and that worker still holds the GPU, so every later cell of the
        # sweep would be measured against it. Collect it here, on the ordinary path, not only
        # on the cap's.
        _, stuck = kill_cell_group(proc)
        leftovers = (
            "the cell left members of its process group behind; they were killed"
            if stuck is cell_group.GONE
            else "the cell left members of its process group behind AND "
            f"{_remaining_note(stuck)} ({stuck.value})"
        )
        print(f"!!! {label}: {leftovers}", flush=True)
    elif not timed_out:
        proc.close()
    if CELL_LOG_DIR:
        # A cell's own output is otherwise discarded on success, which throws away exactly the
        # evidence a measurement sweep is asked to report: with autotune_log=true the search
        # pass's candidate lines (seeded vs timed, FAILED, dedup, split-reduce evictions) live
        # here and nowhere else, and re-running the searches to recover them costs as much as the
        # sweep. Off unless BENCH_CELL_LOG_DIR is set, so the default run is unchanged.
        try:
            CELL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in label)
            (CELL_LOG_DIR / f"{safe}.log").write_text(stdout)
        except OSError as exc:
            # An unwritable log directory is a lost convenience, not a lost sweep.
            print(f"!!! {label}: could not write the cell log ({exc})", flush=True)
    line = next((l for l in reversed(stdout.splitlines()) if l.startswith("{")), None)
    if timed_out:
        # A cell over the cap is a FAILURE, not a slow measurement: whatever it was doing, it was
        # not making progress in the time the same search takes in every other repeat, and a
        # result line salvaged from a killed process would be a partial run's (gh-ocannl-760).
        print(stdout[-4000:])
        note = (
            f"TIMED OUT after {timeout:g}s (cap; --cell-timeout raises it, 0 disables) — "
            "killed the cell's whole process group"
        )
        if remaining is not cell_group.GONE:
            # A member may have outlived SIGKILL, or the platform could not prove the group gone;
            # every cell measured after it in this run was measured against it. The sweep goes on
            # (that is the point of the cap) but no later row from this run is quotable until
            # someone has cleared the survivor and re-run them.
            note += (
                f"; {_remaining_note(remaining)} ({remaining.value}) — every later cell in "
                "this run may have been measured "
                "against it, so clear the survivors and re-run them"
            )
        if cache_note:
            note += f"; {cache_note}"
        print(f"!!! {label} {note}", flush=True)
        if remaining is not cell_group.GONE and _cancellation.held_signal is not None:
            raise cell_group.CleanupFailed(note)
        return None, note
    if stuck is not cell_group.GONE:
        # The cell ran to completion — it may even have printed a result line — but something it
        # spawned outlived SIGKILL and still holds the device. That makes THIS cell's own timing
        # suspect (it shared the device with a process nobody scheduled) and every later cell of
        # the run too, so it is a runner failure whatever the leader's exit code says: a warning
        # on a console nobody keeps is not a record, and the report would otherwise publish the
        # row and everything after it (gh-ocannl-760 review).
        note = (
            "the cell left members of its process group behind AND "
            f"{_remaining_note(stuck)} ({stuck.value}) — it may still hold the device, so this cell's "
            "own timing and every later cell's in this run may have been measured against it; "
            "clear the survivors and re-run"
        )
        # This branch returns early, so the cache handling has to happen here too: part of the
        # search was forcibly interrupted (that is what `stuck` means), which is exactly the torn
        # cache the cap's path quarantines (gh-ocannl-760 review).
        stuck_cache_note = (on_incomplete(True) if on_incomplete else "") or ""
        if stuck_cache_note:
            note += f"; {stuck_cache_note}"
        print(f"!!! {label} {note}", flush=True)
        if _cancellation.held_signal is not None:
            raise cell_group.CleanupFailed(note)
        return None, note
    if proc.returncode != 0 or line is None:
        print(stdout[-4000:])
        print(f"!!! {label} failed (exit {proc.returncode})", flush=True)
        note = f"exit {proc.returncode}" if proc.returncode else "no result line"
        if leftovers:
            note += f"; {leftovers}"
        # A search that exits nonzero partway through leaves the same partial cache a killed one
        # does — some arms committed, the rest never run — and the next attempt over it reports a
        # provenance nobody wrote. The cell says so here as it does on the kill path, with
        # `killed=False`: what differs is not the risk but what may be DONE about it
        # (gh-ocannl-760 review).
        # `killed` is about whether anything of this cell was interrupted, not about how the
        # LEADER ended: if the leftover sweep above killed a member, a candidate worker may have
        # been cut off mid-write, which is the torn cache the kill path quarantines
        # (gh-ocannl-760 review).
        failed_note = (on_incomplete(bool(leftovers)) if on_incomplete else "") or ""
        if failed_note:
            note += f"; {failed_note}"
        if failed_note:
            print(f"!!! {label}: {failed_note}", flush=True)
        return None, note
    result = json.loads(line)
    if leftovers and on_incomplete:
        # The row stands — the cell ran and printed its result — but a member of its group was
        # killed on the way out, possibly mid-write, and the cache it shares with every later
        # beam cell is in the state a kill leaves (gh-ocannl-760 review). Same handling as the
        # failure paths; it is the cache that is at issue here, not this row.
        killed_note = on_incomplete(True)
        if killed_note:
            print(f"!!! {label}: {killed_note}", flush=True)
    print(
        f"    p50 {num(result['step_ms']['p50'], '.3f')} ms, "
        f"compile {num(result['compile_s'], '.2f')} s",
        flush=True,
    )
    return result, None


def tinygrad_cachedb(env=None):
    """Path of the kernel cache a tinygrad cell writes — asked of the library that writes it.

    `CACHEDB` is composed by tinygrad's own helpers (an explicit `CACHEDB`, else `XDG_CACHE_HOME`
    or the platform cache dir), so it is read out of the venv's tinygrad rather than recomputed
    here: a rule copied into this file would go stale silently and quarantine the wrong file, or
    nothing. The composition below is only the fallback for a probe that could not run.
    """
    try:
        probe = run_supporting(
            [str(VENV_PY), "-c", "from tinygrad.helpers import CACHEDB; print(CACHEDB)"],
            env=env,
            cwd=str(HERE),
            capture_output=True,
            timeout=TINYGRAD_PROBE_TIMEOUT_S,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return Path(probe.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # No venv interpreter, or a tinygrad that cannot be imported here. The kill path must
        # still finish and still say something true about the cache.
        pass
    env = env if env is not None else os.environ
    if env.get("CACHEDB"):
        return Path(env["CACHEDB"])
    default_home = "~/Library/Caches" if platform.system() == "Darwin" else "~/.cache"
    cache_dir = env.get("XDG_CACHE_HOME") or os.path.expanduser(default_home)
    return Path(cache_dir) / "tinygrad" / "cache.db"


def quarantine_tinygrad_cache(env=None, enabled=True, killed=True):
    """Move a killed beam search's kernel cache aside; return what was done, for the failure note.

    The consequence a killed search leaves behind, recorded on the HIP leg of gh-ocannl-760: the
    search writes into a single sqlite file, so a kill midway leaves it holding whatever that
    search had committed so far, and the next run over that cache neither replays a complete
    result nor searches from scratch — while `searched` reports one of the two. A retry over it is
    not the pass it claims to be, which is exactly the retry an operator makes after a wedge.

    tinygrad's cache is one file (plus its `-wal`/`-shm` siblings), so the clean handling is
    available: rename it aside, with its siblings, so the retry starts cold and the torn cache is
    still there to look at. Renamed rather than deleted — it is evidence — and only ever after the
    kill, when the cell is already a failure. `enabled=False` (`--no-cache-quarantine`) leaves the
    file alone and only names the risk.

    OCANNL's tuned cell needs no equivalent: its schedule cache is a directory whose entries are
    written through `Utils.Atomic_file` (staged, then committed by rename), so a killed search
    leaves complete entries and at most a staging file the next writer sweeps. Nothing there is
    torn; what a retry over it costs is a mixed pass, which `ocannl_cache_note` states.
    """
    db = tinygrad_cachedb(env)
    if not db.exists():
        return f"tinygrad kernel cache {db} does not exist; nothing to quarantine"
    risk = (
        f"a search killed midway leaves {db} holding a partial result, and the next run over it "
        "reports a `searched` verdict nobody wrote deliberately"
    )
    if not killed:
        # The cell exited on its own, which can mean it never searched at all (a bad flag, a
        # missing compiler) as easily as it can mean it searched halfway. The cache is SHARED
        # with every later cell of the sweep, so moving it aside on an ordinary failure would
        # cost all of them their warm kernels for a risk that may not exist. Name it instead,
        # and let whoever quotes the retry decide (gh-ocannl-760 review).
        return f"CACHE AT RISK: {risk} (left in place: the cell exited rather than being killed)"
    if not enabled:
        return f"CACHE AT RISK: {risk} (--no-cache-quarantine: left in place)"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    quarantined = db.with_name(f"{db.name}.wedged-{stamp}")
    # A second-resolution stamp is not a unique name: two cells killed within the same second (a
    # low `--cell-timeout`, or a sweep whose whole GPU column wedges at once) would land on it
    # twice, and the second `os.replace` would overwrite the first cell's quarantined database
    # with no trace — destroying exactly the evidence the rename exists to keep (gh-ocannl-760
    # review). Take the first free name instead, counting the family as one: the sidecars must
    # end up beside THEIR database.
    suffixes = ("", "-wal", "-shm")
    attempt = 1
    while any(Path(f"{quarantined}{suffix}").exists() for suffix in suffixes):
        attempt += 1
        quarantined = db.with_name(f"{db.name}.wedged-{stamp}.{attempt}")
    moved = []
    rolled_back = []
    # The sidecars are renamed UNDER the quarantined database's name, not beside their own:
    # sqlite finds a write-ahead log only at `<database>-wal` (and its index at `-shm`), so
    # `cache.db-wal.wedged-<stamp>` next to `cache.db.wedged-<stamp>` is a database that opens
    # without the very writes the killed search had not checkpointed — the evidence this move
    # exists to preserve, silently dropped (gh-ocannl-760 review).
    for suffix in suffixes:
        path = Path(f"{db}{suffix}")
        if not path.exists():
            continue
        dest = Path(f"{quarantined}{suffix}")
        try:
            os.replace(path, dest)
        except OSError as exc:
            # All or nothing: a half-moved family is worse than an unmoved one — a database
            # separated from its write-ahead log opens without the writes it holds, and the cache
            # left behind gains a stale sidecar the next tinygrad process will read against a
            # database it never belonged to (gh-ocannl-760 review). Put back what was moved.
            undone = []
            for done, origin in reversed(rolled_back):
                try:
                    os.replace(done, origin)
                except OSError:
                    undone.append(done.name)
            trouble = f" (and {', '.join(undone)} could not be put back)" if undone else ""
            return f"CACHE AT RISK: {risk}; could not quarantine it ({exc}){trouble}"
        rolled_back.append((dest, path))
        moved.append(dest.name)
    return f"quarantined the tinygrad kernel cache to {db.parent}/{{{', '.join(moved)}}} — {risk}"


def beam_cell_env(base_env, beam_parallel):
    """The environment a tinygrad beam cell runs in: `PARALLEL` is the orchestrator's to say.

    `--beam-parallel N` sets it, and the CLI's pinned default passes zero.  The ``None`` case is
    retained for programmatic callers and historical-result tests; it means tinygrad's own
    default, and removes an exported `PARALLEL` rather than letting the invoking shell silently
    choose a pool size that the label and result do not name (gh-ocannl-760 review).
    """
    env = dict(base_env)
    if beam_parallel is None:
        env.pop("PARALLEL", None)
    else:
        env["PARALLEL"] = str(beam_parallel)
    return env


def ocannl_cache_note(_killed=True):
    """What a killed OCANNL search leaves in its schedule cache, for the failure record.

    Nothing to undo — see `quarantine_tinygrad_cache` for why the two caches differ — but the
    retry's provenance is still not a from-scratch search's, and saying so where the failure is
    read is cheaper than rediscovering it from a `(cached)` compile cost.

    What it must NOT claim is that the retry reports REPLAY. A kill lands mid-search, so the
    retry's arms are mixed: those that finished replay, those that did not are searched again —
    and `search_provenance` reads `searched`, which is true whenever ANY arm searched, so the
    mixed pass reports SEARCHED. Its compile cost is then neither a from-scratch search's nor a
    replay's while wearing the from-scratch label, which is precisely the misreading this note
    exists to prevent (gh-ocannl-760 review).
    """
    return (
        "autotune_cache/ keeps the arms that finished before the kill (entries are committed "
        "atomically, so none of them is torn); a retry replays those and searches the rest, and "
        "since the pass reports SEARCHED whenever any arm searched, its search cost is a partial "
        "one wearing a from-scratch label — wipe autotune_cache/ for a comparable search timing"
    )


def finite(x):
    """Whether a JSON number from a result line is a real number this report can compare.

    A runner emits `null` for a value it has but cannot express in JSON -- a diverged loss, a time
    it never measured (gh-ocannl-676) -- so `None` reaching here means "ran, and this number is not
    a number", not "runner failed". `NaN` is accepted by Python's `json.loads` (its non-standard
    extension) and so can still arrive from an older result file, and is the same fact.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def diverged_at(losses):
    """The index of the first non-finite loss, or None for a trajectory that stayed finite."""
    return next((i for i, loss in enumerate(losses) if not finite(loss)), None)


def finite_prefix(losses):
    """The steps before the trajectory left the finite numbers.

    Nothing after the first non-finite loss is evidence about the run: a finite value FOLLOWING a
    NaN is whatever the arithmetic settled on afterwards, not drift the reference can be compared
    against, and not movement. Both readers of a trajectory take this prefix, so neither can report
    a number the DIVERGED verdict says is meaningless (gh-ocannl-676).
    """
    cut = diverged_at(losses)
    return losses if cut is None else losses[:cut]


def json_safe(obj):
    """`obj` with every non-finite float replaced by None, recursively.

    `json.dumps` writes `NaN` / `Infinity` for those, which its own loader accepts and no other
    JSON reader does -- results.jsonl is read by jq, by pandas and by the next session's scripts,
    so what this sweep writes has to be JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def num(x, spec):
    """A number for the report table, or `n/a` when the runner had none to give."""
    return format(x, spec) if finite(x) else "n/a"


def loss_moved(losses):
    """Whether a loss trajectory has more than floating-point-noise-level variation.

    Over the prefix before the first non-finite step: a diverged trajectory is not a stationary
    one, and reporting it as stationary names the wrong defect. A prefix shorter than two steps has
    nothing to say about movement, and the cell is reported as diverged instead.
    """
    prefix = finite_prefix(losses)
    if len(prefix) < 2:
        return False
    scale = max(max(abs(loss) for loss in prefix), 1e-6)
    return max(prefix) - min(prefix) > LOSS_MOVE_MIN_REL * scale


def parity_check(results):
    """Annotate each result with parity vs the reference run of the same workload."""
    by_workload = {}
    for r in results:
        by_workload.setdefault(r["workload"], []).append(r)
    for workload, rs in by_workload.items():
        # The reference is the exact torch CPU eager cell in every regime: an approximate torch
        # cell is a counterpart, gated against the exact one like every other approximate row.
        ref = next(
            (
                r
                for r in rs
                if (r["framework"], r["backend"], r["variant"]) == REFERENCE
                and regime_of(r) == "exact"
            ),
            None,
        )
        ref_prefix = finite_prefix(ref["losses"]) if ref is not None else []
        ref_diverged = ref is not None and len(ref_prefix) < len(ref["losses"])
        for r in rs:
            r["parity_loss_moved"] = loss_moved(r["losses"])
            r["diverged_at"] = diverged_at(r["losses"])
            if ref is not None:
                # Compared over the prefix both trajectories reached while still finite, so a
                # DIVERGED row's parity_max_rel is drift measured BEFORE it went and nothing else.
                n = min(len(finite_prefix(r["losses"])), len(ref_prefix))
                if n:
                    r["parity_max_rel"] = max(
                        abs(a - b) / max(abs(b), 1e-6)
                        for a, b in zip(r["losses"][:n], ref["losses"][:n])
                    )
            if r["diverged_at"] is not None:
                # The cell ran and its training blew up: a gate failure naming its cause, not a
                # missing cell and not a stationary one (gh-ocannl-676). Parity is meaningless past
                # the divergence, so what is reported is where it happened -- and, when there is a
                # finite prefix shared with the reference, how far it had drifted before it did.
                r["parity"] = "DIVERGED"
            elif ref is None or ref_diverged:
                # A diverged reference is no reference: comparing against a trajectory that blew up
                # would report the reference's defect as every other cell's.
                r["parity"] = "NO-REF"
            elif r is ref:
                r["parity"] = "REF"
                r["parity_max_rel"] = 0.0
            else:
                max_rel = r.get("parity_max_rel")
                regime = regime_of(r)
                tol = parity_tol(r.get("precision", "f32"), regime)
                r["parity"] = (
                    "PASS"
                    if max_rel is not None and max_rel < tol and r["parity_loss_moved"]
                    else "FAIL"
                )
                if regime != "exact":
                    # Whether the row's DRIFT also fits the exact envelope is reported, not assumed
                    # (gh-ocannl-719): a rewrite that changes nothing measurable is a finding too,
                    # and the envelope is calibrated from these verdicts rather than from faith.
                    # Drift alone: loss movement is the gate's own condition and is reported on
                    # its own, so a stationary row is not misattributed to excessive drift.
                    exact_tol = parity_tol(r.get("precision", "f32"))
                    r["parity_exact_envelope"] = bool(max_rel is not None and max_rel < exact_tol)


def check_fixture_digests(fixtures, digests_path=None, allow_unpinned=False):
    """What the fixtures ARE, checked before anything is measured on them (gh-ocannl-645).

    Returns `{fixture path: (sha256, origin)}` for stamping onto the results. Exits on a fixture
    whose bytes are nobody's in `fixtures/DIGESTS.txt` unless `allow_unpinned` — a differently
    generated fixture is consumed uniformly by every cell, so the cross-cell parity gate certifies
    it exactly as it certifies the intended workload, and the digest is the only thing that ties a
    published number to the workload the report names.

    `origin` is the box whose recorded bytes these are (gh-ocannl-759), None when nothing records
    them. It rides onto every row and report section, because the same workload spec has different
    bytes on different boxes: the digest says the numbers are on *some* pinned workload, the origin
    says *whose*, and only the pair makes a cross-box comparison an honest one.
    """
    entries = fixture_digest.read_digests(
        digests_path or HERE / "fixtures" / fixture_digest.DIGEST_FILE
    )
    measured = {}
    unpinned = []
    for fx in fixtures:
        verdict, sha, size, origins = fixture_digest.status(fx, entries)
        measured[fx] = (sha, origins)
        where = f" — {origins}'s bytes" if verdict == "MATCH" else ""
        print(f"fixture {fx.name}: sha256 {sha} ({size} bytes) — {verdict}{where}", flush=True)
        if verdict != "MATCH":
            print(f"  recorded: {fixture_digest.describe(fx.name, entries)}", flush=True)
            unpinned.append((fx, verdict))
    named = ", ".join(f"{fx.name} ({verdict})" for fx, verdict in unpinned)
    if unpinned and not allow_unpinned:
        sys.exit(
            f"refusing to measure {named}: these bytes are not the ones "
            "fixtures/DIGESTS.txt records for any origin, so a report of them would name a "
            "workload nothing pins. If they are an unrecorded box's copies, pin them with "
            f"`python3 {fixture_digest.cli_command()} --record --origin <box>` (no regeneration, "
            "and it leaves the other boxes' entries alone); if you deliberately regenerated, "
            "re-run gen_fixtures.py to re-record (the digest diff is the review); or pass "
            "--no-fixture-digest-check to measure them as they are."
        )
    if unpinned:
        print(
            f"MEASURING UNPINNED FIXTURES: {named} — results carry their measured digest, but "
            "nothing checked in describes them",
            flush=True,
        )
    return measured


def fixture_result_stamp(fixture, sha, origin, entries, measurement_boxes):
    """Fixture provenance that survives in every raw result row, partial or final.

    The declaration and its missing-entry reading are facts at measurement time. Re-reading a
    later DIGESTS.txt cannot recover them after the fleet or its records change, so both travel
    with the same stamp as the fixture digest and origin.
    """
    return {
        "fixture": fixture.name,
        "fixture_sha256": sha,
        "fixture_origin": origin,
        "measurement_boxes": list(measurement_boxes),
        "fixture_missing_origins": fixture_digest.missing_origins(
            fixture.name, entries, measurement_boxes
        ),
    }


def search_provenance(result):
    """What one OCANNL process did about searching: SEARCHED, REPLAY, NO-SEARCH or UNKNOWN.

    The single reading of `searched`, shared by every consumer, because the fact is three-valued
    and each boolean spelling of it has been wrong in a different place (gh-ocannl-644 review):
    a process can run a search, replay cached winners, or — under `autotune_search=false`, the
    reproducible profile — do neither and ship the untuned default. `not searched` therefore does
    not mean "replayed", which is what a `(cached)` label on its compile cost would claim.

    The `tune` object's totals are the evidence separating the second case from the third: an
    OCANNL cell that tuned anything reports them, and without them there is nothing to tell those
    two apart with, so the answer is UNKNOWN rather than a guess. Since gh-ocannl-677 the runner
    states the third case outright — `no_searches` counts the arms whose `Autotune` outcome was
    `search-disabled` or `pre-search-failure` — so it is read directly rather than recovered from
    two counters that are both zero. Older artifacts carry no such key, and the zero-zero reading
    stays for them.
    """
    searched = result.get("searched")
    if searched is None:
        return "UNKNOWN"
    if searched:
        return "SEARCHED"
    tune = result.get("tune")
    if not tune:
        return "UNKNOWN"
    if "no_searches" in tune:
        return "REPLAY" if tune.get("replays") else "NO-SEARCH"
    return "NO-SEARCH" if not tune.get("searches") and not tune.get("replays") else "REPLAY"


# gh-ocannl-626: what the timed artifact's emission actually did about tensor cores, for the
# report's `mma` column. A cell whose shipped arm carries a `Tensorize` but whose kernels rendered
# the lane-0 scalar fallback measured scalar code under a tensorized label, and every perf number
# quoted from that row inherits the error — so those two states are shouted, not spelled quietly.
TENSORIZATION_MARK = {
    "TENSORIZED": "tensorized",
    "SCALAR-FALLBACK": "**SCALAR FALLBACK**",
    "NOT-EMITTED": "**NO MMA EMITTED**",
    "NOT-REQUESTED": "—",
    "UNKNOWN": "?",
}
# The two verdicts that mean "this row's tensorized label is not what ran".
TENSORIZATION_MISMATCH = ("SCALAR-FALLBACK", "NOT-EMITTED")


def shipped_arm(result):
    """The `tune` arm whose artifact produced this cell's step times, or None.

    `Train.tune_placements` runs several searches and ships one of them, so reading any arm — or
    the fastest one — would describe a schedule that was discarded (gh-ocannl-638). The runner
    states which arm shipped; that is the only one whose emission the timings measured.
    """
    tune = result.get("tune")
    if not tune:
        return None
    shipped = tune.get("shipped")
    for arm in tune.get("arms") or []:
        if arm.get("arm") == shipped:
            return arm
    return None


def tensorization_verdict(result):
    """Did this cell's timed artifact actually tensorize: the schedule's ask against the emission.

    Four answers, not two, because the interesting cases are the disagreements (gh-ocannl-626):
    TENSORIZED (asked or not, at least one tensor-core / SIMD-tile emission happened),
    SCALAR-FALLBACK (`Tile_mma` statements were emitted and every one of them declined to the
    lane-0 scalar path), NOT-EMITTED (the schedule carries a `Tensorize` and codegen emitted no
    `Tile_mma` at all), NOT-REQUESTED (nothing about this artifact claims tensor cores — not a
    defect, and the only one of the four that is not worth a mark).

    The emission half comes from `tune.shipped_mma`, the census of the routine whose steps were
    TIMED, and not from the arm named as shipped. A crowned arm candidate is not always the shipped
    artifact: a gh-555 flip refinement that wins ships under `shipped: "flip"` and is not an arm at
    all, and on the `timing_ctx` path the tuner recompiles the winner in the production context and
    falls back to the untuned default when that replay is rejected — in both cases the arm
    describes a schedule that was discarded, and reading it could claim `tensorized` over a routine
    that emitted no mma.

    The `tensorized` half — did the schedule ASK — has no such per-routine record, so it is still
    read off the shipped arm and only refines a `not-requested` emission into NOT-EMITTED. Where no
    arm can be identified (the flip case) that refinement is unavailable and the verdict stays
    NOT-REQUESTED: the artifact demonstrably emitted no `Tile_mma`, which is the fact the column
    reports; what is lost is at most the shout, never a false `tensorized`.

    None for a cell with no tune object at all — an eager framework, an untuned default — which has
    no census to consult; UNKNOWN for a runner predating either field, or one that reported arms
    without recording the shipped census, so a missing census never reads as a tensorized one.
    """
    tune = result.get("tune")
    if not tune:
        return None
    arm = shipped_arm(result) or {}
    if "shipped_mma" in tune:
        shipped = tune["shipped_mma"]
        if not shipped:
            # The key is there and empty: the harness reported arms and recorded no census. Not a
            # finding either way, and emphatically not a passing reading.
            return "UNKNOWN"
        label = shipped.get("tensorization")
    else:
        # An artifact predating `shipped_mma`. The arm is all there is; it is right whenever the
        # crowned candidate WAS the shipped artifact, which is the common case.
        if not arm:
            return None
        label = arm.get("tensorization")
    if label == "tensorized":
        return "TENSORIZED"
    if label == "scalar-fallback":
        return "SCALAR-FALLBACK"
    if label == "not-requested":
        return "NOT-EMITTED" if arm.get("tensorized") else "NOT-REQUESTED"
    return "UNKNOWN"


def tensorization_check(results):
    """Annotate each tuned cell with its `tensorization` verdict; return the mismatched ones."""
    mismatched = []
    for r in results:
        verdict = tensorization_verdict(r)
        if verdict is None:
            continue
        r["tensorization"] = verdict
        if verdict in TENSORIZATION_MISMATCH:
            mismatched.append(r)
    return mismatched


# How a two-pass cell's own verdict maps to the report's `pass` column: only a searching process
# is a protocol violation, and only a genuine replay may call the carried-over compile cost cached.
TWO_PASS_VERDICT = {
    "SEARCHED": "SEARCH-PASS",
    "REPLAY": "REPLAY",
    "NO-SEARCH": "NO-SEARCH",
    "UNKNOWN": "UNKNOWN",
}
# What a search pass's verdict says about the `compile s` carried over from it.
COMPILE_S_NOTE = {"REPLAY": " (cached)", "NO-SEARCH": " (no search)"}


def provenance_check(results):
    """Annotate each searching cell with which process produced its step times (gh-ocannl-644).

    OCANNL's two-pass protocol exists because a searching process is measurably slower per launch,
    so pass-1 step times understate the tuned artifact. Both passes emit the same
    framework/backend/variant/precision, and until the runner carried `searched` nothing in the
    artifact distinguished them: `report-gh612-hip.md` quoted pass-1 timings for fifteen revisions
    and only a reviewer reading the driver caught it.

    A two-pass cell is REPLAY (compliant) or SEARCH-PASS (its timings came from a process that
    searched — returned as a violation). NO-SEARCH is the third case: with `autotune_search=false`
    (the reproducible profile) a tuned cell neither searches nor replays, it ships the untuned
    default — nothing to gate, but calling that a replay would credit the row with a tuned artifact
    it does not have. A cell whose protocol searches in the timing process is
    SAME-PROCESS or, when it replayed its framework's own cache instead, CACHED; neither is a
    violation, because nothing yet says those frameworks pay for it (gh-ocannl-675) — the point of
    annotating them is that the report stops implying the question is OCANNL's alone. UNKNOWN is a
    runner predating the field, or a probe that could not read its framework's internals.

    Cells that neither search nor compile — eager, jit, an untuned OCANNL default — get no
    annotation: there is no process distinction for them to be on the wrong side of.
    """
    violations = []
    for r in results:
        cell = (r.get("framework"), r.get("variant"))
        two_pass = cell in TWO_PASS_CELLS
        if not two_pass and cell not in SAME_PROCESS_CELLS:
            continue
        searched = r.get("searched")
        if searched is None:
            r["provenance"] = "UNKNOWN"
        elif two_pass:
            verdict = search_provenance(r)
            r["provenance"] = TWO_PASS_VERDICT[verdict]
            if verdict == "SEARCHED":
                violations.append(r)
        else:
            r["provenance"] = "SAME-PROCESS" if searched else "CACHED"
    return violations


def failure_line(failure):
    """One `(label, note)` runner failure, as the run log and the report name it."""
    label, note = failure
    return f"{label} ({note})" if note else label


def beam_parallel_arg(text):
    """`--beam-parallel` as a worker count: zero (no pool) or more, never negative.

    tinygrad builds its pool for any truthy PARALLEL and passes the value straight to
    `multiprocessing.Pool`, which refuses a negative process count — so `-1` does not mean
    "unset" or "default", it means every beam cell of the sweep dies on a `ValueError` from
    inside the runner (gh-ocannl-760 review).
    """
    workers = int(text)
    if workers < 0:
        raise argparse.ArgumentTypeError(
            f"--beam-parallel must be 0 (no pool) or more (got {text!r})"
        )
    return workers


def cell_timeout_arg(text):
    """`--cell-timeout` as a number of seconds: finite and not negative, or an argparse error.

    Only zero (no cap) and a positive number of seconds mean anything here. A negative cap expires
    every `communicate` at once — killing every cell of the sweep and quarantining their caches on
    the way — and `nan`/`inf` raise from inside `communicate`, which the generic cleanup path
    turns into a killed cell and an aborted run. Both are worth refusing before the first process
    is spawned rather than discovering them cell by cell (gh-ocannl-760 review).
    """
    seconds = float(text)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(
            f"--cell-timeout must be a finite number of seconds, 0 or more (got {text!r})"
        )
    return seconds


def report(results, out_dir, unavailable=(), failures=(), digests_path=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    digests_path = digests_path or HERE / "fixtures" / fixture_digest.DIGEST_FILE
    digest_entries = fixture_digest.read_digests(digests_path)
    current_measurement_boxes = fixture_digest.measurement_boxes(digests_path)
    # Old/in-memory callers may not have gone through main's measurement-time stamp. Enrich them
    # before serialization; rows that already carry the fact remain authoritative if the checked-
    # in declaration changed between measurement and this later report rendering.
    for r in results:
        if "measurement_boxes" not in r:
            r["measurement_boxes"] = list(current_measurement_boxes)
        if "fixture_missing_origins" not in r and r.get("fixture"):
            r["fixture_missing_origins"] = fixture_digest.missing_origins(
                r["fixture"], digest_entries, r["measurement_boxes"]
            )
    recorded_box_sets = {tuple(r["measurement_boxes"]) for r in results}
    if len(recorded_box_sets) > 1:
        raise ValueError(
            "cannot combine result rows recorded under different measurement-box declarations: "
            + ", ".join(repr(boxes) for boxes in sorted(recorded_box_sets))
        )
    measurement_boxes = (
        list(next(iter(recorded_box_sets))) if recorded_box_sets else current_measurement_boxes
    )
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            # allow_nan=False so a non-finite value this sweep computed itself cannot slip out as
            # `NaN`: json_safe has already mapped the ones it expects, and anything left is a bug
            # worth raising over rather than writing an unreadable file (gh-ocannl-676).
            f.write(json.dumps(json_safe(r), allow_nan=False) + "\n")
    lines = []
    commit = run_supporting(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True
    ).stdout.strip()
    lines.append(f"# Benchmark results\n")
    lines.append(
        f"platform: {platform.platform()} {platform.machine()} | "
        f"ocannl commit: {commit} | parity tol: {PARITY_TOL:g} (max rel diff over "
        f"first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: "
        + ", ".join(f"{p} {t:g}" for p, t in sorted(PARITY_TOL_PRECISION.items()))
        + f"; the approximate regime {PARITY_TOL_APPROX:g})\n"
    )
    lines.append(
        "measurement boxes declared by `fixtures/DIGESTS.txt`: "
        + ", ".join(measurement_boxes)
        + "\n"
    )
    for workload in sorted({r["workload"] for r in results}):
        lines.append(f"\n## {workload}\n")
        rows = [r for r in results if r["workload"] == workload]
        # Which bytes these numbers are on (gh-ocannl-645), and whose (gh-ocannl-759). A report is
        # compared across sessions and machines; without this the comparison rests on the
        # assumption that everyone regenerated the same fixture, and today they demonstrably have
        # not. More than one line here means the section mixes fixtures.
        measured_on = sorted(
            {
                (
                    r.get("fixture"),
                    r.get("fixture_sha256"),
                    r.get("fixture_origin"),
                    tuple(r.get("fixture_missing_origins", ())),
                )
                for r in rows
                if r.get("fixture_sha256")
            }
        )
        for fixture, sha, origin, missing in measured_on:
            # Named, not implied by the report's own platform line: the recorded origin is the box
            # whose GENERATOR drew these bytes, which is not necessarily the box now measuring them
            # (fixtures get copied between boxes, and that is fine as long as the report says so).
            whose = f", {origin}'s bytes" if origin else ", bytes no origin records"
            lines.append(f"measured on `{fixture}`, sha256 `{sha}`{whose}\n")
            if missing:
                lines.append(
                    "**MISSING DIGEST RECORD:** declared measurement box(es) with no entry for "
                    f"`{fixture}`: {', '.join(missing)}\n"
                )
        # Precision-major, p50-ascending within a precision: scheduling variants are ranked
        # against the others computing in the same format, and a reduced-precision block reads as
        # its own group rather than being interleaved by a speed it owes to its storage format.
        # Within a precision the exact rows come first: an approximate row's number is read
        # against the exact one it relaxes.
        rows.sort(
            key=lambda r: (
                precision_rank(r.get("precision", "f32")),
                REGIMES.index(regime_of(r)) if regime_of(r) in REGIMES else len(REGIMES),
                r["step_ms"]["p50"] if finite(r["step_ms"]["p50"]) else math.inf,
            )
        )
        precisions = {r.get("precision", "f32") for r in rows}
        if len(precisions) > 1:
            lines.append(
                "Rows are grouped by precision (f32 first), p50-ascending within each group.\n"
            )
        # gh-ocannl-719: the column appears once a row ran in a non-exact regime -- or claims one
        # its runner contradicts -- and then every row of the section names its regime, the exact
        # ones included.
        with_regime = any(regime_of(r) != "exact" or r.get("regime_mismatch") for r in rows)
        if with_regime:
            lines.append(
                "`regime` is the numerics regime the row was measured and gated in (gh-ocannl-719). "
                "`exact` rows are gated at the exact envelope. `approximate` rows ran OCANNL under "
                "`--ocannl_profile=approximate` (tf32 matmuls, fp16 arithmetic, the C compiler's "
                "fast-math and contraction licence, the inlining refinement, and the "
                "algebraic-rewrite gates as they land) or torch under its own defaults (`high` "
                "matmul precision, scaled_dot_product_attention, cudnn.benchmark); they are gated "
                f"at the approximate envelope {PARITY_TOL_APPROX:g}, and their `parity` says "
                "whether they ALSO passed the exact envelope -- a rewrite that changes nothing "
                "measurable is a finding, not a pass. tinygrad has no exact pin, so its row stands "
                "in the approximate regime whenever the sweep has one. Exact rows come first "
                "within each precision. **`REGIME MISMATCH`** is a row whose runner reports "
                "having run under flags its regime does not own (an ambient OCANNL_PROFILE or "
                "OCANNL_TF32_MATMULS reaching an exact cell, say -- the regime is the resolution "
                "of the profile's keys, not the profile's name, and the row names the key and its "
                "source): its number belongs to neither column and is not comparable with "
                "anything -- the sweep fails on it, and the row is kept so the failure is "
                "visible where the numbers are read.\n"
            )
        with_tokens = any(r.get("tokens_per_step") for r in rows)
        # gh-ocannl-644: which process produced the step times. Only a cell that searches or
        # compiles has the distinction, so the column appears only where one was measured.
        with_provenance = any(r.get("provenance") for r in rows)
        if with_provenance:
            lines.append(
                "`pass` says which process produced a searching cell's step times. For the OCANNL "
                "tuned cell, whose protocol splits them: `replay` is the fresh pass-2 process "
                "replaying the cached winner, **`SEARCH PASS`** is the searching process itself — "
                "whose accumulated modules and buffers inflate every launch, so those numbers are "
                "not comparable with the others, and `no search` is a tuned cell that searched "
                "nothing and replayed nothing (autotune_search=false), so it shipped the untuned "
                "default. For a tinygrad `beam` or a `torch.compile` cell, "
                "which search in the timing process by protocol: `same-process` searched here, "
                "`cached` replayed its framework's own cache (so its `compile s` is a replay cost). "
                "A tuned cell's `compile s` carries the same statement about the search pass it "
                "came from: `(cached)` for one that replayed the schedule cache, `(no search)` "
                "for one that searched nothing at all.\n"
            )
        # gh-ocannl-626: only cells that tuned something have an emission census to report.
        with_tensorization = any(r.get("tensorization") for r in rows)
        if with_tensorization:
            lines.append(
                "`mma` says what the timed artifact's kernels actually emitted, which is not what "
                "its schedule asked for: `tensorized` is at least one tensor-core / SIMD-tile "
                "emission, **`SCALAR FALLBACK`** is a schedule carrying a `Tensorize` whose every "
                "`Tile_mma` declined at codegen to the lane-0 scalar loop, **`NO MMA EMITTED`** is "
                "one that carries a `Tensorize` and emitted no `Tile_mma` at all, and `—` is an "
                "artifact that never asked for tensor cores. The two shouted verdicts mean the row "
                "is a scalar timing: quoting it as a tensor-core number is the error this column "
                "exists to stop.\n"
            )
        header = "| framework | backend | variant | precision |"
        rule = "|---|---|---|---|"
        if with_regime:
            header += " regime |"
            rule += "---|"
        header += " step p50 ms | p10 | p90 | queued ms | compile s |"
        rule += "---|---|---|---|---|"
        if with_provenance:
            header += " pass |"
            rule += "---|"
        if with_tensorization:
            header += " mma |"
            rule += "---|"
        header += " parity |"
        rule += "---|"
        if with_tokens:
            header += " tok/s |"
            rule += "---|"
        lines.append(header)
        lines.append(rule)
        for r in rows:
            s = r["step_ms"]
            parity = r["parity"]
            notes = []
            if parity not in ("REF", "NO-REF") and finite(r.get("parity_max_rel")):
                notes.append(f"{r['parity_max_rel']:.1e}")
            if regime_of(r) != "exact" and r.get("parity_exact_envelope") is not None:
                notes.append(
                    "within exact envelope"
                    if r["parity_exact_envelope"]
                    else "beyond exact envelope"
                )
            if notes:
                parity += f" ({', '.join(notes)})"
            if r.get("diverged_at") is not None:
                parity += f" (loss non-finite from step {r['diverged_at']})"
            elif not r["parity_loss_moved"]:
                parity += " (loss stationary)"
            tokens = ""
            if with_tokens:
                tps = r.get("tokens_per_step")
                tokens = (
                    f" {tps * 1000 / s['p50']:,.0f} |" if tps and finite(s["p50"]) else " |"
                )
            compile_s = num(r["compile_s"], ".2f")
            compile_s += COMPILE_S_NOTE.get(r.get("search_pass"), "")
            provenance = ""
            if with_provenance:
                provenance = " %s |" % PROVENANCE_MARK.get(r.get("provenance"), "—")
            if with_tensorization:
                provenance += " %s |" % TENSORIZATION_MARK.get(r.get("tensorization"), "—")
            regime = ""
            if with_regime:
                mismatch = r.get("regime_mismatch")
                regime = (
                    f"| **REGIME MISMATCH** (dispatched {regime_of(r)}; ran {mismatch}) "
                    if mismatch
                    else f"| {regime_of(r)} "
                )
            lines.append(
                f"| {r['framework']} | {r['backend']} | {rendered_variant(r)} "
                f"| {r.get('precision', 'f32')} {regime}"
                f"| {num(s['p50'], '.3f')} | {num(s['p10'], '.3f')} | {num(s['p90'], '.3f')} "
                f"| {num(r['queued_step_ms'], '.3f')} | {compile_s} |{provenance} {parity} |{tokens}"
            )
    if unavailable:
        # Stated where the matrix is read: a cell missing because the workload cannot express it
        # is not a cell someone forgot to run (gh-ocannl-551).
        lines.append("\n## Cells not applicable\n")
        lines.append(
            "Requested cells this matrix cannot contain — the workload cannot express them, so "
            "their absence above is structural, not an unrun measurement.\n"
        )
        lines.append("| workload | precision | why |")
        lines.append("|---|---|---|")
        for workload, precision, reason in unavailable:
            lines.append(f"| {workload} | {precision} | {reason} |")
    if failures:
        # A cell that produced no result is absent from every table above, and an absent row and a
        # failed one read identically once the report outlives the run log — which is the whole
        # cost of an unattended sweep meeting a wedge (gh-ocannl-760). Named here, with what the
        # sweep did about it, including any cache it quarantined on the way out.
        lines.append("\n## Runner failures\n")
        lines.append(
            "Cells that produced no result line. Their absence from the tables above is a "
            "failure, not a measurement — nothing here is comparable with anything.\n"
        )
        lines.append("| cell | why |")
        lines.append("|---|---|")
        for label, note in failures:
            lines.append(f"| {label} | {note or 'no result line'} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text)
    print("\n" + text)


def build_arg_parser():
    """The sweep's command line, built without touching anything.

    Separated from `main` so a test can read what the flags actually parse to.  A default
    that lives only in the parser is a claim no other test can reach: `DEFAULT_BEAM_PARALLEL`
    being zero says nothing about the sweep unless `--beam-parallel` is wired to it, and the
    wiring is what an edit here breaks (gh-ocannl-843).  Everything `main` does that touches
    the machine -- the signal handler, the torch probe, the sweep itself -- stays in `main`,
    so calling this is free.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="*", help="workload names (default: all fixtures)")
    ap.add_argument("--tuned", action="store_true", help="add the OCANNL autotuned variant")
    ap.add_argument(
        "--precision",
        nargs="*",
        default=[],
        type=precision_spec,
        metavar="bf16|f16|f16-static|f16-gatedN",
        help="add OCANNL reduced storage precisions (training workloads: master weights + "
        "storage policy, f16 with dynamic loss scaling; the forward-only gpt workload: "
        "load-time weight conversion; gh-ocannl-492). Precision is an axis independent of "
        "the scheduling variant, so each one requested here is run against every selected "
        "variant — `--tuned --precision bf16` measures tuned bf16, which is the only "
        "tensor-core route on RDNA3/3.5 (gh-ocannl-539). f16-static (fixed scale, no gate) "
        "and f16-gatedN (fused on-device gate sampled every N steps) are the task-5 "
        "gate-cost legs; they need a training workload and are reported as not applicable "
        "on a forward-only one (gh-ocannl-551)",
    )
    ap.add_argument(
        "--materialized",
        action="store_true",
        help="add the OCANNL materialized-activations variant",
    )
    ap.add_argument(
        "--profile",
        nargs="+",
        default=["exact"],
        choices=list(REGIMES),
        metavar="exact|approximate",
        help="numerics regimes to run the matrix in (gh-ocannl-719; default: exact only). "
        "`exact` is the everyday matrix: OCANNL under no profile, torch pinned to `highest` "
        "matmul precision with hand-composed attention, gated at PARITY_TOL. `approximate` runs "
        "every OCANNL cell under --ocannl_profile=approximate (tf32 matmuls, fp16 arithmetic, "
        "fast-math and contraction, the inlining refinement, the rewrite gates as they land) and "
        "every torch cell under torch's own defaults (`high` matmul precision, "
        "scaled_dot_product_attention, cudnn.benchmark), gated at the looser PARITY_TOL_APPROX; "
        "each row and report section names its regime, and an approximate row also reports "
        "whether it passed the exact envelope. The parity reference (pytorch/cpu/eager, exact) "
        "runs whichever regimes are selected; tinygrad, which has no exact pin, runs once and "
        "stands in the approximate regime when one is selected. Both regimes in one sweep put "
        "the before/after in one report",
    )
    ap.add_argument("--nojit", action="store_true", help="add the tinygrad nojit variant")
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="add the pytorch torch.compile variant",
    )
    ap.add_argument(
        "--beam",
        type=int,
        default=0,
        metavar="N",
        help="add the tinygrad BEAM=N search variant (0 = off)",
    )
    ap.add_argument(
        "--beam-parallel",
        type=beam_parallel_arg,
        default=DEFAULT_BEAM_PARALLEL,
        metavar="N",
        help="run the tinygrad beam cells with PARALLEL=N — tinygrad's own knob for the "
        f"candidate-compile pool (default {DEFAULT_BEAM_PARALLEL} on the benchmark fleet). N=0 "
        "disables the pool outright and compiles candidates in-process, which is the only "
        "configuration the spawn-pool deadlock cannot occur in; pass N>0 to opt into a parallel "
        "search, with the per-cell cap remaining as its backstop",
    )
    ap.add_argument(
        "--cell-timeout",
        type=cell_timeout_arg,
        default=DEFAULT_CELL_TIMEOUT_S,
        metavar="SECONDS",
        help=f"per-cell wall-clock cap (default {DEFAULT_CELL_TIMEOUT_S:g} s; 0 disables). A "
        "cell over the cap has its whole process group killed and is recorded as a runner "
        "failure, so an unattended sweep meeting a wedged cell loses that cell rather than the "
        "sweep (gh-ocannl-760)",
    )
    ap.add_argument(
        "--no-cache-quarantine",
        action="store_true",
        help="on a killed tinygrad beam cell, leave its kernel cache in place instead of "
        "renaming it aside. The failure still names the risk: a search killed midway leaves a "
        "partial cache.db, and the next run over it reports a `searched` verdict nobody wrote",
    )
    ap.add_argument(
        "--gpu",
        choices=sorted(GPU_DEVICES),
        default="metal" if platform.system() == "Darwin" else "cuda",
        help="GPU backend for the non-CPU column of the matrix (default: metal on macOS, "
        "cuda elsewhere; none = CPU-only matrix)",
    )
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument(
        "--no-fixture-digest-check",
        action="store_true",
        help="measure fixtures whose bytes do not match fixtures/DIGESTS.txt. The digest is "
        "what says which workload a published number is on (gh-ocannl-645); use this only "
        "for a deliberately regenerated fixture you are about to re-record",
    )
    ap.add_argument(
        "--no-skip-cells",
        action="store_true",
        help="run the SKIP_CELLS entries too. Each was observed pathological on one "
        "machine/backend/OS; use this to retest whether the entry still applies here",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=["ocannl", "pytorch", "tinygrad"],
        help="frameworks to run",
    )
    return ap


def main():
    args = build_arg_parser().parse_args()
    install_termination_handler()
    gpu_ocannl, gpu_torch, gpu_tiny = GPU_DEVICES[args.gpu]
    if args.gpu == "hip" and gpu_torch and "pytorch" in args.only:
        # "cuda" only reaches the AMD GPU when torch is a ROCm/HIP build (a stock CPU or
        # CUDA wheel isn't); probe the bench venv and fall back to the CPU-only column.
        probe = run_supporting(
            [str(VENV_PY), "-c", "import sys, torch; sys.exit(0 if torch.version.hip else 1)"],
            capture_output=True,
        )
        if probe.returncode != 0:
            print("torch in the bench venv is not a ROCm/HIP build; "
                  "skipping the PyTorch GPU column", flush=True)
            gpu_torch = None
    if args.gpu == "hip" and gpu_tiny == "AMD" and not os.path.exists("/dev/kfd"):
        # tinygrad's AMD device drives the KFD driver directly; under WSL there is no
        # /dev/kfd (the GPU is reached through /dev/dxg and the WSL HSA runtime), but
        # tinygrad's HIP device goes through the HIP runtime and does reach the GPU.
        print("no /dev/kfd (WSL?); using tinygrad's HIP device for the GPU column", flush=True)
        gpu_tiny = "HIP"

    fixtures = sorted((HERE / "fixtures").glob("*.safetensors"))
    if args.workloads:
        fixtures = [f for f in fixtures if f.stem in args.workloads]
    if not fixtures:
        sys.exit("no fixtures found — run gen_fixtures.py first")

    digests_path = HERE / "fixtures" / fixture_digest.DIGEST_FILE
    digest_entries = fixture_digest.read_digests(digests_path)
    measurement_boxes = fixture_digest.measurement_boxes(digests_path)
    fixture_ids = check_fixture_digests(
        fixtures, digests_path=digests_path, allow_unpinned=args.no_fixture_digest_check
    )

    metas = {fx: read_st_metadata(fx) for fx in fixtures}
    models = {fx: metas[fx].get("model", "mlp") for fx in fixtures}
    if "ocannl" in args.only and not args.skip_build:
        targets = sorted(
            {f"benchmarks/runners/ocannl/bench_{m}.exe" for m in models.values()}
        )
        run_supporting(["dune", "build", "--root", ".", *targets], cwd=ROOT, check=True)

    results = []
    failures = []
    unavailable = []
    partial = HERE / "results" / "partial.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("")  # fresh run
    # Failures stream too, beside the results (gh-ocannl-760 review). A sweep that is interrupted,
    # terminated or crashed before `report()` otherwise leaves an artifact in which the cell that
    # WEDGED is indistinguishable from one that never ran — losing the cap, the survivor and the
    # quarantine record, which for an unattended run is the whole finding.
    partial_failures = HERE / "results" / "partial-failures.jsonl"
    partial_failures.write_text("")

    # The fixture the cells currently being dispatched are measuring — stamped onto every result
    # so a row, and the report built from it, states its own workload identity (gh-ocannl-645)
    # rather than leaving it to how the operator ran the sweep.
    stamp = {}

    def record_failure(label, note):
        """The one path a failed cell takes, wherever in the sweep it failed.

        Both the in-memory list the report is built from and the checkpoint an interrupted run
        leaves behind — appended here rather than at each call site, because the tuned search
        pass's own failure went straight to the list and so vanished from the artifact
        (gh-ocannl-760 review).
        """
        failures.append((label, note))
        with open(partial_failures, "a") as f:
            f.write(json.dumps({"cell": label, "why": note}) + "\n")

    def collect(label, cmd, override=None, **kwargs):
        t0 = time.monotonic()
        r, note = run_cell(label, cmd, timeout=args.cell_timeout, **kwargs)
        if r:
            r.update(stamp)
            if override:
                r.update(override)
            results.append(r)
            # Stream each cell as it lands so an interrupted run keeps its results. Through
            # json_safe like the final file: an interrupted run's rows are read by the same
            # readers, and a diverged cell is exactly the one an operator interrupts around.
            with open(partial, "a") as f:
                f.write(json.dumps(json_safe(r), allow_nan=False) + "\n")
        else:
            record_failure(label, note)
        print(f"    cell took {time.monotonic() - t0:.0f}s", flush=True)

    for fx in fixtures:
        name = fx.stem
        model = models[fx]
        stamp.clear()
        sha, origin = fixture_ids[fx]
        stamp.update(
            fixture_result_stamp(fx, sha, origin, digest_entries, measurement_boxes)
        )
        if "ocannl" in args.only:
            variants = ["default"]
            if args.materialized:
                variants.append("materialized")
            if args.tuned:
                variants.append("tuned")
            # Both training runners implement the mixed-precision recipe (master weights, storage
            # policy, f16 loss scaling and its gate-cost legs); the forward-only gpt fixture takes
            # BENCH_PRECISION as load-time weight conversion instead (gh-ocannl-492 task 4). Every
            # scheduling variant composes with every precision (gh-ocannl-529 lifted the
            # runner-side guard), so the OCANNL cells of a backend are the product of the two axes.
            mode = metas[fx].get("mode", "train")
            precisions = ["f32"]
            for precision in args.precision:
                reason = precision_unavailable(model, mode, precision)
                if reason:
                    unavailable.append((name, precision, reason))
                    print(f"--- {name} ocannl/*/{precision}: NOT APPLICABLE ({reason})")
                else:
                    precisions.append(precision)
            for backend in ["cc"] + ([gpu_ocannl] if gpu_ocannl else []):
                for precision in precisions:
                    for variant in variants:
                        cell = cell_name(variant, precision)
                        if cell_skipped(name, backend, variant, precision) and not args.no_skip_cells:
                            print(f"--- {name} ocannl/{backend}/{cell}: SKIPPED (SKIP_CELLS; "
                                  "--no-skip-cells to run it anyway)")
                            continue
                        env = cell_env(os.environ, fx, variant, precision)
                        for regime in args.profile:
                            # The regime is a third axis over the cells (gh-ocannl-719): the same
                            # variant and precision, dispatched under the profile that IS the
                            # regime, gated at that regime's envelope, labelled as such.
                            cmd = [
                                str(ocannl_exe(model)),
                                f"--ocannl_backend={backend}",
                                *ocannl_regime_args(regime),
                            ]
                            label = f"{name} ocannl/{backend}/{cell}{regime_label(regime)}"
                            # The cell's identity is what was dispatched, not what the runner chose
                            # to call itself: stamp every axis so a report row cannot silently
                            # collapse two cells (a runner predating gh-ocannl-539 reports a
                            # reduced-precision cell's variant as its precision; the regime is
                            # cross-checked against what the runner resolved by `regime_check`).
                            ident = {"variant": variant, "precision": precision, "regime": regime}
                            if variant == "tuned":
                                # Two-pass protocol: the search leaves the process slower (extra
                                # per-launch overhead from accumulated modules/buffers — measured at
                                # +10.3% on small CUDA kernels behind a 16 s search, and ~0 behind a
                                # 4 s one or on a workload whose steps are milliseconds;
                                # gh-ocannl-675), so pass 1 runs the search and
                                # populates autotune_cache (its compile_s is the search cost), and a
                                # fresh pass-2 process replays the cached winner for the step timings.
                                pass1, note = run_cell(
                                    f"{label} (search pass)",
                                    cmd,
                                    env=env,
                                    cwd=HERE,
                                    timeout=args.cell_timeout,
                                    on_incomplete=ocannl_cache_note,
                                )
                                if pass1 is None:
                                    record_failure(f"{label} (search pass)", note)
                                    continue
                                # What the search pass actually did, which is not derivable from the
                                # compile_s it hands over: a warm autotune_cache makes it a replay, and
                                # autotune_search=false makes it neither a search nor a replay. Both
                                # are legitimate, and both would otherwise be published as a search
                                # cost. Stamped so the report can say which.
                                pass1_verdict = search_provenance(pass1)
                                if pass1_verdict != "SEARCHED":
                                    print(f"    search pass verdict {pass1_verdict}: its compile_s is "
                                          "not a from-scratch search cost", flush=True)
                                collect(label, cmd, env=env, cwd=HERE,
                                        override={**ident, "compile_s": pass1["compile_s"],
                                                  "search_pass": pass1_verdict})
                            else:
                                collect(label, cmd, env=env, cwd=HERE, override=ident)
        if "pytorch" in args.only:
            for device in ["cpu"] + ([gpu_torch] if gpu_torch else []):
                for compiled in [False] + ([True] if args.torch_compile else []):
                    regimes = list(args.profile)
                    if device == "cpu" and not compiled and "exact" not in regimes:
                        # The parity reference is the exact CPU eager cell whatever regimes were
                        # selected: without it every approximate row would be NO-REF.
                        regimes.insert(0, "exact")
                    for regime in regimes:
                        collect(
                            f"{name} pytorch/{device}/{'compiled' if compiled else 'eager'}"
                            f"{regime_label(regime)}",
                            [str(VENV_PY), str(HERE / "runners/pytorch/run.py"), "--fixture", str(fx), "--device", device]
                            + (["--compile"] if compiled else [])
                            + torch_regime_args(regime),
                            override={"regime": regime},
                        )
        if "tinygrad" in args.only:
            # No exact pin exists for tinygrad, so its cells run once and stand in the approximate
            # regime whenever the sweep has one (gh-ocannl-719).
            tiny_regime = tinygrad_regime(args.profile)
            for device in ["CPU"] + ([gpu_tiny] if gpu_tiny else []):
                for jit in [1] + ([0] if args.nojit else []):
                    collect(
                        f"{name} tinygrad/{device}/{'jit' if jit else 'nojit'}"
                        f"{regime_label(tiny_regime)}",
                        [str(VENV_PY), str(HERE / "runners/tinygrad/run.py"), "--fixture", str(fx), "--device", device, "--jit", str(jit)],
                        override={"regime": tiny_regime},
                    )
                if args.beam:
                    # PARALLEL sizes tinygrad's candidate-compile pool (its own knob, read where
                    # the pool is created). The fleet default is 0 because that means no pool at
                    # all; a positive --beam-parallel explicitly opts back into the failure-prone
                    # spawn path, with the cell cap as its backstop (gh-ocannl-843).
                    beam_env = beam_cell_env(os.environ, args.beam_parallel)
                    collect(
                        # The pool is in the LABEL as well as in the row, because a failure has
                        # only the label: a wedged cell records no result, and which pool it
                        # wedged with is the first thing anyone asks (gh-ocannl-760 review).
                        f"{name} tinygrad/{device}/beam"
                        + ("" if args.beam_parallel is None else f" P={args.beam_parallel}")
                        + regime_label(tiny_regime),
                        [str(VENV_PY), str(HERE / "runners/tinygrad/run.py"), "--fixture", str(fx), "--device", device, "--beam", str(args.beam)],
                        # What pool the search ran with, recorded where the row is read. Without
                        # it the default, `0` and `--beam-parallel 2` all land as `beam` with no
                        # way to tell them apart, while their search costs differ by a factor of
                        # three or four -- so a compile_s from one configuration reads as the
                        # other's (gh-ocannl-760 review). `null` is tinygrad's own default, which
                        # is one worker per logical core on a GPU device and no pool on CPU.
                        override={
                            "beam": args.beam,
                            "beam_parallel": args.beam_parallel,
                            "regime": tiny_regime,
                        },
                        env=beam_env,
                        # The one cell whose cache a kill can tear: the beam search writes its
                        # winners into a single sqlite file as it goes (gh-ocannl-760).
                        on_incomplete=lambda killed: quarantine_tinygrad_cache(
                            beam_env, enabled=not args.no_cache_quarantine, killed=killed
                        ),
                    )

    parity_check(results)
    provenance_violations = provenance_check(results)
    tensorization_mismatches = tensorization_check(results)
    regime_mismatches = regime_check(results)
    report(results, HERE / "results", unavailable, failures)
    ok = True
    if unavailable:
        # Not a failure: these cells were requested but the workload cannot express them. Saying so
        # is the point — an unavailable cell and an unrun one otherwise look identical.
        print(
            f"NOT APPLICABLE: {len(unavailable)} requested cell(s) the workload cannot express: "
            + ", ".join(f"{w}/{p}" for w, p, _ in unavailable),
            flush=True,
        )
    if failures:
        ok = False
        # One line per cell: a timed-out cell's reason carries what was killed and what became of
        # the cache it was writing, which run together into one line is unreadable exactly where
        # an unattended run is read (gh-ocannl-760).
        print(
            f"RUNNER FAILURES: {len(failures)} cell(s) produced no result:\n"
            + "\n".join(f"  - {failure_line(f)}" for f in failures),
            flush=True,
        )
    no_ref = [r for r in results if r["parity"] == "NO-REF"]
    if no_ref and "pytorch" in args.only:
        # The reference cell was requested but is missing — the gate would be vacuous.
        ok = False
        print(f"PARITY GATE: {len(no_ref)} cell(s) have no reference to compare against", flush=True)
    if provenance_violations:
        # Not a parity problem: these cells computed the right thing. Their step times are the
        # search process's, which the two-pass protocol exists to keep out of a report — and which
        # nothing downstream can detect once the numbers are quoted (gh-ocannl-644).
        ok = False
        labels = ", ".join(
            f"{r['workload']} {r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}{regime_label(regime_of(r))}"
            for r in provenance_violations
        )
        print(
            f"PROVENANCE GATE: {len(provenance_violations)} tuned cell(s) reported step times "
            f"from a process that searched: {labels}",
            flush=True,
        )
    if tensorization_mismatches:
        # Not a gate: a declined `Tile_mma` still computes the right thing, and at some extents
        # declining is the correct decision. It is announced because the number is honest only
        # about a scalar kernel, and the row's variant name says otherwise (gh-ocannl-626) — the
        # failure mode is a reader quoting it as a tensor-core measurement.
        labels = ", ".join(
            f"{r['workload']} {r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}{regime_label(regime_of(r))}"
            f" ({r['tensorization']})"
            for r in tensorization_mismatches
        )
        print(
            f"TENSORIZATION NOTICE: {len(tensorization_mismatches)} tuned cell(s) shipped a "
            f"schedule asking for tensor cores whose kernels did not emit them: {labels}",
            flush=True,
        )
    if regime_mismatches:
        # A row labelled with a regime its process did not run in is worse than a missing row:
        # its number would be quoted as the other regime's (gh-ocannl-719).
        ok = False
        labels = ", ".join(
            f"{r['workload']} {r['framework']}/{r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}{regime_label(regime_of(r))}"
            f" (runner reports {r['regime_mismatch']})"
            for r in regime_mismatches
        )
        print(
            f"REGIME GATE: {len(regime_mismatches)} cell(s) ran under a regime other than the "
            f"one dispatched: {labels}",
            flush=True,
        )
    failed = [r for r in results if r["parity"] == "FAIL"]
    if failed:
        ok = False
        print(f"PARITY GATE: {len(failed)} cell(s) FAILED", flush=True)
    diverged = [r for r in results if r["parity"] == "DIVERGED"]
    if diverged:
        # A cell that ran and blew up, reported as such (gh-ocannl-676). It is a gate failure like
        # any other parity failure, named separately because the finding is different: nothing is
        # wrong with the runner, and the trajectory that shows it is in results.jsonl.
        ok = False
        labels = ", ".join(
            f"{r['workload']} {r['framework']}/{r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}{regime_label(regime_of(r))}"
            f" (from step {r['diverged_at']})"
            for r in diverged
        )
        print(
            f"PARITY GATE: {len(diverged)} cell(s) DIVERGED (non-finite loss): {labels}",
            flush=True,
        )
    stationary = [
        r for r in results if not r["parity_loss_moved"] and r["diverged_at"] is None
    ]
    if stationary:
        ok = False
        labels = ", ".join(
            f"{r['workload']} {r['framework']}/{r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}{regime_label(regime_of(r))}"
            for r in stationary
        )
        print(
            f"LOSS-MOVEMENT GATE: {len(stationary)} stationary cell(s): {labels}",
            flush=True,
        )
    if not ok:
        sys.exit(1)
    print("PARITY GATE: all cells passed", flush=True)


if __name__ == "__main__":
    main()
