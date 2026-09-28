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

Resumable: a cell whose output already holds a result line is not run again.
"""

import argparse
import datetime
import json
import os
import re
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
    try:
        for line in open(str(path) + ".out"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except (OSError, ValueError):
        pass
    return None


def run_cell(out, backend, fixture, treatment, r, artifacts=False):
    base = cell_path(out, backend, fixture, treatment, r)
    if result_line(base) is not None:
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
    if artifacts:
        adir = out / "artifacts" / f"{backend}__{fixture}__{treatment}"
        adir.mkdir(parents=True, exist_ok=True)
        argv += ["--ocannl_output_debug_files_in_build_directory=true",
                 f"--ocannl_build_files_prefix={adir}"]
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
        try:
            status = subprocess.run(argv, cwd=HERE, env=clean_env(extra), stdout=o, stderr=e,
                                    timeout=CELL_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    with open(str(base) + ".status", "w") as f:
        f.write(f"{status}\n")
    res = result_line(base)
    if res is None:
        log(f"FAILED {base.name}: status {status}, no result line")
    else:
        log(f"done {base.name}: p50 {res['step_ms']['p50']:.1f} ms, "
            f"peak {res['peak_memory_bytes'] / 2**20:.1f} MiB")


def preflight(out):
    sys.path.insert(0, str(HERE))
    import fixture_digest  # stdlib-only

    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
    ids = {}
    for fx in MAIN + SWEEP:
        verdict, sha, _, origins = fixture_digest.status(HERE / "fixtures" / f"{fx}.safetensors",
                                                         entries)
        if verdict != "MATCH" or ORIGIN not in origins.split(","):
            sys.exit(f"fixture {fx}: {verdict} against DIGESTS.txt (need a {ORIGIN} match); "
                     "regenerate with gen_fixtures.py --origin m4-max")
        ids[fx] = sha
    meta = {"revision": rev, "dirty_tracked_files": dirty, "host": os.uname().nodename,
            "fixtures": ids, "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out / "preflight.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"revision {rev}{' (DIRTY)' if dirty else ''}; fixtures match {ORIGIN}")
    if not BENCH_GPT.exists():
        sys.exit(f"{BENCH_GPT} is not built: dune build benchmarks/runners/ocannl/bench_gpt.exe")
    import bench_venv

    venv = bench_venv.venv_python(HERE)  # BENCH_VENV_PY is read here, before clean_env drops it
    for fx in MAIN + SWEEP:
        dst = out / "torch" / f"{fx}.out"
        if dst.exists() and dst.read_text().strip():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"torch cpu parity reference {fx}")
        with open(dst, "w") as o, open(out / "torch" / f"{fx}.err", "w") as e:
            subprocess.run([str(venv), str(HERE / "runners" / "pytorch" / "run.py"), "--fixture",
                            str(HERE / "fixtures" / f"{fx}.safetensors"), "--device", "cpu"],
                           cwd=HERE, env=clean_env({}), stdout=o, stderr=e, timeout=CELL_TIMEOUT_S)


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    backends = args.backends.split(",")
    phases = args.phases.split(",") if args.phases else PHASES
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
    ks = []
    try:
        for line in open(str(base) + ".err"):
            m = KERNEL.match(line.rstrip("\n"))
            if m:
                ks.append({"i": int(m.group(1)), "ms": float(m.group(3)), "grid": m.group(4),
                           "block": m.group(5), "w": m.group(7).split()})
    except OSError:
        pass
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
    problems = []
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
                torch_losses = None
                tf = out / "torch" / f"{fx}.out"
                if tf.exists() and tf.read_text().strip():
                    torch_losses = json.loads(tf.read_text().strip().splitlines()[-1])["losses"]
                a_reps = timed(backend, fx, "A")
                for t in NAMES:
                    reps = timed(backend, fx, t)
                    diffs = [rel_losses(reps[r][0]["losses"], a_reps[r][0]["losses"])
                             for r in reps if r in a_reps]
                    vs_torch = ""
                    if t == "A" and torch_losses and a_reps:
                        vs_torch = f"{max(rel_losses(v[0]['losses'], torch_losses) for v in a_reps.values()):.2e}"
                    if diffs:
                        worst = max(diffs)
                        if worst > 1e-4:
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
                        if not ks:
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
                    _, bwd = attention_blocks(kernels(rep[1]))
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
    args = ap.parse_args(argv)
    if args.cmd == "run":
        LIMIT[0] = args.limit
        run(args)
        return 0
    return summarize(args)


if __name__ == "__main__":
    sys.exit(main())
