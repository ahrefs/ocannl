#!/usr/bin/env python3
"""gh-ocannl-1181: the sponsor-tagline matrix for `gpt2_mini` on one box.

One invocation measures one box's GPU column across three frameworks, with the August anchor
re-measured beside today's master in the same window:

- OCANNL tuned at the checkout this file lives in (`master`), under the exact regime and under
  `--ocannl_profile=approximate`, and OCANNL tuned exact at the August commit (`--aug`, a second
  checkout whose `bench_gpt.exe` is run unchanged: history is selected by executable path, never
  patched). Tuned cells keep orchestrate's two-pass protocol: a search process, then a fresh
  process that replays the cached winners and supplies the timings (gh-ocannl-644).
- PyTorch eager and `torch.compile`, each under the exact pin (`highest` matmul precision, cudnn
  tf32 off, composed attention) and under torch's own defaults (`--regime approximate`).
- tinygrad JIT, and BEAM=2 with the fleet's pinned `PARALLEL` (`DEFAULT_BEAM_PARALLEL`).
- The parity reference, `pytorch/cpu/eager` exact, once per repeat.

The process, parity, regime, provenance and tensorization gates are `orchestrate.py`'s own, imported
rather than restated. What this driver adds is the arm list, the order (rotated per repeat and
reversed on odd repeats, so no arm always runs right after the same neighbour), a fresh cache per
arm and repeat, a check that every row ran on the backend this box was asked for, and the
`quote_ok` verdict: a row is quotable only when it and its search pass passed parity, its
provenance is what its arm promises (searched for compiled/beam/search passes, replayed for a
tuned timing pass, nothing for eager/JIT), its runner reports the regime it was dispatched in, and
its p50 is finite. Repeat 0 is a discarded warm-up; `--summarize` medians the rest.

Outputs under `--out`: raw.jsonl (every row as the runner printed it, stamped), checked.jsonl
(after the gates), timings.jsonl (the timing rows), failures.jsonl, order.jsonl, env.json, the
per-cell logs under cells/, summary.json and summary.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Each arm: (framework, tree, regime, variant). The order is the base the rotation starts from;
# it interleaves frameworks so that even an unrotated repeat never runs one framework's arms back
# to back.
ARMS = {
    "ocannl-master-exact": ("ocannl", "master", "exact", "tuned"),
    "torch-exact-eager": ("pytorch", None, "exact", "eager"),
    "tinygrad-jit": ("tinygrad", None, "approximate", "jit"),
    "ocannl-master-approximate": ("ocannl", "master", "approximate", "tuned"),
    "torch-defaults-compiled": ("pytorch", None, "approximate", "compiled"),
    "ocannl-aug-exact": ("ocannl", "aug", "exact", "tuned"),
    "torch-exact-compiled": ("pytorch", None, "exact", "compiled"),
    "tinygrad-beam2": ("tinygrad", None, "approximate", "beam"),
    "torch-defaults-eager": ("pytorch", None, "approximate", "eager"),
}
BEAM = 2
COMPLETION_PASSES = 2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["cuda", "hip", "metal"])
    ap.add_argument("--box", help="box label stamped on every row, e.g. rog-nv-linux")
    ap.add_argument("--aug", type=Path, help="checkout of 7014dc44 with its bench_gpt.exe built")
    ap.add_argument("--no-anchor", metavar="REASON",
                    help="run without the August arm, recording why (it did not build, say)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workloads", nargs="+", default=["gpt2_mini"])
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--repeats", type=int, default=4, help="repeat 0 is a discarded warm-up")
    ap.add_argument("--cell-timeout", type=float, default=1800)
    ap.add_argument("--total-timeout", type=float, default=18000)
    ap.add_argument("--cpus", default="0-15" if platform.system() == "Linux" else "",
                    help="taskset CPU list for every cell (Linux); empty to not pin")
    ap.add_argument("--search-once", action="store_true",
                    help="tuned arms search at repeat 0 only; later repeats replay that cache in a "
                    "fresh process (for boxes where a search costs tens of minutes)")
    ap.add_argument("--summarize", action="store_true",
                    help="only (re)write summary.json/summary.md from an existing --out")
    a = ap.parse_args(argv)
    if a.repeats < 1:
        # One repeat is the smoke mode (every cell runs and is gated, no medians); none runs nothing.
        ap.error("--repeats must be at least 1")
    if not a.summarize:
        if not a.backend or not a.box:
            ap.error("--backend and --box are required to measure")
        if "ocannl-aug-exact" in a.arms and not (a.aug or a.no_anchor):
            ap.error("the August arm needs --aug, or --no-anchor with the reason")
    return a


def supporting_output(cmd):
    """stdout of a helper process, run in its own group like every child of a sweep."""
    import orchestrate as o

    return o.run_supporting(cmd, capture_output=True, check=True, timeout=120).stdout


def git_sha(tree):
    return supporting_output(["git", "-C", str(tree), "rev-parse", "HEAD"]).strip()


def raw_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append(out, name, obj):
    import orchestrate as o

    with (out / name).open("a") as f:
        f.write(json.dumps(o.json_safe(obj), allow_nan=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_backend(framework, backend, reference=False):
    """The `backend` a row must report to have run on this box's GPU (or the CPU reference).

    OCANNL reports the backend its context resolved (`Context.backend_name`), the torch runner
    labels a ROCm build's `cuda` device `cuda(hip)`, and tinygrad reports the device it opened.
    """
    import orchestrate as o

    if reference:
        return "cpu"
    ocannl, torch_dev, tiny_dev = o.GPU_DEVICES[backend]
    if framework == "ocannl":
        return ocannl
    if framework == "pytorch":
        return "cuda(hip)" if backend == "hip" else torch_dev
    return tiny_dev


def environment(a, fixtures):
    import orchestrate as o

    probe = (
        "import json,sys,importlib.metadata as m,torch,numpy\n"
        "d={'python':sys.version.split()[0],'torch':torch.__version__,'numpy':numpy.__version__,"
        "'tinygrad':m.version('tinygrad'),'safetensors':m.version('safetensors'),"
        "'torch_hip':torch.version.hip,'torch_cuda':torch.version.cuda}\n"
        "if torch.cuda.is_available(): d['torch_device']=torch.cuda.get_device_name(0)\n"
        "elif torch.backends.mps.is_available(): d['torch_device']='mps'\n"
        "print(json.dumps(d))\n"
    )
    try:
        versions = json.loads(supporting_output([str(o.VENV_PY), "-c", probe]))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        versions = {"probe_failed": str(exc)}
    driver = Path(__file__).resolve()
    return {
        "box": a.box,
        # The label is the caller's; the host is the machine's own name, kept beside it for audit.
        "hostname": platform.node(),
        "backend": a.backend,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "venv_python": str(o.VENV_PY),
        "versions": versions,
        "driver_sha256": raw_sha256(driver),
        "master_tree": str(HERE.parent),
        "aug_tree": str(a.aug) if a.aug else None,
        "no_anchor": a.no_anchor,
        "fixtures": fixtures,
        "arms": a.arms,
        "repeats": a.repeats,
        "cell_timeout_s": a.cell_timeout,
        "total_timeout_s": a.total_timeout,
        "cpus": a.cpus,
        # Reaches the torch cells unfiltered: where a box's interpreter lacks its dev headers,
        # inductor's triton build of its launcher finds them through this (gh-ocannl-1181, rog).
        "c_include_path": os.environ.get("C_INCLUDE_PATH"),
        "beam": BEAM,
        "beam_parallel": o.DEFAULT_BEAM_PARALLEL,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def measure(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Read by orchestrate at import: every cell's combined output lands in cells/ as it runs.
    os.environ["BENCH_CELL_LOG_DIR"] = str(out / "cells")
    import orchestrate as o
    import gh675_cells as g

    o.install_termination_handler()
    master = HERE.parent
    trees = {"master": master}
    shas = {"master": git_sha(master)}
    arms = list(a.arms)
    if "ocannl-aug-exact" in arms:
        if a.no_anchor:
            arms.remove("ocannl-aug-exact")
        else:
            trees["aug"] = a.aug.resolve()
            shas["aug"] = git_sha(trees["aug"])
            if not shas["aug"].startswith("7014dc44"):
                raise SystemExit(f"--aug is at {shas['aug']}, not the August commit 7014dc44")

    # A row is stamped with its tree's HEAD, so that HEAD must be what runs: a tree with tracked
    # modifications is refused, and each executable's digest is recorded, so a stale build is at
    # least visible against the HEAD it is stamped with.
    executables = {}
    for name, tree in trees.items():
        dirty = supporting_output(["git", "-C", str(tree), "status", "--porcelain",
                                   "--untracked-files=no"]).strip()
        if dirty:
            raise SystemExit(f"the {name} tree {tree} has tracked modifications:\n{dirty}")
        exe = tree / "_build/default/benchmarks/runners/ocannl/bench_gpt.exe"
        # Absent only when no arm of that tree runs; an OCANNL arm then fails at its cell.
        executables[name] = {"path": str(exe), "sha256": raw_sha256(exe) if exe.exists() else None}

    # The fixture is the box's own copy, gated by DIGESTS.txt as every sweep's is: some declared
    # box's bytes, its content digest and origin stamped on every row. The raw sha256 is recorded
    # beside it, since that is the digest the August report quoted.
    digests_path = HERE / "fixtures/DIGESTS.txt"
    entries = o.fixture_digest.read_digests(digests_path)
    boxes = o.fixture_digest.measurement_boxes(digests_path)
    fixtures, stamps = {}, {}
    for w in a.workloads:
        fx = HERE / "fixtures" / f"{w}.safetensors"
        sha, origin = o.check_fixture_digests([fx])[fx]
        fixtures[w] = {"path": str(fx), "content_sha256": sha, "origin": origin,
                       "raw_sha256": raw_sha256(fx), "bytes": fx.stat().st_size}
        stamps[w] = dict(o.fixture_result_stamp(fx, sha, origin, entries, boxes),
                         raw_fixture_sha256=fixtures[w]["raw_sha256"])

    env_record = environment(a, fixtures)
    env_record["shas"] = shas
    env_record["executables"] = executables
    env_record["arms_run"] = arms
    (out / "env.json").write_text(json.dumps(env_record, indent=2) + "\n")
    print(json.dumps(env_record), flush=True)

    base = g.base_env()
    for key in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
        base.pop(key, None)
    ocannl_backend, torch_dev, tiny_dev = o.GPU_DEVICES[a.backend]
    base.update(OCANNL_BACKEND=ocannl_backend, BENCH_DOMINANT_KERNEL="0")
    python = str(o.VENV_PY)
    pin = ["taskset", "-c", a.cpus] if a.cpus else []
    start = time.monotonic()
    failures = []
    searched_once = {}  # (workload, arm) -> the latest search pass, gated with its repeat
    search_cost = {}  # (workload, arm) -> compile_s of that search plus its completion passes

    def fail(workload, arm, repeat, stage, why):
        failures.append({"workload": workload, "arm": arm, "repeat": repeat, "stage": stage,
                         "why": why})
        append(out, "failures.jsonl", failures[-1])

    def run(workload, arm, repeat, stage, cmd, env, cwd, regime):
        remaining = a.total_timeout - (time.monotonic() - start)
        if remaining < 30:
            fail(workload, arm, repeat, stage, "total time cap reached")
            return None
        label = f"{workload}-r{repeat}-{arm}-{stage}"
        row, note = o.run_cell(label, [*pin, *cmd], env=env, cwd=cwd,
                               timeout=min(a.cell_timeout, remaining),
                               on_incomplete=(o.ocannl_cache_note if arm.startswith("ocannl") else None))
        if row is None:
            fail(workload, arm, repeat, stage, note)
            if "SURVIVED SIGKILL" in note or "LIVENESS IS UNKNOWN" in note:
                raise RuntimeError("cell group cleanup unproven: " + note)
            return None
        framework = row.get("framework")
        want = expected_backend(framework, a.backend, reference=(arm == "reference"))
        tree = ARMS.get(arm, (None, None))[1]
        row.update(stamps[workload], box=a.box, arm=arm, repeat=repeat, stage=stage, regime=regime,
                   revision=shas.get(tree) if framework == "ocannl" else None)
        if row.get("backend") != want:
            # Kept in raw.jsonl, marked, as the evidence of where the cell ran; never a timing.
            row["wrong_backend"] = want
            append(out, "raw.jsonl", row)
            fail(workload, arm, repeat, stage,
                 f"ran on backend {row.get('backend')!r}, expected {want!r}")
            return None
        append(out, "raw.jsonl", row)
        return row

    for workload in a.workloads:
        fx = HERE / "fixtures" / f"{workload}.safetensors"
        for repeat in range(a.repeats):
            rows, candidates = [], []
            ref = run(workload, "reference", repeat, "timed",
                      [python, str(HERE / "runners/pytorch/run.py"), "--fixture", str(fx),
                       "--device", "cpu"], base, HERE, "exact")
            if ref:
                rows.append(ref)
            k = repeat % len(arms)
            order = arms[k:] + arms[:k]
            if repeat % 2:
                order.reverse()
            append(out, "order.jsonl", {"workload": workload, "repeat": repeat, "arms": order})
            for arm in order:
                framework, tree, regime, variant = ARMS[arm]
                cache = out / "caches" / workload / f"r{repeat}" / arm
                cache.mkdir(parents=True, exist_ok=False)
                if framework == "ocannl":
                    root = trees[tree]
                    env = o.cell_env(base, fx, variant, "f32")
                    reuse = a.search_once and repeat > 0
                    tuned_cache = (out / "caches" / workload / "r0" / arm) if a.search_once else cache
                    env["OCANNL_AUTOTUNE_CACHE_DIR"] = str(tuned_cache / "autotune")
                    cmd = [str(root / "_build/default/benchmarks/runners/ocannl/bench_gpt.exe"),
                           f"--ocannl_backend={ocannl_backend}", *o.ocannl_regime_args(regime)]
                    if tree == "master":
                        # The August runner predates the progress flag (gh-ocannl-1061).
                        cmd += o.ocannl_variant_args(variant)
                    if reuse:
                        # The search this replay's cache came from, already gated in repeat 0.
                        search = searched_once.get((workload, arm))
                    else:
                        search = run(workload, arm, repeat, "search", cmd, env, root / "benchmarks",
                                     regime)
                        if search is not None:
                            rows.append(search)
                            searched_once[(workload, arm)] = search
                            search_cost[(workload, arm)] = search["compile_s"]
                    if search is None:
                        continue
                    # The tuner caches nothing from a search whose timings it found contended
                    # (`Autotune.search_measurements_cacheable`), so the next process searches that
                    # arm again. Such a pass is a cache completion, not a timing: it is recorded
                    # and followed by a fresh process, up to COMPLETION_PASSES times; only a
                    # process that replayed every arm supplies the timing.
                    # A completion's searches are part of what the replayed cache cost, so they
                    # accumulate into the arm's search cost. The last allowed pass, searched or not,
                    # is the arm's timing row (and fails the provenance gate if it searched).
                    for attempt in range(COMPLETION_PASSES + 1):
                        stage = "replay" if attempt == 0 else f"replay{attempt + 1}"
                        replay = run(workload, arm, repeat, stage, cmd, env, root / "benchmarks",
                                     regime)
                        if (replay is None or o.search_provenance(replay) != "SEARCHED"
                                or attempt == COMPLETION_PASSES):
                            break
                        replay["stage"] = f"completion{attempt + 1}"
                        rows.append(replay)
                        search_cost[(workload, arm)] += replay["compile_s"]
                    if replay is None:
                        continue
                    rows.append(replay)
                    replay["search_compile_s"] = search_cost[(workload, arm)]
                    replay["search_pass"] = o.search_provenance(search)
                    replay["provenance_ok"] = (replay["search_pass"] == "SEARCHED"
                                               and o.search_provenance(replay) == "REPLAY")
                    candidates.append(replay)
                elif framework == "pytorch":
                    env = dict(base, TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                               TORCHINDUCTOR_FX_GRAPH_CACHE="1", TRITON_CACHE_DIR=str(cache / "triton"))
                    cmd = [python, str(HERE / "runners/pytorch/run.py"), "--fixture", str(fx),
                           "--device", torch_dev, *o.torch_regime_args(regime)]
                    if variant == "compiled":
                        cmd.append("--compile")
                    row = run(workload, arm, repeat, "timed", cmd, env, HERE, regime)
                    if row:
                        rows.append(row)
                        row["provenance_ok"] = row.get("searched") is (variant == "compiled")
                        candidates.append(row)
                else:
                    beam = variant == "beam"
                    env = o.beam_cell_env(dict(base, CACHEDB=str(cache / "tinygrad.db")),
                                          o.DEFAULT_BEAM_PARALLEL)
                    env["BEAM"] = str(BEAM) if beam else "0"
                    cmd = [python, str(HERE / "runners/tinygrad/run.py"), "--fixture", str(fx),
                           "--device", tiny_dev, "--jit", "1"]
                    if beam:
                        cmd += ["--beam", str(BEAM)]
                    row = run(workload, arm, repeat, "timed", cmd, env, HERE, regime)
                    if row:
                        rows.append(row)
                        row["beam"] = BEAM if beam else 0
                        row["beam_parallel"] = o.DEFAULT_BEAM_PARALLEL if beam else None
                        row["provenance_ok"] = row.get("searched") is beam
                        candidates.append(row)
            o.parity_check(rows)
            o.regime_check(rows)
            o.provenance_check(candidates)
            o.tensorization_check(candidates)
            n_ref = len(ref["losses"]) if ref else None
            searches = {arm: r for (w, arm), r in searched_once.items() if w == workload}
            for row in candidates:
                search = searches.get(row["arm"])
                stages = [row] + ([search] if search else [])
                row["quote_ok"] = bool(
                    n_ref
                    and all(s.get("parity") == "PASS" and len(s["losses"]) == n_ref for s in stages)
                    and row["provenance_ok"]
                    and not row.get("regime_mismatch")
                    and o.finite(row["step_ms"]["p50"])
                )
            for row in rows:
                append(out, "checked.jsonl", row)
            for row in candidates:
                append(out, "timings.jsonl", row)
    summary = summarize(out)
    summary.update(failures=len(failures), wall_s=round(time.monotonic() - start, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    expected = len(arms) * a.repeats * len(a.workloads)
    complete = summary["timing_rows"] == expected and summary["quote_ok_rows"] == expected
    return 0 if complete and not failures else 1


def summarize(out):
    """Per workload and arm: the median over repeats >= 1 of each quotable row's p50, and the ratios
    the tagline quotes. Rerunnable over an existing --out (`--summarize`)."""
    env = json.loads((out / "env.json").read_text())
    rows = read_jsonl(out / "timings.jsonl")
    failures = read_jsonl(out / "failures.jsonl")
    cells = {}
    for r in rows:
        if r["repeat"] == 0:
            continue
        cell = cells.setdefault((r["workload"], r["arm"]), {"ok": [], "rejected": []})
        (cell["ok"] if r["quote_ok"] else cell["rejected"]).append(r)
    table = {}
    for (w, arm), cell in sorted(cells.items()):
        p50s = [r["step_ms"]["p50"] for r in cell["ok"]]
        entry = {
            "n": len(p50s),
            "rejected": [
                {"repeat": r["repeat"], "parity": r.get("parity"),
                 "provenance_ok": r.get("provenance_ok"), "regime_mismatch": r.get("regime_mismatch")}
                for r in cell["rejected"]
            ],
        }
        if p50s:
            entry.update(median_ms=statistics.median(p50s), min_ms=min(p50s), max_ms=max(p50s))
            compile_s = [r.get("search_compile_s", r.get("compile_s")) for r in cell["ok"]]
            entry["compile_s"] = [c for c in compile_s if c is not None]
            entry["parity_max_rel"] = max(r.get("parity_max_rel") or 0.0 for r in cell["ok"])
        table.setdefault(w, {})[arm] = entry
    ratios = {}
    pairings = {
        "exact": ("ocannl-master-exact", "torch-exact-eager", "torch-exact-compiled"),
        "approximate": ("ocannl-master-approximate", "torch-defaults-eager", "torch-defaults-compiled"),
    }
    for w, arms in table.items():
        def ms(arm):
            return arms.get(arm, {}).get("median_ms")

        r = {}
        for regime, (ocannl, eager, compiled) in pairings.items():
            for other in (eager, compiled, "tinygrad-jit", "tinygrad-beam2"):
                if ms(ocannl) and ms(other):
                    r[f"{ocannl} / {other}"] = ms(ocannl) / ms(other)
        if ms("ocannl-aug-exact") and ms("ocannl-master-exact"):
            r["ocannl-aug-exact / ocannl-master-exact"] = ms("ocannl-aug-exact") / ms("ocannl-master-exact")
        ratios[w] = r
    summary = {"box": env["box"], "backend": env["backend"], "shas": env.get("shas"),
               "timing_rows": len(rows),
               "quote_ok_rows": sum(1 for r in rows if r["quote_ok"]),
               "failures_recorded": failures, "cells": table, "ratios": ratios}
    lines = [f"# {env['box']} ({env['backend']})", ""]
    for w, arms in table.items():
        lines += [f"## {w}", "", "| arm | n | median p50 ms | min | max | compile s | parity max rel | rejected |",
                  "|---|---|---|---|---|---|---|---|"]
        for arm, e in arms.items():
            if "median_ms" in e:
                cs = e["compile_s"]
                comp = f"{min(cs):.1f}-{max(cs):.1f}" if cs else "-"
                lines.append(f"| {arm} | {e['n']} | {e['median_ms']:.3f} | {e['min_ms']:.3f} | "
                             f"{e['max_ms']:.3f} | {comp} | {e['parity_max_rel']:.1e} | {len(e['rejected'])} |")
            else:
                lines.append(f"| {arm} | 0 | - | - | - | - | - | {len(e['rejected'])} |")
        lines += ["", "| ratio | value |", "|---|---|"]
        lines += [f"| {k} | {v:.2f} |" for k, v in ratios[w].items()]
        lines.append("")
    if failures:
        lines += ["## failures", ""] + [f"- {json.dumps(f)}" for f in failures] + [""]
    (out / "summary.md").write_text("\n".join(lines))
    return summary


def main(argv=None):
    a = parse_args(argv)
    if a.summarize:
        out = a.out.resolve()
        summary = summarize(out)
        # The run's own facts are not derived from the rows; keep them from the measured summary.
        old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        summary.update({k: old[k] for k in ("failures", "wall_s") if k in old})
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print((out / "summary.md").read_text())
        return 0
    rc = measure(a)
    print((a.out.resolve() / "summary.md").read_text(), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
