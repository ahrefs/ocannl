#!/usr/bin/env python3
"""Hermetic, supervised, position-balanced cells for the bin/ benchmarks (gh-ocannl-743).

Every executable under bin/ reads its configuration from the ambient OCANNL_* environment and the
nearest `ocannl_config` up from its working directory, so two invocations that look identical on
the command line can measure different treatments, and the output cannot tell. This driver runs a
bench so that the command line IS the treatment, and records what the process actually resolved:

    python3 benchmarks/bin_cells.py --bench narrow_gebp_bench --backend cc --out DIR \\
        --rounds 12 --arm rn4=--rn=4 --arm rn6=--rn=6 -- f16 512 20

The machinery is the shared cell runner's (`cells.py`, gh-ocannl-1565); this file is its front end
for named arms over one checkout or prebuilt executables. `ab.py` is the front end for two
revisions.

- Hermetic. Every OCANNL_*, BENCH_* and OpenMP-runtime (OMP_*, GOMP_*, KMP_*) variable is removed
  from the cells' environment, and each cell runs with `--ocannl_no_config_file=true` from a fresh
  directory of its own under DIR, so neither an exported variable nor any `ocannl_config` on any
  path can reach it. What is left is the command line -- the driver's three pins first, then the
  bench's common arguments (after `--`), then the arm's -- over the defaults of the commit the
  bench was built from. A runner mode or an OpenMP control that IS the treatment is stated: with
  `--env NAME=VALUE` for every cell, or as the leading NAME=VALUE words of an arm's ARGS (as
  `env`(1) reads them) for that arm; a stated OCANNL_* variable is refused, since a library setting
  is stated on the command line, where the trace reads it back. Device and loader runtime controls
  (CUDA_*, HIP_*, HSA_*, MTL_*, DYLD_*, LD_*, ... -- `cells.RUNTIME_PREFIXES`) are the box's: some
  boxes cannot run without them, so they pass through, and each one present is recorded with its
  value beside the numbers (OCAMLRUNPARAM, the OCaml runtime's, among them), as are the toolchain
  variables (PATH, TMPDIR, SDKROOT, ...); every other variable is recorded by name only, since an
  inherited environment carries credentials no name pattern recognizes. On the C backends the
  compiler the kernels are compiled with -- the resolved `cc_backend_compiler_command`, else
  `ocamlc -config`'s `c_compiler` -- is resolved on the cells' PATH and fingerprinted per arm. The
  build, when the driver does it, runs with DUNE_* and OCAMLPARAM cleared and the root and build
  directory named, so the executable measured is the one built from this checkout.
- The resolved configuration is echoed, not assumed. Each cell runs under
  `--ocannl_log_config_sourcing=true`, and the driver reads that trace back: every setting the
  process read, with its value and where it came from. A cell fails when any setting came from the
  environment or a config file, when the backend it ran is not the one pinned, or when the trace is
  unreadable; the run fails when two cells of one arm resolved a setting differently. The report
  lists what the arms differ by as RESOLVED, and the whole resolved configuration beside the
  numbers, so a published number carries the settings that produced it -- the knobs a bench cannot
  pin itself (`autotune_bound_pruning`, the roofline constants, `gpu_graph_capture`, ...) included.
- Supervised. Each cell is a process group under `cell_group` with a cap (`--cell-timeout`), killed
  TERM-then-KILL as a group on expiry; a group not proven gone stops the run, since a survivor holds
  the device every later cell would be timed against. The build, when the driver does it, runs the
  same way.
- Position-balanced. Arms are whole bench processes, visited round by round in `Bench_design`'s
  spread mirrored rotation (bin/bench_design.ml; `cells.design_orders` is held equal to it by
  test/operations/bin_cells.py): every pair of arms exchanges positions within each pair of rounds,
  and every arm's position histogram is equal once the round count reaches the arm count (twice it
  for an odd one).
- Paired. A bench reports a timed variant through `Bench_out.result`'s `bench-result:` line on
  stdout (seconds per run); a cell reporting none, or one variant twice, fails -- an executable
  that does not report (`device_props`, `compilation_speed`) is not something this driver measures.
  With `--reader record` a cell is a runner instead (`bench_gpt`, `bench_conv`, ...), read through
  its result line (`bench_output.result_record`): each `--metric` (a dotted time field, default
  `step_ms.p50`) is a variant, unchecked -- a runner states no output check, so the report compares
  the arms' parity losses instead, informationally (a treatment may change numerics on purpose).
  A stated `BENCH_FIXTURE` is made absolute, content-stamped in the manifest, and puts the run
  behind the box gate (`--box BOX` or `--skip-box-check`, box_gate.py). For example, two placements
  of one `bench_gpt` build:
      python3 benchmarks/bin_cells.py --exe _build/default/benchmarks/runners/ocannl/bench_gpt.exe \
          --backend cuda --out DIR --box rog-nv --reader record --rounds 12 \
          --env BENCH_FIXTURE=benchmarks/fixtures/gpt2_mini.safetensors --env BENCH_TUNE=0 \
          --arm base= --arm treat=--ocannl_gpu_graph_capture=false
  The report gives each arm's median/min/max per variant, and for each
  non-reference arm the PAIRED ratio against the first arm, round by round -- the median ratio, its
  range, in how many rounds the arm was faster, and the exact two-sided sign-test p. Twelve paired
  rounds can resolve a 5% effect that overlapping intervals of single runs read as parity
  (gh-ocannl-755's twelve pairs).

DIR must not exist yet or be empty, and must lie outside the checkout. It receives `manifest.json`
(what ran: commit, host, bench, arms, design, the variables cleared), `cells.jsonl` (one record per
cell: argv, stated environment, exit, wall time, results, resolved configuration), `report.md` (also
printed), and a `cells/<NNN>-<arm>/` directory per cell holding its `stdout`/`stderr` and anything
it wrote. The exit status is 0 only when every cell exited 0 inside its cap with a clean treatment;
3 when the only faults are cells that exited 3 themselves (a bench refusing its measurement for host
contention: the box's verdict, not the treatment's); 1 otherwise.

Boundary: hermeticity here is about CONFIGURATION -- what the library reads. The machine (clock
behaviour, co-tenants, thermal state, device state) is the operator's to control, through an
exclusive measurement window; the driver records the host and the commit but does not query the
device. The resolved configuration is what the process read through the library's traced getters,
which is every library setting. `--exe` runs a prebuilt executable instead of building
`bin/<bench>.exe` (for tests, and for measuring a binary as it is), and `--arm-exe LABEL=PATH` runs
one arm's cells on an executable of its own (two builds of one bench, paired): such a binary's
sha256 is then its identity, and no commit is claimed for it.
"""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import bench_output
import box_gate
import cells
from cells import (TraceError, consistency_problems, design_orders, parse_config_trace,  # noqa: F401
                   sign_test_p, treatment_problems)
from display_safe import code_span, display_value, log_line, table_lines

ROOT = cells.ROOT
RESULT_PREFIX = bench_output.RESULT_PREFIX
LABEL_RE = cells.LABEL_RE
DRIVER = "bin_cells"


# A bench's `bench-result:` lines, read by the one reader of every bench output format.
parse_results = bench_output.bench_results


def bench_result_reader(stdout, _stderr):
    """The cell reader for `Bench_out.result` lines."""
    results = parse_results(stdout)
    return results, cells.timed_results_problems(results, "reported no bench-result line")


RECORD_NOTE = ("a runner's result line states no output check; the arms' parity losses are "
               "compared under Parity losses")


def metric_scale(path):
    """Seconds per unit of the result-line field at the dotted `path`: a `step_ms` percentile or a
    `*_ms` field is milliseconds, a `*_s` field seconds; ValueError for any other (a count, a
    flag), which is not a time this driver pairs."""
    last = path.rsplit(".", 1)[-1]
    if path.startswith("step_ms.") or last.endswith("_ms"):
        return 1e-3
    if last.endswith("_s"):
        return 1.0
    raise ValueError(f"--metric {path}: not a time field (step_ms.*, *_ms, *_s)")


def record_reader(metrics):
    """The cell reader for a runner's result line (`bench_gpt`, `bench_conv`, ...): each of
    `metrics` (dotted paths) becomes one timed variant, in seconds."""
    def read(stdout, _stderr):
        record = bench_output.result_record(stdout)
        if record is None:
            return [], ["reported no result line"]
        results, problems = [], []
        for path in metrics:
            try:
                value = bench_output.field(record, path)
            except KeyError as e:
                problems.append(e.args[0])
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                problems.append(f"{path} is {value!r}, not a positive time")
                continue
            results.append((path, value * metric_scale(path), "unchecked", RECORD_NOTE))
        return results, problems + cells.timed_results_problems(results, "reported no result line")
    return read


def cell_losses(out, rec):
    """The parity losses of a cell's result line, or None."""
    try:
        record = bench_output.result_record(
            (out / "cells" / f"{rec['cell']:03d}-{rec['arm']}" / "stdout").read_text(
                errors="replace"))
    except (OSError, ValueError):
        return None
    losses = record.get("losses") if isinstance(record, dict) else None
    return losses if isinstance(losses, list) else None


def parity_lines(out, manifest, records):
    """The Parity losses section of a record-reading run: per arm, how many distinct loss
    sequences its clean cells reported, and whether they are the reference arm's. Informational:
    a treatment may change numerics on purpose (a precision, a reassociation), so a difference is
    reported, never failed."""
    arms = [a["label"] for a in manifest["arms"]]
    seqs = {a: [] for a in arms}
    for r in records:
        if r["ok"]:
            losses = cell_losses(out, r)
            if losses is not None:
                seqs[r["arm"]].append(json.dumps(losses))
    ref = set(seqs[arms[0]])
    rows = [(display_value(a), str(len(seqs[a])), str(len(set(seqs[a]))),
             "(reference)" if a == arms[0] else "-" if not seqs[a] or not ref else
             "identical" if set(seqs[a]) == ref else "DIFFERENT")
            for a in arms]
    return ["", "## Parity losses", "",
            *table_lines(("arm", "cells", "distinct sequences", "against the reference"), rows,
                         "lrrl")]


# ---------------------------------------------------------------------------------------------
# The report.


def spans(args):
    """A command line's arguments as code spans, one each, or `(none)`."""
    return " ".join(code_span(a) for a in args) or "(none)"


def assignments(env):
    """A stated environment as code spans, or the empty string."""
    return " ".join(code_span(f"{k}={v}") for k, v in sorted(env.items()))


def ms(seconds):
    return f"{seconds * 1e3:.4g}"


def render_report(manifest, records, problems, title="bin/ bench cells", extra=()):
    arms = [a["label"] for a in manifest["arms"]]
    rounds = manifest["rounds"]
    lines = [f"# {title}: {display_value(manifest['bench'])} on {manifest['backend']}", ""]
    lines.append("## Identity")
    lines.append("")
    for key in ("commit", "executable_sha256", "host", "system", "cpu"):
        value = manifest.get(key)
        lines.append(log_line(f"- {key}: " + ("n/a (per arm, below)" if value is None
                                              else display_value(str(value)))))
    for a in manifest["arms"]:
        if "executable_sha256" in a:
            lines.append(log_line(f"- executable ({display_value(a['label'])}): "
                                  f"{display_value(a.get('commit', 'n/a'))}, sha256 "
                                  f"{a['executable_sha256']}"))
    lines.append(log_line("- runtime controls passed through (device, loader, OCaml runtime): " + (
        ", ".join(f"{k}={display_value(v)}" for k, v in manifest["runtime_controls"].items())
        or "none")))
    for arm, c in manifest.get("kernel_compilers", {}).items():
        lines.append(log_line(f"- kernel compiler ({display_value(arm)}): " + (
            f"{display_value(c['resolved'])} ({c['sha256'][:12]}), command "
            + code_span(c["command"]) if "error" not in c else "UNKNOWN: " + c["error"])))
    lines.append("- toolchain variables and every variable's name: manifest.json")
    lines.append(log_line("- cleared from the environment: " + (
        ", ".join(display_value(v) for v in manifest["cleared"]) or "nothing")))
    lines += ["", "## Arms", ""]
    common_env = manifest.get("common_env", {})
    lines.append(log_line("- common arguments: " + spans(manifest["common_args"])
                          + (f"; stated environment {assignments(common_env)}"
                             if common_env else "")))
    for a in manifest["arms"]:
        env = a.get("env", {})
        lines.append(log_line(f"- {display_value(a['label'])}: " + spans(a["args"])
                              + (f"; stated environment {assignments(env)}" if env else "")))
    for path, digest in manifest.get("stated_files", {}).items():
        lines.append(log_line(f"- stated file {display_value(path)}: sha256 {digest}"))
    lines.append(log_line(cells.design_line(len(arms), rounds)))

    ok = [r for r in records if r["ok"]]
    by, unchecked = {}, {}
    for r in ok:
        for variant, seconds, check, note in r["results"]:
            by.setdefault((variant, r["arm"]), {})[r["round"]] = seconds
            if check == "unchecked":
                unchecked.setdefault((variant, note), set()).add(r["arm"])
    variants = sorted({v for v, _ in by})
    lines += ["", "## Times per run (ms)", ""]
    if variants:
        rows = []
        for v in variants:
            for a in arms:
                xs = sorted(by.get((v, a), {}).values())
                rows.append((display_value(v), display_value(a), str(len(xs)),
                             ms(statistics.median(xs)) if xs else "-",
                             ms(xs[0]) if xs else "-", ms(xs[-1]) if xs else "-"))
        lines += table_lines(("variant", "arm", "n", "median", "min", "max"), rows, "llrrrr")
        if unchecked:
            lines += ["", "Variants whose output was NOT checked (their times are not evidence of a "
                      "correct kernel):", ""]
            lines += [log_line(f"- {display_value(v)} ({', '.join(display_value(a) for a in arms if a in on)})"
                               f": {note}")  # the bench's own words; log_line refuses a line break
                      for (v, note), on in sorted(unchecked.items())]
    else:
        lines.append("No cell reported a result.")
    if len(arms) > 1 and variants:
        ref = arms[0]
        lines += ["", f"## Paired against {display_value(ref)} (ratio = arm / reference, per round)",
                  ""]
        rows = [(display_value(v), display_value(a),
                 *cells.paired_cells(cells.paired(by.get((v, ref), {}), by.get((v, a), {}))))
                for v in variants for a in arms[1:]]
        lines += table_lines(("variant", "arm", *cells.PAIRED_HEADER), rows, "llrrrrr")

    lines += list(extra)
    lines += ["", "## Resolved configuration", ""]
    configs = {}
    for r in ok:
        if r["config"]:
            configs.setdefault(r["arm"], r["config"])
    keys = sorted({k for c in configs.values() for k in c})
    differing = [k for k in keys
                 if len({json.dumps(configs[a].get(k)) for a in arms if a in configs}) > 1]
    if differing:
        lines.append("What the arms differ by, as resolved:")
        lines.append("")
        rows = [(code_span("ocannl_" + k), *(
            display_value(configs[a][k]["value"]) + f" ({configs[a][k]['source']})"
            if a in configs and k in configs[a] else "(not read)" for a in arms)) for k in differing]
        lines += table_lines(("setting", *(display_value(a) for a in arms)), rows)
        lines.append("")
    elif len(configs) > 1:
        lines.append("No library setting resolved differently between the arms: they differ only by "
                     "their bench arguments above.")
        lines.append("")
    lines.append(f"Every setting the first successful cell of each arm resolved ({len(keys)} keys):")
    lines.append("")
    rows = []
    for k in keys:
        a = next(a for a in arms if a in configs and k in configs[a])
        e = configs[a][k]
        rows.append((code_span("ocannl_" + k), display_value(e["value"]),
                     e["source"] + (f" {display_value(e['detail'])}" if e["detail"] else ""),
                     "differs" if k in differing else "all arms"))
    if rows:
        lines += table_lines(("setting", "value", "source", "arms"), rows)
    lines += ["", "## Cells", ""]
    rows = [(str(r["cell"]), str(r["round"]), str(r["position"]), display_value(r["arm"]),
             str(r["status"]), str(len(r["results"])),
             "ok" if r["ok"] else display_value("; ".join(r["problems"]))) for r in records]
    lines += table_lines(("cell", "round", "position", "arm", "exit", "results", "verdict"), rows,
                         "rrrlrrl")
    lines += ["", "## Verdict", ""]
    # Driver-composed text over restricted labels and single lines of a cell's output: shown raw,
    # with the line builder's refusal as its guard (display_safe's rule for matched output).
    lines += [log_line(f"- {p}") for p in problems] or ["- every cell clean"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Running.


def parse_arm(text):
    try:
        return cells.parse_arm(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_arm_exe(text):
    label, sep, path = text.partition("=")
    if not sep or not LABEL_RE.fullmatch(label) or not path:
        raise argparse.ArgumentTypeError(f"--arm-exe {text!r}: expected LABEL=PATH")
    return label, Path(path)


def parse_assignment(text):
    try:
        env, rest = cells.split_assignments([text])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--env {text!r}: {e}")
    if rest:
        raise argparse.ArgumentTypeError(f"--env {text!r}: expected NAME=VALUE")
    return env


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bin_cells.py", description=__doc__.split("\n\n", 1)[0],
        usage="%(prog)s [--bench NAME | --exe PATH] --backend B --out DIR [--rounds R] "
              "[--arm LABEL=ARGS]... [--arm-exe LABEL=PATH]... [--env NAME=VALUE]... "
              "[--cell-timeout S] [-- BENCH-ARGS...]")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--bench", help="a bin/ executable's dune name, built as bin/<NAME>.exe")
    which.add_argument("--exe", type=Path, help="run this prebuilt executable instead")
    parser.add_argument("--backend", required=True, choices=cells.BACKENDS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=6,
                        help="even; each round runs every arm once (default 6)")
    parser.add_argument("--arm", action="append", type=parse_arm, default=[],
                        help="LABEL=ARGS, the arm's own arguments (shell-split; leading NAME=VALUE "
                             "words are its stated environment); the first arm is the reference "
                             "of the paired ratios. Default: one arm, 'base', with none")
    parser.add_argument("--arm-exe", action="append", type=parse_arm_exe, default=[],
                        help="LABEL=PATH: the arm LABEL runs this prebuilt executable (an arm "
                             "named only here takes no arguments of its own)")
    parser.add_argument("--env", action="append", type=parse_assignment, default=[],
                        help="NAME=VALUE stated in every cell's environment (BENCH_*, OMP_*, ...; "
                             "never OCANNL_*)")
    parser.add_argument("--cell-timeout", type=float, default=cells.CELL_TIMEOUT_S,
                        help=f"seconds per cell (default {cells.CELL_TIMEOUT_S})")
    parser.add_argument("--reader", choices=("bench-result", "record"), default="bench-result",
                        help="what a cell reports: `Bench_out.result` lines (the bin/ benches; the "
                             "default), or a runner's result line (bench_gpt, bench_conv, ...)")
    parser.add_argument("--metric", action="append", default=[],
                        help="with --reader record: a dotted time field of the result line to "
                             "pair (repeatable; default step_ms.p50)")
    box_gate.add_arguments(parser)  # checked only when a cell is handed a fixture
    return parser


def resolve_arms(args, parser):
    """The arms with their executables, or a usage exit."""
    arms = list(args.arm)
    labels = [a["label"] for a in arms]
    exes = dict(args.arm_exe)
    if len(exes) != len(args.arm_exe):
        parser.exit(2, f"{DRIVER}: one --arm-exe per arm\n")
    arms += [{"label": label, "args": [], "env": {}} for label in exes if label not in labels]
    arms = arms or [{"label": "base", "args": [], "env": {}}]
    labels = [a["label"] for a in arms]
    if len(set(labels)) != len(labels):
        parser.exit(2, f"{DRIVER}: arm labels must be distinct, got {labels}\n")
    if args.bench is None and args.exe is None and any(a["label"] not in exes for a in arms):
        parser.exit(2, f"{DRIVER}: --bench or --exe is required unless every arm has its "
                       "--arm-exe\n")
    for a in arms:
        if a["label"] in exes:
            exe = exes[a["label"]].resolve()
            if not exe.is_file():
                parser.exit(2, f"{DRIVER}: no executable at {exes[a['label']]}\n")
            a["exe"] = exe
    return arms


def stated_fixtures(common_env, arms):
    """Every stated BENCH_FIXTURE, made absolute (a cell runs in a directory of its own, where a
    relative path names nothing); exit when one names no file."""
    for env in [common_env, *(a["env"] for a in arms)]:
        if "BENCH_FIXTURE" in env:
            path = Path(env["BENCH_FIXTURE"]).expanduser().resolve()
            if not path.is_file():
                sys.exit(f"{DRIVER}: BENCH_FIXTURE={env['BENCH_FIXTURE']} names no file")
            env["BENCH_FIXTURE"] = str(path)
    return sorted({env["BENCH_FIXTURE"] for env in [common_env, *(a["env"] for a in arms)]
                   if "BENCH_FIXTURE" in env})


def gate_fixtures(args, fixtures):
    """The box gate over the stated fixtures, per fixture directory (`box_gate.require`)."""
    by_dir = {}
    for f in fixtures:
        path = Path(f)
        by_dir.setdefault(path.parent, []).append(path.name.removesuffix(".safetensors"))
    for fixture_dir, names in sorted(by_dir.items()):
        box_gate.require(DRIVER, args, fixture_dir, names,
                         run=lambda cmd: cells.run_in_group(cmd, ROOT, dict(os.environ), 900,
                                                            None, None))


def stated_files(common_env, arms):
    """{path: sha256} of every stated value naming an existing file by an absolute path: what a
    cell read besides its executable (a fixture), content-stamped beside the numbers."""
    files = {}
    for env in [common_env, *(a["env"] for a in arms)]:
        for value in env.values():
            if os.path.isabs(value) and os.path.isfile(value):
                files[value] = cells.sha256(value)
    return dict(sorted(files.items()))


def main(argv=None, arm_commits=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    parser = build_parser()
    args = parser.parse_args(argv[:split])
    common = argv[split + 1:]
    arms = resolve_arms(args, parser)
    common_env = {k: v for e in args.env for k, v in e.items()}
    if args.metric and args.reader != "record":
        parser.exit(2, f"{DRIVER}: --metric reads a runner's result line: pass --reader record\n")
    metrics = args.metric or ["step_ms.p50"]
    try:
        for m in metrics:
            metric_scale(m)
    except ValueError as e:
        parser.exit(2, f"{DRIVER}: {e}\n")
    reader = bench_result_reader if args.reader == "bench-result" else record_reader(metrics)
    fixtures = stated_fixtures(common_env, arms)
    try:
        orders = design_orders(len(arms), args.rounds)
    except ValueError as e:
        sys.exit(f"{DRIVER}: {e}")
    if not args.cell_timeout > 0:
        sys.exit(f"{DRIVER}: --cell-timeout must be positive")
    out = cells.fresh_out_dir(args.out, DRIVER)
    cells.install_cancellation(DRIVER)
    if fixtures:
        gate_fixtures(args, fixtures)

    shared = [a for a in arms if "exe" not in a]
    exe = None
    if not shared:
        bench, commit = "per-arm executables", "n/a (--arm-exe: prebuilt binaries, identified by sha256)"
    elif args.exe is not None:
        exe = args.exe.resolve()
        if not exe.is_file():
            sys.exit(f"{DRIVER}: no executable at {args.exe}")
        bench, commit = exe.name, "n/a (--exe: a prebuilt binary, identified by its sha256)"
    else:
        bench = args.bench
        if not LABEL_RE.fullmatch(bench):
            sys.exit(f"{DRIVER}: not a dune executable name: {bench!r}")
        commit = cells.tree_commit(ROOT, DRIVER)
        out.mkdir(parents=True, exist_ok=True)
        exe = cells.build_target(ROOT, f"bin/{bench}.exe", out / "build.log", DRIVER)
    for a in shared:
        a["exe"] = exe
    out.mkdir(parents=True, exist_ok=True)

    env, cleared = cells.hermetic_env(os.environ)
    manifest = {
        "bench": bench, "backend": args.backend, "rounds": args.rounds, "commit": commit,
        "executable": None if exe is None else str(exe),
        "executable_sha256": None if exe is None else cells.sha256(exe), **cells.host_identity(),
        "common_args": common, "common_env": common_env,
        "arms": [{"label": a["label"], "args": a["args"], "env": a["env"],
                  **({"executable": str(a["exe"]), "executable_sha256": cells.sha256(a["exe"])}
                     if a["exe"] != exe else {}),
                  **({"commit": arm_commits[a["label"]]}
                     if arm_commits and a["label"] in arm_commits else {})} for a in arms],
        "reader": args.reader, **({"metrics": metrics} if args.reader == "record" else {}),
        "stated_files": stated_files(common_env, arms),
        "pins": cells.pins(args.backend), "cleared": cleared,
        "runtime_controls": cells.runtime_controls(env),
        "environment_values": cells.recorded_values(env),
        "environment_names": sorted(env),
        "cell_timeout_s": args.cell_timeout, "orders": orders,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    records = cells.run_cells(arms=arms, orders=orders, out=out, env=env, backend=args.backend,
                              common=common, common_env=common_env, timeout=args.cell_timeout,
                              reader=reader, driver=DRIVER)
    if args.backend in ("cc", "multidev_cc"):
        manifest["kernel_compilers"] = cells.kernel_compilers(arms, records, env, out)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    problems = cells.run_problems(records, [
        f"arm {arm}: the kernel compiler could not be identified: {c['error']}"
        for arm, c in manifest.get("kernel_compilers", {}).items() if "error" in c])
    report = render_report(manifest, records, problems,
                           extra=parity_lines(out, manifest, records)
                           if args.reader == "record" else ())
    (out / "report.md").write_text(report)
    sys.stdout.write(report)
    return cells.exit_status(records, problems)


if __name__ == "__main__":
    sys.exit(main())
