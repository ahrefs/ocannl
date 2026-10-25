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
  that does not report (`device_props`, `compilation_speed`) is not something this driver measures. The report gives each arm's median/min/max per variant, and for each
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
import os
import statistics
import sys
from pathlib import Path

import bench_output
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


def render_report(manifest, records, problems, title="bin/ bench cells"):
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    parser = build_parser()
    args = parser.parse_args(argv[:split])
    common = argv[split + 1:]
    arms = resolve_arms(args, parser)
    common_env = {k: v for e in args.env for k, v in e.items()}
    try:
        orders = design_orders(len(arms), args.rounds)
    except ValueError as e:
        sys.exit(f"{DRIVER}: {e}")
    if not args.cell_timeout > 0:
        sys.exit(f"{DRIVER}: --cell-timeout must be positive")
    out = cells.fresh_out_dir(args.out, DRIVER)
    cells.install_cancellation(DRIVER)

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
                     if a["exe"] != exe else {})} for a in arms],
        "pins": cells.pins(args.backend), "cleared": cleared,
        "runtime_controls": cells.runtime_controls(env),
        "environment_values": cells.recorded_values(env),
        "environment_names": sorted(env),
        "cell_timeout_s": args.cell_timeout, "orders": orders,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    records = cells.run_cells(arms=arms, orders=orders, out=out, env=env, backend=args.backend,
                              common=common, common_env=common_env, timeout=args.cell_timeout,
                              reader=bench_result_reader, driver=DRIVER)
    if args.backend in ("cc", "multidev_cc"):
        manifest["kernel_compilers"] = cells.kernel_compilers(arms, records, env, out)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    problems = cells.run_problems(records, [
        f"arm {arm}: the kernel compiler could not be identified: {c['error']}"
        for arm, c in manifest.get("kernel_compilers", {}).items() if "error" in c])
    report = render_report(manifest, records, problems)
    (out / "report.md").write_text(report)
    sys.stdout.write(report)
    return cells.exit_status(records, problems)


if __name__ == "__main__":
    sys.exit(main())
