#!/usr/bin/env python3
"""Hermetic, supervised, position-balanced cells for the bin/ benchmarks (gh-ocannl-743).

Every executable under bin/ reads its configuration from the ambient OCANNL_* environment and the
nearest `ocannl_config` up from its working directory, so two invocations that look identical on
the command line can measure different treatments, and the output cannot tell. This driver runs a
bench so that the command line IS the treatment, and records what the process actually resolved:

    python3 benchmarks/bin_cells.py --bench narrow_gebp_bench --backend cc --out DIR \\
        --rounds 12 --arm rn4=--rn=4 --arm rn6=--rn=6 -- f16 512 20

- Hermetic. Every OCANNL_*, BENCH_* and OpenMP-runtime (OMP_*, GOMP_*, KMP_*) variable is removed
  from the cells' environment, and each cell runs with `--ocannl_no_config_file=true` from a fresh
  directory of its own under DIR, so neither an exported variable nor any `ocannl_config` on any
  path can reach it. What is left is the command line -- the driver's three pins first, then the
  bench's common arguments (after `--`), then the arm's -- over the defaults of the commit the
  bench was built from. Device and loader runtime controls (CUDA_*, HIP_*, HSA_*, MTL_*, DYLD_*,
  LD_*, ... -- `RUNTIME_PREFIXES`) are the box's: some boxes cannot run without them, so they pass
  through, and each one present is recorded with its value beside the numbers (OCAMLRUNPARAM, the
  OCaml runtime's, among them); every other variable that passes is recorded by name. The build,
  when the driver does it, runs with DUNE_* and OCAMLPARAM cleared and the root and build directory
  named, so the executable measured is the one built from this checkout.
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
  spread mirrored rotation (bin/bench_design.ml; `design_orders` here is held equal to it by
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
cell: argv, exit, wall time, results, resolved configuration), `report.md` (also printed), and a
`cells/<NNN>-<arm>/` directory per cell holding its `stdout`/`stderr` and anything it wrote. The
exit status is 0 only when every cell exited 0 inside its cap with a clean treatment.

Boundary: hermeticity here is about CONFIGURATION -- what the library reads. The machine (clock
behaviour, co-tenants, thermal state, device state) is the operator's to control, through an
exclusive measurement window; the driver records the host and the commit but does not query the
device. The resolved configuration is what the process read through the library's traced getters,
which is every library setting. `--exe` runs a prebuilt executable instead of building
`bin/<bench>.exe` (for tests, and for measuring a binary as it is): the binary's sha256 is then its
identity, and no commit is claimed for it.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cell_group
import strict_json
from display_safe import code_span, display_value, log_line, table_lines

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKENDS = ("cc", "multidev_cc", "cuda", "hip", "metal")
# The environment families a cell must not inherit: the library's settings, the benchmark runners'
# modes, and the OpenMP runtime's controls (team size and placement shape the cc backend's Grid
# kernels).
CLEARED_PREFIXES = ("OCANNL_", "BENCH_", "OMP_", "GOMP_", "KMP_")
# The environment families that steer a device or loader runtime without being OCANNL configuration
# (`CUDA_LAUNCH_BLOCKING`, `HSA_OVERRIDE_GFX_VERSION`, `MTL_DEBUG_LAYER`, a preloaded library, ...).
# Clearing them could leave a box unable to run at all (some AMD boxes need their HSA override), so
# they pass through, and every one present is RECORDED with its value as part of the treatment.
# Every other variable that passes is recorded by name.
RUNTIME_PREFIXES = ("CUDA_", "NVIDIA_", "CUBLAS", "HIP_", "ROCR_", "HSA_", "AMD_", "GPU_", "MTL_",
                    "METAL_", "OBJC_", "DYLD_", "LD_", "MALLOC", "OCAMLRUNPARAM")
# What the build must not inherit: Dune's own environment and the compiler's flag override.
BUILD_CLEARED_PREFIXES = ("DUNE_", "OCAMLPARAM")
RESULT_PREFIX = "bench-result: "
CELL_TIMEOUT_S = 1800
BUILD_TIMEOUT_S = 3600
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_cancellation = cell_group.CancellationDeferral("bin_cells")


# ---------------------------------------------------------------------------------------------
# The visiting order: bin/bench_design.ml's spread mirrored rotation, line for line.


def design_offset(arms, pairs, p):
    """The rotation offset of mirrored pair `p` of `pairs` (`Bench_design.offset`)."""
    if arms % 2 == 0:
        return 2 * (p * (arms // 2) // pairs)
    return p * arms // pairs


def design_orders(arms, batches):
    """The arm indices each batch visits, in order (`Bench_design.orders`)."""
    if arms < 1:
        raise ValueError(f"{arms} arms; at least one is needed")
    if batches < 2 or batches % 2:
        raise ValueError(
            f"{batches} rounds; the count must be even and positive -- every round is mirrored "
            "by its partner")
    orders = []
    for b in range(batches):
        s = design_offset(arms, batches // 2, b // 2)
        idx = [(i + s) % arms for i in range(arms)]
        if b % 2:
            idx.reverse()
        orders.append(idx)
    return orders


def exact_batches(arms):
    """The least round count at which every arm's position histogram is equal."""
    return arms if arms % 2 == 0 else 2 * arms


def design_balance(arms, orders):
    """(pairs exchanged, histograms equal, (fewest, most) edge visits), as `Bench_design.balance`."""
    hist = [[0] * arms for _ in range(arms)]
    before = [[0] * arms for _ in range(arms)]
    for order in orders:
        for pos, a in enumerate(order):
            hist[a][pos] += 1
            for later in order[pos + 1:]:
                before[a][later] += 1
    exchanged = all(before[a][b] == before[b][a] for a in range(arms) for b in range(arms))
    equal = all(h == hist[0] for h in hist)
    edges = [h[0] if arms == 1 else h[0] + h[-1] for h in hist]
    return exchanged, equal, (min(edges), max(edges))


# ---------------------------------------------------------------------------------------------
# The environment and the configuration trace.


def hermetic_env(environ):
    """`environ` without the cleared families, and the sorted names it cleared."""
    cleared = sorted(k for k in environ if k.startswith(CLEARED_PREFIXES))
    return {k: v for k, v in environ.items() if k not in cleared}, cleared


class TraceError(ValueError):
    """The configuration trace in a cell's stderr cannot be read as the library writes it."""


_RETRIEVING = re.compile(r"Retrieving commandline, environment, or config file variable ocannl_(\S+)")
# `Utils.describe_config_source`'s five shapes; the value is greedy up to the LAST separator, since
# the source part (an argv element, a variable name) carries no such separator itself.
_SOURCES = [
    (re.compile(r"Found (.*), commandline (.*)"), "commandline"),
    (re.compile(r"Found (.*), environment (\S+)"), "environment"),
    (re.compile(r"Found (.*), in the config file"), "config file"),
    (re.compile(r'Found (.*), in the profile "(.*)" picked via (.*)'), "profile"),
    (re.compile(r"Not found, using default (.*)"), "default"),
]


def parse_config_trace(text):
    """{key: {"value", "source", "detail"}} from the `log_config_sourcing` trace in `text`.

    The trace prints a `Retrieving ... ocannl_<key>` line and then where the value came from; other
    stderr may sit between the two, but a second request before the first is answered, a request
    never answered, or one key resolved to two values is refused with `TraceError`."""
    config, pending = {}, None
    for line in text.splitlines():
        line = line.rstrip("\r")
        m = _RETRIEVING.fullmatch(line)
        if m:
            if pending is not None:
                raise TraceError(f"ocannl_{pending} was requested and never resolved")
            pending = m.group(1)
            continue
        if pending is None:
            continue
        for pattern, source in _SOURCES:
            m = pattern.fullmatch(line)
            if not m:
                continue
            if source == "default":
                entry = {"value": m.group(1), "source": "default", "detail": ""}
            elif source == "profile":
                entry = {"value": m.group(1), "source": "profile",
                         "detail": f"{m.group(2)} via {m.group(3)}"}
            elif source == "config file":
                entry = {"value": m.group(1), "source": source, "detail": ""}
            else:
                entry = {"value": m.group(1), "source": source, "detail": m.group(2)}
            known = config.get(pending)
            if known is not None and known["value"] != entry["value"]:
                raise TraceError(
                    f"ocannl_{pending} resolved to both {known['value']!r} and {entry['value']!r}")
            config.setdefault(pending, entry)
            pending = None
            break
    if pending is not None:
        raise TraceError(f"ocannl_{pending} was requested and never resolved")
    return config


def treatment_problems(config, backend):
    """Why a cell's resolved configuration is not the treatment its command line states."""
    if not config:
        return ["no configuration trace: the bench resolved no setting the driver could read"]
    problems = []
    b = config.get("backend")
    if b is None or b["value"] != backend or b["source"] != "commandline":
        problems.append(f"the backend resolved to {b and b['value']!r} from "
                        f"{b and b['source']}, not to the pinned {backend!r}")
    nc = config.get("no_config_file")
    if nc is None or nc["value"] != "true":
        problems.append("the config file was not disabled")
    for key, entry in sorted(config.items()):
        if entry["source"] in ("environment", "config file"):
            problems.append(f"ocannl_{key} came from the {entry['source']}")
        elif entry["source"] == "profile" and not entry["detail"].endswith("via the commandline"):
            problems.append(f"ocannl_{key} came from a profile picked outside the command line")
    return problems


def consistency_problems(records):
    """Settings two successful cells of one arm resolved differently."""
    problems = []
    seen = {}
    for rec in records:
        if rec["config"] is None:
            continue
        for key, entry in rec["config"].items():
            first = seen.setdefault((rec["arm"], key), (entry["value"], rec["cell"]))
            if first[0] != entry["value"]:
                problems.append(
                    f"arm {rec['arm']}: ocannl_{key} resolved to {first[0]!r} in cell {first[1]} "
                    f"and to {entry['value']!r} in cell {rec['cell']}")
    return problems


CHECKS = ("passed", "failed", "unchecked")


def parse_results(text):
    """The cell's `bench-result:` records: [(variant, seconds, check, note)], refusing with
    ValueError a malformed one, a non-finite or non-positive time, or one that does not state its
    output check (`Bench_out.check`)."""
    results = []
    for line in text.splitlines():
        if not line.startswith(RESULT_PREFIX):
            continue
        record = strict_json.loads(line[len(RESULT_PREFIX):])
        variant, seconds = record.get("variant"), record.get("seconds")
        check, note = record.get("check"), record.get("check_note")
        if not isinstance(variant, str) or isinstance(seconds, bool) or not isinstance(
                seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0 \
                or check not in CHECKS or not isinstance(note, str):
            raise ValueError(f"unusable result line: {line}")
        results.append((variant, float(seconds), check, note))
    return results


# ---------------------------------------------------------------------------------------------
# Statistics and the report.


def sign_test_p(faster, n):
    """The exact two-sided sign-test p of `faster` successes in `n` untied pairs."""
    if n == 0:
        return 1.0
    k = min(faster, n - faster)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def spans(args):
    """A command line's arguments as code spans, one each, or `(none)`."""
    return " ".join(code_span(a) for a in args) or "(none)"


def ms(seconds):
    return f"{seconds * 1e3:.4g}"


def render_report(manifest, records, problems):
    arms = [a["label"] for a in manifest["arms"]]
    rounds = manifest["rounds"]
    lines = [f"# bin/ bench cells: {display_value(manifest['bench'])} on {manifest['backend']}", ""]
    lines.append("## Identity")
    lines.append("")
    for key in ("commit", "executable_sha256", "host", "system", "cpu"):
        lines.append(log_line(f"- {key}: {display_value(str(manifest.get(key)))}"))
    lines.append(log_line("- runtime controls passed through (device, loader, OCaml runtime): " + (
        ", ".join(f"{k}={display_value(v)}" for k, v in manifest["runtime_controls"].items())
        or "none")))
    lines.append(log_line("- cleared from the environment: " + (
        ", ".join(display_value(v) for v in manifest["cleared"]) or "nothing")))
    lines += ["", "## Arms", ""]
    lines.append(log_line("- common arguments: " + spans(manifest["common_args"])))
    for a in manifest["arms"]:
        lines.append(log_line(f"- {display_value(a['label'])}: " + spans(a["args"])))
    exchanged, equal, (lo, hi) = design_balance(len(arms), design_orders(len(arms), rounds))
    lines.append(log_line(
        f"- design: spread mirrored rotation, {len(arms)} arm(s) x {rounds} rounds; every pair "
        f"exchanged: {str(exchanged).lower()}; position histograms equal: {str(equal).lower()} "
        f"(exact at multiples of {exact_batches(len(arms))} rounds); edge visits per arm {lo}-{hi}"))

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
                             ms(statistics.median(xs)) if xs else "-", ms(xs[0]) if xs else "-",
                             ms(xs[-1]) if xs else "-"))
        lines += table_lines(("variant", "arm", "n", "median", "min", "max"), rows, "llrrrr")
        if unchecked:
            lines += ["", "Variants whose output was NOT checked (their times are not evidence of a "
                      "correct kernel):", ""]
            lines += [log_line(f"- {display_value(v)} ({', '.join(display_value(a) for a in arms if a in on)})"
                               f": {note}")  # the bench's own words; log_line refuses a line break
                      for (v, note), on in sorted(unchecked.items())]
    else:
        lines.append("No cell reported a `bench-result:` line.")
    if len(arms) > 1 and variants:
        ref = arms[0]
        lines += ["", f"## Paired against {display_value(ref)} (ratio = arm / reference, per round)",
                  ""]
        rows = []
        for v in variants:
            base = by.get((v, ref), {})
            for a in arms[1:]:
                mine = by.get((v, a), {})
                ratios = [mine[r] / base[r] for r in sorted(set(base) & set(mine))]
                untied = [x for x in ratios if x != 1.0]
                faster = sum(1 for x in untied if x < 1.0)
                rows.append((display_value(v), display_value(a), str(len(ratios)),
                             f"{statistics.median(ratios):.3f}" if ratios else "-",
                             f"{min(ratios):.3f}-{max(ratios):.3f}" if ratios else "-",
                             f"{faster} of {len(untied)}",
                             f"{sign_test_p(faster, len(untied)):.3g}"))
        lines += table_lines(("variant", "arm", "pairs", "median ratio", "range", "faster in",
                              "sign-test p"), rows, "llrrrrr")

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


def run_in_group(argv, cwd, env, timeout, stdout, stderr):
    """`argv` as a managed process group under a cap: its exit status, or "timeout"."""
    with _cancellation.deferring():
        proc = cell_group.spawn(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        try:
            with _cancellation.cancellable():
                status = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
        finally:
            cleanup = cell_group.terminate(proc, grace=1.0)
            if cleanup.observation is not cell_group.GONE or not cleanup.reaped:
                raise cell_group.CleanupFailed(
                    f"bin_cells: child group {proc.pid} cleanup {cleanup.observation.value}; clear "
                    "survivors before measuring again")
    return status


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tree_commit():
    """HEAD, marked when the tree carries anything that commit does not (read before the build)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if head.returncode != 0:
        sys.exit(f"bin_cells: {ROOT} is not a git checkout; pass --exe to measure a binary as is")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                           capture_output=True, text=True)
    return head.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")


def host_identity():
    cpu = "unknown"
    if platform.system() == "Darwin":
        probe = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                               text=True)
        cpu = probe.stdout.strip() or cpu
    elif Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"host": platform.node(), "system": platform.platform(), "cpu": cpu}


def parse_arm(text):
    label, sep, args = text.partition("=")
    if not sep or not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(
            f"--arm {text!r}: expected LABEL=ARGS with LABEL of letters, digits, '_', '.', '-'")
    return {"label": label, "args": shlex.split(args)}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bin_cells.py", description=__doc__.split("\n\n", 1)[0],
        usage="%(prog)s (--bench NAME | --exe PATH) --backend B --out DIR [--rounds R] "
              "[--arm LABEL=ARGS]... [--cell-timeout S] [-- BENCH-ARGS...]")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--bench", help="a bin/ executable's dune name, built as bin/<NAME>.exe")
    which.add_argument("--exe", type=Path, help="run this prebuilt executable instead")
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=6,
                        help="even; each round runs every arm once (default 6)")
    parser.add_argument("--arm", action="append", type=parse_arm, default=[],
                        help="LABEL=ARGS, the arm's own arguments (shell-split); the first arm is "
                             "the reference of the paired ratios. Default: one arm, 'base', "
                             "with none")
    parser.add_argument("--cell-timeout", type=float, default=CELL_TIMEOUT_S,
                        help=f"seconds per cell (default {CELL_TIMEOUT_S})")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    args = build_parser().parse_args(argv[:split])
    common = argv[split + 1:]
    arms = args.arm or [{"label": "base", "args": []}]
    labels = [a["label"] for a in arms]
    if len(set(labels)) != len(labels):
        sys.exit(f"bin_cells: arm labels must be distinct, got {labels}")
    try:
        orders = design_orders(len(arms), args.rounds)
    except ValueError as e:
        sys.exit(f"bin_cells: {e}")
    if not args.cell_timeout > 0:
        sys.exit("bin_cells: --cell-timeout must be positive")
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        sys.exit(f"bin_cells: --out must lie outside the checkout {ROOT}: its cells would write "
                 "into the tree they measure")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        sys.exit(f"bin_cells: {out} is not an empty directory; a leftover cell would be read as "
                 "this run's")
    _cancellation.install()

    if args.exe is not None:
        exe = args.exe.resolve()
        if not exe.is_file():
            sys.exit(f"bin_cells: no executable at {args.exe}")
        bench, commit = exe.name, "n/a (--exe: a prebuilt binary, identified by its sha256)"
    else:
        bench = args.bench
        if not LABEL_RE.fullmatch(bench):
            sys.exit(f"bin_cells: not a dune executable name: {bench!r}")
        commit = tree_commit()
        out.mkdir(parents=True, exist_ok=True)
        build_dir = ROOT / "_build"
        # The build is hermetic in the same sense as the cells: Dune's own environment
        # (DUNE_BUILD_DIR moves the artifacts, DUNE_PROFILE changes the flags, ...) and OCAMLPARAM
        # (compiler flags) are cleared, and the build directory is named, so the executable read
        # below is the one this command built from this checkout.
        build_env = {k: v for k, v in os.environ.items()
                     if not k.startswith(BUILD_CLEARED_PREFIXES)}
        with open(out / "build.log", "wb") as log:
            # --root: dune otherwise picks the OUTERMOST project root, so a nested worktree without
            # its dune-workspace would build the parent checkout while the executable read below
            # (and the commit recorded above) are this one's (AGENTS.md, Worktrees).
            status = run_in_group(["dune", "build", "--root", str(ROOT), "--build-dir",
                                   str(build_dir), f"bin/{bench}.exe"], ROOT, build_env,
                                  BUILD_TIMEOUT_S, log, subprocess.STDOUT)
        exe = build_dir / "default/bin" / f"{bench}.exe"
        if status != 0 or not exe.is_file():
            sys.exit(f"bin_cells: building bin/{bench}.exe failed ({status}); see {out}/build.log")
    out.mkdir(parents=True, exist_ok=True)

    env, cleared = hermetic_env(os.environ)
    pins = [f"--ocannl_backend={args.backend}", "--ocannl_no_config_file=true",
            "--ocannl_log_config_sourcing=true"]
    manifest = {
        "bench": bench, "backend": args.backend, "rounds": args.rounds, "commit": commit,
        "executable": str(exe), "executable_sha256": sha256(exe), **host_identity(),
        "common_args": common, "arms": arms, "pins": pins, "cleared": cleared,
        "runtime_controls": {k: env[k] for k in sorted(env) if k.startswith(RUNTIME_PREFIXES)},
        "passed_names": sorted(env),
        "cell_timeout_s": args.cell_timeout, "orders": orders,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    records = []
    with open(out / "cells.jsonl", "w") as jsonl:
        for rnd, order in enumerate(orders):
            for pos, a in enumerate(order):
                arm = arms[a]
                cell = len(records)
                cwd = out / "cells" / f"{cell:03d}-{arm['label']}"
                cwd.mkdir(parents=True)
                cmd = [str(exe), *pins, *common, *arm["args"]]
                started = time.monotonic()
                with open(cwd / "stdout", "wb") as so, open(cwd / "stderr", "wb") as se:
                    status = run_in_group(cmd, cwd, env, args.cell_timeout, so, se)
                wall = time.monotonic() - started
                stdout = (cwd / "stdout").read_text(errors="replace")
                stderr = (cwd / "stderr").read_text(errors="replace")
                problems, config, results = [], None, []
                if status != 0:
                    problems.append("timed out" if status == "timeout" else f"exited {status}")
                try:
                    config = parse_config_trace(stderr)
                    problems += treatment_problems(config, args.backend)
                except TraceError as e:
                    problems.append(f"unreadable configuration trace: {e}")
                try:
                    results = parse_results(stdout)
                except ValueError as e:
                    problems.append(str(e))
                else:
                    # A cell that measured nothing the driver can read is no measurement: either
                    # the executable does not report through Bench_out.result, or every variant
                    # was skipped. Exiting 0 over an empty table would publish nothing as clean.
                    if not results:
                        problems.append("reported no bench-result line")
                    problems += [f"{v} failed its output check: {note}"
                                 for v, _, check, note in results if check == "failed"]
                    names = [v for v, *_ in results]
                    if len(set(names)) != len(names):
                        problems.append("a variant reported twice: "
                                        + ", ".join(sorted({v for v in names if names.count(v) > 1})))
                rec = {"cell": cell, "round": rnd, "position": pos, "arm": arm["label"],
                       "argv": cmd, "status": status, "wall_s": round(wall, 3),
                       "results": results, "config": config, "problems": problems,
                       "ok": not problems}
                records.append(rec)
                jsonl.write(json.dumps(rec) + "\n")
                jsonl.flush()
                print(log_line(
                    f"bin_cells: cell {cell} round {rnd} position {pos} {display_value(arm['label'])}"
                    f": exit {status}, {len(results)} result(s), {wall:.1f} s"
                    + ("" if not problems else " -- " + display_value("; ".join(problems)))),
                    file=sys.stderr, flush=True)

    problems = [f"cell {r['cell']} ({r['arm']}): {'; '.join(r['problems'])}"
                for r in records if not r["ok"]]
    problems += consistency_problems([r for r in records if r["ok"]])
    report = render_report(manifest, records, problems)
    (out / "report.md").write_text(report)
    sys.stdout.write(report)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
