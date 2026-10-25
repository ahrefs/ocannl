"""The shared cell runner of the benchmark drivers: hermetic, supervised, position-balanced, paired
(gh-ocannl-1565).

A CELL is one whole bench process: one arm's executable and command line, run once. This module is
the machinery every driver that measures cells needs and none should re-implement --
`bin_cells.py` (the `bin/` benches and `bench_gpt` with named arms) and `ab.py` (two revisions,
interleaved) are its two front ends:

- The environment (`hermetic_env`). The library's settings (OCANNL_*), the runners' modes (BENCH_*)
  and the OpenMP runtime's controls are removed; device and loader runtime controls pass through
  and are recorded with their values, the toolchain variables too, every other variable by name.
  A driver hands a cell BENCH_*/OMP_* values only as a stated part of its treatment
  (`assignment_problem` says which names may be stated).
- The configuration trace (`parse_config_trace`, `treatment_problems`, `consistency_problems`).
  Every cell runs with `pins(backend)` -- the backend, `--ocannl_no_config_file=true` and
  `--ocannl_log_config_sourcing=true` -- and the trace in its stderr is read back: a setting from
  the environment or a config file, or a backend other than the pinned one, fails the cell.
- Supervision (`run_in_group`). Each process is a group under `cell_group` with a cap, killed as a
  group on expiry; a group not proven gone stops the run.
- The visiting order (`design_orders`): `bin/bench_design.ml`'s spread mirrored rotation, line for
  line.
- The cell loop (`run_cells`), which takes the driver's READER of a cell's output -- what turns its
  streams into timed results (`bench_output.py` holds every format an OCANNL bench prints).
- The paired statistics (`paired_row`, `sign_test_p`).

Exit statuses a front end shares: 0 when every cell is clean; `REFUSED_EXIT` (3) when the only
faults are cells that themselves exited `REFUSED_EXIT` -- a bench saying the machine refused its
measurement (host contention, gh-ocannl-855), a fact about the box rather than a failed run; 1 for
any other fault; 2 for a usage error.

Boundary: hermeticity here is about CONFIGURATION -- what the library reads. The machine (clocks,
co-tenants, thermal and device state) is the operator's, through an exclusive measurement window.
"""

import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cell_group
from display_safe import display_value, log_line

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
# What a build must not inherit: Dune's own environment and the compiler's flag override.
BUILD_CLEARED_PREFIXES = ("DUNE_", "OCAMLPARAM")
CELL_TIMEOUT_S = 1800
BUILD_TIMEOUT_S = 3600
REFUSED_EXIT = 3
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
ASSIGNMENT_RE = re.compile(r"([A-Z_][A-Z0-9_]*)=(.*)", re.S)
_cancellation = cell_group.CancellationDeferral("cells")


def install_cancellation(driver):
    """Turn SIGINT/SIGTERM/SIGHUP into a cancellation the cell loop honours between steps
    (`cell_group.CancellationDeferral`), named for `driver` in its message."""
    _cancellation.label = driver
    _cancellation.install()


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


def design_line(arms, rounds):
    """The report's one line on the design: its balance, as computed, not as claimed."""
    exchanged, equal, (lo, hi) = design_balance(arms, design_orders(arms, rounds))
    return (f"- design: spread mirrored rotation, {arms} arm(s) x {rounds} rounds; every pair "
            f"exchanged: {str(exchanged).lower()}; position histograms equal: {str(equal).lower()} "
            f"(exact at multiples of {exact_batches(arms)} rounds); edge visits per arm {lo}-{hi}")


# ---------------------------------------------------------------------------------------------
# The environment and the configuration trace.


def hermetic_env(environ):
    """`environ` without the cleared families, and the sorted names it cleared."""
    cleared = sorted(k for k in environ if k.startswith(CLEARED_PREFIXES))
    return {k: v for k, v in environ.items() if k not in cleared}, cleared


def assignment_problem(name):
    """Why `name` may not be STATED as a cell's environment, or None when it may.

    A driver states a variable of a cleared family other than OCANNL_* -- a runner mode
    (`BENCH_TUNE`, `BENCH_MATERIALIZE`), an OpenMP control -- as part of a cell's treatment; it is then
    recorded with the treatment. The library's settings are not stated this way: on the command line
    they outrank every other source and the configuration trace reads them back as `commandline`,
    where an environment value is exactly what the treatment check refuses."""
    if name.startswith("OCANNL_"):
        return (f"{name}: a library setting is stated on the command line, where the "
                "configuration trace reads it back, never in the cell's environment")
    if not name.startswith(CLEARED_PREFIXES):
        return (f"{name}: only the cleared families ({', '.join(CLEARED_PREFIXES[1:])}) are stated; "
                "every other variable is the box's and passes through as it is")
    return None


def split_assignments(words):
    """`words` as `env`(1) reads them: the leading NAME=VALUE words, then the rest. A NAME that may
    not be stated (`assignment_problem`) raises ValueError."""
    env, rest = {}, list(words)
    while rest:
        m = ASSIGNMENT_RE.fullmatch(rest[0])
        if not m:
            break
        problem = assignment_problem(m.group(1))
        if problem is not None:
            raise ValueError(problem)
        env[m.group(1)] = m.group(2)
        rest.pop(0)
    return env, rest


class TraceError(ValueError):
    """The configuration trace in a cell's stderr cannot be read as the library writes it."""


_RETRIEVING = re.compile(r"Retrieving commandline, environment, or config file variable ocannl_(\S+)")
# `Utils.describe_config_source`'s five shapes. Four end in a part that cannot hold their own
# separator (a variable name, fixed words, a quoted profile name and a fixed level), so the value is
# greedy up to the LAST separator. The commandline shape ends in the argv element, which repeats
# the value and so can hold `, commandline ` itself: it is matched against the cell's own argv
# instead (`_commandline_entry`), and never by this list.
_COMMANDLINE = "commandline"
_SOURCES = [
    (re.compile(r"Found (.*), environment (\S+)"), "environment"),
    (re.compile(r"Found (.*), in the config file"), "config file"),
    (re.compile(r'Found (.*), in the profile "(.*)" picked via (.*)'), "profile"),
    (re.compile(r"Not found, using default (.*)"), "default"),
]


def _commandline_entry(line, argv):
    """The entry of a `Found <value>, commandline <arg>` line, `arg` being one of `argv` -- the
    longest that fits, so an argument that is a suffix of another cannot steal its line -- or None
    when the line is not of that shape. A commandline-shaped line naming no argv element is
    refused: the cell's command line is known exactly."""
    if not line.startswith("Found ") or ", commandline " not in line:
        return None
    for arg in sorted(set(argv), key=len, reverse=True):
        tail = ", commandline " + arg
        if line.endswith(tail) and len(line) >= len("Found ") + len(tail):
            return {"value": line[len("Found "):-len(tail)], "source": _COMMANDLINE, "detail": arg}
    if any(pattern.fullmatch(line) for pattern, _ in _SOURCES):
        return None  # another source's line whose value happens to hold the separator
    raise TraceError(f"a commandline source naming no argument of the cell: {line!r}")


def parse_config_trace(text, argv):
    """{key: {"value", "source", "detail"}} from the `log_config_sourcing` trace in `text`, for a
    cell run with the command line `argv`.

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
        entry = _commandline_entry(line, argv)
        for pattern, source in [] if entry is not None else _SOURCES:
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
            break
        if entry is not None:
            known = config.get(pending)
            if known is not None and known["value"] != entry["value"]:
                raise TraceError(
                    f"ocannl_{pending} resolved to both {known['value']!r} and {entry['value']!r}")
            config.setdefault(pending, entry)
            pending = None
    if pending is not None:
        raise TraceError(f"ocannl_{pending} was requested and never resolved")
    return config


def treatment_problems(config, backend):
    """Why a cell's resolved configuration is not the treatment its command line states."""
    if not config:
        return ["no configuration trace: the bench resolved no setting the driver could read"]
    problems = []
    b = config.get("backend")
    if b is None or b["value"] != backend or b["source"] != _COMMANDLINE:
        problems.append(f"the backend resolved to {b and b['value']!r} from "
                        f"{b and b['source']}, not to the pinned {backend!r}")
    nc = config.get("no_config_file")
    if nc is None or nc["value"] != "true" or nc["source"] != _COMMANDLINE:
        problems.append("the config file was not disabled")
    for key, entry in sorted(config.items()):
        if entry["source"] in ("environment", "config file"):
            problems.append(f"ocannl_{key} came from the {entry['source']}")
        elif entry["source"] == "profile" and not entry["detail"].endswith("via the commandline"):
            problems.append(f"ocannl_{key} came from a profile picked outside the command line")
    return problems


def consistency_problems(records):
    """Settings the successful cells of one arm did not resolve identically: to two values, or read
    in some cells and not in others (a cell that took a path reading one more setting ran a
    treatment its siblings did not)."""
    problems = []
    arms = {}
    for rec in records:
        if rec["config"] is not None:
            arms.setdefault(rec["arm"], []).append(rec)
    for arm, recs in arms.items():
        for key in sorted({k for r in recs for k in r["config"]}):
            values = {}
            for r in recs:
                entry = r["config"].get(key)
                values.setdefault(None if entry is None else entry["value"], []).append(r["cell"])
            if len(values) > 1:
                problems.append(f"arm {arm}: ocannl_{key} resolved differently -- " + "; ".join(
                    ("not read" if v is None else repr(v)) + " in cell(s) "
                    + ", ".join(map(str, cells)) for v, cells in values.items()))
    return problems


def pins(backend):
    """The three arguments every cell starts with, ahead of the bench's own."""
    return [f"--ocannl_backend={backend}", "--ocannl_no_config_file=true",
            "--ocannl_log_config_sourcing=true"]


# ---------------------------------------------------------------------------------------------
# Statistics.


def sign_test_p(faster, n):
    """The exact two-sided sign-test p of `faster` successes in `n` untied pairs."""
    if n == 0:
        return 1.0
    k = min(faster, n - faster)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(base, mine):
    """The paired comparison of one arm against the reference over the rounds both measured, from
    {round: seconds} maps: {"pairs", "median", "lo", "hi", "faster", "untied", "p"} (None for the
    ratio statistics when no round pairs)."""
    ratios = [mine[r] / base[r] for r in sorted(set(base) & set(mine))]
    untied = [x for x in ratios if x != 1.0]
    faster = sum(1 for x in untied if x < 1.0)
    return {"pairs": len(ratios), "median": statistics.median(ratios) if ratios else None,
            "lo": min(ratios) if ratios else None, "hi": max(ratios) if ratios else None,
            "faster": faster, "untied": len(untied), "p": sign_test_p(faster, len(untied))}


def paired_cells(stats):
    """`paired`'s statistics as the report's table cells (pairs, median ratio, range, faster in, p)."""
    have = stats["median"] is not None
    return (str(stats["pairs"]), f"{stats['median']:.3f}" if have else "-",
            f"{stats['lo']:.3f}-{stats['hi']:.3f}" if have else "-",
            f"{stats['faster']} of {stats['untied']}", f"{stats['p']:.3g}")


PAIRED_HEADER = ("pairs", "median ratio", "range", "faster in", "sign-test p")


# ---------------------------------------------------------------------------------------------
# Supervised processes, identities.


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
                    f"cells: child group {proc.pid} cleanup {cleanup.observation.value}; clear "
                    "survivors before measuring again")
    return status


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# The variables a C toolchain consults, recorded with their values beside `RUNTIME_PREFIXES`'s.
# Every other variable a cell receives is recorded by NAME only: an inherited environment carries
# credentials under names no pattern can recognize (a token inside DATABASE_URL), and the manifest
# travels with published numbers. What the cc backend actually compiles kernels with is
# fingerprinted directly instead (`kernel_compiler`), so a toolchain fact does not rest on this
# list being complete.
TOOLCHAIN_VARS = ("PATH", "TMPDIR", "SDKROOT", "DEVELOPER_DIR", "MACOSX_DEPLOYMENT_TARGET", "CPATH",
                  "C_INCLUDE_PATH", "LIBRARY_PATH", "CC", "CFLAGS", "CPPFLAGS", "LDFLAGS",
                  "OCAMLFIND_TOOLCHAIN")


def recorded_values(env):
    """The variables of `env` recorded with their values: runtime controls and toolchain ones."""
    return {k: env[k] for k in sorted(env)
            if k.startswith(RUNTIME_PREFIXES) or k in TOOLCHAIN_VARS}


def runtime_controls(env):
    return {k: env[k] for k in sorted(env) if k.startswith(RUNTIME_PREFIXES)}


def kernel_compiler(command, env, out):
    """The executable the cc backend compiles kernels with, fingerprinted: the first word of the
    resolved `cc_backend_compiler_command` (`command`), or of `ocamlc -config`'s `c_compiler` when
    that is empty -- `Cc_backend.compiler_command`'s rule -- resolved on the cells' PATH as
    `Cc_backend.resolve_executable` resolves it. {"command", "resolved", "sha256"}, or an
    {"error"} naming what could not be resolved."""
    path = env.get("PATH", "")
    if not command.strip():
        ocamlc = shutil.which("ocamlc", path=path)
        if ocamlc is None:
            return {"error": "no ocamlc on the cells' PATH to ask for its c_compiler"}
        with open(out, "wb") as f:
            status = run_in_group([ocamlc, "-config"], out.parent, env, 120, f, subprocess.STDOUT)
        lines = out.read_text(errors="replace").splitlines() if status == 0 else []
        command = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("c_compiler:")),
                       "")
        if not command:
            return {"error": f"ocamlc -config named no c_compiler (status {status})"}
    try:
        first = shlex.split(command)[0]
    except (ValueError, IndexError):
        return {"command": command, "error": "the compiler command does not split into words"}
    found = first if os.sep in first else shutil.which(first, path=path)
    if found is None or not os.path.isfile(found):
        return {"command": command, "error": f"{first} resolves to no executable"}
    real = os.path.realpath(found)
    return {"command": command, "resolved": real, "sha256": sha256(real)}


def kernel_compilers(arms, records, env, out):
    """`kernel_compiler` per arm, from each arm's first traced cell: {label: fingerprint}."""
    compilers = {}
    for arm in arms:
        config = next((r["config"] for r in records
                       if r["arm"] == arm["label"] and r["config"]), {})
        command = config.get("cc_backend_compiler_command", {}).get("value", "")
        compilers[arm["label"]] = kernel_compiler(command, env,
                                                  out / f"ocamlc-config-{arm['label']}.txt")
    return compilers


def tree_commit(root=ROOT, driver="cells"):
    """HEAD of `root`, marked when the tree carries anything that commit does not (read before the
    build)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    if head.returncode != 0:
        sys.exit(f"{driver}: {root} is not a git checkout; pass --exe to measure a binary as is")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
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


def fresh_out_dir(path, driver):
    """`path` resolved, refused (exit) unless it is outside the checkout and empty or absent."""
    out = path.resolve()
    if out == ROOT or ROOT in out.parents:
        sys.exit(f"{driver}: --out must lie outside the checkout {ROOT}: its cells would write "
                 "into the tree they measure")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        sys.exit(f"{driver}: {out} is not an empty directory; a leftover cell would be read as "
                 "this run's")
    return out


def build_target(root, target, log_path, driver, jobs=None):
    """Build the dune `target` (a path such as `bin/schedule_bench.exe`) of the checkout `root`,
    hermetically, and return the built file's path; exit naming the log when it fails.

    The build is hermetic in the same sense as the cells: Dune's own environment (DUNE_BUILD_DIR
    moves the artifacts, DUNE_PROFILE changes the flags, ...) and OCAMLPARAM (compiler flags) are
    cleared, and the root and build directory are named -- dune otherwise picks the OUTERMOST
    project root, so a nested worktree without its dune-workspace would build the parent checkout
    while the executable read here is this one's (AGENTS.md, Worktrees). `jobs` is dune's -j."""
    build_dir = Path(root) / "_build"
    build_env = {k: v for k, v in os.environ.items() if not k.startswith(BUILD_CLEARED_PREFIXES)}
    with open(log_path, "wb") as log:
        status = run_in_group(["dune", "build", "--root", str(root), "--build-dir",
                               str(build_dir), *([] if jobs is None else ["-j", str(jobs)]),
                               target], root, build_env, BUILD_TIMEOUT_S, log, subprocess.STDOUT)
    exe = build_dir / "default" / target
    if status != 0 or not exe.is_file():
        sys.exit(f"{driver}: building {target} failed ({status}); see {log_path}")
    return exe


# ---------------------------------------------------------------------------------------------
# The cell loop.


def parse_arm(text, flag="--arm"):
    """`LABEL=ARGS` as an arm: {"label", "args", "env"}, the leading NAME=VALUE words of ARGS being
    the arm's stated environment (`split_assignments`). ValueError when malformed."""
    label, sep, args = text.partition("=")
    if not sep or not LABEL_RE.fullmatch(label):
        raise ValueError(f"{flag} {text!r}: expected LABEL=ARGS with LABEL of letters, digits, "
                         "'_', '.', '-'")
    env, rest = split_assignments(shlex.split(args))
    return {"label": label, "args": rest, "env": env}


def run_cells(*, arms, orders, out, env, backend, common, common_env, timeout, reader, driver):
    """Run every cell of the design and return its records, one per cell, also appended to
    `out/cells.jsonl` as they finish.

    `arms`: [{"label", "exe", "args", "env"}], `exe` the arm's executable. A cell's command line is
    the arm's executable, `pins(backend)`, the `common` arguments, then the arm's own; its
    environment is `env` (already hermetic) plus `common_env`, then the arm's stated variables. It
    runs in `out/cells/<NNN>-<label>/`, which keeps its `stdout` and `stderr`.

    `reader(stdout, stderr)` turns a cell's output into ([(variant, seconds, check, note)],
    [problem]), raising ValueError for output it cannot read; a cell is clean when it exited 0, its
    trace states the treatment, and its reader found results and no problem. A cell exiting
    `REFUSED_EXIT` is marked `refused`."""
    records = []
    with open(out / "cells.jsonl", "w") as jsonl:
        for rnd, order in enumerate(orders):
            for pos, a in enumerate(order):
                arm = arms[a]
                cell = len(records)
                cwd = out / "cells" / f"{cell:03d}-{arm['label']}"
                cwd.mkdir(parents=True)
                cmd = [str(arm["exe"]), *pins(backend), *common, *arm["args"]]
                cell_env = {**env, **common_env, **arm.get("env", {})}
                started = time.monotonic()
                with open(cwd / "stdout", "wb") as so, open(cwd / "stderr", "wb") as se:
                    status = run_in_group(cmd, cwd, cell_env, timeout, so, se)
                wall = time.monotonic() - started
                stdout = (cwd / "stdout").read_text(errors="replace")
                stderr = (cwd / "stderr").read_text(errors="replace")
                problems, config, results = [], None, []
                refused = status == REFUSED_EXIT
                if refused:
                    problems.append(f"refused its measurement (exit {REFUSED_EXIT}): the bench "
                                    "says the machine, not the treatment, stopped it")
                elif status != 0:
                    problems.append("timed out" if status == "timeout" else f"exited {status}")
                try:
                    config = parse_config_trace(stderr, cmd)
                    problems += treatment_problems(config, backend)
                except TraceError as e:
                    problems.append(f"unreadable configuration trace: {e}")
                try:
                    results, read_problems = reader(stdout, stderr)
                except ValueError as e:
                    problems.append(str(e))
                else:
                    problems += read_problems
                rec = {"cell": cell, "round": rnd, "position": pos, "arm": arm["label"],
                       "argv": cmd, "env": {**common_env, **arm.get("env", {})},
                       "status": status, "wall_s": round(wall, 3), "results": results,
                       "config": config, "problems": problems, "refused": refused,
                       "ok": not problems}
                records.append(rec)
                jsonl.write(json.dumps(rec) + "\n")
                jsonl.flush()
                print(log_line(
                    f"{driver}: cell {cell} round {rnd} position {pos} {display_value(arm['label'])}"
                    f": exit {status}, {len(results)} result(s), {wall:.1f} s"
                    + ("" if not problems else " -- " + display_value("; ".join(problems)))),
                    file=sys.stderr, flush=True)
    return records


def timed_results_problems(results, nothing):
    """The problems of a cell's timed results whatever format they came in: none at all (`nothing`
    says so in the reader's words), a variant whose output check failed, a variant reported
    twice."""
    # A cell that measured nothing the driver can read is no measurement: either the executable
    # does not report in the format its reader takes, or every variant was skipped. Exiting 0 over
    # an empty table would publish nothing as clean.
    if not results:
        return [nothing]
    problems = [f"{v} failed its output check: {note}"
                for v, _, check, note in results if check == "failed"]
    names = [v for v, *_ in results]
    if len(set(names)) != len(names):
        problems.append("a variant reported twice: "
                        + ", ".join(sorted({v for v in names if names.count(v) > 1})))
    return problems


def run_problems(records, extra=()):
    """The run's verdict lines: each unclean cell, then `extra`, then the arms' consistency."""
    problems = [f"cell {r['cell']} ({r['arm']}): {'; '.join(r['problems'])}"
                for r in records if not r["ok"]]
    problems += list(extra)
    problems += consistency_problems([r for r in records if r["ok"]])
    return problems


def read_run(run_dir):
    """A finished run's `manifest.json` and `cells.jsonl` records, for a driver that judges its cells
    further (a census, a header): (manifest, records). OSError/ValueError when either is missing
    or unreadable."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in (run_dir / "cells.jsonl").read_text().splitlines()]
    return manifest, records


def cell_stream(run_dir, rec, stream):
    """The text of a cell's `stdout` or `stderr`, as its run directory keeps it."""
    return (Path(run_dir) / "cells" / f"{rec['cell']:03d}-{rec['arm']}" / stream).read_text(
        errors="replace")


def exit_status(records, problems):
    """0 when clean; `REFUSED_EXIT` when every fault is a cell that refused its measurement and
    nothing else is wrong; 1 otherwise."""
    if not problems:
        return 0
    unclean = [r for r in records if not r["ok"]]
    only_refusals = (unclean and len(problems) == len(unclean)
                     and all(r["refused"] and len(r["problems"]) == 1 for r in unclean))
    return REFUSED_EXIT if only_refusals else 1
