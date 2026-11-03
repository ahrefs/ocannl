"""What OCANNL's bench executables print, read in one place (gh-ocannl-1443).

Every driver that reads a bench's output reads it through this module, so a format has one reader
and a change to the emitter has one place to land. The formats:

- `bench-result:` lines (`Bench_out.result`, the `bin/` timing benches): one JSON object per timed
  variant, `{"variant", "seconds", "check", "check_note"}` -- `bench_results`.
- The runners' result line (`Bench_json`: `bench_gpt`, `bench_conv`, the other OCANNL runners, and
  the Python runners' `bench_common.emit`): one JSON object on stdout, the last line that opens one
  -- `result_lines`, `result_record`, `only_result_record`. Its fields are the benchmarks README's
  result schema (`step_ms.p50`, `compile_s`, `searched`, `tune.*`, `losses`, ...); `field` walks a
  dotted path into it.
- Checkpoints on stderr (`Bench_json.checkpoint_prefix`, gh-ocannl-1209): the parity losses a cell
  saw before a later stage, all that is left when it is killed there -- `last_checkpoint`.
- Compile timing records on stderr under `BENCH_TIMING_TRACE=1` (`timing-trace: compile ...`) --
  `compile_traces`; the rest of that trace (`timing-trace: summary/attempt/search done`) --
  `trace_summary`, `trace_has`.

The text lines (gh-ocannl-1767), each with ONE reader here, listed in `FORMATS`:

- The GEBP benches' timing line (`bin/schedule_bench`, `bin/narrow_gebp_bench`):
  `<variant> <ms> ms <rate> GFLOP/s (<detail>)  [<mma census>] [<grid census>]` -- `timing_lines`;
  and narrow_gebp_bench's `GEBP n=...` header -- `gebp_header`.
- The per-kernel table's rows (`bench: kernel i/n ...`, Bench_harness.time_shipped_kernels) --
  `kernel_rows`; and its opening legend (`bench: per-kernel table (...; each kernel alone)`,
  Bench_harness.kernel_timing_legend), whose `autotune_timing=` it states -- `kernel_table_timing`.
- `bench_rank_site`'s `rank-site: <kind> key=value ...` lines -- `rank_site_lines`.
- The launch-geometry trace (`schedule: <routine> seg i/n grid=[..] block=[..] stmts=..`, config
  `schedule_log_launches`) -- `schedule_launches`.
- The seeded-default census the untuned compile logs (`seeded_default: ...`) -- `seeded_default`.

A shell driver asks the same readers through the command line at the end of this file (`python3
bench_output.py <command> FILE ...`) rather than grepping a line itself: a pattern of its own is a
second reader, and a second reader is how gh-ocannl-1668's appended `[grid: ...]` census went unread
-- a last-bracket match took it for the mma census. test/operations/bench_line_readers runs every
reader here over lines the producers print, and refuses a bench-line pattern anywhere else in
`benchmarks/` or `tools/`.

Every JSON read is `strict_json`'s: a non-finite number is refused (`strict_json.NonFinite`, a
ValueError), never read as a float, since a runner writes `null` there (gh-ocannl-676).
"""

import math
import os
import re
import sys

# The command line runs under `python3 -I` too, which leaves this file's directory off sys.path.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import strict_json  # noqa: E402
from strict_json import NonFinite, field  # noqa: E402,F401  (re-exported: the result line's readers)

RESULT_PREFIX = "bench-result: "
CHECKS = ("passed", "failed", "unchecked")
CHECKPOINT_PREFIX = "bench: checkpoint "


def bench_results(text):
    """The `bench-result:` records of a cell's stdout: [(variant, seconds, check, note)], refusing
    with ValueError a malformed one, a non-finite or non-positive time, or one that does not state
    its output check (`Bench_out.check`)."""
    results = []
    for line in text.splitlines():
        if not line.startswith(RESULT_PREFIX):
            continue
        record = strict_json.loads(line[len(RESULT_PREFIX):])
        if not isinstance(record, dict):
            raise ValueError(f"unusable result line (not a JSON object): {line}")
        variant, seconds = record.get("variant"), record.get("seconds")
        check, note = record.get("check"), record.get("check_note")
        if not isinstance(variant, str) or isinstance(seconds, bool) or not isinstance(
                seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0 \
                or check not in CHECKS or not isinstance(note, str):
            raise ValueError(f"unusable result line: {line}")
        results.append((variant, float(seconds), check, note))
    return results


def result_lines(text):
    """The lines of a cell's stdout that open a JSON object, stripped, in order. The runner's result
    line is the last of them (`Bench_json`: printed last); a checkpoint starts with its prefix, not
    '{', so it is never among them."""
    return [line.strip() for line in text.splitlines() if line.lstrip().startswith("{")]


def result_record(text):
    """The record of the runner's result line -- the LAST line of `text` that opens a JSON object --
    read strictly; None when there is none. ValueError (`NonFinite` among them) when that line is
    not strict JSON; a reader that tells the two apart catches `NonFinite` first."""
    lines = result_lines(text)
    return strict_json.loads(lines[-1]) if lines else None


def only_result_record(text):
    """The record of the ONE result line `text` holds, for a reader whose protocol is one process,
    one result: ValueError naming the count when there are none or several, or when the line is not
    strict JSON."""
    lines = result_lines(text)
    if len(lines) != 1:
        raise ValueError(f"expected one result line, found {len(lines)}")
    return strict_json.loads(lines[0])


def last_checkpoint(text):
    """The last checkpoint in a cell's output, parsed, or None.

    A checkpoint records the parity losses a cell observed before a later stage -- the timed steps,
    or the dominant-kernel instrument -- and is all that is left of them when the cell is killed
    there. It is not a result line and never stands in for one: it starts with CHECKPOINT_PREFIX,
    not '{', so the result readers cannot pick it up, and it carries no timing.

    A kill can land in the middle of a write, so a final line that does not parse is skipped for
    the one before it.
    """
    for line in reversed(text.splitlines()):
        if not line.startswith(CHECKPOINT_PREFIX):
            continue
        try:
            obj = strict_json.loads(line[len(CHECKPOINT_PREFIX):])
        except ValueError:  # strict_json.NonFinite included: a runner emits null there
            continue
        if isinstance(obj, dict) and obj.get("record") == "checkpoint":
            return obj
    return None


_COMPILE_TRACE = re.compile(
    r'timing-trace: compile \d+ at \S+: (\S+) "(.*)" (\S+), (\S+) s \(lower (\S+), '
    r'transform (\S+), codegen (\S+), device (\S+), link (\S+)\), (\d+) of (\d+) kernels')


def compile_traces(text):
    """The `timing-trace: compile` records of a cell's stderr, in order: dicts of `stage`, `label`,
    `verdict` (`compiled`, ...), the seconds `wall`, `lower`, `transform`, `codegen`, `device`
    (None where the backend reports `none`) and `link`, and `kernels_compiled` of `kernels`."""
    rows = []
    for line in text.splitlines():
        m = _COMPILE_TRACE.search(line)
        if not m:
            continue
        stage, label, verdict, wall, lower, transform, codegen, device, link, kc, kr = m.groups()
        rows.append({"stage": stage, "label": label, "verdict": verdict, "wall": float(wall),
                     "lower": float(lower), "transform": float(transform),
                     "codegen": float(codegen), "device": None if device == "none" else float(device),
                     "link": float(link), "kernels_compiled": int(kc), "kernels": int(kr)})
    return rows


# ---------------------------------------------------------------------------------------------
# The trace's session records (gh-ocannl-834): `timing-trace: summary: ...` (printed at exit, with
# the running totals), `timing-trace: attempt ...` (one per candidate the tuner attempts) and
# `timing-trace: search done: ...` (the end of the measured compile). Bench_harness.install_timing_trace
# and trace_search_done print them.

TRACE_PREFIX = "timing-trace: "
_TRACE_SUMMARY = re.compile(
    r"^timing-trace: summary: (?P<wall>[0-9.]+)s wall, (?P<attempts>\d+) candidate attempts, "
    r"(?P<calls>\d+) timing calls, (?P<launches>\d+) launches, ")


def trace_summary(text):
    """The LAST `timing-trace: summary` record of a cell's stderr, or None: a dict of `wall` (s),
    `attempts`, `calls` (timing calls), `launches` and `incomplete` (the summary names tagged timing
    calls that raised, whose cost is in none of its totals). ValueError for a summary line that
    does not read as one."""
    found = None
    for line in text.splitlines():
        if not line.startswith(TRACE_PREFIX + "summary: "):
            continue
        m = _TRACE_SUMMARY.match(line)
        if not m:
            raise ValueError(f"unusable trace summary: {line}")
        found = {"wall": float(m["wall"]), "attempts": int(m["attempts"]),
                 "calls": int(m["calls"]), "launches": int(m["launches"]),
                 "incomplete": "; INCOMPLETE: " in line}
    return found


TRACE_EVENTS = {"attempt": TRACE_PREFIX + "attempt ", "search done": TRACE_PREFIX + "search done: ",
                "call": TRACE_PREFIX + "call ", "compile": TRACE_PREFIX + "compile "}


def trace_events(text, event):
    """The trace's `event` lines (a key of TRACE_EVENTS), in order."""
    prefix = TRACE_EVENTS[event]
    return [line for line in text.splitlines() if line.startswith(prefix)]


# ---------------------------------------------------------------------------------------------
# The GEBP benches' timing line (bin/schedule_bench, bin/narrow_gebp_bench):
#   `<variant> <ms> ms <rate> GFLOP/s  (<detail: spot, chk a/b, agreement>)  [<mma>] [<grid>]`
# The censuses are read LEFT TO RIGHT and must be exactly the two the producers print
# (gh-ocannl-626's mma census, then gh-ocannl-1668's grid census): a last-bracket match read the grid
# census as the mma census from gh-ocannl-1668 on, so a census added, dropped or moved is refused
# here, by name, rather than read as a different one.

TIMING_CENSUSES = ("mma", "grid")
# The line before gh-ocannl-1668 carried the mma census alone. A driver that times historical
# revisions (gh1051_boundary_split.sh) names this layout explicitly, beside the current one; the
# census count then says which layout a line is in, and any other count is still refused.
TIMING_CENSUSES_BEFORE_1668 = ("mma",)
_TIMING_HEAD = re.compile(r"^(?P<variant>\w+)\s+(?P<ms>\S+) ms\s+(?P<rate>\S+) GFLOP/s\s+")
_TIMING_CHK = re.compile(r"(?:^|[ ,])chk ([^,]+),")


def _balanced(line, start, opening, closing):
    """The index just past the group `line[start]` (== opening) opens, counting nested pairs, or
    None when it never closes."""
    depth = 0
    for j in range(start, len(line)):
        if line[j] == opening:
            depth += 1
        elif line[j] == closing:
            depth -= 1
            if depth == 0:
                return j + 1
    return None


def timing_line(line, layouts=(TIMING_CENSUSES,)):
    """One timing line as a dict -- `variant`, `ms`, `rate` (GFLOP/s), `detail` (the parenthesized
    group), `chk` (its printed checksum, `a/b`) and the censuses by name, in the layout of
    `layouts` (tuples of census names) whose census count the line has -- or None when the line
    does not open like one. ValueError for a line that opens like one and does not read as one,
    its census count among them."""
    head = _TIMING_HEAD.match(line)
    if not head:
        return None
    rest = line[head.end():]
    if not rest.startswith("("):
        raise ValueError(f"timing line without its detail group: {line}")
    end = _balanced(rest, 0, "(", ")")
    if end is None:
        raise ValueError(f"timing line whose detail group never closes: {line}")
    detail, rest = rest[1:end - 1], rest[end:]
    censuses = []
    while rest.strip():
        stripped = rest.lstrip()
        if not stripped.startswith("["):
            raise ValueError(f"timing line with text after its censuses: {line}")
        close = _balanced(stripped, 0, "[", "]")
        if close is None:
            raise ValueError(f"timing line whose census never closes: {line}")
        censuses.append(stripped[1:close - 1])
        rest = stripped[close:]
    layout = next((names for names in layouts if len(names) == len(censuses)), None)
    if layout is None:
        expected = " or ".join(f"{len(names)} ({', '.join(names)})" for names in layouts)
        raise ValueError(f"timing line with {len(censuses)} censuses, not the {expected} it "
                         f"prints: {line}")
    chk = _TIMING_CHK.search(detail)
    try:
        ms, rate = float(head["ms"]), float(head["rate"])
    except ValueError:
        raise ValueError(f"timing line with an unreadable time or rate: {line}") from None
    return {"variant": head["variant"], "ms": ms, "rate": rate, "detail": detail,
            "chk": chk.group(1).strip() if chk else None,
            **dict(zip(layout, censuses))}


def timing_lines(text, layouts=(TIMING_CENSUSES,)):
    """The timing lines of a GEBP bench's stdout, in order (`timing_line`)."""
    return [t for t in (timing_line(line, layouts) for line in text.splitlines()) if t is not None]


GEBP_HEADER_PREFIX = "GEBP "


def gebp_header(text):
    """narrow_gebp_bench's `GEBP n=..., register tile ...` header line, or None."""
    return next((line for line in text.splitlines() if line.startswith(GEBP_HEADER_PREFIX)), None)


# ---------------------------------------------------------------------------------------------
# The per-kernel table's rows (Bench_harness.time_shipped_kernels, BENCH_KERNEL_TABLE=1 and the
# dominant-kernel instrument). A timed row:
#   `bench: kernel <i>/<n> <ms> ms ms_lo=.. ms_hi=.. ms_retimed=<a>/<c> grid=[..] block=[..]
#    io:<read>/<written><flags> mma:<census> vol:<census> w: <written nodes>`
# `io:` (global-memory bytes by the cost model; `~` an upper bound, `?` code the model cannot see)
# is gh-ocannl-1444's and the range is gh-ocannl-1694's: both are absent from older rows, whose `ms`
# is a min-of-20, so a recorded run still reads. The groups are named so a field added to the row
# renumbers nothing. A kernel with no time (DECLINED: it would not compile alone; UNTIMED: none of
# its timing calls was admitted) prints `bench: kernel <i>/<n> grid=[..] block=[..] <LABEL> (<detail>)
# w: ..`, a DECLINED detail running over several lines before its `w:`.

KERNEL_ROW_PREFIX = "bench: kernel "
KERNEL = re.compile(r"^bench: kernel (?P<i>\d+)/(?P<n>\d+) (?P<ms>[0-9.]+) ms "
                    r"(?:ms_lo=(?P<lo>-?nan|[0-9.]+) ms_hi=(?P<hi>-?nan|[0-9.]+) "
                    r"ms_retimed=(?P<admitted>\d+)/(?P<calls>\d+) )?"
                    r"grid=\[(?P<grid>[0-9;]+)\] block=\[(?P<block>[0-9;]+)\] "
                    r"(?:io:(?P<read>\d+)/(?P<written>\d+)(?P<io_flags>[~?]*) )?"
                    r"mma:(?P<mma>.*?) (?:vol:.*?)?w: (?P<writes>.*)$")
KERNEL_FAILURE = re.compile(r"^bench: kernel (?P<i>\d+)/(?P<n>\d+) "
                            r"(?:grid=\[(?P<grid>[0-9;]+)\] block=\[(?P<block>[0-9;]+)\] )?"
                            r"(?P<failure>DECLINED|UNTIMED) \(")
_KERNEL_ROW_HEAD = re.compile(r"^bench: kernel \d+/\d+ ")
_KERNEL_TABLE_LEGEND = re.compile(r"^bench: per-kernel table \((?P<legend>.*); each kernel alone\)$",
                                  re.M)
_LEGEND_TIMING = re.compile(r"\bautotune_timing=(\w+)")


def kernel_table_timing(text):
    """The `autotune_timing` the per-kernel table was re-timed under, as its opening legend states it
    (`isolated` or `queued`), or None when the text has no legend: a cell from before gh-ocannl-1694,
    or one whose table never opened. ValueError for a legend that states no timing, and for two
    legends that disagree."""
    timings = set()
    for m in _KERNEL_TABLE_LEGEND.finditer(text):
        t = _LEGEND_TIMING.search(m["legend"])
        if t is None:
            raise ValueError(f"the per-kernel table's legend states no autotune_timing: {m[0]}")
        timings.add(t[1])
    if len(timings) > 1:
        raise ValueError(f"the per-kernel table's legends disagree: {sorted(timings)}")
    return next(iter(timings), None)


def kernel_rows(text):
    """The kernel table's rows, in order: dicts of `i`, `n`, `ms` (None on a row with no time),
    `lo`/`hi` (the row's range: None on a row without one, NaN where it states none), `grid`,
    `block`, `io` ((read, written, flags), None on a row without it), `mma`, `writes` (the written
    nodes; [] on a row with no time) and `failure` (None, `DECLINED` or `UNTIMED`). ValueError for a
    line that opens like a row and reads as neither kind."""
    rows = []
    for line in text.splitlines():
        if not _KERNEL_ROW_HEAD.match(line):
            continue
        m = KERNEL.match(line)
        if m:
            rows.append({"i": int(m["i"]), "n": int(m["n"]), "ms": float(m["ms"]),
                         "lo": None if m["lo"] is None else float(m["lo"]),
                         "hi": None if m["hi"] is None else float(m["hi"]),
                         "grid": m["grid"], "block": m["block"],
                         "io": None if m["read"] is None
                         else (int(m["read"]), int(m["written"]), m["io_flags"]),
                         "mma": m["mma"], "writes": m["writes"].split(), "failure": None})
            continue
        f = KERNEL_FAILURE.match(line)
        if not f:
            raise ValueError(f"unusable kernel row: {line}")
        rows.append({"i": int(f["i"]), "n": int(f["n"]), "ms": None, "lo": None, "hi": None,
                     "grid": f["grid"], "block": f["block"], "io": None, "mma": None, "writes": [],
                     "failure": f["failure"]})
    return rows


# ---------------------------------------------------------------------------------------------
# bench_rank_site's lines (gh-ocannl-1442 and on): `rank-site: <kind> key=value key=value ...`,
# a value either a run of non-blanks or an OCaml string literal (`label=%S`). Read as fields by name,
# so neither their order nor a field added between them moves another; the literals are decoded as
# OCaml writes them (`\ddd` is DECIMAL there, octal to Python's `unicode_escape`).

RANK_SITE_PREFIX = "rank-site: "
_RANK_SITE_HEAD = re.compile(r"^rank-site: (?P<kind>\w+)(?= |$)")
_RANK_SITE_TOKEN = re.compile(r'\s*(?:(?P<key>\w+)=(?:"(?P<lit>(?:[^"\\]|\\.)*)"|(?P<raw>[^\s"]*))'
                              r'|(?P<bare>\S+))(?=\s|$)')
_OCAML_ESCAPE = re.compile(r'\\(?:(?P<dec>\d{3})|x(?P<hex>[0-9a-fA-F]{2})|o(?P<oct>[0-3][0-7]{2})'
                           r'|(?P<char>[\\"\'ntbr ]))')
_OCAML_CHARS = {"\\": "\\", '"': '"', "'": "'", "n": "\n", "t": "\t", "b": "\b", "r": "\r", " ": " "}


def ocaml_string(literal):
    """The text of an OCaml string literal's body (`%S` without its quotes), decoded byte for byte
    as OCaml reads it, then as UTF-8. ValueError on an escape OCaml does not write."""
    out, pos = bytearray(), 0
    for m in _OCAML_ESCAPE.finditer(literal):
        out += literal[pos:m.start()].encode()
        if m["dec"]:
            out.append(int(m["dec"]))
        elif m["hex"]:
            out.append(int(m["hex"], 16))
        elif m["oct"]:
            out.append(int(m["oct"], 8))
        else:
            out += _OCAML_CHARS[m["char"]].encode()
        pos = m.end()
    tail = literal[pos:]
    if "\\" in tail:
        raise ValueError(f"not an OCaml string literal: {literal!r}")
    out += tail.encode()
    return out.decode("utf-8", errors="replace")


def rank_site_line(line):
    """(kind, fields, bare words) of one `rank-site:` line, or None for another line: `fields` maps
    each `key=` to its value (a literal decoded by `ocaml_string`), `bare` keeps the words that are
    no `key=value` (`rank-site: pick site=.. no report`). ValueError for one that does not
    tokenize."""
    head = _RANK_SITE_HEAD.match(line)
    if not head:
        return None
    fields, bare, pos, rest = {}, [], 0, line[head.end():]
    while rest[pos:].strip():
        m = _RANK_SITE_TOKEN.match(rest, pos)
        if not m:
            raise ValueError(f"unusable rank-site line: {line}")
        if m["key"] is not None:
            if m["key"] in fields:
                raise ValueError(f"rank-site line naming {m['key']} twice: {line}")
            fields[m["key"]] = ocaml_string(m["lit"]) if m["lit"] is not None else m["raw"]
        else:
            bare.append(m["bare"])
        pos = m.end()
    return head["kind"], fields, bare


def rank_site_lines(text, kind=None):
    """The `rank-site:` lines of a bench_rank_site stdout, in order, as `rank_site_line` reads them
    -- only those of `kind` when it is given."""
    out = []
    for line in text.splitlines():
        r = rank_site_line(line)
        if r is not None and (kind is None or r[0] == kind):
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------
# The launch-geometry trace (config `schedule_log_launches`, Backends' compile), one line per
# segment of a compile: `schedule: <routine> seg <i>/<n> grid=[x;y;z] block=[x;y;z] stmts=<s>`.

SCHEDULE_LAUNCH_PREFIX = "schedule: "
SCHEDULE_LAUNCH = re.compile(r"^schedule: (?P<routine>\S+) seg (?P<seg>\d+)/(?P<segs>\d+) "
                             r"grid=\[(?P<grid>\d+;\d+;\d+)\] block=\[(?P<block>\d+;\d+;\d+)\] "
                             r"stmts=(?P<stmts>\d+)$")
_SCHEDULE_LAUNCH_HEAD = re.compile(r"^schedule: \S+ seg \d")


def schedule_launches(text):
    """The launch-geometry lines, in order: dicts of `routine`, `seg`, `segs`, `grid` and `block`
    (3-tuples of ints) and `stmts`. ValueError for a line that opens like one and does not read as
    one."""
    out = []
    for line in text.splitlines():
        if not _SCHEDULE_LAUNCH_HEAD.match(line):
            continue
        m = SCHEDULE_LAUNCH.match(line)
        if not m:
            raise ValueError(f"unusable launch line: {line}")
        out.append({"routine": m["routine"], "seg": int(m["seg"]), "segs": int(m["segs"]),
                    "grid": tuple(int(x) for x in m["grid"].split(";")),
                    "block": tuple(int(x) for x in m["block"].split(";")),
                    "stmts": int(m["stmts"])})
    return out


# ---------------------------------------------------------------------------------------------
# The seeded-default census the untuned compile logs under config `autotune_log`
# (Autotune.seeded_apply, gh-ocannl-1253; CUDA and HIP only): `autotune: seeded_default: plan derived
# in <t> s` and `autotune: seeded_default: <k> of <n> segments seeded (...): <segments>`.

_SEEDED_DERIVED = re.compile(r"seeded_default: plan derived in ([0-9.]+ s)")
_SEEDED_CENSUS = re.compile(r"seeded_default: \d+ of \d+ segments seeded.*")


def seeded_default(text):
    """(the plan-derivation times as printed, `<t> s`; the census lines from `seeded_default:` on),
    each in order."""
    return _SEEDED_DERIVED.findall(text), _SEEDED_CENSUS.findall(text)


# ---------------------------------------------------------------------------------------------
# The registry of text-line formats (gh-ocannl-1767). test/operations/bench_line_readers reads it
# twice: every entry's producer must still print its `producer_token`, and every other file under
# benchmarks/ and tools/ whose reading code names an `anchor` is refused -- that line's reader
# belongs here. `pinned` is how the test runs the readers over a line a producer printed; an entry
# whose producer has no cheap run on a CPU says why it has none instead.

FORMATS = {
    "gebp_timing": {"readers": ("timing_lines", "timing_line"), "anchor": r"GFLOP/s",
                    "producers": {"bin/schedule_bench.ml": "GFLOP/s  (spot [%d]",
                                  "bin/narrow_gebp_bench.ml": "GFLOP/s  (spot check [%d]"},
                    "pinned": True},
    "gebp_header": {"readers": ("gebp_header",), "anchor": r"""['"]GEBP """,
                    "producers": {"bin/narrow_gebp_bench.ml": '"GEBP n=%d, '}, "pinned": True},
    "kernel_row": {"readers": ("kernel_rows",), "anchor": r"bench: kernel",
                   "producers": {"benchmarks/runners/ocannl/bench_harness.ml":
                                 '"bench: kernel %d/%d %s %s'}, "pinned": True},
    "kernel_table_legend": {"readers": ("kernel_table_timing",),
                            "anchor": r"bench: per-kernel table \\?\(",
                            "producers": {"benchmarks/runners/ocannl/bench_harness.ml":
                                          '"bench: per-kernel table (%s; each kernel alone)'},
                            "pinned": True},
    "timing_trace": {"readers": ("compile_traces", "trace_summary", "trace_events"),
                     "anchor": r"timing-trace:",
                     "producers": {"benchmarks/runners/ocannl/bench_harness.ml": '"timing-trace: '},
                     "pinned": True},
    "rank_site": {"readers": ("rank_site_lines", "rank_site_line"), "anchor": r"rank-site:",
                  "producers": {"benchmarks/runners/ocannl/bench_rank_site.ml": '"rank-site: '},
                  "pinned": True},
    "schedule_launch": {"readers": ("schedule_launches",), "anchor": r"seg \S+/|grid=\(?\\\[",
                        "producers": {"arrayjit/lib/backends.ml": '"schedule: %s seg %d/%d grid=['},
                        "pinned": True},
    "seeded_default": {"readers": ("seeded_default",), "anchor": r"seeded_default:",
                       "producers": {"arrayjit/lib/autotune.ml": '"seeded_default: '},
                       "pinned": "Autotune.seeded_apply logs it on CUDA and HIP only "
                                 "(seeded_default_backend); no CPU run prints it"},
}


# ---------------------------------------------------------------------------------------------
# The command line, for the shell drivers. Each command reads FILE through the readers above and
# exits 1 on a refusal (the reason on stderr), 2 on a usage error.

def _launches_of(text, routine):
    return [s for s in schedule_launches(text) if s["routine"] == routine]


def _cmd_trace_has(text, what):
    """Exit status 0 when the trace has `what`: `summary`, `timing-calls` (a summary recording at
    least one timing call), `incomplete` (a summary naming raised timing calls), `attempt` or
    `search-done`; 3 when it has not."""
    summary = trace_summary(text)
    verdicts = {"summary": lambda: summary is not None,
                "timing-calls": lambda: summary is not None and summary["calls"] > 0,
                "incomplete": lambda: summary is not None and summary["incomplete"],
                "attempt": lambda: bool(trace_events(text, "attempt")),
                "search-done": lambda: bool(trace_events(text, "search done"))}
    if what not in verdicts:
        raise _Usage(f"trace-has: no predicate {what} (one of {', '.join(verdicts)})")
    return 0 if verdicts[what]() else 3


def _cmd_trace_digest(text):
    """The trace's attempt count, then its last two call/summary lines (a session's display)."""
    print(f"candidate attempts {len(trace_events(text, 'attempt'))}")
    tail = [line for line in text.splitlines()
            if line.startswith((TRACE_EVENTS["call"], TRACE_PREFIX + "summary: "))]
    for line in tail[-2:]:
        print(line)
    return 0


def _cmd_rank_site_field(text, kind, key):
    """The value of `key` on the LAST `kind` line that has it; nothing when none has it."""
    values = [f[key] for _, f, _ in rank_site_lines(text, kind) if key in f]
    if values:
        print(values[-1])
    return 0


def _cmd_schedule_segs(text, routine):
    """The segment count of `routine`'s first FISSIONED compile (more than one segment), or exit 3
    when it has none."""
    n = next((s["segs"] for s in _launches_of(text, routine) if s["seg"] == 0 and s["segs"] > 1),
             None)
    if n is None:
        return 3
    print(n)
    return 0


def _cmd_schedule_geometry(text, routine, segs):
    """`seg <i> grid=[..] block=[..]` for every launch line of `routine` at `segs` segments."""
    dims = lambda d: "[" + ";".join(map(str, d)) + "]"  # noqa: E731
    for s in _launches_of(text, routine):
        if s["segs"] == int(segs):
            print(f"seg {s['seg']} grid={dims(s['grid'])} block={dims(s['block'])}")
    return 0


class _Usage(Exception):
    pass


COMMANDS = {"trace-has": (_cmd_trace_has, 1), "trace-digest": (_cmd_trace_digest, 0),
            "rank-site-field": (_cmd_rank_site_field, 2),
            "schedule-segs": (_cmd_schedule_segs, 1),
            "schedule-geometry": (_cmd_schedule_geometry, 2)}


def main(argv):
    usage = ("usage: bench_output.py trace-has FILE PREDICATE | trace-digest FILE | rank-site-field "
             "FILE KIND KEY | schedule-segs FILE ROUTINE | schedule-geometry FILE ROUTINE SEGS")
    if len(argv) < 2 or argv[0] not in COMMANDS or len(argv) != 2 + COMMANDS[argv[0]][1]:
        print(usage, file=sys.stderr)
        return 2
    cmd, path, *args = argv
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return COMMANDS[cmd][0](text, *args)
    except _Usage as e:
        print(f"bench_output: {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"bench_output: {path}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
