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
  `compile_traces`.

Every JSON read is `strict_json`'s: a non-finite number is refused (`strict_json.NonFinite`, a
ValueError), never read as a float, since a runner writes `null` there (gh-ocannl-676).
"""

import math
import re

import strict_json
from strict_json import NonFinite, field  # noqa: F401  (re-exported: the result line's readers)

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
