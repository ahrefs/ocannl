"""JSON as every reader of a benchmark runner's result line has it (stdlib only).

`json.loads` extends JSON in two ways that matter to a result line. It accepts the constants `NaN`,
`Infinity` and `-Infinity`, and it reads an overflowing literal such as `1e999` as infinity. The
runners emit `null` for a number JSON cannot express (gh-ocannl-676: `Bench_json.num`,
`bench_common.json_safe`), so a result line that carries either spelling comes from an emitter that
regressed, and every consumer other than Python -- jq, pandas, the next session's scripts -- refuses
it or reads it differently. `loads` refuses it too, so that a Python reader cannot accept a
non-finite number that no other reader would.

Every reader of a runner's result or checkpoint line goes through `loads`, the inline Python of the
shell drivers included (they put this directory on sys.path from STRICT_JSON_DIR). JSON this
tooling writes and reads back itself (preflight records, matrices, results.jsonl, fixture headers,
workload specs) is out of scope; test_orchestrate's ResultLineReaderInventoryTest holds each such
`json.loads` call with its reason and fails on any other.

A shell driver that needs ONE field of a cell's result line asks for it by name instead of
grepping the line's text (gh-ocannl-1355):

  python3 benchmarks/strict_json.py field CELL.out step_ms.p50

prints the field of the cell's last result line (`bench_output.result_record`, `field`): a string as it is,
anything else as JSON. It exits 1, naming the problem on stderr, when the file holds no result
line, the line is not strict JSON, or the field is missing -- so `step_ms` being a
`{p10,p50,p90}` object, or a nested `tune.shipped`, is the reader's business, not a pattern's.
test_orchestrate's ShellFieldGrepInventoryTest fails on a driver line that greps a quoted JSON key.
"""
import json
import math
import sys


class NonFinite(ValueError):
    """The text is JSON-shaped but holds a number JSON cannot express: one of Python's `NaN` /
    `Infinity` / `-Infinity` constants, or a literal that overflows a float."""


def _reject_constant(name):
    raise NonFinite(f"non-JSON constant {name}")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise NonFinite(f"number {text} overflows to {value}")
    return value


def loads(text):
    """`json.loads(text)`, refusing any non-finite number with `NonFinite`.

    `NonFinite` is a `ValueError`, as `json.JSONDecodeError` is, so a reader that already treats an
    unparseable line as no result treats this one the same way; a reader that skips noise lines
    can catch `NonFinite` first to name the problem instead.
    """
    return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)


def field(record, path):
    """`record`'s value at the dotted `path` (`step_ms.p50`, `tune.shipped`), refusing with
    `KeyError` -- naming the path so far -- where a key is missing or its parent is not an
    object."""
    value, walked = record, []
    for key in path.split("."):
        walked.append(key)
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"the result line has no {'.'.join(walked)}")
        value = value[key]
    return value


def _main(argv):
    if len(argv) != 3 or argv[0] != "field":
        print("usage: strict_json.py field FILE KEY[.KEY...]", file=sys.stderr)
        return 2
    _, path, key = argv
    import bench_output  # the one reader of result lines (gh-ocannl-1443); it imports this module
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            record = bench_output.result_record(f.read())
        if record is None:
            raise ValueError("no result line")
        value = field(record, key)
    except (OSError, ValueError, KeyError) as e:
        print(f"strict_json: {path}: {e.args[0] if isinstance(e, KeyError) else e}", file=sys.stderr)
        return 1
    print(value if isinstance(value, str) else json.dumps(value))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
