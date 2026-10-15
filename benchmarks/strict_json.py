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
"""
import json
import math


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
