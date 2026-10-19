"""How every report-writing script in benchmarks/ shows a string it did not write itself (stdlib only).

A report table or run-log line that interpolates a fixture's file name, a workload a runner named, a
cache path from the environment, a treatment list from the command line or text a runner printed
can be broken by that string: a raw `|` adds a column to a Markdown table row, a raw line break
ends the row or the log line, a backtick opens a code span the report did not mean. The cure has two
halves, and both live here so that every script has them the same way (gh-ocannl-1355):

- `display_value` shows such a string safely: plain values as they are, anything else JSON-quoted
  with `|` and the backtick escaped too. `code_span` is a code span around it.
- `table_row`/`table_lines` build every Markdown table row and `log_line`/`log_lines` every
  run-log line that carries such a string. They REFUSE, at render time, a cell holding a raw `|` or
  line break (`UnsafeTableRow`) and a log line holding a raw line break (`UnsafeLogLine`): a value
  that reached them without `display_value` fails loudly instead of shifting every column after it
  or forging a line.

test_orchestrate's `TableRowTest` scans every shipped script, the shell drivers' inline Python
included, for a table row written around `table_row`.
"""
import json
import re

_DISPLAY_PLAIN_RE = re.compile(r"[A-Za-z0-9_.,:/=+@%~-]*")


def display_value(value):
    """An operator-, environment- or runner-supplied value as a label, a report cell, a failure note
    or a log line shows it.

    Every human-facing rendering of a string the operator or the environment supplied goes through
    this (or through `json.dumps`, as orchestrate.py's report header lines do): a `--cell-flag`
    value, a `--skip-cell` workload, a fixture's name and the workload a result row or checkpoint
    names after it, a cache path from the environment, an exception's text, the value a regime knob
    resolved to, a driver's treatment or label argument. Only the records keep the raw value. A
    label built from such a string (`<workload> ocannl/...`) is a display string from the start, so
    a report and a run log print labels as they are. Origins need nothing:
    `fixture_digest.check_origin` holds them to a portable alphabet. Text a runner or the host
    printed and the script's own pattern matched -- a census, a kernel's written nodes, a CPU name
    -- is shown raw: the builders' refusal is its guard, and quoting it would only make every
    report noisier.

    Plain values (`16`, `/tmp/cache`) are shown as they are. Anything else is JSON-quoted with `|`
    and the backtick escaped as well, so a value cannot add a column to a report table, a line to
    the report or the run log, or open a code span -- `debug_highlight_pcre=a|b` is a legitimate
    setting -- while the shown text still decodes to the exact value. The raw value is what the
    command line and the JSON records carry.
    """
    if _DISPLAY_PLAIN_RE.fullmatch(value):
        return value
    return json.dumps(value).replace("|", "\\u007c").replace("`", "\\u0060")


def code_span(value):
    """`value` as a Markdown code span: `display_value` escapes the backtick, so the span cannot be
    closed early."""
    return f"`{display_value(value)}`"


# What ends a line early: every character `str.splitlines` breaks a line at (a renderer ends a table
# row there, and so does any reader that splits a report or a run log into lines). `display_value`
# escapes all of them, so a value shown through it never trips this.
_LINE_BREAKERS = "\n\r\v\f\x1c\x1d\x1e\x85  "
_LINE_BREAKERS_RE = re.compile(f"[{_LINE_BREAKERS}]")
# What ends a Markdown table cell early: the column separator, and every line break.
_TABLE_CELL_BREAKERS_RE = re.compile(f"[|{_LINE_BREAKERS}]")


class UnsafeTableRow(ValueError):
    """A report table row whose cell would add a column or end the row (a raw `|` or line break),
    or whose cell count is not its table's column count.

    Raised by `table_row` and `table_lines` when the row is rendered, so a string that reached a
    cell without going through `display_value` fails the report loudly instead of shifting every
    column after it.
    """


class UnsafeLogLine(ValueError):
    """A run-log line that would be more than one line: it holds a raw line break.

    Raised by `log_line` and `log_lines`, so a string that reached a log line without going through
    `display_value` fails loudly instead of forging a line a reader of the log takes as the
    driver's own (a `PARITY GATE: all cells passed` planted by a fixture's name, say).
    """


def table_row(*cells):
    """One row of a report table -- its header and rule included -- from its cell texts.

    Every Markdown table a report writes is built here and only here, so display safety is checked
    where the row is rendered rather than remembered at each interpolation (post-wave follow-up of
    gh-ocannl-1273): a cell holding a raw `|` or line break raises `UnsafeTableRow`, naming the
    cell. Show an operator- or environment-supplied string through `display_value` first; that is
    the fix the refusal asks for, never escaping at the call site.
    """
    for column, cell in enumerate(cells):
        if not isinstance(cell, str):
            raise TypeError(f"table cell {column} is {type(cell).__name__}, not str: {cell!r}")
        breaker = _TABLE_CELL_BREAKERS_RE.search(cell)
        if breaker:
            raise UnsafeTableRow(
                f"table cell {column} holds a raw {breaker.group()!r}, which would "
                f"{'add a column' if breaker.group() == '|' else 'end the row'}: {cell!r} "
                "(show the value through display_value)"
            )
    return "|" + "|".join(f" {cell} " if cell else " " for cell in cells) + "|"


_RULES = {"l": "---", "r": "---:"}


def table_lines(columns, rows, align=None):
    """A report table as its lines: the header row naming `columns`, the rule, then `rows`, each a
    sequence of cell texts as long as `columns` (a short or long row is refused as a malformed one
    would be read: every column after the gap shifts).

    `align`, when given, is one letter per column: `l` for the default rule `---`, `r` for a
    right-aligned `---:` (a numeric column)."""
    align = align or "l" * len(columns)
    if len(align) != len(columns) or set(align) - set(_RULES):
        raise UnsafeTableRow(f"alignment {align!r} for {len(columns)} columns (one of l/r each)")
    lines = [table_row(*columns), "|" + "".join(_RULES[a] + "|" for a in align)]
    for row in rows:
        row = tuple(row)
        if len(row) != len(columns):
            raise UnsafeTableRow(
                f"table row has {len(row)} cells for {len(columns)} columns: {row!r}"
            )
        lines.append(table_row(*row))
    return lines


def log_line(text):
    """`text` as one line of a run log (or of a report outside its tables), refused with
    `UnsafeLogLine` when it holds a raw line break.

    A line naming an operator- or runner-supplied value -- a gate's cell labels, a failure note, a
    fixture's name -- is built here, so a value that skipped `display_value` cannot add a line.
    """
    if not isinstance(text, str):
        raise TypeError(f"log line is {type(text).__name__}, not str: {text!r}")
    breaker = _LINE_BREAKERS_RE.search(text)
    if breaker:
        raise UnsafeLogLine(
            f"log line holds a raw {breaker.group()!r}, which would add a line: {text!r} "
            "(show the value through display_value)"
        )
    return text


def log_lines(lines):
    """Several log lines, each checked by `log_line`, joined into one text (no trailing newline):
    a message that is deliberately more than one line says where its lines are."""
    return "\n".join(log_line(line) for line in lines)
