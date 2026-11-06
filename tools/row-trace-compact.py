#!/usr/bin/env python3
r"""Digest a flushing row-solver trace: one line per distinct `solve_row_ineq`, plus template mints.

Usage: tools/row-trace-compact.py TRACE

TRACE is the flat text ppx_minidebug's flushing backend writes when `row.ml` is built with
OCANNL_LOG_LEVEL_ROW=9 and run with debug_backend=flushing and log_level=9; the recipe, and where
the text lands, are in .claude/skills/ocannl-debug-tracing/SKILL.md. Both forms read: stdout's
prefixes every line with the default log_file_stem `debug`, a log file's only the scope edges. Such
a trace runs to hundreds of megabytes, so it is read as a stream.

Each distinct (stage, res, opnd) prints once, at its first occurrence, as

  LINE STAGE RES >= OPND

LINE being the 1-based trace line of the `solve_row_ineq` scope, to go back to. A row reads
`BEG,..V..,DIMS` as in the solver, V the id of the row variable, or `BEG,<bc>,DIMS` for a closed
(Broadcastable) row, with one comma-separated element per axis:

  3          a solved dim of the default basis
  rgb=3      a solved dim of another basis (`bcast_if_1=1` is the broadcast top, not a fixed 1)
  v12        a dim variable
  2*v12+1    an Affine dim, in `Row.dim_to_string`'s notation, as is conv(v12+v13) with a kernel;
             conv_pad(v12+v13) with padding, which that notation leaves out and the solver does not
  (a^b)      a Concat dim

Any other dim prints as its raw sexp, still one element, so that axes are never miscounted. Dropped:
`prov`, `proj_id` and the names of dim variables. A row variable or a dim variable greps back as
`Row_var V` / `(id V)` in the trace.

Each template row variable prints once, indented, as `LINE templ V`, where the cache first hands
it out: the first sighting is the mint, as every later one is a cache hit.

A record that does not parse prints as `LINE STAGE <unparsed>` where it would have stood. The
counts go to stderr. Exit status: 0 for a whole digest; 1 when a record did not parse or the trace
holds no `solve_row_ineq` at all (a build without the gate looks just like a solver with nothing
to do), or its reader left; 2 for a usage error.
"""

import argparse
import os
import re
import sys

# Value lines carry the `debug ` prefix only in stdout's form; scope edges carry it in both.
ARGUMENT = re.compile(r"\s*(?:debug )?(\w+) = (.*)$")
INEQ_BEGIN = re.compile(r"\s*debug solve_row_ineq begin\b")
# Any scope edge ends the argument list: the arguments are the scope's first log entries.
SCOPE_EDGE = re.compile(r"\s*debug \w+ begin\b|.* - debug \w+ end\s*$")
TEMPLATE = re.compile(r"\s*(?:debug )?templ_v = \(Row_var (\d+)\)")
TOKEN = re.compile(r'[()]|"(?:[^"\\]|\\.)*"|[^\s()"]+')


def parse_sexp(text):
    """One sexp as nested lists of atoms; ValueError unless `text` is exactly one, balanced."""
    stack = [[]]
    for token in TOKEN.findall(text):
        if token == "(":
            stack.append([])
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token)
    if len(stack) != 1 or len(stack[0]) != 1:
        raise ValueError("not one sexp")
    return stack[0][0]


def fields(pairs):
    """A record's sexp as a dict."""
    return {name: value for name, value in pairs}


def raw(sexp):
    return sexp if isinstance(sexp, str) else "(" + " ".join(map(raw, sexp)) + ")"


def render_dim(sexp):
    """One axis, as `Row.dim_to_string` spells it where the two share a form."""
    kind = sexp[0]
    if kind == "Var":
        return "v" + fields(sexp[1])["id"]
    if kind == "Dim":
        solved = fields(sexp[1])
        basis = solved["basis"]
        return solved["d"] if basis == "default" else f"{basis}={solved['d']}"
    if kind == "Concat":
        return "(" + "^".join(map(render_dim, sexp[1])) + ")"
    if kind == "Affine":
        affine = fields(sexp[1:])
        text = render_dim(affine["over"])
        if affine["stride"] != "1":
            text = f"{affine['stride']}*{text}"
        if affine["stride_offset"] != "0":
            text = f"{text}+{affine['stride_offset']}"
        if affine["conv"]:
            conv = fields(affine["conv"][0])
            kernel = render_dim(conv["kernel"])
            if conv["dilation"] != "1":
                kernel = f"{conv['dilation']}*{kernel}"
            form = "conv_pad" if conv["use_padding"] == "true" else "conv"
            text = f"{form}({text}+{kernel})"
        return text
    return raw(sexp)


def render_row(text):
    row = fields(parse_sexp(text))
    bcast = row["bcast"]
    if bcast == "Broadcastable":
        middle = "<bc>"
    else:
        _, (_, var) = bcast
        middle = f"..{var}.."
    return ",".join([*map(render_dim, row["beg_dims"]), middle, *map(render_dim, row["dims"])])


def events(trace):
    """Yields (LINE, "ineq", ARGUMENTS) and (LINE, "templ", V) in trace order. ARGUMENTS maps each
    logged argument to its text, which may span lines, and is yielded at the scope's next edge or
    at the end of the trace: a crash can end a trace right inside the call."""
    begun, arguments, name = 0, None, None
    for number, line in enumerate(trace, 1):
        if arguments is not None:
            if not SCOPE_EDGE.match(line):
                logged = ARGUMENT.match(line)
                if logged:
                    name = logged[1]
                    arguments[name] = logged[2]
                elif name is not None:
                    arguments[name] += line
                continue
            yield begun, "ineq", arguments
            arguments = None
        if INEQ_BEGIN.match(line):
            begun, arguments, name = number, {}, None
            continue
        template = TEMPLATE.match(line)
        if template:
            yield number, "templ", template[1]
    if arguments is not None:
        yield begun, "ineq", arguments


def digest(trace):
    seen, minted = set(), set()
    calls = unparsed = 0
    for line, kind, value in events(trace):
        if kind == "templ":
            if value not in minted:
                minted.add(value)
                print(f"    {line} templ {value}")
            continue
        calls += 1
        stage = value.get("stage", "?").strip()
        try:
            res, opnd = render_row(value["res"]), render_row(value["opnd"])
        except (IndexError, KeyError, TypeError, ValueError):
            unparsed += 1
            print(f"{line} {stage} <unparsed>")
            continue
        key = (stage, res, opnd)
        if key not in seen:
            seen.add(key)
            print(f"{line} {stage} {res} >= {opnd}")
    return calls, len(seen), len(minted), unparsed


def main(argv):
    parser = argparse.ArgumentParser(
        prog="tools/row-trace-compact.py",
        usage=argparse.SUPPRESS,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace", metavar="TRACE")
    trace_path = parser.parse_args(argv).trace
    try:
        trace = open(trace_path, encoding="utf-8", errors="surrogateescape")
    except OSError as e:
        print(f"row-trace-compact: cannot read {trace_path}: {e.strerror}", file=sys.stderr)
        return 2
    with trace:
        calls, distinct, mints, unparsed = digest(trace)
    # Here, not at interpreter exit: that is where the last block of a short digest would meet a
    # reader that has left, past any handler.
    sys.stdout.flush()
    print(
        f"row-trace-compact: {calls} solve_row_ineq, {distinct} distinct, {mints} template mints, "
        f"{unparsed} unparsed",
        file=sys.stderr,
    )
    if calls == 0:
        print(
            f"row-trace-compact: no solve_row_ineq in {trace_path}: is it a flushing trace of a "
            "build with OCANNL_LOG_LEVEL_ROW=9?",
            file=sys.stderr,
        )
    return 1 if unparsed or calls == 0 else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="surrogateescape")
    try:
        sys.exit(main(sys.argv[1:]))
    except BrokenPipeError:
        # A reader such as `head` has what it wanted; the buffered rest must not retry at exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
