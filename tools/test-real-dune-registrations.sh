#!/usr/bin/env bash
# Check that every manifest harness leg calling `harness_real_dune` is
# registered in BOTH `real_dune_legs` (tools/test-harnesses.sh) and
# `tools/harness-skips-ci-shell-dune.txt` (gh-ocannl-2486).
#
# Usage: tools/test-real-dune-registrations.sh [--keep|--help|--list-legs|--only REGEX]
#
# Leg 1 reads every manifest harness (plus any local sourced `.sh` helper
# files); Leg 2 runs the discovered real-dune legs under NO_DUNE=1.
# affected-reads: scripts/test-*.sh tools/test-*.sh benchmarks/test-*.sh test/operations/apt_fallback.sh test/operations/ci_matrix.sh --only '^1:'
# affected-reads: tools/test-test-run-options.sh tools/test-test-run-digest.sh tools/test-test-run-bisect.sh tools/test-merge-build-check.sh --only '^2:'
#
# Boundary (fail-closed):
#   - Manifest membership comes from `tools/test-harnesses.sh --list` and
#     `--shell --list`; `real_dune_legs` is read from `tools/test-harnesses.sh`;
#     hidden-dune skips come from `tools/harness-skips-ci-shell-dune.txt` under
#     the three-line-form allowlist of `tools/test-harnesses.sh`.
#   - Static reader (Leg 1): scans every manifest harness (and same-directory
#     `.sh` helpers sourced via `.` / `source`, excluding
#     `scripts/harness-support.sh`) for live command-position calls to
#     `harness_real_dune`, including inside `$(...)` command substitutions and
#     after prefix keywords/builtins (`!`, `if`, `then`, `elif`, `else`, `do`,
#     `while`, `until`, `time`, `command`, `builtin`, `exec`) and leading
#     `VAR=val` assignments. Comments, single-quoted strings, `$'...'`,
#     double-quoted text outside `$(...)`, and quoted heredoc bodies (`<<'EOF'`,
#     `<<"EOF"`, `<<\EOF`) are treated as data and not scanned as live calls;
#     unquoted heredoc bodies (`<<EOF`) are scanned for live `$(...)` and
#     backtick command substitutions. Live backticks containing
#     `harness_real_dune` are refused as unattributed.
#   - Recognised leg attribution forms (fail-closed):
#       1. `harness_leg_run <LITERAL_NAME> <FN>` or
#          `harness_leg_run_errexit <LITERAL_NAME> <FN>` where `<FN>` calls
#          `harness_real_dune` directly or through same-file/sourced helper
#          functions.
#       2. Inline `if harness_leg <LITERAL_NAME>; then ... fi` blocks calling
#          `harness_real_dune` directly or through same-file/sourced helper
#          functions.
#     Any live `harness_real_dune` call outside those forms (ungated top-level
#     code, uncalled function, non-literal leg/function registration beside a
#     live call, or inside a `toolchain`/`promotion` harness) is refused.
#     Indirect invocation (`eval`, variable command names `$fn`) is outside
#     the static reader's scope.
#   - Runtime hidden-dune skip check (Leg 2): runs each statically discovered
#     `harness_real_dune` leg individually with an exact anchored `--only`
#     selector under `OCANNL_TOOL_HARNESS_NO_DUNE=1` (with ambient
#     strict/skip/record/fleet variables cleared so nested runs under
#     `--ci-shell` stay isolated) and verifies that each leg emits at least one
#     hidden-dune skip of its own and that the union of emitted `(HARNESS,
#     LABEL)` pairs equals `tools/harness-skips-ci-shell-dune.txt`.
set -u
export LC_ALL=C

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/.." && pwd)
. "$root/scripts/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch test-real-dune-registrations

checker=$TMP/check_real_dune.py
cat >"$checker" <<'PY_CHECKER'
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BASH_BIN = os.environ.get("BASH") or "bash"

CLEAN_ENV_DROP = {
    "OCANNL_TOOL_HARNESS_STRICT",
    "OCANNL_TOOL_HARNESS_EXPECTED_SKIPS",
    "OCANNL_TOOL_HARNESS_SKIP_RECORD",
    "OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD",
    "OCANNL_TOOL_HARNESS_SKIP_LEGS",
    "OCANNL_TOOL_HARNESS_NAME",
    "OCANNL_TOOL_HARNESS_NO_DUNE",
    "OCANNL_TOOL_HARNESS_SUBJECT",
    "OCANNL_TOOL_HARNESS_TIMINGS_FILE",
    "OCANNL_TOOL_HARNESS_LEG_TIMES",
    "OCANNL_TOOL_HARNESSES_IN_SLOT",
}

CMD_PREFIX_WORDS = {
    "!",
    "if",
    "then",
    "elif",
    "else",
    "do",
    "while",
    "until",
    "time",
    "command",
    "builtin",
    "exec",
}

CMD_SEP_TOKENS = {";", ";;", ";&", ";;&", "&", "&&", "|", "||", "\n"}

ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\+?)=.*$", re.S)
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SOURCED_SH_RE = re.compile(r"([A-Za-z0-9_.-]+\.sh)$")


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in CLEAN_ENV_DROP}
    env["OCANNL_TOOL_FLEET_WORKER"] = "none"
    env["LC_ALL"] = "C"
    if extra:
        env.update(extra)
    return env


def ere_escape(text):
    return re.sub(r"([.\\^$*+?()\[\]{}|])", r"\\\1", text)


class Token:
    __slots__ = ("kind", "raw", "literal", "line")

    def __init__(self, kind, raw, literal=None, line=1):
        self.kind = kind  # 'WORD', 'OP', 'NL'
        self.raw = raw
        self.literal = literal  # str if purely literal word, else None
        self.line = line


def tokenize_bash(text, start_line=1):
    """Fail-closed Bash lexer for static harness_real_dune discovery.

    Returns (tokens, subst_spans, backtick_hits) where:
      - tokens is a list of Token
      - subst_spans is a list of (line, code_str) extracted from $(...)
      - backtick_hits is a list of (line, raw_str) where live backticks mention
        harness_real_dune
    """
    tokens = []
    subst_spans = []
    backtick_hits = []
    n = len(text)
    i = 0
    line = start_line
    pending_heredocs = []  # list of (delim, strip_tabs, was_quoted)

    def scan_unquoted_heredoc(body, body_start_ln):
        """Scan an unquoted heredoc body for live $(...) and backtick substitutions."""
        m = len(body)
        k = 0
        ln = body_start_ln
        while k < m:
            c = body[k]
            if c == "\\" and k + 1 < m:
                if body[k + 1] == "\n":
                    ln += 1
                k += 2
                continue
            if c == "$" and k + 1 < m and body[k + 1] == "(":
                if k + 2 < m and body[k + 2] == "(":
                    k += 3
                    ad = 2
                    while k < m and ad > 0:
                        if body[k] == "(":
                            ad += 1
                        elif body[k] == ")":
                            ad -= 1
                        elif body[k] == "\n":
                            ln += 1
                        k += 1
                    continue
                s_ln = ln
                end_k, end_ln, inner = scan_cmd_subst_in(body, k + 2, ln)
                subst_spans.append((s_ln, inner))
                k = end_k
                ln = end_ln
                continue
            if c == "`":
                bt_ln = ln
                k += 1
                bt_start = k
                while k < m and body[k] != "`":
                    if body[k] == "\\" and k + 1 < m:
                        if body[k + 1] == "\n":
                            ln += 1
                        k += 2
                    else:
                        if body[k] == "\n":
                            ln += 1
                        k += 1
                bt_body = body[bt_start:k]
                if "harness_real_dune" in bt_body:
                    backtick_hits.append((bt_ln, bt_body))
                if k < m:
                    k += 1
                continue
            if c == "\n":
                ln += 1
            k += 1

    def consume_heredocs():
        nonlocal i, line, pending_heredocs
        while pending_heredocs and i < n:
            delim, strip_tabs, was_quoted = pending_heredocs.pop(0)
            body_start_ln = line
            body_lines = []
            while i < n:
                nl = text.find("\n", i)
                if nl == -1:
                    seg = text[i:]
                    i = n
                else:
                    seg = text[i:nl]
                    i = nl + 1
                    line += 1
                if seg.endswith("\r"):
                    seg = seg[:-1]
                cmp_seg = seg.lstrip("\t") if strip_tabs else seg
                if cmp_seg == delim:
                    break
                if not was_quoted:
                    body_lines.append(seg)
            if not was_quoted and body_lines:
                scan_unquoted_heredoc("\n".join(body_lines), body_start_ln)

    def scan_cmd_subst_in(buf, start_idx, start_ln):
        """Scan inside $( ... ) starting right after '$('. Returns (end_idx, end_ln, inner_text)."""
        m = len(buf)
        j = start_idx
        ln = start_ln
        depth = 1
        in_sq = False
        in_dq = False
        while j < m and depth > 0:
            ch = buf[j]
            if in_sq:
                if ch == "'":
                    in_sq = False
                elif ch == "\n":
                    ln += 1
                j += 1
                continue
            if in_dq:
                if ch == "\\" and j + 1 < m:
                    if buf[j + 1] == "\n":
                        ln += 1
                    j += 2
                    continue
                if ch == '"':
                    in_dq = False
                    j += 1
                    continue
                if ch == "$" and j + 1 < m and buf[j + 1] == "(":
                    if j + 2 < m and buf[j + 2] == "(":
                        j += 3
                        ad = 2
                        while j < m and ad > 0:
                            if buf[j] == "(":
                                ad += 1
                            elif buf[j] == ")":
                                ad -= 1
                            elif buf[j] == "\n":
                                ln += 1
                            j += 1
                        continue
                    end_j, end_ln, _ = scan_cmd_subst_in(buf, j + 2, ln)
                    j = end_j
                    ln = end_ln
                    continue
                if ch == "\n":
                    ln += 1
                j += 1
                continue
            # Unquoted inside $(...)
            if ch == "\\" and j + 1 < m:
                if buf[j + 1] == "\n":
                    ln += 1
                j += 2
                continue
            if ch == "'":
                in_sq = True
                j += 1
                continue
            if ch == '"':
                in_dq = True
                j += 1
                continue
            if ch == "#" and (j == start_idx or buf[j - 1] in " \t\n;|&("):
                nl = buf.find("\n", j)
                if nl == -1:
                    j = m
                else:
                    j = nl + 1
                    ln += 1
                continue
            if ch == "(":
                depth += 1
                j += 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    return j + 1, ln, buf[start_idx:j]
                j += 1
                continue
            if ch == "\n":
                ln += 1
            j += 1
        return j, ln, buf[start_idx:j]

    def scan_cmd_subst(start_idx, start_ln):
        return scan_cmd_subst_in(text, start_idx, start_ln)

    def read_heredoc_delim():
        nonlocal i, line
        while i < n and text[i] in " \t":
            i += 1
        chars = []
        was_quoted = False
        while i < n:
            ch = text[i]
            if ch in " \t\n\r;&|()<>":
                break
            if ch == "\\" and i + 1 < n:
                if text[i + 1] == "\n":
                    i += 2
                    line += 1
                    continue
                was_quoted = True
                chars.append(text[i + 1])
                i += 2
                continue
            if ch == "'":
                was_quoted = True
                i += 1
                while i < n and text[i] != "'":
                    if text[i] == "\n":
                        line += 1
                    chars.append(text[i])
                    i += 1
                if i < n:
                    i += 1
                continue
            if ch == '"':
                was_quoted = True
                i += 1
                while i < n and text[i] != '"':
                    if text[i] == "\\" and i + 1 < n and text[i + 1] in '"\\$\n':
                        if text[i + 1] == "\n":
                            line += 1
                        else:
                            chars.append(text[i + 1])
                        i += 2
                    else:
                        if text[i] == "\n":
                            line += 1
                        chars.append(text[i])
                        i += 1
                if i < n:
                    i += 1
                continue
            chars.append(ch)
            i += 1
        return "".join(chars), was_quoted

    while i < n:
        ch = text[i]
        # Line continuation outside quotes
        if ch == "\\" and i + 1 < n and text[i + 1] == "\n":
            i += 2
            line += 1
            continue
        if ch in " \t\r":
            i += 1
            continue
        if ch == "\n":
            tokens.append(Token("NL", "\n", None, line))
            i += 1
            line += 1
            if pending_heredocs:
                consume_heredocs()
            continue
        if ch == "#":
            nl = text.find("\n", i)
            if nl == -1:
                i = n
            else:
                i = nl
            continue
        # Arithmetic command (( ... )) at command start or inside $(( ... ))
        if ch == "(" and i + 1 < n and text[i + 1] == "(":
            # Skip balanced (( ... )) so << inside arithmetic is not a heredoc.
            prev_tok = tokens[-1] if tokens else None
            at_cmd = prev_tok is None or prev_tok.kind == "NL" or prev_tok.raw in CMD_SEP_TOKENS or prev_tok.raw in {"(", "{", "if", "then", "elif", "else", "do", "while", "until", "!"}
            if at_cmd:
                start_ln = line
                i += 2
                depth = 2
                while i < n and depth > 0:
                    if text[i] == "(":
                        depth += 1
                    elif text[i] == ")":
                        depth -= 1
                    elif text[i] == "\n":
                        line += 1
                    i += 1
                tokens.append(Token("WORD", "((arith))", None, start_ln))
                continue
        # Operators
        if ch == "<":
            if text[i : i + 3] == "<<<":
                tokens.append(Token("OP", "<<<", None, line))
                i += 3
                continue
            if text[i : i + 2] == "<<":
                strip_tabs = i + 2 < n and text[i + 2] == "-"
                op = "<<-" if strip_tabs else "<<"
                tokens.append(Token("OP", op, None, line))
                i += 3 if strip_tabs else 2
                delim, was_quoted = read_heredoc_delim()
                if delim:
                    pending_heredocs.append((delim, strip_tabs, was_quoted))
                continue
            if i + 1 < n and text[i + 1] in "&<>":
                tokens.append(Token("OP", text[i : i + 2], None, line))
                i += 2
                continue
            tokens.append(Token("OP", "<", None, line))
            i += 1
            continue
        if ch == ">":
            if i + 1 < n and text[i + 1] in ">&|":
                tokens.append(Token("OP", text[i : i + 2], None, line))
                i += 2
                continue
            tokens.append(Token("OP", ">", None, line))
            i += 1
            continue
        if ch == ";":
            if text[i : i + 3] == ";;&":
                tokens.append(Token("OP", ";;&", None, line))
                i += 3
            elif i + 1 < n and text[i + 1] in ";&":
                tokens.append(Token("OP", text[i : i + 2], None, line))
                i += 2
            else:
                tokens.append(Token("OP", ";", None, line))
                i += 1
            continue
        if ch in "&|":
            if i + 1 < n and text[i + 1] == ch:
                tokens.append(Token("OP", ch + ch, None, line))
                i += 2
            elif ch == "|" and i + 1 < n and text[i + 1] == "&":
                tokens.append(Token("OP", "|&", None, line))
                i += 2
            elif ch == "&" and i + 1 < n and text[i + 1] == ">":
                tokens.append(Token("OP", "&>", None, line))
                i += 2
            else:
                tokens.append(Token("OP", ch, None, line))
                i += 1
            continue
        if ch in "()":
            tokens.append(Token("OP", ch, None, line))
            i += 1
            continue

        # Word token
        start_i = i
        start_ln = line
        is_literal = True
        lit_parts = []
        while i < n:
            ch = text[i]
            if ch in " \t\r\n;&|()<>":
                break
            if ch == "\\" and i + 1 < n:
                if text[i + 1] == "\n":
                    i += 2
                    line += 1
                    continue
                lit_parts.append(text[i + 1])
                i += 2
                continue
            if ch == "'":
                i += 1
                sq_start = i
                while i < n and text[i] != "'":
                    if text[i] == "\n":
                        line += 1
                    i += 1
                lit_parts.append(text[sq_start:i])
                if i < n:
                    i += 1
                continue
            if ch == "$" and i + 1 < n and text[i + 1] == "'":
                # ANSI-C string $'...'
                is_literal = False
                i += 2
                while i < n and text[i] != "'":
                    if text[i] == "\\" and i + 1 < n:
                        if text[i + 1] == "\n":
                            line += 1
                        i += 2
                    else:
                        if text[i] == "\n":
                            line += 1
                        i += 1
                if i < n:
                    i += 1
                continue
            if ch == '"':
                i += 1
                while i < n and text[i] != '"':
                    c = text[i]
                    if c == "\\" and i + 1 < n:
                        nxt = text[i + 1]
                        if nxt == "\n":
                            line += 1
                        elif nxt in '"\\$`':
                            lit_parts.append(nxt)
                        else:
                            lit_parts.append("\\" + nxt)
                        i += 2
                        continue
                    if c == "$":
                        is_literal = False
                        if i + 1 < n and text[i + 1] == "(":
                            if i + 2 < n and text[i + 2] == "(":
                                i += 3
                                ad = 2
                                while i < n and ad > 0:
                                    if text[i] == "(":
                                        ad += 1
                                    elif text[i] == ")":
                                        ad -= 1
                                    elif text[i] == "\n":
                                        line += 1
                                    i += 1
                                continue
                            s_ln = line
                            end_j, end_ln, inner = scan_cmd_subst(i + 2, line)
                            subst_spans.append((s_ln, inner))
                            i = end_j
                            line = end_ln
                            continue
                        if i + 1 < n and text[i + 1] == "{":
                            i += 2
                            bd = 1
                            while i < n and bd > 0:
                                if text[i] == "\\" and i + 1 < n:
                                    if text[i + 1] == "\n":
                                        line += 1
                                    i += 2
                                    continue
                                if text[i] == "{":
                                    bd += 1
                                elif text[i] == "}":
                                    bd -= 1
                                elif text[i] == "\n":
                                    line += 1
                                i += 1
                            continue
                        i += 1
                        continue
                    if c == "`":
                        is_literal = False
                        bt_ln = line
                        i += 1
                        bt_start = i
                        while i < n and text[i] != "`":
                            if text[i] == "\\" and i + 1 < n:
                                if text[i + 1] == "\n":
                                    line += 1
                                i += 2
                            else:
                                if text[i] == "\n":
                                    line += 1
                                i += 1
                        bt_body = text[bt_start:i]
                        if "harness_real_dune" in bt_body:
                            backtick_hits.append((bt_ln, bt_body))
                        if i < n:
                            i += 1
                        continue
                    if c == "\n":
                        line += 1
                    lit_parts.append(c)
                    i += 1
                if i < n:
                    i += 1
                continue
            if ch == "$":
                is_literal = False
                if i + 1 < n and text[i + 1] == "(":
                    if i + 2 < n and text[i + 2] == "(":
                        i += 3
                        ad = 2
                        while i < n and ad > 0:
                            if text[i] == "(":
                                ad += 1
                            elif text[i] == ")":
                                ad -= 1
                            elif text[i] == "\n":
                                line += 1
                            i += 1
                        continue
                    s_ln = line
                    end_j, end_ln, inner = scan_cmd_subst(i + 2, line)
                    subst_spans.append((s_ln, inner))
                    i = end_j
                    line = end_ln
                    continue
                if i + 1 < n and text[i + 1] == "{":
                    i += 2
                    bd = 1
                    while i < n and bd > 0:
                        if text[i] == "\\" and i + 1 < n:
                            if text[i + 1] == "\n":
                                line += 1
                            i += 2
                            continue
                        if text[i] == "'":
                            i += 1
                            while i < n and text[i] != "'":
                                if text[i] == "\n":
                                    line += 1
                                i += 1
                            if i < n:
                                i += 1
                            continue
                        if text[i] == '"':
                            i += 1
                            while i < n and text[i] != '"':
                                if text[i] == "\\" and i + 1 < n:
                                    if text[i + 1] == "\n":
                                        line += 1
                                    i += 2
                                else:
                                    if text[i] == "\n":
                                        line += 1
                                    i += 1
                            if i < n:
                                i += 1
                            continue
                        if text[i] == "{":
                            bd += 1
                        elif text[i] == "}":
                            bd -= 1
                        elif text[i] == "\n":
                            line += 1
                        i += 1
                    continue
                i += 1
                continue
            if ch == "`":
                is_literal = False
                bt_ln = line
                i += 1
                bt_start = i
                while i < n and text[i] != "`":
                    if text[i] == "\\" and i + 1 < n:
                        if text[i + 1] == "\n":
                            line += 1
                        i += 2
                    else:
                        if text[i] == "\n":
                            line += 1
                        i += 1
                bt_body = text[bt_start:i]
                if "harness_real_dune" in bt_body:
                    backtick_hits.append((bt_ln, bt_body))
                if i < n:
                    i += 1
                continue
            lit_parts.append(ch)
            i += 1

        raw = text[start_i:i]
        tokens.append(Token("WORD", raw, "".join(lit_parts) if is_literal else None, start_ln))

    return tokens, subst_spans, backtick_hits


def parse_script_structure(text, rel_path, start_line=1):
    """Parse a Bash script and extract:
      - functions: dict[fn_name] -> {'calls': set[str], 'real_dune_lines': list[int]}
      - top_real_dune_lines: list[int] (outside any function and outside any if harness_leg)
      - inline_legs: list[(leg_name, calls_set, real_dune_lines)]
      - leg_runs: list[(leg_name, fn_name, line)]
      - nonliteral_legs: list[(line, detail)]
      - sourced_names: set[str] (basenames of sourced *.sh files excluding harness-support.sh)
      - backtick_hits: list[(line, body)]
    """
    tokens, subst_spans, backtick_hits = tokenize_bash(text, start_line)

    functions = {}  # fn_name -> {'calls': set, 'real_dune_lines': list}
    top_real_dune_lines = []
    inline_legs = []  # list of {'leg': str, 'calls': set, 'real_dune_lines': list}
    leg_runs = []  # list of (leg_name, fn_name, line)
    nonliteral_legs = []  # list of (line, detail)
    sourced_names = set()

    # Track enclosing function bodies and inline `if harness_leg ...; then ... fi`
    fn_stack = []  # list of {'name': str, 'delim': '{' or '(', 'depth': int}
    if_stack = []  # list of {'leg': str|None, 'in_cond': bool, 'inline_rec': dict|None}
    case_depth = 0
    in_case_pattern = False

    def current_fn():
        return fn_stack[-1]["name"] if fn_stack else None

    def current_inline_leg():
        for frame in reversed(if_stack):
            if not frame["in_cond"] and frame["inline_rec"] is not None:
                return frame["inline_rec"]
        return None

    def record_call(cmd_name, ln):
        fn = current_fn()
        inl = current_inline_leg()
        if fn is not None:
            rec = functions.setdefault(fn, {"calls": set(), "real_dune_lines": []})
            rec["calls"].add(cmd_name)
            if cmd_name == "harness_real_dune":
                rec["real_dune_lines"].append(ln)
        elif inl is not None:
            inl["calls"].add(cmd_name)
            if cmd_name == "harness_real_dune":
                inl["real_dune_lines"].append(ln)
        else:
            if cmd_name == "harness_real_dune":
                top_real_dune_lines.append(ln)

    # Also scan command substitutions for harness_real_dune calls; attribute them
    # by line number against the enclosing construct? Better: attach each subst
    # directly by re-running a lightweight check on tokens inside $(...).
    # Even cleaner: scan subst_spans for command-position calls to harness_real_dune
    # and check where their start_line falls; since we record function/inline spans
    # with (start_line, end_line), we can attribute subst calls accurately too!
    fn_ranges = []  # (start_line, end_line, fn_name)
    inline_ranges = []  # (start_line, end_line, inline_rec)

    pending_fn_name = None
    cmd_pos = True
    current_cmd = None  # command name of current simple command
    current_cmd_args = []  # list of Token
    expect_redir_target = False

    def flush_simple_cmd():
        nonlocal current_cmd, current_cmd_args
        if current_cmd is None:
            return
        cmd = current_cmd
        args = current_cmd_args
        current_cmd = None
        current_cmd_args = []

        if cmd in (".", "source") and args:
            m = SOURCED_SH_RE.search(args[0].raw.strip("\"'"))
            if m and m.group(1) != "harness-support.sh":
                sourced_names.add(m.group(1))
        elif cmd in ("harness_leg_run", "harness_leg_run_errexit"):
            if len(args) >= 2 and args[0].literal is not None and args[1].literal is not None:
                leg_runs.append((args[0].literal, args[1].literal, args[0].line))
            else:
                nonliteral_legs.append((args[0].line if args else 0, f"{cmd} with non-literal leg or function name"))
        elif cmd == "harness_leg":
            if if_stack and if_stack[-1]["in_cond"]:
                if args and args[0].literal is not None:
                    rec = {"leg": args[0].literal, "calls": set(), "real_dune_lines": [], "start_line": args[0].line}
                    if_stack[-1]["leg"] = args[0].literal
                    if_stack[-1]["inline_rec"] = rec
                    inline_legs.append(rec)
                else:
                    nonliteral_legs.append((args[0].line if args else 0, "harness_leg with non-literal leg name"))
            else:
                nonliteral_legs.append((args[0].line if args else 0, "harness_leg outside an if-condition"))

    idx = 0
    nt = len(tokens)
    while idx < nt:
        tok = tokens[idx]

        # Handle pending function body opening '{' or '('
        if pending_fn_name is not None:
            if tok.kind == "NL":
                idx += 1
                continue
            if tok.kind == "WORD" and tok.raw == "{":
                fn_name = pending_fn_name
                pending_fn_name = None
                functions.setdefault(fn_name, {"calls": set(), "real_dune_lines": []})
                fn_stack.append({"name": fn_name, "delim": "{", "depth": 1, "start_line": tok.line})
                cmd_pos = True
                idx += 1
                continue
            if tok.kind == "OP" and tok.raw == "(":
                fn_name = pending_fn_name
                pending_fn_name = None
                functions.setdefault(fn_name, {"calls": set(), "real_dune_lines": []})
                fn_stack.append({"name": fn_name, "delim": "(", "depth": 1, "start_line": tok.line})
                cmd_pos = True
                idx += 1
                continue
            # Not a standard body; drop pending
            pending_fn_name = None

        if tok.kind == "NL" or (tok.kind == "OP" and tok.raw in CMD_SEP_TOKENS):
            flush_simple_cmd()
            expect_redir_target = False
            if tok.raw in (";;", ";&", ";;&"):
                in_case_pattern = True
            cmd_pos = True
            idx += 1
            continue

        if tok.kind == "OP" and tok.raw in ("<", "<<", "<<-", "<<<", "<&", "<>", ">", ">>", ">&", ">|", "&>"):
            expect_redir_target = True
            idx += 1
            continue

        if tok.kind == "OP" and tok.raw == "(":
            flush_simple_cmd()
            if not in_case_pattern and fn_stack and fn_stack[-1]["delim"] == "(":
                fn_stack[-1]["depth"] += 1
            if not in_case_pattern:
                cmd_pos = True
            idx += 1
            continue

        if tok.kind == "OP" and tok.raw == ")":
            flush_simple_cmd()
            if in_case_pattern:
                in_case_pattern = False
                cmd_pos = True
            elif fn_stack and fn_stack[-1]["delim"] == "(":
                fn_stack[-1]["depth"] -= 1
                if fn_stack[-1]["depth"] == 0:
                    popped = fn_stack.pop()
                    fn_ranges.append((popped["start_line"], tok.line, popped["name"]))
                cmd_pos = True
            else:
                cmd_pos = True
            idx += 1
            continue

        if expect_redir_target:
            expect_redir_target = False
            idx += 1
            continue

        if in_case_pattern:
            if tok.kind == "WORD" and tok.raw == "esac":
                in_case_pattern = False
                if case_depth > 0:
                    case_depth -= 1
                cmd_pos = True
            idx += 1
            continue

        # WORD token
        w = tok.raw
        if cmd_pos:
            # Check function definition: `name() {` or `name () {`
            if (
                IDENT_RE.match(w)
                and idx + 2 < nt
                and tokens[idx + 1].kind == "OP"
                and tokens[idx + 1].raw == "("
                and tokens[idx + 2].kind == "OP"
                and tokens[idx + 2].raw == ")"
            ):
                pending_fn_name = w
                idx += 3
                continue
            if w == "function" and idx + 1 < nt and tokens[idx + 1].kind == "WORD" and IDENT_RE.match(tokens[idx + 1].raw):
                pending_fn_name = tokens[idx + 1].raw
                idx += 2
                if (
                    idx + 1 < nt
                    and tokens[idx].kind == "OP"
                    and tokens[idx].raw == "("
                    and tokens[idx + 1].kind == "OP"
                    and tokens[idx + 1].raw == ")"
                ):
                    idx += 2
                continue
            if w == "{":
                if fn_stack and fn_stack[-1]["delim"] == "{":
                    fn_stack[-1]["depth"] += 1
                cmd_pos = True
                idx += 1
                continue
            if w == "}":
                if fn_stack and fn_stack[-1]["delim"] == "{":
                    fn_stack[-1]["depth"] -= 1
                    if fn_stack[-1]["depth"] == 0:
                        popped = fn_stack.pop()
                        fn_ranges.append((popped["start_line"], tok.line, popped["name"]))
                cmd_pos = True
                idx += 1
                continue
            if w == "if":
                if_stack.append({"leg": None, "in_cond": True, "inline_rec": None})
                cmd_pos = True
                idx += 1
                continue
            if w == "then":
                if if_stack:
                    if_stack[-1]["in_cond"] = False
                cmd_pos = True
                idx += 1
                continue
            if w == "elif":
                if if_stack:
                    prev = if_stack[-1]["inline_rec"]
                    if prev is not None:
                        inline_ranges.append((prev["start_line"], tok.line, prev))
                    if_stack[-1]["leg"] = None
                    if_stack[-1]["inline_rec"] = None
                    if_stack[-1]["in_cond"] = True
                cmd_pos = True
                idx += 1
                continue
            if w == "else":
                if if_stack:
                    prev = if_stack[-1]["inline_rec"]
                    if prev is not None:
                        inline_ranges.append((prev["start_line"], tok.line, prev))
                    if_stack[-1]["leg"] = None
                    if_stack[-1]["inline_rec"] = None
                    if_stack[-1]["in_cond"] = False
                cmd_pos = True
                idx += 1
                continue
            if w == "fi":
                if if_stack:
                    popped = if_stack.pop()
                    if popped["inline_rec"] is not None:
                        inline_ranges.append((popped["inline_rec"]["start_line"], tok.line, popped["inline_rec"]))
                cmd_pos = True
                idx += 1
                continue
            if w == "case":
                case_depth += 1
                # Advance past `case WORD in`
                idx += 1
                while idx < nt and not (tokens[idx].kind == "WORD" and tokens[idx].raw == "in"):
                    idx += 1
                if idx < nt:
                    idx += 1  # consume 'in'
                in_case_pattern = True
                cmd_pos = True
                continue
            if w == "esac":
                if case_depth > 0:
                    case_depth -= 1
                cmd_pos = True
                idx += 1
                continue
            if w in ("done", "in"):
                cmd_pos = True
                idx += 1
                continue
            if w in CMD_PREFIX_WORDS:
                cmd_pos = True
                idx += 1
                continue
            if ASSIGN_RE.match(w):
                # Leading VAR=val assignment keeps command position.
                cmd_pos = True
                idx += 1
                continue

            # Regular command word!
            cmd_name = tok.literal if tok.literal is not None else w
            current_cmd = cmd_name
            current_cmd_args = []
            record_call(cmd_name, tok.line)
            cmd_pos = False
            idx += 1
            continue

        # Argument to current simple command
        current_cmd_args.append(tok)
        idx += 1

    flush_simple_cmd()

    # Check command substitutions $(...) recursively and feed all command-position
    # calls (both direct harness_real_dune calls and helper calls) into the same
    # call graph as ordinary commands.
    def find_subst_calls(spans):
        hits = []
        for s_ln, code in spans:
            sub_toks, sub_spans, sub_bt = tokenize_bash(code, s_ln)
            backtick_hits.extend(sub_bt)
            cpos = True
            for st in sub_toks:
                if st.kind == "NL" or (st.kind == "OP" and st.raw in CMD_SEP_TOKENS | {"(", ")"}):
                    cpos = True
                    continue
                if st.kind == "OP":
                    continue
                if cpos:
                    if st.raw in CMD_PREFIX_WORDS or st.raw in ("{", "}", "done") or ASSIGN_RE.match(st.raw):
                        continue
                    cname = st.literal if st.literal is not None else st.raw
                    hits.append((st.line, cname))
                    cpos = False
            hits.extend(find_subst_calls(sub_spans))
        return hits

    for h_ln, cname in find_subst_calls(subst_spans):
        # Attribute to innermost enclosing function or inline leg by line range
        attributed = False
        for s_l, e_l, fn_name in reversed(fn_ranges):
            if s_l <= h_ln <= e_l:
                functions[fn_name]["calls"].add(cname)
                if cname == "harness_real_dune":
                    functions[fn_name]["real_dune_lines"].append(h_ln)
                attributed = True
                break
        if not attributed:
            for s_l, e_l, rec in reversed(inline_ranges):
                if s_l <= h_ln <= e_l:
                    rec["calls"].add(cname)
                    if cname == "harness_real_dune":
                        rec["real_dune_lines"].append(h_ln)
                    attributed = True
                    break
        if not attributed and cname == "harness_real_dune":
            top_real_dune_lines.append(h_ln)

    return {
        "functions": functions,
        "top_real_dune_lines": top_real_dune_lines,
        "inline_legs": inline_legs,
        "leg_runs": leg_runs,
        "nonliteral_legs": nonliteral_legs,
        "sourced_names": sourced_names,
        "backtick_hits": backtick_hits,
    }


def analyze_harness(repo_root, rel_path):
    """Discover all legs of rel_path that call harness_real_dune (directly or via helpers)."""
    full = repo_root / rel_path
    text = full.read_text(encoding="utf-8")
    parsed = parse_script_structure(text, rel_path)
    errors = []

    for bt_ln, _ in parsed["backtick_hits"]:
        errors.append(f"{rel_path}:{bt_ln}: live backtick command substitution mentions harness_real_dune (unattributed)")

    # Merge any sourced same-directory .sh helper files (excluding harness-support.sh)
    functions = {k: {"calls": set(v["calls"]), "real_dune_lines": list(v["real_dune_lines"])} for k, v in parsed["functions"].items()}
    seen_sourced = set()
    queue = list(parsed["sourced_names"])
    parent_dir = (repo_root / rel_path).parent
    while queue:
        sname = queue.pop(0)
        if sname in seen_sourced:
            continue
        seen_sourced.add(sname)
        cand = parent_dir / sname
        if not cand.is_file():
            continue
        srel = str(cand.relative_to(repo_root))
        sp = parse_script_structure(cand.read_text(encoding="utf-8"), srel)
        for bt_ln, _ in sp["backtick_hits"]:
            errors.append(f"{srel}:{bt_ln}: live backtick command substitution mentions harness_real_dune (sourced by {rel_path})")
        for ln in sp["top_real_dune_lines"]:
            errors.append(f"{srel}:{ln}: top-level harness_real_dune call in helper sourced by {rel_path}")
        for fn, info in sp["functions"].items():
            rec = functions.setdefault(fn, {"calls": set(), "real_dune_lines": []})
            rec["calls"].update(info["calls"])
            rec["real_dune_lines"].extend(info["real_dune_lines"])
        for nxt in sp["sourced_names"]:
            if nxt not in seen_sourced:
                queue.append(nxt)

    # Compute transitive closure of functions that reach harness_real_dune
    direct_rd_fns = {fn for fn, info in functions.items() if info["real_dune_lines"]}
    rd_fns = set(direct_rd_fns)
    changed = True
    while changed:
        changed = False
        for fn, info in functions.items():
            if fn not in rd_fns and (info["calls"] & rd_fns):
                rd_fns.add(fn)
                changed = True

    discovered_legs = set()
    used_rd_fns = set()

    def mark_reachable(fn):
        stack = [fn]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in functions:
                continue
            visited.add(cur)
            if cur in direct_rd_fns:
                used_rd_fns.add(cur)
            for nxt in functions[cur]["calls"]:
                if nxt in functions and nxt not in visited:
                    stack.append(nxt)

    for leg_name, fn_name, _ in parsed["leg_runs"]:
        if fn_name in rd_fns:
            discovered_legs.add(leg_name)
            mark_reachable(fn_name)

    for rec in parsed["inline_legs"]:
        called_rd = rec["calls"] & rd_fns
        if rec["real_dune_lines"] or called_rd:
            discovered_legs.add(rec["leg"])
            for fn in called_rd:
                mark_reachable(fn)

    for ln in parsed["top_real_dune_lines"]:
        errors.append(f"{rel_path}:{ln}: live harness_real_dune call outside any recognised leg")

    unreached_rd_fns = sorted(direct_rd_fns - used_rd_fns)
    if unreached_rd_fns:
        for fn in unreached_rd_fns:
            lns = ",".join(str(x) for x in functions[fn]["real_dune_lines"])
            errors.append(f"{rel_path}: function `{fn}` calls harness_real_dune (line {lns}) but is not reached by any recognised leg")

    # If the file has ANY live harness_real_dune call and also non-literal leg gates, fail closed
    has_any_rd = bool(parsed["top_real_dune_lines"] or direct_rd_fns or any(r["real_dune_lines"] for r in parsed["inline_legs"]))
    if has_any_rd and parsed["nonliteral_legs"]:
        for ln, detail in parsed["nonliteral_legs"]:
            errors.append(f"{rel_path}:{ln}: {detail} in a harness with live harness_real_dune calls")

    return discovered_legs, errors


def read_manifest(test_harnesses_path, group=None):
    cmd = [BASH_BIN, str(test_harnesses_path)]
    if group:
        cmd.append(f"--{group}")
    cmd.append("--list")
    res = subprocess.run(cmd, env=clean_env(), text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {res.returncode}: {res.stderr.strip()}")
    return [line for line in res.stdout.splitlines() if line]


def read_real_dune_legs(test_harnesses_path, shell_members):
    text = test_harnesses_path.read_text(encoding="utf-8")
    m = re.search(r"(?m)^real_dune_legs\(\) \{\n(.*?^\})", text, re.S)
    if not m:
        return {}, [f"{test_harnesses_path}: could not find real_dune_legs() definition"]
    snippet = m.group(0) + "\nreal_dune_legs\n"
    res = subprocess.run([BASH_BIN, "-u", "-c", snippet], env=clean_env(), text=True, capture_output=True)
    if res.returncode != 0:
        return {}, [f"{test_harnesses_path}: real_dune_legs() exited {res.returncode}: {res.stderr.strip()}"]
    legs = {}
    errors = []
    shell_set = set(shell_members)
    for lineno, line in enumerate(res.stdout.splitlines(), 1):
        if "\r" in line:
            errors.append(f"real_dune_legs line {lineno}: contains carriage return")
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            errors.append(f"real_dune_legs line {lineno}: expected HARNESS<TAB>ERE, got {line!r}")
            continue
        h, ere = parts
        if h in legs:
            errors.append(f"real_dune_legs line {lineno}: duplicate entry for {h}")
            continue
        if h not in shell_set:
            errors.append(f"real_dune_legs line {lineno}: {h} is not a shell-tier manifest member")
        # Validate ERE with Bash itself
        chk = subprocess.run([BASH_BIN, "-c", '[[ "" =~ $1 ]]', "_", ere], env=clean_env())
        if chk.returncode == 2:
            errors.append(f"real_dune_legs line {lineno}: {h}: invalid Bash ERE {ere!r}")
            continue
        legs[h] = ere
    return legs, errors


def read_dune_skip_list(dune_list_path, shell_members):
    entries = set()
    by_harness = {}
    errors = []
    shell_set = set(shell_members)
    raw_bytes = dune_list_path.read_bytes()
    for lineno, raw_line in enumerate(raw_bytes.split(b"\n"), 1):
        if not raw_line:
            continue
        if raw_line.startswith(b"#"):
            continue
        if b"\r" in raw_line:
            errors.append(f"{dune_list_path}:{lineno}: carriage return")
            continue
        line = raw_line.decode("utf-8")
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            errors.append(f"{dune_list_path}:{lineno}: expected HARNESS<TAB>LABEL, got {line!r}")
            continue
        h, label = parts
        if h not in shell_set:
            errors.append(f"{dune_list_path}:{lineno}: {h} is not a shell-tier manifest member")
        pair = (h, label)
        if pair in entries:
            errors.append(f"{dune_list_path}:{lineno}: duplicate entry {h}: {label}")
            continue
        entries.add(pair)
        by_harness.setdefault(h, set()).add(label)
    return entries, by_harness, errors


def list_harness_legs(repo_root, rel_path):
    res = subprocess.run(
        [BASH_BIN, str(repo_root / rel_path), "--list-legs"],
        cwd=str(repo_root),
        env=clean_env(),
        text=True,
        capture_output=True,
    )
    if res.returncode != 0:
        return None, f"{rel_path} --list-legs exited {res.returncode}: {(res.stderr or res.stdout).strip()}"
    legs = []
    for line in res.stdout.splitlines():
        if line.startswith("LEG  "):
            legs.append(line[5:])
    if not legs:
        return None, f"{rel_path} --list-legs reported no legs"
    return legs, None


def bash_ere_filter(ere, candidates):
    if not candidates:
        return set()
    script = 'ere=$1; shift; for l in "$@"; do [[ $l =~ $ere ]] && printf "%s\\n" "$l"; done'
    res = subprocess.run([BASH_BIN, "-c", script, "_", ere, *candidates], env=clean_env(), text=True, capture_output=True)
    if res.returncode not in (0, 1):
        raise RuntimeError(f"Bash ERE match failed for {ere!r}: {res.stderr.strip()}")
    return set(res.stdout.splitlines())


def glob_to_regex(glob_str):
    parts = glob_str.split("*")
    return re.compile(r"\A" + "[^/]*".join(re.escape(p) for p in parts) + r"\Z")


def check_affected_reads(self_path, repo_root, all_members, real_dune_map):
    errors = []
    lines = self_path.read_text(encoding="utf-8").splitlines()
    all_globs = []
    leg2_globs = []
    for i, line in enumerate(lines):
        if i >= 1 and not line.startswith("#"):
            break
        m = re.match(r"^# affected-reads: (\S+(?: \S+)*?)(?: --only '([^'\t\n\r]+)')?$", line)
        if not m:
            continue
        globs = [glob_to_regex(g) for g in m.group(1).split(" ")]
        ere = m.group(2)
        all_globs.extend(globs)
        if ere is None or ere == "^2:":
            leg2_globs.extend(globs)

    self_rel = str(self_path.resolve().relative_to(repo_root.resolve()))
    for member in all_members:
        if member == self_rel:
            continue
        if not any(rx.match(member) for rx in all_globs):
            errors.append(f"{self_rel}: # affected-reads does not cover manifest member {member}")

    leg2_matched = {m for m in all_members if any(rx.match(m) for rx in leg2_globs)}
    rd_set = set(real_dune_map.keys())
    for missing in sorted(rd_set - leg2_matched):
        errors.append(f"{self_rel}: # affected-reads (--only '^2:') omits real_dune_legs member {missing}")
    for extra in sorted(leg2_matched - rd_set):
        errors.append(f"{self_rel}: # affected-reads (--only '^2:') names non-real_dune_legs member {extra}")
    return errors


def discover_all(repo_root, all_members, shell_members):
    shell_set = set(shell_members)
    discovered = {}
    errors = []
    for h in all_members:
        legs, errs = analyze_harness(repo_root, h)
        errors.extend(errs)
        if legs:
            if h not in shell_set:
                errors.append(f"{h}: calls harness_real_dune in legs {sorted(legs)}, but is not a shell-tier member")
            else:
                discovered[h] = legs
    return discovered, errors


def run_static(args):
    repo_root = Path(args.repo_root).resolve()
    th_path = Path(args.test_harnesses).resolve() if args.test_harnesses else repo_root / "tools/test-harnesses.sh"
    dl_path = Path(args.dune_list).resolve() if args.dune_list else repo_root / "tools/harness-skips-ci-shell-dune.txt"

    all_members = read_manifest(th_path)
    shell_members = read_manifest(th_path, "shell")

    errors = []
    rd_map, rd_errs = read_real_dune_legs(th_path, shell_members)
    errors.extend(rd_errs)

    _, dl_by_harness, dl_errs = read_dune_skip_list(dl_path, shell_members)
    errors.extend(dl_errs)

    discovered, disc_errs = discover_all(repo_root, all_members, shell_members)
    errors.extend(disc_errs)

    all_rd_harnesses = sorted(set(discovered) | set(rd_map) | set(dl_by_harness))
    for h in all_rd_harnesses:
        disc_legs = discovered.get(h, set())
        ere = rd_map.get(h)
        has_dl = bool(dl_by_harness.get(h))

        if disc_legs and ere is None:
            errors.append(f"{h}: has harness_real_dune leg(s) {sorted(disc_legs)}, but no entry in real_dune_legs")
        if disc_legs and not has_dl:
            errors.append(f"{h}: has harness_real_dune leg(s) {sorted(disc_legs)}, but no entry in {dl_path.name}")
        if ere is not None and not disc_legs:
            errors.append(f"{h}: listed in real_dune_legs ({ere!r}), but no live harness_real_dune leg was found")
        if has_dl and not disc_legs:
            errors.append(f"{h}: listed in {dl_path.name}, but no live harness_real_dune leg was found")
        if has_dl and ere is None:
            errors.append(f"{h}: listed in {dl_path.name}, but has no entry in real_dune_legs")

        if disc_legs or ere is not None:
            live_legs, list_err = list_harness_legs(repo_root, h)
            if list_err:
                errors.append(list_err)
                continue
            live_set = set(live_legs)
            for l in sorted(disc_legs - live_set):
                errors.append(f"{h}: discovered harness_real_dune leg {l!r} is not listed by --list-legs")
            if ere is not None:
                selected = bash_ere_filter(ere, live_legs)
                for l in sorted(disc_legs - selected):
                    errors.append(f"{h}: harness_real_dune leg {l!r} is not selected by real_dune_legs ({ere!r})")
                for l in sorted(selected - disc_legs):
                    errors.append(f"{h}: real_dune_legs ({ere!r}) selects leg {l!r}, which does not call harness_real_dune")

    if args.check_affected_reads:
        errors.extend(check_affected_reads(Path(args.check_affected_reads), repo_root, all_members, rd_map))

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    total_legs = sum(len(v) for v in discovered.values())
    print(f"OK: {len(discovered)} harness(es), {total_legs} harness_real_dune leg(s) matched real_dune_legs and {dl_path.name}")
    return 0


def run_runtime(args):
    repo_root = Path(args.repo_root).resolve()
    th_path = Path(args.test_harnesses).resolve() if args.test_harnesses else repo_root / "tools/test-harnesses.sh"
    dl_path = Path(args.dune_list).resolve() if args.dune_list else repo_root / "tools/harness-skips-ci-shell-dune.txt"

    all_members = read_manifest(th_path)
    shell_members = read_manifest(th_path, "shell")

    errors = []
    expected_entries, _, dl_errs = read_dune_skip_list(dl_path, shell_members)
    errors.extend(dl_errs)

    discovered, disc_errs = discover_all(repo_root, all_members, shell_members)
    errors.extend(disc_errs)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    target_harnesses = sorted(discovered.keys())
    if args.only_harness:
        if args.only_harness not in discovered:
            print(f"ERROR: --only-harness {args.only_harness} has no discovered harness_real_dune legs", file=sys.stderr)
            return 1
        target_harnesses = [args.only_harness]
        expected_entries = {(h, lbl) for (h, lbl) in expected_entries if h == args.only_harness}

    observed_entries = set()
    with tempfile.TemporaryDirectory(prefix="rd-runtime-") as td:
        td_path = Path(td)
        run_idx = 0
        for h in target_harnesses:
            for l in sorted(discovered[h]):
                run_idx += 1
                exact_ere = "^" + ere_escape(l) + "$"
                dune_rec = td_path / f"dune-{run_idx}.tsv"
                dune_rec.write_text("", encoding="utf-8")
                env = clean_env(
                    {
                        "OCANNL_TOOL_HARNESS_NO_DUNE": "1",
                        "OCANNL_TOOL_HARNESS_NAME": h,
                        "OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD": str(dune_rec),
                    }
                )
                res = subprocess.run(
                    [BASH_BIN, str(repo_root / h), "--only", exact_ere],
                    cwd=str(repo_root),
                    env=env,
                    text=True,
                    capture_output=True,
                )
                if res.returncode != 0:
                    tail = "\n".join((res.stdout + "\n" + res.stderr).splitlines()[-12:])
                    errors.append(
                        f"{h}: leg {l!r} exited {res.returncode} under OCANNL_TOOL_HARNESS_NO_DUNE=1:\n{tail}"
                    )
                    continue
                leg_dune_count = 0
                for line in dune_rec.read_text(encoding="utf-8").splitlines():
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 2:
                        errors.append(f"{h}: leg {l!r}: malformed dune skip record line {line!r}")
                        continue
                    observed_entries.add((parts[0], parts[1]))
                    if parts[0] == h:
                        leg_dune_count += 1
                if leg_dune_count == 0:
                    errors.append(
                        f"{h}: harness_real_dune leg {l!r} emitted no hidden-dune skip under OCANNL_TOOL_HARNESS_NO_DUNE=1"
                    )

    for h, lbl in sorted(observed_entries - expected_entries):
        errors.append(f"{h}: hidden-dune skip label not listed in {dl_path.name}: {lbl}")
    for h, lbl in sorted(expected_entries - observed_entries):
        errors.append(f"{h}: listed in {dl_path.name}, but not emitted by any harness_real_dune leg: {lbl}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: {len(observed_entries)} hidden-dune skip(s) across {len(target_harnesses)} harness(es) match {dl_path.name}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("static", "runtime"), required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--test-harnesses", default=None)
    p.add_argument("--dune-list", default=None)
    p.add_argument("--check-affected-reads", default=None)
    p.add_argument("--only-harness", default=None)
    args = p.parse_args()
    if args.mode == "static":
        sys.exit(run_static(args))
    sys.exit(run_runtime(args))


if __name__ == "__main__":
    main()
PY_CHECKER

run_checker() { # OUT_FILE ARGS...
  local out=$1
  shift
  check_rc=0
  python3 "$checker" "$@" >"$out" 2>&1 || check_rc=$?
}

leg_1_static() {
  run_checker "$TMP/live-static.out" \
    --mode static \
    --repo-root "$root" \
    --check-affected-reads "$here/test-real-dune-registrations.sh"
  if [ "$check_rc" = 0 ] && grep -q '^OK: ' "$TMP/live-static.out"; then
    report 0 'every manifest harness_real_dune leg is registered in real_dune_legs and harness-skips-ci-shell-dune.txt'
  else
    report 1 'every manifest harness_real_dune leg is registered in real_dune_legs and harness-skips-ci-shell-dune.txt' \
      "exit $check_rc: $(tr '\n' '|' <"$TMP/live-static.out")"
  fi
}
harness_leg_run '1: every manifest harness_real_dune leg is registered in real_dune_legs' leg_1_static

leg_2_runtime() {
  run_checker "$TMP/live-runtime.out" \
    --mode runtime \
    --repo-root "$root"
  if [ "$check_rc" = 0 ] && grep -q '^OK: ' "$TMP/live-runtime.out"; then
    report 0 'every harness_real_dune leg skip label under NO_DUNE=1 matches harness-skips-ci-shell-dune.txt'
  else
    report 1 'every harness_real_dune leg skip label under NO_DUNE=1 matches harness-skips-ci-shell-dune.txt' \
      "exit $check_rc: $(tr '\n' '|' <"$TMP/live-runtime.out")"
  fi
}
harness_leg_run '2: every harness_real_dune leg hidden-dune skip label is registered in harness-skips-ci-shell-dune.txt' leg_2_runtime

leg_3_controls() {
  # Build a clean synthetic repository exercising both harness_leg_run and inline
  # `if harness_leg`, a helper function calling harness_real_dune, plus a
  # harness mentioning harness_real_dune only inside a heredoc and comment
  # (nearest honest counterpart).
  local fx=$TMP/fx
  rm -rf "$fx"
  mkdir -p "$fx/tools" "$fx/scripts"
  cp "$root/scripts/harness-support.sh" "$fx/scripts/harness-support.sh"

  write_fx_th() { # [EXTRA_MANIFEST] [REAL_DUNE_BODY]
    local extra_m=${1:-}
    local rd_body=${2:-"  printf '%s\t%s\n' tools/test-fn.sh '^(1|2):'
  printf '%s\t%s\n' tools/test-inline.sh '#real'"}
    cat >"$fx/tools/test-harnesses.sh" <<EOF
#!/usr/bin/env bash
set -u
group=all
while [ \$# -gt 0 ]; do
  case \$1 in
    --shell) group=shell ;;
    --toolchain) group=toolchain ;;
    --list) ;;
    *) exit 2 ;;
  esac
  shift
done
manifest() {
  cat <<'HARNESS_LIST'
shell tools/test-fn.sh
shell tools/test-inline.sh
shell tools/test-heredoc.sh
${extra_m}HARNESS_LIST
}
real_dune_legs() {
${rd_body}
}
while read -r tier script; do
  [ -n "\$tier" ] || continue
  [ "\$group" = all ] || [ "\$tier" = "\$group" ] || continue
  printf '%s\n' "\$script"
done < <(manifest)
EOF
    chmod +x "$fx/tools/test-harnesses.sh"
  }

  cat >"$fx/tools/test-fn.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-fn
probe_helper() {
  if ! harness_real_dune; then
    skip "$1" "$harness_dune_absent"
    return 1
  fi
  return 0
}
leg_1() {
  local label='fn leg one (real dune)'
  if ! harness_real_dune; then skip "$label" "$harness_dune_absent"; return; fi
  report 0 "$label"
}
leg_2() {
  local label='fn leg two via helper (real dune)'
  probe_helper "$label" || return
  report 0 "$label"
}
leg_3() {
  report 0 'plain leg without real dune'
}
harness_leg_run '1: direct real dune leg' leg_1
harness_leg_run '2: helper real dune leg' leg_2
harness_leg_run '3: plain leg' leg_3
finish
EOF

  cat >"$fx/tools/test-inline.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-inline
if harness_leg 'inline real dune leg #real'; then
  label='inline real dune check'
  if ! harness_real_dune; then
    skip "$label" "$harness_dune_absent"
  else
    report 0 "$label"
  fi
fi
finish
EOF

  cat >"$fx/tools/test-heredoc.sh" <<'EOF'
#!/usr/bin/env bash
# Mentions harness_real_dune in a comment and inside a heredoc stub only.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-heredoc
cat >"$TMP/stub.sh" <<'STUB'
harness_real_dune || skip 'stub only' "$harness_dune_absent"
STUB
echo "harness_real_dune" >/dev/null
report 0 'heredoc and string mentions are data, not live calls'
finish
EOF
  chmod +x "$fx/tools/test-fn.sh" "$fx/tools/test-inline.sh" "$fx/tools/test-heredoc.sh"

  write_fx_th
  cat >"$fx/tools/harness-skips-ci-shell-dune.txt" <<'EOF'
# Fixture hidden-dune skip list
tools/test-fn.sh	fn leg one (real dune)
tools/test-fn.sh	fn leg two via helper (real dune)
tools/test-inline.sh	inline real dune check
EOF

  # 1. Clean fixture passes both static and runtime checks.
  run_checker "$TMP/fx-clean-static.out" --mode static --repo-root "$fx"
  local s_rc=$check_rc
  run_checker "$TMP/fx-clean-runtime.out" --mode runtime --repo-root "$fx"
  if [ "$s_rc" = 0 ] && [ "$check_rc" = 0 ]; then
    report 0 'control: clean fixture (function, helper, inline, heredoc/comment) passes both checks'
  else
    report 1 'control: clean fixture (function, helper, inline, heredoc/comment) passes both checks' \
      "static=$s_rc ($(cat "$TMP/fx-clean-static.out")); runtime=$check_rc ($(cat "$TMP/fx-clean-runtime.out"))"
  fi

  # 2. Negative control: dropping one leg from real_dune_legs fails static check.
  write_fx_th "" "  printf '%s\t%s\n' tools/test-fn.sh '^1:'
  printf '%s\t%s\n' tools/test-inline.sh '#real'"
  run_checker "$TMP/fx-drop-rd.out" --mode static --repo-root "$fx"
  if [ "$check_rc" = 1 ] && grep -qF "2: helper real dune leg" "$TMP/fx-drop-rd.out" &&
     grep -qF "is not selected by real_dune_legs" "$TMP/fx-drop-rd.out"; then
    report 0 'negative control: dropping a leg from real_dune_legs fails naming the unregistered leg'
  else
    report 1 'negative control: dropping a leg from real_dune_legs fails naming the unregistered leg' \
      "exit $check_rc: $(cat "$TMP/fx-drop-rd.out")"
  fi

  # 3. Negative control: over-selecting a non-real-dune leg in real_dune_legs fails.
  write_fx_th "" "  printf '%s\t%s\n' tools/test-fn.sh '^(1|2|3):'
  printf '%s\t%s\n' tools/test-inline.sh '#real'"
  run_checker "$TMP/fx-over-rd.out" --mode static --repo-root "$fx"
  if [ "$check_rc" = 1 ] && grep -qF "3: plain leg" "$TMP/fx-over-rd.out" &&
     grep -qF "does not call harness_real_dune" "$TMP/fx-over-rd.out"; then
    report 0 'negative control: over-selecting a non-real-dune leg in real_dune_legs fails'
  else
    report 1 'negative control: over-selecting a non-real-dune leg in real_dune_legs fails' \
      "exit $check_rc: $(cat "$TMP/fx-over-rd.out")"
  fi
  write_fx_th

  # 4. Negative control: dropping one entry from harness-skips-ci-shell-dune.txt fails runtime check.
  grep -vF 'fn leg two via helper (real dune)' "$fx/tools/harness-skips-ci-shell-dune.txt" >"$TMP/fx-drop-dl.txt"
  run_checker "$TMP/fx-drop-dl.out" --mode runtime --repo-root "$fx" --dune-list "$TMP/fx-drop-dl.txt"
  if [ "$check_rc" = 1 ] && grep -qF 'hidden-dune skip label not listed' "$TMP/fx-drop-dl.out" &&
     grep -qF 'fn leg two via helper (real dune)' "$TMP/fx-drop-dl.out"; then
    report 0 'negative control: dropping a skip label from harness-skips-ci-shell-dune.txt fails naming it'
  else
    report 1 'negative control: dropping a skip label from harness-skips-ci-shell-dune.txt fails naming it' \
      "exit $check_rc: $(cat "$TMP/fx-drop-dl.out")"
  fi

  # 5. Negative control: stale entry in harness-skips-ci-shell-dune.txt fails runtime check.
  cp "$fx/tools/harness-skips-ci-shell-dune.txt" "$TMP/fx-stale-dl.txt"
  printf 'tools/test-fn.sh\tstale label that never skips\n' >>"$TMP/fx-stale-dl.txt"
  run_checker "$TMP/fx-stale-dl.out" --mode runtime --repo-root "$fx" --dune-list "$TMP/fx-stale-dl.txt"
  if [ "$check_rc" = 1 ] && grep -qF 'not emitted by any harness_real_dune leg' "$TMP/fx-stale-dl.out" &&
     grep -qF 'stale label that never skips' "$TMP/fx-stale-dl.out"; then
    report 0 'negative control: a stale entry in harness-skips-ci-shell-dune.txt fails naming it'
  else
    report 1 'negative control: a stale entry in harness-skips-ci-shell-dune.txt fails naming it' \
      "exit $check_rc: $(cat "$TMP/fx-stale-dl.out")"
  fi

  # 6. Negative control (PR #2343 shape): adding a new harness_real_dune leg missing from BOTH lists fails both checks.
  cp "$fx/tools/test-fn.sh" "$TMP/test-fn.orig"
  python3 - "$fx/tools/test-fn.sh" <<'PY_ADD_LEG' || { harness_abandon 'could not inject unregistered leg'; return; }
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text()
extra = """leg_4() {
  local label='unregistered leg 4 (real dune)'
  if ! harness_real_dune; then skip "$label" "$harness_dune_absent"; return; fi
  report 0 "$label"
}
harness_leg_run '4: unregistered real dune leg' leg_4
"""
p.write_text(text.replace("finish\n", extra + "finish\n"))
PY_ADD_LEG
  run_checker "$TMP/fx-pr2343-static.out" --mode static --repo-root "$fx"
  local p_s=$check_rc
  run_checker "$TMP/fx-pr2343-runtime.out" --mode runtime --repo-root "$fx"
  if [ "$p_s" = 1 ] && grep -qF '4: unregistered real dune leg' "$TMP/fx-pr2343-static.out" &&
     [ "$check_rc" = 1 ] && grep -qF 'unregistered leg 4 (real dune)' "$TMP/fx-pr2343-runtime.out"; then
    report 0 'negative control: an unregistered new harness_real_dune leg fails both static and runtime checks'
  else
    report 1 'negative control: an unregistered new harness_real_dune leg fails both static and runtime checks' \
      "static=$p_s ($(cat "$TMP/fx-pr2343-static.out")); runtime=$check_rc ($(cat "$TMP/fx-pr2343-runtime.out"))"
  fi
  cp "$TMP/test-fn.orig" "$fx/tools/test-fn.sh"

  # 7. Fail-closed controls: ungated call, uncalled helper, wrong-reason skip, and non-shell tier.
  cat >"$fx/tools/test-ungated.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-ungated
harness_real_dune || skip 'ungated check' "$harness_dune_absent"
finish
EOF
  chmod +x "$fx/tools/test-ungated.sh"
  write_fx_th "shell tools/test-ungated.sh
"
  run_checker "$TMP/fx-ungated.out" --mode static --repo-root "$fx"
  if [ "$check_rc" = 1 ] && grep -qF 'outside any recognised leg' "$TMP/fx-ungated.out"; then
    report 0 'negative control: an ungated harness_real_dune call is refused fail-closed'
  else
    report 1 'negative control: an ungated harness_real_dune call is refused fail-closed' \
      "exit $check_rc: $(cat "$TMP/fx-ungated.out")"
  fi

  cat >"$fx/tools/test-uncalled.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-uncalled
orphan_helper() { harness_real_dune || skip 'orphan' "$harness_dune_absent"; }
leg_ok() { report 0 'ok'; }
harness_leg_run '1: ok leg' leg_ok
finish
EOF
  chmod +x "$fx/tools/test-uncalled.sh"
  write_fx_th "shell tools/test-uncalled.sh
"
  run_checker "$TMP/fx-uncalled.out" --mode static --repo-root "$fx"
  if [ "$check_rc" = 1 ] && grep -qF 'orphan_helper' "$TMP/fx-uncalled.out" &&
     grep -qF 'not reached by any recognised leg' "$TMP/fx-uncalled.out"; then
    report 0 'negative control: an uncalled function invoking harness_real_dune is refused fail-closed'
  else
    report 1 'negative control: an uncalled function invoking harness_real_dune is refused fail-closed' \
      "exit $check_rc: $(cat "$TMP/fx-uncalled.out")"
  fi

  cat >"$fx/tools/test-wrong-reason.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-wrong-reason
leg_wr() {
  if ! harness_real_dune; then skip 'wrong reason check' 'custom non-dune reason'; return; fi
  report 0 'wrong reason check'
}
harness_leg_run '1: wrong reason leg' leg_wr
finish
EOF
  chmod +x "$fx/tools/test-wrong-reason.sh"
  write_fx_th "shell tools/test-wrong-reason.sh
" "  printf '%s\t%s\n' tools/test-fn.sh '^(1|2):'
  printf '%s\t%s\n' tools/test-inline.sh '#real'
  printf '%s\t%s\n' tools/test-wrong-reason.sh '^1:'"
  cp "$fx/tools/harness-skips-ci-shell-dune.txt" "$TMP/fx-wrong-dl.txt"
  printf 'tools/test-wrong-reason.sh\twrong reason check\n' >>"$TMP/fx-wrong-dl.txt"
  run_checker "$TMP/fx-wrong-reason.out" --mode runtime --repo-root "$fx" --dune-list "$TMP/fx-wrong-dl.txt"
  if [ "$check_rc" = 1 ] && grep -qF 'emitted no hidden-dune skip' "$TMP/fx-wrong-reason.out"; then
    report 0 'negative control: a harness_real_dune leg skipping with a non-hidden-dune reason fails runtime check'
  else
    report 1 'negative control: a harness_real_dune leg skipping with a non-hidden-dune reason fails runtime check' \
      "exit $check_rc: $(cat "$TMP/fx-wrong-reason.out")"
  fi

  cat >"$fx/tools/test-shared-label.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-shared
leg_a() {
  if ! harness_real_dune; then skip 'shared check label' "$harness_dune_absent"; return; fi
  report 0 'shared check label'
}
leg_b() {
  if ! harness_real_dune; then skip 'shared check label' 'different non-dune reason'; return; fi
  report 0 'shared check label'
}
harness_leg_run '1: shared label dune skip' leg_a
harness_leg_run '2: shared label wrong reason' leg_b
finish
EOF
  chmod +x "$fx/tools/test-shared-label.sh"
  write_fx_th "shell tools/test-shared-label.sh
" "  printf '%s\t%s\n' tools/test-fn.sh '^(1|2):'
  printf '%s\t%s\n' tools/test-inline.sh '#real'
  printf '%s\t%s\n' tools/test-shared-label.sh '^(1|2):'"
  cp "$fx/tools/harness-skips-ci-shell-dune.txt" "$TMP/fx-shared-dl.txt"
  printf 'tools/test-shared-label.sh\tshared check label\n' >>"$TMP/fx-shared-dl.txt"
  run_checker "$TMP/fx-shared-label.out" --mode runtime --repo-root "$fx" --dune-list "$TMP/fx-shared-dl.txt"
  if [ "$check_rc" = 1 ] && grep -qF "2: shared label wrong reason" "$TMP/fx-shared-label.out" &&
     grep -qF 'emitted no hidden-dune skip' "$TMP/fx-shared-label.out"; then
    report 0 'negative control: shared label with wrong reason in another leg is not credited to that leg'
  else
    report 1 'negative control: shared label with wrong reason in another leg is not credited to that leg' \
      "exit $check_rc: $(cat "$TMP/fx-shared-label.out")"
  fi

  # Controls for command-substitution helper calls and unquoted heredoc expansions.
  cat >"$fx/tools/test-subst-helper.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-subst-helper
probe_in_subst() {
  harness_real_dune || return 1
  printf '%s' "$harness_dune"
}
leg_sh() {
  local found
  found=$(probe_in_subst) || { skip 'subst helper check' "$harness_dune_absent"; return; }
  report 0 "subst helper check ($found)"
}
harness_leg_run '1: helper invoked inside command substitution' leg_sh
finish
EOF
  chmod +x "$fx/tools/test-subst-helper.sh"
  write_fx_th "shell tools/test-subst-helper.sh
" "  printf '%s\t%s\n' tools/test-fn.sh '^(1|2):'
  printf '%s\t%s\n' tools/test-inline.sh '#real'
  printf '%s\t%s\n' tools/test-subst-helper.sh '^1:'"
  cp "$fx/tools/harness-skips-ci-shell-dune.txt" "$TMP/fx-subst-dl.txt"
  printf 'tools/test-subst-helper.sh\tsubst helper check\n' >>"$TMP/fx-subst-dl.txt"
  run_checker "$TMP/fx-subst-helper.out" --mode static --repo-root "$fx" --dune-list "$TMP/fx-subst-dl.txt"
  if [ "$check_rc" = 0 ]; then
    report 0 'control: helper calling harness_real_dune inside $(...) is attributed to its calling leg'
  else
    report 1 'control: helper calling harness_real_dune inside $(...) is attributed to its calling leg' \
      "exit $check_rc: $(cat "$TMP/fx-subst-helper.out")"
  fi

  cat >"$fx/tools/test-unquoted-heredoc.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-uq-heredoc
leg_uq() {
  cat <<UNQUOTED_EOF >/dev/null
$(harness_real_dune)
UNQUOTED_EOF
  report 0 'unquoted heredoc'
}
harness_leg_run '1: unquoted heredoc expansion' leg_uq
finish
EOF
  chmod +x "$fx/tools/test-unquoted-heredoc.sh"
  write_fx_th "shell tools/test-unquoted-heredoc.sh
"
  run_checker "$TMP/fx-uq-heredoc.out" --mode static --repo-root "$fx"
  if [ "$check_rc" = 1 ] && grep -qF '1: unquoted heredoc expansion' "$TMP/fx-uq-heredoc.out" &&
     grep -qF 'no entry in real_dune_legs' "$TMP/fx-uq-heredoc.out"; then
    report 0 'negative control: live $(harness_real_dune) inside an unquoted heredoc is detected'
  else
    report 1 'negative control: live $(harness_real_dune) inside an unquoted heredoc is detected' \
      "exit $check_rc: $(cat "$TMP/fx-uq-heredoc.out")"
  fi

  cat >"$fx/tools/test-toolchain-rd.sh" <<'EOF'
#!/usr/bin/env bash
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fx-tc
leg_tc() { harness_real_dune || skip 'tc check' "$harness_dune_absent"; }
harness_leg_run '1: toolchain leg' leg_tc
finish
EOF
  chmod +x "$fx/tools/test-toolchain-rd.sh"
  write_fx_th "toolchain tools/test-toolchain-rd.sh
"
  run_checker "$TMP/fx-tc.out" --mode static --repo-root "$fx"
  if [ "$check_rc" = 1 ] && grep -qF 'not a shell-tier member' "$TMP/fx-tc.out"; then
    report 0 'negative control: a toolchain-tier harness calling harness_real_dune is refused'
  else
    report 1 'negative control: a toolchain-tier harness calling harness_real_dune is refused' \
      "exit $check_rc: $(cat "$TMP/fx-tc.out")"
  fi
  write_fx_th

  # 8. Live-repository negative controls: derive one live registration from
  # real_dune_legs and one live entry from harness-skips-ci-shell-dune.txt so
  # renumbering or renaming legs never breaks the control.
  local live_rd_harness
  live_rd_harness=$(python3 - "$root/tools/test-harnesses.sh" "$TMP/live-mut-th.sh" <<'PY_LIVE_RD'
from pathlib import Path
import re, sys
text = Path(sys.argv[1]).read_text()
m = re.search(r"(?m)^(real_dune_legs\(\) \{\n)(.*?)(\n^\})", text, re.S)
assert m, "real_dune_legs() not found"
body_lines = m.group(2).splitlines()
assert body_lines, "real_dune_legs() is empty"
dropped = body_lines[0]
hm = re.search(r"([A-Za-z0-9_./-]+\.sh)", dropped)
assert hm, f"no harness path in {dropped!r}"
new_body = "\n".join(body_lines[1:])
new_text = text[:m.start(2)] + new_body + text[m.end(2):]
Path(sys.argv[2]).write_text(new_text)
print(hm.group(1))
PY_LIVE_RD
  ) || { harness_abandon 'could not mutate real_dune_legs copy'; return; }
  chmod +x "$TMP/live-mut-th.sh"
  run_checker "$TMP/live-mut-rd.out" --mode static --repo-root "$root" --test-harnesses "$TMP/live-mut-th.sh"
  if [ "$check_rc" = 1 ] && grep -qF "$live_rd_harness" "$TMP/live-mut-rd.out" &&
     grep -qF 'no entry in real_dune_legs' "$TMP/live-mut-rd.out"; then
    report 0 'live negative control: dropping a derived real_dune_legs entry fails against the live repository'
  else
    report 1 'live negative control: dropping a derived real_dune_legs entry fails against the live repository' \
      "exit $check_rc: $(cat "$TMP/live-mut-rd.out")"
  fi

  local live_leg_info live_leg_harness live_leg_token
  live_leg_info=$(python3 - "$root/tools/test-harnesses.sh" "$TMP/live-mut-leg-th.sh" <<'PY_LIVE_LEG'
from pathlib import Path
import re, sys
text = Path(sys.argv[1]).read_text()
m = re.search(r"(?m)^(real_dune_legs\(\) \{\n)(.*?)(\n^\})", text, re.S)
assert m, "real_dune_legs() not found"
body_lines = m.group(2).splitlines()
target_idx = None
harness = None
dropped_tok = None
mutated_line = None
for idx, line in enumerate(body_lines):
    lm = re.search(r"([A-Za-z0-9_./-]+\.sh)\s+'([^']*\|[^']*)'", line)
    if not lm:
        continue
    harness, sel = lm.group(1), lm.group(2)
    gm = re.search(r"\(([^()|]+)\|([^()]+)\)", sel)
    if gm:
        dropped_tok = gm.group(1)
        new_sel = sel[:gm.start()] + "(" + gm.group(2) + ")" + sel[gm.end():]
    else:
        parts = sel.split("|", 1)
        dropped_tok = parts[0]
        new_sel = parts[1]
    target_idx = idx
    mutated_line = line[:lm.start(2)] + new_sel + line[lm.end(2):]
    break
assert target_idx is not None, "no multi-leg selector found in real_dune_legs()"
new_lines = list(body_lines)
new_lines[target_idx] = mutated_line
new_text = text[:m.start(2)] + "\n".join(new_lines) + text[m.end(2):]
Path(sys.argv[2]).write_text(new_text)
print(f"{harness}\t{dropped_tok}")
PY_LIVE_LEG
  ) || { harness_abandon 'could not drop one leg from a multi-leg real_dune_legs selector'; return; }
  live_leg_harness=${live_leg_info%%$'\t'*}
  live_leg_token=${live_leg_info#*$'\t'}
  chmod +x "$TMP/live-mut-leg-th.sh"
  run_checker "$TMP/live-mut-leg.out" --mode static --repo-root "$root" --test-harnesses "$TMP/live-mut-leg-th.sh"
  if [ "$check_rc" = 1 ] && grep -qF "$live_leg_harness" "$TMP/live-mut-leg.out" &&
     grep -qF "$live_leg_token" "$TMP/live-mut-leg.out" &&
     grep -qF 'is not selected by real_dune_legs' "$TMP/live-mut-leg.out"; then
    report 0 'live negative control: dropping one leg from a multi-leg real_dune_legs selector fails naming it'
  else
    report 1 'live negative control: dropping one leg from a multi-leg real_dune_legs selector fails naming it' \
      "exit $check_rc: $(cat "$TMP/live-mut-leg.out")"
  fi

  local live_dl_pair live_dl_harness live_dl_label
  live_dl_pair=$(awk -F '\t' '/^[^#]/ && NF == 2 { print $1 "\t" $2; exit }' \
    "$root/tools/harness-skips-ci-shell-dune.txt")
  live_dl_harness=${live_dl_pair%%$'\t'*}
  live_dl_label=${live_dl_pair#*$'\t'}
  if [ -z "$live_dl_harness" ] || [ -z "$live_dl_label" ]; then
    harness_abandon 'could not derive a live entry from harness-skips-ci-shell-dune.txt'
    return
  fi
  grep -vxF "$live_dl_pair" \
    "$root/tools/harness-skips-ci-shell-dune.txt" >"$TMP/live-mut-dl.txt"
  run_checker "$TMP/live-mut-dl.out" --mode runtime --repo-root "$root" \
    --dune-list "$TMP/live-mut-dl.txt" --only-harness "$live_dl_harness"
  if [ "$check_rc" = 1 ] && grep -qF 'hidden-dune skip label not listed' "$TMP/live-mut-dl.out" &&
     grep -qF "$live_dl_label" "$TMP/live-mut-dl.out"; then
    report 0 'live negative control: dropping a derived harness-skips-ci-shell-dune.txt entry fails against the live repository'
  else
    report 1 'live negative control: dropping a derived harness-skips-ci-shell-dune.txt entry fails against the live repository' \
      "exit $check_rc: $(cat "$TMP/live-mut-dl.out")"
  fi
}
harness_leg_run '3: negative controls catch dropped, stale, over-selected and unattributed entries' leg_3_controls

finish
