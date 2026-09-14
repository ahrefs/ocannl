#!/usr/bin/env bash
# Evaluate ci.yml's actual matrix expressions.
# Requires only python3 and bash; run on the Ubuntu main leg before opam setup.
# ocannl-harness: standalone
# Usage: test/operations/ci_matrix.sh [--keep|--help]
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
. "$root/scripts/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch ci-matrix
rc=0
python3 - "$root/.github/workflows/ci.yml" "$TMP" <<'PY' || rc=$?
import ast
import itertools
import json
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text()


def field(text, name):
    match = re.search(r'^( +)' + re.escape(name) + r': >-\n', text, re.M)
    assert match, name
    indent = len(match[1])
    lines = text[match.end():].splitlines()
    content = list(itertools.takewhile(lambda line: len(line) - len(line.lstrip()) > indent, lines))
    return ' '.join(line.strip() for line in content)


def expression(value, event):
    # This is the workflow's small expression vocabulary, not a second matrix.
    # Parse and whitelist the translated AST so a new operator fails loudly.
    value = value.removeprefix('${{').removesuffix('}}').strip()
    tokens = re.findall(r"'(?:[^']*)'|github\.event_name|inputs\.[a-z_]+|fromJSON|&&|\|\||!=|==|!|[(),]|\s+", value)
    assert ''.join(tokens) == value, value
    mapping = {'github.event_name': 'event', '&&': ' and ', '||': ' or ', '!': ' not '}
    translated = ''.join(mapping.get(token, token) for token in tokens).strip()
    tree = ast.parse(translated, mode='eval')
    permitted = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
                 ast.Compare, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant, ast.Call)
    assert all(isinstance(node, permitted) for node in ast.walk(tree)), translated
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id == 'fromJSON'
    return eval(compile(tree, '<workflow expression>', 'eval'), {'__builtins__': {}},
                dict(event=event, fromJSON=json.loads))


def matrix(text, event):
    systems = expression(field(text, 'os'), event)
    includes = expression(field(text, 'include'), event)
    axes = []
    for name in ('ocaml-compiler', 'suite'):
        match = re.search(r'^        ' + name + r':\n((?:          - .+\n)+)', text, re.M)
        assert match, name
        axes.append([line.strip().removeprefix('- ') for line in match[1].splitlines()])
    jobs = list(itertools.product(systems, *axes))
    jobs += [(entry['os'], entry['ocaml-compiler'], entry['suite']) for entry in includes]
    fmt = re.search(r'^  fmt:\n    if: (.*)$', text, re.M)
    return sorted(jobs), bool(expression(fmt[1], event)) if fmt else True


normal = sorted([('ubuntu-latest', '5.5.x', 'main'),
                 ('macos-latest', '5.5.x', 'main'), ('macos-latest', '5.5.x', 'train')])
full = sorted(normal + [('windows-latest', '5.5.x', 'main'),
                        ('windows-latest', '5.5.x', 'train'), ('ubuntu-latest', '5.3.x', 'main')])


def controls(text):
    for event in ('pull_request', 'push', 'workflow_dispatch'):
        assert matrix(text, event) == (normal, True), event
    assert matrix(text, 'schedule') == (full, True), 'scheduled full coverage'


controls(source)
print('PASS PR, push and manual runs exclude Windows; schedule retains full coverage')
for label, mutant in (
    ('manual Windows dispatch', source.replace("github.event_name == 'schedule'",
                                              "github.event_name != 'push'")),
    ('PR Windows jobs', source.replace("github.event_name == 'schedule'",
                                      "github.event_name != 'workflow_dispatch'")),
    ('push Windows jobs', source.replace("github.event_name == 'schedule'",
                                        "github.event_name != 'pull_request'")),
    ('missing scheduled coverage', source.replace("github.event_name == 'schedule'",
                                                 "github.event_name == 'never'")),
    ('missing scheduled training', source.replace('"os": "windows-latest"', '"os": "macos-latest"')),
    ('missing manual formatting', source.replace('  fmt:\n', "  fmt:\n    if: github.event_name != 'workflow_dispatch'\n")),
):
    assert mutant != source, label
    try:
        controls(mutant)
    except AssertionError:
        print('PASS rejected mutant:', label)
    else:
        raise AssertionError('accepted mutant: ' + label)
PY
report "$rc" "ci matrix controls"
finish
