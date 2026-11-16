#!/usr/bin/env bash
# Offline fixture for every "Update apt package index" step in .github/workflows/*.yml.
# Boundary: the current named-step/literal run: | dialect only; unknown scalar styles
# refuse. Compare bodies ignoring blank/full-comment lines and the YAML block margin,
# not step metadata (the Linux if differs). This is not a general YAML/shell parser.
# Execute only extracted bodies, with all /etc/apt paths redirected into scratch;
# refuse any occurrence left over. A private PATH exposes only Python and strict
# sudo/timeout/apt-get stubs: no real sudo, timeout, apt, network or root is needed.
# The historical control covers PR #1934's first rewriter, not its whole workflow.
# affected-reads: .github/workflows/*.yml
# ocannl-harness: standalone
# Usage: test/operations/apt_fallback.sh [--keep|--help]
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
. "$root/scripts/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch apt-fallback
rc=0
python3 - "$root" "$TMP" "$BASH" <<'PY' || rc=$?
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

root, scratch = map(Path, sys.argv[1:3])
bash = sys.argv[3]


def extract(source):
    lines = source.splitlines()
    result = []
    for i, line in enumerate(lines):
        if not re.fullmatch(r'\s*- name: Update apt package index', line):
            continue
        indent = len(line) - len(line.lstrip())
        end = i + 1
        while end < len(lines) and (not lines[end].strip() or
                len(lines[end]) - len(lines[end].lstrip()) > indent):
            end += 1
        block = lines[i + 1:end]
        runs = [j for j, value in enumerate(block) if value.strip().startswith('run:')]
        assert len(runs) == 1 and block[runs[0]].strip() == 'run: |', 'unsupported apt run block'
        j = runs[0]
        margin = len(block[j]) - len(block[j].lstrip())
        body = block[j + 1:]
        assert body and all(not x.strip() or len(x) - len(x.lstrip()) > margin for x in body)
        result.append(textwrap.dedent('\n'.join(body)) + '\n')
    return result


def normalized(script):
    return '\n'.join(line for line in script.splitlines()
                     if line.strip() and not line.lstrip().startswith('#'))


def equivalent(copies):
    assert len(copies) >= 2, 'apt copy census unexpectedly empty/singular'
    reference = normalized(copies[0][1])
    for name, body in copies:
        assert normalized(body) == reference, 'apt step drift: ' + name


copies = [(f'{p.name}:{i + 1}', body)
          for p in sorted((root / '.github/workflows').glob('*.yml'))
          for i, body in enumerate(extract(p.read_text()))]
# Exactly today's five: a new or lost copy must update this census.
assert len(copies) == 5, [name for name, _ in copies]
equivalent(copies)
print('PASS all five discovered apt steps are equivalent')
for label, old, new in (
        ('timeout drift', '240 apt-get', '241 apt-get'),
        ('Python indentation drift', '    path.write_text(source)', '        path.write_text(source)'),
        ('shell continuation drift', 'Acquire::Retries=3 ' + chr(92) + chr(10),
         'Acquire::Retries=3 ' + chr(92) + ' ' + chr(10))):
    drift = copies[:-1] + [(copies[-1][0], copies[-1][1].replace(old, new))]
    assert drift != copies, 'control did not change the script: ' + label
    try:
        equivalent(drift)
    except AssertionError as error:
        assert 'apt step drift:' in str(error)
        print('PASS rejected one drifting workflow copy: ' + label)
    else:
        raise AssertionError('accepted drifting copy: ' + label)

azure = 'azure.archive.ubuntu.com'
other = 'deb https://example.invalid/third-party stable main\n'


def sources(layout, indirect, fallback=False):
    archive = 'https://archive.ubuntu.com/ubuntu'
    security = 'https://security.ubuntu.com/ubuntu'
    uri = 'mirror+file:/etc/apt/apt-mirrors.txt' if indirect else f'https://{azure}/ubuntu'
    if layout == 'deb822':
        def block(url, suites):
            return f'Types: deb deb-src\nURIs: {url}\nSuites: {suites}\nComponents: main universe\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg'
        untouched = block('https://example.invalid/third-party', 'stable')
        active = (block(archive, 'noble noble-updates noble-backports') + '\n\n' +
                  block(security, 'noble-security')) if fallback else block(uri, 'noble noble-updates noble-security noble-backports')
        return 'sources.list.d/ubuntu.sources', active + '\n\n' + untouched + '\n'
    return 'sources.list', (f'deb {archive if fallback else uri} noble main universe\n'
                           f'deb-src {security if fallback else uri} noble-security main universe\n' + other)


# All external commands reachable through PATH are fixture-owned. Absolute Python
# shebangs also work on macOS, which has no system timeout command.
bin_dir = scratch / 'bin'
bin_dir.mkdir()
(bin_dir / 'python3').symlink_to(sys.executable)
stub = '''import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
run = Path(os.environ['APT_FIXTURE'])
def record(value):
    with (run / 'calls').open('a') as out:
        out.write(json.dumps([name, value]) + '\\n')
record(args)
if name == 'sudo':
    assert args[:1] == ['timeout'] or args == ['python3', '-'], args
    os.execv(str(Path(sys.argv[0]).parent / args[0]), args)
elif name == 'timeout':
    assert args[:4] == ['--kill-after=15', '240', 'apt-get', 'update'], args
    os.execv(str(Path(sys.argv[0]).parent / 'apt-get'), args[2:])
else:
    assert args == ['update', '-o', 'Acquire::Retries=3', '-o', 'Acquire::http::Timeout=30',
                    '-o', 'Acquire::https::Timeout=30', '-o', 'APT::Update::Error-Mode=any'], args
    state = run / 'count'
    n = int(state.read_text()) if state.exists() else 0
    state.write_text(str(n + 1))
    (run / ('snapshot-' + str(n))).write_text((run / 'apt' / os.environ['APT_SOURCE']).read_text())
    codes = json.loads(os.environ['APT_CODES'])
    assert n < len(codes), 'unexpected extra apt attempt'
    sys.exit(codes[n])
'''
for name in ('sudo', 'timeout', 'apt-get'):
    path = bin_dir / name
    path.write_text(f'#!{sys.executable}\n' + stub)
    path.chmod(0o755)


def redirect(script, apt):
    redirected = script.replace('/etc/apt', str(apt))
    assert '/etc/apt' not in redirected, 'unrewritten host apt path'
    return redirected


serial = 0


def run_case(script, layout, indirect, codes):
    global serial
    serial += 1
    run = scratch / str(serial)
    apt = run / 'apt'
    (apt / 'sources.list.d').mkdir(parents=True)
    filename, original = sources(layout, indirect)
    original = original.replace('/etc/apt', str(apt))
    (apt / filename).write_text(original)
    mirror_list = '# ignored mirror\nhttps://azure.archive.ubuntu.com/ubuntu\n'
    (apt / 'apt-mirrors.txt').write_text(mirror_list)
    body = redirect(script, apt)
    env = dict(PATH=str(bin_dir), APT_FIXTURE=str(run), APT_SOURCE=filename,
               APT_CODES=json.dumps(codes), LC_ALL='C')
    result = subprocess.run([bash, '--noprofile', '--norc', '-e', '-c', body],
                            env=env, cwd=run, text=True, capture_output=True, timeout=15)
    output = result.stdout + result.stderr
    count = int((run / 'count').read_text())
    expected_count = 1 if codes[0] == 0 else 2
    assert count == expected_count, (layout, indirect, codes, count, output)
    assert (run / 'snapshot-0').read_text() == original
    expected = sources(layout, indirect, fallback=True)[1] if expected_count == 2 else original
    assert (apt / filename).read_text() == expected, (layout, indirect, output)
    if expected_count == 2:
        assert (run / 'snapshot-1').read_text() == expected
        assert '::warning title=apt-get update::attempt 1 of 2' in output
        reason = 'stalled past 240s (apt mirror stall)' if codes[0] in (124, 137) else f'failed with exit {codes[0]}'
        assert reason in output, output
    assert (apt / 'apt-mirrors.txt').read_text() == mirror_list
    calls = [json.loads(line) for line in (run / 'calls').read_text().splitlines()]
    assert sum(name == 'timeout' for name, _ in calls) == expected_count
    assert sum(name == 'sudo' for name, _ in calls) == expected_count + (expected_count == 2)
    failed = codes[expected_count - 1] != 0
    assert result.returncode == (1 if failed else 0), output
    if failed:
        reason = 'stalled past 240s (apt mirror stall)' if codes[-1] in (124, 137) else f'failed with exit {codes[-1]}'
        assert ('::error title=apt-get update::sudo apt-get ' + 'update did not complete in 2 capped attempts; the last ' + reason) in output, output
    else:
        assert '::error' not in output, output
    return output


script = copies[0][1]
for layout in ('deb822', 'one-line'):
    for indirect in (False, True):
        for codes in ([0], [124, 0], [137, 0], [100, 0], [100, 100], [124, 137]):
            run_case(script, layout, indirect, codes)
            print(f'PASS {layout} indirect={indirect} apt exits={codes}')

old = (root / 'test/operations/fixtures/apt_fallback_1934_first.py').read_text()
pattern = r"(sudo python3 - <<'PYTHON'\n).*?(\nPYTHON)"
assert len(re.findall(pattern, script, re.S)) == 1
historical = re.sub(pattern, lambda m: m[1] + old.rstrip() + m[2], script, flags=re.S)
# The old implementation must still pass direct layouts, and fail specifically
# on both mirror-list layouts (one attempt, Cannot switch mirrors), not syntax.
for layout in ('deb822', 'one-line'):
    run_case(historical, layout, False, [124, 0])
    try:
        run_case(historical, layout, True, [124, 0])
    except AssertionError as error:
        assert 'Cannot switch mirrors: neither Ubuntu source file contains azure.archive.ubuntu.com' in str(error), error
        print(f'PASS rejected PR #1934 first implementation on {layout} mirror-list layout')
    else:
        raise AssertionError('historical layout bug accepted')
PY
report "$rc" "apt mirror fallback fixtures and controls"
finish
