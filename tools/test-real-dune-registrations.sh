#!/usr/bin/env bash
# Check that every manifest harness leg calling `harness_real_dune` is
# registered in BOTH `real_dune_legs` (tools/test-harnesses.sh) and
# `tools/harness-skips-ci-shell-dune.txt` (gh-ocannl-2486).
#
# Usage: tools/test-real-dune-registrations.sh [--keep|--help|--list-legs|--only REGEX]
#
# Leg 1 compares the live registry with the independently generated
# discovery output. Leg 2 runs the discovered legs with NO_DUNE=1.
# affected-reads: scripts/test-*.sh tools/test-*.sh benchmarks/test-*.sh test/operations/apt_fallback.sh test/operations/ci_matrix.sh --only '^1:'
# affected-reads: tools/test-test-run-options.sh tools/test-test-run-digest.sh tools/test-test-run-bisect.sh tools/test-merge-build-check.sh --only '^2:'
# affected-reads: test/operations/harness_real_dune_legs.expected
#
# Boundary (fail-closed):
#   - Manifest membership comes from `tools/test-harnesses.sh --list` and
#     `--shell --list`. The literal `real_dune_legs` function supplies selectors.
#     Hidden-dune skips use the three-line-form allowlist of test-harnesses.sh.
#   - Static discovery comes from shell_scripts_parse's checked TSV output.
#     It reads literal calls, leg gates, local functions, sourced siblings and
#     command substitutions through Shell_context. The scan fails when the
#     checked table is stale. This script reads HARNESS<TAB>LEG rows and
#     bare paths for members with no discovered legs. The full path set
#     must equal the live manifest. The table supplies data only.
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

def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in CLEAN_ENV_DROP}
    env["OCANNL_TOOL_FLEET_WORKER"] = "none"
    env["LC_ALL"] = "C"
    if extra:
        env.update(extra)
    return env


def ere_escape(text):
    return re.sub(r"([.\\^$*+?()\[\]{}|])", r"\\\1", text)


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


def read_discovered(repo_root, all_members, shell_members):
    """Read sorted member paths, with one row per discovered leg when present."""
    path = repo_root / "test/operations/harness_real_dune_legs.expected"
    discovered = {}
    errors = []
    all_set = set(all_members)
    shell_set = set(shell_members)
    scanned = set()
    without_legs = set()
    previous = None
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as exc:
        return {}, [f"{path}: cannot read static discovery: {exc}"]
    if text and not text.endswith("\n"):
        return {}, [f"{path}: discovered rows must end with a newline"]
    for number, line in enumerate(text.split("\n")[:-1], 1):
        parts = line.split("\t")
        where = f"{path}:{number}"
        if len(parts) not in (1, 2) or not all(parts) or "\r" in line:
            errors.append(f"{where}: expected HARNESS or HARNESS<TAB>LEG, got {line!r}")
            continue
        harness = parts[0]
        leg = parts[1] if len(parts) == 2 else ""
        if previous is not None and (harness, leg) <= previous:
            errors.append(f"{where}: discovery rows must be sorted and unique")
        previous = (harness, leg)
        scanned.add(harness)
        if harness not in all_set:
            errors.append(f"{where}: {harness} is not a manifest member")
        elif not leg:
            without_legs.add(harness)
        elif harness not in shell_set:
            errors.append(f"{where}: {harness} is not a shell-tier member")
        else:
            discovered.setdefault(harness, set()).add(leg)
    for harness in sorted(all_set - scanned):
        errors.append(f"{path}: manifest member {harness} is missing from static discovery")
    for harness in sorted(without_legs & discovered.keys()):
        errors.append(f"{path}: {harness} has both a bare path and discovered legs")
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

    discovered, disc_errs = read_discovered(repo_root, all_members, shell_members)
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

    discovered, disc_errs = read_discovered(repo_root, all_members, shell_members)
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
  # These fixtures exercise the registry and runtime consumer. The Dune scan
  # separately exercises discovery on whole shell sources.
  local fx=$TMP/fx
  mkdir -p "$fx/tools" "$fx/scripts" "$fx/test/operations"
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

  local discovered=$fx/test/operations/harness_real_dune_legs.expected
  {
    printf '%s\t%s\n' \
      tools/test-fn.sh '1: direct real dune leg' \
      tools/test-fn.sh '2: helper real dune leg'
    printf '%s\n' tools/test-heredoc.sh
    printf '%s\t%s\n' tools/test-inline.sh 'inline real dune leg #real'
  } >"$discovered"
  cp "$discovered" "$TMP/fx-discovered.tsv"

  # 1. The supplied discovery table agrees with the registries and drives the runtime check.
  run_checker "$TMP/fx-clean-static.out" --mode static --repo-root "$fx"
  local s_rc=$check_rc
  run_checker "$TMP/fx-clean-runtime.out" --mode runtime --repo-root "$fx"
  if [ "$s_rc" = 0 ] && [ "$check_rc" = 0 ]; then
    report 0 'control: supplied discovery table passes registry and runtime checks'
  else
    report 1 'control: supplied discovery table passes registry and runtime checks' \
      "static=$s_rc ($(cat "$TMP/fx-clean-static.out")); runtime=$check_rc ($(cat "$TMP/fx-clean-runtime.out"))"
  fi

  grep -vxF 'tools/test-heredoc.sh' "$TMP/fx-discovered.tsv" >"$discovered"
  run_checker "$TMP/fx-root.out" --mode static --repo-root "$fx"
  if [ "$check_rc" = 1 ] &&
     grep -qF 'manifest member tools/test-heredoc.sh is missing from static discovery' "$TMP/fx-root.out"; then
    report 0 'negative control: a live manifest member missing from discovery is refused'
  else
    report 1 'negative control: a live manifest member missing from discovery is refused' \
      "exit $check_rc: $(cat "$TMP/fx-root.out")"
  fi
  cp "$TMP/fx-discovered.tsv" "$discovered"

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
  printf '%s\t%s\n' tools/test-fn.sh '4: unregistered real dune leg' >"$TMP/fx-extra.tsv"
  LC_ALL=C sort "$TMP/fx-discovered.tsv" "$TMP/fx-extra.tsv" >"$discovered"
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

  cp "$TMP/fx-discovered.tsv" "$discovered"

  # 7. A wrong skip reason does not count, even when another leg emits the expected label.
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
  printf '%s\t%s\n' tools/test-wrong-reason.sh '1: wrong reason leg' >>"$discovered"
  run_checker "$TMP/fx-wrong-reason.out" --mode runtime --repo-root "$fx" --dune-list "$TMP/fx-wrong-dl.txt"
  if [ "$check_rc" = 1 ] && grep -qF 'emitted no hidden-dune skip' "$TMP/fx-wrong-reason.out"; then
    report 0 'negative control: a harness_real_dune leg skipping with a non-hidden-dune reason fails runtime check'
  else
    report 1 'negative control: a harness_real_dune leg skipping with a non-hidden-dune reason fails runtime check' \
      "exit $check_rc: $(cat "$TMP/fx-wrong-reason.out")"
  fi

  cp "$TMP/fx-discovered.tsv" "$discovered"
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
  printf '%s\t%s\n' \
    tools/test-shared-label.sh '1: shared label dune skip' \
    tools/test-shared-label.sh '2: shared label wrong reason' >>"$discovered"
  run_checker "$TMP/fx-shared-label.out" --mode runtime --repo-root "$fx" --dune-list "$TMP/fx-shared-dl.txt"
  if [ "$check_rc" = 1 ] && grep -qF "2: shared label wrong reason" "$TMP/fx-shared-label.out" &&
     grep -qF 'emitted no hidden-dune skip' "$TMP/fx-shared-label.out"; then
    report 0 'negative control: shared label with wrong reason in another leg is not credited to that leg'
  else
    report 1 'negative control: shared label with wrong reason in another leg is not credited to that leg' \
      "exit $check_rc: $(cat "$TMP/fx-shared-label.out")"
  fi

  cp "$TMP/fx-discovered.tsv" "$discovered"
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
  printf '%s\t%s\n' tools/test-toolchain-rd.sh '1: toolchain leg' >>"$discovered"
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
