#!/usr/bin/env bash
# Hermetic regression and mutation tests for the pin-revisions action.
#
# The action's production shell lives in resolve.sh rather than embedded YAML
# specifically so this harness can run that exact file. Fake `opam` output is
# shaped like the opam 2.5.2 Actions run that invalidated the old repository-
# stamp implementation; fake `git` makes revision resolution deterministic and
# keeps every leg offline. OPAMCOLOR=always is hostile on purpose.
#
# Each shipping assertion has a fault-injected twin below. A mutation must make
# the same oracle reject the subject, proving the test would go red if that
# defect were reintroduced instead of merely restating today's implementation.
#
#   tools/test-pin-revisions.sh          # run every leg
#   tools/test-pin-revisions.sh --keep   # keep the scratch directory
#
# A failing leg reports the run directory holding the subject's stdout, stderr,
# `$GITHUB_OUTPUT` and recorded fake-tool calls -- which is the only way to see
# what the subject actually did, and which cleanup deletes on the way out, so
# the printed path is dead by the time it is read. `--keep` is what makes that
# report actionable; it is the same flag, spelled the same way, as the sibling
# harnesses tools/test-promote.sh and tools/test-test-run.sh.

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/.github/actions/pin-revisions/resolve.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }


harness_require bash awk sed grep sort git
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  skip "every digest leg" "neither sha256sum nor shasum is on PATH"
  finish
fi
harness_scratch "test-pin-revisions"


mkdir -p "$TMP/bin" "$TMP/project"
printf 'opam-version: "2.0"\n' >"$TMP/project/arrayjit.opam"
printf 'opam-version: "2.0"\n' >"$TMP/project/neural_nets_lib.opam"

# Recorded-output fixture. It deliberately returns an unsorted solution with a
# duplicate, the 2.5.2 pin-table shape containing both local git+file pins and
# duplicate remote pins, and ANSI wrappers unless every query says
# `--color=never`. Calls unsupported by the current CLI solution approach fail;
# in particular, `opam var root` exposes the retired repository-stamp approach.
cat >"$TMP/bin/opam" <<'FAKE_OPAM'
#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_OPAM_CALLS"

has_arg() {
  local wanted=$1 arg
  shift
  for arg in "$@"; do [ "$arg" = "$wanted" ] && return 0; done
  return 1
}
emit() {
  if [ "${OPAMCOLOR:-}" = always ] && ! has_arg --color=never "$@"; then
    while IFS= read -r line; do printf '\033[36m%s\033[0m\n' "$line"; done
  else
    cat
  fi
}

case " $* " in
  *" var root "*)
    printf '%s\n' "$FAKE_OPAM_ROOT"
    ;;
  *" pin list "*)
    has_arg --cli=2.1 "$@" && has_arg --safe "$@" \
      || { echo "unexpected opam pin list contract: $*" >&2; exit 64; }
    case "${FAKE_PIN_FIXTURE:-mixed-a}" in
      empty)
        printf '%s\n' 'package version kind target' | emit "$@"
        ;;
      mixed-a)
        printf '%s\n' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'arrayjit.dev git git+file:///checkout#deadbeef' \
          'alpha.dev git git+https://example.invalid/alpha.git' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'neural_nets_lib.dev git git+file:///checkout#deadbeef' | emit "$@"
        ;;
      mixed-b)
        printf '%s\n' \
          'alpha.dev git git+https://example.invalid/alpha.git' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'neural_nets_lib.dev git git+file:///checkout#deadbeef' \
          'zeta.dev git git+https://example.invalid/zeta.git#main' \
          'arrayjit.dev git git+file:///checkout#deadbeef' | emit "$@"
        ;;
      *) echo "unknown pin fixture: $FAKE_PIN_FIXTURE" >&2; exit 64 ;;
    esac
    ;;
  *" list "*)
    has_arg --cli=2.1 "$@" && has_arg --safe "$@" \
      && has_arg --with-test "$@" && has_arg --with-doc "$@" \
      && has_arg --columns=package "$@" && has_arg --short "$@" \
      && has_arg --resolve=arrayjit,neural_nets_lib "$@" \
      || { echo "unexpected opam list contract: $*" >&2; exit 64; }
    case "${FAKE_LIST_FIXTURE:-mixed}" in
      # The solution carries the project's own packages, as `--resolve` does.
      mixed)
        printf '%s\n' beta.2.0 ocannl.dev arrayjit.dev alpha.1.0 \
          neural_nets_lib.dev beta.2.0 | emit "$@"
        ;;
      project-only)
        printf '%s\n' neural_nets_lib.dev arrayjit.dev | emit "$@"
        ;;
      *) echo "unknown list fixture: $FAKE_LIST_FIXTURE" >&2; exit 64 ;;
    esac
    ;;
  *" show "*)
    # The project's pinned definitions carry the checkout's git ref, so they
    # must never reach the definition digest (gh-ocannl-889).
    if has_arg arrayjit.dev "$@" || has_arg neural_nets_lib.dev "$@"; then
      echo "project package reached opam show: $*" >&2
      exit 64
    fi
    # No package argument at all: answer silently, so the guard mutant below
    # completes with a digest and the loud-failure oracle proves the guard
    # itself rather than this fixture's refusal.
    if ! has_arg alpha.1.0 "$@" && ! has_arg beta.2.0 "$@" && ! has_arg ocannl.dev "$@"; then
      exit 0
    fi
    has_arg --cli=2.1 "$@" && has_arg --safe "$@" \
      && has_arg --raw "$@" && has_arg --sort "$@" \
      && has_arg alpha.1.0 "$@" && has_arg beta.2.0 "$@" \
      && has_arg ocannl.dev "$@" \
      || { echo "unexpected opam show contract: $*" >&2; exit 64; }
    {
      printf '%s\n' \
        'opam-version: "2.0"' \
        'synopsis: "Alpha fixture"' \
        'version: "1.0"' \
        'description: """' \
        'The alpha definition has a multi-line description.' \
        'Its fields are deliberately out of order.' \
        '"""' \
        'name: "alpha"' \
        'url {' \
        '  src: "https://example.invalid/alpha-1.0.tbz"' \
        '  checksum: "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' \
        '}' \
        'opam-version: "2.0"' \
        'url {' \
        '  src: "https://example.invalid/beta-2.0.tbz"' \
        '  checksum: "sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"' \
        '}' \
        'description: """' \
        'The beta definition puts its URL before its identity.' \
        'This text keeps the definition realistically multi-line.' \
        '"""' \
        'name: "beta"' \
        'maintainer: "beta@example.invalid"' \
        'version: "2.0"'
      # A short opam answer is the count guard's fault-injected input. The
      # complete fixture's last definition also omits identity fields so the
      # parser's deliberately uninformative ?.? fallback remains exercised.
      if [ "${FAKE_SHOW_FIXTURE:-complete}" != drop-one ]; then
        printf '%s\n' \
          'opam-version: "2.0"' \
          'description: """' \
          'This definition deliberately has no name or version.' \
          'It pins the fallback label for malformed opam output.' \
          '"""' \
          'url {' \
          '  src: "https://example.invalid/ocannl-dev.tbz"' \
          '  checksum: "sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"' \
          '}' \
          'synopsis: "Missing identity fixture"'
      fi
    } | emit "$@"
    ;;
  *)
    echo "unsupported opam 2.5.2 fixture call: $*" >&2
    exit 64
    ;;
esac
FAKE_OPAM

cat >"$TMP/bin/git" <<'FAKE_GIT'
#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >>"$FAKE_GIT_CALLS"
[ "${1:-}" = ls-remote ] || { echo "unsupported fake git call: $*" >&2; exit 64; }
case "${2:-}" in
  https://example.invalid/alpha.git)
    [ "${FAKE_RESOLUTION:-ok}" = fail ] \
      || printf '%s\trefs/heads/main\n' 1111111111111111111111111111111111111111
    ;;
  https://example.invalid/zeta.git)
    printf '%s\trefs/heads/main\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    ;;
  git+file:*|file:*)
    printf '%s\trefs/heads/local\n' cccccccccccccccccccccccccccccccccccccccc
    ;;
  '')
    # Lets the empty-registry mutant complete with a silent digest, so the loud
    # failure oracle proves the guard itself rather than fake git's refusal.
    printf '%s\trefs/heads/empty\n' ffffffffffffffffffffffffffffffffffffffff
    ;;
  *) echo "unknown fake remote: ${2:-}" >&2; exit 64 ;;
esac
FAKE_GIT
chmod +x "$TMP/bin/opam" "$TMP/bin/git"

printf '%s\n' \
  'solution-digest=34956fded534' \
  'digest=623c3bf9bb0e' >"$TMP/expected-output"
printf '%s\n' \
  'ls-remote https://example.invalid/alpha.git HEAD' \
  'ls-remote https://example.invalid/zeta.git main' >"$TMP/expected-git-calls"

run_subject() { # run_subject SUBJECT LABEL PIN_FIXTURE RESOLUTION
  local subject=$1 label=$2 pins=$3 resolution=$4 dir="$TMP/runs/$2" status
  rm -rf "$dir"
  mkdir -p "$dir"
  : >"$dir/github-output"
  : >"$dir/opam.calls"
  : >"$dir/git.calls"
  (
    cd "$TMP/project" || exit 125
    PATH="$TMP/bin:$PATH" \
      OPAMCOLOR=always \
      FAKE_PIN_FIXTURE="$pins" \
      FAKE_LIST_FIXTURE="${FAKE_LIST_FIXTURE:-mixed}" \
      FAKE_SHOW_FIXTURE="${FAKE_SHOW_FIXTURE:-complete}" \
      FAKE_RESOLUTION="$resolution" \
      FAKE_OPAM_ROOT="$TMP/opam-root" \
      FAKE_OPAM_CALLS="$dir/opam.calls" \
      FAKE_GIT_CALLS="$dir/git.calls" \
      GITHUB_OUTPUT="$dir/github-output" \
      bash "$subject" >"$dir/stdout" 2>"$dir/stderr"
  )
  status=$?
  printf '%s\n' "$status" >"$dir/status"
  return 0
}

has_escape() { # has_escape FILE...
  LC_ALL=C grep -q "$(printf '\033')" "$@"
}

lacks_match() { # lacks_match PATTERN FILE
  if grep -q -- "$1" "$2"; then
    return 1
  fi
  return 0
}

# The per-definition listing is what names the package whose definition moved
# when two runs digest differently -- the listings above are identical in that
# case, which is what forced gh-ocannl-889 to be diagnosed locally. Each line
# must carry its own digest. The three realistically shaped definitions must
# produce three DIFFERENT hashes, or the listing is a per-run constant that
# names nothing. The ?.? entry additionally pins the parser's fallback when a
# definition supplies neither identity field.
oracle_definition_digests() { # oracle_definition_digests DIR
  local dir=$1
  grep -q '^Definition digests:$' "$dir/stdout" \
    && grep -qE '^  alpha\.1\.0 [0-9a-f]{12}$' "$dir/stdout" \
    && grep -qE '^  beta\.2\.0 [0-9a-f]{12}$' "$dir/stdout" \
    && grep -qE '^  \?\.\? [0-9a-f]{12}$' "$dir/stdout" \
    && [ "$(grep -cE '^  (alpha\.1\.0|beta\.2\.0|\?\.\?) [0-9a-f]{12}$' \
      "$dir/stdout")" -eq 3 ] \
    && [ "$(grep -oE '^  (alpha\.1\.0|beta\.2\.0|\?\.\?) [0-9a-f]{12}$' \
      "$dir/stdout" | awk '{ print $2 }' | LC_ALL=C sort -u | wc -l)" -eq 3 ]
}

oracle_happy() { # oracle_happy SUBJECT LABEL PIN_FIXTURE
  local subject=$1 label=$2 pins=${3:-mixed-a} dir="$TMP/runs/$2"
  run_subject "$subject" "$label" "$pins" ok
  [ "$(cat "$dir/status")" -eq 0 ] \
    && cmp -s "$TMP/expected-output" "$dir/github-output" \
    && cmp -s "$TMP/expected-git-calls" "$dir/git.calls" \
    && grep -q '^  alpha\.1\.0$' "$dir/stdout" \
    && grep -q '^  beta\.2\.0$' "$dir/stdout" \
    && grep -q '^  arrayjit\.dev$' "$dir/stdout" \
    && grep -q '^Project packages left out of the definition digest: arrayjit neural_nets_lib$' "$dir/stdout" \
    && oracle_definition_digests "$dir" \
    && grep -q '^  git+https://example\.invalid/alpha\.git$' "$dir/stdout" \
    && ! grep -q 'git+file:' "$dir/stdout" \
    && [ "$(grep -c -- '--color=never' "$dir/opam.calls")" -eq 3 ] \
    && ! has_escape "$dir/stdout" "$dir/stderr" "$dir/github-output"
}

oracle_deterministic() { # oracle_deterministic SUBJECT LABEL
  local subject=$1 label=$2 a="$TMP/runs/$2-a" b="$TMP/runs/$2-b"
  oracle_happy "$subject" "$label-a" mixed-a \
    && oracle_happy "$subject" "$label-b" mixed-b \
    && cmp -s "$a/github-output" "$b/github-output" \
    && cmp -s "$a/git.calls" "$b/git.calls"
}

oracle_empty_loud() { # oracle_empty_loud SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  run_subject "$subject" "$label" empty ok
  [ "$(cat "$dir/status")" -ne 0 ] \
    && grep -q '^no remote git pins found in opam pin registry$' "$dir/stderr" \
    && ! grep -q '^digest=' "$dir/github-output"
}

oracle_resolution_loud() { # oracle_resolution_loud SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  run_subject "$subject" "$label" mixed-a fail
  [ "$(cat "$dir/status")" -ne 0 ] \
    && grep -q '^could not resolve HEAD of https://example.invalid/alpha.git$' "$dir/stderr" \
    && ! grep -q '^digest=' "$dir/github-output"
}

oracle_project_only_loud() { # oracle_project_only_loud SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  FAKE_LIST_FIXTURE=project-only run_subject "$subject" "$label" mixed-a ok
  [ "$(cat "$dir/status")" -ne 0 ] \
    && grep -q '^opam dependency solution holds only project packages$' "$dir/stderr" \
    && ! grep -q '^solution-digest=' "$dir/github-output"
}

oracle_partial_definitions_loud() { # oracle_partial_definitions_loud SUBJECT LABEL
  local subject=$1 label=$2 dir="$TMP/runs/$2"
  FAKE_SHOW_FIXTURE=drop-one run_subject "$subject" "$label" mixed-a ok
  [ "$(cat "$dir/status")" -ne 0 ] \
    && grep -q '^opam show returned 2 definitions for 3 requested packages$' "$dir/stderr" \
    && lacks_match '^solution-digest=' "$dir/github-output"
}

if oracle_happy "$SRC" shipping-happy mixed-a; then
  report 0 "opam 2.5.2 output: exact solution and pin digests"
  report 0 "local git+file pins: excluded from resolution and digest"
  report 0 "project packages: excluded from the definition digest"
  report 0 "definition digests: one distinct per-package hash per definition"
  report 0 "OPAMCOLOR=always: no ANSI reaches names, URLs, or outputs"
else
  report 1 "shipping happy path" "see $TMP/runs/shipping-happy"
fi
if oracle_project_only_loud "$SRC" shipping-project-only; then
  report 0 "all-project solution: fails loudly without a solution digest"
else
  report 1 "all-project solution" "see $TMP/runs/shipping-project-only"
fi
if oracle_partial_definitions_loud "$SRC" shipping-partial-definitions; then
  report 0 "partial opam show answer: fails loudly without a solution digest"
else
  report 1 "partial opam show answer" "see $TMP/runs/shipping-partial-definitions"
fi
if oracle_deterministic "$SRC" shipping-order; then
  report 0 "pin ordering and duplicates: one stable resolution order and digest"
else
  report 1 "pin ordering and duplicates" "see $TMP/runs/shipping-order-{a,b}"
fi
if oracle_empty_loud "$SRC" shipping-empty; then
  report 0 "empty pin registry: fails loudly without a pin digest"
else
  report 1 "empty pin registry" "see $TMP/runs/shipping-empty"
fi
if oracle_resolution_loud "$SRC" shipping-resolution; then
  report 0 "empty git resolution: fails loudly without a pin digest"
else
  report 1 "empty git resolution" "see $TMP/runs/shipping-resolution"
fi


# Each mutant must reach its intended wrong result. A failed launch or unrelated
# error must not satisfy the shipping oracle's rejection.
reason_local_pin() { grep -q 'git+file:' "$TMP/runs/$1/stdout"; }
reason_project_package() { grep -q '^project package reached opam show:' "$TMP/runs/$1/stderr"; }
reason_project_only() { grep -q '^no package definitions parsed from opam show output$' "$TMP/runs/$1/stderr"; }
reason_definitions() {
  [ "$(cat "$TMP/runs/$1/status")" = 0 ] && lacks_match '^Definition digests:' "$TMP/runs/$1/stdout"
}
reason_solution_published() {
  [ "$(cat "$TMP/runs/$1/status")" = 0 ] && grep -q '^solution-digest=' "$TMP/runs/$1/github-output"
}
reason_digest_published() {
  [ "$(cat "$TMP/runs/$1/status")" = 0 ] && grep -q '^digest=' "$TMP/runs/$1/github-output"
}
reason_storage() { grep -q 'repo/default/repo' "$TMP/runs/$1/stderr"; }
reason_color() { has_escape "$TMP/runs/$1/stdout" "$TMP/runs/$1/stderr"; }
reason_order() {
  [ "$(cat "$TMP/runs/$1-a/status")" = 0 ] \
    && ! cmp -s "$TMP/expected-output" "$TMP/runs/$1-a/github-output"
}

local_mutant=$(mutant local-pin-filter \
  'index($0, "| sed") && index($0, "git+file:") { changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$local_mutant" ]; then
  expect_rejected "removing local-pin exclusion is detected" "$local_mutant" oracle_happy "" reason_local_pin
else
  report 1 "negative control: local-pin mutant constructed"
fi

project_mutant=$(mutant project-package-filter \
  'index($0, "grep -qxF -- \"$name\"") { print "  if false; then"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$project_mutant" ]; then
  expect_rejected "removing project-package exclusion is detected" "$project_mutant" oracle_happy "" reason_project_package
else
  report 1 "negative control: project-package mutant constructed"
fi

definitions_mutant=$(mutant definition-digest-listing \
  'index($0, "echo \"Definition digests:\"") { skip = 1 } skip { changed++; if (index($0, "paste -d")) skip = 0; next } { print } END { if (changed != 2) exit 9 }')
if [ -n "$definitions_mutant" ]; then
  expect_rejected "dropping the per-definition listing is detected" "$definitions_mutant" oracle_happy "" reason_definitions
else
  report 1 "negative control: definition-listing mutant constructed"
fi

project_guard_mutant=$(mutant project-only-guard \
  'index($0, "#definition_packages[@]") && index($0, "-gt 0") { print "true \\"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$project_guard_mutant" ]; then
  expect_rejected "silent all-project digest is detected" "$project_guard_mutant" oracle_project_only_loud "" reason_project_only
else
  report 1 "negative control: project-only guard mutant constructed"
fi

definition_count_mutant=$(mutant definition-count-guard \
  '/^\[/ && index($0, "wc -l <\"$work_dir/labels\"") { print "true \\"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$definition_count_mutant" ]; then
  expect_rejected "silent partial definition listing is detected" "$definition_count_mutant" oracle_partial_definitions_loud "" reason_solution_published
else
  report 1 "negative control: definition-count guard mutant constructed"
fi

empty_mutant=$(mutant empty-registry-guard \
  'index($0, "[ -n \"$specs\" ] ||") { print "if [ -z \"$specs\" ]; then"; print "  echo \"digest=$(printf %s \\\"\\\" | hash12)\" >>\"$GITHUB_OUTPUT\""; print "  exit 0"; print "fi"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$empty_mutant" ]; then
  expect_rejected "silent empty-registry digest is detected" "$empty_mutant" oracle_empty_loud "" reason_digest_published
else
  report 1 "negative control: empty-registry mutant constructed"
fi

storage_mutant=$(mutant opam-storage \
  '/^set -euo pipefail$/ { print; print "root=$(opam var root)"; print "repo_file=\"$root/repo/default/repo\""; print "stamp=$(sed -n '\''s/^stamp:.*\"\\([^\"]*\\)\".*/\\1/p'\'' \"$repo_file\")"; print "[ -n \"$stamp\" ] || exit 1"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$storage_mutant" ]; then
  expect_rejected "opam 2.5.0 repository-stamp assumption is detected" "$storage_mutant" oracle_happy "" reason_storage
else
  report 1 "negative control: opam-storage mutant constructed"
fi

color_mutant="$TMP/opam-color.sh"
sed 's/ --color=never//g' "$SRC" >"$color_mutant"
if [ "$(grep -c -- '--color=never' "$SRC")" -eq 3 ] \
  && ! grep -q -- '--color=never' "$color_mutant" \
  && bash -n "$color_mutant"; then
  expect_rejected "removing color suppression is detected" "$color_mutant" oracle_happy "" reason_color
else
  report 1 "negative control: OPAMCOLOR mutant constructed"
fi

sort_mutant=$(mutant pin-sort \
  'BEGIN { in_specs=0 } /^specs=\$\(/ { in_specs=1 } in_specs && /LC_ALL=C sort -u\)/ { sub(/LC_ALL=C sort -u/, "cat"); changed++; in_specs=0 } { print } END { if (changed != 1) exit 9 }')
if [ -n "$sort_mutant" ]; then
  expect_rejected "removing pin sort/dedup is detected" "$sort_mutant" oracle_deterministic "" reason_order
else
  report 1 "negative control: pin-sort mutant constructed"
fi

resolution_mutant=$(mutant resolution-guard \
  'index($0, "[ -n \"$sha\" ] ||") { print "  [ -n \"$sha\" ] || sha=0000000000000000000000000000000000000000"; changed++; next } { print } END { if (changed != 1) exit 9 }')
if [ -n "$resolution_mutant" ]; then
  expect_rejected "silent empty resolution is detected" "$resolution_mutant" oracle_resolution_loud "" reason_digest_published
else
  report 1 "negative control: resolution mutant constructed"
fi

if [ "$failures" -ne 0 ]; then
  printf '%s pin-revisions test failure(s)\n' "$failures" >&2
  # The run directories named above are inside $TMP, so without --keep the EXIT
  # trap removes them before anyone can look.
  [ "$KEEP" = 1 ] || printf 're-run with --keep to retain the run directories named above\n' >&2
fi

finish
