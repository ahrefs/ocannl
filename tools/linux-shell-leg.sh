#!/usr/bin/env bash
# Run the scoped Linux shell leg (harness members and/or sweep-driver Dune shell tests) on a
# Linux host and write a verifiable receipt for tools/owed-run.sh (gh-ocannl-2414).
# Usage:
#   tools/linux-shell-leg.sh [--jobs N] (--all-harnesses | --harness MEMBER...) [--dune @DIR/runtest-NAME...]
#   tools/linux-shell-leg.sh [--jobs N] --dune @DIR/runtest-NAME...
#
# Boundary (fail-closed allowlist):
#   - Must run on Linux (`uname -s` = Linux); refuses on any other host before creating a receipt.
#   - `--harness MEMBER` (repeatable) must match a member of `tools/test-harnesses.sh --list` with
#     no duplicates; mutually exclusive with `--all-harnesses`.
#   - `--dune @DIR/runtest-NAME` (repeatable) must be a canonical `@DIR/runtest-NAME` operand
#     present in `dune show aliases --root . DIR` with no duplicates.
#   - `--jobs N` is a positive integer (default 4). At least one harness or Dune target is required.
#   - Captures checkout state (`head`, `dirty`) at launch (with Git's local env vars cleared and
#     `GIT_OPTIONAL_LOCKS=0`), writes receipt metadata outside the worktree under
#     `${OCANNL_TOOL_TEST_RUNS:-$HOME/.ocannl-test-runs}/linux-shell/`, binds `FLEET_LEG_RUN_RECORD`
#     and `FLEET_LEG_RECEIPT_LINK` when invoked inside a fleet execution leg, and atomically records
#     `exit` only on completion.
#   - Requires `dune` unconditionally (sourcing `tools/opam-env.sh` when absent from PATH) and
#     `ocamlc` when `--all-harnesses` or a toolchain harness is requested. Runs `--dune` targets
#     under `OCANNL_BACKEND=cc` (`dune build --root . -j N ...`), then `tools/test-harnesses.sh
#     --jobs N` (with `--member ...` or whole), and refuses (exit 1) if any harness member reports
#     zero `PASS` checks or any `SKIP` outside `tools/harness-skips-ci-shell.txt` (plus the
#     absent-zsh skip in `scripts/test-setup-ocaml-env.sh`).
#   - Fails with exit 2 if completion status cannot be written atomically to `$receipt_dir/exit`.
#   - Standalone runs on a fleet Linux box should wrap this command in `fleet-worker.sh execution
#     slot --cpu -- ...`; fleet execution legs (`execution leg --cpu`) already hold that slot and
#     publish the receipt through `.github/fleet-leg`'s `owed-record` hook (`tools/owed-run-report.sh`).
#   - Recovery: on a manual Linux run (or if hook publication failed), certify and publish the printed
#     receipt directory with `tools/owed-run.sh record <receipt-dir>` (or `--print` + `add`).
#
# Exit status: 0 every requested target passed, 1/nonzero a target failed or was interrupted,
#              2 usage, host or setup refusal (no passing receipt written).
set -u
me=linux-shell-leg
die() { echo "$me: $*" >&2; exit 2; }
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P) || die "cannot resolve repository root"

host_os=$(uname -s 2>/dev/null || echo unknown)
[ "$host_os" = Linux ] || die "must run on Linux (host OS is '$host_os')"

jobs=4 all_harnesses=0
harness_targets=() dune_targets=()
while [ $# -gt 0 ]; do
  case $1 in
    --all-harnesses) all_harnesses=1 ;;
    --harness)
      [ $# -ge 2 ] && [ -n "$2" ] || die "--harness requires a member path"
      harness_targets+=("$2")
      shift ;;
    --dune)
      [ $# -ge 2 ] && [ -n "$2" ] || die "--dune requires a @DIR/runtest-NAME alias"
      dune_targets+=("$2")
      shift ;;
    --jobs)
      [ $# -ge 2 ] && [[ $2 =~ ^[1-9][0-9]*$ ]] || die "--jobs requires a positive integer"
      jobs=$2
      shift ;;
    -h | --help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument '$1'" ;;
  esac
  shift
done

if [ "$all_harnesses" = 1 ] && [ ${#harness_targets[@]} -gt 0 ]; then
  die "--all-harnesses and --harness cannot be combined"
fi
if [ "$all_harnesses" = 0 ] && [ ${#harness_targets[@]} -eq 0 ] && [ ${#dune_targets[@]} -eq 0 ]; then
  die "specify at least one of --all-harnesses, --harness MEMBER, or --dune @DIR/runtest-NAME"
fi

valid_harness_member() {
  local LC_ALL=C
  case $1 in
    '' | /* | *..* | *//* | *[!A-Za-z0-9_./-]*) return 1 ;;
    *.sh) return 0 ;;
    *) return 1 ;;
  esac
}

valid_runtest_alias() {
  local LC_ALL=C name
  case $1 in
    '' | *[!A-Za-z0-9_@./-]* | *..* | *//* | @/* | *@*@*) return 1 ;;
    @?*/runtest-?*) ;;
    *) return 1 ;;
  esac
  name=${1##*/}
  case $name in runtest-?*) return 0 ;; *) return 1 ;; esac
}

find_dune() {
  if ! command -v dune >/dev/null 2>&1; then
    if ! command -v opam >/dev/null 2>&1; then
      for prefix in /opt/homebrew/bin /usr/local/bin; do
        if [ -x "$prefix/opam" ]; then PATH=$prefix:$PATH; break; fi
      done
    fi
    if [ -r "$root/tools/opam-env.sh" ]; then
      # shellcheck source=tools/opam-env.sh
      . "$root/tools/opam-env.sh" || return 1
    fi
    command -v dune >/dev/null 2>&1 || return 1
  fi
}

# Validate harness members against tools/test-harnesses.sh --list.
if [ ${#harness_targets[@]} -gt 0 ]; then
  manifest_list=$("${BASH:-bash}" "$root/tools/test-harnesses.sh" --list 2>/dev/null) ||
    die "could not read harness manifest from tools/test-harnesses.sh --list"
  manifest_nl=$'\n'$manifest_list$'\n'
  seen_h=$'\n'
  for m in "${harness_targets[@]}"; do
    valid_harness_member "$m" || die "invalid harness member '$m'"
    case $manifest_nl in
      *$'\n'"$m"$'\n'*) ;;
      *) die "harness member '$m' is not in tools/test-harnesses.sh --list" ;;
    esac
    case $seen_h in
      *$'\n'"$m"$'\n'*) die "duplicate harness member '$m'" ;;
    esac
    seen_h=$seen_h$m$'\n'
  done
fi

# Validate Dune targets against canonical syntax and `dune show aliases`.
# Require dune unconditionally so real-dune and toolchain harnesses cannot silently skip.
find_dune || die "dune is not on PATH and tools/opam-env.sh did not supply it"
if [ "$all_harnesses" = 1 ]; then
  command -v ocamlc >/dev/null 2>&1 || die "ocamlc is not on PATH (required for --all-harnesses)"
elif [ ${#harness_targets[@]} -gt 0 ]; then
  toolchain_list=$("${BASH:-bash}" "$root/tools/test-harnesses.sh" --toolchain --list 2>/dev/null || true)
  toolchain_nl=$'\n'$toolchain_list$'\n'
  for m in "${harness_targets[@]}"; do
    case $toolchain_nl in
      *$'\n'"$m"$'\n'*)
        command -v ocamlc >/dev/null 2>&1 || die "ocamlc is not on PATH (required for toolchain harness '$m')"
        break ;;
    esac
  done
fi
if [ ${#dune_targets[@]} -gt 0 ]; then
  seen_d=$'\n'
  for a in "${dune_targets[@]}"; do
    valid_runtest_alias "$a" || die "invalid Dune alias '$a' (expected @DIR/runtest-NAME)"
    case $seen_d in
      *$'\n'"$a"$'\n'*) die "duplicate Dune alias '$a'" ;;
    esac
    seen_d=$seen_d$a$'\n'
    dir=${a%/*} dir=${dir#@} name=${a##*/}
    aliases_out=$(cd "$root" && dune show aliases --root . "$dir" 2>&1 </dev/null) ||
      die "dune could not list aliases in $dir: $aliases_out"
    printf '%s\n' "$aliases_out" | grep -qxF -- "$name" ||
      die "Dune alias '$a' does not exist in this tree"
  done
fi

# Build canonical sorted comma-separated target list.
raw_targets=()
if [ "$all_harnesses" = 1 ]; then
  raw_targets+=(harnesses)
else
  for m in ${harness_targets[@]+"${harness_targets[@]}"}; do
    raw_targets+=("member:$m")
  done
fi
for a in ${dune_targets[@]+"${dune_targets[@]}"}; do
  raw_targets+=("$a")
done
canonical_targets=$(printf '%s\n' "${raw_targets[@]}" | LC_ALL=C sort -u | tr '\n' ',')
canonical_targets=${canonical_targets%,}

# Read checkout head and dirty state with repository-local Git env vars cleared.
git_clean() {
  (
    git_vars=$(git -C "$root" rev-parse --local-env-vars 2>/dev/null) || exit 1
    # shellcheck disable=SC2086
    unset $git_vars
    GIT_OPTIONAL_LOCKS=0 git -C "$root" "$@"
  )
}
top=$(git_clean rev-parse --show-toplevel 2>/dev/null) || die "$root is not a Git checkout"
top=$(CDPATH= cd -- "$top" && pwd -P) || die "cannot resolve Git top level"
[ "$top" = "$root" ] || die "$root is below Git top level $top"

head1= head2= dirty=
for _ in 1 2 3 4 5; do
  head1=$(git_clean rev-parse --verify HEAD 2>/dev/null) || die "cannot read HEAD commit"
  dirty=$(git_clean status --porcelain --untracked-files=all --ignore-submodules=none 2>/dev/null) ||
    die "cannot read git status"
  head2=$(git_clean rev-parse --verify HEAD 2>/dev/null) || die "cannot read HEAD commit"
  [ "$head1" = "$head2" ] && break
done
[ "$head1" = "$head2" ] || die "HEAD moved while recording launch state"

# Allocate receipt directory outside the worktree.
state_root=${OCANNL_TOOL_TEST_RUNS:-${HOME:-}/.ocannl-test-runs}
[ -n "$state_root" ] && [ "$state_root" != "/.ocannl-test-runs" ] ||
  die "cannot determine receipt directory (set HOME or OCANNL_TOOL_TEST_RUNS)"
case $state_root in
  /*) ;;
  *) state_root=$root/$state_root ;;
esac
mkdir -p "$state_root/linux-shell" || die "cannot create $state_root/linux-shell"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
receipt_dir=$(mktemp -d "$state_root/linux-shell/$stamp.XXXXXX") ||
  die "cannot create receipt directory under $state_root/linux-shell"

wt_key="$(basename -- "$root" | tr -c 'A-Za-z0-9' '_')$(printf %s "$root" | cksum | awk '{print $1}')"
last_ptr="$state_root/linux-shell-last-$wt_key"

{
  printf '%s\n' linux-shell >"$receipt_dir/kind" &&
  printf '%s\n' "$head1" >"$receipt_dir/head" &&
  printf '%s' "$dirty" >"$receipt_dir/dirty" &&
  printf '%s\n' "$host_os" >"$receipt_dir/os" &&
  printf '%s\n' "$canonical_targets" >"$receipt_dir/targets" &&
  printf '%s\n' "$root" >"$receipt_dir/wt"
} || die "cannot write receipt metadata in $receipt_dir"

if [ -n "${FLEET_LEG_RUN_RECORD:-}" ]; then
  {
    printf '%s\n' "$FLEET_LEG_RUN_RECORD" >"$receipt_dir/fleet-leg-run-record" &&
    printf '%s\n' "$canonical_targets" >"$receipt_dir/fleet-leg-targets"
  } || die "cannot write fleet leg binding in $receipt_dir"
  if [ -n "${FLEET_LEG_RECEIPT_LINK:-}" ] &&
     [ ! -e "$FLEET_LEG_RECEIPT_LINK" ] && [ ! -L "$FLEET_LEG_RECEIPT_LINK" ]; then
    ln -s "$receipt_dir" "$FLEET_LEG_RECEIPT_LINK" ||
      die "cannot link receipt pointer $FLEET_LEG_RECEIPT_LINK"
  fi
fi

# Isolate child harnesses and Dune actions from the outer fleet leg's receipt variables.
unset FLEET_LEG_RUN_RECORD FLEET_LEG_RECEIPT_LINK FLEET_LEG_RECEIPT_DIR

printf '%s\n' "$receipt_dir" >"$last_ptr.tmp.$$" && mv -f "$last_ptr.tmp.$$" "$last_ptr" ||
  die "cannot update last receipt pointer $last_ptr"

write_exit() {
  printf '%s\n' "$1" >"$receipt_dir/exit.tmp.$$" && mv -f "$receipt_dir/exit.tmp.$$" "$receipt_dir/exit"
}
on_signal() {
  write_exit "$1" || :
  echo "$me: interrupted (exit $1; receipt: $receipt_dir)" >&2
  exit "$1"
}
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

verify_harness_output() {
  local out_file=$1 allow_file=$receipt_dir/allowed-skips.txt
  : >"$allow_file"
  if [ -r "$root/tools/harness-skips-ci-shell.txt" ]; then
    cat "$root/tools/harness-skips-ci-shell.txt" >>"$allow_file"
  fi
  if ! command -v zsh >/dev/null 2>&1; then
    printf 'scripts/test-setup-ocaml-env.sh\tsourcing (b-d): zsh legs\n' >>"$allow_file"
  fi
  awk -v allow_file="$allow_file" '
    FILENAME == allow_file {
      if ($0 ~ /^[[:space:]]*(#|$)/) next
      allowed[$0] = 1
      next
    }
    /^HARNESS [^ ]+ \(/ {
      cur = $2
      if (!(cur in seen)) { seen[cur] = 1; order[++n] = cur; pass[cur] = 0 }
      next
    }
    cur != "" && /^PASS[[:space:]]/ { pass[cur]++; next }
    cur != "" && /^SKIP  / {
      lbl = substr($0, 7)
      key = cur "\t" lbl
      if (!(key in allowed)) {
        printf "linux-shell-leg: REFUSED unexpected skip in %s: %s\n", cur, lbl
        bad = 1
      }
      next
    }
    END {
      if (n == 0) {
        print "linux-shell-leg: REFUSED harness run produced no HARNESS blocks"
        exit 1
      }
      for (i = 1; i <= n; i++) {
        m = order[i]
        if (pass[m] < 1) {
          printf "linux-shell-leg: REFUSED %s reported 0 PASS checks (skip-only or empty run)\n", m
          bad = 1
        }
      }
      exit bad ? 1 : 0
    }
  ' "$allow_file" "$out_file" >&2
}

cd "$root" || { write_exit 2 || :; die "cannot enter $root"; }

rc=0
if [ ${#dune_targets[@]} -gt 0 ]; then
  echo "$me: running Dune shell tests: ${dune_targets[*]}"
  env OCANNL_BACKEND=cc dune build --root . -j "$jobs" "${dune_targets[@]}" || rc=$?
fi

if [ "$rc" = 0 ] && { [ "$all_harnesses" = 1 ] || [ ${#harness_targets[@]} -gt 0 ]; }; then
  h_out="$receipt_dir/harnesses.out"
  if [ "$all_harnesses" = 1 ]; then
    echo "$me: running whole harness tier (--jobs $jobs)"
    "${BASH:-bash}" "$root/tools/test-harnesses.sh" --jobs "$jobs" >"$h_out" 2>&1 || rc=$?
  else
    h_args=(--jobs "$jobs")
    for m in "${harness_targets[@]}"; do
      h_args+=(--member "$m")
    done
    echo "$me: running harness members: ${harness_targets[*]}"
    "${BASH:-bash}" "$root/tools/test-harnesses.sh" "${h_args[@]}" >"$h_out" 2>&1 || rc=$?
  fi
  cat "$h_out"
  if [ "$rc" = 0 ]; then
    verify_harness_output "$h_out" || rc=1
  fi
fi

if ! write_exit "$rc"; then
  echo "$me: FAILED to record completion status in $receipt_dir/exit" >&2
  exit 2
fi
if [ "$rc" = 0 ]; then
  echo "$me: passed (targets: $canonical_targets; receipt: $receipt_dir)"
else
  echo "$me: FAILED (exit $rc; targets: $canonical_targets; receipt: $receipt_dir)" >&2
fi
exit "$rc"
