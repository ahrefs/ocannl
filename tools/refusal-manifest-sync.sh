#!/usr/bin/env bash
# Regenerate the refusal-control manifest and its goldens after a scanner diagnostic is reworded
# (gh-ocannl-1429): one command where a rewording used to cost three or four test runs.
# Usage: tools/refusal-manifest-sync.sh [--dry-run] [--rounds N] [--base REV] [SOURCE.ml ...]
#
# 1. tools/refusal_manifest_sync.exe extracts every catalogued scanner's diagnostics and rewrites,
#    in test/support/refusal_control_manifest.ml, each stale `raw_entries` row (the extraction, in
#    order), each `raw_direct_evidence`/`raw_catalogue_only` key of a direct failure the rewording
#    replaced, and each `raw_direct_evidence` label naming a control claim reworded in that
#    scanner's own source. A key or label follows its diagnostic only where the registered row
#    and the extraction pair it one for one and no other scanner shares its text; a control claim
#    reworded or removed in another test source changed since REV (default HEAD) is never followed
#    by text (its label may back another scanner's evidence) -- anything of that kind is printed
#    UNRESOLVED, with the new text as a hint, and left for a hand edit: then NOTHING is written or run (exit 1),
#    so the next invocation, once the items are settled, still sees every source the rewording
#    touched. Every argument is checked before anything is written (exit 2).
# 2. ocamlformat reformats the manifest, and its diff is shown.
# 3. Every test/operations golden holding a replaced marker has it swapped for its successor,
#    and a row `<label>: true` of a reworded claim (no substitutions) for the new label -- the
#    audit's coverage claims read the goldens for the NEW markers, so without this its first run
#    could not pass. Those goldens, each touched or named source's catalogued ones (the
#    refusal_control_scan_cases stanza's argument list) and the audit's own then run through
#    tools/test-run.sh, the first round forced (`dune build --force`), so every one of them runs
#    and says its piece even where nothing it reads changed. A source whose manifest entries differ
#    from REV's (a hand edit, say) has its goldens re-run too, as does a named SOURCE.
# 4. To a fixpoint (gh-ocannl-1571): while a run fails and dune's promotion list, recorded by that
#    run, holds corrections of those goldens, they are promoted (tools/promote.sh) and the goldens
#    run again -- a promotion can change what another golden reads (the audit reads the scanners'
#    goldens), so one round's correction can surface the next. A run that passes ends it (exit 0);
#    one that fails with nothing to promote is a failed check no promotion settles, and it says so
#    (the run's exit status); still failing with corrections after N runs (--rounds, default 5,
#    at least 2) is STALE (exit 3), naming what was left. Each golden promoted is named with its
#    round, and a correction dune lists for a file outside the set is named, never promoted.
# 5. Each direct failure a row lists with no `raw_direct_evidence` or `raw_catalogue_only` entry
#    is offered (Refusal_control_manifest.offer_lines, read back from the runs' logs): the entry
#    naming each passed claim whose refused run printed it, ready to paste, or the word that no
#    claim of the run answers for it. Which claim answers is a judgment, so it is never written.
# --dry-run prints the plan and changes nothing. The plan is a point-in-time reading of the source
# text: what a run prints is the golden's business, and the confirming run is what says it held.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
cd "$root" || exit 2
dry_run=0 rounds=5 engine_args=()
while [ $# -gt 0 ]; do
  case $1 in
    --rounds)
      case ${2-} in
        '' | *[!0-9]*) echo "refusal-manifest-sync.sh: --rounds needs a number of runs" >&2; exit 2 ;;
      esac
      [ "$2" -ge 2 ] || { echo "refusal-manifest-sync.sh: --rounds is at least 2 (run, promote, run)" >&2; exit 2; }
      rounds=$2
      shift 2 ;;
    *)
      [ "$1" != --dry-run ] || dry_run=1
      engine_args+=("$1")
      shift ;;
  esac
done
command -v dune >/dev/null 2>&1 || . tools/opam-env.sh
# Credentials never reach dune, which records every spawned process's environment in
# `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies.
# shellcheck source=credential-env.sh
. tools/credential-env.sh
eval "$(credential_env_scrub_text)" || {
  echo "refusal-manifest-sync.sh: cannot remove credential variables:$credential_env_left" >&2
  exit 2
}
manifest=test/support/refusal_control_manifest.ml
before=$(git hash-object "$manifest") || exit 2
# Built apart, so that a build failure (dune's exit 1) is never read as the engine's UNRESOLVED.
dune build tools/refusal_manifest_sync.exe || exit 2
plan=$(dune exec --no-print-directory tools/refusal_manifest_sync.exe -- ${engine_args[@]+"${engine_args[@]}"})
status=$?
printf '%s\n' "$plan"
case $status in 0 | 1) ;; *) exit "$status" ;; esac
[ "$dry_run" = 0 ] || exit "$status"
if [ "$(git hash-object "$manifest")" != "$before" ]; then
  if command -v ocamlformat >/dev/null 2>&1; then
    ocamlformat --inplace "$manifest" ||
      echo "refusal-manifest-sync.sh: ocamlformat failed on $manifest; run \`dune fmt\`" >&2
  else
    echo "refusal-manifest-sync.sh: no ocamlformat on PATH; run \`dune fmt\` before pushing" >&2
  fi
  git --no-pager diff -- "$manifest"
fi
if [ "$status" = 1 ]; then
  echo "refusal-manifest-sync.sh: nothing was written; settle the UNRESOLVED items by hand," \
    "then run this again" >&2
  exit 1
fi
aliases=()
goldens=()
while IFS= read -r line; do
  case $line in
    'rerun: @'*)
      alias=${line#rerun: }
      aliases+=("$alias")
      dir=${alias#@}
      dir=${dir%/runtest-*}
      goldens+=("$dir/${alias##*/runtest-}.expected") ;;
  esac
done <<EOF
$plan
EOF
[ "${#aliases[@]}" -gt 0 ] || exit 0
echo "refusal-manifest-sync.sh: running ${#aliases[@]} golden(s)"
logs=()
round=1
force=(--force)
while :; do
  tools/test-run.sh run build "${force[@]}" "${aliases[@]}"
  status=$?
  force=()
  run=$(tools/test-run.sh paths run last) || {
    echo "refusal-manifest-sync.sh: cannot find the run just made" >&2
    exit 2
  }
  logs+=("$run/log")
  [ "$status" != 0 ] || break
  # The corrections this run's dune listed, split into the set's goldens and the rest.
  if [ ! -f "$run/promotions" ]; then
    echo "refusal-manifest-sync.sh: round $round failed and its run recorded no promotion list:" \
      "promote by hand (tools/promote.sh) and run this again (log: $run/log)" >&2
    break
  fi
  pending=() foreign=()
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    case " ${goldens[*]} " in
      *" $path "*) pending+=("$path") ;;
      *) foreign+=("$path") ;;
    esac
  done <"$run/promotions"
  for path in ${foreign[@]+"${foreign[@]}"}; do
    echo "refusal-manifest-sync.sh: dune lists a correction of $path, outside the goldens this" \
      "sync runs: left unpromoted"
  done
  if [ "${#pending[@]}" = 0 ]; then
    echo "refusal-manifest-sync.sh: round $round failed with no correction of its goldens to" \
      "promote: a check failed that no promotion settles (log: $run/log)" >&2
    break
  fi
  if [ "$round" -ge "$rounds" ]; then
    echo "refusal-manifest-sync.sh: STALE: after $rounds runs these goldens still have" \
      "corrections (log: $run/log):" >&2
    printf '  %s\n' "${pending[@]}" >&2
    status=3
    break
  fi
  echo "refusal-manifest-sync.sh: round $round: promoting ${pending[*]}"
  tools/promote.sh "${pending[@]}" || exit $?
  round=$((round + 1))
done
dune exec --no-print-directory tools/refusal_manifest_sync.exe -- --offers "${logs[@]}" || exit 2
exit "$status"
