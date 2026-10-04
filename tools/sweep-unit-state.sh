#!/usr/bin/env bash
# Sweep comparison cursor, sourced by tools/sweep.sh and direct fixtures.
# The caller owns UNIT_STATES, TARGET, SLOW, REF, MAIN and full_sha, and provides
# say and die. Only sweep_state_update publishes state; sourcing does no I/O.

# The history remains the append-only coverage record. This smaller state is
# the comparison cursor for one exact unit scope: its immediately previous
# verdict, and the previous failing fingerprint plus the commit that last
# touched each failing golden. A target smoke must not become the predecessor
# of a full sweep (nor a weekday run the predecessor of a slow one), hence all
# scope columns participate in the key.
sweep_state_path() { # machine backend -> path
  local raw readable crc
  # The requested logical ref is part of the experiment scope. A one-off old
  # or feature ref must not become origin/master's green/red predecessor even
  # when both happen to resolve to related commits.
  raw=$(printf '%s\t%s\t%s\t%s\t%s' "$1" "$2" "${TARGET:-<all>}" "$SLOW" "$REF")
  readable=$(printf '%s' "$1-$2-${TARGET:-all}-$SLOW-$REF" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-96)
  crc=$(printf '%s' "$raw" | cksum | awk '{print $1}')
  printf '%s/%s-%s.state' "$UNIT_STATES" "$readable" "$crc"
}

sweep_state_field() { # state key -> first value
  awk -F '\t' -v key="$2" '$1 == key { print $2; exit }' "$1"
}

sweep_state_goldens_from_log() { # log destination -- source-tree paths proved to have failed a diff
  local log=$1 destination=$2 candidates token
  candidates=$destination.candidates.$$
  : >"$destination" || die "cannot stage failing golden paths"

  # An ordinary `(test)` failure names its expected file directly. Explicit
  # rules name only their dune stanza, but a diff that ACTUALLY RAN and found a
  # mismatch emits a resolved `diff --git` header. Reading that header—rather
  # than guessing from the stanza—distinguishes a failed diff from an earlier
  # command in a run-then-diff `progn`, and naturally carries `%{read:...}`
  # expansions plus PPX's `*_expected.ml` naming.
  {
    sed -n 's/^File "\([^"]*\.expected\)".*/\1/p' "$log"
    sed -n 's|^diff --git a/_build/default/\([^ ]*\) b/_build/default/.*$|\1|p' "$log"
    # Inline ppx_expect compares the source baseline directly with a generated
    # .corrected file, so its first operand has no _build/default prefix.
    sed -n 's|^diff --git a/\([^ ]*\) b/_build/default/[^ ]*\.corrected$|\1|p' "$log"
  } | sort -u >"$candidates" || die "cannot extract proven failing goldens"
  while IFS= read -r token; do
    [ -n "$token" ] || continue
    token=${token#./}
    git -C "$MAIN" cat-file -e "$full_sha:$token" 2>/dev/null || continue
    printf '%s\n' "$token" >>"$destination" ||
      die "cannot stage failing golden path $token"
  done <"$candidates"
  sort -u "$destination" -o "$destination" || die "cannot normalize failing golden paths"
  rm -f "$candidates"
}

sweep_state_update() { # machine backend outcome [fingerprint] [log]
  local machine=$1 backend=$2 outcome=$3 fp=${4:-} log=${5:-}
  local label state stage previous_verdict previous_failure_ref
  local previous_fp current_goldens golden_paths path commit old_commit short old_short
  label=$machine/$backend
  # skip/gate/error/timeout are recorded outcomes but not verdicts: they judged no
  # test result. Letting one replace a prior green would hide the next red's
  # regression transition merely because a machine slept or a run timed out.
  case $outcome in skip | gate | error | timeout) return 0 ;; esac
  state=$(sweep_state_path "$machine" "$backend")
  stage=$state.stage.$$
  previous_fp=$state.previous-fingerprint.$$
  current_goldens=$state.current-goldens.$$
  golden_paths=$state.golden-paths.$$
  : >"$current_goldens" || die "cannot stage unit state for $label"

  if [ -f "$state" ]; then
    [ "$(head -1 "$state")" = "$(printf 'schema\t1')" ] ||
      die "$state has an unknown unit-state schema"
    previous_verdict=$(sweep_state_field "$state" last_verdict)
    [ -n "$previous_verdict" ] || die "$state has no last verdict"
    previous_failure_ref=$(sweep_state_field "$state" last_failure_ref)
    if [ -n "$previous_failure_ref" ] &&
       ! awk -F '\t' '$1 == "fingerprint" { found=1 } END { exit !found }' "$state"; then
      die "$state has a previous failure but no fingerprint"
    fi
  else
    previous_verdict=
    previous_failure_ref=
  fi

  if [ "$outcome" = fail ]; then
    [ -n "$fp" ] && [ -s "$fp" ] || die "no current failure fingerprint for $label"
    [ -n "$log" ] && [ -f "$log" ] || die "no current failure log for $label"
    case $previous_verdict in
      pass | incremental-pass | legacy-pass)
        say "  $label: REGRESSION OR FIX DID NOT TAKE -- previous verdict was $previous_verdict"
        ;;
    esac

    # Compare with the previous FAILURE, even if green or unavailable runs sat
    # between it and this one. That is the experiment whose identity matters:
    # same failure vs a moving one, not merely same as yesterday's outcome.
    if [ -n "$previous_failure_ref" ]; then
      awk -F '\t' '$1 == "fingerprint" { sub(/^[^\t]*\t/, ""); print }' \
        "$state" >"$previous_fp" || die "cannot read the previous fingerprint for $label"
      if [ -s "$previous_fp" ] && ! cmp -s "$previous_fp" "$fp"; then
        short=$(printf '%s' "$previous_failure_ref" | cut -c1-8)
        say "  $label: fingerprint moved since the previous failure at $short"
      fi
    fi

    sweep_state_goldens_from_log "$log" "$golden_paths"
    while IFS= read -r path; do
      [ -n "$path" ] || continue
      commit=$(git -C "$MAIN" log -1 --format=%H "$full_sha" -- "$path" 2>/dev/null) ||
        die "cannot read golden history for $path"
      [ -n "$commit" ] || continue
      printf 'golden\t%s\t%s\n' "$commit" "$path" >>"$current_goldens" ||
        die "cannot stage golden state for $label"
      if [ -f "$state" ]; then
        old_commit=$(awk -F '\t' -v path="$path" \
          '$1 == "golden" && $3 == path { print $2; exit }' "$state")
        if [ -n "$old_commit" ] && [ "$old_commit" != "$commit" ]; then
          short=$(printf '%s' "$commit" | cut -c1-8)
          old_short=$(printf '%s' "$old_commit" | cut -c1-8)
          say "  $label: REGRESSION OR FIX DID NOT TAKE -- $path last changed at $short (previous failing copy: $old_short)"
        fi
      fi
    done <"$golden_paths"
  fi

  {
    printf 'schema\t1\n'
    printf 'last_verdict\t%s\n' "$outcome"
    printf 'last_ref\t%s\n' "$full_sha"
    if [ "$outcome" = fail ]; then
      printf 'last_failure_ref\t%s\n' "$full_sha"
      while IFS= read -r line; do printf 'fingerprint\t%s\n' "$line"; done <"$fp"
      cat "$current_goldens"
    elif [ -f "$state" ] && [ -n "$previous_failure_ref" ]; then
      awk -F '\t' '$1 == "last_failure_ref" || $1 == "fingerprint" || $1 == "golden"' "$state"
    fi
  } >"$stage" && mv "$stage" "$state" ||
    die "cannot publish unit state for $label"
  rm -f "$previous_fp" "$current_goldens" "$golden_paths"
}

