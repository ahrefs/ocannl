#!/usr/bin/env bash
# Hermetic fixtures for tools/merge-build-replay.py over a synthetic repository: which first-parent
# merges each input set would build (a path both sides changed, by either name of a rename; the
# any-input trigger; an octopus merge and a merge inside a side branch left out), the paths it
# reports, the built-in sets read from where the fixture defines them (its own
# .github/ship-pr-merge-build and tools/merge-build-check.sh), the shell-tier replay agreeing with
# merge-build-check.sh's own selection merge by merge (its fail-closed quoted-path fallback
# included), an unreadable merge base failing the run, and this repository's definitions parsing.
# Needs git and python3; a host without either skips every leg.
# Usage: tools/test-merge-build-replay.sh [--keep] [--only REGEX]
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require git python3
harness_scratch test-merge-build-replay
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/merge-build-replay.py
# The fixtures' own git: no ambient configuration, identity or repository selection.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_COMMON_DIR GIT_CEILING_DIRECTORIES
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid \
  GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid

# The history, on `main` (first parent), each feature branched from main's tip, main then moving:
#   m1  both sides edit x.ml                                  (an OCaml path shared)
#   m2  both sides edit n.md                                  (a Markdown path shared)
#   m3  main renames r.txt to r.sh, the side edits r.txt      (shared by its other name)
#   m4  main edits a.ml, the side b.ml                        (inputs on each side, none shared)
#   m5  the side edits c.ml, main does not move               (one side empty)
#   m6  an octopus merge of two branches                      (not a two-parent merge)
#   m7  both sides edit s/t.sh, the side holding a merge      (`*` crossing `/`; the side's own
#                                                              merge is not first-parent)
# Commits are dated a day apart from 2001-01-02 (m1 on 2001-01-05, m2 on 2001-01-08), so
# --since/--until select by date.
R=$TMP/repo
day=0
# stamp: sets d to the next day's date, day N being 2001-01-01 plus N days, midnight UTC.
stamp() { day=$((day + 1)) d="@$((978307200 + day * 86400)) +0000"; }
commit() { # MESSAGE
  stamp
  git -C "$R" add -A && GIT_COMMITTER_DATE=$d GIT_AUTHOR_DATE=$d git -C "$R" commit -q -m "$1"
}
# first FILE / last FILE: an edit at either end, so both sides' edits merge cleanly.
first() { { echo first; tail -n +2 "$R/$1"; } >"$R/$1.new" && mv "$R/$1.new" "$R/$1"; }
last() { echo last >>"$R/$1"; }
merge() { # MESSAGE BRANCH...: main merges the branches, no fast-forward
  local msg=$1
  shift
  stamp
  git -C "$R" checkout -q main &&
    GIT_COMMITTER_DATE=$d GIT_AUTHOR_DATE=$d git -C "$R" merge -q --no-ff -m "$msg" "$@" >/dev/null
}
side() { git -C "$R" checkout -q -b "$1" main; }
build_fixture() {
  mkdir -p "$R/tools" "$R/.github" "$R/s" &&
    cp "$HERE/merge-build-check.sh" "$R/tools/" &&
    printf '# the fixture config\ncommand tools/merge-build-check.sh\ninput *\ntimeout 60\n' \
      >"$R/.github/ship-pr-merge-build" &&
    for f in x.ml a.ml b.ml c.ml n.md r.txt s/t.sh o1.txt o2.txt; do printf '1\n2\n3\n4\n5\n6\n' >"$R/$f"; done &&
    git -C "$R" init -q && git -C "$R" checkout -q -b main && commit init || return 1
  side f1 && last x.ml && commit f1 && git -C "$R" checkout -q main && first x.ml && commit main1 &&
    merge m1 f1 || return 1
  side f2 && last n.md && commit f2 && git -C "$R" checkout -q main && first n.md && commit main2 &&
    merge m2 f2 || return 1
  side f3 && last r.txt && commit f3 && git -C "$R" checkout -q main && git -C "$R" mv r.txt r.sh &&
    commit main3 && merge m3 f3 || return 1
  side f4 && last b.ml && commit f4 && git -C "$R" checkout -q main && first a.ml && commit main4 &&
    merge m4 f4 || return 1
  side f5 && last c.ml && commit f5 && merge m5 f5 || return 1
  side f6a && last o1.txt && commit f6a && side f6b && last o2.txt && commit f6b &&
    git -C "$R" checkout -q main && last a.ml && commit main6 && merge m6 f6a f6b || return 1
  side f7 && side f7b && echo extra >"$R/extra.txt" && commit f7b && git -C "$R" checkout -q f7 &&
    last s/t.sh && commit f7 && stamp &&
    GIT_COMMITTER_DATE=$d GIT_AUTHOR_DATE=$d git -C "$R" merge -q --no-ff -m inner f7b >/dev/null &&
    git -C "$R" checkout -q main && first s/t.sh && commit main7 && merge m7 f7 || return 1
}
FIXTURE=ok
build_fixture || { report 1 'the synthetic history builds' "see $TMP (rerun with --keep)"; FIXTURE=; }

out=$TMP/out
replay() { # [ARG...]: the replay, in R unless --repo is given, into $out; rc set
  rc=0
  (cd "$R" && python3 "$SRC" "$@") >"$out" 2>&1 || rc=$?
}
has() { grep -qF -- "$1" "$out"; }
# row NAME: the summary table's row for set NAME, globs column dropped.
row() { awk -F' [|] ' -v n="| $1" '$1 == n { print $3 " " $4 " " $5 }' "$out" | tr -d ' |'; }
SETS=(--set 'ml=*.ml' --set 'md=@ml,*.md' --set shell-tier --set 'all=*')

if harness_leg 'each set counts the merges it would build, and what each step adds #replay'; then
  if [ -z "$FIXTURE" ]; then report 1 'replay counts' 'no fixture'; else
    replay "${SETS[@]}" main
    # m1..m7 listed, m6 not replayed; ml builds m1; md adds m2; shell-tier builds m3 (r.txt by its
    # other name) and m7 (s/t.sh), beyond md both; all builds m1 m2 m3 m7, beyond shell-tier m1 m2.
    if [ "$rc" = 0 ] && has 'merge-build-replay: 7 first-parent merge(s) of main, 6 replayed; not replayed: 1 with 3 parents' &&
       has 'merge-build-replay: trigger same-input (the default: .github/ship-pr-merge-build names none)' &&
       [ "$(row ml)" = 61- ] && [ "$(row md)" = 621 ] && [ "$(row shell-tier)" = 622 ] &&
       [ "$(row all)" = 642 ]; then
      report 0 'per-set counts and the beyond-the-previous-set column'
    else report 1 'per-set counts and the beyond-the-previous-set column' "exit $rc; $(cat "$out")"; fi
    # The paths listed under each set: from the second on, only those of the merges it adds.
    want=$'set md:\n  replayed 6, would build 2\n  beyond ml: 1 merge(s); the paths that triggered those, by merges:\n        1  n.md\n'
    want+=$'\nset shell-tier: *.sh  (tools/merge-build-check.sh)\n  replayed 6, would build 2\n'
    want+=$'  beyond md: 2 merge(s); the paths that triggered those, by merges:\n        1  r.txt\n        1  s/t.sh'
    got=$(sed -n '/^set md:/,/^set all:/p' "$out" | sed '/^set all:/d; s/^set md:.*/set md:/')
    if [ "$got" = "$want" ]; then
      report 0 'the paths that triggered the merges each set adds'
    else report 1 'the paths that triggered the merges each set adds' "got:
$got
want:
$want"; fi
  fi
fi

if harness_leg 'any-input builds on inputs each side changed, never on an empty side #replay'; then
  if [ -z "$FIXTURE" ]; then report 1 'any-input' 'no fixture'; else
    replay --trigger any-input --set 'ml=*.ml' --verbose main
    # m1 (x.ml on both) and m4 (a.ml against b.ml); not m5, whose main side changed nothing.
    if [ "$rc" = 0 ] && has 'merge-build-replay: trigger any-input (--trigger)' && [ "$(row ml)" = 62- ] &&
       grep -qE '^  [0-9a-f]{10} m4: ml$' "$out" && grep -qE '^  [0-9a-f]{10} m5: -$' "$out"; then
      report 0 'any-input builds m1 and m4, not the one-sided m5'
    else report 1 'any-input builds m1 and m4, not the one-sided m5' "exit $rc; $(cat "$out")"; fi
  fi
fi

if harness_leg 'since and until select merges by date #replay'; then
  if [ -z "$FIXTURE" ]; then report 1 'date window' 'no fixture'; else
    # m1 is committed on 2001-01-05, m2 on 2001-01-08: a window from 2001-01-06 starts at m2.
    replay --since 2001-01-06T00:00:00Z --set 'ml=*.ml' main
    if [ "$rc" = 0 ] && has "6 first-parent merge(s) of main --since '2001-01-06T00:00:00Z', 5 replayed" &&
       [ "$(row ml)" = 50- ]; then
      report 0 'a window from 2001-01-06 leaves m1 out'
    else report 1 'a window from 2001-01-06 leaves m1 out' "exit $rc; $(cat "$out")"; fi
  fi
fi

# The replay's shell-tier set must decide as the gate itself does: run merge-build-check.sh --plan
# on every merge the replay reports and compare. A clone, so HEAD moves outside the shared fixture,
# extended by m8: main adds q"uote.txt, a name git quotes, while the side edits b.ml -- nothing
# shared, but the gate fails closed on the quoted name and selects the shell tier.
if harness_leg 'the shell-tier replay agrees with merge-build-check.sh on every merge #gate'; then
  if [ -z "$FIXTURE" ]; then report 1 'shell-tier agreement' 'no fixture'; else
    C=$TMP/gate
    rm -rf "$C" && git clone -q "$R" "$C" 2>/dev/null
    shared=$R R=$C
    side f8 && first b.ml && commit f8 && git -C "$R" checkout -q main && printf 'x\n' >"$R/q\"uote.txt" &&
      commit main8 && merge m8 f8 || bad=' (m8 could not be built)'
    R=$shared
    replay --repo "$C" --set shell-tier --verbose main
    n=0 bad=${bad-} selected=0
    while read -r sha subject sets; do
      n=$((n + 1))
      git -C "$C" checkout -q --detach "$sha" 2>/dev/null || { bad+=" $sha(checkout)"; continue; }
      plan=$(cd "$C" && "$BASH" tools/merge-build-check.sh --plan 2>&1)
      case $plan in
        *'shell tier: SELECTED'*) gate=1 ;;
        *'shell tier: skipped'*) gate=0 ;;
        *) bad+=" $sha(plan: $plan)"; continue ;;
      esac
      case $sets in shell-tier) mine=1 ;; *) mine=0 ;; esac
      selected=$((selected + gate))
      [ "$gate" = "$mine" ] || bad+=" $subject(gate $gate, replay $mine)"
    done < <(sed -n 's/^  \([0-9a-f]\{10\}\) \([^:]*\): \(.*\)$/\1 \2 \3/p' "$out")
    if [ "$rc" = 0 ] && [ "$n" = 7 ] && [ "$selected" = 3 ] && [ -z "$bad" ]; then
      report 0 'the replay and the gate select the shell tier on the same 3 of 7 merges'
    else report 1 'the replay and the gate select the shell tier on the same 3 of 7 merges' \
      "exit $rc, $n merge(s) compared, gate selected $selected; disagreements:${bad:- none}; $(cat "$out")"; fi
    # Negative controls: a replay that ignores renames' other names loses m3, and one without the
    # quoted-path fallback loses m8; the comparison must see each.
    for control in 'rename-blind:m3:s/for n in (p, \*other.get(p, \[\]))/for n in (p,)/' \
                   'quote-blind:m8:s/if src == CHECK_PATH and quoted:/if False:/'; do
      what=${control%%:*} rest=${control#*:}
      merge_name=${rest%%:*} program=${rest#*:}
      label="negative control: a $what replay disagrees with the gate on $merge_name"
      M=$TMP/$what.py
      sed "$program" "$SRC" >"$M"
      if cmp -s "$SRC" "$M"; then
        report 1 "$label" 'the mutation matched nothing'
        continue
      fi
      rc=0
      (cd "$C" && python3 "$M" --set shell-tier --verbose main) >"$out" 2>&1 || rc=$?
      if [ "$rc" = 0 ] && grep -qE "^  [0-9a-f]{10} $merge_name: -\$" "$out" && [ "$(row shell-tier)" = 72- ]; then
        report 0 "$label"
      else report 1 "$label" "exit $rc; $(cat "$out")"; fi
    done
  fi
fi

# A merge base git cannot compute because an object is missing (a damaged or partial clone) is a
# failed read, exit 1 -- never a merge quietly left out as having no merge base.
if harness_leg 'a merge base that cannot be read fails the replay #replay'; then
  if [ -z "$FIXTURE" ]; then report 1 'missing object' 'no fixture'; else
    B=$TMP/broken
    rm -rf "$B" && git clone -q --no-hardlinks "$R" "$B" 2>/dev/null
    tip=$(git -C "$B" rev-parse origin/f1 2>/dev/null)
    obj=$B/.git/objects/${tip:0:2}/${tip:2}
    if [ -z "$tip" ] || [ ! -f "$obj" ]; then
      report 1 'a missing side commit exits 1' "no loose object for f1's tip ($tip) in the clone"
    else
      rm -f "$obj"
      replay --repo "$B" --set 'ml=*.ml' origin/main
      if [ "$rc" = 1 ] && has 'merge-build-replay: git merge-base' && ! has 'no merge base'; then
        report 0 'a missing side commit exits 1'
      else report 1 'a missing side commit exits 1' "exit $rc; $(cat "$out")"; fi
    fi
  fi
fi

# Paths are bytes, and printing them is the replay's to make safe: both sides add odd<0xff>.ml,
# which is not UTF-8, and new<LF>line.ml, whose newline would forge a report line. The replay
# decodes both and lists them escaped as git quotes them, never raw. Built with plumbing, so no
# file of either name touches the work tree (APFS refuses the first).
if harness_leg 'paths that are not UTF-8 or hold a newline are replayed and printed escaped #replay'; then
  E=$TMP/bytes
  odd=$'odd\xff.ml' nl=$'new\nline.ml'
  side_with() { # BRANCH CONTENT: a commit on BRANCH adding $odd and $nl with CONTENT, through the index
    local blob
    blob=$(printf '%s\n' "$2" | git -C "$E" hash-object -w --stdin) &&
      git -C "$E" update-index --add --cacheinfo "100644,$blob,$odd" &&
      git -C "$E" update-index --add --cacheinfo "100644,$blob,$nl" &&
      git -C "$E" commit -q -m "$1"
  }
  rm -rf "$E" && mkdir -p "$E" && printf '1\n' >"$E/a.ml" &&
    git -C "$E" init -q && git -C "$E" checkout -q -b main && git -C "$E" add -A &&
    git -C "$E" commit -q -m init && git -C "$E" branch f &&
    side_with main-odd main && git -C "$E" symbolic-ref HEAD refs/heads/f && git -C "$E" read-tree f &&
    side_with f-odd side && m=$(git -C "$E" commit-tree -p main -p f -m m-odd 'main^{tree}') &&
    git -C "$E" update-ref refs/heads/main "$m"
  built=$?
  rc=0
  python3 "$SRC" --repo "$E" --set 'ml=*.ml' main >"$out" 2>&1 || rc=$?
  label='the shared non-UTF-8 and newline paths build, and are listed escaped'
  if [ "$built" != 0 ]; then
    report 1 "$label" "the fixture did not build: $(cat "$out")"
  elif [ "$rc" = 0 ] && [ "$(row ml)" = 11- ] && grep -qxF '        1  "odd\377.ml"' "$out" &&
       grep -qxF '        1  "new\nline.ml"' "$out" && ! grep -qx 'line.ml"\{0,1\}' "$out" &&
       ! LC_ALL=C grep -q $'\xff' "$out"; then
    report 0 "$label"
  else report 1 "$label" "exit $rc; $(cat "$out")"; fi
  # A shallow clone hides boundary merges and merge bases: refused, never a short count.
  rm -rf "$TMP/shallow" && git clone -q --depth 1 "file://$E" "$TMP/shallow" 2>/dev/null
  rc=0
  python3 "$SRC" --repo "$TMP/shallow" --set 'ml=*.ml' HEAD >"$out" 2>&1 || rc=$?
  if [ "$rc" = 2 ] && has 'is a shallow clone, whose boundary hides merges and merge bases'; then
    report 0 'a shallow clone is refused'
  else report 1 'a shallow clone is refused' "exit $rc; $(cat "$out")"; fi
fi

# The built-in sets come from the replayed tree's own definitions, read at run time: change them
# (uncommitted, in a clone) and the replay follows.
if harness_leg 'built-in sets are read from where they are defined #definitions'; then
  if [ -z "$FIXTURE" ]; then report 1 'definitions' 'no fixture'; else
    D=$TMP/defs
    rm -rf "$D" && git clone -q "$R" "$D" 2>/dev/null
    printf 'command tools/merge-build-check.sh\ninput *.md\ninput *.txt\ntrigger any-input\n' >"$D/.github/ship-pr-merge-build"
    sed "s/^SHELL_GLOB='\\*\\.sh'\$/SHELL_GLOB='*.ml'/" "$D/tools/merge-build-check.sh" >"$D/check.new" &&
      mv "$D/check.new" "$D/tools/merge-build-check.sh"
    replay --repo "$D" origin/main
    # any-input from the config: config (*.md *.txt) builds where each side changed one: m2 (n.md)
    # and m3 (r.txt); shell-tier, now *.ml, builds m1 and m4.
    if [ "$rc" = 0 ] && has 'set config: *.md *.txt  (.github/ship-pr-merge-build)' &&
       has 'set shell-tier: *.ml  (tools/merge-build-check.sh)' &&
       has 'trigger any-input (.github/ship-pr-merge-build)' &&
       [ "$(row config)" = 62- ] && [ "$(row shell-tier)" = 622 ]; then
      report 0 'edited definitions change the default sets and trigger'
    else report 1 'edited definitions change the default sets and trigger' "exit $rc; $(cat "$out")"; fi
    printf "SHELL_GLOB='*.sh'\n" >>"$D/tools/merge-build-check.sh"
    replay --repo "$D" --set shell-tier origin/main
    if [ "$rc" = 2 ] && has "tools/merge-build-check.sh has 2 \`SHELL_GLOB='...'\` assignment(s), not exactly one"; then
      report 0 'an ambiguous shell-tier definition is refused'
    else report 1 'an ambiguous shell-tier definition is refused' "exit $rc; $(cat "$out")"; fi
    rm "$D/.github/ship-pr-merge-build"
    replay --repo "$D" --set config origin/main
    if [ "$rc" = 2 ] && has 'cannot read .github/ship-pr-merge-build'; then
      report 0 'a missing config is refused when its set is asked for'
    else report 1 'a missing config is refused when its set is asked for' "exit $rc; $(cat "$out")"; fi
    replay --repo "$D" --set 'ml=*.ml' origin/main
    if [ "$rc" = 0 ] && has 'trigger same-input (the default: no config)'; then
      report 0 'without a config, named sets replay under same-input'
    else report 1 'without a config, named sets replay under same-input' "exit $rc; $(cat "$out")"; fi
  fi
fi

if harness_leg 'usage errors exit 2 #usage'; then
  if [ -z "$FIXTURE" ]; then report 1 'usage' 'no fixture'; else
    for args in "--set nosuch" "--set ml=*.ml --set ml" "--set ml=a,,b" "--set ml=@ml" "--top -1" "--trigger sometimes"; do
      # shellcheck disable=SC2086 # each case is a word list, its globs unexpanded
      set -f
      replay $args main
      set +f
      if [ "$rc" = 2 ]; then report 0 "\`$args\` exits 2"; else report 1 "\`$args\` exits 2" "exit $rc; $(cat "$out")"; fi
    done
  fi
fi

# This repository's own definitions parse: a format change at either definition site that the
# reader no longer follows fails here, not in the next replay someone runs. Copied into a fresh
# one-commit repository, since the checkout itself may be shallow (CI's is, and the replay refuses
# one); an empty revision range reads no history.
if harness_leg "this repository's built-in sets parse #definitions"; then
  O=$TMP/own
  rm -rf "$O" && mkdir -p "$O/.github" "$O/tools" &&
    cp "$HERE/../.github/ship-pr-merge-build" "$O/.github/" && cp "$HERE/merge-build-check.sh" "$O/tools/" &&
    git -C "$O" init -q && git -C "$O" add -A && git -C "$O" commit -q -m own
  rc=0
  python3 "$SRC" --repo "$O" HEAD..HEAD >"$out" 2>&1 || rc=$?
  if [ "$rc" = 0 ] && has 'merge-build-replay: 0 first-parent merge(s) of HEAD..HEAD, 0 replayed' &&
     grep -q '^set config: .*(.github/ship-pr-merge-build)$' "$out" &&
     grep -q '^set shell-tier: .*(tools/merge-build-check.sh)$' "$out"; then
    report 0 "this repository's config and merge-build-check.sh give both built-in sets"
  else report 1 "this repository's config and merge-build-check.sh give both built-in sets" "exit $rc; $(cat "$out")"; fi
fi
finish
