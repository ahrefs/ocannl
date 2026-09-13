#!/usr/bin/env bash
# Hand-run tests for the mid-merge guard in tools/promote.sh -- the trap where
# a promotion made after `git add` is silently dropped from the merge commit
# (staging PR #487, ~90 minutes).
#
#   tools/test-promote.sh          # run every leg
#   tools/test-promote.sh --keep   # keep the temp dir for inspection
#
# It is the sibling of tools/test-test-run.sh, and is deliberately NOT wired
# into any dune alias for the same shape of reason: every leg runs `dune` on a
# throwaway project of its own, and a dune nested inside `dune runtest` is a
# poor bargain (its own lock, its own `_build`, the outer run's DUNE_* leaking
# in). What dune does check about this file is that it parses --
# test/operations/shell_scripts_parse globs tools/.
#
# It tests the WORKING-TREE copy: tools/promote.sh is copied into each scratch
# repository, so the legs exercise the text that ships. That copy is also what
# makes the legs possible at all -- promote.sh resolves its own repository as
# `dirname $0/..`, so the repository it acts on is chosen by where the script
# sits, and a copy under the scratch repo's tools/ points it there. It never
# touches this repository.
#
# The scenario every leg builds is the real one: two branches that both edit a
# generator and its `.expected` golden, merged, so the golden conflicts; the
# conflict resolved the way a hurried session resolves it (take one side, `git
# add`, run the tests); and a promotion afterwards.
#
# Legs:
#   1. the trap REPRODUCES -- a bare `dune promotion apply` in that scenario
#      commits the pre-promotion golden while the built output says otherwise.
#      This is the negative control: without it, leg 2 could pass because
#      `git commit` picks up the working tree, and the guard would be untested.
#   2. promote.sh closes it -- same scenario, and the COMMITTED blob matches
#      the built output.
#   3. a promoted golden still UNMERGED is not staged, and the warning names it
#      and the `git add` that would accept it. Staging it would be recording a
#      conflict resolution, which is the caller's call.
#   4. outside a merge the guard is a no-op: a promotion does not reach the
#      index, so `git add -p`-style staging discipline is left intact.
#   5. a golden that is NEW mid-merge (untracked, absent from the index) is
#      staged too -- `git commit -a` would not have carried it either, so it
#      is the same trap wearing a different hat.
#   6. CRLF stripping survives the guard, on the same new-file path: what
#      reaches the INDEX is LF, not merely what reaches the working tree.

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/promote.sh"
[ -f "$SRC" ] || {
  echo "no $SRC" >&2
  exit 2
}
command -v dune >/dev/null 2>&1 || . "$HERE/opam-env.sh"
harness_require dune
harness_require perl git


# Checked, not assumed: nothing here uses `set -e`, so a `mktemp` that fails
# would leave TMP empty and `rm -rf "$TMP"` would be handed the ROOT.
harness_scratch "test-promote"


echo "testing $SRC"
printf '  digest %s\n' "$( (cksum <"$SRC") 2>/dev/null || echo '?')"
printf '  dune   %s\n' "$(dune --version 2>/dev/null || echo '?')"

# git in a scratch repo, with an identity and none of the caller's config: a
# global commit.gpgsign or a hook template would otherwise decide whether the
# legs can commit at all.
g() { git -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }

# Build a repository holding a dune rule that diffs `<name>.expected` against
# the output of `cat gen.txt`, on `main`, plus two branches that each rewrite
# both files. `gen.txt` is what the test "computes"; the golden is what it is
# pinned to. Echoes the repo path.
scenario() { # scenario NAME
  local repo="$TMP/$1"
  mkdir -p "$repo/tools" || return 1
  cp "$SRC" "$repo/tools/promote.sh" || return 1
  chmod +x "$repo/tools/promote.sh" || return 1
  (
    cd "$repo" || exit 1
    printf '(lang dune 3.20)\n' >dune-project
    cat >dune <<'DUNE'
(rule
 (target foo.output)
 (deps gen.txt)
 (action
  (with-stdout-to
   foo.output
   (run cat gen.txt))))

(rule
 (alias runtest)
 (action
  (diff foo.expected foo.output)))
DUNE
    # `-text` keeps git's own line-ending normalization out of the way, and it
    # is load-bearing for leg 6 rather than tidiness: with the Windows default
    # `core.autocrlf=true` and no attribute, `git add` strips the CRs itself,
    # so leg 6 would pass on a promote.sh that stripped nothing. Measured both
    # ways -- with `-text` the staged blob keeps its `\r`, without it it does
    # not. This repository's real `.gitattributes` pins goldens the opposite
    # way (`*.expected text eol=lf`), which is a belt to promote.sh's braces;
    # the leg has to remove the belt to test the braces.
    printf '* -text\n' >.gitattributes
    printf 'base\n' >gen.txt
    printf 'base\n' >foo.expected
    g init -q -b main . >/dev/null 2>&1 || exit 1
    g add -A >/dev/null 2>&1 || exit 1
    g commit -qm base >/dev/null 2>&1 || exit 1

    g checkout -q -b side-a || exit 1
    printf 'from A\n' >gen.txt
    printf 'from A\n' >foo.expected
    g commit -qam A >/dev/null 2>&1 || exit 1

    g checkout -q main || exit 1
    g checkout -q -b side-b || exit 1
    printf 'from B\n' >gen.txt
    printf 'from B\n' >foo.expected
    g commit -qam B >/dev/null 2>&1 || exit 1

    g checkout -q main || exit 1
    g merge -q side-a -m 'merge A' >/dev/null 2>&1 || exit 1
    # Conflicts in both files; leaving the merge in progress is the point.
    g merge side-b >/dev/null 2>&1
    [ -f "$(g rev-parse --git-dir)/MERGE_HEAD" ] || exit 1
  ) || return 1
  printf '%s' "$repo"
}

# The hurried resolution: take B's generator, take A's golden (i.e. leave the
# golden stale), stage both, and run the test so a promotion is pending.
resolve_and_test() { # resolve_and_test REPO
  (
    cd "$1" || exit 1
    printf 'from B\n' >gen.txt
    printf 'from A\n' >foo.expected
    g add gen.txt foo.expected >/dev/null 2>&1 || exit 1
    dune build @runtest >/dev/null 2>&1
    # The golden diff must be PENDING; a leg that promotes nothing proves
    # nothing, and would let legs 1 and 2 agree for the wrong reason.
    [ -n "$(dune promotion list --root . 2>/dev/null)" ] || exit 1
  )
}

blob() { # blob REPO REV:PATH -- the committed content, or empty
  (cd "$1" && g show "$2" 2>/dev/null)
}

# ---------------------------------------------------------------- leg 1
# Negative control: the trap is real in this scenario.
repo="$(scenario trap)"
if [ -z "$repo" ] || ! resolve_and_test "$repo"; then
  report 1 "leg 1 setup: conflicted scenario with a pending promotion"
else
  (cd "$repo" && dune promotion apply --root . >/dev/null 2>&1 && g commit -q --no-edit) \
    >/dev/null 2>&1
  built="$(cat "$repo/_build/default/foo.output" 2>/dev/null)"
  committed="$(blob "$repo" HEAD:foo.expected)"
  worktree="$(cat "$repo/foo.expected" 2>/dev/null)"
  if [ "$committed" != "$built" ] && [ "$worktree" = "$built" ]; then
    report 0 "leg 1  bare \`dune promotion apply\` mid-merge: promotion dropped from the commit"
  else
    report 1 "leg 1  bare \`dune promotion apply\` mid-merge: promotion dropped from the commit" \
      "expected committed != built and worktree == built; got committed='$committed' built='$built' worktree='$worktree'"
  fi
fi

# ---------------------------------------------------------------- leg 2
# The guard: same scenario, through promote.sh.
repo="$(scenario guard)"
if [ -z "$repo" ] || ! resolve_and_test "$repo"; then
  report 1 "leg 2 setup: conflicted scenario with a pending promotion"
else
  out="$( (cd "$repo" && tools/promote.sh) 2>&1)"
  (cd "$repo" && g commit -q --no-edit) >/dev/null 2>&1
  built="$(cat "$repo/_build/default/foo.output" 2>/dev/null)"
  committed="$(blob "$repo" HEAD:foo.expected)"
  if [ "$committed" = "$built" ] && [ -n "$built" ]; then
    report 0 "leg 2  promote.sh mid-merge: the COMMITTED golden matches the built output"
  else
    report 1 "leg 2  promote.sh mid-merge: the COMMITTED golden matches the built output" \
      "committed='$committed' built='$built'; promote.sh said: $out"
  fi
  case "$out" in
    *"staged"*foo.expected*) report 0 "leg 2b promote.sh names what it staged" ;;
    *) report 1 "leg 2b promote.sh names what it staged" "said: $out" ;;
  esac
fi

# ---------------------------------------------------------------- leg 3
# A promoted golden still unmerged in the index: warned about, not staged.
repo="$(scenario unmerged)"
if [ -z "$repo" ]; then
  report 1 "leg 3 setup: conflicted scenario"
else
  # Resolve only the generator, so the build can run, and leave the golden
  # unmerged -- overwriting its conflict markers in the WORKING TREE without
  # `git add`, which is what a dune diff needs to produce a promotion.
  ok=1
  (
    cd "$repo" || exit 1
    printf 'from B\n' >gen.txt
    g add gen.txt >/dev/null 2>&1 || exit 1
    printf 'from A\n' >foo.expected
    dune build @runtest >/dev/null 2>&1
    [ -n "$(dune promotion list --root . 2>/dev/null)" ] || exit 1
    [ -n "$(g ls-files --unmerged -- foo.expected)" ] || exit 1
  ) || ok=0
  if [ "$ok" = 0 ]; then
    report 1 "leg 3 setup: golden left unmerged with a pending promotion"
  else
    out="$( (cd "$repo" && tools/promote.sh) 2>&1)"
    still_unmerged=1
    [ -n "$( (cd "$repo" && g ls-files --unmerged -- foo.expected) )" ] || still_unmerged=0
    if [ "$still_unmerged" = 1 ]; then
      report 0 "leg 3  an UNMERGED promoted golden is left unstaged"
    else
      report 1 "leg 3  an UNMERGED promoted golden is left unstaged" \
        "promote.sh staged it; said: $out"
    fi
    case "$out" in
      *WARNING*UNMERGED*foo.expected*"git add --"*foo.expected*)
        report 0 "leg 3b the warning names the file and the \`git add\` to accept it"
        ;;
      *)
        report 1 "leg 3b the warning names the file and the \`git add\` to accept it" \
          "said: $out"
        ;;
    esac
  fi
fi

# ---------------------------------------------------------------- leg 4
# Outside a merge the guard does nothing: a promotion stays out of the index.
repo="$TMP/nomerge"
mkdir -p "$repo/tools" && cp "$SRC" "$repo/tools/promote.sh" && chmod +x "$repo/tools/promote.sh"
ok=1
(
  cd "$repo" || exit 1
  printf '(lang dune 3.20)\n' >dune-project
  cat >dune <<'DUNE'
(rule
 (target foo.output)
 (deps gen.txt)
 (action
  (with-stdout-to
   foo.output
   (run cat gen.txt))))

(rule
 (alias runtest)
 (action
  (diff foo.expected foo.output)))
DUNE
  printf '* -text\n' >.gitattributes
  printf 'stale\n' >foo.expected
  printf 'fresh\n' >gen.txt
  g init -q -b main . >/dev/null 2>&1 || exit 1
  g add -A >/dev/null 2>&1 || exit 1
  g commit -qm base >/dev/null 2>&1 || exit 1
  g rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1 && exit 1 # not mid-merge
  dune build @runtest >/dev/null 2>&1
  [ -n "$(dune promotion list --root . 2>/dev/null)" ] || exit 1
) || ok=0
if [ "$ok" = 0 ]; then
  report 1 "leg 4 setup: non-merge repository with a pending promotion"
else
  out="$( (cd "$repo" && tools/promote.sh) 2>&1)"
  staged_blob="$( (cd "$repo" && g show :foo.expected) 2>/dev/null)"
  worktree="$(cat "$repo/foo.expected" 2>/dev/null)"
  if [ "$staged_blob" = "stale" ] && [ "$worktree" = "fresh" ]; then
    report 0 "leg 4  outside a merge: promotion reaches the working tree, not the index"
  else
    report 1 "leg 4  outside a merge: promotion reaches the working tree, not the index" \
      "index='$staged_blob' worktree='$worktree'; promote.sh said: $out"
  fi
  case "$out" in
    *staged* | *WARNING*)
      report 1 "leg 4b outside a merge promote.sh says nothing about staging" "said: $out"
      ;;
    *) report 0 "leg 4b outside a merge promote.sh says nothing about staging" ;;
  esac
fi

# ---------------------------------------------------------------- legs 5, 6
# A golden created mid-merge is absent from the index, so `git commit` -- with
# or without -a -- would not carry it. It is staged too. And what lands in the
# INDEX is LF: leg 6 promotes CRLF output, which is the Windows case, and reads
# the staged blob rather than the file.
repo="$(scenario newfile)"
if [ -z "$repo" ]; then
  report 1 "leg 5 setup: conflicted scenario"
else
  ok=1
  (
    cd "$repo" || exit 1
    printf 'from B\n' >gen.txt
    printf 'from B\n' >foo.expected
    g add gen.txt foo.expected >/dev/null 2>&1 || exit 1
    # A second golden, tracked by nothing, whose generated output carries CRs.
    printf 'crlf line\r\n' >gen2.txt
    cat >>dune <<'DUNE'

(rule
 (target bar.output)
 (deps gen2.txt)
 (action
  (with-stdout-to
   bar.output
   (run cat gen2.txt))))

(rule
 (alias runtest)
 (action
  (diff bar.expected bar.output)))
DUNE
    : >bar.expected # empty golden: untracked, so absent from the index
    g add dune gen2.txt >/dev/null 2>&1 || exit 1
    dune build @runtest >/dev/null 2>&1
    dune promotion list --root . 2>/dev/null | grep -q '^bar\.expected$' || exit 1
    [ -z "$(g ls-files -- bar.expected)" ] || exit 1 # genuinely untracked
  ) || ok=0
  if [ "$ok" = 0 ]; then
    report 1 "leg 5 setup: an untracked golden with a pending CRLF promotion"
  else
    out="$( (cd "$repo" && tools/promote.sh) 2>&1)"
    if [ -n "$( (cd "$repo" && g ls-files -- bar.expected) )" ]; then
      report 0 "leg 5  a golden created mid-merge is staged, not left untracked"
    else
      report 1 "leg 5  a golden created mid-merge is staged, not left untracked" \
        "still untracked; promote.sh said: $out"
    fi
    staged_bytes="$( (cd "$repo" && g show :bar.expected 2>/dev/null | od -c | tr -d '\n') )"
    case "$staged_bytes" in
      *'\r'*)
        report 1 "leg 6  the CRs are stripped before the golden reaches the index" \
          "staged blob still holds a CR: $staged_bytes"
        ;;
      '') report 1 "leg 6  the CRs are stripped before the golden reaches the index" \
        "nothing staged at all" ;;
      *) report 0 "leg 6  the CRs are stripped before the golden reaches the index" ;;
    esac
  fi
fi

# ---------------------------------------------------------------- leg 7
# The guard learns what to stage from `dune promotion list`. If that call
# fails, `promoted` is empty for a reason that is NOT "nothing to promote", and
# staying quiet would reinstate the trap while looking like a clean run. Fault
# injection, since nothing about a repository makes `list` fail on its own: a
# `dune` earlier on PATH that refuses `promotion list` and delegates the rest,
# so `apply` still promotes and only the guard's input is lost.
repo="$(scenario listfails)"
if [ -z "$repo" ] || ! resolve_and_test "$repo"; then
  report 1 "leg 7 setup: conflicted scenario with a pending promotion"
else
  mkdir -p "$repo/shim"
  cat >"$repo/shim/dune" <<SHIM
#!/usr/bin/env bash
if [ "\${1:-}" = promotion ] && [ "\${2:-}" = list ]; then
  echo "shim: refusing promotion list" >&2
  exit 1
fi
exec "$(command -v dune)" "\$@"
SHIM
  chmod +x "$repo/shim/dune"
  out="$( (cd "$repo" && PATH="$repo/shim:$PATH" tools/promote.sh) 2>&1)"
  rc=$?
  worktree="$(cat "$repo/foo.expected" 2>/dev/null)"
  built="$(cat "$repo/_build/default/foo.output" 2>/dev/null)"
  # The promotion must still have happened -- a guard that cannot read its
  # input must not take the promotion down with it.
  if [ "$worktree" = "$built" ] && [ -n "$built" ] && [ "$rc" -eq 0 ]; then
    report 0 "leg 7  a failed \`promotion list\` still promotes, and exits 0"
  else
    report 1 "leg 7  a failed \`promotion list\` still promotes, and exits 0" \
      "rc=$rc worktree='$worktree' built='$built'; said: $out"
  fi
  case "$out" in
    *WARNING*"staged NOTHING"*)
      report 0 "leg 7b it says it staged nothing, rather than passing quietly"
      ;;
    *)
      report 1 "leg 7b it says it staged nothing, rather than passing quietly" \
        "said: $out"
      ;;
  esac
fi

finish
