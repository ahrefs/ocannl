#!/usr/bin/env bash
# Bring the OCANNL development environment up, incrementally.
#
#   scripts/setup-ocaml-env.sh          # check and self-heal; seconds, never compiles
#   scripts/setup-ocaml-env.sh --deps   # additionally install/refresh opam dependencies
#
# Every step inspects the current state and acts only when something is missing
# or broken, so running it repeatedly is safe and each run advances the setup by
# whatever is still outstanding.
#
# This runs as a SessionStart hook (.claude/settings.json), which is why the
# default mode is cheap: it queries opam, repairs the cygwin pkgconf breakage
# described below, and reports what is left, but never builds anything. Pass
# --deps when you actually want the (10-20 minute) dependency install.
#
# Steps that would install software are only taken automatically in Claude Code
# cloud sessions (CLAUDE_CODE_REMOTE=true), where the machine is disposable. On
# a real workstation they are reported as suggestions instead — this script does
# not install opam or create switches behind your back.
#
# Note: this file used to restore a pre-built switch from a GitHub release. That
# "fast setup" machinery was reverted in 9cf51f51 and the release it referenced
# no longer exists, so that path is gone; docs/fast-setup.md describes the
# abandoned design.

set -u

MODE_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --deps|--full) MODE_DEPS=1 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup-ocaml-env.sh: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

# Only cloud sessions may install things unattended.
if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then MAY_INSTALL=1; else MAY_INSTALL=0; fi

status=0
ok()   { printf '  ok    %s\n' "$*"; }
fixed(){ printf '  fixed %s\n' "$*"; }
todo() { printf '  todo  %s\n' "$*"; status=1; }
fail() { printf '  FAIL  %s\n' "$*" >&2; status=1; }

script_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
if [ -z "$script_dir" ] || [ ! -r "$script_dir/process-group.sh" ]; then
  echo "setup-ocaml-env.sh: cannot read scripts/process-group.sh" >&2
  exit 2
fi
# shellcheck source=process-group.sh
. "$script_dir/process-group.sh"

echo "=== OCANNL environment ==="

# --- worktree dune root --------------------------------------------------
# Claude Code creates worktrees under `.claude/worktrees/`, i.e. INSIDE the
# repository. Dune takes as its root the outermost ancestor holding a
# `dune-workspace` (failing that, a `dune-project`) and ignores dot-directories,
# so from such a worktree the main checkout wins and the worktree is invisible to
# dune: targeted commands fail with "Don't know about directory
# .claude/worktrees/...", and — the quiet one — a bare `dune build` / `dune
# runtest` builds and tests the PARENT checkout instead of the branch you are on.
#
# A `dune-workspace` at this checkout's root makes it the root again and gives it
# its own `_build`. That only holds while no ancestor has one (outermost wins),
# which is why the file is generated per worktree and gitignored rather than
# committed. This step comes before the opam section: it needs no toolchain, and
# a missing opam exits the script early below.
#
# The test is on the ancestor DIRECTORIES, not on git topology: a checkout can be
# nested inside another checkout that is itself a linked worktree (living
# anywhere), in which case `--git-common-dir` names the primary checkout, which
# is not the one dune would root at. Asking the question dune asks covers every
# arrangement — nested worktree, worktree of a worktree, worktree outside the
# repo — without enumerating them.
self_top="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd || true)"
wt_top="$(git -C "${self_top:-.}" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$wt_top" ] || wt_top="$self_top"
if [ -n "$wt_top" ] && [ -e "$wt_top/dune-project" ]; then
  # Walk outwards, keeping the LAST hit, so each variable ends at the outermost
  # ancestor holding that file — which is the one dune would pick.
  shadow="" outer="" dir="$(dirname "$wt_top")"
  while [ -n "$dir" ]; do
    [ -e "$dir/dune-workspace" ] && shadow="$dir"
    [ -e "$dir/dune-project" ] && outer="$dir"
    next="$(dirname "$dir")"
    [ "$next" = "$dir" ] && break
    dir="$next"
  done
  if [ -n "$shadow" ]; then
    # Outermost wins, so a dune-workspace above us cannot be overridden from here.
    fail "dune-workspace in $shadow shadows this checkout's — dune will build $shadow instead"
  elif [ -n "$outer" ]; then
    # No ancestor dune-workspace, but an ancestor dune-project would take the root.
    if [ -e "$wt_top/dune-workspace" ]; then
      ok "worktree dune root"
    else
      lang="$(grep -m1 '^(lang dune ' "$wt_top/dune-project" 2>/dev/null || true)"
      if printf '%s\n' "${lang:-(lang dune 3.18)}" >"$wt_top/dune-workspace"; then
        fixed "worktree dune root (dune-workspace written; dune would have built $outer)"
      else
        fail "could not write $wt_top/dune-workspace"
      fi
    fi
  fi
  # No ancestor holds either file: dune already roots at this checkout.
fi

# --- staleness against origin/master ------------------------------------
# Claude Code creates a worktree from the MAIN checkout's HEAD, and the main
# checkout's `master` only advances when someone fast-forwards it after a PR
# merge — so a fresh worktree can start dozens of commits behind `origin/master`
# (79 on 2026-08-22, and a full CUDA suite run tested stale code before anyone
# noticed). A session never sees that by itself: dune builds what is checked
# out. So fetch `origin master` and count.
#
# Best-effort and bounded, and the bound must not depend on the transport or
# on the platform: `GIT_TERMINAL_PROMPT=0` silences git's prompts but not
# OpenSSH's (host-key confirmation, a passphrase), `http.lowSpeed*` reaches only
# HTTP transfers, and `timeout` is not something to lean on: absent on a default
# macOS and on some Git Bash installs, and where present not necessarily GNU's —
# uutils' (Ubuntu's since 25.10) takes `-k` but does not signal the process
# group on the KILL escalation, so git's ssh child outlives it. So the fetch
# runs under `bounded`, a watchdog of this script's own that kills the
# command's whole process group after N seconds whatever it is stuck on, the
# same on every platform — and the SSH leg additionally gets
# `BatchMode=yes` (no prompts) with connect and keepalive timeouts, appended to
# whatever ssh command the user already configured. Any failure prints `skip`.
# The count is then taken against whatever `origin/master` the repository
# holds — after a failed fetch that is the previous fetch's, which is still
# newer than the parent checkout's `master` when the latter is the thing that
# lagged. The warning changes nothing and does not mark the environment
# incomplete; it names the recovery instead.
bounded() {
  # bounded SECS CMD...: run CMD, killing it if still running after SECS.
  # What GNU timeout does, done here so that it holds whatever `timeout` is on
  # PATH: the command runs in its own process group and the watchdog signals
  # the GROUP, so the ssh or credential helper git is blocked on dies with git
  # instead of being orphaned by a PID-only kill and accumulating across
  # session starts; TERM then CONT (for a job stopped on a tty read), then KILL
  # for anything still there 5s later, since an ignoring parent's children
  # inherit the ignore. Job control is what gives a background job its own
  # group; it is switched off again once both jobs are spawned. The contract is
  # that nothing of the group survives the return: the command exiting does not
  # by itself cancel the watchdog — git dying on TERM while its ssh child
  # ignores it would otherwise return before the KILL — only an empty group does.
  # "Empty" cannot be asked of signals, though: `kill -0` succeeds on a ZOMBIE
  # exactly as on a live process, and when git exits its ssh child is one,
  # having been reparented and not yet reaped. Reading that as work in progress
  # sat out the entire bound after a fetch that had already failed in
  # milliseconds — 30s added to every session start whose ssh remote is
  # unreachable — and where the reaper is a PID 1 that does not reap (the
  # common container case) the zombie is permanent, so no amount of waiting for
  # it to clear would help. The shared `group_alive` predicate therefore reads
  # process STATES, from /proc where there is one and from `ps` otherwise, and
  # only a member that is not a zombie counts as work.
  local secs="$1"; shift
  local pid watchdog rc
  set -m
  "$@" </dev/null & pid=$!
  # The killer gets no inherited fds: a lingering `sleep` holding the hook's
  # stdout would keep the harness waiting for EOF after the script exits.
  ( sleep "$secs"; kill -TERM -- -"$pid" 2>/dev/null; kill -CONT -- -"$pid" 2>/dev/null
    for _ in 1 2 3 4 5; do sleep 1; group_alive "$pid" || exit 0; done
    kill -KILL -- -"$pid" 2>/dev/null ) >/dev/null 2>&1 </dev/null &
  watchdog=$!
  set +m
  wait "$pid" 2>/dev/null; rc=$?
  if group_alive "$pid"; then
    wait "$watchdog" 2>/dev/null
  else
    kill -TERM -- -"$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
  fi
  return $rc
}
if [ -n "$wt_top" ] && git -C "$wt_top" remote get-url origin >/dev/null 2>&1; then
  # Git picks its SSH launcher as GIT_SSH_COMMAND, else core.sshCommand, else
  # GIT_SSH, else `ssh`; the probe keeps that choice and only APPENDS the OpenSSH
  # options where OpenSSH is CERTAIN: a shell-string launcher (the first two, or
  # the default) whose program is `ssh` itself, or whose variant is explicitly
  # `ssh` (`GIT_SSH_VARIANT`, which outranks `ssh.variant`). Everything else —
  # plink and its kin, `simple`, a custom wrapper git would probe with `-G`, a
  # `GIT_SSH` program (a path, not a shell string) — is passed through exactly
  # as git would run it. This is deliberately narrower than git's own variant
  # detection rather than a copy of it: the options are a courtesy bound, and
  # `bounded` is the bound wherever they are not appended, so nothing is lost
  # by declining to guess, while a wrong guess breaks every startup fetch.
  ssh_opts="-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2"
  ssh_variant="${GIT_SSH_VARIANT:-$(git -C "$wt_top" config ssh.variant 2>/dev/null || true)}"
  fetch_env=()
  if [ -n "${GIT_SSH_COMMAND:-}" ]; then ssh_launcher="$GIT_SSH_COMMAND"
  elif ssh_launcher="$(git -C "$wt_top" config core.sshCommand 2>/dev/null)" && [ -n "$ssh_launcher" ]; then :
  elif [ -n "${GIT_SSH:-}" ]; then ssh_launcher=""
  else ssh_launcher="ssh"
  fi
  if [ -n "$ssh_launcher" ]; then
    ssh_prog="$(basename "${ssh_launcher%% *}" | tr 'A-Z' 'a-z')"
    case "${ssh_variant:-auto}:$ssh_prog" in
      ssh:*|auto:ssh|auto:ssh.exe) fetch_env=(GIT_SSH_COMMAND="$ssh_launcher $ssh_opts") ;;
    esac
  fi
  # `--no-write-fetch-head` (git >= 2.29): a startup probe must not clobber a
  # FETCH_HEAD someone kept for a later `git merge FETCH_HEAD`;
  # `--no-auto-maintenance`: housekeeping the watchdog would cut short at the
  # bound, leaving its trigger in place for the next start to hit. The refspec
  # names `refs/heads/master` in full because a remote TAG called `master`
  # would otherwise win the short name, `--no-tags` notwithstanding, and the
  # forced update would then write the tag's commit into the tracking ref.
  # `${a[@]+"${a[@]}"}`, not `"${a[@]}"`: under `set -u`, bash before 4.4 (macOS
  # ships 3.2) treats an EMPTY array's expansion as an unbound variable and
  # exits the script — which is every leg where no options are appended.
  if bounded 30 env GIT_TERMINAL_PROMPT=0 ${fetch_env[@]+"${fetch_env[@]}"} \
       git -C "$wt_top" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=15 \
       fetch --quiet --no-tags --no-write-fetch-head --no-auto-maintenance origin \
       "+refs/heads/master:refs/remotes/origin/master" >/dev/null 2>&1; then
    fetched="origin/master"
  else
    echo "  skip  fetching origin/master failed (offline?)"
    fetched=""
  fi
  # The tracking ref is spelled in full here too: a branch or tag named
  # `origin/master` would make the short form ambiguous. A comparison that
  # fails anyway (an unborn HEAD, say) is reported, not read as 0.
  upstream=refs/remotes/origin/master
  if git -C "$wt_top" rev-parse --verify -q "$upstream" >/dev/null; then
    behind="$(git -C "$wt_top" rev-list --count "HEAD..$upstream" 2>/dev/null || true)"
    ahead="$(git -C "$wt_top" rev-list --count "$upstream..HEAD" 2>/dev/null || true)"
    asof="${fetched:+}"; [ -n "$fetched" ] || asof=" (as of the last successful fetch)"
    if [ -z "$behind" ] || [ -z "$ahead" ]; then
      echo "  skip  could not compare HEAD with origin/master"
    elif [ "$behind" = 0 ]; then
      ok "up to date with origin/master$asof"
    else
      if [ "$ahead" = 0 ]; then
        # Spelled in full for the same reason as the comparison above.
        recovery="git merge --ff-only $upstream"
      else
        recovery="git rebase $upstream  ($ahead local commit(s) to replay)"
      fi
      printf '  WARNING HEAD is %s commit(s) behind origin/master%s — a suite run here tests stale code.\n' "$behind" "$asof"
      printf '          recover with: %s\n' "$recovery"
    fi
  fi
fi

# --- opam itself ---------------------------------------------------------
if command -v opam >/dev/null 2>&1; then
  ok "opam $(opam --version)"
elif [ "$MAY_INSTALL" = 1 ]; then
  echo "  ...   installing opam"
  if bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)" -- --no-backup >/dev/null 2>&1; then
    fixed "opam $(opam --version)"
  else
    fail "could not install opam"
  fi
else
  todo "opam is not on PATH — see https://opam.ocaml.org/doc/Install.html"
fi

command -v opam >/dev/null 2>&1 || { echo "=== stopped: opam required ==="; exit $status; }

# --- opam root -----------------------------------------------------------
OPAM_ROOT_RAW="$(opam var root 2>/dev/null || true)"
if [ -z "$OPAM_ROOT_RAW" ]; then
  if [ "$MAY_INSTALL" = 1 ]; then
    echo "  ...   initialising opam"
    opam init -y --bare --disable-sandboxing >/dev/null 2>&1 \
      && fixed "opam root initialised" || fail "opam init failed"
    OPAM_ROOT_RAW="$(opam var root 2>/dev/null || true)"
  else
    todo "opam root not initialised — run: opam init --bare"
  fi
else
  ok "opam root $OPAM_ROOT_RAW"
fi

# opam prints a native path; the rest of this script needs a POSIX one.
if command -v cygpath >/dev/null 2>&1 && [ -n "$OPAM_ROOT_RAW" ]; then
  OPAM_ROOT="$(cygpath -u "$OPAM_ROOT_RAW")"
else
  OPAM_ROOT="$OPAM_ROOT_RAW"
fi

# --- switch --------------------------------------------------------------
# Any switch will do as long as it satisfies the packages' `ocaml >= 5.3.0`; the
# repo does not mandate a local `_opam`, and creating one unasked would be a
# surprising, expensive side effect.
ocaml_version="$(opam exec -- ocamlopt -version 2>/dev/null || true)"
if [ -z "$ocaml_version" ]; then
  todo "no usable switch — run: opam switch create . 5.3.0 --no-install"
else
  case "$ocaml_version" in
    5.[3-9]*|5.[1-9][0-9]*|[6-9]*) ok "ocaml $ocaml_version" ;;
    *) todo "ocaml $ocaml_version is below the required 5.3.0" ;;
  esac
fi

# --- cygwin pkgconf ------------------------------------------------------
# Cygwin's pkgconf 3.0.4-1 (2026-07-26) broke `pkgconf --personality=<triplet>`:
# personality files are only looked up in /usr/lib/pkgconfig/personality.d while
# the package installs them to /etc/pkgconfig/personality.d, and even once loaded
# the personality's DefaultSearchPaths are ignored. Every opam conf-mingw-w64-*
# package is exactly that invocation, so dependency installs fail outright.
#
# The probe is behavioural rather than a version comparison, so this repairs a
# broken pkgconf whatever its version and goes quiet by itself once cygwin ships
# a fix. Restoring 2.5.1 needs libpkgconf7 too: pkgconf.exe links
# cygpkgconf-7.dll, and 3.0.4 only brought in cygpkgconf-8.dll.
CYGROOT="${OPAM_ROOT:+$OPAM_ROOT/.cygwin/root}"
if [ -n "${CYGROOT:-}" ] && [ -x "$CYGROOT/bin/pkgconf.exe" ]; then
  personality_ok() {
    "$CYGROOT/bin/pkgconf.exe" --personality=x86_64-w64-mingw32 --dump-personality 2>/dev/null \
      | grep -q '^Triplet: x86_64-w64-mingw32'
  }
  # Pick a tar that understands zstd: Git Bash's GNU tar shells out to a zstd
  # binary that is not installed, while Windows' bundled bsdtar has it built in.
  zstd_tar() {
    for candidate in "${SYSTEMROOT:-/c/Windows}/System32/tar.exe" /c/Windows/System32/tar.exe tar; do
      [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1 || continue
      if "$candidate" -tf "$1" >/dev/null 2>&1; then echo "$candidate"; return 0; fi
    done
    return 1
  }

  if personality_ok; then
    ok "cygwin pkgconf personality ($("$CYGROOT/bin/pkgconf.exe" --version 2>&1))"
  else
    echo "  ...   repairing cygwin pkgconf (personality lookup broken)"
    work="$(mktemp -d)"
    base="https://cygwin.mirror.constant.com/x86_64/release/pkgconf"
    repaired=1
    for pkg in pkgconf-2.5.1-1-x86_64 libpkgconf7/libpkgconf7-2.5.1-1-x86_64; do
      file="$work/$(basename "$pkg").tar.zst"
      curl -sSLf -o "$file" "$base/$pkg.tar.zst" || { repaired=0; break; }
      tarbin="$(zstd_tar "$file")" || { repaired=0; break; }
      "$tarbin" -xf "$file" -C "$work" || { repaired=0; break; }
    done
    if [ "$repaired" = 1 ] && [ -d "$work/usr/bin" ]; then
      # Cygwin mounts /usr/bin at <root>/bin and nothing reads a physical
      # <root>/usr/bin, so the payload is copied rather than unpacked over the root.
      cp -f "$work"/usr/bin/* "$CYGROOT/bin/" || repaired=0
    else
      repaired=0
    fi
    rm -rf "$work"
    if [ "$repaired" = 1 ] && personality_ok; then
      fixed "cygwin pkgconf restored to $("$CYGROOT/bin/pkgconf.exe" --version 2>&1)"
    else
      fail "could not repair cygwin pkgconf; conf-mingw-w64-* packages will not install"
    fi
  fi
fi

# --- pins ----------------------------------------------------------------
# Kept in step with .github/workflows/ci.yml; see the comments there for why each
# one is pinned. `opam pin -n` is only invoked for pins that are actually absent,
# so a repeat run neither re-resolves nor reports spurious changes.
pinned="$(opam pin list --short 2>/dev/null || true)"
# grep -qx, not a substring test: these lists are newline-separated, and a
# substring match would also let `printbox` satisfy a check for `printbox-text`.
pin_if_missing() {
  if printf '%s\n' "$pinned" | grep -qx "$1"; then
    ok "pin $1"
  elif opam pin -n "$1" "$2" >/dev/null 2>&1; then
    fixed "pin $1"
  else
    fail "could not pin $1"
  fi
}
pin_if_missing printbox-text https://github.com/c-cube/printbox.git
pin_if_missing pcre https://github.com/mmottl/pcre-ocaml.git
pin_if_missing dataprep https://github.com/lukstafi/ocaml-dataprep.git

# --- dependencies --------------------------------------------------------
# A representative sample rather than a full solve: `opam install --dry-run`
# would be authoritative but costs a solver run on every session start.
installed="$(opam list --installed --short 2>/dev/null || true)"
missing=""
for pkg in base ppxlib ppx_minidebug dataprep; do
  printf '%s\n' "$installed" | grep -qx "$pkg" || missing="$missing $pkg"
done
# The `with-dev-setup` tooling (dune-project): part of a development switch,
# but `opam install . --deps-only` alone never installs it, so a fresh switch
# has neither until asked. The install below passes --with-dev-setup.
dev_missing=""
for pkg in ocamlformat ocaml-lsp-server; do
  printf '%s\n' "$installed" | grep -qx "$pkg" || dev_missing="$dev_missing $pkg"
done

if [ -z "$missing" ] && [ -z "$dev_missing" ]; then
  ok "dependencies present"
elif [ "$MODE_DEPS" = 1 ]; then
  echo "  ...   installing dependencies (this takes 10-20 minutes)"
  if opam install . -y --deps-only --with-test --with-doc --with-dev-setup; then
    fixed "dependencies installed"
  else
    fail "opam install failed"
  fi
else
  [ -n "$missing" ] && todo "missing:$missing — re-run with --deps to install" || ok "dependencies present"
  [ -z "$dev_missing" ] || todo "dev tools missing:$dev_missing — re-run with --deps, or: opam install . --deps-only --with-dev-setup"
fi

# ocamlformat refuses to run at any release other than the one .ocamlformat
# pins, so an installed-but-wrong release is as unusable as none; dune-project
# pins the same release (move the two together). Only reported: the fix is an
# opam install, and installing is not this script's to do on a workstation.
fmt_pin="$(sed -n 's/^version *= *//p' "$script_dir/../.ocamlformat" 2>/dev/null | tr -d '[:space:]')"
if printf '%s\n' "$installed" | grep -qx ocamlformat; then
  # tr -d '\r': ocamlformat.exe emits CRLF on Windows; substitution strips only the LF.
  fmt_have="$(opam exec -- ocamlformat --version 2>/dev/null | tr -d '\r' || true)"
  if [ -z "$fmt_pin" ]; then
    todo ".ocamlformat carries no version pin"
  elif [ "$fmt_have" = "$fmt_pin" ]; then
    ok "ocamlformat $fmt_pin"
  else
    todo "ocamlformat ${fmt_have:-?} does not match the .ocamlformat pin $fmt_pin — opam install ocamlformat.$fmt_pin"
  fi
fi

# --- verification --------------------------------------------------------
dune_version="$(opam exec -- dune --version 2>/dev/null || true)"
[ -n "$dune_version" ] && ok "dune $dune_version" || todo "dune not available"

# Claude Code picks up environment for subsequent commands from this file.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  opam env --shell=sh >>"$CLAUDE_ENV_FILE" 2>/dev/null && ok "environment persisted"
fi

if [ "$status" = 0 ]; then echo "=== ready ==="; else echo "=== incomplete (see 'todo' above) ==="; fi
# A hook must not fail the session; report state through the summary instead.
exit 0
