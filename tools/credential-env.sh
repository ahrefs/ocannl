#!/bin/sh
# The credential variables no runner hands to dune (gh-ocannl-1280). Dune writes the environment of
# every process it spawns into `_build/trace.csexp`, inside the checkout -- a tree agents grep, mine
# and quote -- so a token a session exports would land there on every build. The runners remove
# these from dune's environment on the side where dune runs, after any opam environment is
# applied: tools/test-run.sh for everything it launches, tools/sweep.sh inside every `opam exec` of
# a unit leg's shell text (local, or sent to a remote box, whose own environment holds its own
# token), tools/machine-verify-far.sh inside every `opam exec` on the verified machine, and every
# other script here that starts dune itself (fmt-check.sh, promote.sh, dune-quiet.sh,
# api-drift.sh, ci-shard.sh). `test/operations/env_var_deps` reads `credential_env_patterns` below
# and refuses a dune stanza that declares a match as an `(env_var ...)` dependency or reads one
# through `%{env:NAME=...}`, and a test source that reads one by name, so no test can need one.
#
# Sourced, never executed. POSIX sh: machine-verify carries the scrub's text to a far side that
# runs it under dash.

# The deny-list, on ONE line, which is the form the env_var_deps scan reads: `|`-separated case
# patterns, each an exact variable name or `*` followed by a suffix.
credential_env_patterns='GH_TOKEN|GITHUB_TOKEN|GH_ENTERPRISE_TOKEN|CLAUDE_CODE_MESSAGING_TOKEN|*_TOKEN|*_API_KEY'

# Prints shell text that unsets, in the shell evaluating it, every exported variable the deny-list
# matches; the text's status is 0 exactly when none is left. Text rather than a function, so that
# it runs where dune will: sweep.sh and machine-verify's far side run it inside `opam exec -- sh
# -c`, on a remote box, where nothing on this side can name the environment's variables.
#
# It fails CLOSED. The one external step is listing the environment (`env`), and its status is
# checked: an `env` that cannot run -- a PATH an opam environment broke -- is a failure, never an
# empty environment. Each line is cut at its first `=` by the shell itself (parameter expansion,
# under `set -f` and a newline IFS, both restored), so no parser can fail silently. Only names are
# read, never a value. A deny-listed name that is not a shell identifier (`odd-name_TOKEN`, which
# a parent process can still pass) cannot be unset here, so it is reported and the status is
# non-zero rather than leaving it behind. A value spanning lines can make a later line look like
# `NAME=...`: unsetting a name nothing set is harmless, and such a line that matches the deny-list
# without being an identifier refuses -- a false refusal, never a pass.
credential_env_scrub_text() {
  printf '%s' 'credential_env_left=; credential_env_all=$(env) || credential_env_left=" (env failed: the environment could not be listed)"; '
  printf '%s' 'credential_env_ifs_set=${IFS+x}; credential_env_ifs=${IFS-}; IFS=$(printf '\''\n_'\''); IFS=${IFS%_}; '
  printf '%s' 'case $- in *f*) credential_env_noglob=1 ;; *) credential_env_noglob=; set -f ;; esac; '
  printf '%s' 'for credential_env_line in $credential_env_all; do case $credential_env_line in *=*) ;; *) continue ;; esac; '
  printf '%s' 'credential_env_name=${credential_env_line%%=*}; case $credential_env_name in '
  printf '%s' "$credential_env_patterns"
  printf '%s' ') case $credential_env_name in "" | [!A-Za-z_]* | *[!A-Za-z0-9_]*) credential_env_left="$credential_env_left $credential_env_name (not a shell identifier)" ;; '
  printf '%s' '*) unset "$credential_env_name" || credential_env_left="$credential_env_left $credential_env_name" ;; esac ;; esac; done; '
  printf '%s' 'if [ -n "$credential_env_ifs_set" ]; then IFS=$credential_env_ifs; else unset IFS; fi; [ -n "$credential_env_noglob" ] || set +f; '
  printf '%s' 'unset credential_env_all credential_env_line credential_env_name credential_env_ifs credential_env_ifs_set credential_env_noglob; [ -z "$credential_env_left" ]'
}
