#!/bin/sh
# The credential variables no runner hands to dune (gh-ocannl-1280). Dune writes the environment of
# every process it spawns into `_build/trace.csexp`, inside the checkout -- a tree agents grep, mine
# and quote -- so a token a session exports would land there on every build. The runners remove
# these from dune's environment on the side where dune runs, after any opam environment is
# applied: tools/test-run.sh for everything it launches, tools/sweep.sh inside every `opam exec` of
# a unit leg's shell text (local, or sent to a remote box, whose own environment holds its own
# token), tools/machine-verify-far.sh inside every `opam exec` on the verified machine, and every
# other script here that starts dune itself (fmt-check.sh, promote.sh, api-drift.sh). `test/operations/env_var_deps` reads
# `credential_env_patterns` below and refuses a dune stanza that declares a match as an
# `(env_var ...)` dependency, so no test can come to need one.
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
# Only names are read (`env`, cut at the first `=`), never a value. A value spanning lines can
# make a later line look like `NAME=...`; unsetting a name nothing set is harmless.
credential_env_scrub_text() {
  printf '%s' 'credential_env_left=; for credential_env_name in $(env | sed -n '\''s/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p'\''); do case $credential_env_name in '
  printf '%s' "$credential_env_patterns"
  printf '%s' ') unset "$credential_env_name" || credential_env_left="$credential_env_left $credential_env_name" ;; esac; done; unset credential_env_name; [ -z "$credential_env_left" ]'
}
