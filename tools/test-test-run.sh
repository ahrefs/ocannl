#!/usr/bin/env bash
# Run the harness of tools/test-run.sh, which is several members
# (gh-ocannl-2002): each tools/test-test-run-NAME.sh that
# tools/test-harnesses.sh lists, one after another.
#
#   tools/test-test-run.sh                 # every member
#   tools/test-test-run.sh --only '^81:'   # the members holding a leg whose name matches
#   tools/test-test-run.sh --only '#lifecycle'   # one named group of legs, wherever its legs sit
#   tools/test-test-run.sh --subject FILE --only '^99:'   # test FILE, not tools/test-run.sh
#   tools/test-test-run.sh --members       # list the members, and run nothing
#   tools/test-test-run.sh --list-legs     # every member's legs
#
# Every word but --members goes to each member it runs, unread but for
# --only's selector: --keep, --subject and the rest are the members'
# (tools/test-test-run-common.sh). A leg's number names it across the members,
# so a selector needs no member: with --only, a member is run when
# `--list-legs` shows it a leg the selector matches, by the matcher the gates
# use. A selector matching no leg of any member is refused (exit 2) as nothing
# tested, as one member refuses it.
#
# The members are read from tools/test-harnesses.sh --list, CI's one list of
# harnesses, never restated here: a member that list gains is run. That script
# is what runs them beside one another (--jobs) and under a slot; this one is
# for a person or an agent iterating on a leg, and takes none.
#
# Exit: 0 every member run passed; 1 one failed; 2 a usage error, a member
# that refused its arguments, or no member to run. HUP, INT and TERM in a
# member stop the run and keep their status (129, 130, 143).
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
only= only_set=0 list_members=0 args=("$@")
while [ $# -gt 0 ]; do
  case $1 in
    --members) list_members=1 ;;
    --only=*) only=${1#--only=} only_set=1 ;;
    --only)
      [ $# -ge 2 ] || { echo "${0##*/}: --only needs a regular expression" >&2; exit 2; }
      only=$2 only_set=1
      shift ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
  shift
done
members=()
while IFS= read -r member; do
  case $member in tools/test-test-run-*.sh) members+=("${member#tools/}") ;; esac
done < <("${BASH:-bash}" "$here/test-harnesses.sh" --list)
[ ${#members[@]} -gt 0 ] ||
  { echo "${0##*/}: $here/test-harnesses.sh --list names no tools/test-test-run-NAME.sh" >&2; exit 2; }
if [ "$list_members" = 1 ]; then
  [ ${#args[@]} = 1 ] || { echo "${0##*/}: --members takes no other argument" >&2; exit 2; }
  printf 'tools/%s\n' "${members[@]}"
  exit 0
fi
if [ "$only_set" = 1 ]; then
  # A member refuses an empty or malformed selector itself; so does this.
  rc=0
  [ -n "$only" ] && { [[ '' =~ $only ]] || rc=$?; [ "$rc" != 2 ]; } ||
    { echo "${0##*/}: --only '$only' is not a non-empty extended regular expression" >&2; exit 2; }
  holding=()
  for member in "${members[@]}"; do
    while IFS= read -r leg; do
      case $leg in 'LEG  '*) ;; *) continue ;; esac
      [[ ${leg#LEG  } =~ $only ]] || continue
      holding+=("$member")
      break
    done < <("${BASH:-bash}" "$here/$member" --list-legs 2>/dev/null)
  done
  [ ${#holding[@]} -gt 0 ] ||
    { echo "${0##*/}: --only '$only' selected no leg of any member (${members[*]}): nothing was tested" >&2; exit 2; }
  members=("${holding[@]}")
fi
failed=0 refused=0
for member in "${members[@]}"; do
  printf '\nMEMBER tools/%s\n' "$member"
  rc=0
  "${BASH:-bash}" "$here/$member" ${args[@]+"${args[@]}"} || rc=$?
  printf 'MEMBER tools/%s exit: %s\n' "$member" "$rc"
  case $rc in
    0) ;;
    129|130|143)
      printf '%s: interrupted (exit %s)\n' "${0##*/}" "$rc" >&2
      exit "$rc" ;;
    2) refused=$((refused + 1)) ;;
    *) failed=$((failed + 1)) ;;
  esac
done
printf '\n%s: %s member(s) run, %s failed, %s refused\n' "${0##*/}" "${#members[@]}" "$failed" "$refused"
[ "$refused" = 0 ] || exit 2
exit $((failed > 0 ? 1 : 0))
