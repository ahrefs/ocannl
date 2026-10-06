#!/usr/bin/env bash
# Opposing controls for the aggregate: run all selected members, preserve
# failures, select the CI groups, and refuse arguments before executing work.
# Usage: tools/test-test-harnesses.sh [--keep|--help]
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3 perl
harness_scratch test-harnesses
fixture="$TMP/repository with spaces"
mkdir -p "$fixture/tools"
fixture=$(CDPATH= cd -- "$fixture" && pwd)
cp "$here/test-harnesses.sh" "$fixture/tools/test-harnesses.sh"
subject="$fixture/tools/test-harnesses.sh"
# Derive fixture membership through the public interface, rather than keeping
# another list that would go stale as CI gains a harness.
"$subject" --list >"$TMP/all" || exit 2
"$subject" --shell --list >"$TMP/shell" || exit 2
"$subject" --toolchain --list >"$TMP/toolchain" || exit 2
"$subject" --promotion --list >"$TMP/promotion" || exit 2
[ -s "$TMP/shell" ] && [ -s "$TMP/toolchain" ] && [ -s "$TMP/promotion" ] || exit 2
cat "$TMP/shell" "$TMP/toolchain" "$TMP/promotion" | sort >"$TMP/groups"
sort "$TMP/all" >"$TMP/sorted"
if cmp -s "$TMP/groups" "$TMP/sorted" \
   && [ "$(sort -u "$TMP/all" | wc -l)" -eq "$(wc -l <"$TMP/all")" ]; then
  report 0 'groups partition the complete list without duplicates'
else report 1 'groups partition the complete list without duplicates'; fi
while IFS= read -r script; do
  mkdir -p "$fixture/$(dirname "$script")"
  cat >"$fixture/$script" <<'STUB'
#!/usr/bin/env bash
name=${0#"$FIXTURE/"}
printf '%s\n' "$name" >>"$RECORD"
[ "$PWD" = "$FIXTURE" ] || exit 90
if [ "$name" = "${FAIL_MEMBER:-}" ]; then
  if [ -n "${FAIL_SIGNAL:-}" ]; then
    # Background launchers can inherit ignored INT; reset the member itself
    # before the real signal so the oracle is independent of that parent.
    exec perl -e '$SIG{$ARGV[0]} = "DEFAULT"; kill $ARGV[0], $$;' "$FAIL_SIGNAL"
  fi
  exit 7
fi
exit 0
STUB
  chmod +x "$fixture/$script"
done <"$TMP/all"
export FIXTURE="$fixture" RECORD="$TMP/record"
run() {
  : >"$RECORD"
  got=0
  (cd "$TMP" && "$subject" "$@") >"$TMP/out" 2>&1 || got=$?
}
for group in all shell toolchain promotion; do
  case $group in all) run ;; *) run "--$group" ;; esac
  if [ "$got" = 0 ] && cmp -s "$TMP/$group" "$RECORD"; then
    report 0 "$group executes every selected member from the repository root"
  else report 1 "$group executes every selected member from the repository root"; fi
done
export FAIL_MEMBER
FAIL_MEMBER=$(head -n 1 "$TMP/all")
run
if [ "$got" = 1 ] && cmp -s "$TMP/all" "$RECORD" \
   && grep -q 'exit: 7' "$TMP/out" && grep -q '1 failed$' "$TMP/out"; then
  report 0 'a failing first harness stays red while every later harness runs'
else report 1 'a failing first harness stays red while every later harness runs'; fi
# A fail-fast twin must lose the continuation property, proving the oracle
# above sees the defect instead of merely observing a failing command.
python3 - "$subject" "$TMP/fail-fast.sh" <<'PY_CONTROL'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
assert '"$root/$script" || rc=$?' in text
Path(sys.argv[2]).write_text(text.replace('"$root/$script" || rc=$?',
                                        '"$root/$script" || exit 1'))
PY_CONTROL
# Keep the twin at the same tools/ depth so root resolution stays identical.
cp "$TMP/fail-fast.sh" "$fixture/tools/fail-fast.sh"
chmod +x "$fixture/tools/fail-fast.sh"
: >"$RECORD"
twin_rc=0
"$fixture/tools/fail-fast.sh" >"$TMP/twin-out" 2>&1 || twin_rc=$?
if [ "$twin_rc" = 1 ] && ! cmp -s "$TMP/all" "$RECORD"; then
  report 0 'the fail-fast twin loses the all-members oracle'
else report 1 'the fail-fast twin loses the all-members oracle'; fi
# Actual signals on the member preserve cancellation and suppress later work.
export FAIL_SIGNAL FAIL_MEMBER
FAIL_MEMBER=$(head -n 1 "$TMP/all")
head -n 1 "$TMP/all" >"$TMP/first"
python3 - "$subject" "$fixture/tools/continue.sh" <<'PY_INTERRUPT'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
assert '129|130|143)' in text
Path(sys.argv[2]).write_text(text.replace('129|130|143)', '250)'))
PY_INTERRUPT
chmod +x "$fixture/tools/continue.sh"
for signal in HUP INT TERM; do
  case $signal in HUP) want=129 ;; INT) want=130 ;; TERM) want=143 ;; esac
  FAIL_SIGNAL=$signal
  run
  if [ "$got" = "$want" ] && cmp -s "$TMP/first" "$RECORD" \
     && grep -q "^harnesses: interrupted (exit $want)$" "$TMP/out"; then
    report 0 "$signal stops after the interrupted member and preserves exit $want"
  else report 1 "$signal stops after the interrupted member and preserves exit $want"; fi
  : >"$RECORD"
  continuation_rc=0
  "$fixture/tools/continue.sh" >"$TMP/continuation-out" 2>&1 || continuation_rc=$?
  if [ "$continuation_rc" = 1 ] && cmp -s "$TMP/all" "$RECORD"; then
    report 0 "$signal continuation twin launches the later members and loses the interrupt oracle"
  else report 1 "$signal continuation twin launches the later members and loses the interrupt oracle"; fi
done
unset FAIL_MEMBER FAIL_SIGNAL
# Empty membership is an invocation error, never a vacuous green tier.
python3 - "$subject" "$fixture/tools/empty.sh" <<'PY_EMPTY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
head, tail = text.split("cat <<'HARNESS_LIST'\n", 1)
_, tail = tail.split('HARNESS_LIST\n', 1)
Path(sys.argv[2]).write_text(head + "cat <<'HARNESS_LIST'\nHARNESS_LIST\n" + tail)
PY_EMPTY
chmod +x "$fixture/tools/empty.sh"
empty_rc=0
"$fixture/tools/empty.sh" --list >"$TMP/empty-out" 2>&1 || empty_rc=$?
if [ "$empty_rc" = 2 ] && grep -q '^no harnesses selected$' "$TMP/empty-out"; then
  report 0 'empty membership is refused even for read-only listing'
else report 1 'empty membership is refused even for read-only listing'; fi
for args in unknown conflicting; do
  case $args in unknown) run --unknown ;; conflicting) run --shell --toolchain ;; esac
  if [ "$got" = 2 ] && [ ! -s "$RECORD" ]; then
    report 0 "$args arguments refuse before any harness executes"
  else report 1 "$args arguments refuse before any harness executes"; fi
done
run --list
if [ "$got" = 0 ] && [ ! -s "$RECORD" ] && cmp -s "$TMP/all" "$TMP/out"; then
  report 0 'listing membership executes no harness'
else report 1 'listing membership executes no harness'; fi
finish
