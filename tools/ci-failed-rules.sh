#!/usr/bin/env bash
# Which dune rules failed, read from a build log (gh-ocannl-1304). A red CI
# build job's log otherwise ends with whatever test printed last, and the rule
# that failed is named far above it, under a `File "<dir>/dune", line N`
# header: gh-ocannl-1271's OCaml-5.3 red was blamed on the last test to print
# for three scheduled runs in a row, while the failure was a scan rule
# thousands of lines up. CI's build jobs end red runs with this listing.
#
# Usage: tools/ci-failed-rules.sh [--github] [--host FILE] [LOG...]
#   LOG        a build log: dune's own output as `tee` saved it, or a job log
#              GitHub serves (`gh api .../actions/jobs/<id>/logs`, or
#              `gh run view <id> --log-failed`, whose JOB<TAB>STEP<TAB>
#              prefixes are dropped). Timestamps, colour escapes, a byte-order
#              mark and CRLF line ends are all read through. No LOG reads stdin.
#   --github   CI mode: also emit one `::error file=...,line=...` annotation
#              per failure, so each is listed on the run's summary page, its
#              message led by the entry's `<path>:<lines>`; say
#              so when the log holds none; and say so when LOG is missing,
#              rather than refusing.
#   --host FILE  print FILE (the host-capability line, gh-ocannl-1371) above
#              the listing when it exists.
#
# Prints one entry per dune error header -- `File "<path>", line N, ...:` or
# `lines N-M` -- in the order dune reported them, a repeated one once:
#   test/operations/dune:7709-7715  (rule) alias runtest-foo
#       ran: (cd _build/default/test/operations && ./foo.exe) > ...
# For a header in a dune file the rule is named by its own fields (alias,
# aliases, name, names, target, targets), read from the source file when it is
# present under the working directory and from dune's excerpt otherwise; the
# second line is the command dune ran for it, or the error it printed. A
# header in any other file is a compile error or an expectation diff and says
# which. Warning and alert blocks that did not fail the build are left out.
# Dune's locationless `Error:` lines (an unknown alias, say) are listed after
# them: every one when the log has no header, and otherwise those printed
# before the first header (after one, such a line may be a failing action's
# own output). A log with neither prints nothing (except under
# --github), which is what a green build's log gives.
#
# Exit 0 after reading the log, 2 for a usage error or an unreadable LOG.

set -u

github=0
host=
while [ $# -gt 0 ]; do
  case $1 in
    --github) github=1 ;;
    --host)
      [ $# -ge 2 ] || { echo "ci-failed-rules.sh: --host needs a file" >&2; exit 2; }
      host=$2
      shift ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    -?*) echo "ci-failed-rules.sh: unknown argument '$1'" >&2; exit 2 ;;
    *) break ;;
  esac
  shift
done

if [ -n "$host" ] && [ -f "$host" ]; then
  cat -- "$host"
fi

for log in "$@"; do
  if [ ! -f "$log" ] || [ ! -r "$log" ]; then
    if [ "$github" = 1 ]; then
      # In CI a missing log is a finding, not a usage error: the job failed
      # before any dune build step wrote one.
      echo "No dune build log at $log: this job failed before its dune build ran."
      exit 0
    fi
    echo "ci-failed-rules.sh: cannot read $log" >&2
    exit 2
  fi
done

# Escapes as variables: a regular expression literal holding \033 or \t is
# read differently by gawk, mawk and the BSD awk macOS ships.
esc=$(printf '\033')
tab=$(printf '\t')
bom=$(printf '\357\273\277')
# Bytes, not characters: under a UTF-8 locale an awk may read the BOM as one
# character, or refuse a log's invalid UTF-8.
LC_ALL=C awk -v esc="$esc" -v tab="$tab" -v bom="$bom" -v github="$github" '
function clean(s) {
  if (index(s, bom) == 1) s = substr(s, length(bom) + 1)
  sub(/\r$/, "", s)
  gsub(esc "\\[[0-9;]*[A-Za-z]", "", s)
  # `gh run view --log` puts JOB<TAB>STEP<TAB> before the timestamp.
  if (match(s, "^[^" tab "]*" tab "[^" tab "]*" tab ts)) {
    sub("^[^" tab "]*" tab "[^" tab "]*" tab, "", s)
  }
  sub("^" ts, "", s)
  return s
}
function trim(s, n) { return length(s) > n ? substr(s, 1, n - 3) "..." : s }
# A depth-1 field of a dune stanza names it. dune files are formatted, so a
# field of the stanza itself is the one indented by exactly one space.
function name_field(src,   m) {
  if (match(src, /^ \((alias|aliases|name|names|target|targets) [^()]*\)?/)) {
    m = substr(src, 3, RLENGTH - 2)
    sub(/\)$/, "", m)
    return m
  }
  return ""
}
function rstrip(s) { sub(/[ \t]+$/, "", s); return s }
# The stanza at FIRST..LAST of PATH, when the file under the working directory
# is the one the log was written from: every line dune quoted (XL, by line
# number; at least the first) must read the same there, or a log from another
# revision would name whatever stanza has since moved onto those lines.
function stanza_from_source(path, first, last,   line, n, kind, fields, f, ok) {
  n = 0; kind = ""; fields = ""; ok = (first in xl)
  if (!ok || (getline line < path) <= 0) return ""
  do {
    n++
    if (n < first) continue
    if (n > last) break
    if ((n in xl) && rstrip(line) != rstrip(xl[n])) { ok = 0; break }
    if (n == first && match(line, /^\([A-Za-z_]+/)) kind = substr(line, 2, RLENGTH - 1)
    f = name_field(line)
    if (f != "") fields = fields (fields == "" ? "" : ", ") f
  } while ((getline line < path) > 0)
  close(path)
  if (!ok || kind == "") return ""
  return "(" kind ")" (fields == "" ? "" : " " fields)
}
function close_block() {
  if (cur == "") return
  if (!dropped) {
    if (!(cur in seen)) {
      seen[cur] = 1
      count++
      cpath[count] = path; cfirst[count] = first; clast[count] = last
      what = ""
      if (is_dune) {
        what = stanza_from_source(path, first, last)
        if (what == "") what = (xkind == "" ? "(stanza)" : "(" xkind ")") (xfields == "" ? "" : " " xfields)
      } else if (detail ~ /^Error/) {
        what = "compile error"
      } else if (path ~ /\.expected$/) {
        what = "output differs from this golden"
      } else {
        what = "differs from its expectation"
      }
      cwhat[count] = what
      cdetail[count] = detail
    }
  }
  cur = ""
}
BEGIN {
  ts = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9](\\.[0-9]+)?Z ?"
  header = "^File \"[^\"]+\", lines? [0-9]+(-[0-9]+)?, characters [0-9]+-[0-9]+:$"
  cur = ""; count = 0; nerr = 0
}
{
  s = clean($0)
  if (s ~ header) {
    close_block()
    cur = s
    path = s; sub(/^File "/, "", path); sub(/".*/, "", path)
    range = s; sub(/^[^,]*, lines? /, "", range); sub(/,.*/, "", range)
    first = range; sub(/-.*/, "", first)
    last = range; sub(/^[0-9]+-?/, "", last); if (last == "") last = first
    first += 0; last += 0
    is_dune = (path ~ /(^|\/)dune$/ || path ~ /(^|\/)dune-project$/)
    decided = 0; dropped = 0; detail = ""; xkind = ""; xfields = ""; split("", xl)
    next
  }
  if (cur == "" || decided) {
    if (s ~ /^Error: /) { if (!(s in errseen)) { errseen[s] = 1; errs[++nerr] = s; errafter[nerr] = (cur != "") } }
    next
  }
  # dune excerpt of the located source: `NNN | text`, a row of dots where it
  # elides the middle of a long stanza, and carets.
  if (match(s, /^ *[0-9]+ \|( |$)/)) {
    src = substr(s, RLENGTH + 1)
    num = s; sub(/^ */, "", num); sub(/ .*/, "", num)
    xl[num + 0] = src
    if (xkind == "" && match(src, /^\([A-Za-z_]+/)) xkind = substr(src, 2, RLENGTH - 1)
    f = name_field(src)
    if (f != "") xfields = xfields (xfields == "" ? "" : ", ") f
    next
  }
  if (s ~ /^\.+$/ || s ~ /^ *\^+ *$/ || s ~ /^ *$/) next
  decided = 1
  if (s ~ /^(Warning|Alert)/) { dropped = 1; next }
  # A header outside a dune file followed by the diff (or the command that
  # printed it) is an expectation mismatch: the header already says where.
  if (!is_dune && (s ~ /(^|[ \/(])diff / || s ~ /^--- /)) detail = ""
  else if (s ~ /^\(cd /) detail = "ran: " s
  else detail = s
}
END {
  close_block()
  if (count > 0) {
    printf "Failed dune rules (%d), in the order dune reported them:\n", count
    for (i = 1; i <= count; i++) {
      loc[i] = cpath[i] ":" cfirst[i] (clast[i] != cfirst[i] ? "-" clast[i] : "")
      printf "  %s  %s\n", loc[i], cwhat[i]
      if (cdetail[i] != "") printf "      %s\n", trim(cdetail[i], 240)
    }
    if (github) {
      for (i = 1; i <= count; i++) {
        # The message leads with the location the listing gives: the annotation
        # list on the summary page shows the message, not `file=`, and a golden
        # mismatch carries no detail to name it otherwise (gh-ocannl-1464).
        msg = loc[i] "  " cwhat[i] (cdetail[i] == "" ? "" : " -- " trim(cdetail[i], 200))
        gsub(/%/, "%25", msg); gsub(/\r/, "%0D", msg); gsub(/\n/, "%0A", msg)
        printf "::error file=%s,line=%d,endLine=%d,title=Failed dune rule::%s\n", cpath[i], cfirst[i], clast[i], msg
      }
    }
  }
  # Every locationless error when no header was found; beside located failures, those dune
  # printed before its first header -- after one, an `Error:` line may be the output of a
  # failing action.
  nshow = 0
  for (i = 1; i <= nerr; i++) if (count == 0 || !errafter[i]) shown[++nshow] = errs[i]
  if (nshow > 0) {
    printf "dune errors without a location (%d):\n", nshow
    for (i = 1; i <= nshow; i++) printf "  %s\n", trim(shown[i], 240)
    if (github) for (i = 1; i <= nshow; i++) {
      msg = trim(shown[i], 200); gsub(/%/, "%25", msg)
      printf "::error title=dune error::%s\n", msg
    }
  }
  if (count == 0 && nshow == 0 && github) {
    print "No failed dune rule in this job'"'"'s build log: the failure lies in another step."
  }
}
' "$@"
