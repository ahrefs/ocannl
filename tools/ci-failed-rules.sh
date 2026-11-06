#!/usr/bin/env bash
# Which dune rules failed, read from a build log (gh-ocannl-1304). A red CI
# build job's log otherwise ends with whatever test printed last, and the rule
# that failed is named far above it, under a `File "<dir>/dune", line N`
# header: gh-ocannl-1271's OCaml-5.3 red was blamed on the last test to print
# for three scheduled runs in a row, while the failure was a scan rule
# thousands of lines up. CI's build jobs end red runs with this listing.
#
# Usage: tools/ci-failed-rules.sh [--github] [--host FILE] [LOG...]
#        tools/ci-failed-rules.sh --names [LOG...]
#   LOG        a build log: dune's own output as `tee` saved it, or a job log
#              GitHub serves (`gh api .../actions/jobs/<id>/logs`, or
#              `gh run view <id> --log-failed`, whose JOB<TAB>STEP<TAB>
#              prefixes are dropped). Timestamps, colour escapes (raw, or as
#              the `^[[...m` text gh prints them as), a byte-order mark and
#              CRLF line ends are all read through. No LOG reads stdin.
#   --github   CI mode: also emit one `::error file=...,line=...` annotation
#              per failure, so each is listed on the run's summary page, its
#              message led by the entry's `<path>:<lines>` and carrying the
#              lines quoted under it, on one line; say
#              so when the log holds none; and say so when LOG is missing,
#              rather than refusing.
#   --host FILE  print FILE (the host-capability line, gh-ocannl-1371) above
#              the listing when it exists.
#   --names    print only one short name per line, in the listing's order, for
#              a verdict line that says what failed (gh-ocannl-2005): a
#              stanza's `name`, else its `target` less a golden-diff rule's
#              `.actual`, else its `alias`; a stanza with none of these as
#              `<path>:<line>`; any other entry's file, without its directory;
#              a locationless error as its message, cut to 80 characters. One
#              failure is one line: another header in the same file is the same
#              non-dune entry, and two different places sharing a name keep
#              both, each led by its directory. Not with --github or --host.
#
# Prints one entry per dune error header -- `File "<path>", line N, ...:` or
# `lines N-M` -- in the order dune reported them, a repeated one once:
#   test/operations/dune:7709-7715  (rule) alias runtest-foo
#       ran: (cd _build/default/test/operations && ./foo.exe) > ...
#       | FAIL: the claim the test makes, as it printed it
#       (2 more such lines)
#   test/operations/dune:7720-7726  (rule) alias runtest-bar
#       ran: (cd _build/default/test/operations && ./bar.exe)
#       last output: /bin/sh: ./bar.exe: Permission denied
# For a header in a dune file the rule is named by its own fields (alias,
# aliases, name, names, target, targets), read from the source file when it is
# present under the working directory and from dune's excerpt otherwise; the
# second line is the command dune ran for it, or the error it printed. A
# header in any other file is a compile error or an expectation diff and says
# which. Warning and alert blocks that did not fail the build are left out.
# Under each entry, behind `| `, are the first 3 lines of its own block that
# say why (gh-ocannl-1831): those holding `FAIL:`, `FAILED:`,
# `AssertionError`, `Error:` (so a Python `KeyError:` too) or OCaml's
# `Fatal error:`, and the count of the rest. An entry's block is what dune
# printed after its header and up to the first of: the next header, another
# action's `(cd ...)` command line (CI's display prints one above any output),
# or a dune error of its own (below). Of an expectation diff only the lines it
# adds (`+`) are read, since a context or removed line is not what the run
# said. Under a display that prints no command lines (not CI's), a later
# action's output can run on into the block above it. In a job log GitHub
# serves, a block also ends at a `##[` line (`##[error]Process completed ...`
# closes the build step), so the steps after the build lend it nothing.
# An entry that shows the command it ran and has no such line -- a shell error,
# a signal, a bare exit status -- shows the last non-blank line of its block
# instead, behind `last output:` so that `| ` keeps its one meaning
# (gh-ocannl-1927). The last and not the first: dune keeps the end of a long
# output and elides its middle, and the first line is usually a banner. That
# line is dune's own `Command exited with code N.` or `Command got signal
# SEGV.` when the action printed nothing. Any header ends a block, one left
# out of the listing too, so a test that prints a bare dune block shows the
# line above it.
# A header is read only at the start of its line (after the timestamps and
# prefixes GitHub adds), so a replay a test prefixes is never one; and a dune-
# file header whose excerpt omits its first or last line is not dune's -- a
# failing test's own output, quoting a fixture -- and is left out too
# (gh-ocannl-1590).
# Dune's locationless `Error:` lines (an unknown alias, say) are listed after
# them, each with the `-> required by` lines dune printed under it, which name
# the target it failed for and are its only attribution (a message dune
# wrapped above them is read as one line; ahrefs/ocannl#1720
# read a chained CreateProcess() error as the listed rule's): every one when
# the log has no header, and otherwise those printed before the first header
# and those followed by a `-> required by` line anywhere. An unchained one after
# a header may be a failing action's own output, and is quoted under that
# action's entry instead when it falls in its block. A log with neither prints
# nothing (except under --github), which is what a green build's log gives.
#
# Exit 0 after reading the log, 2 for a usage error or an unreadable LOG.

set -u

github=0
host=
names=0
while [ $# -gt 0 ]; do
  case $1 in
    --github) github=1 ;;
    --host)
      [ $# -ge 2 ] || { echo "ci-failed-rules.sh: --host needs a file" >&2; exit 2; }
      host=$2
      shift ;;
    --names) names=1 ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    -?*) echo "ci-failed-rules.sh: unknown argument '$1'" >&2; exit 2 ;;
    *) break ;;
  esac
  shift
done

if [ "$names" = 1 ] && { [ "$github" = 1 ] || [ -n "$host" ]; }; then
  echo "ci-failed-rules.sh: --names takes neither --github nor --host" >&2
  exit 2
fi

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
LC_ALL=C awk -v esc="$esc" -v tab="$tab" -v bom="$bom" -v github="$github" -v names="$names" '
function clean(s) {
  if (index(s, bom) == 1) s = substr(s, length(bom) + 1)
  sub(/\r$/, "", s)
  gsub(esc "\\[[0-9;]*[A-Za-z]", "", s)
  # gh 2.x prints a log'"'"'s ESC as the two characters `^[` (and refuses the raw
  # bytes from `gh api` without --allow-escape-sequences), so a coloured header
  # reached this reader as `^[[0;1mFile ...` and nothing was listed.
  gsub(/\^\[\[[0-9;]*[A-Za-z]/, "", s)
  # `gh run view --log` puts JOB<TAB>STEP<TAB> before the timestamp.
  if (match(s, "^[^" tab "]*" tab "[^" tab "]*" tab ts)) {
    sub("^[^" tab "]*" tab "[^" tab "]*" tab, "", s)
  }
  sub("^" ts, "", s)
  return s
}
function trim(s, n) { return length(s) > n ? substr(s, 1, n - 3) "..." : s }
# What --names calls entry I (see the header).
function rule_name(i,   n) {
  n = cwhat[i]
  if (n !~ /^\(/) { n = cpath[i]; sub(/.*\//, "", n); return n }
  if (match(n, / (name|names) [^,]*/) || match(n, / (target|targets) [^,]*/) ||
      match(n, / (alias|aliases) [^,]*/)) {
    n = substr(n, RSTART + 1, RLENGTH - 1)
    sub(/^[a-z]+ /, "", n)
    sub(/\.actual$/, "", n)
    return n
  }
  return cpath[i] ":" cfirst[i]
}
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
  # dune quotes the lines its header names, the first and the last at least
  # (it elides only the middle). A dune-file header whose excerpt leaves either
  # out is text a test printed -- a harness replaying a fixture log as its own
  # failure output -- not a dune report (gh-ocannl-1590). A header with no
  # excerpt at all is kept: dune prints none when it cannot read the file.
  if (is_dune && nx > 0 && !((first in xl) && (last in xl))) dropped = 1
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
      cq[count] = qn
      clastout[count] = (qn == 0 && detail ~ /^ran: /) ? qlast : ""
      for (j = 1; j <= qn && j <= qcap; j++) cql[count, j] = q[j]
    }
  }
  cur = ""; qopen = 0
}
# A line of the open block that says why its action failed (see the header).
function quote(s) {
  if (!qopen) return
  if (qdiff) { if (s !~ /^\+/ || s ~ /^\+\+\+ /) return }
  if (s !~ /FAIL(ED)?:|AssertionError|Error:|Fatal error:/) return
  if (++qn <= qcap) q[qn] = s
}
# An `Error:` line at the start of its line waits for what follows it: dune
# wraps a long message onto further lines (a long path, say), and a
# `-> required by` line below them makes it dune'"'"'s own error, printed apart
# from any action'"'"'s output. Anything else -- a header, a `(cd ...)` line,
# another `Error:`, a blank line, or more lines than a wrapped message takes --
# leaves it as before: listed when it came before every header, and otherwise
# also quoted in the open block; the lines held behind it are then read in turn.
function flush_pend(   k, n) {
  if (pend == "") return
  errs[++nerr] = pend; echain[nerr] = ""; echained[nerr] = 0; errafter[nerr] = pend_after
  if (pend_after) quote(pend)
  pend = ""; n = npb; npb = 0
  for (k = 1; k <= n; k++) step(pb[k])
}
# One log line, past any `Error:` line waiting above it.
function step(s) {
  if (s ~ header) {
    close_block()
    cur = s
    path = s; sub(/^File "/, "", path); sub(/".*/, "", path)
    range = s; sub(/^[^,]*, lines? /, "", range); sub(/,.*/, "", range)
    first = range; sub(/-.*/, "", first)
    last = range; sub(/^[0-9]+-?/, "", last); if (last == "") last = first
    first += 0; last += 0
    is_dune = (path ~ /(^|\/)dune$/ || path ~ /(^|\/)dune-project$/)
    decided = 0; dropped = 0; detail = ""; xkind = ""; xfields = ""; nx = 0; split("", xl)
    qopen = 0; qdiff = 0; qn = 0; qlast = ""
    return
  }
  if (cur == "" || decided) {
    if (s ~ /^Error: /) { pend = s; pend_after = (cur != ""); npb = 0; return }
    if (qopen) {
      if (s ~ /^\(cd /) qopen = 0
      else if (s ~ /^##\[/) qopen = 0
      else {
        quote(s)
        if (s !~ /^[ \t]*$/) qlast = s
      }
    }
    return
  }
  # dune excerpt of the located source: `NNN | text`, a row of dots where it
  # elides the middle of a long stanza, and carets.
  if (match(s, /^ *[0-9]+ \|( |$)/)) {
    src = substr(s, RLENGTH + 1)
    num = s; sub(/^ */, "", num); sub(/ .*/, "", num)
    xl[num + 0] = src; nx++
    if (xkind == "" && match(src, /^\([A-Za-z_]+/)) xkind = substr(src, 2, RLENGTH - 1)
    f = name_field(src)
    if (f != "") xfields = xfields (xfields == "" ? "" : ", ") f
    return
  }
  if (s ~ /^\.+$/ || s ~ /^ *\^+ *$/ || s ~ /^ *$/) return
  decided = 1
  if (s ~ /^(Warning|Alert)/) { dropped = 1; return }
  qopen = 1
  # A header outside a dune file followed by the diff (or the command that
  # printed it) is an expectation mismatch: the header already says where.
  if (!is_dune && (s ~ /(^|[ \/(])diff / || s ~ /^--- /)) { detail = ""; qdiff = 1 }
  else if (s ~ /^\(cd /) detail = "ran: " s
  else detail = s
}
BEGIN {
  ts = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9](\\.[0-9]+)?Z ?"
  header = "^File \"[^\"]+\", lines? [0-9]+(-[0-9]+)?, characters [0-9]+-[0-9]+:$"
  cur = ""; count = 0; nerr = 0; qcap = 3; qopen = 0; qn = 0; pend = ""; npb = 0; inchain = 0
}
{
  s = clean($0)
  if (pend != "") {
    if (s ~ /^-> required by /) {
      # A wrapped message reads as dune printed it before wrapping.
      for (k = 1; k <= npb; k++) pend = pend " " pb[k]
      errs[++nerr] = pend; echain[nerr] = s; echained[nerr] = 1; errafter[nerr] = pend_after
      pend = ""; npb = 0; inchain = 1
      # dune prints a message whole, so the action output above it has ended.
      qopen = 0
      next
    }
    if (npb < 8 && s !~ header && s !~ /^\(cd / && s !~ /^Error: / && s !~ /^ *$/) { pb[++npb] = s; next }
    flush_pend()
  }
  if (inchain) {
    if (s ~ /^-> required by /) { echain[nerr] = echain[nerr] "\n" s; next }
    inchain = 0
  }
  step(s)
}
END {
  flush_pend()
  close_block()
  if (count > 0 && !names) {
    printf "Failed dune rules (%d), in the order dune reported them:\n", count
    for (i = 1; i <= count; i++) {
      loc[i] = cpath[i] ":" cfirst[i] (clast[i] != cfirst[i] ? "-" clast[i] : "")
      printf "  %s  %s\n", loc[i], cwhat[i]
      if (cdetail[i] != "") printf "      %s\n", trim(cdetail[i], 240)
      for (j = 1; j <= cq[i] && j <= qcap; j++) printf "      | %s\n", trim(cql[i, j], 240)
      if (cq[i] > qcap) printf "      (%d more such lines)\n", cq[i] - qcap
      if (clastout[i] != "") printf "      last output: %s\n", trim(clastout[i], 240)
    }
    if (github) {
      for (i = 1; i <= count; i++) {
        # The message leads with the location the listing gives: the annotation
        # list on the summary page shows the message, not `file=`, and a golden
        # mismatch carries no detail to name it otherwise (gh-ocannl-1464). The
        # quoted lines follow on the same line, so the job log GitHub echoes the
        # annotation into holds no line a later read of it could take for dune'"'"'s.
        msg = loc[i] "  " cwhat[i] (cdetail[i] == "" ? "" : " -- " trim(cdetail[i], 200))
        for (j = 1; j <= cq[i] && j <= qcap; j++) msg = msg (j == 1 ? " -- " : " | ") trim(cql[i, j], 200)
        if (cq[i] > qcap) msg = msg " | (" (cq[i] - qcap) " more such lines)"
        if (clastout[i] != "") msg = msg " -- last output: " trim(clastout[i], 200)
        gsub(/%/, "%25", msg); gsub(/\r/, "%0D", msg); gsub(/\n/, "%0A", msg)
        printf "::error file=%s,line=%d,endLine=%d,title=Failed dune rule::%s\n", cpath[i], cfirst[i], clast[i], msg
      }
    }
  }
  # Every locationless error when no header was found; beside located failures, those dune
  # printed before its first header, and those it chained to a target anywhere -- after a
  # header, an unchained `Error:` line may be the output of a failing action. A repeated
  # one, chain and all, once.
  nshow = 0
  for (i = 1; i <= nerr; i++) {
    if (count > 0 && errafter[i] && !echained[i]) continue
    k = errs[i] SUBSEP echain[i]
    if (k in eseen) continue
    eseen[k] = 1; shown[++nshow] = errs[i]; schain[nshow] = echain[i]
  }
  if (nshow > 0 && !names) {
    printf "dune errors without a location (%d):\n", nshow
    for (i = 1; i <= nshow; i++) {
      printf "  %s\n", trim(shown[i], 240)
      nc = split(schain[i], ch, "\n")
      for (j = 1; j <= nc; j++) printf "    %s\n", trim(ch[j], 240)
    }
    if (github) for (i = 1; i <= nshow; i++) {
      msg = shown[i] (schain[i] == "" ? "" : " " schain[i])
      gsub(/\n/, " ", msg); msg = trim(msg, 400); gsub(/%/, "%25", msg); gsub(/\r/, "%0D", msg)
      printf "::error title=dune error::%s\n", msg
    }
  }
  if (names) {
    m = 0
    for (i = 1; i <= count; i++) {
      n = rule_name(i)
      # A dune stanza is told by where it stands, and any other entry by its file.
      k = cpath[i] SUBSEP n SUBSEP (cpath[i] ~ /(^|\/)dune(-project)?$/ ? cfirst[i] : "")
      if (k in nseen) continue
      nseen[k] = 1
      uname[++m] = n; upath[m] = cpath[i]
      if (!((n, cpath[i]) in nplace)) { nplace[n, cpath[i]] = 1; places[n]++ }
    }
    for (i = 1; i <= m; i++) {
      n = uname[i]; d = upath[i]
      if (places[n] > 1 && sub(/\/[^\/]*$/, "", d)) n = d "/" n
      print n
    }
    for (i = 1; i <= nshow; i++) print trim(shown[i], 80)
  }
  if (count == 0 && nshow == 0 && github) {
    print "No failed dune rule in this job'"'"'s build log: the failure lies in another step."
  }
}
' "$@"
