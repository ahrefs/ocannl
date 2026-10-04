#!/usr/bin/env bash
# Sweep diagnostic normalization, sourced by tools/sweep.sh and direct fixtures.
# The caller provides window_fingerprint_lines (kernel-window.sh), say and die.
# sweep_fingerprint_write sets WRITTEN_FINGERPRINT to the published path.

# The error SITES in a log, one per line, in BOTH of dune's spellings: a
# diagnostic anchored to one line says `line N`, one anchored to a span --
# notably a whole stanza whose action exited non-zero, which is how every
# explicit-rule test here fails -- says `lines N-M`. Shared by `sweep_fingerprint`,
# which sorts and bounds them, and by `rerun_aliases`, which needs every one.
sweep_fingerprint_sites() { # log
  {
    # Matching only the singular left a unit whose ONLY failure
    # had that shape with an EMPTY fingerprint, and empty compares equal to
    # empty, so the consumer that diffs against the previous non-pass run read a
    # red suite as "unchanged since the last sweep" and said nothing.
    #
    # A location in a dune FILE is additionally reduced to the stanza it names.
    # Line numbers there shift under any edit to that file, so a fingerprint
    # keyed on them reports wholesale change whenever an unrelated stanza is
    # inserted above -- overstating exactly the thing the diff is asked to
    # measure. The stanza's own alias/name survives such edits, and is what a
    # reader needs anyway. A stanza is named by whichever of alias/name/target
    # it declares first -- a bare `(rule (target x.actual) ...)` has no alias to
    # give. Dune elides the middle of a long excerpt, so nothing identifying is
    # always quoted; the location stands in when none was.
    awk '
      function clear_names( i) {
        for (i in names) delete names[i]
        names_count = 0
      }
      function flush( i) {
        if (loc == "") return
        if (name != "") print prefix ", " name
        else if (names_count > 0) {
          for (i = 1; i <= names_count; i++) print prefix ", names " names[i]
        } else print loc
        loc = ""; name = ""; want = ""; opened = 0; names_done = 0
        clear_names()
      }
      /^File "[^"]+", lines? [0-9]+/ {
        flush()
        match($0, /^File "[^"]+", lines? [0-9]+(-[0-9]+)?/)
        here = substr($0, 1, RLENGTH)
        match($0, /^File "[^"]+"/)
        head = substr($0, 1, RLENGTH)
        if (head ~ /\/dune"$/ || head == "File \"dune\"") {
          loc = here; prefix = head; next
        }
        print here
        next
      }
      loc != "" {
        # The quoted excerpt: numbered source lines, plus the elision marker
        # dune prints for a long one. Anything else ends the excerpt, which
        # then never named its stanza.
        if ($0 ~ /^\.\.\.+$/) { want = ""; opened = 0; next }
        if ($0 !~ /^[0-9 ]*[0-9] \|/) { flush(); next }
        if (name != "" || names_done) next
        text = $0
        sub(/^[0-9 ]*[0-9] \| ?/, "", text)
        # Tokenized rather than matched as one regex, because the identifier is
        # not reliably a bare word sitting on its keywords line: it can be
        # quoted, and dune wraps a long field so that `(targets` ends one line
        # and its first target begins the next. A same-line regex reads both as
        # unnamed and falls back to the shifting span -- which is the failure
        # this normalization exists to avoid.
        gsub(/\(/, " ( ", text)
        gsub(/\)/, " ) ", text)
        n = split(text, tok, /[ \t]+/)
        for (i = 1; i <= n; i++) {
          if (tok[i] == "") continue
          # A dune comment runs to end of line: never the stanzas identifier.
          if (tok[i] ~ /^;/) break
          # An opening paren abandons a pending keyword: the field held a
          # nested form, as `(alias (name slow))` does, and the name is inside.
          if (tok[i] == "(") {
            opened = 1
            if (want != "names") want = ""
            continue
          }
          if (tok[i] == ")") {
            if (want == "names" && names_count > 0) names_done = 1
            opened = 0; want = ""
            if (names_done) break
            continue
          }
          if (want == "names") { names[++names_count] = tok[i]; continue }
          if (want != "") { name = want " " tok[i]; break }
          if (opened && tok[i] ~ /^(alias|name|names|target|targets)$/) want = tok[i]
          opened = 0
        }
        next
      }
      END { flush() }
    ' "$1"
  } 2>/dev/null
}

# A compact, diffable summary of what went wrong, so a caller can tell a NEW
# failure from a standing one. Metal's operations suite carries known-red tests,
# and a sweep that shouts on every red is a sweep nobody reads.
sweep_fingerprint() {
  {
    sweep_fingerprint_sites "$1"
    grep -hoE '^(Error|Fatal error|Exception)[^,]*' "$1"
    # A production compiler option vector appended to the exception message by
    # `cuda_to_ptx`, `hip_to_code`, or `compile_metal_source`. The selectors above
    # cannot reach it (it starts neither at an error site nor at
    # `Error`/`Fatal error`/`Exception`), so match the prefix each backend writes.
    # A changed option set then appears as a fingerprint diff rather than as a
    # missing line (gh-ocannl-849; Codex P2 on PR #510).
    grep -hoE '^(nvrtc|hiprtc|metal) options: .*' "$1"
  } 2>/dev/null | sort -u | head -60
  # The rtc-context block a failing GPU unit appended (see rtc_context_cmd),
  # verbatim and unsorted: it is a small fixed-size report whose ORDER is what
  # makes it readable, not a set of error sites to deduplicate. Carried into the
  # fingerprint rather than left in the log because the fingerprint is what a
  # caller diffs against yesterday's -- a toolkit upgrade or a changed option
  # vector then shows up as a diff beside the failure it explains, which is the
  # whole point (gh-ocannl-784).
  sed -n '/^=== rtc-context /,/^=== end rtc-context ===$/p' "$1" 2>/dev/null | head -40
  # The kernel window's STABLE half (window_fingerprint_lines): which signatures the
  # window held, and whether the device was being refused at all. Not the block
  # verbatim -- the window instants, the kernel timestamps and the exact count all
  # differ between two equally broken runs, and a fingerprint is compared bytewise
  # against the previous failure's, so the verbatim block would report `fingerprint
  # moved` on every repeat of a standing environment red, costing the suppression
  # that keeps this output readable. The full block stays in the log, and the
  # window and count are fields of the run record.
  # UNCAPPED, unlike everything above it, and deliberately: this list is already
  # deduplicated, so it is bounded by the number of distinct kernel message shapes
  # the bridge or the drivers can produce -- a handful, where the raw lines it summarises run to
  # hundreds. A cap here would drop exactly what the list exists for, a signature
  # never seen before, and would do it to the lexicographically last ones, which is
  # no one's idea of the least interesting. The verdict follows them for the reason
  # the serial rerun's line does: it is the one line that must survive.
  window_fingerprint_lines "$1"
  # The serial rerun's verdict (serial_rerun), after the sorted block and
  # outside its bound: which of the red stanzas stayed red on their own is the
  # first line a reader of an environment-red unit needs, and the one a
  # 60-entry bound must not be able to drop.
  grep -h '^serial rerun: ' "$1" 2>/dev/null
}

# An outcome that is not a pass, with nothing extractable from its log, is its
# own condition -- not a fingerprint of zero failures. The consumer diffs this
# file against the previous non-pass run's, and an empty file compares equal to
# an empty file, so such a unit was filed as "unchanged since the last sweep"
# and reported to nobody; that is how the missing `lines N-M` spelling above
# survived two sweeps. The sentinel makes the file differ from a real
# fingerprint in either direction, and the summary line is what a human
# actually sees: the scheduled routine quotes sweep output, so a finding that
# lives only in a written file is one nobody reads (gh-ocannl-792).
EMPTY_FINGERPRINT='(no fingerprintable diagnostics -- read the log)'

sweep_fingerprint_write() {
  local log=$1 label=$2 fp=${1%.log}.fingerprint
  sweep_fingerprint "$log" >"$fp"
  if [ ! -s "$fp" ]; then
    printf '%s\n' "$EMPTY_FINGERPRINT" >"$fp"
    say "  $label: $EMPTY_FINGERPRINT -- $log"
  fi
  WRITTEN_FINGERPRINT=$fp
}

