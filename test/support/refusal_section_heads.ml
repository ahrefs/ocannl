(** The fixed text {!Refusal_control_manifest.print_section} writes for readers outside its process
    (gh-ocannl-1840): the row its stdout section ends on and the heads of the two kinds of stderr
    line it writes. [tools/mutation-run.sh] reads {!closing_row}, {!not_printed_head} and the marker
    grammar; [tools/refusal-manifest-sync.sh]'s engine reads the offer lines through
    {!Refusal_control_manifest.parse_offer}. This module is their one owner: the printer and every
    reader take the text from here.

    [tools/mutation-run.sh] is Perl and reads this file's TEXT, by a grammar that is deliberately
    small and fails closed: a value it reads is exactly one line [let <name> = "<text>"], whole,
    with no double quote or backslash in the text, so the line is the value byte for byte. A named
    value in any other shape -- absent, wrapped onto a second line, written with an escape, or bound
    twice -- is refused at the runner's startup, never guessed at; [refusal_control_scan_cases]
    holds every value here to that grammar against its compiled value. Keep each binding on one line
    under the formatter's margin. *)

(** The last row of every section {!Refusal_control_manifest.print_section} prints, unconditionally:
    a golden ending on it is complete exactly when a run's stdout ends on it. *)
let closing_row = "End of synthetic controls."

(** The head of the stderr line {!Refusal_control_manifest.print_section} writes for a marker of the
    row that it does not print because a claim behind it failed, followed by the marker, [" -- "]
    and the reason -- or because its diagnostic is gone from the source. [tools/mutation-run.sh]
    lists each marker a run omitted with the reason this line gives, and one with no such line as
    one whose claim or refusal did not run (gh-ocannl-1499). *)
let not_printed_head = "Refusal marker not printed (not part of the golden): "

(** The head of a run's offer line ({!Refusal_control_manifest.offer_lines}): the head, the key,
    {!offer_answered_by} and the answering claim. *)
let offer_head = "Refusal evidence offer (not part of the golden): "

let offer_answered_by = " answered by: "

(** Marker grammar: an indented row, this prefix, a lowercase hexadecimal digest, and the separator
    before its diagnostic fragment. These strings also use the verbatim binding grammar above so the
    Perl reader derives its regex from the same values. *)
let marker_indent = "  "

let marker_prefix = "[scanner-refusal:"
let marker_hex_chars = "0123456789abcdef"
let marker_hex_length = "32"
let marker_separator = "] "
let marker ~identity ~fragment = marker_prefix ^ identity ^ marker_separator ^ fragment

let marker_identity marker =
  let open Base in
  Option.bind (String.chop_prefix marker ~prefix:marker_prefix) ~f:(fun rest ->
      let length = Int.of_string marker_hex_length in
      let identity = String.prefix rest length in
      Option.some_if
        (String.length identity = length
        && String.for_all identity ~f:(String.mem marker_hex_chars)
        && String.is_prefix (String.drop_prefix rest length) ~prefix:marker_separator)
        identity)
