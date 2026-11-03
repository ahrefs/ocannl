(** The fixed text {!Refusal_control_manifest.print_section} writes for readers outside its process
    (gh-ocannl-1840): the row its stdout section ends on and the heads of the two kinds of stderr
    line it writes. [tools/mutation-run.sh] reads {!closing_row} and {!not_printed_head};
    [tools/refusal-manifest-sync.sh]'s engine reads the offer lines through
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
