(* How a run that links Verdict ends, on each path out of the process (gh-ocannl-1067).

   Verdict turns a failed check into a nonzero status from an [at_exit] teardown. OCaml runs
   [at_exit] BEFORE it reports an uncaught exception, and the teardown used to call [exit 1] -- so
   once any check had failed, an exception escaping a later case was never printed, and every row
   after it silently never ran. gh-ocannl-1016's negative control read exactly such a run as "six
   rows failed, the rest passed". The teardown now raises instead, and Verdict's uncaught-exception
   handler reports what happened.

   Every property here is about a process that FAILS, so each path runs as a child whose streams
   this one captures (the `verdict_quantified` construction): a failing child prints this
   repository's failure markers, which a green run's log must not carry. The children print through
   [Format] without flushing it too, because [Format]'s own [at_exit] flush is registered before
   Verdict's teardown, and raising out of [at_exit] skips exactly those handlers unless the ending
   runs them again. *)

open Base
open Verdict.Claims

let run_child mode = Fresh_process.run [ mode ]

(* Echoes are prefixed so a child's Verdict markers cannot become the parent's report. *)
let about mode claim ((status, stdout_text, stderr_text) as child) ~holds =
  let ok = holds status stdout_text stderr_text in
  if not ok then Fresh_process.report ~label:("child " ^ mode) child;
  p claim ok

let exited n status = Poly.equal status (Unix.WEXITED n)
let has text ~substring = String.is_substring text ~substring
let escaped_after_fail = "escaped after a failed check"
let formatted = "printed through Format, never flushed by the test"

(* The teardown line of a child whose one failed check is ["the claim"]. *)
let one_failed = Verdict.teardown_line [ "the claim" ] ^ "\n"

(* A label longer than the teardown line keeps, with a three-byte character straddling the cut: the
   cut backs off to the character's start rather than splitting it. *)
let long_label = String.make 58 'x' ^ "\u{2014}and the rest of a very long label"
let last_line text = List.last (String.split_lines text)

(* A COMPUTED label carrying line breaks, as a label built from a message or a rendered value can:
   [verdict_ratchet] refuses a literal one, so only a computed one reaches Verdict's printing. *)
let broken_label = String.concat ~sep:"\n" [ "a computed"; "label\r" ]

(* What the [label_line_breaks] child prints: each claim on ONE line of each stream it reports on,
   its label's line breaks rendered as spaces -- the rows [tools/mutation-run.sh] counts. *)
let one_line_out =
  "a computed label : false\n\
   a computed label : FAIL (got 1 want 2)\n\
   a computed label  (empty): false\n"

let one_line_err =
  "FAIL: a computed label : false\n\
   FAIL: a computed label : false\n\
   FAIL: a computed label : FAIL (got 1 want 2)\n\
   FAIL: a computed label  (empty): false\n"
  ^ Verdict.teardown_line [ broken_label; broken_label; broken_label; broken_label ]
  ^ "\n"

(* A scan's ending, through the printers the scans share: its claims, [Verdict.success_trailer],
   then the refusal-control marker section, where [Refusal_control_manifest.print_section] prints a
   claim's marker only once Verdict recorded the claim passing, a direct failure's only once its
   refusal ran, and the section's closing row whatever came before it. The diagnostics come from
   this source text, by the extractor [print] itself reads a scan's source with. *)
let manifest_source =
  {|let () =
  p "the first manifest claim" ok;
  p "the last manifest claim" true;
  if refused then Verdict.fail "the manifest refusal"|}

let manifest_ending ~first_holds ~refusal_runs =
  p "the first manifest claim" first_holds;
  p "the last manifest claim" true;
  Verdict.success_trailer ~blank_line:true [ "every manifest claim held." ];
  let source = "test/operations/verdict_teardown.ml" in
  (* The refusal's caught branch, as a scan's control records it -- or, in the silenced child, a
     mutant that keeps every claim passing while the refusal no longer runs (gh-ocannl-1251). *)
  if refusal_runs then
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format:"the manifest refusal";
  (* What a scan writes on stderr beside its golden, so that each stream has a transcript. *)
  Stdio.eprintf "Manifest diagnostics: 3 (not part of the golden)\n";
  let diagnostics = Test_utils.Refusal_control_scan.diagnostics manifest_source in
  Test_utils.Refusal_control_manifest.print_section ~source
    ~expected:(List.map diagnostics ~f:Test_utils.Refusal_control_scan.marker)
    diagnostics

(* What [tools/test-mutation-run.sh] replays through [tools/mutation-run.sh]: the runner reads
   Verdict's own text -- a raising case's claim, the teardown line, an exit inside a case, the
   [FAIL: ] prefix on every line of a failure -- and that harness, which builds no OCaml, takes the
   text from these transcripts, which this golden pins byte for byte, rather than from
   [verdict.ml]'s source. A refactor of Verdict that keeps its text keeps the harness green; a
   rewording the runner no longer reads turns it red. Every line is prefixed with its child and
   stream, so no Verdict marker starts a row of this golden. *)
let print_transcript mode ((_, stdout_text, stderr_text) : Fresh_process.t) =
  List.iter
    [ ("stdout", stdout_text); ("stderr", stderr_text) ]
    ~f:(fun (stream, text) ->
      List.iter (String.split_lines text) ~f:(fun line ->
          (* An empty line ends on the bar: no trailing blank in the golden. *)
          Stdio.printf "  %s %s |%s\n" mode stream (if String.is_empty line then "" else " " ^ line)))

(* Byte-for-byte comparisons below, so both streams are binary (Windows would rewrite "\n"): the
   children's stderr is matched line-exactly too. *)
let () =
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  Stdlib.set_binary_mode_out Stdlib.stderr true

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _ ] ->
      let fail_then_end = run_child "fail_then_end" in
      about "fail_then_end" "a failed check ends a run that reaches its end with status 1"
        fail_then_end ~holds:(fun status _ err ->
          exited 1 status && has err ~substring:one_failed && not (has err ~substring:"Fatal error"));
      about "fail_then_end" "the handlers registered before Verdict's still run at a failing end"
        fail_then_end ~holds:(fun _ out _ -> has out ~substring:formatted);
      let fail_then_raise = run_child "fail_then_raise" in
      about "fail_then_raise" "an exception escaping after a failed check is still reported"
        fail_then_raise ~holds:(fun _ _ err ->
          has err ~substring:(Printf.sprintf "Failure(%S)" escaped_after_fail));
      about "fail_then_raise" "a run an exception ended says it stopped early" fail_then_raise
        ~holds:(fun _ _ err -> has err ~substring:"STOPPED EARLY");
      about "fail_then_raise" "a run an exception ended still counts its failed checks"
        fail_then_raise ~holds:(fun _ _ err -> has err ~substring:one_failed);
      about "fail_then_raise" "a run an exception ended exits with the runtime's status 2"
        fail_then_raise ~holds:(fun status _ _ -> exited 2 status);
      about "fail_then_raise"
        "the handlers registered before Verdict's still run when it stops early" fail_then_raise
        ~holds:(fun _ out _ -> has out ~substring:formatted);
      about "fail_then_raise" "nothing after the escaping exception ran" fail_then_raise
        ~holds:(fun _ out _ -> not (has out ~substring:"never reached"));
      about "raise_clean"
        "an exception in a run with no failed check is reported as the runtime reports it"
        (run_child "raise_clean") ~holds:(fun status _ err ->
          exited 2 status
          && has err ~substring:"Failure(\"escaped from a clean run\")"
          && (not (has err ~substring:"FAILED"))
          && not (has err ~substring:"STOPPED EARLY"));
      about "fail_then_exit0" "an explicit exit 0 after a failed check exits 1"
        (run_child "fail_then_exit0") ~holds:(fun status _ err ->
          exited 1 status && has err ~substring:one_failed
          && not (has err ~substring:"STOPPED EARLY"));
      about "exit_swallowed" "a catch-all that swallows the exit still ends the run with status 1"
        (run_child "exit_swallowed") ~holds:(fun status _ _ -> exited 1 status);
      let case_raise = run_child "case_raise" in
      about "case_raise" "a case that raises is a failed claim naming the case and the exception"
        case_raise ~holds:(fun status out _ ->
          exited 1 status
          && has out
               ~substring:"first: the case ran to completion (raised Failure(\"boom\")): false\n");
      about "case_raise" "the case after a raising case still runs" case_raise
        ~holds:(fun _ out _ -> has out ~substring:"second: reached: true\n");
      let case_exit = run_child "case_exit" in
      about "case_exit" "an exit inside a case ends the run rather than failing the case" case_exit
        ~holds:(fun status out _ ->
          exited 1 status
          && (not (has out ~substring:"after the exit"))
          && not (has out ~substring:"the case ran to completion"));
      (* gh-ocannl-1084: what tools/mutation-run.sh reads to tell this ending from a run whose
         earlier case raised and whose later cases all ran. *)
      about "case_exit" "an exit inside a case after a failed check says which case stopped the run"
        case_exit ~holds:(fun _ _ err ->
          has err
            ~substring:
              "STOPPED EARLY: an exit inside case \"exits\" ended the run, so no case after it ran\n");
      (* What tools/mutation-run.sh relies on to skip a failure's rows when it matches a mutated
         run's stdout against the golden: every line of a multi-line message is a [FAIL: ] row. *)
      let multi_line_fail = "FAIL: the first line\nFAIL:   expected a\nFAIL:   got      b\n" in
      about "fail_multiline" "every line of a multi-line failure carries the FAIL prefix"
        (run_child "fail_multiline") ~holds:(fun status out err ->
          exited 1 status && String.equal out multi_line_fail && has err ~substring:multi_line_fail);
      (* What a reader of a truncated log is left with (dune keeps a long stderr's head and tail):
         the last line names the claims that failed, not only how many. *)
      about "fail_three" "a run with three failed checks names all three on its last stderr line"
        (run_child "fail_three") ~holds:(fun status _ err ->
          exited 1 status
          && Option.equal String.equal (last_line err)
               (Some
                  {|FAILED: 3 checks did not hold: "first claim", "second claim", "third claim".|}));
      about "fail_many"
        "a run with more failed checks than the line shows names the first ones, cut and on one \
         line, and counts the rest"
        (run_child "fail_many") ~holds:(fun status _ err ->
          exited 1 status
          && Option.equal String.equal (last_line err)
               (Some
                  (Printf.sprintf
                     {|FAILED: 5 checks did not hold: "%s...", "second claim", "third claim " and 2 more.|}
                     (String.make 58 'x'))));
      (* tools/mutation-run.sh lists a run's false claims as the [FAIL: …: false] (and [FAIL: …:
         FAIL …]) lines of its log and counts its stdout rows against the golden's: a claim split
         over two lines was neither listed nor one row. *)
      about "label_line_breaks"
        "a claim whose computed label carries line breaks reports on one line of each stream"
        (run_child "label_line_breaks") ~holds:(fun status out err ->
          exited 1 status && String.equal out one_line_out && String.equal err one_line_err);
      (* A scan's shape: its success trailer prints only when every claim held. *)
      let trailer_omitted = run_child "trailer_omitted" in
      about "trailer_omitted"
        "a failing run omits the success trailer it prints only when no claim failed"
        trailer_omitted ~holds:(fun status out _ ->
          exited 1 status && not (has out ~substring:"OK: "));
      (* What tools/mutation-run.sh lists a failed claim by: its stderr line is its stdout row
         behind [FAIL: ] -- for pass_fail too, whose row ends on [FAIL], bare or with a detail. A
         Verdict.fail message, which is no claim, is a [FAIL: ] row on both streams. *)
      let pass_fail_forms = run_child "pass_fail_forms" in
      about "pass_fail_forms"
        "a failed pass_fail claim's stderr line is its stdout row behind FAIL:, bare or detailed"
        pass_fail_forms ~holds:(fun status out err ->
          let rows = String.split_lines out and lines = String.split_lines err in
          let echoed suffix =
            List.exists rows ~f:(fun row ->
                String.is_suffix row ~suffix && List.mem lines ("FAIL: " ^ row) ~equal:String.equal)
          in
          exited 1 status && echoed ": FAIL" && echoed ": FAIL (got 1 want 2)");
      (* What tools/mutation-run.sh reads a failing scan's ending by: a failed claim prints no
         refusal marker, and no success trailer prints once a claim failed; every other row is the
         passing run's. *)
      let manifest_held = run_child "manifest_held"
      and manifest_failed = run_child "manifest_failed" in
      about "manifest_failed"
        "a failing scan's ending omits the failed claim's marker and the success trailer, and only \
         those"
        manifest_failed ~holds:(fun status out _ ->
          let held_status, held_out, _ = manifest_held in
          (* The failing ending derived from the passing one: the first claim false, its marker row
             (the row carrying [Refusal_control_scan.marker] of its diagnostic) and the trailer's
             rows (each [OK: ] row and the empty row before it) dropped -- and nothing else, so a
             printer dropping a row from both endings fails here. *)
          let first_marker =
            Test_utils.Refusal_control_scan.(marker (List.hd_exn (diagnostics manifest_source)))
          in
          let rec derive = function
            | "" :: ok :: rows when String.is_prefix ok ~prefix:"OK: " -> derive rows
            | ok :: rows when String.is_prefix ok ~prefix:"OK: " -> derive rows
            | "the first manifest claim: true" :: rows ->
                "the first manifest claim: false" :: derive rows
            | row :: rows when String.is_substring row ~substring:first_marker -> derive rows
            | row :: rows -> row :: derive rows
            | [] -> []
          in
          let held = String.split_lines held_out in
          exited 0 held_status && exited 1 status
          && List.length (derive held) = List.length held - 3
          && List.equal String.equal (String.split_lines out) (derive held));
      (* What it reads a mutant by that silences one refusal while every claim still holds: no
         failed check, so no teardown line -- only the refusal's marker missing, and the section's
         closing row still printed after it (gh-ocannl-1251, gh-ocannl-1318). *)
      let manifest_silenced = run_child "manifest_silenced" in
      about "manifest_silenced"
        "a scan whose refusal no longer runs, every claim holding, exits 0 and omits that \
         refusal's marker and only it, still ending on the section's closing row"
        manifest_silenced ~holds:(fun status out err ->
          let _, held_out, _ = manifest_held in
          let refusal_marker =
            Test_utils.Refusal_control_scan.(marker (List.last_exn (diagnostics manifest_source)))
          in
          let held = String.split_lines held_out in
          let derived =
            List.filter held ~f:(fun row -> not (String.is_substring row ~substring:refusal_marker))
          in
          exited 0 status
          && (not (has err ~substring:"FAILED"))
          && List.length derived = List.length held - 1
          && List.equal String.equal (String.split_lines out) derived
          && String.equal (List.last_exn held) Test_utils.Refusal_control_manifest.closing_row);
      let pass_status, pass_out, _ = run_child "pass" in
      let case_status, case_out, _ = run_child "case_pass" in
      p "a case that returns prints only what its claims print"
        (exited 0 pass_status && exited 0 case_status
        && String.equal pass_out "the claim: true\n"
        && String.equal pass_out case_out);
      Stdio.printf
        "\nTranscripts tools/test-mutation-run.sh replays through tools/mutation-run.sh:\n";
      print_transcript "case_raise" case_raise;
      print_transcript "case_exit" case_exit;
      print_transcript "trailer_omitted" trailer_omitted;
      print_transcript "pass_fail_forms" pass_fail_forms;
      print_transcript "manifest_held" manifest_held;
      print_transcript "manifest_failed" manifest_failed;
      print_transcript "manifest_silenced" manifest_silenced
  | [ _; mode ] -> (
      (* Only on the paths whose stdout is not compared byte for byte. *)
      if String.is_prefix mode ~prefix:"fail_then_" then Stdlib.Format.printf "%s@\n" formatted;
      match mode with
      | "fail_then_end" -> p "the claim" false
      | "fail_then_raise" ->
          p "the claim" false;
          ignore (failwith escaped_after_fail : unit);
          p "never reached" true
      | "raise_clean" ->
          p "the claim" true;
          failwith "escaped from a clean run"
      | "fail_then_exit0" ->
          p "the claim" false;
          Stdlib.exit 0
      | "exit_swallowed" -> (
          p "the claim" false;
          try Stdlib.exit 0 with _ -> ())
      | "case_raise" ->
          (* No backtrace, which names [verdict.ml]'s lines: its transcript is replayed, and a
             refactor of Verdict must leave it as it is. *)
          Stdlib.Printexc.record_backtrace false;
          case "first" (fun () -> failwith "boom");
          case "second" (fun () -> p "second: reached" true)
      | "case_exit" ->
          p "the claim" false;
          case "exits" (fun () -> Stdlib.exit 0);
          p "after the exit" true
      | "fail_multiline" -> fail "the first line\n  expected a\n  got      b\n"
      | "fail_three" ->
          p "first claim" false;
          p "a passing claim" true;
          fail "second claim\n  expected a\n  got      b";
          pass_fail "third claim" false
      | "fail_many" ->
          p long_label false;
          p "second claim" false;
          (* A label carrying line breaks: the teardown names it on its own one line. *)
          claim (String.concat ~sep:"\n" [ "third"; "claim\r" ]) false;
          p "fourth claim" false;
          p "fifth claim" false
      | "label_line_breaks" ->
          p broken_label false;
          claim broken_label false;
          pass_fail ~detail:(fun () -> "got 1\nwant 2") broken_label false;
          p_all broken_label [] ~f:Fn.id
      | "trailer_omitted" ->
          p "first claim" false;
          fail "second claim\n  expected a, got b";
          p "third claim" false;
          p "last claim" true;
          Verdict.success_trailer ~blank_line:true [ "every claim held."; "four claims checked." ]
      | "pass_fail_forms" ->
          pass_fail "held" true;
          pass_fail "failed" false;
          fail "a message, not a claim";
          pass_fail ~detail:(fun () -> "got 1 want 2") "detailed" false;
          (* A message ending as a claim would: still no claim (gh-ocannl-1364). *)
          fail "a message ending like a claim: false"
      | "manifest_held" -> manifest_ending ~first_holds:true ~refusal_runs:true
      | "manifest_failed" -> manifest_ending ~first_holds:false ~refusal_runs:true
      | "manifest_silenced" -> manifest_ending ~first_holds:true ~refusal_runs:false
      | "pass" -> p "the claim" true
      | "case_pass" -> case "the case" (fun () -> p "the claim" true)
      | other -> failwith ("verdict_teardown: unknown mode " ^ other))
  | _ -> failwith "verdict_teardown: expected at most one mode argument"
