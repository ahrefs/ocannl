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

(* Byte-for-byte comparisons below, so stdout is binary (Windows would rewrite "\n"). *)
let () = Stdlib.set_binary_mode_out Stdlib.stdout true

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _ ] ->
      let fail_then_end = run_child "fail_then_end" in
      about "fail_then_end" "a failed check ends a run that reaches its end with status 1"
        fail_then_end ~holds:(fun status _ err ->
          exited 1 status
          && has err ~substring:"FAILED: 1 check did not hold."
          && not (has err ~substring:"Fatal error"));
      about "fail_then_end" "the handlers registered before Verdict's still run at a failing end"
        fail_then_end ~holds:(fun _ out _ -> has out ~substring:formatted);
      let fail_then_raise = run_child "fail_then_raise" in
      about "fail_then_raise" "an exception escaping after a failed check is still reported"
        fail_then_raise ~holds:(fun _ _ err ->
          has err ~substring:(Printf.sprintf "Failure(%S)" escaped_after_fail));
      about "fail_then_raise" "a run an exception ended says it stopped early" fail_then_raise
        ~holds:(fun _ _ err -> has err ~substring:"STOPPED EARLY");
      about "fail_then_raise" "a run an exception ended still counts its failed checks"
        fail_then_raise ~holds:(fun _ _ err -> has err ~substring:"FAILED: 1 check did not hold.");
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
          exited 1 status
          && has err ~substring:"FAILED: 1 check did not hold."
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
      let pass_status, pass_out, _ = run_child "pass" in
      let case_status, case_out, _ = run_child "case_pass" in
      p "a case that returns prints only what its claims print"
        (exited 0 pass_status && exited 0 case_status
        && String.equal pass_out "the claim: true\n"
        && String.equal pass_out case_out)
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
          case "first" (fun () -> failwith "boom");
          case "second" (fun () -> p "second: reached" true)
      | "case_exit" ->
          p "the claim" false;
          case "exits" (fun () -> Stdlib.exit 0);
          p "after the exit" true
      | "pass" -> p "the claim" true
      | "case_pass" -> case "the case" (fun () -> p "the claim" true)
      | other -> failwith ("verdict_teardown: unknown mode " ^ other))
  | _ -> failwith "verdict_teardown: expected at most one mode argument"
