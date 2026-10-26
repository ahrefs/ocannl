(* Count claims over a physical search's report, gated on what moves the counts (gh-ocannl-1632).

   A search drops a candidate's digest from its dedup set when that digest's window is refused (for
   any cause), cut as hopeless (gh-ocannl-1527) or stopped by the budget, so an equivalent later
   seed is timed afresh instead of deduplicated (gh-ocannl-1484: on a loaded host a refused
   whole-routine window let its per-segment twin be timed). A claim pinning how many candidates a
   search timed or deduplicated is therefore exact only on a report where nothing moved, and which
   departures move it depends on what the count counts:

   - [`Reached]: a count of windows REACHED, refused ones included -- the family counters
   ([fiss_sketch_timed], [split_reduce_timed], [mma_timed]) -- or of dedups. Refusals and hopeless
   cuts do not move it; a re-timed departed digest ([timings_retimed]) adds a window, and a budget
   cut ends the seeding early. - [`Admitted]: a count of ADMITTED windows ([candidates_timed], what
   ranks). Every departure moves it: a refused, hopeless or budget-stopped window is not admitted,
   and a re-timed digest may be.

   The gate reads only those departure fields, never the count under test, so a regression that
   changes the count on a quiet host still fails the claim. Where the gate is closed the claim is an
   [`Environment] skip ({!Verdict.gated}): announced on stderr with the departures, recorded for the
   sweep, and never evaluated, so a closed gate cannot pass a claim silently. The skip names the
   host's measurement, not a backend: the search ran, its counts just do not say what the claim
   asks. *)

open Base

type count = [ `Reached | `Admitted ]

(* The departures a report records that can move a [count] claim, rendered for stderr; empty when
   none did. *)
let departures (count : count) (r : Autotune.report) =
  List.filter_opt
    [
      (match count with
      | `Admitted when r.timings_contended > 0 ->
          Some
            (Printf.sprintf "refused=%d (unbatched=%d unreadable=%d)" r.timings_contended
               r.timings_unbatched r.timings_unreadable)
      | `Admitted | `Reached -> None);
      (match count with
      | `Admitted when r.timings_hopeless > 0 ->
          Some (Printf.sprintf "hopeless=%d" r.timings_hopeless)
      | `Admitted | `Reached -> None);
      (if r.timings_retimed > 0 then Some (Printf.sprintf "retimed=%d" r.timings_retimed) else None);
      (if r.budget_cut then Some "budget_cut" else None);
    ]

(* What a skip is announced on: the condition, since the backend did run the search. *)
let skipped_on = "a search whose timing windows left its dedup set"

(** [claim ~count ~tag reports label ok] is [Verdict.p label ok] when none of [reports] records a
    departure that moves a [count] claim, and an [`Environment] skip of [label] otherwise, with each
    report's departures on stderr under [tag] (not part of the golden). A claim comparing two
    searches passes both reports. *)
let claim ~(count : count) ~tag reports label ok =
  let moved = List.map reports ~f:(departures count) |> List.filter ~f:(Fn.non List.is_empty) in
  List.iter moved ~f:(fun ds ->
      Stdio.eprintf "%s: the count claim is skipped, its search's windows departed: %s\n%!" tag
        (String.concat ~sep:" " ds));
  Verdict.gated ~aggregation:`Environment ~when_:(List.is_empty moved) ~on:skipped_on label ok
