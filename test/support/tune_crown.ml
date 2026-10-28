(* What a tuning call crowned, and how the beam finals decided it, as one line of fields.

   A [rounds:0] training test ships the seed that won its timing race, so a golden diff on another
   box is often a different seed, not a different computation; and a value defect a schedule
   introduces lives in one seed, not in the tuner. Without this line the crown is readable only
   through [autotune_log], so the training tests and the tuning benches print it, the tests on
   stderr tagged [(not part of the golden)] since which seed wins is the timing race's outcome. One
   renderer, so a bench line and a test line spell the same report the same way. *)

open Base

(** {!Autotune.finals_outcome_name} as one field word: ["not run"] reads as [not_run]. *)
let finals_word (o : Autotune.finals_outcome) =
  String.substr_replace_all ~pattern:" " ~with_:"_" (Autotune.finals_outcome_name o)

(** The finals' outcome and how many contenders they re-timed (zero when they did not run). *)
let finals_fields (r : Autotune.report) =
  Printf.sprintf "finals=%s finals_contenders=%d"
    (finals_word r.Autotune.finals_outcome)
    (List.length r.Autotune.finals)

(** The call's outcome, the crowned candidate's label and search reading, then {!finals_fields}. The
    label is [""] exactly when nothing was timed, which [outcome] then explains. *)
let fields (r : Autotune.report) =
  Printf.sprintf "outcome=%s crowned=%S best_ms=%.4f %s"
    (Autotune.outcome_name r.Autotune.outcome)
    r.Autotune.best_label r.Autotune.best_ms (finals_fields r)

(** A [~report] callback for a test: prints [<site>: tune <fields> (not part of the golden)] on
    stderr. *)
let eprint ~site (r : Autotune.report) =
  Stdio.eprintf "%s: tune %s (not part of the golden)\n%!" site (fields r)
