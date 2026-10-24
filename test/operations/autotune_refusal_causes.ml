(* gh-ocannl-1517: a refused timing window names its cause, in the report and in what the search
   writes about it.

   The admission gate refuses a window for one of three causes: a queued calibration that measured
   no batch ([unbatched]), a degenerate clock reading (a minimum that is not finite and positive),
   and host contention ([contended]). The report used to count the last two as one, and one [NOT
   TIMED] line named both, so a skip excused by a refused window could not say which it met -- and a
   zero reading, against which every positive sample reads as a stall, was usually named contention.
   [Autotune.timing_admission] now names one cause per window, the reading before contention, and
   the report keeps the degenerate readings apart ([timings_unreadable], with
   [Autotune.contention_refusals] the part left to contention).

   Each cause the search can meet on cc is driven on every host through an existing or a test-only
   seam, never left to the machine: the degenerate reading through [Autotune.on_candidate_measured]
   returning NaN, with host contention cleared so the scripted reading is the one refused, and
   contention through [Autotune.with_contended_test_windows]. Each search's report, its
   [search_done] progress line and its log lines must all name that cause and only that one.
   ([unbatched] is a CUDA/HIP queued calibration's verdict, pinned by autotune_timing_modes.)

   The program is a row sum, as in autotune_report_drops: no matmul site, so each search is a
   handful of candidates. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Progress = Stderr_capture.Progress
open Verdict.Claims

(* --- The classifier: one cause per window, the reading before contention. --- *)

let reading ?(contended = false) ?(unbatched = false) ms =
  { Autotune.ms; contended; unbatched; samples = 16; dispersed = contended; spread = 1. }

let cases =
  [
    ("a positive reading", reading 1.0, Autotune.Admitted);
    ("a contended positive reading", reading ~contended:true 1.0, Autotune.Refused_contended);
    ("a zero reading", reading 0., Autotune.Refused_unreadable);
    (* The case the merged count misnamed: a zero minimum is also the dispersion floor. *)
    ("a contended zero reading", reading ~contended:true 0., Autotune.Refused_unreadable);
    ("a contended negative reading", reading ~contended:true (-1.), Autotune.Refused_unreadable);
    ("a NaN reading", reading Float.nan, Autotune.Refused_unreadable);
    ("an infinite reading", reading Float.infinity, Autotune.Refused_unreadable);
    ("an unbatched reading", reading ~unbatched:true 1.0, Autotune.Refused_unbatched);
    ( "an unbatched contended NaN reading",
      reading ~unbatched:true ~contended:true Float.nan,
      Autotune.Refused_unbatched );
  ]

let () =
  p_all "each window's admission names its cause, the reading before contention" cases
    ~f:(fun (what, r, want) ->
      let got = Autotune.timing_admission r in
      if not (Poly.equal got want) then
        Stdio.eprintf "%s: %s, expected %s\n%!" what
          (Autotune.batch_admission_name got)
          (Autotune.batch_admission_name want);
      Poly.equal got want);
  p_all "the admission gate admits exactly the windows the classifier admits" cases
    ~f:(fun (_, r, _) ->
      Bool.equal
        (Option.is_some (Autotune.admitted_timing_ms r))
        (Poly.equal (Autotune.timing_admission r) Autotune.Admitted))

(* --- The test-window scopes: what the admission sees, innermost scope first. --- *)

let admitted_as r =
  Autotune.timing_admission (Autotune.apply_measurement_seam ~label:"scope" ~digest:"scope" r)

let () =
  p "outside the scopes a window keeps its own contention flag"
    (Poly.equal (admitted_as (reading 1.0)) Autotune.Admitted
    && Poly.equal (admitted_as (reading ~contended:true 1.0)) Autotune.Refused_contended);
  p "the contended scope refuses a positive reading for contention"
    (Poly.equal
       (Autotune.with_contended_test_windows (fun () -> admitted_as (reading 1.0)))
       Autotune.Refused_contended);
  p "the contended scope leaves a degenerate reading its own cause"
    (Poly.equal
       (Autotune.with_contended_test_windows (fun () -> admitted_as (reading 0.)))
       Autotune.Refused_unreadable);
  p "the innermost scope applies"
    (Poly.equal
       (Autotune.with_contended_test_windows (fun () ->
            Autotune.with_uncontended_test_windows (fun () ->
                admitted_as (reading ~contended:true 1.0))))
       Autotune.Admitted
    && Poly.equal
         (Autotune.with_uncontended_test_windows (fun () ->
              Autotune.with_contended_test_windows (fun () -> admitted_as (reading 1.0))))
         Autotune.Refused_contended);
  p "leaving the scopes restores the window's own flag"
    (Poly.equal (admitted_as (reading 1.0)) Autotune.Admitted)

(* --- Searches whose every window is refused for one cause. --- *)

let rows = 8
let cols = 16

let av =
  Array.init (rows * cols)
    ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus:11 ~offset:(-2.5) ~stride:0.5)

type observed = {
  report : Autotune.report;
  dones : Progress.line list;  (** The call's [search_done] lines: one, if the line is honest. *)
  refusals : string list;  (** The log lines about a refused window, the baseline's included. *)
  measured : int;  (** Windows that reached [on_candidate_measured]. *)
}

(* One search of a fresh row sum on cc under [scope], with every window that reaches the measurement
   seam read as [script]'s value, and its progress and log lines captured. *)
let search ~tag ~scope ~script =
  let a = TDSL.ndarray av ~label:[ tag ^ "_a" ] ~output_dims:[ rows; cols ] () in
  let%op s = a ++ "ij => i" in
  let comp = Train.forward s in
  let report = ref None and measured = ref 0 in
  let ctx, lines =
    Stderr_capture.lines (fun () ->
        Autotune.Seam.with_value Autotune.on_candidate_measured
          (fun ~label:_ ~digest:_ ms ->
            Int.incr measured;
            script ms)
          (fun () ->
            scope (fun () ->
                fst
                  (Autotune.tune ~search:true ~rounds:0 ~repeats:1 ~timing:Autotune.Isolated
                     ~cache_dir:"" ~progress:true ~log:true
                     ~report:(fun r -> report := Some r)
                     (Context.cpu ()) comp Ir.Indexing.Empty))))
  in
  Context.release ctx;
  let report = Option.value_exn ~here:[%here] !report in
  let refusals =
    List.filter lines ~f:(fun line ->
        String.is_prefix line ~prefix:"autotune: "
        && (String.is_substring line ~substring:": NOT TIMED, "
           || String.is_substring line ~substring:"baseline timing refused: "))
  in
  Stdio.eprintf
    "%s (not part of the golden): %s, timed %d, refused %d (unreadable %d, unbatched %d, \
     contention %d), baseline refusal %s, %d refusal log lines, %d windows measured\n\
     %!"
    tag
    (Autotune.outcome_name report.Autotune.outcome)
    report.Autotune.candidates_timed report.Autotune.timings_contended
    report.Autotune.timings_unreadable report.Autotune.timings_unbatched
    (Autotune.contention_refusals report)
    (Option.value_map report.Autotune.baseline_refusal ~default:"none"
       ~f:Autotune.batch_admission_name)
    (List.length refusals) !measured;
  { report; dones = Progress.events lines "search_done"; refusals; measured = !measured }

(* The claims every one-cause search must satisfy: [counted] is the report's count for that cause,
   [names] the phrase its log lines carry and [other] the other cause's phrase. *)
let one_cause ~tag (o : observed) ~cause ~counted ~names ~other =
  let r = o.report in
  p
    (tag ^ ": the search refused windows and timed none")
    (r.Autotune.timings_contended > 0 && r.Autotune.candidates_timed = 0);
  p (tag ^ ": every refused window is counted under that cause") (counted r = r.timings_contended);
  p
    (tag ^ ": the named causes are subsets of the refused windows")
    (Autotune.contention_refusals r >= 0);
  p
    (tag ^ ": the baseline's refusal names that cause")
    (Option.equal Poly.equal r.baseline_refusal (Some cause));
  p
    (tag ^ ": its search_done line carries the report's counts")
    (match o.dones with
    | [ line ] ->
        Option.equal Int.equal (Progress.int_field line "contended") (Some r.timings_contended)
        && Option.equal Int.equal (Progress.int_field line "unreadable") (Some r.timings_unreadable)
        && Option.equal Int.equal (Progress.int_field line "unbatched") (Some r.timings_unbatched)
        && Option.equal String.equal
             (Progress.field line "baseline_refused")
             (Some (Bool.to_string (Option.is_some r.baseline_refusal)))
    | _ -> false);
  p (tag ^ ": one log line per refused window") (List.length o.refusals = r.timings_contended);
  p_all (tag ^ ": every refused window's log line names that cause and not the other") o.refusals
    ~f:(fun line ->
      String.is_substring line ~substring:names && not (String.is_substring line ~substring:other))

(* What each cause's lines say, and the bare word the other cause's lines would carry: the merged
   line ("contention or a degenerate clock reading") carries both, and fails both searches. *)
let degenerate_phrase = "a degenerate clock reading"
let contention_phrase = "host contention"

(* A clock that resolved nothing, scripted for every window, with host contention cleared so the
   scripted reading is the one the gate refuses. *)
let () =
  let o =
    search ~tag:"degenerate" ~scope:Autotune.with_uncontended_test_windows ~script:(fun _ ->
        Float.nan)
  in
  one_cause ~tag:"degenerate clock" o ~cause:Autotune.Refused_unreadable
    ~counted:(fun r -> r.Autotune.timings_unreadable)
    ~names:degenerate_phrase ~other:"contention"

(* Contention on every window: none reaches the measurement seam, which sees admitted windows
   only. *)
let () =
  let o = search ~tag:"contended" ~scope:Autotune.with_contended_test_windows ~script:Fn.id in
  one_cause ~tag:"contention" o ~cause:Autotune.Refused_contended
    ~counted:Autotune.contention_refusals ~names:contention_phrase ~other:"degenerate";
  p "contention: no refused window reached the measurement seam" (o.measured = 0)
