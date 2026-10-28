(* gh-ocannl-1438: two facts a search used to keep to itself, read back from real searches on cc.

   [privatize_drops]: the privatized preset candidates ([F_preset[... priv]]) propose a [Privatize]
   for each materialized accumulator and drop the proposals the op refuses
   ({!Autotune.privatize_extension}); the search used to discard those drops. No workload of the
   tuning corpus reaches a real refusal on cc ([privatize_census]), so the refusal is injected
   through [Autotune.on_privatize_proposal], which raises inside the extension's own refusal
   boundary: the proposal is dropped as the op's refusal would drop it. Every proposal the seam saw
   refused must be in the report under the injected cause -- counted from the seam's own calls, not
   restated -- and none of them may count as a declined candidate. A control search without the
   fault shows the presets did propose (so the faulted claims are not vacuous) and drop nothing.

   The [search_done] progress line: it now carries the report's [baseline_refused] and the privatize
   drop total. The baseline's window is refused through [Autotune.on_candidate_measured] (a NaN
   reading), as autotune_replay_refusal's control leg refuses it, so the line is read on a search
   whose record says [true] as well as on searches whose record says [false].

   The program is a row sum: one accumulation under a serial loop, the privatized presets' case, and
   no matmul site, so no sketch family seeds and each search is a handful of candidates. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Outcome = Ir.Schedule_outcome
module Progress = Stderr_capture.Progress
open Verdict.Claims

let rows = 8
let cols = 16

(* Halves of small integers: every row sum is exact in any order. *)
let av =
  Array.init (rows * cols)
    ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus:11 ~offset:(-2.5) ~stride:0.5)

let expected =
  Array.init rows ~f:(fun r ->
      Array.fold (Array.sub av ~pos:(r * cols) ~len:cols) ~init:0. ~f:( +. ))

let fault_cause =
  Outcome.Illegal_schedule
    { check = "autotune_report_drops"; detail = "the test refuses every Privatize proposal" }

type observed = {
  report : Autotune.report;
  dones : Progress.line list;  (** The call's [search_done] lines: one, if the line is honest. *)
  values : float array;
}

(* One search of a fresh row sum on cc, with progress lines captured and the routine run. *)
let search ~tag () =
  let a = TDSL.ndarray av ~label:[ tag ^ "_a" ] ~output_dims:[ rows; cols ] () in
  let%op s = a ++ "ij => i" in
  let comp = Train.forward s in
  let report = ref None in
  let (ctx, routine), lines =
    Stderr_capture.lines (fun () ->
        Autotune.tune ~search:true ~rounds:0 ~repeats:1 ~cache_dir:"" ~progress:true
          ~report:(fun r -> report := Some r)
          (Context.cpu ()) comp Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  let report = Option.value_exn ~here:[%here] !report in
  Stdio.eprintf
    "%s (not part of the golden): %s, timed %d, failed %d, baseline_refused %b, privatize drops [%s]\n\
     %!"
    tag
    (Autotune.outcome_name report.Autotune.outcome)
    report.Autotune.candidates_timed report.Autotune.candidates_failed
    (Option.is_some report.Autotune.baseline_refusal)
    (String.concat ~sep:"; "
       (List.map report.Autotune.privatize_drops ~f:(fun d ->
            Printf.sprintf "%s x%d"
              (Sexp.to_string (Outcome.sexp_of_rejection_key d.Autotune.key))
              d.Autotune.count)));
  {
    report;
    dones = Progress.events lines "search_done";
    values = Context.get_values ctx s.Tensor.value;
  }

let drop_total (r : Autotune.report) =
  List.sum (module Int) r.Autotune.privatize_drops ~f:(fun d -> d.Autotune.count)

(* The call's one [search_done] line agrees with its report on both new fields. *)
let line_agrees (o : observed) =
  match o.dones with
  | [ line ] ->
      Option.equal String.equal
        (Progress.field line "baseline_refused")
        (Some (Bool.to_string (Option.is_some o.report.Autotune.baseline_refusal)))
      && Option.equal Int.equal
           (Progress.int_field line "privatize_dropped")
           (Some (drop_total o.report))
  | _ -> false

(* --- The control: the presets propose, and nothing is dropped. --- *)

(* Every proposal the presets made, by its target's name. *)
let proposed = ref []

let control =
  Autotune.Seam.with_value Autotune.on_privatize_proposal
    (fun ~target ~over:_ -> proposed := Ir.Tnode.label_name target :: !proposed)
    (search ~tag:"control")

let () =
  Stdio.eprintf "control (not part of the golden): proposals for [%s]\n%!"
    (String.concat ~sep:", " !proposed);
  p_empty "control: the search dropped none of the proposals the presets made" ~over:!proposed
    control.report.Autotune.privatize_drops;
  p_all2 "control: the routine computes the row sums" control.values expected ~f:Float.equal;
  p "control: its search_done line carries its report's baseline refusal and drop count"
    (line_agrees control)

(* --- Every proposal refused. --- *)

let refused = ref 0

let faulted =
  Autotune.Seam.with_value Autotune.on_privatize_proposal
    (fun ~target:_ ~over:_ ->
      Int.incr refused;
      raise (Outcome.Cause_at (Outcome.Transform, fault_cause)))
    (search ~tag:"faulted")

let () =
  let r = faulted.report in
  Stdio.eprintf "faulted (not part of the golden): %d proposals refused\n%!" !refused;
  p "faulted: the privatized presets proposed a Privatize" (!refused > 0);
  p "faulted: every refused proposal is in the report, under the refusal's own cause"
    (match r.Autotune.privatize_drops with
    | [ d ] ->
        Outcome.equal_rejection_key d.Autotune.key (Outcome.key_of_cause fault_cause)
        && d.Autotune.count = !refused
    | _ -> false);
  p "faulted: a dropped proposal is not a declined candidate"
    (r.Autotune.candidates_failed = control.report.Autotune.candidates_failed
    && List.equal Outcome.equal_rejection_key
         (List.map r.Autotune.declines ~f:(fun d -> d.Autotune.key))
         (List.map control.report.Autotune.declines ~f:(fun d -> d.Autotune.key)));
  p_all2 "faulted: the routine computes the row sums" faulted.values expected ~f:Float.equal;
  p "faulted: its search_done line carries its report's baseline refusal and drop count"
    (line_agrees faulted)

(* --- The baseline's window refused. --- *)

let baseline_refused =
  Autotune.Seam.with_value Autotune.on_candidate_measured
    (fun ~label ~digest:_ ms -> if String.equal label "baseline" then Float.nan else ms)
    (search ~tag:"baseline_refused")

let () =
  p "refused baseline: the report records the baseline's refused window"
    (Option.is_some baseline_refused.report.Autotune.baseline_refusal);
  p "refused baseline: its search_done line says so too" (line_agrees baseline_refused)
