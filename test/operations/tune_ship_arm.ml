(* gh-ocannl-638: shipping a CHOSEN placement arm of [Train.tune_placements] (config
   [tune_ship_arm], argument [?ship_arm]).

   The measurement gap this closes: the search ships whichever arm is faster, so a report that
   profiles arm A's kernels and quotes arm A's ratios can rest on routines that were compiled,
   dispatched and timed but never executed against a reference — benchmarks/report-gh612-hip.md
   states exactly that for three of its four cells. So the claims here are not only "the requested
   arm ships": each forced run's routine is EXECUTED and its values compared against an independent
   plain compile, which is the check AGENTS.md requires and the one a forced arm exists to make
   possible.

   Discriminating, rather than merely present: the two forced runs must ship placements that
   actually differ (the intermediate virtual under A, materialized under B) — otherwise "arm A
   shipped" and "arm B shipped" would be the same artifact and the selector would be untested. And
   both runs must still report two arms, since forcing changes what ships, not what is measured. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
open Verdict.Claims

let approx a b = Float.(abs (a -. b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 8

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "tsa_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "tsa_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  let comp = named "tsa" (Train.forward t2) in
  (* The independent reference: a plain compile, no search, no placement decisions of its own. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in
  (* The spellings the config key accepts, and the rejection of one it does not: [tune_ship_arm] is
     read through this parser, so a typo must fail loudly rather than resolve to the default. *)
  let parses s expected =
    match Train.placement_arm_of_string ~source:"test" s with
    | a -> Poly.equal a expected
    | exception _ -> false
  in
  p "\"auto\" parses as the measured winner" (parses "auto" Train.Measured_winner);
  p "\"a\" and \"default\" parse as arm A"
    (parses "a" Train.Force_arm_a && parses "default" Train.Force_arm_a);
  p "\"B\" and \"materialize-all\" parse as arm B (case-insensitively)"
    (parses "B" Train.Force_arm_b && parses "materialize-all" Train.Force_arm_b);
  p "an unknown spelling is rejected rather than defaulted"
    (match Train.placement_arm_of_string ~source:"test" "arm-a" with
    | _ -> false
    | exception Invalid_argument _ -> true);
  let run ?ship_arm ?inline_flips () =
    let reports = ref [] and shipped = ref None in
    let ctx_t, routine_t =
      Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
        ~report:(fun r -> reports := r :: !reports)
        ~on_ship:(fun what -> shipped := Some what)
        ?ship_arm ?inline_flips (Context.auto ()) t2 comp Ir.Indexing.Empty
    in
    let ctx_t = Context.run ctx_t routine_t in
    let got = Context.get_values ctx_t t2.Tensor.value in
    let materialized =
      Tn.Placements.is_materialized_peek (Context.placements ctx_t) mc.Tensor.value
    in
    (List.rev !reports, !shipped, materialized, got)
  in
  (* --- Forced arm A: the default-placement artifact ships and is executed. --- *)
  let a_reports, a_shipped, a_materialized, a_got = run ~ship_arm:Train.Force_arm_a () in
  p "forcing arm A ships arm A" (Option.equal String.equal a_shipped (Some "A"));
  p_all2 "the forced arm A routine executes and matches the plain compile" a_got expected ~f:approx;
  p "the shipped arm A context leaves the intermediate virtual" (not a_materialized);
  p "forcing arm A still searched both arms" (List.length a_reports = 2);
  (* --- Forced arm B: the materialize-all artifact ships and is executed. --- *)
  let b_reports, b_shipped, b_materialized, b_got = run ~ship_arm:Train.Force_arm_b () in
  p "forcing arm B ships arm B" (Option.equal String.equal b_shipped (Some "B"));
  p_all2 "the forced arm B routine executes and matches the plain compile" b_got expected ~f:approx;
  p "the shipped arm B context materializes the intermediate" b_materialized;
  p "forcing arm B still searched both arms" (List.length b_reports = 2);
  p "the two forced runs ship different placements, so the arms are distinguishable artifacts"
    (not (Bool.equal a_materialized b_materialized));
  (* --- A forced arm suppresses the flip refinement: the shipped result is the arm itself. --- *)
  let _, f_shipped, _, f_got = run ~ship_arm:Train.Force_arm_a ~inline_flips:3 () in
  p "a flip budget under a forced arm still ships that arm, not a refined vector"
    (Option.equal String.equal f_shipped (Some "A"));
  p_all2 "the flip-budgeted forced run also matches the plain compile" f_got expected ~f:approx;
  (* --- The default: [on_ship] describes what actually shipped, which the measured comparison
         decides. This is what a benchmark attributes its losses by (gh-ocannl-546's arms JSON). --- *)
  (* --- A raising [on_ship] is the caller's failure and propagates unchanged, like [report]'s: it is
         not reclassified as an arm failure (which would hide it and ship something anyway). The
         other half of that path -- releasing the routine the caller never received a handle to, so
         a repeatedly-failing callback cannot accumulate one rooted footprint per call -- is not
         observable from here; this pins the half that is. --- *)
  let raised =
    match
      Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
        ~on_ship:(fun _ -> failwith "tsa: on_ship says no")
        ~ship_arm:Train.Force_arm_a (Context.auto ()) t2 comp Ir.Indexing.Empty
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"tsa: on_ship says no"
    | exception _ -> false
  in
  p "an exception from on_ship propagates to the caller unchanged" raised;
  (* --- Both arms failed, with an arm forced: the forced arm's failure is the one that propagates.
     The caller asked for that artifact, so reporting the other search's exception would name a
     search it did not select. Injected through [Autotune.on_candidate_attempt] and told apart by
     the arm-report count, the way autotune_arm_containment.ml does: arm A's search is the one
     running before any arm has reported. --- *)
  let both_arms_fail ship_arm =
    let reported = ref 0 in
    (Autotune.on_candidate_attempt :=
       fun _ -> failwith (Printf.sprintf "tsa arm %s" (if !reported = 0 then "A" else "B")));
    Exn.protect
      ~f:(fun () ->
        match
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun _ -> Int.incr reported)
            ~ship_arm (Context.auto ()) t2 comp Ir.Indexing.Empty
        with
        | _ -> "no failure"
        | exception Failure msg -> msg
        | exception exn -> Exn.to_string exn)
      ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  (* Discriminating, not merely nonzero: the two settings must name DIFFERENT arms, or the claim
     would hold for a [tune_placements] that always propagated arm A. *)
  let failed_a = both_arms_fail Train.Force_arm_a in
  let failed_b = both_arms_fail Train.Force_arm_b in
  p "with both arms failed, forcing arm A propagates arm A's failure"
    (String.is_substring failed_a ~substring:"tsa arm A");
  p "with both arms failed, forcing arm B propagates arm B's failure"
    (String.is_substring failed_b ~substring:"tsa arm B");
  let d_reports, d_shipped, _, d_got = run () in
  p_all2 "the default run ships and matches the plain compile" d_got expected ~f:approx;
  match d_reports with
  | [ a; b ] ->
      let measured = if Float.( <= ) a.Autotune.best_ms b.Autotune.best_ms then "A" else "B" in
      p "with no forcing, the shipped arm is the faster-measured one"
        (Option.equal String.equal d_shipped (Some measured))
  | _ -> Verdict.fail "the default run did not report exactly two arms"
