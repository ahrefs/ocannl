(* gh-ocannl-1199: the [BENCH_TIMING_TRACE] decision line says what the queued calibration decided,
   read from the policy's own metadata ([Autotune.on_batch_decision]) rather than re-derived from
   probe readings. Two synthetic CUDA/HIP devices on an injected clock, sharing the singles (9.75
   ms, which owe a batch) and the depth-2/depth-3 pair geometry of the gh-ocannl-1184 boundary:

   - [retained]: 1.25 ms fixed plus 9 ms a launch. The fit puts one launch at 10.25 ms, just above
   the 10 ms target, but its marginal work fits the target and its fixed term is below it, so the
   floor keeps the measured depth-2 batch, which is timed and admitted; - [refused]: no fixed term
   and 10.5 ms a launch, marginal work over the target. The same shallower-crossing branch settles
   at depth 1 without the floor, the isolated objective, and the call is refused as unbatched.

   Dyadic costs keep every printed float exact, so the lines are the golden. The claims tie each
   line to the decision the call actually took: its reading's admission and its timed depth. *)

open Base
open Verdict.Claims

let call ~fixed ~marginal =
  let decisions = ref [] and depth = ref None in
  let old_decision = !Autotune.on_batch_decision and old_depth = !Autotune.on_batch_depth in
  Exn.protect
    ~finally:(fun () ->
      Autotune.on_batch_decision := old_decision;
      Autotune.on_batch_depth := old_depth)
    ~f:(fun () ->
      (Autotune.on_batch_decision := fun d -> decisions := d :: !decisions);
      (Autotune.on_batch_depth := fun d ~calibration_samples:_ -> depth := Some d);
      let reading =
        Autotune.calibrate_and_time ~retry_contended:false ~timing:Autotune.Queued ~repeats:3
          ~queue_depth_cap:(Autotune.queue_depth_cap_for_backend "hip") ~batch:(fun d ->
            if d = 1 then 9.75 else fixed +. (marginal *. Float.of_int d))
      in
      (reading, !depth, !decisions))

let () =
  Test_utils.set_binary_stdout ();
  let traced =
    List.map
      [ ("retained", 1.25, 9.); ("refused", 0., 10.5) ]
      ~f:(fun (what, fixed, marginal) ->
        let reading, depth, decisions = call ~fixed ~marginal in
        let lines =
          List.mapi (List.rev decisions) ~f:(fun i d -> Bench_harness.decision_line ~call:(i + 1) d)
        in
        List.iter lines ~f:(fun line -> Stdio.printf "%s: %s\n" what line);
        (what, reading, depth, lines))
  in
  let says sub line = String.is_substring line ~substring:sub in
  p_all "each call prints exactly one decision line" traced ~f:(fun (_, _, _, lines) ->
      List.length lines = 1);
  p_all "the line's depth is the depth the call timed" traced ~f:(fun (_, _, depth, lines) ->
      match (depth, lines) with
      | Some depth, [ line ] -> says (Printf.sprintf ", depth %d by " depth) line
      | _ -> false);
  p_all "the line says admitted exactly when the call's reading is admitted" traced
    ~f:(fun (_, reading, _, lines) ->
      match lines with
      | [ line ] ->
          Bool.equal (says ", admitted" line) (Option.is_some (Autotune.admitted_timing_ms reading))
      | _ -> false);
  p_all "the line says refused unbatched exactly when the call refused as unbatched" traced
    ~f:(fun (_, reading, _, lines) ->
      match lines with
      | [ line ] -> Bool.equal (says ", refused unbatched" line) reading.Autotune.unbatched
      | _ -> false);
  match traced with
  | [ (_, retained, _, [ kept ]); (_, refused, _, [ refusal ]) ] ->
      p "the retained boundary is admitted and its line names the floor"
        (Option.is_some (Autotune.admitted_timing_ms retained) && says "fit boundary_floor" kept);
      p "the opposing refusal is unbatched and its line names the same crossing without the floor"
        (refused.Autotune.unbatched && says "fit shallower_crossing" refusal)
  | _ ->
      p "the retained boundary is admitted and its line names the floor" false;
      p "the opposing refusal is unbatched and its line names the same crossing without the floor"
        false
