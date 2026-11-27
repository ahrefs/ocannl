(* gh-ocannl-1613: end-to-end [Train.tune_placements] winner-replay substitution through the
   benchmark harness and JSON result line.

   [test/operations/bench_ship_substituted] drives [Bench_harness.collect_ship] on fabricated
   records; [test/operations/autotune_replay_fallback] exercises [Autotune.tune] and
   [Autotune.tune_deferred] directly. Here
   [Ir.Resource_fault_injection.Autotune_before_winner_replay] forces real [Train.tune_placements]
   shipping compiles to fail their first working-pool allocation ([Delta_pool_allocated]) with
   [Device_memory_exhausted], wired through the same [Bench_harness] callbacks and
   [Bench_json.result_line] emitter the benchmark runners use, and writes the resulting lines for
   [tune_ship_substituted.py] to feed to [orchestrate.py], [gh728_protocol.py] and
   [gh834_session.py].

   Four scenarios on scripted timings (arm A wins over arm B, dyadic inputs so every schedule
   computes identical bits): 1. [both_refused]: arm A's replay substitutes -> [tune_placements]
   falls back to arm B -> arm B's replay also substitutes -> nothing left to fall back to -> arm B
   ships substituted without recording a placement decision. 2. [fallback_ok]: only arm A's replay
   substitutes -> [tune_placements] falls back to arm B, whose replay succeeds -> arm B ships
   unsubstituted. 3. [forced_refused]: [Force_arm_a] has no fallback arm -> arm A ships substituted
   immediately. 4. [control]: no refusal -> arm A ships unsubstituted. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module FI = Ir.Resource_fault_injection
module SO = Ir.Schedule_outcome
module H = Bench_harness

let member = Yojson.Safe.Util.member
let n = 8

let cells modulus offset =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus ~offset ~stride:0.5)

let av = cells 7 (-3.)
let bv = cells 5 (-2.)
let nd label values = TDSL.ndarray values ~label:[ label ] ~output_dims:[ n; n ] ()
let a = nd "tss_a" av
let b = nd "tss_b" bv
let%op mid = a + b
let%op out = mid ++ "ij => ji"
let comp = Train.forward out
let expected = Scripted_search.untuned ~ctx:(Context.cpu ()) out comp

(* Arm A (first search, [!delivered = 0]) beats arm B deterministically; baseline trails both so
   each arm crowns a non-baseline candidate when one exists. *)
let delivered = ref 0

let measure (w : Scripted_search.window) =
  if String.equal w.label "baseline" then 1000.
  else
    let base = if !delivered = 0 then 4. else 50. in
    base +. (Float.of_int w.ordinal /. 64.)

type refusal_mode = Refuse_none | Refuse_first | Refuse_all

type scenario_result = {
  reports : Autotune.report list;
  shipped : Train.shipped;
  output : float array;
  refused : int;
  stderr_lines : string list;
  tune_json : Yojson.Safe.t;
}

let run_scenario ~mode ~cache_dir ?ship_arm ?out_file tag =
  let arms = H.tune_arms () in
  let reports = ref [] and shipped = ref None in
  delivered := 0;
  let replay_armed = ref false in
  let refused = ref 0 in
  let callback (point : FI.point) =
    match point with
    | FI.Autotune_before_winner_replay -> (
        match mode with
        | Refuse_none -> ()
        | Refuse_first -> if !refused = 0 then replay_armed := true
        | Refuse_all -> replay_armed := true)
    | FI.Delta_pool_allocated when !replay_armed ->
        replay_armed := false;
        Int.incr refused;
        raise
          (SO.Device_memory_exhausted
             {
               backend = "simulated";
               requested = 1;
               free = Some 0;
               detail = "gh-ocannl-1613 simulated refusal of winner replay";
             })
    | _ -> ()
  in
  let output, stderr_lines =
    Stderr_capture.lines (fun () ->
        let output, _record =
          Scripted_search.scripted ~measure (fun () ->
              FI.with_callback callback ~f:(fun () ->
                  let ctx, routine =
                    Train.tune_placements ~name:tag ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
                      ?ship_arm
                      ~report:(fun r ->
                        Int.incr delivered;
                        H.collect_arm arms r;
                        reports := r :: !reports)
                      ~flip_report:(H.collect_search arms)
                      ~on_ship:(fun s ->
                        H.collect_ship arms s;
                        shipped := Some s)
                      (Context.cpu ()) out comp Ir.Indexing.Empty
                  in
                  H.collect_shipped arms (H.Plain routine);
                  let ctx = Context.run ctx routine in
                  let values = Context.get_values ctx out.Tensor.value in
                  Context.release ctx;
                  values))
        in
        output)
  in
  let shipped =
    match !shipped with Some s -> s | None -> failwith (tag ^ ": on_ship was not called")
  in
  let reports = List.rev !reports in
  let tune_str =
    match H.tune_json arms with
    | Some s -> s
    | None -> failwith (tag ^ ": Bench_harness.tune_json returned None")
  in
  let line =
    Bench_json.result_line ~backend:"cc" ~variant:"tuned" ~precision:"f32" ~storage_precision:"f32"
      ~profile:None ~regime_knobs:[] ~ambient_resolved:[] ~workload:"tune_ship_substituted"
      ~compile_s:1.0 ~searched:(H.searched arms) ~tune:tune_str ~p10:1.0 ~p50:1.0 ~p90:1.0
      ~queued_ms:1.0 ~timed_steps:2 ~peak_memory:None ~losses:[| 2.5; 2.0 |] ()
  in
  Option.iter out_file ~f:(fun path ->
      Stdio.Out_channel.write_all path ~data:("bench: done\n" ^ line ^ "\n"));
  let parsed = Yojson.Safe.from_string line in
  let tune_json = member "tune" parsed in
  Stdio.eprintf "%s (not part of the golden): shipped %s substituted=%b refused=%d reports=%d\n%!"
    tag shipped.Train.ship_arm shipped.Train.ship_substituted !refused (List.length reports);
  { reports; shipped; output; refused = !refused; stderr_lines; tune_json }

let arm_label reports idx = (List.nth_exn reports idx).Autotune.best_label

let () =
  (* --- 1. Both placement arms' winner replays fail: arm B ships substituted as last fallback.
     --- *)
  let cache_dir = "autotune_cache_tune_ship_substituted" in
  Scripted_search.clear_cache_dir cache_dir;
  let both =
    run_scenario ~mode:Refuse_all ~cache_dir ~out_file:"tss_both_refused.out" "tss_both_refused"
  in
  let b_label = arm_label both.reports 1 in
  p "both refused: winner replay was refused for both arm A and fallback arm B" (both.refused = 2);
  p "both refused: arm B shipped substituted with its report and non-empty crown label"
    (List.length both.reports = 2
    && String.equal both.shipped.ship_arm "B"
    && both.shipped.ship_substituted
    && (not (String.is_empty b_label))
    &&
    match both.shipped.ship_report with
    | Some r -> String.equal r.best_label b_label
    | None -> false);
  p "both refused: result line records ship_substituted=true and arm B's discarded crown"
    (Yojson.Safe.equal (member "shipped" both.tune_json) (`String "B")
    && Yojson.Safe.equal (member "ship_substituted" both.tune_json) (`Bool true)
    && Yojson.Safe.equal (member "substituted_from" both.tune_json) (`String b_label));
  p_exists "both refused: stderr warns that the searched winner did not replay" both.stderr_lines
    ~f:(fun l ->
      String.is_substring l ~substring:"did not replay"
      && String.is_substring l ~substring:"shipping the untuned default in its place");
  p_all2 "both refused: the substituted untuned routine computes the reference bit for bit"
    both.output expected ~f:Float.equal;
  (* A follow-up call over the same cache directory must not see a recorded placement decision: a
     substituted ship is never committed to the placement store. *)
  let after_sub = run_scenario ~mode:Refuse_none ~cache_dir "tss_after_sub" in
  p "both refused: the substituted ship did not record a placement-store decision"
    ((not after_sub.shipped.ship_replayed)
    && (not after_sub.shipped.ship_substituted)
    && String.equal after_sub.shipped.ship_arm "A");

  (* --- 2. Only arm A's winner replay fails: fallback arm B replays cleanly and ships measured.
     --- *)
  let fallback =
    run_scenario ~mode:Refuse_first ~cache_dir:"" ~out_file:"tss_fallback_ok.out" "tss_fallback_ok"
  in
  p "fallback ok: only arm A's winner replay was refused" (fallback.refused = 1);
  p "fallback ok: arm B shipped unsubstituted after arm A's substitute was rejected"
    (List.length fallback.reports = 2
    && String.equal fallback.shipped.ship_arm "B"
    && not fallback.shipped.ship_substituted);
  p "fallback ok: result line records ship_substituted=false and null substituted_from"
    (Yojson.Safe.equal (member "shipped" fallback.tune_json) (`String "B")
    && Yojson.Safe.equal (member "ship_substituted" fallback.tune_json) (`Bool false)
    && Yojson.Safe.equal (member "substituted_from" fallback.tune_json) `Null);
  p_all2 "fallback ok: arm B's shipped routine computes the reference bit for bit" fallback.output
    expected ~f:Float.equal;

  (* --- 3. Forced arm A refuses replay: no fallback arm, so arm A ships substituted. --- *)
  let forced =
    run_scenario ~mode:Refuse_all ~cache_dir:"" ~ship_arm:Train.Force_arm_a
      ~out_file:"tss_forced_refused.out" "tss_forced_refused"
  in
  let a_label = arm_label forced.reports 0 in
  p "forced refused: arm A's single winner replay was refused" (forced.refused = 1);
  p "forced refused: forced arm A shipped substituted with its crown label"
    (List.length forced.reports = 2
    && String.equal forced.shipped.ship_arm "A"
    && forced.shipped.ship_substituted
    && not (String.is_empty a_label));
  p "forced refused: result line records ship_substituted=true and arm A's discarded crown"
    (Yojson.Safe.equal (member "shipped" forced.tune_json) (`String "A")
    && Yojson.Safe.equal (member "ship_substituted" forced.tune_json) (`Bool true)
    && Yojson.Safe.equal (member "substituted_from" forced.tune_json) (`String a_label));
  p_all2 "forced refused: the substituted routine computes the reference bit for bit" forced.output
    expected ~f:Float.equal;

  (* --- 4. Control: no refusal; arm A ships measured and unsubstituted. --- *)
  let control =
    run_scenario ~mode:Refuse_none ~cache_dir:"" ~out_file:"tss_control.out" "tss_control"
  in
  p "control: nothing was refused and arm A shipped unsubstituted"
    (control.refused = 0
    && String.equal control.shipped.ship_arm "A"
    && not control.shipped.ship_substituted);
  p "control: result line records ship_substituted=false and null substituted_from"
    (Yojson.Safe.equal (member "shipped" control.tune_json) (`String "A")
    && Yojson.Safe.equal (member "ship_substituted" control.tune_json) (`Bool false)
    && Yojson.Safe.equal (member "substituted_from" control.tune_json) `Null);
  p_all2 "control: arm A's shipped routine computes the reference bit for bit" control.output
    expected ~f:Float.equal
