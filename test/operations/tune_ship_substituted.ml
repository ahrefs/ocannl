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
   ships substituted without attempting a placement-store write. 2. [fallback_ok]: only arm A's
   replay substitutes -> [tune_placements] falls back to arm B, whose replay succeeds -> arm B ships
   unsubstituted. 3. [forced_refused]: [Force_arm_a] has no fallback arm -> arm A ships substituted
   immediately. 4. [control]: no refusal -> arm A ships unsubstituted. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module FI = Ir.Resource_fault_injection
module SC = Ir.Schedule_cache
module SO = Ir.Schedule_outcome
module H = Bench_harness
module Cache_plant = Test_utils.Cache_plant

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
let identity = Option.is_some (Context.timing_identity (Context.cpu ()))

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
  ios : SC.cache_io list;
  tune_json : Yojson.Safe.t;
}

let run_scenario ~mode ~cache_dir ~placement_store ~ship_arm ~out_file tag =
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
  let (output, ios), stderr_lines =
    Stderr_capture.lines (fun () ->
        (* Pin search enabled regardless of caller profile (e.g. OCANNL_PROFILE=reproducible). *)
        Utils.with_stated_config
          [ ("autotune_search", "true") ]
          (fun () ->
            let (output, ios), _record =
              Scripted_search.scripted ~measure (fun () ->
                  SC.recording_cache_io (fun () ->
                      FI.with_callback callback ~f:(fun () ->
                          let ctx, routine =
                            Train.tune_placements ~name:tag ~beam_width:1 ~rounds:0 ~repeats:1
                              ~cache_dir ~placement_store ~ship_arm
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
                          values)))
            in
            (output, ios)))
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
  Stdio.Out_channel.write_all out_file ~data:("bench: done\n" ^ line ^ "\n");
  let parsed = Yojson.Safe.from_string line in
  let tune_json = member "tune" parsed in
  Stdio.eprintf "%s (not part of the golden): shipped %s substituted=%b refused=%d reports=%d\n%!"
    tag shipped.Train.ship_arm shipped.Train.ship_substituted !refused (List.length reports);
  { reports; shipped; output; refused = !refused; stderr_lines; ios; tune_json }

let is_placement_io op (io : SC.cache_io) =
  Poly.equal io.SC.op op && String.is_prefix io.SC.key ~prefix:"placements-"

let ship_report_is shipped expected_report =
  match shipped.Train.ship_report with Some r -> phys_equal r expected_report | None -> false

let () =
  (* --- 1. Both placement arms' winner replays fail: arm B ships substituted as last fallback.
     --- *)
  let cache_dir = "autotune_cache_tune_ship_substituted" in
  Scripted_search.clear_cache_dir cache_dir;
  let both =
    run_scenario ~mode:Refuse_all ~cache_dir ~placement_store:true ~ship_arm:Train.Measured_winner
      ~out_file:"tss_both_refused.out" "tss_both_refused"
  in
  let b_report = List.nth_exn both.reports 1 in
  let b_label = b_report.Autotune.best_label in
  p "both refused: winner replay was refused for both arm A and fallback arm B" (both.refused = 2);
  p "both refused: arm B shipped substituted with arm B's own report and non-empty crown label"
    (List.length both.reports = 2
    && String.equal both.shipped.ship_arm "B"
    && both.shipped.ship_substituted
    && (not (String.is_empty b_label))
    && ship_report_is both.shipped b_report);
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
  (* Verify directly on the cache-I/O record: placement lookup was active on this call, and no
     placement Store was attempted when the substituted fallback shipped. *)
  Cache_plant.unless_unkeyed ~identity "both refused: placement lookup was active on the call"
    (List.exists both.ios ~f:(is_placement_io SC.Lookup));
  p_empty "both refused: the substituted ship attempted no placement-store write" ~over:both.ios
    (List.filter both.ios ~f:(is_placement_io SC.Store));

  (* --- 2. Only arm A's winner replay fails: fallback arm B replays cleanly and ships measured.
     --- *)
  let fallback =
    run_scenario ~mode:Refuse_first ~cache_dir:"" ~placement_store:false
      ~ship_arm:Train.Measured_winner ~out_file:"tss_fallback_ok.out" "tss_fallback_ok"
  in
  let fb_report = List.nth_exn fallback.reports 1 in
  p "fallback ok: only arm A's winner replay was refused" (fallback.refused = 1);
  p
    "fallback ok: arm B shipped unsubstituted with arm B's own report after arm A's substitute was \
     rejected"
    (List.length fallback.reports = 2
    && String.equal fallback.shipped.ship_arm "B"
    && (not fallback.shipped.ship_substituted)
    && ship_report_is fallback.shipped fb_report);
  p "fallback ok: result line records ship_substituted=false and null substituted_from"
    (Yojson.Safe.equal (member "shipped" fallback.tune_json) (`String "B")
    && Yojson.Safe.equal (member "ship_substituted" fallback.tune_json) (`Bool false)
    && Yojson.Safe.equal (member "substituted_from" fallback.tune_json) `Null);
  p_all2 "fallback ok: arm B's shipped routine computes the reference bit for bit" fallback.output
    expected ~f:Float.equal;

  (* --- 3. Forced arm A refuses replay: no fallback arm, so arm A ships substituted. --- *)
  let forced =
    run_scenario ~mode:Refuse_all ~cache_dir:"" ~placement_store:false ~ship_arm:Train.Force_arm_a
      ~out_file:"tss_forced_refused.out" "tss_forced_refused"
  in
  let forced_a_report = List.nth_exn forced.reports 0 in
  let a_label = forced_a_report.Autotune.best_label in
  p "forced refused: arm A's single winner replay was refused" (forced.refused = 1);
  p "forced refused: forced arm A shipped substituted with arm A's own report and crown label"
    (List.length forced.reports = 2
    && String.equal forced.shipped.ship_arm "A"
    && forced.shipped.ship_substituted
    && (not (String.is_empty a_label))
    && ship_report_is forced.shipped forced_a_report);
  p "forced refused: result line records ship_substituted=true and arm A's discarded crown"
    (Yojson.Safe.equal (member "shipped" forced.tune_json) (`String "A")
    && Yojson.Safe.equal (member "ship_substituted" forced.tune_json) (`Bool true)
    && Yojson.Safe.equal (member "substituted_from" forced.tune_json) (`String a_label));
  p_all2 "forced refused: the substituted routine computes the reference bit for bit" forced.output
    expected ~f:Float.equal;

  (* --- 4. Control: no refusal; arm A ships measured and unsubstituted. --- *)
  let control =
    run_scenario ~mode:Refuse_none ~cache_dir:"" ~placement_store:false
      ~ship_arm:Train.Measured_winner ~out_file:"tss_control.out" "tss_control"
  in
  let control_a_report = List.nth_exn control.reports 0 in
  p "control: nothing was refused and arm A shipped unsubstituted with arm A's own report"
    (control.refused = 0
    && String.equal control.shipped.ship_arm "A"
    && (not control.shipped.ship_substituted)
    && ship_report_is control.shipped control_a_report);
  p "control: result line records ship_substituted=false and null substituted_from"
    (Yojson.Safe.equal (member "shipped" control.tune_json) (`String "A")
    && Yojson.Safe.equal (member "ship_substituted" control.tune_json) (`Bool false)
    && Yojson.Safe.equal (member "substituted_from" control.tune_json) `Null);
  p_all2 "control: arm A's shipped routine computes the reference bit for bit" control.output
    expected ~f:Float.equal
