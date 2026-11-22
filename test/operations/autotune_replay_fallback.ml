(* gh-ocannl-2416: a direct [Autotune.tune] search whose winner replay fails and substitutes the
   untuned default reports [Searched_replay_fallback] with its crown fields cleared, rather than
   handing back the discarded timed crown's report under [Searched].

   After the search crowns a winner and commits its cache entry, [Autotune.tune] replays the saved
   winner into the caller's context before delivering its report. Here [Resource_fault_injection]
   refuses the first working-pool allocation after [Schedule_cache_before_commit] (the winner
   replay's link) with [Device_memory_exhausted], forcing the classified-failure fallback that sets
   [substituted] and ships the untuned default. The search's census counters stay intact while the
   crown fields are cleared so no caller credits the untuned default with the discarded winner.

   Controls: (1) the same search without the replay refusal reports [Searched] with a finite
   [best_ms] and non-empty [best_label]; (2) [Autotune.tune_deferred] with the same refusal during
   [compile_deferred] still delivers [Searched] before the deferred compile runs and reports the
   substitution via [Autotune.shipped_substitute]. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module FI = Ir.Resource_fault_injection
module SO = Ir.Schedule_outcome

let chain = Fissioned_chain.chain_with_output ()
let reference = Scripted_search.untuned (fst chain) (snd chain)
let cache_dir = "autotune_cache_replay_fallback"

(* Crown a fissioned candidate rather than [baseline] so the control carries non-trivial
   [fissioned], [best_schedule], [best_grid], [best_spread] and [best_steps], making the
   crown-clearing assertions under [Searched_replay_fallback] discriminating. *)
let measure (w : Scripted_search.window) =
  let base = if String.is_prefix w.Scripted_search.label ~prefix:"F_" then 1. else 50. in
  base +. (Float.of_int w.Scripted_search.ordinal /. 64.)

let run_direct ~fail_replay ~tag =
  Scripted_search.clear_cache_dir cache_dir;
  let qg, comp = chain in
  let report = ref None in
  let cache_committed = ref false in
  let refused = ref 0 in
  let callback (point : FI.point) =
    match point with
    | FI.Schedule_cache_before_commit -> cache_committed := true
    | FI.Delta_pool_allocated when fail_replay && !cache_committed && !refused = 0 ->
        Int.incr refused;
        raise
          (SO.Device_memory_exhausted
             {
               backend = "simulated";
               requested = 1;
               free = Some 0;
               detail = "gh-ocannl-2416 simulated refusal of winner replay";
             })
    | _ -> ()
  in
  let (values, attempts), lines =
    Stderr_capture.lines (fun () ->
        let values, record =
          Scripted_search.scripted ~measure (fun () ->
              FI.with_callback callback ~f:(fun () ->
                  let ctx, routine =
                    Autotune.tune ~name:tag ~search:true ~beam_width:1 ~rounds:0 ~repeats:1
                      ~cache_dir ~log:false ~progress:false
                      ~report:(fun r -> report := Some r)
                      (Context.cpu ()) comp Ir.Indexing.Empty
                  in
                  let ctx = Context.run ctx routine in
                  let values = Context.get_values ctx qg.Tensor.value in
                  Context.release ctx;
                  values))
        in
        (values, record.Scripted_search.attempts))
  in
  let r = Option.value_exn ~here:[%here] !report in
  Stdio.eprintf "%s (not part of the golden): %s, refused %d, timed %d, crown %S\n%!" tag
    (Autotune.outcome_name r.Autotune.outcome)
    !refused r.Autotune.candidates_timed r.Autotune.best_label;
  (r, values, attempts, !refused, lines)

let () =
  (* --- 1. Direct Autotune.tune with forced winner-replay refusal. --- *)
  let r, values, attempts, refused, lines = run_direct ~fail_replay:true ~tag:"replay_fallback" in
  p "forced replay refusal: the simulated refusal fired once on winner replay" (refused = 1);
  p "forced replay refusal: the search timed at least one candidate before replay"
    (r.Autotune.candidates_timed >= 1);
  p "forced replay refusal: outcome is Searched_replay_fallback"
    (match r.Autotune.outcome with Autotune.Searched_replay_fallback -> true | _ -> false);
  p "forced replay refusal: outcome_name is searched-replay-fallback"
    (String.equal (Autotune.outcome_name r.Autotune.outcome) "searched-replay-fallback");
  p "forced replay refusal: crown_fields spells searched-replay-fallback with no crown"
    (String.is_prefix (Autotune.crown_fields r)
       ~prefix:"outcome=searched-replay-fallback crowned=\"\" ");
  p "forced replay refusal: the report credits no timed crown"
    (Float.is_inf r.Autotune.best_ms && String.is_empty r.Autotune.best_label);
  p "forced replay refusal: the report credits no spread, tensorization, grid or fissioned flag"
    (Option.is_none r.Autotune.best_spread
    && (not r.Autotune.best_tensorized)
    && Option.is_none r.Autotune.best_tensorization
    && r.Autotune.best_mma_statements = 0
    && r.Autotune.best_mma_scalar_fallbacks = 0
    && Option.is_none r.Autotune.best_grid
    && not r.Autotune.fissioned);
  p_empty "forced replay refusal: the report carries no saved schedule" ~over:attempts
    r.Autotune.best_schedule;
  p_empty "forced replay refusal: the report keeps no best-so-far record of the discarded crown"
    ~over:attempts r.Autotune.best_steps;
  p_exists "forced replay refusal: stderr warns that the winner did not replay" lines ~f:(fun l ->
      String.is_substring l ~substring:"did not replay"
      && String.is_substring l ~substring:"shipping the untuned default in its place");
  p_all2 "forced replay refusal: the shipped untuned default computes the reference bit for bit"
    values reference ~f:Float.equal;

  (* --- 2. Control: direct Autotune.tune without winner-replay refusal. --- *)
  let r_ok, values_ok, _, refused_ok, _ = run_direct ~fail_replay:false ~tag:"replay_ok" in
  p "control: nothing was refused" (refused_ok = 0);
  p "control: outcome stays Searched"
    (match r_ok.Autotune.outcome with Autotune.Searched -> true | _ -> false);
  p "control: the report carries the timed crown and its metadata"
    (Float.is_finite r_ok.Autotune.best_ms
    && (not (String.is_empty r_ok.Autotune.best_label))
    && r_ok.Autotune.fissioned
    && (not (List.is_empty r_ok.Autotune.best_schedule))
    && Option.is_some r_ok.Autotune.best_grid
    && Option.is_some r_ok.Autotune.best_spread
    && not (List.is_empty r_ok.Autotune.best_steps));
  p_all2 "control: the shipped routine computes the reference bit for bit" values_ok reference
    ~f:Float.equal;

  (* --- 3. Deferred control: tune_deferred reports before compile_deferred and exposes substitution
     via shipped_substitute. --- *)
  Scripted_search.clear_cache_dir cache_dir;
  let qg, comp = chain in
  let deferred_report = ref None in
  let deferred, _record =
    Scripted_search.scripted ~measure (fun () ->
        Autotune.tune_deferred ~name:"replay_deferred" ~search:true ~beam_width:1 ~rounds:0
          ~repeats:1 ~cache_dir ~log:false ~progress:false
          ~report:(fun r -> deferred_report := Some r)
          (Context.cpu ()) comp Ir.Indexing.Empty)
  in
  let r_def = Option.value_exn ~here:[%here] !deferred_report in
  p "deferred: before compile_deferred, outcome is Searched and shipped_substitute is false"
    ((match r_def.Autotune.outcome with Autotune.Searched -> true | _ -> false)
    && Float.is_finite r_def.Autotune.best_ms
    && (not (String.is_empty r_def.Autotune.best_label))
    && not (Autotune.shipped_substitute deferred));
  let def_refused = ref 0 in
  let def_ctx, def_routine =
    FI.with_callback
      (function
        | FI.Delta_pool_allocated when !def_refused = 0 ->
            Int.incr def_refused;
            raise
              (SO.Device_memory_exhausted
                 {
                   backend = "simulated";
                   requested = 1;
                   free = Some 0;
                   detail = "gh-ocannl-2416 simulated refusal of deferred winner replay";
                 })
        | _ -> ())
      ~f:(fun () -> Autotune.compile_deferred deferred)
  in
  let def_ctx = Context.run def_ctx def_routine in
  let def_values = Context.get_values def_ctx qg.Tensor.value in
  Context.release def_ctx;
  p "deferred: compile_deferred hit the refusal and sets shipped_substitute"
    (!def_refused = 1 && Autotune.shipped_substitute deferred);
  p_all2 "deferred: the substituted routine computes the reference bit for bit" def_values reference
    ~f:Float.equal
