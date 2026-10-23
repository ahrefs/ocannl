(* gh-ocannl-1480: arm A's crown ships as the schedule it was timed as, whatever ran after it.

   [Train.tune_placements] compiles what ships only after every search (gh-ocannl-1408), so arm A's
   crowned schedule is replayed from its saved form after arm B has searched. On cc this fixture's
   arm-A crown is a fissioned single-kernel form whose head is the loss's whole-node zero, left
   whole although a fold is provable. Its segmentation used to be recorded as a plain [`Normal]
   segment, which a replay reads as a folded one and rebuilds with the zero expanded -- code the
   search never timed -- so the replay's digest guard refused it and the untuned default shipped in
   its place; its cache entry refused the same way and re-searched on every warm call. The single
   segment now records [`Unfolded] ([Schedule.segment_kind]), and a replay rebuilds exactly what was
   timed.

   Scripted (uncontended, deterministic) timings make arm A the winner, and arm B searches between
   A's search and A's ship in every two-arm call. The claims: the cold call ships arm A's crown
   unsubstituted ([Train.shipped.ship_substituted]) and records its decision; the warm call replays
   that decision through a schedule-cache hit; with the placement store bypassed both arms replay
   their cached schedules and A ships; an arm A forced over a faster arm B ships unsubstituted too;
   all of it again with a compatible [timing_ctx]; and on [Autotune.tune_deferred] directly, arm A's
   deferred compile after arm B's search ships no substitute, and a second [compile_deferred]
   returns the same routine. Every shipped routine computes the untuned output bit for bit (dyadic
   inputs). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module SC = Ir.Schedule_cache
module Cache_plant = Test_utils.Cache_plant

let n = 8

let nd label modulus offset =
  TDSL.ndarray
    (Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus ~offset ~stride:0.5))
    ~label:[ label ] ~output_dims:[ n; n ] ()

(* Dyadic inputs: every product and partial sum is exact, so any schedule computes the same bits. *)
let a = nd "a" 7 (-3.)
let b = nd "b" 5 (-2.)
let%op x = a *. b
let%op y = relu x
let%op loss = y ++ "ij => 0"
let comp = Train.forward loss
let backend = Context.backend_name (Context.auto ())
let () = Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend
let expected = Scripted_search.untuned ~ctx:(Context.auto ()) loss comp

(* The searches are told apart by how many reports have been delivered when a window is admitted:
   the first search is arm A's. [a_wins]: arm A far faster than arm B; otherwise the reverse, for
   the forced-arm call. *)
let delivered = ref 0

let measure ~a_wins (w : Scripted_search.window) =
  if String.equal w.label "baseline" then 1000.
  else
    let first = !delivered = 0 in
    (if Bool.equal first a_wins then 4. else 50.) +. (Float.of_int w.ordinal /. 64.)

type call = {
  reports : Autotune.report list;
  shipped : Train.shipped;
  output : float array;
  ios : SC.cache_io list;
}

let run ?placement_store ?ship_arm ?(a_wins = true) ~timing ~cache_dir tag =
  let reports = ref [] and shipped = ref None in
  delivered := 0;
  let timing_ctx = if timing then Some (Context.auto ()) else None in
  let ((ctx, routine), ios), _record =
    Scripted_search.scripted ~measure:(measure ~a_wins) (fun () ->
        SC.recording_cache_io (fun () ->
            Train.tune_placements ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir ?timing_ctx
              ?placement_store ?ship_arm
              ~report:(fun r ->
                Int.incr delivered;
                reports := r :: !reports)
              ~on_ship:(fun s -> shipped := Some s)
              (Context.auto ()) loss comp Ir.Indexing.Empty))
  in
  let ctx = Context.run ctx routine in
  let output = Context.get_values ctx loss.Tensor.value in
  Context.release ctx;
  Option.iter timing_ctx ~f:Context.release;
  let shipped =
    match !shipped with Some s -> s | None -> failwith (tag ^ ": on_ship was not called")
  in
  let reports = List.rev !reports in
  Stdio.eprintf
    "%s (not part of the golden): shipped %s replayed %b substituted %b, reports [%s]\n%!" tag
    shipped.Train.ship_arm shipped.Train.ship_replayed shipped.Train.ship_substituted
    (String.concat ~sep:"; "
       (List.map reports ~f:(fun r ->
            Printf.sprintf "%s %s" (Autotune.outcome_name r.Autotune.outcome) r.Autotune.best_label)));
  { reports; shipped; output; ios }

let replay (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let searched (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let identity = Option.is_some (Context.timing_identity (Context.auto ()))
let hit_claim ios label b = Cache_plant.unless_unkeyed_or_refused ~identity ios label b
let is_cc = String.equal backend "cc"
let cache_dir = "autotune_cache_tune_arm_a_replay"

(* The cold, warm and bypass calls over one cache directory, with or without a [timing_ctx]. *)
let scenario ~timing =
  let tag s = if timing then s ^ " (timing_ctx)" else s in
  (* One directory, emptied per scenario: each scenario's cold call must search. *)
  Scripted_search.clear_cache_dir cache_dir;
  let cold = run ~timing ~cache_dir (tag "cold") in
  let warm = run ~timing ~cache_dir (tag "warm") in
  let bypass = run ~timing ~cache_dir ~placement_store:false (tag "bypass") in
  Cache_plant.not_every_store_refused ~expected:identity
    (tag "cold: not every schedule-cache store was refused")
    cold.ios;
  p
    (tag "cold: both arms searched, arm B after arm A, and arm A's crown shipped unsubstituted")
    (List.length cold.reports = 2
    && List.for_all cold.reports ~f:searched
    && String.equal cold.shipped.ship_arm "A"
    && (not cold.shipped.ship_replayed) && not cold.shipped.ship_substituted);
  p
    (tag "cold: the shipped report is arm A's")
    (match (cold.reports, cold.shipped.ship_report) with
    | a_report :: _, Some r -> phys_equal a_report r
    | _ -> false);
  (* The shape this issue's defect needed, so the claims above are not vacuous where it applies: on
     cc arm A's crown is a fissioned form (gh-ocannl-1480's single kernel with a whole head
     zero). *)
  gated ~when_:is_cc ~on:backend
    (tag "cold: on cc, arm A's crown is a fissioned form")
    (match cold.reports with a_report :: _ -> a_report.Autotune.fissioned | [] -> false);
  let warm_ios = cold.ios @ warm.ios in
  hit_claim warm_ios
    (tag "warm: the recorded decision replayed arm A's schedule from the cache, unsubstituted")
    (warm.shipped.ship_replayed
    && (not warm.shipped.ship_substituted)
    && match warm.reports with [ r ] -> replay r | _ -> false);
  let bypass_ios = cold.ios @ bypass.ios in
  hit_claim bypass_ios
    (tag "bypass: both arms replayed their cached schedules, and arm A shipped unsubstituted")
    (List.length bypass.reports = 2
    && List.for_all bypass.reports ~f:replay
    && String.equal bypass.shipped.ship_arm "A"
    && not bypass.shipped.ship_substituted);
  p_all (tag "every shipped routine computes the untuned output bit for bit") [ cold; warm; bypass ]
    ~f:(fun c -> Array.equal Float.equal c.output expected)

let () = scenario ~timing:false
let () = scenario ~timing:true

(* An arm A forced over a measurably faster arm B: the arm the caller asked for ships, as the
   schedule it was timed as. *)
let () =
  let cache_dir = "autotune_cache_tune_arm_a_replay_forced" in
  Scripted_search.clear_cache_dir cache_dir;
  let forced = run ~timing:false ~cache_dir ~a_wins:false ~ship_arm:Train.Force_arm_a "forced" in
  p "forced: arm B measured faster, both arms searched, and arm A's crown shipped unsubstituted"
    (List.length forced.reports = 2
    && List.for_all forced.reports ~f:searched
    && (match forced.reports with
      | [ ra; rb ] -> Float.( < ) rb.Autotune.best_ms ra.Autotune.best_ms
      | _ -> false)
    && String.equal forced.shipped.ship_arm "A"
    && not forced.shipped.ship_substituted);
  p_all "forced: the shipped routine computes the untuned output bit for bit" [ forced ]
    ~f:(fun c -> Array.equal Float.equal c.output expected)

(* --- On [Autotune.tune_deferred] directly: arm A's search, arm B's search, then arm A's compile.
   --- *)

let materialized ctx =
  let embedded = ref [] in
  Tensor.iter_embedded loss ~f:(fun tn -> embedded := tn :: !embedded);
  Context.decide_materialized ctx !embedded

let () =
  let cache_dir = "autotune_cache_tune_arm_a_replay_deferred" in
  Scripted_search.clear_cache_dir cache_dir;
  let reports = ref [] in
  delivered := 0;
  let (deferred_a, _deferred_b), _record =
    Scripted_search.scripted ~measure:(measure ~a_wins:true) (fun () ->
        let tune ctx =
          Autotune.tune_deferred ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
            ~report:(fun r ->
              Int.incr delivered;
              reports := r :: !reports)
            ctx comp Ir.Indexing.Empty
        in
        let deferred_a = tune (Context.auto ()) in
        let deferred_b = tune (materialized (Context.auto ())) in
        (deferred_a, deferred_b))
  in
  let first = Autotune.compile_deferred deferred_a in
  let second = Autotune.compile_deferred deferred_a in
  p_all ~min:2 "deferred: both searches searched, arm B's after arm A's" !reports ~f:searched;
  p "deferred: arm A's compile after arm B's search ships its crown, no substitute"
    (not (Autotune.shipped_substitute deferred_a));
  p "deferred: a second compile_deferred returns the same routine"
    (phys_equal (fst first) (fst second) && phys_equal (snd first) (snd second));
  let ctx = Context.run (fst first) (snd first) in
  let output = Context.get_values ctx loss.Tensor.value in
  Context.release ctx;
  p "deferred: it computes the untuned output bit for bit" (Array.equal Float.equal output expected)

let () = Stdio.printf "tune_arm_a_replay: done\n"
