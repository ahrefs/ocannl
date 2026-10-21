(* gh-ocannl-1440: a deferred schedule-cache hit compiles its schedule once.

   [Autotune.tune_deferred] validates a cached schedule by compiling its replay -- that compile
   tells a stale entry from a sound one and supplies the report's emission census -- and then
   releases it, so a caller running several searches holds nothing resident (gh-ocannl-1408). The
   routine it ships used to be compiled a second time, from the same entry. Now the validating
   compile's code is kept (it holds no working pools) and the shipped routine is that code linked
   again ([Context.relink_outcome]).

   Counted, never timed: [Context.backend_compiles] counts completed backend compiles, each report
   carries the compiles its search completed ([compiles], of which [replay_compiles] are a cached
   schedule's), and [Train.shipped.ship_compiles] what the shipped routine's own compile took. The
   claims, through [Train.tune_placements] with scripted (uncontended, deterministic) timings: - a
   cold call ships a searched winner, recompiled once from its saved form; - a warm call replaying
   the recorded placement decision, and a warm call comparing both arms afresh (placement store
   bypassed, both arms cache hits), each ship with no compile; - for every call, the compiles
   counted over the call are its reports' plus the ship's; - every shipped routine computes the cold
   call's output bit for bit. And on [Autotune.tune_deferred] directly: a second [compile_deferred]
   returns the same routine and compiles nothing more; a relink the device refuses ships the untuned
   default in the schedule's place ([Autotune.shipped_substitute]), and that routine still computes
   the output. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module FI = Ir.Resource_fault_injection
module SO = Ir.Schedule_outcome
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

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name (Context.auto ()))

let cache_dir = "autotune_cache_deferred_hit_compiles"
let () = Scripted_search.clear_cache_dir cache_dir

(* Arm B (materialize-all) faster than arm A, so the measured winner and the recorded decision are
   B's. The searches are told apart by how many reports have been delivered when a window is
   admitted. B rather than A: on cc this fixture's arm-A crown does not replay into the caller's
   lineage once arm B has searched ("the saved segmentation no longer applies"), so arm A's winner
   ships substituted and its warm entry re-searches -- a drift of the lineage, not of this test's
   subject, which a B winner keeps out of the claims. *)
let delivered = ref 0

let measure (w : Scripted_search.window) =
  if String.equal w.label "baseline" then 1000.
  else (if !delivered = 0 then 50. else 4.) +. (Float.of_int w.ordinal /. 64.)

type call = {
  reports : Autotune.report list;
  shipped : Train.shipped;
  compiled : int;  (** [Context.backend_compiles] over the whole call. *)
  output : float array;
  ios : SC.cache_io list;  (** The call's schedule-cache and placement-store operations. *)
}

let run ?placement_store tag =
  let reports = ref [] and shipped = ref None in
  delivered := 0;
  let before = Context.backend_compiles () in
  let ((ctx, routine), ios), _record =
    Scripted_search.scripted ~measure (fun () ->
        SC.recording_cache_io (fun () ->
            Train.tune_placements ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir ?placement_store
              ~report:(fun r ->
                Int.incr delivered;
                reports := r :: !reports)
              ~on_ship:(fun s -> shipped := Some s)
              (Context.auto ()) loss comp Ir.Indexing.Empty))
  in
  let compiled = Context.backend_compiles () - before in
  let ctx = Context.run ctx routine in
  let output = Context.get_values ctx loss.Tensor.value in
  Context.release ctx;
  let shipped =
    match !shipped with Some s -> s | None -> failwith (tag ^ ": on_ship was not called")
  in
  let reports = List.rev !reports in
  Stdio.eprintf
    "%s (not part of the golden): shipped %s replayed %b, ship_compiles %d, call compiles %d, \
     reports [%s]\n\
     %!"
    tag shipped.Train.ship_arm shipped.Train.ship_replayed shipped.Train.ship_compiles compiled
    (String.concat ~sep:"; "
       (List.map reports ~f:(fun r ->
            Printf.sprintf "%s compiles %d replay %d"
              (Autotune.outcome_name r.Autotune.outcome)
              r.Autotune.compiles r.Autotune.replay_compiles)));
  { reports; shipped; compiled; output; ios }

let replay (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let searched (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let accounted tag c =
  p
    (tag ^ ": the call's compiles are its reports' plus the ship's")
    (c.compiled
    = List.sum (module Int) c.reports ~f:(fun r -> r.Autotune.compiles) + c.shipped.ship_compiles)

(* What one compile of the routine counts: 1 where the device compile happens with the code
   generation (cc, CUDA, HIP), 2 where the backend defers it to the first link (Metal). *)
let one_compile =
  let before = Context.backend_compiles () in
  let ctx, _routine = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let counted = Context.backend_compiles () - before in
  Context.release ctx;
  counted

let cold = run "cold"
let warm = run "warm"
let bypass = run ~placement_store:false "bypass"

(* Every claim about a warm hit rests on the cold call's stores and the warm lookups: where the
   context has no timing identity (no cache key) or the filesystem refused one of them, it is an
   environment skip, never a pass (review round 2). What no refusal waives: the cold call's stores
   were not all refused. *)
let identity = Option.is_some (Context.timing_identity (Context.auto ()))
let hit_claim ios label b = Cache_plant.unless_unkeyed_or_refused ~identity ios label b

let () =
  Cache_plant.not_every_store_refused ~expected:identity
    "cold: not every schedule-cache store was refused" cold.ios

let () =
  p "cold: both arms searched, and the A/B winner, arm B, shipped un-replayed"
    (List.length cold.reports = 2
    && List.for_all cold.reports ~f:searched
    && String.equal cold.shipped.ship_arm "B"
    && not cold.shipped.ship_replayed);
  p "cold: the searched winner's shipped routine is one compile, from its saved form"
    (one_compile >= 1 && cold.shipped.ship_compiles = one_compile);
  p_all "cold: no search replayed a cached schedule" cold.reports ~f:(fun r ->
      r.Autotune.replay_compiles = 0);
  accounted "cold" cold;
  let warm_ios = cold.ios @ warm.ios in
  hit_claim warm_ios "warm: the recorded decision replayed in one search, a schedule-cache hit"
    (warm.shipped.ship_replayed && match warm.reports with [ r ] -> replay r | _ -> false);
  hit_claim warm_ios "warm: the hit compiled its schedule once, to validate it"
    (match warm.reports with [ r ] -> r.Autotune.replay_compiles = one_compile | _ -> false);
  hit_claim warm_ios "warm: the shipped routine took no compile -- the validated code, linked again"
    (warm.shipped.ship_compiles = 0);
  accounted "warm" warm;
  let shipped_report = bypass.shipped.ship_report in
  let bypass_ios = cold.ios @ bypass.ios in
  hit_claim bypass_ios "bypass: both arms compared afresh, and arm B shipped, a schedule-cache hit"
    (List.length bypass.reports = 2
    && (not bypass.shipped.ship_replayed)
    && String.equal bypass.shipped.ship_arm "B"
    && Option.exists shipped_report ~f:replay);
  hit_claim bypass_ios "bypass: that hit compiled its schedule once"
    (Option.exists shipped_report ~f:(fun r -> r.Autotune.replay_compiles = one_compile));
  hit_claim bypass_ios "bypass: the shipped arm took no compile" (bypass.shipped.ship_compiles = 0);
  accounted "bypass" bypass;
  p_all2 "warm: the relinked routine computes the cold output bit for bit" warm.output cold.output
    ~f:Float.equal;
  p_all2 "bypass: the relinked routine computes the cold output bit for bit" bypass.output
    cold.output ~f:Float.equal

(* --- On [Autotune.tune_deferred] directly, warm. --- *)

(* Arm B's context: what its cached entry was searched in. *)
let materialized () =
  let embedded = ref [] in
  Tensor.iter_embedded loss ~f:(fun tn -> embedded := tn :: !embedded);
  Context.decide_materialized (Context.auto ()) !embedded

let deferred ?refuse () =
  let report = ref None in
  let (d, ios), _record =
    Scripted_search.scripted ~measure (fun () ->
        SC.recording_cache_io (fun () ->
            Autotune.tune_deferred ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
              ~report:(fun r -> report := Some r)
              (materialized ()) comp Ir.Indexing.Empty))
  in
  let refused = ref 0 in
  let compile () =
    match refuse with
    | None -> Autotune.compile_deferred d
    | Some () ->
        FI.with_callback
          (function
            | FI.Delta_pool_allocated when !refused = 0 ->
                Int.incr refused;
                raise
                  (SO.Device_memory_exhausted
                     {
                       backend = "simulated";
                       requested = 1;
                       free = Some 0;
                       detail = "gh-ocannl-1440 simulated refusal of the shipping relink";
                     })
            | _ -> ())
          ~f:(fun () -> Autotune.compile_deferred d)
  in
  (d, !report, compile, refused, cold.ios @ ios)

let run_routine (ctx, routine) =
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx loss.Tensor.value in
  Context.release ctx;
  values

let () =
  let d, report, compile, _, ios = deferred () in
  hit_claim ios "deferred: the warm search is a cache hit" (Option.exists report ~f:replay);
  p "deferred: nothing compiled for the ship before compile_deferred"
    (Autotune.deferred_compiles d = 0);
  let first = compile () in
  let second = compile () in
  hit_claim ios "deferred: the shipped routine is the validated code linked again, no compile"
    (Autotune.deferred_compiles d = 0 && not (Autotune.shipped_substitute d));
  p "deferred: a second compile_deferred returns the same routine"
    (phys_equal (fst first) (fst second) && phys_equal (snd first) (snd second));
  p_all2 "deferred: it computes the cold output bit for bit" (run_routine first) cold.output
    ~f:Float.equal;
  let d, _, compile, refused, ios = deferred ~refuse:() () in
  let shipped = compile () in
  hit_claim ios "refused relink: the device refused the relink's allocation once" (!refused = 1);
  hit_claim ios "refused relink: the untuned default shipped in the schedule's place"
    (Autotune.shipped_substitute d && Autotune.deferred_compiles d >= 1);
  p_all2 "refused relink: the untuned default computes the cold output bit for bit"
    (run_routine shipped) cold.output ~f:Float.equal

(* --- A [retain] callback that raises (review round 1): the routine linked before it ran is
   released, so the caller, who receives no context, leaks nothing. --- *)

let () =
  let live () = (Ir.Alloc_census.snapshot ()).Ir.Alloc_census.live_working_bytes in
  let before = live () in
  let raised =
    match
      Context.compile_outcome ~provenance:Ir.Schedule_outcome.User_schedule
        ~retain:(fun _ -> raise Stdlib.Exit)
        (Context.auto ()) comp Ir.Indexing.Empty
    with
    | _ -> false
    | exception Stdlib.Exit -> true
  in
  p "retain raising: its exception propagates" raised;
  p "retain raising: the routine linked before it holds no working bytes" (live () = before)

let () = Stdio.printf "deferred_hit_compiles: done\n"
