(* gh-ocannl-1326: a schedule search must be able to tune a step that uses over half the device.

   The reproduction that motivated this needs a 12 GiB GPU and a b256 gpt2_mini training step, whose
   working set is 6.6 GiB: a tuned search timed its first candidate and then declined every later
   one at link with [CUDA_ERROR_OUT_OF_MEMORY], because the incumbent's buffers stayed resident
   while each challenger allocated its own; every warm process re-searched, because the cache replay
   was refused for the same reason while the base compile's buffers were resident; and the
   materialize-all arm, next to the first arm's shipped routine, declined all of its candidates and
   died at its fallback compile. Here the device is simulated instead: the allocator seam's
   fault-injection callback refuses an allocation, with the typed
   [Ir.Schedule_outcome.Device_memory_exhausted] the CUDA and HIP pool allocators raise, whenever
   the live working bytes would exceed a capacity sized from the routine's own measured footprint.
   Sizes are this run's own, so the scenarios mean the same thing on every backend.

   Asserted: under room for one and a half footprints, a cold search declines nothing for memory
   (its incumbents are not resident beside a challenger) and a warm search replays its winner (the
   replay is not co-resident with the base compile); under room for half a footprint, the search is
   refused before any candidate is compiled, with a typed [Pre_search_failure]; and, through
   [Train.tune_placements], an arm that does not fit beside the other arm's shipped routine is
   refused the same way while that arm ships, is cached, and replays in the next call without a
   single search. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module FI = Ir.Resource_fault_injection
module AC = Ir.Alloc_census
module SO = Ir.Schedule_outcome
open Verdict.Claims

let n = 32

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let live_working () = (AC.snapshot ()).AC.live_working_bytes

(* The bytes of working pools one compile of [comp] from a fresh context of [decide] allocates,
   released again before returning. [unscheduled] compiles the serial form the search's base compile
   links: on a GPU backend the default compile needs more, since fission promotes the locals that
   cross its segments to device buffers. *)
let footprint ?(decide = Fn.id) ?(unscheduled = false) comp =
  let before = live_working () in
  let lowered_transform = if unscheduled then Some (fun opt -> [ opt ]) else None in
  let ctx, _routine =
    Context.compile ?lowered_transform (decide (Context.auto ())) comp Ir.Indexing.Empty
  in
  let bytes = live_working () - before in
  Context.release ctx;
  bytes

(* A device with room for [capacity] working bytes: every allocation at the shared seams that would
   leave more live is refused, after the census recorded it, so the seam's own unwind frees it. The
   count of refusals is returned beside [f]'s outcome. *)
let with_capacity ~capacity f =
  let refused = ref 0 in
  let refuse point =
    match point with
    | FI.Delta_pool_allocated | FI.Transfer_pool_allocated ->
        let live = live_working () in
        if live > capacity then (
          Int.incr refused;
          raise
            (SO.Device_memory_exhausted
               {
                 backend = "simulated";
                 requested = live - capacity;
                 free = Some 0;
                 detail = "gh-ocannl-1326 simulated capacity";
               }))
    | _ -> ()
  in
  let result = Result.try_with (fun () -> FI.with_callback refuse ~f) in
  (result, !refused)

let outcome_is f (r : Autotune.report) = f r.Autotune.outcome
let searched = outcome_is (function Autotune.Searched -> true | _ -> false)
let replayed = outcome_is (function Autotune.Cache_replay -> true | _ -> false)

let refused_for_memory (r : Autotune.report) =
  (match r.Autotune.outcome with
    | Autotune.Pre_search_failure { phase; _ } -> SO.equal_phase phase SO.Backend_link
    | _ -> false)
  && List.exists r.Autotune.declines ~f:(fun d ->
      SO.equal_rejection_key d.Autotune.key (SO.Resource_exceeded_key SO.Device_memory))

let memory_declines (r : Autotune.report) =
  List.sum
    (module Int)
    r.Autotune.declines
    ~f:(fun d ->
      if SO.equal_rejection_key d.Autotune.key (SO.Resource_exceeded_key SO.Device_memory) then
        d.Autotune.count
      else 0)

let approx a b = Float.(abs (a - b) < 1e-4)

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "dm_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "dm_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = Train.forward t2 in
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in
  let check_values what (ctx, routine) =
    let ctx = Context.run ctx routine in
    p_all2 what (Context.get_values ctx t2.Tensor.value) expected ~f:approx
  in
  let tune ~cache_dir ~report =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ~report (Context.auto ()) comp
      Ir.Indexing.Empty
  in
  let materialize_all c = Context.decide_materialized c (Train.embedded_nodes t2) in
  let fp = footprint comp in
  let fp_base = footprint ~unscheduled:true comp in
  let fp_b_base = footprint ~decide:materialize_all ~unscheduled:true comp in
  Stdio.eprintf
    "footprints (not part of the golden): default %d B, unscheduled %d B, materialize-all \
     unscheduled %d B\n\
     %!"
    fp fp_base fp_b_base;
  p "a compile of the routine allocates working memory" (fp_base > 0 && fp >= fp_base);
  (* The capacities below are read against the bytes live when each scenario starts, so whatever the
     process already holds (the reference routine above) is outside them. *)
  let room ?(of_ = fp) factor = live_working () + Float.to_int (factor *. Float.of_int of_) in

  (* --- 1. Room for one and a half footprints: the incumbent must not be resident while a
     challenger links. --- *)
  let cache_dir = "autotune_cache_device_memory" in
  clean_cache cache_dir;
  let report1 = ref None in
  let result, refused =
    with_capacity ~capacity:(room 1.5) (fun () ->
        tune ~cache_dir ~report:(fun r -> report1 := Some r))
  in
  let r1 = Option.value_exn !report1 in
  Stdio.eprintf "cold search (not part of the golden): timed %d failed %d refused %d\n%!"
    r1.Autotune.candidates_timed r1.Autotune.candidates_failed refused;
  p "the cold search completed" (Result.is_ok result && searched r1);
  p "it timed more than one candidate" (r1.Autotune.candidates_timed >= 2);
  p "no candidate was declined for device memory" (memory_declines r1 = 0 && refused = 0);
  Result.iter result ~f:(check_values "the cold search's winner computes the right values");

  (* --- 2. The same room, warm: the cache replay must not find the base compile resident. A
     contention-refused window keeps a search cache-cold (gh-ocannl-855), which waives this
     scenario's replay claim and nothing else. --- *)
  let cacheable = r1.Autotune.timings_contended = 0 in
  let report2 = ref None in
  let result, refused =
    with_capacity ~capacity:(room 1.5) (fun () ->
        tune ~cache_dir ~report:(fun r -> report2 := Some r))
  in
  let r2 = Option.value_exn !report2 in
  p "the warm call replays the cached winner, or the cold search was contended"
    (replayed r2 || not cacheable);
  p "nothing was refused for device memory in the warm call" (refused = 0);
  Result.iter result ~f:(check_values "the replayed winner computes the right values");

  (* --- 3. Room for half the unscheduled footprint: the base compile is refused, and with it the
     search, before any candidate is compiled. --- *)
  let cache_dir3 = "autotune_cache_device_memory_refused" in
  clean_cache cache_dir3;
  let report3 = ref None in
  let attempts = ref [] in
  let result, refused =
    Autotune.Seam.with_value Autotune.on_candidate_attempt
      (fun label -> attempts := label :: !attempts)
      (fun () ->
        with_capacity ~capacity:(room ~of_:fp_base 0.5) (fun () ->
            tune ~cache_dir:cache_dir3 ~report:(fun r -> report3 := Some r)))
  in
  let r3 = Option.value_exn !report3 in
  p "a routine that does not fit is refused, not searched" (Result.is_error result);
  p "the refusal is a typed pre-search failure at link, with the device-memory decline"
    (refused_for_memory r3);
  p "only the base compile was attempted" (List.equal String.equal !attempts [ "baseline" ]);
  p "the device refused exactly one allocation" (refused = 1);

  (* --- 4. Through the placement A/B: the materialize-all arm cannot fit beside the default arm's
     shipped routine. It is refused; the default arm ships, is cached, and replays. The placement
     store records no decision over a failed arm, so the second call compares the arms again, which
     is the path a warm process takes. --- *)
  p "the materialize-all arm's base compile does not fit beside the default arm's routine"
    (fp + fp_b_base > Float.to_int (1.5 *. Float.of_int fp));
  let cache_dir4 = "autotune_cache_device_memory_arms" in
  clean_cache cache_dir4;
  let run_arms () =
    let reports = ref [] and shipped = ref [] in
    let capacity = room 1.5 in
    let result, refused =
      with_capacity ~capacity (fun () ->
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:cache_dir4
            ~report:(fun r -> reports := r :: !reports)
            ~on_ship:(fun what -> shipped := what :: !shipped)
            (Context.auto ()) t2 comp Ir.Indexing.Empty)
    in
    (result, refused, List.rev !reports, !shipped)
  in
  let result, _refused, reports, shipped = run_arms () in
  p "both arms reported, in position" (List.length reports = 2);
  let arm_a = List.nth_exn reports 0 and arm_b = List.nth_exn reports 1 in
  p "arm A searched without a device-memory decline" (searched arm_a && memory_declines arm_a = 0);
  p "arm B was refused for device memory before its search" (refused_for_memory arm_b);
  p "arm A shipped" (List.equal String.equal shipped [ "A" ]);
  Result.iter result ~f:(check_values "the shipped arm computes the right values");
  let cacheable = arm_a.Autotune.timings_contended = 0 in
  let result, _refused, reports, shipped = run_arms () in
  let arm_a2 = List.nth_exn reports 0 and arm_b2 = List.nth_exn reports 1 in
  p "the next call replays arm A's winner, or its search was contended"
    (replayed arm_a2 || not cacheable);
  p "and searches nothing: arm B is refused again" (refused_for_memory arm_b2);
  p "arm A shipped again" (List.equal String.equal shipped [ "A" ]);
  Result.iter result ~f:(check_values "the replayed arm computes the right values")
