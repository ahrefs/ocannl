(* gh-ocannl-594: construction limits describe link-anywhere validity; timed evidence belongs to one
   concrete device/toolchain. Missing identity must not even open either persistent store. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module SC = Ir.Schedule_cache
module BI = Ir.Backend_intf
module FI = Ir.Resource_fault_injection

let cache_dir = "autotune_cache_device"
let absent_cache_dir = "autotune_cache_device_absent"

let clean dir =
  if Stdlib.Sys.file_exists dir then (
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f));
    Stdlib.Sys.rmdir dir)

let schedule_entry canon backend =
  {
    SC.version = SC.entry_version;
    backend;
    numerics = SC.numerics_tag ();
    codegen = None;
    objective = None;
    source_digest = SC.digest canon;
    saved = [];
    segments = None;
    finer_fission = None;
    best_ms = 1.;
    baseline_ms = 2.;
    default_ms = None;
    mma_best_ms = None;
    default_fingerprint = None;
  }

let () =
  let registry uuid = "  \"IOPlatformUUID\" = \"" ^ uuid ^ "\"\n" in
  let host_a = "01234567-89AB-CDEF-0123-456789ABCDEF" in
  let host_b = "11234567-89AB-CDEF-0123-456789ABCDEF" in
  let host = Utils.macos_platform_uuid in
  p "hardware UUID discovery normalizes case"
    (Option.equal String.equal (host (registry host_a)) (host (registry (String.lowercase host_a))));
  p "distinct hardware UUIDs separate otherwise identical hosts"
    (match (host (registry host_a), host (registry host_b)) with
    | Some a, Some b -> not (String.equal a b)
    | _ -> false);
  p_none "missing, malformed and placeholder UUIDs decline identity"
    [ ""; registry ""; registry "not-a-uuid"; registry "00000000-0000-0000-0000-000000000000" ]
    ~f:(fun text -> Option.is_some (host text));
  clean cache_dir;
  clean absent_cache_dir;
  let x = TDSL.ndarray [| 1.; 2.; 3.; 4. |] ~label:[ "device_identity_x" ] ~output_dims:[ 4 ] () in
  let%op y = x + x in
  let comp = Train.forward y in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let construction = BI.sexp_of_hardware_limits limits in
  let identity = Context.timing_identity ctx in
  Stdio.eprintf "backend=%s; timing identity (not part of the golden): %s\n%!" backend
    (Sexp.to_string_hum ([%sexp_of: BI.timing_identity option] identity));
  p "identity queries preserve conservative construction limits"
    (Sexp.equal construction (BI.sexp_of_hardware_limits (Context.hardware_limits ctx)));
  p "a concrete context's identity is stable"
    (Option.equal BI.equal_timing_identity identity (Context.timing_identity ctx));
  (match identity with
  | Some identity ->
      p "the available identity contains concrete device facts"
        (not (String.is_empty identity.device_signature))
  | None ->
      skipped ~aggregation:`Environment ~backend
        "the available identity contains concrete device facts");
  let canon = ref None in
  let _, _ =
    Context.compile
      ~lowered_transform:(fun opt ->
        canon := Some (SC.canonicalize ~static_indices:[] opt);
        [ opt ])
      ctx comp Ir.Indexing.Empty
  in
  let canon = Option.value_exn !canon in
  let fixture =
    {
      BI.device_signature = "model-A:sm80:memory320";
      toolchain_signature = Some "driver-A:compiler-A";
    }
  in
  let identities =
    [
      fixture;
      { fixture with device_signature = "model-B:sm40:memory160" };
      { fixture with toolchain_signature = Some "driver-B:compiler-A" };
      { fixture with toolchain_signature = Some "driver-A:compiler-B" };
    ]
  in
  let key timing_identity = SC.cache_key ~timing_identity ~limits canon ~backend in
  let placement_key timing_identity = SC.placement_key ~timing_identity ~limits canon ~backend in
  p_pairwise_distinct "device, driver and compiler changes independently separate schedule keys"
    (List.map identities ~f:(fun i -> Option.value_exn (key (Some i))))
    ~equal:String.equal ~to_string:Fn.id;
  p_pairwise_distinct "device, driver and compiler changes independently separate placement keys"
    (List.map identities ~f:(fun i -> Option.value_exn (placement_key (Some i))))
    ~equal:String.equal ~to_string:Fn.id;
  p "unavailable identity produces no key for either store"
    (Option.is_none (key None) && Option.is_none (placement_key None));
  let entry = schedule_entry canon backend in
  let placement =
    {
      SC.version = SC.placement_entry_version;
      backend;
      numerics = SC.numerics_tag ();
      codegen = SC.codegen_tag ~limits ();
      objective = SC.objective_tag ();
      problem_digest = SC.digest canon;
      decision = SC.Materialize_all;
      outcome_digest = SC.digest canon;
      shipped_ms = 1.;
      arm_a_ms = 2.;
      arm_b_ms = 1.;
    }
  in
  let valid_key = key (Some fixture) and valid_placement = placement_key (Some fixture) in
  let lock_hits = ref 0 in
  let probe f =
    FI.with_callback (function FI.Schedule_cache_before_lock -> Int.incr lock_hits | _ -> ()) ~f
  in
  probe (fun () ->
      SC.store ~dir:cache_dir ~key:valid_key entry;
      SC.store_placements ~dir:cache_dir ~key:valid_placement placement);
  p "concrete identities reach both stores' lock boundary" (!lock_hits = 2);
  p "concrete identity round-trips a schedule entry"
    (Option.value_map (SC.lookup ~dir:cache_dir ~key:valid_key) ~default:false ~f:(fun e ->
         String.equal e.source_digest entry.source_digest));
  p "concrete identity round-trips a placement entry"
    (Option.value_map (SC.lookup_placements ~dir:cache_dir ~key:valid_placement) ~default:false
       ~f:(fun e -> SC.equal_placement_decision e.decision placement.decision));
  let device_only = Some { fixture with toolchain_signature = None } in
  let device_key = key device_only and device_placement_key = placement_key device_only in
  SC.store ~dir:cache_dir ~key:device_key entry;
  SC.store_placements ~dir:cache_dir ~key:device_placement_key placement;
  p "absent toolchain metadata preserves schedule persistence"
    (Option.is_some (SC.lookup ~dir:cache_dir ~key:device_key));
  p "absent toolchain metadata preserves placement persistence"
    (Option.is_some (SC.lookup_placements ~dir:cache_dir ~key:device_placement_key));
  let stamp = Stdlib.Filename.concat cache_dir SC.regime_stamp_filename in
  Stdio.Out_channel.write_all stamp ~data:"0\n";
  let snapshot () =
    Stdlib.Sys.readdir cache_dir |> Array.to_list |> List.sort ~compare:String.compare
    |> List.map ~f:(fun f -> (f, Stdio.In_channel.read_all (Stdlib.Filename.concat cache_dir f)))
  in
  let before = snapshot () in
  lock_hits := 0;
  probe (fun () ->
      SC.store ~dir:cache_dir ~key:(key None) entry;
      SC.store ~dir:absent_cache_dir ~key:(key None) entry;
      SC.store_placements ~dir:cache_dir ~key:(placement_key None) placement;
      SC.store_placements ~dir:absent_cache_dir ~key:(placement_key None) placement;
      p_none "unavailable identity never reads a schedule, even from a populated store"
        [
          SC.lookup ~dir:cache_dir ~key:(key None); SC.lookup ~dir:absent_cache_dir ~key:(key None);
        ]
        ~f:Option.is_some;
      p_none "unavailable identity never reads a placement, even from a populated store"
        [
          SC.lookup_placements ~dir:cache_dir ~key:(placement_key None);
          SC.lookup_placements ~dir:absent_cache_dir ~key:(placement_key None);
        ]
        ~f:Option.is_some);
  p "unavailable identity never reaches the cache lock" (!lock_hits = 0);
  p "unavailable identity creates no directory or lock"
    (not (Stdlib.Sys.file_exists absent_cache_dir));
  p "unavailable identity neither writes entries nor advances or sweeps a stale regime"
    (List.equal
       (fun (name_a, contents_a) (name_b, contents_b) ->
         String.equal name_a name_b && String.equal contents_a contents_b)
       before (snapshot ()));
  clean cache_dir;
  let reports = ref [] in
  let run () =
    let ctx, routine =
      Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
        ~report:(fun r -> reports := r :: !reports)
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx y.Tensor.value
  in
  let first = run () in
  let second = run () in
  p_all2 "the cold routine computes the reference" first [| 2.; 4.; 6.; 8. |] ~f:Float.equal;
  p_all2 "the second routine computes the reference" second first ~f:Float.equal;
  let second_report, first_report =
    match !reports with [ b; a ] -> (b, a) | _ -> failwith "two reports"
  in
  let replayed =
    match second_report.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false
  in
  p "runtime replay requires a concrete device identity" ((not replayed) || Option.is_some identity);
  p "unavailable identity leaves the runtime store absent"
    (Option.is_some identity || not (Stdlib.Sys.file_exists cache_dir));
  p "the identity is stable across allocation, execution and replay"
    (Option.equal BI.equal_timing_identity identity (Context.timing_identity ctx));
  if Option.is_none identity then (
    Stdio.eprintf "concrete device identity unavailable: persistent replay disabled\n";
    skipped ~aggregation:`Environment ~backend
      "a complete cold search replays on the same concrete device")
  else if first_report.Autotune.timings_contended > 0 then
    skipped ~aggregation:`Environment ~backend
      "a complete cold search replays on the same concrete device"
  else p "a complete cold search replays on the same concrete device" replayed
