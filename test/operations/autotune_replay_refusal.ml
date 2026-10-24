(* gh-ocannl-1242: a stale or corrupt schedule-cache entry reads as a typed refusal followed by a
   miss, never as the end of [Autotune.tune].

   [Schedule_cache.of_saved] replays a saved schedule against this process's canonical form. A
   reference the form cannot resolve -- a symbol minted by an op the schedule does not have, a
   tensor-node index past the routine's nodes -- used to raise a bare [Invalid_argument], and under
   the default [strict_failure_classification] an untyped exception inside a candidate compile is
   FATAL, cache-replay provenance included: one bad file ended the call instead of re-searching. It
   now raises a typed [Illegal_schedule] cause, the shape gh-ocannl-1164 gave a drifted fission
   replay.

   The test plants two corrupt entries under the key the routine's own lowering derives, decodable
   (a file that does not decode is already a plain miss) and carrying the routine's own source
   digest, so the lookup finds them and the replay is what refuses:

   1. a whole-routine ([W_saved]) winner whose [Swap] names the outer loop of a [Split] the schedule
   does not contain -- a dangling [Minted] symbol reference. With search off, the call compiles the
   untuned default; with search on, it re-searches. Both reports carry the refusal. 2. a
   split-reduce winner whose whole-routine prelude names tensor node 99 of a routine with three --
   the prelude is resolved before fission, so its refusal precedes any segment check.

   The cache absorbs filesystem refusals by design, so the claims resting on a replay are waived (an
   environment skip, never a pass) where the plant's store or the call's lookup was refused, and one
   unwaived claim keeps a deterministic refusal from turning them into a permanent skip.

   A loaded host refuses timing windows too (gh-ocannl-1423), so the claim that the re-search times
   the baseline it relinked reads the search's own record of the baseline's window,
   [baseline_refusal]: the re-search is retried a bounded number of times while that window is
   refused, and the claim is an environment skip only when every attempt's was, must-pass whenever
   one was admitted. The skip names the refusal's cause (gh-ocannl-1517): contention, a queued
   calibration with no batch, or a degenerate clock reading. A control (leg 4) refuses the window
   through [Autotune.on_candidate_measured], with host contention cleared so the scripted reading is
   the one refused, and pins that the record names that cause and that the claim it gates would have
   failed -- a skip, not a pass.

   Pinned to cc ([Context.cpu]): the refusal is backend-independent, and cc is always available. It
   is also a backend whose baseline is timed, which is what makes the re-search's relink of the
   baseline a replay released (gh-ocannl-1326) observable. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
module Outcome = Ir.Schedule_outcome
open Verdict.Claims
module Cache_plant = Test_utils.Cache_plant

let cache_dir = "autotune_cache_replay_refusal"

(* The dune sandbox persists across runs: a previous run's re-searched winner would replay. *)
let () =
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f))

(* The refusal under test is the difference strict classification makes: permissive classification
   contained the untyped exception too. *)
let () =
  p "strict failure classification is on"
    (Utils.registered_flag ~arg_name:"strict_failure_classification")

let n = 8
let k = 4
let m = 8

let av =
  Array.init (n * k) ~f:(Ll_test.cycle_flat ~dims:[| n; k |] ~modulus:5 ~offset:(-2.) ~stride:0.25)

let bv =
  Array.init (k * m) ~f:(Ll_test.cycle_flat ~dims:[| k; m |] ~modulus:3 ~offset:(-1.) ~stride:0.5)

let expected =
  Array.init (n * m) ~f:(fun idx ->
      let i = idx / m and j = idx % m in
      let acc = ref 0. in
      for kk = 0 to k - 1 do
        acc := !acc +. (av.((i * k) + kk) *. bv.((kk * m) + j))
      done;
      !acc)

let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ n; k ] ()
let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ k; m ] ()
let%op c = a +* "ik;kj=>ij" b
let comp = Train.forward c

(* The base lowering [tune] replays against, captured hermetically (codegen decides the remaining
   placements into the table the captured record shares), and the key its entry lives under. *)
let base_ctx = Context.cpu ()

let canon =
  let captured = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured :=
          Some
            {
              opt with
              LL.traced_store = Hashtbl.copy opt.LL.traced_store;
              LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
            };
        [ opt ])
      (Context.cpu ()) comp Ir.Indexing.Empty
  in
  SC.canonicalize (Option.value_exn ~here:[%here] !captured)

let backend = Context.backend_name base_ctx
let timing_identity = Context.timing_identity base_ctx
let limits = Context.hardware_limits base_ctx
let capabilities = Context.codegen_capabilities base_ctx
let key = Cache_plant.key ~backend ~timing_identity ~limits ~capabilities canon

(* Timed-cache reuse needs a concrete device identity; without one nothing is looked up. *)
let cache_available = Option.is_some key

(* Each leg records its I/O, and a claim resting on the replay is waived where the plant's store or
   the call's lookup was refused ([Cache_plant.unless_unkeyed_or_refused]). *)
let replay_claim = Cache_plant.unless_unkeyed_or_refused ~identity:cache_available

(* Every planted store, for the one claim no refusal waives: a deterministic refusal (a rejected
   filename, a commit that always fails) would otherwise turn every claim above into a permanent
   skip. *)
let planted = ref []

let plant ~saved ~segments =
  let ios =
    Cache_plant.plant ~saved ?segments ~cache_dir ~backend ~timing_identity ~limits ~capabilities
      canon
  in
  planted := !planted @ ios;
  ios

(* The call's report and values, and its lookups -- what a replay claim rests on beside the plant
   (the re-search's own store does not). *)
let tune ~search =
  let report = ref None in
  let (ctx, routine), ios =
    SC.recording_cache_io (fun () ->
        Autotune.tune ~search ~name:"replay_refusal" ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
          ~log:false ~progress:false
          ~report:(fun r -> report := Some r)
          (Context.cpu ()) comp Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx c.Tensor.value in
  Context.release ctx;
  let lookups =
    List.filter ios ~f:(fun (io : SC.cache_io) ->
        match io.SC.op with SC.Lookup -> true | SC.Store -> false)
  in
  (values, Option.value_exn ~here:[%here] !report, lookups)

let approx_tol _ = 1e-4

(* The report names the refusal, and it is the typed one [of_saved] raises for [check]. *)
let refused_by ~check (r : Autotune.report) =
  match r.Autotune.cache_replay_refusal with
  | Some { Outcome.phase = Outcome.Transform; cause = Outcome.Illegal_schedule c; _ } ->
      String.equal c.check check
  | Some _ | None -> false

let outcome_is name (r : Autotune.report) = String.equal (Autotune.outcome_name r.outcome) name

(* gh-ocannl-1326: what the re-search did with the baseline it relinked -- on cc, where it is
   dispatched, timed it. *)
let baseline_timed (r : Autotune.report) =
  (not r.Autotune.baseline_declined) && Float.is_finite r.Autotune.baseline_ms

(* gh-ocannl-1423: how many times a re-search whose baseline window the host refused is re-run
   before the claim on that window is skipped. *)
let baseline_attempts = 3

(* The skip's reason, naming the refusal's cause (gh-ocannl-1517). *)
let baseline_refused_on (cause : Autotune.batch_admission) =
  "baseline timing "
  ^
  match cause with
  | Autotune.Refused_contended -> "refused under contention"
  | Refused_unbatched -> "refused: queued calibration measured no batch"
  | Refused_unreadable -> "refused: a degenerate clock reading"
  | Admitted -> "admitted"

(* A claim on the baseline's timing: skipped where the search recorded the baseline's own window as
   refused, otherwise a replay claim like the others. The gate is the record of the window, never
   [baseline_ms]: a baseline the re-search never relinked or never timed has no refused window, and
   fails. *)
let baseline_claim (r : Autotune.report) ios label b =
  match r.Autotune.baseline_refusal with
  | Some cause ->
      Verdict.skipped ~aggregation:`Environment ~backend:(baseline_refused_on cause) label
  | None -> replay_claim ios label b

let () =
  (* --- 1. A whole-routine winner with a dangling minted-symbol reference. --- *)
  let dangling = [ SC.Swap { outer = SC.Minted (0, SC.Split_outer); inner = SC.Base 0 } ] in
  let planted_io = plant ~saved:dangling ~segments:None in
  let values, r, lookups = tune ~search:false in
  let ios = planted_io @ lookups in
  replay_claim ios "with search off, the corrupt winner's replay is refused with a typed cause"
    (refused_by ~check:"Schedule_cache.of_saved" r);
  replay_claim ios "with search off, the call compiles the untuned default"
    (outcome_is "search-disabled" r);
  p_close "with search off, the untuned default computes the right values" values expected
    ~tol:approx_tol;
  (* Re-run, from a fresh plant, while the host refuses the relinked baseline's window. *)
  let rec re_search attempt =
    let planted_io = plant ~saved:dangling ~segments:None in
    let values, r, lookups = tune ~search:true in
    if Option.is_some r.Autotune.baseline_refusal && attempt < baseline_attempts then (
      Stdio.eprintf
        "re-search attempt %d of %d (not part of the golden): the baseline's timing window was \
         refused (%s); retrying\n\
         %!"
        attempt baseline_attempts
        (Option.value_map r.Autotune.baseline_refusal ~default:"-" ~f:Autotune.batch_admission_name);
      re_search (attempt + 1))
    else (planted_io @ lookups, values, r)
  in
  let ios, values, r = re_search 1 in
  replay_claim ios "the corrupt winner's replay is refused with a typed cause"
    (refused_by ~check:"Schedule_cache.of_saved" r);
  replay_claim ios "the refused replay re-searches" (outcome_is "searched" r);
  (* gh-ocannl-1326: the baseline is released before a replay compile, so that the two are never
     resident together; the re-search a refused replay leads to relinks it, and on cc, where it is
     dispatched, times it. *)
  baseline_claim r ios "the re-search times the baseline the replay released" (baseline_timed r);
  p_close "the re-searched routine computes the right values" values expected ~tol:approx_tol;

  (* --- 2. A split-reduce winner whose prelude names a tensor node the routine lacks. --- *)
  p "the routine numbers fewer than 100 tensor nodes"
    (match SC.tn_of_ref canon 99 with
    | _ -> false
    | exception Outcome.Cause_at (_, Outcome.Illegal_schedule _) -> true);
  let planted_io =
    plant
      ~saved:[ SC.Split_reduce { axis = SC.Base 0; target = 99; num_blocks = 2 } ]
      ~segments:(Some [ { SC.seg_kind = `Normal; seg_units = 1; seg_digest = ""; seg_saved = [] } ])
  in
  let values, r, lookups = tune ~search:true in
  let ios = planted_io @ lookups in
  replay_claim ios "the corrupt split-reduce prelude's replay is refused with a typed cause"
    (refused_by ~check:"Schedule_cache.tn_of_ref" r);
  replay_claim ios "the refused prelude re-searches" (outcome_is "searched" r);
  p_close "the routine re-searched past the refused prelude computes the right values" values
    expected ~tol:approx_tol;
  (* --- 3. gh-ocannl-1326: the baseline a replay released is relinked when the replay is refused,
     and a device that refuses that relink for memory is asked the floor every schedule must
     allocate (gh-ocannl-1554) -- the relink's delta bounds no candidate from below. 3a: the device
     refuses the relink alone and admits the floor, so the relink's decline is the baseline's
     account and the search runs on its candidates. 3b: it refuses the floor as well, so the search
     is refused, exactly as a refused base compile refuses it -- rather than declining every
     candidate in turn. The device is simulated: the first working-pool allocation after the cache
     lookup (the stale replay is refused at its transform, before it allocates) is the relink's, the
     next one the floor's. --- *)
  let relink_leg ~refusals =
    let planted_io = plant ~saved:dangling ~segments:None in
    let report = ref None and attempts = ref [] in
    let looked_up = ref false and refused = ref 0 in
    let refuse_relink (point : Ir.Resource_fault_injection.point) =
      match point with
      | Schedule_cache_before_replay -> looked_up := true
      | Delta_pool_allocated when !looked_up && !refused < refusals ->
          Int.incr refused;
          raise
            (Outcome.Device_memory_exhausted
               { backend = "simulated"; requested = 1; free = Some 0; detail = "gh-ocannl-1326" })
      | _ -> ()
    in
    let raised, ios =
      SC.recording_cache_io (fun () ->
          match
            Autotune.Seam.with_value Autotune.on_candidate_attempt
              (fun label -> attempts := label :: !attempts)
              (fun () ->
                Ir.Resource_fault_injection.with_callback refuse_relink ~f:(fun () ->
                    Autotune.tune ~search:true ~name:"replay_refusal" ~beam_width:1 ~rounds:0
                      ~repeats:1 ~cache_dir ~log:false ~progress:false
                      ~report:(fun r -> report := Some r)
                      (Context.cpu ()) comp Ir.Indexing.Empty))
          with
          | _ -> false
          | exception Utils.User_error _ -> true)
    in
    let lookups =
      List.filter ios ~f:(fun (io : SC.cache_io) ->
          match io.SC.op with SC.Lookup -> true | SC.Store -> false)
    in
    ( planted_io @ lookups,
      raised,
      !refused,
      Option.value_exn ~here:[%here] !report,
      List.rev !attempts )
  in
  let memory_decline (r : Autotune.report) =
    List.exists r.Autotune.declines ~f:(fun d ->
        Outcome.equal_rejection_key d.Autotune.key
          (Outcome.Resource_exceeded_key Outcome.Device_memory))
  in
  let ios, raised, refused, r, attempts = relink_leg ~refusals:1 in
  replay_claim ios
    "a device that refuses the relinked baseline alone, admitting the floor, still searches"
    (refused = 1 && (not raised) && outcome_is "searched" r && r.Autotune.baseline_declined);
  replay_claim ios "the refused relink is the baseline's device-memory decline" (memory_decline r);
  replay_claim ios "candidates are compiled after the refused relink"
    (List.exists attempts ~f:(fun l -> not (String.equal l "baseline")));
  let ios, raised, refused, r, attempts = relink_leg ~refusals:2 in
  replay_claim ios "a device that refuses the relinked baseline and the floor refuses the search"
    (raised && refused = 2 && outcome_is "pre-search-failure" r && r.Autotune.baseline_declined);
  replay_claim ios "the refusal carries the device-memory decline" (memory_decline r);
  replay_claim ios "no candidate is compiled after the refused floor"
    (List.equal String.equal attempts [ "baseline" ]);
  (* --- 4. gh-ocannl-1423's control: the relinked baseline's window refused at admission, as a
     loaded host refuses it, here on every host. The record names the baseline's refusal, and the
     claim that record gates would have failed, so leg 1 skips it rather than passing it. Waived
     like the other replay claims: where the plant was refused, an earlier leg's winner can replay
     and time nothing. --- *)
  let planted_io = plant ~saved:dangling ~segments:None in
  let _values, r, lookups =
    Autotune.Seam.with_value Autotune.on_candidate_measured
      (fun ~label ~digest:_ ms -> if String.equal label "baseline" then Float.nan else ms)
      (fun () -> Autotune.with_uncontended_test_windows (fun () -> tune ~search:true))
  in
  let ios = planted_io @ lookups in
  replay_claim ios
    "a refused baseline window is recorded as the baseline's refusal, by a degenerate clock reading"
    (Option.equal Poly.equal r.Autotune.baseline_refusal (Some Autotune.Refused_unreadable));
  replay_claim ios "a refused baseline window leaves the baseline untimed" (not (baseline_timed r));
  (* A keyed plant always attempts its store. *)
  Cache_plant.not_every_store_refused ~expected:cache_available
    "not every planted entry's store was refused" !planted
