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

   Pinned to cc ([Context.cpu]): the refusal is backend-independent, and cc is always available. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
module Outcome = Ir.Schedule_outcome
open Verdict.Claims

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
    (Utils.get_global_flag ~default:true ~arg_name:"strict_failure_classification")

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

let limits = Context.hardware_limits base_ctx
let capabilities = Context.codegen_capabilities base_ctx

let key =
  SC.cache_key
    ~timing_identity:(Context.timing_identity base_ctx)
    ~limits ~capabilities canon ~backend:(Context.backend_name base_ctx)

(* Timed-cache reuse needs a concrete device identity; without one nothing is looked up. *)
let cache_available = Option.is_some key

let () =
  if not cache_available then
    Stdio.eprintf "timed-cache reuse unavailable: concrete device identity missing\n"

let cache_claim label value =
  gated ~aggregation:`Environment ~when_:cache_available ~on:"no-device-identity" label value

(* The cache absorbs a filesystem refusal by design (gh-ocannl-1040): a refused plant or lookup
   leaves the call with no entry to refuse, for a reason none of the claims below observes. Each leg
   records its I/O, and a claim resting on the replay is waived, as an environment skip, where the
   plant's store or the call's lookup was refused -- named on stderr. *)
let refused_in ios label =
  let refusals = List.filter ios ~f:(fun (io : SC.cache_io) -> Option.is_some io.SC.refusal) in
  List.iter refusals ~f:(fun (io : SC.cache_io) ->
      Stdio.eprintf "%s (not part of the golden): cache I/O refused: %s\n%!" label
        (Option.value io.SC.refusal ~default:""));
  not (List.is_empty refusals)

let replay_claim ios label value =
  if not cache_available then cache_claim label value
  else
    gated ~aggregation:`Environment
      ~when_:(not (refused_in ios label))
      ~on:"refused cache I/O" label value

(* Every planted store, for the one claim no refusal waives: a deterministic refusal (a rejected
   filename, a commit that always fails) would otherwise turn every claim above into a permanent
   skip. *)
let planted = ref []

let plant ~saved ~segments =
  let (), ios =
    SC.recording_cache_io (fun () ->
        SC.store ~dir:cache_dir ~key
          {
            SC.version = SC.entry_version;
            backend = Context.backend_name base_ctx;
            numerics = SC.numerics_tag ();
            codegen =
              Some
                (SC.codegen_tag ~backend:(Context.backend_name base_ctx) ~limits ~capabilities ());
            objective = Some (SC.objective_tag ());
            source_digest = SC.digest canon;
            saved;
            segments;
            best_ms = 0.;
            baseline_ms = 0.;
            default_ms = None;
            mma_best_ms = None;
            default_fingerprint = None;
            best_steps = None;
          })
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

let approx x y = Float.(abs (x - y) < 1e-4)

(* The report names the refusal, and it is the typed one [of_saved] raises for [check]. *)
let refused_by ~check (r : Autotune.report) =
  match r.Autotune.cache_replay_refusal with
  | Some { Outcome.phase = Outcome.Transform; cause = Outcome.Illegal_schedule c; _ } ->
      String.equal c.check check
  | Some _ | None -> false

let outcome_is name (r : Autotune.report) = String.equal (Autotune.outcome_name r.outcome) name

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
  p_all2 "with search off, the untuned default computes the right values" values expected ~f:approx;
  let planted_io = plant ~saved:dangling ~segments:None in
  let values, r, lookups = tune ~search:true in
  let ios = planted_io @ lookups in
  replay_claim ios "the corrupt winner's replay is refused with a typed cause"
    (refused_by ~check:"Schedule_cache.of_saved" r);
  replay_claim ios "the refused replay re-searches" (outcome_is "searched" r);
  p_all2 "the re-searched routine computes the right values" values expected ~f:approx;

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
  p_all2 "the routine re-searched past the refused prelude computes the right values" values
    expected ~f:approx;
  cache_claim "not every planted entry's store was refused"
    (List.exists !planted ~f:(fun (io : SC.cache_io) -> Option.is_none io.SC.refusal))
