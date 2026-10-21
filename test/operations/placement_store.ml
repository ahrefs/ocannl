(* gh-ocannl-786: the placement-decision store of [Train.tune_placements].

   The schedule cache persists what each search crowned; this store persists what the placement A/B
   and the flip refinement DECIDED, keyed by the decision problem's structural identity
   ([Schedule_cache.canonicalize_source]). A cold run records its decision; a warm run replays it --
   one search from the recorded placement instead of two arms and a flip chain -- and ships the same
   placement vector. Pinned here, on a matmul-plus-relu routine with a policy-virtual intermediate
   (so the refinement has a [`Materialize] candidate and a refined result is possible):

   - the identity is structural: a second graph of the same shape, built from different tensors, has
   the same problem digest, and a lineage that inherits a decision has a different one; - a cold run
   records its decision exactly under clean evidence, and never under a forced arm; - a warm run
   replays: one report, no flip reports, the recorded label, the recorded placements, the right
   values -- and the one search is a schedule-cache replay, which is what shows the recorded
   decision reproduces the very lowering the cold run tuned; - a stale entry -- a decision that no
   longer reproduces its program, or one naming a node the problem lacks -- is ignored, re-tuned,
   and overwritten; - a bypassed store (gh-ocannl-1020, [~placement_store:false], config
   [tune_placement_store]) neither replays nor records: on the warm schedule cache both arms report
   in position, each a schedule-cache replay, and the recorded entry is untouched; - the entry's
   guard is what was tuned (gh-ocannl-1022): its [outcome_digest] is the shipped search's own
   [Autotune.report.source_digest], a fresh lowering of the recorded decision
   ([Train.placement_outcome_digest], the replay guard's recomputation) reproduces it -- an equality
   on lowering determinism, where the schedule-cache replay above pins it only by consequence -- and
   under a timing context that lowers the program differently from the caller's lineage, both the
   digest and the replay guard are the timing lineage's.

   - the identity holds across a process boundary (gh-ocannl-1021): this executable, re-run in a
   role, records in one child process; a second child builds the same computation from other
   tensors, after other nodes have taken the uids the recorder's graph held, and tunes it against
   that directory: it poses the same problem digest and replays the recorded decision -- one search,
   a schedule-cache replay of the recorded lowering, the recorded label and placement, the right
   values -- while a control child whose lineage inherits a decision poses a different problem and
   replays nothing. Either leaves the recorded entry untouched. The leg runs first, while this
   process holds no device.

   Times never enter the golden. A recording rests on clean evidence -- every search the run
   observed had no contention-refused window, did not fail, and timed something -- which the load
   can spoil; such a run records nothing, by design. So every recording run (run 1, run 6, the
   cross-process recorder) is retried in a fresh directory until one attempt records, up to
   [recording_attempts] times (gh-ocannl-1086). Where none did, the claims resting on the recording
   are not waived as a silent [true]: they are skipped through [Verdict], an environment skip naming
   why, which the sweep counts. The claims that rest on a store or a lookup having happened are
   likewise skipped on the filesystem's own evidence: a refusal the cache absorbed, which
   [Schedule_cache.recording_cache_io] reports (gh-ocannl-1040), or a run whose own context has no
   timing identity (gh-ocannl-1043). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module SC = Ir.Schedule_cache
open Verdict.Claims
module Cache_plant = Test_utils.Cache_plant

let approx_tol _ = 1e-4
let n = 8
let cache_dir = "autotune_cache_placement_store"

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* The placement entries in the directory, by key: the keys the entries store, since a filename is a
   digest of its key (gh-ocannl-1228). *)
let placement_keys ?(cache_dir = cache_dir) () =
  List.filter (SC.entry_keys ~dir:cache_dir) ~f:(String.is_prefix ~prefix:"placements-")

let read_entry key = Option.value_exn (SC.lookup_placements ~dir:cache_dir ~key:(Some key))

let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let uncontended (r : Autotune.report) = r.Autotune.timings_contended = 0
let source_digest (r : Autotune.report) = r.Autotune.source_digest

(* The evidence a recording rests on, as [Train.tune_placements] judges it: its own predicate, so
   the test cannot disagree with the recorder (a flip the abandonment rule stopped early, or a
   lineage-refused arm, does not spoil the evidence). *)
let clean_report = Train.placement_evidence_clean
let clean_evidence reports = List.for_all reports ~f:clean_report

(* A search that failed outright: the evidence is unclean, but not by the load, so a retry would
   only fail again. *)
let failed_search reports =
  List.exists reports ~f:(fun r -> Option.is_some (Autotune.terminal_failure r))

(* --- Recording retries (gh-ocannl-1086). A recording run whose evidence was unclean records
   nothing, and every claim about what it recorded -- and every replay of it -- then has nothing to
   check. Retried in a FRESH directory, so the attempt the claims judge is a cold run like the
   first: in the same one, the searches that were clean the first time would replay their schedule
   entries instead. --- *)

let recording_attempts = 3

(* [attempt], each time after [fresh ()], until [settled] holds of its result or
   [recording_attempts] attempts ran: every attempt's result, in order, so the last is the run's. *)
let attempts_until ~fresh ~settled attempt =
  let rec go k acc =
    fresh ();
    let r = attempt () in
    let acc = r :: acc in
    if settled r || k >= recording_attempts then List.rev acc else go (k + 1) acc
  in
  go 1 []

(* Why a recording run's entry is absent, as the skip reason of the claims resting on it: no timing
   identity to key on, a refused store, a compared search that failed, or unclean evidence on every
   attempt. A clean, keyed, unrefused run that recorded nothing is no environment: its record claim
   fails. *)
let unrecorded_on ~identity ~failed ~clean stores =
  if not identity then Cache_plant.unkeyed_on
  else if List.exists stores ~f:Cache_plant.refused then Cache_plant.refused_on
  else if failed then "a compared search failed"
  else if not clean then
    Printf.sprintf "unclean evidence on all %d recording attempts" recording_attempts
  else "a clean recording run that recorded nothing"

let graph ~label =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ label ^ "_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ label ^ "_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  (mc, t2, Train.forward t2)

let problem_digest ctx loss comp =
  SC.digest (Train.placement_problem ctx loss comp Ir.Indexing.Empty)

(* What a run's persistence came to (gh-ocannl-1040, gh-ocannl-1043). The store is best-effort by
   design, so a claim that a clean run recorded something -- or that a later run replayed it -- also
   rests on two facts the timing evidence never observes: that the run's OWN context has a concrete
   timing identity (without one the key is [None] and nothing is consulted; a reference context's
   identity says nothing about this one), and that the cache I/O the claim rests on was not refused
   (on Windows a commit can outlive [Atomic_file]'s bounded retry, or the lock can refuse; the cache
   absorbs both, and [Schedule_cache.recording_cache_io] reports them). *)
type persistence = { identity : bool; io : SC.cache_io list }

let is_store (r : SC.cache_io) = match r.SC.op with SC.Store -> true | SC.Lookup -> false
let is_placement r = String.equal (Cache_plant.store_kind r) "placement"

(* The operations a claim rests on, by kind: a record claim on its run's placement stores, a
   decision replay on its run's placement lookups, a schedule-cache replay on the recording run's
   schedule stores and the replaying run's lookups. *)
let placement_stores p = List.filter p.io ~f:(fun r -> is_store r && is_placement r)
let placement_lookups p = List.filter p.io ~f:(fun r -> (not (is_store r)) && is_placement r)
let schedule_stores p = List.filter p.io ~f:(fun r -> is_store r && not (is_placement r))
let lookups p = List.filter p.io ~f:(Fn.non is_store)
let refused = Cache_plant.refused
let refused_on = Cache_plant.refused_on

(* Whether any of the operations [ops] rests on was refused, naming on stderr, for the claim [label]
   it waives, what refused and why ([Cache_plant.refused_in]). *)
let refused_in ops label = Cache_plant.refused_in (List.concat ops) label

(* [label] over [b], evaluated where none of the operations [ops] was refused; a refusal waives it
   as an environment skip, never as a pass. A refusal that is no transient -- a filename the
   filesystem rejects, a commit that always fails -- would waive these on every run, which the
   process-wide claim before run 8 keeps red. *)
let unless_refused ops label b = Cache_plant.unless_refused (List.concat ops) label b

(* [label] over [b], evaluated where the recording it rests on happened ([recorded]) and none of the
   operations [ops] was refused. Where the recording is absent it is skipped on [on] (from
   {!unrecorded_on}), a counted environment skip rather than the vacuous [true] it used to be
   (gh-ocannl-1086); a refusal skips it as [unless_refused] does. *)
let given_recording ~recorded ~on ops label b =
  if refused_in ops label then skipped ~aggregation:`Environment ~backend:refused_on label
  else gated ~aggregation:`Environment ~when_:recorded ~on label b

(* A recording run's own record claim: [b] (it recorded) where its context is keyed, none of its
   placement stores [ops] was refused, and its evidence was [clean]; skipped otherwise, on the
   reason that holds. *)
let records_when_clean ~identity ~failed ~clean ops label b =
  if not identity then Cache_plant.unless_unkeyed ~identity label b
  else
    given_recording ~recorded:clean
      ~on:(unrecorded_on ~identity ~failed ~clean (List.concat ops))
      ops label b

(* gh-ocannl-1421: the measurement seam under [~rank_fissioned_first], which ranks every
   whole-routine candidate (every label but the [F_] ones [Autotune.spec_label] gives fissioned
   specs) one second behind its reading, so a search crowns a fissioned schedule whenever it timed
   one -- the winner that used to die compiling into a caller lineage that lowers differently (run
   6). The windows it sees are admitted ones only: contention still refuses as in production.
   Counted, so stderr shows the ranking reached candidates at all. *)
let whole_ranked_last = ref 0

let fissioned_first_ranking ~label ~digest:_ ms =
  if String.is_prefix label ~prefix:"F_" then ms
  else (
    Int.incr whole_ranked_last;
    ms +. 1000.)

(* What [on_ship] said (gh-ocannl-1438), with the two facts about it only the run can check: that
   its routine name is the returned routine's own, and that its report is one the run delivered to
   [report] or [flip_report] -- the very value, not an equal one. *)
type ship = { s : Train.shipped; routine_agrees : bool; report_delivered : bool }

let arm s = s.s.Train.ship_arm

(* One placement tune of [graph]'s routine from [ctx] (a fresh [Context.auto ()] by default) with
   [cache_dir] as the cache directory, run and read back: the arm and flip reports in order, what
   shipped, whether the intermediate shipped materialized, the values, and what its persistence came
   to. [~rank_fissioned_first] runs it under {!fissioned_first_ranking}. *)
let tune_in ?ship_arm ?placement_store ?timing_ctx ?(rank_fissioned_first = false) ?ctx ~cache_dir
    (mc, t2, comp) =
  let ctx = match ctx with Some ctx -> ctx | None -> Context.auto () in
  let identity = Option.is_some (Context.timing_identity ctx) in
  let arms = ref [] and flips = ref [] and shipped = ref None in
  let tune () =
    Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ~inline_flips:2
      ~report:(fun r -> arms := r :: !arms)
      ~flip_report:(fun r -> flips := r :: !flips)
      ~on_ship:(fun s -> shipped := Some s)
      ?ship_arm ?placement_store ?timing_ctx ctx t2 comp Ir.Indexing.Empty
  in
  let (ctx_t, routine_t), io =
    SC.recording_cache_io (fun () ->
        if rank_fissioned_first then
          Autotune.Seam.with_value Autotune.on_candidate_measured fissioned_first_ranking tune
        else tune ())
  in
  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  let materialized =
    Tn.Placements.is_materialized_peek (Context.placements ctx_t) mc.Tensor.value
  in
  let shipped = Option.value_exn !shipped in
  let ship =
    {
      s = shipped;
      routine_agrees = String.equal shipped.Train.ship_routine routine_t.Context.name;
      report_delivered =
        Option.exists shipped.Train.ship_report ~f:(fun r ->
            List.exists (!arms @ !flips) ~f:(phys_equal r));
    }
  in
  (List.rev !arms, List.rev !flips, ship, materialized, got, { identity; io })

(* --- The cross-process leg's children (gh-ocannl-1021). The test re-runs this executable in a
   role, one child at a time: the recorder tunes cold into [xproc_cache_dir] after a plain compile
   of the same routine (its reference values), the replayer tunes the same computation against that
   directory, and the control tunes another problem against it. Each prints what it observed as
   [xproc <field> <value>] lines on its stdout, a pipe the parent reads; no child claims anything,
   so every verdict is the parent's, in the golden's order.

   Nothing but the directory crosses between them. The replayer builds a throwaway graph first, so
   the graph it tunes -- built from other tensors under another label -- holds other uids than the
   recorder's (the parent checks that, as the leg's precondition). The control tunes from a lineage
   that has already decided the intermediate materialized: another problem, which must replay
   nothing recorded.

   Every device this leg touches is a child's, and the children run one after another while the
   parent does nothing but read files and pipes: the leg runs before the parent's own in-process
   runs initialize a backend, so a GPU run of this test never holds two device contexts at once (the
   width caps of AGENTS.md count one test action as one GPU process). --- *)

let xproc_cache_dir = "autotune_cache_placement_store_xproc"

type role = Record | Replay | Control

let role_flag = function
  | Record -> "--placement-store-record-child"
  | Replay -> "--placement-store-replay-child"
  | Control -> "--placement-store-control-child"

let child role =
  (match role with Replay -> ignore (graph ~label:"xpad" : _ * _ * _) | Record | Control -> ());
  let ((mc, t2, comp) as g) = graph ~label:(match role with Record -> "xr" | _ -> "xp") in
  let reference =
    match role with
    | Record ->
        let ctx, routine = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
        Context.get_values (Context.run ctx routine) t2.Tensor.value
    | Replay | Control -> [||]
  in
  let ctx () =
    match role with
    | Control -> Context.decide_materialized (Context.auto ()) [ mc.Tensor.value ]
    | Record | Replay -> Context.auto ()
  in
  let digest = problem_digest (ctx ()) t2 comp in
  let arms, flips, shipped, materialized, got, persisted =
    tune_in ~ctx:(ctx ()) ~cache_dir:xproc_cache_dir g
  in
  let floats a = String.concat_array ~sep:" " (Array.map a ~f:(Printf.sprintf "%h")) in
  let out field value = Stdio.printf "xproc %s %s\n" field value in
  out "uid" (Int.to_string t2.Tensor.value.Tn.id);
  out "digest" digest;
  out "arms" (Int.to_string (List.length arms));
  out "flips" (Int.to_string (List.length flips));
  out "shipped" (arm shipped);
  out "materialized" (Bool.to_string materialized);
  out "clean" (Bool.to_string (clean_evidence (arms @ flips)));
  out "failed" (Bool.to_string (failed_search (arms @ flips)));
  out "replayed" (Bool.to_string (match arms with [ r ] -> replayed r | _ -> false));
  out "source" (match arms with [ r ] -> source_digest r | _ -> "-");
  out "values" (floats got);
  out "reference" (floats reference);
  out "identity" (Bool.to_string persisted.identity);
  List.iter persisted.io ~f:(fun (r : SC.cache_io) ->
      out "io"
        (String.concat ~sep:"\t"
           [
             (if is_store r then "store" else "lookup");
             r.SC.key;
             Option.value r.SC.refusal ~default:"";
           ]));
  Stdio.Out_channel.flush Stdio.stdout;
  Stdlib.exit 0

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: flag :: _ -> (
      match List.find [ Record; Replay; Control ] ~f:(fun r -> String.equal flag (role_flag r)) with
      | Some role -> child role
      | None -> ())
  | _ -> ()

type child_report = {
  uid : int;
  digest : string;
  arms : int;
  flips : int;
  shipped : string;
  materialized : bool;
  clean : bool;
  failed : bool;
  replayed : bool;
  source : string;
  values : float array;
  reference : float array;
  persisted : persistence;
}

(* Runs this executable in [role] -- forwarding this process's own [--ocannl_*] flags, so the child
   is configured as the parent is (the environment and [ocannl_config] it inherits) -- waits for it,
   and reads its report. [None] when the child did not exit 0 or did not report every field. *)
let spawn_child role =
  let exe = Stdlib.Sys.executable_name in
  let forwarded = Array.filter Stdlib.Sys.argv ~f:(String.is_prefix ~prefix:"--ocannl_") in
  let ic = Unix.open_process_args_in exe (Array.append [| exe; role_flag role |] forwarded) in
  let lines = Stdio.In_channel.input_lines ic in
  let status = Unix.close_process_in ic in
  let fields =
    List.filter_map lines ~f:(fun l ->
        Option.bind (String.chop_prefix l ~prefix:"xproc ") ~f:(String.lsplit2 ~on:' '))
  in
  let field k = List.Assoc.find_exn fields k ~equal:String.equal in
  let floats k =
    String.split (field k) ~on:' '
    |> List.filter ~f:(Fn.non String.is_empty)
    |> List.map ~f:Float.of_string |> Array.of_list
  in
  match status with
  | Unix.WEXITED 0 ->
      Option.try_with (fun () ->
          {
            uid = Int.of_string (field "uid");
            digest = field "digest";
            arms = Int.of_string (field "arms");
            flips = Int.of_string (field "flips");
            shipped = field "shipped";
            materialized = Bool.of_string (field "materialized");
            clean = Bool.of_string (field "clean");
            failed = Bool.of_string (field "failed");
            replayed = Bool.of_string (field "replayed");
            source = field "source";
            values = floats "values";
            reference = floats "reference";
            persisted =
              {
                identity = Bool.of_string (field "identity");
                io =
                  List.filter_map fields ~f:(fun (k, v) ->
                      if not (String.equal k "io") then None
                      else
                        match String.split v ~on:'\t' with
                        | op :: key :: reason ->
                            let reason = String.concat ~sep:"\t" reason in
                            Some
                              {
                                SC.op = (if String.equal op "store" then SC.Store else SC.Lookup);
                                dir = xproc_cache_dir;
                                key;
                                refusal = (if String.is_empty reason then None else Some reason);
                              }
                        | _ -> None);
              };
          })
  | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ -> None

(* --- Run 7 (gh-ocannl-1021): across a process boundary, ahead of everything that would give this
   process a device. The recorder child tunes cold into a fresh directory; the replaying child must
   pose the same problem and replay its decision; the control child poses another problem against
   the same directory. The recorder is retried, each time into a fresh directory, until one attempt
   records (gh-ocannl-1086). The replay claims are skipped exactly where run 2's are: when no
   attempt recorded (its evidence was unclean every time, or the store is unavailable), and each
   claim when the cache I/O it rests on was refused (gh-ocannl-1040). --- *)

(* The stores the children attempted, for the process-wide claim before run 8. *)
let xproc_stores = ref []

let () =
  (* The one placement entry the directory holds, with its rendering; [None] for none or several. *)
  let held () =
    match placement_keys ~cache_dir:xproc_cache_dir () with
    | [ k ] ->
        Option.map (SC.lookup_placements ~dir:xproc_cache_dir ~key:(Some k)) ~f:(fun e ->
            (k, e, SC.sexp_of_placement_entry e))
    | _ -> None
  in
  let untouched held =
    match held with
    | None -> true
    | Some (k, _, rendered) ->
        Option.equal Sexp.equal (Some rendered)
          (Option.map
             (SC.lookup_placements ~dir:xproc_cache_dir ~key:(Some k))
             ~f:SC.sexp_of_placement_entry)
  in
  (* A recorder that failed outright, or whose search did, settles too: retrying would only fail
     again. *)
  let record_attempts =
    attempts_until
      ~fresh:(fun () -> clean_cache xproc_cache_dir)
      ~settled:(function
        | None -> true
        | Some c -> c.clean || c.failed || Option.is_some (held ()) || not c.persisted.identity)
      (fun () -> spawn_child Record)
  in
  let record = List.last_exn record_attempts in
  p "the recording process runs to completion and reports what it observed" (Option.is_some record);
  let on_record f = Option.value_map record ~default:false ~f in
  let reference = Option.value_map record ~default:[||] ~f:(fun c -> c.reference) in
  p_close "the recording process's tuned routine computes its plain compile's values"
    (Option.value_map record ~default:[||] ~f:(fun c -> c.values))
    reference ~tol:approx_tol;
  let recorded = held () in
  let entry = Option.map recorded ~f:(fun (_, e, _) -> e) in
  let recorder_clean = on_record (fun c -> c.clean) in
  Stdio.eprintf
    "run 7 (not part of the golden): recorder shipped %s, clean %b, stored %b, after %d attempt(s)\n\
     %!"
    (Option.value_map record ~default:"nothing" ~f:(fun c -> c.shipped))
    recorder_clean (Option.is_some entry) (List.length record_attempts);
  let replay = spawn_child Replay in
  let on_replay f = Option.value_map replay ~default:false ~f in
  let ops select c = Option.value_map c ~default:[] ~f:(fun c -> select c.persisted) in
  let unrecorded =
    unrecorded_on
      ~identity:(on_record (fun c -> c.persisted.identity))
      ~failed:(on_record (fun c -> c.failed))
      ~clean:recorder_clean (ops placement_stores record)
  in
  let given_entry = given_recording ~recorded:(Option.is_some entry) ~on:unrecorded in
  p "the replaying process runs to completion and reports what it observed" (Option.is_some replay);
  p "the replaying process numbers the graph's tensor nodes differently"
    (on_record (fun r -> on_replay (fun c -> c.uid <> r.uid)));
  p "the replaying process poses the recording process's problem digest"
    (on_record (fun r -> on_replay (fun c -> String.equal c.digest r.digest)));
  p_close "the replaying process's routine computes the right values"
    (Option.value_map replay ~default:[||] ~f:(fun c -> c.values))
    reference ~tol:approx_tol;
  unless_refused
    [ ops placement_lookups replay ]
    "across processes, the replay is exact when the recording process recorded a decision: one \
     search, no flips"
    (on_replay (fun c -> if Option.is_some entry then c.arms = 1 && c.flips = 0 else c.arms = 2));
  given_entry
    [ ops placement_lookups replay ]
    "a cross-process replay ships the recorded label"
    (on_replay (fun c ->
         match entry with
         | None -> false
         | Some e -> String.equal c.shipped (SC.shipped_label e.SC.decision)));
  given_entry
    [ ops placement_lookups replay ]
    "a cross-process replay ships the recorded placement of the intermediate"
    (on_replay (fun c -> on_record (fun r -> Bool.equal c.materialized r.materialized)));
  (* An entry is recorded only over clean evidence, so the gate's second half is the record's own
     precondition, kept explicit: the schedule entry this replays is the recorder's clean
     search's. *)
  given_recording
    ~recorded:(Option.is_some entry && recorder_clean)
    ~on:unrecorded
    [ ops schedule_stores record; ops lookups replay ]
    "a cross-process replay's one search is a schedule-cache replay"
    (on_replay (fun c -> c.replayed));
  given_entry
    [ ops placement_lookups replay ]
    "a cross-process replay's one search tunes the lowering the recorded decision was measured on"
    (on_replay (fun c ->
         match entry with None -> false | Some e -> String.equal c.source e.SC.outcome_digest));
  (* Skipped when the recorder recorded nothing: the replaying child's tune is then a cold one of
     its own, free to record what its evidence allows. *)
  given_entry []
    "a cross-process replay leaves the recorded decision untouched, and records nothing beside it"
    (untouched recorded && List.length (placement_keys ~cache_dir:xproc_cache_dir ()) = 1);
  (* The control: a too-coarse identity -- one blind to what the lineage decided -- would hand it
     the entry the store now holds for the recorded problem (the recorder's, or, when the recorder's
     evidence was unclean, the replaying child's re-tune): one search where the arms belong. With no
     entry at all neither control claim can discriminate, so both are skipped then. *)
  let held_before_control = held () in
  Stdio.eprintf "control (not part of the golden): the store holds %s for the recorded problem\n%!"
    (if Option.is_some held_before_control then "an entry" else "no entry");
  let control = spawn_child Control in
  let on_control f = Option.value_map control ~default:false ~f in
  p "the control process runs to completion and reports what it observed" (Option.is_some control);
  p "the control process, whose lineage inherits a decision, poses a different problem digest"
    (on_record (fun r -> on_control (fun c -> not (String.equal c.digest r.digest))));
  p_close "the control process's routine computes the right values"
    (Option.value_map control ~default:[||] ~f:(fun c -> c.values))
    reference ~tol:approx_tol;
  let given_held = given_recording ~recorded:(Option.is_some held_before_control) ~on:unrecorded in
  given_held [] "the control process replays nothing recorded: both arms report"
    (on_control (fun c -> c.arms = 2));
  given_held [] "the control process leaves the entry recorded for the other problem untouched"
    (untouched held_before_control);
  xproc_stores :=
    List.concat_map
      (record_attempts @ [ replay; control ])
      ~f:(ops (fun p -> List.filter p.io ~f:is_store))

let () =
  clean_cache cache_dir;
  let mc, t2, comp = graph ~label:"ps" in
  (* Reference values from a plain compile. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in
  (* --- The identity. --- *)
  let mc', t2', comp' = graph ~label:"ps2" in
  let d = problem_digest (Context.auto ()) t2 comp in
  p "the problem digest is structural: a same-shape graph over other tensors has the same digest"
    (String.equal d (problem_digest (Context.auto ()) t2' comp'));
  p "the problem digest is the same for a fresh lineage"
    (String.equal d (problem_digest (Context.auto ()) t2 comp));
  (* gh-ocannl-1421: with a timing context the key also carries how each lineage's lowering accesses
     every node, per node, so it is still free of process-local node ids. *)
  let timed_digest ~decide loss comp =
    SC.digest
      (Train.placement_problem
         ~timing_ctx:(decide (Context.auto ()))
         (Context.auto ()) loss comp Ir.Indexing.Empty)
  in
  let materialize_mid mc c = Context.decide_materialized c [ mc.Tensor.value ] in
  p "with a timing context, a same-shape graph over other tensors has the same digest"
    (String.equal
       (timed_digest ~decide:(materialize_mid mc) t2 comp)
       (timed_digest ~decide:(materialize_mid mc') t2' comp'));
  p "with a timing context, how the timing lineage lowers the routine enters the digest"
    (not
       (String.equal
          (timed_digest ~decide:Fn.id t2 comp)
          (timed_digest ~decide:(materialize_mid mc) t2 comp)));
  p "a lineage that inherits a decision poses a different problem"
    (not
       (String.equal d
          (problem_digest
             (Context.decide_materialized (Context.auto ()) [ mc'.Tensor.value ])
             t2' comp')));
  (* Arm B is defined by the loss's embedded set, so the same computation against another loss is
     another problem: its materialize-all arm materializes different nodes. *)
  p "the same computation tuned against a different loss poses a different problem"
    (not (String.equal d (problem_digest (Context.auto ()) mc comp)));
  (* The arms are measured in the timing lineage, so what it inherits is part of the problem. *)
  let timing_ctx = Context.decide_materialized (Context.auto ()) [ mc.Tensor.value ] in
  p "a timing context that inherits a decision poses a different problem"
    (not
       (String.equal d
          (SC.digest
             (Train.placement_problem ~timing_ctx (Context.auto ()) t2 comp Ir.Indexing.Empty))));
  (* --- The runs. --- *)
  let run ?ship_arm ?placement_store ?timing_ctx ?rank_fissioned_first ?(cache_dir = cache_dir) () =
    tune_in ?ship_arm ?placement_store ?timing_ctx ?rank_fissioned_first ~cache_dir (mc, t2, comp)
  in
  (* A recording run, retried into a fresh [cache_dir] until an attempt records (gh-ocannl-1086):
     every attempt's run with the placement keys it left. Settled by a recording, by clean evidence
     (whose record claim then judges it), by a search that failed outright, or by a context with no
     timing identity to key on. *)
  let recording_run ?timing_ctx ?rank_fissioned_first ~cache_dir () =
    attempts_until
      ~fresh:(fun () -> clean_cache cache_dir)
      ~settled:(fun ((arms, flips, _, _, _, persisted), keys) ->
        (not (List.is_empty keys))
        || clean_evidence (arms @ flips)
        || failed_search (arms @ flips)
        || not persisted.identity)
      (fun () ->
        let r = run ?timing_ctx ?rank_fissioned_first ~cache_dir () in
        (r, placement_keys ~cache_dir ()))
  in
  (* --- Run 1: cold. --- *)
  let attempts1 = recording_run ~cache_dir () in
  let (arms1, flips1, shipped1, materialized1, got1, persisted1), keys1 = List.last_exn attempts1 in
  p "the cold run reports both arms in position" (List.length arms1 = 2);
  p_close "the cold run's routine computes the right values" got1 expected ~tol:approx_tol;
  let clean1 = clean_evidence (arms1 @ flips1) in
  (* Non-emptiness is the recorded case; a run whose evidence the load spoiled records nothing. *)
  let stored1 = List.length keys1 = 1 in
  Stdio.eprintf
    "run 1 (not part of the golden): shipped %s, %d arm and %d flip reports, clean %b, stored %b, \
     after %d attempt(s)\n\
     %!"
    (arm shipped1) (List.length arms1) (List.length flips1) clean1 stored1 (List.length attempts1);
  (* Gated on the run's own context (gh-ocannl-1043): the store keys on that context's timing
     identity, so a reference context's says nothing about whether this run could record. *)
  let failed1 = failed_search (arms1 @ flips1) in
  records_when_clean ~identity:persisted1.identity ~failed:failed1 ~clean:clean1
    [ placement_stores persisted1 ]
    "the cold run records exactly one decision, whenever its evidence was clean" stored1;
  (* Over every attempt, the discarded ones included: an unclean attempt is exactly where a decision
     must NOT have been recorded. *)
  p_all "a decision is never recorded over refused windows or a failed search"
    (List.concat_map attempts1 ~f:(fun ((arms, flips, _, _, _, _), keys) ->
         List.map (arms @ flips) ~f:(fun r -> (keys, r))))
    ~f:(fun (keys, r) -> List.is_empty keys || clean_report r);
  let unrecorded1 =
    unrecorded_on ~identity:persisted1.identity ~failed:failed1 ~clean:clean1
      (placement_stores persisted1)
  in
  let given_stored1 = given_recording ~recorded:stored1 ~on:unrecorded1 in
  given_stored1 [] "the recorded decision is the shipped one"
    (match keys1 with
    | [ k ] -> String.equal (SC.shipped_label (read_entry k).SC.decision) (arm shipped1)
    | _ -> false);
  (* gh-ocannl-1022: the guard is what was tuned. The arms tune distinct lowerings (the intermediate
     is policy-virtual in A and materialized in B), which is what keeps the equalities below from
     being satisfied by accident. The shipped search is exact for an arm; a refined winner is one of
     the flip searches. *)
  p "the two arms report the distinct digests of the lowerings they tuned"
    (match arms1 with
    | [ ra; rb ] ->
        (not (String.is_empty (source_digest ra)))
        && (not (String.is_empty (source_digest rb)))
        && not (String.equal (source_digest ra) (source_digest rb))
    | _ -> false);
  let entry1 = if stored1 then Some (read_entry (List.hd_exn keys1)) else None in
  (* gh-ocannl-1438: what shipped, as [on_ship] names it -- the arm, the search, the routine --
     against what the run itself shows. A cold run replays no placement; its shipped search is the
     shipping arm's own report, in that arm's position, or one of the flips'. *)
  p "the cold run's ship is not a placement replay" (not shipped1.s.Train.ship_replayed);
  p "the cold run's ship names the routine it returned" shipped1.routine_agrees;
  p "the cold run's ship carries the report of the search it shipped, in that search's position"
    (shipped1.report_delivered
    &&
    match (arm shipped1, arms1, shipped1.s.Train.ship_report) with
    | "A", [ ra; _ ], Some r -> phys_equal r ra
    | "B", [ _; rb ], Some r -> phys_equal r rb
    | "flip", _, Some r -> List.exists flips1 ~f:(phys_equal r)
    | _ -> false);
  given_stored1 [] "the recorded outcome digest is the shipped search's own source digest"
    (match (entry1, shipped1.s.Train.ship_report) with
    | Some e, Some r -> String.equal (source_digest r) e.SC.outcome_digest
    | _ -> false);
  given_stored1 []
    "a fresh lowering of the recorded decision reproduces the digest the shipped search tuned"
    (match entry1 with
    | None -> false
    | Some e ->
        String.equal e.SC.outcome_digest
          (Train.placement_outcome_digest (Context.auto ()) t2 comp Ir.Indexing.Empty e.SC.decision));
  (* --- Run 2: warm. --- *)
  let arms2, flips2, shipped2, materialized2, got2, persisted2 = run () in
  p_close "the warm run's routine computes the right values" got2 expected ~tol:approx_tol;
  (* A refused lookup makes the warm run a cold one; the claims resting on the replay are skipped
     then, as on a refused store, and where run 1 recorded nothing. The first claim reads both ways:
     without an entry the warm run must search both arms. *)
  unless_refused
    [ placement_lookups persisted2 ]
    "the warm run replays the decision exactly when the cold run recorded one: one search, no flips"
    (if stored1 then List.length arms2 = 1 && List.length flips2 = 0 else List.length arms2 = 2);
  given_stored1
    [ placement_lookups persisted2 ]
    "a replay ships the recorded label"
    (String.equal (arm shipped2) (arm shipped1));
  unless_refused
    [ placement_lookups persisted2 ]
    "the warm run's ship says it replayed the placement exactly when it ran the one search"
    (Bool.equal shipped2.s.Train.ship_replayed stored1
    && Bool.equal shipped2.s.Train.ship_replayed (List.length arms2 = 1));
  p "the warm run's ship names the routine it returned" shipped2.routine_agrees;
  given_stored1
    [ placement_lookups persisted2 ]
    "a replay's ship carries its one search's report"
    (match (arms2, shipped2.s.Train.ship_report) with
    | [ r ], Some sr -> phys_equal r sr
    | _ -> false);
  given_stored1
    [ placement_lookups persisted2 ]
    "a replay ships the recorded placement of the intermediate"
    (Bool.equal materialized2 materialized1);
  (* The recorded decision reproduces the lowering the cold run tuned: the one search is a
     schedule-cache replay of the winner the cold run's shipped search crowned, which a clean run 1
     -- the only kind that records -- stored. *)
  given_recording ~recorded:(stored1 && clean1) ~on:unrecorded1
    [ schedule_stores persisted1; lookups persisted2 ]
    "a replay's one search is a schedule-cache replay"
    (match arms2 with [ r ] -> replayed r | _ -> false);
  (* The same fact without the cache in between (gh-ocannl-1022): the replayed search tuned the very
     lowering whose digest the entry records. *)
  given_stored1
    [ placement_lookups persisted2 ]
    "a replay's one search tunes the lowering the recorded decision was measured on"
    (match (entry1, arms2) with
    | Some e, [ r ] -> String.equal (source_digest r) e.SC.outcome_digest
    | _ -> false);
  (* --- Run 2b: a replayed search that fails without poisoning the lineage is a losing arm, not a
     failed tune: the recorded decision is treated as stale and the arms are searched. The failure
     is injected at the first candidate attempt of the process from here on -- the replayed search's
     base compile when there is an entry to replay, otherwise arm A's -- so with an entry the failed
     replay is not reported and both arms are, in position; without one, arm A dies in its slot and
     arm B ships, which the cold path already handles (autotune_arm_containment.ml). Either way the
     caller sees the two arms. --- *)
  let attempts = ref 0 in
  let arms2b, _, _, _, got2b, persisted2b =
    Autotune.Seam.with_value Autotune.on_candidate_attempt
      (fun _ ->
        Int.incr attempts;
        if !attempts = 1 then failwith "ps: injected replay failure")
      run
  in
  p_close "after a failed replay, the routine computes the right values" got2b expected
    ~tol:approx_tol;
  p "a failed replay is not reported; the arms it falls back to report in position"
    (List.length arms2b = 2);
  (* --- Run 3: a forced arm neither consults nor records the store. Whichever run recorded the
     entry -- run 1, or run 2 re-tuning after an unclean run 1 -- is what must survive; where no run
     recorded anything the entry-level claims are skipped. --- *)
  let recorded = Option.map (List.hd (placement_keys ())) ~f:(fun k -> (k, read_entry k)) in
  let snapshot () =
    Option.map recorded ~f:(fun (k, _) -> SC.sexp_of_placement_entry (read_entry k))
  in
  let before = snapshot () in
  (* Whichever run recorded it -- run 1 under a clean load, else the re-tune that run 2 or 2b fell
     back to -- the entry the store holds is one a fresh lowering of its decision reproduces
     (gh-ocannl-1022). Run 1's claims above are skipped exactly when run 1 recorded nothing; this
     one only when no run did. *)
  gated ~aggregation:`Environment ~when_:(Option.is_some recorded) ~on:unrecorded1
    "the entry the store holds is reproduced by a fresh lowering of its decision"
    (match recorded with
    | None -> false
    | Some (_, e) ->
        String.equal e.SC.outcome_digest
          (Train.placement_outcome_digest (Context.auto ()) t2 comp Ir.Indexing.Empty e.SC.decision));
  let arms3, _, shipped3, _, got3, persisted3 = run ~ship_arm:Train.Force_arm_b () in
  p_close "the forced run's routine computes the right values" got3 expected ~tol:approx_tol;
  p "a forced arm searches both arms whatever the store holds" (List.length arms3 = 2);
  p "a forced arm ships the forced arm" (String.equal (arm shipped3) "B");
  p "a forced arm's ship is not a placement replay" (not shipped3.s.Train.ship_replayed);
  p "a forced run leaves the recorded decision untouched"
    (Option.equal Sexp.equal before (snapshot ())
    && List.length (placement_keys ()) = Option.length recorded);
  (* --- Run 3b: a bypassed store (gh-ocannl-1020) is the placement A/B on a warm schedule cache:
     whatever the store holds, both arms are compared -- each replaying the schedule run 1 crowned
     for it, exactly when run 1's evidence was clean -- and nothing is recorded or overwritten.
     --- *)
  Stdio.eprintf "run 3b (not part of the golden): bypassing a store that %s\n%!"
    (if Option.is_some recorded then "holds an entry" else "is empty");
  let arms3b, _, _, _, got3b, persisted3b = run ~placement_store:false () in
  p_close "the bypassed-store run's routine computes the right values" got3b expected
    ~tol:approx_tol;
  p "a bypassed store compares both arms, in position, whatever the store holds"
    (List.length arms3b = 2);
  (* Skipped where run 1 could not cache (gh-ocannl-1043: its own context's identity) or never had
     clean evidence, and on a refused store or lookup in between. *)
  let label =
    "a bypassed store's arms replay the schedule cache, whenever the cold run cached them"
  in
  if refused_in [ schedule_stores persisted1; lookups persisted3b ] label then
    skipped ~aggregation:`Environment ~backend:refused_on label
  else
    gated_all ~aggregation:`Environment ~when_:(persisted1.identity && clean1) ~on:unrecorded1 label
      arms3b ~f:replayed;
  p "a bypassed store leaves the recorded decision untouched"
    (Option.equal Sexp.equal before (snapshot ())
    && List.length (placement_keys ()) = Option.length recorded);
  (* --- Runs 4 and 5: stale entries are ignored, re-tuned, and overwritten. Each corruption comes
     with the fact that tells the re-tune's entry from it. --- *)
  let stale_decision = SC.Refined [ { SC.node = 100_000; flip = `Inline } ] in
  let stale =
    [
      ( "no longer reproduces its program",
        (fun (e : SC.placement_entry) -> { e with SC.outcome_digest = "0" ^ e.SC.outcome_digest }),
        fun (e : SC.placement_entry) (e' : SC.placement_entry) ->
          String.equal e'.SC.outcome_digest e.SC.outcome_digest );
      ( "names a node the problem lacks",
        (fun (e : SC.placement_entry) -> { e with SC.decision = stale_decision }),
        fun (e : SC.placement_entry) (e' : SC.placement_entry) ->
          String.equal e'.SC.outcome_digest e.SC.outcome_digest
          && not (SC.equal_placement_decision e'.SC.decision stale_decision) );
    ]
  in
  let stale_placement_stores = ref [] and stale_runs = ref [] in
  List.iter stale ~f:(fun (how, corrupt, overwritten) ->
      (* The corruption is a store too, and a refused one leaves the valid entry in place. *)
      let (), corrupting =
        SC.recording_cache_io (fun () ->
            Option.iter recorded ~f:(fun (key, e) ->
                SC.store_placements ~dir:cache_dir ~key:(Some key) (corrupt e)))
      in
      let corrupting = List.filter corrupting ~f:is_placement in
      let arms, flips, _, _, got, persisted = run () in
      stale_placement_stores := !stale_placement_stores @ corrupting @ placement_stores persisted;
      stale_runs := !stale_runs @ [ (persisted, arms) ];
      p_close
        (Printf.sprintf "after an entry that %s, the routine computes the right values" how)
        got expected ~tol:approx_tol;
      (* Skipped also on a refused read of it: the run then never saw the stale entry, and tunes
         both arms cold. *)
      given_recording ~recorded:(Option.is_some recorded) ~on:unrecorded1
        [ corrupting; placement_lookups persisted ]
        (Printf.sprintf "an entry that %s is ignored and the placements re-tuned" how)
        (List.length arms = 2);
      (* The re-tune records under the same rule as any run, so its flips count too. Not retried:
         the stale entry would have to be planted again, and the warm directory would replay a
         cached abandonment. *)
      let clean = clean_evidence (arms @ flips) in
      given_recording
        ~recorded:(clean && Option.is_some recorded)
        ~on:
          (if Option.is_none recorded then unrecorded1
           else
             unrecorded_on ~identity:persisted.identity
               ~failed:(failed_search (arms @ flips))
               ~clean (placement_stores persisted))
        [ corrupting; placement_stores persisted ]
        (Printf.sprintf "an entry that %s is overwritten by the re-tune, given clean evidence" how)
        (Option.value_map recorded ~default:false ~f:(fun (key, e) ->
             Option.value_map
               (SC.lookup_placements ~dir:cache_dir ~key:(Some key))
               ~default:false ~f:(overwritten e))));
  (* --- Run 6 (gh-ocannl-1022): the arms searched in a timing lineage that inherits a decision --
     the intermediate materialized -- which the caller's lineage does not. The shipped search tuned
     the TIMING lineage's lowering, so that is the digest recorded, and the replay guard recomputes
     it in the timing lineage: recomputed in the caller's, which lowers the default placements
     differently (the precondition claim), a recorded default would never replay. A new problem,
     recorded with the same retries as run 1.

     Its own directory, and fissioned candidates ranked FIRST (gh-ocannl-1421). Every schedule these
     searches crown is tuned on the timing lineage's lowering, where the intermediate is
     materialized; in the caller's lineage it is virtual, so no arm-A schedule applies there, and
     [Autotune.tune] refuses arm A up front ([Lineage_refused], their lowerings differ): it searches
     nothing, ships the untuned default and ranks last, while arm B, which materializes the
     intermediate in both lineages, searches and ships its own crown. A fissioned winner used to die
     instead, promoting the virtual intermediate to a device buffer -- fatal under strict
     classification, so the replayed search failed and the warm run went back to the arms
     (gh-ocannl-1386's flake, whenever the timing noise crowned a fissioned schedule; on Metal a
     whole-routine winner's compile failure was fatal too). Ranking fissioned candidates first makes
     that winner deterministic, and the claim that no compared search failed -- which no environment
     waives -- fails on the old behavior every time. The materialize-all lineage of run 1's arm B
     lowers to the same program as this timing lineage, so a shared directory would hand these
     searches run 1's unranked winner as a schedule-cache hit. --- *)
  let timing_ctx () = Context.decide_materialized (Context.auto ()) [ mc.Tensor.value ] in
  p "the caller's and the timing lineage lower the default placements differently"
    (not
       (String.equal
          (Train.placement_outcome_digest (Context.auto ()) t2 comp Ir.Indexing.Empty SC.Default)
          (Train.placement_outcome_digest ~timing_ctx:(timing_ctx ()) (Context.auto ()) t2 comp
             Ir.Indexing.Empty SC.Default)));
  let timing_cache_dir = "autotune_cache_placement_store_timing" in
  let attempts6 =
    recording_run ~timing_ctx:(timing_ctx ()) ~rank_fissioned_first:true ~cache_dir:timing_cache_dir
      ()
  in
  let (arms6, flips6, shipped6, _, got6, persisted6), keys6 = List.last_exn attempts6 in
  p_close "with a timing context, the cold run's routine computes the right values" got6 expected
    ~tol:approx_tol;
  let clean6 = clean_evidence (arms6 @ flips6) in
  let entry6 =
    match keys6 with [ k ] -> SC.lookup_placements ~dir:timing_cache_dir ~key:(Some k) | _ -> None
  in
  Stdio.eprintf
    "run 6 (not part of the golden): shipped %s, clean %b, stored %b, after %d attempt(s); %d \
     whole-routine candidate timings ranked last\n\
     %!"
    (arm shipped6) clean6 (Option.is_some entry6) (List.length attempts6) !whole_ranked_last;
  let failed6 = failed_search (arms6 @ flips6) in
  p_none
    "with a timing context, no compared search failed: a schedule the caller's lineage cannot take \
     is never compiled into it"
    (arms6 @ flips6) ~f:(fun r -> Option.is_some (Autotune.terminal_failure r));
  p
    "with a timing context, arm A -- whose lowering differs between the two lineages -- is refused \
     before searching, and arm B, which materializes the intermediate in both, searches"
    (match arms6 with
    | [ { Autotune.outcome = Autotune.Lineage_refused _; best_ms; _ }; rb ] -> (
        Float.is_inf best_ms
        &&
        match rb.Autotune.outcome with
        | Autotune.Searched | Autotune.Cache_replay -> true
        | _ -> false)
    | _ -> false);
  records_when_clean ~identity:persisted6.identity ~failed:failed6 ~clean:clean6
    [ placement_stores persisted6 ]
    "with a timing context, the cold run records a decision of its own, whenever its evidence was \
     clean"
    (Option.is_some entry6);
  let given_entry6 =
    given_recording ~recorded:(Option.is_some entry6)
      ~on:
        (unrecorded_on ~identity:persisted6.identity ~failed:failed6 ~clean:clean6
           (placement_stores persisted6))
  in
  given_entry6 []
    "with a timing context, the recorded outcome digest is the shipped search's own source digest"
    (shipped6.report_delivered
    &&
    match (entry6, shipped6.s.Train.ship_report) with
    | Some e, Some r -> String.equal (source_digest r) e.SC.outcome_digest
    | _ -> false);
  given_entry6 []
    "with a timing context, a fresh lowering of the recorded decision in the timing lineage \
     reproduces it"
    (match entry6 with
    | None -> false
    | Some e ->
        String.equal e.SC.outcome_digest
          (Train.placement_outcome_digest ~timing_ctx:(timing_ctx ()) (Context.auto ()) t2 comp
             Ir.Indexing.Empty e.SC.decision));
  let arms6w, flips6w, shipped6w, _, got6w, persisted6w =
    run ~timing_ctx:(timing_ctx ()) ~rank_fissioned_first:true ~cache_dir:timing_cache_dir ()
  in
  p_close "with a timing context, the warm run's routine computes the right values" got6w expected
    ~tol:approx_tol;
  Stdio.eprintf
    "run 6w (not part of the golden): arms [%s], %d flip reports, shipped %s, entry %s, source \
     digests [%s] vs recorded %s\n\
     %!"
    (String.concat ~sep:"; "
       (List.map arms6w ~f:(fun r ->
            Printf.sprintf "%s contended=%d best=%.4f"
              (Autotune.outcome_name r.Autotune.outcome)
              r.Autotune.timings_contended r.Autotune.best_ms)))
    (List.length flips6w) (arm shipped6w)
    (Option.value_map entry6 ~default:"none" ~f:(fun e -> SC.shipped_label e.SC.decision))
    (String.concat ~sep:"; " (List.map arms6w ~f:source_digest))
    (Option.value_map entry6 ~default:"-" ~f:(fun e -> e.SC.outcome_digest));
  (* "No flips" is an absence: an empty flip-report list is the passing case, as in run 2. *)
  given_entry6
    [ placement_lookups persisted6w ]
    "with a timing context, the warm run replays the recorded decision: one search, no flips, the \
     recorded label, tuning the recorded lowering"
    (match (entry6, arms6w) with
    | Some e, [ r ] ->
        List.length flips6w = 0
        && String.equal (arm shipped6w) (SC.shipped_label e.SC.decision)
        && shipped6w.s.Train.ship_replayed
        && String.equal (source_digest r) e.SC.outcome_digest
    | _ -> false);
  (* --- Every claim above that rests on a store or a lookup is waived when it was refused, which is
     right for a transient refusal and wrong for one that is no accident: a filename the filesystem
     rejects, a permission, a commit that always fails. Those refuse EVERY store (of a directory),
     and would turn this test from red into a permanent skip that hosted CI aggregates nowhere. So,
     over the stores this process and its children attempted under real names (run 8's injected
     refusals excluded), not all of any directory's placement or schedule stores may be refused --
     never waived by a refusal. --- *)
  let stores p = List.filter p.io ~f:is_store in
  let attempt_persisted attempts = List.map attempts ~f:(fun ((_, _, _, _, _, p), _) -> p) in
  let attempted =
    !xproc_stores
    @ List.concat_map
        (attempt_persisted attempts1
        @ [ persisted2; persisted2b; persisted3; persisted3b ]
        @ attempt_persisted attempts6 @ [ persisted6w ])
        ~f:stores
    @ !stale_placement_stores
  in
  (* Judged per cache directory and per kind of entry, so the cross-process leg's directory, or the
     schedule entries, refused alone cannot hide behind another's successes. Expected wherever two
     arm searches of this process's keyed runs stored their winners -- a search that ran, timed
     something and had no refused window stores into [cache_dir] ([Autotune.tune]'s rule) -- so that
     directory's schedule group has two attempts by construction; told from the reports, never from
     the stores the claim judges (gh-ocannl-1115). Contention that leaves fewer is an environment
     the claim cannot judge. Flips, the cross-process leg and run 6's own directory go uncounted:
     counting fewer only skips more. *)
  let stores_winner (r : Autotune.report) =
    (match r.Autotune.outcome with Autotune.Searched -> true | _ -> false)
    && uncontended r
    && Float.is_finite r.Autotune.best_ms
  in
  let storing_searches =
    List.sum
      (module Int)
      (List.map attempts1 ~f:(fun ((arms, _, _, _, _, p), _) -> (p, arms))
      @ [ (persisted2, arms2); (persisted2b, arms2b); (persisted3, arms3); (persisted3b, arms3b) ]
      @ !stale_runs)
      ~f:(fun (p, arms) -> if p.identity then List.count arms ~f:stores_winner else 0)
  in
  Stdio.eprintf "arm searches that stored a winner (not part of the golden): %d\n%!"
    storing_searches;
  Cache_plant.not_every_store_refused ~expected:(storing_searches >= 2)
    "in each cache directory, the filesystem refused not every placement store, nor every schedule \
     store, the runs attempted"
    attempted;
  (* --- Run 8 (gh-ocannl-1040): the waivers above rest on the cache REPORTING a refusal it
     absorbed. A cold run into a fresh directory whose every commit the filesystem refuses -- the
     [Sys_error] a Windows commit raises once its bounded retry runs out -- runs to the right
     values, records no decision, and reports the placement store's refusal among its own. --- *)
  let refused_cache_dir = "autotune_cache_placement_store_refused" in
  clean_cache refused_cache_dir;
  let arms8, flips8, _, _, got8, persisted8 =
    Ir.Resource_fault_injection.with_callback
      (fun point ->
        if Ir.Resource_fault_injection.equal_point point Schedule_cache_before_commit then
          raise (Stdlib.Sys_error "ps: injected commit refusal"))
      ~f:(fun () -> tune_in ~cache_dir:refused_cache_dir (mc, t2, comp))
  in
  p_close "under refused commits, the cold run's routine computes the right values" got8 expected
    ~tol:approx_tol;
  (* Over the arms the run tuned: the decision it reached is what must have left no entry. *)
  p_empty "refused commits leave no decision recorded" ~over:arms8
    (placement_keys ~cache_dir:refused_cache_dir ());
  (* Whether the recorder attempted a store is its own decision ([Train.tune_placements]'s
     [persist], which an abandoned flip does not stop), read from the record rather than re-derived:
     every placement store run 8 attempted must be reported refused -- for any reason: on a host
     that refuses the staging file's creation itself (a Windows path beyond MAX_PATH,
     gh-ocannl-1040's rog leg) the injected hook never fires, and [schedule_cache_regime] pins the
     injected reason -- and the run must have attempted one. The gate is the keyed run being clean,
     told from its reports and never from the stores the claim judges (gh-ocannl-1115,
     gh-ocannl-1361): a contended run is skipped whether or not it stored, since a gate that opened
     on the stores being there would turn a regression that stops attempting them into a skip. *)
  let stores8 = placement_stores persisted8 in
  let clean8 = clean_evidence (arms8 @ flips8) in
  Stdio.eprintf
    "run 8 (not part of the golden): clean %b, %d placement stores, %d refused cache operations\n%!"
    clean8 (List.length stores8)
    (List.count persisted8.io ~f:refused);
  gated ~aggregation:`Environment ~when_:(persisted8.identity && clean8)
    ~on:"no timing identity, or a contended run"
    "under refused commits, the cold run reports its placement store's refusal"
    ((not (List.is_empty stores8)) && List.for_all stores8 ~f:refused)
