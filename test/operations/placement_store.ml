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

   Times never enter the golden; the claims that involve one are waived by the load's own evidence
   (contention-refused windows), as autotune_arm_containment.ml does. The claims that rest on a
   store or a lookup having happened are waived, as environment skips, by the filesystem's own
   evidence: a refusal the cache absorbed, which [Schedule_cache.recording_cache_io] reports
   (gh-ocannl-1040), or a run whose own context has no timing identity (gh-ocannl-1043). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module SC = Ir.Schedule_cache
open Verdict.Claims

let approx a b = Float.(abs (a -. b) < 1e-4)
let n = 8
let cache_dir = "autotune_cache_placement_store"

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* The placement entries in the directory, by key (the filename without its extension is the key:
   [Schedule_cache.cache_file] sanitizes a key that is already filename-safe). *)
let placement_keys ?(dir = cache_dir) () =
  if not (Stdlib.Sys.file_exists dir) then []
  else
    Stdlib.Sys.readdir dir |> Array.to_list
    |> List.filter_map ~f:(fun f ->
        if String.is_prefix f ~prefix:"placements-" then String.chop_suffix f ~suffix:".sexp"
        else None)

let read_entry key = Option.value_exn (SC.lookup_placements ~dir:cache_dir ~key:(Some key))

let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched | Autotune.Cache_replay -> true | _ -> false

let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let uncontended (r : Autotune.report) = r.Autotune.timings_contended = 0
let source_digest (r : Autotune.report) = r.Autotune.source_digest

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
   identity says nothing about this one), and that none of the run's cache I/O was refused (on
   Windows a commit can outlive [Atomic_file]'s bounded retry, or the lock can refuse; the cache
   absorbs both, [Schedule_cache.recording_cache_io] reports them). *)
type persistence = { identity : bool; refusals : (string * string) list (* key, reason *) }

let refusals io =
  List.filter_map io ~f:(fun (r : SC.cache_io) ->
      Option.map r.SC.refusal ~f:(fun reason -> (r.SC.key, reason)))

(* Whether any of [runs]' cache I/O was refused, naming on stderr, for the claim [label] it waives,
   what refused and why. *)
let refused_in runs label =
  let refused = List.concat_map runs ~f:(fun r -> r.refusals) in
  List.iter refused ~f:(fun (key, reason) ->
      Stdio.eprintf "%s (not part of the golden): cache I/O under %s refused: %s\n%!" label key
        reason);
  not (List.is_empty refused)

let refused_on = "refused cache I/O"

(* [label] over [b], evaluated where none of [runs]' cache I/O was refused; a refusal waives it as
   an environment skip, never as a pass. *)
let unless_refused runs label b =
  gated ~aggregation:`Environment ~when_:(not (refused_in runs label)) ~on:refused_on label b

(* One placement tune of [graph]'s routine from [ctx] (a fresh [Context.auto ()] by default) with
   [cache_dir] as the cache directory, run and read back: the arm and flip reports in order, what
   shipped, whether the intermediate shipped materialized, the values, and what its persistence came
   to. *)
let tune_in ?ship_arm ?placement_store ?timing_ctx ?ctx ~cache_dir (mc, t2, comp) =
  let ctx = match ctx with Some ctx -> ctx | None -> Context.auto () in
  let identity = Option.is_some (Context.timing_identity ctx) in
  let arms = ref [] and flips = ref [] and shipped = ref None in
  let (ctx_t, routine_t), io =
    SC.recording_cache_io (fun () ->
        Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ~inline_flips:2
          ~report:(fun r -> arms := r :: !arms)
          ~flip_report:(fun r -> flips := r :: !flips)
          ~on_ship:(fun what -> shipped := Some what)
          ?ship_arm ?placement_store ?timing_ctx ctx t2 comp Ir.Indexing.Empty)
  in
  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  let materialized =
    Tn.Placements.is_materialized_peek (Context.placements ctx_t) mc.Tensor.value
  in
  ( List.rev !arms,
    List.rev !flips,
    Option.value_exn !shipped,
    materialized,
    got,
    { identity; refusals = refusals io } )

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
  out "shipped" shipped;
  out "materialized" (Bool.to_string materialized);
  out "clean"
    (Bool.to_string
       (List.for_all (arms @ flips) ~f:(fun r ->
            completed r && uncontended r && Float.is_finite r.Autotune.best_ms)));
  out "replayed" (Bool.to_string (match arms with [ r ] -> replayed r | _ -> false));
  out "source" (match arms with [ r ] -> source_digest r | _ -> "-");
  out "values" (floats got);
  out "reference" (floats reference);
  out "identity" (Bool.to_string persisted.identity);
  List.iter persisted.refusals ~f:(fun (key, reason) -> out "refusal" (key ^ "\t" ^ reason));
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
            replayed = Bool.of_string (field "replayed");
            source = field "source";
            values = floats "values";
            reference = floats "reference";
            persisted =
              {
                identity = Bool.of_string (field "identity");
                refusals =
                  List.filter_map fields ~f:(fun (k, v) ->
                      if String.equal k "refusal" then String.lsplit2 v ~on:'\t' else None);
              };
          })
  | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ -> None

(* --- Run 7 (gh-ocannl-1021): across a process boundary, ahead of everything that would give this
   process a device. The recorder child tunes cold into a fresh directory; the replaying child must
   pose the same problem and replay its decision; the control child poses another problem against
   the same directory. The replay claims are waived exactly where run 2's are: when the recorder
   recorded nothing (its evidence was unclean, or the store is unavailable), the schedule-cache
   replay also when its evidence was contended, and each claim when the cache I/O it rests on was
   refused (gh-ocannl-1040). --- *)
let () =
  clean_cache xproc_cache_dir;
  (* The one placement entry the directory holds, with its rendering; [None] for none or several. *)
  let held () =
    match placement_keys ~dir:xproc_cache_dir () with
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
  let record = spawn_child Record in
  p "the recording process runs to completion and reports what it observed" (Option.is_some record);
  let on_record f = Option.value_map record ~default:false ~f in
  let reference = Option.value_map record ~default:[||] ~f:(fun c -> c.reference) in
  p_all2 "the recording process's tuned routine computes its plain compile's values"
    (Option.value_map record ~default:[||] ~f:(fun c -> c.values))
    reference ~f:approx;
  let recorded = held () in
  let entry = Option.map recorded ~f:(fun (_, e, _) -> e) in
  let recorder_clean = on_record (fun c -> c.clean) in
  Stdio.eprintf "run 7 (not part of the golden): recorder shipped %s, clean %b, stored %b\n%!"
    (Option.value_map record ~default:"nothing" ~f:(fun c -> c.shipped))
    recorder_clean (Option.is_some entry);
  let replay = spawn_child Replay in
  let on_replay f = Option.value_map replay ~default:false ~f in
  let persisted c = Option.value_map c ~default:[] ~f:(fun c -> [ c.persisted ]) in
  p "the replaying process runs to completion and reports what it observed" (Option.is_some replay);
  p "the replaying process numbers the graph's tensor nodes differently"
    (on_record (fun r -> on_replay (fun c -> c.uid <> r.uid)));
  p "the replaying process poses the recording process's problem digest"
    (on_record (fun r -> on_replay (fun c -> String.equal c.digest r.digest)));
  p_all2 "the replaying process's routine computes the right values"
    (Option.value_map replay ~default:[||] ~f:(fun c -> c.values))
    reference ~f:approx;
  unless_refused (persisted replay)
    "across processes, the replay is exact when the recording process recorded a decision: one \
     search, no flips"
    (on_replay (fun c -> if Option.is_some entry then c.arms = 1 && c.flips = 0 else c.arms = 2));
  unless_refused (persisted replay) "a cross-process replay ships the recorded label"
    (on_replay (fun c ->
         match entry with
         | None -> true
         | Some e -> String.equal c.shipped (SC.shipped_label e.SC.decision)));
  unless_refused (persisted replay)
    "a cross-process replay ships the recorded placement of the intermediate"
    (on_replay (fun c ->
         Option.is_none entry || on_record (fun r -> Bool.equal c.materialized r.materialized)));
  unless_refused
    (persisted record @ persisted replay)
    "a cross-process replay's one search is a schedule-cache replay"
    (on_replay (fun c -> Option.is_none entry || (not recorder_clean) || c.replayed));
  unless_refused (persisted replay)
    "a cross-process replay's one search tunes the lowering the recorded decision was measured on"
    (on_replay (fun c ->
         match entry with None -> true | Some e -> String.equal c.source e.SC.outcome_digest));
  (* Waived when the recorder recorded nothing: the replaying child's tune is then a cold one of its
     own, free to record what its evidence allows. *)
  p "a cross-process replay leaves the recorded decision untouched, and records nothing beside it"
    (Option.is_none recorded
    || (untouched recorded && List.length (placement_keys ~dir:xproc_cache_dir ()) = 1));
  (* The control: a too-coarse identity -- one blind to what the lineage decided -- would hand it
     the entry the store now holds for the recorded problem (the recorder's, or, when the recorder's
     evidence was unclean, the replaying child's re-tune): one search where the arms belong. With no
     entry at all the claim cannot discriminate, which stderr says. *)
  let held_before_control = held () in
  Stdio.eprintf "control (not part of the golden): the store holds %s for the recorded problem\n%!"
    (if Option.is_some held_before_control then "an entry" else "no entry");
  let control = spawn_child Control in
  let on_control f = Option.value_map control ~default:false ~f in
  p "the control process runs to completion and reports what it observed" (Option.is_some control);
  p "the control process, whose lineage inherits a decision, poses a different problem digest"
    (on_record (fun r -> on_control (fun c -> not (String.equal c.digest r.digest))));
  p_all2 "the control process's routine computes the right values"
    (Option.value_map control ~default:[||] ~f:(fun c -> c.values))
    reference ~f:approx;
  p "the control process replays nothing recorded: both arms report"
    (on_control (fun c -> c.arms = 2));
  p "the control process leaves the entry recorded for the other problem untouched"
    (untouched held_before_control)

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
  let run ?ship_arm ?placement_store ?timing_ctx () =
    tune_in ?ship_arm ?placement_store ?timing_ctx ~cache_dir (mc, t2, comp)
  in
  (* --- Run 1: cold. --- *)
  let arms1, flips1, shipped1, materialized1, got1, persisted1 = run () in
  p "the cold run reports both arms in position" (List.length arms1 = 2);
  p_all2 "the cold run's routine computes the right values" got1 expected ~f:approx;
  let observed1 = arms1 @ flips1 in
  let clean1 =
    List.for_all observed1 ~f:(fun r ->
        completed r && uncontended r && Float.is_finite r.Autotune.best_ms)
  in
  let keys1 = placement_keys () in
  (* Non-emptiness is the recorded case; a run whose evidence the load spoiled records nothing. *)
  let stored1 = List.length keys1 = 1 in
  Stdio.eprintf
    "run 1 (not part of the golden): shipped %s, %d arm and %d flip reports, clean %b, stored %b\n\
     %!"
    shipped1 (List.length arms1) (List.length flips1) clean1 stored1;
  (* Gated on the run's own context (gh-ocannl-1043): the store keys on that context's timing
     identity, so a reference context's says nothing about whether this run could record. *)
  if persisted1.identity then
    unless_refused [ persisted1 ]
      "the cold run records exactly one decision, whenever its evidence was clean"
      ((not clean1) || stored1)
  else (
    Stdio.eprintf "concrete device identity unavailable: placement persistence disabled\n";
    skipped ~aggregation:`Environment ~backend:(Context.backend_name ctx_ref)
      "the cold run records exactly one decision, whenever its evidence was clean");
  p_all "a decision is never recorded over refused windows or a failed search" observed1
    ~f:(fun r -> (not stored1) || (completed r && uncontended r));
  p "the recorded decision is the shipped one"
    ((not stored1)
    || String.equal (SC.shipped_label (read_entry (List.hd_exn keys1)).SC.decision) shipped1);
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
  let shipped_searches1 =
    match (shipped1, arms1) with "A", [ ra; _ ] -> [ ra ] | "B", [ _; rb ] -> [ rb ] | _ -> flips1
  in
  p "the recorded outcome digest is the shipped search's own source digest"
    (match entry1 with
    | None -> not stored1
    | Some e ->
        List.exists shipped_searches1 ~f:(fun r ->
            String.equal (source_digest r) e.SC.outcome_digest));
  p "a fresh lowering of the recorded decision reproduces the digest the shipped search tuned"
    (match entry1 with
    | None -> not stored1
    | Some e ->
        String.equal e.SC.outcome_digest
          (Train.placement_outcome_digest (Context.auto ()) t2 comp Ir.Indexing.Empty e.SC.decision));
  (* --- Run 2: warm. --- *)
  let arms2, flips2, shipped2, materialized2, got2, persisted2 = run () in
  p_all2 "the warm run's routine computes the right values" got2 expected ~f:approx;
  (* A refused lookup makes the warm run a cold one; the claims resting on the replay are waived
     then, as on a refused store. *)
  unless_refused [ persisted2 ]
    "the warm run replays the decision exactly when the cold run recorded one: one search, no flips"
    (if stored1 then List.length arms2 = 1 && List.length flips2 = 0 else List.length arms2 = 2);
  unless_refused [ persisted2 ] "a replay ships the recorded label"
    ((not stored1) || String.equal shipped2 shipped1);
  unless_refused [ persisted2 ] "a replay ships the recorded placement of the intermediate"
    ((not stored1) || Bool.equal materialized2 materialized1);
  (* The recorded decision reproduces the lowering the cold run tuned: the one search is a
     schedule-cache replay of the winner the cold run's shipped search crowned. Waived only by the
     cold run's shipped search having stored nothing, which under a clean run 1 it did not. *)
  unless_refused [ persisted1; persisted2 ] "a replay's one search is a schedule-cache replay"
    ((not stored1) || (not clean1) || match arms2 with [ r ] -> replayed r | _ -> false);
  (* The same fact without the cache in between (gh-ocannl-1022): the replayed search tuned the very
     lowering whose digest the entry records. Not waived by contention. *)
  unless_refused [ persisted2 ]
    "a replay's one search tunes the lowering the recorded decision was measured on"
    (match (entry1, arms2) with
    | None, _ -> not stored1
    | Some e, [ r ] -> String.equal (source_digest r) e.SC.outcome_digest
    | Some _, _ -> false);
  (* --- Run 2b: a replayed search that fails without poisoning the lineage is a losing arm, not a
     failed tune: the recorded decision is treated as stale and the arms are searched. The failure
     is injected at the first candidate attempt of the process from here on -- the replayed search's
     base compile when there is an entry to replay, otherwise arm A's -- so with an entry the failed
     replay is not reported and both arms are, in position; without one, arm A dies in its slot and
     arm B ships, which the cold path already handles (autotune_arm_containment.ml). Either way the
     caller sees the two arms. --- *)
  let attempts = ref 0 in
  (Autotune.on_candidate_attempt :=
     fun _ ->
       Int.incr attempts;
       if !attempts = 1 then failwith "ps: injected replay failure");
  let arms2b, _, _, _, got2b, _ =
    Exn.protect ~f:run ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  p_all2 "after a failed replay, the routine computes the right values" got2b expected ~f:approx;
  p "a failed replay is not reported; the arms it falls back to report in position"
    (List.length arms2b = 2);
  (* --- Run 3: a forced arm neither consults nor records the store. Whichever run recorded the
     entry -- run 1, or run 2 re-tuning after a contended run 1 -- is what must survive; a run the
     load kept from recording anything waives the entry-level claims. --- *)
  let recorded = Option.map (List.hd (placement_keys ())) ~f:(fun k -> (k, read_entry k)) in
  let snapshot () =
    Option.map recorded ~f:(fun (k, _) -> SC.sexp_of_placement_entry (read_entry k))
  in
  let before = snapshot () in
  (* Whichever run recorded it -- run 1 under a clean load, else the re-tune that run 2 or 2b fell
     back to -- the entry the store holds is one a fresh lowering of its decision reproduces
     (gh-ocannl-1022). Run 1's claims above are waived exactly when run 1 recorded nothing; this one
     only when no run did. *)
  p "the entry the store holds is reproduced by a fresh lowering of its decision"
    (match recorded with
    | None -> true
    | Some (_, e) ->
        String.equal e.SC.outcome_digest
          (Train.placement_outcome_digest (Context.auto ()) t2 comp Ir.Indexing.Empty e.SC.decision));
  let arms3, _, shipped3, _, got3, _ = run ~ship_arm:Train.Force_arm_b () in
  p_all2 "the forced run's routine computes the right values" got3 expected ~f:approx;
  p "a forced arm searches both arms whatever the store holds" (List.length arms3 = 2);
  p "a forced arm ships the forced arm" (String.equal shipped3 "B");
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
  p_all2 "the bypassed-store run's routine computes the right values" got3b expected ~f:approx;
  p "a bypassed store compares both arms, in position, whatever the store holds"
    (List.length arms3b = 2);
  (* Waived where run 1 could not cache (gh-ocannl-1043: its own context's identity), and on a
     refused store or lookup in between. *)
  let label =
    "a bypassed store's arms replay the schedule cache, whenever the cold run cached them"
  in
  if refused_in [ persisted1; persisted3b ] label then
    skipped ~aggregation:`Environment ~backend:refused_on label
  else p_all label arms3b ~f:(fun r -> (not (persisted1.identity && clean1)) || replayed r);
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
  List.iter stale ~f:(fun (how, corrupt, overwritten) ->
      (* The corruption is a store too, and a refused one leaves the valid entry in place. *)
      let (), corrupting =
        SC.recording_cache_io (fun () ->
            Option.iter recorded ~f:(fun (key, e) ->
                SC.store_placements ~dir:cache_dir ~key:(Some key) (corrupt e)))
      in
      let corrupting = { identity = true; refusals = refusals corrupting } in
      let arms, _, _, _, got, persisted = run () in
      p_all2
        (Printf.sprintf "after an entry that %s, the routine computes the right values" how)
        got expected ~f:approx;
      unless_refused [ corrupting ]
        (Printf.sprintf "an entry that %s is ignored and the placements re-tuned" how)
        (Option.is_none recorded || List.length arms = 2);
      let clean =
        List.for_all arms ~f:(fun r ->
            completed r && uncontended r && Float.is_finite r.Autotune.best_ms)
      in
      unless_refused [ corrupting; persisted ]
        (Printf.sprintf "an entry that %s is overwritten by the re-tune, given clean evidence" how)
        ((not clean)
        || Option.value_map recorded ~default:true ~f:(fun (key, e) ->
            Option.value_map
              (SC.lookup_placements ~dir:cache_dir ~key:(Some key))
              ~default:false ~f:(overwritten e))));
  (* --- Run 6 (gh-ocannl-1022): the arms searched in a timing lineage that inherits a decision --
     the intermediate materialized -- which the caller's lineage does not. The shipped search tuned
     the TIMING lineage's lowering, so that is the digest recorded, and the replay guard recomputes
     it in the timing lineage: recomputed in the caller's, which lowers the default placements
     differently (the precondition claim), a recorded default would never replay. A new problem, so
     a new entry beside run 1's. --- *)
  let timing_ctx () = Context.decide_materialized (Context.auto ()) [ mc.Tensor.value ] in
  p "the caller's and the timing lineage lower the default placements differently"
    (not
       (String.equal
          (Train.placement_outcome_digest (Context.auto ()) t2 comp Ir.Indexing.Empty SC.Default)
          (Train.placement_outcome_digest ~timing_ctx:(timing_ctx ()) (Context.auto ()) t2 comp
             Ir.Indexing.Empty SC.Default)));
  let keys_before6 = placement_keys () in
  let arms6, flips6, shipped6, _, got6, persisted6 = run ~timing_ctx:(timing_ctx ()) () in
  p_all2 "with a timing context, the cold run's routine computes the right values" got6 expected
    ~f:approx;
  let clean6 =
    List.for_all (arms6 @ flips6) ~f:(fun r ->
        completed r && uncontended r && Float.is_finite r.Autotune.best_ms)
  in
  let entry6 =
    List.find_map (placement_keys ()) ~f:(fun k ->
        if List.mem keys_before6 k ~equal:String.equal then None else Some (read_entry k))
  in
  Stdio.eprintf "run 6 (not part of the golden): shipped %s, clean %b, stored %b\n%!" shipped6
    clean6 (Option.is_some entry6);
  if persisted6.identity then
    unless_refused [ persisted6 ]
      "with a timing context, the cold run records a decision of its own, whenever its evidence \
       was clean"
      ((not clean6) || Option.is_some entry6)
  else
    skipped ~aggregation:`Environment ~backend:(Context.backend_name ctx_ref)
      "with a timing context, the cold run records a decision of its own, whenever its evidence \
       was clean";
  let shipped_searches6 =
    match (shipped6, arms6) with "A", [ ra; _ ] -> [ ra ] | "B", [ _; rb ] -> [ rb ] | _ -> flips6
  in
  p "with a timing context, the recorded outcome digest is the shipped search's own source digest"
    (match entry6 with
    | None -> true
    | Some e ->
        List.exists shipped_searches6 ~f:(fun r ->
            String.equal (source_digest r) e.SC.outcome_digest));
  p
    "with a timing context, a fresh lowering of the recorded decision in the timing lineage \
     reproduces it"
    (match entry6 with
    | None -> true
    | Some e ->
        String.equal e.SC.outcome_digest
          (Train.placement_outcome_digest ~timing_ctx:(timing_ctx ()) (Context.auto ()) t2 comp
             Ir.Indexing.Empty e.SC.decision));
  let arms6w, flips6w, shipped6w, _, got6w, persisted6w = run ~timing_ctx:(timing_ctx ()) () in
  p_all2 "with a timing context, the warm run's routine computes the right values" got6w expected
    ~f:approx;
  (* "No flips" is an absence: an empty flip-report list is the passing case, as in run 2. *)
  unless_refused [ persisted6w ]
    "with a timing context, the warm run replays the recorded decision: one search, no flips, the \
     recorded label, tuning the recorded lowering"
    (match (entry6, arms6w) with
    | None, _ -> true
    | Some e, [ r ] ->
        List.length flips6w = 0
        && String.equal shipped6w (SC.shipped_label e.SC.decision)
        && String.equal (source_digest r) e.SC.outcome_digest
    | Some _, _ -> false);
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
  p_all2 "under refused commits, the cold run's routine computes the right values" got8 expected
    ~f:approx;
  (* Over the arms the run tuned: the decision it reached is what must have left no entry. *)
  p_empty "refused commits leave no decision recorded" ~over:arms8
    (placement_keys ~dir:refused_cache_dir ());
  (* The recorder's own rule ([Train.tune_placements]'s [persist]), which is looser than [clean]
     above: an abandoned flip search is no terminal failure, and this run's flip is often one. *)
  let clean8 =
    List.for_all (arms8 @ flips8) ~f:(fun r ->
        r.Autotune.timings_contended = 0
        && Option.is_none (Autotune.terminal_failure r)
        && Float.is_finite r.Autotune.best_ms)
  in
  Stdio.eprintf "run 8 (not part of the golden): clean %b, %d refused cache operations\n%!" clean8
    (List.length persisted8.refusals);
  gated ~aggregation:`Environment ~when_:persisted8.identity ~on:"no concrete timing identity"
    "under refused commits, a clean cold run reports its placement store's refusal"
    ((not clean8)
    || List.exists persisted8.refusals ~f:(fun (key, reason) ->
        String.is_prefix key ~prefix:"placements-"
        && String.is_substring reason ~substring:"ps: injected commit refusal"))
