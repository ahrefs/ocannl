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

   Times never enter the golden; the claims that involve one are waived by the load's own evidence
   (contention-refused windows), as autotune_arm_containment.ml does. *)

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
let placement_keys () =
  if not (Stdlib.Sys.file_exists cache_dir) then []
  else
    Stdlib.Sys.readdir cache_dir |> Array.to_list
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
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ label ^ "_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ label ^ "_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  (mc, t2, Train.forward t2)

let problem_digest ctx loss comp =
  SC.digest (Train.placement_problem ctx loss comp Ir.Indexing.Empty)

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
    let arms = ref [] and flips = ref [] and shipped = ref None in
    let ctx_t, routine_t =
      Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ~inline_flips:2
        ~report:(fun r -> arms := r :: !arms)
        ~flip_report:(fun r -> flips := r :: !flips)
        ~on_ship:(fun what -> shipped := Some what)
        ?ship_arm ?placement_store ?timing_ctx (Context.auto ()) t2 comp Ir.Indexing.Empty
    in
    let ctx_t = Context.run ctx_t routine_t in
    let got = Context.get_values ctx_t t2.Tensor.value in
    let materialized =
      Tn.Placements.is_materialized_peek (Context.placements ctx_t) mc.Tensor.value
    in
    (List.rev !arms, List.rev !flips, Option.value_exn !shipped, materialized, got)
  in
  (* --- Run 1: cold. --- *)
  let arms1, flips1, shipped1, materialized1, got1 = run () in
  p "the cold run reports both arms in position" (List.length arms1 = 2);
  p_all2 "the cold run's routine computes the right values" got1 expected ~f:approx;
  let cache_available = Option.is_some (Context.timing_identity ctx_ref) in
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
  if cache_available then
    p "the cold run records exactly one decision, whenever its evidence was clean"
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
  let arms2, flips2, shipped2, materialized2, got2 = run () in
  p_all2 "the warm run's routine computes the right values" got2 expected ~f:approx;
  p "the warm run replays the decision exactly when the cold run recorded one: one search, no flips"
    (if stored1 then List.length arms2 = 1 && List.length flips2 = 0 else List.length arms2 = 2);
  p "a replay ships the recorded label" ((not stored1) || String.equal shipped2 shipped1);
  p "a replay ships the recorded placement of the intermediate"
    ((not stored1) || Bool.equal materialized2 materialized1);
  (* The recorded decision reproduces the lowering the cold run tuned: the one search is a
     schedule-cache replay of the winner the cold run's shipped search crowned. Waived only by the
     cold run's shipped search having stored nothing, which under a clean run 1 it did not. *)
  p "a replay's one search is a schedule-cache replay"
    ((not stored1) || (not clean1) || match arms2 with [ r ] -> replayed r | _ -> false);
  (* The same fact without the cache in between (gh-ocannl-1022): the replayed search tuned the very
     lowering whose digest the entry records. Not waived by contention. *)
  p "a replay's one search tunes the lowering the recorded decision was measured on"
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
  let arms2b, _, _, _, got2b =
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
  let arms3, _, shipped3, _, got3 = run ~ship_arm:Train.Force_arm_b () in
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
  let arms3b, _, _, _, got3b = run ~placement_store:false () in
  p_all2 "the bypassed-store run's routine computes the right values" got3b expected ~f:approx;
  p "a bypassed store compares both arms, in position, whatever the store holds"
    (List.length arms3b = 2);
  p_all "a bypassed store's arms replay the schedule cache, whenever the cold run cached them"
    arms3b ~f:(fun r -> (not (cache_available && clean1)) || replayed r);
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
      Option.iter recorded ~f:(fun (key, e) ->
          SC.store_placements ~dir:cache_dir ~key:(Some key) (corrupt e));
      let arms, _, _, _, got = run () in
      p_all2
        (Printf.sprintf "after an entry that %s, the routine computes the right values" how)
        got expected ~f:approx;
      p
        (Printf.sprintf "an entry that %s is ignored and the placements re-tuned" how)
        (Option.is_none recorded || List.length arms = 2);
      let clean =
        List.for_all arms ~f:(fun r ->
            completed r && uncontended r && Float.is_finite r.Autotune.best_ms)
      in
      p
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
  let arms6, flips6, shipped6, _, got6 = run ~timing_ctx:(timing_ctx ()) () in
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
  if cache_available then
    p
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
  let arms6w, flips6w, shipped6w, _, got6w = run ~timing_ctx:(timing_ctx ()) () in
  p_all2 "with a timing context, the warm run's routine computes the right values" got6w expected
    ~f:approx;
  p
    "with a timing context, the warm run replays the recorded decision: one search, no flips, the \
     recorded label, tuning the recorded lowering"
    (match (entry6, arms6w) with
    | None, _ -> true
    | Some e, [ r ] ->
        List.is_empty flips6w
        && String.equal shipped6w (SC.shipped_label e.SC.decision)
        && String.equal (source_digest r) e.SC.outcome_digest
    | Some _, _ -> false)
