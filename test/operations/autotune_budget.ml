(* gh-ocannl-1409: a search's wall-clock budget ([autotune_budget_seconds]). Once spent, the search
   compiles nothing more beyond the untuned floors and crowns its incumbent; it stores that crown
   marked as cut, which a later call with a budget replays and a call without one declines,
   searching in full and overwriting it.

   A wall clock cannot place a cut at a chosen candidate, so every search here reads a virtual clock
   through [Autotune.budget_elapsed]: the attempts so far, the baseline's included. A budget of [n]
   "seconds" then lets the first [n] attempts start. The timings are scripted
   ([Autotune.on_candidate_measured]) so which candidate wins is the script's choice, not the
   machine's; the crowned routines are still compiled, run and compared with the untuned output. The
   seams are [Scripted_search]'s. Searched on cc, over [Fissioned_chain]'s small menu so the one
   full search here stays cheap. *)

open Base
open Verdict.Claims
module Cache_plant = Test_utils.Cache_plant

let () = Fissioned_chain.small_menu ()
let budget_key = "autotune_budget_seconds"
let backend_name = Context.backend_name (Context.cpu ())

(* The untuned floor a cc search attempts whatever its budget: the default-pipeline seed. *)
let default_label = Autotune.default_seed_label ~backend_name
let is_floor label = Option.exists default_label ~f:(String.equal label)

(* The baseline slowest, the untuned default next, every other candidate a distinct faster time. *)
let cost label =
  if String.equal label "baseline" then 1000.
  else if is_floor label then 100.
  else 2. +. (Float.of_int (Stdlib.Hashtbl.hash label % 1000) /. 100.)

let chain = Fissioned_chain.chain_with_output ()
let reference = Scripted_search.untuned (fst chain) (snd chain)

(* One search, with [budget] stated as [autotune_budget_seconds] for it alone (none when absent). *)
let search ?budget ?(rounds = 0) ~cache_dir ~tag () =
  let qg, comp = chain in
  let (o : Scripted_search.observed) =
    Scripted_search.search
      ~settings:(Option.value_map budget ~default:[] ~f:(fun data -> [ (budget_key, data) ]))
      ~attempt_clock:true ~output:qg ~rounds ~cache_dir ~tag
      ~measure:(fun w -> cost w.label)
      comp
  in
  Stdio.eprintf "%s attempts (not part of the golden): %s | measured: %s | default %s\n%!" tag
    (String.concat ~sep:"; " o.attempts)
    (String.concat ~sep:"; " o.measured)
    (Option.value default_label ~default:"none");
  Stdio.eprintf
    "%s (not part of the golden): %s, budget_cut %b, %d attempts, timed %d, rounds %d, crown %s \
     %.2f ms\n\
     %!"
    tag
    (Autotune.outcome_name o.report.Autotune.outcome)
    o.report.Autotune.budget_cut (List.length o.attempts) o.report.Autotune.candidates_timed
    o.report.Autotune.rounds_run o.report.Autotune.best_label o.report.Autotune.best_ms;
  o

let cut_cache_dir = "autotune_cache_budget_cut"
let replay_cache_dir = "autotune_cache_budget_replay"
let floor_cache_dir = "autotune_cache_budget_floor"

let () =
  List.iter [ cut_cache_dir; replay_cache_dir; floor_cache_dir ] ~f:Scripted_search.clear_cache_dir

let searched (o : Scripted_search.observed) =
  match o.report.Autotune.outcome with Autotune.Searched -> true | _ -> false

let replayed (o : Scripted_search.observed) =
  match o.report.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let fastest_measured (o : Scripted_search.observed) =
  List.map o.measured ~f:cost |> List.min_elt ~compare:Float.compare

(* --- the setting --- *)

let parse v = Scripted_search.with_settings [ (budget_key, v) ] Autotune.budget_setting

(* gh-ocannl-1438: the budget a report carries is the one its call resolved from the setting. *)
let carries ?budget (o : Scripted_search.observed) =
  Option.equal Float.equal o.report.Autotune.budget_s (Option.bind budget ~f:parse)

let () =
  p "setting: a positive number of seconds is a budget"
    (Option.equal Float.equal (parse "2.5") (Some 2.5));
  p_all "setting: empty, zero and negative mean no budget" [ ""; "0"; "-3" ] ~f:(fun v ->
      Option.is_none (parse v));
  (* gh-ocannl-1332: a registered key's unparsable value is refused, naming the key. *)
  p "setting: unparsable text is refused, naming the key"
    (match parse "soon" with
    | _ -> false
    | exception Utils.Invalid_config_value { key; _ } -> String.equal key budget_key)

(* --- the unmeasured-floor rule, case by case (the executed case is autotune_budget_unfloored)
   --- *)

let () =
  let unmeasured ~budget_cut ~auto_sched ~baseline_dispatched ~floor_measured =
    Autotune.budget_floor_unmeasured ~budget_cut ~auto_sched ~baseline_dispatched ~floor_measured
  in
  p "floor rule: a measured floor keeps the crown"
    (not
       (unmeasured ~budget_cut:true ~auto_sched:true ~baseline_dispatched:true ~floor_measured:true));
  p "floor rule: an uncut search keeps its crown"
    (not
       (unmeasured ~budget_cut:false ~auto_sched:true ~baseline_dispatched:true
          ~floor_measured:false));
  p "floor rule: an unmeasured scheduled default ships the default"
    (unmeasured ~budget_cut:true ~auto_sched:true ~baseline_dispatched:false ~floor_measured:false);
  (* Automatic scheduling off: the default is the serial baseline, whose window can be refused. *)
  p "floor rule: an unmeasured dispatched serial default ships the default"
    (unmeasured ~budget_cut:true ~auto_sched:false ~baseline_dispatched:true ~floor_measured:false);
  p "floor rule: an undispatched serial default (GPU, scheduling off) keeps the crown"
    (not
       (unmeasured ~budget_cut:true ~auto_sched:false ~baseline_dispatched:false
          ~floor_measured:false))

(* --- cut: the budget lets four attempts start --- *)

let cut = search ~budget:"4" ~cache_dir:cut_cache_dir ~tag:"cut" ()

(* The same search with no budget, over the cut search's directory: it declines the cut crown and
   searches in full, so it is also the reference the cut search's attempts are a prefix of. *)
let full = search ~cache_dir:cut_cache_dir ~tag:"full" ()

let () =
  let r = cut.report in
  p "cut: the search completed, cut by its budget" (searched cut && r.Autotune.budget_cut);
  p "cut: its attempts are the full search's first four"
    (List.length cut.attempts = 4
    && List.equal String.equal cut.attempts (List.take full.attempts 4));
  p "cut: the full search attempted more" (List.length full.attempts > 4);
  p "cut: the crown is the fastest window it timed"
    (Option.exists (fastest_measured cut) ~f:(Float.equal r.Autotune.best_ms));
  p "cut: the crown is no slower than the untuned default it timed"
    (Option.exists r.Autotune.default_ms ~f:(fun d -> Float.(r.Autotune.best_ms <= d)));
  p_all2 "cut: the crowned routine computes the untuned output bit for bit" cut.output reference
    ~f:Float.equal;
  p "cut: its report carries the budget it was cut against" (carries ~budget:"4" cut);
  p "full: no budget, so the cut crown was not replayed and nothing was cut"
    (searched full && not full.report.Autotune.budget_cut);
  p "full: its report carries no budget" (carries full);
  p_all2 "full: the crowned routine computes the untuned output bit for bit" full.output reference
    ~f:Float.equal

(* --- replays: a budgeted call replays whatever crown is stored --- *)

let () =
  (* Before [full] overwrote the entry, a budgeted call would have replayed the cut crown: check
     that on a directory holding only the cut search's entry. *)
  let cut2 = search ~budget:"4" ~cache_dir:replay_cache_dir ~tag:"cut2" () in
  let warm = search ~budget:"1000" ~cache_dir:replay_cache_dir ~tag:"warm" () in
  let ios = cut2.ios @ warm.ios in
  Cache_plant.unless_unkeyed_or_refused ~identity:warm.identity ios
    "warm: a budgeted call replays the cut crown, and says it was cut"
    (replayed warm && warm.report.Autotune.budget_cut
    && Float.equal warm.report.Autotune.best_ms cut2.report.Autotune.best_ms);
  p_all2 "warm: the replayed routine computes the untuned output bit for bit" warm.output reference
    ~f:Float.equal;
  (* The budget is the call's, the cut the crown's: the replay names its own budget, not the one the
     crown was cut against. *)
  p "warm: a replay's report carries its own call's budget" (carries ~budget:"1000" warm);
  (* [full] stored over the cut crown in [cut_cache_dir]: what a budgeted call replays now is
     uncut. *)
  let after = search ~budget:"1000" ~cache_dir:cut_cache_dir ~tag:"after" () in
  Cache_plant.unless_unkeyed_or_refused ~identity:after.identity (full.ios @ after.ios)
    "after: the full search's store replaced the cut crown"
    (replayed after
    && (not after.report.Autotune.budget_cut)
    && Float.equal after.report.Autotune.best_ms full.report.Autotune.best_ms);
  Cache_plant.not_every_store_refused ~expected:cut.identity "stores: not every store was refused"
    (cut.ios @ full.ios @ cut2.ios)

(* --- floor: the budget is spent before the first candidate --- *)

let () =
  let o = search ~budget:"1" ~rounds:1 ~cache_dir:floor_cache_dir ~tag:"floor" () in
  let r = o.report in
  match default_label with
  | Some label ->
      p "floor: only the baseline and the untuned default were attempted"
        (List.equal String.equal o.attempts [ "baseline"; label ]);
      p "floor: the full search attempted another candidate before the untuned default"
        (match List.nth full.attempts 1 with Some l -> not (String.equal l label) | None -> false);
      p "floor: the search was cut and ran no beam round"
        (searched o && r.Autotune.budget_cut && r.Autotune.rounds_run = 0);
      (* On this small chain cc's default pipeline leaves the code serial, so the default seed
         dedups against the baseline and its time is the baseline's: the crown is that code. *)
      p "floor: the crown is the untuned default's time"
        (Option.exists r.Autotune.default_ms ~f:(Float.equal r.Autotune.best_ms));
      p_all2 "floor: the crowned routine computes the untuned output bit for bit" o.output reference
        ~f:Float.equal
  | None -> p "floor: the cc search attributes an untuned default seed" false
