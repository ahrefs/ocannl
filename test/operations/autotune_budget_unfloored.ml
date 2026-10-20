(* gh-ocannl-1409: a budget-cut search that never measured the untuned default. With automatic
   scheduling on and [schedule_fission=false], no candidate reproduces the untuned default
   ([Autotune.default_seed_label] is [None]) and cc has no seeded plan, so nothing is exempt from
   the budget. A budget spent before the first candidate then leaves the serial baseline as the only
   timing; crowning it could ship and cache code slower than not tuning, so the search ships the
   untuned default and stores nothing. [schedule_fission] is read once per process, hence a process
   of its own (the cut, the floor and the replay rules are [autotune_budget]'s).

   The virtual clock and scripted timings are [autotune_budget]'s ([Scripted_search]'s seams): the
   attempts so far, the baseline first, and a baseline timed fastest, so a search allowed to crown
   it would. *)

open Base
open Verdict.Claims
module Cache_plant = Test_utils.Cache_plant

let budget_key = "autotune_budget_seconds"
let () = Hashtbl.set Utils.config_file_args ~key:"schedule_fission" ~data:"false"
let backend_name = Context.backend_name (Context.cpu ())
let chain = Fissioned_chain.chain_with_output ()
let reference = Scripted_search.untuned (fst chain) (snd chain)
let cache_dir = "autotune_cache_budget_unfloored"

(* An earlier run's entry would turn the cold search into a replay. *)
let () = Scripted_search.clear_cache_dir cache_dir

(* One search under a budget of one attempt: the baseline's. *)
let search ~tag =
  let qg, comp = chain in
  let (o : Scripted_search.observed) =
    Scripted_search.search
      ~settings:[ (budget_key, "1") ]
      ~attempt_clock:true ~output:qg ~cache_dir ~tag
      ~measure:(fun w -> if String.equal w.label "baseline" then 1. else 50.)
      comp
  in
  let r = o.report in
  Stdio.eprintf "%s (not part of the golden): %s, budget_cut %b, attempts %s, crown %s\n%!" tag
    (Autotune.outcome_name r.Autotune.outcome)
    r.Autotune.budget_cut
    (String.concat ~sep:"; " o.attempts)
    r.Autotune.best_label;
  o

let () =
  p "setup: no candidate reproduces the untuned default"
    (Option.is_none (Autotune.default_seed_label ~backend_name));
  let cold = search ~tag:"cold" in
  let r = cold.report and attempts = cold.attempts and output = cold.output in
  p "cold: the budget cut the search before any candidate"
    ((match r.Autotune.outcome with Autotune.Searched -> true | _ -> false)
    && r.Autotune.budget_cut
    && List.equal String.equal attempts [ "baseline" ]);
  (* It reports what it shipped: an untimed untuned default, not the baseline it timed. *)
  p "cold: the report credits no timed crown"
    (Float.is_inf r.Autotune.best_ms && String.equal r.Autotune.best_label "");
  p_empty "cold: the report keeps no best-so-far record of the discarded candidates" ~over:attempts
    r.Autotune.best_steps;
  p_all2 "cold: the shipped routine computes the untuned output bit for bit" output reference
    ~f:Float.equal;
  (* Stored, the baseline crown would replay here. *)
  let warm = search ~tag:"warm" in
  Cache_plant.unless_unkeyed_or_refused ~identity:warm.identity warm.ios
    "warm: nothing was stored, so the next budgeted call searches again"
    (match warm.report.Autotune.outcome with Autotune.Searched -> true | _ -> false)
