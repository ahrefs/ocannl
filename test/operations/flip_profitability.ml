(* gh-ocannl-579: the profitability term of the flip chain's enablement promotion, driven with
   synthetic search reports.

   The enablement prior (gh-ocannl-514, from the gh-ocannl-558 lesson) promotes a flip because it
   makes a sketch family EXPRESSIBLE, and prices nothing about whether that family pays. On gh-514's
   metal/f16 mlp_wide cell (benchmarks/report-gh514-eval.md, cells C) that promotion took budget
   slots 1-2 of a budget-5 chain for a family whose candidates timed 79-92 ms against the arm's 7.5
   ms, pushing the cheap `inline n32_relu.grad` flip — rank 5 under cost ordering, and the actual
   winner — out of the budget: cost-ordered chains shipped 6.55/6.64 ms, enablement-ordered ones
   7.03-7.14 ms.

   The evidence that settles it is already paid for when the chain starts: [Train.tune_placements]
   searches arm B, the all-materialized specialization the prior derives its enablement set FROM,
   before walking the chain. This test drives that derivation and the ordering it selects with the
   three boxes' measured numbers, plus the two controls that matter — a family that plausibly pays,
   and a family nobody timed — since a term that demoted those would have deleted enablement rather
   than weighed it.

   Pure functions over records: no context, no compile, no device. *)

open Base
open Verdict.Claims

(* A completed search that timed [mma_timed] tensorized candidates, the best of them at
   [mma_best_ms], with the search's overall best at [best_ms]. The objective is the default one and
   nothing here depends on it: profitability is a RATIO of two times from one search, so it is the
   one quantity that reads the same under either (see [Autotune.report.timing]). *)
let searched ~best_ms ~mma_timed ~mma_best_ms =
  {
    (Autotune.no_search_report ~timing:Autotune.Queued) with
    Autotune.best_ms;
    mma_timed;
    mma_best_ms;
    mma_candidates = 16;
    outcome = Autotune.Searched;
  }

(* An arm that seeded tensorized candidates and timed none of them (the gh-ocannl-521 state): a fact
   about candidate compilation, not about the family's speed. *)
let seeded_none ~best_ms = searched ~best_ms ~mma_timed:0 ~mma_best_ms:Float.infinity

(* A beam round appending a [Tensorize] to a saved or preset incumbent: the candidate promises
   nothing in its LABEL, so [mma_timed] stays zero, while [mma_best_ms] records its structural
   measurement. The family was measured; reading the label counter instead would keep the prior
   standing against a family that lost tenfold. *)
let beam_appended ~best_ms ~mma_best_ms = searched ~best_ms ~mma_timed:0 ~mma_best_ms

let profit_name = function
  | Autotune.Unmeasured -> "unmeasured"
  | Autotune.Pays _ -> "pays"
  | Autotune.Loses _ -> "loses"

let ratio = function Autotune.Unmeasured -> Float.nan | Autotune.Pays r | Autotune.Loses r -> r

(* No tie values among these ratios, so decimal rounding is portable. *)
let show_ratio r = if Float.is_nan r then "--" else Printf.sprintf "%.3f" r
let ordering_name = function `Cost -> "cost" | `Enablement -> "enablement"
let resolves profit = Autotune.effective_flip_ordering ~ordering:`Profitable ~profit

(* The three gh-514 phase-6 cells, as their arm reports. Arm A is the default-placement arm: on
   metal the reduced-precision cast twins are virtual there, so the site reads f32 masters and no
   tensorized candidate exists at all. *)
let metal = [ seeded_none ~best_ms:7.5; searched ~best_ms:7.5 ~mma_timed:3 ~mma_best_ms:92.0 ]
let cuda = [ seeded_none ~best_ms:4.4; searched ~best_ms:4.4 ~mma_timed:16 ~mma_best_ms:10.2 ]

(* On hip the arm's own winner IS tensorized: [mma_best_ms] equals [best_ms]. *)
let hip = [ seeded_none ~best_ms:2.05; searched ~best_ms:1.28 ~mma_timed:16 ~mma_best_ms:1.28 ]

let () =
  let cells =
    [ ("metal f16", metal); ("cuda f16", cuda); ("hip bf16", hip) ]
    |> List.map ~f:(fun (name, reports) -> (name, Autotune.family_profit_of_reports reports))
  in
  Stdio.printf "%-12s %-11s %-8s %s\n" "cell" "evidence" "ratio" "ranks the surface by";
  List.iter cells ~f:(fun (name, profit) ->
      Stdio.printf "%-12s %-11s %-8s %s\n" name (profit_name profit)
        (show_ratio (ratio profit))
        (ordering_name (resolves profit)));
  let profit_of name = List.Assoc.find_exn cells name ~equal:String.equal in
  (* The displacement case: measured to lose by ~12x, so the prior is void here. *)
  p "metal's cell is measured out of profit"
    (match profit_of "metal f16" with Autotune.Loses _ -> true | _ -> false);
  p "metal's losing family ranks the surface by cost"
    (match resolves (profit_of "metal f16") with `Cost -> true | `Enablement -> false);
  p "cuda's cell is measured out of profit too"
    (match resolves (profit_of "cuda f16") with `Cost -> true | `Enablement -> false);
  (* The negative control gh-558 closed on: the family IS the arm's winner there, so the prior — and
     with it the budget-5 reachability the enablement ordering bought — stands. *)
  p "hip's winning family keeps the enablement prior"
    (match profit_of "hip bf16" with Autotune.Pays _ -> true | _ -> false);
  p "hip's paying family ranks the surface by enablement"
    (match resolves (profit_of "hip bf16") with `Enablement -> true | `Cost -> false);
  (* The other control: a family that lost by a hair could be won back by one more placement flip,
     so it stays promoted. Only a loss beyond the margin voids the prior. *)
  let hair =
    Autotune.family_profit_of_reports [ searched ~best_ms:7.5 ~mma_timed:3 ~mma_best_ms:7.6 ]
  in
  p "a family that lost by 1% is still within profit"
    (match hair with Autotune.Pays _ -> true | _ -> false);
  p "a family that lost by 1% keeps the enablement prior"
    (match resolves hair with `Enablement -> true | `Cost -> false);
  (* Absence of a confirmation is not evidence against: an arm that seeded tensorized candidates and
     timed none measured nothing about the family. *)
  let unmeasured = Autotune.family_profit_of_reports [ seeded_none ~best_ms:7.5 ] in
  p "an arm that timed no tensorized candidate measures nothing"
    (match unmeasured with Autotune.Unmeasured -> true | _ -> false);
  p "unmeasured keeps the enablement prior"
    (match resolves unmeasured with `Enablement -> true | `Cost -> false);
  let contention_affected =
    { (searched ~best_ms:1.0 ~mma_timed:4 ~mma_best_ms:0.5) with Autotune.timings_contended = 1 }
  in
  p "an incomplete contention-affected search contributes no profitability evidence"
    (match Autotune.family_profit_of_reports [ contention_affected ] with
    | Autotune.Unmeasured -> true
    | _ -> false);
  (* The verdict must not depend on cache state. A warm run's report comes from the replay path,
     whose COUNTERS describe this call (all zero, nothing was timed here) while its TIMES are the
     storing search's — so it must reach the verdict that search reached, or the same workload would
     rank its decision surface one way cold and the other way warm. *)
  let replayed ~best_ms ~mma_best_ms =
    {
      (Autotune.no_search_report ~timing:Autotune.Queued) with
      Autotune.outcome = Autotune.Cache_replay;
      best_ms;
      mma_best_ms;
    }
  in
  p "a cache replay reaches the verdict of the search that stored it"
    (Float.equal
       (ratio (Autotune.family_profit_of_reports [ replayed ~best_ms:7.5 ~mma_best_ms:92.0 ]))
       (ratio (profit_of "metal f16")));
  p "and a replay of a search that timed none still measures nothing"
    (match
       Autotune.family_profit_of_reports [ replayed ~best_ms:7.5 ~mma_best_ms:Float.infinity ]
     with
    | Autotune.Unmeasured -> true
    | _ -> false);
  (* Which the cache entry carries as an OPTIONAL field, so entries written before it stay readable
     without an [entry_version] bump — and replay as "nothing is known" rather than as a verdict. *)
  let round_trip e = Ir.Schedule_cache.entry_of_sexp (Ir.Schedule_cache.sexp_of_entry e) in
  let entry mma_best_ms =
    {
      Ir.Schedule_cache.version = Ir.Schedule_cache.entry_version;
      backend = "metal";
      numerics = "n";
      codegen = None;
      objective = None;
      source_digest = "d";
      saved = [];
      segments = None;
      best_ms = 7.5;
      baseline_ms = 9.0;
      mma_best_ms;
      default_ms = None;
      default_fingerprint = None;
      best_steps = None;
    }
  in
  p "a stored tensorized best survives the entry round-trip"
    (match (round_trip (entry (Some 92.0))).Ir.Schedule_cache.mma_best_ms with
    | Some v -> Float.equal v 92.0
    | None -> false);
  p "an entry without the field is readable and claims nothing"
    (Option.is_none (round_trip (entry None)).Ir.Schedule_cache.mma_best_ms);
  p "omitting it keeps the entry's sexp free of the field"
    (not
       (String.is_substring
          (Sexp.to_string (Ir.Schedule_cache.sexp_of_entry (entry None)))
          ~substring:"mma_best_ms"));
  (* The label counter and the structural best are different populations. *)
  let beam_only =
    Autotune.family_profit_of_reports [ beam_appended ~best_ms:7.5 ~mma_best_ms:92.0 ]
  in
  p "a beam-appended Tensorize measures the family though no label promised one"
    (match beam_only with Autotune.Loses _ -> true | _ -> false);
  p "and its losing measurement voids the prior"
    (match resolves beam_only with `Cost -> true | `Enablement -> false);
  p "a report that never searched measures nothing"
    (match
       Autotune.family_profit_of_reports [ Autotune.no_search_report ~timing:Autotune.Queued ]
     with
    | Autotune.Unmeasured -> true
    | _ -> false);
  (* Over several arms the most favourable evidence wins: the arms search different placements, and
     the promotion is a bet on the best placement the chain can reach. *)
  let mixed =
    Autotune.family_profit_of_reports
      [
        searched ~best_ms:7.5 ~mma_timed:3 ~mma_best_ms:92.0;
        searched ~best_ms:1.28 ~mma_timed:16 ~mma_best_ms:1.28;
      ]
  in
  p "one arm's paying family outweighs another's losing one"
    (match mixed with Autotune.Pays _ -> true | _ -> false);
  p "two losing arms report the more favourable ratio"
    (Float.equal
       (ratio
          (Autotune.family_profit_of_reports
             [
               searched ~best_ms:7.5 ~mma_timed:3 ~mma_best_ms:92.0;
               searched ~best_ms:4.4 ~mma_timed:16 ~mma_best_ms:10.2;
             ]))
       (ratio (profit_of "cuda f16")));
  (* The margin is configuration, and an unusable one fails the run rather than quietly becoming the
     default: a run that asked for a profitability policy it also made impossible should not get a
     different policy without saying so. A ratio below 1.0 would demote a family that WON. *)
  let rejects raw =
    match Autotune.flip_profit_margin_of_string raw with
    | _ -> false
    | exception Utils.User_error _ -> true
  in
  p "a well-formed margin parses"
    (Float.equal (Autotune.flip_profit_margin_of_string " 1.25 ") 1.25);
  p "a margin below 1.0 is rejected" (rejects "0.5");
  p "an unparseable margin is rejected" (rejects "generous");
  p "an empty margin is rejected" (rejects "");
  p "a non-finite margin is rejected" (rejects "inf");
  (* The margin is the knob, and it is the only thing that moves these verdicts. *)
  p "a margin above metal's ratio re-admits its family"
    (match Autotune.family_profit_of_reports ~margin:20.0 metal with
    | Autotune.Pays _ -> true
    | _ -> false);
  p "a margin below hip's ratio is not reachable (its family won outright)"
    (Float.( <= ) (ratio (profit_of "hip bf16")) 1.0);
  (* The two configured orderings are unconditional: they are the evaluation baselines the gh-514
     report's cells C were measured under, and a term that moved them would make those cells
     irreproducible. *)
  List.iter
    [
      ("unmeasured", unmeasured); ("paying", profit_of "hip bf16"); ("losing", profit_of "metal f16");
    ]
    ~f:(fun (name, profit) ->
      p
        (Printf.sprintf "ordering=cost stays cost under %s evidence" name)
        (match Autotune.effective_flip_ordering ~ordering:`Cost ~profit with
        | `Cost -> true
        | `Enablement -> false);
      p
        (Printf.sprintf "ordering=enablement stays enablement under %s evidence" name)
        (match Autotune.effective_flip_ordering ~ordering:`Enablement ~profit with
        | `Enablement -> true
        | `Cost -> false))
