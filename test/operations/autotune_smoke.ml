(* Autotune (docs/proposals/schedule-ir-optops.md, the search-harness half of the OptOps port):
   canonical schedule identities and the empirical schedule search.

   Covered here, backend-independent (every printed verdict holds on every backend):

   - Canonical digests are process-stable across sibling compiles of the same computation (each
   backend [compile] re-lowers with fresh symbols), and distinguish different computations. - A
   schedule serialized against one compile ([Schedule_cache.to_saved]) replays against another
   ([of_saved]) with structural fidelity: the replayed transform produces code with the same
   canonical digest as the original transform, and the executed values match the unscheduled twin. -
   [Autotune.tune] returns a working routine whose values match the unscheduled twin, reports a
   cache miss on the first call and a cache hit on the second (same computation, fresh context), and
   the cached winner replays to correct values. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments
open Verdict.Claims

(* The report's outcome as the questions this test asks of it (gh-ocannl-677): the outcome is a
   variant naming one of five mutually exclusive states, so a claim names the state it means. In
   particular [not (replayed r)] is NOT "a search ran" — that mis-derivation is what the variant
   exists to stop, and the states section below pins the distinction. *)
let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let approx a b = Float.(abs (a - b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let rec first_loop (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> ( match first_loop a with Some r -> Some r | None -> first_loop b)
  | LL.For_loop { index; body; _ } -> Some (index, body)
  | _ -> None

let first_loop_exn llc = Option.value_exn ~here:[%here] (first_loop llc)

let () =
  let av = Array.init 32 ~f:(fun i -> Float.of_int i *. 0.5) in
  let bv = Array.init 32 ~f:(fun i -> Float.of_int (i % 7) -. 3.) in
  let expected_c = Array.init 32 ~f:(fun i -> av.(i) +. bv.(i)) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ 4; 8 ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ 4; 8 ] () in
  let%op c = a + b in
  let comp = named "canon_ab" (Train.forward c) in

  (* --- Digest stability across sibling compiles (fresh lowering, fresh symbols) --- *)
  let capture () =
    let captured = ref None in
    let ctx = Context.auto () in
    let _ctx, _routine =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some opt;
          [ opt ])
        ctx comp Ir.Indexing.Empty
    in
    Option.value_exn ~here:[%here] !captured
  in
  let opt1 = capture () in
  let opt2 = capture () in
  let canon1 = SC.canonicalize opt1 in
  let canon2 = SC.canonicalize opt2 in
  p "canonical digest stable across sibling compiles"
    (String.equal (SC.digest canon1) (SC.digest canon2));
  p "canonical form complete" (SC.complete canon1);
  let mav = Array.init 20 ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init 30 ~f:(fun i -> Float.of_int (i % 11) -. 4.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ 5 ] ~output_dims:[ 4 ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ 6 ] ~output_dims:[ 5 ] () in
  let%op mc = ma * mb in
  let mm_comp = named "canon_mm" (Train.forward mc) in
  let capture_mm () =
    let captured = ref None in
    let ctx = Context.auto () in
    let _ctx, _routine =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some opt;
          [ opt ])
        ctx mm_comp Ir.Indexing.Empty
    in
    Option.value_exn ~here:[%here] !captured
  in
  let mm_opt = capture_mm () in
  p "different computations have different digests"
    (not (String.equal (SC.digest canon1) (SC.digest (SC.canonicalize mm_opt))));

  (* --- Local scope ids are alpha-numbered in the digest (Codex P2 on PR #103): [get_scope] draws
     from a process-global counter, so the same schedule applied in two compiles consumes different
     raw ids. [Privatize] introduces [Declare_local]/[Set_local]/[Get_local]: without
     alpha-numbering, the digest of a fresh compile's replay could never match the digest of the
     direct application. --- *)
  let mm_expected = Array.create ~len:24 0. in
  Array.iteri mm_expected ~f:(fun idx _ ->
      let i = idx / 6 and j = idx % 6 in
      let acc = ref 0. in
      for k = 0 to 4 do
        acc := !acc +. (mav.((i * 5) + k) *. mbv.((k * 6) + j))
      done;
      mm_expected.(idx) <- !acc);
  let mm_canon = SC.canonicalize mm_opt in
  let _i, mm_b1 = first_loop_exn mm_opt.LL.llc in
  let _j, mm_b2 = first_loop_exn mm_b1 in
  let mm_k, _ = first_loop_exn mm_b2 in
  let priv_sched = [ Sched.Privatize { target = mc.Tensor.value; over = mm_k } ] in
  let priv_saved, _reg = SC.to_saved (SC.base_registry mm_canon) priv_sched in
  let priv_direct = Sched.apply priv_sched mm_opt in
  let priv_direct_digest = SC.digest (SC.canonicalize priv_direct) in
  (* Two structurally identical codes whose scope ids differ (as two lowering / transform runs
     produce, [LL.get_scope] being a process-global counter) must digest equally. *)
  let scope_tn =
    Ir.Tnode.create (Ir.Tnode.Specified Ir.Ops.single) ~id:9001 ~label:[ "scope_probe" ]
      ~unpadded_dims:(lazy [| 4 |])
      ~padding:(lazy None)
      ()
  in
  let build_scoped () =
    let s = Ir.Indexing.get_symbol () in
    let scope = LL.get_scope scope_tn in
    let fake llc : LL.optimized =
      {
        LL.traced_store = Hashtbl.create (module Ir.Tnode);
        optimize_ctx = LL.empty_optimize_ctx ();
        llc;
        merge_node = None;
        workgroup_shared = Set.empty (module Ir.Tnode);
        simdgroup_fragments = Set.empty (module Ir.Tnode);
        swizzled = Map.empty (module Ir.Tnode);
        pipelined = Map.empty (module Ir.Tnode);
        zero_fringe = Set.empty (module Ir.Tnode);
        flip_candidates = [];
        spliced_rbw = Base.Set.empty (module Ir.Tnode);
        source = llc;
      }
    in
    fake
      (LL.For_loop
         {
           index = s;
           from_ = 0;
           to_ = 3;
           axis = LL.Serial;
           body =
             LL.Seq
               ( LL.Declare_local { id = scope; needs_init = false },
                 LL.Seq
                   ( LL.Set_local (scope, LL.Constant 1.),
                     LL.Set
                       {
                         tn = scope_tn;
                         idcs = [| Ir.Indexing.Iterator s |];
                         llsc = LL.Get_local scope;
                         debug = "";
                       } ) );
         })
  in
  p "scope ids alpha-numbered in the digest"
    (String.equal
       (SC.digest (SC.canonicalize (build_scoped ())))
       (SC.digest (SC.canonicalize (build_scoped ()))));
  let priv_replay_digest = ref "" in
  let pctx = Context.auto () in
  let pctx, proutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let canon = SC.canonicalize opt in
        assert (String.equal (SC.digest canon) (SC.digest mm_canon));
        let sched, _reg = SC.of_saved canon priv_saved in
        let opt' = Sched.apply sched opt in
        priv_replay_digest := SC.digest (SC.canonicalize opt');
        [ opt' ])
      pctx mm_comp Ir.Indexing.Empty
  in
  let pctx = Context.run pctx proutine in
  let priv_got = Context.get_values pctx mc.Tensor.value in
  p_all2 "privatized replay values correct" priv_got mm_expected ~f:approx;
  p "privatized replay structurally equals the direct application"
    (String.equal !priv_replay_digest priv_direct_digest);

  (* --- to_saved / of_saved roundtrip: serialize a split+swap schedule built against compile #1,
     replay it inside a fresh compile, check executed values and structural (digest) parity --- *)
  let sym1, _ = first_loop_exn opt1.LL.llc in
  let split_op, _outer, inner =
    Sched.split ~axis:sym1 ~factor:2 ~outer:LL.Serial ~inner:LL.Serial
  in
  let sched1 = [ split_op; Sched.Unroll { axis = inner; materialize = true } ] in
  let saved, _reg = SC.to_saved (SC.base_registry canon1) sched1 in
  let direct = Sched.apply sched1 opt1 in
  let direct_digest = SC.digest (SC.canonicalize direct) in
  let replay_digest = ref "" in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let canon = SC.canonicalize opt in
        assert (String.equal (SC.digest canon) (SC.digest canon1));
        let sched, _reg = SC.of_saved canon saved in
        let opt' = Sched.apply sched opt in
        replay_digest := SC.digest (SC.canonicalize opt');
        [ opt' ])
      ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx c.Tensor.value in
  p_all2 "replayed schedule values correct" got expected_c ~f:approx;
  p "replayed schedule structurally equals the direct application"
    (String.equal !replay_digest direct_digest);
  p "saved schedule roundtrips through sexp"
    (SC.equal_saved_schedule saved (SC.saved_schedule_of_sexp (SC.sexp_of_saved_schedule saved)));

  (* --- End-to-end tune on a combo computation (an elementwise nest and a matmul nest): first call
     searches (cache miss), second call (same computation, fresh context) hits the cache; both
     return routines computing correct values --- *)
  let%op tc1 = a + b in
  let%op tc2 = ma * mb in
  let tune_comp = named "tune_combo" (Asgns.sequence [ Train.forward tc1; Train.forward tc2 ]) in
  let cache_dir = "autotune_cache_test" in
  (* The dune sandbox persists across runs: a stale entry written by an older binary (digest
     ingredients and the saved-schedule format evolve) breaks the miss-then-hit assertions below, so
     start from a clean cache. *)
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let reports = ref [] in
  let tune_once () =
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir
        ~report:(fun r -> reports := r :: !reports)
        ctx tune_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    (Context.get_values ctx tc1.Tensor.value, Context.get_values ctx tc2.Tensor.value)
  in
  let got1, got_mm1 = tune_once () in
  let got2, got_mm2 = tune_once () in
  let r2, r1 =
    match !reports with [ r2; r1 ] -> (r2, r1) | _ -> failwith "expected two reports"
  in
  p_all2 "tuned elementwise values correct" got1 expected_c ~f:approx;
  p_all2 "tuned matmul values correct" got_mm1 mm_expected ~f:approx;
  p "first tune call searches (cache miss)" (completed r1);
  (* Which candidates a search times is host-dependent (gh-ocannl-892): a timing window that is
     mostly host stalls is refused ([Autotune.admitted_timing_ms]) and grows [timings_contended]
     instead of [candidates_timed], and processes sharing one GPU refuse whole searches this small
     (the 08-31 cuda sweep ran test/operations in parallel). So a search that timed nothing passes
     only on the load's own evidence: one whose every candidate failed compile or dispatch shows no
     refusals and still fails here. The accounting goes to stderr so a red run names the numbers. *)
  Stdio.eprintf
    "search (not part of the golden): %d candidate(s) timed, %d timing(s) refused, %d candidate(s) \
     failed\n"
    r1.Autotune.candidates_timed r1.Autotune.timings_contended r1.Autotune.candidates_failed;
  p "first tune call timed at least the baseline, or was refused its timings by contention"
    (r1.Autotune.candidates_timed >= 1 || r1.Autotune.timings_contended > 0);
  p "a completed search carries no terminal failure" (Option.is_none (Autotune.terminal_failure r1));
  p "decline census sums to candidates_failed"
    (r1.Autotune.candidates_failed
    = List.sum (module Int) r1.Autotune.declines ~f:(fun d -> d.Autotune.count));
  let first_cacheable = r1.Autotune.timings_contended = 0 in
  p "second tune call replays exactly when the first search had no contention refusals"
    (Bool.equal (replayed r2) first_cacheable);
  p "a replayed second report has no declines"
    ((not (replayed r2))
    || (List.is_empty r2.Autotune.declines && r2.Autotune.candidates_failed = 0));
  (* The crowned candidate's identity in the report (gh-ocannl-546): backend-independent contract,
     since which candidate wins is not. A per-arm win that never reaches a shipping artifact is
     invisible in every end-to-end number unless the report says which candidate won and whether it
     tensorizes. *)
  let tensorized_schedule (r : Autotune.report) =
    List.exists r.Autotune.best_schedule ~f:(function SC.Tensorize _ -> true | _ -> false)
  in
  (* Why [mma_best_ms >= best_ms] is not an invariant, which is what flaked on macOS CI
     (gh-ocannl-716). The beam accepts a round only when it improves on the incumbent by at least
     [Autotune.min_progress] (0.01, private). A round that improves by LESS is rejected and the
     incumbent stays crowned — but its candidates were timed, and a tensorized one among them has
     already lowered [mma_best_ms]. So [mma_best_ms] can sit strictly below [best_ms], by less than
     that band: not a report-assembly fault, just what "the winner is the incumbent" means.

     Whether the band is REACHABLE is a property of the clock against the workload, which is why
     repeats on a quiet machine do not reproduce it. Times land on the platform's timer ticks (41.7
     ns on Apple silicon's 24 MHz mach timebase), so the half-open interval [0.99 * best_ms,
     best_ms) contains a representable time only once [best_ms] exceeds ~100 ticks, i.e. ~4 us. On
     [cc] here these kernels time at ~0.25 us — 6 ticks — and the interval is empty, which is why 23
     forced repeats could not flip the conjunct; on [metal] the same computation times at ~0.23 ms,
     where the band spans ~57 ticks and is wide open, and a contended CI runner inflates the [cc]
     times the same way.

     The other three conjuncts below are decided by the winner's schedule and its rendering census,
     not by measured times, and cannot flip on a repeat. So the two timing-dependent ones state
     their precondition instead: the band here, and [<=] rather than [=] below. *)
  let beam_min_progress = 0.01 in
  let winner_contract ~which (r : Autotune.report) =
    let d fmt = Printf.ksprintf (fun s () -> s) fmt in
    let ms f = if Float.is_inf f then "inf" else Printf.sprintf "%.6f" f in
    (* Timing-dependent, so it stays OFF stdout: a golden that pinned these numbers — or any boolean
       derived from them — would flake exactly where the conjunction did. Printed unconditionally,
       so the next investigation reads the winner off a passing run's stderr. [mma-best] is the
       signed margin whose SIGN the old conjunction asserted: negative is a tensorized round
       candidate that undercut the crowned incumbent sub-threshold, which the band above says is
       legal. *)
    Stdio.eprintf
      "%s: label=%S best_ms=%s mma_best_ms=%s mma-best=%+.6f tensorized=%b mma=%d/%d scalar\n%!"
      which r.Autotune.best_label (ms r.Autotune.best_ms) (ms r.Autotune.mma_best_ms)
      (r.Autotune.mma_best_ms -. r.Autotune.best_ms)
      r.Autotune.best_tensorized r.Autotune.best_mma_scalar_fallbacks r.Autotune.best_mma_statements;
    Verdict.pass_fail
      (which ^ " labels a winner exactly when it has a winning time")
      (Bool.equal (String.is_empty r.Autotune.best_label) (Float.is_inf r.Autotune.best_ms))
      ~detail:(d "label=%S best_ms=%s" r.Autotune.best_label (ms r.Autotune.best_ms));
    (* The flag agrees with the SCHEDULE it describes, never with the winning spec's label promise:
       a beam-appended [Tensorize] on a saved incumbent promises nothing in its label (gh-ocannl-545
       / 546). Both fields are read off the winner's applied schedule today, so this holds by
       construction — which is the point: it fails the day either one is re-derived from a label or
       from [mma_timed]. *)
    Verdict.pass_fail
      (which ^ "'s tensorized flag agrees with the schedule it carries")
      (Bool.equal r.Autotune.best_tensorized (tensorized_schedule r))
      ~detail:
        (d "flag=%b, best_schedule carries %d Tensorize op(s)" r.Autotune.best_tensorized
           (List.count r.Autotune.best_schedule ~f:(function SC.Tensorize _ -> true | _ -> false)));
    Verdict.pass_fail
      (which ^ "'s scalar fallbacks do not exceed its Tile_mma statements")
      (r.Autotune.best_mma_scalar_fallbacks <= r.Autotune.best_mma_statements)
      ~detail:
        (d "%d scalar of %d statements" r.Autotune.best_mma_scalar_fallbacks
           r.Autotune.best_mma_statements);
    (* The best timed tensorized candidate cannot beat the crowned winner outright — only undercut
       it inside the progress band above, which is the precondition the bare inequality was missing.
       [infinity >= infinity *. 0.99] holds, so a search that timed nothing passes; a finite
       [mma_best_ms] under an infinite [best_ms] would not, and should not. *)
    Verdict.pass_fail
      (which ^ "'s best tensorized time undercuts the winner only inside the beam's progress band")
      Float.(r.Autotune.mma_best_ms >= r.Autotune.best_ms *. (1. -. beam_min_progress))
      ~detail:
        (d "mma_best_ms=%s best_ms=%s (band %.0f%%)" (ms r.Autotune.mma_best_ms)
           (ms r.Autotune.best_ms) (beam_min_progress *. 100.));
    (* A winner that tensorizes was itself timed as a tensorized candidate, so it is a member of the
       population [mma_best_ms] minimizes over: [<=], not [=] — a sub-threshold round candidate may
       hold the minimum, which is the same band as above and equally not a fault. Keyed on labels
       instead, [mma_best_ms] would stay [infinity] for a beam-appended [Tensorize] winner and this
       would fail, which is the regression it is here to catch. Excluded on a cache hit, which times
       nothing in this process — and which is also where the implication genuinely does not hold: an
       entry written before [mma_best_ms] existed replays [infinity] for it while [best_tensorized]
       still comes off the saved schedule. *)
    Verdict.pass_fail
      (which ^ "'s tensorizing winner was itself timed in the tensorized family")
      ((not r.Autotune.best_tensorized) || replayed r
      || Float.(r.Autotune.mma_best_ms <= r.Autotune.best_ms))
      ~detail:
        (d "tensorized=%b mma_best_ms=%s best_ms=%s" r.Autotune.best_tensorized
           (ms r.Autotune.mma_best_ms) (ms r.Autotune.best_ms))
  in
  winner_contract ~which:"searched report" r1;
  winner_contract ~which:"second report" r2;
  (* A winner is named exactly when one was timed (the conjunct above), so the second call names one
     unless its own windows were refused: a replay carries the cached label, and a re-search that
     the load emptied has none to carry (gh-ocannl-892). EVERY window, not merely one (Codex P2 on
     PR #608): a search that admitted a timing and still crowned nothing is the winner-selection
     regression the five [winner_contract] conjuncts accept as internally consistent, and it is this
     claim that catches it. *)
  p "second report names its winner, or the load refused it every timing"
    ((not (String.is_empty r2.Autotune.best_label))
    || (r2.Autotune.candidates_timed = 0 && r2.Autotune.timings_contended > 0));
  p_all2 "cached schedule replays to correct elementwise values" got2 expected_c ~f:approx;
  p_all2 "cached schedule replays to correct matmul values" got_mm2 mm_expected ~f:approx;

  (* --- gh-ocannl-559: with the search off (config [autotune_search], which the [reproducible]
     profile sets), nothing is timed. A committed cache entry still replays -- a pinned schedule is
     deterministic, so a reproducible run can keep a tuned one -- and a miss compiles the untuned
     default pipeline, which is what the caller would have gotten without calling [tune]. The cache
     populated by the two calls above is the "committed" one. --- *)
  let tune_no_search ~cache_dir () =
    let ctx = Context.auto () in
    let report = ref None in
    let ctx, routine =
      Autotune.tune ~search:false ~cache_dir
        ~report:(fun r -> report := Some r)
        ctx tune_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    ( Option.value_exn ~here:[%here] !report,
      Context.get_values ctx tc1.Tensor.value,
      Context.get_values ctx tc2.Tensor.value )
  in
  let r3, got3, got_mm3 = tune_no_search ~cache_dir:"" () in
  p "search off without a cache times nothing"
    (r3.Autotune.candidates_timed = 0 && r3.Autotune.rounds_run = 0
    && List.is_empty r3.Autotune.best_schedule
    && String.is_empty r3.Autotune.best_label);
  p_all2 "search off without a cache returns correct untuned elementwise values" got3 expected_c
    ~f:approx;
  p_all2 "search off without a cache returns correct untuned matmul values" got_mm3 mm_expected
    ~f:approx;
  let cache_committed = first_cacheable || (completed r2 && r2.Autotune.timings_contended = 0) in
  let r4, got4, got_mm4 = tune_no_search ~cache_dir () in
  p "search off replays exactly when a complete search committed a cache entry"
    (Bool.equal (replayed r4) cache_committed && r4.Autotune.candidates_timed = 0);
  p_all2 "cache replay under search off gives correct elementwise values" got4 expected_c ~f:approx;
  p_all2 "cache replay under search off gives correct matmul values" got_mm4 mm_expected ~f:approx;
  (* gh-ocannl-644, gh-ocannl-677: what a call did about searching is ONE state, and the states a
     measurement harness must tell apart are reached right here -- a completed search, a cached
     winner replayed, and a call that neither searched nor replayed because the search is off and
     nothing was cached (the reproducible profile), which ships the untuned default. Pinning the
     state's own name is the point: the boolean spellings of it were independently derivable, and
     "[not cache_hit] means this process searched" is false for exactly r3, which is the reading
     that cost a benchmark sweep every tuned cell under that profile. The two failing states are
     pinned in autotune_arm_containment.ml, which is where a search can be made to die. *)
  let named_states =
    [
      ("completed search", r1);
      ("cache replay", r2);
      ("search off, no cache", r3);
      ("search off, cached", r4);
    ]
  in
  List.iter named_states ~f:(fun (what, r) ->
      Stdio.eprintf "state (not part of the golden): %-22s -> %s\n" what
        (Autotune.outcome_name r.Autotune.outcome));
  p "the four calls report outcomes consistent with contention-aware cacheability"
    (List.equal String.equal
       (List.map named_states ~f:(fun (_, r) -> Autotune.outcome_name r.Autotune.outcome))
       [
         "searched";
         (if first_cacheable then "cache-replay" else "searched");
         "search-disabled";
         (if cache_committed then "cache-replay" else "search-disabled");
       ]);
  p_all "a report that neither searched nor replayed timed nothing" named_states ~f:(fun (_, r) ->
      match r.Autotune.outcome with
      | Autotune.Searched | Autotune.Search_died _ | Autotune.Cache_replay -> true
      | Autotune.Search_disabled | Autotune.Pre_search_failure _ -> r.Autotune.candidates_timed = 0);
  (* Only a CHOSEN cache replays (Codex P2 on PR #291): the SAME directory is replayed or ignored
     depending on whether someone asked for it. A search with no [cache_dir] populates the built-in
     default directory; a search-less call that likewise names no directory must not pick that entry
     up -- otherwise two reproducible runs would differ on whether an earlier local search happened
     to leave one there -- while naming the same directory explicitly replays it. *)
  let default_cache_dir = "autotune_cache" in
  let default_search_report = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1
      ~report:(fun r -> default_search_report := Some r)
      ctx tune_comp Ir.Indexing.Empty
  in
  let default_search_report = Option.value_exn ~here:[%here] !default_search_report in
  let r5, _, _ = tune_no_search ~cache_dir:default_cache_dir () in
  p "search off replays the chosen default cache exactly after a complete search"
    (Bool.equal (replayed r5) (default_search_report.Autotune.timings_contended = 0));
  let r6 = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~search:false ~report:(fun r -> r6 := Some r) ctx tune_comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got6 = Context.get_values ctx tc1.Tensor.value in
  let r6 = Option.value_exn ~here:[%here] !r6 in
  p "search off ignores the unchosen default cache directory"
    (match r6.Autotune.outcome with Autotune.Search_disabled -> true | _ -> false);
  p_all2 "the ignored-cache fallback is still a correct routine" got6 expected_c ~f:approx
