open Base

(* The sketch families — matmul/conv site detection, the composed schedule pipelines they
   parameterize, and the refinement trees whose leaves are the seed lists — live in their own module
   (gh-ocannl-580). Included rather than opened: the search harness below refers to the family types
   and helpers unqualified, and {!sketch_params} and the site types are part of this module's public
   interface. The harness keeps its own aliases for the IR modules. *)
include Sketch_families
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Outcome = Ir.Schedule_outcome
module Host_load = Utils.Host_load
module SC = Ir.Schedule_cache

type decline_summary = { key : Outcome.rejection_key; count : int; sample_details : string list }
type terminal_failure = { phase : Outcome.phase; candidate : string option; detail : string }

(* gh-ocannl-1110: why a search stopped early by its caller's rule. See the interface. *)
type abandonment = { ab_timed : int; ab_best_ms : float; ab_incumbent_ms : float; ab_ratio : float }

(* gh-ocannl-677: the one thing a [tune] call did about searching, as a state rather than as
   independent flags. The states are mutually exclusive and each carries exactly its own data, so
   "replayed a cached winner AND ran a search" and "died mid-search but carries no failure" are not
   expressible, and a consumer that forgets a state stops compiling instead of answering [false].
   The counters below describe how much work the state got through; they never identify it. *)
type outcome =
  | Searched  (** A search ran and completed. *)
  | Search_died of terminal_failure
      (** A search ran and terminated on a fatal failure. The counters hold what it had reached. *)
  | Cache_replay  (** A cached winner replayed; nothing was searched in this process. *)
  | Abandonment_replay of abandonment
      (** Cached timings justify abandoning this call; nothing was searched or returned. *)
  | Search_disabled
      (** [autotune_search=false] with nothing to replay: the untuned default ships. *)
  | Pre_search_failure of terminal_failure
      (** A failure before (or instead of) the search proper: the base compile, the baseline link or
          timing, a fatal cache replay, an untuned fallback compile. *)
  | Abandoned of abandonment
      (** A search its caller's [?abandon] rule stopped after its first timed candidates
          (gh-ocannl-1110). Not a failure: nothing was compiled for the caller. *)
  | Lineage_refused of { caller_digest : string; timing_digest : string }
      (** The [timing_ctx] lowers the routine to another program than the caller's context does
          (gh-ocannl-1421): no schedule tuned there applies to the caller, so nothing is searched or
          replayed and the untuned default ships. *)

(* gh-ocannl-755: what a timed candidate's number is a measurement OF. [Isolated] times one launch
   followed by a host synchronization, so the number is the latency of a lone dispatch — the kernel
   plus one submit/sync round trip. [Queued] dispatches a calibrated number of launches back to back
   and synchronizes once, dividing by the count, so the round trip is amortized and what is left is
   what the kernel sustains inside a stream that already has work in it.

   They are not the same objective and they do not crown the same candidate. On gfx1151 the round
   trip is ~50-60 us; at the gpt2_mini out-projection shape (134 MFLOP) the fastest candidates run
   in 60-70 us, so [Isolated] reads about 2x their steady-state cost — and the offset is not a
   constant the ranking could ignore: measured per candidate over that site's ten seeded geometries
   it spans 39-86 us, varying with the block count and the per-launch queue work, and within a
   single run it spans up to 45 us across candidates that are 5-8 us apart in steady state. Two such
   candidates therefore swap places once each has its own round trip added to it, and at that site
   they do -- in 2 of 8 measured runs, against 0 of 8 for [Queued] against an independent batched
   instrument. See the tables in gh-ocannl-755. Re-measured there on 2026-09-27 (gh-ocannl-833) the
   offset had shrunk to 0-6 us (at most 1.08x), and the crown still moved in 3 of 4 site-runs
   against 0 of 4 for [Queued]: the offset's variation between candidates, not its size, reorders a
   close pair. The same site-runs on M4 Max Metal (offset 42-143 us) moved the isolated crown in 3
   of 4 too, and on CUDA sm_120 (a uniform 3.8-5.6 us against a 10-18% gap between the top two) in 0
   of 4; [Queued] crowned the batched winner in every site-run on all three backends.

   [Queued] is the default because it is the objective the workload presents: a training step queues
   every kernel of a layer into one stream and synchronizes at the end, so no kernel in it pays a
   round trip of its own. [Isolated] remains selectable for a workload that really does dispatch one
   kernel and wait — and because it is what every schedule crowned before gh-ocannl-755 was ranked
   by. *)
type timing_mode = Isolated | Queued
type timing_sample = { per_launch_ms : float; contention_ms : float }

(* A timing can produce a numeric minimum and still fail to establish that the minimum is
   representative: under host contention most samples can be orders of magnitude above the one clean
   sample the min-of-N eventually finds. Keep that fact beside the number so calibration and ranking
   cannot silently consume it as an ordinary measurement (gh-ocannl-855). [unbatched] is the other
   refusal, kept apart from contention so a report can tell them apart (gh-ocannl-1098): a queued
   calibration that measured no batch within its target for a candidate its single launch said
   should batch.

   [dispersed] and [spread] describe the window apart from its verdict (gh-ocannl-1558). [dispersed]
   is the 2x-majority rule of [window_result]: at least half of the window's raw batch walls exceed
   twice its fastest. [spread] is the window's median raw batch wall over that fastest one -- the
   statistic the rule reads, recorded so a reader can see how noisy an admitted reading was. On a
   GPU backend a dispersed window is a contended one; on a CPU backend it is refused only when the
   host-load signal shows a co-tenant ([attribute_dispersion]), so a routine's own variance on a
   quiet many-core host is admitted with its spread kept. *)
type timing_result = {
  ms : float;
  contended : bool;
  unbatched : bool;
  dispersed : bool;
  spread : float;
  samples : int;
}

(* A diagnostic stream's switch: a config flag read once per process, which one {!tune} or
   [Train.tune_placements] call can override for its own duration through its argument. The override
   is domain-local, like the phase a timing decision carries (gh-ocannl-1199): a per-call argument
   must not switch the stream of a search in another domain. A call that passes nothing inherits the
   enclosing call's override, so a [tune_placements] argument reaches the searches it runs. *)
type stream_switch = { setting : bool Lazy.t; override : bool option Stdlib.Domain.DLS.key }

let stream_switch setting = { setting; override = Stdlib.Domain.DLS.new_key (fun () -> None) }

let stream_enabled { setting; override } =
  match Stdlib.Domain.DLS.get override with Some on -> on | None -> Lazy.force setting

(* Restores the enclosing setting on return or exception, so nested calls compose. *)
let with_stream { override; _ } on f =
  match on with
  | None -> f ()
  | Some _ ->
      let previous = Stdlib.Domain.DLS.get override in
      Exn.protect
        ~finally:(fun () -> Stdlib.Domain.DLS.set override previous)
        ~f:(fun () ->
          Stdlib.Domain.DLS.set override on;
          f ())

(* Per-candidate search diagnostics on stderr, gated by config [autotune_log] -- or, for the
   duration of one call, by its [?log] argument. Kept above the timing policy because a cap-bound
   queued batch reports the target wall it could not reach. *)
let log_switch =
  stream_switch
    (lazy
      (match
         String.lowercase
           (String.strip (Utils.get_global_arg ~arg_name:"autotune_log" ~default:"false"))
       with
      | "true" | "1" -> true
      | _ -> false))

let log_enabled () = stream_enabled log_switch

let logf fmt =
  Printf.ksprintf (fun s -> if log_enabled () then Stdio.eprintf "autotune: %s\n%!" s) fmt

(* gh-ocannl-1407: the seeding census on stderr, gated by config [autotune_log_seeding]. Seeding
   refusals happen before any candidate exists, so the decline report cannot hold them; this is
   where an evidence gate citing a seeding gate is checked. Separate from [autotune_log]: that
   stream pays for an extra control compile, and a census of a whole search wants nothing else. *)
let log_seeding_enabled =
  let on = lazy (Utils.get_global_flag ~default:false ~arg_name:"autotune_log_seeding") in
  fun () -> Lazy.force on

let print_seeding_lines lines =
  List.iter lines ~f:(fun l -> Stdio.eprintf "%s\n" l);
  Stdio.Out_channel.flush Stdio.stderr

(* One lowering's census, taken where the caller seeds it; a no-op unless the key is on. *)
let log_seeding ?coalesced ?capabilities ~where ~is_gpu ~is_cpu ~limits opt =
  if log_seeding_enabled () then
    print_seeding_lines
      (seeding_census_lines ~where
         (seeding_census ?coalesced ?capabilities ~is_gpu ~is_cpu ~limits opt))

(* gh-ocannl-1061: the search's cost record, gated by config [autotune_progress] -- or, for the
   duration of one call, by its [?progress] argument. A search can run for hours and be killed by a
   cap before it reports anything, and [autotune_log] is no substitute: it pays for an extra
   untuned-default control compile, so it moves the cost it would be recording. These lines cost a
   clock read per candidate, and each is flushed as it is written, so a kill keeps everything up to
   it.

   The per-candidate line is written at EVERY attempt's start, not at a time-bounded rate: the
   candidate a cap kills the search inside is the one the record most needs to name, and without a
   thread beside the blocking compile or timing call only the line written before it can name it. A
   time bound always leaves a window after the previous line (review rounds 2-3 on PR #817). The
   rate is bounded by the attempts themselves -- each is a candidate compile, a timing window or a
   pruning decision -- and the phase, arm and flip lines by the search's own structure. The format
   is the interface's contract; see {!progressf}. *)
let progress_switch =
  stream_switch (lazy (Utils.get_global_flag ~default:false ~arg_name:"autotune_progress"))

let progress_enabled () = stream_enabled progress_switch

let with_streams ?log ?progress f =
  with_stream log_switch log (fun () -> with_stream progress_switch progress f)

(* Taken when this module is initialized, i.e. at program start: [wall_s] places a line within the
   process, which for a benchmark cell is within the cell's own wall clock. *)
let process_clock = Mtime_clock.counter ()
let seconds_since counter = Mtime.Span.to_float_ns (Mtime_clock.count counter) /. 1e9

let progressf fmt =
  Printf.ksprintf
    (fun s ->
      if progress_enabled () then
        Stdio.eprintf "autotune-progress: wall_s=%.1f %s\n%!" (seconds_since process_clock) s)
    fmt

let progress_ms ms = if Float.is_finite ms then Printf.sprintf "%.4f" ms else "none"

let progress_stopwatch () =
  let c = Mtime_clock.counter () in
  fun () -> seconds_since c

let timing_string = function Isolated -> "isolated" | Queued -> "queued"

let timing_of_setting s =
  match String.lowercase (String.strip s) with
  | "isolated" -> Isolated
  | "queued" -> Queued
  | other ->
      invalid_arg ("ocannl_autotune_timing setting should be isolated or queued; found: " ^ other)

(* Which exit of the calibration chose the depth before the post-calibration bounds
   (gh-ocannl-1199). Each branch of [calibrate_and_time_from] returns its own, so the trace reports
   the decision taken rather than a reconstruction of it. *)
type batch_settle =
  | Isolated_objective
  | Single_estimate
  | Singles_reach_target
  | Singles_far_over_target
  | Measured_batch
  | Affine_projection
  | Wall_scaled
  | Cap_measured
  | Cap_unresolved
  | Budget_exhausted

type batch_admission = Admitted | Refused_unbatched | Refused_contended | Refused_unreadable

(* The spellings the benchmark trace and result line use, one per constructor. *)
let batch_settle_name = function
  | Isolated_objective -> "isolated"
  | Single_estimate -> "single_estimate"
  | Singles_reach_target -> "singles_reach_target"
  | Singles_far_over_target -> "singles_far_over_target"
  | Measured_batch -> "measured_batch"
  | Affine_projection -> "affine_projection"
  | Wall_scaled -> "wall_scaled"
  | Cap_measured -> "cap_measured"
  | Cap_unresolved -> "cap_unresolved"
  | Budget_exhausted -> "budget_exhausted"

let batch_admission_name = function
  | Admitted -> "admitted"
  | Refused_unbatched -> "refused_unbatched"
  | Refused_contended -> "refused_contended"
  | Refused_unreadable -> "refused_unreadable"

(* The one admission verdict for a timing window, naming the cause of a refusal (gh-ocannl-1517). An
   unreadable reading is named before contention: a non-positive minimum is also the dispersion
   check's floor, against which every positive sample reads as a stall, so a clock that resolved
   nothing used to be counted as a contended host. [unbatched] comes first because it is set on the
   depth-1 reading calibration fell back to, whose own minimum and dispersion are beside the
   point. *)
let timing_admission { ms; contended; unbatched; _ } =
  if unbatched then Refused_unbatched
  else if (not (Float.is_finite ms)) || not (Float.is_positive ms) then Refused_unreadable
  else if contended then Refused_contended
  else Admitted

(* The one admission gate for a timing verdict, for the consumers that RANK: candidate selection,
   the calibration rows, the roofline consistency check, cache attribution. Keeping it next to the
   result type prevents such a caller from proving only one half of usability (usually [not
   contended]) and accidentally feeding an unresolved zero/NaN clock reading to ranking.

   Deliberately not the gate for [queued_batch_depth] (gh-ocannl-888): a batch depth is a scale
   estimate, not a measurement, and both of its error directions are bounded. *)
let admitted_timing_ms result =
  match timing_admission result with
  | Admitted -> Some result.ms
  | Refused_unbatched | Refused_unreadable | Refused_contended -> None

(* What a refused window's log line says about it; one phrase per cause, so the line names it. *)
let refusal_reason = function
  | Refused_unbatched -> "queued calibration measured no batch within the target"
  | Refused_unreadable ->
      "a degenerate clock reading (a non-positive or non-finite minimum) made the sample window \
       unusable"
  | Refused_contended -> "host contention made the sample window unusable"
  | Admitted -> "admitted"

(* gh-ocannl-1360: what each beam-expansion round offered, per beam entry and per unit. See the
   interface. *)
type round_unit_moves = { unit_segment : int option; unit_expanded : bool; unit_moves : int }

type round_entry_moves = {
  entry_digest_tag : string;
  entry_ms : float;
  entry_segments_differing : int list option;
  entry_units : round_unit_moves list;
}

type round_moves = {
  round_index : int;
  round_expansion_rule : string;
  round_entries : round_entry_moves list;
}

(* gh-ocannl-1635: how the beam finals decided the crown. See the interface. *)
type finals_outcome = Finals_not_run | Finals_kept | Finals_swapped | Finals_inconclusive

let finals_outcome_name = function
  | Finals_not_run -> "not run"
  | Finals_kept -> "kept"
  | Finals_swapped -> "swapped"
  | Finals_inconclusive -> "inconclusive"

type finals_contender = {
  contender_label : string;
  contender_search_ms : float;
  contender_finals_ms : float option;
}

type report = {
  outcome : outcome;
  candidates_timed : int;
  timings_contended : int;
  timings_retried : int;
  timings_unbatched : int;
  timings_unreadable : int;
      (** Of [timings_contended], the windows refused for a degenerate clock reading
          (gh-ocannl-1517). See the interface. *)
  timings_dispersed : int;
      (** Admitted windows that were dispersed (gh-ocannl-1558). See the interface. *)
  timings_hopeless : int;
      (** Windows cut as hopeless, their candidates unranked (gh-ocannl-1527). See the interface. *)
  timing_decisions : (batch_settle * batch_admission * int) list;
      (** How this call's search timing calls settled and were admitted (gh-ocannl-1263). See the
          interface. *)
  candidates_contended : int;
  default_refused : bool;
  candidates_failed : int;
  baseline_declined : bool;
  baseline_refusal : batch_admission option;
      (** Why the baseline's own timing window was refused, if it was (gh-ocannl-1423,
          gh-ocannl-1517). See the interface. *)
  declines : decline_summary list;
  privatize_drops : decline_summary list;
      (** The privatized presets' dropped [Privatize] proposals, by cause (gh-ocannl-1438). See the
          interface. *)
  cache_replay_refusal : Outcome.classified_cause option;
  rounds_run : int;
  round_moves : round_moves list;
  finals : finals_contender list;
      (** The beam finals' contenders and readings (gh-ocannl-1635). See the interface. *)
  finals_timed : int;
  finals_admitted : int;
  finals_outcome : finals_outcome;
  budget_cut : bool;
  budget_s : float option;
      (** The wall-clock budget this call resolved (gh-ocannl-1438). See the interface. *)
  call_budget_s : float option;
      (** The enclosing [Train.tune_placements] call's budget (gh-ocannl-1440). See the interface.
      *)
  call_share_s : float option;
      (** This search's share of [call_budget_s] (gh-ocannl-1440). See the interface. *)
  compiles : int;
      (** Backend compiles this call completed before it reported (gh-ocannl-1440). See the
          interface. *)
  replay_compiles : int;
      (** Compiles of a cached schedule's replay among [compiles] (gh-ocannl-1440). See the
          interface. *)
  beam_width : int;
  sketch_candidates : int;
  epilogue_sketch_candidates : int;
  fiss_sketch_candidates : int;
  fiss_sketch_timed : int;
  fiss_sketch_composite : [ `Ineligible | `Singles_refused | `Proposed | `Refused | `Timed ];
  fiss_sketch_playoff_timed : int;
  fiss_sketch_playoff_swaps : int;
  fiss_sketch_staffed_timed : int;
  fiss_sketch_backdrop_fallbacks : int;
  split_reduce_candidates : int;
  split_reduce_timed : int;
  split_reduce_composite_eligible : bool;
  split_reduce_composite_timed : bool;
  split_reduce_backdrop_fallbacks : int;
  mma_candidates : int;
      (** Candidates whose label promises a tensorized pipeline ([spec_expects_mma]) that the search
          put through candidate compile: whole-routine and per-fission-segment seeds, the
          cross-segment recombination composite, and beam-expansion candidates. *)
  fiss_mma_candidates : int;
      (** Of [mma_candidates], candidates built from per-fission-segment MMA sketches. *)
  mma_timed : int;
      (** How many of [mma_candidates] survived candidate compile far enough to be TIMED. A search
          with [mma_candidates > 0] and [mma_timed = 0] never measured a tensorized pipeline at all
          — the state gh-ocannl-521 records for every GPU backend. Dedup'd candidates do not count:
          a duplicate digest means an identical candidate was already timed. *)
  model_scored : int;
  model_pruned : int;
  bound_pruned : int;
      (** Candidates the measured-incumbent bound pruning skipped before compile (gh-ocannl-514
          phase 4b, config [autotune_bound_pruning]): their schedule-invariant roofline floor met
          the best measured time so far. Counted apart from [model_pruned] (the keep-fraction
          pre-filter) so the fathomed-vs-timed ledger attributes each mechanism. *)
  fissioned : bool;
  baseline_ms : float;
  default_ms : float option;
  seeded_ms : float option;
  best_ms : float;
  best_spread : float option;
      (** The winning window's spread (gh-ocannl-1558). See the interface. *)
  timing : timing_mode;
  best_label : string;
  best_tensorized : bool;
  best_tensorization : Ir.C_syntax.tensorization option;
  best_mma_statements : int;
  best_mma_scalar_fallbacks : int;
  mma_best_ms : float;
      (** The best timed tensorized candidate's time (gh-ocannl-546), [infinity] when none was
          timed. Its margin against [best_ms] is what tells a crowned tensorization apart from one
          that lost by 1% and one that lost by 40%. Structural, not label-keyed (see its set site),
          and on a [Cache_replay] report it is the storing search's measurement, like [best_ms] —
          the counters describe this call, the times describe the program. *)
  best_schedule : SC.saved_schedule;
  source_digest : string;
      (** The digest of the base lowering this call tuned (gh-ocannl-1022), [""] when it never
          reached one. See the interface. *)
  best_steps : (int * float) list;
      (** The best-so-far as a step function of the admitted timings (gh-ocannl-1110). See the
          interface. *)
  compile_cost : Ir.Compile_cost.t;
      (** The compiles this call made before delivering the report, summed by phase
          (gh-ocannl-1314). See the interface. *)
}

(* The refused windows the report puts down to host contention alone: what [timings_contended]
   counts minus its two named subsets (gh-ocannl-1517). See the interface. *)
let contention_refusals r = r.timings_contended - r.timings_unbatched - r.timings_unreadable

(* gh-ocannl-1635: whether the beam finals crowned [r]'s winner over a contender that read
   [search_ms] in the search -- the one way [best_ms] can exceed the search reading of an entry the
   beam ranked. See the interface. *)
let finals_outranked r ~search_ms =
  match r.finals_outcome with
  | Finals_swapped -> (
      match List.filter_map r.finals ~f:(fun c -> c.contender_finals_ms) with
      | [] -> false
      | readings ->
          let crown_ms = Option.value_exn (List.min_elt readings ~compare:Float.compare) in
          List.exists r.finals ~f:(fun c ->
              Float.equal c.contender_search_ms search_ms
              && Option.exists c.contender_finals_ms ~f:(fun f -> Float.(f >= crown_ms))))
  | Finals_not_run | Finals_kept | Finals_inconclusive -> false

(** The report of a [tune] call that never searched (config [autotune_search=false], gh-ocannl-559):
    every counter zero and every time [infinity], like a search whose candidates all failed. The
    caller gets the untuned default compile; [outcome] says why. Also the base that the [census]
    below and the pre-search failure reports build on — the census keeps [Search_disabled] (it
    describes exactly that call), a pre-search failure replaces it.

    A function of the objective rather than a constant (gh-ocannl-755 follow-up): [timing] describes
    what every millisecond of a report was measured under, and a report that never resolved one is
    not a state [tune] can be in — a call resolves its objective before it can emit anything. Made a
    constant, this would have to carry [None] and make the field an option for the sake of a
    template, so every consumer of a real report would handle an absence no report has. *)
let no_search_report ~timing =
  {
    outcome = Search_disabled;
    candidates_timed = 0;
    timings_contended = 0;
    timings_retried = 0;
    timings_unbatched = 0;
    timings_unreadable = 0;
    timings_dispersed = 0;
    timings_hopeless = 0;
    timing_decisions = [];
    candidates_contended = 0;
    default_refused = false;
    candidates_failed = 0;
    baseline_declined = false;
    baseline_refusal = None;
    declines = [];
    privatize_drops = [];
    cache_replay_refusal = None;
    rounds_run = 0;
    round_moves = [];
    finals = [];
    finals_timed = 0;
    finals_admitted = 0;
    finals_outcome = Finals_not_run;
    budget_cut = false;
    budget_s = None;
    call_budget_s = None;
    call_share_s = None;
    compiles = 0;
    replay_compiles = 0;
    beam_width = 0;
    sketch_candidates = 0;
    epilogue_sketch_candidates = 0;
    fiss_sketch_candidates = 0;
    fiss_sketch_timed = 0;
    fiss_sketch_composite = `Ineligible;
    fiss_sketch_playoff_timed = 0;
    fiss_sketch_playoff_swaps = 0;
    fiss_sketch_staffed_timed = 0;
    fiss_sketch_backdrop_fallbacks = 0;
    split_reduce_candidates = 0;
    split_reduce_timed = 0;
    split_reduce_composite_eligible = false;
    split_reduce_composite_timed = false;
    split_reduce_backdrop_fallbacks = 0;
    mma_candidates = 0;
    fiss_mma_candidates = 0;
    mma_timed = 0;
    model_scored = 0;
    model_pruned = 0;
    bound_pruned = 0;
    fissioned = false;
    baseline_ms = Float.infinity;
    default_ms = None;
    seeded_ms = None;
    best_ms = Float.infinity;
    timing;
    (* Nothing was timed, so there is no winner to name — and since gh-ocannl-677 the state is in
       [outcome] rather than smuggled through this string. Keeps [best_label]'s contract exact:
       empty exactly when [best_ms] is [infinity]. *)
    best_label = "";
    best_tensorized = false;
    best_tensorization = None;
    best_mma_statements = 0;
    best_mma_scalar_fallbacks = 0;
    mma_best_ms = Float.infinity;
    best_spread = None;
    best_schedule = [];
    (* No base lowering reached yet: [tune] fills this in once its base compile has captured one,
       and every report from then on carries it. *)
    source_digest = "";
    best_steps = [];
    compile_cost = Ir.Compile_cost.empty;
  }

(** The stable one-word name of an outcome state, for logs, JSON records and test goldens. *)
let outcome_name = function
  | Searched -> "searched"
  | Search_died _ -> "search-died"
  | Cache_replay -> "cache-replay"
  | Search_disabled -> "search-disabled"
  | Pre_search_failure _ -> "pre-search-failure"
  | Abandoned _ -> "abandoned"
  | Abandonment_replay _ -> "abandonment-replay"
  | Lineage_refused _ -> "lineage-refused"

(** The fatal failure that ended the call, from whichever of the two failing states it was. A
    projection over the outcome, not a re-derivation of it: "did this call fail" is a question that
    spans two states, and every caller that ranks or attributes arms asks exactly that. *)
let terminal_failure (r : report) =
  match r.outcome with
  | Search_died tf | Pre_search_failure tf -> Some tf
  | Searched | Cache_replay | Abandonment_replay _ | Search_disabled | Abandoned _
  | Lineage_refused _ ->
      None

(* gh-ocannl-1110: the flip chain's early abandonment. *)
exception Search_abandoned of abandonment

type abandon_rule = { incumbent_steps : (int * float) list; trailing_ratio : float }

let best_after steps k =
  List.fold steps ~init:Float.infinity ~f:(fun acc (n, ms) ->
      if n <= k then Float.min acc ms else acc)

let abandon_verdict rule ~k ~steps =
  let incumbent_ms = best_after rule.incumbent_steps k in
  let best_ms = best_after steps k in
  (* Strict, so a flip trailing by exactly the ratio keeps searching; and nothing to compare against
     when the incumbent timed nothing (a cache replay of an entry older than the field). *)
  if Float.is_finite incumbent_ms && Float.(best_ms > rule.trailing_ratio *. incumbent_ms) then
    Some
      {
        ab_timed = k;
        ab_best_ms = best_ms;
        ab_incumbent_ms = incumbent_ms;
        ab_ratio = rule.trailing_ratio;
      }
  else None

(* Best-effort reporting must stay best-effort for ordinary callback errors and NOT for these: an
   interrupt or a runtime-fatal condition raised inside a [report] callback is about the process,
   and swallowing it (on a path that is already failing) would, for a caller that CONTAINS the
   failure per arm, let a long search carry on through a Ctrl-C (gh-ocannl-550). Same set
   {!Ir.Schedule_outcome.classify_raw} refuses to classify. *)
let process_fatal_exn = function
  | Out_of_memory | Stdlib.Sys.Break | Stack_overflow | Assert_failure _ -> true
  | _ -> false

type decline_acc = { mutable da_count : int; mutable da_details : string list }

(* Where the candidate died, for the per-candidate log line. Compile-side phases are already
   apparent from the message; the launch/sync split is not, and it is the difference between "this
   schedule could never run" and "it ran and the device complained". *)
let phase_label (phase : Outcome.phase) = Sexp.to_string (Outcome.sexp_of_phase phase)

let record_decline ?(warn = true) declines (classified : Outcome.classified_cause) =
  let key = Outcome.key_of_cause classified.cause in
  let detail = Outcome.detail_of_cause classified.cause in
  let first_for_key = not (Hashtbl.mem declines key) in
  Hashtbl.update declines key ~f:(function
    | None -> { da_count = 1; da_details = [ detail ] }
    | Some acc ->
        acc.da_count <- acc.da_count + 1;
        if
          List.length acc.da_details < 3 && not (List.mem acc.da_details detail ~equal:String.equal)
        then acc.da_details <- acc.da_details @ [ detail ];
        acc);
  if warn && first_for_key then
    match classified.cause with
    (* Unclassified by construction, and contained under strict classification too (gh-ocannl-564),
       so the warning below — about a compile-side failure only permissiveness absorbed — would be
       saying something false about it. *)
    | Outcome.Unclassified { phase = Outcome.Preflight; _ } -> ()
    | Outcome.Unclassified _ ->
        Stdio.eprintf
          "autotune: WARNING: permissive failure classification contained an unclassified compiler \
           failure (%s); strict_failure_classification=true would stop the search\n\
           %!"
          detail
    | _ -> ()

let decline_summaries declines =
  Hashtbl.to_alist declines
  |> List.sort ~compare:(fun (a, _) (b, _) -> Outcome.compare_rejection_key a b)
  |> List.map ~f:(fun (key, acc) -> { key; count = acc.da_count; sample_details = acc.da_details })

let failed_count declines =
  Hashtbl.fold declines ~init:0 ~f:(fun ~key:_ ~data:acc count -> count + acc.da_count)

(* A candidate round-improvement below this fraction of the incumbent ends the search. *)
let min_progress = 0.01

(* Which units of each beam entry a round expands (gh-ocannl-1229; config
   [autotune_round_expansion]). The beam is there to give a round more than one starting point. On a
   fissioned routine a starting point is new only on the units where it differs from the head: on a
   unit both hold alike, the entry's move is the head's same move over a backdrop already measured
   slower, because the routine's time is the sum of its segments' and the entry trails the head.
   After the composite playoff (gh-ocannl-1166) every runner-up is the head with one to four of
   122-150 segments swapped, so expanding each one in full re-timed the head's menu once per entry:
   #1137's probe paid 635 candidates for a 157-move menu. [Expand_differing] (the default) expands
   the head in full and every other entry only on the units where it differs from the head; an entry
   that segments differently, or a whole-routine one, has no shared units and is expanded in full.
   [Expand_full] is the previous rule; [Expand_head], expanding the head alone, is the single-entry
   control. *)
type round_expansion = Expand_differing | Expand_full | Expand_head

let round_expansion_of_setting = function
  | "differing" -> Some Expand_differing
  | "full" -> Some Expand_full
  | "head" -> Some Expand_head
  | _ -> None

let round_expansion_name = function
  | Expand_differing -> "differing"
  | Expand_full -> "full"
  | Expand_head -> "head"

(* The beam finals (gh-ocannl-1635). Every candidate is timed in a window of its own, and the beam
   ranks entries by those readings, so the head is the entry whose window read fastest -- and a
   window's reading carries the window's own offset: drift, or contention that only ever slows a
   window down, which the 2x-majority rule admits when it is uniform. The crown was therefore often
   the luckiest window rather than the fastest schedule, the winner's curse [Bench_select] fixed for
   [projection_shape_bench] (gh-ocannl-1531). Re-timing the head alone does not repair that: it
   sharpens the number of the wrong pick. The finals re-time the near-tie contenders TOGETHER, in
   adjacent windows, and crown by those readings alone.

   [finals_contenders]: the entries [pool] offers (the search passes the final beam, the beams
   accepted rounds displaced and the round a rejected round discarded, deduplicated), by first
   reading, ties in [pool] order, keeping those with a finite reading within [margin] of the
   fastest, at most [width]; none when fewer than two qualify, or [margin <= 0] (the off setting).
   [finals_order]: the windows' order, the contenders and then the same in reverse, so each
   contender is timed twice and a drift across the finals reaches every contender alike.
   [finals_crown]: a contender's finals reading is the least of its admitted ones, and the crown the
   contender with the least, ties to the earlier; [None] when some contender has no admitted
   reading, since a contender with none cannot be compared in the shared round (the crown then stays
   the first readings' head). *)
let finals_contenders ~margin ~width pool =
  let ranked =
    List.filter pool ~f:(fun (_, ms) -> Float.is_finite ms)
    |> List.stable_sort ~compare:(fun (_, a) (_, b) -> Float.compare a b)
  in
  match ranked with
  | [] -> []
  | (_, best_ms) :: _ ->
      let within =
        List.filter ranked ~f:(fun (_, ms) -> Float.(ms <= best_ms *. (1. +. margin)))
        |> Fn.flip List.take width
      in
      if Float.(margin <= 0.) || List.length within < 2 then [] else within

let finals_order contenders = contenders @ List.rev contenders

let finals_crown readings =
  if List.exists readings ~f:(fun (_, rs) -> List.is_empty rs) then None
  else
    List.fold readings ~init:None ~f:(fun acc (c, rs) ->
        let ms = Option.value_exn (List.min_elt rs ~compare:Float.compare) in
        match acc with Some (_, best) when Float.(best <= ms) -> acc | _ -> Some (c, ms))

(* The composite playoff (gh-ocannl-1166). A per-segment single is timed as the WHOLE routine with
   every other segment on its untuned preset, so its time carries that backdrop: on gpt2_mini/CUDA
   tf32 the q/k/v singles measure about 23.2 ms where the composite they staff runs at 2.4 ms, and
   two replicate searches put the same single 0.02-0.03 ms apart, which is as large as the whole gap
   between the site's top four geometries. The per-key crown among them was a lottery (one replicate
   crowned [mma 32x32x0 bgrid], the other [mma 16x32x0 bgrid-in]). The playoff re-ranks those
   near-ties where the backdrop is small: inside the composite. A single within the playoff margin
   of its key's best single is close enough that the backdrop's noise could have inverted the order:
   config [autotune_composite_playoff_margin], a fraction of that best, default 0.005; a box whose
   backdrop is noisier inverts wider gaps, which only a wider margin reaches. At most
   [playoff_width] of them per key go to the playoff, nearest first: config
   [autotune_composite_playoff_width], default 2, and 0 turns the playoff off (gh-ocannl-1259). Both
   defaults are their keys' [Utils.config_key_defaults] rows (gh-ocannl-1332). Changing either
   default changes the default search menu, so bump [Schedule_cache.entry_version]; a stated margin
   or width is [Search_shaping], like the beam width, so a crown found under one setting replays
   under another. *)

(* gh-ocannl-1138, config [autotune_staffed_singles]: whether the per-segment singles are timed
   against the running composite (see [tune]'s staffed sweep) rather than against every other
   segment's preset. The default is the key's [Utils.config_key_defaults] row; changing it bumps
   [Schedule_cache.entry_version]: the search then times candidates the other one never did. *)
let staffed_singles_setting () = Utils.registered_flag ~arg_name:"autotune_staffed_singles"

(* gh-ocannl-1409: the wall-clock budget of one search, in seconds, or [None] for none. Read per
   call, like the other search settings, so a test can state it for one search. Only a positive
   number sets it ([Utils.Positive_float_or_unset], which is how the trajectory identity spells it):
   empty, zero and negative mean no budget, and text that is no number is refused
   (gh-ocannl-1332). *)
let budget_setting () = Utils.registered_positive_float_or_unset ~arg_name:"autotune_budget_seconds"

(* gh-ocannl-1409: whether a search its budget cut must ship the untuned default rather than its
   crown. A cut search crowns nothing slower than the untuned default because it measured that
   default: the floors (the seeded plan, the default seed) are exempt from the budget. Where neither
   was measured -- no candidate reproduces the default (automatic scheduling with
   [schedule_fission=false] and no seeded plan, where [default_seed_label] is [None]), or the
   floor's windows were refused, the dispatched serial baseline's included when it is the default
   (automatic scheduling off) -- the cut crown has no such reference, and the serial baseline can be
   all it timed. Such a search ships the untuned default and stores nothing, reporting as one that
   timed nothing. Except where the default is an undispatched serial form (a GPU with automatic
   scheduling off, gh-ocannl-532), which no timed candidate can be slower than. A policy seam like
   {!search_measurements_cacheable}, so the cases are pinned without manufacturing each regime. *)
let budget_floor_unmeasured ~budget_cut ~auto_sched ~baseline_dispatched ~floor_measured =
  budget_cut && (auto_sched || baseline_dispatched) && not floor_measured

(* The beam holds no compiled candidate exactly when nothing was timed, which every consumer of the
   winner tests first ([nothing_timed]). *)
let timed_winner_exists = "Autotune.tune: a finite best time without a compiled candidate"

(** {2 Timing} *)

let set_test_bindings routine =
  List.iter routine.Context.bindings ~f:(fun (ss, r) ->
      match ss.Idx.static_range with
      | Some range when range > 0 && ss.Idx.used_as_extent ->
          (* gh-490 symbolic extents: tune at the upper bound. The schedule digest is
             extent-value-independent (the extent is a kernel parameter), so one tuned entry serves
             every extent; measuring at the maximum makes the tuned schedule's cost model
             conservative for smaller runtime extents. *)
          r := range
      | Some range when range > 0 -> r := range / 2
      | _ ->
          (* No range to pick from. Time a value the caller assigned at the caller's value, and
             otherwise at 0, written explicitly: linking leaves the parameter unassigned
             (gh-ocannl-940), and a compile-time search runs before the caller could assign it. *)
          if not (Idx.launch_param_assigned r) then r := 0)

(* Fast routines get extra timed runs beyond [repeats], until this much accumulated PER-SAMPLE time
   (or [max_timing_runs]); every routine gets [min_timing_samples]. A caller's larger [repeats]
   floor still wins over that top-up limit. On sub-millisecond kernels a min-of-3 is dominated by
   launch jitter, and under contention a wall budget used to stop after the three worst samples
   because each host stall spent the whole budget. Noise only ever adds time, so min-of-N converges
   monotonically to the true best case and more samples reduce mis-selection. *)
let min_timing_ms = 25.
let min_timing_samples = 16
let max_timing_runs = 64

(* This is a refusal threshold, not an estimate of ordinary jitter. One slow outlier says nothing
   about the minimum; a MAJORITY this far above it says the sample window was mostly measuring host
   stalls. A 2x majority threshold catches a fixed 20 ms host stall on a calibrated ~10 ms queued
   batch, while a consistently slow routine has no dispersion and remains valid. *)
let contention_ratio = 2.

(* The policy seam shared by calibration and the timed loop. The budget accumulates [per_launch_ms],
   the quantity being ranked; contention is detected independently on [contention_ms], the raw batch
   wall before queued mode divides it by depth. A deep queue therefore neither spends the budget on
   a whole batch at once nor divides a fixed host stall out of the refusal signal. The sample floor
   keeps a burst from ending the min-of-N after the caller's usual three repeats.

   Resumable (gh-ocannl-1074): [prior] is a window already taken, oldest first, and the loop
   continues it under the same stopping rule rather than starting a fresh one -- so a resumed window
   stops exactly where one uninterrupted window over the same samples would. The timed loop resumes
   the queued calibration's synchronized singles this way when the depth settles at 1, where they
   are samples of the very quantity it would otherwise time again. Returns the whole window, oldest
   first.

   [hopeless] (gh-ocannl-1527, default never) is asked after every sample with the window so far,
   oldest first; when it answers, the loop raises {!Timing_hopeless} at once, so a window that
   already proves its candidate out of the running stops taking samples. *)
type hopeless = { hopeless_ms : float; hopeless_threshold_ms : float; hopeless_samples : int }

exception Timing_hopeless of hopeless

let sample_window ?(prior = []) ?(hopeless = fun _ -> None) ~repeats ~sample () =
  let samples = ref (List.rev prior) in
  let total = ref (List.fold prior ~init:0. ~f:(fun total s -> total +. s.per_launch_ms)) in
  let count = ref (List.length prior) in
  while
    !count < Int.max min_timing_samples (Int.max 1 repeats)
    || (Float.(!total < min_timing_ms) && !count < max_timing_runs)
  do
    let ({ per_launch_ms; _ } as timing_sample) = sample () in
    samples := timing_sample :: !samples;
    total := !total +. per_launch_ms;
    Int.incr count;
    Option.iter (hopeless (List.rev !samples)) ~f:(fun h -> raise (Timing_hopeless h))
  done;
  List.rev !samples

(* The median of a list of floats; [nan] for an empty one. *)
let median_float values =
  let sorted = Array.of_list values in
  Array.sort sorted ~compare:Float.compare;
  let n = Array.length sorted in
  if n = 0 then Float.nan
  else if n % 2 = 1 then sorted.(n / 2)
  else (sorted.((n / 2) - 1) +. sorted.(n / 2)) /. 2.

let window_result samples =
  let ms =
    List.fold samples ~init:Float.infinity ~f:(fun best s -> Float.min best s.per_launch_ms)
  in
  let contention_floor =
    List.fold samples ~init:Float.infinity ~f:(fun best s -> Float.min best s.contention_ms)
  in
  let stalled =
    List.count samples ~f:(fun s -> Float.(s.contention_ms > contention_floor * contention_ratio))
  in
  let count = List.length samples in
  (* Dispersion only. A window whose minimum is zero, NaN or infinite is a clock that resolved
     nothing -- a different fact, and one [admitted_timing_ms] already refuses on the number itself.
     Folding the two lost the distinction exactly where it is load-bearing (gh-ocannl-888): the
     depth policy consults one and must ignore the other. The verdict here is the dispersion-only
     policy; a CPU backend's caller then attributes it ([attribute_dispersion], gh-ocannl-1558). *)
  let dispersed = stalled * 2 >= count in
  let spread = median_float (List.map samples ~f:(fun s -> s.contention_ms)) /. contention_floor in
  { ms; contended = dispersed; unbatched = false; samples = count; dispersed; spread }

let sample_min ~repeats ~sample = window_result (sample_window ~repeats ~sample ())

(* gh-ocannl-1527: when a timing window may end before its sample floor because its candidate is out
   of the running. A caller that knows what the reading will be ranked against passes it as the
   window's [reference]; the window is HOPELESS once its first [hopeless_min_samples] or more
   samples have a minimum over [hopeless_ratio] times that reference AND are not dispersed by the
   window's own 2x-majority rule ([window_result]).

   Why that cannot flip a ranking within the noise the timing rules already bound. Noise only adds
   time, so a window's minimum only falls as it grows: the full window could rank below the
   reference only if a later sample undercut every sample so far by more than [hopeless_ratio], i.e.
   only if every sample so far were inflated beyond twice the [contention_ratio] at which the
   admission rule calls a sample a stall. A stall that leaves some samples clean leaves the minimum
   clean; one that inflates most of them but not all disperses the prefix, which therefore never
   cuts -- the same composition as the admission rule itself, which trusts a non-dispersed window
   and refuses (or, on a CPU backend, attributes: gh-ocannl-1558) a dispersed one. Three samples is
   the depth at which the calibration's batch probes already refuse a single stall
   ([queue_batch_probe_min_runs]). The boundary, stated: a co-tenant that delays every one of the
   first samples uniformly by more than [hopeless_ratio] and then leaves is not detected -- exactly
   the uniform delay the admission rule already accepts as a valid reading of a whole window.

   [at_least_ms] gates the calibration's synchronized singles, which are the reading only when the
   depth settles at 1: on cc and Metal that is exactly when their minimum reaches the batch target
   ({!queued_batch_ms}), and a minimum only falls, so a cut there is a cut of the reading itself. On
   CUDA/HIP singles under {!slow_singles_ratio} times the target are followed by a depth-2 probe,
   which could, in principle, settle deeper on a fixed-dominated pair; the per-launch reading at any
   depth is at least the single's marginal part, so a single of at least the target over
   [hopeless_ratio] times the reference is out of the running unless the fixed per-batch
   synchronization term exceeds three quarters of it -- 7.5 ms at the target, two orders of
   magnitude above the ~60 us round trip the batching amortizes. *)
let hopeless_ratio = 4.
let hopeless_min_samples = 3

let hopeless_window ?(at_least_ms = 0.) ~reference_ms samples =
  if
    List.length samples < hopeless_min_samples
    || (not (Float.is_finite reference_ms))
    || Float.(reference_ms <= 0.)
  then None
  else
    let threshold_ms = hopeless_ratio *. reference_ms in
    let r = window_result samples in
    if
      (not r.dispersed) && Float.is_finite r.ms
      && Float.(r.ms > threshold_ms)
      && Float.(r.ms >= at_least_ms)
    then
      Some
        { hopeless_ms = r.ms; hopeless_threshold_ms = threshold_ms; hopeless_samples = r.samples }
    else None

(* How a timing call turns a dispersed window into a verdict (gh-ocannl-1558). [Refuse_dispersed] is
   the dispersion-only rule, kept on GPU backends: host CPU accounting cannot see a device
   co-tenant, and Metal's queued retry is for a transient device-scheduler burst.
   [Attribute_dispersion] is the CPU backends' rule: a dispersed window is refused only when the
   host-load signal over it shows a co-tenant. A many-thread cc routine's own run-to-run spread
   meets the 2x-majority rule on a quiet 32-thread host (step p90 1.7-4.9x p10 on minix,
   gh-ocannl-1482), and refusing it made every search there uncacheable. *)
type dispersion_policy = Refuse_dispersed | Attribute_dispersion

let dispersion_policy_for_backend backend =
  if Sched.backend_is_cpu backend then Attribute_dispersion else Refuse_dispersed

(* The attribution itself, pure. Only a dispersed window changes, and only on a [Quiet] verdict: a
   co-tenant keeps the refusal, and so does an [Unavailable] signal -- no evidence of quiet is not
   evidence of it. A window that is not dispersed is admitted under load exactly as before: a
   uniformly delayed window is a valid reading (Codex P2, round 4 on PR #735), and refusing on load
   alone would make every cc search on a busy box or a parallel CI run uncacheable. *)
let attribute_dispersion (verdict : Host_load.verdict) result =
  match verdict with
  | Quiet _ when result.dispersed && result.contended -> { result with contended = false }
  | Quiet _ | Cotenant _ | Unavailable -> result

(* A usable winner drawn from an incomplete measurement set may ship for this call, but must not
   become the answer to every later call through the schedule cache. A later idle process needs to
   retry any schedule whose timing was refused for contention. Kept as a policy seam so the cache
   gate is pinned without manufacturing a contended device run. *)
let search_measurements_cacheable ~nothing_timed ~timings_contended =
  (not nothing_timed) && timings_contended = 0

(* Queued mode's batch depth is calibrated per candidate rather than fixed. A fixed depth is either
   too shallow to amortize the round trip on a fast kernel, or minutes of uninterruptible dispatches
   on a slow one — and the tuner meets both within one search. [queued_batch_ms] is the wall time
   one batch aims for: at a ~60 us round trip it keeps the overhead under 1% of the reading, and it
   makes each sample long enough to amortize the host round trip. The sampling budget is per-launch,
   not batch wall (gh-ocannl-855), so [max_timing_runs] rather than [min_timing_ms] bounds the wall
   cost of queued timing once the caller's [repeats] floor is met. [max_queue_depth] stops a
   sub-microsecond CUDA/HIP kernel from minting an unbounded in-memory dispatch queue. The cc and
   Metal paths retain the historical 200 cap: Metal's observed target is already below it, and
   multiplying cc's timed batch did not serve this GPU dispatch-scale correction. A genuinely slow
   routine gets depth 1 and is then measured exactly as [Isolated] measures it; later target-sized
   batch probes require a depth-separated marginal confirmation, so one stall-inflated window does
   not silently take that same path. The 10 ms target is also the Metal long-command-buffer safety
   bound established by gh-ocannl-828: on M4 Max / macOS 26.6.2, two UNORDERED command buffers (the
   probe's raw-queued and wait-after-kernel arms -- the latter was misread as the backend's
   SharedEvent chain until gh-ocannl-909) changed scheduling at about 1.2 s per kernel, while
   [queued_batch_depth] is already 1 at 10 ms -- about 120x below that regime. The backend's own
   launches wait for the previous all-work signal and serialize at every length. See
   [benchmarks/runners/ocannl/metal_queue_probe.ml] for the four-arm discriminator. *)
let queued_batch_ms = 10.
let max_queue_depth = 2048
let legacy_queue_depth = 200

let queue_depth_cap_for_backend = function
  | "cuda" | "hip" -> max_queue_depth
  | _ -> legacy_queue_depth

let retry_contended_window_for_backend backend = String.equal backend "metal"

(* The depth is calibrated from timed single launches, not from the warmup: the warmup absorbs lazy
   initialization and module loading, so on a fast kernel it can overestimate by enough to collapse
   the depth to 1 and silently turn a queued search into an isolated one. The synchronized-single
   estimate uses the shared 16-sample floor because it is dominated by round-trip jitter. A batch
   probe is already milliseconds long and only chooses scale rather than ranking a candidate, so
   twelve minima are enough; imposing the timed window's 16-sample floor on up to six probes added
   nearly 800 ms per candidate and exhausted CI's 45-minute suite ceiling. *)
let queue_calibration_runs = 3
let queue_batch_probe_runs = 12
let max_depth_validation_probes = 4

(* The fit's noise tolerance, as a fraction of the base batch's wall (gh-ocannl-1098). A pair whose
   inferred fixed term is more negative than this fraction of the base reading attributes more than
   125% of that reading to launch work -- the probe's slope contradicts the base by more than the
   25% confirmation step, which is a stall or a non-linearity rather than noise. Relative, not
   absolute: the absolute quarter-target it replaces (2.5 ms) was under 1% of a 64 ms launch's
   batch, so a slow candidate's slightly superlinear pairs never fit and its calibration doubled
   until the validation loop ran out (gh-ocannl-1096). Floored at the target's wall, so for batches
   up to the target the tolerance is the historical 2.5 ms exactly. *)
let fixed_fit_noise_fraction = 0.25

(* Wall budgets on the calibration's batch probes (gh-ocannl-1098). The affine refinement and its
   validation are budgeted in probes and launches, and a launch can cost anything: an unresolved 64
   ms candidate's doubling retries spent 760 launches (~49 s) before its fallback, and a device
   whose cost jumps past a queue threshold spent ~1600 s doubling toward the cap.

   Per probe: the twelve minima continue only while the probe's summed wall is under two
   target-sized probes, and never stop before three. Every probe whose batches are within twice the
   target -- every probe of a calibration that is converging -- still takes all twelve; an
   over-target batch is already evidence against its depth, and three minima still refuse a single
   stall.

   Per calibration: once the probes' summed wall reaches eight target-sized probes -- one for each
   probe that checks this budget, i.e. every probe [queue_calibration_max_probes] below counts
   except the rescue, which checks none -- no further validation, confirmation or retry starts, and
   the calibration ends unresolved, so the wall-bounded fallback decides the depth from what was
   measured. The synchronized singles are not charged: they are the timed window whenever the depth
   settles at 1. *)
let queue_batch_probe_min_runs = 3
let queue_batch_probe_wall_ms = 2. *. Float.of_int queue_batch_probe_runs *. queued_batch_ms
let queue_calibration_wall_ms = 8. *. Float.of_int queue_batch_probe_runs *. queued_batch_ms

(* The most batch probes one CUDA/HIP calibration dispatches: the provisional probe, four
   validations, a confirmation, its stall retry and a sampled shallower crossing -- eight -- then
   the rescue. The calibration's branches already stop there; the count is also enforced, as if the
   wall budget were spent, so that [time_routine]'s documented dispatch maximum is a bound the code
   keeps rather than a reading of its control flow. *)
let queue_calibration_max_probes = 9

(* gh-ocannl-1627: how far over the batch target a CUDA/HIP candidate's synchronized singles must
   read for the calibration to keep depth 1 without its provisional depth-2 probe. The probe exists
   to expose an inflated single window, and for singles of [S] over the target it can move the depth
   off 1 only through a fit whose marginal launch cost is below the target ([Fit_fixed_dominated],
   or a non-monotone pair's retries): a depth-2 batch costing less than [S + target], i.e. a fixed
   per-synchronization term above [S - target]. At twice the target that term is at least the target
   itself, 10 ms -- over a hundred times the ~60 us round trip the batching amortizes, the same
   order of implausibility {!hopeless_window}'s single gate already rests on; a stall that inflated
   every one of the sixteen singles uniformly by that much is the boundary, as it is there. Below
   the ratio the probe still runs: a ~16 ms candidate can genuinely be a few ms of fixed cost over a
   sub-target launch.

   What it saves is the probe's batches: three depth-2 batches, six launches, of the ~22 a slow
   candidate dispatches. On minix (gfx1151) a gpt2_mini_train per-segment single times its whole
   routine around an all-preset backdrop at ~130 ms, so the probe was ~0.8 s of its ~3 s timing,
   once per single and with no effect on the depth it settled on. *)
let slow_singles_ratio = 2.

(* How far past its deepest batch probe a CUDA/HIP calibration may settle, as a multiple of that
   depth (gh-ocannl-1100): one doubling, the step the calibration's own unresolved retries take into
   unmeasured depth. Several exits settle on a depth no probe measured -- the last validation's
   affine projection, a linear scale from a confirmation that read below the target, a fit that
   wants the cap -- and a fixed-dominated or non-monotone pair projects far past every measured
   batch (a pair (2, 12.25 ms) / (3, 12.5 ms) wants depth 40), where a queue cost can jump and
   neither the wall budget nor the fallback would ever look. The timed window would then run at a
   per-launch cost the calibration never saw, and its 2x contention rule would judge a wall it did
   not measure. A bound in depth, not in wall: a jump inside the factor is the bet it states, and no
   probe is spent confirming it. *)
let queue_depth_projection_factor = 2

(* The calibration policy itself, as a function of the estimate, so a test can pin it without a
   device: what [Queued] measures depends on it, and its two boundaries are the ones a regression
   would silently cross -- a depth stuck at 1 turns a queued search back into an isolated one, and
   an uncapped depth turns a microsecond kernel's batch into an unbounded dispatch. A non-positive
   or NaN estimate is a clock that resolved nothing, not a zero-cost kernel: batch as deeply as the
   cap allows rather than degenerating on the very routines queueing exists for. An infinite one is
   not that case and is left to the arithmetic, which floors it at depth 1 — an unboundedly slow
   routine is the far end of the scale the policy is FOR, not a reading it failed to take. The cap
   is applied to the FLOAT, before the conversion: [Float.iround_up_exn] raises on a value outside
   the integer range, and an estimate of a few times [Float.min_positive_subnormal_value] produces
   exactly such a value — the clamp has to happen while the quantity can still hold it.

   Total, and [contended] is deliberately unread (gh-ocannl-888). The 2x-majority contention rule is
   a statement about a ~10 ms batch; the estimate it judges here is ONE dispatch plus one host
   synchronization, whose dispersion on a GPU is the round trip's own tail rather than a host stall.
   Refusing on it starved every search on both GPU backends: the calibration was refused, the
   refusal was returned as the candidate's timing, and so nothing was ever timed on the machines
   queued batching exists for. A depth is not a measurement -- an overestimated one only shortens
   the batch and an underestimated one is capped -- and a deeper batch is precisely the remedy for
   the dispersion that made the estimate look contended. The refusal belongs downstream, in the
   timed loop, where the window being judged IS a batch. *)
let queued_batch_depth_with_cap ~max_depth { ms = est_ms; _ } =
  if Float.is_nan est_ms || Float.(est_ms <= 0.) then max_depth
  else
    let want = queued_batch_ms /. est_ms in
    if Float.(want >= of_int max_depth) then max_depth else Int.max 1 (Float.iround_up_exn want)

let queued_batch_depth = queued_batch_depth_with_cap ~max_depth:max_queue_depth

(* Split the synchronized single-launch and provisional-batch minima into fixed synchronization and
   marginal launch terms. Dividing the provisional wall by its depth leaves a share of the fixed
   term in every launch, so a shallow probe can still select a batch below the contention scale.
   When the marginal term is unresolved, the memory cap is the only honest depth bound. Return the
   predicted whole-batch wall too, so a cap-bound shortfall is logged from this affine model rather
   than from an overhead-polluted per-launch average.

   The third component is what the fit concluded, the decision metadata the timing trace reports
   (gh-ocannl-1199): each verdict names the branch below that produced the depth, so a consumer
   reads the policy's own conclusion rather than re-deriving it from the probe readings. *)
type fit_verdict =
  | Fit_unordered
  | Fit_unreadable
  | Fit_nonpositive_marginal
  | Fit_negative_fixed
  | Fit_fixed_dominated
  | Fit_base_crossing
  | Fit_boundary_floor
  | Fit_shallower_crossing
  | Fit_deeper_crossing

type affine_fit = {
  base_depth : int;
  probe_depth : int;
  fixed_ms : float;
  marginal_ms : float;
  verdict : fit_verdict;
}

let fit_queued_batch_depth_between_with_cap ~min_depth ~max_depth ~base_depth ~base_ms ~probe_depth
    ~probe_ms =
  let retry_depth =
    if probe_depth >= max_depth / 2 then max_depth
    else Int.min max_depth (Int.max (probe_depth + 1) (2 * probe_depth))
  in
  let fit ?(fixed_ms = Float.nan) ?(marginal_ms = Float.nan) verdict =
    { base_depth; probe_depth; fixed_ms; marginal_ms; verdict }
  in
  if probe_depth <= base_depth then (retry_depth, Float.nan, fit Fit_unordered)
  else if
    (not (Float.is_finite base_ms))
    || (not (Float.is_positive base_ms))
    || (not (Float.is_finite probe_ms))
    || not (Float.is_positive probe_ms)
  then (retry_depth, Float.nan, fit Fit_unreadable)
  else
    let marginal_ms = (probe_ms -. base_ms) /. Float.of_int (probe_depth - base_depth) in
    if (not (Float.is_finite marginal_ms)) || not (Float.is_positive marginal_ms) then
      (retry_depth, Float.nan, fit ~marginal_ms Fit_nonpositive_marginal)
    else
      let fixed_ms = base_ms -. (marginal_ms *. Float.of_int base_depth) in
      let fit = fit ~fixed_ms ~marginal_ms in
      let tolerance_ms = fixed_fit_noise_fraction *. Float.max queued_batch_ms base_ms in
      (* A synchronized single below the target owes a queued batch. Two BATCH observations can
         nevertheless put the fitted depth-1 wall just above it: their fixed term and the singles'
         round trip need not agree (gfx1102, gh-ocannl-1184). Keep the measured depth-2 batch when
         this resolved pair proves that its marginal work still fits the target and its fixed term
         is below it. Fixed-dominated fits retain their existing marginal-work policy. Returning to
         the singles would change the objective; refusing would make these ordinary boundary
         candidates permanently uncacheable. An unresolved fit, or marginal work above the target,
         establishes no such supported batch and keeps the existing fallback/refusal. *)
      let min_depth =
        if base_depth = 2 && Float.(fixed_ms < queued_batch_ms && marginal_ms <= queued_batch_ms)
        then min_depth
        else 1
      in
      if Float.(fixed_ms < -.tolerance_ms) then (retry_depth, Float.nan, fit Fit_negative_fixed)
      else if Float.(fixed_ms >= queued_batch_ms) then
        (* The whole-wall target is unattainable, but a positive depth-separated slope still gives a
           safe scale. Target one batch-wall worth of marginal launch work: accepting the shallow
           base would let a shared fixed stall mimic a confirmation, while treating the clean fit as
           unresolved can jump to a many-second cap batch on genuinely slow kernels. *)
        let wanted = queued_batch_ms /. marginal_ms in
        let depth =
          if (not (Float.is_finite wanted)) || Float.(wanted >= of_int max_depth) then max_depth
          else Int.max min_depth (Float.iround_up_exn wanted)
        in
        (depth, fixed_ms +. (marginal_ms *. Float.of_int depth), fit Fit_fixed_dominated)
      else if
        Float.(base_ms >= queued_batch_ms)
        && (base_depth = 1
           || Float.(fixed_ms +. (marginal_ms *. of_int Int.(base_depth - 1)) < queued_batch_ms))
      then (base_depth, base_ms, fit Fit_base_crossing)
      else if Float.(base_ms >= queued_batch_ms) then
        (* A base the fit says is not the model's first target crossing: one launch fewer would
           still fill the target. Keeping it kept a batch of any length -- a slow candidate whose
           doubling retries reached depth 4 would be timed in 256 ms batches, whose launch work
           alone is 25 times the target (gh-ocannl-1098). The pair is resolved, so its crossing is
           the answer; the fixed term is below the target here, so the crossing is at least one
           launch and strictly shallower than the base. *)
        let crossing = Float.iround_up_exn ((queued_batch_ms -. fixed_ms) /. marginal_ms) in
        let depth = Int.max min_depth crossing in
        ( depth,
          fixed_ms +. (marginal_ms *. Float.of_int depth),
          fit (if depth > crossing then Fit_boundary_floor else Fit_shallower_crossing) )
      else
        let wanted = (queued_batch_ms -. fixed_ms) /. marginal_ms in
        let depth =
          if (not (Float.is_finite wanted)) || Float.(wanted >= of_int max_depth) then max_depth
          else Int.max (base_depth + 1) (Float.iround_up_exn wanted)
        in
        (depth, fixed_ms +. (marginal_ms *. Float.of_int depth), fit Fit_deeper_crossing)

let refine_queued_batch_depth_between_with_cap ~min_depth ~max_depth ~base_depth ~base_ms
    ~probe_depth ~probe_ms =
  let depth, wall_ms, _ =
    fit_queued_batch_depth_between_with_cap ~min_depth ~max_depth ~base_depth ~base_ms ~probe_depth
      ~probe_ms
  in
  (depth, wall_ms)

let refine_queued_batch_depth_between =
  refine_queued_batch_depth_between_with_cap ~min_depth:1 ~max_depth:max_queue_depth

let refine_queued_batch_depth_with_cap ~max_depth ~single_ms ~probe_depth ~probe_ms =
  if probe_depth <= 1 then (1, probe_ms)
  else
    refine_queued_batch_depth_between_with_cap ~min_depth:1 ~max_depth ~base_depth:1
      ~base_ms:single_ms ~probe_depth ~probe_ms

let refine_queued_batch_depth = refine_queued_batch_depth_with_cap ~max_depth:max_queue_depth

(* A bounded validation loop can end with two non-monotone batch minima even though its latest
   observation is itself usable scale evidence. Never return to the earlier, possibly inflated
   target crossing in that case. Project from the deepest measured batch wall; at this scale the
   fixed synchronization term is already amortized, and flooring at the measured depth means a
   target-sized result is never made shallower because of jitter. *)
let depth_from_batch_wall_with_cap ~max_depth ~depth ~wall_ms =
  if (not (Float.is_finite wall_ms)) || not (Float.is_positive wall_ms) then (max_depth, Float.nan)
  else
    let wanted = Float.of_int depth *. queued_batch_ms /. wall_ms in
    let depth' =
      if Float.(wanted >= of_int max_depth) then max_depth
      else Int.max depth (Float.iround_up_exn wanted)
    in
    (depth', wall_ms *. Float.of_int depth' /. Float.of_int depth)

(* The wall bound on a depth the affine fits did not choose (gh-ocannl-1096). When calibration ends
   unresolved -- every fit refused, or the cap taken without a measurement there -- the depth it
   returns is a bet that the per-launch cost is negligible against a fixed synchronization cost the
   fits could not separate. The cap bounds that bet in launches, not in wall: a ~61 ms gfx1151
   candidate whose superlinear batches never fit settled at 2048 and spent 126 s per batch, 2016 s
   on one timing call.

   [observed] is every finite positive [(depth, wall)] minimum the calibration measured, the
   synchronized singles as depth 1. The fallback never goes deeper than the deepest depth measured
   within the batch target, and is depth 1 when none was. That is sound for any cost that grows with
   depth -- the true wall is monotone in depth and a measured wall bounds the true one -- and it is
   the only bound that is. The readings that left the fits unresolved cannot tell a host stall from
   a cost growing faster than linearly past a queue threshold, so any bound extrapolated past the
   measured depths through a per-launch cost is defeated by the second (Codex P1, rounds 1 and 2 on
   PR #846: the least [wall / depth] kept a threshold kernel's cheap single-launch ratio, and
   restricting its refutation to depths at or below its own was dodged by an over-target probe one
   rounding step above, at ~600 ms). A slow candidate therefore falls back to depth 1 -- measured
   exactly as [Isolated] measures it -- and a fast one whose deeper probes stalled keeps the deepest
   batch it measured within the target, still batching rather than turning isolated.

   Only ever shortens the depth; with no finite positive reading at all (a clock that resolved
   nothing) there is no evidence and the depth stands, as [queued_batch_depth] batches such
   estimates at the cap. *)
let wall_bounded_fallback_depth ~observed depth =
  if List.is_empty observed then depth
  else
    Int.min depth
    @@ List.fold observed ~init:1 ~f:(fun deepest (d, wall_ms) ->
        if Float.(wall_ms <= queued_batch_ms) then Int.max deepest d else deepest)

(* The rescue probe's depth (gh-ocannl-1098), for a calibration about to refuse a candidate whose
   single launch owed it a batch but which measured none within the target. The shallowest
   over-target batch [(d, w)] projected linearly to the target: [floor (d * target / w)], at least 2
   (depth 1 is the singles, already measured) and below [d]. For any cost whose per-launch average
   [wall / depth] does not fall with depth -- linear, a queue threshold, a superlinear queue -- that
   depth reads within the target, because its average is at most [w / d]. A cost whose average does
   fall is a fixed term the affine fits resolve, and never reaches the refusal. [None] when no depth
   lies strictly between 1 and [d]. Only the shallowest over-target batch is consulted: every deeper
   one bounds nothing below it, and the readings cannot tell a stall from a threshold, so the probe
   is the evidence rather than a further extrapolation. *)
let rescue_depth ~observed =
  let shallowest =
    List.fold observed ~init:None ~f:(fun acc (d, wall_ms) ->
        if d < 2 || Float.(wall_ms <= queued_batch_ms) then acc
        else
          match acc with
          | Some (d', w') when d' < d || (d' = d && Float.(w' <= wall_ms)) -> acc
          | _ -> Some (d, wall_ms))
  in
  Option.bind shallowest ~f:(fun (d, wall_ms) ->
      let projected = Float.of_int d *. queued_batch_ms /. wall_ms in
      let rescue = if Float.(projected >= of_int d) then d - 1 else Int.of_float projected in
      let rescue = Int.max 2 rescue in
      if rescue < d then Some rescue else None)

(* The search's test seams, per domain (the treatment [timing_call_observer] got, gh-ocannl-1263): a
   value set in one domain is what that domain's searches call, a search running in another domain
   neither reaches nor disturbs it, and a newly spawned domain starts with the default. Unlike the
   timing-call observers, a seam holds ONE value -- a later setting replaces it rather than chaining
   -- because several of them return or raise into the search. *)
module Seam = struct
  type 'a t = 'a Stdlib.Domain.DLS.key

  let make default : 'a t = Stdlib.Domain.DLS.new_key (fun () -> default)
  let get = Stdlib.Domain.DLS.get
  let set = Stdlib.Domain.DLS.set

  let with_value seam value body =
    let previous = get seam in
    set seam value;
    Exn.protect ~f:body ~finally:(fun () -> set seam previous)
end

(* gh-ocannl-1352: one search at a time, process-wide. The compile path a search drives was never
   made safe for two domains at once -- [Tnode]'s uid counter, [Low_level]'s analysis cache and
   pricing counter, [Context]'s print proxies, a device's constant buffer cache, Base hash tables
   throughout (the codegen census counters alone are per domain) -- so a second domain's search is
   refused, loudly and typed, rather than left to race them. The holder is a domain, and the nesting
   depth is per domain, so a search nested in its own domain's search ([Train.tune_placements]'s
   arms, a [?report] callback that tunes) proceeds, while one from any other domain is refused for
   as long as the holder's outermost search runs. *)
exception Concurrent_search of { holder : int; caller : int }

let () =
  Stdlib.Printexc.register_printer (function
    | Concurrent_search { holder; caller } ->
        Some
          (Printf.sprintf
             "Autotune.Concurrent_search: domain %d started a search while domain %d's is running; \
              searches run one at a time (gh-ocannl-1352)"
             caller holder)
    | _ -> None)

let search_holder : int option Stdlib.Atomic.t = Stdlib.Atomic.make None
let search_depth : int Stdlib.Domain.DLS.key = Stdlib.Domain.DLS.new_key (fun () -> 0)

let exclusive_search f =
  let depth = Stdlib.Domain.DLS.get search_depth in
  (if depth = 0 then
     let self = (Stdlib.Domain.self () :> int) in
     let rec acquire () =
       match Stdlib.Atomic.get search_holder with
       | Some holder -> raise (Concurrent_search { holder; caller = self })
       | None ->
           (* [None] is an immediate, so the physical comparison is the value's. *)
           if not (Stdlib.Atomic.compare_and_set search_holder None (Some self)) then acquire ()
     in
     acquire ());
  Stdlib.Domain.DLS.set search_depth (depth + 1);
  Exn.protect ~f ~finally:(fun () ->
      Stdlib.Domain.DLS.set search_depth depth;
      if depth = 0 then Stdlib.Atomic.set search_holder None)

(* Sibling fault-injection seam to [on_candidate_attempt], at a timing run's pre-dispatch validation
   rather than at a candidate's compile (gh-ocannl-564). Default no-op, no config key selects it.
   Needed because the causes this phase contains — an unsatisfied dependency, an out-of-range
   binding — belong to the lineage and the bindings, so a genuine one hits every candidate at once
   and cannot express "this one declined, the search went on". *)
let on_candidate_preflight : (string -> unit) Seam.t = Seam.make (fun _routine_name -> ())

(* Observation seam for the containment tests (gh-ocannl-898), fired with a routine's name exactly
   where a timing run's window yields an ADMITTED measurement — the moment the [candidates_timed]
   accounting grows, for the dispatched baseline and candidates alike — and with [timed_so_far], the
   search's own accounting value after this admission (what a report cut at this instant would state
   as [candidates_timed]). A preflight is upstream of that verdict: under the queued objective a
   preflighted run can still be refused (a contended window, a degenerate clock reading;
   gh-ocannl-855), so a fault-injection precondition of the form "this arm has timed N candidates"
   counted in preflights fires on an arm the report says timed nothing. The count is passed rather
   than left to the observer to re-derive, so such a precondition pins the tuner's number instead of
   maintaining a second counter that can drift from it (PR #593 P1). Default a no-op; no
   configuration selects it. *)
let on_candidate_timed : (string -> timed_so_far:int -> unit) Seam.t =
  Seam.make (fun _routine_name ~timed_so_far:_ -> ())

(* Fault-injection seam at the two post-admission callback boundaries (gh-ocannl-975). The measured
   candidate and incumbent times let a cleanup probe select an actual nonwinner, rather than
   guessing from attempt order. Default no-op; no configuration selects it. *)
let on_candidate_callback :
    ([ `Timed | `Calibration ] -> candidate_ms:float -> incumbent_ms:float -> unit) Seam.t =
  Seam.make (fun _ ~candidate_ms:_ ~incumbent_ms:_ -> ())

(* Measurement seam (gh-ocannl-1027): the one place a test can decide what an ADMITTED window
   measured. Without it, whether a search ever admits a candidate slower than its incumbent is a
   property of the machine's timings, so a regression about the nonwinner paths (gh-ocannl-975)
   could only re-roll the search and skip when every roll came out monotone. Called once per
   admitted timing window -- the dispatched baseline's and each candidate's -- by
   [apply_measurement_seam] below, before the admission gate that everything ranking or recording
   the time goes through. Default the identity; no configuration selects it. *)
let physical_measurement ~label:_ ~digest:_ ms = ms

let on_candidate_measured : (label:string -> digest:string -> float -> float) Seam.t =
  Seam.make physical_measurement

(* gh-ocannl-1527: whether an admitted window's reading is the window's own. A scripted measurement
   decides what a candidate measured after its window, so the window's physical samples say nothing
   about how it will rank, and a hopeless cut on them would drop candidates the script ranks. *)
let measurement_is_physical () = phys_equal (Seam.get on_candidate_measured) physical_measurement

(* Explicit test-only admission seam (gh-ocannl-1156). Scripted candidate measurements need their
   branches to run even when the host stalls every physical window. The [Cleared] scope clears ONLY
   the contention flag before candidate admission; calibration, dispatch, invalid clock readings and
   unresolved queued batching retain their ordinary rules. No configuration selects it. Production
   callers of [on_candidate_measured] still see only admitted windows. Restore the prior mode even
   on an injected failure, including when test scopes nest.

   [Forced] is the opposite scope (gh-ocannl-1517): it sets the contention flag on every window, so
   a test can drive the contention refusal's own accounting on any host, as [on_candidate_measured]
   drives the degenerate-reading one. Only the flag: a reading that is unreadable or unbatched keeps
   that cause, which [timing_admission] names first. *)
type test_window_contention = Physical | Cleared | Forced

let test_window_contention : test_window_contention Seam.t = Seam.make Physical
let with_uncontended_test_windows f = Seam.with_value test_window_contention Cleared f
let with_contended_test_windows f = Seam.with_value test_window_contention Forced f

(* The host-load signal a CPU backend's timing call reads around its window (gh-ocannl-1558), as a
   seam so a test supplies load evidence independently of its clock: a scripted quiet host admits a
   dispersed window, a scripted co-tenant refuses it, on any machine. Default {!Host_load.read}; no
   configuration selects another. *)
let host_load_reading : (unit -> Host_load.reading option) Seam.t = Seam.make Host_load.read

(* The scripted clock for an executed search (gh-ocannl-1558): applied by [time_routine] to every
   batch wall it measures -- warmup excluded -- given the batch's depth and a per-call ordinal
   counted from 0, so a test can make a real cc routine's windows dispersed and drive the ordinary
   admission path end to end. Default the identity; no configuration selects another. *)
let on_batch_wall : (depth:int -> ordinal:int -> float -> float) Seam.t =
  Seam.make (fun ~depth:_ ~ordinal:_ wall -> wall)

(* Unless the test-only scope above is active, only an admitted window reaches the seam; its result
   then passes the same [admitted_timing_ms] gate as a real reading, so a value that gate refuses
   (zero, negative, NaN, infinite) refuses the window exactly as a degenerate clock reading would --
   through the caller's ordinary refusal path, with nothing raised outside the search's cleanup
   boundaries. *)
let apply_measurement_seam ~label ~digest timing_result =
  let timing_result =
    match Seam.get test_window_contention with
    | Physical -> timing_result
    | Cleared -> { timing_result with contended = false }
    | Forced -> { timing_result with contended = true }
  in
  match admitted_timing_ms timing_result with
  | None -> timing_result
  | Some ms -> { timing_result with ms = Seam.get on_candidate_measured ~label ~digest ms }

(* Which of the calibration's branches dispatched a batch probe. *)
type calibration_probe_role =
  | Provisional_probe
  | Validation_probe
  | Confirmation_probe
  | Stall_retry_probe
  | Crossing_probe
  | Rescue_probe

(* One CUDA/HIP calibration batch probe as it ended (gh-ocannl-1119): the branch that started it,
   how many minima it took, the wall it charged to [queue_calibration_wall_ms] and its minimum. A
   test that saw only the [batch] calls had to reconstruct probes from runs of same-depth batches,
   which merged a stall retry into the confirmation it repeats and counted probes only as a lower
   bound. *)
type calibration_probe = {
  role : calibration_probe_role;
  depth : int;
  runs : int;
  wall_ms : float;
  min_ms : float;
}

type batch_decision = {
  phase : string option;
  timing : timing_mode;
  depth : int;
  cap : int;
  target_ms : float;
  estimated_wall_ms : float option;
  settle : batch_settle;
  fit : affine_fit option;
  cap_directed_probes : int;
  bounded_from : int option;
  fallback_from : int option;
  rescue : (int * float) option;
  probes : int;
  probe_wall_ms : float;
  budget_spent : bool;
  admission : batch_admission;
  retried : bool;
  host_load : Host_load.verdict option;
}

(* A window of the timed loop (gh-ocannl-994): how many batches it ran, counted by the loop itself
   rather than restated from its result, and their summed and median wall. A bound written on the
   reading needs THAT window, not the whole call: the call's wall also holds the warmup and the
   calibration's synchronized singles, and on a backend whose host round trip is two orders of
   magnitude above an amortized launch (Linux multidev_cc: ~57 us against ~0.44 us) those few dozen
   singles are a comparable share of the call to the tens of thousands of timed dispatches -- about
   40% of it -- so the whole-call mean is diluted by construction, and a host stall landing in the
   untimed part moves it without moving the reading. The MEDIAN accompanies the sum because the
   window's mean is still a stalled minority's to move: [contended] is declared on a MAJORITY of the
   window's batches exceeding twice its floor, so a bound meant to hold whenever the claim is not
   bypassed belongs on a statistic a minority cannot move. [reused] says how many of [samples] the
   loop did NOT dispatch (gh-ocannl-1074): a queued call that settles at depth 1 resumes the
   calibration's synchronized singles as its window, and those launches are already counted as
   calibration -- so the launches the loop itself dispatched are [depth * (samples - reused)].
   [spread] is the window's [timing_result.spread], from the same samples (gh-ocannl-1558). *)
type timed_window = {
  samples : int;
  reused : int;
  wall_ms : float;
  median_wall_ms : float;
  spread : float;
}

(* Everything one timing call did, reported once, as it returns (gh-ocannl-1263). It replaces four
   seams fired at different moments of the call -- the depth decision (gh-ocannl-851), each
   calibration probe (gh-ocannl-1119), a discarded contention window (gh-ocannl-1060) and the
   returned window (gh-ocannl-994) -- beside the decision record (gh-ocannl-1199), which their
   consumers had to stitch together by arrival order. *)
type timing_call = {
  decision : batch_decision;
  tagged : bool;
  calibration_singles : int;
  calibration_launches : int;
  calibration_probes : calibration_probe list;
  discarded_window : timed_window option;
  window : timed_window;
  setup_ms : float;
  timed_ms : float;
}

(* Trace-only observation of every timing call, per domain (gh-ocannl-1263) -- the treatment
   [autotune_log] got for searches in different domains: a search in another domain neither reaches
   nor disturbs this domain's observers. Observers compose, each installation calling the enclosing
   observer first. Observing changes nothing a search measures or ranks. Default a no-op; no
   configuration selects it. A search's own accounting goes through [time_routine]'s per-call
   [on_call] instead, so it never depends on who else observes. *)
let timing_call_observer : (timing_call -> unit) Stdlib.Domain.DLS.key =
  Stdlib.Domain.DLS.new_key (fun () _ -> ())

let chain_timing_call_observer f =
  let enclosing = Stdlib.Domain.DLS.get timing_call_observer in
  Stdlib.Domain.DLS.set timing_call_observer (fun call ->
      enclosing call;
      f call);
  enclosing

let install_timing_call_observer f = ignore (chain_timing_call_observer f : timing_call -> unit)

let with_timing_call_observer f body =
  let enclosing = chain_timing_call_observer f in
  Exn.protect ~f:body ~finally:(fun () -> Stdlib.Domain.DLS.set timing_call_observer enclosing)

(* gh-ocannl-1314: what one compile of a [tune] call cost, by phase. See the interface. *)
type compile_record = {
  compile_routine : string;
  compile_stage : string;
  compile_attempt : string;
  compile_verdict : string;
  compile_wall_s : float;
  compile_cost : Ir.Compile_cost.t;
}

(* Trace-only, per domain, composed like [timing_call_observer]. *)
let compile_observer : (compile_record -> unit) Stdlib.Domain.DLS.key =
  Stdlib.Domain.DLS.new_key (fun () _ -> ())

let install_compile_observer f =
  let enclosing = Stdlib.Domain.DLS.get compile_observer in
  (* Each observer is contained on its own: one that raises is reported and dropped for this record,
     and the observers after it still see it. *)
  let contained r =
    try f r
    with exn when not (process_fatal_exn exn) ->
      Stdio.eprintf "autotune: a compile observer raised (ignored): %s\n%!" (Exn.to_string exn)
  in
  Stdlib.Domain.DLS.set compile_observer (fun r ->
      enclosing r;
      contained r)

(* The measurement proper, after the warmup: the calibration and the timed loop, with the device
   reduced to [batch depth], which dispatches [depth] launches back to back, synchronizes once and
   returns the wall in milliseconds. Separated from [time_routine] so a test can drive the whole
   policy -- which depth a call settles on, which window it times, how many launches each costs --
   on an injected clock, with no device and no machine-dependent routine (gh-ocannl-1074). *)
let calibrate_and_time_from ~hopeless_against ~started ~tagged ~on_retry ~on_call ~phase
    ~retry_contended ~dispersion ~timing ~repeats ~queue_depth_cap ~batch =
  (* gh-ocannl-1527: the cut of a hopeless window, against the reading the caller ranks it by. *)
  let hopeless ?at_least_ms () =
    Option.map hopeless_against ~f:(fun reference_ms samples ->
        hopeless_window ?at_least_ms ~reference_ms samples)
  in
  (* The host-load reading the window's attribution starts from (gh-ocannl-1558): the span covers
     exactly the window's samples. Taken here, before the calibration's first single, for a depth-1
     window that resumes those singles; retaken just before a fresh window's first batch (below), so
     quiet calibration time never dilutes a co-tenant that arrived with the window. Read only under
     [Attribute_dispersion]. *)
  let read_load () =
    match dispersion with
    | Refuse_dispersed -> None
    | Attribute_dispersion -> Seam.get host_load_reading ()
  in
  let load_before = ref (read_load ()) in
  (* Every finite positive batch minimum the calibration measured, as [(depth, wall)]: the evidence
     that bounds an unresolved calibration's fallback depth. *)
  let observed = ref [] and no_supported_batch = ref false in
  let observe ~depth wall_ms =
    if Float.is_finite wall_ms && Float.is_positive wall_ms then
      observed := (depth, wall_ms) :: !observed
  in
  (* The probes' summed wall, against [queue_calibration_wall_ms]. Only finite positive readings are
     charged: a clock that resolved nothing spends no measurable wall, and must not end a probe on a
     NaN comparison. *)
  let probe_wall_ms = ref 0. and probes_started = ref 0 and probes_ended = ref [] in
  (* The decision metadata a [timing_call] record reports (gh-ocannl-1199): the latest affine fit
     and the validation probes a fit wanting the unmeasured cap directed (gh-ocannl-1144). *)
  let last_fit = ref None and cap_directed_probes = ref 0 in
  let refine ~min_depth ~base_depth ~base_ms ~probe_depth ~probe_ms =
    let depth, wall_ms, fit =
      fit_queued_batch_depth_between_with_cap ~min_depth ~max_depth:queue_depth_cap ~base_depth
        ~base_ms ~probe_depth ~probe_ms
    in
    last_fit := Some fit;
    (depth, wall_ms)
  in
  (* The deepest batch any probe dispatched, the synchronized singles as depth 1: what
     [queue_depth_projection_factor] multiplies. Every probe counts, whatever it read: a clock that
     resolved nothing still dispatched its provisional probe at the cap, which the bound must not
     shorten. *)
  let deepest_probed = ref 1 in
  (* The last of [queue_calibration_max_probes] is kept for the rescue, which checks no budget. *)
  let probe_budget_spent () =
    Float.(!probe_wall_ms >= queue_calibration_wall_ms)
    || !probes_started >= queue_calibration_max_probes - 1
  in
  let probe_batch ~role depth =
    Int.incr probes_started;
    deepest_probed := Int.max !deepest_probed depth;
    let best_ms = ref Float.infinity and runs = ref 0 and wall_ms = ref 0. in
    while
      !runs < queue_batch_probe_runs
      && (!runs < queue_batch_probe_min_runs || Float.(!wall_ms < queue_batch_probe_wall_ms))
    do
      let wall = batch depth in
      best_ms := Float.min !best_ms wall;
      if Float.is_finite wall && Float.is_positive wall then wall_ms := !wall_ms +. wall;
      Int.incr runs
    done;
    probe_wall_ms := !probe_wall_ms +. !wall_ms;
    observe ~depth !best_ms;
    probes_ended :=
      { role; depth; runs = !runs; wall_ms = !wall_ms; min_ms = !best_ms } :: !probes_ended;
    {
      ms = !best_ms;
      contended = false;
      unbatched = false;
      samples = !runs;
      dispersed = false;
      spread = Float.nan;
    }
  in
  (* [singles] is the calibration's window of synchronized single launches, kept rather than reduced
     to its minimum: they are depth-1 batches, sampled under the timed loop's own stopping rule, so
     when the depth settles at 1 they are the timed window's first samples (below). *)
  let ( calibration_dispatches,
        depth,
        estimated_batch_wall_ms,
        singles,
        (settle, bounded_from, fallback_from, rescue) ) =
    match timing with
    | Isolated -> (0, 1, None, [], (Isolated_objective, None, None, None))
    | Queued ->
        let singles =
          sample_window ~repeats:queue_calibration_runs
            ?hopeless:(hopeless ~at_least_ms:queued_batch_ms ())
            ~sample:(fun () ->
              let wall = batch 1 in
              { per_launch_ms = wall; contention_ms = wall })
            ()
        in
        let single_estimate = window_result singles in
        observe ~depth:1 single_estimate.ms;
        let calibration_dispatches, depth, estimated_batch_wall_ms, settled =
          if queue_depth_cap <> max_queue_depth then
            (* The affine refinement repairs the CUDA/HIP dispatch-scale defect in gh-ocannl-892.
               Preserve the historical single-estimate policy on cc and Metal: Metal's measured
               depth is already below the old cap, while extra probes and a deeper timed queue on cc
               multiplied the repository's CPU autotune-suite wall past its CI ceiling. *)
            ( single_estimate.samples,
              queued_batch_depth_with_cap ~max_depth:queue_depth_cap single_estimate,
              None,
              (Single_estimate, None, None, None) )
          else if Float.(single_estimate.ms >= slow_singles_ratio *. queued_batch_ms) then
            (* gh-ocannl-1627: singles this far over the target settle at depth 1 whatever a depth-2
               probe reads, short of a fixed synchronization term of at least the target
               ({!slow_singles_ratio}); their window is the reading, as at every depth-1 settle. *)
            ( single_estimate.samples,
              1,
              Some single_estimate.ms,
              (Singles_far_over_target, None, None, None) )
          else
            let provisional_depth =
              queued_batch_depth_with_cap ~max_depth:queue_depth_cap single_estimate
            in
            let min_depth = if provisional_depth > 1 then 2 else 1 in
            (* Depth 1 is probed at depth 2 before it is retained: a genuinely slow routine's affine
               pair confirms depth 1, while an inflated synchronized-single window enters the same
               retry path as every other suspect target crossing. *)
            let probe_depth = Int.max 2 provisional_depth in
            (* A synchronized single dispatch includes the round trip queueing is meant to amortize,
               so its estimate safely seeds a probe but is not the batch's steady-state per-launch
               cost. On HIP's STREAM kernels it selected 105--209 launches whose actual batch wall
               was only 1.3--2.6 ms. Measure that provisional queue and split its wall into fixed
               synchronization and marginal launch costs; their affine model selects the depth whose
               whole batch reaches the target even when the provisional depth is shallow. Budget the
               probe on batch wall rather than per-launch time: it is calibration, not a candidate
               measurement, and need not spend 64 whole batches to learn the scale. *)
            let probe = probe_batch ~role:Provisional_probe probe_depth in
            let depth, estimated_batch_wall_ms =
              (* [probe_depth] is at least 2, so this is [refine_queued_batch_depth_with_cap]'s
                 affine pair, recorded. *)
              refine ~min_depth:1 ~base_depth:1 ~base_ms:single_estimate.ms ~probe_depth
                ~probe_ms:probe.ms
            in
            let confirm_interpolated calibration_dispatches depth ~upper_depth ~upper_ms =
              if probe_budget_spent () then
                (calibration_dispatches, depth, Float.nan, Budget_exhausted)
              else
                let measured = probe_batch ~role:Crossing_probe depth in
                let calibration_dispatches = calibration_dispatches + (measured.samples * depth) in
                let confirmed_depth, confirmed_wall_ms =
                  refine ~min_depth ~base_depth:depth ~base_ms:measured.ms ~probe_depth:upper_depth
                    ~probe_ms:upper_ms
                in
                if confirmed_depth = depth then
                  (calibration_dispatches, depth, measured.ms, Measured_batch)
                else if Float.is_finite confirmed_wall_ms && confirmed_depth > upper_depth then
                  (* A resolved fit can project past the upper batch: a fixed-dominated pair targets
                     a batch of marginal work, far deeper than either point (Codex P1, round 2 on PR
                     #847). That depth is unmeasured and beyond every measured one, where a queue
                     cost may jump, and nothing would check it -- neither the wall budget nor the
                     fallback. Stay with the measured upper batch instead. *)
                  (calibration_dispatches, upper_depth, upper_ms, Measured_batch)
                else if Float.is_finite confirmed_wall_ms then
                  (calibration_dispatches, confirmed_depth, confirmed_wall_ms, Affine_projection)
                else if Float.(upper_ms >= queued_batch_ms && upper_ms >= measured.ms) then
                  (calibration_dispatches, queue_depth_cap, Float.nan, Cap_unresolved)
                else
                  let depth, wall_ms =
                    depth_from_batch_wall_with_cap ~max_depth:queue_depth_cap ~depth:upper_depth
                      ~wall_ms:upper_ms
                  in
                  (calibration_dispatches, depth, wall_ms, Wall_scaled)
            in
            (* A resolved pair whose target crossing is shallower than the batch it was measured
               against (gh-ocannl-1098): sample the crossing and fit it against the shallowest
               measured batch above it, exactly as an interpolation inside a bracket is checked.
               Depth 1 needs no probe -- the synchronized singles are its measurement. *)
            let settle_shallower calibration_dispatches depth ~upper_depth ~upper_ms =
              if depth = 1 then (calibration_dispatches, 1, single_estimate.ms, Measured_batch)
              else confirm_interpolated calibration_dispatches depth ~upper_depth ~upper_ms
            in
            let rec confirm_or_scale ?(retry_stall = true) calibration_dispatches depth wall_ms =
              if depth = queue_depth_cap then
                (* The cap itself cannot provide a depth-separated confirmation. Its directly
                   measured wall is still the best scale evidence; repeating the same depth only
                   spends another queue and can replace that observation with [nan] on noise. *)
                (calibration_dispatches, depth, wall_ms, Cap_measured)
              else if probe_budget_spent () then
                (calibration_dispatches, depth, Float.nan, Budget_exhausted)
              else
                let confirmation_depth = Int.min queue_depth_cap (depth + Int.max 1 (depth / 4)) in
                let confirmation =
                  probe_batch
                    ~role:(if retry_stall then Confirmation_probe else Stall_retry_probe)
                    confirmation_depth
                in
                let calibration_dispatches =
                  calibration_dispatches + (confirmation.samples * confirmation_depth)
                in
                let confirmed_depth, confirmed_wall_ms =
                  refine ~min_depth ~base_depth:depth ~base_ms:wall_ms
                    ~probe_depth:confirmation_depth ~probe_ms:confirmation.ms
                in
                if confirmed_depth = depth then
                  (calibration_dispatches, depth, wall_ms, Measured_batch)
                else if confirmed_depth < depth && Float.is_finite confirmed_wall_ms then
                  settle_shallower calibration_dispatches confirmed_depth ~upper_depth:depth
                    ~upper_ms:wall_ms
                else if
                  Float.is_nan confirmed_wall_ms
                  && Float.(confirmation.ms >= queued_batch_ms && confirmation.ms >= wall_ms)
                then
                  if retry_stall && Float.(confirmation.ms > wall_ms * contention_ratio) then
                    (* This confirmation is itself a contention outlier against the supported
                       target-sized base. Retry the same depth once: jumping straight to the cap
                       would inflate the timed batch and its 2x refusal threshold precisely when
                       calibration observed the stall that rule is meant to reject. *)
                    confirm_or_scale ~retry_stall:false calibration_dispatches depth wall_ms
                  else (calibration_dispatches, queue_depth_cap, Float.nan, Cap_unresolved)
                else
                  let depth, wall_ms =
                    depth_from_batch_wall_with_cap ~max_depth:queue_depth_cap
                      ~depth:confirmation_depth ~wall_ms:confirmation.ms
                  in
                  (calibration_dispatches, depth, wall_ms, Wall_scaled)
            in
            let rec validate_depth probes_left calibration_dispatches base_depth base_ms depth
                estimated_wall_ms =
              if depth = queue_depth_cap && !deepest_probed = queue_depth_cap then
                (calibration_dispatches, depth, estimated_wall_ms, Cap_measured)
              else if probe_budget_spent () then
                (calibration_dispatches, depth, Float.nan, Budget_exhausted)
              else
                (* A fit wanting the cap is not evidence at the cap (gh-ocannl-1144). Walk there
                   through measured doublings while validation probes remain, including when the
                   provisional pair already projects to the cap. Each probe spends the same wall and
                   count budgets as every other validation. *)
                let depth =
                  if depth = queue_depth_cap then (
                    Int.incr cap_directed_probes;
                    Int.min queue_depth_cap (queue_depth_projection_factor * !deepest_probed))
                  else depth
                in
                let validation = probe_batch ~role:Validation_probe depth in
                let calibration_dispatches =
                  calibration_dispatches + (validation.samples * depth)
                in
                let next_depth, next_wall_ms =
                  refine ~min_depth ~base_depth ~base_ms ~probe_depth:depth ~probe_ms:validation.ms
                in
                if next_depth = base_depth then
                  (* A valid pair can confirm its earlier, already-target-sized observation. Stop
                     there: probing that shallower depth again would reverse the refinement order
                     and can oscillate until unrelated noise forces the cap. *)
                  (calibration_dispatches, base_depth, base_ms, Measured_batch)
                else if Float.(base_ms < queued_batch_ms && validation.ms >= queued_batch_ms) then
                  if next_depth < depth then
                    (* The measured pair brackets the target. Interpolate inside that bracket before
                       confirming: retaining an overshooting validation would turn a well-resolved
                       10 ms target into a 20 ms batch and blunt the 2x rule. *)
                    if probes_left <= 1 then
                      confirm_interpolated calibration_dispatches next_depth ~upper_depth:depth
                        ~upper_ms:validation.ms
                    else
                      validate_depth (probes_left - 1) calibration_dispatches base_depth base_ms
                        next_depth next_wall_ms
                  else confirm_or_scale calibration_dispatches depth validation.ms
                else if next_depth < depth && Float.is_finite next_wall_ms then
                  (* Past the bracket, so the base is itself over the target and the resolved pair
                     puts the crossing below the validation, possibly below the base too. Probing it
                     against the deeper point would reverse the pair's order and read as unresolved,
                     doubling back up (gh-ocannl-1098). *)
                  if next_depth < base_depth then
                    settle_shallower calibration_dispatches next_depth ~upper_depth:base_depth
                      ~upper_ms:base_ms
                  else
                    settle_shallower calibration_dispatches next_depth ~upper_depth:depth
                      ~upper_ms:validation.ms
                else if next_depth = queue_depth_cap && depth < queue_depth_cap && probes_left > 1
                then
                  validate_depth (probes_left - 1) calibration_dispatches depth validation.ms
                    next_depth next_wall_ms
                else if next_depth = queue_depth_cap then
                  if
                    Float.is_nan next_wall_ms
                    && Float.(validation.ms >= queued_batch_ms && validation.ms >= base_ms)
                  then (calibration_dispatches, queue_depth_cap, Float.nan, Cap_unresolved)
                  else if Float.is_nan next_wall_ms then
                    let depth, wall_ms =
                      depth_from_batch_wall_with_cap ~max_depth:queue_depth_cap ~depth
                        ~wall_ms:validation.ms
                    in
                    (calibration_dispatches, depth, wall_ms, Wall_scaled)
                  else (calibration_dispatches, next_depth, next_wall_ms, Affine_projection)
                else if probes_left <= 1 && Float.is_nan next_wall_ms then
                  (* Do not fall back to the earlier target crossing: it may be the inflated window
                     this validation was meant to expose. Scale from the deepest measured batch when
                     it is non-monotone; a monotone, fixed-dominated pair still binds at the cap. *)
                  if Float.(validation.ms >= queued_batch_ms && validation.ms >= base_ms) then
                    (calibration_dispatches, queue_depth_cap, Float.nan, Cap_unresolved)
                  else
                    let depth, wall_ms =
                      depth_from_batch_wall_with_cap ~max_depth:queue_depth_cap ~depth
                        ~wall_ms:validation.ms
                    in
                    (calibration_dispatches, depth, wall_ms, Wall_scaled)
                else if probes_left <= 1 then
                  (* Keep the latest supported affine projection after the bounded validation loop.
                     Jumping to the cap here would turn a noisy near-target probe into a 20--30 ms
                     batch whose 2x contention threshold no longer catches the fixed host stall this
                     policy exists to detect. *)
                  (calibration_dispatches, next_depth, next_wall_ms, Affine_projection)
                else
                  validate_depth (probes_left - 1) calibration_dispatches depth validation.ms
                    next_depth next_wall_ms
            in
            let calibration_dispatches = single_estimate.samples + (probe.samples * probe_depth) in
            let calibration_dispatches, depth, estimated_batch_wall_ms, settle =
              if provisional_depth = 1 && depth = 1 then
                (calibration_dispatches, 1, single_estimate.ms, Singles_reach_target)
              else if Float.(probe.ms >= queued_batch_ms) && depth < probe_depth then
                validate_depth max_depth_validation_probes calibration_dispatches 1
                  single_estimate.ms depth estimated_batch_wall_ms
              else if depth = probe_depth && Float.(probe.ms >= queued_batch_ms) then
                confirm_or_scale calibration_dispatches probe_depth probe.ms
              else
                validate_depth max_depth_validation_probes calibration_dispatches probe_depth
                  probe.ms depth estimated_batch_wall_ms
            in
            (* Every exit above, bounded at once rather than branch by branch (gh-ocannl-1100): no
               settle goes past [queue_depth_projection_factor] times the deepest probe. The wall
               estimate keeps its per-launch share; an unresolved one stays unresolved, for the
               stricter fallback below. *)
            let bounded_from = depth in
            let depth, estimated_batch_wall_ms =
              let bound = queue_depth_projection_factor * !deepest_probed in
              if depth <= bound then (depth, estimated_batch_wall_ms)
              else (
                logf
                  "queued batch depth %d projected past the deepest calibration probe (%d): \
                   settled at %d"
                  depth !deepest_probed bound;
                (bound, estimated_batch_wall_ms *. Float.of_int bound /. Float.of_int depth))
            in
            (* An unresolved outcome is the only one whose depth no measured wall or fit supports;
               every resolved one carries its own wall estimate. *)
            let fallback_from = depth in
            let depth =
              if Float.is_nan estimated_batch_wall_ms then
                wall_bounded_fallback_depth ~observed:!observed depth
              else depth
            in
            (* What the two bounds changed, for the trace: the depth each started from, when it
               moved. *)
            let bounded_from = Option.some_if (fallback_from < bounded_from) bounded_from
            and fallback_from = Option.some_if (depth < fallback_from) fallback_from in
            (* A candidate whose single launch owed it a batch, but whose every batched probe read
               over the target, has no depth this calibration can time as [Queued]: the fallback
               leaves only depth 1, whose reading is the isolated objective, and timing it would
               crown under the wrong objective silently (Codex P1, round 3 on PR #846). The readings
               cannot tell that from a stall. Before refusing, one rescue probe below the shallowest
               over-target batch ({!rescue_depth}): a kernel with a genuine queue threshold below
               its provisional depth is timed there rather than refused on every search, since its
               refusal made it permanently uncacheable. A probe that also reads over the target
               leaves the call refused, not ranked, not cached, retried by a later search -- under a
               reason of its own ([unbatched]): not a diagnosis, since host load stalling every
               batched probe reads the same, but what the calibration measured, which a threshold
               repeats on every rerun and a stall does not. Any settle at depth 1 counts, not only
               the unresolved one: a resolved fit that puts the crossing at one launch contradicts
               the singles just the same. *)
            let calibration_dispatches, depth, estimated_batch_wall_ms, rescue =
              if depth = 1 && provisional_depth > 1 then
                match rescue_depth ~observed:!observed with
                | None -> (calibration_dispatches, depth, estimated_batch_wall_ms, None)
                | Some rescue ->
                    (* One rescue probe before refusing (gh-ocannl-1098), charged to no budget: its
                       depth is chosen so that a cost whose per-launch average does not fall with
                       depth reads within the target there, which is a bounded wall. *)
                    let measured = probe_batch ~role:Rescue_probe rescue in
                    let calibration_dispatches =
                      calibration_dispatches + (measured.samples * rescue)
                    in
                    let probed = Some (rescue, measured.ms) in
                    if
                      Float.is_finite measured.ms && Float.is_positive measured.ms
                      && Float.(measured.ms <= queued_batch_ms)
                    then (calibration_dispatches, rescue, measured.ms, probed)
                    else (calibration_dispatches, depth, estimated_batch_wall_ms, probed)
              else (calibration_dispatches, depth, estimated_batch_wall_ms, None)
            in
            if depth = 1 && provisional_depth > 1 then no_supported_batch := true;
            ( calibration_dispatches,
              depth,
              Some estimated_batch_wall_ms,
              (settle, bounded_from, fallback_from, rescue) )
        in
        (calibration_dispatches, depth, estimated_batch_wall_ms, singles, settled)
  in
  Option.iter estimated_batch_wall_ms ~f:(fun estimated_wall_ms ->
      if depth = queue_depth_cap then
        if Float.is_finite estimated_wall_ms && Float.is_positive estimated_wall_ms then (
          if Float.(estimated_wall_ms < queued_batch_ms) then
            logf
              "queued batch capped at depth %d: estimated wall %.4f ms, %.4f ms short of the %.1f \
               ms target"
              depth estimated_wall_ms
              (queued_batch_ms -. estimated_wall_ms)
              queued_batch_ms)
        else
          logf
            "queued batch capped at depth %d: batch wall estimate is unresolved, so the shortfall \
             from the %.1f ms target cannot be quantified"
            depth queued_batch_ms);
  let setup_ms = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e6
  and decided = Mtime_clock.counter () in
  (* The calibration's own contention verdict is not consulted (gh-ocannl-888): it judged single
     dispatches, and the window that gets judged for refusal is the batch below.

     At depth 1 that batch IS a synchronized single, so the calibration has already sampled the very
     quantity the timed loop would measure, under the same stopping rule -- which is also exactly
     what [Isolated] measures: a warmup, then a min-of-N over single launches. Timing a fresh
     depth-1 window re-measured it, and on a search dominated by candidates slower than the batch
     target that repeat was a large share of the session's wall (gh-ocannl-1074: 215 of 220 calls of
     a gfx1151 gpt2_mini search settled at depth 1). So the window resumes from the singles, topping
     up only as far as the caller's [repeats] floor asks beyond the calibration's, and the resumed
     window is judged for contention whole, exactly as a fresh one would be. At any other depth the
     singles are a different quantity from the batch and are left out.

     What reuse gives up is independence between the depth decision and the reading (Codex P2, round
     1 on PR #839): depth 1 is chosen because these singles' minimum reached the batch target, so a
     depth-1 reading cannot fall below the target the way an independent window's could. That
     conditioning moves a reading only for a candidate whose min-of-16 straddles the target, and
     only by less than that minimum's own spread -- the band in which an independent window's
     ordering was already decided by jitter. The reading has a larger discontinuity at the same
     place anyway: a candidate whose minimum falls just short settles at depth 2 and reads the
     batched, amortized cost. The candidates reuse exists for sit far above the target (55-70 ms
     gpt2_mini steps against 10 ms), where the condition holds with or without the conditioning. *)
  let time_window reused =
    let timed_wall_ms = ref (List.fold reused ~init:0. ~f:(fun total s -> total +. s.contention_ms))
    and timed_batches = ref (List.length reused)
    and timed_walls = ref (List.rev_map reused ~f:(fun s -> s.contention_ms)) in
    let result =
      window_result
      @@ sample_window ~prior:reused ?hopeless:(hopeless ()) ~repeats
           ~sample:(fun () ->
             let wall = batch depth in
             timed_wall_ms := !timed_wall_ms +. wall;
             timed_walls := wall :: !timed_walls;
             Int.incr timed_batches;
             { per_launch_ms = wall /. Float.of_int depth; contention_ms = wall })
           ()
    in
    let median_wall_ms = if List.is_empty !timed_walls then 0. else median_float !timed_walls in
    ( result,
      {
        samples = !timed_batches;
        reused = List.length reused;
        wall_ms = !timed_wall_ms;
        median_wall_ms;
        spread = result.spread;
      } )
  in
  let reused = if depth = 1 then singles else [] in
  if List.is_empty reused then load_before := read_load ();
  let ((first, discarded) as window) = time_window reused in
  let retried =
    retry_contended && Poly.equal timing Queued && first.contended && Float.is_finite first.ms
    && Float.is_positive first.ms
  in
  let (result, window), discarded_window =
    if retried then (
      (* Keep the depth and the 2x-majority rule. A fresh independent window can outlast a transient
         Metal scheduler burst; mixing windows would instead dilute its refusal evidence. Depth-1
         retries must likewise dispatch fresh singles rather than resume the refused ones. *)
      logf "queued timing retry: contention refused the first window (%d samples)" first.samples;
      on_retry ();
      (* A fresh window gets fresh load evidence: the refused one's must not vouch for it. *)
      load_before := read_load ();
      (time_window [], Some discarded))
    else (window, None)
  in
  let result =
    if !no_supported_batch then (
      logf
        "queued timing refused: every batched calibration probe read over the %.1f ms target, so \
         only an isolated depth-1 reading was left"
        queued_batch_ms;
      { result with unbatched = true })
    else result
  in
  (* gh-ocannl-1558: the attribution of a dispersed window. For a dispersed window shorter than
     [Host_load.min_span_s] the second reading waits out the remainder, so the tick-sampled busy
     counter covers a span it can resolve; the process is idle meanwhile, so the wait measures the
     co-tenant and not this routine. Every foreign CPU-second of the padded span is then charged to
     the window itself (worst case, [~window_s]): the padding can add evidence, never dilute a
     co-tenant that ran only during the window. *)
  let host_load =
    match dispersion with
    | Refuse_dispersed -> None
    | Attribute_dispersion ->
        Some
          (match !load_before with
          | None -> Host_load.Unavailable
          | Some before -> (
              let after, window_s =
                match read_load () with
                | Some after
                  when result.dispersed
                       && Float.(after.wall_s -. before.wall_s < Host_load.min_span_s) ->
                    let window_s = after.wall_s -. before.wall_s in
                    Unix.sleepf (Host_load.min_span_s -. window_s);
                    (read_load (), Some window_s)
                | after -> (after, None)
              in
              match after with
              | None -> Host_load.Unavailable
              | Some after -> Host_load.classify ?window_s ~before ~after ()))
  in
  let result =
    match host_load with
    | None -> result
    | Some verdict ->
        let attributed = attribute_dispersion verdict result in
        (if result.dispersed then
           match verdict with
           | Quiet foreign when not attributed.contended ->
               logf
                 "admitted a dispersed window (spread %.2fx, foreign load %.2f cores): the \
                  routine's own variance, not a co-tenant (gh-ocannl-1558)"
                 result.spread foreign
           | Cotenant foreign ->
               logf
                 "a dispersed window (spread %.2fx) under a co-tenant (foreign load %.2f cores): \
                  refused as host contention"
                 result.spread foreign
           | Quiet _ | Unavailable -> ());
        attributed
  in
  let decision =
    {
      phase;
      timing;
      depth;
      cap = queue_depth_cap;
      target_ms = queued_batch_ms;
      estimated_wall_ms = estimated_batch_wall_ms;
      settle;
      fit = !last_fit;
      cap_directed_probes = !cap_directed_probes;
      bounded_from;
      fallback_from;
      rescue;
      probes = !probes_started;
      probe_wall_ms = !probe_wall_ms;
      budget_spent = probe_budget_spent ();
      admission = timing_admission result;
      retried;
      host_load;
    }
  in
  let call =
    {
      decision;
      tagged;
      calibration_singles = List.length singles;
      calibration_launches = calibration_dispatches;
      calibration_probes = List.rev !probes_ended;
      discarded_window;
      window;
      setup_ms;
      timed_ms = Mtime.Span.to_float_ns (Mtime_clock.count decided) /. 1e6;
    }
  in
  on_call call;
  Stdlib.Domain.DLS.get timing_call_observer call;
  result

(* Exposed for tests with an injected clock: the per-call observers [time_routine] gives a search,
   timed from this call's own start, and the dispersion policy it picks by backend. *)
let calibrate_and_time_with_retry_observer ~hopeless_against ~on_retry ~on_call ~phase
    ~retry_contended ~dispersion ~timing ~repeats ~queue_depth_cap ~batch =
  calibrate_and_time_from ~hopeless_against ~started:(Mtime_clock.counter ()) ~tagged:false
    ~on_retry ~on_call ~phase ~retry_contended ~dispersion ~timing ~repeats ~queue_depth_cap ~batch

(* Preserve the original all-labelled helper's source interface: an optional observer would not
   erase without adding a positional argument. *)
let calibrate_and_time ~retry_contended ~timing ~repeats ~queue_depth_cap ~batch =
  calibrate_and_time_with_retry_observer ~hopeless_against:None
    ~on_retry:(fun () -> ())
    ~on_call:(fun _ -> ())
    ~phase:None ~retry_contended ~dispersion:Refuse_dispersed ~timing ~repeats ~queue_depth_cap
    ~batch

(* gh-ocannl-1556: raised by [time_routine] when its [stop] answers [true] -- before the warmup, or
   before a batch -- abandoning the call with nothing dispatched past the last synchronized
   batch. *)
exception Timing_stopped

(* [routine.bindings] exposes the routine's live binding refs — restore them after timing (Codex P2
   on PR #103), or the returned winner would stay bound to the tuner's midpoint test values. *)
let time_routine ?(on_retry = fun () -> ()) ?(on_call = fun _ -> ()) ?(tag_failures = false) ?phase
    ?(stop = fun () -> false) ?hopeless_against ~timing ~repeats cctx routine =
  let check_stop () = if stop () then raise Timing_stopped in
  (* The call's [setup_ms] runs from here: the pre-dispatch validation, the warmup and the
     calibration. *)
  let started = Mtime_clock.counter () in
  let saved_bindings = List.map routine.Context.bindings ~f:(fun (_ss, r) -> (r, !r)) in
  let run ctx =
    if tag_failures then Outcome.tag Outcome.Launch (fun () -> Context.run ctx routine)
    else Context.run ctx routine
  in
  let sync ctx =
    if tag_failures then Outcome.tag Outcome.Sync (fun () -> Context.sync ctx) else Context.sync ctx
  in
  Exn.protect
    ~finally:(fun () -> List.iter saved_bindings ~f:(fun (r, v) -> r := v))
    ~f:(fun () ->
      set_test_bindings routine;
      (* The runs' pre-dispatch validation, in its own phase so an unattributed failure of it is
         contained rather than condemning the lineage (gh-ocannl-564). Here and once: what it checks
         (lineage, initialized nodes, dependencies, the bindings just written) is settled before the
         warmup and only becomes more satisfied as the loop dispatches. [Context.run] re-validates
         per iteration inside the [Launch] tag, where it can no longer fail. *)
      (* Only the PER-CANDIDATE half of the pre-dispatch validation is contained here. The
         lineage-wide half ({!Context.check_lineage_runnable}) is run by the callers below, outside
         their failure boundaries, because it fails every candidate of every arm identically —
         see the comments at those two sites (gh-ocannl-569). *)
      if tag_failures then
        Outcome.tag Outcome.Preflight (fun () ->
            Seam.get on_candidate_preflight routine.Context.name;
            Context.check_launch_bindings routine);
      (* Warmup run: absorbs lazy initialization and fills caches like a steady-state iteration. *)
      check_stop ();
      let ctx = ref (run cctx) in
      sync !ctx;
      (* Monotonic high-resolution clock: on Windows, [Unix.gettimeofday] ticks at ~1 ms, which
         makes sub-millisecond candidates indistinguishable (they all measure 0). *)
      let ordinal = ref 0 in
      let batch depth =
        check_stop ();
        let c0 = Mtime_clock.counter () in
        for _ = 1 to depth do
          ctx := run !ctx
        done;
        sync !ctx;
        let wall = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
        let wall = Seam.get on_batch_wall ~depth ~ordinal:!ordinal wall in
        Int.incr ordinal;
        wall
      in
      calibrate_and_time_from ~hopeless_against ~started ~tagged:tag_failures ~on_retry ~on_call
        ~phase ~timing ~repeats
        ~retry_contended:(retry_contended_window_for_backend (Context.backend_name cctx))
        ~dispersion:(dispersion_policy_for_backend (Context.backend_name cctx))
        ~queue_depth_cap:(queue_depth_cap_for_backend (Context.backend_name cctx))
        ~batch)

(* gh-ocannl-532: on a GPU backend, code that binds no hardware dimension runs the whole routine in
   a single work-item — every nest a serial scalar loop, at one lane's throughput. Such a candidate
   cannot win a search whose other candidates are parallel, so dispatching it is pure cost, and the
   cost is unbounded: a training step of a few GFLOP is minutes to hours per run, and [time_routine]
   does four of them (a warmup plus [autotune_repeats]). The dispatch is also uninterruptible and
   shares the device with the display — the sessions in gh-ocannl-532 produced driver-timeout
   reports and, once, loss of display output. So an unparallelized GPU candidate is never
   dispatched: not timed, and not eligible to win. This covers the identity-transform serial
   baseline, which is where it bites (the default annotator that parallelizes an untuned compile is
   bypassed whenever a [?lowered_transform] is supplied, so the tuner's base compile is always the
   unscheduled form). On CPU backends the serial form runs at full single-core speed and stays a
   legitimate competitor — the rule is GPU-only. *)
let binds_hardware_dims (opt : LL.optimized) = not (List.is_empty (LL.hardware_axes opt.LL.llc))

(* A candidate is dispatchable when it is on a CPU backend, or at least one of its kernels binds a
   hardware dimension. Whole-candidate rather than per-kernel: a fissioned candidate legitimately
   leaves small segments serial next to parallel ones, and only an entirely serial routine has the
   unbounded single-work-item cost. *)
let dispatchable ~is_gpu (opts : LL.optimized list) =
  (not is_gpu) || List.exists opts ~f:binds_hardware_dims

let axis_type_is_hardware = function
  | LL.Grid | LL.Workgroup | LL.Workgroup_reduce -> true
  | LL.Serial | LL.Unrolled | LL.Vectorized -> false

(* Whether a menu move could turn a form that binds no hardware dimension into one that does. Only
   two families can: a placement retype (or a [Split] whose halves are hardware-typed), and
   [Tensorize], whose lane loop is a fresh [Workgroup] axis — which is exactly the move the seeding
   comments call the beam's one path out of the serial baseline. The moves [menu] actually proposes
   otherwise rewrite serial loops into serial loops ([Split] Serial/Serial, [Swap], [Unroll],
   [Retype] to [Vectorized]), so extending an undispatchable incumbent with them yields another
   undispatchable candidate — provable without compiling it (gh-ocannl-543). Families [menu] does
   not emit answer [true]: not pruning is the conservative side, so a future menu addition is never
   silently dropped. *)
let optop_can_bind_hardware (op : SC.saved_optop) =
  match op with
  | SC.Split { outer; inner; _ } -> axis_type_is_hardware outer || axis_type_is_hardware inner
  | SC.Retype { ty; _ } -> axis_type_is_hardware ty
  | SC.Swap _ | SC.Unroll _ -> false
  | SC.Tensorize _ | SC.Partition _ | SC.Pad _ | SC.Coalesce _ | SC.Stage _ | SC.Privatize _
  | SC.Expand_zero _ | SC.Fuse_epilogue _ | SC.Split_reduce _ | SC.Fold_mma _ ->
      true

let optop_family (op : SC.saved_optop) =
  match op with
  | SC.Split _ -> "Split"
  | SC.Swap _ -> "Swap"
  | SC.Retype _ -> "Retype"
  | SC.Unroll _ -> "Unroll"
  | SC.Partition _ -> "Partition"
  | SC.Pad _ -> "Pad"
  | SC.Coalesce _ -> "Coalesce"
  | SC.Stage _ -> "Stage"
  | SC.Privatize _ -> "Privatize"
  | SC.Expand_zero _ -> "Expand_zero"
  | SC.Tensorize _ -> "Tensorize"
  | SC.Fuse_epilogue _ -> "Fuse_epilogue"
  | SC.Fold_mma _ -> "Fold_mma"
  | SC.Split_reduce _ -> "Split_reduce"

(** {2 The privatized fission flavor}

    A variant of the per-segment preset that contracts each materialized read-modify-write
    accumulator into a per-thread register tile ({!Sched.optop.Privatize}) over its serial reduction
    loop. A routine-local accumulator beats a device-memory RMW on every backend, and on Metal it
    additionally sidesteps the volatile-RMW miscompile workaround tax (c_syntax.ml
    [volatile_serial_accumulation]). Detection is permissive: each proposal is validated by
    try-applying against the segment (Privatize's own preconditions — single index vector, uniform
    iteration-invariant guards, etc.), and dropped rather than failing the candidate. *)

let rec subtree_has_hardware_loop (llc : LL.t) =
  match llc with
  | LL.For_loop { axis = LL.Grid | LL.Workgroup | LL.Workgroup_reduce; _ } -> true
  | LL.For_loop { body; _ } -> subtree_has_hardware_loop body
  | LL.Seq (a, b) -> subtree_has_hardware_loop a || subtree_has_hardware_loop b
  | LL.If { body; _ } -> subtree_has_hardware_loop body
  | _ -> false

(* Materialized RMW accumulation sites of the (post-preset) scheduled segment, each paired with the
   outermost enclosing Serial loop eligible to privatize over: the access vector must not mention
   its symbol (so the accumulation is carried across it), and no hardware-typed loop may sit inside
   its subtree (the private tile is per-thread; spanning other threads' iterations would store back
   their elements). *)
let privatize_proposals (post : LL.optimized) : (Ir.Tnode.t * Idx.symbol) list =
  let plc = post.LL.optimize_ctx.LL.placements in
  let proposals = ref [] in
  let rec walk stack (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        walk stack a;
        walk stack b
    | LL.If { body; _ } -> walk stack body
    | LL.For_loop { index; from_; body; axis; _ } -> walk ((index, from_, axis, body) :: stack) body
    | LL.Set { tn; idcs; llsc; _ }
      when Ir.Tnode.Placements.is_materialized_peek plc tn
           && List.exists (collect_gets llsc) ~f:(fun (t, i) ->
               phys_equal t tn && Array.equal Idx.equal_axis_index i idcs) ->
        List.find (List.rev stack) ~f:(fun (index, from_, axis, body) ->
            LL.equal_axis_type axis LL.Serial && from_ = 0
            && (not (Array.exists idcs ~f:(Idx.axis_index_mentions_symbol index)))
            && not (subtree_has_hardware_loop body))
        |> Option.iter ~f:(fun (index, _, _, _) ->
            if
              not
                (List.exists !proposals ~f:(fun (t, s) ->
                     Ir.Tnode.equal t tn && Idx.equal_symbol s index))
            then proposals := (tn, index) :: !proposals)
    | _ -> ()
  in
  walk [] post.LL.llc;
  List.rev !proposals

type privatize_drop = { pd_target : Ir.Tnode.t; pd_over : Idx.symbol; pd_cause : Outcome.cause }

(* gh-ocannl-1438: called with each proposal just before {!privatize_extension} validates it, inside
   the validation's refusal boundary, so a typed refusal raised here drops the proposal exactly as
   the op's own would. No workload in the tuning corpus reaches a real refusal on cc
   ([test/operations/privatize_census]), so this is how a test puts a dropped proposal through a
   real search. Not a production seam: default no-op, and no config key selects it. *)
let on_privatize_proposal : (target:Ir.Tnode.t -> over:Idx.symbol -> unit) Seam.t =
  Seam.make (fun ~target:_ ~over:_ -> ())

(** The preset schedule extended with a [Privatize] per detected accumulator, beside the proposals
    it dropped. Proposals are detected on the preset-scheduled segment and validated one at a time
    by re-applying the growing schedule; a proposal violating an op precondition is dropped, with
    the refusal that dropped it. The exploratory applies run against a hermetic copy of the segment:
    [Privatize] registers its (fresh) tile in the traced store and placements, and abandoned tiles
    would otherwise be emitted as dead local declarations when the caller applies the returned
    schedule to the real segment. *)
let privatize_extension ~accum_prec ~static_indices sched (seg : LL.optimized) :
    Sched.schedule * privatize_drop list =
  let scratch () =
    {
      seg with
      LL.traced_store = Hashtbl.copy seg.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx seg.LL.optimize_ctx;
    }
  in
  match Sched.apply_classified ~static_indices sched (scratch ()) with
  | exception Outcome.Cause_at _ -> (sched, [])
  | post ->
      let sched, drops =
        List.fold (privatize_proposals post) ~init:(sched, [])
          ~f:(fun (acc, drops) (target, over) ->
            let acc' = acc @ [ Sched.privatize ~accum_prec ~target ~over ] in
            match
              (Seam.get on_privatize_proposal) ~target ~over;
              Sched.apply_classified ~static_indices acc' (scratch ())
            with
            | (_ : LL.optimized) -> (acc', drops)
            | exception Outcome.Cause_at (_, cause) ->
                (acc, { pd_target = target; pd_over = over; pd_cause = cause } :: drops))
      in
      (sched, List.rev drops)

let extend_with_privatize ~accum_prec ~static_indices sched seg =
  fst (privatize_extension ~accum_prec ~static_indices sched seg)

(** {2 Split-reduce site detection (gh-ocannl-484 task 3)}

    Reduction-dominated sites: an accumulation whose target has few cells (little output
    parallelism) fed by a long serial reduction loop — the bias/weight-gradient reductions of the
    conv benchmarks, softmax denominators, and skinny (split-K) GEMMs alike. The gh-476 sweep
    attribution: on both Metal and CUDA one such fission segment is 60-95% of the default conv
    training step, and the tuner had no move into the split-reduction region of the schedule space —
    [Sched.Split_reduce] existed but nothing seeded it. Detection is deliberately cheap and
    over-approximate: any rmw [Set] (or gh-466 [Set_dynamic] scatter) qualifies structurally, and
    each candidate axis is settled by the hermetic {!Sched.op_legality} probe — the op's own
    recognizer decides the static-form pinning discipline, never a re-implementation here.

    {3 The enabling interchange (gh-ocannl-537)}

    A bare [Split_reduce] reaches none of the conv-gradient accumulations it was filed for: OCANNL
    lowers them with the accumulated channel loop {e innermost} and the reduction loops (batch, y,
    x) outside it, so every axis is rejected for "the accumulation cell mentions a symbol not bound
    by a loop enclosing the reduction loop" — measured on HIP lenet, where that one segment is 89%
    of the step. That cause, and only that cause, a loop interchange removes. So a rejected
    candidate is re-probed after hoisting exactly the symbols {!Sched.split_reduce_hoist} names,
    each bubbled outside the reduction loop by adjacent [Swap]s (relative order preserved); the site
    records the chain and the [F_split] prelude replays it before the split. Every [Swap] is
    confirmed [Op_legal] on the code it is applied to — [Swap]'s reassociation license covers the
    accumulation it reorders, but it is checked per site, not assumed — and the [Split_reduce] is
    re-probed on the interchanged code, so a returned site is still seedable exactly as proposed.

    {3 The coalesced reduction pair (gh-ocannl-1414)}

    A reduction over two loops can be splittable on neither. Backprop lowers a small-batch parameter
    gradient as [for b, s, o: g[o] += dy[b,s,o]] with [b = 8]: [b] is below [sr_red_min], and
    splitting [s] fails because [b], a reduction loop, encloses it and pins no component of the
    cell. Where no single loop of a site splits, its two outermost reduction loops are tried
    together: the cell loops below the outer one are hoisted as above, the pair is merged by a
    [Coalesce] ({!coalesce_reduction_pair}), and the split is re-probed on the merged loop. The site
    records the pair; the prelude replays the interchange, then the coalescing, then the split. The
    mechanism knows nothing about batches: an attention out projection's [(h, e)] contraction over a
    small output is the same two-loop band. Where the coalescing declines -- an operand read along
    one loop of the pair only, such as the positional table [wpe[s, o]] inlined into the first
    layers' residual stream, has no flattened address -- the pair is interchanged instead, one more
    [Swap] in the chain, and the inner loop is split with the outer one running inside each block. A
    site that one loop already splits keeps that split, so every site detected before is proposed
    unchanged. *)

type sr_site = {
  sr_axis : Idx.symbol;  (** The reduction loop to split: the largest-extent legal candidate. *)
  sr_target : Ir.Tnode.t;  (** The accumulated node. *)
  sr_red : int;  (** The [sr_axis] loop's extent. *)
  sr_out : int;  (** The target's cell count — the site's whole output parallelism. *)
  sr_cost : int;
      (** Estimated segment cost: the accumulation statement's trip count — the product of every
          enclosing loop extent, i.e. how many accumulate steps the serial nest spends on this site.
          Ranks the sites (gh-ocannl-541): the earlier [sr_red / sr_out] integer-division ratio sent
          every large-output site to 0, silently excluding the very sites (conv weight gradients)
          with the most serial work to recover. *)
  sr_dynamic : bool;  (** The gh-466 scatter form ([Set_dynamic]). *)
  sr_swaps : (Idx.symbol * Idx.symbol) list;
      (** The enabling interchange (gh-ocannl-537), as [(outer, inner)] pairs applied {e in order}
          before the [Split_reduce]: each hoists an accumulation-cell loop outside [sr_axis]. Empty
          when the site is splittable as lowered. *)
  sr_coalesce : (Idx.symbol * Idx.symbol) option;
      (** The reduction pair [(outer, inner)] that a [Coalesce], applied after [sr_swaps] and before
          the split, merges into the loop [sr_axis] names (gh-ocannl-1414,
          {!coalesce_reduction_pair}). [None] when [sr_axis] is a loop of the lowering. *)
  sr_block_hoist : Idx.symbol list;
      (** The accumulation-cell loops enclosing [sr_axis] once [sr_swaps] and [sr_coalesce] are
          applied, outermost first, which the prelude interchanges the split's block loop above
          (gh-ocannl-1412): the first pass then iterates [for block: for cells: ..], and the presets
          give its lanes the cell axes rather than the blocks. Empty for a scatter site and where
          that interchange is not [Op_legal]. *)
}
[@@deriving sexp_of]

(* Sites with more output cells than this have enough output parallelism that the default presets
   already fill a device; splitting the reduction would only add combine traffic. *)
let sr_out_max = 4096

(* Reduction extents below this are not worth a second kernel pass (the combine reads [num_blocks]
   partial cells per output cell). *)
let sr_red_min = 64

(* The adjacent-interchange chain hoisting [needed] outside [axis] within the write's loop [path]
   (outermost first), or [None] when some symbol is not a loop of that path — e.g. a static index —
   and hence not hoistable. Each symbol is bubbled up one loop at a time until it encloses [axis];
   taking them in path order leaves their relative order intact, so the resulting enclosing prefix
   iterates the accumulation cell exactly as the original nest did. *)
let sr_hoist_swaps ~path ~axis ~needed : (Idx.symbol * Idx.symbol) list option =
  let pos order s = List.findi order ~f:(fun _ x -> Idx.equal_symbol x s) |> Option.map ~f:fst in
  match
    (pos path axis, List.map needed ~f:(fun s -> Option.map (pos path s) ~f:(fun i -> (i, s))))
  with
  | None, _ -> None
  | Some _, indexed -> (
      match Option.all indexed with
      | None -> None
      | Some indexed ->
          let ordered =
            List.sort indexed ~compare:(fun (a, _) (b, _) -> Int.compare a b) |> List.map ~f:snd
          in
          let order = ref path and swaps = ref [] in
          List.iter ordered ~f:(fun h ->
              let continue_ = ref true in
              while !continue_ do
                (* Both are in [order] by construction and interchange only permutes it. *)
                let ih = Option.value_exn (pos !order h) in
                let ia = Option.value_exn (pos !order axis) in
                if ih <= ia then continue_ := false
                else
                  let parent = List.nth_exn !order (ih - 1) in
                  swaps := (parent, h) :: !swaps;
                  order :=
                    List.mapi !order ~f:(fun i x ->
                        if i = ih - 1 then h else if i = ih then parent else x)
              done);
          Some (List.rev !swaps))

(* A copy whose traced store and optimize context a probe may extend freely. *)
let sr_hermetic (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* One site's prelude up to and including its split: the enabling interchange, the coalescing of its
   reduction pair (re-minting the merged loop under [axis]), then [split], the [Sched.Split_reduce]
   op. *)
let sr_split_ops ~swaps ~coalesce ~axis split : Sched.schedule =
  List.map swaps ~f:(fun (outer, inner) -> Sched.Swap { outer; inner })
  @ (match coalesce with
    | None -> []
    | Some (outer, inner) -> [ Sched.Coalesce { outer; inner; merged = axis } ])
  @ [ split ]

(* The adjacent interchanges hoisting the split's block loop [block] above [cells], the loops
   enclosing it outermost first: innermost first, so each [Swap] finds [block] directly below the
   loop it passes. *)
let sr_block_hoist_ops ~block cells : Sched.schedule =
  List.rev_map cells ~f:(fun outer -> Sched.Swap { outer; inner = block })

(* The loops enclosing the loop [sym] in [llc], outermost first, or [None] when no loop is [sym]. *)
let rec sr_loops_above sym (llc : LL.t) : Idx.symbol list option =
  match llc with
  | LL.For_loop { index; _ } when Idx.equal_symbol index sym -> Some []
  | LL.For_loop { index; body; _ } -> Option.map (sr_loops_above sym body) ~f:(fun l -> index :: l)
  | LL.Seq (a, b) -> (
      match sr_loops_above sym a with Some _ as r -> r | None -> sr_loops_above sym b)
  | LL.If { body; _ } -> sr_loops_above sym body
  | _ -> None

(* gh-ocannl-1412: the cell loops a static site's split leaves around its block loop, when hoisting
   the block loop above all of them is [Op_legal] on the split code; else none. The split puts its
   block loop where the reduction loop was, below the cell loops the pinning discipline requires
   ([for cells: for block: { init; for inner: .. }]), and the GPU presets map the outer loop of such
   a chain to the grid and the inner one to the lanes. The lanes then walked the blocks, [chunk]
   rows apart, so every lane of a wavefront loaded its own cache line on every row: on
   gpt2_mini_train b256 the bias-gradient first passes cost HIP's unified-memory gfx1151 about 400
   ms of a 1260 ms step, where CUDA's caches absorbed it. Hoisted, the lanes walk the cell axes, as
   the elementwise nests that produce the summed rows do. Each cell's contributions keep their
   order: a cell is one tuple of cell-loop values, accumulated in its own partials row by one
   block. *)
let sr_block_hoist ~static_indices (opt : LL.optimized) ~axis ~tn ~swaps ~coalesce =
  let split, block, _, _ = Sched.split_reduce ~axis ~target:tn ~num_blocks:2 in
  match
    Sched.apply_classified ~static_indices
      (sr_split_ops ~swaps ~coalesce ~axis split)
      (sr_hermetic opt)
  with
  | exception Outcome.Cause_at _ -> []
  | split_opt -> (
      match sr_loops_above block split_opt.LL.llc with
      | None | Some [] -> []
      | Some cells ->
          let ops = sr_block_hoist_ops ~block cells in
          let verdicts = Sched.schedule_legality split_opt ops in
          if
            List.length verdicts = List.length ops
            && List.for_all verdicts ~f:(fun (_, v) -> Sched.equal_op_verdict v Sched.Op_legal)
          then cells
          else [])

(* [coalesce_reduction_pair] (gh-ocannl-1414) lives in {!Sketch_families}, which this module
   includes: the contraction layout of the coalesced sketch branch (gh-ocannl-1279) prepares its
   pair through it too, so a split of the merged loop and a tile of it start from one
   preparation. *)

let split_reduce_sites ?(static_indices = []) (opt : LL.optimized) : sr_site list =
  let acc = ref [] in
  let hermetic = sr_hermetic in
  (* Each loop's body, by its index: a site's split runs this body inside its block loop. *)
  let loop_body = Hashtbl.create (module Idx.Symbol) in
  (* The interchanged code, once every [Swap] of the chain is confirmed [Op_legal] against the code
     it is applied to ({!Sched.schedule_legality} walks the chain exactly as application will —
     [Swap]'s reassociation license covers accumulations, but each site is checked, not assumed).
     Anything short of all-legal drops the site. *)
  let apply_swaps swaps =
    let ops = List.map swaps ~f:(fun (outer, inner) -> Sched.Swap { outer; inner }) in
    let verdicts = Sched.schedule_legality opt ops in
    if
      List.length verdicts <> List.length ops
      || not (List.for_all verdicts ~f:(fun (_, v) -> Sched.equal_op_verdict v Sched.Op_legal))
    then None
    else
      match Sched.apply_classified ~static_indices ops (hermetic opt) with
      | opt' -> Some opt'
      | exception Outcome.Cause_at _ -> None
  in
  let splittable o ~axis ~tn =
    let op, _, _, _ = Sched.split_reduce ~axis ~target:tn ~num_blocks:2 in
    match Sched.op_legality o op with
    | Sched.Op_legal -> `Legal
    | Sched.Op_illegal _ | Sched.Op_unknown _ -> (
        (* The one rejection an interchange removes; empty for every other cause. *)
        match Sched.split_reduce_hoist o op with
        | [] -> `No
        | needed -> `Hoist needed)
  in
  let consider ~enclosing ~tn ~idcs ~dynamic =
    let out = try Ir.Tnode.num_elems tn with _ -> 0 in
    if out >= 1 && out <= sr_out_max then
      let path = List.map enclosing ~f:(fun (s, _, _) -> s) in
      let candidates =
        List.filter enclosing ~f:(fun (s, n, ty) ->
            LL.equal_axis_type ty LL.Serial && n >= sr_red_min
            && not (Array.exists idcs ~f:(Idx.axis_index_mentions_symbol s)))
        (* Largest extent first: the probe stops at the first legal candidate, and loops enclosing
           an inner reduction loop fail the pinning discipline anyway (an enclosing reduction loop
           pins no component), so outer/larger candidates dominate. *)
        |> List.sort ~compare:(fun (_, a, _) (_, b, _) -> Int.compare b a)
      in
      let single =
        List.find_map candidates ~f:(fun (s, n, _) ->
            match splittable opt ~axis:s ~tn with
            | `Legal -> Some (s, n, [], None)
            | `No -> None
            | `Hoist needed -> (
                (* gh-537: hoist and re-probe. Both the interchange and the split are settled on the
                   code they act on, so the recorded chain is replayable as recorded. *)
                match sr_hoist_swaps ~path ~axis:s ~needed with
                | None -> None
                | Some swaps -> (
                    match apply_swaps swaps with
                    | None -> None
                    | Some swapped -> (
                        match splittable swapped ~axis:s ~tn with
                        | `Legal -> Some (s, n, swaps, None)
                        | `No | `Hoist _ -> None))))
      in
      (* gh-ocannl-1414: no single loop splits, so try the two outermost reduction loops together,
         after hoisting the cell loops below the outer one, which leaves the pair adjacent: merged
         into one loop, or, where the coalescing declines, interchanged so that the inner loop leads
         and the outer one runs inside its split. Each step is settled on the code it acts on: the
         interchange and the coalescing by their own probes, the split on the loop it splits. *)
      let pair () =
        let mentioned s = Array.exists idcs ~f:(Idx.axis_index_mentions_symbol s) in
        match List.filter enclosing ~f:(fun (s, _, _) -> not (mentioned s)) with
        | (outer, p, ty) :: (inner, q, ty') :: _
          when LL.equal_axis_type ty LL.Serial && LL.equal_axis_type ty' LL.Serial -> (
            let below = List.drop_while path ~f:(fun s -> not (Idx.equal_symbol s outer)) in
            match sr_hoist_swaps ~path ~axis:outer ~needed:(List.filter below ~f:mentioned) with
            | None -> None
            | Some swaps -> (
                let merged () =
                  if p * q < sr_red_min then None
                  else
                    match apply_swaps swaps with
                    | None -> None
                    | Some swapped -> (
                        match coalesce_reduction_pair ~static_indices swapped ~outer ~inner with
                        | None -> None
                        | Some (merged, coalesced) -> (
                            match splittable coalesced ~axis:merged ~tn with
                            | `Legal -> Some (merged, p * q, swaps, Some (outer, inner))
                            | `No | `Hoist _ -> None))
                in
                (* A broadcast operand read along only one loop of the pair -- a positional table
                   [wpe[s, o]] inlined into a residual stream -- has no flattened address, but the
                   interchange needs none. *)
                let interchanged () =
                  if q < sr_red_min then None
                  else
                    let swaps = swaps @ [ (outer, inner) ] in
                    match apply_swaps swaps with
                    | None -> None
                    | Some swapped -> (
                        match splittable swapped ~axis:inner ~tn with
                        | `Legal -> Some (inner, q, swaps, None)
                        | `No | `Hoist _ -> None)
                in
                match merged () with Some _ as site -> site | None -> interchanged ()))
        | _ -> None
      in
      let legal = match single with Some _ -> single | None -> pair () in
      (* The split redirects only its target to partials: every other write under the loops it
         rewrites would keep running once per block, and the block loop is the one the presets
         parallelize, so blocks would race on it (or, kept serial, a non-additive write would see
         its iterations reordered). So a site whose outermost rewritten loop -- the outer loop of a
         pair, else the split loop -- encloses a write of any other node is declined. Lowering gives
         each assignment its own nest, so only hand-built or fused code meets this. *)
      let legal =
        Option.filter legal ~f:(fun (s, _, _, coalesce) ->
            let outermost = match coalesce with Some (o, _) -> o | None -> s in
            match Hashtbl.find loop_body outermost with
            | None -> false
            | Some body -> Set.for_all (Sched.written_nodes body) ~f:(Ir.Tnode.equal tn))
      in
      (* The loops a site's prelude rewrites: its split loop, and the pair a coalescing merges into
         it. Two accumulations in one nest find the same loops; a site sharing any with a site
         already taken would replay a rewrite of loops the first prelude has removed (two
         coalescings of one pair mint two merged symbols, so comparing the split loops alone would
         keep both). For two uncoalesced sites this is the split loop's identity, as before. *)
      let rewritten s coalesce =
        s :: (match coalesce with Some (o, i) -> [ o; i ] | None -> [])
      in
      Option.iter legal ~f:(fun (s, n, swaps, coalesce) ->
          let mine = rewritten s coalesce in
          if
            not
              (List.exists !acc ~f:(fun site ->
                   List.exists (rewritten site.sr_axis site.sr_coalesce) ~f:(fun x ->
                       List.mem mine x ~equal:Idx.equal_symbol)))
          then
            acc :=
              {
                sr_axis = s;
                sr_target = tn;
                sr_red = n;
                sr_out = out;
                sr_cost = List.fold enclosing ~init:1 ~f:(fun c (_, n, _) -> c * max 1 n);
                sr_dynamic = dynamic;
                sr_swaps = swaps;
                sr_coalesce = coalesce;
                sr_block_hoist =
                  (if dynamic then []
                   else sr_block_hoist ~static_indices opt ~axis:s ~tn ~swaps ~coalesce);
              }
              :: !acc)
  in
  let rec walk enclosing (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        walk enclosing a;
        walk enclosing b
    | LL.If { body; _ } -> walk enclosing body
    | LL.For_loop { index; from_; to_; body; axis; _ } ->
        Hashtbl.set loop_body ~key:index ~data:body;
        walk (enclosing @ [ (index, to_ - from_ + 1, axis) ]) body
    | LL.Set { tn; idcs; llsc; _ } ->
        (* rmw accumulation: the value re-reads the written node ([op_legality] then enforces the
           exact same-cell and operator discipline). *)
        if List.exists (collect_gets llsc) ~f:(fun (t, _) -> Ir.Tnode.equal t tn) then
          consider ~enclosing ~tn ~idcs ~dynamic:false
    | LL.Set_dynamic { tn; idcs; _ } -> consider ~enclosing ~tn ~idcs ~dynamic:true
    | _ -> ()
  in
  walk [] opt.LL.llc;
  (* Estimated segment cost, descending — the site with the most serial work to recover ranks first
     (gh-ocannl-541). Stable, so equal-cost sites keep detection (program) order. The
     candidate-volume cap is NOT applied here: it belongs to the search ([tune]'s
     [max_split_reduce_sites]), which records the sites it evicts in the decline census. *)
  List.stable_sort (List.rev !acc) ~compare:(fun a b -> Int.compare b.sr_cost a.sr_cost)

(* The whole-routine prelude splitting [sites], each at its block count: per site, the gh-537
   enabling interchange (empty for a site splittable as lowered), the gh-ocannl-1414 coalescing of
   its reduction pair (re-minting the merged loop under the symbol detection probed), the split
   itself, then the gh-ocannl-1412 interchange of the block loop above the cell loops
   ([sr_block_hoist]). Sites are distinct statements, so their preludes compose. Shared by the
   search's [F_split] seeds and the untuned plan's split ({!seeded_split_sites}, gh-ocannl-1325). *)
let split_reduce_prelude (sites : (sr_site * int) list) : Sched.schedule =
  List.concat_map sites ~f:(fun (s, num_blocks) ->
      let op, block, _, _ = Sched.split_reduce ~axis:s.sr_axis ~target:s.sr_target ~num_blocks in
      sr_split_ops ~swaps:s.sr_swaps ~coalesce:s.sr_coalesce ~axis:s.sr_axis op
      @ sr_block_hoist_ops ~block s.sr_block_hoist)

(** {2 Analytic cost-model scoring (gh-ocannl-491, the selection half)}

    The extraction half lives in {!Ir.Cost_model}; here it is consumed for ranking candidate
    schedules — the beam pre-filter of {!tune} and the untuned-default selection of
    {!model_default}. The model is advisory throughout: a candidate class without model coverage
    (opaque code, a schedule the model cannot apply, missing envelope constants) is never dropped,
    only measured — consistent with never overriding a measured result, and keeping the search
    independent of enumeration order. *)

module CM = Ir.Cost_model

let scratch_of (opt : LL.optimized) =
  {
    opt with
    LL.traced_store = Hashtbl.copy opt.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
  }

(* Fission can expose new sketch sites without adding a kernel: GPU zero expansion makes a covering
   initializer a per-cell companion even when the routine stays in one segment. Only an unchanged
   singleton is already covered by whole-routine sketches. *)
let fission_exposes_sites ~static_indices (opt : LL.optimized) tuples =
  match tuples with
  | [] -> false
  | [ (_, pre, _, _) ] ->
      let key o = SC.digest (SC.canonicalize ~static_indices ~with_placements:false o) in
      not (String.equal (key pre) (key opt))
  | _ -> true

(* gh-ocannl-1483: the default preset a segment gets in the tuner's fission pipelines. [`Search] is
   the search's ungated presets ([min_parallel:1] on GPU, [min_work:0] on CPU), [`Config] the
   configured thresholds -- which, with no [block_size], reproduce the untuned default pipeline's
   presets. *)
let segment_preset ?block_size ~presets ~is_gpu ~is_cpu ~limits seg =
  let search = match presets with `Search -> true | `Config -> false in
  if is_gpu then Sched.default_gpu ?block_size ?min_parallel:(Option.some_if search 1) ~limits seg
  else if is_cpu then Sched.default_cpu ?min_work:(Option.some_if search 0) seg
  else []

(* gh-ocannl-1483: the fission pipeline every tuner call site segments under, in place on [opt]:
   statement-crossing [Local]s promoted on GPU and the GPU zero policy, as the untuned default
   pipeline ([Sched.maybe_default_schedules]) has them, and the same schedule-aware merge probes
   ([Sched.fission_keep_mapping]), so a fissioned candidate and the untuned baseline schedule the
   same code. *)
let tuner_fission ?arity_cuts ?fold_zeros ?replay ~is_gpu ~limits ~preset ~static_indices opt =
  Sched.fission_segmented ~promote_locals:is_gpu ?arity_cuts ?fold_zeros
    ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
    ?replay ~preset
    ~zero_sched:(fun tns -> if is_gpu then Sched.zero_expansion ~limits tns else [])
    ~static_indices opt

(* gh-ocannl-1483: the segmentation the per-segment sketch candidates are enumerated, modeled and
   explained under, derived on a hermetic copy of [opt]: {!tuner_fission} with the zero fold
   (gh-ocannl-1175) and the default presets [presets] names. *)
let sketch_segmentation ?arity_cuts ~presets ~is_gpu ~is_cpu ~limits ~static_indices opt =
  tuner_fission ?arity_cuts ~fold_zeros:true ~is_gpu ~limits
    ~preset:(segment_preset ~presets ~is_gpu ~is_cpu ~limits)
    ~static_indices (scratch_of opt)

(* Per-machine calibrated envelope constants from the config beat the backend's class-level advisory
   constants ([Backend_intf.hardware_limits]'s [peak_flops] / [peak_memory_bandwidth]) — fitting
   them from [autotune_calibration_file] data is the intended workflow. Empty and non-positive
   values leave the advisory in place; text that is no number is refused
   ([Utils.Positive_float_or_unset]). *)
let peak_flops_override =
  lazy (Utils.registered_positive_float_or_unset ~arg_name:"model_peak_flops")

let peak_bandwidth_override =
  lazy (Utils.registered_positive_float_or_unset ~arg_name:"model_peak_memory_bandwidth")

let envelope_legs ~(limits : Ir.Backend_intf.hardware_limits) =
  let leg override advisory =
    match Lazy.force override with
    | Some v -> Some (v, `Config)
    | None -> Option.map advisory ~f:(fun v -> (v, `Backend))
  in
  ( leg peak_flops_override limits.Ir.Backend_intf.peak_flops,
    leg peak_bandwidth_override limits.Ir.Backend_intf.peak_memory_bandwidth )

(* The same constants without their provenance: one resolution, so the report's column and the model
   cannot read different envelopes. *)
let envelope ~limits =
  let flops, bandwidth = envelope_legs ~limits in
  (Option.map flops ~f:fst, Option.map bandwidth ~f:fst)

(* The roofline lower bound summed over a candidate's kernels; [None] — no model coverage — when any
   kernel is opaque (its counts may UNDER-estimate, so ranking on them could prune the true winner)
   or when no envelope constant is present. The kernels run sequentially, so the bound is per-kernel
   max-of-legs, summed — aggregating flops/bytes first and applying the roofline once would
   under-price a compute-bound + bandwidth-bound mix to roughly its larger leg. *)
let summaries_roofline ~peak_flops ~peak_memory_bandwidth (summaries : CM.summary list) :
    float option =
  if List.exists summaries ~f:(fun s -> s.CM.opaque) then None
  else
    (* [roofline_seconds] is [None] exactly when no envelope constant is given, uniformly across the
       folds — the [~flops:0 ~bytes:0] seed keeps that contract for the empty list. *)
    List.fold summaries
      ~init:(CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:0 ~bytes:0 ())
      ~f:(fun acc s ->
        Option.both acc
          (CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:s.CM.flops
             ~bytes:(CM.total_bytes s) ())
        |> Option.map ~f:(fun (a, b) -> a +. b))

(* gh-ocannl-1253: the ranking score's cache capacity — config [model_cache_capacity] (bytes;
   [none]: no capacity, so no DRAM tier) beats the backend's advisory [cache_capacity_bytes], and
   its cache bandwidth — config [model_cache_bandwidth] (bytes/s; [none]: re-reads within capacity
   uncharged) beats the advisory [cache_bandwidth]. Like those fields, [Untimed] readings: only the
   untimed choosers below call [ranking_capacity] / [ranking_cache_bandwidth], and nothing [tune]
   reaches does ([test/operations/limits_field_classes]). *)
let cache_capacity_override =
  lazy
    (let raw = String.strip (Utils.get_global_arg ~arg_name:"model_cache_capacity" ~default:"") in
     match String.lowercase raw with
     | "" -> `Advisory
     | "none" -> `Off
     | _ -> (
         match Float.of_string raw with
         | f when Float.is_finite f && Float.(f >= 0.) && Float.(f < of_int Int.max_value) ->
             `Value (Int.of_float f)
         | _ | (exception _) ->
             raise
             @@ Utils.User_error
                  ("Autotune: ocannl_model_cache_capacity should be a non-negative byte count, or \
                    none; found: " ^ raw)))

let cache_bandwidth_override =
  lazy
    (let raw = String.strip (Utils.get_global_arg ~arg_name:"model_cache_bandwidth" ~default:"") in
     match String.lowercase raw with
     | "" -> `Advisory
     | "none" -> `Off
     | _ -> (
         match Float.of_string raw with
         | f when Float.is_finite f && Float.(f > 0.) -> `Value f
         | _ | (exception _) ->
             raise
             @@ Utils.User_error
                  ("Autotune: ocannl_model_cache_bandwidth should be a positive rate in bytes/s, \
                    or none; found: " ^ raw)))

(* gh-ocannl-1442: the ranking score's own rates -- config [model_issue_rate] (operations/s) and
   [model_dram_bandwidth] (bytes/s), each [none]: priced at the envelope's ceiling -- beat the
   backend's advisory [issue_rate] / [dram_bandwidth]. Same readers and class as the two above. *)
let parse_rate ~key ~what raw =
  let raw = String.strip raw in
  match String.lowercase raw with
  | "" -> `Advisory
  | "none" -> `Off
  | _ -> (
      match Float.of_string raw with
      | f when Float.is_finite f && Float.(f > 0.) -> `Value f
      | _ | (exception _) ->
          raise
          @@ Utils.User_error
               (Printf.sprintf "Autotune: ocannl_%s should be a positive %s, or none; found: %s" key
                  what raw))

let issue_rate_override =
  lazy
    (parse_rate ~key:"model_issue_rate" ~what:"rate in operations/s"
       (Utils.get_global_arg ~arg_name:"model_issue_rate" ~default:""))

let dram_bandwidth_override =
  lazy
    (parse_rate ~key:"model_dram_bandwidth" ~what:"rate in bytes/s"
       (Utils.get_global_arg ~arg_name:"model_dram_bandwidth" ~default:""))

(* gh-ocannl-1495: the tile-MMA rate (operations/s; [none]: accepted tile-MMA work priced at the
   envelope's [peak_flops]) and the device-gather rate (per-lane gathers/s; [none]: no gather term)
   beat the backend's advisory [mma_rate] / [device_gather_rate]. Same readers and class. *)
let mma_rate_override =
  lazy
    (parse_rate ~key:"model_mma_rate" ~what:"rate in operations/s"
       (Utils.get_global_arg ~arg_name:"model_mma_rate" ~default:""))

let device_gather_rate_override =
  lazy
    (parse_rate ~key:"model_device_gather_rate" ~what:"rate in gathers/s"
       (Utils.get_global_arg ~arg_name:"model_device_gather_rate" ~default:""))

let resolve override advisory =
  match Lazy.force override with `Advisory -> advisory | `Off -> None | `Value v -> Some v

(* gh-ocannl-1479: the launch fill's two constants -- config [model_multiprocessors] and
   [model_fill_threads] (positive integers; [none]: no launch fill) -- beat the backend's advisory
   [multiprocessors] / [fill_threads]. Same readers and class as the rates above. *)
let parse_count ~key ~what raw =
  let raw = String.strip raw in
  match String.lowercase raw with
  | "" -> `Advisory
  | "none" -> `Off
  | _ -> (
      match Int.of_string_opt raw with
      | Some n when n > 0 -> `Value n
      | _ ->
          raise
          @@ Utils.User_error
               (Printf.sprintf "Autotune: ocannl_%s should be a positive %s, or none; found: %s" key
                  what raw))

let multiprocessors_override =
  lazy
    (parse_count ~key:"model_multiprocessors" ~what:"multiprocessor count"
       (Utils.get_global_arg ~arg_name:"model_multiprocessors" ~default:""))

let fill_threads_override =
  lazy
    (parse_count ~key:"model_fill_threads" ~what:"thread count per multiprocessor"
       (Utils.get_global_arg ~arg_name:"model_fill_threads" ~default:""))

let ranking_capacity ~limits =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  (resolve cache_capacity_override u.cache_capacity, u.resident_threads)

let ranking_cache_bandwidth ~limits =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  resolve cache_bandwidth_override u.cache_rate

let ranking_issue_rate ~limits =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  resolve issue_rate_override u.scalar_rate

let ranking_dram_bandwidth ~limits =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  resolve dram_bandwidth_override u.dram_rate

let ranking_mma_rate ~limits =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  resolve mma_rate_override u.matrix_rate

let ranking_device_gather_rate ~limits =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  resolve device_gather_rate_override u.gather_rate

let ranking_fill_model ~limits : CM.fill_model option =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  match
    ( resolve multiprocessors_override u.core_count,
      resolve fill_threads_override u.saturation_threads )
  with
  | Some fm_multiprocessors, Some fm_fill_threads -> Some { CM.fm_multiprocessors; fm_fill_threads }
  | _ -> None

type kernel_legs = {
  kl_compute : float option;
  kl_memory : float option;
  kl_spilled : bool;
  kl_fill : float;
      (** gh-ocannl-1479: the kernel's scalar launch fill ({!Ir.Cost_model.launch_fill}), 1 without
          a fill model. A diagnostic: the compute leg sums per context. *)
  kl_mma_work : float;
  kl_device_gathers : float;
      (** gh-ocannl-1495: the accepted tile-MMA work and the device gathers as the compute leg
          divides them by their rates ({!Ir.Cost_model.ranking_mma_terms}). *)
}

(* Per kernel, the ranking score's two legs and whether its working set left the cache, beside the
   candidate's roofline lower bound ([summaries_roofline]'s). [None] exactly where
   [summaries_roofline] is. *)
let segments_legs ?capabilities ?capacity ?cache_bandwidth ?issue_rate ?dram_bandwidth ?mma_rate
    ?device_gather_rate ?fill_model ~peak_flops ~peak_memory_bandwidth ~limits
    (segs : LL.optimized list) : (float * kernel_legs list) option =
  (* Each reading only where no explicit value replaces it: a caller's gate isolates the score from
     the configuration it overrides, a malformed override of it included. *)
  let max_resident_threads = (Ir.Backend_intf.untimed_limits limits).resident_threads in
  let cache_capacity_bytes =
    match capacity with Some c -> c | None -> fst (ranking_capacity ~limits)
  in
  let cache_bandwidth =
    match cache_bandwidth with Some cb -> cb | None -> ranking_cache_bandwidth ~limits
  in
  let issue_rate = match issue_rate with Some r -> r | None -> ranking_issue_rate ~limits in
  let dram_bandwidth =
    match dram_bandwidth with Some r -> r | None -> ranking_dram_bandwidth ~limits
  in
  let mma_rate = match mma_rate with Some r -> r | None -> ranking_mma_rate ~limits in
  let device_gather_rate =
    match device_gather_rate with Some r -> r | None -> ranking_device_gather_rate ~limits
  in
  let fill_model = match fill_model with Some f -> f | None -> ranking_fill_model ~limits in
  let summaries = List.map segs ~f:(fun o -> (o, CM.analyze o.LL.llc)) in
  match summaries_roofline ~peak_flops ~peak_memory_bandwidth (List.map summaries ~f:snd) with
  | None -> None
  | Some bound ->
      Some
        ( bound,
          List.map summaries ~f:(fun (o, s) ->
              let reuse =
                match (cache_capacity_bytes, cache_bandwidth) with
                | None, None -> None
                | _ ->
                    Some
                      (CM.reuse_traffic ?max_resident_threads ~is_global:(CM.global_in o) o.LL.llc)
              in
              (* gh-ocannl-1479: the compute nests' own launches, whether or not the cache tiers are
                 on. *)
              let launch = Option.map fill_model ~f:(fun fm -> (fm, CM.launch_contexts o.LL.llc)) in
              (* gh-ocannl-1442: tile-MMA work runs on the matrix units only where the backend's own
                 hook accepts it; without the capabilities to ask, all of it is priced as scalar
                 work. gh-ocannl-1495: the per-lane gathers its operands issue from device memory,
                 under the backend's own description of its arms. *)
              let mma_flops, device_gathers =
                Option.value_map capabilities ~default:(0, 0)
                  ~f:(fun (c : Ir.Backend_intf.codegen_capabilities) ->
                    ( CM.accepted_tile_mma_flops ~operand_accepts:c.mma_accepts o,
                      CM.device_operand_gathers ~operand_accepts:c.mma_accepts
                        ~operand_fragments:c.mma_operand_fragments
                        ~operand_loads:c.mma_operand_loads o ))
              in
              let kl_compute, kl_memory =
                CM.ranking_legs_of ?peak_flops ?peak_memory_bandwidth ?cache_capacity_bytes
                  ?cache_bandwidth ?issue_rate ?dram_bandwidth ?mma_rate ~mma_flops ~device_gathers
                  ?device_gather_rate ?launch s reuse
              in
              let kl_mma_work, kl_device_gathers =
                CM.ranking_mma_terms ~mma_flops ~device_gathers ?launch s
              in
              {
                kl_mma_work;
                kl_device_gathers;
                kl_compute;
                kl_memory;
                kl_spilled = Option.exists reuse ~f:(CM.spills ?cache_capacity_bytes);
                kl_fill =
                  Option.value_map launch ~default:1. ~f:(fun (fm, contexts) ->
                      CM.launch_fill fm contexts);
              }) )

(* A kernel's ranking score: the larger leg ({!Ir.Cost_model.ranking_seconds_of}). *)
let kernel_rank { kl_compute; kl_memory; _ } =
  match List.filter_opt [ kl_compute; kl_memory ] with
  | [] -> 0.
  | legs -> List.reduce_exn legs ~f:Float.max

(* The ranking score summed over a candidate's kernels, beside the roofline lower bound of the same
   kernels ([summaries_roofline]'s): [(rank, bound)], [rank >= bound] by
   {!Ir.Cost_model.ranking_seconds_of}' contract. [None] exactly where [summaries_roofline] is. Read
   by the untimed choosers only — [model_default], [model_explain] and the [auto] seeding gate. An
   explicit [capacity] / [cache_bandwidth] (the gate's own, or a test's) overrides the readings. *)
let segments_ranking ?capabilities ?capacity ?cache_bandwidth ?issue_rate ?dram_bandwidth ?mma_rate
    ?device_gather_rate ?fill_model ~peak_flops ~peak_memory_bandwidth ~limits
    (segs : LL.optimized list) : (float * float) option =
  Option.map
    (segments_legs ?capabilities ?capacity ?cache_bandwidth ?issue_rate ?dram_bandwidth ?mma_rate
       ?device_gather_rate ?fill_model ~peak_flops ~peak_memory_bandwidth ~limits segs)
    ~f:(fun (bound, legs) ->
      (List.fold legs ~init:0. ~f:(fun acc l -> acc +. kernel_rank l), bound))

let ranking_ms ?capabilities ~limits segs =
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  Option.map (segments_ranking ?capabilities ~peak_flops ~peak_memory_bandwidth ~limits segs)
    ~f:(fun (rank, bound) -> (rank *. 1e3, bound *. 1e3))

let ranking_legs_ms ?capabilities ~limits segs =
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  Option.map (segments_legs ?capabilities ~peak_flops ~peak_memory_bandwidth ~limits segs)
    ~f:(fun (_, legs) ->
      let sum f =
        List.fold legs ~init:(Some 0.) ~f:(fun acc l ->
            Option.map2 acc (f l) ~f:(fun a b -> a +. (b *. 1e3)))
      in
      (sum (fun l -> l.kl_compute), sum (fun l -> l.kl_memory)))

let model_score ~static_indices ~limits (opt : LL.optimized) (sched : Sched.schedule) : float option
    =
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  match Sched.apply_classified ~static_indices sched (scratch_of opt) with
  | exception Outcome.Cause_at _ -> None
  | post -> summaries_roofline ~peak_flops ~peak_memory_bandwidth [ CM.analyze post.LL.llc ]

let model_prefilter ~keep_fraction (scored : ('a * float option) list) : ('a * float option) list =
  let scores = List.filter_map scored ~f:snd in
  let n = List.length scores in
  if Float.(keep_fraction >= 1.) || n <= 1 then scored
  else
    let n_keep =
      Int.min n (Int.max 1 (Int.of_float (Float.round_up (keep_fraction *. Float.of_int n))))
    in
    let cutoff = List.nth_exn (List.sort scores ~compare:Float.compare) (n_keep - 1) in
    (* Ties at the cutoff are all kept: which of two equal-scored candidates survives must not
       depend on enumeration order. Unscored candidates ([None]) always pass — the no-coverage
       exemption. *)
    List.filter scored ~f:(fun (_, s) ->
        match s with None -> true | Some v -> Float.(v <= cutoff))

(** {2 Candidate compilation}

    A candidate is a recipe producing schedules against a {e fresh} lowering: backend [compile]
    re-lowers (with fresh symbols) on every call, so schedules are rebound structurally inside the
    transform closure, after checking the fresh code's canonical digest against the base compile's.
    Whole-routine candidates return a singleton from the [?lowered_transform] seam; fissioned
    candidates return one element per segment, with per-segment schedules keyed by the pre-schedule
    segment's canonical digest. *)

(* A beam-expansion candidate's provenance (gh-ocannl-1230): the parent it extends, by its digest
   tag ({!dshort}, the tag the round's beam log and every timing line print), the unit the move
   extends ([None] whole-routine, [Some i] the [i]th segment), and the appended move. Read only by
   {!spec_label}: the saved forms already end in the move, so compilation never consults it, but
   without it every round candidate of one parent prints the same label. [None] for a replay. *)
type move = { mv_parent : string; mv_segment : int option; mv_op : SC.saved_optop }

type whole_flavor =
  | W_saved of SC.saved_schedule * move option
  | W_preset of { block_size : int option }
  | W_sketch of sketch_params

(* Which segments of a recorded segmentation a per-segment sketch is for (gh-ocannl-1243): each
   one's first unit in the segmentation, and their common structural pre-schedule digest. The digest
   leaves placements out, so it cannot name a segment on its own: two segments of one structure can
   differ in placements and want different sketches. Segments share a key only when their
   placement-aware digests agree too ({!keyed_segments}) -- one code, one sketch, as for a model's
   repeated layers. *)
type seg_key = { key_units : int list; key_digest : string }

let equal_seg_key a b =
  List.equal Int.equal a.key_units b.key_units && String.equal a.key_digest b.key_digest

(* Each segment's first unit in a segmentation, by position. *)
let segment_starts (plan : Sched.segmentation) =
  Array.of_list (List.folding_map plan ~init:0 ~f:(fun at (_, n) -> (at + n, at)))

(* The keyable segments of a fission result ([plan] and its tuples, in order): its [`Normal]
   segments, grouped by their structural AND their placement-aware pre-schedule digests, in order of
   first occurrence -- each group's key, and its first member's pre- and post-schedule slices. *)
let keyed_segments ~static_indices plan tuples =
  let starts = segment_starts plan in
  let digest ~with_placements pre =
    SC.digest (SC.canonicalize ~static_indices ~with_placements pre)
  in
  List.filter_mapi tuples ~f:(fun i (kind, pre, _, post) ->
      match kind with
      | `Zeros | `Solo -> None
      | `Normal ->
          Some
            ( (digest ~with_placements:false pre, digest ~with_placements:true pre),
              starts.(i),
              pre,
              post ))
  |> List.fold ~init:[] ~f:(fun groups (ids, at, pre, post) ->
      if List.exists groups ~f:(fun (ids', _, _, _) -> Poly.equal ids ids') then
        List.map groups ~f:(fun ((ids', ats, pre', post') as g) ->
            if Poly.equal ids ids' then (ids', at :: ats, pre', post') else g)
      else (ids, [ at ], pre, post) :: groups)
  |> List.rev_map ~f:(fun ((structural, _), ats, pre, post) ->
      ({ key_units = List.rev ats; key_digest = structural }, pre, post))

(* The replay a per-segment sketch candidate compiles under ({!Sched.fission_segmented}'s [replay]):
   [plan], with each segment that a key of [entries] names by position scheduled by [sketch] -- once
   its structural digest is the key's -- and every other segment left to its kind's preset. *)
let sketch_replay ~static_indices ~sketch plan entries =
  let starts = segment_starts plan in
  let schedule i seg =
    let digest = lazy (SC.digest (SC.canonicalize ~static_indices ~with_placements:false seg)) in
    List.find_map entries ~f:(fun (key, p) ->
        if
          List.mem key.key_units starts.(i) ~equal:Int.equal
          && String.equal key.key_digest (Lazy.force digest)
        then Some (sketch p seg)
        else None)
  in
  (plan, schedule)

(* gh-ocannl-1388: the seeded untuned plan as a value, so the search can seed it, name it and (in
   gh-ocannl-1138) extend it. Deciding it and replaying it are separate steps: the untuned compile
   does both inside one transform, the search decides once on its base lowering and replays the plan
   in a candidate compile, and both replay through {!seeded_replay}, so the candidate compiles the
   code the untuned compile ships. *)
type seeded_segment =
  | Seeded_preset
  | Seeded_sketch of { digest : string; params : sketch_params }
  | Seeded_sunk of { digest : string; swaps : int }

(* gh-ocannl-1138: the plan's per-segment decision rule, for the staffed search's start, which
   applies it to the search's own segments; and a rendering of the gate it reads, which joins the
   search shape when that start is used. Carried in the plan so that the device readings the gate
   rests on reach [tune] only through it (gh-ocannl-1253's [limits_field_classes]). *)
type seeded_start = { decide : LL.optimized -> sketch_params option; gate_tag : string }

type seeded_plan = {
  sp_split : (sr_site * int) list;
  sp_segments : (Sched.segmentation * seeded_segment list) option;
  sp_refined : int;
  sp_start : seeded_start option;
  sp_admit_narrow : bool;
}

type fiss_flavor =
  | F_preset of {
      block_size : int option;
      privatize : bool;
      config_thresholds : bool;
          (** Use the config-default thresholds ([min_parallel] on GPU, [min_work] on CPU) instead
              of the search's ungated presets ([min_parallel:1] / [min_work:0]) — with
              [block_size = None] this reproduces the untuned default pipeline
              ({!Sched.maybe_default_schedules}) exactly, so the candidate pool always contains the
              behavior the user gets without tuning: on launch-overhead-bound workloads the
              aggressive ungated presets can all lose to it. *)
    }
  | F_saved of SC.saved_segment list * move option
      (** Replay of a fissioned winner (gh-ocannl-1164): the routine is cut where the saved segments
          say ({!Sched.fission_segmented}'s [replay]), never re-segmented under the current policy,
          and the [i]th segment gets the [i]th saved schedule — by position, never by digest — once
          its structural digest matches the saved [i]th; a segmentation that no longer fits, or a
          digest that differs, fails the candidate — for a cache entry, a re-search. Whichever mode
          derived the winner's segmentation (the finer [arity_cuts] one included, gh-ocannl-574) is
          in the data. *)
  | F_sketch of {
      plan : Sched.segmentation;
      entries : (seg_key * sketch_params) list;
      staff : (seg_key * sketch_params) list;
      fine : bool;
    }
      (** Per-segment matmul sketches over the segmentation the seed-time enumeration derived
          ([plan], replayed like [F_saved]'s, gh-ocannl-1243): each keyed segment gets the composed
          sketch pipeline instantiated with its entry's parameters, every other segment its kind's
          plain default preset — the pipeline the enumeration ran. Entries match segments by
          POSITION, the way a replay matches saved schedules, never by digest alone: the digest
          leaves placements out. A keyed segment whose digest differs from its key's degrades to the
          preset (the candidate then dedups away by digest); unlike [F_saved] it never replays a
          cache entry, so no loud drift guard is needed. [fine]: [plan] is the {e finer}
          [arity_cuts] segmentation, which frees a matmul site whose segment otherwise carries a
          companion that cannot follow the site's full arity (the lm_head's max-logits reduction,
          gh-ocannl-574). [staff] (gh-ocannl-1138): further keyed segments the candidate schedules
          by their entries' parameters, as [entries]' are -- the running composite's choices for the
          keys other than the one being swept, so a single is timed against the composite it would
          join rather than against every other segment's preset. [entries] stays the candidate's
          focus: what it is a single OF, what the search ranks it as, and the part of its label
          before [" | staff "]; empty [staff] is the bare single. *)
  | F_split of { sites : (sr_site * int) list; staff : (sr_site * int) list }
      (** Split-reduce seeds (gh-ocannl-484 task 3): per listed site, a
          [Sched.Split_reduce { axis = sr_axis; target = sr_target; num_blocks }] — applied
          {e whole-routine, before fission}, unlike the per-segment flavors: the two passes must
          compile as separate kernels (annotating the block loop with both passes in one kernel
          races — the combine needs grid-wide synchronization), and the partials producer/consumer
          pair is exactly the materialized cross-nest edge fission cuts at. Each resulting segment
          then gets the aggressive default preset — the block loop parallelizes pass 1, the combine
          nest annotates like any small kernel. [staff] (gh-ocannl-1138): further sites split by the
          same prelude, the running split composite's choices for the sites other than the one being
          swept; [sites] stays the candidate's focus, and empty [staff] is the bare single. *)
  | F_split_saved of SC.saved_schedule * SC.saved_segment list * move option
      (** Replay of a split-reduce winner: the whole-routine prelude (resolved against the base
          canonical form, re-minting the partials node and fresh symbols via [SC.of_saved]), then
          the saved segments of the {e post-prelude} routine, replayed exactly like [F_saved]. *)
  | F_seeded of seeded_plan
      (** The seeded untuned plan (gh-ocannl-1388): its split prelude whole-routine, as [F_split]'s,
          then its segments through {!seeded_replay}, the replay the recipes' untuned compile runs
          ({!seeded_default}), so the candidate is the step that ships untuned on CUDA and HIP
          (where [model_default_schedule] is off). Seeded FIRST, so the search's floor is that step
          by construction: a search cut short after its first timing, or one whose other candidates
          are all refused, still crowns it. *)

type spec = Whole of whole_flavor | Fiss of fiss_flavor

(* The replayable/cacheable description of a compiled candidate: a fissioned one carries every
   segment of its segmentation in order (gh-ocannl-1164). *)
type form =
  | Whole_saved of SC.saved_schedule
  | Fiss_saved of SC.saved_segment list
  | Split_saved of SC.saved_schedule * SC.saved_segment list

type unit_gen = {
  u_segment : int option;
      (** [Some i] for the [i]th segment of a fissioned form; [None] whole-routine. *)
  u_saved : SC.saved_schedule;
  u_registry : SC.registry;
  u_opt : LL.optimized;  (** The transformed unit, for menu generation. *)
}

type compiled = {
  form : form;
  cctx : Context.t;
  routine : Context.routine;
  units : unit_gen list;
  all_opts : LL.optimized list;
      (** Every compiled segment ([`Zeros] and [`Solo] segments included, unlike [units]) — the code
          the timing runs actually execute, for calibration analysis. *)
  digest_after : string;
}

(* Log tag for a (possibly '+'-concatenated, fissioned) digest: a plain prefix only reflects the
   first segment — two fissioned programs identical in segment 1 would read as "the same digest"
   (misled the CUDA round-4 analysis on PR #140) — so fold the whole string into the tag. *)
let dshort d =
  String.prefix d 8 ^ "/" ^ String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string d)) 8

let digest_tag = dshort
let bs_label = function None -> "cfg" | Some b -> Int.to_string b

(* The two forms' segment lists side by side when they segment alike -- the same form, and every
   segment of the same kind, length and pre-schedule digest, the identity a replay checks -- with
   the split-reduce preludes for a [Split_saved] pair; [None] when no unit-by-unit comparison exists
   (different forms, a different segmentation, or two whole-routine forms). *)
let aligned_segments (a : compiled) (b : compiled) =
  let same_cut (x : SC.saved_segment) (y : SC.saved_segment) =
    Poly.equal x.seg_kind y.seg_kind && x.seg_units = y.seg_units
    && String.equal x.seg_digest y.seg_digest
  in
  let aligned xs ys = List.length xs = List.length ys && List.for_all2_exn xs ys ~f:same_cut in
  match (a.form, b.form) with
  | Fiss_saved xs, Fiss_saved ys when aligned xs ys -> Some (None, List.zip_exn xs ys)
  | Split_saved (px, xs), Split_saved (py, ys) when aligned xs ys ->
      Some (Some (px, py), List.zip_exn xs ys)
  | _ -> None

(* The positions of the segments of [b] whose saved schedule differs from the same segment of [a],
   in the indexing [unit_gen.u_segment] uses; [None] when the two do not segment alike
   ([aligned_segments]). *)
let segments_differing (a : compiled) (b : compiled) =
  Option.map (aligned_segments a b) ~f:(fun (_, pairs) ->
      List.filter_mapi pairs ~f:(fun i ((x : SC.saved_segment), (y : SC.saved_segment)) ->
          if Poly.equal x.seg_saved y.seg_saved then None else Some i))

(* How many units of [b] carry a different saved schedule than the same unit of [a] (a split-reduce
   prelude counting as one); [None] when the two forms do not segment alike, so no unit-by-unit
   comparison exists. *)
let units_differing (a : compiled) (b : compiled) =
  let differ x y = if Poly.equal x y then 0 else 1 in
  match (a.form, b.form) with
  | Whole_saved x, Whole_saved y -> Some (differ x y)
  | _ ->
      Option.map (aligned_segments a b) ~f:(fun (preludes, pairs) ->
          List.sum
            (module Int)
            pairs
            ~f:(fun ((x : SC.saved_segment), (y : SC.saved_segment)) ->
              differ x.seg_saved y.seg_saved)
          + Option.value_map preludes ~default:0 ~f:(fun (px, py) -> differ px py))

(* Calibration output (gh-ocannl-491 task 4) and the bound-agreement invariant (gh-ocannl-514 phase
   0): the model score next to the measured time — every tuning run is free calibration data for the
   envelope constants, and every timed candidate is a test of the roofline bound's soundness.
   Human-readable stderr lines under config [autotune_log]; durable tab-separated rows (schema owned
   by {!CM.Calibration}) appended under config [autotune_calibration_file].

   The analysis runs on the candidate's actual compiled segments ([compiled.all_opts]), so a row
   prices exactly the code that was timed. For an exact-count candidate, a roofline LOWER bound
   exceeding a measured time can only mean the envelope constants understate this machine's
   achievable peaks — a search fathoming on that bound would prune true winners — so the violation
   warns unconditionally, not gated by [autotune_log]: per the gh-ocannl-498 lesson, an invariant
   between a scorer and reality is checked continuously against every sample, never spot-checked.
   Approximate counts ([CM.approximate]: guards-taken / union upper bounds) make an exceedance
   ambiguous — mostly-failing guards over-count without implicating the envelope — so those log as
   diagnostics and their rows are flagged for the fitter to exclude. Refitting the constants from
   the accumulated rows ([CM.Calibration.fit], [tools/fit_envelope.exe]) restores soundness. The
   analysis therefore also runs whenever envelope constants are present, even with logging and the
   calibration file off — one [CM.analyze] per compiled segment, trivial next to the compile and
   timing runs the candidate already paid for. *)
let calibration_file = lazy (Utils.registered_text ~arg_name:"autotune_calibration_file")

(* The same candidate can be timed repeatedly within a process (a test tuning the same preset in two
   arms, a re-tune after a cache miss). A repeat violation restates the first one — the implied
   minima move only by timing jitter — so the unconditional warning fires once per distinct
   (backend, device, digest tag), while every timing still contributes its own calibration row and
   autotune_log line. The backend and device belong in the key: the digest is schedule-level, so one
   process tuning the same code on another backend — or on another device of the same backend —
   produces an identical tag, and those measurements are exactly the ones [tune] refuses to
   substitute for each other (the [timing_ctx] backend-and-device check), so their implied minima
   are independent evidence.

   Claiming a key is the module's only mutation of process-wide state, so it takes a mutex rather
   than assume its caller's threading: [tune] runs on whichever domain called it, and a test-and-set
   torn across two domains would both duplicate the warning and race Base's hash table internals.
   Uncontended, the lock is nothing next to the compile and timing runs the candidate already paid
   for. *)
let warned_bound_violations = Hash_set.create (module String)
let warned_bound_violations_mutex = Stdlib.Mutex.create ()

(* [true] exactly once per key per process: the winner of the test-and-set warns. *)
let claim_bound_violation_warning key =
  Stdlib.Mutex.protect warned_bound_violations_mutex (fun () ->
      let fresh = not (Hash_set.mem warned_bound_violations key) in
      if fresh then Hash_set.add warned_bound_violations key;
      fresh)

(* Bound pruning against the measured incumbent (gh-ocannl-514 phase 4b): a sketch candidate whose
   schedule-invariant roofline floor meets the best measured time so far provably cannot win, so its
   compile and timing are skipped — the admissible-direction pruning of the issue's tuned regime.
   Default off: it changes which candidates get timed (reports, test goldens), and its soundness
   rests on honest envelope constants — the continuous agreement check guards them, but an
   understated envelope over-prunes, so the gate is explicit. Only the enumerative sketch flavors
   are prunable: presets, saved schedules and the baseline keep their reporting and cache-replay
   roles regardless of winnability, mirroring the keep-fraction pre-filter's exemptions. *)
let bound_pruning_enabled = lazy (Utils.registered_flag ~arg_name:"autotune_bound_pruning")

let bound_prunable = function
  | Whole (W_sketch _) | Fiss (F_sketch _) | Fiss (F_split _) -> true
  | _ -> false

let emit_calibration_unchecked ~backend ~device ~limits ~routine ~label ~digest ~measured_ms
    (opts : LL.optimized list) =
  let file = Lazy.force calibration_file in
  (* Everything this emits names the computation as well as the candidate (gh-ocannl-635): a process
     tunes several routines, and a row (or a stderr line, or a fit witness quoting one) saying only
     [W_preset[bs=512]] cannot be traced back to the kernel it measured. *)
  let named = CM.Calibration.qualified ~routine ~label in
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let have_envelope = Option.is_some peak_flops || Option.is_some peak_memory_bandwidth in
  if log_enabled () || (not (String.is_empty file)) || have_envelope then (
    let summaries = List.map opts ~f:(fun o -> CM.analyze o.LL.llc) in
    let flops = List.sum (module Int) summaries ~f:(fun s -> s.CM.flops) in
    let bytes = List.sum (module Int) summaries ~f:CM.total_bytes in
    let opaque = List.exists summaries ~f:(fun s -> s.CM.opaque) in
    let flops_approx = List.exists summaries ~f:(fun s -> s.CM.flops_approx) in
    let bytes_approx = List.exists summaries ~f:CM.footprint_approximate in
    let model_ms =
      Option.map (summaries_roofline ~peak_flops ~peak_memory_bandwidth summaries) ~f:(fun s ->
          s *. 1e3)
    in
    let dtag = dshort digest in
    (let seconds = Float.max 1e-12 (measured_ms *. 1e-3) in
     (* Per-leg audit: an exact aggregate leg exceeding the measurement indicts the envelope no
        matter what the other leg's counts are (the aggregate leg lower-bounds the per-kernel sum).
        The whole-bound check additionally catches the fully-exact multi-kernel case where the
        per-kernel max-of-legs sum exceeds the measurement without either aggregate leg doing so.
        The implied minima name only legs that are configured AND exact — an absent leg cannot have
        caused the exceedance, and an approximate one is not evidence. *)
     let leg_exceeds exact count peak =
       match peak with
       | Some p -> exact && Float.(Float.of_int count /. seconds > p)
       | None -> false
     in
     let flops_leg = leg_exceeds (not flops_approx) flops peak_flops in
     let bytes_leg = leg_exceeds (not bytes_approx) bytes peak_memory_bandwidth in
     let bound_exceeds = match model_ms with Some m -> Float.(m > measured_ms) | None -> false in
     if flops_leg || bytes_leg || (bound_exceeds && (not flops_approx) && not bytes_approx) then
       let warn_key = Printf.sprintf "%s|%d|%s" backend device dtag in
       if not (claim_bound_violation_warning warn_key) then ()
       else
         let minima =
           String.concat ~sep:" and "
             (List.filter_opt
                [
                  (if Option.is_some peak_flops && not flops_approx then
                     Some
                       (Printf.sprintf "model_peak_flops >= %.6g" (Float.of_int flops /. seconds))
                   else None);
                  (if Option.is_some peak_memory_bandwidth && not bytes_approx then
                     Some
                       (Printf.sprintf "model_peak_memory_bandwidth >= %.6g"
                          (Float.of_int bytes /. seconds))
                   else None);
                ])
         in
         Stdio.eprintf
           "autotune: BOUND VIOLATION: roofline lower bound %s ms > measured %.4f ms for %s \
            (digest %s) on %s device %d — the envelope constants understate this machine's peaks \
            (this row implies %s as necessary minima); refit with tools/fit_envelope.exe over \
            autotune_calibration_file data\n\
            %!"
           (match model_ms with Some m -> Printf.sprintf "%.6f" m | None -> "?")
           measured_ms named dtag backend device minima
     else if bound_exceeds then
       (* Only an approximate leg can explain the exceedance: possibly over-counting (guards-taken /
          union upper bounds), not the envelope — a diagnostic, no unconditional warning, no
          implied-minima claim. *)
       logf
         "model bound %.6f ms > measured %.4f ms for %s (digest %s), but its counts are \
          approximate upper bounds (guarded/masked code) — possibly over-counting, not the \
          envelope"
         (Option.value_exn model_ms) measured_ms named dtag);
    let n_kernels = List.length summaries in
    logf "calibration: %s measured %.4f ms, model %s, %d kernel%s, flops %d, bytes %d%s" named
      measured_ms
      (match model_ms with Some m -> Printf.sprintf "%.6f ms" m | None -> "n/a")
      n_kernels
      (if n_kernels = 1 then "" else "s")
      flops bytes
      (if opaque then " (opaque: counts may under-estimate)" else "");
    if not (String.is_empty file) then
      let line =
        CM.Calibration.to_line
          {
            CM.Calibration.backend;
            digest = dtag;
            routine;
            label;
            measured_ms;
            model_ms;
            kernels = n_kernels;
            flops;
            bytes;
            flops_approx;
            bytes_approx;
            opaque;
          }
        ^ "\n"
      in
      try
        Stdio.Out_channel.with_file file ~append:true ~f:(fun oc ->
            Stdio.Out_channel.output_string oc line)
      with _ -> logf "calibration: cannot append to %s" file)

let emit_calibration ~backend ~device ~limits ~routine ~label ~digest ~timing_result opts =
  Option.iter (admitted_timing_ms timing_result) ~f:(fun measured_ms ->
      emit_calibration_unchecked ~backend ~device ~limits ~routine ~label ~digest ~measured_ms opts)

(* Whether the spec's label promises a tensorized pipeline — used to flag "no Tile_mma emitted"
   census anomalies (gh-ocannl-479). *)
let spec_expects_mma = function
  | Whole (W_sketch p) -> p.sk_mma
  | Fiss (F_sketch { entries; staff; _ }) ->
      List.exists (entries @ staff) ~f:(fun (_, p) -> p.sk_mma)
  | _ -> false

(* A sketch's label renders EVERY field of its [sketch_params] (gh-ocannl-1230): the tuner seeds
   twins that differ in a single field — the swizzled, pipelined, widened-pack, batch-grid and
   register-tile twins of one geometry — so a field the label leaves out makes two candidates read
   identically in every search log, and either can be crowned under that one label.
   test/operations/autotune_labels enumerates the fields from the type's derived s-expression and
   checks that changing any one of them changes the label, so a new field fails there until it is
   rendered here. Tokens print only where a field departs from its common value, so a typical label
   stays short. *)
let sketch_params_label p =
  let flag b token = if b then token else "" in
  String.concat
    [
      flag p.sk_conv "conv-";
      flag p.sk_mma "mma-";
      (if p.sk_gpu then "gpu" else "cpu");
      Printf.sprintf " %dx%dx%d" p.sk_bm p.sk_bn p.sk_bk;
      (* Register-tile factors: the scalar GPU pipeline's; 0x0 (unused) elsewhere. *)
      (if p.sk_tm = 0 && p.sk_tn = 0 then "" else Printf.sprintf "/%dx%d" p.sk_tm p.sk_tn);
      (if p.sk_simd = 0 then "" else Printf.sprintf " simd%d" p.sk_simd);
      (* A widened-pack candidate (gh-ocannl-575) differs from a plain sibling only in the packed
         tiles' precision. *)
      (match p.sk_pack_prec with
      | Some pr -> Printf.sprintf " pack%s" (Ir.Ops.prec_string pr)
      | None -> "");
      (* The swizzled staged twin (gh-ocannl-481 item 3, D3). *)
      (match p.sk_swizzle with
      | None -> ""
      | Some LL.Swizzle_elem -> " swz-elem"
      | Some LL.Swizzle_b128 -> " swz-b128");
      (* The pipelined staged twin (gh-ocannl-487): the cooperative-stage depth; 1 = unpipelined. *)
      (if p.sk_depth = 1 then "" else Printf.sprintf " pd%d" p.sk_depth);
      flag p.sk_hoist " hoist";
      flag p.sk_grid " grid";
      flag p.sk_pack_rest " packrest";
      (* The batch flavors (gh-ocannl-643, gh-ocannl-728): [bgrid] for the batch-grid twin,
         [bgrid-in] for its interior-batch-inside-the-row-blocks order; [sk_batch_inner] is never
         seeded without [sk_batch_grid], and [bin] keeps that combination apart anyway. *)
      (match (p.sk_batch_grid, p.sk_batch_inner) with
      | false, false -> ""
      | true, false -> " bgrid"
      | true, true -> " bgrid-in"
      | false, true -> " bin");
      (* The coalesced layouts' seeds, whose tile fields describe the coalesced site: [coal-n] for
         the interior batch loop merged into the column role (gh-ocannl-1165), [coal-k] for the
         contraction pair merged into one k (gh-ocannl-1279). *)
      (match p.sk_coalesce with
      | None -> ""
      | Some Coalesce_columns -> " coal-n"
      | Some Coalesce_contraction -> " coal-k");
      flag p.sk_epilogue " ep";
      (* The register-tile twins of one CPU tensorized leaf (gh-ocannl-619, gh-ocannl-1166). *)
      (match p.sk_tile with
      | None -> ""
      | Some { Ir.Register_tile.rm; rn; lanes } -> Printf.sprintf " tile-rm%d-rn%d-l%d" rm rn lanes);
    ]

(* A round candidate's suffix (gh-ocannl-1230): the parent's digest tag, then the unit the move
   extended and the move itself, printed from its derived s-expression so every field of the move
   shows: round candidates that differ in parent, unit or move print apart, where the bare
   [F_saved[N segs]] they all used to share named none of the three. A replay ([None]) prints
   nothing. *)
let move_label = function
  | None -> ("", "")
  | Some { mv_parent; mv_segment; mv_op } ->
      ( " from " ^ mv_parent,
        Printf.sprintf "+%s%s"
          (Option.value_map mv_segment ~default:"" ~f:(Printf.sprintf "seg%d:"))
          (Sexp.to_string_mach (SC.sexp_of_saved_optop mv_op)) )

(* A segment key's tag: {!dshort} of the structural digest, then the keyed segments' first units.
   Segments of one structure but different placements share the digest and are keyed apart
   (gh-ocannl-1243), so the units are what tells their keys apart. *)
let seg_key_tag { key_units; key_digest } =
  Printf.sprintf "%s:u%s" (dshort key_digest)
    (String.concat ~sep:"+u" (List.map key_units ~f:Int.to_string))

(* A per-segment sketch candidate's label: each keyed entry renders its parameters and its segment's
   key tag. Two segments can admit the same parameters -- two matmul segments sharing a tile
   geometry -- and their one-entry candidates schedule different kernels, so without the key they
   would print alike (Codex P2 on PR #1108). A staffed candidate (gh-ocannl-1138) renders its staff
   after [" | staff "], so the focus part reads as it does bare. *)
let fiss_sketch_label ?(staff = []) ~fine entries =
  let render entries =
    String.concat ~sep:","
      (List.map entries ~f:(fun (key, p) -> sketch_params_label p ^ " @" ^ seg_key_tag key))
  in
  Printf.sprintf "F_sketch[%s%s%s]"
    (if fine then "fine " else "")
    (render entries)
    (match staff with [] -> "" | _ -> " | staff " ^ render staff)

(* gh-ocannl-1512: the per-segment view [model_default] ranks and [model_explain] lists -- exactly
   what {!tune}'s seed enumeration derives: {!sketch_segmentation} at [`Search] of [base], when it
   splits the routine or exposes a changed lowering, as the plan and one [(key, pre, post)] per
   keyed segment, [post] being the segment under the search's preset: the threshold its sketches
   must strictly beat to enter a substitution. Not the configured thresholds: they leave a
   below-threshold nest unannotated, and fission coalesces adjacent unannotated segments back into
   one kernel, so on cc below [cpu_schedule_min_parallel] that segmentation was the single kernel
   and the per-segment path never fired. Nor the configured preset as the threshold: the ranking
   score charges no operand re-reads to an unannotated kernel (one block), so a segment's sketches
   could at best tie its serial preset and no substitution would ever form. (What a substitution
   SHIPS on the segments it leaves unsketched is the configured preset: [model_default]'s
   [apply_fiss].) [None] where fission exposes no site; raises {!Outcome.Cause_at} as
   {!Sched.fission_segmented} does. [tune] enumerates its per-segment seeds on it
   ({!search_segment_entries}, gh-ocannl-1490), its finer pass with [arity_cuts]. *)
let model_segments ?arity_cuts ~is_gpu ~is_cpu ~limits ~static_indices (base : LL.optimized) =
  match
    sketch_segmentation ?arity_cuts ~presets:`Search ~is_gpu ~is_cpu ~limits ~static_indices base
  with
  | _, tuples when not (fission_exposes_sites ~static_indices base tuples) -> None
  | plan, tuples -> Some (plan, keyed_segments ~static_indices plan tuples)

(* gh-ocannl-1490: the search's seeding, as one value a driver can take without a search -- the
   seeding census without a budget hack, and the seeds a per-seed parity check executes. [tune]
   enumerates through it, so the census and the parity driver see exactly what a search seeds.
   [census] takes each seeded lowering's census line where [tune] takes it (default: the
   [autotune_log_seeding] stream). *)
let default_census ~capabilities ~is_gpu ~is_cpu ~limits ~where opt =
  log_seeding ~coalesced:true ~capabilities ~where ~is_gpu ~is_cpu ~limits opt

let search_seed_params ?census ~capabilities ~is_gpu ~is_cpu ~limits ~where opt =
  (match census with
  | Some census -> census ~where opt
  | None -> default_census ~capabilities ~is_gpu ~is_cpu ~limits ~where opt);
  sketch_seed_params ~capabilities ~is_gpu ~is_cpu ~limits opt
  @ coalesced_seed_params ~capabilities ~is_gpu ~is_cpu ~limits opt

(* The search's per-fission-segment seeds on one segmentation (coarse, or the finer [arity_cuts]
   one): the keyed segments of {!sketch_segmentation} at the search's presets, each with the seeds
   it admits; a segment with none is left out. Empty off the C and GPU backends, where the
   segmentation refuses, and where fission exposes no site the whole-routine seeds miss. *)
let search_segment_entries ?census ~capabilities ~arity_cuts ~is_gpu ~is_cpu ~limits ~static_indices
    base_opt =
  if not (is_gpu || is_cpu) then ([], [])
  else
    match model_segments ~arity_cuts ~is_gpu ~is_cpu ~limits ~static_indices base_opt with
    | exception Outcome.Cause_at _ -> ([], [])
    | None -> ([], [])
    | Some (plan, keyed) ->
        ( plan,
          List.filter_map keyed ~f:(fun (key, pre, _) ->
              let where =
                Printf.sprintf "tune %s segment %s"
                  (if arity_cuts then "fine" else "coarse")
                  (seg_key_tag key)
              in
              match search_seed_params ?census ~capabilities ~is_gpu ~is_cpu ~limits ~where pre with
              | [] -> None
              | params -> Some (key, pre, params)) )

type segment_seeds = seg_key * LL.optimized * sketch_params list

type search_seeding = {
  ss_whole : sketch_params list;
  ss_coarse_plan : Sched.segmentation;
  ss_coarse : segment_seeds list;
  ss_fine_plan : Sched.segmentation;
  ss_fine : segment_seeds list;
  ss_fine_new : segment_seeds list;
}

let search_seeding ?census ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices base_opt =
  let ss_whole =
    search_seed_params ?census ~capabilities ~is_gpu ~is_cpu ~limits ~where:"tune whole-routine"
      base_opt
  in
  let entries ~arity_cuts =
    search_segment_entries ?census ~capabilities ~arity_cuts ~is_gpu ~is_cpu ~limits ~static_indices
      base_opt
  in
  let ss_coarse_plan, ss_coarse = entries ~arity_cuts:false in
  (* The finer segmentation (gh-ocannl-574) is GPU-only, and is dropped whole when it cuts nothing
     new: see [tune]. *)
  let coarse_keys = List.map ss_coarse ~f:(fun (k, _, _) -> k) in
  let is_new (k, _, _) = not (List.mem coarse_keys k ~equal:equal_seg_key) in
  let ss_fine_plan, ss_fine =
    if not is_gpu then ([], [])
    else
      let plan, fine = entries ~arity_cuts:true in
      if List.exists fine ~f:is_new then (plan, fine) else ([], [])
  in
  {
    ss_whole;
    ss_coarse_plan;
    ss_coarse;
    ss_fine_plan;
    ss_fine;
    ss_fine_new = List.filter ss_fine ~f:is_new;
  }

(* A split-reduce candidate's label renders EVERY field of each [(site, num_blocks)] entry
   (post-wave task of gh-ocannl-1230), as {!sketch_params_label} does for a sketch: the target by
   its debug name AND its id, since two nodes can share a debug name (two [x.grad]s of one model),
   the split axis, and the interchange chain pair by pair, since two chains of equal length hoist
   different loops. test/operations/autotune_labels enumerates the fields from the type's derived
   s-expression, so a new field fails there until it is rendered here. No token contains [','], the
   entries' separator. *)
let split_site_label (s, num_blocks) =
  let name = Ir.Tnode.debug_name s.sr_target and id = Ir.Tnode.id s.sr_target in
  Printf.sprintf "%s %s red%d out%d cost%d b%d%s%s%s%s"
    (if String.equal name id then name else name ^ "#" ^ id)
    (Idx.symbol_ident s.sr_axis) s.sr_red s.sr_out s.sr_cost num_blocks
    (if s.sr_dynamic then " dyn" else "")
    (match s.sr_swaps with
    | [] -> ""
    | swaps ->
        " swap:"
        ^ String.concat ~sep:"+"
            (List.map swaps ~f:(fun (outer, inner) ->
                 Idx.symbol_ident outer ^ "/" ^ Idx.symbol_ident inner)))
    (match s.sr_coalesce with
    | None -> ""
    | Some (outer, inner) -> " coalesce:" ^ Idx.symbol_ident outer ^ "*" ^ Idx.symbol_ident inner)
    (match s.sr_block_hoist with
    | [] -> ""
    | cells -> " hoist:" ^ String.concat ~sep:"+" (List.map cells ~f:Idx.symbol_ident))

let split_sites_label ?(staff = []) sites =
  let render sites = String.concat ~sep:"," (List.map sites ~f:split_site_label) in
  Printf.sprintf "F_split[%s%s]" (render sites)
    (match staff with [] -> "" | _ -> " | staff " ^ render staff)

(* gh-ocannl-1388: the seeded plan's identity, from its content: each split site by its target's
   debug name, extents, block count and interchange length (the ids and symbols {!split_site_label}
   prints are process-local), then the segmentation and every segment's decision with the structural
   digest it was decided on. A plan derived from the same lowering under the same settings renders
   alike in every process. *)
let seeded_plan_digest (plan : seeded_plan) =
  let split =
    List.map plan.sp_split ~f:(fun (s, b) ->
        Printf.sprintf "split %s red%d out%d cost%d b%d swaps%d hoist%d%s"
          (Ir.Tnode.debug_name s.sr_target) s.sr_red s.sr_out s.sr_cost b (List.length s.sr_swaps)
          (List.length s.sr_block_hoist)
          (if s.sr_dynamic then " dyn" else ""))
  in
  let segments =
    match plan.sp_segments with
    | None -> [ "default segments" ]
    | Some (segmentation, segs) ->
        Sexp.to_string_mach (Sched.sexp_of_segmentation segmentation)
        :: List.map segs ~f:(function
          | Seeded_preset -> "preset"
          | Seeded_sketch { digest; params } ->
              "sketch " ^ digest ^ " " ^ Sexp.to_string_mach (sexp_of_sketch_params params)
          | Seeded_sunk { digest; swaps } -> Printf.sprintf "sunk %s %d" digest swaps)
  in
  (* gh-ocannl-1405: the replay re-derives each sunk segment's chain under the plan's narrow-storage
     admission, so the admission is part of what the plan means; rendered only when on, so every
     digest that predates the key is unchanged. *)
  let narrow = if plan.sp_admit_narrow then [ "admit narrow storage" ] else [] in
  Stdlib.Digest.to_hex (Stdlib.Digest.string (String.concat ~sep:"\n" (split @ segments @ narrow)))

let seeded_plan_label (plan : seeded_plan) =
  let count f = function None -> 0 | Some (_, segs) -> List.count segs ~f in
  Printf.sprintf "F_seeded[%s, %d seeded, %d sunk, %d split, plan %s]"
    (match plan.sp_segments with
    | None -> "default segs"
    | Some (segmentation, _) -> Printf.sprintf "%d segs" (List.length segmentation))
    (count (function Seeded_sketch _ -> true | _ -> false) plan.sp_segments)
    (count (function Seeded_sunk _ -> true | _ -> false) plan.sp_segments)
    (List.length plan.sp_split)
    (String.prefix (seeded_plan_digest plan) 8)

let spec_label = function
  | Whole (W_saved (s, move)) ->
      let from, extension = move_label move in
      Printf.sprintf "W_saved[%d ops%s]%s" (List.length s) from extension
  | Whole (W_preset { block_size }) -> Printf.sprintf "W_preset[bs=%s]" (bs_label block_size)
  | Whole (W_sketch p) -> Printf.sprintf "W_sketch[%s]" (sketch_params_label p)
  | Fiss (F_preset { block_size; privatize; config_thresholds }) ->
      Printf.sprintf "F_preset[bs=%s%s%s]" (bs_label block_size)
        (if privatize then " priv" else "")
        (if config_thresholds then " cfg-thresh" else "")
  | Fiss (F_saved (segs, move)) ->
      let from, extension = move_label move in
      Printf.sprintf "F_saved[%d segs%s]%s" (List.length segs) from extension
  | Fiss (F_sketch { entries; staff; fine; _ }) -> fiss_sketch_label ~staff ~fine entries
  | Fiss (F_split { sites; staff }) -> split_sites_label ~staff sites
  | Fiss (F_seeded plan) -> seeded_plan_label plan
  | Fiss (F_split_saved (prelude, segs, move)) ->
      let from, extension = move_label move in
      Printf.sprintf "F_split_saved[%d prelude ops, %d segs%s]%s" (List.length prelude)
        (List.length segs) from extension

(* Which candidate [tune] attributes as the untuned default, by the label the measurement seam
   receives for it -- and the ONE place that decides it: the [default_seed_digest] attribution in
   [tune] reads this, so a test keyed on it cannot drift from what the report calls the default.
   With automatic scheduling inactive the untuned default is the unscheduled serial form, the
   baseline -- which a GPU backend never dispatches (gh-ocannl-532), so there the default reaches no
   seam at all; with it active, the [config_thresholds] seed reproduces the default pipeline
   exactly, but only when that pipeline fissions -- the whole-routine annotation it is otherwise has
   no reproducing candidate (Codex P1 on PR #279). *)
let default_seed_label ~backend_name =
  if not (Sched.automatic_schedule_active ~backend_name) then
    if Sched.backend_is_gpu backend_name then None else Some "baseline"
  else if Sched.default_pipeline_fissions () then
    Some
      (spec_label
         (Fiss (F_preset { block_size = None; privatize = false; config_thresholds = true })))
  else None

(* The model ranks several scheduled forms without compiling them. Keep this eager validation in
   that ranking loop: removing it made an invalid tensorized argmin displace a viable schedule and
   then fall back all the way to the default. This is no longer needed for exception attribution or
   advisory containment -- codegen carries the same typed cause -- only to preserve "best viable
   model candidate" selection without compiling every contender. [limits]' shuffle width lets it see
   the barriers the warp-shuffle renderings insert (gh-ocannl-1238), as codegen's does. *)
let validate_segments_for_model ?(fork_join = false) ~(limits : Ir.Backend_intf.hardware_limits)
    (segs : LL.optimized list) =
  let warp_size = Option.value limits.simdgroup_width ~default:0 in
  List.iter segs ~f:(fun (o : LL.optimized) ->
      LL.validate_parallel_classified ~warp_size ~reassociable:Ir.Online_softmax.reassociable_local
        ~fork_join o.LL.optimize_ctx.LL.placements o.LL.llc);
  segs

let seeded_preset ~limits seg = Sched.default_gpu ~limits seg

(* gh-ocannl-1348: a segment no sketch takes gets its accumulations' reduction loops sunk below
   their write loops, then the default preset of the interchanged code: the number of [Swap]s and
   the schedule, or [None] where nothing sinks. *)
let seeded_sunk_schedule ~limits ~accum_prec ~admit_narrow ~static_indices seg =
  match reduction_sink ~admit_narrow ~accum_prec seg with
  | [] -> None
  | swaps -> (
      match Sched.apply_classified ~static_indices swaps (scratch_of seg) with
      | exception Outcome.Cause_at _ -> None
      | swapped -> Some (List.length swaps, swaps @ seeded_preset ~limits swapped))

(* The seeded plan's segments of [opt], the routine its split (if any) has been applied to: the
   segmentation and [Sched.fission_segmented]'s tuples, and whether the plan's segments applied.
   Where the plan has none, or its segmentation is refused, these are the default pipeline's own
   segments ([maybe_default_schedules]'s, derived the same way). A plan segment is matched by
   position, as a replayed segmentation is; a slice whose code is not the decided one's keeps its
   preset. Every applied segment is validated, which raises on a parallelization the backend would
   refuse. *)
let seeded_replay ~limits ~accum_prec ~admit_narrow ~static_indices segments (opt : LL.optimized) =
  let preset = seeded_preset ~limits in
  let zero_sched tns = Sched.zero_expansion ~limits tns in
  let default () =
    Sched.fission_segmented ~promote_locals:true
      ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:true ~limits)
      ~preset ~zero_sched ~static_indices opt
  in
  match segments with
  | None -> (default (), false)
  | Some (plan, segs) -> (
      let segs = Array.of_list segs in
      let digest seg = SC.digest (SC.canonicalize ~static_indices ~with_placements:false seg) in
      let schedule i seg =
        match segs.(i) with
        | Seeded_sketch { digest = d; params = p } when String.equal d (digest seg) ->
            Some (sketch_schedule ~accum_prec ~p seg)
        | Seeded_sunk { digest = d; _ } when String.equal d (digest seg) ->
            Option.map
              (seeded_sunk_schedule ~limits ~accum_prec ~admit_narrow ~static_indices seg)
              ~f:snd
        | Seeded_preset | Seeded_sketch _ | Seeded_sunk _ -> None
      in
      match
        Sched.fission_segmented ~promote_locals:true ~replay:(plan, schedule) ~preset ~zero_sched
          ~static_indices opt
      with
      | (_, tuples) as segmented ->
          ignore
            (validate_segments_for_model ~limits (List.map tuples ~f:(fun (_, _, _, post) -> post))
              : LL.optimized list);
          (segmented, true)
      | exception Outcome.Cause_at (_, cause) ->
          logf "seeded_default: the seeded pipeline FAILED to apply (%s); using the default"
            (Outcome.detail_of_cause cause);
          (default (), false))

(* Every candidate derives its CODE from the ONE base lowering ([base_opt] with [canon] its
   canonical form, captured together in [tune]) rather than from the compile's own fresh lowering,
   whose llc the transform ignores. Re-lowering per candidate was subtly unsound: timing runs settle
   tensor-node value bounds, so later fresh lowerings can fold guards (and even re-segment fission)
   differently from the base — failing digest checks at best (the CUDA rounds on PR #140: whole arms
   degenerating to their serial baselines) and silently replaying the winner with empty per-segment
   schedules at worst (a 296 ms winner returning as a 2614 ms routine). Deriving from the base makes
   candidates and the winner replay drift-immune and byte-comparable by construction; the
   fresh-lowering digest check survives only in spirit via the disk cache's [source_digest] guard
   (cross-process compatibility).

   The rebased code keeps the fresh compile's OWN [optimize_ctx] (the per-compile fork of the
   context's lineage): link-time buffer allocation consults that fork, so placement mutations by
   schedule ops — fission's Local promotions above all — must land there or the allocator would miss
   buffers the kernels reference. Candidate hermeticity is unchanged: each compile forks the lineage
   table anew. The traced store is copied from the base (schedule ops register their tiles in it).

   gh-ocannl-1421: that is sound only while the fresh lowering is the base's program up to that
   drift. Within the lineage the base was lowered in it is. A compile into ANOTHER lineage -- the
   caller's, after a search on a [timing_ctx] -- can lower to another program: a node the timing
   lineage materialized and the caller's leaves virtual is inlined there, so the base's code reads a
   buffer the caller's placements never allocate. [tune] refuses that case up front
   ([Lineage_refused]), comparing the two lineages' lowerings taken together, so both see the same
   value bounds ([lineage_identity]); a comparison here, with a lowering taken after the timing
   runs, could not tell drift from a lineage difference.

   gh-ocannl-1340: two verdicts are decided inside the transform, on the forms it is about to hand
   to codegen, so a candidate they refuse costs no codegen, no device compile and no link — on HIP a
   serial whole-routine form is minutes of hiprtc (303 s per arm on a gfx1102). A form [seen]
   already holds is a [`Duplicate] (the search's dedup, which used to be decided after the compile
   it made redundant); on a GPU backend a form that binds no hardware dimension is refused with the
   typed [Not_dispatched] cause of {!dispatchable} (gh-ocannl-532), which every caller already
   rejected after its compile. The order is the search's: a duplicate of a refused form dedups
   rather than counting a second refusal, which is why a refused digest joins [seen]. The verdicts
   read the transform's forms, not what codegen shipped (zero-only segments can be elided after the
   transform), so the callers keep their post-codegen check, on [routine.segments]. *)
type gated_compile = Compiled of compiled | Duplicate of string | Failed of Outcome.failure

let not_dispatched_origin = function
  | Outcome.Cache_replay -> "cache_replay"
  | Outcome.Candidate | Outcome.Advisory | Outcome.User_schedule -> "candidate"

(* gh-ocannl-1480, gh-ocannl-1560: every whole-routine transform ahead of fission applies through
   here -- the split-reduce prelude, in the search's candidates, their replays and the seeded plan.
   Blind to node bounds candidates, so its trailing fold is a function of the code alone: the
   segments fission cuts from the result are what a replay's digests check, and a bounds candidate
   proposed by a link between a candidate's compile and the winner's replay (on GPU the search's
   first candidate compiles before anything has linked) used to fold a gather guard in the replay
   only -- the saved segmentation "no longer applied" and the untuned default shipped. The
   per-segment schedules still fold against the bounds as they stand: nothing checks their
   output. *)
let apply_before_fission ~static_indices prelude (opt : LL.optimized) =
  if List.is_empty prelude then opt
  else LL.without_node_bounds (fun () -> Sched.apply_classified ~static_indices prelude opt)

(* The replay a saved fissioned form compiles under ({!Sched.fission_segmented}'s [replay]): the
   routine cut where the saved segments say, the [i]th segment given the [i]th saved schedule -- by
   position, never by digest: the digest leaves placements out, and two segments of one structure
   can differ in placements, hence in kind and schedule. The replay-validity check rides on the same
   position: the segment cut there must be the one its schedule was saved against, otherwise the
   entry no longer applies and the candidate fails (a cache replay then re-searches) rather than
   running a segment under a schedule saved for other code. The check compares the STRUCTURAL canon
   ([with_placements:false]): placement classes can render differently across compilation lineages
   on byte-identical segments (decided in one, undecided in the other -- e.g. tuning with
   [timing_ctx]), which used to fail winner replays wholesale. *)
let saved_fission_replay ~static_indices (saved : SC.saved_segment list) =
  let segs = Array.of_list saved in
  let schedule i seg =
    let seg_canon = SC.canonicalize ~static_indices ~with_placements:false seg in
    let s = segs.(i) in
    if not (String.equal (SC.digest seg_canon) s.SC.seg_digest) then
      raise
        (Outcome.Cause_at
           ( Outcome.Transform,
             Outcome.Illegal_schedule
               {
                 check = "Autotune.fission_replay";
                 detail =
                   Printf.sprintf
                     "segment %d differs from the one its schedule was saved against: the saved \
                      segmentation no longer applies"
                     i;
               } ));
    Some (fst (SC.of_saved seg_canon s.SC.seg_saved))
  in
  (SC.segmentation_of saved, schedule)

(* gh-ocannl-1560: the [autotune_strict_replay] check. Read per search, so a test can state it for
   one search. *)
let strict_replay_enabled () =
  Utils.get_global_flag ~default:false ~arg_name:"autotune_strict_replay"

(* Whether a fissioned candidate's saved form replays: the code a winner's replay ({!tune}'s
   [winner_compile]) or a cache hit would derive from it -- the saved prelude decoded and applied,
   then the saved segmentation cut with every saved schedule digest-checked by position and applied
   -- rebuilt now, after the candidate's own link, on [base_opt] and [pristine] (a copy of the
   candidate's compile-fork placements taken as its transform began). [None] when it replays, else
   the refusal's detail. A whole-routine form has no segmentation to check: its replay decodes its
   schedule against [canon] and checks no digest. Point-in-time: it sees the process state of this
   moment, so it catches a transform reading state the candidate's own link changed
   (gh-ocannl-1480's CUDA shape, where the first candidate's link uploads the index tensor) and any
   encoding asymmetry between what a candidate records and what a replay reads back
   (gh-ocannl-1480's single-kernel head zero), not state a later, unrelated link changes. *)
let strict_replay_refusal ~static_indices ~canon ~limits ~is_gpu ~(base_opt : LL.optimized)
    ~pristine form =
  match form with
  | Whole_saved _ -> None
  | Fiss_saved segs | Split_saved (_, segs) -> (
      let opt =
        {
          base_opt with
          LL.traced_store = Hashtbl.copy base_opt.LL.traced_store;
          LL.optimize_ctx = pristine;
        }
      in
      match
        let opt =
          match form with
          | Split_saved (psaved, _) ->
              apply_before_fission ~static_indices (fst (SC.of_saved canon psaved)) opt
          | Whole_saved _ | Fiss_saved _ -> opt
        in
        tuner_fission ~is_gpu ~limits
          ~replay:(saved_fission_replay ~static_indices segs)
          ~preset:(fun _ -> [])
          ~static_indices opt
      with
      | (_ : Sched.segmentation * _) -> None
      | exception Outcome.Cause_at (_, cause) -> Some (Outcome.detail_of_cause cause))

let compile_candidate_gated ?name ?label ?seen ?(on_privatize_drops = fun _ -> ()) ?retain
    ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu ~provenance ctx comp bindings spec :
    gated_compile =
  let candidate = match label with Some l -> l | None -> spec_label spec in
  let duplicate = ref None in
  let gate ~digest_after opts =
    if Option.exists seen ~f:(fun seen -> Hash_set.mem seen digest_after) then (
      duplicate := Some digest_after;
      (* Unwinds out of the backend's compile before codegen; the [duplicate] cell, not this cause,
         is what the result reports. *)
      raise
        (Outcome.Cause_at
           ( Outcome.Transform,
             Outcome.Not_dispatched { origin = "dedup"; detail = candidate ^ " is a duplicate" } )));
    if not (dispatchable ~is_gpu opts) then (
      Option.iter seen ~f:(fun seen -> Hash_set.add seen digest_after);
      raise
        (Outcome.Cause_at
           ( Outcome.Transform,
             Outcome.Not_dispatched
               {
                 origin = not_dispatched_origin provenance;
                 detail =
                   Printf.sprintf
                     "%s binds no hardware dimension (gh-ocannl-532), refused before codegen \
                      (digest %s)"
                     candidate (dshort digest_after);
               } )))
  in
  let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
  let rebase (fresh : LL.optimized) =
    {
      base_opt with
      LL.traced_store = Hashtbl.copy base_opt.LL.traced_store;
      LL.optimize_ctx = fresh.LL.optimize_ctx;
    }
  in
  let preset_sched ?block_size ?(config_thresholds = false) opt =
    segment_preset ?block_size
      ~presets:(if config_thresholds then `Config else `Search)
      ~is_gpu ~is_cpu ~limits opt
  in
  let captured = ref None in
  (* gh-ocannl-1560: under [autotune_strict_replay], a searched candidate's compile-fork placements
     as its transform received them, for {!strict_replay_refusal}. *)
  let strict_replay =
    match provenance with
    | Outcome.Candidate -> strict_replay_enabled ()
    | Outcome.Cache_replay | Outcome.Advisory | Outcome.User_schedule -> false
  in
  let pristine = ref None in
  let compile_ctx () =
    match spec with
    | Whole flavor ->
        let transform fresh =
          let opt = rebase fresh in
          let sched, saved, registry =
            match flavor with
            | W_saved (saved, _) ->
                let sched, registry = SC.of_saved canon saved in
                (sched, saved, registry)
            | W_preset { block_size } ->
                let sched = preset_sched ?block_size opt in
                let saved, registry = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved, registry)
            | W_sketch p ->
                let sched = sketch_schedule ~accum_prec ~p opt in
                let saved, registry = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved, registry)
          in
          let opt' = Sched.apply_classified ~static_indices sched opt in
          let digest_after = SC.digest (SC.canonicalize ~static_indices opt') in
          captured :=
            Some
              ( Whole_saved saved,
                [ { u_segment = None; u_saved = saved; u_registry = registry; u_opt = opt' } ],
                [ opt' ],
                digest_after );
          gate ~digest_after [ opt' ];
          [ opt' ]
        in
        Context.compile_outcome ?name ~lowered_transform:transform ~provenance ~candidate ?retain
          ctx comp bindings
    | Fiss flavor ->
        let transforms fresh =
          if strict_replay then pristine := Some (LL.copy_optimize_ctx fresh.LL.optimize_ctx);
          let opt = rebase fresh in
          (* The split-reduce prelude (gh-ocannl-484 task 3) applies whole-routine BEFORE fission:
             the partials edge it mints is what fission cuts at, giving the two passes separate
             kernels and the event-chain synchronization the combine needs. *)
          let prelude, prelude_saved =
            match flavor with
            | F_preset _ | F_saved _ | F_sketch _ -> ([], [])
            | F_split { sites; staff } ->
                let sched = split_reduce_prelude (sites @ staff) in
                let saved, _ = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved)
            | F_seeded { sp_split = _ :: _ as sites; _ } ->
                let sched = split_reduce_prelude sites in
                let saved, _ = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved)
            | F_seeded { sp_split = []; _ } -> ([], [])
            | F_split_saved (psaved, _, _) ->
                let sched, _ = SC.of_saved canon psaved in
                (sched, psaved)
          in
          let opt = apply_before_fission ~static_indices prelude opt in
          (* A replay cuts where its saved segments say ({!saved_fission_replay}); the sketch
             candidates replay the segmentation their seeds were enumerated under the same way
             ({!sketch_replay}). *)
          let replay =
            match flavor with
            | F_preset _ | F_split _ | F_seeded _ -> None
            | F_sketch { plan; entries; staff; _ } ->
                Some
                  (sketch_replay ~static_indices
                     ~sketch:(fun p seg -> sketch_schedule ~accum_prec ~p seg)
                     plan (entries @ staff))
            | F_saved (saved, _) | F_split_saved (_, saved, _) ->
                Some (saved_fission_replay ~static_indices saved)
          in
          let preset seg =
            match flavor with
            | F_preset { block_size; privatize; config_thresholds } ->
                let sched = preset_sched ?block_size ~config_thresholds seg in
                if privatize then (
                  (* gh-ocannl-1438: what the extension dropped reaches the search's report. *)
                  let sched, drops = privatize_extension ~accum_prec ~static_indices sched seg in
                  on_privatize_drops drops;
                  sched)
                else sched
            | F_saved _ | F_split_saved _ | F_seeded _ ->
                [] (* Never consulted: a replay brings its own. *)
            | F_sketch _ | F_split _ -> preset_sched seg
          in
          (* Neither the [arity_cuts] (finer) segmentation (gh-ocannl-574) nor the zero fold
             (gh-ocannl-1175) reaches a replay: the sketches' recorded segmentation was enumerated
             under both, and carries them as data. The preset flavors keep the untuned pipeline's
             coarse cuts and separate zero. *)
          let segmentation, tuples =
            match flavor with
            | F_seeded plan ->
                (* The untuned compile's own replay, so the candidate is the step it ships, its
                   fallback to the default segments included. *)
                let segmented, applied =
                  seeded_replay ~limits ~accum_prec ~admit_narrow:plan.sp_admit_narrow
                    ~static_indices plan.sp_segments opt
                in
                if Option.is_some plan.sp_segments && not applied then
                  logf "%s: the plan's segments did not apply; compiled the default segments"
                    candidate;
                segmented
            | F_preset _ | F_saved _ | F_sketch _ | F_split _ | F_split_saved _ ->
                (* Match the default pipeline's placements (statement-crossing [Local]s promoted on
                   GPU), so fissioned candidates and the untuned baseline schedule the same code. *)
                tuner_fission ~is_gpu ~limits ?replay ~preset ~static_indices opt
          in
          let saved_with_registries = SC.save_segments ~static_indices segmentation tuples in
          let posts = List.map tuples ~f:(fun (_, _, _, post) -> post) in
          let units =
            List.filter_mapi (List.zip_exn saved_with_registries posts)
              ~f:(fun i ((s, registry), post) ->
                match Sched.segment_class s.SC.seg_kind with
                | `Zeros | `Solo -> None
                | `Normal ->
                    Some
                      {
                        u_segment = Some i;
                        u_saved = s.SC.seg_saved;
                        u_registry = registry;
                        u_opt = post;
                      })
          in
          let digest_after =
            String.concat ~sep:"+"
              (List.map posts ~f:(fun post -> SC.digest (SC.canonicalize ~static_indices post)))
          in
          let segs = List.map saved_with_registries ~f:fst in
          let form =
            if List.is_empty prelude_saved then Fiss_saved segs
            else Split_saved (prelude_saved, segs)
          in
          captured := Some (form, units, posts, digest_after);
          gate ~digest_after posts;
          posts
        in
        Context.compile_outcome ?name ~lowered_transform:transforms ~provenance ~candidate ?retain
          ctx comp bindings
  in
  (* The [Tile_mma] rendering census travels on the routine ([Context.routine.mma], gh-ocannl-626):
     [Context.compile] collects it around this candidate's kernel compiles, fissioned segments
     included, so no bracket is needed here and a candidate cannot be timed without it. *)
  match compile_ctx () with
  | Error _ when Option.is_some !duplicate -> Duplicate (Option.value_exn !duplicate)
  | Error failure -> Failed failure
  | Ok (cctx, routine) -> (
      match !captured with
      | Some (form, units, all_opts, digest_after) -> (
          let refusal =
            match !pristine with
            | Some pristine when strict_replay ->
                strict_replay_refusal ~static_indices ~canon ~limits ~is_gpu ~base_opt ~pristine
                  form
            | Some _ | None -> None
          in
          match refusal with
          | None -> Compiled { form; cctx; routine; units; all_opts; digest_after }
          | Some detail -> (
              (* Fatal whatever [strict_failure_classification] says: the check was asked for, and a
                 form the search timed that its own replay refuses is what it exists to stop. *)
              Context.release cctx;
              match
                Outcome.protect ~strict:true
                  ~classify_backend:(fun _ _ -> None)
                  ~provenance ~phase:Outcome.Transform ~candidate
                  (fun () ->
                    failwith
                      (Printf.sprintf
                         "Autotune: autotune_strict_replay: candidate %s does not replay from its \
                          saved form (gh-ocannl-1560): %s"
                         candidate detail))
              with
              | Ok (_ : compiled) -> assert false
              | Error failure -> Failed failure))
      | None -> (
          match
            Outcome.protect
              ~classify_backend:(fun _ _ -> None)
              ~provenance ~phase:Outcome.Transform ~candidate
              (fun () -> failwith "Autotune: the transform was not invoked")
          with
          | Ok (_ : compiled) -> assert false
          | Error failure -> Failed failure))

(* The callers that do not deduplicate: a refused form arrives as the typed [Not_dispatched]
   failure, which each of them already handles as a failed or degenerate compile. *)
let compile_candidate ?name ?label ?on_privatize_drops ?retain ~static_indices ~base_opt ~canon
    ~limits ~is_gpu ~is_cpu ~provenance ctx comp bindings spec : compiled Outcome.outcome =
  match
    compile_candidate_gated ?name ?label ?on_privatize_drops ?retain ~static_indices ~base_opt
      ~canon ~limits ~is_gpu ~is_cpu ~provenance ctx comp bindings spec
  with
  | Compiled c -> Ok c
  | Failed failure -> Error failure
  | Duplicate _ -> assert false (* No [seen], no duplicate. *)

(* The post-codegen check behind the gate: whether the kernels codegen SHIPPED ([routine.segments],
   gh-ocannl-1006) bind a hardware dimension -- the artifact, not the transform's forms the gate
   judged. A routine that recorded no segments is judged by its forms. *)
let shipped_dispatchable ~is_gpu (c : compiled) =
  dispatchable ~is_gpu (match c.routine.Context.segments with [] -> c.all_opts | segs -> segs)

(* Whether a failure is the gate's pre-codegen refusal above. *)
let is_not_dispatched = function
  | Outcome.Classified { cause = Outcome.Not_dispatched _; _ } -> true
  | Outcome.Classified _ | Outcome.Fatal _ -> false

(** {2 The action menu} *)

type loop_desc = {
  ld_ref : SC.sym_ref;
  ld_sym : Idx.symbol;  (** The raw binder, for consulting {!Sched.op_legality}. *)
  ld_from_ : int;
      (** The loop's lower bound. [Partition] segments after the first start at their breakpoint
          (segment ranges stay absolute), and only [Split] among the proposed ops requires a
          zero-origin loop — [Swap], [Unroll] (either representation) and non-hardware [Retype]s are
          origin-agnostic, so nonzero-origin loops stay enumerated for them (Codex P2 on PR #403).
      *)
  ld_extent : int;
  ld_axis : LL.axis_type;
  ld_innermost : bool;
  ld_accumulating : bool;
  ld_inlined : bool;
      (** Reached by descending through a virtualization-inlined [Local_scope] (gh-ocannl-687).
          Decides which action CATEGORIES may propose for it, not whether it is enumerated. *)
  ld_perfect_child : (SC.sym_ref * Idx.symbol * LL.axis_type) option;
}

(* The [Local_scope] bodies in the scalar positions of one statement (empty for non-writes) — where
   the accumulation mints of [Unroll ~materialize:true] and [Partition] (gh-ocannl-639) and
   virtualization's inlined computations put loops. This is the scalar-position reach of
   [Schedule.rewrite_loop] and [Schedule.find_loops_env]; like them it enters neither [If]
   conditions nor [Tile_mma] fallbacks (transforming those is never profitable and often invalid).

   Each body comes with its scope's provenance (gh-ocannl-687), which every caller here descends
   into but which decides what the menu then proposes for the loops found there. *)
let stmt_scope_bodies (stmt : LL.t) : (LL.t * LL.scope_mint) list =
  let rec scalar (llsc : LL.scalar_t) =
    match llsc with
    | LL.Local_scope { body; mint; _ } -> [ (body, mint) ]
    | LL.Get_dynamic { dyn_value = v, _; _ } -> scalar v
    | LL.Ternop (_, (a, _), (b, _), (c, _)) -> scalar a @ scalar b @ scalar c
    | LL.Binop (_, (a, _), (b, _)) -> scalar a @ scalar b
    | LL.Unop (_, (a, _)) -> scalar a
    | LL.Get_local _ | LL.Get _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _
    | LL.Embed_index _ ->
        []
  in
  match stmt with
  | LL.Set { llsc; _ } | LL.Set_local (_, llsc) -> scalar llsc
  | LL.Set_dynamic { dyn_value = v, _; llsc; _ } -> scalar v @ scalar llsc
  | LL.Set_from_vec { arg = a, _; _ } -> scalar a
  | _ -> []

(* Whether [llc] contains a nested loop, [Local_scope] bodies of BOTH provenances included
   (gh-ocannl-666), so a loop whose only inner loops sit inside a scope does not read as innermost.
   Provenance-blind on purpose (PR #424 review round 2): innermost-ness decides which loop a
   [Vectorized] retype is proposed for, and the renderer answers that question structurally. An
   outer loop whose body holds a [Local_scope] cannot be explicitly vectorized at all — [C_syntax]'s
   elementwise vectorizer bails on [Local_scope] / [Get_local], and an accumulating bailout falls
   back to a plain serial loop — so calling it innermost would propose a retype that renders exactly
   like the baseline. The inlined reduction one level down is the loop that has a renderer
   ([try_vectorize_reduce] recognizes its [Set_local] accumulation), which is why [collect_loops]
   keeps enumerating it. *)
let rec contains_loop = function
  | LL.Seq (a, b) -> contains_loop a || contains_loop b
  | LL.If { body; _ } -> contains_loop body
  | LL.For_loop _ -> true
  | stmt -> List.exists (stmt_scope_bodies stmt) ~f:(fun (b, _) -> contains_loop b)

(* Loops proposable for schedule ops: the statement-level nest structure plus the loops inside
   [Local_scope] bodies (gh-ocannl-666) — since gh-ocannl-639 the accumulation mints of [Unroll
   ~materialize:true] and [Partition] move segment/inner loops inside the accumulator's scope, and
   [Schedule.rewrite_loop] reaches them there, so the menu must enumerate them or the moment such an
   op joins an incumbent every inner loop vanishes from the rest of the search. Restricted to loops
   whose binder the registry can name (Stage-internal copy loops cannot be referenced by a persisted
   schedule), and deduplicated by binder: a materializing mint copies its body per step/segment
   WITHOUT refreshing loop symbols, so sibling copies share binders and [rewrite_loop] rewrites them
   all — one binder is one scheduling decision. Scope-nested descriptors are safe for every op the
   menu proposes from them (serial [Split]s, [Swap]s, [Unroll]s, [Vectorized] retypes — none
   introduces a hardware annotation, which [Low_level.validate_parallel] rejects inside a
   [Local_scope]); [Tensorize] is the exception, which is why [collect_serial_triples] below stays
   out of scopes.

   Both provenances are entered, and each descriptor records which (gh-ocannl-687's [ld_inlined]).
   The distinction is NOT about reachability — [Schedule.rewrite_loop] descends every [Local_scope],
   so a proposal naming an inlined loop applies — and it is not about which loops exist. It is about
   which CATEGORY is worth a candidate compile on a loop virtualization re-instantiates per use site
   (PR #424 review round 2, correcting a first attempt that dropped such loops wholesale):

   - [Vectorized] retypes stay proposable there, and must. That is one descriptor, and it is the
   only one with a renderer built for the shape: an inlined reduction's [Set_local] accumulation is
   exactly what [C_syntax.try_vectorize_reduce] recognizes, while the enclosing loop cannot be
   explicitly vectorized at all (its body holds a [Local_scope]). Excluding the inner loop does not
   move the candidate outward, it destroys it. - [Split]s, [Swap]s and [Unroll]s do not. Up to eight
   descriptors per loop, no evidence any of them pays on a per-use-site inline, and each one costs a
   candidate compile and — under the per-unit cap — displaces a proposal for the main nest. Nothing
   proposed them before gh-666, whose widening was aimed at the accumulation mints and swept these
   in for want of provenance. *)
let collect_loops registry llc =
  let acc = ref [] in
  let seen = Hash_set.create (module Idx.Symbol) in
  let rec walk ~inlined = function
    | LL.Seq (a, b) ->
        walk ~inlined a;
        walk ~inlined b
    | LL.If { body; _ } -> walk ~inlined body
    | LL.For_loop { index; from_; to_; body; axis; _ } ->
        (match SC.resolve registry index with
        | Some ld_ref when not (Hash_set.mem seen index) ->
            Hash_set.add seen index;
            let ld_perfect_child =
              match body with
              | LL.For_loop { index = ci; axis = cax; _ } ->
                  Option.map (SC.resolve registry ci) ~f:(fun r -> (r, ci, cax))
              | _ -> None
            in
            acc :=
              {
                ld_ref;
                ld_sym = index;
                ld_from_ = from_;
                ld_extent = to_ - from_ + 1;
                ld_axis = axis;
                ld_innermost = not (contains_loop body);
                ld_accumulating = LL.has_accumulation body;
                ld_inlined = inlined;
                ld_perfect_child;
              }
              :: !acc
        | _ -> ());
        walk ~inlined body
    | stmt ->
        List.iter (stmt_scope_bodies stmt) ~f:(fun (body, mint) ->
            walk ~inlined:(inlined || LL.equal_scope_mint mint LL.Inlined_computation) body)
  in
  walk ~inlined:false llc;
  List.rev !acc

(* Perfectly nested serial triples (with extents), for Tensorize proposals. Statement-level only,
   deliberately (gh-ocannl-666): [Tensorize] wraps the micro-kernel in a hardware-annotated
   [Workgroup] lane loop, which the scope grammar forbids inside a [Local_scope] body — a
   scope-nested triple can never apply ([Schedule.op_legality] would prune it, gh-ocannl-1467, but
   only after probing the transform on it, which skipping it saves). No candidates are lost: the
   loops inside an accumulation mint all reduce into the scope's single loop-invariant cell, so none
   can play the micro-kernel's output-dim [i]/[j] roles and every role assignment over such a triple
   would be refuted anyway. Deduplicated by the outer binder, since a non-minting materializing
   [Unroll] leaves statement-level copies sharing loop symbols. *)
let collect_serial_triples registry llc =
  let acc = ref [] in
  let seen = Hash_set.create (module Idx.Symbol) in
  let rec walk = function
    | LL.Seq (a, b) ->
        walk a;
        walk b
    | LL.If { body; _ } -> walk body
    | LL.For_loop { index = i; from_ = 0; to_ = ti; axis = LL.Serial; body; _ }
      when not (Hash_set.mem seen i) ->
        (match body with
        | LL.For_loop
            {
              index = j;
              from_ = 0;
              to_ = tj;
              axis = LL.Serial;
              body = LL.For_loop { index = k; from_ = 0; to_ = tk; axis = LL.Serial; body = b3; _ };
              _;
            }
          when not (contains_loop b3) -> (
            match (SC.resolve registry i, SC.resolve registry j, SC.resolve registry k) with
            | Some ri, Some rj, Some rk ->
                Hash_set.add seen i;
                acc := ((ri, i, ti + 1), (rj, j, tj + 1), (rk, k, tk + 1)) :: !acc
            | _ -> ())
        | _ -> ());
        walk body
    | LL.For_loop { body; _ } -> walk body
    | _ -> ()
  in
  walk llc;
  List.rev !acc

let split_factors = [ 2; 4; 8; 16; 32 ]
let max_actions_per_unit = 48

(* gh-ocannl-685: share a cap across the menu's action categories instead of spending it in category
   order. The menu list is a concatenation ordered by category and NOT ranked (unlike the placement
   surface's [rank_flip_candidates] prefix, where top-N is the intended semantics), so a plain
   prefix is arbitrary: a unit whose tensorizes alone reach the cap offered the search no split,
   swap, unroll or vectorize action at all -- not fewer, none -- and those are exactly the
   categories a unit needs when its tensorizes turn out [Op_illegal] or unprofitable.

   Round-robin, one proposal per category per round, in category order: every non-empty category is
   represented before any category gets a second, and a category that runs out simply stops taking
   turns, so its unused share spills to the others without anyone naming a ranking the tuner is
   supposed to discover. Survivors are emitted in the original category order, so a menu that fits
   under the cap comes out exactly as before. Returns the kept proposals and the per-category drop
   counts, which the caller logs -- the cap must say what it dropped, not what it was offered. *)
let share_cap ~cap (categories : (string * 'a list) list) : 'a list * (string * int) list =
  let sizes = Array.of_list_map categories ~f:(fun (_, l) -> List.length l) in
  let keep = Array.map sizes ~f:(fun _ -> 0) in
  let budget = ref (max 0 cap) in
  let progressed = ref true in
  while !budget > 0 && !progressed do
    progressed := false;
    Array.iteri sizes ~f:(fun idx n ->
        if !budget > 0 && keep.(idx) < n then (
          keep.(idx) <- keep.(idx) + 1;
          Int.decr budget;
          progressed := true))
  done;
  let kept = List.concat_mapi categories ~f:(fun idx (_, l) -> List.take l keep.(idx)) in
  let dropped =
    List.filter_mapi categories ~f:(fun idx (name, l) ->
        let d = List.length l - keep.(idx) in
        if d > 0 then Some (name, d) else None)
  in
  (kept, dropped)

let menu ?(admits = fun (_ : SC.saved_optop) -> true) ~is_cpu ~is_gpu
    ~(limits : Ir.Backend_intf.hardware_limits) ~registry (opt : LL.optimized) : SC.saved_optop list
    =
  let loops = collect_loops registry opt.LL.llc in
  (* gh-ocannl-687: how many enumerated loops are offered the [Vectorized] retype alone because
     virtualization inlined them, so the narrowing is visible rather than silent. *)
  let inlined_loops = List.count loops ~f:(fun ld -> ld.ld_inlined) in
  (* Menu proposals carry their raw-symbol counterpart so the op-legality oracle (gh-494 waypoint 3)
     can veto proven-illegal ones before they cost a candidate compile; [Op_unknown] proposals
     proceed to compile-and-time exactly as before (the oracle's Unknown is never a rejection). *)
  let gate (saved, raw) =
    match Sched.op_legality opt raw with
    | Sched.Op_illegal witness ->
        logf "menu prune (illegal): %s" witness;
        None
    | Sched.Op_legal | Sched.Op_unknown _ -> Some saved
  in
  let splits =
    List.concat_map loops ~f:(fun ld ->
        (* [Sched.Split]'s index arithmetic requires a zero-origin loop (its apply raises
           otherwise); nonzero-origin loops — [Partition] segments after the first — are still in
           [loops] for the origin-agnostic families below. *)
        if (not (LL.equal_axis_type ld.ld_axis LL.Serial)) || ld.ld_from_ <> 0 || ld.ld_inlined then
          []
        else
          List.filter_map split_factors ~f:(fun factor ->
              if factor < ld.ld_extent && ld.ld_extent % factor = 0 then
                let raw, _, _ =
                  Sched.split ~axis:ld.ld_sym ~factor ~outer:LL.Serial ~inner:LL.Serial
                in
                gate
                  (SC.Split { axis = ld.ld_ref; factor; outer = LL.Serial; inner = LL.Serial }, raw)
              else None))
  in
  let swaps =
    List.filter_map loops ~f:(fun ld ->
        match (ld.ld_axis, ld.ld_perfect_child) with
        | LL.Serial, Some (child, child_sym, LL.Serial) when not ld.ld_inlined ->
            gate
              ( SC.Swap { outer = ld.ld_ref; inner = child },
                Sched.Swap { outer = ld.ld_sym; inner = child_sym } )
        | _ -> None)
  in
  let unrolls =
    List.concat_map loops ~f:(fun ld ->
        if LL.equal_axis_type ld.ld_axis LL.Serial && ld.ld_extent <= 8 && not ld.ld_inlined then
          List.filter_map [ true; false ] ~f:(fun materialize ->
              gate
                ( SC.Unroll { axis = ld.ld_ref; materialize },
                  Sched.Unroll { axis = ld.ld_sym; materialize } ))
        else [])
  in
  let vectorizes =
    (* CPU renders eligible retyped loops via vector extensions (or vectorization pragmas); GPU
       backends render them as 128-bit packed loads/stores (gh-ocannl-463). Ineligible candidates
       fall back to plain serial loops, so a proposal that fails codegen eligibility merely times
       like the baseline. Accumulating bodies are proposable on CPU (gh-ocannl-468): the renderer
       either emits the reduction-chains rendering or falls back to a plain serial loop — never to a
       vectorization pragma, which would assert iteration independence the loop-carried accumulation
       does not satisfy. On GPU the reduction rendering does not exist (reductions parallelize via
       [Workgroup_reduce] instead), so accumulations stay excluded. *)
    if not (is_cpu || is_gpu) then []
    else
      List.filter_map loops ~f:(fun ld ->
          if
            LL.equal_axis_type ld.ld_axis LL.Serial
            && ld.ld_innermost
            && ((not ld.ld_accumulating) || is_cpu)
          then
            gate
              ( SC.Retype { axis = ld.ld_ref; ty = LL.Vectorized },
                Sched.Retype { axis = ld.ld_sym; ty = LL.Vectorized } )
          else None)
  in
  let triples = collect_serial_triples registry opt.LL.llc in
  let tensorizes =
    match limits.Ir.Backend_intf.mma with
    | None -> []
    | Some { Ir.Backend_intf.mma_simd_width; mma_tile = tm, tn, tk; _ } ->
        (* The nesting order need not match the (i, j, k) roles — the roles are fixed by the
           accumulation pattern. The op-legality oracle decides role-assignment validity (gh-494
           waypoint 3 follow-up): invalid permutations — most of the 6 per triple — are proven
           [Op_illegal] by the probe of apply's micro-kernel recognition and pruned before they cost
           a candidate compile, instead of failing at compile time. Propose role assignments
           compatible with the intrinsic tile's divisibility per role. *)
        List.concat_map triples ~f:(fun (t1, t2, t3) ->
            List.filter_map
              [ (t1, t2, t3); (t1, t3, t2); (t2, t1, t3); (t2, t3, t1); (t3, t1, t2); (t3, t2, t1) ]
              ~f:(fun ((i, si, ei), (j, sj, ej), (k, sk, ek)) ->
                if ei % tm = 0 && ej % tn = 0 && ek % tk = 0 then
                  let raw, _lane =
                    Sched.tensorize ~i:si ~j:sj ~k:sk ~simd_width:mma_simd_width ()
                  in
                  gate (SC.Tensorize { i; j; k; simd_width = mma_simd_width; tile = None }, raw)
                else None))
  in
  logf
    "menu: %d serial triple(s) -> %d tensorize proposal(s); %d split, %d swap, %d unroll, %d \
     vectorize"
    (List.length triples) (List.length tensorizes) (List.length splits) (List.length swaps)
    (List.length unrolls) (List.length vectorizes);
  if inlined_loops > 0 then
    logf
      "menu: %d loop(s) inside virtualization-inlined scopes offered the Vectorized retype only \
       (gh-ocannl-687: no Split/Swap/Unroll proposals for a per-use-site inline)"
      inlined_loops;
  (* gh-ocannl-685 review: [admits] runs BEFORE the cap, so the budget is shared over the moves the
     caller can actually use rather than over moves it is about to discard. The beam's GPU rule is
     the case that matters: expanding an incumbent that binds no hardware dimension is worthwhile
     only through moves that can bind one, and sharing 48 slots across five categories first would
     leave a tensorize-rich unit ~10 tensorizes plus dozens of proposals the beam drops — where the
     old prefix, by accident of ordering, handed all 48 to the tensorizes. Filtering first makes
     that outcome the rule rather than the accident, and leaves the sharing to decide between
     categories the caller is actually choosing among. It may record its refusals; the drop log
     below is about the cap alone. *)
  let kept, dropped =
    share_cap ~cap:max_actions_per_unit
      (List.map
         [
           ("tensorize", tensorizes);
           ("split", splits);
           ("swap", swaps);
           ("unroll", unrolls);
           ("vectorize", vectorizes);
         ]
         ~f:(fun (name, l) -> (name, List.filter l ~f:admits)))
  in
  if not (List.is_empty dropped) then
    logf "menu: the per-unit cap of %d dropped %s" max_actions_per_unit
      (String.concat ~sep:", "
         (List.map dropped ~f:(fun (name, d) -> Printf.sprintf "%d %s" d name)));
  kept

(* Extend one unit of a compiled candidate with a menu action: a segment's unit extends that segment
   alone, by position — a structurally identical segment elsewhere can differ in placements and
   keeps its own schedule. *)
let extend_spec (elem : compiled) (u : unit_gen) (op : SC.saved_optop) : spec option =
  let extend segs i =
    List.mapi segs ~f:(fun j (s : SC.saved_segment) ->
        if i = j then { s with seg_saved = u.u_saved @ [ op ] } else s)
  in
  let move mv_segment = { mv_parent = dshort elem.digest_after; mv_segment; mv_op = op } in
  match (elem.form, u.u_segment) with
  | Whole_saved _, None -> Some (Whole (W_saved (u.u_saved @ [ op ], Some (move None))))
  | Fiss_saved segs, Some i -> Some (Fiss (F_saved (extend segs i, Some (move (Some i)))))
  | Split_saved (prelude, segs), Some i ->
      Some (Fiss (F_split_saved (prelude, extend segs i, Some (move (Some i)))))
  | _ -> None

(** {2 The placement decision surface (gh-ocannl-514, the placement-space search)}

    The per-node inline/materialize levels of the joint decision space, prepared for search: the
    deduplicated flip candidates ranked enablement-first (the gh-ocannl-558 lesson — a flip's value
    includes which sketch families become expressible under it, which the recompute-cost bound has
    no term for), and the roofline floor of a partial placement vector (phase 3's
    [Cost_model.completion_floor] on the all-materialized specialization — the bound that
    differentiates placement commitments, where the family levels' floor is schedule-invariant). *)

(* The mma-eligible matmul sites of a lowering, seen the way the seeders see them: whole-routine and
   per-fission-segment (the [F_sketch] granularity — [fission_scheduled] with empty per-segment
   schedules, since only the pre-schedule segment slices are consulted). Fission not applying
   degrades to the whole-routine site; a classified rejection degrades likewise rather than failing
   the caller (the classification is a ranking input, not a legality fact).

   Segmented as THIS backend's seeders segment ([tune]'s [F_sketch] flavor and [model_default] pass
   [Sched.fission_keep_mapping ~is_gpu]): cc advertises an [mma] descriptor too, and segmenting it
   the GPU way both judged its sites on a segmentation its own families never see and read the GPU
   lane keys ([gpu_fission_keep_mapping], [gpu_serial_lanes]) on a backend whose searches never read
   them (gh-ocannl-1430, raised on gh-ocannl-1331). *)
let mma_eligible_sites ~is_gpu ~(limits : Ir.Backend_intf.hardware_limits) ~static_indices
    (opt : LL.optimized) : matmul_site list =
  match limits.Ir.Backend_intf.mma with
  | None -> []
  | Some mma ->
      let segments =
        match
          Sched.fission_scheduled
            ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
            ~preset:(fun _ -> [])
            ~zero_sched:(fun _ -> [])
            ~static_indices (scratch_of opt)
        with
        | tuples ->
            List.filter_map tuples ~f:(fun (kind, pre, _sched, _post) ->
                match kind with `Normal -> Some pre | `Zeros | `Solo -> None)
        | exception Outcome.Cause_at _ -> [ opt ]
      in
      List.filter_map segments ~f:(fun seg ->
          Option.map (detect_matmul_canonical seg) ~f:(fun (site, _, _) -> site))
      |> List.filter ~f:(fun site ->
          let a_prec = Lazy.force site.m_a.Ir.Tnode.storage_prec in
          let b_prec = Lazy.force site.m_b.Ir.Tnode.storage_prec in
          let d_prec = Lazy.force site.m_d.Ir.Tnode.storage_prec in
          Option.is_some
            (mma_tile_for_precisions_in_scope mma ~scope:(matmul_mma_scope site ~bk:0) ~a_prec
               ~b_prec ~d_prec))

let placement_enablement ~is_gpu ~limits ~static_indices ~(base : LL.optimized)
    ~(allmat : LL.optimized) =
  let site_tns sites =
    List.fold sites
      ~init:(Set.empty (module Ir.Tnode))
      ~f:(fun acc site -> Set.add (Set.add (Set.add acc site.m_a) site.m_b) site.m_d)
  in
  let base_sites = mma_eligible_sites ~is_gpu ~limits ~static_indices base in
  let allmat_sites = mma_eligible_sites ~is_gpu ~limits ~static_indices allmat in
  (* An all-materialized site whose destination already carries an eligible default-placement site
     is not enablement: the family is expressible either way, and promoting its operands would rank
     ordinary mma-adjacent flips above genuinely family-unlocking ones. *)
  let base_dests =
    List.fold base_sites
      ~init:(Set.empty (module Ir.Tnode))
      ~f:(fun acc site -> Set.add acc site.m_d)
  in
  let enabling_sites =
    List.filter allmat_sites ~f:(fun site -> not (Set.mem base_dests site.m_d))
  in
  (site_tns enabling_sites, site_tns base_sites)

(* gh-ocannl-579: the profitability term. The enablement prior above prices EXPRESSIBILITY — which
   sketch families a placement makes reachable — and nothing else, so it promotes a flip whose
   family this device has already been measured to lose with. The evidence that settles it is in
   hand at the only place the prior is consumed: [Train.tune_placements] searches arm B
   (materialize-all) — the very specialization [placement_enablement] derives [enablement] from —
   before the flip chain walks, and its report says what the tensorized family was worth here
   ([mma_best_ms] against [best_ms], same device, same computation, same session, already paid
   for). *)

type family_profit = Unmeasured | Pays of float | Loses of float

(* Most favourable evidence wins: the prior is deleted only by evidence that contradicts it, never
   by the mere absence of a confirmation, and a single arm that measured a competitive family
   outranks another that measured a losing one (the arms search different placements; the promotion
   is a bet on the best placement reachable, not on the average). *)
let combine_family_profit a b =
  match (a, b) with
  | Unmeasured, x | x, Unmeasured -> x
  | Pays x, Pays y | Loses x, Loses y -> if Float.(x <= y) then a else b
  | (Pays _ as p), Loses _ | Loses _, (Pays _ as p) -> p

let flip_profit_margin_of_string raw =
  let raw = String.strip raw in
  let bad () =
    raise
    @@ Utils.User_error
         ("Autotune: ocannl_tune_flip_profit_margin should be a ratio of at least 1.0 (how much \
           worse than a search's best the best tensorized candidate may be and still be worth a \
           flip-budget slot); found: " ^ raw)
  in
  match Float.of_string raw with
  | m when Float.(is_finite m && m >= 1.) -> m
  | _ -> bad ()
  | exception _ -> bad ()

let flip_profit_margin () =
  flip_profit_margin_of_string
    (Utils.get_global_arg ~arg_name:"tune_flip_profit_margin" ~default:"1.25")

let flip_abandon_ratio ?margin () =
  let margin = match margin with Some m -> m | None -> flip_profit_margin () in
  margin *. margin

(** What one completed search measured about the tensorized family's profitability. A search that
    timed no tensorized candidate measured nothing about it — including one that seeded many and
    timed none, the gh-ocannl-521 state, which is a fact about candidate compilation rather than
    about the family's speed.

    "Was one timed" is [mma_best_ms] being finite, NOT [mma_timed > 0]: those are deliberately
    different populations (see where [mma_best_ms] is set). [mma_timed] counts candidates whose
    LABEL promised a tensorized pipeline, while a beam round appending a [Tensorize] to a saved or
    preset incumbent promises nothing in its label and is exactly as tensorized — and can win. A
    search whose only tensorized measurement came that way has measured the family, and keying this
    guard on the label would keep the prior standing against a family that lost tenfold. *)
let family_profit_of_report ?margin (r : report) =
  let margin = match margin with Some m -> m | None -> flip_profit_margin () in
  if
    r.timings_contended > 0
    || (not (Float.is_finite r.mma_best_ms))
    || (not (Float.is_finite r.best_ms))
    || Float.(r.best_ms <= 0.)
  then Unmeasured
  else
    let ratio = r.mma_best_ms /. r.best_ms in
    if Float.(ratio <= margin) then Pays ratio else Loses ratio

let family_profit_of_reports ?margin reports =
  List.fold reports ~init:Unmeasured ~f:(fun acc r ->
      combine_family_profit acc (family_profit_of_report ?margin r))

let family_profit_summary = function
  | Unmeasured -> "the tensorized family was never timed, so the enablement prior stands"
  | Pays r ->
      Printf.sprintf "the tensorized family measured %.2fx the best time, within the profit margin"
        r
  | Loses r -> Printf.sprintf "the tensorized family measured %.2fx the best time, out of profit" r

(** The ordering a ranking actually uses. [`Profitable] is the prior weighed against the measured
    evidence: with none, or with evidence that the family is competitive, it IS [`Enablement]; with
    measured evidence that the family loses here, both of the prior's classes are void at once — the
    promotion of family-unlocking flips and the demotion of family-breaking ones are the same bet on
    the same family — and the ranking degenerates to [`Cost]. *)
let effective_flip_ordering ~ordering ~profit =
  match (ordering, profit) with
  | `Cost, _ -> `Cost
  | `Enablement, _ -> `Enablement
  | `Profitable, (Unmeasured | Pays _) -> `Enablement
  | `Profitable, Loses _ -> `Cost

let flip_ordering () =
  match
    String.lowercase
      (String.strip (Utils.get_global_arg ~arg_name:"tune_flip_ordering" ~default:"profitable"))
  with
  | "cost" -> `Cost
  | "enablement" -> `Enablement
  | _ -> `Profitable

let rank_flip_candidates ~ordering ?(profit = Unmeasured) ~enablement ~disablement candidates =
  let deduped =
    List.fold candidates ~init:[] ~f:(fun acc (fc : LL.flip_candidate) ->
        (* Identity is [Tn.uid] ([Tn.equal]), not the session [id], which can repeat across
           namespaces and reinitializations. A candidate is the whole node (gh-ocannl-1017), so a
           repeated node is a repeated candidate. *)
        if List.exists acc ~f:(fun c -> Ir.Tnode.equal c.LL.fc_tn fc.LL.fc_tn) then acc
        else fc :: acc)
    |> List.rev
    (* gh-ocannl-1093: a flip the pricer's world shows refused replays to the materialized
       placement, so it is no decision to rank; a node left with no other alternative is no
       candidate. The dedup runs first, so a refused repeat cannot resurrect a node's first
       record. *)
    |> List.filter_map ~f:(fun (fc : LL.flip_candidate) ->
        match List.filter fc.LL.fc_alternatives ~f:(fun fa -> Option.is_none fa.LL.fa_refused) with
        | [] -> None
        | fc_alternatives -> Some { fc with LL.fc_alternatives })
  in
  let effective = effective_flip_ordering ~ordering ~profit in
  (* An alternative's class: [`Cost] has one; [`Enablement] has three — family-unlocking
     [`Materialize] flips first (their acceptance changes the feasible set, not just the objective),
     neutral flips in the middle, family-breaking [`Inline] flips last — inlining an operand or
     destination of an eligible site (whether reached by default placements or only by further
     materialization) can only move away from the tensorized family. *)
  let cls tn (fa : LL.flip_alternative) =
    match (effective, fa.LL.fa_flip) with
    | `Cost, _ -> 1
    | `Enablement, `Materialize when Set.mem enablement tn -> 0
    | `Enablement, (`Inline | `Footprint) when Set.mem enablement tn || Set.mem disablement tn ->
        (* A footprint-scoped node is virtual in the lineage like an inlined one: neither reading
           materializes the operand a tensorized site would stage. *)
        2
    | `Enablement, (`Materialize | `Inline | `Footprint) -> 1
  in
  (* Class ascending, cost descending within a class. *)
  let by_rank tn_a (a : LL.flip_alternative) tn_b (b : LL.flip_alternative) =
    match Int.compare (cls tn_a a) (cls tn_b b) with
    | 0 -> Int.compare b.LL.fa_recompute_cost a.LL.fa_recompute_cost
    | c -> c
  in
  (* A node's alternatives in rank order, and the node ranked where its best-ranked alternative is:
     the position its first record held when the readings were separate records. *)
  List.map deduped ~f:(fun (fc : LL.flip_candidate) ->
      {
        fc with
        LL.fc_alternatives =
          List.stable_sort fc.LL.fc_alternatives ~compare:(fun a b ->
              by_rank fc.LL.fc_tn a fc.LL.fc_tn b);
      })
  |> List.sort ~compare:(fun (a : LL.flip_candidate) (b : LL.flip_candidate) ->
      match
        by_rank a.LL.fc_tn
          (List.hd_exn a.LL.fc_alternatives)
          b.LL.fc_tn
          (List.hd_exn b.LL.fc_alternatives)
      with
      | 0 -> Ir.Tnode.compare a.LL.fc_tn b.LL.fc_tn
      | c -> c)

let offers_materialize (fc : LL.flip_candidate) =
  List.exists fc.LL.fc_alternatives ~f:(fun fa -> LL.equal_reading fa.LL.fa_flip `Materialize)

(* A node that is not materialized by default and offers no [`Materialize] alternative — the one an
   earlier routine left virtual and this consumer footprint-scoped (gh-ocannl-616), which no routine
   here writes — is one no materialize-all specialization reaches: that specialization keeps its
   prologue and the scratch's traffic, which its inline completion would not have. The floor is then
   not a lower bound over that node's completions, and no floor is better than a wrong one. Read off
   the candidate's default (gh-ocannl-1017), never off which alternatives exist: [`Inline] alone is
   also what a cap-materialized node with no smaller footprint offers. *)
let placement_floor_withheld candidates =
  List.exists candidates ~f:(fun (fc : LL.flip_candidate) ->
      (not (LL.equal_reading fc.LL.fc_default `Materialize)) && not (offers_materialize fc))

type placement_surface = {
  ps_candidates : LL.flip_candidate list;
  ps_ordering : [ `Cost | `Enablement ];
  ps_profit : family_profit option;
  ps_enablement : Set.M(Ir.Tnode).t;
  ps_disablement : Set.M(Ir.Tnode).t;
  ps_floor_ms : materialized:Ir.Tnode.t list -> float option;
  ps_twin_identity :
    node_tag:(Ir.Tnode.t -> string option) ->
    chain:(Ir.Tnode.t -> LL.reading option) ->
    LL.flip_candidate ->
    LL.flip_alternative ->
    string option;
}

(* gh-ocannl-1274: the flip chain's structural-twin identity of one alternative. The neighbourhood
   is every top-level statement of the raw program ([base.source], where a node the default policy
   inlines is still present) that sets or reads the flipped node: its producers and its consumers,
   backward ones included. Rendered by the schedule cache's canonical walk, so node, loop-binder and
   scope names are normalized by first occurrence and each node carries its dims, precision and
   constancy.

   What the identity claims is bounded, and stated here as a list rather than as "everything the
   flip's outcome depends on", which no local reading can be: per rendered node, the placement the
   lineage brought ([SC.lineage_tag]), the placement this specialization decided, the reading the
   chain has committed, the caller's [node_tag] (the flip chain's timing lineage; [None] refuses),
   observability (an
   observable buffer is never aliased), the surface's enablement classes, and the compile-time
   bounds candidate (the simplifier folds guards on it); then the flipped node's mark, its default
   and target readings, its recompute cost and whether the cost is modeled. No identity -- never a
   twin -- where the rendering cannot be the flip's whole local reading: an incomplete rendering (an
   opaque construct, a free symbol); a source holding an opaque construct anywhere (a
   [Staged_compilation] touches no node [code_touches_tn] can see, so the filter would drop it); a
   flipped node no statement here assigns; and a rendered node with a computation an earlier routine
   of the lineage stored ([inherited]), whose replayed prefix the source does not show; and a
   statement-level local read that no statement defines ([close]).
   Whole-routine context -- where the neighbourhood sits, hence live ranges -- is deliberately out:
   per-layer twins differ in exactly that. A false twin costs one unmeasured flip, never a wrong
   value. *)
(* The scope locals a fragment reads ([Get_local]) and defines ([Set_local], [Declare_local], a
   [Local_scope]'s own id, a scan's carried state), everything visited. *)
let scope_locals (code : LL.t) =
  let module F = LL.Access_fold in
  let policy =
    {
      F.discarded_operands = F.Visit;
      gated_operands = F.Visit;
      dead_loops = F.Visit;
      local_scopes = F.Visit;
      guards = F.Ignore;
      scan_implicit = F.Visit;
    }
  in
  let empty = Set.empty (module LL.Scope_id) in
  let hooks =
    {
      (F.hooks ()) with
      F.statement =
        (fun _ (reads, defs) (st : LL.t) ->
          match st with
          | LL.Set_local (id, _) | LL.Declare_local { id; _ } -> F.Continue (reads, Set.add defs id)
          | _ -> F.Continue (reads, defs));
      scalar =
        (fun _ (reads, defs) (sc : LL.scalar_t) ->
          match sc with
          | LL.Get_local id -> F.Continue (Set.add reads id, defs)
          | LL.Local_scope { id; _ } -> F.Continue (reads, Set.add defs id)
          | _ -> F.Continue (reads, defs));
    }
  in
  F.fold ~policy ~hooks ~init:(empty, empty) code

let flip_twin_identity ~static_indices ~lineage ~inherited ~enablement ~disablement
    (base : LL.optimized) =
  let stmts = LL.flat_lines [ base.LL.source ] in
  (* A neighbourhood closed over the statement-level locals it reads: a top-level statement that
     defines one (the [Declare_local] + [Set_local] a rewrite such as online softmax emits) joins
     it, in source order, until every local read is defined inside; [None] when one never is. The
     filter by node cannot see those statements, and an unbound local would render as complete. *)
  let rec close hood =
    let reads, defs = scope_locals (LL.unflat_lines hood) in
    let unbound = Set.diff reads defs in
    if Set.is_empty unbound then Some hood
    else
      let defines st =
        (not (List.mem hood st ~equal:phys_equal))
        && not (Set.is_empty (Set.inter unbound (snd (scope_locals st))))
      in
      if not (List.exists stmts ~f:defines) then None
      else close (List.filter stmts ~f:(fun st -> List.mem hood st ~equal:phys_equal || defines st))
  in
  let octx = base.LL.optimize_ctx in
  let decided = octx.LL.placements in
  (* The whole source's completeness, read once per surface on first use. *)
  let source_complete = ref None in
  let source_complete () =
    match !source_complete with
    | Some c -> c
    | None ->
        let c =
          SC.complete (SC.canonicalize_code ~static_indices ~node_tag:(fun _ -> "") base.LL.source)
        in
        source_complete := Some c;
        c
  in
  fun ~node_tag:caller_tag ~chain (fc : LL.flip_candidate) (fa : LL.flip_alternative) ->
    let tn = fc.LL.fc_tn in
    let hood = close (List.filter stmts ~f:(LL.code_touches_tn tn)) in
    let produced_here =
      Option.exists (Hashtbl.find base.LL.traced_store tn) ~f:(fun ta -> ta.LL.has_assignment)
    in
    (* The walk tags each node once, at its first occurrence: every rendered node passes here. *)
    let any_inherited = ref false in
    let node_tag n =
      if inherited n then any_inherited := true;
      SC.lineage_tag lineage octx n
      ^ (match caller_tag n with
        | Some tag -> tag
        | None ->
            any_inherited := true;
            "")
      ^ (if Ir.Tnode.is_observable n then ";obs" else "")
      ^ (if Set.mem enablement n then ";en" else "")
      ^ (if Set.mem disablement n then ";dis" else "")
      ^ (match Ir.Tnode.bounds_candidate n with
        | None -> ""
        | Some iv -> ";b=" ^ Sexp.to_string (Ir.Interval.sexp_of_t iv))
      ^ (match Ir.Tnode.Placements.get decided n with
        | None -> ";d=u"
        | Some (m, _) -> ";d=" ^ Sexp.to_string (Ir.Tnode.sexp_of_memory_mode m))
      ^ (match chain n with None -> "" | Some r -> ";chain=" ^ LL.reading_to_string r)
      ^ if Ir.Tnode.equal n tn then ";flip" else ""
    in
    Option.bind hood ~f:(fun hood ->
        let c = SC.canonicalize_code ~static_indices ~node_tag (LL.unflat_lines hood) in
        Option.some_if
          (produced_here && (not !any_inherited) && SC.complete c && source_complete ())
          (Printf.sprintf "%s;%s->%s;cost=%d;modeled=%b" (SC.digest c)
             (LL.reading_to_string fc.LL.fc_default)
             (LL.reading_to_string fa.LL.fa_flip)
             fa.LL.fa_recompute_cost fa.LL.fa_modeled))

let placement_surface ?name ?ordering ?(evidence = []) ctx comp bindings =
  let ordering = match ordering with Some o -> o | None -> flip_ordering () in
  (* The evidence is derived only on the path that consults it, so an unconditional ordering does
     not depend on [tune_flip_profit_margin] at all — it is not merely ignored: a malformed or
     out-of-range margin must not abort a run pinned to a baseline the term plays no part in. *)
  let profit =
    match ordering with
    | `Profitable -> Some (family_profit_of_reports evidence)
    | `Cost | `Enablement -> None
  in
  let limits = Context.hardware_limits ctx in
  let static_indices = Idx.bound_symbols bindings in
  let base = Context.lowered_for_decisions ?name ctx comp bindings in
  let candidates = base.LL.flip_candidates in
  (* The all-materialized specialization of the decision surface: the [`Materialize] flips are the
     default-virtual candidates, so deciding exactly those materialized makes every open node's work
     sit in its own producer statement — the form [completion_floor]'s [open_placement] contract
     asks for. *)
  let to_materialize =
    List.filter_map candidates ~f:(fun fc -> Option.some_if (offers_materialize fc) fc.LL.fc_tn)
  in
  let allmat = Context.lowered_for_decisions ?name ~materialized:to_materialize ctx comp bindings in
  let enablement, disablement =
    placement_enablement
      ~is_gpu:(Sched.backend_is_gpu (Context.backend_name ctx))
      ~limits ~static_indices ~base ~allmat
  in
  let ps_candidates = rank_flip_candidates ~ordering ?profit ~enablement ~disablement candidates in
  let candidate_set =
    Set.of_list (module Ir.Tnode) (List.map ps_candidates ~f:(fun fc -> fc.LL.fc_tn))
  in
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let floor_withheld = placement_floor_withheld candidates in
  let ps_floor_ms ~materialized =
    if floor_withheld then None
    else
      let mat = Set.of_list (module Ir.Tnode) materialized in
      let open_placement tn = Set.mem candidate_set tn && not (Set.mem mat tn) in
      let f = CM.completion_floor ~open_placement allmat.LL.llc in
      CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:f.CM.fr_flops
        ~bytes:f.CM.fr_bytes ()
      |> Option.map ~f:(fun s -> s *. 1e3)
  in
  {
    ps_candidates;
    ps_ordering =
      effective_flip_ordering ~ordering ~profit:(Option.value profit ~default:Unmeasured);
    ps_profit = profit;
    ps_enablement = enablement;
    ps_disablement = disablement;
    ps_floor_ms;
    ps_twin_identity =
      flip_twin_identity ~static_indices ~lineage:(Context.placements ctx)
        ~inherited:(Context.inherits_computation ctx)
        ~enablement ~disablement base;
  }

(** {2 Model-picked untuned defaults (gh-ocannl-491 task 3)}

    A drop-in for [Context.compile] that raises the untuned floor: with no measurement at all, the
    default pipeline and the sketch families are scored with the roofline model inside the compile's
    own transform seam, and the model-argmin schedule is applied. Advisory by construction — a
    candidate without model coverage is never picked over the default, ties go to the default, and
    any scoring or application failure falls back to the ordinary default pipeline. *)

type model_choice = {
  mc_label : string;
      (** ["default"] or the winning candidate's spec label (matching {!tune}'s [autotune_log]
          labels). *)
  mc_model_ms : float option;
      (** The winner's roofline lower bound in ms — not a runtime prediction; [None] when selection
          did not run (no envelope constants, automatic scheduling disabled, or the default itself
          had no model coverage). *)
  mc_rank_ms : float option;
      (** The winner's ranking score in ms ({!Ir.Cost_model.ranking_seconds}, gh-ocannl-1253), the
          quantity the pick minimized; at least [mc_model_ms]. [None] exactly when [mc_model_ms] is.
      *)
  mc_scored : int;
      (** Model evaluations that produced a score (the default pipeline included; the fissioned flow
          also scores per segment). *)
  mc_skipped : int;  (** Model evaluations without coverage, excluded from ranking. *)
  mc_rejected : int;
      (** Candidates excluded from the ranking because their scheduled form fails
          {!Ir.Low_level.validate_parallel} — it could not have compiled (gh-ocannl-522). *)
  mc_excluded : int;
      (** Sketch candidates kept out of the ranking on a GPU backend as not numerics-preserving
          (gh-ocannl-1253). *)
}

let model_default_enabled =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"model_default_schedule")

(* gh-ocannl-514 phase 5: whether [model_default]'s family search lifts the tile-size lattice
   exclusions ([lift_geometry_lattice]) — the full dividing lattice searched under non-uniform
   bounds instead of the curated menus alone. Never affects [tune]'s seed lists. *)
let geometry_lattice_enabled =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"model_default_geometry_lattice")

let compile_advisory ?name ?on_fallback ?fallback_if ?on_unrecovered lowered_transform ctx comp
    bindings =
  match
    Context.compile_outcome ?name ~lowered_transform ~provenance:Outcome.Advisory ctx comp bindings
  with
  | Ok result -> result
  | Error (Outcome.Fatal _ as failure) -> Outcome.raise_failure failure
  | Error (Outcome.Classified classified as failure)
    when not (Option.value_map fallback_if ~default:true ~f:(fun f -> f ())) -> (
      (* [fallback_if] is what keeps the retry from duplicating a genuine failure: a transform that
         already degraded to the default pipeline has nothing to fall back TO, so recompiling would
         just repeat the same failing compile (and, on a resource failure, aggravate it) before
         raising the same exception. Such callers say [false] here and the original exception
         propagates through the public exception contract -- or, given [on_unrecovered], the
         classified cause goes to the caller, which may know a fallback of its own (gh-ocannl-1390:
         [model_default]'s placement pick). A fatal failure never reaches it. *)
      match on_unrecovered with
      | Some f -> f classified
      | None -> Outcome.raise_failure failure)
  | Error (Outcome.Classified classified) -> (
      (* Typed compiler rejection is the advisory fallback boundary. Fatal failures are propagated
         above without paying for a second compile. *)
      Option.iter on_fallback ~f:(fun f -> f (Outcome.exception_of_cause classified.cause));
      (* [Context.compile], spelled out so that a classified failure of the fallback itself also
         reaches [on_unrecovered]: there is then nothing left to fall back to here either. *)
      match Context.compile_outcome ?name ~provenance:Outcome.User_schedule ctx comp bindings with
      | Ok result -> result
      | Error (Outcome.Classified classified) when Option.is_some on_unrecovered ->
          (Option.value_exn on_unrecovered) classified
      | Error failure -> Outcome.raise_failure failure)

(* gh-ocannl-1254: a candidate's tile-MMA operand loads under the backend's own description of its
   arms: their load rule and fragment structure. *)
let model_operand_loads ~(capabilities : Ir.Backend_intf.codegen_capabilities) (opt : LL.optimized)
    =
  CM.operand_load_count ~operand_accepts:capabilities.mma_accepts
    ~operand_fragments:capabilities.mma_operand_fragments
    ~operand_loads:capabilities.mma_operand_loads opt

(* The branch-and-bound walk over the factored matmul family (gh-ocannl-514 phase 4), shared by
   [model_default] and the tests that drive it with stand-in limits and load rules: verdict-carrying
   children are the construction-time fathoms, and the bound is the schedule-invariant roofline
   floor — sketch completions share the base program's semantics, so [completion_floor] lower-bounds
   every one; it fathoms the whole family exactly when the incumbent already achieves it (the
   memory-bound kernels where the default preset is optimal) — raised per subtree by the committed
   staging decisions' certain traffic (phase 5, [sketch_path_traffic_floor]). The epilogue twins are
   the claiming family's root level (gh-ocannl-613), below the combined view's [family] root whose
   other child is refuted (gh-ocannl-1488), so they compete inside the walk, after the unfused
   leaves and at the threshold those tightened to, with the same bound and on the same stats ledger.
   Leaves whose scores tie exactly are ordered by their tile-MMA operand loads under [operand_loads]
   (gh-ocannl-1254, {!model_operand_loads}): fewer load instructions win, so a staged twin that
   brings its operands in through [ldmatrix] beats its gathering twin of equal flops and bytes; a
   leaf with no tile-MMA site has no count and stays out of the comparison (its operands load by
   paths the count does not see), and the incumbent still wins every tie. The walk covers whichever
   family claims the lowering ({!sketch_family_tree}): the conv family is a tree under the same
   fusion root since gh-ocannl-697, and since gh-ocannl-1489 its committed row geometries raise the
   bound by their two staged operand tiles' certain traffic, so conv geometry subtrees fathom as
   matmul ones do. [None] = neither family has a site. *)
let model_family_search ?(on_rejected = fun () -> ()) ?(admit = fun _ -> true)
    ?(on_excluded = fun () -> ()) ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices ~peak_flops
    ~peak_memory_bandwidth ~score ~incumbent (base_opt : LL.optimized) =
  let accum_prec = capabilities.Ir.Backend_intf.accum_prec in
  match sketch_family_tree ~capabilities ~is_gpu ~is_cpu ~limits base_opt with
  | None -> None
  | Some tree ->
      (* gh-ocannl-514 phase 5: the tile-size lattice beyond the curated menus enters the searched
         space when lifted (config [model_default_geometry_lattice]), and the bound is no longer
         uniform across the family — each subtree's committed staging decisions contribute their
         certain traffic ([sketch_path_traffic_floor]) on top of the schedule-invariant floor, so
         whole boxes of the lattice fathom without expansion. *)
      let tree = if Lazy.force geometry_lattice_enabled then lift_geometry_lattice tree else tree in
      let f = CM.completion_floor base_opt.LL.llc in
      let path_inc = sketch_path_traffic_floor ~limits base_opt in
      let bound_at inc =
        CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:f.CM.fr_flops
          ~bytes:(f.CM.fr_bytes + inc) ()
      in
      let applied p =
        match
          Sched.apply_classified ~static_indices
            (sketch_schedule ~accum_prec ~p base_opt)
            (scratch_of base_opt)
        with
        | exception Outcome.Cause_at _ -> None
        | post -> Some post
      in
      let score_sketch p =
        if not (admit p) then (
          on_excluded ();
          None)
        else
          match applied p with
          | None ->
              on_rejected ();
              None
          | Some post -> score [ post ]
      in
      (* Computed only for leaves that tie; a leaf that cannot be applied scored nothing, so never
         reaches a tie. *)
      let tie_break p =
        match applied p with
        | None -> None
        | Some post -> (
            match model_operand_loads ~capabilities post with
            | CM.Loads n -> Some n
            | CM.No_tile_mma | CM.Undescribed -> None)
      in
      let best, stats =
        Sspace.search
          ~bound:(fun ~path _sub -> bound_at (path_inc path))
          ~incumbent ~tie_break ~score:score_sketch tree
      in
      Some (best, stats, bound_at 0)

(** {2 The seeded untuned default (gh-ocannl-1324)} *)

type seeded_choice = {
  sd_seeded : (int * sketch_params) list;
  sd_sunk : (int * int) list;
  sd_refined : int;
  sd_split : (Ir.Tnode.t * int) list;
  sd_fell_back : bool;
}

let site_reduction_extent (site : matmul_site) =
  List.fold site.m_ko ~init:site.m_nk ~f:(fun acc (_, n) -> acc * n)

(* CUDA and HIP only: the backends the untiled-contraction gap was measured on (gh-ocannl-1183) and
   the seeded plan's before/after was measured on. Metal keeps the plain default until a Metal
   measurement says otherwise. *)
let seeded_default_backend backend = Sched.backend_seeds_untuned_default backend

type seed_gate =
  | Min_reduction of int
  | Ranked of {
      capacity : int option;
      cache_rate : float option;
      scalar_rate : float option;
      dram_rate : float option;
      matrix_rate : float option;
      gather_rate : float option;
      fill : Ir.Cost_model.fill_model option;
    }

(* The threshold [auto] falls back to where the device publishes no capacity (gh-ocannl-1253):
   gh-ocannl-1324's measured value. *)
let seed_gate_fallback = 32

(* The gate rendered with every reading its decisions rest on: the staffed start's search shape
   ([seeded_start.gate_tag]) and the untuned compile's log line (gh-ocannl-1641: a survey of the
   seeding's decisions can then say which gate, and which launch fill, decided them). *)
let seed_gate_tag = function
  | Min_reduction n -> Printf.sprintf "min_reduction=%d" n
  | Ranked { capacity; cache_rate; scalar_rate; dram_rate; matrix_rate; gather_rate; fill } ->
      let rate = Option.value_map ~default:"none" ~f:Float.to_string in
      (* gh-ocannl-1495: the tile-MMA and device-gather rates only where set, so a gate without them
         keeps its tag. *)
      let set key =
        Option.value_map ~default:"" ~f:(fun r -> Printf.sprintf " %s=%s" key (rate (Some r)))
      in
      Printf.sprintf "ranked capacity=%s cache_rate=%s scalar_rate=%s dram_rate=%s%s%s%s"
        (Option.value_map capacity ~default:"none" ~f:Int.to_string)
        (rate cache_rate) (rate scalar_rate) (rate dram_rate) (set "matrix_rate" matrix_rate)
        (set "gather_rate" gather_rate)
        (Option.value_map fill ~default:""
           ~f:(fun { Ir.Cost_model.fm_multiprocessors; fm_fill_threads } ->
             Printf.sprintf " fill=%dx%d" fm_multiprocessors fm_fill_threads))

(* The configured gate, unresolved: [`Auto] or a threshold. *)
let seed_gate_setting () =
  Utils.registered_keyword ~arg_name:"gpu_default_sketch_min_reduction"
    ~expected:"auto or a non-negative integer (0 disables the seeded default)" ~parse:(function
    | "auto" -> Some `Auto
    | raw -> (
        match Int.of_string_opt raw with Some n when n >= 0 -> Some (`Threshold n) | _ -> None))

let seeded_default_gate ~limits =
  match seed_gate_setting () with
  | `Threshold n -> Min_reduction n
  | `Auto -> (
      (* The fallback is for a device nothing has calibrated -- no advisory and no override. An
         explicit [none] override is a calibration too (no DRAM tier, or uncharged re-reads): its
         ranking stands, even when both tiers resolve to [None]. *)
      let stated = function `Advisory -> false | `Off | `Value _ -> true in
      let explicit =
        stated (Lazy.force cache_capacity_override) || stated (Lazy.force cache_bandwidth_override)
      in
      match (fst (ranking_capacity ~limits), ranking_cache_bandwidth ~limits) with
      | None, None when not explicit -> Min_reduction seed_gate_fallback
      | capacity, cache_rate ->
          Ranked
            {
              capacity;
              cache_rate;
              scalar_rate = ranking_issue_rate ~limits;
              dram_rate = ranking_dram_bandwidth ~limits;
              matrix_rate = ranking_mma_rate ~limits;
              gather_rate = ranking_device_gather_rate ~limits;
              fill = ranking_fill_model ~limits;
            })

(* The seeded geometries in preference order: the largest block tile first, for its operand reuse,
   then the 4x4 register tile at a smaller block, then the 2x2 register tiles, which put four times
   as many threads on the same outputs. A geometry is taken once its launch has [seeded_min_blocks]
   blocks — about two per multiprocessor on the RTX 5070 Ti (70 SMs), more per compute unit on the
   HIP parts — so a small output does not leave most of the device idle behind a few large tiles. *)
(* gh-ocannl-1348: whether the untuned compile sinks the reduction loops of the segments it does not
   seed below their write loops ({!reduction_sink}). *)
let reduction_sink_enabled () = Utils.registered_flag ~arg_name:"gpu_default_reduction_sink"

(* gh-ocannl-1405: whether the untuned plan also takes the sites whose accumulator precision widens
   their storage precision (a bf16 weight gradient on CUDA, fp8 on CUDA and HIP): seeded, sunk and
   admitted to the model's numerics-preserving set. Seeding one keeps its result: the sketch's
   [Privatize] mints its tile wide only where the default's serial rendering widens too
   (gh-ocannl-1151), so a backprop weight gradient, which the default narrows after every batch row,
   keeps a storage-precision tile and the same narrowing points (measured bitwise on CUDA,
   test/operations/seeded_narrow_storage). Sinking one does not: the sunk reduction's accumulator is
   localized below its write loops, where code generation widens it, so it rounds to storage once
   instead of after every row -- an accuracy gain that changes the result, hence a numerics gate:
   off by default, on in the [approximate] profile, pinned off in [reproducible]. *)
let seed_narrow_storage_enabled () =
  Utils.get_global_flag ~default:false ~arg_name:"gpu_default_seed_narrow_storage"

let seeded_geometries =
  [ (64, 64, 8, 4, 4); (32, 32, 8, 4, 4); (32, 32, 16, 2, 2); (16, 16, 8, 2, 2); (16, 16, 8, 4, 4) ]

let seeded_min_blocks = 128

(* The numerics-preserving candidate set (gh-ocannl-1324, gh-ocannl-1253): the unfused scalar
   blocktile pipeline with no twin flavor. Tensorized forms change numerics and the conv family
   reorders the reduction. The coalesced layouts do NOT: each keeps every cell's summation order
   (test/operations/coalesce_numerics executes it), and they stay out, as the twins do, because they
   are the search's to time -- the ranking score cannot tell a coalesced leaf from its uncoalesced
   twin (gh-ocannl-1278). Shared by the seeding and by [model_default]'s GPU family search, so the
   two cannot drift. *)
let scalar_blocktile p =
  p.sk_gpu && (not p.sk_mma) && (not p.sk_conv) && (not p.sk_epilogue)
  && Option.is_none p.sk_coalesce && (not p.sk_batch_inner) && Option.is_none p.sk_swizzle
  && p.sk_depth = 1

(* Whether a site's output keeps its accumulator at its storage precision under [accum_prec]: a
   sketch keeps its accumulator at [accum_prec] across the whole reduction, where the default
   pipeline stores a wider accumulator back at the output's precision after every contraction step
   whose loop encloses the write loops (a backprop weight gradient) -- such a sketch would round
   less often than the default. *)
let accumulates_at_storage ~accum_prec (site : matmul_site) =
  let d_prec = Lazy.force site.m_d.Ir.Tnode.storage_prec in
  Ir.Ops.equal_prec (accum_prec d_prec) d_prec

(* [Error] names the refusing gate, for the seeding census (gh-ocannl-1407). [capabilities] are the
   compiling context's, so the family's workgroup-memory refutation charges the tiles as the gate
   that judges the seeded kernel does ([Sched.kernel_workgroup_bytes], gh-ocannl-1330). *)
let seeded_params_judged ?capabilities ?(admit_narrow = false) ~limits ~accum_prec ~min_reduction
    (seg : LL.optimized) : (sketch_params, string) Result.t =
  match detect_matmul_canonical seg with
  | None -> Error "no matmul site"
  | Some (site, _, _) when site_reduction_extent site < min_reduction ->
      Error
        (Printf.sprintf "site %s: reduction extent %d is under the seeding threshold %d"
           (Ir.Tnode.debug_name site.m_d) (site_reduction_extent site) min_reduction)
  | Some (site, _, _) when not (admit_narrow || accumulates_at_storage ~accum_prec site) ->
      (* Left to the default unless the narrow-storage admission (gh-ocannl-1405, the [approximate]
         profile's) takes it, which {!seed_narrow_storage_enabled} explains. *)
      Error
        (Printf.sprintf
           "site %s: the accumulator precision widens its %s storage, so a sketch would change the \
            result"
           (Ir.Tnode.debug_name site.m_d)
           (Ir.Ops.prec_string (Lazy.force site.m_d.Ir.Tnode.storage_prec)))
  | Some (site, _, _) -> (
      match matmul_sketch_tree ?capabilities ~is_gpu:true ~is_cpu:false ~limits seg with
      | None -> Error "no matmul site"
      | Some tree -> (
          (* The tree's own leaves, so every refutation the search applies (launch caps,
             workgroup-memory and thread floors, companion coverage) applies here too; of those, the
             unfused scalar blocktile pipeline with no twin flavor: tensorized forms change
             numerics, and the twins are the search's to time. *)
          let leaves = List.filter (Sspace.leaves tree) ~f:scalar_blocktile in
          let batch = List.map (site.m_bo @ site.m_bi) ~f:snd in
          (* A batched site's batch loops launch as grid blocks where the device's [.z] cap admits
             them (the tree refutes the flavor otherwise); serial inside each block otherwise. *)
          let leaves =
            match List.filter leaves ~f:(fun p -> p.sk_batch_grid) with
            | [] -> leaves
            | grid when not (List.is_empty batch) -> grid
            | _ -> leaves
          in
          let blocks p =
            (site.m_ni + p.sk_bm - 1)
            / p.sk_bm
            * ((site.m_nj + p.sk_bn - 1) / p.sk_bn)
            * if p.sk_batch_grid then List.fold batch ~init:1 ~f:( * ) else 1
          in
          let threads p = blocks p * (p.sk_bm / p.sk_tm) * (p.sk_bn / p.sk_tn) in
          let ranked =
            List.filter_map seeded_geometries ~f:(fun (bm, bn, bk, tm, tn) ->
                List.find leaves ~f:(fun p ->
                    p.sk_bm = bm && p.sk_bn = bn && p.sk_bk = bk && p.sk_tm = tm && p.sk_tn = tn))
          in
          (* A block tile wider than the output pads it with cells every thread computes and
             discards — attention's value product (a 32-column output) under a 64-column tile wastes
             half its threads. Geometries padding the output by at most an eighth go first. *)
          let padded p =
            8
            * ((site.m_ni + p.sk_bm - 1) / p.sk_bm * p.sk_bm)
            * ((site.m_nj + p.sk_bn - 1) / p.sk_bn * p.sk_bn)
            > 9 * site.m_ni * site.m_nj
          in
          let ranked =
            match List.filter ranked ~f:(fun p -> not (padded p)) with [] -> ranked | snug -> snug
          in
          match List.find ranked ~f:(fun p -> blocks p >= seeded_min_blocks) with
          | Some p -> Ok p
          | None -> (
              (* Nothing reaches the block count: the geometry putting the most threads on the
                 output, the earlier one on a tie. *)
              match
                List.fold ranked ~init:None ~f:(fun best p ->
                    match best with Some b when threads b >= threads p -> best | _ -> Some p)
              with
              | Some p -> Ok p
              | None ->
                  Error
                    (Printf.sprintf
                       "site %s: no seeded geometry is a scalar blocktile leaf of its family tree \
                        (%d such leaves, %d refutations)"
                       (Ir.Tnode.debug_name site.m_d) (List.length leaves)
                       (List.length (Sspace.refutations tree))))))

let seeded_params ?capabilities ?admit_narrow ~limits ~accum_prec ~min_reduction seg =
  Result.ok
    (seeded_params_judged ?capabilities ?admit_narrow ~limits ~accum_prec ~min_reduction seg)

(* The untuned split runs wherever the seeded default does (gh-ocannl-1325, gh-ocannl-1412). It was
   CUDA's alone until its first pass ran the block loop outside the cell loops ([sr_block_hoist]):
   with the lanes walking the blocks, the HIP b256 step went from 885 to 1256 ms on gfx1151's
   unified memory where CUDA's caches absorbed the scattered loads. *)
let seeded_split_backend backend = seeded_default_backend backend

(* gh-ocannl-1331: whether a search on [backend] can read the [Search_shaping] key [key]: the seeded
   plan's keys where {!seeded_settings} resolves them (its backend tests come first, so another
   backend never reads them), the register-tile twin gate where the matmul family's CPU branch reads
   it ([matmul_flavor_tree]'s [is_cpu] arm, the register-tiled rendering), the default pipeline's
   per {!Sched.search_shaping_key_read}. The other [autotune_*] keys are read on every backend: the
   search's own knobs, and [autotune_pad_waste_max] in the GPU and CPU menus alike. *)
let search_shaping_key_read ~backend key =
  match key with
  | "autotune_register_tile_rm_twin" -> Sched.backend_is_cpu backend
  | "gpu_default_sketch_min_reduction" | "gpu_default_reduction_sink" ->
      seeded_default_backend backend
  | "gpu_default_split_reduce_min_serial" ->
      seeded_default_backend backend && seeded_split_backend backend
  | _ -> Sched.search_shaping_key_read ~backend_name:backend key

let search_shaping_identity ~backend ~limits =
  Utils.config_class_fingerprint
    ~reads:(search_shaping_key_read ~backend)
    ~resolve:(Sched.resolve_search_shaping_value limits)
    Utils.Search_shaping

let seeded_split_min_serial () =
  match Utils.registered_int ~arg_name:"gpu_default_split_reduce_min_serial" with
  | n when n >= 0 -> n
  | n ->
      raise
        (Utils.Invalid_config_value
           {
             key = "gpu_default_split_reduce_min_serial";
             value = Int.to_string n;
             expected = "a non-negative integer (0 disables the untuned split)";
           })

(* The first pass's thread count a split aims for: every output cell gets [num_blocks] partial sums
   computed in parallel. About two waves of the RTX 5070 Ti's 70 multiprocessors at 256 threads
   each; deliberately a constant rather than a device query, so that which split the plan takes --
   hence the summation order, hence the result -- depends on the routine alone. *)
let seeded_split_threads = 32768

(* The untuned plan's split reductions (gh-ocannl-1325): the {!split_reduce_sites} -- so the split's
   own legality and the gh-537 enabling interchange are the search's -- whose serial chain per
   output cell (the accumulation's trip count over its cell count) is at least [min_serial], each at
   the block count of the search's menu ([32; 128; 512], at most half the split loop) that comes
   closest to {!seeded_split_threads} first-pass threads without exceeding it, else the smallest. A
   scatter site ([Set_dynamic]) is left to the search.

   The block count also stays within the square root of the chain (gh-ocannl-1414): the combine
   reads one partial per block, so beyond it the combine's serial chain outgrows the first pass's. A
   1024-row chain into one cell would otherwise take 512 blocks of 2 rows each; it takes 32 of 32.
   The cap binds the fallback too: a chain shorter than 32 * 32, reachable only below the default
   threshold, is not split. Every split of gh-ocannl-1325's b256 measurement chains 32768 rows and
   keeps its count. *)
let seeded_split_sites ~backend ~min_serial ~static_indices (opt : LL.optimized) :
    (sr_site * int) list =
  if min_serial <= 0 || not (seeded_split_backend backend) then []
  else
    List.filter_map (split_reduce_sites ~static_indices opt) ~f:(fun s ->
        let chain = s.sr_cost / s.sr_out in
        if s.sr_dynamic || chain < min_serial then None
        else
          let fitting =
            List.filter [ 512; 128; 32 ] ~f:(fun b -> 2 * b <= s.sr_red && b * b <= chain)
          in
          match List.find fitting ~f:(fun b -> s.sr_out * b <= seeded_split_threads) with
          | Some b -> Some (s, b)
          | None -> Option.map (List.last fitting) ~f:(fun b -> (s, b)))

(* The site refinement of a segmentation (gh-ocannl-574's [arity_cuts] pieces, gh-ocannl-1413):
   [coarse] and [fine] are two segmentations of one routine as [(first unit, kind, units,
   pre-schedule slice)] per segment, and [seed kind pre] decides a segment's seed. They are compared
   block by block, a block being a maximal run of units whose two ends are segment boundaries in
   both, and a block takes its fine segments when more of them seed than of its coarse ones; every
   other block keeps its coarse segments, so no launch is added that buys no seed.

   A kernel has one launch geometry, so a sketch tiles one product per segment and covers the others
   only untiled ([companion_geometry]), and a companion that cannot follow the site's full arity
   refutes every seed of the segment. Fission keeps a weight gradient with the bias gradient
   reducing the same rows, and a projection's weight gradient with its input's data gradients
   ([w_k.grad] with the two [x.grad] products and [w_q.grad] in a transformer layer): the fine
   segments give each its own kernel. A block, not a single coarse segment, because the finer cuts
   can also move a boundary: attention's score product shares its coarse segment with the row
   maximum's initialization, which the fine segmentation keeps with the maximum instead -- the
   segments [(qk, init) (max)] become [(qk) (init, max)], and no coarse segment is partitioned
   exactly. Returns the refined segmentation, each segment with its decided seed, and the number of
   coarse segments replaced. *)
let refine_segments ~coarse ~(fine : _ list option Lazy.t) ~seed =
  let decide (at, kind, n, pre) = (at, kind, n, seed kind pre, pre) in
  let seeds segs = List.count segs ~f:(fun (_, _, _, s, _) -> Option.is_some s) in
  let end_of (at, _, n, _) = at + n in
  (* The blocks, as [(coarse segments, fine segments)] in order; [None] when the two do not cover
     the same units. *)
  let rec blocks acc cb fb coarse fine =
    match (coarse, fine) with
    | [], [] when List.is_empty cb && List.is_empty fb -> Some (List.rev acc)
    | c :: cs, f :: fs ->
        if end_of c = end_of f then
          blocks ((List.rev (c :: cb), List.rev (f :: fb)) :: acc) [] [] cs fs
        else if end_of c < end_of f then blocks acc (c :: cb) fb cs fine
        else blocks acc cb (f :: fb) coarse fs
    | _ -> None
  in
  match Option.bind (Lazy.force fine) ~f:(blocks [] [] [] coarse) with
  | None -> (List.map coarse ~f:decide, 0)
  | Some blocks ->
      let segs, refined =
        List.fold blocks ~init:([], 0) ~f:(fun (acc, refined) (cb, fb) ->
            let whole = List.map cb ~f:decide in
            match fb with
            | [ _ ] when List.length cb = 1 -> (List.rev_append whole acc, refined)
            | _ ->
                let pieces = List.map fb ~f:decide in
                if seeds pieces > seeds whole then
                  (List.rev_append pieces acc, refined + List.length cb)
                else (List.rev_append whole acc, refined))
      in
      (List.rev segs, refined)

let no_seeded_choice =
  { sd_seeded = []; sd_sunk = []; sd_refined = 0; sd_split = []; sd_fell_back = false }

(* The resolved thresholds, or [None] where the untuned compile is the plain default. The backend
   first: the thresholds are CUDA/HIP settings, so a malformed value must not break a compile on a
   backend that ignores them. *)
let seeded_settings ?min_reduction ?seed_gate ?min_serial ?reduction_sink:sink_override ?seed_narrow
    ~backend ~configured_gate () =
  let gate () =
    match (seed_gate, min_reduction) with
    | Some _, Some _ ->
        invalid_arg "Autotune.seeded_transform: pass ~min_reduction or ~seed_gate, not both"
    | Some g, None -> g
    | None, Some n -> Min_reduction n
    | None, None -> configured_gate ()
  in
  let gate_off () = match gate () with Min_reduction n -> n <= 0 | Ranked _ -> false in
  (* Resolved only where the split runs, so a malformed value breaks no compile that ignores it. *)
  let min_serial () =
    if not (seeded_split_backend backend) then 0
    else match min_serial with Some n -> n | None -> seeded_split_min_serial ()
  in
  let sink_enabled () =
    match sink_override with Some b -> b | None -> reduction_sink_enabled ()
  in
  let admit_narrow () =
    match seed_narrow with Some b -> b | None -> seed_narrow_storage_enabled ()
  in
  if
    (not (seeded_default_backend backend))
    || (gate_off () && min_serial () <= 0 && not (sink_enabled ()))
    || (not (Sched.automatic_schedule_active ~backend_name:backend))
    || not (Sched.default_pipeline_fissions ())
  then None
  else Some (gate (), min_serial (), sink_enabled (), admit_narrow ())

(* The split reductions first, whole-routine and before any segmentation, as the search's [F_split]
   seeds apply them: the partials edge each split mints is what fission cuts the two passes apart
   at, and every later step -- the default segments, the seeding -- works on the split routine. A
   copy of the traced store only, as the search's [F_split] candidates rebase: the placement table
   is the caller's, which the partials nodes the prelude mints must be registered in for the routine
   to link. A prelude that fails leaves at most placements for nodes no routine mentions, and the
   routine unsplit. *)
let seeded_split ~backend ~min_serial ~static_indices (opt : LL.optimized) =
  match seeded_split_sites ~backend ~min_serial ~static_indices opt with
  | [] -> (opt, [])
  | sites -> (
      match
        (* Blind to node bounds, as the search's prelude is (gh-ocannl-1480): the plan's seeded
           digests are taken on these segments, and an [F_seeded] candidate replays the prelude. *)
        apply_before_fission ~static_indices (split_reduce_prelude sites)
          { opt with LL.traced_store = Hashtbl.copy opt.LL.traced_store }
      with
      | split_opt ->
          logf "seeded_default: %d reduction(s) split: %s" (List.length sites)
            (String.concat ~sep:", " (List.map sites ~f:split_site_label));
          (split_opt, sites)
      | exception Outcome.Cause_at (_, cause) ->
          logf "seeded_default: the split reductions FAILED to apply (%s); left unsplit"
            (Outcome.detail_of_cause cause);
          (opt, []))

(* gh-ocannl-1253, the [auto] gate's test: the seeded sketch's ranking score strictly below the
   preset's, on the segment alone, under the gate's capacity and cache bandwidth. Either failing to
   build, to apply or to score keeps the preset. Only a typed refusal fails: any other exception is
   a builder or transform bug and propagates (gh-ocannl-1390). *)
let seeded_ranker ~capabilities ~limits ~accum_prec ~static_indices ~capacity ~cache_bandwidth
    ~issue_rate ~dram_bandwidth ~mma_rate ~device_gather_rate ~fill_model p pre =
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let rank sched =
    match Sched.apply_classified ~static_indices (sched ()) (scratch_of pre) with
    | exception Outcome.Cause_at _ -> None
    | post ->
        Option.map
          (segments_ranking ~capabilities ~capacity ~cache_bandwidth ~issue_rate ~dram_bandwidth
             ~mma_rate ~device_gather_rate ~fill_model ~peak_flops ~peak_memory_bandwidth ~limits
             [ post ])
          ~f:fst
  in
  match
    (rank (fun () -> sketch_schedule ~accum_prec ~p pre), rank (fun () -> seeded_preset ~limits pre))
  with
  | Some sketch, Some plain -> Float.(sketch < plain)
  | _ -> false

(* What the seeded plan does with each segment of the (split) routine: the segmentation and one
   decision per segment, plus how many default segments were refined, or [None] where the plan is
   the default segments (nothing seeds or sinks, or the segmentation is refused). Hermetic: every
   probe runs on a copy. *)
(* The seeded plan's decision for one segment: the sketch its gate admits there, probed and validated
   on the segment alone, or [None] where the segment keeps its preset. Also the staffed search's
   start (gh-ocannl-1138), which applies it to each key's own segment. The judgment's [Error]
   names the refusing gate, for the seeding census (gh-ocannl-1407). *)
let seeded_segment_judgment ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
    ~ranked_below (pre : LL.optimized) : (sketch_params, string) Result.t =
  let candidate =
    match gate with
    | Min_reduction n when n <= 0 ->
        Error "sketch seeding is off (gpu_default_sketch_min_reduction=0)"
    | Min_reduction min_reduction ->
        seeded_params_judged ~capabilities ~admit_narrow ~limits ~accum_prec ~min_reduction pre
    | Ranked { capacity; cache_rate; scalar_rate; dram_rate; matrix_rate; gather_rate; fill } -> (
        match
          seeded_params_judged ~capabilities ~admit_narrow ~limits ~accum_prec ~min_reduction:1 pre
        with
        | Ok p
          when ranked_below ~capacity ~cache_bandwidth:cache_rate ~issue_rate:scalar_rate
                 ~dram_bandwidth:dram_rate ~mma_rate:matrix_rate ~device_gather_rate:gather_rate
                 ~fill_model:fill p pre ->
            Ok p
        | Ok p ->
            Error
              (sketch_params_label p
             ^ ": the auto gate's ranking score does not place it below the preset")
        | Error _ as refused -> refused)
  in
  match candidate with
  | Error _ as refused -> refused
  | Ok p -> (
      match
        validate_segments_for_model ~limits
          [
            Sched.apply_classified ~static_indices
              (sketch_schedule ~accum_prec ~p pre)
              (scratch_of pre);
          ]
      with
      | _ -> Ok p
      | exception Outcome.Cause_at (_, cause) ->
          (* Typed refusals only (gh-ocannl-1390): an untyped exception from the builder or the
             transforms is a bug, not this segment's decline. *)
          logf "seeded_default: %s declined on a segment; keeping the preset"
            (sketch_params_label p);
          Error
            (sketch_params_label p ^ " declined on the segment: " ^ Outcome.detail_of_cause cause))

let seeded_segment_params ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
    ~ranked_below pre =
  Result.ok
    (seeded_segment_judgment ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
       ~ranked_below pre)

let seeded_decisions ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
    ~ranked_below ~sink_enabled (opt : LL.optimized) =
  let preset = seeded_preset ~limits in
  let zero_sched tns = Sched.zero_expansion ~limits tns in
  let keep_mapping = Sched.fission_keep_mapping ~is_gpu:true ~limits in
  let digest ~with_placements seg =
    SC.digest (SC.canonicalize ~static_indices ~with_placements seg)
  in
  (* One decision per distinct segment code — a model's repeated layers decide once. Probed on the
     segment alone: a sketch that does not apply or validate there leaves the segment to the default
     preset rather than costing the routine its other seeds. *)
  let decided = Hashtbl.create (module String) in
  (* Each segment code's census verdict, every kind of segment included (gh-ocannl-1407). Recorded
     as decided and printed once the plan's segmentation is chosen ([census_plan]), so a segment of
     the segmentation the refinement discards prints nothing. *)
  let census_verdicts = Hashtbl.create (module String) in
  let census ~key verdict =
    if log_seeding_enabled () then Hashtbl.set census_verdicts ~key:(Lazy.force key) ~data:verdict
  in
  let census_plan segments =
    if log_seeding_enabled () then
      print_seeding_lines
        (List.map segments ~f:(fun (at, _, _, _, pre) ->
             Printf.sprintf "%s where=\"seeded_default segment u%d\" %s" seeding_census_prefix at
               (Option.value ~default:"undecided"
                  (Hashtbl.find census_verdicts
                     (digest ~with_placements:false pre ^ "|" ^ digest ~with_placements:true pre)))))
  in
  let seed_of kind pre =
    let key = lazy (digest ~with_placements:false pre ^ "|" ^ digest ~with_placements:true pre) in
    match (kind, gate) with
    | `Zeros, _ ->
        census ~key "refused: a zero-fill segment, which no sketch seeds";
        None
    | `Solo, _ ->
        census ~key "refused: a solo segment, which no sketch seeds";
        None
    | `Normal, Min_reduction n when n <= 0 ->
        census ~key "refused: sketch seeding is off (gpu_default_sketch_min_reduction=0)";
        None
    | `Normal, _ ->
        Hashtbl.find_or_add decided (Lazy.force key) ~default:(fun () ->
            let judged =
              seeded_segment_judgment ~capabilities ~limits ~accum_prec ~admit_narrow
                ~static_indices ~gate ~ranked_below pre
            in
            census ~key
              (match judged with
              | Ok p -> "seeded: " ^ sketch_params_label p
              | Error why -> "refused: " ^ why);
            Result.ok judged)
  in
  (* Probed on the segment alone like a seed; a chain that does not validate leaves the plain
     preset. *)
  let sink_of kind pre =
    match kind with
    | `Zeros | `Solo -> None
    | `Normal when not sink_enabled -> None
    | `Normal -> (
        match seeded_sunk_schedule ~limits ~accum_prec ~admit_narrow ~static_indices pre with
        | None -> None
        | Some (n_swaps, sched) -> (
            match
              validate_segments_for_model ~limits
                [ Sched.apply_classified ~static_indices sched (scratch_of pre) ]
            with
            | _ -> Some n_swaps
            | exception Outcome.Cause_at _ ->
                logf "seeded_default: the reduction sink declined on a segment; keeping the preset";
                None))
  in
  (* A segmentation derived on a hermetic copy, as [(first unit, kind, units, pre-schedule slice)]
     per segment. *)
  let derive ~arity_cuts =
    match
      Sched.fission_segmented ~promote_locals:true ?keep_mapping ~arity_cuts ~preset ~zero_sched
        ~static_indices (scratch_of opt)
    with
    | exception Outcome.Cause_at _ -> None
    | plan, tuples ->
        Some
          (List.map2_exn
             (Array.to_list (segment_starts plan))
             (List.zip_exn plan tuples)
             ~f:(fun at ((kind, n), (_, pre, _, _)) -> (at, kind, n, pre)))
  in
  (* The default pipeline's own segmentation — derived exactly as [maybe_default_schedules] derives
     it, no zero fold — so every segment the seeding leaves alone is the default's segment under the
     default's schedule. *)
  match derive ~arity_cuts:false with
  | None -> None
  | Some coarse ->
      (* A default segment can hold sites its own seed cannot reach: fission keeps aligned nests
         together, so a weight gradient shares its kernel with the bias gradient reducing the same
         rows, and a projection's weight gradient with its input's data gradients, and only the
         first product is tiled. The finer [arity_cuts] segmentation (gh-ocannl-574) cuts those
         apart; {!refine_segments} takes its pieces where more of them seed (gh-ocannl-1413), and
         elsewhere the default segmentation stands, so no launch is added that buys no seed. *)
      let hybrid, refined =
        refine_segments ~coarse
          ~fine:(lazy (derive ~arity_cuts:true))
          ~seed:(fun k pre ->
            Option.map
              (seed_of (Sched.segment_class k) pre)
              ~f:(fun p -> (p, digest ~with_placements:false pre)))
      in
      census_plan hybrid;
      (* The segments no sketch takes, one decision per distinct segment code like the seeds. *)
      let sink_decided = Hashtbl.create (module String) in
      let segments =
        List.map hybrid ~f:(fun (_, k, _, s, pre) ->
            match s with
            | Some (params, digest) -> Seeded_sketch { digest; params }
            | None -> (
                let d = digest ~with_placements:false pre in
                match
                  Hashtbl.find_or_add sink_decided
                    (d ^ "|" ^ digest ~with_placements:true pre)
                    ~default:(fun () -> sink_of (Sched.segment_class k) pre)
                with
                | Some swaps -> Seeded_sunk { digest = d; swaps }
                | None -> Seeded_preset))
      in
      if List.for_all segments ~f:(function Seeded_preset -> true | _ -> false) then None
      else Some (List.map hybrid ~f:(fun (_, k, n, _, _) -> (k, n)), segments, refined)

let seeded_plan_choice (plan : seeded_plan) =
  let split = List.map plan.sp_split ~f:(fun (s, b) -> (s.sr_target, b)) in
  match plan.sp_segments with
  | None -> { no_seeded_choice with sd_split = split }
  | Some (segmentation, segs) ->
      let starts = segment_starts segmentation in
      {
        no_seeded_choice with
        sd_split = split;
        sd_seeded =
          List.filter_mapi segs ~f:(fun i -> function
            | Seeded_sketch { params; _ } -> Some (starts.(i), params)
            | _ -> None);
        sd_sunk =
          List.filter_mapi segs ~f:(fun i -> function
            | Seeded_sunk { swaps; _ } -> Some (starts.(i), swaps)
            | _ -> None);
        sd_refined = plan.sp_refined;
      }

(* The plan on [opt], the routine [split] was applied to. *)
let seeded_plan_on ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
    ~ranked_below ~sink_enabled ~split opt =
  match
    seeded_decisions ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
      ~ranked_below ~sink_enabled opt
  with
  | None ->
      {
        sp_split = split;
        sp_segments = None;
        sp_refined = 0;
        sp_start = None;
        sp_admit_narrow = admit_narrow;
      }
  | Some (segmentation, segs, refined) ->
      {
        sp_split = split;
        sp_segments = Some (segmentation, segs);
        sp_refined = refined;
        sp_start = None;
        sp_admit_narrow = admit_narrow;
      }

let seeded_plan ?min_reduction ?seed_gate ?min_serial ?reduction_sink ?seed_narrow
    ?(capabilities = Ir.Backend_intf.no_codegen_capabilities) ~backend ~limits ~accum_prec
    ~static_indices (opt : LL.optimized) =
  match
    seeded_settings ?min_reduction ?seed_gate ?min_serial ?reduction_sink ?seed_narrow ~backend
      ~configured_gate:(fun () -> seeded_default_gate ~limits)
      ()
  with
  | None -> None
  | Some (gate, min_serial, sink_enabled, admit_narrow) -> (
      (* Hermetic, prelude included: deriving the plan must leave the caller's lowering as it
         was. *)
      let opt, split = seeded_split ~backend ~min_serial ~static_indices (scratch_of opt) in
      let ranked_below = seeded_ranker ~capabilities ~limits ~accum_prec ~static_indices in
      match
        seeded_plan_on ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
          ~ranked_below ~sink_enabled ~split opt
      with
      | { sp_split = []; sp_segments = None; _ } -> None
      | plan ->
          (* Every reading the per-segment decisions rest on (gh-ocannl-1442, gh-ocannl-1479): the
             staffed start re-decides on the search's own segments, which the plan digest does not
             cover. *)
          let gate_tag = seed_gate_tag gate in
          Some
            {
              plan with
              sp_start =
                Some
                  {
                    decide =
                      seeded_segment_params ~capabilities ~limits ~accum_prec ~admit_narrow
                        ~static_indices ~gate ~ranked_below;
                    gate_tag;
                  };
            })

(* The seeded pipeline on [opt] (mutated as a compile's transform input is): the splits, the plan
   decided on the split routine, its replay. Shared by the untuned compile's transform and by
   [model_default], which starts from it (gh-ocannl-1253). *)
let seeded_apply ?(on_choice = fun (_ : seeded_choice) -> ()) ~capabilities ~backend ~limits
    ~accum_prec ~static_indices ~gate ~min_serial ~sink_enabled ~admit_narrow (opt : LL.optimized) :
    LL.optimized list =
  let t0 = Unix.gettimeofday () in
  let opt, split = seeded_split ~backend ~min_serial ~static_indices opt in
  on_choice
    (seeded_plan_choice
       {
         sp_split = split;
         sp_segments = None;
         sp_refined = 0;
         sp_start = None;
         sp_admit_narrow = admit_narrow;
       });
  let plan =
    seeded_plan_on ~capabilities ~limits ~accum_prec ~admit_narrow ~static_indices ~gate
      ~ranked_below:(seeded_ranker ~capabilities ~limits ~accum_prec ~static_indices)
      ~sink_enabled ~split opt
  in
  (* gh-ocannl-1253: the derivation [tune] repeats (as [seeded_plan]) on a trajectory-matched warm
     hit, timed so its cost can be read off any untuned compile. *)
  logf "seeded_default: plan derived in %.3f s" (Unix.gettimeofday () -. t0);
  logf "seeded_default: gate %s" (seed_gate_tag gate);
  let (_, tuples), applied =
    seeded_replay ~limits ~accum_prec ~admit_narrow ~static_indices plan.sp_segments opt
  in
  if applied then (
    let choice = seeded_plan_choice plan in
    logf
      "seeded_default: %d of %d segments seeded (%d default segments refined), %d with reductions \
       sunk: %s"
      (List.length choice.sd_seeded)
      (List.length (fst (Option.value_exn plan.sp_segments)))
      choice.sd_refined (List.length choice.sd_sunk)
      (String.concat ~sep:", "
         (List.map choice.sd_seeded ~f:(fun (at, p) ->
              Printf.sprintf "u%d %s" at (sketch_params_label p))));
    on_choice choice);
  List.map tuples ~f:(fun (_, _, _, post) -> post)

let seeded_transform ?min_reduction ?seed_gate ?min_serial ?reduction_sink ?seed_narrow
    ?(on_choice = fun (_ : seeded_choice) -> ()) ctx bindings =
  let backend = Context.backend_name ctx in
  match
    seeded_settings ?min_reduction ?seed_gate ?min_serial ?reduction_sink ?seed_narrow ~backend
      ~configured_gate:(fun () -> seeded_default_gate ~limits:(Context.hardware_limits ctx))
      ()
  with
  | None -> None
  | Some (gate, min_serial, sink_enabled, admit_narrow) ->
      let limits = Context.hardware_limits ctx in
      let capabilities = Context.codegen_capabilities ctx in
      let accum_prec = capabilities.Ir.Backend_intf.accum_prec in
      let static_indices = Idx.bound_symbols bindings in
      Some
        (seeded_apply ~on_choice ~capabilities ~backend ~limits ~accum_prec ~static_indices ~gate
           ~min_serial ~sink_enabled ~admit_narrow)

(* gh-ocannl-1479: the launch sweep's pinned sketch -- one whole-routine scalar blocktile leaf of
   the matmul family, whatever the seeding would pick at the size, so every timed point of a fit
   runs the same geometry. Applied and validated as [model_default] applies a whole-routine pick. *)
let pinned_sketch_transform ~select ?(on_leaf = fun (_ : sketch_params) -> ()) ctx bindings =
  let backend = Context.backend_name ctx in
  if not (Sched.backend_is_gpu backend) then
    raise
      (Utils.User_error
         (Printf.sprintf "Autotune.pinned_sketch_transform: backend %s has no GPU sketch family"
            backend));
  let limits = Context.hardware_limits ctx in
  let capabilities = Context.codegen_capabilities ctx in
  let accum_prec = capabilities.Ir.Backend_intf.accum_prec in
  let static_indices = Idx.bound_symbols bindings in
  fun (opt : LL.optimized) ->
    match matmul_sketch_tree ~capabilities ~is_gpu:true ~is_cpu:false ~limits opt with
    | None -> raise (Utils.User_error "Autotune.pinned_sketch_transform: no matmul site")
    | Some tree -> (
        match List.find (Sspace.leaves tree) ~f:(fun p -> scalar_blocktile p && select p) with
        | None ->
            raise
              (Utils.User_error
                 "Autotune.pinned_sketch_transform: no scalar blocktile leaf of the family matches")
        | Some p ->
            on_leaf p;
            validate_segments_for_model ~limits
              [ Sched.apply_classified ~static_indices (sketch_schedule ~accum_prec ~p opt) opt ])

let ranking_multiprocessors ~limits =
  let (u : Ir.Backend_intf.untimed_limits) = Ir.Backend_intf.untimed_limits limits in
  resolve multiprocessors_override u.core_count

let seeded_default ?name ?min_reduction ?seed_gate ?min_serial ?reduction_sink ?seed_narrow ?report
    ctx comp bindings =
  let choice = ref no_seeded_choice in
  let emit () = Option.iter report ~f:(fun f -> f !choice) in
  match
    seeded_transform ?min_reduction ?seed_gate ?min_serial ?reduction_sink ?seed_narrow
      ~on_choice:(fun c -> choice := c)
      ctx bindings
  with
  | None ->
      emit ();
      Context.compile ?name ctx comp bindings
  | Some transform ->
      let on_fallback exn =
        logf "seeded_default: compiling the seeded pipeline FAILED (%s); recompiling the default"
          (Exn.to_string exn);
        choice := { no_seeded_choice with sd_fell_back = true }
      in
      let result =
        compile_advisory ?name ~on_fallback
          ~fallback_if:(fun () ->
            not
              (List.is_empty !choice.sd_seeded && List.is_empty !choice.sd_split
             && List.is_empty !choice.sd_sunk))
          transform ctx comp bindings
      in
      emit ();
      result

(* gh-ocannl-1253: on CUDA and HIP the model starts from the seeded untuned plan
   ({!seeded_default}'s, [auto] gate included) rather than the plain default pipeline: it is the
   incumbent, and the action taken when no candidate strictly beats it -- measured, on each searched
   program, against its own sketch there ({!model_seeded_anchors}, gh-ocannl-1477). [None] where the
   untuned compile is the plain default (another backend, seeding off). Shared by [model_default]
   and [model_explain]. *)
let model_seeding ~backend ~limits =
  if not (Sched.backend_is_gpu backend) then None
  else seeded_settings ~backend ~configured_gate:(fun () -> seeded_default_gate ~limits) ()

(* Where the model starts from the seeded plan, the family is restricted to the numerics-preserving
   candidate set the seeding uses ({!scalar_blocktile}, at a site accumulating at its storage
   precision), so a model pick never changes a result the seeded plan would not: tensorized forms
   reround, the conv family reorders the reduction. (The coalesced branch is not enumerated here at
   all: it preserves numerics, but the score ties it with its uncoalesced twins, gh-ocannl-1278.)
   Other leaves are kept out of the ranking, counted as not numerics-preserving, never an error.
   Metal, the CPU backends and an unseeded CUDA/HIP compile keep the whole family. *)
let model_admits ~seeding ~accum_prec (base_opt : LL.optimized) =
  (* gh-ocannl-1405: a narrow-storage site is in the set exactly where the seeded plan admits it, so
     the model's pick still changes no result the seeded plan would not. *)
  let admit_narrow = match seeding with Some (_, _, _, admit) -> admit | None -> false in
  let preserving =
    lazy
      (admit_narrow
      ||
      match detect_matmul_canonical base_opt with
      | Some (site, _, _) -> accumulates_at_storage ~accum_prec site
      | None -> true)
  in
  fun p -> Option.is_none seeding || (scalar_blocktile p && Lazy.force preserving)

(* gh-ocannl-1477: the seeded plan's own sketches on a program [model_default] searches -- the whole
   routine, or one keyed segment of its per-segment substitution: each scalar blocktile the plan
   applied ([seeded], its [sd_seeded] parameters, deduplicated), applied to [prog] and validated as
   a family leaf is, those that apply. Empty without seeding, or where the plan seeds nothing. The
   caller's threshold is the best-scoring one.

   Why the walks need it: the seeded plan runs its zero-fill as a kernel of its own (the default
   segmentation, no zero fold), while a family leaf keeps the zero nest inside its one kernel, and
   the ranking score sums [max (compute, memory)] per kernel. So the seeded plan pays the zero
   kernel's memory leg as a separate term that a leaf hides under its compute leg; with every scalar
   leaf of a site tied on the compute leg, the first leaf of the walk -- a plain one, its batch
   loops serial inside one block, measured 2.6-35x slower -- "strictly" beat the plan. Scored in the
   candidates' own decomposition, the seeded geometry ties them, and the incumbent wins every tie.

   The parameters are the plan's RECORD, never its gate re-run on [prog]: under the [auto] gate the
   ranked judgment of the whole routine (zero nest and companions inside) can decline a site whose
   own segment the plan seeds, which left the walk unanchored and the defect in place under the
   [performance] and [approximate] profiles. A segment of a multi-site routine is held to every
   seeded geometry that applies to it, not only its own site's: the stricter threshold, never a
   looser one. Shared by [model_default] and [model_explain]. *)
let model_seeded_anchors ~limits ~accum_prec ~static_indices ~(seeded : sketch_params list)
    (prog : LL.optimized) : (sketch_params * LL.optimized list) list =
  List.dedup_and_sort seeded ~compare:(fun a b ->
      String.compare (sketch_params_label a) (sketch_params_label b))
  |> List.filter_map ~f:(fun p ->
      match
        validate_segments_for_model ~limits
          [
            Sched.apply_classified ~static_indices
              (sketch_schedule ~accum_prec ~p prog)
              (scratch_of prog);
          ]
      with
      | segs -> Some (p, segs)
      | exception Outcome.Cause_at _ -> None)

(* The seeded plan's record ([sd_seeded]: first unit, parameters) for one keyed segment of a fission
   of the same routine ([plan]): the entries whose unit lies in one of the segment's occurrences.
   Both segmentations number the routine's units alike; the fold-zeros one merges a zero-fill into
   the segment of the site it initializes, which still holds the site's unit. *)
let seeded_record_for_key ~(plan : Sched.segmentation) ~record (key : seg_key) =
  let starts = segment_starts plan in
  let spans = List.mapi plan ~f:(fun i (_, n) -> (starts.(i), n)) in
  List.filter_map record ~f:(fun (u, p) ->
      Option.some_if
        (List.exists spans ~f:(fun (start, n) ->
             List.mem key.key_units start ~equal:Int.equal && start <= u && u < start + n))
        p)

let model_default ?name ?report ctx comp bindings =
  let backend = Context.backend_name ctx in
  let is_gpu = Sched.backend_is_gpu backend and is_cpu = Sched.backend_is_cpu backend in
  let limits = Context.hardware_limits ctx in
  let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
  let static_indices = Idx.bound_symbols bindings in
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let emit r = Option.iter report ~f:(fun f -> f r) in
  let no_selection =
    {
      mc_label = "default";
      mc_model_ms = None;
      mc_rank_ms = None;
      mc_scored = 0;
      mc_skipped = 0;
      mc_rejected = 0;
      mc_excluded = 0;
    }
  in
  if
    (Option.is_none peak_flops && Option.is_none peak_memory_bandwidth)
    || not (Sched.automatic_schedule_active ~backend_name:backend)
  then (
    emit no_selection;
    Context.compile ?name ctx comp bindings)
  else
    let choice = ref no_selection in
    (* Whether the segments the compile actually received came from a model pick rather than the
       default pipeline — the condition for the compile-level fallback below to have anywhere to
       fall back to. *)
    let applied_pick = ref false in
    (* Counters and scoring helpers at [model_default] scope rather than per-compile: the placement
       pre-search (config [model_default_placements]) scores hermetic lowerings before any compile
       runs, and its work accumulates into the same reported totals as the in-compile selection. *)
    let n_scored = ref 0 and n_skipped = ref 0 and n_rejected = ref 0 and n_excluded = ref 0 in
    (* Candidates rank by the ranking score (gh-ocannl-1253), which charges operand re-reads once
       the working set leaves the cache — under [analyze]'s distinct-cell roofline an untiled
       contraction scores at the family's floor and no sketch can win. The family search's bound
       below stays [completion_floor]'s roofline: the ranking score is never below [analyze]'s, so
       it is never below the bound, and fathoming stays sound. *)
    let score opts =
      match
        segments_ranking
          ~capabilities:(Context.codegen_capabilities ctx)
          ~peak_flops ~peak_memory_bandwidth ~limits opts
      with
      | Some (rank, _bound) ->
          Int.incr n_scored;
          Some rank
      | None ->
          Int.incr n_skipped;
          None
    in
    (* The model must rank the best viable schedule, not crown an invalid argmin and fall all the
       way back to default. The validator is typed, so only an expected schedule rejection is
       excluded; compiler assertions and other failures still escape. *)
    let score_valid opts =
      match validate_segments_for_model ~fork_join:is_cpu ~limits opts with
      | opts -> score opts
      | exception Outcome.Cause_at _ ->
          Int.incr n_rejected;
          None
    in
    (* gh-ocannl-1253: on CUDA and HIP the model starts from the seeded untuned plan
       ({!seeded_default}'s, [auto] gate included) rather than the plain default pipeline: it is the
       incumbent, and the action taken when no candidate strictly beats it (and its own sketch,
       [anchored] below). [None] where the untuned compile is the plain default (another backend,
       seeding off). *)
    let seeding_here = lazy (model_seeding ~backend ~limits) in
    (* Where the model starts from the seeded plan, the family is restricted to the
       numerics-preserving candidate set the seeding uses ({!scalar_blocktile}, at a site
       accumulating at its storage precision), so a model pick never changes a result the seeded
       plan would not: tensorized forms reround, the conv family reorders the reduction (see
       [model_admits] on the coalesced branch). Other leaves are kept out of the ranking, counted as
       not numerics-preserving ([n_excluded]), never an error. Metal, the CPU backends and an
       unseeded CUDA/HIP compile keep the whole family. *)
    let admits base_opt = model_admits ~seeding:(Lazy.force seeding_here) ~accum_prec base_opt in
    let capabilities = Context.codegen_capabilities ctx in
    (* gh-ocannl-1477: under seeding, a walk's threshold is the seeded plan's own sketch on the
       searched program, scored as its leaves are ({!model_seeded_anchors}), where it is below the
       incumbent: a leaf that only ties the seeded geometry never displaces the seeded plan, while
       one the score separates from it (a memory leg past the cache, say) still can. Placement
       leaves reuse [select], so they are anchored too. Returns the walk's threshold on [prog] and
       the anchor setting it: the lower of [incumbent] and the best-scoring of the seeded plan's own
       sketches that apply to [prog] ([seeded], the plan's record for the units [prog] covers). *)
    let anchored ~seeded ~incumbent prog =
      List.fold (model_seeded_anchors ~limits ~accum_prec ~static_indices ~seeded prog)
        ~init:(incumbent, None) ~f:(fun ((threshold, best) as acc) (p, segs) ->
          match score segs with
          | Some a ->
              logf
                "model_default: anchored at the seeded sketch %s (rank %.6f ms, incumbent %.6f ms)"
                (sketch_params_label p) (a *. 1e3) (incumbent *. 1e3);
              let best =
                match best with Some (_, b) when Float.(b <= a) -> best | _ -> Some (p, a)
              in
              (Float.min threshold a, best)
          | None -> acc)
    in
    (* [model_family_search] with the default as incumbent, logging its ledger; under the
       numerics-preserving restriction, a leaf outside it is excluded before it is scheduled.
       Returns the first leaf strictly better than [incumbent] (exact ties broken by operand loads,
       gh-ocannl-1254). *)
    let tree_search ~incumbent base_opt =
      (* Snapshot the caller-side counters so the log can split the driver's unscored leaves into
         compiler rejections vs genuine no-coverage — st_unscored alone would misclassify rejections
         as cost-model gaps in the phase-6 ledger. *)
      let r0 = !n_rejected and k0 = !n_skipped and e0 = !n_excluded in
      match
        model_family_search
          ~on_rejected:(fun () -> Int.incr n_rejected)
          ~admit:(admits base_opt)
          ~on_excluded:(fun () -> Int.incr n_excluded)
          ~capabilities ~is_gpu ~is_cpu ~limits ~static_indices ~peak_flops ~peak_memory_bandwidth
          ~score:score_valid ~incumbent base_opt
      with
      | None -> None
      | Some (best, stats, fb) ->
          logf
            "model_default: family search: %d expanded, %d scored, %d unscored (%d rejected, %d \
             without coverage, %d not numerics-preserving), %d fathomed (bound %s), %d refuted, %d \
             excluded"
            stats.Sspace.st_expanded stats.Sspace.st_scored stats.Sspace.st_unscored
            (!n_rejected - r0) (!n_skipped - k0) (!n_excluded - e0) stats.Sspace.st_fathomed
            (match fb with Some b -> Printf.sprintf "%.6f ms" (b *. 1e3) | None -> "n/a")
            stats.Sspace.st_refuted stats.Sspace.st_excluded;
          Some best
    in
    (* A per-segment substitution, as [tune]'s [F_sketch] candidates apply one: [plan] replayed,
       each keyed segment found by position (gh-ocannl-1243), the others on their kind's CONFIGURED
       preset (gh-ocannl-1512) -- what the incumbent ships, not the search's [min_parallel:1] one
       its segments were ranked against ({!model_segments}). The model times nothing: on GPU the
       search's preset can give a small segment a lane reduction, which reorders its sums, where the
       configured one keeps it serial -- under seeding, a result the seeded plan would not compute;
       on cc it would fan a tiny segment out over the pool, an overhead the score does not see. So a
       pick labeled [F_sketch[...]] is the search's candidate of that label up to the presets of the
       segments it leaves unsketched; the substituted pipeline is scored whole and must still
       strictly beat the incumbent. *)
    let preset = segment_preset ~presets:`Config ~is_gpu ~is_cpu ~limits in
    let apply_fiss ~plan ~entries opt =
      let replay =
        sketch_replay ~static_indices
          ~sketch:(fun p seg -> sketch_schedule ~accum_prec ~p seg)
          plan entries
      in
      List.map
        (snd (tuner_fission ~is_gpu ~limits ~replay ~preset ~static_indices opt))
        ~f:(fun (_, _, _, post) -> post)
    in
    (* The model-argmin pipeline choice for one lowering — label, roofline score (seconds), and the
       action reproducing it. Shared between the in-compile transform seam and the placement
       pre-search's leaf scoring (hermetic lowerings of decided placement vectors). *)
    let seeded_segs ?on_choice seeding opt =
      let gate, min_serial, sink_enabled, admit_narrow = seeding in
      seeded_apply ?on_choice ~capabilities ~backend ~limits ~accum_prec ~static_indices ~gate
        ~min_serial ~sink_enabled ~admit_narrow opt
    in
    (* Whether the applied seeded plan differs from the default pipeline, as [seeded_default] judges
       it: a plan that seeds, splits and sinks nothing IS the default pipeline, so a compilation
       failure of it is the default's and is not retried. *)
    let seeded_changed = ref false in
    (* The seeded plan's split prelude (CUDA, gh-ocannl-1325); the program itself where the model
       does not start from the seeded plan. On a hermetic copy for scoring; in place for the pick's
       application, as the [`Seeded] action's [seeded_apply] splits the compile's own lowering: a
       pick applied to a copy registers the nodes its transforms create (a fissioned segment's
       promoted locals) in the copy's placements, not the compiling context's, and the compile then
       fails on an undecided node (gh-ocannl-1477's fissioned chain, on HIP). *)
    let prelude ?(hermetic = true) (opt : LL.optimized) =
      match Lazy.force seeding_here with
      | Some (_, min_serial, _, _) ->
          fst
            (seeded_split ~backend ~min_serial ~static_indices
               (if hermetic then scratch_of opt else opt))
      | None -> opt
    in
    let select (opt : LL.optimized) =
      let seeded_record = ref [] in
      try
        (* The incumbent -- the seeded plan where there is one, else the untuned default pipeline --
           scored on a hermetic copy: both the anchor candidate and the fallback. *)
        let incumbent_label, incumbent_action, incumbent_scratch =
          match Lazy.force seeding_here with
          | Some seeding ->
              ( "seeded",
                `Seeded seeding,
                seeded_segs
                  ~on_choice:(fun c -> seeded_record := c.sd_seeded)
                  seeding (scratch_of opt) )
          | None ->
              ( "default",
                `Default,
                Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices
                  (scratch_of opt) )
        in
        (* gh-ocannl-1477: what the incumbent plan seeded, by first unit -- the anchors' source. *)
        let seeded_record = !seeded_record in
        let default_score = score incumbent_scratch in
        (* The candidates compete on the program the seeded plan starts from, its split prelude
           included, so a winner keeps the reductions the untuned compile splits. *)
        let base = prelude opt in
        match default_score with
        | None ->
            (* No coverage of the incumbent itself: nothing to honestly compare against. *)
            (incumbent_label, None, incumbent_action)
        | Some ds -> (
            (* Whole-routine sketch candidates. A candidate without coverage is skipped — it is
               never picked over the default without a measured run ({!tune} covers that). *)
            let capabilities = Context.codegen_capabilities ctx in
            log_seeding ~capabilities ~where:"model_default whole-routine" ~is_gpu ~is_cpu ~limits
              base;
            (* The claiming family's tree (matmul or conv), twins included, searched with the
               default as incumbent. *)
            let whole_best =
              Option.join
                (tree_search
                   ~incumbent:
                     (fst (anchored ~seeded:(List.map seeded_record ~f:snd) ~incumbent:ds base))
                   base)
            in
            let contenders =
              match whole_best with
              | Some (p, sc) -> [ (spec_label (Whole (W_sketch p)), sc, `Whole p) ]
              | None -> []
            in
            (* Per-segment sketch substitution when the search's segmentation splits the routine or
               exposes a new lowering (a covering GPU zero companion) for sketching
               ({!model_segments}, gh-ocannl-1512). Mirrors [tune]'s [F_sketch] flavor: the derived
               segmentation replayed, keyed segments found in it by position (gh-ocannl-1243), a
               digest mismatch degrading to the configured preset. *)
            let fiss =
              match model_segments ~is_gpu ~is_cpu ~limits ~static_indices base with
              | exception Outcome.Cause_at _ ->
                  Int.incr n_rejected;
                  None
              | None -> None
              | Some (plan, keyed) -> (
                  let decided =
                    List.filter_map keyed ~f:(fun (key, pre, post) ->
                        match score [ post ] with
                        | None -> None
                        | Some bs -> (
                            log_seeding ~capabilities
                              ~where:("model_default segment " ^ seg_key_tag key)
                              ~is_gpu ~is_cpu ~limits pre;
                            (* The segment's family tree searched with the segment's own
                               default-preset score as incumbent, anchored at the seeded plan's own
                               sketch on the segment under seeding (gh-ocannl-1477). *)
                            let threshold, anchor =
                              anchored
                                ~seeded:(seeded_record_for_key ~plan ~record:seeded_record key)
                                ~incumbent:bs pre
                            in
                            match Option.join (tree_search ~incumbent:threshold pre) with
                            | Some (p, _s) -> Some (`Strict (key, p))
                            | None -> Option.map anchor ~f:(fun (p, _) -> `Anchor (key, p))))
                  in
                  (* A substitution only where some segment strictly improves on its anchor; every
                     other segment the seeded plan seeds keeps its seeded sketch (round 1 of
                     staging#1332), so the contender is the seeded plan with strict improvements,
                     not those improvements over default presets. *)
                  let entries =
                    if List.exists decided ~f:(function `Strict _ -> true | `Anchor _ -> false)
                    then List.map decided ~f:(function `Strict e | `Anchor e -> e)
                    else []
                  in
                  if List.is_empty entries then None
                  else
                    (* Score the substituted pipeline whole, so it competes on the same footing as
                       the other candidates. *)
                    match apply_fiss ~plan ~entries (scratch_of base) with
                    | exception Outcome.Cause_at _ ->
                        Int.incr n_rejected;
                        None
                    | posts -> Option.map (score_valid posts) ~f:(fun s -> (plan, entries, s)))
            in
            let contenders =
              contenders
              @
              match fiss with
              | Some (plan, entries, s) ->
                  [
                    ( spec_label (Fiss (F_sketch { plan; entries; staff = []; fine = false })),
                      s,
                      `Fiss (plan, entries) );
                  ]
              | None -> []
            in
            (* Argmin with ties to the default: the model only displaces the honest default on a
               strict improvement. *)
            let best =
              List.min_elt contenders ~compare:(fun (_, a, _) (_, b, _) -> Float.compare a b)
            in
            match best with
            | Some (lbl, s, act) when Float.(s < ds) -> (lbl, Some s, act)
            | _ -> (incumbent_label, Some ds, incumbent_action))
      with Outcome.Cause_at (_, cause) ->
        logf "model_default: scoring declined (%s); using the default pipeline"
          (Outcome.detail_of_cause cause);
        ("default", None, `Default)
    in
    let transforms (opt : LL.optimized) : LL.optimized list =
      let default_segs () =
        Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices opt
      in
      let label, model_s, action = select opt in
      choice :=
        {
          mc_label = label;
          mc_model_ms = None;
          mc_rank_ms = Option.map model_s ~f:(fun s -> s *. 1e3);
          mc_scored = !n_scored;
          mc_skipped = !n_skipped;
          mc_rejected = !n_rejected;
          mc_excluded = !n_excluded;
        };
      let apply_action () =
        (* Schedule application uses the typed seam. Backend validation is deliberately left to
           [compile_advisory], which now receives its classified cause directly from codegen. *)
        match action with
        | `Default -> default_segs ()
        | `Seeded seeding ->
            seeded_changed := false;
            validate_segments_for_model ~fork_join:is_cpu ~limits
              (seeded_segs
                 ~on_choice:(fun c ->
                   seeded_changed :=
                     not
                       (List.is_empty c.sd_seeded && List.is_empty c.sd_split
                      && List.is_empty c.sd_sunk))
                 seeding opt)
        | `Whole p ->
            let opt = prelude ~hermetic:false opt in
            validate_segments_for_model ~fork_join:is_cpu ~limits
              [ Sched.apply_classified ~static_indices (sketch_schedule ~accum_prec ~p opt) opt ]
        | `Fiss (plan, entries) ->
            let opt = prelude ~hermetic:false opt in
            validate_segments_for_model ~fork_join:is_cpu ~limits (apply_fiss ~plan ~entries opt)
      in
      match apply_action () with
      | segs ->
          (* The lower bound of what is applied, beside the rank the pick minimized. *)
          (if Option.is_some model_s then
             let bound =
               summaries_roofline ~peak_flops ~peak_memory_bandwidth
                 (List.map segs ~f:(fun o -> CM.analyze o.LL.llc))
             in
             choice :=
               {
                 !choice with
                 mc_model_ms = Option.map bound ~f:(fun s -> s *. 1e3);
                 mc_rank_ms = Option.bind bound ~f:(fun _ -> !choice.mc_rank_ms);
               });
          logf
            "model_default: chose %s (rank %s; %d scored, %d without coverage, %d unbuildable, %d \
             not numerics-preserving)"
            label
            (match model_s with Some s -> Printf.sprintf "%.6f ms" (s *. 1e3) | None -> "n/a")
            !n_scored !n_skipped !n_rejected !n_excluded;
          (applied_pick :=
             match action with
             | `Default -> false
             | `Seeded _ -> !seeded_changed
             | `Whole _ | `Fiss _ -> true);
          segs
      | exception Outcome.Cause_at (_, cause) ->
          logf
            "model_default: winner %s FAILED to apply or validate (%s); using the default pipeline"
            label (Outcome.detail_of_cause cause);
          choice :=
            {
              no_selection with
              mc_scored = !n_scored;
              mc_skipped = !n_skipped;
              mc_rejected = !n_rejected;
              mc_excluded = !n_excluded;
            };
          applied_pick := false;
          default_segs ()
    in
    let on_fallback exn =
      logf "model_default: compiling the pick %s FAILED (%s); recompiling the default pipeline"
        !choice.mc_label (Exn.to_string exn);
      choice := { !choice with mc_label = "default"; mc_model_ms = None; mc_rank_ms = None }
    in
    (* gh-ocannl-514, the placement levels of the untuned regime (config [model_default_placements]
       = N > 0): before the compile, branch-and-bound over the top-N ranked flip candidates of the
       decision surface — one node per level, kept at its default reading or flipped to one of its
       alternatives, the all-keep leaf visited first so the default placements' own selection score
       is the running incumbent (ties stay with the default placements), [select] pricing each
       leaf's hermetic lowering ([Context.lowered_for_decisions]), and the partial-vector roofline
       floor ([placement_surface.ps_floor_ms], monotone in the committed materializations) fathoming
       subtrees that cannot beat it. This is where the bound differentiates {e within} the tree: the
       family levels' floor is schedule-invariant, the placement levels' is not (phase 3). *)
    let placement_budget =
      Int.of_string
        (String.strip (Utils.get_global_arg ~arg_name:"model_default_placements" ~default:"0"))
    in
    let placement_pick =
      if placement_budget <= 0 then None
      else
        match
          let surface = placement_surface ?name ctx comp bindings in
          (* The cut counts nodes (gh-ocannl-1017): a candidate is the whole node, so a level always
             decides every reading it has. *)
          let cands = List.take surface.ps_candidates placement_budget in
          if List.is_empty cands then None
          else
            let level_name (fc : LL.flip_candidate) =
              Printf.sprintf "placement#%d %s %s" fc.LL.fc_tn.Ir.Tnode.uid
                (Ir.Tnode.debug_name fc.LL.fc_tn)
                (String.concat ~sep:"/"
                   (List.map fc.LL.fc_alternatives ~f:(fun fa -> LL.reading_to_string fa.LL.fa_flip)))
            in
            (* The placement levels commit to DATA like the family levels do (gh-ocannl-591): each
               child carries the candidate it decides and which way, so the bound below reads the
               path instead of finding the candidate back through the level name and the commitment
               back through the label. [level_name] is the display name only. A leaf is the vector
               of the flips taken, one per flipped node. *)
            let rec build vector = function
              | [] -> Sspace.Leaf (List.rev vector)
              | (fc : LL.flip_candidate) :: rest ->
                  (* One MULTIWAY level per node (gh-ocannl-616): its alternatives are mutually
                     exclusive readings, so a level keeps the default or takes exactly one. *)
                  Sspace.Choice
                    {
                      level = level_name fc;
                      children =
                        ((fc, `Keep), Sspace.Child (lazy (build vector rest)))
                        :: List.map fc.LL.fc_alternatives ~f:(fun (fa : LL.flip_alternative) ->
                            ( (fc, `Flip fa.LL.fa_flip),
                              Sspace.Child
                                (lazy (build ((fc.LL.fc_tn, fa.LL.fa_flip) :: vector) rest)) ));
                    }
            in
            let score vector =
              let mat, inl, fp = LL.partition_readings vector in
              match
                Context.lowered_for_decisions ?name ~materialized:mat ~inline:inl ~footprint:fp ctx
                  comp bindings
              with
              | opt_v ->
                  let _lbl, s, _act = select opt_v in
                  s
              | exception Outcome.Cause_at _ -> None
            in
            let bound ~path _sub =
              let mat =
                List.filter_map path ~f:(fun (_level, ((fc : LL.flip_candidate), commitment)) ->
                    (* Certainly materialized below this node: a committed [`Materialize] flip, or a
                       kept node whose default reading is [`Materialize] — the candidate's own field
                       (gh-ocannl-1017), not an inference from which alternatives it offers. A level
                       decides its whole node, so a kept level is the whole node kept. The other
                       commitments (and every open level) contribute zero. *)
                    match commitment with
                    | `Flip `Materialize -> Some fc.LL.fc_tn
                    | `Keep ->
                        Option.some_if (LL.equal_reading fc.LL.fc_default `Materialize) fc.LL.fc_tn
                    | `Flip (`Inline | `Footprint) -> None)
              in
              (* [ps_floor_ms] is milliseconds; [select]'s scores are roofline seconds. *)
              Option.map (surface.ps_floor_ms ~materialized:mat) ~f:(fun ms -> ms /. 1e3)
            in
            let best, stats = Sspace.search ~bound ~score (build [] cands) in
            logf
              "model_default: placement search over %d level(s): %d expanded, %d scored, %d \
               unscored, %d fathomed"
              (List.length cands) stats.Sspace.st_expanded stats.Sspace.st_scored
              stats.Sspace.st_unscored stats.Sspace.st_fathomed;
            match best with
            | Some (vector, s) when not (List.is_empty vector) ->
                let mat, inl, fp = LL.partition_readings vector in
                let names tns = String.concat ~sep:"," (List.map tns ~f:Ir.Tnode.debug_name) in
                logf
                  "model_default: placement pick: materialize [%s], inline [%s], footprint [%s] \
                   (model %.6f ms)"
                  (names mat) (names inl) (names fp) (s *. 1e3);
                Some (mat, inl, fp)
            | _ -> None
        with
        | pick -> pick
        | exception Outcome.Cause_at (_, cause) ->
            logf
              "model_default: placement search declined (%s); selecting from the default placements"
              (Outcome.detail_of_cause cause);
            None
    in
    (* With the model on the default pipeline (no sketch strictly improved on it, or the pick failed
       validation above), the compile that just failed IS the fallback: retrying it would duplicate
       an expensive failure and delay the honest error, so the exception propagates instead. *)
    let compile_from ?on_unrecovered base_ctx =
      compile_advisory ?name ~on_fallback ?on_unrecovered
        ~fallback_if:(fun () -> !applied_pick)
        transforms base_ctx comp bindings
    in
    let result =
      match placement_pick with
      | None -> compile_from ctx
      | Some (mat, inl, fp) ->
          (* Every reading gets its call, empty or not — the historical context lineage of this
             path, kept as is; skipping would change only context identity, not the decisions. *)
          let ctx' = Context.decide_readings ~skip_empty:false ctx (mat, inl, fp) in
          let picked = ref true in
          (* A CLASSIFIED rejection that [compile_advisory] had nothing to fall back to under the
             picked placements: [applied_pick] was false (the pick's win was the placement move
             itself, so the failing compile was already the default pipeline — under [ctx'], not
             under [ctx]), or its fallback to the default pipeline under [ctx'] failed classified
             too. The pick is advisory, so abandon it and rerun the ordinary selection from the
             caller's own placements. Only the classified cause comes here (gh-ocannl-1390): a fatal
             failure — an untyped exception a transform or the builder raised, under
             [strict_failure_classification] — propagates, rather than being retried from the
             default placements as one more decline. *)
          let on_unrecovered (classified : Outcome.classified_cause) =
            logf
              "model_default: compiling under the picked placements FAILED (%s); recompiling \
               from                the default placements"
              (Exn.to_string (Outcome.exception_of_cause classified.cause));
            picked := false;
            applied_pick := false;
            compile_from ctx
          in
          let result = compile_from ~on_unrecovered ctx' in
          (if !picked then
             (* The emitted label carries the placement decision: the in-compile selection only
                names the pipeline it chose under those placements. *)
             let names tns = String.concat ~sep:"," (List.map tns ~f:Ir.Tnode.debug_name) in
             choice :=
               {
                 !choice with
                 mc_label =
                   Printf.sprintf "placements[mat:%s inl:%s fp:%s] %s" (names mat) (names inl)
                     (names fp) !choice.mc_label;
               });
          result
    in
    emit !choice;
    result

(* gh-ocannl-1254, the #1269 interface: what the model sees of each candidate [model_default] ranks
   for one computation — its incumbent (the seeded plan where it starts from one, else the default
   pipeline) and every whole-routine sketch leaf of the program it searches (the seeded plan's split
   prelude included), each marked with whether the numerics restriction admits it — without
   choosing. *)
type explain = {
  ex_floor_ms : float option;
  ex_rank_ms : float option;
  ex_compute_ms : float option;
  ex_memory_ms : float option;
  ex_spilled : bool option;
  ex_fills : float list option;
  ex_mma_work : float list option;
  ex_device_gathers : float list option;
  ex_operand_loads : CM.load_count;
  ex_admitted : bool;
  ex_anchor : bool;
}

type segment_explain = { xs_segment : string; xs_candidates : (string * explain) list }
type explanation = { x_routine : (string * explain) list; x_segments : segment_explain list }

let model_explain ?name ?(coalesced = false) ctx comp bindings =
  let backend = Context.backend_name ctx in
  let is_gpu = Sched.backend_is_gpu backend and is_cpu = Sched.backend_is_cpu backend in
  let limits = Context.hardware_limits ctx in
  let caps = Context.codegen_capabilities ctx in
  let accum_prec = caps.Ir.Backend_intf.accum_prec in
  let static_indices = Idx.bound_symbols bindings in
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let opt = Context.lowered_for_decisions ?name ctx comp bindings in
  let ms = Option.map ~f:(fun s -> s *. 1e3) in
  let explain ?(ex_admitted = true) ?(ex_anchor = false) segs =
    let ex_operand_loads =
      CM.combine_load_counts (List.map segs ~f:(model_operand_loads ~capabilities:caps))
    in
    match segments_legs ~capabilities:caps ~peak_flops ~peak_memory_bandwidth ~limits segs with
    | None ->
        {
          ex_floor_ms = None;
          ex_rank_ms = None;
          ex_compute_ms = None;
          ex_memory_ms = None;
          ex_spilled = None;
          ex_fills = None;
          ex_mma_work = None;
          ex_device_gathers = None;
          ex_operand_loads;
          ex_admitted;
          ex_anchor;
        }
    | Some (bound, legs) ->
        (* A leg is present on every kernel or on none: its peak is one constant. *)
        let total leg =
          List.fold legs ~init:(Some 0.) ~f:(fun acc l -> Option.map2 acc (leg l) ~f:Float.( + ))
        in
        {
          ex_floor_ms = Some (bound *. 1e3);
          ex_rank_ms = Some (List.fold legs ~init:0. ~f:(fun acc l -> acc +. kernel_rank l) *. 1e3);
          ex_compute_ms = ms (total (fun l -> l.kl_compute));
          ex_memory_ms = ms (total (fun l -> l.kl_memory));
          ex_spilled = Some (List.exists legs ~f:(fun l -> l.kl_spilled));
          ex_fills = Some (List.map legs ~f:(fun l -> l.kl_fill));
          ex_mma_work = Some (List.map legs ~f:(fun l -> l.kl_mma_work));
          ex_device_gathers = Some (List.map legs ~f:(fun l -> l.kl_device_gathers));
          ex_operand_loads;
          ex_admitted;
          ex_anchor;
        }
  in
  (* [model_default]'s early exits: without envelope constants, or with the default annotator off,
     it ranks nothing and compiles the default pipeline. *)
  if
    (Option.is_none peak_flops && Option.is_none peak_memory_bandwidth)
    || not (Sched.automatic_schedule_active ~backend_name:backend)
  then
    {
      x_routine =
        [
          ( "default",
            explain
              (Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices
                 (scratch_of opt)) );
        ];
      x_segments = [];
    }
  else
    (* [model_default]'s incumbent and searched program, as it derives them. *)
    let seeding = model_seeding ~backend ~limits in
    let record = ref [] in
    let incumbent_label, incumbent =
      match seeding with
      | Some (gate, min_serial, sink_enabled, admit_narrow) ->
          ( "seeded",
            seeded_apply
              ~on_choice:(fun c -> record := c.sd_seeded)
              ~capabilities:caps ~backend ~limits ~accum_prec ~static_indices ~gate ~min_serial
              ~sink_enabled ~admit_narrow (scratch_of opt) )
      | None ->
          ( "default",
            Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices
              (scratch_of opt) )
    in
    let base =
      match seeding with
      | Some (_, min_serial, _, _) ->
          fst (seeded_split ~backend ~min_serial ~static_indices (scratch_of opt))
      | None -> opt
    in
    (* The sketch leaves [model_default]'s family search visits on [prog] (a whole routine or one
       segment), in its visiting order. Exactly the leaves it can score: a schedule that fails to
       apply, or whose scheduled form fails validation, is excluded from its ranking too. *)
    let explain_leaves ~label ~admits prog leaves =
      List.filter_map leaves ~f:(fun p ->
          match
            validate_segments_for_model ~fork_join:is_cpu ~limits
              [
                Sched.apply_classified ~static_indices
                  (sketch_schedule ~accum_prec ~p prog)
                  (scratch_of prog);
              ]
          with
          | exception Outcome.Cause_at _ -> None
          | segs -> Some (label p, explain ~ex_admitted:(admits p) segs))
    in
    let record = !record in
    (* gh-ocannl-1477: the leaf [model_default] anchors the walk on [prog] at -- the best-ranked of
       the seeded plan's own sketches there ({!model_seeded_anchors}, from [seeded], the plan's
       record for the units [prog] covers) -- marked [ex_anchor]; listed after the leaves when the
       walk's leaves do not include it. *)
    let leaf_explains ~seeded ~label prog =
      let leaves =
        match sketch_family_tree ~capabilities:caps ~is_gpu ~is_cpu ~limits prog with
        | Some tree ->
            Sspace.leaves
              (if Lazy.force geometry_lattice_enabled then lift_geometry_lattice tree else tree)
        | None -> []
      in
      let listed =
        explain_leaves ~label ~admits:(model_admits ~seeding ~accum_prec prog) prog leaves
      in
      let rank (_, ex) = Option.value ex.ex_rank_ms ~default:Float.infinity in
      let anchors =
        List.map (model_seeded_anchors ~limits ~accum_prec ~static_indices ~seeded prog)
          ~f:(fun (p, segs) -> (label p, explain ~ex_anchor:true segs))
      in
      match
        List.min_elt
          (List.filter anchors ~f:(fun (_, ex) -> Option.is_some ex.ex_rank_ms))
          ~compare:(fun a b -> Float.compare (rank a) (rank b))
      with
      | None -> listed
      | Some ((anchor, _) as row) ->
          if List.exists listed ~f:(fun (l, _) -> String.equal l anchor) then
            List.map listed ~f:(fun (l, ex) ->
                if String.equal l anchor then (l, { ex with ex_anchor = true }) else (l, ex))
          else listed @ [ row ]
    in
    (* gh-ocannl-1278: the coalesced tuner branch's whole-routine leaves, which [model_default] does
       not rank, listed on request to check the score's ranking of them against timings. A leaf is
       marked admitted when its uncoalesced form is: the set [model_default] would rank were the
       coalesced layouts admitted as numerics-preserving (neither reorders a cell's summation). *)
    let coalesced_explains prog =
      if not coalesced then []
      else
        let admits = model_admits ~seeding ~accum_prec prog in
        explain_leaves
          ~label:(fun p -> spec_label (Whole (W_sketch p)))
          ~admits:(fun p -> admits { p with sk_coalesce = None })
          prog
          (coalesced_seed_params ~capabilities:caps ~is_gpu ~is_cpu ~limits prog)
    in
    (* gh-ocannl-1438: [model_default]'s per-segment substitution, as it derives it
       ({!model_segments}: the search's fold-zeros segmentation of the searched program, when it
       splits the routine or exposes a changed lowering, gh-ocannl-1512) -- one entry per keyed
       segment: the segment under the search's preset, the incumbent its sketches must strictly
       beat, then the sketch leaves it searches there. *)
    let x_segments =
      match model_segments ~is_gpu ~is_cpu ~limits ~static_indices base with
      | exception Outcome.Cause_at _ -> []
      | None -> []
      | Some (plan, keyed) ->
          List.map keyed ~f:(fun (key, pre, post) ->
              let tag = seg_key_tag key in
              {
                xs_segment = tag;
                xs_candidates =
                  ("preset", explain [ post ])
                  :: leaf_explains
                       ~seeded:(seeded_record_for_key ~plan ~record key)
                       ~label:(fun p -> sketch_params_label p ^ " @" ^ tag)
                       pre;
              })
    in
    {
      x_routine =
        (incumbent_label, explain incumbent)
        :: (leaf_explains ~seeded:(List.map record ~f:snd)
              ~label:(fun p -> spec_label (Whole (W_sketch p)))
              base
           @ coalesced_explains base);
      x_segments;
    }

let untuned_compile ?name ctx comp bindings =
  if Lazy.force model_default_enabled then model_default ?name ctx comp bindings
  else seeded_default ?name ctx comp bindings

(** {2 The search} *)

(* gh-ocannl-550: the containment properties of the search — a failed candidate costs that
   candidate, a failed search costs that search and not its sibling arm — are only testable with a
   candidate that fails, and the reproduction that motivated them needs a 12 GB GPU and a half-hour
   search. This seam manufactures the failure instead. It is called with the candidate's label
   before each candidate compile; raising from it emulates the shape the device OOM had, a failure
   that is NOT contained as a candidate decline (there it escaped after the search had concluded,
   when the exhausted device defeated both the winner replay and its untuned fallback). Not a
   production seam: default no-op, and no config key selects it. Called for the baseline compile too
   — it is a candidate (gh-ocannl-533) — which is what makes a failure BEFORE the search has
   reported anything injectable, the case the positional-arm-slot handling in
   [Train.tune_placements] exists for. *)
let on_candidate_attempt : (string -> unit) Seam.t = Seam.make (fun _label -> ())

(* gh-ocannl-1409: the elapsed seconds a budget check reads, given the call's real elapsed seconds.
   A wall clock cannot place a cut at a chosen candidate, and a test of what a cut search attempts,
   crowns and stores needs exactly that: it substitutes a virtual clock (the attempts so far, say).
   Not a production seam: default the identity, and no config key selects it. *)
let budget_elapsed : (float -> float) Seam.t = Seam.make Fn.id

(* gh-ocannl-1440: a search's share of an enclosing call's budget ([Train.tune_placements]'
   [tune_call_budget_seconds]). [cb_share_s] may be zero or negative, a call budget already spent:
   the search then attempts only its untuned floors. *)
type call_budget = { cb_total_s : float; cb_share_s : float }

(* gh-ocannl-1440: the elapsed seconds a call budget's split reads, given the call's real elapsed
   seconds -- [budget_elapsed]'s counterpart one level up, read by [Train.tune_placements]. Not a
   production seam: default the identity, and no config key selects it. *)
let call_budget_elapsed : (float -> float) Seam.t = Seam.make Fn.id

(* Parsed by [autotune_budget_seconds]' grammar, but not through its typed getter: the typed getters
   serve the keys an identity renders ([Utils.config_key_defaults]), and this one is
   [Placement_shaping], which no identity renders. *)
let call_budget_setting () =
  let value = Utils.get_global_arg ~arg_name:"tune_call_budget_seconds" ~default:"" in
  match Utils.canonical_config_value Utils.Positive_float_or_unset value with
  | Some "" -> None
  | Some positive -> Some (Float.of_string positive)
  | None ->
      raise
        (Utils.Invalid_config_value
           {
             key = "tune_call_budget_seconds";
             value;
             expected = "a number, or nothing to leave it unset";
           })

(* gh-ocannl-1138: the running composite a staffed sweep starts from. Called once per search that
   sweeps, with the coarse keys and their seeded parameters, and the starting composite derived from
   the seeded untuned plan; its result is where the sweep starts. On cc no seeded plan exists, so
   the composite starts empty there, and only this seam can give a key a choice before its own sweep
   -- the case where a slower replacement must leave the composite as it was. Not a production seam:
   default the identity, and no config key selects it. *)
let on_staffed_start :
    (menu:(seg_key * sketch_params list) list ->
    (seg_key * sketch_params) list ->
    (seg_key * sketch_params) list)
    Seam.t =
  Seam.make (fun ~menu:_ start -> start)

(* gh-ocannl-1441: the seeded untuned plan [tune] derived on its base lowering (forced lazily, at
   most once per call), which the search then seeds first, falls back to and names in its trajectory
   identity ([|seed=<digest>]). A test substitutes another plan -- [None] included -- to run two
   seeds against one trajectory: no setting moves the plan alone, since the keys that steer it are
   [Search_shaping] and render in the trajectory's settings themselves, and the [Untimed] device
   readings its [auto] gate consults are fixed per process. Not a production seam: default the
   identity, and no config key selects it. *)
let on_seeded_plan : (seeded_plan option -> seeded_plan option) Seam.t = Seam.make Fn.id
let search_setting () = Utils.registered_flag ~arg_name:"autotune_search"

(* The schedule cache directory a tuning call uses, or [""] for none. Whether the directory was
   CHOSEN, as opposed to being the built-in default -- passed by the caller, or set at some config
   source (a profile payload included) -- is only relevant with the search off, where it is the
   difference between replaying a cache someone committed and replaying whatever an earlier local
   search happened to leave in ./autotune_cache (gh-ocannl-559; Codex P2 on PR #291): the latter
   would make two reproducible runs differ on local state, which is the leak that turning the search
   off exists to close. Shared with [Train.tune_placements], whose placement-decision store lives in
   the same directory (gh-ocannl-786) and must open or not open it by the same rule. *)
let resolve_cache_dir ?cache_dir ~search () =
  let cache_dir_chosen =
    Option.is_some cache_dir
    ||
    match snd (Utils.get_global_arg_with_source ~arg_name:"autotune_cache_dir" ~default:"") with
    | Utils.From_default -> false
    | _ -> true
  in
  let cache_dir =
    Option.value cache_dir
      ~default:(Utils.get_global_arg ~arg_name:"autotune_cache_dir" ~default:"autotune_cache")
  in
  (* A search-less [tune] replays only a cache someone asked for. *)
  if search || cache_dir_chosen then cache_dir else ""

(* gh-ocannl-1421: how a lowering's code accesses each tensor node -- ["r"], ["w"], ["rw"], or ["-"]
   for not at all. Stable under the drift between two lowerings of one lineage (folding a gather
   guard removes no accessed node), and per node, so a caller can tag nodes with it inside a
   canonically numbered key. *)
let lineage_accesses (opt : LL.optimized) =
  let table = Hashtbl.create (module Ir.Tnode) in
  List.iter (LL.affine_accesses opt.LL.llc) ~f:(fun (a : Ir.Tnode.t Ir.Affine.access) ->
      Hashtbl.update table a.a_tn ~f:(fun prev ->
          let r, w = Option.value prev ~default:(false, false) in
          if a.a_write then (r, true) else (true, w)));
  fun tn ->
    match Hashtbl.find table tn with
    | None -> "-"
    | Some (r, w) -> (if r then "r" else "") ^ if w then "w" else ""

(* gh-ocannl-1421: the identity two lowerings of the routine, from two lineages, must share for a
   schedule tuned on the one to apply to the other: the code, with loop binders canonically numbered
   and every tensor node anchored to its actual identity -- the canonical form alone alpha-renames
   nodes, so [u := x] against [u := y], or [u := x - y] against [u := y - x], would compare equal
   (review rounds 6-7). Process-local (node ids): for comparing two lowerings in one call, never for
   a persisted key. The two are lowered together, at the same moment: the code drifts between
   lowerings of one lineage taken at different times (a gather guard folds once a compile has
   settled its index tensor's bounds, so a lowering taken after the base compile folds what the base
   kept), but two lowerings taken together see the same bounds. *)
let lineage_identity ~static_indices (opt : LL.optimized) =
  SC.digest
    (SC.canonicalize_code ~static_indices ~node_tag:(fun tn -> "#" ^ Ir.Tnode.id tn) opt.LL.llc)

(* gh-ocannl-1621: the search lineage's training state, restored before every timing window. Every
   candidate compiles from the search lineage, so a node that lineage already holds a buffer for is
   ONE buffer every candidate's runs write -- parameters, optimizer state, counters -- while a node
   it lacks is allocated afresh in each candidate's own compile (its link-time initialization
   re-runs), so it cannot carry one candidate's runs into the next. The state that drifts across a
   search is therefore exactly what the computation writes among the buffers the lineage holds: the
   base lowering's written nodes ([has_assignment] or [zeroed_out]; a hosted constant's writes are
   gone from the code) present in [search_ctx], a written slice ALIAS standing for its parent, whose
   buffer it writes and which host access goes through. A node the lineage placed [Local] has no
   observable buffer and holds no state between runs. Read-only nodes (the data batch, an uploaded
   dataset) are never written, so they are not copied. *)
let timing_state_nodes search_ctx (opt : LL.optimized) =
  let placements = Context.placements search_ctx in
  Hashtbl.fold opt.LL.traced_store
    ~init:(Set.empty (module Ir.Tnode))
    ~f:(fun ~key:_ ~data:(traced : LL.traced_array) acc ->
      if not (traced.has_assignment || traced.zeroed_out) then acc
      else
        (* A confirmed slice alias owns no buffer: its writes land in its parent's. An ineligible
           (copy) slice owns its own buffer, recomputed from its parent on every run -- no state,
           and host access refuses it -- so it stands for nothing here; mapping it to its parent
           would copy a read-only dataset in and out per window. *)
        let tn =
          match (Ir.Tnode.alias_of traced.tn, Ir.Tnode.slice_of traced.tn) with
          | Some (parent, _), _ -> Some parent
          | None, Some _ -> None
          | None, None -> Some traced.tn
        in
        Option.fold tn ~init:acc ~f:(fun acc tn ->
            let local =
              match Ir.Tnode.Placements.get placements tn with
              | Some (Ir.Tnode.Local, _) -> true
              | _ -> false
            in
            if local || not (Context.mem search_ctx tn) then acc else Set.add acc tn))

(* Host copies, not a second device buffer per node: a device snapshot would keep a second working
   set of the state resident through the search, against the one-live-candidate memory bound
   (gh-ocannl-1326), while a copy per candidate is noise beside the candidate's compile. *)
let snapshot_timing_state search_ctx opt =
  List.map
    (Set.to_list (timing_state_nodes search_ctx opt))
    ~f:(fun tn -> (tn, Context.to_host search_ctx tn))

(* Through the lineage's own handle: the bytes land in the buffer every candidate binds, and the
   ledger records an external write, as a caller's [Context.set_values] before a run would. The
   returned context is dropped -- the search keeps compiling from [search_ctx], whose nodes are all
   initialized already. *)
let restore_timing_state search_ctx snapshot =
  List.iter snapshot ~f:(fun (tn, nd) -> ignore (Context.from_host search_ctx tn nd : Context.t))

(* How a search's end restores its lineage (gh-ocannl-1621): [settle] is set once the search has
   taken its snapshot, and [with_timing_state_restored] calls it on the way out of the call. *)
type timing_state = { mutable settle : (after_exn:bool -> unit) option }

(* The search leaves every buffer it found as it found it, however it ends: completed, budget-cut,
   abandoned ([Search_abandoned]) or failed -- unless the lineage is poisoned, which is unusable by
   contract and which a search can return over (a contained candidate failure on a [timing_ctx]). So
   the next search of the lineage -- the other arm and the flips of [Train.tune_placements], whose
   lineages share these buffers -- starts from the caller's values, and a search without a
   [timing_ctx] does not train the caller's context. After an exception the restore is best effort:
   never after a process-fatal exception, and a failure of the restore itself is logged, never
   raised over the exception being propagated. *)
let with_timing_state_restored f =
  let state = { settle = None } in
  match f state with
  | result ->
      Option.iter state.settle ~f:(fun settle -> settle ~after_exn:false);
      result
  | exception exn when not (process_fatal_exn exn) ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      Option.iter state.settle ~f:(fun settle -> settle ~after_exn:true);
      Stdlib.Printexc.raise_with_backtrace exn backtrace

(* What a search hands back: the compile of what it ships, as a thunk. Without [defer] the thunk is
   already forced (it returns the routine [tune] compiled before reporting); with it
   (gh-ocannl-1408, {!tune_deferred}) the search has reported, released everything it compiled, and
   the thunk compiles the shipped schedule on first call. [substituted] is set when that compile
   ships the untuned default in place of the schedule the report named (a winner or cache replay
   that failed or degenerated), which a deferred caller ranking by the report must hear about. *)
let tune_in_stream_scope ~timing_state ~defer ~substituted ?name ?search ?beam_width ?rounds
    ?repeats ?timing ?seed_block_sizes ?cache_dir ?keep_fraction ?max_split_reduce_sites ?timing_ctx
    ?abandon ?call_budget ?report ctx comp bindings : unit -> Context.t * Context.routine =
  (* gh-ocannl-559: with the search off, [tune] still replays an explicitly provided cache -- a
     pinned schedule is deterministic, and committing one is how a reproducible run keeps a tuned
     schedule -- but never times candidates, whose crowning is the largest cross-machine determinism
     leak. A miss compiles the untuned default pipeline, exactly like the nothing-was-timed fallback
     below. *)
  let search = Option.value_or_thunk search ~default:search_setting in
  let beam_width =
    max 1
      (Option.value_or_thunk beam_width ~default:(fun () ->
           Utils.registered_int ~arg_name:"autotune_beam_width"))
  in
  let rounds =
    Option.value_or_thunk rounds ~default:(fun () ->
        Utils.registered_int ~arg_name:"autotune_rounds")
  in
  let repeats =
    Option.value_or_thunk repeats ~default:(fun () ->
        Utils.registered_int ~arg_name:"autotune_repeats")
  in
  (* Not [Option.value ~default:…] (Codex P2 on PR #512): that evaluates the default eagerly, so an
     unparsable ambient [autotune_timing] would be rejected even for a caller that passed its own
     mode — and for a [~search:false] call, which times nothing at all. The settings above read in
     [Option.value_or_thunk] for the same reason: each refuses a value that does not parse
     (gh-ocannl-1332), so the read has to be in the branch that needs it. *)
  let timing =
    match timing with
    | Some t -> t
    | None ->
        timing_of_setting @@ Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued"
  in
  (* Every report this call emits starts here, so the objective its times were taken under travels
     with them (gh-ocannl-755): a consumer comparing a [best_ms] across processes, or storing one in
     a benchmark artifact, otherwise has to guess it from ambient configuration that a [?timing]
     override may not match. Which is why the objective is resolved above this line and not below
     it. *)
  (* [budget_s] reports the per-search setting, a fact about the call that a shrinking share must not
     replace; [budget] is what the search is cut at: the tighter of the setting and the share of an
     enclosing call's budget (gh-ocannl-1440). *)
  let budget_s = budget_setting () in
  let budget =
    match (budget_s, call_budget) with
    | b, None -> b
    | None, Some cb -> Some (Float.max 0. cb.cb_share_s)
    | Some b, Some cb -> Some (Float.min b (Float.max 0. cb.cb_share_s))
  in
  let call_budget_s = Option.map call_budget ~f:(fun cb -> cb.cb_total_s) in
  let call_share_s = Option.map call_budget ~f:(fun cb -> cb.cb_share_s) in
  let base_report =
    { (no_search_report ~timing) with beam_width; budget_s; call_budget_s; call_share_s }
  in
  (* gh-ocannl-1440: the backend compiles this call completes, stamped on every report it emits. *)
  let compiles_at_start = Context.backend_compiles () in
  let replay_compiles = ref 0 in
  let max_split_reduce_sites =
    max 0
      (Option.value_or_thunk max_split_reduce_sites ~default:(fun () ->
           Utils.registered_int ~arg_name:"autotune_split_reduce_max_sites"))
  in
  let staffed_singles = staffed_singles_setting () in
  let playoff_width = max 0 (Utils.registered_int ~arg_name:"autotune_composite_playoff_width") in
  (* A negative margin admits no single and so means what width 0 says; it is floored at 0 (exact
     ties only) rather than given a second spelling of off. A malformed value, NaN included, is
     refused (gh-ocannl-1332). *)
  let playoff_margin =
    Float.max 0. (Utils.registered_float ~arg_name:"autotune_composite_playoff_margin")
  in
  (* gh-ocannl-1635: the beam finals' margin ([finals_contenders]); 0 or less turns them off. A
     malformed value, NaN included, is refused. *)
  let finals_margin = Utils.registered_float ~arg_name:"autotune_finals_margin" in
  (* Read only where a round can run, so a malformed value refuses the searches it would shape and
     no other call (like [autotune_timing] above). *)
  let round_expansion =
    if search && rounds > 0 then
      Utils.registered_keyword ~arg_name:"autotune_round_expansion"
        ~expected:"differing, full or head" ~parse:round_expansion_of_setting
    else Expand_differing
  in
  let seed_block_sizes = Option.value seed_block_sizes ~default:[ 64; 128; 256; 512 ] in
  let cache_dir = resolve_cache_dir ?cache_dir ~search () in
  let keep_fraction =
    Option.value_or_thunk keep_fraction ~default:(fun () ->
        Utils.registered_float ~arg_name:"autotune_keep_fraction")
  in
  let static_indices = Idx.bound_symbols bindings in
  let backend = Context.backend_name ctx in
  let device = Context.ordinal ctx in
  (* The tuned computation's name, for the calibration rows and log lines of every candidate
     (gh-ocannl-635). READ from [name] rather than re-derived (gh-ocannl-669): every compile below
     is passed the same [?name], and this is the same [Option.value_or_thunk ... get_name_exn] they
     resolve it by ([Backends.lower_assignments]), so a row names the code exactly the way its
     generated sources and debug artifacts are named — by construction now, rather than by the
     coincidence that no compile here passed a name. Lazy and total on purpose: this is a diagnostic
     label, while the "a comp must be named" contract belongs to the compiles, and deriving it
     eagerly would move that failure ahead of them (and impose it on a search that emits
     nothing). *)
  let routine_name =
    lazy
      (Option.value_or_thunk name ~default:(fun () ->
           match Ir.Assignments.get_name_exn comp.Ir.Assignments.asgns with
           | derived -> derived
           | exception Invalid_argument _ -> ""))
  in
  (* gh-ocannl-1061: this call's progress state. [progress_compile_s] and [progress_timing_s]
     accumulate the candidates' compiles and timing windows, so [elapsed_s] minus the two is what
     the search spent elsewhere (the base compile, analyses, replays, the winner's recompile). *)
  let progress_clock = Mtime_clock.counter () in
  let progress_compile_s = ref 0. and progress_timing_s = ref 0. in
  let progress_attempts = ref 0 in
  let progress_line event fields =
    if progress_enabled () then
      let elapsed = seconds_since progress_clock in
      progressf "event=%s routine=%S elapsed_s=%.1f %s" event (Lazy.force routine_name) elapsed
        fields
  in
  let progress_costs () =
    Printf.sprintf "attempts=%d compile_s=%.1f timing_s=%.1f" !progress_attempts !progress_compile_s
      !progress_timing_s
  in
  (* A line BEFORE every step outside the candidates that can block for long -- the base compile, a
     cache replay, the baseline's timing window, the seed enumeration's lowerings, the winner's
     recompile, an untuned fallback -- for the same reason a candidate's line precedes it: a search
     killed inside the step is inside the one its last line names. *)
  (* The stage the last [stage] line named: the stage of a compile made outside the candidates. *)
  let current_stage = ref "base_compile" in
  let progress_stage stage =
    current_stage := stage;
    progress_line "stage" (Printf.sprintf "stage=%s %s" stage (progress_costs ()))
  in
  let timed_into acc f =
    let c = Mtime_clock.counter () in
    Exn.protect ~f ~finally:(fun () -> acc := !acc +. seconds_since c)
  in
  (* gh-ocannl-1314: every compile this call makes, by phase, for [report.compile_cost]; and each
     one's record as it returns: a [compiled] progress event, an [autotune_log] line and the
     domain's compile observers. An exception is recorded as [failed] and re-raised unchanged. *)
  let compile_cost_total = ref Ir.Compile_cost.empty in
  let costed ?stage ~attempt ~verdict_of f =
    let stage = Option.value stage ~default:!current_stage in
    let c = Mtime_clock.counter () in
    let result, cost =
      Ir.Compile_cost.with_census (fun () ->
          match f () with
          | r -> Ok r
          | exception exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ()))
    in
    let wall_s = seconds_since c in
    let verdict = match result with Ok r -> verdict_of r | Error _ -> "failed" in
    compile_cost_total := Ir.Compile_cost.add !compile_cost_total cost;
    let device = Option.value_map cost.device_s ~default:"none" ~f:(Printf.sprintf "%.3f") in
    progress_line "compiled"
      (Printf.sprintf
         "stage=%s attempt=%S verdict=%s took_s=%.3f lower_s=%.3f transform_s=%.3f codegen_s=%.3f \
          device_s=%s link_s=%.3f kernels=%d kernels_compiled=%d"
         stage attempt verdict wall_s cost.lower_s cost.transform_s (Ir.Compile_cost.codegen_s cost)
         device cost.link_s cost.kernels cost.kernels_compiled);
    logf
      "%s: compile %s in %.3f s (lower %.3f, transform %.3f, codegen %.3f, device %s, link %.3f; \
       %d of %d kernels compiled)"
      attempt verdict wall_s cost.lower_s cost.transform_s (Ir.Compile_cost.codegen_s cost) device
      cost.link_s cost.kernels_compiled cost.kernels;
    (* Trace-only: each observer is contained where it is installed ([install_compile_observer]), so
       an observer that raises cannot fail a compiled candidate nor strand its context. *)
    Stdlib.Domain.DLS.get compile_observer
      {
        compile_routine = Lazy.force routine_name;
        compile_stage = stage;
        compile_attempt = attempt;
        compile_verdict = verdict;
        compile_wall_s = wall_s;
        compile_cost = cost;
      };
    match result with
    | Ok r -> r
    | Error (exn, backtrace) -> Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  let outcome_verdict = function
    | Ok _ -> "compiled"
    | Error failure when is_not_dispatched failure -> "refused"
    | Error _ -> "failed"
  in
  progress_line "search_start"
    (let opt = Option.value_map ~default:"none" ~f:(Printf.sprintf "%g") in
     Printf.sprintf
       "backend=%s device=%d search=%b beam=%d rounds=%d budget_s=%s call_budget_s=%s \
        call_share_s=%s"
       backend device search beam_width rounds (opt budget_s) (opt call_budget_s) (opt call_share_s));
  (* gh-ocannl-1408: set once a deferred call has delivered its report. The deferred compile runs
     after that, and its failure paths ([raise_pre_search], [emit_partial_and_raise]) would
     otherwise report a second time. *)
  let sealed = ref false in
  (* Every report goes through here, exactly once per call on every path (gh-ocannl-550), so this is
     where a search's closing progress line is written: before the callback, which may raise. *)
  let emit_report (r : report) =
    let r =
      {
        r with
        compiles = Context.backend_compiles () - compiles_at_start;
        replay_compiles = !replay_compiles;
        compile_cost = !compile_cost_total;
      }
    in
    if not !sealed then (
      progress_line "search_done"
        (Printf.sprintf
           "outcome=%s timed=%d contended=%d timings_retried=%d unbatched=%d unreadable=%d \
            dispersed=%d hopeless=%d failed=%d rounds=%d budget_cut=%b baseline_refused=%b \
            privatize_dropped=%d %s best_ms=%s best=%S"
           (outcome_name r.outcome) r.candidates_timed r.timings_contended r.timings_retried
           r.timings_unbatched r.timings_unreadable r.timings_dispersed r.timings_hopeless
           r.candidates_failed r.rounds_run r.budget_cut
           (Option.is_some r.baseline_refusal)
           (List.sum (module Int) r.privatize_drops ~f:(fun d -> d.count))
           (progress_costs ()) (progress_ms r.best_ms) r.best_label);
      Option.iter report ~f:(fun f -> f r))
  in
  (* [tune] reports exactly once per call, on every path (gh-ocannl-550). The failures that happen
     before (or instead of) the search proper — the base compile failing before its lowering is
     captured, a fatal baseline link, a fatal cache replay, a baseline timing failure, and either
     untuned fallback compile of a search-less call — used to raise with no report at all, which
     leaves a caller that attributes arms by arrival order (the positional [?report] of
     [Train.tune_placements]) with no slot for this search. The phase reported is the one the
     failure itself carries, so the diagnostic names where it actually died — at codegen, at link,
     at launch, at sync — instead of guessing. Reporting is best-effort here, as on the search's own
     fatal path: it must not replace the compiler failure. [base] carries whatever the call did
     learn before failing (e.g. a decline census). *)
  let n_timings_retried = ref 0 in
  let on_retry () = Int.incr n_timings_retried in
  (* gh-ocannl-1263: how the search's timing calls settled and were admitted, from each call's own
     record through the per-call [on_call] -- the baseline's and every candidate's, never the
     diagnostic untuned control's -- so the counts belong to this call whatever else observes, and
     whatever runs in another domain. *)
  let decision_counts = Hashtbl.Poly.create () in
  let on_call (c : timing_call) =
    Hashtbl.incr decision_counts (c.decision.settle, c.decision.admission)
  in
  let timing_decisions () =
    Hashtbl.to_alist decision_counts
    |> List.sort ~compare:(fun (a, _) (b, _) -> Poly.compare a b)
    |> List.map ~f:(fun ((settle, admission), calls) -> (settle, admission, calls))
  in
  let emit_pre_search_failure ?(base = base_report) ~phase ~candidate ~detail () =
    let r =
      {
        base with
        outcome = Pre_search_failure { phase; candidate; detail };
        timings_retried = !n_timings_retried;
        timing_decisions = timing_decisions ();
      }
    in
    try emit_report r
    with report_exn when not (process_fatal_exn report_exn) ->
      Stdio.eprintf "autotune: pre-search failure report callback failed: %s\n%!"
        (Exn.to_string report_exn)
  in
  (* gh-ocannl-550: every [raise_pre_search] leaves [tune] without returning a routine, so the base
     compile's artifact is dead on all of them — but the base is linked further down, after this is
     defined, so the release action arrives by hook rather than by reference. A hook rather than a
     call at each raise site on purpose: the previous rounds of this work fixed such sites one at a
     time and each new one was a fresh leak (the fatal cache replay was the last of them), whereas a
     family with one member cannot be partially updated. Harmless where nothing is linked yet — the
     two raises above the base compile invoke the no-op default. *)
  let release_baseline_hook = ref (fun () -> ()) in
  (* The ONE way this function releases anything (gh-ocannl-550). Releasing is best-effort
     everywhere: it runs on failure paths where the device may already be refusing work, and a
     failure to give memory back must never replace the outcome the caller has to act on.
     Process-fatal conditions still propagate. A helper rather than the ad-hoc guard this started
     as, because "is this call wrapped?" produced its own review finding once already. *)
  let release_quietly ~what ctx =
    try Context.release ctx
    with exn when not (process_fatal_exn exn) ->
      logf "release of %s failed: %s" what (Exn.to_string exn)
  in
  (* [emit_report] on a path that then hands a routine back: the callback's exception propagates by
     design, so the caller never receives [result] and its buffers become unreachable while the pool
     table goes on rooting them. Every site that reports a compiled routine reports through this. *)
  let report_or_release r ~result =
    match emit_report r with
    | () -> ()
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        release_quietly ~what:"the routine of a failed completion report" (fst result);
        Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  (* gh-ocannl-1408: the deferred compile of a {!tune_deferred} call. The report has been delivered
     and sealed, so the compile's own failure paths raise without a second one; the streams this
     call ran under are re-entered, since the compile runs after the call's scope has closed; and
     the compile happens at most once, a second call returning the same routine. *)
  let deferred_compile compile =
    sealed := true;
    let log = log_enabled () and progress = progress_enabled () in
    let compiled = ref None in
    let outcome () =
      match !compiled with
      | Some (Ok result) -> result
      | Some (Error (exn, backtrace)) -> Stdlib.Printexc.raise_with_backtrace exn backtrace
      | None -> invalid_arg "Autotune.compile_deferred: no outcome recorded"
    in
    fun () ->
      if Option.is_some !compiled then outcome ()
      else
        (* The search's own compile, so it holds the search like the search did (gh-ocannl-1352) --
           only where it runs: a memoized outcome is returned unguarded. A failure is kept too: the
           compile runs once, and a caller catching its exception and asking again gets the same
           exception, not a second attempt's side effects. *)
        match
          exclusive_search (fun () ->
              (* Rechecked under the guard: another domain may have compiled between the read above
                 and the acquisition. *)
              if Option.is_none !compiled then
                compiled :=
                  Some
                    (match with_streams ~log ~progress compile with
                    | result -> Ok result
                    | exception exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ())))
        with
        | () -> outcome ()
        (* The guard refused before anything ran: not an outcome, so a later turn compiles. *)
        | exception (Concurrent_search _ as exn) -> raise exn
  in
  (* The end of every path that ships a routine: [compile] produces it. Eagerly, it is compiled
     first and reported after -- a report is a record of what the call achieved, and a compile that
     raises reports its own failure instead (gh-ocannl-550). Deferred, the report comes first and
     the compile is handed back. *)
  let finish r compile =
    if defer then (
      emit_report r;
      deferred_compile compile)
    else
      let result = compile () in
      report_or_release r ~result;
      Fn.const result
  in
  let raise_pre_search ?base (failure : Outcome.failure) =
    !release_baseline_hook ();
    (match failure with
    | Outcome.Classified c ->
        emit_pre_search_failure ?base ~phase:c.Outcome.phase ~candidate:None
          ~detail:(Outcome.detail_of_cause c.Outcome.cause)
          ()
    | Outcome.Fatal f ->
        emit_pre_search_failure ?base ~phase:f.Outcome.phase ~candidate:f.Outcome.candidate
          ~detail:(Exn.to_string f.Outcome.exn) ());
    Outcome.raise_failure failure
  in
  (* The untuned fallback of a call that does not search (search off, a lineage refusal, a deferred
     cache replay that failed), through the containment-aware form so a failure reports the phase it
     carries. It is {!seeded_default}'s compile, the recipes' untuned step, wherever
     {!seeded_settings} admits the seeded plan (CUDA and HIP; gh-ocannl-1441) -- the plan derived
     inside this compile's own transform, on [ctx]'s lowering -- and the plain default pipeline
     elsewhere. As in {!seeded_default}, a classified failure of a compile that seeded, split or
     sank something recompiles the plain default; any other failure goes to [raise_pre_search].
     [Context.compile] is the plain form plus [raise_failure], which is what [raise_pre_search] ends
     with, so the caller sees the same exception either way. *)
  let compile_untuned_default ?base () =
    progress_stage "untuned_default_compile";
    let plain () =
      match
        costed ~attempt:"untuned default" ~verdict_of:outcome_verdict (fun () ->
            Context.compile_outcome ?name ~provenance:Ir.Schedule_outcome.User_schedule ctx comp
              bindings)
      with
      | Ok result -> result
      | Error failure -> raise_pre_search ?base failure
    in
    let choice = ref no_seeded_choice in
    (* Before the compile, so outside its classification: on CUDA and HIP this reads the device's
       limits (device work, as the search path's limits query is) and the seeding settings, either
       of which can raise, and the call must still report once. *)
    let seeded =
      match seeded_transform ~on_choice:(fun c -> choice := c) ctx bindings with
      | transform -> transform
      | exception exn when not (process_fatal_exn exn) ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          !release_baseline_hook ();
          let phase =
            match exn with
            | Utils.Invalid_config_value _ -> Outcome.Transform
            | _ -> Outcome.Hardware_limits
          in
          emit_pre_search_failure ?base ~phase ~candidate:None ~detail:(Exn.to_string exn) ();
          Stdlib.Printexc.raise_with_backtrace exn backtrace
    in
    match seeded with
    | None -> plain ()
    | Some lowered_transform -> (
        match
          costed ~attempt:"seeded untuned default" ~verdict_of:outcome_verdict (fun () ->
              Context.compile_outcome ?name ~lowered_transform ~provenance:Outcome.Advisory ctx comp
                bindings)
        with
        | Ok result -> result
        | Error (Outcome.Classified classified)
          when not
                 (List.is_empty !choice.sd_seeded && List.is_empty !choice.sd_split
                && List.is_empty !choice.sd_sunk) ->
            logf "the seeded untuned default FAILED (%s); compiling the plain default"
              (Outcome.detail_of_cause classified.cause);
            plain ()
        | Error failure -> raise_pre_search ?base failure)
  in
  (* Without a cache to replay there is nothing for a search-less [tune] to do, so it does not even
     take the base compile that computes the cache key: the caller gets the untuned default compile
     it would have gotten from {!untuned_compile} without [model_default_schedule]. *)
  if (not search) && String.is_empty cache_dir then (
    logf
      "search disabled (autotune_search=false) and no chosen cache: compiling the untuned default";
    (* Report AFTER the fallback compile ([finish]): a report is a record of what this call
       achieved, and the base report says the untuned default shipped. Emitting it first would leave
       a consumer holding a clean, non-partial report for a call that then raised (Codex P2 on PR
       #291); a compile that raises reports its own failure instead (gh-ocannl-550). *)
    finish base_report (fun () -> compile_untuned_default ()))
  else
    let is_gpu = Sched.backend_is_gpu backend and is_cpu = Sched.backend_is_cpu backend in
    (* With [timing_ctx], the search (candidate compiles and timing runs) happens against that
       scratch lineage's buffers, and only the winner is compiled from [ctx] — so the timing runs
       never mutate the caller's live state (parameters, accumulators). The scratch context must
       contain the nodes the computation requires from a prior context (e.g. initialized
       parameters), typically by repeating the caller's initialization on a fresh root context. It
       must live on the same backend and device as [ctx] (Codex P2 on PR #109): candidates timed
       elsewhere do not predict this device, and the winner would be cached under this backend's key
       without ever having been timed on it. *)
    Option.iter timing_ctx ~f:(fun tctx ->
        if
          (not (String.equal (Context.backend_name tctx) backend))
          || Context.ordinal tctx <> Context.ordinal ctx
        then
          invalid_arg
            (Printf.sprintf
               "Autotune.tune: timing_ctx must be on the same backend and device as the target \
                context (timing: %s device %d, target: %s device %d)"
               (Context.backend_name tctx) (Context.ordinal tctx) backend (Context.ordinal ctx)));

    (* Device work, not a pure query: the GPU backends lazily initialize the device and read driver
       attributes here, so a driver or enumeration error surfaces at this line — the first thing
       this call does that can fail, and squarely inside the reporting contract. *)
    let limits =
      match Context.hardware_limits ctx with
      | limits -> limits
      | exception exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          emit_pre_search_failure ~phase:Outcome.Hardware_limits ~candidate:None
            ~detail:(Exn.to_string exn) ();
          Stdlib.Printexc.raise_with_backtrace exn backtrace
    in
    (* gh-ocannl-1110: what shapes which candidates this search times, and in what order, that the
       schedule cache's key does not carry: every [Search_shaping] configuration key (the class the
       key leaves out by definition), plus every argument of this function that resolves one — each
       resolved value, whether passed or read. A cached [best_steps] replays only under the same
       shape: the flip chain compares at equal depth, which presumes the same candidate order and
       sampling. The keys that steer the placement level above this call ([Utils.Placement_shaping]:
       the flip budget and ordering, the shipped arm, the placement store) are not in it: no search
       reads them, and with them in it a warm chain run under [tune_placement_store=false] searched
       every abandoned flip in full (gh-ocannl-1192). Nor are the keys of the untuned regime's model
       pick ([Utils.Untimed_model]: [model_default_*]), which only {!model_default} reads
       (gh-ocannl-1241). A key renders only when its resolved value is not its default, as this
       device resolves the default (gh-ocannl-1272): stating a key at its default must not cost a
       warm chain its records. Hence computed here, once [limits] is known: the cache key the
       trajectory is stored under hashes the limits record's [Key]-class projection, so two
       processes sharing it resolve every [Key]-derived [auto] alike. The one input an [Untimed]
       field steers -- the seeded plan's [auto] gate (gh-ocannl-1253) -- joins below as the plan it
       resolved to, once the base lowering exists ([search_shape]). *)
    let search_shape_settings =
      Printf.sprintf "search=%b beam=%d rounds=%d repeats=%d keep=%h split_sites=%d blocks=%s|%s"
        search beam_width rounds repeats keep_fraction max_split_reduce_sites
        (String.concat ~sep:"," (List.map seed_block_sizes ~f:Int.to_string))
        (search_shaping_identity ~backend ~limits)
    in
    (* The accumulator residency the seeded [Privatize] tiles are minted at: the same resolution the
       backend's code generation widens the serial rendering by (gh-ocannl-1116). *)
    let capabilities = Context.codegen_capabilities ctx in
    let accum_prec = capabilities.Ir.Backend_intf.accum_prec in
    let search_ctx = Option.value timing_ctx ~default:ctx in
    (* The base compile: identity transform (= the serial baseline candidate), capturing the
       optimized code every candidate derives from (see [compile_candidate]) and its canonical form.
       Canonicalize INSIDE the transform: after the transform returns, codegen forces the remaining
       undecided placements into the very placements table the captured [opt] references, and
       placement classes enter the digest (Schedule_cache.canonicalize) — the disk-cache key must be
       the deterministic transform-time form so that storing and replaying processes agree.

       The baseline is a candidate, so its compile is protected like every other candidate's
       (gh-ocannl-533): a typed rejection — the HIP scratch validator declining the unscheduled
       serial form at [Backend_link] is the case that motivated this — declines the baseline and
       lets the search proceed with the scheduled candidates, instead of killing the run before a
       single candidate has been tried. This is sound because the capture happens INSIDE the
       transform, which runs before codegen and link: [base_opt] survives the rejection, so every
       candidate still derives from the same base lowering. Only the timing of the serial form is
       lost, and on a GPU backend it was never going to be timed anyway ([dispatchable] below).
       Unclassified failures stay fatal: provenance [Candidate] under strict classification.

       gh-ocannl-552 settled whether this base compile should instead be the default-annotated
       pipeline (the shared cause behind gh-ocannl-532 and gh-ocannl-533): it cannot be. The default
       form is [maybe_default_schedules] — fission, then per-segment annotation — so in general it
       is several kernels, not one [optimized] to rebase candidates on; every candidate family
       (presets, the sketch detectors, fission enumeration, beam menu moves) assumes the serial zero
       point; and annotation consults [hardware_limits], which would bake per-device decisions into
       [source_digest]. The consequences that motivated the question are each handled where they
       arise: the scratch hazard by this compile's candidate-grade protection (gh-ocannl-533), the
       GPU dispatch hazard by [dispatchable] (gh-ocannl-532), and the missing "did tuning beat the
       default?" reference by [report.default_ms] — the [config_thresholds] seed's measurement, not
       a new baseline. *)
    let base_capture = ref None in
    (* gh-ocannl-1388: the base lowering as the transform received it, for deriving the seeded
       untuned plan exactly as the untuned compile derives it inside its own transform: codegen
       later forces the undecided placements into the table the captured [opt] references, and the
       plan's decisions key on placement-sensitive digests. GPU only: the plan is CUDA/HIP's. *)
    let base_pristine = ref None in
    let base_undispatched = ref false in
    progress_stage "base_compile";
    let base_outcome =
      costed ~stage:"base_compile" ~attempt:"baseline" ~verdict_of:outcome_verdict @@ fun () ->
      Context.compile_outcome ?name
        ~lowered_transform:(fun opt ->
          (* Inside the transform, so an injected fault is classified by the ordinary machinery
             (phase [Transform], provenance [Candidate]) and reaches [raise_pre_search] below with a
             real phase and a report — rather than escaping the whole call unreported, which would
             break the exactly-once contract for direct [tune] callers. *)
          Seam.get on_candidate_attempt "baseline";
          base_capture := Some (opt, SC.canonicalize ~static_indices opt);

          if is_gpu then base_pristine := Some (scratch_of opt);
          (* gh-ocannl-1340: an undispatchable baseline is refused here, before codegen, like any
             candidate ([compile_candidate_gated]): it is never timed, so its codegen, device
             compile and link were pure cost -- minutes of hiprtc per call on HIP. *)
          if not (dispatchable ~is_gpu [ opt ]) then (
            base_undispatched := true;
            raise
              (Outcome.Cause_at
                 ( Outcome.Transform,
                   Outcome.Not_dispatched
                     {
                       origin = "baseline";
                       detail =
                         Printf.sprintf
                           "the serial baseline binds no hardware dimension on %s (gh-ocannl-532)"
                           backend;
                     } )));
          [ opt ])
        ~provenance:Outcome.Candidate ~candidate:"baseline" search_ctx comp bindings
    in
    let base_opt, canon =
      match (!base_capture, base_outcome) with
      | Some oc, _ -> oc
      (* Failed before reaching the transform: there is no base lowering, hence no search. *)
      | None, Error failure -> raise_pre_search failure
      | None, Ok _ -> failwith "Autotune.tune: backend compile did not invoke lowered_transform"
    in
    (* gh-ocannl-1621: every timing window starts from the state the search lineage held before the
       search's first dispatch. The snapshot is taken at the first window -- after the base compile,
       before any routine of this search has run, since [time_routine] is a search's only dispatch;
       a call that times nothing (a cache replay, a search that declines everything) copies nothing.
       Each later window first restores it, outside [time_routine] and so outside its clock and its
       [setup_ms]: what a candidate is timed on is the routine it ships, optimizer update included,
       started from the same values as every other candidate. The runs of one window still train
       from there (the boundary; restoring per batch would put a host transfer before every
       calibration probe). [with_timing_state_restored] restores once more when the call ends. *)
    let timing_snapshot =
      lazy
        (let snapshot = snapshot_timing_state search_ctx base_opt in
         logf
           "timing state: %d written node(s) of the search lineage, %d bytes, restored before each \
            timing window"
           (List.length snapshot)
           (List.sum (module Int) snapshot ~f:(fun (_, nd) -> Ir.Ndarray.size_in_bytes nd));
         snapshot)
    in
    let timing_state_dirty = ref false in
    let restore_timing_snapshot () =
      let snapshot = Lazy.force timing_snapshot in
      if !timing_state_dirty then (
        restore_timing_state search_ctx snapshot;
        timing_state_dirty := false)
    in
    (* Called right before every [time_routine] of the search, after the lineage-wide preflight (so
       a restore never vouches for an input nobody initialized) and outside the candidate's failure
       boundary: a failed restore is the lineage's, not a candidate's decline. *)
    let before_timing_window () =
      restore_timing_snapshot ();
      timing_state_dirty := true
    in
    timing_state.settle <-
      Some
        (fun ~after_exn ->
          (* A poisoned lineage is unusable by contract, and a search can still return over one:
             with a [timing_ctx], a contained candidate failure poisons the scratch lineage while
             the winner ships from [ctx] -- restoring there would refuse, and raise over a result
             the caller is owed. *)
          if !timing_state_dirty && Option.is_none (Context.poisoned_failure search_ctx) then
            if not after_exn then restore_timing_snapshot ()
            else
              match restore_timing_snapshot () with
              | () -> ()
              | exception exn when not (process_fatal_exn exn) ->
                  logf "timing state: restoring the search lineage after a failure FAILED: %s"
                    (Exn.to_string exn));
    (* The key this call's schedule entry is stored and looked up under, and (gh-ocannl-1022) the
       [source_digest] of every report from here on, failures included: they all describe a search
       of this lowering. Shadowing [base_report] reaches [census] and every [~base] passed below;
       the report closures defined above keep the empty-digest one, which only the failures that
       precede the capture use without a [~base]. *)
    let base_digest = SC.digest canon in
    let base_report = { base_report with source_digest = base_digest } in
    (* gh-ocannl-1388 + gh-ocannl-1253: the seeded untuned plan, decided once on the base lowering.
       It is the search's first candidate, and since its [auto] gate reads the device description's
       [Untimed] capacity and cache bandwidth, the RESOLVED plan -- its content digest, never the
       raw constants -- joins the trajectory identity: a crown searched under another seed is not an
       equal-depth record for early stopping. An absent plan is named too ([seed=none]), so no
       trajectory recorded before the seed joined the identity -- one whose first step may have been
       a seed -- matches a search without one. Lazy: a warm hit whose entry carries no record never
       derives it. *)
    let seeded_plan_value =
      lazy
        (Seam.get on_seeded_plan
           (Option.bind !base_pristine ~f:(fun opt ->
                let t0 = Unix.gettimeofday () in
                let plan =
                  seeded_plan ~capabilities ~backend ~limits ~accum_prec ~static_indices opt
                in
                logf "seeded untuned plan derived in %.3f s: %s"
                  (Unix.gettimeofday () -. t0)
                  (Option.value_map plan ~default:"none" ~f:seeded_plan_label);
                plan)))
    in
    let search_shape =
      lazy
        (match Lazy.force seeded_plan_value with
        | None -> search_shape_settings ^ "|seed=none"
        | Some plan ->
            search_shape_settings ^ "|seed=" ^ seeded_plan_digest plan
            (* gh-ocannl-1138: the staffed search's start reads the plan's gate on the search's own
               segments, which the digest does not record. *)
            ^ Option.value_map plan.sp_start ~default:"" ~f:(fun start ->
                if staffed_singles then "|staffed_start=" ^ start.gate_tag else ""))
    in
    (* A stored shape whose settings already differ is decided without deriving the plan. *)
    let same_search_shape stored =
      String.is_prefix stored ~prefix:search_shape_settings
      && String.equal stored (Lazy.force search_shape)
    in
    let baseline_linked, baseline_decline =
      match base_outcome with
      | Ok (bctx, broutine) -> (Some (bctx, broutine), None)
      (* Refused before codegen, not declined: the search records the refusal where it always has,
         as [Not_dispatched_key "baseline"], and the two accounts stay exclusive. *)
      | Error failure when !base_undispatched && is_not_dispatched failure -> (None, None)
      | Error (Outcome.Classified classified) -> (None, Some classified)
      | Error (Outcome.Fatal _ as failure) -> raise_pre_search ~base:base_report failure
    in
    (* gh-ocannl-550: the base compile runs BEFORE the cache is consulted — its lowering is what
       every candidate and every replay derives from — so on the two paths that do not search, its
       linked artifact is dead as soon as that decision is taken, and nothing downstream can reach
       it. On the search path it is timed (where it is dispatchable) and then released like any
       candidate. Without this, a warm-cache process leaked one full base-candidate pool per [tune]
       call, permanently (the pool table roots it), which for a repeatedly-tuning process is the
       very accumulation this issue is about.

       A ref, because a cache replay releases it BEFORE the replay compile (gh-ocannl-1326) and a
       stale entry that sends the call on to search relinks it where it is to be timed. *)
    let baseline_linked = ref baseline_linked in
    let baseline_released = ref false in
    let release_baseline () =
      Option.iter !baseline_linked ~f:(fun (bctx, _) ->
          baseline_released := true;
          release_quietly ~what:"the baseline compile" bctx)
    in
    release_baseline_hook := release_baseline;
    let codegen_tag = SC.codegen_tag ~backend ~limits ~capabilities () in
    let objective = timing_string timing in
    let key_of canon =
      SC.cache_key
        ~timing_identity:(Context.timing_identity search_ctx)
        ~objective ~limits ~capabilities canon ~backend
    in
    (* Whether a stored winner answers this call, for the base whose digest is [digest]. The
       numerics and codegen checks are belt-and-braces: the key already carries both tags
       (gh-ocannl-568, gh-ocannl-572), so a regime-mismatched entry normally lives in a different
       file and is never looked up. They catch a hand-moved or hand-written entry, which is the
       shape of the misdirection this guards against — a tf32-vs-default A/B whose cache directories
       got crossed. An entry from before the codegen field existed carries no claim about its
       regime, so it is not rejected on that ground. *)
    let entry_replayable ~digest (entry : SC.entry) =
      String.equal entry.SC.source_digest digest
      && String.equal entry.SC.numerics (SC.numerics_tag ())
      && Option.value_map entry.SC.codegen ~default:true ~f:(String.equal codegen_tag)
      && Option.value_map entry.SC.objective ~default:true ~f:(String.equal objective)
      (* gh-ocannl-1409: a budget-cut crown answers only a call that has a budget too. *)
      && (Option.is_some budget || not entry.SC.budget_cut)
    in
    (* gh-ocannl-1614: the base lowering reads node value bounds (gh-ocannl-1560's audit), so a base
       lowered after a link that proposed some -- a [Host_inits] upload of an index tensor -- folds
       guards a base lowered before it kept, and digests to another key: a tune after a run missed
       the entry its cold twin stored and searched again. The base the call keeps is the bounded one
       (its folds are free speed for a fresh search), but when no winner is stored under its key and
       one is stored under the key of the same routine lowered blind to node bounds
       ([LL.without_node_bounds]), the call takes the blind lowering as its base instead. The swap
       is whole: every candidate, the replay, a re-search on a stale entry (whose baseline relinks
       from [base_opt]) and the store all derive from the blind base, so the stored entry keeps
       naming the code it was searched on, and a replay applies a schedule to exactly the code it
       was saved against -- never the bounded code to a schedule searched on the blind one. A cold
       base is blind already when nothing has proposed bounds for its nodes, which is when its entry
       is stored. What it costs a hit: the guard folds of the bounded lowering, which the cold call
       that stored the winner shipped without too. Neither key is widened: each is still the digest
       of the code it names, so two bases that differ in anything but those folds keep apart. Gated
       on a node the routine mentions carrying a bounds candidate (nothing can fold otherwise, so
       the blind lowering would be the base itself), then on a miss under the bounded key, checked
       before the blind lowering: a call with no bounds pays nothing, and a warm bounded hit one
       extra entry read but no second lowering. *)
    let blind_rebase =
      let bounds_proposed () =
        let ins, outs = Ir.Assignments.collect_nodes_guess_output comp.Ir.Assignments.asgns in
        Set.exists (Set.union ins outs) ~f:(fun tn -> Option.is_some (Ir.Tnode.bounds_candidate tn))
      in
      let stored ~digest canon =
        Option.exists (SC.lookup ~dir:cache_dir ~key:(key_of canon)) ~f:(entry_replayable ~digest)
      in
      if
        String.is_empty cache_dir
        || (not (SC.complete canon))
        || Option.is_none (key_of canon)
        || (not (bounds_proposed ()))
        || stored ~digest:base_digest canon
      then None
      else
        match
          LL.without_node_bounds (fun () ->
              Context.lowered_for_decisions ?name search_ctx comp bindings)
        with
        | exception exn when not (process_fatal_exn exn) ->
            logf "blind base lowering failed, keeping the bounded base: %s" (Exn.to_string exn);
            None
        | blind_opt ->
            let blind_canon = SC.canonicalize ~static_indices blind_opt in
            let blind_digest = SC.digest blind_canon in
            if String.equal blind_digest base_digest || not (SC.complete blind_canon) then None
            else if stored ~digest:blind_digest blind_canon then
              Some (blind_opt, blind_canon, blind_digest)
            else None
    in
    let base_opt, canon, base_digest, base_report =
      match blind_rebase with
      | None -> (base_opt, canon, base_digest, base_report)
      | Some (blind_opt, blind_canon, blind_digest) ->
          logf
            "base lowered after a link (digest %s); a winner is stored for its bounds-blind \
             lowering (digest %s), which this call rebases on (gh-ocannl-1614)"
            (dshort base_digest) (dshort blind_digest);
          if is_gpu then base_pristine := Some (scratch_of blind_opt);
          (blind_opt, blind_canon, blind_digest, { base_report with source_digest = blind_digest })
    in
    let key = key_of canon in
    let use_cache = (not (String.is_empty cache_dir)) && SC.complete canon && Option.is_some key in
    (* gh-ocannl-1384: unconditional, not [logf]. Declining the cache this way is invisible
       otherwise -- the call searches, returns a good routine and stores nothing, so every later
       call searches again -- and for weeks it was why no padded-conv cell ever replayed. *)
    if (not (String.is_empty cache_dir)) && not (SC.complete canon) then
      Stdio.eprintf
        "autotune: WARNING: schedule cache not used for routine %S (digest %s): its canonical form \
         is incomplete (%s), so this search's winner is neither looked up nor stored\n\
         %!"
        (Lazy.force routine_name) base_digest
        (Option.value (SC.incomplete_reason canon) ~default:"no reason recorded");
    if Option.is_none key then
      logf "persistent timed cache disabled for %s: concrete device identity unavailable" backend;
    (* gh-ocannl-1438: the privatized presets' dropped proposals, by the same aggregation, in a
       table of their own: a drop costs a proposal inside a candidate that still compiles, so
       counting it in [declines] would break that census's sum to [candidates_failed]. *)
    let privatize_declines : (Outcome.rejection_key, decline_acc) Hashtbl.Poly.t =
      Hashtbl.Poly.create ()
    in
    let on_privatize_drops drops =
      List.iter drops ~f:(fun (d : privatize_drop) ->
          logf "privatize proposal for %s over %s dropped: %s" (Ir.Tnode.debug_name d.pd_target)
            (Idx.symbol_ident d.pd_over)
            (Outcome.detail_of_cause d.pd_cause);
          record_decline ~warn:false privatize_declines
            {
              Outcome.phase = Outcome.Transform;
              cause = d.pd_cause;
              execution_effect = Outcome.No_device_writes;
            })
    in
    let compile_spec spec =
      costed ~attempt:(spec_label spec) ~verdict_of:outcome_verdict (fun () ->
          compile_candidate ?name ~on_privatize_drops ~static_indices ~base_opt ~canon ~limits
            ~is_gpu ~is_cpu ~provenance:Outcome.Candidate search_ctx comp bindings spec)
    in
    (* The search's own candidate compiles, deduplicated against [seen] before codegen. *)
    let compile_spec_deduped ~seen =
      compile_candidate_gated ?name ~seen ~on_privatize_drops ~static_indices ~base_opt ~canon
        ~limits ~is_gpu ~is_cpu ~provenance:Outcome.Candidate search_ctx comp bindings
    in
    (* Winner (and cache-hit) compiles target the caller's context; they replay against the same
       base lowering as the search's candidates, the caller's lineage lowering to it
       ([Lineage_refused] below). *)
    let compile_spec_real ?label ?retain provenance spec =
      costed
        ~attempt:(Option.value label ~default:(spec_label spec))
        ~verdict_of:outcome_verdict
        (fun () ->
          compile_candidate ?name ?label ?retain ~static_indices ~base_opt ~canon ~limits ~is_gpu
            ~is_cpu ~provenance ctx comp bindings spec)
    in
    let flat_schedule = function
      | Whole_saved saved -> saved
      | Fiss_saved segs -> List.concat_map segs ~f:(fun s -> s.SC.seg_saved)
      | Split_saved (prelude, segs) -> prelude @ List.concat_map segs ~f:(fun s -> s.SC.seg_saved)
    in
    let is_fissioned = function Whole_saved _ -> false | Fiss_saved _ | Split_saved _ -> true in
    (* A timed candidate's saved form as the spec that recompiles it: the winner replay's, and the
       beam finals' re-timings (gh-ocannl-1635). *)
    let spec_of_form = function
      | Whole_saved saved -> Whole (W_saved (saved, None))
      | Fiss_saved segs -> Fiss (F_saved (segs, None))
      | Split_saved (prelude, segs) -> Fiss (F_split_saved (prelude, segs, None))
    in
    (* Whether the crowned schedule tensorizes is read off the schedule, not off the winning spec's
       label (gh-ocannl-546): the beam can extend a plainly-labeled incumbent with a [Tensorize]
       move, and a sketch label promises tensorization the transform may not have kept. *)
    let saved_is_tensorized (saved : SC.saved_schedule) =
      List.exists saved ~f:(function SC.Tensorize _ -> true | _ -> false)
    in
    (* What the emission actually did, straight off the compiled routine: the schedule-side
       [saved_is_tensorized] says what was ASKED, [c.routine.mma] says what was DELIVERED, and the
       gap between the two is exactly the false "tensorized" timing (gh-ocannl-626). *)
    let mma_summary c = c.routine.Context.mma in
    let mma_scalar_fallbacks c = (mma_summary c).Ir.C_syntax.scalar_fallbacks in
    let mma_statements c = (mma_summary c).Ir.C_syntax.statements in
    (* The decline census outlives the cache branch: the baseline compile happens before the lookup
       and can be declined whether or not a cached winner then replays (gh-ocannl-533), so a
       cache-hit report has to carry that rejection too — [baseline_declined] with an empty census
       would be an internally inconsistent diagnostic on exactly the warm-cache runs of the workload
       that motivated the fix (Codex review, PR #271). *)
    let declines : (Outcome.rejection_key, decline_acc) Hashtbl.Poly.t = Hashtbl.Poly.create () in
    (* A declined baseline is an ordinary entry in the census: it is the same evidence about the
       same device as any candidate's rejection, and dropping it would report a smaller
       [candidates_failed] than the work actually attempted. It is recorded HERE and NOT as a
       [Not_dispatched] refusal below — the two are mutually exclusive accounts of one baseline, and
       the gh-ocannl-532 refusal asserts a reason ("binds no hardware dimension") that is not why
       this baseline did not run. *)
    Option.iter baseline_decline ~f:(record_decline declines);
    (* What the call has learned by now: everything after this point reports on top of it, success
       or failure, so a pre-search failure never understates the work already attempted (a declined
       baseline in particular must not read back as [baseline_declined = false], [declines =
       []]). *)
    (* gh-ocannl-1242: why a cache entry under this key did not replay, when it was refused with a
       typed cause — set by the lookup below and carried by every report after it. *)
    let cache_replay_refusal = ref None in
    let census () =
      {
        base_report with
        candidates_failed = failed_count declines;
        baseline_declined = Option.is_some baseline_decline;
        declines = decline_summaries declines;
        privatize_drops = decline_summaries privatize_declines;
        cache_replay_refusal = !cache_replay_refusal;
      }
    in
    (* gh-ocannl-1326: a device that refuses what every schedule of the base lowering must allocate
       has answered for the whole search: every candidate, a cache replay and the untuned fallback.
       Refused here, typed, rather than by declining every candidate in turn at link (a b256
       [gpt2_mini_train] arm on a 12 GiB card spent 12 minutes doing that before its fallback
       compile died of the same cause), and before the cache is consulted, whose replay would be
       refused alike. A [Pre_search_failure] is no search and no replay, which is how
       {!Train.tune_placements}' other arm still ships and how a benchmark harness tells this from a
       search. What "must allocate" means is the floor ({!Context.probe_link_floor}, gh-ocannl-1554,
       [probe_floor] below): not the base link's delta, which bounds no candidate from below under
       the liveness planner or on-chip fusion; and not the first search candidate to reach link
       (gh-ocannl-1493), which can be larger than the base. *)
    let is_device_memory = function
      | { Outcome.cause = Outcome.Resource_exceeded { resource = Outcome.Device_memory; _ }; _ } ->
          true
      | _ -> false
    in
    let refuse_if_device_memory ~probe = function
      | Some classified when is_device_memory classified ->
          logf "%s: REFUSED, the search cannot run: the floor under every schedule does not fit: %s"
            probe
            (Outcome.detail_of_cause classified.cause);
          (* The search is refused on a declined baseline -- the floor's refusal stands for it --
             which [census] reads off the base compile alone; the baseline's one account is in the
             census already (its own decline, or the floor's in its place). *)
          raise_pre_search
            ~base:{ (census ()) with baseline_declined = true }
            (Outcome.Classified classified)
      | Some _ | None -> ()
    in
    (* gh-ocannl-1421: a timing lineage that lowers the routine to another program than the caller's
       can tune nothing the caller can take: every candidate, cached entry and winner derives from
       the timing lineage's base lowering. Decided here, before the cache and the search, so the
       call reports what it ships -- the untuned default, timed at nothing -- rather than a crown
       that a refused winner compile would leave a placement A/B ranking (and recording) by. And
       before the device-memory refusal below: that is the TIMING lineage's base refused, which says
       nothing about the caller's untuned routine, the one thing a refused lineage ships. *)
    let lineage_refusal =
      Option.bind timing_ctx ~f:(fun tctx ->
          let identity c =
            lineage_identity ~static_indices (Context.lowered_for_decisions ?name c comp bindings)
          in
          match (identity ctx, identity tctx) with
          | caller_digest, timing_digest when String.equal caller_digest timing_digest -> None
          | caller_digest, timing_digest -> Some (caller_digest, timing_digest)
          | exception exn when not (process_fatal_exn exn) ->
              (* A lineage that cannot lower the routine fails the winner's and the untuned
                 fallback's compiles alike: said now, not after a search that cannot ship. *)
              let backtrace = Stdlib.Printexc.get_raw_backtrace () in
              raise_pre_search ~base:(census ())
                (Outcome.Fatal
                   {
                     exn;
                     backtrace;
                     phase = Outcome.Transform;
                     candidate = Some "lineage comparison";
                     cause = None;
                   })
          | exception exn ->
              (* A process-level failure propagates as it is, but not past the linked baseline. *)
              let backtrace = Stdlib.Printexc.get_raw_backtrace () in
              !release_baseline_hook ();
              Stdlib.Printexc.raise_with_backtrace exn backtrace)
    in
    (* gh-ocannl-1554: the footprint question every search-wide device-memory refusal asks -- does
       the device admit the FLOOR under every schedule of the base lowering
       ({!Context.probe_link_floor})? Not the base link's own delta, which bounds no candidate from
       below under the liveness planner (the serial base is one kernel, one GPU planner position, so
       it overlaps nothing a fissioned candidate's segments separate: 954 KB against 791 KB on
       [link_delta_parity]'s training step) nor under on-chip fusion (a shared epilogue placement
       keeps its target off the device), nor on the CPU under aliasing (a [Split_reduce] candidate
       refines the base's statement-grained spans). Asked in the same place for every path: after
       the lineage check (a refused timing lineage says nothing of the caller's routine) and before
       the cache, whose replay would be refused alike. A floor the device admits answers nothing
       about any candidate, which is that candidate's own decline at link.

       [~account]: where the baseline has no account in the census yet (it was refused before
       codegen, gh-ocannl-1340), a refused floor is its one account -- declined at [Backend_link]
       for device memory, as a refused base link's was before gh-ocannl-1340, and not beside its
       [Not_dispatched] refusal, which the search records only once it runs. Where the baseline's
       own link (or relink) was declined, that decline is its account already. *)
    let probe_floor ~account =
      progress_stage "footprint_probe";
      let probe = "baseline footprint floor probe" in
      match
        Outcome.protect ~classify_backend:(Context.failure_classifier search_ctx)
          ~provenance:Outcome.Candidate ~phase:Outcome.Backend_link ~candidate:probe (fun () ->
            Context.probe_link_floor ?name search_ctx base_opt)
      with
      | Ok (f : Context.link_floor) ->
          logf
            "%s: %s: %d working + %d constant bytes in %d pool(s); left out: %d bytes some \
             schedule keeps off the device, %d a packed tile can replace, %d the liveness planner \
             may overlap. It promises nothing about any candidate"
            probe
            (if f.lf_pools = 0 then "empty floor, nothing to ask" else "the floor fits")
            f.lf_working_bytes f.lf_constant_bytes f.lf_pools f.lf_off_device_bytes
            f.lf_replaceable_bytes f.lf_overlap_bytes
      | Error (Outcome.Classified classified) when is_device_memory classified ->
          if account then record_decline declines classified;
          refuse_if_device_memory ~probe (Some classified)
      | Error (Outcome.Classified classified) ->
          (* Not the device's memory: no answer to the footprint question, so the search
             proceeds. *)
          logf "%s: no answer, %s %s" probe (phase_label classified.phase)
            (Outcome.detail_of_cause classified.cause)
      | Error (Outcome.Fatal _ as failure) -> raise_pre_search ~base:(census ()) failure
    in
    (* The base compile's own device-memory decline refuses nothing by itself (it bounds no
       candidate from below, above): the floor answers. Where the baseline was refused before
       codegen (every GPU backend's), it allocated nothing, and the floor answers in its place. *)
    if Option.is_none lineage_refusal then
      if Option.exists baseline_decline ~f:is_device_memory then probe_floor ~account:false
      else if !base_undispatched then probe_floor ~account:true;
    let cached =
      if use_cache && Option.is_none lineage_refusal then
        match SC.lookup ~dir:cache_dir ~key with
        | Some entry when entry_replayable ~digest:base_digest entry -> (
            let spec =
              match entry.SC.segments with
              (* A fissioned entry with a non-empty [saved] is a split-reduce winner: [saved] is the
                 whole-routine prelude, [segments] the post-prelude per-segment schedules. *)
              | Some segs when not (List.is_empty entry.SC.saved) ->
                  Fiss (F_split_saved (entry.SC.saved, segs, None))
              | Some segs -> Fiss (F_saved (segs, None))
              | None -> Whole (W_saved (entry.SC.saved, None))
            in
            (* gh-ocannl-1326: the replay must not find the baseline's working set still resident:
               on a step that uses over half the device it was refused at link while the baseline
               held the rest, so every warm process re-searched. Dead on a hit, and relinked by the
               search if this entry turns out stale. *)
            release_baseline ();
            progress_stage "cache_replay";
            (* An entry written before the gh-ocannl-532 rule can name the serial baseline as the
               winner: it was timed then, and it won by default whenever every candidate failed to
               compile — the state gh-ocannl-521 recorded for every GPU backend. Replaying it would
               reintroduce the single-work-item dispatch through the cache, permanently and without
               ever timing anything. Rejected like a stale entry: the fresh search below overwrites
               it. Rejecting the replay (rather than bumping [entry_version]) keeps every sound
               entry, on this backend and on the CPU backends, where an empty schedule is a
               legitimate winner. The rejection is decided before codegen
               ([compile_candidate_gated]) and arrives as the typed [Not_dispatched] failure, which
               the [Error] arm below records in [cache_replay_refusal] like any other refused replay
               (gh-ocannl-1340). This arm is the post-codegen check on what was shipped.

               gh-ocannl-1440: a deferred call keeps the compiled code ([retained]), so the routine
               it ships is this compile's code linked again rather than a second compile. *)
            let retained = ref None in
            let retain = if defer then Some (fun rt -> retained := Some rt) else None in
            (* Counted as the backend compiles the replay completed, a replay whose link then failed
               included (review round 2). *)
            let before = Context.backend_compiles () in
            let replay_compile = compile_spec_real ?retain Outcome.Cache_replay spec in
            replay_compiles := !replay_compiles + (Context.backend_compiles () - before);
            match replay_compile with
            | Ok c when not (shipped_dispatchable ~is_gpu c) ->
                logf "cache entry replays to an unparallelized routine, re-searching: %s"
                  (spec_label spec);
                cache_replay_refusal :=
                  Some
                    {
                      Outcome.phase = Outcome.Backend_codegen;
                      cause =
                        Outcome.Not_dispatched
                          {
                            origin = "cache_replay";
                            detail =
                              Printf.sprintf
                                "%s shipped kernels binding no hardware dimension (gh-ocannl-532)"
                                (spec_label spec);
                          };
                      execution_effect = Outcome.No_device_writes;
                    };
                (* gh-ocannl-550: rejected, so its buffers are dead — and the fresh search below is
                   about to want them. *)
                release_quietly ~what:"a rejected cache replay" c.cctx;
                None
            | Ok c ->
                logf "cache hit: %s (best %.4f ms, baseline %.4f ms)" (spec_label spec)
                  entry.SC.best_ms entry.SC.baseline_ms;
                (* gh-ocannl-550: the report happens INSIDE the construction of [cached], so a
                   [report] callback that raises here never reaches the [Some result] arm below that
                   releases the baseline, and abandons the replayed winner too — two rooted routine
                   footprints per call, for a caller that retries. Both released before the
                   callback's exception propagates; the exception and its backtrace are
                   unchanged. *)
                let emit_report report =
                  match emit_report report with
                  | () -> ()
                  | exception exn ->
                      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                      release_quietly ~what:"the replay of a failed cache-hit report" c.cctx;
                      release_baseline ();
                      Stdlib.Printexc.raise_with_backtrace exn backtrace
                in
                let hit_report =
                  {
                    outcome = Cache_replay;
                    (* The entry's times are the storing search's, but the objective is this call's:
                       since gh-ocannl-755 the objective is a cache-key component, so an entry under
                       this key was measured under this objective. *)
                    timing;
                    candidates_timed = 0;
                    timings_contended = 0;
                    timings_retried = 0;
                    timings_unbatched = 0;
                    timings_unreadable = 0;
                    timings_dispersed = 0;
                    timings_hopeless = 0;
                    timing_decisions = timing_decisions ();
                    candidates_contended = 0;
                    default_refused = false;
                    (* No search ran, so the only rejection this can carry is the baseline's. *)
                    candidates_failed = failed_count declines;
                    baseline_declined = Option.is_some baseline_decline;
                    (* Nothing is timed on a replay, the baseline included. *)
                    baseline_refusal = None;
                    declines = decline_summaries declines;
                    privatize_drops = [];
                    cache_replay_refusal = None;
                    rounds_run = 0;
                    round_moves = [];
                    finals = [];
                    finals_timed = 0;
                    finals_admitted = 0;
                    finals_outcome = Finals_not_run;
                    (* A fact about the crown, like [best_ms] (gh-ocannl-1409). *)
                    budget_cut = entry.SC.budget_cut;
                    budget_s;
                    call_budget_s;
                    call_share_s;
                    compiles = 0 (* Stamped by [emit_report]. *);
                    replay_compiles = 0;
                    beam_width;
                    sketch_candidates = 0;
                    epilogue_sketch_candidates = 0;
                    fiss_sketch_candidates = 0;
                    fiss_sketch_timed = 0;
                    fiss_sketch_composite = `Ineligible;
                    fiss_sketch_playoff_timed = 0;
                    fiss_sketch_playoff_swaps = 0;
                    fiss_sketch_staffed_timed = 0;
                    fiss_sketch_backdrop_fallbacks = 0;
                    split_reduce_candidates = 0;
                    split_reduce_timed = 0;
                    split_reduce_composite_eligible = false;
                    split_reduce_composite_timed = false;
                    split_reduce_backdrop_fallbacks = 0;
                    mma_candidates = 0;
                    fiss_mma_candidates = 0;
                    mma_timed = 0;
                    model_scored = 0;
                    model_pruned = 0;
                    bound_pruned = 0;
                    fissioned = is_fissioned c.form;
                    baseline_ms = entry.SC.baseline_ms;
                    default_ms =
                      (* The entry's [default_ms] describes the default pipeline under the config
                         that ran the search; the cache key covers neither, so a config change can
                         redefine the default without missing the cache. Fingerprint mismatch drops
                         the stale diagnostic — the winner replay itself stays valid (Codex P2 on PR
                         #279). *)
                      (match (entry.SC.default_ms, entry.SC.default_fingerprint) with
                      | (Some _ as d), Some fp
                        when String.equal fp
                               (Sched.default_schedule_fingerprint ~backend_name:backend) ->
                          d
                      | _ -> None);
                    (* Not stored: a measurement of one seed, read only next to the search that
                       timed it. *)
                    seeded_ms = None;
                    best_ms = entry.SC.best_ms;
                    best_label = spec_label spec;
                    best_tensorized = saved_is_tensorized (flat_schedule c.form);
                    best_tensorization = Some (mma_summary c).Ir.C_syntax.tensorization;
                    best_mma_statements = mma_statements c;
                    best_mma_scalar_fallbacks = mma_scalar_fallbacks c;
                    (* Nothing was timed in this process — [mma_timed = 0] like every other COUNTER
                       here, which describes this call. The TIMES describe the program, and are
                       replayed from the entry exactly as [best_ms] and [baseline_ms] above are:
                       without that, the flip chain's profitability term (gh-ocannl-579) would rank
                       the decision surface one way on the cold run that measured the family and the
                       other way on every warm-cache run after it. [None] for a search that timed
                       none, and for entries written before the field existed. *)
                    mma_best_ms = Option.value entry.SC.mma_best_ms ~default:Float.infinity;
                    (* A time of the program, replayed with [best_ms] (gh-ocannl-1558). *)
                    best_spread = entry.SC.best_spread;
                    best_schedule = flat_schedule c.form;
                    source_digest = base_digest;
                    (* Replayed for the same reason as [mma_best_ms]: a flip chain whose incumbent
                       replayed would otherwise have no timed record to abandon a hopeless flip
                       against (gh-ocannl-1110). Empty for entries older than the field, and when
                       the storing search had another shape, whose candidate order an equal-depth
                       comparison cannot assume. *)
                    best_steps =
                      (match entry.SC.best_steps with
                      | Some t when same_search_shape t.SC.search_shape -> t.SC.steps
                      | Some _ | None -> []);
                    (* Filled in by [emit_report], like every report's. *)
                    compile_cost = Ir.Compile_cost.empty;
                  }
                in
                if not defer then (
                  emit_report hit_report;
                  Some (Fn.const (c.cctx, c.routine)))
                else (
                  (* gh-ocannl-1408: the replay above validated the entry -- a stale one goes on to
                     search, which only a compile can tell -- and supplied the emission census the
                     report carries. Its routine is not kept: a deferred call leaves nothing
                     resident. gh-ocannl-1440: its compiled code is, holding no working pools, and
                     the routine the caller asks for is that code linked again on [ctx]
                     ({!Context.relink_outcome}) -- the census the report carries is the shipped
                     code's, and the schedule compiles once. A link that fails anyway ships the
                     untuned default rather than a search, which has already been reported as not
                     needed. *)
                  release_quietly ~what:"the validating cache replay of a deferred call" c.cctx;
                  emit_report hit_report;
                  let retained = !retained in
                  Some
                    (deferred_compile (fun () ->
                         progress_stage "cache_replay_relink";
                         match
                           match retained with
                           | Some rt -> Context.relink_outcome rt
                           | None ->
                               failwith
                                 "Autotune: a deferred cache replay compiled without retaining its \
                                  code"
                         with
                         | Ok result -> result
                         | Error (Outcome.Classified classified) ->
                             logf
                               "the deferred cache replay's link FAILED (%s), compiling the \
                                untuned default"
                               (Outcome.detail_of_cause classified.cause);
                             substituted := true;
                             compile_untuned_default ~base:(census ()) ()
                         | Error (Outcome.Fatal _ as failure) ->
                             raise_pre_search ~base:(census ()) failure)))
            | Error (Outcome.Classified classified) ->
                (* Stale or corrupt entry: fall through to a fresh search, whose store then
                   overwrites the entry. A reference the canonical form cannot resolve
                   ([SC.of_saved]), a drifted fission segment and a refused transform all arrive
                   here typed (gh-ocannl-1242, gh-ocannl-1164); the report carries the cause. *)
                logf "cache entry replay FAILED, re-searching: %s"
                  (Outcome.detail_of_cause classified.cause);
                cache_replay_refusal := Some classified;
                None
            | Error (Outcome.Fatal _ as failure) -> raise_pre_search ~base:(census ()) failure)
        | Some entry when entry.SC.budget_cut && String.equal entry.SC.source_digest base_digest ->
            (* Reached only without a budget (the arm above takes it with one): the entry is the
               best of a prefix of the menu, and this call was not asked to stop early. Its search's
               store overwrites the entry, so the next budgeted call replays a full crown. *)
            logf
              "cache entry is the crown of a budget-cut search and this call has no budget: \
               searching in full (gh-ocannl-1409)";
            None
        | _ -> None
      else None
    in
    (* A winner takes precedence. Without one, replay an abandoned prefix only under the shape that
       measured it and a current rule that still rejects it. A weaker incumbent, changed margin,
       absent rule or changed sampling cannot inherit yesterday's verdict. *)
    let replayed_abandonment =
      if use_cache && Option.is_none cached && Option.is_none lineage_refusal then
        Option.bind abandon ~f:(fun rule ->
            Option.bind (SC.lookup_abandonment ~dir:cache_dir ~key) ~f:(fun entry ->
                if
                  String.equal entry.SC.source_digest base_digest
                  && same_search_shape entry.SC.trajectory.SC.search_shape
                then
                  Option.map
                    (abandon_verdict rule ~k:beam_width ~steps:entry.SC.trajectory.SC.steps)
                    ~f:(fun ab -> (ab, entry))
                else None))
      else None
    in
    match (lineage_refusal, cached, replayed_abandonment) with
    | Some (caller_digest, timing_digest), _, _ ->
        Stdio.eprintf
          "autotune: WARNING: routine %S: the caller's context lowers it differently from the \
           timing context (structural digest %s vs %s), so no schedule tuned there can ship into \
           it: not searching, compiling the untuned default (gh-ocannl-1421)\n\
           %!"
          (Lazy.force routine_name) caller_digest timing_digest;
        release_baseline ();
        let reached =
          { (census ()) with outcome = Lineage_refused { caller_digest; timing_digest } }
        in
        finish reached (fun () -> compile_untuned_default ~base:reached ())
    | None, Some result, _ ->
        release_baseline ();
        result
    | None, None, Some (ab, entry) ->
        release_baseline ();
        emit_report
          {
            (census ()) with
            outcome = Abandonment_replay ab;
            best_ms = ab.ab_best_ms;
            best_label = "cached-abandonment";
            best_steps = entry.SC.trajectory.SC.steps;
          };
        raise (Search_abandoned ab)
    | None, None, None when not search ->
        logf
          "search disabled (autotune_search=false) and no cache entry: compiling the untuned \
           default";
        (* Before the fallback compile, which wants the memory. *)
        release_baseline ();
        (* After the compile, as in the no-cache branch above. The census the base compile already
           produced is carried whether this succeeds or fails. *)
        let reached = census () in
        finish reached (fun () -> compile_untuned_default ~base:reached ())
    | None, None, None ->
        let seen = Hash_set.create (module String) in
        Hash_set.add seen base_digest;
        (* Every gh-ocannl-532 refusal enters the same decline census (gh-ocannl-543). Without it a
           GPU search that timed a single candidate reports [candidates_timed = 1] with an empty
           census — the same report a computation with a one-element schedule space would give — and
           the difference (how many candidates existed and were refused, and why) was only ever
           visible in the [autotune_log] stderr stream. *)
        let record_not_dispatched ~origin ~detail =
          record_decline declines
            {
              Outcome.phase = Outcome.Transform;
              cause = Outcome.Not_dispatched { origin; detail };
              execution_effect = Outcome.No_device_writes;
            }
        in
        (* A stale cache entry released the baseline before its replay (gh-ocannl-1326); relink it
           where it is to be timed. Not where it is not: an undispatched baseline is only ever
           released. *)
        let baseline_decline =
          if !baseline_released && dispatchable ~is_gpu [ base_opt ] then (
            progress_stage "baseline_relink";
            match compile_spec (Whole (W_saved ([], None))) with
            | Ok c ->
                baseline_linked := Some (c.cctx, c.routine);
                baseline_decline
            | Error (Outcome.Classified classified) ->
                baseline_linked := None;
                record_decline declines classified;
                (* The same footprint question as the base compile's (gh-ocannl-1554): a relink the
                   device refused bounds no candidate from below, the floor does. *)
                if is_device_memory classified then probe_floor ~account:false;
                Some classified
            | Error (Outcome.Fatal _ as failure) -> raise_pre_search ~base:(census ()) failure)
          else baseline_decline
        in
        (* [None] when the baseline compile was declined (gh-ocannl-533): there is no routine to
           time and none to return, and the search runs on the scheduled candidates alone. *)
        let baseline =
          Option.map !baseline_linked ~f:(fun (bctx, broutine) ->
              {
                form = Whole_saved [];
                cctx = bctx;
                routine = broutine;
                units =
                  [
                    {
                      u_segment = None;
                      u_saved = [];
                      u_registry = SC.base_registry canon;
                      u_opt = base_opt;
                    };
                  ];
                all_opts = [ base_opt ];
                digest_after = base_digest;
              })
        in
        (* Baseline timing failures are the user's bug (e.g. uninitialized inputs) and propagate as
           the exception [Context.run] would give — reported first, with the phase they carry, so
           the arm still occupies its slot (gh-ocannl-550). On a GPU backend the baseline is the
           unscheduled serial form and is not dispatched at all (see [dispatchable]); [infinity] is
           its rank, so every timed candidate beats it and the search never returns it (see the
           fallback at the end), and a declined baseline ranks the same way. *)
        let baseline_dispatched = Option.is_some baseline && dispatchable ~is_gpu [ base_opt ] in
        (* The baseline's refused window, by its cause (gh-ocannl-1517); [None] when it was
           admitted, or never ran one. *)
        let baseline_refusal = ref None in
        let baseline_timing_result = ref None in
        let baseline_ms =
          match baseline with
          | Some b when baseline_dispatched -> (
              (* Still uncaught in the sense that matters — the caller sees the same exception
                 [Context.run] would raise, unwrapped and with its own backtrace. The tagging is
                 only so the report can name the phase (pre-dispatch validation vs. launch vs. sync)
                 before it propagates.

                 The lineage effect is NOT optional, though, and it is why this consults the
                 backend's classifier like the candidate timing below (gh-ocannl-550): a baseline
                 launch that may have written buffers leaves the lineage unusable, and a caller that
                 CONTAINS this failure — [Train.tune_placements] does, per arm — would otherwise go
                 on to time its other arm against buffers the failed baseline had already modified.
                 Proven write-free, the routine's execution claim is withdrawn instead;
                 unattributed, the device's state is unknown and the lineage is condemned, exactly
                 as an unattributed candidate launch failure condemns it. *)
              let condemn phase exn =
                match phase with
                (* Nothing to judge and nothing to withdraw (gh-ocannl-564): the routine never ran,
                   and the execution claim is only made after a dispatch. Without this arm an
                   unsatisfied dependency would fall to [None] below on every C backend and condemn
                   the lineage the caller is meant to fix and retry in. *)
                | Outcome.Preflight -> ()
                | _ -> (
                    match Context.failure_classifier b.cctx phase exn with
                    | Some { Ir.Schedule_outcome.execution_effect = Outcome.No_device_writes; _ } ->
                        Context.rollback_execution b.cctx b.routine.Context.routine_id
                    | Some _ | None ->
                        Context.poison_lineage b.cctx ~routine_name:b.routine.Context.name exn)
              in
              match
                (* Lineage-wide validation, tagged so [condemn] above reads it for what it is —
                   pre-dispatch, nothing to withdraw — and raised here rather than inside the timing
                   so a baseline failure keeps propagating as the pre-search failure it is. This is
                   the site that made the containment gap invisible on the C backends: the serial
                   baseline is dispatched there, hits this first, and takes the search down with the
                   caller's error, where a GPU backend refuses the baseline outright (gh-ocannl-532)
                   and never reaches it (gh-ocannl-569). *)
                Outcome.tag Outcome.Preflight (fun () ->
                    Context.check_lineage_runnable b.cctx b.routine);
                (* gh-ocannl-1621: the search's first window, so this takes the snapshot. Tagged
                   pre-dispatch like the check above: nothing has run, so a failure here withdraws
                   nothing and condemns nothing, and propagates as the pre-search failure. *)
                Outcome.tag Outcome.Preflight before_timing_window;
                progress_stage "baseline_timing";
                time_routine ~on_retry ~on_call ~tag_failures:true ~timing ~repeats b.cctx b.routine
              with
              | timing_result -> (
                  let timing_result =
                    apply_measurement_seam ~label:"baseline" ~digest:base_digest timing_result
                  in
                  match admitted_timing_ms timing_result with
                  | Some ms ->
                      baseline_timing_result := Some timing_result;
                      ms
                  | None ->
                      let refusal = timing_admission timing_result in
                      baseline_refusal := Some refusal;
                      logf "baseline timing refused: %s" (refusal_reason refusal);
                      Float.infinity)
              | exception Outcome.Raised_at (phase, exn, backtrace) ->
                  condemn phase exn;
                  emit_pre_search_failure ~base:(census ()) ~phase ~candidate:(Some "baseline")
                    ~detail:(Exn.to_string exn) ();
                  (* gh-ocannl-550: it never reaches the beam, so nothing downstream can release it
                     — and a caller that CONTAINS this (a write-free preflight decline, a
                     backend-classified failure) goes on to another arm or retries. *)
                  release_baseline ();
                  Stdlib.Printexc.raise_with_backtrace exn backtrace
              | exception exn ->
                  let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                  condemn Outcome.Launch exn;
                  emit_pre_search_failure ~base:(census ()) ~phase:Outcome.Launch
                    ~candidate:(Some "baseline") ~detail:(Exn.to_string exn) ();
                  release_baseline ();
                  Stdlib.Printexc.raise_with_backtrace exn backtrace)
          | _ -> Float.infinity
        in
        let baseline_timed = baseline_dispatched && Float.is_finite baseline_ms in
        (* A contended timing established nothing about this digest. Let an identical seed retry it
           later in the search; successful timing and every definitive structural refusal keep the
           ordinary dedup ownership. *)
        if Option.is_some !baseline_refusal then Hash_set.remove seen base_digest;
        (match baseline_decline with
        | Some classified ->
            logf "baseline: DECLINED at %s %s" (phase_label classified.phase)
              (Outcome.detail_of_cause classified.cause)
        | None ->
            if baseline_timed then (
              (* Ranked like any timed candidate, and released like one (gh-ocannl-1326). *)
              release_baseline ();
              logf "baseline: %.4f ms (digest %s)" baseline_ms (dshort base_digest);
              emit_calibration ~backend ~device ~limits ~routine:(Lazy.force routine_name)
                ~label:"baseline" ~digest:base_digest
                ~timing_result:(Option.value_exn !baseline_timing_result)
                [ base_opt ])
            else if baseline_dispatched then release_baseline ()
            else (
              (* Refused before codegen ([base_undispatched], gh-ocannl-1340), so nothing is linked;
                 the release stays for a baseline this rule ever lets link. *)
              release_baseline ();
              (* No calibration row: the model column is only meaningful next to a measurement. *)
              logf
                "baseline: NOT DISPATCHED, binds no hardware dimension on %s -- the whole routine \
                 would run in one work-item (gh-ocannl-532); refused before codegen (digest %s)"
                backend (dshort base_digest);
              record_not_dispatched ~origin:"baseline"
                ~detail:
                  (Printf.sprintf
                     "the serial baseline binds no hardware dimension on %s (gh-ocannl-532)" backend)));
        let n_timed = ref (if baseline_timed then 1 else 0) in
        (* Fired from the counter itself rather than from the admission match above, so the reported
           [timed_so_far] is the accounting value and not a restatement of it. *)
        if baseline_timed then
          Option.iter baseline ~f:(fun b ->
              Seam.get on_candidate_timed b.routine.Context.name ~timed_so_far:!n_timed);
        let n_timings_contended = ref 0 in
        (* The subset of those windows refused because queued calibration measured no batch within
           its target (gh-ocannl-1098). Not a diagnosis (a stall on every probe reads the same), but
           a count that persists across idle reruns is a queue threshold's signature. *)
        let n_timings_unbatched = ref 0 in
        (* The subset refused for a degenerate clock reading (gh-ocannl-1517): a clock that resolved
           nothing is not a loaded host, and a skip excused by one names that cause. *)
        let n_timings_unreadable = ref 0 in
        (* Every refused window is counted here, by its cause, so the per-cause counts stay subsets
           of [n_timings_contended] by construction. *)
        let count_refusal refusal =
          Int.incr n_timings_contended;
          match refusal with
          | Refused_unbatched -> Int.incr n_timings_unbatched
          | Refused_unreadable -> Int.incr n_timings_unreadable
          | Refused_contended | Admitted -> ()
        in
        Option.iter !baseline_refusal ~f:count_refusal;
        (* [n_timings_contended] counts WINDOWS, which is the right shape for "was this search's
           measurement set complete?" but the wrong one for any claim about WHICH candidate went
           unmeasured (Codex P2 on PR #608): a refused digest is dropped from [seen] so an
           equivalent seed can retry it, so one candidate refused twice is two windows. The digests
           are therefore kept alongside, and the report derives from them a distinct-candidate count
           and the one per-candidate fact a caller cannot reconstruct — whether the untuned-default
           seed was the refused one, which is what separates a contention-refused reference from the
           gh-ocannl-552 regression of never proposing or attributing it. *)
        let contended_digests = Hash_set.create (module String) in
        if Option.is_some !baseline_refusal then Hash_set.add contended_digests base_digest;
        (* Live search state for an honest partial report. Each counter starts at the amount of work
           completed so far and is updated at its ordinary accounting site below. [best_so_far] is
           updated after every successful timing, including midway through seed enumeration. *)
        let n_mma_proposed = ref 0 and n_fiss_mma_proposed = ref 0 and n_mma_timed = ref 0 in
        (* gh-ocannl-546: the crowned candidate's identity, and how close tensorization came to it.
           Labels are keyed by digest rather than carried on the candidate, because the winner is
           picked from the beam pool (and the beam's own expansions time through the same site), so
           the timing site is the one place every timed candidate passes exactly once. *)
        let label_by_digest = Hashtbl.create (module String) in
        if baseline_timed then Hashtbl.set label_by_digest ~key:base_digest ~data:"baseline";
        let mma_best_ms = ref Float.infinity in
        let winner_label best_c =
          Option.value_map best_c ~default:"" ~f:(fun c ->
              Option.value (Hashtbl.find label_by_digest c.digest_after) ~default:"")
        in
        (* gh-ocannl-1558: each admitted window's spread, keyed like the labels, and how many
           admitted windows were dispersed -- a CPU routine's own variance on a quiet host, which
           the dispersion rule alone would have refused. *)
        let spread_by_digest = Hashtbl.create (module String) and n_timings_dispersed = ref 0 in
        (* gh-ocannl-1527: the windows cut as hopeless, their candidates unranked. *)
        let n_timings_hopeless = ref 0 in
        let note_admitted_window ~digest (r : timing_result) =
          Hashtbl.set spread_by_digest ~key:digest ~data:r.spread;
          if r.dispersed then Int.incr n_timings_dispersed
        in
        if baseline_timed then
          Option.iter !baseline_timing_result ~f:(note_admitted_window ~digest:base_digest);
        let winner_spread best_c =
          Option.bind best_c ~f:(fun c -> Hashtbl.find spread_by_digest c.digest_after)
        in
        let winner_tensorized best_c =
          Option.exists best_c ~f:(fun c -> saved_is_tensorized (flat_schedule c.form))
        in
        let n_model_scored = ref 0 and n_model_pruned = ref 0 in
        let n_bound_pruned = ref 0 in
        (* The schedule-invariant floor (gh-ocannl-514 phases 3-4): sketch completions share the
           base program's semantics, so one floor bounds every prunable candidate; computed once,
           only under the explicit gate. *)
        let floor_bound_ms =
          lazy
            (if not (Lazy.force bound_pruning_enabled) then None
             else
               let peak_flops, peak_memory_bandwidth = envelope ~limits in
               let f = CM.completion_floor base_opt.LL.llc in
               Option.map
                 (CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:f.CM.fr_flops
                    ~bytes:f.CM.fr_bytes ()) ~f:(fun sec -> sec *. 1e3))
        in
        let n_fiss_sketch_timed = ref 0
        and fs_composite = ref `Ineligible
        (* gh-ocannl-1166: while the composite playoff runs, its F_sketch windows are counted as
           playoff windows rather than as singles or as the coarse composite's own outcome. *)
        and in_playoff = ref false
        and n_playoff_timed = ref 0
        and n_playoff_swaps = ref 0
        (* gh-ocannl-1138: the staffed singles' windows, and the staffed singles retried bare. *)
        and n_staffed_timed = ref 0
        and n_backdrop_fallbacks = ref 0
        and n_sr_backdrop_fallbacks = ref 0
        and n_sr_timed = ref 0
        and sr_composite_eligible = ref false
        and sr_composite_timed = ref false in
        let sr_single_results = ref [] in
        let coarse_single_measured = Hash_set.Poly.create () in
        let coarse_single_refused = Hash_set.Poly.create () in
        let rounds_run = ref 0 in
        (* [report.round_moves], newest first (gh-ocannl-1360). *)
        let round_moves = ref [] in
        (* gh-ocannl-1635: the beam finals' accounting, published as each window ends. *)
        let finals_record = ref []
        and n_finals_timed = ref 0
        and n_finals_admitted = ref 0
        and finals_outcome = ref Finals_not_run in
        (* gh-ocannl-1409: set once the search's budget has stopped a candidate or a round. *)
        let budget_cut = ref false in
        let n_sketch_candidates = ref 0
        and n_epilogue_sketch_candidates = ref 0
        and n_fiss_sketch_candidates = ref 0
        and n_split_reduce_candidates = ref 0 in
        let best_so_far = ref ((if baseline_timed then baseline else None), baseline_ms) in
        (* [report.best_steps], newest first: each time [best_so_far] improves, at which admitted
           timing it did (gh-ocannl-1110). *)
        let best_steps = ref (if baseline_timed then [ (1, baseline_ms) ] else []) in
        let by_time (_, a) (_, b) = Float.compare a b in
        (* [beam] holds the fastest [beam_width] entries seen so far: the incumbents the rounds
           expand. It starts with the baseline when the baseline is eligible; a declined one
           contributes no entry, so the beam can be empty and every consumer below takes that as
           "nothing was timed" (gh-ocannl-533). Keeping the k smallest incrementally keeps the k
           smallest overall, with one difference from sorting at the end: a tie between exactly
           equal times resolves by arrival rather than by seed order.

           Its entries are descriptions, not artifacts (gh-ocannl-1326): every candidate's buffers
           are released as soon as its timing window is over, the incumbent's included, and the
           winner is recompiled from its saved form once the search ends. A candidate's device
           buffers are invisible to the OCaml GC (the backends' pool tables root every slab, see
           {!Context.release}), so whatever the search keeps holds device memory. gh-ocannl-550
           bounded that at [beam_width + 2] routine footprints, which is still too many for a step
           that uses over half the device: on a 12 GiB card a b256 [gpt2_mini_train] search timed
           its first candidate and then declined every later one at link with
           [CUDA_ERROR_OUT_OF_MEMORY], because the incumbent's 6.6 GiB stayed resident. What a
           ranked candidate is still read for — its saved form, units, compiled segments and the
           routine's tensorization census — is host data, and all a round needs is its saved form:
           every compile derives from [base_opt], never from another candidate's context. *)
        let beam =
          ref
            (Option.to_list
               (Option.map
                  (if baseline_timed then baseline else None)
                  ~f:(fun b -> (b, baseline_ms))))
        in
        (* The beam-expansion round's own bounded accumulator. Reset at the top of each round. *)
        let round = ref [] in
        (* Owns the just-compiled candidate until its timing window is over and it is released, so
           that a failure in between (a seam, a fatal timing failure) still frees it. *)
        let pending = ref None in
        let release_candidate c =
          if Option.exists !pending ~f:(phys_equal c) then pending := None;
          (* Best-effort: a failure to free must not replace the candidate's own outcome, and this
             runs on failure paths too, where the device may already be refusing work. Process-fatal
             conditions still propagate. Idempotent ({!Context.release}), so a ranked candidate
             released at its timing can meet this again harmlessly. A released candidate's digest
             deliberately STAYS in [seen]: it must keep deduplicating, and dedup cannot resurrect an
             artifact, since [seen], [timed_ms_by_digest] and [label_by_digest] hold strings and
             floats and never a [compiled]. *)
          release_quietly ~what:("candidate " ^ dshort c.digest_after) c.cctx
        in
        let admit entry =
          let kept, _evicted =
            List.split_n (List.sort (entry :: !beam) ~compare:by_time) beam_width
          in
          beam := kept
        in
        (* The exit sweep: whatever the search still owns when it stops. Every ranked candidate was
           released at its timing, so that is the pending one, cut short between its compile and the
           end of its timing window; the ranked ones are swept too, which costs nothing ([release]
           is idempotent) and keeps the sweep's meaning independent of where each one was released.
           It must run BEFORE the compiles that follow a search, which are the two an exhausted
           device used to defeat (the winner replay and the untuned-default fallback behind it). *)
        let release_all_candidates () =
          let live =
            List.map !beam ~f:fst @ List.map !round ~f:fst
            @ Option.to_list (fst !best_so_far)
            @ Option.to_list !pending
          in
          beam := [];
          round := [];
          pending := None;
          List.iter live ~f:release_candidate
        in
        (* The gh-ocannl-552 reference point. [baseline_ms] is the serial form's time ([infinity] on
           GPU), so it cannot answer "did tuning beat what the user gets without tuning?". The
           untuned default pipeline is already in the pool — the [config_thresholds] seed reproduces
           it exactly — and its measurement is attributed by digest, so a seed that dedups against
           an identical earlier candidate (the timed baseline included, on CPU backends whose config
           thresholds leave the code unparallelized) still reports the time of that code.

           The attribution honors the scheduling gates (Codex P1 on PR #279): the seed reproduces
           [maybe_default_schedules] only on its main path. With automatic scheduling inactive
           ([automatic_gpu_schedule]/[automatic_cpu_schedule] off, or [debug_log_from_routines] on),
           the untuned default IS the unscheduled serial form, so the reference is the base digest —
           timed on CPU, deliberately unmeasured on GPU (gh-ocannl-532). With [schedule_fission]
           off, the untuned default is the whole-routine config-thresholds annotation, which no
           candidate reproduces (the whole-routine presets are ungated): no attribution, rather than
           labeling a differently-scheduled pipeline as the default. *)
        let auto_sched = Sched.automatic_schedule_active ~backend_name:backend in
        (* A candidate label only when scheduling is active: inactive, the default is the base
           digest attributed right here, never a seed. *)
        let default_candidate_label =
          if auto_sched then default_seed_label ~backend_name:backend else None
        in
        let timed_ms_by_digest = Hashtbl.create (module String) in
        if baseline_timed then Hashtbl.set timed_ms_by_digest ~key:base_digest ~data:baseline_ms;
        let default_seed_digest = ref (if auto_sched then None else Some base_digest) in
        let default_ms () = Option.bind !default_seed_digest ~f:(Hashtbl.find timed_ms_by_digest) in
        (* gh-ocannl-1388: the seeded untuned plan's seed, attributed by digest like the
           default's. *)
        let seeded_seed_digest = ref None in
        let seeded_ms () = Option.bind !seeded_seed_digest ~f:(Hashtbl.find timed_ms_by_digest) in
        (* A digest that was refused and later timed by an equivalent seed is not an unmeasured
           candidate, so the count is over the refusals that stuck. *)
        let candidates_contended () =
          Hash_set.count contended_digests ~f:(fun d -> not (Hashtbl.mem timed_ms_by_digest d))
        in
        let default_refused () =
          Option.value_map !default_seed_digest ~default:false ~f:(Hash_set.mem contended_digests)
        in
        let partial_emitted = ref false in
        (* The report of a search cut short, from the state it reached: a fatal failure's
           ([emit_partial_and_raise]) and an abandonment's. *)
        let partial_report outcome =
          let summaries = decline_summaries declines in
          let best_c, best_ms = !best_so_far in
          {
            outcome;
            timing;
            candidates_timed = !n_timed;
            timings_contended = !n_timings_contended;
            timings_retried = !n_timings_retried;
            timings_unbatched = !n_timings_unbatched;
            timings_unreadable = !n_timings_unreadable;
            timings_dispersed = !n_timings_dispersed;
            timings_hopeless = !n_timings_hopeless;
            timing_decisions = timing_decisions ();
            candidates_contended = candidates_contended ();
            default_refused = default_refused ();
            candidates_failed = failed_count declines;
            baseline_declined = Option.is_some baseline_decline;
            baseline_refusal = !baseline_refusal;
            declines = summaries;
            privatize_drops = decline_summaries privatize_declines;
            cache_replay_refusal = !cache_replay_refusal;
            rounds_run = !rounds_run;
            round_moves = List.rev !round_moves;
            finals = !finals_record;
            finals_timed = !n_finals_timed;
            finals_admitted = !n_finals_admitted;
            finals_outcome = !finals_outcome;
            budget_cut = !budget_cut;
            budget_s;
            call_budget_s;
            call_share_s;
            compiles = 0 (* Stamped by [emit_report]. *);
            replay_compiles = 0;
            beam_width;
            sketch_candidates = !n_sketch_candidates;
            epilogue_sketch_candidates = !n_epilogue_sketch_candidates;
            fiss_sketch_candidates = !n_fiss_sketch_candidates;
            fiss_sketch_timed = !n_fiss_sketch_timed;
            fiss_sketch_composite = !fs_composite;
            fiss_sketch_playoff_timed = !n_playoff_timed;
            fiss_sketch_playoff_swaps = !n_playoff_swaps;
            fiss_sketch_staffed_timed = !n_staffed_timed;
            fiss_sketch_backdrop_fallbacks = !n_backdrop_fallbacks;
            split_reduce_candidates = !n_split_reduce_candidates;
            split_reduce_timed = !n_sr_timed;
            split_reduce_composite_eligible = !sr_composite_eligible;
            split_reduce_composite_timed = !sr_composite_timed;
            split_reduce_backdrop_fallbacks = !n_sr_backdrop_fallbacks;
            mma_candidates = !n_mma_proposed;
            fiss_mma_candidates = !n_fiss_mma_proposed;
            mma_timed = !n_mma_timed;
            model_scored = !n_model_scored;
            model_pruned = !n_model_pruned;
            bound_pruned = !n_bound_pruned;
            fissioned = Option.exists best_c ~f:(fun c -> is_fissioned c.form);
            baseline_ms;
            default_ms = default_ms ();
            seeded_ms = seeded_ms ();
            best_ms;
            best_label = winner_label best_c;
            best_tensorized = winner_tensorized best_c;
            best_tensorization =
              Option.map best_c ~f:(fun c -> (mma_summary c).Ir.C_syntax.tensorization);
            best_mma_statements = Option.value_map best_c ~default:0 ~f:mma_statements;
            best_mma_scalar_fallbacks = Option.value_map best_c ~default:0 ~f:mma_scalar_fallbacks;
            mma_best_ms = !mma_best_ms;
            best_spread = winner_spread best_c;
            best_schedule = Option.value_map best_c ~default:[] ~f:(fun c -> flat_schedule c.form);
            source_digest = base_digest;
            best_steps = List.rev !best_steps;
            (* Filled in by [emit_report], like every report's. *)
            compile_cost = Ir.Compile_cost.empty;
          }
        in
        let emit_partial_and_raise (fatal : Outcome.fatal) =
          (* Shadowing the projection of the same name would be gratuitous here: this is the failure
             being constructed, not one being read off a report. *)
          let failure =
            { phase = fatal.phase; candidate = fatal.candidate; detail = Exn.to_string fatal.exn }
          in
          let partial_report = partial_report (Search_died failure) in
          (* Reporting is best-effort on the exceptional path and must not replace the compiler
             failure or its raw backtrace. *)
          partial_emitted := true;
          (try emit_report partial_report
           with report_exn when not (process_fatal_exn report_exn) ->
             Stdio.eprintf "autotune: partial-report callback failed: %s\n%!"
               (Exn.to_string report_exn));
          (* gh-ocannl-550: this arm is over and returns no routine, so every artifact it still
             holds is dead. It matters most exactly here: a caller that CONTAINS this failure per
             arm ([Train.tune_placements]) goes on to search its other arm, and used to do so
             against a device still holding everything this arm had compiled. *)
          release_all_candidates ();
          Outcome.raise_failure (Outcome.Fatal fatal)
        in
        (* The post-search fallbacks to the untuned default (nothing timed; the winner replay failed
           or degenerated). Through the containment-aware form, so a failure here reports the phase
           it carries — the outer catch-all would otherwise record every one of them as [Transform],
           which for a link failure is simply wrong. The exception the caller sees is unchanged:
           [emit_partial_and_raise] ends in [raise_failure], exactly as [Context.compile] does. *)
        let plain_default_or_raise () =
          match
            costed ~attempt:"untuned default" ~verdict_of:outcome_verdict (fun () ->
                Context.compile_outcome ?name ~provenance:Ir.Schedule_outcome.User_schedule ctx comp
                  bindings)
          with
          | Ok result -> result
          | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
          | Error (Outcome.Classified classified) ->
              emit_partial_and_raise
                (Outcome.fatal_of_classified ~candidate:"untuned default fallback" classified)
        in
        (* gh-ocannl-1253: where the search derived a seeded plan, the untuned step the caller would
           have shipped is that plan, so the fallbacks compile it -- replayed as its [F_seeded]
           candidate is, from the plan already in the trajectory identity -- and keep the seeded
           floor. A replay that fails or degenerates ends on the plain default. *)
        let untuned_default_or_raise () =
          progress_stage "untuned_default_compile";
          match Lazy.force seeded_plan_value with
          | None -> plain_default_or_raise ()
          | Some plan -> (
              let spec = Fiss (F_seeded plan) in
              match compile_spec_real ~label:(spec_label spec) Outcome.Candidate spec with
              | Ok c when shipped_dispatchable ~is_gpu c -> (c.cctx, c.routine)
              | Ok c ->
                  release_quietly ~what:"the degenerate seeded fallback" c.cctx;
                  plain_default_or_raise ()
              | Error (Outcome.Classified classified) ->
                  logf "the seeded untuned fallback FAILED (%s); compiling the plain default"
                    (Outcome.detail_of_cause classified.cause);
                  plain_default_or_raise ()
              | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal)
        in
        let exception Abandon_now of abandonment in
        let search () =
          progress_stage "seed_enumeration";
          (* gh-ocannl-1061: where the search is, on the [autotune_progress] stream (the rest of its
             state is defined with [progress_phase_begin] below), and passed to each candidate's
             timing call for the trace's decisions (gh-ocannl-1199). *)
          let progress_phase = ref "seeds" in
          (* gh-ocannl-521: tensorized candidates are counted where they are TIMED, not where they
         are enumerated — a family can be seeded in bulk and rejected in bulk at candidate
         compile, and the enumerated count alone reads as coverage it does not have. Both
         counters are taken HERE rather than off [seed_specs], so they cover the same population
         by construction: the cross-segment recombination composite and the beam-expansion
         candidates also reach [try_spec] without appearing in the seed list, and counting only
         seeds in the denominator would let [mma_timed] exceed [mma_candidates] on a
         multi-segment routine. *)
          (* gh-ocannl-1138: how the last attempt that returned no timing ended, for the staffed
         sweep: a refusal ([`Declined], a compile or run decline that wrote nothing) is retried
         without its staff, and a dedup carries the time its code was already measured at, if
         it was. *)
          let last_attempt = ref `Other in
          (* gh-ocannl-1409, gh-ocannl-1556: the search's wall-clock budget, measured from the start
             of the call. Nothing in flight can be cancelled -- a compile runs to its end -- so the
             budget is kept by three checks, each against the same clock ([budget_clock]): - as each
             attempt (and each beam round) starts, it is refused once the budget is spent, or once
             the elapsed time plus the costliest candidate compile this search has made
             ([costliest_compile_s], the projection) would pass it: a compile is the part of an
             attempt nothing can stop, so one is started only where one as costly as the worst seen
             still ends within the budget; - as each batch of a candidate's timing starts, and
             before its warmup, the timing is abandoned once the budget is passed ([budget_stop]):
             the candidate is not ranked, its compile is spent, and the search ends as at a refused
             attempt. Once the budget is spent (or projected to be), nothing more is compiled -- the
             seeds left, the composites, the playoff and the rounds are skipped where they would
             have been attempted, which costs no compile -- and the search crowns its incumbent
             exactly as a completed one does. The untuned floors are exempt from all three
             ([floor]): the seeded untuned plan (gh-ocannl-1388) and the default-pipeline seed, so a
             cut search has measured what the caller ships without tuning and crowns nothing slower.
             A skipped spec leaves [last_attempt] at [`Other]: the staffed sweep must neither retry
             it bare nor rank a stale dedup time for it. *)
          let budget_clock () = Seam.get budget_elapsed (seconds_since progress_clock) in
          let budget_spent = ref false in
          let costliest_compile_s = ref 0. in
          let spend_budget ~b ~elapsed reason =
            budget_spent := true;
            logf "budget of %gs %s (%.1f s elapsed, %d timed): crowning the incumbent" b reason
              elapsed !n_timed;
            progress_line "budget_spent"
              (Printf.sprintf "budget_s=%g timed=%d %s" b !n_timed (progress_costs ()))
          in
          let budget_allows ~what ~floor =
            match budget with
            | None -> true
            | Some b ->
                (if not !budget_spent then
                   let elapsed = budget_clock () in
                   if Float.(elapsed >= b) then spend_budget ~b ~elapsed "spent"
                   else if Float.(elapsed + !costliest_compile_s > b) then
                     spend_budget ~b ~elapsed
                       (Printf.sprintf
                          "would be overrun by a compile as costly as the costliest so far (%.1f s)"
                          !costliest_compile_s));
                if (not !budget_spent) || floor then true
                else (
                  if not !budget_cut then
                    logf "budget spent: skipping %s and every later non-floor candidate" what;
                  budget_cut := true;
                  false)
          in
          let budget_floor spec =
            match spec with
            | Fiss (F_seeded _) -> true
            | _ -> Option.exists default_candidate_label ~f:(String.equal (spec_label spec))
          in
          (* The projection's evidence: a candidate compile's cost on the budget clock, kept as the
             costliest so far. Floors' compiles count -- they are compiles of the same computation
             -- while the base compile, which precedes every check, does not. *)
          let budget_compile f =
            match budget with
            | None -> f ()
            | Some _ ->
                let c0 = budget_clock () in
                Exn.protect ~f ~finally:(fun () ->
                    costliest_compile_s := Float.max !costliest_compile_s (budget_clock () -. c0))
          in
          (* The timing check, for a non-floor candidate: [time_routine] raises [Timing_stopped]
             before the first batch it starts past the budget. *)
          let budget_stop spec =
            match budget with
            | Some b when not (budget_floor spec) ->
                fun () ->
                  let elapsed = budget_clock () in
                  Float.(elapsed > b)
                  &&
                  (if not !budget_spent then
                     spend_budget ~b ~elapsed "passed during a candidate's timing";
                   budget_cut := true;
                   true)
            | Some _ | None -> fun () -> false
          in
          (* [hopeless_against] (gh-ocannl-1527) is what the caller ranks this candidate's reading
             against, asked as its timing starts: [Some ms] lets the window end as soon as it is
             hopeless against [ms] ({!hopeless_window}); [None], the default, times it in full. *)
          (* One timing window of the compiled candidate [c], timed under [label] -- a candidate's
             in [try_spec], a contender's re-timing in the beam finals (gh-ocannl-1635).

             The backend's own classifier decides whether a launch or sync failure is this
             candidate's fault: the driver error is all the evidence there is, and only the backend
             can read it. With the always-[None] classifier this used to pass, no backend could ever
             declare one, so every launch failure was fatal by phase default and there was nowhere
             for a backend to plug one in (gh-ocannl-536; the HIP scratch-overflow arm of
             gh-ocannl-533 is what fills this seam). The phase reaching the report is the tagged one
             inside [time_routine], so a report distinguishes a launch refusal from an asynchronous
             failure at sync.

             And from the third case, which is not the backend's to judge: [time_routine]'s
             pre-dispatch validation carries [Preflight] and is contained without asking the
             classifier (gh-ocannl-564). Tagged [Launch] it was fatal on every C backend, so a
             scratch context missing one of the caller's initializations condemned the search
             instead of declining a candidate. *)
          let timing_window ~label ~stop ?hopeless_against c =
            (* The lineage-wide validation is OUTSIDE the boundary (gh-ocannl-569): a poisoned
               lineage, an uninitialized input and an unexecuted dependency are properties of the
               context and the computation, so a genuine one fails every candidate of every arm at
               once. Contained as a decline it is silent — on a backend whose serial baseline is not
               dispatched (every GPU backend) every candidate declines for the one reason, nothing
               is timed, and the search ships the untuned default out of an unusable lineage under a
               report that says it completed. It reaches the caller instead, which is the only party
               that can fix it.

               Tagged, though not contained: the tag carries no boundary here, it only labels the
               phase so the fallback handler at the end of [search] reports a pre-dispatch
               validation failure as [Preflight] rather than as its [Transform] default. *)
            Outcome.tag Outcome.Preflight (fun () ->
                Context.check_lineage_runnable c.cctx c.routine);
            (* gh-ocannl-1621: the state every window starts from, restored outside the boundary for
               the same reason -- it is the lineage's, not this candidate's. *)
            Outcome.tag Outcome.Preflight before_timing_window;
            Outcome.protect ~classify_backend:(Context.failure_classifier c.cctx)
              ~provenance:Outcome.Candidate ~phase:Outcome.Launch ~candidate:label (fun () ->
                timed_into progress_timing_s (fun () ->
                    (* gh-ocannl-1556: [`Stopped] is a timing the budget stopped, and
                       gh-ocannl-1527: [`Hopeless] a window cut as out of the running -- neither a
                       failure of the candidate, so caught inside the boundary. *)
                    match
                      time_routine ~on_retry ~on_call ~tag_failures:true ~phase:!progress_phase
                        ~stop ?hopeless_against ~timing ~repeats c.cctx c.routine
                    with
                    | timing_result -> `Timed timing_result
                    | exception Timing_stopped -> `Stopped
                    | exception Timing_hopeless h -> `Hopeless h))
            (* Outside the boundary: the seam is not a candidate failure to classify. *)
            |> Result.map ~f:(function
              | `Timed timing_result ->
                  `Timed (apply_measurement_seam ~label ~digest:c.digest_after timing_result)
              | (`Stopped | `Hopeless _) as other -> other)
          in
          let try_spec ?(hopeless_against = fun () -> None) spec =
            last_attempt := `Other;
            Seam.get on_candidate_attempt (spec_label spec);
            let pruned_by_bound =
              bound_prunable spec
              && Option.value_map (Lazy.force floor_bound_ms) ~default:false ~f:(fun fb ->
                  (* Equality prunes: displacing the incumbent needs strict improvement. *)
                  Float.(fb >= snd !best_so_far))
            in
            if pruned_by_bound then (
              Int.incr n_bound_pruned;
              logf "%s: BOUND-PRUNED (floor %.4f ms >= best %.4f ms)" (spec_label spec)
                (Option.value_exn (Lazy.force floor_bound_ms))
                (snd !best_so_far);
              None)
            else (
              (* Counted only past the pruning gate: [mma_candidates]' contract is candidates put
                 through candidate compilation, and a bound-pruned sketch never was. *)
              if spec_expects_mma spec then begin
                Int.incr n_mma_proposed;
                match spec with
                | Fiss (F_sketch _) -> Int.incr n_fiss_mma_proposed
                | Whole _ | Fiss (F_preset _ | F_saved _ | F_split _ | F_split_saved _ | F_seeded _)
                  ->
                    ()
              end;
              (* Recorded whether or not this compile goes on to be timed: on dedup the code was (or
                 will not be) timed under the same digest, and the [default_ms] lookup follows the
                 digest, not the seed (gh-ocannl-552). Which seed, if any, is [default_seed_label]'s
                 to say. *)
              let attribute digest_after =
                if Option.exists default_candidate_label ~f:(String.equal (spec_label spec)) then
                  default_seed_digest := Some digest_after;
                match spec with
                | Fiss (F_seeded _) -> seeded_seed_digest := Some digest_after
                | _ -> ()
              in
              match
                timed_into progress_compile_s (fun () ->
                    budget_compile @@ fun () ->
                    costed ~stage:"candidate" ~attempt:(spec_label spec)
                      ~verdict_of:(function
                        | Compiled _ -> "compiled"
                        | Duplicate _ -> "dedup"
                        | Failed failure when is_not_dispatched failure -> "refused"
                        | Failed _ -> "failed")
                      (fun () -> compile_spec_deduped ~seen spec))
              with
              | Duplicate digest_after ->
                  (* Decided before codegen (gh-ocannl-1340): a duplicate pays for the transform
                     only, and its identical twin, timed or refused, is the one the search reasons
                     about. *)
                  attribute digest_after;
                  last_attempt := `Dedup (Hashtbl.find timed_ms_by_digest digest_after);
                  logf "%s: dedup before codegen (digest %s)" (spec_label spec)
                    (dshort digest_after);
                  None
              | Failed (Outcome.Classified classified as failure) when is_not_dispatched failure ->
                  (* Degenerated to the serial form (gh-ocannl-532), refused before codegen; the
                     gate recorded the digest as seen, so an equivalent later candidate dedups
                     rather than re-deriving the same refusal. Never timed, so its seed attribution
                     ([attribute]) would look up nothing. *)
                  logf "%s: NOT DISPATCHED, binds no hardware dimension, refused before codegen"
                    (spec_label spec);
                  record_decline declines classified;
                  None
              | Failed (Outcome.Classified classified) ->
                  (* A candidate refused for device memory is that candidate's decline: the
                     footprint question was the baseline's to answer (gh-ocannl-1493). *)
                  record_decline declines classified;
                  last_attempt := `Declined;
                  logf "%s: FAILED at %s %s" (spec_label spec) (phase_label classified.phase)
                    (Outcome.detail_of_cause classified.cause);
                  None
              | Failed (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
              | Compiled c ->
                  pending := Some c;
                  attribute c.digest_after;
                  if not (shipped_dispatchable ~is_gpu c) then (
                    (* The post-codegen check: the gate admitted the transform's forms, but the
                       kernels codegen shipped bind no hardware dimension (zero-only segments are
                       elided after the transform). *)
                    Hash_set.add seen c.digest_after;
                    logf
                      "%s: NOT DISPATCHED, its shipped kernels bind no hardware dimension (digest \
                       %s)"
                      (spec_label spec) (dshort c.digest_after);
                    record_not_dispatched ~origin:"candidate"
                      ~detail:
                        (Printf.sprintf
                           "%s shipped kernels binding no hardware dimension (gh-ocannl-532)"
                           (spec_label spec));
                    release_candidate c;
                    None)
                  else (
                    Hash_set.add seen c.digest_after;
                    match
                      timing_window ~label:(spec_label spec) ~stop:(budget_stop spec)
                        ?hopeless_against:
                          (if measurement_is_physical () then hopeless_against () else None)
                        c
                    with
                    | Ok (`Hopeless h) ->
                        (* gh-ocannl-1527: not ranked, and not a refusal -- the window proved the
                           candidate out of the running against what its caller ranks it by, so the
                           search stays cacheable. Its digest leaves [seen] as a refused one's does:
                           the reference can move, so an equivalent later proposal is timed against
                           its own. *)
                        Int.incr n_timings_hopeless;
                        if spec_expects_mma spec then Int.incr n_mma_timed;
                        Hash_set.remove seen c.digest_after;
                        logf
                          "%s: NOT RANKED, hopeless: its first %d samples read %.4f ms, over \
                           the                            %.4f ms (%gx what it would have to beat) \
                           that ends the window (digest                            %s)"
                          (spec_label spec) h.hopeless_samples h.hopeless_ms h.hopeless_threshold_ms
                          hopeless_ratio (dshort c.digest_after);
                        release_candidate c;
                        None
                    | Ok `Stopped ->
                        (* The budget passed during this candidate's timing: it is not ranked, and a
                           later floor of the same digest must compile and time afresh rather than
                           dedup against a timing that never finished. *)
                        Hash_set.remove seen c.digest_after;
                        logf "%s: NOT TIMED, the budget passed during its timing (digest %s)"
                          (spec_label spec) (dshort c.digest_after);
                        release_candidate c;
                        None
                    | Ok (`Timed timing_result)
                      when Option.is_none (admitted_timing_ms timing_result) ->
                        (* The family counters answer whether a candidate compiled and reached a
                           timing window, independently of whether that window yielded a usable
                           verdict. Keep that accounting stable under refusal; the historical
                           [timings_contended] counter covers every unusable timing result. *)
                        (match spec with
                        | Fiss (F_sketch _) when !in_playoff -> Int.incr n_playoff_timed
                        | Fiss (F_sketch { entries; staff; fine; _ }) -> (
                            Int.incr n_fiss_sketch_timed;
                            if not (List.is_empty staff) then Int.incr n_staffed_timed;
                            if not fine then
                              match entries with
                              | [ (key, _) ] -> Hash_set.add coarse_single_refused key
                              | _ :: _ :: _ -> fs_composite := `Refused
                              | [] -> ())
                        | Fiss (F_split { sites; staff }) ->
                            Int.incr n_sr_timed;
                            if List.length sites >= 2 && List.is_empty staff then
                              sr_composite_timed := true
                        | _ -> ());
                        if spec_expects_mma spec then Int.incr n_mma_timed;
                        let refusal = timing_admission timing_result in
                        count_refusal refusal;
                        Hash_set.add contended_digests c.digest_after;
                        (* Unlike a launch/compile decline, a refused window (contention, a
                           degenerate reading) is not a property of this schedule. An equivalent
                           later seed is a useful retry, not a dedup. *)
                        Hash_set.remove seen c.digest_after;
                        logf "%s: NOT TIMED, %s (digest %s)" (spec_label spec)
                          (refusal_reason refusal) (dshort c.digest_after);
                        release_candidate c;
                        None
                    | Ok (`Timed timing_result) ->
                        (* gh-ocannl-1326: the timing is all a candidate is for, the incumbent-to-be
                           included, so its buffers go before anything else happens: the next
                           compile must not find this one's working set still resident. A winner is
                           recompiled from its saved form when the search ends. *)
                        release_candidate c;
                        let ms = Option.value_exn (admitted_timing_ms timing_result) in
                        Int.incr n_timed;
                        note_admitted_window ~digest:c.digest_after timing_result;
                        (* Publish window accounting before the post-admission injection seam: it
                           can raise, and the partial report still owns this completed window. *)
                        (match spec with
                        | Fiss (F_sketch _) when !in_playoff -> Int.incr n_playoff_timed
                        | Fiss (F_sketch { entries; staff; fine; _ }) -> (
                            Int.incr n_fiss_sketch_timed;
                            if not (List.is_empty staff) then Int.incr n_staffed_timed;
                            if not fine then
                              match (entries, staff) with
                              | [ (key, _) ], [] -> Hash_set.add coarse_single_measured key
                              | [ (key, _) ], _ :: _ ->
                                  (* A staffed single schedules a composite of the coarse keys
                                     (gh-ocannl-1138): published here, before the callbacks that can
                                     cut the search, as an explicit composite's window is. *)
                                  Hash_set.add coarse_single_measured key;
                                  fs_composite := `Timed
                              | _ :: _ :: _, _ -> fs_composite := `Timed
                              | [], _ -> ())
                        | Fiss (F_split { sites; staff }) -> (
                            Int.incr n_sr_timed;
                            match sites with
                            | _ when not (List.is_empty staff) ->
                                (* A staffed single, ranked by its sweep in its own backdrop, splits
                                   several sites: a multi-site composite's window, published before
                                   the callbacks as an explicit composite's is. *)
                                sr_composite_eligible := true;
                                sr_composite_timed := true
                            | [ (s, b) ] ->
                                sr_single_results := (s, b, ms) :: !sr_single_results;
                                sr_composite_eligible :=
                                  List.exists !sr_single_results ~f:(fun (s2, _, _) ->
                                      not (Idx.equal_symbol s2.sr_axis s.sr_axis))
                            | _ :: _ :: _ -> sr_composite_timed := true
                            | [] -> ())
                        | _ -> ());
                        if spec_expects_mma spec then Int.incr n_mma_timed;
                        Hashtbl.set timed_ms_by_digest ~key:c.digest_after ~data:ms;
                        Hashtbl.set label_by_digest ~key:c.digest_after ~data:(spec_label spec);
                        (* Structural, not label-promised, and deliberately a different population
                           from [n_mma_timed]: with [rounds > 0] the beam menu appends a [Tensorize]
                           to a saved or preset incumbent, and the resulting [W_saved]/[F_saved]
                           spec promises nothing in its label — yet it is exactly as tensorized as a
                           sketch seed, and it can win. Keying this on the label would let the
                           placement A/B report "no tensorized candidate was timed" about a search
                           whose winner tensorizes. *)
                        if saved_is_tensorized (flat_schedule c.form) && Float.(ms < !mma_best_ms)
                        then mma_best_ms := ms;
                        if Float.(ms < snd !best_so_far) then (
                          best_so_far := (Some c, ms);
                          best_steps := (!n_timed, ms) :: !best_steps);
                        Seam.get on_candidate_callback `Timed ~candidate_ms:ms
                          ~incumbent_ms:(snd !best_so_far);
                        Seam.get on_candidate_timed c.routine.Context.name ~timed_so_far:!n_timed;
                        logf "%s: %.4f ms (digest %s)" (spec_label spec) ms (dshort c.digest_after);
                        Seam.get on_candidate_callback `Calibration ~candidate_ms:ms
                          ~incumbent_ms:(snd !best_so_far);
                        emit_calibration ~backend ~device ~limits ~routine:(Lazy.force routine_name)
                          ~label:(spec_label spec) ~digest:c.digest_after ~timing_result c.all_opts;
                        (* The rendering census next to the timing (gh-ocannl-479): a candidate
                           labeled tensorized whose [Tile_mma] statements all declined at emission
                           timed the scalar fallback — report it, or every number off this tuning
                           run inherits the ambiguity. *)
                        let summary = mma_summary c in
                        let scalar = summary.Ir.C_syntax.scalar_fallbacks in
                        let total = summary.Ir.C_syntax.statements in
                        if scalar > 0 then
                          logf
                            "%s: NOTE %s, %d/%d Tile_mma statement(s) rendered as the lane-0 \
                             scalar fallback (config schedule_log_declines=true names the failed \
                             rule)"
                            (spec_label spec)
                            (Ir.C_syntax.tensorization_name summary.Ir.C_syntax.tensorization)
                            scalar total
                        else if total = 0 && spec_expects_mma spec then
                          logf
                            "%s: NOTE not-requested, tensorized candidate emitted no Tile_mma \
                             statement"
                            (spec_label spec);
                        Some (c, ms)
                    | Error (Outcome.Classified classified) -> (
                        record_decline declines classified;
                        logf "%s: RUN FAILED at %s %s" (spec_label spec)
                          (phase_label classified.phase)
                          (Outcome.detail_of_cause classified.cause);
                        match classified.execution_effect with
                        | Outcome.No_device_writes ->
                            last_attempt := `Declined;
                            (* [Context.run] marks a routine executed before the later [sync] can
                               report an asynchronous failure. A rejection the backend proved wrote
                               nothing withdraws that claim, so the next candidate compiled in this
                               lineage does not wait on a routine that never completed. A no-op for
                               a [Preflight] decline, which precedes the dispatch that makes the
                               claim. *)
                            Context.rollback_execution c.cctx c.routine.Context.routine_id;
                            (* gh-ocannl-550: a candidate that failed to run is as dead as one that
                               lost, and on the failure that motivated all of this it is deader — an
                               out-of-memory decline is exactly when the freed buffers are worth
                               most. *)
                            release_candidate c;
                            None
                        | Outcome.Writes_may_have_occurred ->
                            (* Counted once as a decline (its cause is real evidence about the
                               candidate) and then escalated: the timing lineage may hold partially
                               written buffers, and there is no restore API to rebuild its inputs
                               and parameters, so timing the next candidate on it would score
                               suspect data. *)
                            Context.poison_lineage c.cctx ~routine_name:c.routine.Context.name
                              (Outcome.exception_of_cause classified.cause);
                            (* gh-ocannl-550: the exit sweep in [emit_partial_and_raise] can only
                               reach what the beam or [best_so_far] holds, and this candidate is in
                               neither — it failed before being admitted. Releasing it here is what
                               keeps the in-flight one from outliving the arm, which matters
                               precisely because [Train.tune_placements] CONTAINS this failure and
                               goes on to search its sibling arm on the same device. *)
                            release_candidate c;
                            emit_partial_and_raise
                              (Outcome.fatal_of_classified ~candidate:(spec_label spec) classified))
                    | Error (Outcome.Fatal fatal) ->
                        (* An unattributed launch/sync failure says nothing about what the device
                           did, so the lineage is condemned before the exception unwinds — a caller
                           that catches it cannot reuse a ledger claiming the failed routine
                           completed. *)
                        Context.poison_lineage c.cctx ~routine_name:c.routine.Context.name
                          fatal.Outcome.exn;
                        (* Not in the beam either (see above). *)
                        release_candidate c;
                        emit_partial_and_raise fatal))
          in
          (* gh-ocannl-550: the per-candidate allocation census, on the same [autotune_log] stream
             as the candidate lines it follows, so a growth curve can be read against the classes
             that produce it instead of against wall-clock samples from outside the process. One
             line per attempt, whether the candidate was timed, declined or deduped — a class that
             grows on the DECLINE path is a different bug from one that grows on the timed path, and
             only per-attempt lines distinguish them. The device figure is the backend's own
             accounting, which the census does not replace: it covers pools the shared seam does not
             allocate (the merge buffer) and, on [cc], counts host allocations whose GC finalizer
             has not yet run. *)
          let try_spec ?hopeless_against spec =
            let result = try_spec ?hopeless_against spec in
            (* Gated explicitly, not just by [logf]: [logf]'s arguments are evaluated whether or not
               the flag is on, and both readings here fold a hashtable. *)
            if log_enabled () then
              logf "census after %s: %s | device %.1f MiB" (spec_label spec)
                (Ir.Alloc_census.to_string (Ir.Alloc_census.snapshot ()))
                (Float.of_int (Context.get_used_memory search_ctx) /. 1048576.);
            result
          in
          (* gh-ocannl-1061: where the search is, on the [autotune_progress] stream. A phase is the
             seed pass, the recombination composites that follow it, the composite playoff
             (gh-ocannl-1166), or one beam round; its candidate total is known up front except for
             the composites' and the playoff's. *)
          let progress_total = ref None in
          let progress_tried = ref 0 in
          let progress_best () =
            let best_c, best_ms = !best_so_far in
            let label =
              match best_c with
              | None -> if Float.is_finite best_ms then "baseline" else ""
              | Some c ->
                  Option.value (Hashtbl.find label_by_digest c.digest_after) ~default:"baseline"
            in
            Printf.sprintf "best_ms=%s best=%S" (progress_ms best_ms) label
          in
          let progress_where () =
            Printf.sprintf "phase=%s tried=%d/%s timed=%d %s %s" !progress_phase !progress_tried
              (Option.value_map !progress_total ~default:"?" ~f:Int.to_string)
              !n_timed (progress_costs ()) (progress_best ())
          in
          let progress_phase_begin phase total =
            progress_phase := phase;
            progress_total := total;
            progress_tried := 0;
            Option.iter total ~f:(fun n ->
                progress_line "phase"
                  (Printf.sprintf "phase=%s candidates=%d timed=%d %s %s" phase n !n_timed
                     (progress_costs ()) (progress_best ())))
          in
          (* gh-ocannl-1110: the caller's abandonment rule, decided once, as soon as the search has
             [beam_width] admitted timings. Checked as the NEXT attempt starts, so every candidate
             already timed has been admitted (the exit sweep owns it) and none is cut short. *)
          let abandon_decided = ref false in
          let check_abandon () =
            match abandon with
            | Some rule when (not !abandon_decided) && !n_timed >= beam_width -> (
                abandon_decided := true;
                match abandon_verdict rule ~k:beam_width ~steps:!best_steps with
                | Some ab -> raise (Abandon_now ab)
                | None -> ())
            | Some _ | None -> ()
          in
          (* Written as an attempt STARTS, naming it: a search killed inside a long candidate then
             leaves the candidate it was in, and the costs of everything before it. Counted before
             the attempt too, so a fatal candidate -- which writes [search_done] from inside itself
             and raises -- is in the closing record's [attempts]. *)
          let try_spec ?hopeless_against spec =
            check_abandon ();
            if not (budget_allows ~what:(spec_label spec) ~floor:(budget_floor spec)) then (
              last_attempt := `Other;
              None)
            else (
              Int.incr progress_attempts;
              Int.incr progress_tried;
              (* Guarded here, not only inside [progress_line]: its arguments render a label and
                 read the best-so-far table on every attempt of a search that prints nothing. *)
              if progress_enabled () then
                progress_line "candidate"
                  (Printf.sprintf "%s attempt=%S" (progress_where ()) (spec_label spec));
              try_spec ?hopeless_against spec)
          in
          (* The search's seeds for each lowering it seeds ({!search_seeding}, gh-ocannl-1490): the
             family seeds, then the coalesced layout's branch beside them (gh-ocannl-1165) —
             prepared hermetically from the same lowering, its candidates' schedules carry their
             structural [Coalesce] prefix. *)
          let seeding =
            search_seeding
              ~capabilities:(Context.codegen_capabilities ctx)
              ~is_gpu ~is_cpu ~limits ~static_indices base_opt
          in
          let block_size_presets mk =
            mk None
            :: (if is_gpu then List.map seed_block_sizes ~f:(fun bs -> mk (Some bs)) else [])
          in
          (* The model pre-filter of the sketch seeding (gh-ocannl-491 task 3): rank each candidate
             family (the whole-routine sketches; each fission segment's sketches) with the roofline
             model and keep the best [keep_fraction] of the scored candidates before any compilation
             or timing. Only candidates the model fully covers are droppable — a candidate without
             model coverage (opaque code, a schedule the model cannot apply, missing envelope
             constants) is always kept, only measured — so the pre-filter never precludes a measured
             result and its outcome is independent of enumeration order. Presets, saved schedules
             and the baseline are never pruned. A candidate its builder refuses typed
             ([Sketch_families.refuse], gh-ocannl-1390) is dropped here as the decline its compile
             would have been, under the same cause in the census (gh-ocannl-1437): the pre-filter
             builds outside the candidates' failure boundary, where the refusal used to end the
             search. An untyped raise is a builder bug and still propagates. *)
          let model_prefilter_params ~seg_opt ~family params =
            if Float.(keep_fraction >= 1.) || List.length params <= 1 then params
            else
              let scored =
                List.filter_map params ~f:(fun p ->
                    match sketch_schedule ~accum_prec ~p seg_opt with
                    | exception Outcome.Cause_at (phase, cause)
                      when not (Outcome.uncontainable phase cause) ->
                        logf "model pre-filter (%s): %s refused by its builder, declined: %s" family
                          (spec_label (Whole (W_sketch p))) (Outcome.detail_of_cause cause);
                        record_decline declines
                          { Outcome.phase; cause; execution_effect = Outcome.No_device_writes };
                        None
                    | sched -> Some (p, model_score ~static_indices ~limits seg_opt sched))
              in
              n_model_scored :=
                !n_model_scored + List.count scored ~f:(fun (_, s) -> Option.is_some s);
              let kept = model_prefilter ~keep_fraction scored in
              List.iter scored ~f:(fun ((p, s) as entry) ->
                  if not (List.mem kept entry ~equal:phys_equal) then (
                    Int.incr n_model_pruned;
                    logf "model prune (%s, keep %.2f): %s scored %.3e s" family keep_fraction
                      (spec_label (Whole (W_sketch p))) (Option.value_exn s)));
              List.map kept ~f:fst
          in
          let sketch_params =
            model_prefilter_params ~seg_opt:base_opt ~family:"whole-routine" seeding.ss_whole
          in
          n_sketch_candidates := List.length sketch_params;
          n_epilogue_sketch_candidates := List.count sketch_params ~f:(fun p -> p.sk_epilogue);
          (* Per-fission-segment sketch seeds (the [F_sketch] flavor): heavily fissioned graphs tune
             per segment, where the whole-routine sketches never apply. Enumerate the fission
             segmentation once, on a hermetic copy of the base lowering with the same pipeline
             settings the candidate transform uses ([preset_sched]'s defaults), and detect a matmul
             site per keyed segment ({!keyed_segments}): the candidates replay this segmentation and
             find their segments in it by position (gh-ocannl-1243). Enumerated in [seeding]
             above. *)
          let prefilter_entries ~tag entries =
            (* Per-segment pre-filtering: each segment's sketches are their own family —
               cross-segment scores are incomparable (different code volumes), and the singles below
               are also ranked per segment by the recombination step. *)
            List.map entries ~f:(fun (key, pre, ps) ->
                (key, model_prefilter_params ~seg_opt:pre ~family:(tag ^ seg_key_tag key) ps))
          in
          let coarse_plan, coarse_entries = (seeding.ss_coarse_plan, seeding.ss_coarse) in
          (* Each coarse key's pre-schedule segment, which the staffed sweep weighs its key by. *)
          let coarse_pre = List.map coarse_entries ~f:(fun (key, pre, _) -> (key, pre)) in
          let fiss_sketch_entries = prefilter_entries ~tag:"segment " coarse_entries in
          (* The finer ([arity_cuts]) segmentation (gh-ocannl-574): cutting apart a companion that
             cannot follow its site's full arity — the lm_head's max-logits reduction and that
             reduction's initialization nest — frees the site's kernel to seed at full arity, where
             the shared segment's every seed declines on companion coverage. GPU-only: the
             constraint the cut relieves is the GPU sketches' kernel-global launch geometry. Only
             segments whose digest is {e new} versus the coarse segmentation seed singles — an
             unchanged segment's parameters are already timed by its coarse single, and the extra
             cuts elsewhere in a fine twin could only add launches — but the full fine key list is
             kept so the fine recombination below can staff unchanged segments from coarse-timed
             bests. A segment the finer cuts leave unchanged keeps its first unit and its digest, so
             its key is the coarse one. *)
          let fine_plan, fine_all_entries, fine_pre =
            (* Empty off GPU, and when the finer mode cut nothing new: the coarse seeds cover every
               segment. *)
            ( seeding.ss_fine_plan,
              prefilter_entries ~tag:"fine segment " seeding.ss_fine,
              List.map seeding.ss_fine ~f:(fun (key, pre, _) -> (key, pre)) )
          in
          let plan_of ~fine = if fine then fine_plan else coarse_plan in
          let fine_new_entries =
            List.filter fine_all_entries ~f:(fun (k, _) ->
                List.exists seeding.ss_fine_new ~f:(fun (k', _, _) -> equal_seg_key k k'))
          in
          let fiss_sketch_specs =
            (* Single-segment specs: each parameter set of each keyed segment is proposed alone,
               every other segment falling back to its default preset (an absent key degrades to the
               preset in the transform closure). Any zipping of segments' seeds into shared combos —
               index pairing, or pinning the other segments to their first set — lets one segment's
               invalid seed mask another segment's seeds from ever being timed (observed on
               cifar_conv: the fc matmul's invalid packrest-grid seed masked the conv segments'
               row-block seed; and a segment's FIRST seed can itself be the invalid one, e.g. GPU
               conv seeds with a companion tail). Cross-segment combination is recovered below by
               recombining each segment's best-timed single into one composite candidate. *)
            let singles ~plan ~fine entries =
              List.concat_map entries ~f:(fun (key, ps) ->
                  List.map ps ~f:(fun p ->
                      Fiss (F_sketch { plan; entries = [ (key, p) ]; staff = []; fine })))
            in
            ( singles ~plan:coarse_plan ~fine:false fiss_sketch_entries,
              singles ~plan:fine_plan ~fine:true fine_new_entries )
          in
          n_fiss_sketch_candidates :=
            List.length (fst fiss_sketch_specs) + List.length (snd fiss_sketch_specs);
          (* Split-reduce seeds (gh-ocannl-484 task 3), detected on the base lowering — the prelude
             applies whole-routine, so no segment enumeration is needed first — and proposed as
             single-site candidates over a few [num_blocks] values (the tunable of the family; [2*b
             <= extent] keeps chunks at least two elements, below which the split is all combine
             overhead). On GPU the block loop is the bulk of pass 1's launch parallelism at these
             low-output sites, so the sweep leans larger; the CPU pool saturates at core counts.
             Multi-site combination is recovered below by recombining the best-timed singles. *)
          let sr_ranked =
            if is_gpu || is_cpu then split_reduce_sites ~static_indices base_opt else []
          in
          let sr_sites = List.take sr_ranked max_split_reduce_sites in
          (* The candidate-volume cap binding is an eviction, not a judgement about the site: it was
             reachable and ranked, and lost only to the cap. Record each evicted site in the decline
             census — the gh-ocannl-541 blind spot was exactly a previously-seeded site silently
             dropping out of the proposal set when newly-reachable sites filled the cap. *)
          List.iter (List.drop sr_ranked max_split_reduce_sites) ~f:(fun s ->
              let detail =
                Printf.sprintf
                  "site %s red%d out%d cost%d%s%s evicted by autotune_split_reduce_max_sites=%d"
                  (Ir.Tnode.debug_name s.sr_target) s.sr_red s.sr_out s.sr_cost
                  (match List.length s.sr_swaps with 0 -> "" | n -> Printf.sprintf " swap%d" n)
                  (if Option.is_some s.sr_coalesce then " coalesced" else "")
                  max_split_reduce_sites
              in
              logf "split_reduce: %s" detail;
              record_decline declines
                {
                  Outcome.phase = Outcome.Transform;
                  cause = Outcome.Seed_evicted { family = "split_reduce"; detail };
                  execution_effect = Outcome.No_device_writes;
                });
          let sr_num_blocks = if is_gpu then [ 32; 128; 512 ] else [ 8; 32; 128 ] in
          let sr_specs =
            List.concat_map sr_sites ~f:(fun s ->
                List.filter_map sr_num_blocks ~f:(fun b ->
                    if 2 * b <= s.sr_red then
                      Some (Fiss (F_split { sites = [ (s, b) ]; staff = [] }))
                    else None))
          in
          n_split_reduce_candidates := List.length sr_specs;
          (* gh-ocannl-1388: the seeded untuned plan, the step the recipes' untuned compile ships on
             CUDA and HIP ({!seeded_default}), seeded FIRST: wherever the search stops after its
             first timing, or refuses every later window, it crowns that step, so tuned is never
             worse than untuned by construction. Absent where the plan is the default pipeline (no
             seed, split or sink; another backend), which the [config_thresholds] preset seed
             reproduces, so those searches keep their candidate order. *)
          let seeded_specs =
            match Lazy.force seeded_plan_value with
            | None -> []
            | Some plan -> [ Fiss (F_seeded plan) ]
          in
          let coarse_single_specs, fine_single_specs = fiss_sketch_specs in
          (* gh-ocannl-1138: with staffed singles the coarse singles are not seeded as a list but
             swept key by key against the running composite (below), where the list would time them:
             after the whole-routine sketches, before the fine singles. *)
          let staffed = staffed_singles in
          let pre_sweep_specs =
            seeded_specs
            @ block_size_presets (fun block_size -> Whole (W_preset { block_size }))
            @ (if is_gpu || is_cpu then
                 (* Each fissioned preset is seeded plain and privatized (the latter dedups away by
                    digest when no accumulator is eligible). The [config_thresholds] seeds reproduce
                    the untuned default pipeline exactly (plus its privatized variant), so the
                    winner is never worse than not tuning — the aggressive ungated presets can all
                    lose to it on launch-overhead-bound workloads. *)
                 List.concat_map [ false; true ] ~f:(fun privatize ->
                     Fiss (F_preset { block_size = None; privatize; config_thresholds = true })
                     :: block_size_presets (fun block_size ->
                         Fiss (F_preset { block_size; privatize; config_thresholds = false })))
               else [])
            @ List.map sketch_params ~f:(fun p -> Whole (W_sketch p))
          in
          let post_sweep_specs = if staffed then [] else fine_single_specs @ sr_specs in
          let fiss_single_results = ref [] in
          progress_phase_begin "seeds"
            (Some
               (List.length pre_sweep_specs
               + (if staffed then 0 else List.length coarse_single_specs)
               + List.length post_sweep_specs));
          (* gh-ocannl-1527: what a seed's reading is ranked against, for its window's hopeless cut.
             A whole-routine seed -- a preset, or a sketch applied to the whole routine -- feeds
             only the beam: with rounds to run every one of its [beam_width] entries is expanded, so
             once it is full its slowest entry is the reference (a reading over it enters nothing);
             with none to run, only its head is ever read. A per-segment single feeds its key's
             ranking instead -- the recombination's per-key minimum and the playoff's near-ties, at
             the playoff margin -- timed around a backdrop that can be ten times the composite it
             joins (gh-ocannl-1166), so against the search's best every single would look hopeless
             while its rank among its key's singles is the very thing it measures: singles,
             composites and the floors are timed in full. *)
          let seed_hopeless_against spec () =
            match spec with
            | (Whole (W_preset _ | W_sketch _) | Fiss (F_preset _)) when not (budget_floor spec) ->
                if rounds = 0 then Option.map (List.hd !beam) ~f:snd
                else if List.length !beam >= beam_width then Option.map (List.last !beam) ~f:snd
                else None
            | Whole (W_preset _ | W_sketch _ | W_saved _)
            | Fiss (F_preset _ | F_saved _ | F_sketch _ | F_split _ | F_split_saved _ | F_seeded _)
              ->
                None
          in
          let run_seed spec =
            let result = try_spec ~hopeless_against:(seed_hopeless_against spec) spec in
            (match (spec, result) with
            | Fiss (F_sketch { entries = [ (key, p) ]; fine; _ }), Some (_, ms) ->
                fiss_single_results := (key, fine, (p, ms)) :: !fiss_single_results
            | _ -> ());
            Option.iter result ~f:admit
          in
          List.iter pre_sweep_specs ~f:run_seed;
          (* gh-ocannl-1138: the running composite. A single timed with every other segment on its
             preset measures the routine around a backdrop that can be ten times the composite it
             would join (gh-ocannl-1166: 23.2 ms around a 2.4 ms composite on gpt2_mini/CUDA), so
             the search's best stayed flat until the recombination composites it timed last. Instead
             the coarse keys are swept one at a time, each single staffed with the running
             composite's choice for every other key: the composite starts from the seeded untuned
             plan's sketches (matched to the keys by structural digest) and, after a key's sweep,
             takes that key's fastest single. Times are compared only within one key's sweep, where
             the backdrop is the same; the composite's own choice for the key is swept first, at the
             composite's time when that composite was already timed, so a slower replacement leaves
             it as it was. A staffed single that is refused, not merely contended, is retried once
             bare, so one invalid choice in the staff cannot mask the key's seeds; once a key's bare
             retry is refused too, its seeds fail on their own and its refusals stop being
             retried. *)
          let without key choices =
            List.filter choices ~f:(fun (k, _) -> not (equal_seg_key k key))
          in
          (* One running composite over [keys], each with one choice: a segmentation's sketches
             (coarse, or fine: gh-ocannl-574), or the split-reduce sites' block counts. The
             composite, its time when a window timed exactly it, per swept key the staff its sweep
             ran against (for the playoff), and the sweep of one key, which returns that key's
             results in its own backdrop. [single key staff choice] is the candidate; [describe]
             names a choice for the log. Keys compare by [equal_key] and choices structurally. *)
          let sweeper ~equal_key ~keys ~single ~describe ~fallbacks =
            let running = ref [] and running_ms = ref None and swept_against = ref [] in
            let without key choices =
              List.filter choices ~f:(fun (k, _) -> not (equal_key k key))
            in
            let in_key_order choices =
              List.filter_map keys ~f:(fun k ->
                  Option.map (List.Assoc.find choices ~equal:equal_key k) ~f:(fun p -> (k, p)))
            in
            let sweep_key (key, ps) =
              let staff = without key !running in
              let bare = ref [] in
              (* A staffed single of this key admitted shows the staff compatible with it, so any
                 later refusal is the seed's own and is not retried bare (Codex P1 on PR #1215: a
                 cap on the first refused bare retry stopped the retries of a key whose staff was
                 the culprit after one seed that was invalid on its own). Until then, every refusal
                 is retried. *)
              let staff_admitted = ref false in
              swept_against := (key, staff) :: !swept_against;
              let current = List.Assoc.find !running ~equal:equal_key key in
              let ps =
                match current with
                | Some q -> q :: List.filter ps ~f:(fun p -> not (Poly.equal p q))
                | None -> ps
              in
              let results =
                List.filter_map ps ~f:(fun p ->
                    match (current, !running_ms) with
                    | Some q, Some ms when Poly.equal p q ->
                        (* The running composite itself, timed already: not compiled again, and
                           proof the staff is compatible with the key (Codex P2 on PR #1215). *)
                        staff_admitted := true;
                        Some (p, ms)
                    | _ -> (
                        match try_spec (single key staff p) with
                        | Some ((_, ms) as r) ->
                            admit r;
                            staff_admitted := true;
                            Some (p, ms)
                        | None -> (
                            match !last_attempt with
                            | `Dedup (Some ms) ->
                                staff_admitted := true;
                                Some (p, ms)
                            | `Declined when (not (List.is_empty staff)) && not !staff_admitted ->
                                Int.incr fallbacks;
                                logf "%s: refused with its staff; retrying it bare"
                                  (spec_label (single key staff p));
                                (match try_spec (single key [] p) with
                                | Some ((_, ms) as r) ->
                                    admit r;
                                    bare := (p, ms) :: !bare
                                | None -> (
                                    (* A bare twin already timed under its digest still ranks (Codex
                                       P2 on PR #1215). *)
                                    match !last_attempt with
                                    | `Dedup (Some ms) -> bare := (p, ms) :: !bare
                                    | _ -> ()));
                                None
                            | _ -> None)))
              in
              let by_time (_, a) (_, b) = Float.compare a b in
              (* The incumbent in this backdrop: the composite's own choice for the key as timed
                 here, or else the running composite itself, timed whole -- for a key with no
                 choice, the key on its preset (Codex P1 on PR #1215: a key whose every single was
                 slower than its preset used to join anyway, and later keys were swept around the
                 slower composite). *)
              let current_ms =
                Option.bind current ~f:(fun q -> List.Assoc.find results ~equal:Poly.equal q)
              in
              let incumbent_ms = Option.first_some current_ms !running_ms in
              (match List.min_elt results ~compare:by_time with
              | Some (p, ms) ->
                  (* The fastest in this backdrop replaces the incumbent only by being faster; a tie
                     keeps it. *)
                  if Option.for_all incumbent_ms ~f:(fun i -> Float.(ms < i)) then (
                    logf "staffed: %s joins the running composite (%.4f ms%s)" (describe key p) ms
                      (Option.value_map incumbent_ms ~default:"" ~f:(Printf.sprintf " < %.4f ms"));
                    running := in_key_order ((key, p) :: without key !running);
                    running_ms := Some ms)
                  else Option.iter current_ms ~f:(fun q_ms -> running_ms := Some q_ms)
              | None -> (
                  (* Every staffed single of the key was refused, and some of its bare retries were
                     admitted (Codex P1 on PR #1215): the key cannot join this staff, since the
                     composite holding both is exactly the code refused. A bare single IS the
                     composite of its key alone, a whole routine like the running composite, so the
                     faster of the two carries on; a running composite never timed as a whole
                     yields. *)
                  match List.min_elt !bare ~compare:by_time with
                  | Some (p, ms)
                    when Option.value_map !running_ms ~default:true ~f:(fun r -> Float.(ms < r)) ->
                      logf "staffed: %s replaces the running composite, refused with it (%.4f ms%s)"
                        (describe key p) ms
                        (Option.value_map !running_ms ~default:"" ~f:(Printf.sprintf " < %.4f ms"));
                      running := [ (key, p) ];
                      running_ms := Some ms
                  | Some _ | None -> ()));
              (* What the key's sweep timed in one backdrop: its staffed singles, or failing those
                 its bare retries. *)
              if List.is_empty results then List.rev !bare else results
            in
            let start choices = running := in_key_order choices in
            (running, running_ms, swept_against, start, sweep_key)
          in
          (* A segmentation's sketch sweeper. Heaviest key first: its weight is its segment count
             times the roofline score of its best-scored sketch, so the keys whose geometries differ
             most, whose gaps can absorb a noisy backdrop, are decided first, and the near-tie keys
             last, in the smallest backdrop. A key the model cannot score keeps its enumeration
             order, after the scored ones. *)
          let sketch_sweeper ~plan ~fine ~keys =
            let running, running_ms, swept_against, start, sweep_key =
              sweeper ~equal_key:equal_seg_key ~keys ~fallbacks:n_backdrop_fallbacks
                ~single:(fun key staff p ->
                  Fiss (F_sketch { plan; entries = [ (key, p) ]; staff; fine }))
                ~describe:(fun key p ->
                  Printf.sprintf "%ssegment %s: %s"
                    (if fine then "fine " else "")
                    (seg_key_tag key) (spec_label (Whole (W_sketch p))))
            in
            let sweep ~pre_of ~start:choices entries =
              start choices;
              if not (List.is_empty !running) then
                logf "staffed: the running %scomposite starts from %s"
                  (if fine then "fine " else "")
                  (fiss_sketch_label ~fine !running);
              let weighed =
                List.map entries ~f:(fun ((key, ps) as entry) ->
                    let weight =
                      Option.bind (List.Assoc.find pre_of ~equal:equal_seg_key key) ~f:(fun pre ->
                          (* A sketch that does not construct here is left unscored, as the
                             candidate compile would decline it. *)
                          List.filter_map ps ~f:(fun p ->
                              match sketch_schedule ~accum_prec ~p pre with
                              | exception Outcome.Cause_at _ -> None
                              | sched -> model_score ~static_indices ~limits pre sched)
                          |> List.min_elt ~compare:Float.compare
                          |> Option.map ~f:(fun s -> Float.of_int (List.length key.key_units) *. s))
                    in
                    logf "staffed: %ssegment %s weighs %s"
                      (if fine then "fine " else "")
                      (seg_key_tag key)
                      (Option.value_map weight ~default:"unscored" ~f:(Printf.sprintf "%.6e"));
                    (entry, weight))
              in
              let scored, unscored =
                List.partition_tf weighed ~f:(fun (_, w) -> Option.is_some w)
              in
              List.iter
                (List.stable_sort scored ~compare:(fun (_, a) (_, b) ->
                     Option.compare Float.compare b a)
                @ unscored)
                ~f:(fun (((key, _) as entry), _) ->
                  let results = sweep_key entry in
                  fiss_single_results :=
                    List.rev_map results ~f:(fun (p, ms) -> (key, fine, (p, ms)))
                    @ !fiss_single_results)
            in
            (running, running_ms, swept_against, sweep)
          in
          (* The seeded plan's choice for a key: its decision rule ({!seeded_params}, probed and
             validated as {!seeded_decisions} does) applied to the key's own segment. The plan's
             recorded decisions cannot be matched to the keys: they were taken on the split routine,
             under the untuned pipeline's segmentation without the search's zero fold, so a recorded
             digest equals no key's (on gpt2_mini/CUDA the composite started empty), and two keys of
             one structure but different placements would share a digest's decision (Codex P2 on PR
             #1215). The rule is the plan's whole gate ({!seeded_segment_params}: a threshold, or
             gh-ocannl-1253's ranked [auto]); where it seeds nothing
             ([gpu_default_sketch_min_reduction=0], another backend) no key has a start. *)
          let seeded_choice ~pre_of key =
            Option.bind (Lazy.force seeded_plan_value) ~f:(fun plan ->
                Option.bind plan.sp_start ~f:(fun start ->
                    Option.bind (List.Assoc.find pre_of ~equal:equal_seg_key key) ~f:(fun pre ->
                        Option.map (start.decide pre) ~f:(fun p -> (key, p)))))
          in
          let running, running_ms, swept_against, sweep =
            sketch_sweeper ~plan:coarse_plan ~fine:false ~keys:(List.map fiss_sketch_entries ~f:fst)
          in
          let fine_running, fine_running_ms, fine_swept_against, fine_sweep =
            sketch_sweeper ~plan:fine_plan ~fine:true ~keys:(List.map fine_all_entries ~f:fst)
          in
          (* The split-reduce sites' running composite: each site's block count, swept over the
             sites in [sr_sites]' cost order (heaviest first already), each single's prelude staffed
             with the other sites' choices. It starts empty, not from the seeded plan's split: every
             segment of a split candidate runs its preset whatever the prelude, so the seeded split
             would not shrink the backdrop the way the seeded sketches do, and its sites, detected
             on the pristine lowering, would match the search's only by target and extents. *)
          let same_site a b = Idx.equal_symbol a.sr_axis b.sr_axis in
          let sr_running, sr_running_ms, _, _, sr_sweep_key =
            sweeper ~equal_key:same_site ~keys:sr_sites ~fallbacks:n_sr_backdrop_fallbacks
              ~single:(fun s staff b -> Fiss (F_split { sites = [ (s, b) ]; staff }))
              ~describe:(fun s b -> "split site " ^ split_site_label (s, b))
          in
          (* Its own phase, with no total announced: a bare retry or a start choice outside the menu
             adds an attempt, and a start choice already timed takes one away (Codex P2 on PR
             #1215). *)
          if staffed then progress_phase_begin "staffed" None;
          if staffed then (
            sweep ~pre_of:coarse_pre
              ~start:
                (Seam.get on_staffed_start ~menu:fiss_sketch_entries
                   (List.filter_map fiss_sketch_entries ~f:(fun (key, _) ->
                        seeded_choice ~pre_of:coarse_pre key)))
              fiss_sketch_entries;
            (* The fine segmentation's own composite (gh-ocannl-574), swept over the keys the finer
               cuts made new. A key the cuts left unchanged keeps its digest and first unit, so it
               is the coarse key and carries the coarse composite's choice; a new one starts from
               the seeded plan's, where its refinement cut the same segment. *)
            if not (List.is_empty fine_new_entries) then
              fine_sweep ~pre_of:fine_pre
                ~start:
                  (List.filter_map fine_all_entries ~f:(fun (key, _) ->
                       (* An unchanged key is the coarse key: its coarse sweep decided it, a key
                          left on its preset included, and the fine sweep does not revisit it (Codex
                          P2 on PR #1215). Only a new key takes the seeded rule's start. *)
                       if List.Assoc.mem fine_new_entries key ~equal:equal_seg_key then
                         seeded_choice ~pre_of:fine_pre key
                       else
                         Option.map (List.Assoc.find !running ~equal:equal_seg_key key) ~f:(fun p ->
                             (key, p))))
                fine_new_entries)
          else List.iter coarse_single_specs ~f:run_seed;
          List.iter post_sweep_specs ~f:run_seed;
          if staffed && not (List.is_empty sr_specs) then
            List.iter sr_sites ~f:(fun s ->
                match List.filter sr_num_blocks ~f:(fun b -> 2 * b <= s.sr_red) with
                | [] -> ()
                | blocks -> ignore (sr_sweep_key (s, blocks) : (int * float) list));
          (* gh-ocannl-1388: a refused window drops its digest from [seen] so that an equivalent
             later seed can measure it, but no later seed reproduces the seeded plan, so its floor
             would be lost to one contended window. Retried once here, after the seed pass. Still
             unmeasured after that, [seeded_ms] stays [None] and says so. *)
          List.iter seeded_specs ~f:(fun spec ->
              if
                Option.is_none (seeded_ms ())
                && Option.exists !seeded_seed_digest ~f:(Hash_set.mem contended_digests)
              then (
                logf "%s: its window was refused; retrying the seeded plan once" (spec_label spec);
                Option.iter (try_spec spec) ~f:admit));
          (* At most three composites, each proposed only if its singles justify it: no total to
             announce, so no phase line either; a composite's attempt still counts and can print. *)
          progress_phase_begin "recombine" None;
          Option.iter (seeded_ms ()) ~f:(fun ms ->
              logf "seeded untuned plan: %.4f ms (gh-ocannl-1388 floor)" ms);
          (match default_ms () with
          | Some ms -> logf "untuned-default pipeline: %.4f ms (gh-ocannl-552 reference)" ms
          | None ->
              logf
                "untuned-default pipeline: not timed (gated to a form outside the pool, not \
                 seeded, failed, or not dispatched)");
          (* Cross-segment recombination: the singles time every parameter set unmasked, but the
             best full routine may sketch several segments at once. One extra composite candidate
             applies each keyed segment's best-timed single simultaneously — informed by the
             singles' own timings, where the full cartesian product would be exponential. *)
          let best_single_for ~fine_ok key =
            List.filter !fiss_single_results ~f:(fun (k, fine, _) ->
                equal_seg_key k key && (fine_ok || not fine))
            |> List.min_elt ~compare:(fun (_, _, (_, a)) (_, _, (_, b)) -> Float.compare a b)
            |> Option.map ~f:(fun (_, _, (p, _)) -> (key, p))
          in
          (* Staffed, the running composite is the recombination: each key's choice is its fastest
             single in its own sweep's backdrop, where per-key minima over every backdrop would
             compare times taken around different composites. *)
          let recombined =
            if staffed then !running
            else
              List.filter_map fiss_sketch_entries ~f:(fun (key, _) ->
                  best_single_for ~fine_ok:false key)
          in
          (* A staffed composite window admitted during the sweep stays the outcome, published there
             (Codex P2 on PR #1215): the running composite can shrink back to one key after it, when
             a later key's bare single replaced it, and the window still completed. *)
          let composite_window_timed = staffed && Poly.equal !fs_composite `Timed in
          (* Missing singles justify undecided only when every missing coarse key had a viable
             window refused and no later equivalent single supplied a measurement. A refusal from an
             unrelated segment or family cannot excuse a missing proposal. *)
          fs_composite :=
            if List.length recombined >= 2 then `Proposed
            else if
              List.length fiss_sketch_entries >= 2
              && List.for_all fiss_sketch_entries ~f:(fun (key, _) ->
                  List.Assoc.mem recombined key ~equal:equal_seg_key
                  || Hash_set.mem coarse_single_refused key
                     && not (Hash_set.mem coarse_single_measured key))
            then `Singles_refused
            else `Ineligible;
          (* A timed composite, as the playoff's starting incumbent: its entries, its segmentation
             and its time. *)
          let try_composite ~fine entries =
            Option.map
              (try_spec (Fiss (F_sketch { plan = plan_of ~fine; entries; staff = []; fine })))
              ~f:(fun ((_, ms) as r) ->
                admit r;
                (entries, fine, ms))
          in
          (* Staffed, the last accepted window already timed the running composite -- the last key's
             winning staffed single schedules every key's choice -- so it is the coarse composite's
             outcome, stated here rather than re-derived from a label. Only a composite no window
             has timed yet (the seeded plan's choices, every sweep refused) is tried. *)
          let coarse_composite =
            if not (Poly.equal !fs_composite `Proposed) then None
            else
              match (staffed, !running_ms) with
              | true, Some ms ->
                  fs_composite := `Timed;
                  Some (recombined, false, ms)
              | true, None | false, _ -> try_composite ~fine:false recombined
          in
          if composite_window_timed then fs_composite := `Timed;
          (* The fine composite (gh-ocannl-574): the fine winner in a multi-segment routine needs
             the freed site's best AND the other segments' bests in one candidate. Keys address the
             fine segmentation; segments unchanged by the finer cuts share their digest with the
             coarse segmentation, so their coarse-timed bests staff the composite directly (the
             segment code behind a digest is identical, hence the parameters transfer). Proposed
             only when a fine single was actually timed — otherwise the composite is the coarse one
             plus extra launches. *)
          let fine_recombined =
            if List.exists !fiss_single_results ~f:(fun (_, fine, _) -> fine) then
              List.filter_map fine_all_entries ~f:(fun (key, _) ->
                  best_single_for ~fine_ok:true key)
            else []
          in
          (* Staffed, the fine sweep's composite: timed by its last accepted window, which a fine
             single's acceptance is the only way to set, so a fine single was timed. *)
          let fine_composite =
            match (staffed, !fine_running_ms) with
            | true, Some ms when List.length !fine_running >= 2 -> Some (!fine_running, true, ms)
            | true, _ -> None
            | false, _ ->
                if List.length fine_recombined >= 2 then try_composite ~fine:true fine_recombined
                else None
          in
          (* Multi-site split-reduce recombination: apply each detected site's best-timed
             [num_blocks] simultaneously — the sites are distinct statements, so their preludes
             compose. Same rationale as the sketch recombination above: singles keep every value
             unmasked, one composite recovers the combination. *)
          let recombined =
            List.filter_map sr_sites ~f:(fun s ->
                List.filter !sr_single_results ~f:(fun (s2, _, _) ->
                    Idx.equal_symbol s2.sr_axis s.sr_axis)
                |> List.min_elt ~compare:(fun (_, _, a) (_, _, b) -> Float.compare a b)
                |> Option.map ~f:(fun (s2, b, _) -> (s2, b)))
          in
          (* Staffed, the running split composite, timed by its last accepted window when that
             schedules every site. *)
          let recombined = if staffed then !sr_running else recombined in
          (* As for the sketch composite above, a staffed multi-site window admitted during the
             sweep keeps the outcome eligible and timed, whatever the running composite ended as. *)
          let eligible_now = List.length recombined >= 2 in
          sr_composite_eligible := (staffed && !sr_composite_eligible) || eligible_now;
          if eligible_now then
            if staffed && Option.is_some !sr_running_ms then sr_composite_timed := true
            else Option.iter (try_spec (Fiss (F_split { sites = recombined; staff = [] }))) ~f:admit;
          (* The composite playoff (gh-ocannl-1166), after every recombination composite so the
             [playoff] progress phase holds only its windows: starting from the faster timed
             composite, each key in turn tries its near-tie singles in the composite's place, and a
             faster alternate becomes the incumbent the later keys build on. Coordinate descent over
             the keys, at most [playoff_width] alternates each, so the cost is linear in the keyed
             segments rather than the cartesian product the recombination avoids. *)
          let playoff_contenders ~fine_ok ~current key =
            let singles =
              List.filter_map !fiss_single_results ~f:(fun (k, fine, (p, ms)) ->
                  if equal_seg_key k key && (fine_ok || not fine) then Some (p, ms) else None)
              |> List.sort ~compare:(fun (_, a) (_, b) -> Float.compare a b)
            in
            match singles with
            | [] -> []
            | (_, best_ms) :: _ ->
                List.fold singles ~init:[] ~f:(fun acc (p, ms) ->
                    if
                      Float.(ms <= best_ms *. (1. +. playoff_margin))
                      && (not (Poly.equal p current))
                      && not (List.mem acc p ~equal:Poly.equal)
                    then p :: acc
                    else acc)
                |> List.rev |> Fn.flip List.take playoff_width
          in
          (* With width 0 the playoff, its progress phase included, is skipped outright: the search
             times exactly what it timed before the playoff existed (gh-ocannl-1259). *)
          (match
             List.min_elt
               (List.filter_opt [ coarse_composite; fine_composite ])
               ~compare:(fun (_, _, a) (_, _, b) -> Float.compare a b)
           with
          | _ when playoff_width = 0 -> ()
          | None -> ()
          | Some (entries, fine, ms) ->
              progress_phase_begin "playoff" None;
              in_playoff := true;
              Exn.protect
                ~finally:(fun () -> in_playoff := false)
                ~f:(fun () ->
                  ignore
                    (List.fold entries ~init:(entries, ms)
                       ~f:(fun (incumbent, incumbent_ms) (key, _) ->
                         let current = List.Assoc.find_exn incumbent ~equal:equal_seg_key key in
                         (* gh-ocannl-1138: a key whose staffed sweep ran against exactly this
                            incumbent's other choices timed its contenders inside it already;
                            re-timing them would compile the same code again. *)
                         let swept_here =
                           Option.exists
                             (List.Assoc.find
                                (if fine then !fine_swept_against else !swept_against)
                                ~equal:equal_seg_key key)
                             ~f:(fun staff -> List.equal Poly.equal staff (without key incumbent))
                         in
                         List.fold
                           (if swept_here then [] else playoff_contenders ~fine_ok:fine ~current key)
                           ~init:(incumbent, incumbent_ms)
                           ~f:(fun (incumbent, incumbent_ms) p ->
                             (* What [p] would replace: [current] until an earlier alternate of this
                                key swapped in. *)
                             let replaced =
                               List.Assoc.find_exn incumbent ~equal:equal_seg_key key
                             in
                             let alternate =
                               List.map incumbent ~f:(fun (k, q) ->
                                   if equal_seg_key k key then (k, p) else (k, q))
                             in
                             match try_composite ~fine alternate with
                             | Some (_, _, alt_ms) when Float.(alt_ms < incumbent_ms) ->
                                 Int.incr n_playoff_swaps;
                                 logf "playoff: segment %s: %s replaces %s (%.4f ms < %.4f ms)"
                                   (seg_key_tag key) (spec_label (Whole (W_sketch p)))
                                   (spec_label (Whole (W_sketch replaced)))
                                   alt_ms incumbent_ms;
                                 (alternate, alt_ms)
                             | Some _ | None -> (incumbent, incumbent_ms)))
                      : (seg_key * sketch_params) list * float)));
          (* [None] iff the beam is empty: no candidate timed and the baseline was not eligible (an
             undispatched GPU baseline never enters the beam with a finite rank; a declined one does
             not enter it at all). *)
          let best = ref (List.hd !beam) in
          (* gh-ocannl-1635: the entries a round's decision set aside -- the beam an accepted round
             replaced, the round a rejected one discarded -- oldest first. That decision compares
             the round's readings with the incumbent's, taken in earlier windows, so the beam finals
             below re-time the near-ties among them beside the final beam. *)
          let displaced = ref [] in
          let continue_ = ref true in
          while
            !continue_ && !rounds_run < rounds
            && budget_allows ~what:(Printf.sprintf "round %d" (!rounds_run + 1)) ~floor:false
          do
            Int.incr rounds_run;
            let head = Option.map (List.hd !beam) ~f:fst in
            let per_entry =
              List.mapi !beam ~f:(fun i (elem, ms) ->
                  (* On a GPU backend the beam can hold an incumbent that was never dispatched — the
                     serial baseline, whose [infinity] rank keeps it in the pool when fewer than
                     [beam_width] candidates were timed. Expanding it is worthwhile only through the
                     moves that can bind a hardware dimension (the [Tensorize] path the sketch
                     comments describe); every other move provably yields another undispatchable
                     candidate, which [try_spec]'s dispatchability skip drops after paying for its
                     transform, codegen, compile and link (16 such compiles per round on the
                     gh-ocannl-543 chain). Pruned moves are still counted in the census, so the
                     refusal stays visible where it was before. *)
                  let elem_dispatchable = dispatchable ~is_gpu elem.all_opts in
                  (* Passed INTO [menu] so the refusal precedes its per-unit cap (gh-ocannl-685
                     review): applied afterwards, the cap would first share its 48 slots across
                     categories whose moves this predicate is about to reject, and a tensorize-rich
                     unit would lose the very proposals that are the beam's only route out of an
                     undispatchable incumbent. The census recording is unchanged and still happens
                     per refused move, so the refusal stays as visible as it was. *)
                  let admits op =
                    if elem_dispatchable || optop_can_bind_hardware op then true
                    else (
                      logf "menu prune (cannot parallelize an undispatched incumbent): %s"
                        (optop_family op);
                      record_not_dispatched ~origin:"beam_move"
                        ~detail:
                          (Printf.sprintf
                             "%s on an incumbent binding no hardware dimension cannot bind one \
                              either"
                             (optop_family op));
                      false)
                  in
                  (* gh-ocannl-1229 (see [round_expansion]): the units this entry's moves extend.
                     The head always expands in full. Units outside the scope are not even offered
                     to [menu], so they cost no legality check and leave no census record. *)
                  let differing =
                    match (round_expansion, head) with
                    | Expand_differing, Some head when i > 0 -> segments_differing head elem
                    | _ -> None
                  in
                  let expanded (u : unit_gen) =
                    match (round_expansion, differing) with
                    | _ when i = 0 -> true
                    | Expand_full, _ | Expand_differing, None -> true
                    | Expand_head, _ -> false
                    | Expand_differing, Some segs ->
                        Option.exists u.u_segment ~f:(List.mem segs ~equal:Int.equal)
                  in
                  (* Per unit, in [elem.units]' order: whether the rule offered it to [menu], and
                     the round candidates its moves became (gh-ocannl-1360). *)
                  let per_unit =
                    List.map elem.units ~f:(fun u ->
                        if expanded u then
                          ( u,
                            true,
                            List.filter_map
                              (menu ~admits ~is_cpu ~is_gpu ~limits ~registry:u.u_registry u.u_opt)
                              ~f:(fun op -> extend_spec elem u op) )
                        else (u, false, []))
                  in
                  let entry =
                    {
                      entry_digest_tag = dshort elem.digest_after;
                      entry_ms = ms;
                      entry_segments_differing =
                        Option.bind head ~f:(fun head -> segments_differing head elem);
                      entry_units =
                        List.map per_unit ~f:(fun ((u : unit_gen), unit_expanded, c) ->
                            {
                              unit_segment = u.u_segment;
                              unit_expanded;
                              unit_moves = List.length c;
                            });
                    }
                  in
                  ( elem,
                    ms,
                    List.count per_unit ~f:(fun (_, e, _) -> e),
                    List.concat_map per_unit ~f:(fun (_, _, c) -> c),
                    entry ))
            in
            let cands = List.concat_map per_entry ~f:(fun (_, _, _, c, _) -> c) in
            (* Recorded before any candidate is attempted, so a report emitted while the round is in
               flight -- a fatal failure's, an abandonment's -- already carries it. *)
            round_moves :=
              {
                round_index = !rounds_run;
                round_expansion_rule = round_expansion_name round_expansion;
                round_entries = List.map per_entry ~f:(fun (_, _, _, _, e) -> e);
              }
              :: !round_moves;
            (* gh-ocannl-1137: what the round expands. The beam is the [beam_width] fastest timed
               candidates, and a near-tie sweep (the gh-ocannl-1166 composite playoff, a block-size
               ladder) can fill it with variants of one schedule; [units differing] counts each
               entry's units whose saved schedule differs from the incumbent's, so a beam of
               near-copies reads as a column of small numbers, and [units expanded] says how many of
               its units the round's [round_expansion] rule offered to the menu (gh-ocannl-1229). *)
            (match per_entry with
            | [] -> ()
            | (head, _, _, _, _) :: _ ->
                logf "round %d: %d candidate(s) from a beam of %d, expansion %s: %s" !rounds_run
                  (List.length cands) (List.length per_entry)
                  (round_expansion_name round_expansion)
                  (String.concat ~sep:"; "
                     (List.map per_entry ~f:(fun (elem, ms, n_units, c, _) ->
                          Printf.sprintf
                            "%s %.4f ms, %s units differing, %d of %d units expanded, %d move(s)"
                            (dshort elem.digest_after) ms
                            (Option.value_map (units_differing head elem) ~default:"resegmented"
                               ~f:Int.to_string)
                            n_units (List.length elem.units) (List.length c)))));
            (* Ranked like the seed pass, but in a SECOND accumulator, because a round's decision
               compares its own best against the incumbent and, if it wins, replaces the beam
               wholesale. Its entries were released at their timings, like every candidate's. *)
            round := [];
            let round_admit entry =
              let kept, _evicted =
                List.split_n (List.sort (entry :: !round) ~compare:by_time) beam_width
              in
              round := kept
            in
            progress_phase_begin (Printf.sprintf "round%d" !rounds_run) (Some (List.length cands));
            (* gh-ocannl-1527: a round candidate feeds only the round's own accumulator, which keeps
               its [beam_width] fastest; once that is full, a reading over its slowest enters
               nothing, so that is its window's hopeless reference. *)
            let round_hopeless_against () =
              if List.length !round >= beam_width then Option.map (List.last !round) ~f:snd
              else None
            in
            List.iter cands ~f:(fun spec ->
                Option.iter (try_spec ~hopeless_against:round_hopeless_against spec) ~f:round_admit);
            match !round with
            | [] -> continue_ := false
            | (_, round_best_ms) :: _ ->
                let incumbent_ms = Option.value_map !best ~default:Float.infinity ~f:snd in
                if Float.(round_best_ms < incumbent_ms *. (1. -. min_progress)) then (
                  displaced := !displaced @ !beam;
                  beam := !round;
                  best := List.hd !beam)
                else (
                  (* The round did not beat the incumbent by enough: the beam is unchanged, and the
                     search stops. *)
                  displaced := !displaced @ !round;
                  continue_ := false;
                  round := [])
          done;
          (* gh-ocannl-1635: the beam finals ([finals_contenders]). The pool is every entry the
             search ranked as an incumbent: the final beam first, then [displaced], each entry once.
             Its contenders are recompiled from their saved forms -- every candidate's buffers went
             at its timing (gh-ocannl-1326) -- in [finals_order], one compile per run of equal
             contenders and one resident at a time, each window under the contender's own label and
             digest and after the timing-state restore every window starts from (gh-ocannl-1621).
             The crown then follows the finals' readings alone; [best_ms] stays the crown's search
             reading, the number its label, spread and digest were recorded with. A refused or
             failed finals window is the finals' own: it is not a candidate refusal or decline, so
             it neither enters [timings_contended] nor makes the search uncacheable -- a contender
             left with no reading makes the finals inconclusive, and the crown is then the first
             readings' head, as without finals. The finals run only while the budget allows a
             window, and a window stops once it passes, without marking the search as cut: every
             candidate the menu proposed was already timed. *)
          (let finals_pool =
             let seen_digests = Hash_set.create (module String) in
             List.filter (!beam @ !displaced) ~f:(fun ((c : compiled), _) ->
                 if Hash_set.mem seen_digests c.digest_after then false
                 else (
                   Hash_set.add seen_digests c.digest_after;
                   true))
           in
           let contenders = finals_contenders ~margin:finals_margin ~width:beam_width finals_pool in
           let label_of (c : compiled) =
             Option.value
               (Hashtbl.find label_by_digest c.digest_after)
               ~default:(dshort c.digest_after)
           in
           let finals_stop () =
             match budget with Some b -> Float.(budget_clock () > b) | None -> false
           in
           let budget_allows_finals () =
             match budget with
             | None -> true
             | Some b -> (not !budget_spent) && Float.(budget_clock () < b)
           in
           if not (List.is_empty contenders) then
             if not (budget_allows_finals ()) then
               logf "finals: skipped, the budget is spent (%d contender(s))"
                 (List.length contenders)
             else (
               logf "finals: %d contender(s) re-timed together: %s" (List.length contenders)
                 (String.concat ~sep:"; "
                    (List.map contenders ~f:(fun (c, ms) ->
                         Printf.sprintf "%s %.4f ms (digest %s)" (label_of c) ms
                           (dshort c.digest_after))));
               (* No total, like the playoff's: a finals window is no attempt, so no candidate line
                  follows the phase. *)
               progress_phase_begin "finals" None;
               let readings = Hashtbl.create (module String) in
               let groups =
                 List.group (finals_order contenders) ~break:(fun ((a : compiled), _) (b, _) ->
                     not (String.equal a.digest_after b.digest_after))
               in
               List.iter groups ~f:(fun group ->
                   let entry, _ = List.hd_exn group in
                   let label = label_of entry in
                   match
                     timed_into progress_compile_s (fun () ->
                         compile_spec (spec_of_form entry.form))
                   with
                   | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
                   | Error (Outcome.Classified classified) ->
                       logf "finals: %s: recompile FAILED at %s %s" label
                         (phase_label classified.phase)
                         (Outcome.detail_of_cause classified.cause)
                   | Ok c when not (String.equal c.digest_after entry.digest_after) ->
                       logf "finals: %s: recompiled to digest %s, not %s: not re-timed" label
                         (dshort c.digest_after) (dshort entry.digest_after);
                       release_quietly ~what:("finals contender " ^ label) c.cctx
                   | Ok c ->
                       pending := Some c;
                       let rec windows k =
                         if k > 0 then (
                           Int.incr n_finals_timed;
                           match timing_window ~label ~stop:finals_stop c with
                           | Ok (`Timed timing_result) -> (
                               match admitted_timing_ms timing_result with
                               | Some ms ->
                                   Int.incr n_finals_admitted;
                                   Hashtbl.add_multi readings ~key:c.digest_after ~data:ms;
                                   logf "finals: %s: %.4f ms (digest %s)" label ms
                                     (dshort c.digest_after);
                                   windows (k - 1)
                               | None ->
                                   logf "finals: %s: NOT TIMED, %s (digest %s)" label
                                     (refusal_reason (timing_admission timing_result))
                                     (dshort c.digest_after);
                                   windows (k - 1))
                           | Ok `Stopped ->
                               logf "finals: %s: NOT TIMED, the budget passed during its timing"
                                 label
                           | Ok (`Hopeless _) ->
                               (* Unreachable: a finals window is timed in full. *)
                               logf "finals: %s: NOT TIMED, cut as hopeless" label
                           | Error (Outcome.Classified classified) -> (
                               logf "finals: %s: RUN FAILED at %s %s" label
                                 (phase_label classified.phase)
                                 (Outcome.detail_of_cause classified.cause);
                               (* As a candidate's run failure (see [try_spec]): one that wrote
                                  nothing withdraws its executed claim; one that may have written
                                  condemns the lineage. *)
                               match classified.execution_effect with
                               | Outcome.No_device_writes ->
                                   Context.rollback_execution c.cctx c.routine.Context.routine_id
                               | Outcome.Writes_may_have_occurred ->
                                   Context.poison_lineage c.cctx
                                     ~routine_name:c.routine.Context.name
                                     (Outcome.exception_of_cause classified.cause);
                                   release_candidate c;
                                   emit_partial_and_raise
                                     (Outcome.fatal_of_classified ~candidate:label classified))
                           | Error (Outcome.Fatal fatal) ->
                               Context.poison_lineage c.cctx ~routine_name:c.routine.Context.name
                                 fatal.Outcome.exn;
                               release_candidate c;
                               emit_partial_and_raise fatal)
                       in
                       windows (List.length group);
                       release_candidate c);
               finals_record :=
                 List.map contenders ~f:(fun (c, ms) ->
                     {
                       contender_label = label_of c;
                       contender_search_ms = ms;
                       contender_finals_ms =
                         List.min_elt
                           (Hashtbl.find_multi readings c.digest_after)
                           ~compare:Float.compare;
                     });
               let incumbent_digest =
                 Option.map !best ~f:(fun ((c : compiled), _) -> c.digest_after)
               in
               match
                 finals_crown
                   (List.map contenders ~f:(fun ((c, _) as e) ->
                        (e, Hashtbl.find_multi readings c.digest_after)))
               with
               | None ->
                   finals_outcome := Finals_inconclusive;
                   logf
                     "finals: inconclusive, a contender has no admitted reading: the crown stays %s"
                     (Option.value_map !best ~default:"none" ~f:(fun (c, _) -> label_of c))
               | Some ((((crown : compiled), first_ms) as e), finals_ms) ->
                   let kept = Option.exists incumbent_digest ~f:(String.equal crown.digest_after) in
                   finals_outcome := if kept then Finals_kept else Finals_swapped;
                   logf "finals: %s %s (%.4f ms in the finals, %.4f ms in the search)"
                     (if kept then "kept" else "crowned")
                     (label_of crown) finals_ms first_ms;
                   best := Some e));
          let best_c, best_ms =
            match !best with Some (c, ms) -> (Some c, ms) | None -> (None, Float.infinity)
          in
          (* Nothing was timed exactly when every candidate failed and (on GPU) the serial baseline
             was never run — or, since gh-ocannl-533, was itself declined. Nothing measured means
             nothing to cache: a stored entry would pin future processes to a never-timed
             schedule. *)
          let nothing_timed = Float.is_inf best_ms in
          (* gh-ocannl-1409: see [budget_floor_unmeasured]. *)
          let floor_unmeasured =
            budget_floor_unmeasured ~budget_cut:!budget_cut ~auto_sched ~baseline_dispatched
              ~floor_measured:(Option.is_some (default_ms ()) || Option.is_some (seeded_ms ()))
          in
          (if use_cache then
             if floor_unmeasured then
               logf
                 "the budget cut the search before it measured the untuned default: storing no \
                  cache entry (gh-ocannl-1409)"
             else if
               not
                 (search_measurements_cacheable ~nothing_timed
                    ~timings_contended:!n_timings_contended)
             then
               if nothing_timed then
                 logf "nothing was timed: storing no cache entry (gh-ocannl-532)"
               else
                 logf "%d timing window(s) were refused as unusable: storing no cache entry"
                   !n_timings_contended
             else
               let saved, segments =
                 let best_c = Option.value_exn best_c ~message:timed_winner_exists in
                 match best_c.form with
                 | Whole_saved saved -> (saved, None)
                 | Fiss_saved segs -> ([], Some segs)
                 | Split_saved (prelude, segs) -> (prelude, Some segs)
               in
               SC.store ~dir:cache_dir ~key
                 {
                   SC.version = SC.entry_version;
                   backend;
                   numerics = SC.numerics_tag ();
                   codegen = Some codegen_tag;
                   objective = Some objective;
                   source_digest = base_digest;
                   saved;
                   segments;
                   best_ms;
                   baseline_ms;
                   (* gh-ocannl-579: a measurement of the program, stored like the two above so the
                      flip chain's profitability term reads the same evidence on a warm cache as on
                      the cold run that measured it. Absent when nothing tensorized was timed. *)
                   mma_best_ms = (if Float.is_finite !mma_best_ms then Some !mma_best_ms else None);
                   best_spread = winner_spread best_c;
                   default_ms = default_ms ();
                   default_fingerprint =
                     Option.map (default_ms ()) ~f:(fun _ ->
                         Sched.default_schedule_fingerprint ~backend_name:backend);
                   best_steps =
                     Some
                       { SC.search_shape = Lazy.force search_shape; steps = List.rev !best_steps };
                   budget_cut = !budget_cut;
                 });
          (* Diagnostic control (config [autotune_log], or [tune]'s [?log]): compile and time the
             UNTUNED default pipeline in this very process, on the search context — discriminates a
             genuinely slow winner from process-state effects when the winner's code nominally
             equals the untuned program yet a separately-run untuned process measures faster (PR
             #140 round 6: same digest, 3.4x runtime difference across processes on cuda). *)
          (if log_enabled () then
             match
               progress_stage "untuned_control";
               costed ~attempt:"untuned control"
                 ~verdict_of:(fun _ -> "compiled")
                 (fun () -> Context.compile ?name search_ctx comp bindings)
             with
             | cctx, croutine ->
                 (match
                    before_timing_window ();
                    time_routine ~timing ~repeats cctx croutine
                  with
                 | timing_result -> (
                     match admitted_timing_ms timing_result with
                     | Some ms -> logf "untuned-default in-process control: %.4f ms" ms
                     | None ->
                         logf "untuned-default in-process control: refused (%s)"
                           (refusal_reason (timing_admission timing_result)))
                 | exception exn ->
                     logf "untuned-default control run failed: %s" (Exn.to_string exn));
                 (* A diagnostic's artifacts are dead the moment it has printed its number
                    (gh-ocannl-550) — and the diagnostic is on exactly when the memory question is
                    being measured, so leaving them behind would show up in the very census that
                    reads it. Best-effort, like [release_candidate]: this runs after a timing
                    failure the control deliberately swallowed, and [release] awaits the device, so
                    a backend still reporting that failure must not be allowed to turn a completed
                    search with a valid winner into a fatal one. *)
                 release_quietly ~what:"the untuned-default control" cctx
             | exception exn ->
                 logf "untuned-default control compile failed: %s" (Exn.to_string exn));
          let completed_report =
            {
              outcome = Searched;
              timing;
              candidates_timed = !n_timed;
              timings_contended = !n_timings_contended;
              timings_retried = !n_timings_retried;
              timings_unbatched = !n_timings_unbatched;
              timings_unreadable = !n_timings_unreadable;
              timings_dispersed = !n_timings_dispersed;
              timings_hopeless = !n_timings_hopeless;
              timing_decisions = timing_decisions ();
              candidates_contended = candidates_contended ();
              default_refused = default_refused ();
              candidates_failed = failed_count declines;
              baseline_declined = Option.is_some baseline_decline;
              baseline_refusal = !baseline_refusal;
              declines = decline_summaries declines;
              privatize_drops = decline_summaries privatize_declines;
              cache_replay_refusal = !cache_replay_refusal;
              rounds_run = !rounds_run;
              round_moves = List.rev !round_moves;
              finals = !finals_record;
              finals_timed = !n_finals_timed;
              finals_admitted = !n_finals_admitted;
              finals_outcome = !finals_outcome;
              budget_cut = !budget_cut;
              budget_s;
              call_budget_s;
              call_share_s;
              compiles = 0 (* Stamped by [emit_report]. *);
              replay_compiles = 0;
              beam_width;
              sketch_candidates = List.length sketch_params;
              epilogue_sketch_candidates = List.count sketch_params ~f:(fun p -> p.sk_epilogue);
              fiss_sketch_candidates = !n_fiss_sketch_candidates;
              fiss_sketch_timed = !n_fiss_sketch_timed;
              fiss_sketch_composite = !fs_composite;
              fiss_sketch_playoff_timed = !n_playoff_timed;
              fiss_sketch_playoff_swaps = !n_playoff_swaps;
              fiss_sketch_staffed_timed = !n_staffed_timed;
              fiss_sketch_backdrop_fallbacks = !n_backdrop_fallbacks;
              split_reduce_candidates = List.length sr_specs;
              split_reduce_timed = !n_sr_timed;
              split_reduce_composite_eligible = !sr_composite_eligible;
              split_reduce_composite_timed = !sr_composite_timed;
              split_reduce_backdrop_fallbacks = !n_sr_backdrop_fallbacks;
              mma_candidates = !n_mma_proposed;
              fiss_mma_candidates = !n_fiss_mma_proposed;
              mma_timed = !n_mma_timed;
              model_scored = !n_model_scored;
              model_pruned = !n_model_pruned;
              bound_pruned = !n_bound_pruned;
              fissioned = Option.exists best_c ~f:(fun c -> is_fissioned c.form);
              baseline_ms;
              default_ms = default_ms ();
              seeded_ms = seeded_ms ();
              best_ms;
              best_label = winner_label best_c;
              best_tensorized = winner_tensorized best_c;
              best_tensorization =
                Option.map best_c ~f:(fun c -> (mma_summary c).Ir.C_syntax.tensorization);
              best_mma_statements = Option.value_map best_c ~default:0 ~f:mma_statements;
              best_mma_scalar_fallbacks = Option.value_map best_c ~default:0 ~f:mma_scalar_fallbacks;
              mma_best_ms = !mma_best_ms;
              best_spread = winner_spread best_c;
              best_schedule = Option.value_map best_c ~default:[] ~f:(fun c -> flat_schedule c.form);
              source_digest = base_digest;
              best_steps = List.rev !best_steps;
              (* Filled in by [emit_report], like every report's. *)
              compile_cost = Ir.Compile_cost.empty;
            }
          in
          (* What ships, as the compile that produces it: run here unless the call is deferred
             (gh-ocannl-1408). Every branch sweeps the candidates first, so a deferred call leaves
             nothing of its search resident. *)
          let ship =
            if floor_unmeasured && not nothing_timed then (
              logf
                "the budget cut the search before it measured the untuned default: shipping the \
                 untuned default compile (gh-ocannl-1409)";
              release_all_candidates ();
              untuned_default_or_raise)
            else if nothing_timed then (
              (* Returning the incumbent here would hand the caller the very serial routine this
                 search refused to dispatch (gh-ocannl-532) — slower than not tuning at all, and on
                 GPU unbounded. The untuned compile is the honest fallback: the same code the caller
                 would have compiled without the tuner, the seeded plan where there is one. *)
              logf "nothing was timed: falling back to the untuned default compile (gh-ocannl-532)";
              release_all_candidates ();
              untuned_default_or_raise)
            else
              (* [nothing_timed] is false, so the beam holds a timed winner. *)
              let best_c = Option.value_exn best_c ~message:timed_winner_exists in
              (* Compile the winner from the caller's context out of its saved schedule, as the
                 cache-hit path does: every candidate's buffers were released at its timing
                 (gh-ocannl-1326), and with [timing_ctx] the search ran against the scratch lineage
                 anyway. Digest mismatch or replay failure falls back to the production default
                 schedule. *)
              let spec = spec_of_form best_c.form in
              (* The rebuilt spec has lost what the winner was timed as: a round candidate's parent
                 and move, a sketch's or preset's parameters -- its saved form alone would print as
                 the bare [F_saved[N segs]] every round candidate of one parent shares. So the
                 replay, and a failure it reports, carry the label the winner was timed under, which
                 is also the report's [best_label]. *)
              let label = match winner_label (Some best_c) with "" -> spec_label spec | l -> l in
              (* Nothing the replay needs is an artifact — [spec] above is the winner's saved
                 schedule — so whatever the search still owns goes before the compile that
                 reproduces it (gh-ocannl-550). *)
              release_all_candidates ();
              fun () ->
                progress_stage "winner_compile";
                match compile_spec_real ~label Outcome.Candidate spec with
                | Ok c when not (shipped_dispatchable ~is_gpu c) ->
                    (* Completes the invariant rather than fixing an observed bug: the winner was
                       timed, so it was dispatchable when measured, and the replay is
                       digest-guarded. But this is the last of the three ways [tune] hands back a
                       routine, and none of them may return an unparallelized GPU routine
                       (gh-ocannl-532). The default compile is the same fallback a failed replay
                       takes. *)
                    logf "winner replay produced an unparallelized routine, falling back: %s" label;
                    (* gh-ocannl-550: rejected, so dead — and the fallback compile below wants the
                       memory. Same one-liner as the rejected cache replay above; the pre-replay
                       sweep could not cover this context, which did not exist yet. *)
                    release_quietly ~what:"the rejected winner replay" c.cctx;
                    substituted := true;
                    untuned_default_or_raise ()
                | Ok c ->
                    logf "winner replay ok: %s" label;
                    (c.cctx, c.routine)
                | Error (Outcome.Classified classified) ->
                    logf "winner replay FAILED (%s), falling back to the default compile: %s" label
                      (Outcome.detail_of_cause classified.cause);
                    (* gh-ocannl-1480: unconditional, not [logf]. The saved form of a candidate this
                       very search timed must replay; when it does not, the search's work is
                       discarded and the untuned default ships, which a caller otherwise learns only
                       from [Train.shipped.ship_substituted] or an [autotune_log] trace. *)
                    Stdio.eprintf
                      "autotune: WARNING: routine %S: the searched winner %s did not replay (%s); \
                       shipping the untuned default in its place\n\
                       %!"
                      (Lazy.force routine_name) label
                      (Outcome.detail_of_cause classified.cause);
                    substituted := true;
                    untuned_default_or_raise ()
                | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
          in
          let result = if defer then ship else Fn.const (ship ()) in
          (* The report describes the routine returned (Codex P1 on PR #1234): a caller ranking arms
             by [best_ms] must not credit this one with a candidate it discarded, and the untuned
             default it ships was never timed -- the nothing-timed report's shape. *)
          let completed_report =
            if floor_unmeasured && not nothing_timed then
              {
                completed_report with
                best_ms = Float.infinity;
                best_label = "";
                best_tensorized = false;
                best_tensorization = None;
                best_mma_statements = 0;
                best_mma_scalar_fallbacks = 0;
                best_schedule = [];
                fissioned = false;
                best_steps = [];
              }
            else completed_report
          in
          (result, completed_report)
        in
        let result, completed_report =
          let escaped ~phase exn backtrace =
            if !partial_emitted then Stdlib.Printexc.raise_with_backtrace exn backtrace
            else emit_partial_and_raise { exn; backtrace; phase; candidate = None; cause = None }
          in
          try search () with
          | Abandon_now ab ->
              logf
                "abandoned after %d timed candidates: best %.4f ms trails the incumbent's %.4f ms \
                 at the same depth by more than %.4gx"
                ab.ab_timed ab.ab_best_ms ab.ab_incumbent_ms ab.ab_ratio;
              (* Built before the sweep, which the report's winner label outlives (see
                 [release_all_candidates]). No winner is crowned, but the clean measured prefix can
                 spare a replay this same search (gh-ocannl-1136). *)
              let r = partial_report (Abandoned ab) in
              release_all_candidates ();
              if
                use_cache
                && search_measurements_cacheable ~nothing_timed:false
                     ~timings_contended:r.timings_contended
              then
                SC.store_abandonment ~dir:cache_dir ~key
                  {
                    SC.version = SC.entry_version;
                    source_digest = base_digest;
                    trajectory = { SC.search_shape = Lazy.force search_shape; steps = r.best_steps };
                  };
              (* The callback's own exception propagates, as on the completion path. *)
              emit_report r;
              raise (Search_abandoned ab)
          (* A raise that carries its phase keeps it: the lineage-wide pre-dispatch validation is
             deliberately raised outside the candidate loop's failure boundary (gh-ocannl-569), so
             it arrives here rather than at a classifier, and reporting it under the [Transform]
             default below would tell the caller a validation error was a transform failure. The
             original exception is re-raised, not the wrapper, so the caller still sees its
             message. *)
          | Outcome.Raised_at (phase, exn, backtrace) -> escaped ~phase exn backtrace
          | exn -> escaped ~phase:Outcome.Transform exn (Stdlib.Printexc.get_raw_backtrace ())
        in
        (* A callback failure on the ordinary completion path is the callback's own exception and
           propagates normally; only fatal-path callbacks are best-effort. But propagating means the
           caller never receives [result], so its buffers become unreachable while the pool table
           keeps rooting them (gh-ocannl-550) — one full winner's footprint per aborted report,
           which for a caller that retries would accumulate exactly like the candidates used to. The
           exit sweep above deliberately kept this one; nothing is keeping it now. *)
        if defer then (
          emit_report completed_report;
          deferred_compile result)
        else (
          report_or_release completed_report ~result:(result ());
          result)

(* [?log] and [?progress] scope [autotune_log] and [autotune_progress] to this call: every
   diagnostic the search emits, the untuned control it gates, the cache-hit and fallback lines, and
   every progress line read {!log_enabled} and {!progress_enabled} inside the scope. *)
let tune ?name ?search ?beam_width ?rounds ?repeats ?timing ?seed_block_sizes ?cache_dir
    ?keep_fraction ?max_split_reduce_sites ?timing_ctx ?abandon ?log ?progress ?report ctx comp
    bindings =
  exclusive_search @@ fun () ->
  with_streams ?log ?progress (fun () ->
      with_timing_state_restored (fun timing_state ->
          tune_in_stream_scope ~timing_state ~defer:false ~substituted:(ref false) ?name ?search
            ?beam_width ?rounds ?repeats ?timing ?seed_block_sizes ?cache_dir ?keep_fraction
            ?max_split_reduce_sites ?timing_ctx ?abandon ?report ctx comp bindings ()))

type deferred = {
  compile : unit -> Context.t * Context.routine;
  substituted : bool ref;
  compiles : int Atomic.t;
}

let tune_deferred ?name ?search ?beam_width ?rounds ?repeats ?timing ?seed_block_sizes ?cache_dir
    ?keep_fraction ?max_split_reduce_sites ?timing_ctx ?abandon ?call_budget ?log ?progress ?report
    ctx comp bindings =
  let substituted = ref false in
  exclusive_search @@ fun () ->
  with_streams ?log ?progress (fun () ->
      {
        substituted;
        compiles = Atomic.make 0;
        compile =
          (* The search runs here, so its lineage is restored here, before the thunk is handed
             back. *)
          with_timing_state_restored (fun timing_state ->
              tune_in_stream_scope ~timing_state ~defer:true ~substituted ?name ?search ?beam_width
                ?rounds ?repeats ?timing ?seed_block_sizes ?cache_dir ?keep_fraction
                ?max_split_reduce_sites ?timing_ctx ?abandon ?call_budget ?report ctx comp bindings);
      })

let compile_deferred d =
  let before = Context.backend_compiles () in
  (* Atomic (review round 3): a memoized call on another domain adds its zero concurrently with the
     compiling call's count. The counter itself is per domain, and the compile runs on the
     caller's. *)
  Exn.protect ~f:d.compile ~finally:(fun () ->
      ignore (Atomic.fetch_and_add d.compiles (Context.backend_compiles () - before) : int))

let deferred_compiles d = Atomic.get d.compiles
let shipped_substitute d = !(d.substituted)
