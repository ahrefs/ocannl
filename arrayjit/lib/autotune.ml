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
   should batch. *)
type timing_result = { ms : float; contended : bool; unbatched : bool; samples : int }

(* Per-candidate search diagnostics on stderr, gated by config [autotune_log]. Kept above the timing
   policy because a cap-bound queued batch reports the target wall it could not reach. *)
let log_enabled =
  lazy
    (match
       String.lowercase
         (String.strip (Utils.get_global_arg ~arg_name:"autotune_log" ~default:"false"))
     with
    | "true" | "1" -> true
    | _ -> false)

let logf fmt =
  Printf.ksprintf (fun s -> if Lazy.force log_enabled then Stdio.eprintf "autotune: %s\n%!" s) fmt

(* gh-ocannl-1061: the search's cost record, gated by config [autotune_progress]. A search can run
   for hours and be killed by a cap before it reports anything, and [autotune_log] is no substitute:
   it pays for an extra untuned-default control compile, so it moves the cost it would be recording.
   These lines cost a clock read per candidate, and each is flushed as it is written, so a kill
   keeps everything up to it.

   The per-candidate line is written at EVERY attempt's start, not at a time-bounded rate: the
   candidate a cap kills the search inside is the one the record most needs to name, and without a
   thread beside the blocking compile or timing call only the line written before it can name it. A
   time bound always leaves a window after the previous line (review rounds 2-3 on PR #817). The
   rate is bounded by the attempts themselves -- each is a candidate compile, a timing window or a
   pruning decision -- and the phase, arm and flip lines by the search's own structure. The format
   is the interface's contract; see {!progressf}. *)
let progress_enabled =
  let on = lazy (Utils.get_global_flag ~default:false ~arg_name:"autotune_progress") in
  fun () -> Lazy.force on

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

(* The one admission gate for a timing verdict, for the consumers that RANK: candidate selection,
   the calibration rows, the roofline consistency check, cache attribution. Keeping it next to the
   result type prevents such a caller from proving only one half of usability (usually [not
   contended]) and accidentally feeding an unresolved zero/NaN clock reading to ranking.

   Deliberately not the gate for [queued_batch_depth] (gh-ocannl-888): a batch depth is a scale
   estimate, not a measurement, and both of its error directions are bounded. *)
let admitted_timing_ms { ms; contended; unbatched; samples = _ } =
  if contended || unbatched || (not (Float.is_finite ms)) || not (Float.is_positive ms) then None
  else Some ms

let timing_string = function Isolated -> "isolated" | Queued -> "queued"

let timing_of_setting s =
  match String.lowercase (String.strip s) with
  | "isolated" -> Isolated
  | "queued" -> Queued
  | other ->
      invalid_arg ("ocannl_autotune_timing setting should be isolated or queued; found: " ^ other)

type report = {
  outcome : outcome;
  candidates_timed : int;
  timings_contended : int;
  timings_retried : int;
  timings_unbatched : int;
  candidates_contended : int;
  default_refused : bool;
  candidates_failed : int;
  baseline_declined : bool;
  declines : decline_summary list;
  rounds_run : int;
  beam_width : int;
  sketch_candidates : int;
  epilogue_sketch_candidates : int;
  fiss_sketch_candidates : int;
  fiss_sketch_timed : int;
  fiss_sketch_composite : [ `Ineligible | `Singles_refused | `Proposed | `Refused | `Timed ];
  fiss_sketch_playoff_timed : int;
  fiss_sketch_playoff_swaps : int;
  split_reduce_candidates : int;
  split_reduce_timed : int;
  split_reduce_composite_eligible : bool;
  split_reduce_composite_timed : bool;
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
  best_ms : float;
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
}

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
    candidates_contended = 0;
    default_refused = false;
    candidates_failed = 0;
    baseline_declined = false;
    declines = [];
    rounds_run = 0;
    beam_width = 0;
    sketch_candidates = 0;
    epilogue_sketch_candidates = 0;
    fiss_sketch_candidates = 0;
    fiss_sketch_timed = 0;
    fiss_sketch_composite = `Ineligible;
    fiss_sketch_playoff_timed = 0;
    fiss_sketch_playoff_swaps = 0;
    split_reduce_candidates = 0;
    split_reduce_timed = 0;
    split_reduce_composite_eligible = false;
    split_reduce_composite_timed = false;
    mma_candidates = 0;
    fiss_mma_candidates = 0;
    mma_timed = 0;
    model_scored = 0;
    model_pruned = 0;
    bound_pruned = 0;
    fissioned = false;
    baseline_ms = Float.infinity;
    default_ms = None;
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
    best_schedule = [];
    (* No base lowering reached yet: [tune] fills this in once its base compile has captured one,
       and every report from then on carries it. *)
    source_digest = "";
    best_steps = [];
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

(** The fatal failure that ended the call, from whichever of the two failing states it was. A
    projection over the outcome, not a re-derivation of it: "did this call fail" is a question that
    spans two states, and every caller that ranks or attributes arms asks exactly that. *)
let terminal_failure (r : report) =
  match r.outcome with
  | Search_died tf | Pre_search_failure tf -> Some tf
  | Searched | Cache_replay | Abandonment_replay _ | Search_disabled | Abandoned _ -> None

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

let record_decline declines (classified : Outcome.classified_cause) =
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
  if first_for_key then
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

(* These parse a setting the caller has already read, rather than reading it themselves: the key has
   to be a string literal at the [Utils.get_global_arg] call site, because that literal is how the
   consistency tests find a configuration read (test/support/config_key_scan.ml). A reader helper
   taking the key as a parameter would hide every key routed through it. *)
let int_setting ~default s = try Int.of_string (String.strip s) with _ -> default
let float_setting ~default s = try Float.of_string (String.strip s) with _ -> default

(* A candidate round-improvement below this fraction of the incumbent ends the search. *)
let min_progress = 0.01

(* The composite playoff (gh-ocannl-1166). A per-segment single is timed as the WHOLE routine with
   every other segment on its untuned preset, so its time carries that backdrop: on gpt2_mini/CUDA
   tf32 the q/k/v singles measure about 23.2 ms where the composite they staff runs at 2.4 ms, and
   two replicate searches put the same single 0.02-0.03 ms apart, which is as large as the whole gap
   between the site's top four geometries. The per-key crown among them was a lottery (one replicate
   crowned [mma 32x32x0 bgrid], the other [mma 16x32x0 bgrid-in]). The playoff re-ranks those
   near-ties where the backdrop is small: inside the composite. A single within [playoff_margin] of
   its key's best single is close enough that the backdrop's noise could have inverted the order. At
   most [playoff_width] of them per key go to the playoff, nearest first. Changing either constant
   changes the search menu, so bump [Schedule_cache.entry_version]. *)
let playoff_margin = 0.005
let playoff_width = 2

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
      | _ -> ())

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
   first. *)
let sample_window ?(prior = []) ~repeats ~sample () =
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
    Int.incr count
  done;
  List.rev !samples

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
     depth policy consults one and must ignore the other. *)
  { ms; contended = stalled * 2 >= count; unbatched = false; samples = count }

let sample_min ~repeats ~sample = window_result (sample_window ~repeats ~sample ())

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

(* Sibling fault-injection seam to [on_candidate_attempt], at a timing run's pre-dispatch validation
   rather than at a candidate's compile (gh-ocannl-564). Default no-op, no config key selects it.
   Needed because the causes this phase contains — an unsatisfied dependency, an out-of-range
   binding — belong to the lineage and the bindings, so a genuine one hits every candidate at once
   and cannot express "this one declined, the search went on". *)
let on_candidate_preflight : (string -> unit) ref = ref (fun _routine_name -> ())

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
let on_candidate_timed : (string -> timed_so_far:int -> unit) ref =
  ref (fun _routine_name ~timed_so_far:_ -> ())

(* Fault-injection seam at the two post-admission callback boundaries (gh-ocannl-975). The measured
   candidate and incumbent times let a cleanup probe select an actual nonwinner, rather than
   guessing from attempt order. Default no-op; no configuration selects it. *)
let on_candidate_callback :
    ([ `Timed | `Calibration ] -> candidate_ms:float -> incumbent_ms:float -> unit) ref =
  ref (fun _ ~candidate_ms:_ ~incumbent_ms:_ -> ())

(* Measurement seam (gh-ocannl-1027): the one place a test can decide what an ADMITTED window
   measured. Without it, whether a search ever admits a candidate slower than its incumbent is a
   property of the machine's timings, so a regression about the nonwinner paths (gh-ocannl-975)
   could only re-roll the search and skip when every roll came out monotone. Called once per
   admitted timing window -- the dispatched baseline's and each candidate's -- by
   [apply_measurement_seam] below, before the admission gate that everything ranking or recording
   the time goes through. Default the identity; no configuration selects it. *)
let on_candidate_measured : (label:string -> digest:string -> float -> float) ref =
  ref (fun ~label:_ ~digest:_ ms -> ms)

(* Explicit test-only admission seam (gh-ocannl-1156). Scripted candidate measurements need their
   branches to run even when the host stalls every physical window. This scope clears ONLY the
   contention flag before candidate admission; calibration, dispatch, invalid clock readings and
   unresolved queued batching retain their ordinary rules. No configuration selects it. Production
   callers of [on_candidate_measured] still see only admitted windows. Restore the prior mode even
   on an injected failure, including when test scopes nest. *)
let test_windows_uncontended = ref false

let with_uncontended_test_windows f =
  let previous = !test_windows_uncontended in
  Exn.protect
    ~finally:(fun () -> test_windows_uncontended := previous)
    ~f:(fun () ->
      test_windows_uncontended := true;
      f ())

(* Unless the test-only scope above is active, only an admitted window reaches the seam; its result
   then passes the same [admitted_timing_ms] gate as a real reading, so a value that gate refuses
   (zero, negative, NaN, infinite) refuses the window exactly as a degenerate clock reading would --
   through the caller's ordinary refusal path, with nothing raised outside the search's cleanup
   boundaries. *)
let apply_measurement_seam ~label ~digest timing_result =
  let timing_result =
    if !test_windows_uncontended then { timing_result with contended = false } else timing_result
  in
  match admitted_timing_ms timing_result with
  | None -> timing_result
  | Some ms -> { timing_result with ms = !on_candidate_measured ~label ~digest ms }

(* Observation seam for the timing tests (gh-ocannl-851), reporting the batch depth each
   [time_routine] call settles on -- after calibration, before the timed loop; [Isolated] reports 1.
   The negative control for a twice-divided queued reading needs the depth the call ACTUALLY used:
   the call recalibrates independently, so re-applying the policy to an estimate taken outside it
   guesses wrong exactly on the busy runners the control must survive. The calibration dispatch
   count accompanies it so the instrument can account for both the single-launch estimate and the
   provisional queued probe. Default no-op, no config key selects it. *)
let on_batch_depth : (int -> calibration_samples:int -> unit) ref =
  ref (fun _depth ~calibration_samples:_ -> ())

(* Observation seam for the timing tests (gh-ocannl-994), reporting the window the returned minimum
   was actually taken over: how many batches the timed loop ran, counted by the loop itself rather
   than restated from its result, and their summed wall. A bound written on the reading needs THAT
   window's mean, not the whole call's: the call's wall also holds the warmup and the calibration's
   synchronized singles, and on a backend whose host round trip is two orders of magnitude above an
   amortized launch (Linux multidev_cc: ~57 us against ~0.44 us) those few dozen singles are a
   comparable share of the call to the tens of thousands of timed dispatches -- about 40% of it --
   so the whole-call mean is diluted by construction, and a host stall landing in the untimed part
   moves it without moving the reading. The MEDIAN batch wall accompanies the sum because this
   window's mean is still a stalled minority's to move: [contended] is declared on a MAJORITY of the
   window's batches exceeding twice its floor, so a bound meant to hold whenever the claim is not
   bypassed belongs on a statistic a minority cannot move -- which the median is over exactly the
   regime the contention rule leaves to it. [reused] says how many of the window's [samples] the
   loop did NOT dispatch (gh-ocannl-1074): a queued call that settles at depth 1 resumes the
   calibration's synchronized singles as its window, and those launches are already counted in
   [on_batch_depth]'s [calibration_samples] -- so the launches the timed loop itself dispatched are
   [depth * (samples - reused)]. Default no-op; no configuration key selects it. *)
let on_timed_window :
    (samples:int -> reused:int -> wall_ms:float -> median_wall_ms:float -> unit) ref =
  ref (fun ~samples:_ ~reused:_ ~wall_ms:_ ~median_wall_ms:_ -> ())

(* A discarded contention window, immediately before its one fresh retry (gh-ocannl-1060).
   [on_timed_window] describes only the returned window. These counts account for the extra
   dispatches without treating the discarded window as a candidate refusal. *)
let on_timing_retry : (samples:int -> reused:int -> unit) ref = ref (fun ~samples:_ ~reused:_ -> ())

(* Which of the calibration's branches dispatched a batch probe. *)
type calibration_probe_role =
  | Provisional_probe
  | Validation_probe
  | Confirmation_probe
  | Stall_retry_probe
  | Crossing_probe
  | Rescue_probe

type calibration_probe = {
  role : calibration_probe_role;
  depth : int;
  runs : int;
  wall_ms : float;
  min_ms : float;
}

(* Observation seam for the timing tests (gh-ocannl-1119), fired once per CUDA/HIP calibration batch
   probe as it ends, in dispatch order, with the branch that started it, how many minima it took,
   the wall it charged to [queue_calibration_wall_ms] and its minimum. A test that saw only the
   [batch] calls had to reconstruct probes from runs of same-depth batches, which merged a stall
   retry into the confirmation it repeats and counted probes only as a lower bound. Default no-op;
   no configuration key selects it. *)
let on_calibration_probe : (calibration_probe -> unit) ref = ref (fun _ -> ())

(* Which exit of the calibration chose the depth before the post-calibration bounds
   (gh-ocannl-1199). Each branch of [calibrate_and_time_with_retry_observer] returns its own, so the
   trace reports the decision taken rather than a reconstruction of it. *)
type batch_settle =
  | Isolated_objective
  | Single_estimate
  | Singles_within_target
  | Measured_batch
  | Affine_projection
  | Wall_scaled
  | Cap_measured
  | Cap_unresolved
  | Budget_exhausted

type batch_admission = Admitted | Refused_unbatched | Refused_contended | Refused_unreadable

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
}

(* Trace-only seam (gh-ocannl-1199), fired once per timing call after its window, with the depth
   decision's own metadata and the call's admission verdict. Observing it changes nothing a search
   measures or ranks. Default a no-op; no configuration selects it. *)
let on_batch_decision : (batch_decision -> unit) ref = ref (fun _ -> ())

(* The measurement proper, after the warmup: the calibration and the timed loop, with the device
   reduced to [batch depth], which dispatches [depth] launches back to back, synchronizes once and
   returns the wall in milliseconds. Separated from [time_routine] so a test can drive the whole
   policy -- which depth a call settles on, which window it times, how many launches each costs --
   on an injected clock, with no device and no machine-dependent routine (gh-ocannl-1074). *)
let calibrate_and_time_with_retry_observer ~on_retry ~phase ~retry_contended ~timing ~repeats
    ~queue_depth_cap ~batch =
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
  let probe_wall_ms = ref 0. and probes_started = ref 0 in
  (* The decision metadata [on_batch_decision] reports (gh-ocannl-1199): the latest affine fit and
     the validation probes a fit wanting the unmeasured cap directed (gh-ocannl-1144). *)
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
    !on_calibration_probe { role; depth; runs = !runs; wall_ms = !wall_ms; min_ms = !best_ms };
    { ms = !best_ms; contended = false; unbatched = false; samples = !runs }
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
                (calibration_dispatches, 1, single_estimate.ms, Singles_within_target)
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
  !on_batch_depth depth ~calibration_samples:calibration_dispatches;
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
      @@ sample_window ~prior:reused ~repeats
           ~sample:(fun () ->
             let wall = batch depth in
             timed_wall_ms := !timed_wall_ms +. wall;
             timed_walls := wall :: !timed_walls;
             Int.incr timed_batches;
             { per_launch_ms = wall /. Float.of_int depth; contention_ms = wall })
           ()
    in
    let report () =
      let median_wall_ms =
        let sorted = Array.of_list !timed_walls in
        Array.sort sorted ~compare:Float.compare;
        let n = Array.length sorted in
        if n = 0 then 0.
        else if n % 2 = 1 then sorted.(n / 2)
        else (sorted.((n / 2) - 1) +. sorted.(n / 2)) /. 2.
      in
      !on_timed_window ~samples:!timed_batches ~reused:(List.length reused) ~wall_ms:!timed_wall_ms
        ~median_wall_ms
    in
    (result, report)
  in
  let reused = if depth = 1 then singles else [] in
  let ((result, _) as window) = time_window reused in
  let retried =
    retry_contended && Poly.equal timing Queued && result.contended && Float.is_finite result.ms
    && Float.is_positive result.ms
  in
  let result, report =
    if retried then (
      (* Keep the depth and the 2x-majority rule. A fresh independent window can outlast a transient
         Metal scheduler burst; mixing windows would instead dilute its refusal evidence. Depth-1
         retries must likewise dispatch fresh singles rather than resume the refused ones. *)
      logf "queued timing retry: contention refused the first window (%d samples)" result.samples;
      on_retry ();
      !on_timing_retry ~samples:result.samples ~reused:(List.length reused);
      time_window [])
    else window
  in
  report ();
  let result =
    if !no_supported_batch then (
      logf
        "queued timing refused: every batched calibration probe read over the %.1f ms target, so \
         only an isolated depth-1 reading was left"
        queued_batch_ms;
      { result with unbatched = true })
    else result
  in
  !on_batch_decision
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
      admission =
        (if result.unbatched then Refused_unbatched
         else if result.contended then Refused_contended
         else if Option.is_none (admitted_timing_ms result) then Refused_unreadable
         else Admitted);
      retried;
    };
  result

(* Preserve the original all-labelled helper's source interface: an optional observer would not
   erase without adding a positional argument. *)
let calibrate_and_time ~retry_contended ~timing ~repeats ~queue_depth_cap ~batch =
  calibrate_and_time_with_retry_observer
    ~on_retry:(fun () -> ())
    ~phase:None ~retry_contended ~timing ~repeats ~queue_depth_cap ~batch

(* [routine.bindings] exposes the routine's live binding refs — restore them after timing (Codex P2
   on PR #103), or the returned winner would stay bound to the tuner's midpoint test values. *)
let time_routine ?(on_retry = fun () -> ()) ?(tag_failures = false) ?phase ~timing ~repeats cctx
    routine =
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
            !on_candidate_preflight routine.Context.name;
            Context.check_launch_bindings routine);
      (* Warmup run: absorbs lazy initialization and fills caches like a steady-state iteration. *)
      let ctx = ref (run cctx) in
      sync !ctx;
      (* Monotonic high-resolution clock: on Windows, [Unix.gettimeofday] ticks at ~1 ms, which
         makes sub-millisecond candidates indistinguishable (they all measure 0). *)
      let batch depth =
        let c0 = Mtime_clock.counter () in
        for _ = 1 to depth do
          ctx := run !ctx
        done;
        sync !ctx;
        Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6
      in
      calibrate_and_time_with_retry_observer ~on_retry ~phase ~timing ~repeats
        ~retry_contended:(retry_contended_window_for_backend (Context.backend_name cctx))
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
  | SC.Tensorize _ | SC.Partition _ | SC.Pad _ | SC.Stage _ | SC.Privatize _ | SC.Expand_zero _
  | SC.Fuse_epilogue _ | SC.Split_reduce _ | SC.Fold_mma _ ->
      true

let optop_family (op : SC.saved_optop) =
  match op with
  | SC.Split _ -> "Split"
  | SC.Swap _ -> "Swap"
  | SC.Retype _ -> "Retype"
  | SC.Unroll _ -> "Unroll"
  | SC.Partition _ -> "Partition"
  | SC.Pad _ -> "Pad"
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

(** The preset schedule extended with a [Privatize] per detected accumulator. Proposals are detected
    on the preset-scheduled segment and validated one at a time by re-applying the growing schedule;
    a proposal violating an op precondition is dropped. The exploratory applies run against a
    hermetic copy of the segment: [Privatize] registers its (fresh) tile in the traced store and
    placements, and abandoned tiles would otherwise be emitted as dead local declarations when the
    caller applies the returned schedule to the real segment. *)
let extend_with_privatize ~accum_prec ~static_indices sched (seg : LL.optimized) : Sched.schedule =
  let scratch () =
    {
      seg with
      LL.traced_store = Hashtbl.copy seg.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx seg.LL.optimize_ctx;
    }
  in
  match Sched.apply_classified ~static_indices sched (scratch ()) with
  | exception Outcome.Cause_at _ -> sched
  | post ->
      List.fold (privatize_proposals post) ~init:sched ~f:(fun acc (target, over) ->
          let acc' = acc @ [ Sched.privatize ~accum_prec ~target ~over ] in
          match Sched.apply_classified ~static_indices acc' (scratch ()) with
          | (_ : LL.optimized) -> acc'
          | exception Outcome.Cause_at _ -> acc)

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
    re-probed on the interchanged code, so a returned site is still seedable exactly as proposed. *)

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
}

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

let split_reduce_sites ?(static_indices = []) (opt : LL.optimized) : sr_site list =
  let acc = ref [] in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
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
      match Sched.apply ~static_indices ops (hermetic opt) with
      | opt' -> Some opt'
      | exception Invalid_argument _ -> None
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
      let legal =
        List.find_map candidates ~f:(fun (s, n, _) ->
            match splittable opt ~axis:s ~tn with
            | `Legal -> Some (s, n, [])
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
                        | `Legal -> Some (s, n, swaps)
                        | `No | `Hoist _ -> None))))
      in
      Option.iter legal ~f:(fun (s, n, swaps) ->
          if not (List.exists !acc ~f:(fun site -> Idx.equal_symbol site.sr_axis s)) then
            acc :=
              {
                sr_axis = s;
                sr_target = tn;
                sr_red = n;
                sr_out = out;
                sr_cost = List.fold enclosing ~init:1 ~f:(fun c (_, n, _) -> c * max 1 n);
                sr_dynamic = dynamic;
                sr_swaps = swaps;
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

(* Per-machine calibrated envelope constants from the config beat the backend's class-level advisory
   constants ([Backend_intf.hardware_limits]'s [peak_flops] / [peak_memory_bandwidth]) — fitting
   them from [autotune_calibration_file] data is the intended workflow. *)
(* Takes the read as a thunk, both to keep it lazy and to keep the key a literal at its call
   site -- see [int_setting]. *)
let peak_override read =
  lazy
    (let s = String.strip (read ()) in
     if String.is_empty s then None
     else
       match Float.of_string s with
       | f when Float.(f > 0.) -> Some f
       | _ -> None
       | exception _ -> None)

let peak_flops_override =
  peak_override (fun () -> Utils.get_global_arg ~arg_name:"model_peak_flops" ~default:"")

let peak_bandwidth_override =
  peak_override (fun () -> Utils.get_global_arg ~arg_name:"model_peak_memory_bandwidth" ~default:"")

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

type whole_flavor =
  | W_saved of SC.saved_schedule
  | W_preset of { block_size : int option }
  | W_sketch of sketch_params

type fiss_flavor =
  | F_preset of {
      block_size : int option;
      privatize : bool;
      config_thresholds : bool;
          (** Use the config-default [min_parallel] thresholds instead of the search's
              [min_parallel:1] — with [block_size = None] this reproduces the untuned default
              pipeline ({!Sched.maybe_default_schedules}) exactly, so the candidate pool always
              contains the behavior the user gets without tuning: on launch-overhead-bound workloads
              the aggressive [min_parallel:1] presets can all lose to it. *)
    }
  | F_saved of SC.saved_segment list
      (** Replay of a fissioned winner (gh-ocannl-1164): the routine is cut where the saved segments
          say ({!Sched.fission_segmented}'s [replay]), never re-segmented under the current policy,
          and the [i]th segment gets the [i]th saved schedule — by position, never by digest — once
          its structural digest matches the saved [i]th; a segmentation that no longer fits, or a
          digest that differs, fails the candidate — for a cache entry, a re-search. Whichever mode
          derived the winner's segmentation (the finer [arity_cuts] one included, gh-ocannl-574) is
          in the data. *)
  | F_sketch of { entries : (string * sketch_params) list; fine : bool }
      (** Per-segment matmul sketches: for each listed segment (keyed by its pre-schedule structural
          digest, like [F_saved]), the composed sketch pipeline instantiated with the given
          parameters; every other segment gets the plain default preset — the same pipeline the
          seed-time segment enumeration ran, so the segmentation converges. On a key miss
          (segmentation drift) the candidate degrades to the plain fissioned preset and dedups away
          by digest; unlike [F_saved] it never replays a cache entry, so no loud drift guard is
          needed. [fine]: the {e finer} [arity_cuts] segmentation, which frees a matmul site whose
          segment otherwise carries a companion that cannot follow the site's full arity (the
          lm_head's max-logits reduction, gh-ocannl-574). *)
  | F_split of { sites : (sr_site * int) list }
      (** Split-reduce seeds (gh-ocannl-484 task 3): per listed site, a
          [Sched.Split_reduce { axis = sr_axis; target = sr_target; num_blocks }] — applied
          {e whole-routine, before fission}, unlike the per-segment flavors: the two passes must
          compile as separate kernels (annotating the block loop with both passes in one kernel
          races — the combine needs grid-wide synchronization), and the partials producer/consumer
          pair is exactly the materialized cross-nest edge fission cuts at. Each resulting segment
          then gets the aggressive default preset — the block loop parallelizes pass 1, the combine
          nest annotates like any small kernel. *)
  | F_split_saved of SC.saved_schedule * SC.saved_segment list
      (** Replay of a split-reduce winner: the whole-routine prelude (resolved against the base
          canonical form, re-minting the partials node and fresh symbols via [SC.of_saved]), then
          the saved segments of the {e post-prelude} routine, replayed exactly like [F_saved]. *)

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

let bs_label = function None -> "cfg" | Some b -> Int.to_string b

(* How many units of [b] carry a different saved schedule than the same unit of [a] (a split-reduce
   prelude counting as one); [None] when the two forms do not segment alike -- different forms, or a
   segment whose kind, length or pre-schedule digest differs, the identity a replay checks -- so no
   unit-by-unit comparison exists. *)
let units_differing (a : compiled) (b : compiled) =
  let differ x y = if Poly.equal x y then 0 else 1 in
  let segments xs ys =
    let same_cut (x : SC.saved_segment) (y : SC.saved_segment) =
      Poly.equal x.seg_kind y.seg_kind && x.seg_units = y.seg_units
      && String.equal x.seg_digest y.seg_digest
    in
    if List.length xs = List.length ys && List.for_all2_exn xs ys ~f:same_cut then
      Some
        (List.fold2_exn xs ys ~init:0 ~f:(fun n (x : SC.saved_segment) (y : SC.saved_segment) ->
             n + differ x.seg_saved y.seg_saved))
    else None
  in
  match (a.form, b.form) with
  | Whole_saved x, Whole_saved y -> Some (differ x y)
  | Fiss_saved xs, Fiss_saved ys -> segments xs ys
  | Split_saved (px, xs), Split_saved (py, ys) ->
      Option.map (segments xs ys) ~f:(fun n -> n + differ px py)
  | _ -> None

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
let calibration_file =
  lazy (String.strip (Utils.get_global_arg ~arg_name:"autotune_calibration_file" ~default:""))

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
let bound_pruning_enabled =
  lazy
    (match
       String.lowercase
         (String.strip (Utils.get_global_arg ~arg_name:"autotune_bound_pruning" ~default:"false"))
     with
    | "true" | "1" -> true
    | _ -> false)

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
  if Lazy.force log_enabled || (not (String.is_empty file)) || have_envelope then (
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
  | Fiss (F_sketch { entries; _ }) -> List.exists entries ~f:(fun (_, p) -> p.sk_mma)
  | _ -> false

(* The swizzled staged twin is labeled apart from its plain sibling (gh-ocannl-481 item 3, D3): the
   two are otherwise identical, so a timing report that could not name which is which would be
   reporting the same candidate twice. *)
let swz_label p =
  match p.sk_swizzle with
  | None -> ""
  | Some LL.Swizzle_elem -> " swz-elem"
  | Some LL.Swizzle_b128 -> " swz-b128"

(* The pipelined staged twin likewise (gh-ocannl-487): identical to its plain sibling except the
   cooperative-stage depth, so the label must carry it. *)
let depth_label p = if p.sk_depth > 1 then Printf.sprintf " pd%d" p.sk_depth else ""

(* A widened-pack candidate (gh-ocannl-575) differs from a plain sibling only in the packed tiles'
   precision, so the label must carry it. *)
let pack_prec_label p =
  match p.sk_pack_prec with
  | Some pr -> Printf.sprintf " pack%s" (Ir.Ops.prec_string pr)
  | None -> ""

(* The batch flavor's label (gh-ocannl-643, gh-ocannl-728): " bgrid" for the batch-grid twin, "
   bgrid-in" for its interior-batch-inside-the-row-blocks order. *)
let batch_label p =
  if not p.sk_batch_grid then "" else if p.sk_batch_inner then " bgrid-in" else " bgrid"

let spec_label = function
  | Whole (W_saved s) -> Printf.sprintf "W_saved[%d ops]" (List.length s)
  | Whole (W_preset { block_size }) -> Printf.sprintf "W_preset[bs=%s]" (bs_label block_size)
  | Whole (W_sketch p) when p.sk_mma ->
      Printf.sprintf "W_sketch[%smma-%s %dx%dx%d%s%s%s%s%s%s%s%s%s]"
        (if p.sk_conv then "conv-" else "")
        (if p.sk_gpu then "gpu" else "cpu")
        p.sk_bm p.sk_bn p.sk_bk
        (if p.sk_bk > 0 then if p.sk_gpu then " staged" else " pack" else "")
        (pack_prec_label p) (swz_label p) (depth_label p)
        (if p.sk_hoist then " hoist" else "")
        (if p.sk_grid then " grid" else "")
        (if p.sk_pack_rest then " packrest" else "")
        (batch_label p)
        (if p.sk_epilogue then " ep" else "")
  | Whole (W_sketch p) ->
      Printf.sprintf "W_sketch[%s %dx%dx%d/%dx%d%s%s%s]"
        (if p.sk_gpu then "gpu" else "cpu")
        p.sk_bm p.sk_bn p.sk_bk p.sk_tm p.sk_tn
        (if p.sk_hoist then " hoist" else "")
        (batch_label p)
        (if p.sk_epilogue then " ep" else "")
  | Fiss (F_preset { block_size; privatize; config_thresholds }) ->
      Printf.sprintf "F_preset[bs=%s%s%s]" (bs_label block_size)
        (if privatize then " priv" else "")
        (if config_thresholds then " cfg-thresh" else "")
  | Fiss (F_saved segs) -> Printf.sprintf "F_saved[%d segs]" (List.length segs)
  | Fiss (F_sketch { entries; fine }) ->
      Printf.sprintf "F_sketch[%s%s]"
        (if fine then "fine " else "")
        (String.concat ~sep:","
           (List.map entries ~f:(fun (_, p) ->
                Printf.sprintf "%s%s%s %dx%dx%d%s%s%s%s%s%s%s%s%s"
                  (if p.sk_conv then "conv-" else "")
                  (if p.sk_mma then "mma-" else "")
                  (if p.sk_gpu then "gpu" else "cpu")
                  p.sk_bm p.sk_bn p.sk_bk
                  (if p.sk_mma then "" else Printf.sprintf "/%dx%d" p.sk_tm p.sk_tn)
                  (pack_prec_label p) (swz_label p) (depth_label p)
                  (if p.sk_hoist then " hoist" else "")
                  (if p.sk_grid then " grid" else "")
                  (if p.sk_pack_rest then " packrest" else "")
                  (batch_label p)
                  (if p.sk_epilogue then " ep" else ""))))
  | Fiss (F_split { sites }) ->
      Printf.sprintf "F_split[%s]"
        (String.concat ~sep:","
           (List.map sites ~f:(fun (s, b) ->
                Printf.sprintf "%s%s red%d out%d b%d%s" (Ir.Tnode.debug_name s.sr_target)
                  (if s.sr_dynamic then " dyn" else "")
                  s.sr_red s.sr_out b
                  (match List.length s.sr_swaps with 0 -> "" | n -> Printf.sprintf " swap%d" n))))
  | Fiss (F_split_saved (prelude, segs)) ->
      Printf.sprintf "F_split_saved[%d prelude ops, %d segs]" (List.length prelude)
        (List.length segs)

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
   table anew. The traced store is copied from the base (schedule ops register their tiles in
   it). *)
let compile_candidate ?name ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu ~provenance ctx
    comp bindings spec : compiled Outcome.outcome =
  let candidate = spec_label spec in
  let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
  let rebase (fresh : LL.optimized) =
    {
      base_opt with
      LL.traced_store = Hashtbl.copy base_opt.LL.traced_store;
      LL.optimize_ctx = fresh.LL.optimize_ctx;
    }
  in
  let preset_sched ?block_size ?(config_thresholds = false) opt =
    let min_parallel = if config_thresholds then None else Some 1 in
    if is_gpu then Sched.default_gpu ?block_size ?min_parallel ~limits opt
    else if is_cpu then Sched.default_cpu ?min_parallel opt
    else []
  in
  let captured = ref None in
  let compile_ctx () =
    match spec with
    | Whole flavor ->
        let transform fresh =
          let opt = rebase fresh in
          let sched, saved, registry =
            match flavor with
            | W_saved saved ->
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
          [ opt' ]
        in
        Context.compile_outcome ?name ~lowered_transform:transform ~provenance ~candidate ctx comp
          bindings
    | Fiss flavor ->
        let transforms fresh =
          let opt = rebase fresh in
          (* The split-reduce prelude (gh-ocannl-484 task 3) applies whole-routine BEFORE fission:
             the partials edge it mints is what fission cuts at, giving the two passes separate
             kernels and the event-chain synchronization the combine needs. *)
          let prelude, prelude_saved =
            match flavor with
            | F_preset _ | F_saved _ | F_sketch _ -> ([], [])
            | F_split { sites } ->
                let sched =
                  (* Per site: the gh-537 enabling interchange (empty for a site splittable as
                     lowered), then the split itself. Sites are distinct statements, so their
                     preludes compose. *)
                  List.concat_map sites ~f:(fun (s, num_blocks) ->
                      let op, _, _, _ =
                        Sched.split_reduce ~axis:s.sr_axis ~target:s.sr_target ~num_blocks
                      in
                      List.map s.sr_swaps ~f:(fun (outer, inner) -> Sched.Swap { outer; inner })
                      @ [ op ])
                in
                let saved, _ = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved)
            | F_split_saved (psaved, _) ->
                let sched, _ = SC.of_saved canon psaved in
                (sched, psaved)
          in
          let opt =
            if List.is_empty prelude then opt
            else Sched.apply_classified ~static_indices prelude opt
          in
          let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
          (* Per-segment schedule matching keys on the STRUCTURAL canon ([with_placements:false]):
             placement classes can render differently across compilation lineages on byte-identical
             segments (decided in one, undecided in the other — e.g. tuning with [timing_ctx]),
             which used to fail winner replays wholesale. *)
          let seg_key seg =
            SC.digest (SC.canonicalize ~static_indices ~with_placements:false seg)
          in
          (* A replay cuts where its saved segments say and gives the [i]th segment the [i]th saved
             schedule — by position, never by digest: the digest leaves placements out, and two
             segments of one structure can differ in placements, hence in kind and schedule. The
             replay-validity check rides on the same position: the segment cut there must be the one
             its schedule was saved against, otherwise the entry no longer applies and the candidate
             fails (a cache replay then re-searches) rather than running a segment under a schedule
             saved for other code. *)
          let replay =
            match flavor with
            | F_preset _ | F_sketch _ | F_split _ -> None
            | F_saved saved | F_split_saved (_, saved) ->
                let saved = Array.of_list saved in
                let schedule i seg =
                  let seg_canon = SC.canonicalize ~static_indices ~with_placements:false seg in
                  let s = saved.(i) in
                  if not (String.equal (SC.digest seg_canon) s.SC.seg_digest) then
                    raise
                      (Outcome.Cause_at
                         ( Outcome.Transform,
                           Outcome.Illegal_schedule
                             {
                               check = "Autotune.fission_replay";
                               detail =
                                 Printf.sprintf
                                   "segment %d differs from the one its schedule was saved \
                                    against: the saved segmentation no longer applies"
                                   i;
                             } ));
                  fst (SC.of_saved seg_canon s.SC.seg_saved)
                in
                Some (SC.segmentation_of (Array.to_list saved), schedule)
          in
          let preset seg =
            match flavor with
            | F_preset { block_size; privatize; config_thresholds } ->
                let sched = preset_sched ?block_size ~config_thresholds seg in
                if privatize then extend_with_privatize ~accum_prec ~static_indices sched seg
                else sched
            | F_saved _ | F_split_saved _ -> [] (* Never consulted: a replay brings its own. *)
            | F_sketch { entries; _ } -> (
                match List.Assoc.find entries ~equal:String.equal (seg_key seg) with
                | Some p -> sketch_schedule ~accum_prec ~p seg
                | None -> preset_sched seg)
            | F_split _ -> preset_sched seg
          in
          (* The [arity_cuts] (finer) segmentation is part of the candidate's identity: the seeds
             enumerated their keyed segments under it (gh-ocannl-574). A replay's segmentation is
             its data, so the mode does not reach it. *)
          let arity_cuts =
            match flavor with
            | F_sketch { fine; _ } -> fine
            | F_preset _ | F_split _ | F_saved _ | F_split_saved _ -> false
          in
          let segmentation, tuples =
            (* Match the default pipeline's placements (statement-crossing [Local]s promoted on
               GPU), so fissioned candidates and the untuned baseline schedule the same code. *)
            Sched.fission_segmented ~promote_locals:is_gpu
              ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
              ?replay ~arity_cuts ~preset ~zero_sched ~static_indices opt
          in
          let saved_with_registries = SC.save_segments ~static_indices segmentation tuples in
          let posts = List.map tuples ~f:(fun (_, _, _, post) -> post) in
          let units =
            List.filter_mapi (List.zip_exn saved_with_registries posts)
              ~f:(fun i ((s, registry), post) ->
                match s.SC.seg_kind with
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
          posts
        in
        Context.compile_outcome ?name ~lowered_transform:transforms ~provenance ~candidate ctx comp
          bindings
  in
  (* The [Tile_mma] rendering census travels on the routine ([Context.routine.mma], gh-ocannl-626):
     [Context.compile] collects it around this candidate's kernel compiles, fissioned segments
     included, so no bracket is needed here and a candidate cannot be timed without it. *)
  match compile_ctx () with
  | Error failure -> Error failure
  | Ok (cctx, routine) -> (
      match !captured with
      | Some (form, units, all_opts, digest_after) ->
          Ok { form; cctx; routine; units; all_opts; digest_after }
      | None ->
          Outcome.protect
            ~classify_backend:(fun _ _ -> None)
            ~provenance ~phase:Outcome.Transform ~candidate
            (fun () -> failwith "Autotune: the transform was not invoked"))

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
   [Workgroup] lane loop, which [Low_level.validate_parallel] rejects inside a [Local_scope] body —
   a scope-nested triple can never compile ([Schedule.op_legality] would not prune it: it answers
   about races, not scope nesting). No candidates are lost: the loops inside an accumulation mint
   all reduce into the scope's single loop-invariant cell, so none can play the micro-kernel's
   output-dim [i]/[j] roles and every role assignment over such a triple would be refuted anyway.
   Deduplicated by the outer binder, since a non-minting materializing [Unroll] leaves
   statement-level copies sharing loop symbols. *)
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
  match (elem.form, u.u_segment) with
  | Whole_saved _, None -> Some (Whole (W_saved (u.u_saved @ [ op ])))
  | Fiss_saved segs, Some i -> Some (Fiss (F_saved (extend segs i)))
  | Split_saved (prelude, segs), Some i -> Some (Fiss (F_split_saved (prelude, extend segs i)))
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
   the caller (the classification is a ranking input, not a legality fact). *)
let mma_eligible_sites ~(limits : Ir.Backend_intf.hardware_limits) ~static_indices
    (opt : LL.optimized) : matmul_site list =
  match limits.Ir.Backend_intf.mma with
  | None -> []
  | Some mma ->
      let segments =
        match
          (* [mma] is a GPU capability: segment as the GPU seeders do. *)
          Sched.fission_scheduled
            ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:true ~limits)
            ~preset:(fun _ -> [])
            ~zero_sched:(fun _ -> [])
            ~static_indices (scratch_of opt)
        with
        | tuples ->
            List.filter_map tuples ~f:(fun (kind, pre, _sched, _post) ->
                match kind with `Normal -> Some pre | `Zeros | `Solo -> None)
        | exception Outcome.Cause_at _ -> [ opt ]
      in
      List.filter_map segments ~f:(fun seg -> detect_matmul seg.LL.llc)
      |> List.filter ~f:(fun site ->
          let a_prec = Lazy.force site.m_a.Ir.Tnode.storage_prec in
          let b_prec = Lazy.force site.m_b.Ir.Tnode.storage_prec in
          let d_prec = Lazy.force site.m_d.Ir.Tnode.storage_prec in
          Option.is_some
            (mma_tile_for_precisions_in_scope mma ~scope:(matmul_mma_scope site ~bk:0) ~a_prec
               ~b_prec ~d_prec))

let placement_enablement ~limits ~static_indices ~(base : LL.optimized) ~(allmat : LL.optimized) =
  let site_tns sites =
    List.fold sites
      ~init:(Set.empty (module Ir.Tnode))
      ~f:(fun acc site -> Set.add (Set.add (Set.add acc site.m_a) site.m_b) site.m_d)
  in
  let base_sites = mma_eligible_sites ~limits ~static_indices base in
  let allmat_sites = mma_eligible_sites ~limits ~static_indices allmat in
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
}

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
  let enablement, disablement = placement_enablement ~limits ~static_indices ~base ~allmat in
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
      (** The winner's roofline lower bound in ms — a ranking score, not a runtime prediction;
          [None] when selection did not run (no envelope constants, automatic scheduling disabled,
          or the default itself had no model coverage). *)
  mc_scored : int;
      (** Model evaluations that produced a score (the default pipeline included; the fissioned flow
          also scores per segment). *)
  mc_skipped : int;  (** Model evaluations without coverage, excluded from ranking. *)
  mc_rejected : int;
      (** Candidates excluded from the ranking because their scheduled form fails
          {!Ir.Low_level.validate_parallel} — it could not have compiled (gh-ocannl-522). *)
}

let model_default_enabled =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"model_default_schedule")

(* gh-ocannl-514 phase 5: whether [model_default]'s family search lifts the tile-size lattice
   exclusions ([lift_geometry_lattice]) — the full dividing lattice searched under non-uniform
   bounds instead of the curated menus alone. Never affects [tune]'s seed lists. *)
let geometry_lattice_enabled =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"model_default_geometry_lattice")

(* The model ranks several scheduled forms without compiling them. Keep this eager validation in
   that ranking loop: removing it made an invalid tensorized argmin displace a viable schedule and
   then fall back all the way to the default. This is no longer needed for exception attribution or
   advisory containment -- codegen carries the same typed cause -- only to preserve "best viable
   model candidate" selection without compiling every contender. *)
let validate_segments_for_model (segs : LL.optimized list) =
  List.iter segs ~f:(fun (o : LL.optimized) ->
      LL.validate_parallel_classified o.LL.optimize_ctx.LL.placements o.LL.llc);
  segs

let compile_advisory ?name ?on_fallback ?fallback_if lowered_transform ctx comp bindings =
  match
    Context.compile_outcome ?name ~lowered_transform ~provenance:Outcome.Advisory ctx comp bindings
  with
  | Ok result -> result
  | Error (Outcome.Fatal _ as failure) -> Outcome.raise_failure failure
  | Error (Outcome.Classified classified as failure) ->
      (* [fallback_if] is what keeps the retry from duplicating a genuine failure: a transform that
         already degraded to the default pipeline has nothing to fall back TO, so recompiling would
         just repeat the same failing compile (and, on a resource failure, aggravate it) before
         raising the same exception. Such callers say [false] here and the original exception
         propagates through the public exception contract. *)
      if not (Option.value_map fallback_if ~default:true ~f:(fun f -> f ())) then
        Outcome.raise_failure failure;
      (* Typed compiler rejection is the advisory fallback boundary. Fatal failures are propagated
         above without paying for a second compile. *)
      Option.iter on_fallback ~f:(fun f -> f (Outcome.exception_of_cause classified.cause));
      Context.compile ?name ctx comp bindings

let model_default ?name ?report ctx comp bindings =
  let backend = Context.backend_name ctx in
  let is_gpu = Sched.backend_is_gpu backend and is_cpu = Sched.backend_is_cpu backend in
  let limits = Context.hardware_limits ctx in
  let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
  let static_indices = Idx.bound_symbols bindings in
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let emit r = Option.iter report ~f:(fun f -> f r) in
  let no_selection =
    { mc_label = "default"; mc_model_ms = None; mc_scored = 0; mc_skipped = 0; mc_rejected = 0 }
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
    let n_scored = ref 0 and n_skipped = ref 0 and n_rejected = ref 0 in
    let score opts =
      match
        summaries_roofline ~peak_flops ~peak_memory_bandwidth
          (List.map opts ~f:(fun o -> CM.analyze o.LL.llc))
      with
      | Some s ->
          Int.incr n_scored;
          Some s
      | None ->
          Int.incr n_skipped;
          None
    in
    (* The model must rank the best viable schedule, not crown an invalid argmin and fall all the
       way back to default. The validator is typed, so only an expected schedule rejection is
       excluded; compiler assertions and other failures still escape. *)
    let score_valid opts =
      match validate_segments_for_model opts with
      | opts -> score opts
      | exception Outcome.Cause_at _ ->
          Int.incr n_rejected;
          None
    in
    let score_sketch base_opt p =
      match
        Sched.apply_classified ~static_indices
          (sketch_schedule ~accum_prec ~p base_opt)
          (scratch_of base_opt)
      with
      | exception Outcome.Cause_at _ ->
          Int.incr n_rejected;
          None
      | post -> score_valid [ post ]
    in
    (* The branch-and-bound walk over the factored matmul family (gh-ocannl-514 phase 4):
       verdict-carrying children are the construction-time fathoms, and the bound is the
       schedule-invariant roofline floor — sketch completions share the base program's semantics, so
       [completion_floor] lower-bounds every one; it fathoms the whole family exactly when the
       incumbent already achieves it (the memory-bound kernels where the default preset is optimal)
       — raised per subtree by the committed staging decisions' certain traffic (phase 5,
       [sketch_path_traffic_floor]). The epilogue twins are the tree's root level (gh-ocannl-613),
       so they compete inside the walk, after the unfused leaves and at the threshold those
       tightened to, with the same bound and on the same stats ledger. [None] = no matmul site: the
       caller keeps the flat path (conv seeds, which factor as a follow-up). Returns the first leaf
       strictly better than [incumbent]. *)
    let tree_search ~incumbent base_opt =
      match matmul_sketch_tree ~is_gpu ~is_cpu ~limits base_opt with
      | None -> None
      | Some tree ->
          (* gh-ocannl-514 phase 5: the tile-size lattice beyond the curated menus enters the
             searched space when lifted (config [model_default_geometry_lattice]), and the bound is
             no longer uniform across the family — each subtree's committed staging decisions
             contribute their certain traffic ([sketch_path_traffic_floor]) on top of the
             schedule-invariant floor, so whole boxes of the lattice fathom without expansion. *)
          let tree =
            if Lazy.force geometry_lattice_enabled then lift_geometry_lattice tree else tree
          in
          let f = CM.completion_floor base_opt.LL.llc in
          let path_inc = sketch_path_traffic_floor ~limits base_opt in
          let bound_at inc =
            CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:f.CM.fr_flops
              ~bytes:(f.CM.fr_bytes + inc) ()
          in
          let fb = bound_at 0 in
          (* Snapshot the caller-side counters so the log can split the driver's unscored leaves
             into compiler rejections vs genuine no-coverage — st_unscored alone would misclassify
             rejections as cost-model gaps in the phase-6 ledger. *)
          let r0 = !n_rejected and k0 = !n_skipped in
          let best, stats =
            Sspace.search
              ~bound:(fun ~path _sub -> bound_at (path_inc path))
              ~incumbent ~score:(score_sketch base_opt) tree
          in
          logf
            "model_default: family search: %d expanded, %d scored, %d unscored (%d rejected, %d \
             without coverage), %d fathomed (bound %s), %d refuted, %d excluded"
            stats.Sspace.st_expanded stats.Sspace.st_scored stats.Sspace.st_unscored
            (!n_rejected - r0) (!n_skipped - k0) stats.Sspace.st_fathomed
            (match fb with Some b -> Printf.sprintf "%.6f ms" (b *. 1e3) | None -> "n/a")
            stats.Sspace.st_refuted stats.Sspace.st_excluded;
          Some best
    in
    (* First leaf strictly under [threshold], in list order — the flat counterpart the
       not-yet-factored conv family goes through. *)
    let best_flat ~threshold base_opt ps =
      List.fold ps ~init:(None, threshold) ~f:(fun (best, th) p ->
          match score_sketch base_opt p with
          | Some sc when Float.(sc < th) -> (Some (p, sc), sc)
          | _ -> (best, th))
      |> fst
    in
    let preset seg = if is_gpu then Sched.default_gpu ~limits seg else Sched.default_cpu seg in
    let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
    let seg_key seg = SC.digest (SC.canonicalize ~static_indices ~with_placements:false seg) in
    (* The model-argmin pipeline choice for one lowering — label, roofline score (seconds), and the
       action reproducing it. Shared between the in-compile transform seam and the placement
       pre-search's leaf scoring (hermetic lowerings of decided placement vectors). *)
    let select (opt : LL.optimized) =
      try
        (* The untuned default pipeline, scored on a hermetic copy — it is both the anchor candidate
           and the fallback. *)
        let default_scratch =
          Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices
            (scratch_of opt)
        in
        let default_score = score default_scratch in
        match default_score with
        | None ->
            (* No coverage of the default itself: nothing to honestly compare against. *)
            ("default", None, `Default)
        | Some ds -> (
            (* Whole-routine sketch candidates. A candidate without coverage is skipped — it is
               never picked over the default without a measured run ({!tune} covers that). *)
            let whole_best =
              match tree_search ~incumbent:ds opt with
              | Some tree_best ->
                  (* Matmul site: the tree's leaves, twins included, searched with the default as
                     incumbent. *)
                  tree_best
              | None ->
                  (* No matmul site: the flat path covers the conv family. *)
                  best_flat ~threshold:ds opt (sketch_seed_params ~is_gpu ~is_cpu ~limits opt)
            in
            let contenders =
              match whole_best with
              | Some (p, sc) -> [ (spec_label (Whole (W_sketch p)), sc, `Whole p) ]
              | None -> []
            in
            (* Per-segment sketch substitution over the default fission segmentation (only when the
               default actually fissioned; otherwise the whole-routine sketches cover the site).
               Mirrors [tune]'s [F_sketch] flavor: segments keyed by their structural pre-schedule
               digest, a key miss degrading to the default preset. *)
            let fiss =
              if List.length default_scratch <= 1 then None
              else
                match
                  Sched.fission_scheduled ~promote_locals:is_gpu
                    ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
                    ~preset ~zero_sched ~static_indices (scratch_of opt)
                with
                | exception Outcome.Cause_at _ ->
                    Int.incr n_rejected;
                    None
                | tuples -> (
                    let entries =
                      List.filter_map tuples ~f:(fun (kind, pre, _sched, post) ->
                          match kind with
                          | `Zeros | `Solo -> None
                          | `Normal -> (
                              match score [ post ] with
                              | None -> None
                              | Some bs -> (
                                  (* The segment's family tree searched with the segment's own
                                     default-preset score as incumbent; conv segments keep the flat
                                     path. *)
                                  let best_sketch =
                                    match tree_search ~incumbent:bs pre with
                                    | Some tree_best -> tree_best
                                    | None ->
                                        best_flat ~threshold:bs pre
                                          (sketch_seed_params ~is_gpu ~is_cpu ~limits pre)
                                  in
                                  match best_sketch with
                                  | Some (p, _s) -> Some (seg_key pre, p)
                                  | None -> None)))
                    in
                    if List.is_empty entries then None
                    else
                      let subst_preset seg =
                        match List.Assoc.find entries ~equal:String.equal (seg_key seg) with
                        | Some p -> sketch_schedule ~accum_prec ~p seg
                        | None -> preset seg
                      in
                      (* Score the substituted pipeline whole, so it competes on the same footing as
                         the other candidates. *)
                      match
                        Sched.fission_scheduled ~promote_locals:is_gpu
                          ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
                          ~preset:subst_preset ~zero_sched ~static_indices (scratch_of opt)
                      with
                      | exception Outcome.Cause_at _ ->
                          Int.incr n_rejected;
                          None
                      | tuples2 ->
                          let posts = List.map tuples2 ~f:(fun (_, _, _, post) -> post) in
                          Option.map (score_valid posts) ~f:(fun s -> (entries, s)))
            in
            let contenders =
              contenders
              @
              match fiss with
              | Some (entries, s) ->
                  [ (spec_label (Fiss (F_sketch { entries; fine = false })), s, `Fiss entries) ]
              | None -> []
            in
            (* Argmin with ties to the default: the model only displaces the honest default on a
               strict improvement. *)
            let best =
              List.min_elt contenders ~compare:(fun (_, a, _) (_, b, _) -> Float.compare a b)
            in
            match best with
            | Some (lbl, s, act) when Float.(s < ds) -> (lbl, Some s, act)
            | _ -> ("default", Some ds, `Default))
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
          mc_model_ms = Option.map model_s ~f:(fun s -> s *. 1e3);
          mc_scored = !n_scored;
          mc_skipped = !n_skipped;
          mc_rejected = !n_rejected;
        };
      let apply_action () =
        (* Schedule application uses the typed seam. Backend validation is deliberately left to
           [compile_advisory], which now receives its classified cause directly from codegen. *)
        match action with
        | `Default -> default_segs ()
        | `Whole p ->
            validate_segments_for_model
              [ Sched.apply_classified ~static_indices (sketch_schedule ~accum_prec ~p opt) opt ]
        | `Fiss entries ->
            let subst_preset seg =
              match List.Assoc.find entries ~equal:String.equal (seg_key seg) with
              | Some p -> sketch_schedule ~accum_prec ~p seg
              | None -> preset seg
            in
            validate_segments_for_model
              (List.map
                 (Sched.fission_scheduled ~promote_locals:is_gpu
                    ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
                    ~preset:subst_preset ~zero_sched ~static_indices opt)
                 ~f:(fun (_, _, _, post) -> post))
      in
      match apply_action () with
      | segs ->
          logf "model_default: chose %s (model %s; %d scored, %d without coverage, %d unbuildable)"
            label
            (match model_s with Some s -> Printf.sprintf "%.6f ms" (s *. 1e3) | None -> "n/a")
            !n_scored !n_skipped !n_rejected;
          (applied_pick := match action with `Default -> false | `Whole _ | `Fiss _ -> true);
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
            };
          applied_pick := false;
          default_segs ()
    in
    let on_fallback exn =
      logf "model_default: compiling the pick %s FAILED (%s); recompiling the default pipeline"
        !choice.mc_label (Exn.to_string exn);
      choice := { !choice with mc_label = "default"; mc_model_ms = None }
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
    let compile_from base_ctx =
      compile_advisory ?name ~on_fallback
        ~fallback_if:(fun () -> !applied_pick)
        transforms base_ctx comp bindings
    in
    let result =
      match placement_pick with
      | None -> compile_from ctx
      | Some (mat, inl, fp) -> (
          (* Every reading gets its call, empty or not — the historical context lineage of this
             path, kept as is; skipping would change only context identity, not the decisions. *)
          let ctx' = Context.decide_readings ~skip_empty:false ctx (mat, inl, fp) in
          match compile_from ctx' with
          | result ->
              (* The emitted label carries the placement decision: the in-compile selection only
                 names the pipeline it chose under those placements. *)
              let names tns = String.concat ~sep:"," (List.map tns ~f:Ir.Tnode.debug_name) in
              choice :=
                {
                  !choice with
                  mc_label =
                    Printf.sprintf "placements[mat:%s inl:%s fp:%s] %s" (names mat) (names inl)
                      (names fp) !choice.mc_label;
                };
              result
          | exception ((Utils.User_error _ | Invalid_argument _) as exn) ->
              (* A classified rejection that [compile_advisory] had nothing to fall back to under
                 the picked placements ([applied_pick] was false: the pick's win was the placement
                 move itself, so the failing compile was already the default pipeline — under
                 [ctx'], not under [ctx]). The pick is advisory, so abandon it and rerun the
                 ordinary selection from the caller's own placements; fatal failures propagate
                 above. *)
              logf
                "model_default: compiling under the picked placements FAILED (%s); recompiling \
                 from the default placements"
                (Exn.to_string exn);
              applied_pick := false;
              compile_from ctx)
    in
    emit !choice;
    result

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
let on_candidate_attempt : (string -> unit) ref = ref (fun _label -> ())
let search_setting () = Utils.get_global_flag ~arg_name:"autotune_search" ~default:true

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

let tune ?name ?search ?beam_width ?rounds ?repeats ?timing ?seed_block_sizes ?cache_dir
    ?keep_fraction ?max_split_reduce_sites ?timing_ctx ?abandon ?report ctx comp bindings =
  (* gh-ocannl-559: with the search off, [tune] still replays an explicitly provided cache -- a
     pinned schedule is deterministic, and committing one is how a reproducible run keeps a tuned
     schedule -- but never times candidates, whose crowning is the largest cross-machine determinism
     leak. A miss compiles the untuned default pipeline, exactly like the nothing-was-timed fallback
     below. *)
  let search = Option.value_or_thunk search ~default:search_setting in
  let beam_width =
    max 1
      (Option.value beam_width
         ~default:
           (int_setting ~default:2
           @@ Utils.get_global_arg ~arg_name:"autotune_beam_width" ~default:"2"))
  in
  let rounds =
    Option.value rounds
      ~default:
        (int_setting ~default:2 @@ Utils.get_global_arg ~arg_name:"autotune_rounds" ~default:"2")
  in
  let repeats =
    Option.value repeats
      ~default:
        (int_setting ~default:3 @@ Utils.get_global_arg ~arg_name:"autotune_repeats" ~default:"3")
  in
  (* Not [Option.value ~default:…] (Codex P2 on PR #512): that evaluates the default eagerly, so an
     unparsable ambient [autotune_timing] would be rejected even for a caller that passed its own
     mode — and for a [~search:false] call, which times nothing at all. The other settings escape
     this only because [int_setting] swallows a bad value; this one is meant to refuse it, so the
     read has to be in the branch that needs it. *)
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
  let base_report = { (no_search_report ~timing) with beam_width } in
  let max_split_reduce_sites =
    max 0
      (Option.value max_split_reduce_sites
         ~default:
           (int_setting ~default:8
           @@ Utils.get_global_arg ~arg_name:"autotune_split_reduce_max_sites" ~default:"8"))
  in
  let seed_block_sizes = Option.value seed_block_sizes ~default:[ 64; 128; 256; 512 ] in
  let cache_dir = resolve_cache_dir ?cache_dir ~search () in
  let keep_fraction =
    Option.value keep_fraction
      ~default:
        (float_setting ~default:1.
        @@ Utils.get_global_arg ~arg_name:"autotune_keep_fraction" ~default:"1.")
  in
  (* gh-ocannl-1110: what shapes which candidates this search times, and in what order, that the
     schedule cache's key does not carry: every [Search_shaping] configuration key (the class the
     key leaves out by definition), plus every argument of this function that resolves one — each
     resolved value, whether passed or read. A cached [best_steps] replays only under the same
     shape: the flip chain compares at equal depth, which presumes the same candidate order and
     sampling. *)
  let search_shape =
    Printf.sprintf "search=%b beam=%d rounds=%d repeats=%d keep=%h split_sites=%d blocks=%s|%s"
      search beam_width rounds repeats keep_fraction max_split_reduce_sites
      (String.concat ~sep:"," (List.map seed_block_sizes ~f:Int.to_string))
      (Utils.config_class_fingerprint Utils.Search_shaping)
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
  let progress_stage stage =
    progress_line "stage" (Printf.sprintf "stage=%s %s" stage (progress_costs ()))
  in
  let timed_into acc f =
    let c = Mtime_clock.counter () in
    Exn.protect ~f ~finally:(fun () -> acc := !acc +. seconds_since c)
  in
  progress_line "search_start"
    (Printf.sprintf "backend=%s device=%d search=%b beam=%d rounds=%d" backend device search
       beam_width rounds);
  (* Every report goes through here, exactly once per call on every path (gh-ocannl-550), so this is
     where a search's closing progress line is written: before the callback, which may raise. *)
  let emit_report (r : report) =
    progress_line "search_done"
      (Printf.sprintf
         "outcome=%s timed=%d contended=%d timings_retried=%d unbatched=%d failed=%d rounds=%d %s \
          best_ms=%s best=%S"
         (outcome_name r.outcome) r.candidates_timed r.timings_contended r.timings_retried
         r.timings_unbatched r.candidates_failed r.rounds_run (progress_costs ())
         (progress_ms r.best_ms) r.best_label);
    Option.iter report ~f:(fun f -> f r)
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
  let emit_pre_search_failure ?(base = base_report) ~phase ~candidate ~detail () =
    let r =
      {
        base with
        outcome = Pre_search_failure { phase; candidate; detail };
        timings_retried = !n_timings_retried;
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
  (* The untuned fallback of a search-less call, through the containment-aware form so a failure
     reports the phase it carries. [Context.compile] is exactly this plus [raise_failure], which is
     what [raise_pre_search] ends with, so the caller sees the same exception either way. *)
  let compile_untuned_default ?base () =
    progress_stage "untuned_default_compile";
    match
      Context.compile_outcome ?name ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
    with
    | Ok result -> result
    | Error failure -> raise_pre_search ?base failure
  in
  (* Without a cache to replay there is nothing for a search-less [tune] to do, so it does not even
     take the base compile that computes the cache key: the caller gets the untuned default compile
     it would have gotten from [Context.compile]. *)
  if (not search) && String.is_empty cache_dir then (
    logf
      "search disabled (autotune_search=false) and no chosen cache: compiling the untuned default";
    (* Report AFTER the fallback compile: a report is a record of what this call achieved, and the
       base report says the untuned default shipped. Emitting it first would leave a consumer
       holding a clean, non-partial report for a call that then raised (Codex P2 on PR #291); a
       compile that raises reports its own failure instead (gh-ocannl-550). *)
    let result = compile_untuned_default () in
    report_or_release base_report ~result;
    result)
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
    progress_stage "base_compile";
    let base_outcome =
      Context.compile_outcome ?name
        ~lowered_transform:(fun opt ->
          (* Inside the transform, so an injected fault is classified by the ordinary machinery
             (phase [Transform], provenance [Candidate]) and reaches [raise_pre_search] below with a
             real phase and a report — rather than escaping the whole call unreported, which would
             break the exactly-once contract for direct [tune] callers. *)
          !on_candidate_attempt "baseline";
          base_capture := Some (opt, SC.canonicalize ~static_indices opt);
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
    (* The key this call's schedule entry is stored and looked up under, and (gh-ocannl-1022) the
       [source_digest] of every report from here on, failures included: they all describe a search
       of this lowering. Shadowing [base_report] reaches [census] and every [~base] passed below;
       the report closures defined above keep the empty-digest one, which only the failures that
       precede the capture use without a [~base]. *)
    let base_digest = SC.digest canon in
    let base_report = { base_report with source_digest = base_digest } in
    let baseline_linked, baseline_decline =
      match base_outcome with
      | Ok (bctx, broutine) -> (Some (bctx, broutine), None)
      | Error (Outcome.Classified classified) -> (None, Some classified)
      | Error (Outcome.Fatal _ as failure) -> raise_pre_search ~base:base_report failure
    in
    (* gh-ocannl-550: the base compile runs BEFORE the cache is consulted — its lowering is what
       every candidate and every replay derives from — so on the two paths that do not search, its
       linked artifact is dead as soon as that decision is taken, and nothing downstream can reach
       it. On the search path it enters the beam instead and is released there. Without this, a
       warm-cache process leaked one full base-candidate pool per [tune] call, permanently (the pool
       table roots it), which for a repeatedly-tuning process is the very accumulation this issue is
       about. *)
    let release_baseline () =
      Option.iter baseline_linked ~f:(fun (bctx, _) ->
          release_quietly ~what:"the baseline compile" bctx)
    in
    release_baseline_hook := release_baseline;
    let codegen_tag = SC.codegen_tag ~limits ~capabilities () in
    let objective = timing_string timing in
    let key =
      SC.cache_key
        ~timing_identity:(Context.timing_identity search_ctx)
        ~objective ~limits ~capabilities canon ~backend
    in
    let use_cache = (not (String.is_empty cache_dir)) && SC.complete canon && Option.is_some key in
    if Option.is_none key then
      logf "persistent timed cache disabled for %s: concrete device identity unavailable" backend;
    let compile_spec =
      compile_candidate ?name ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu
        ~provenance:Outcome.Candidate search_ctx comp bindings
    in
    (* Winner (and cache-hit) compiles target the caller's context; they replay against the same
       base lowering as the search's candidates. *)
    let compile_spec_real provenance =
      compile_candidate ?name ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu ~provenance
        ctx comp bindings
    in
    let flat_schedule = function
      | Whole_saved saved -> saved
      | Fiss_saved segs -> List.concat_map segs ~f:(fun s -> s.SC.seg_saved)
      | Split_saved (prelude, segs) -> prelude @ List.concat_map segs ~f:(fun s -> s.SC.seg_saved)
    in
    let is_fissioned = function Whole_saved _ -> false | Fiss_saved _ | Split_saved _ -> true in
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
    let census () =
      {
        base_report with
        candidates_failed = failed_count declines;
        baseline_declined = Option.is_some baseline_decline;
        declines = decline_summaries declines;
      }
    in
    let cached =
      if use_cache then
        match SC.lookup ~dir:cache_dir ~key with
        (* The numerics and codegen checks are belt-and-braces: [key] already carries both tags
           (gh-ocannl-568, gh-ocannl-572), so a regime-mismatched entry normally lives in a
           different file and is never looked up. They catch a hand-moved or hand-written entry,
           which is the shape of the misdirection this guards against — a tf32-vs-default A/B whose
           cache directories got crossed. An entry from before the codegen field existed carries no
           claim about its regime, so it is not rejected on that ground. *)
        | Some entry
          when String.equal entry.SC.source_digest base_digest
               && String.equal entry.SC.numerics (SC.numerics_tag ())
               && Option.value_map entry.SC.codegen ~default:true ~f:(String.equal codegen_tag)
               && Option.value_map entry.SC.objective ~default:true ~f:(String.equal objective) -> (
            let spec =
              match entry.SC.segments with
              (* A fissioned entry with a non-empty [saved] is a split-reduce winner: [saved] is the
                 whole-routine prelude, [segments] the post-prelude per-segment schedules. *)
              | Some segs when not (List.is_empty entry.SC.saved) ->
                  Fiss (F_split_saved (entry.SC.saved, segs))
              | Some segs -> Fiss (F_saved segs)
              | None -> Whole (W_saved entry.SC.saved)
            in
            progress_stage "cache_replay";
            match compile_spec_real Outcome.Cache_replay spec with
            | Ok c when not (dispatchable ~is_gpu c.all_opts) ->
                (* An entry written before the gh-ocannl-532 rule can name the serial baseline as
                   the winner: it was timed then, and it won by default whenever every candidate
                   failed to compile — the state gh-ocannl-521 recorded for every GPU backend.
                   Replaying it would reintroduce the single-work-item dispatch through the cache,
                   permanently and without ever timing anything. Rejected like a stale entry: the
                   fresh search below overwrites it. Rejecting the replay (rather than bumping
                   [entry_version]) keeps every sound entry, on this backend and on the CPU
                   backends, where an empty schedule is a legitimate winner. *)
                logf "cache entry replays to an unparallelized routine, re-searching: %s"
                  (spec_label spec);
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
                emit_report
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
                    candidates_contended = 0;
                    default_refused = false;
                    (* No search ran, so the only rejection this can carry is the baseline's. *)
                    candidates_failed = failed_count declines;
                    baseline_declined = Option.is_some baseline_decline;
                    declines = decline_summaries declines;
                    rounds_run = 0;
                    beam_width;
                    sketch_candidates = 0;
                    epilogue_sketch_candidates = 0;
                    fiss_sketch_candidates = 0;
                    fiss_sketch_timed = 0;
                    fiss_sketch_composite = `Ineligible;
                    fiss_sketch_playoff_timed = 0;
                    fiss_sketch_playoff_swaps = 0;
                    split_reduce_candidates = 0;
                    split_reduce_timed = 0;
                    split_reduce_composite_eligible = false;
                    split_reduce_composite_timed = false;
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
                    best_schedule = flat_schedule c.form;
                    source_digest = base_digest;
                    (* Replayed for the same reason as [mma_best_ms]: a flip chain whose incumbent
                       replayed would otherwise have no timed record to abandon a hopeless flip
                       against (gh-ocannl-1110). Empty for entries older than the field, and when
                       the storing search had another shape, whose candidate order an equal-depth
                       comparison cannot assume. *)
                    best_steps =
                      (match entry.SC.best_steps with
                      | Some t when String.equal t.SC.search_shape search_shape -> t.SC.steps
                      | Some _ | None -> []);
                  };
                Some (c.cctx, c.routine)
            | Error (Outcome.Classified classified) ->
                (* Stale or corrupt entry: fall through to a fresh search. *)
                logf "cache entry replay FAILED, re-searching: %s"
                  (Outcome.detail_of_cause classified.cause);
                None
            | Error (Outcome.Fatal _ as failure) -> raise_pre_search ~base:(census ()) failure)
        | _ -> None
      else None
    in
    (* A winner takes precedence. Without one, replay an abandoned prefix only under the shape that
       measured it and a current rule that still rejects it. A weaker incumbent, changed margin,
       absent rule or changed sampling cannot inherit yesterday's verdict. *)
    let replayed_abandonment =
      if use_cache && Option.is_none cached then
        Option.bind abandon ~f:(fun rule ->
            Option.bind (SC.lookup_abandonment ~dir:cache_dir ~key) ~f:(fun entry ->
                if
                  String.equal entry.SC.source_digest base_digest
                  && String.equal entry.SC.trajectory.SC.search_shape search_shape
                then
                  Option.map
                    (abandon_verdict rule ~k:beam_width ~steps:entry.SC.trajectory.SC.steps)
                    ~f:(fun ab -> (ab, entry))
                else None))
      else None
    in
    match (cached, replayed_abandonment) with
    | Some result, _ ->
        release_baseline ();
        result
    | None, Some (ab, entry) ->
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
    | None, None when not search ->
        logf
          "search disabled (autotune_search=false) and no cache entry: compiling the untuned \
           default";
        (* Before the fallback compile, which wants the memory. *)
        release_baseline ();
        (* After the compile, as in the no-cache branch above. The census the base compile already
           produced is carried whether this succeeds or fails. *)
        let reached = census () in
        let result = compile_untuned_default ~base:reached () in
        report_or_release reached ~result;
        result
    | None, None ->
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
        (* [None] when the baseline compile was declined (gh-ocannl-533): there is no routine to
           time and none to return, and the search runs on the scheduled candidates alone. *)
        let baseline =
          Option.map baseline_linked ~f:(fun (bctx, broutine) ->
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
        let baseline_contended = ref false and baseline_unbatched = ref false in
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
                progress_stage "baseline_timing";
                time_routine ~on_retry ~tag_failures:true ~timing ~repeats b.cctx b.routine
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
                      baseline_contended := true;
                      baseline_unbatched := timing_result.unbatched;
                      if timing_result.unbatched then
                        logf
                          "baseline timing refused: queued calibration measured no batch within \
                           the target"
                      else
                        logf
                          "baseline timing refused: contention or a degenerate clock reading made \
                           the sample window unusable";
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
        if !baseline_contended then Hash_set.remove seen base_digest;
        (match baseline_decline with
        | Some classified ->
            logf "baseline: DECLINED at %s %s" (phase_label classified.phase)
              (Outcome.detail_of_cause classified.cause)
        | None ->
            if baseline_timed then (
              logf "baseline: %.4f ms (digest %s)" baseline_ms (dshort base_digest);
              emit_calibration ~backend ~device ~limits ~routine:(Lazy.force routine_name)
                ~label:"baseline" ~digest:base_digest
                ~timing_result:(Option.value_exn !baseline_timing_result)
                [ base_opt ])
            else if baseline_dispatched then release_baseline ()
            else (
              (* An undispatched baseline never enters the beam, so the exit sweep cannot own it. *)
              release_baseline ();
              (* No calibration row: the model column is only meaningful next to a measurement. *)
              logf
                "baseline: NOT DISPATCHED, binds no hardware dimension on %s -- the whole routine \
                 would run in one work-item (gh-ocannl-532) (digest %s)"
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
              !on_candidate_timed b.routine.Context.name ~timed_so_far:!n_timed);
        let n_timings_contended = ref (if !baseline_contended then 1 else 0) in
        (* The subset of those windows refused because queued calibration measured no batch within
           its target (gh-ocannl-1098). Not a diagnosis (a stall on every probe reads the same), but
           a count that persists across idle reruns is a queue threshold's signature. *)
        let n_timings_unbatched = ref (if !baseline_unbatched then 1 else 0) in
        (* [n_timings_contended] counts WINDOWS, which is the right shape for "was this search's
           measurement set complete?" but the wrong one for any claim about WHICH candidate went
           unmeasured (Codex P2 on PR #608): a refused digest is dropped from [seen] so an
           equivalent seed can retry it, so one candidate refused twice is two windows. The digests
           are therefore kept alongside, and the report derives from them a distinct-candidate count
           and the one per-candidate fact a caller cannot reconstruct — whether the untuned-default
           seed was the refused one, which is what separates a contention-refused reference from the
           gh-ocannl-552 regression of never proposing or attributing it. *)
        let contended_digests = Hash_set.create (module String) in
        if !baseline_contended then Hash_set.add contended_digests base_digest;
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
        and n_sr_timed = ref 0
        and sr_composite_eligible = ref false
        and sr_composite_timed = ref false in
        let sr_single_results = ref [] in
        let coarse_single_measured = Hash_set.create (module String) in
        let coarse_single_refused = Hash_set.create (module String) in
        let rounds_run = ref 0 in
        let n_sketch_candidates = ref 0
        and n_epilogue_sketch_candidates = ref 0
        and n_fiss_sketch_candidates = ref 0
        and n_split_reduce_candidates = ref 0 in
        let best_so_far = ref ((if baseline_timed then baseline else None), baseline_ms) in
        (* [report.best_steps], newest first: each time [best_so_far] improves, at which admitted
           timing it did (gh-ocannl-1110). *)
        let best_steps = ref (if baseline_timed then [ (1, baseline_ms) ] else []) in
        let by_time (_, a) (_, b) = Float.compare a b in
        (* gh-ocannl-550: the search's live artifacts are bounded by [beam_width], not by candidates
           processed. [beam] IS the candidate pool — it holds the fastest [beam_width] entries seen
           so far, and [admit] releases whatever falls out of it. It starts with the baseline when
           the baseline is eligible; a declined one contributes no entry, so the beam can be empty
           and every consumer below takes that as "nothing was timed" (gh-ocannl-533).

           Bounding as we go is equivalent to the old "keep every timed candidate, sort, then take
           [beam_width]" — keeping the k smallest incrementally keeps the k smallest overall — with
           one difference: a tie between exactly equal times now resolves by arrival rather than by
           seed order.

           Why bound it at all: a candidate's device buffers are invisible to the OCaml GC, because
           the backends' pool tables root every slab they allocate (see {!Context.release}), so a
           pool holding every ranked candidate holds its device memory too — a cold tf32 gpt2_mini
           search filled a 12 GB card a fifth of the way through and then ran the remaining
           candidates, its winner replay and its fallback compile against a full device. The tune
           loop is the one place that needs no allocator to fix that: it knows each candidate's
           exact lifetime — timed, then dead unless it is a beam survivor. *)
        let beam =
          ref
            (Option.to_list
               (Option.map
                  (if baseline_timed then baseline else None)
                  ~f:(fun b -> (b, baseline_ms))))
        in
        (* The beam-expansion round's own bounded accumulator, hoisted to this scope for one reason:
           the exit sweep has to be able to see it. A fatal launch/sync failure part way through a
           round used to abandon up to [beam_width] already-timed survivors that were in neither
           [beam] nor [best_so_far] (gh-ocannl-550, round-three review). Reset at the top of each
           round. *)
        let round = ref [] in
        (* Own the just-compiled candidate until admission transfers it to beam/round. In
           particular, a nonwinner is not in [best_so_far] when either callback raises. *)
        let pending = ref None in
        (* Set by the exit sweep: past it there is no reader left for any candidate the search
           compiled, so retention stops applying. A flag rather than clearing [best_so_far], which
           the reports still read for the winner's label after the sweep has freed its buffers. *)
        let search_over = ref false in
        let release_candidate c =
          (* Physical identity, not digest: the beam is the authority on what is live, and a
             released candidate's digest deliberately STAYS in [seen] — it must keep deduplicating,
             and dedup cannot resurrect an artifact, since [seen], [timed_ms_by_digest] and
             [label_by_digest] hold strings and floats and never a [compiled]. [best_so_far] is
             normally the beam's head, but it can lag one round behind it (a sub-threshold
             improvement updates the former and not the latter), so it is checked separately. *)
          if
            !search_over
            || not
                 (List.exists !beam ~f:(fun (c', _) -> phys_equal c c')
                 || List.exists !round ~f:(fun (c', _) -> phys_equal c c')
                 || Option.exists (fst !best_so_far) ~f:(phys_equal c))
          then (
            if Option.exists !pending ~f:(phys_equal c) then pending := None;
            (* Best-effort: a failure to free must not replace the candidate's own outcome, and this
               runs on failure paths too, where the device may already be refusing work.
               Process-fatal conditions still propagate. *)
            release_quietly ~what:("candidate " ^ dshort c.digest_after) c.cctx)
        in
        let admit entry =
          let kept, evicted =
            List.split_n (List.sort (entry :: !beam) ~compare:by_time) beam_width
          in
          beam := kept;
          pending := None;
          List.iter evicted ~f:(fun (c, _) -> release_candidate c)
        in
        (* The exit sweep. Once the search has produced its report, the beam survivors and the
           running best have no reader left either — and on the [timing_ctx] path not even the
           winner does, since it is recompiled from the caller's context out of its saved schedule,
           which is data. Ordering matters twice: the sweep must run AFTER the report record has
           been built (it reads [best_so_far]) and BEFORE the compiles that follow it, which are the
           two the exhausted device used to defeat (the winner replay and the untuned-default
           fallback behind it). *)
        let release_all_candidates ~keep () =
          search_over := true;
          let live =
            List.map !beam ~f:fst @ List.map !round ~f:fst
            @ Option.to_list (fst !best_so_far)
            @ Option.to_list !pending
          in
          beam := [];
          round := [];
          pending := None;
          List.iter live ~f:(fun c ->
              if not (List.exists keep ~f:(phys_equal c)) then release_candidate c)
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
           candidate reproduces (the whole-routine presets use [min_parallel:1]): no attribution,
           rather than labeling a differently-scheduled pipeline as the default. *)
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
            candidates_contended = candidates_contended ();
            default_refused = default_refused ();
            candidates_failed = failed_count declines;
            baseline_declined = Option.is_some baseline_decline;
            declines = summaries;
            rounds_run = !rounds_run;
            beam_width;
            sketch_candidates = !n_sketch_candidates;
            epilogue_sketch_candidates = !n_epilogue_sketch_candidates;
            fiss_sketch_candidates = !n_fiss_sketch_candidates;
            fiss_sketch_timed = !n_fiss_sketch_timed;
            fiss_sketch_composite = !fs_composite;
            fiss_sketch_playoff_timed = !n_playoff_timed;
            fiss_sketch_playoff_swaps = !n_playoff_swaps;
            split_reduce_candidates = !n_split_reduce_candidates;
            split_reduce_timed = !n_sr_timed;
            split_reduce_composite_eligible = !sr_composite_eligible;
            split_reduce_composite_timed = !sr_composite_timed;
            mma_candidates = !n_mma_proposed;
            fiss_mma_candidates = !n_fiss_mma_proposed;
            mma_timed = !n_mma_timed;
            model_scored = !n_model_scored;
            model_pruned = !n_model_pruned;
            bound_pruned = !n_bound_pruned;
            fissioned = Option.exists best_c ~f:(fun c -> is_fissioned c.form);
            baseline_ms;
            default_ms = default_ms ();
            best_ms;
            best_label = winner_label best_c;
            best_tensorized = winner_tensorized best_c;
            best_tensorization =
              Option.map best_c ~f:(fun c -> (mma_summary c).Ir.C_syntax.tensorization);
            best_mma_statements = Option.value_map best_c ~default:0 ~f:mma_statements;
            best_mma_scalar_fallbacks = Option.value_map best_c ~default:0 ~f:mma_scalar_fallbacks;
            mma_best_ms = !mma_best_ms;
            best_schedule = Option.value_map best_c ~default:[] ~f:(fun c -> flat_schedule c.form);
            source_digest = base_digest;
            best_steps = List.rev !best_steps;
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
          release_all_candidates ~keep:[] ();
          Outcome.raise_failure (Outcome.Fatal fatal)
        in
        (* The post-search fallbacks to the untuned default (nothing timed; the winner replay failed
           or degenerated). Through the containment-aware form, so a failure here reports the phase
           it carries — the outer catch-all would otherwise record every one of them as [Transform],
           which for a link failure is simply wrong. The exception the caller sees is unchanged:
           [emit_partial_and_raise] ends in [raise_failure], exactly as [Context.compile] does. *)
        let untuned_default_or_raise () =
          progress_stage "untuned_default_compile";
          match
            Context.compile_outcome ?name ~provenance:Ir.Schedule_outcome.User_schedule ctx comp
              bindings
          with
          | Ok result -> result
          | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
          | Error (Outcome.Classified classified) ->
              emit_partial_and_raise
                (Outcome.fatal_of_classified ~candidate:"untuned default fallback" classified)
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
          let try_spec spec =
            !on_candidate_attempt (spec_label spec);
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
                | Whole _ | Fiss (F_preset _ | F_saved _ | F_split _ | F_split_saved _) -> ()
              end;
              match timed_into progress_compile_s (fun () -> compile_spec spec) with
              | Error (Outcome.Classified classified) ->
                  record_decline declines classified;
                  logf "%s: FAILED at %s %s" (spec_label spec) (phase_label classified.phase)
                    (Outcome.detail_of_cause classified.cause);
                  None
              | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
              | Ok c ->
                  pending := Some c;
                  (* Recorded whether or not this compile goes on to be timed: on dedup the code was
                     (or will not be) timed under the same digest, and the [default_ms] lookup
                     follows the digest, not the seed (gh-ocannl-552). Which seed, if any, is
                     [default_seed_label]'s to say. *)
                  if Option.exists default_candidate_label ~f:(String.equal (spec_label spec)) then
                    default_seed_digest := Some c.digest_after;
                  if Hash_set.mem seen c.digest_after then (
                    logf "%s: dedup (digest %s)" (spec_label spec) (dshort c.digest_after);
                    (* gh-ocannl-550: a dedup still PAID for a compile and a link, so it holds a
                       candidate's worth of device buffers — and its identical twin, already in the
                       beam or already released, is the one the search reasons about. This one is
                       dead on arrival. The digest stays in [seen]. *)
                    release_candidate c;
                    None)
                  else if not (dispatchable ~is_gpu c.all_opts) then (
                    (* Degenerated to the serial form (gh-ocannl-532): recorded as seen, so an
                       equivalent later candidate dedups rather than re-deriving the same skip. *)
                    Hash_set.add seen c.digest_after;
                    logf "%s: NOT DISPATCHED, binds no hardware dimension (digest %s)"
                      (spec_label spec) (dshort c.digest_after);
                    record_not_dispatched ~origin:"candidate"
                      ~detail:
                        (Printf.sprintf "%s degenerated to a form binding no hardware dimension"
                           (spec_label spec));
                    release_candidate c;
                    None)
                  else (
                    Hash_set.add seen c.digest_after;
                    match
                      (* The backend's own classifier decides whether a launch or sync failure is this
                   candidate's fault: the driver error is all the evidence there is, and only the
                   backend can read it. With the always-[None] classifier this used to pass, no
                   backend could ever declare one, so every launch failure was fatal by phase
                   default and there was nowhere for a backend to plug one in (gh-ocannl-536; the
                   HIP scratch-overflow arm of gh-ocannl-533 is what fills this seam). The phase
                   reaching the report is the tagged one inside [time_routine], so a report
                   distinguishes a launch refusal from an asynchronous failure at sync.

                   And from the third case, which is not the backend's to judge: [time_routine]'s
                   pre-dispatch validation carries [Preflight] and is contained without asking the
                   classifier (gh-ocannl-564). Tagged [Launch] it was fatal on every C backend, so a
                   scratch context missing one of the caller's initializations condemned the search
                   instead of declining a candidate. *)
                      (* The lineage-wide validation is OUTSIDE the boundary (gh-ocannl-569): a poisoned
                   lineage, an uninitialized input and an unexecuted dependency are properties of
                   the context and the computation, so a genuine one fails every candidate of every
                   arm at once. Contained as a decline it is silent — on a backend whose serial
                   baseline is not dispatched (every GPU backend) every candidate declines for the
                   one reason, nothing is timed, and the search ships the untuned default out of an
                   unusable lineage under a report that says it completed. It reaches the caller
                   instead, which is the only party that can fix it.

                   Tagged, though not contained: the tag carries no boundary here, it only labels
                   the phase so the fallback handler at the end of [search] reports a pre-dispatch
                   validation failure as [Preflight] rather than as its [Transform] default. *)
                      Outcome.tag Outcome.Preflight (fun () ->
                          Context.check_lineage_runnable c.cctx c.routine);
                      Outcome.protect ~classify_backend:(Context.failure_classifier c.cctx)
                        ~provenance:Outcome.Candidate ~phase:Outcome.Launch
                        ~candidate:(spec_label spec) (fun () ->
                          timed_into progress_timing_s (fun () ->
                              time_routine ~on_retry ~tag_failures:true ~phase:!progress_phase
                                ~timing ~repeats c.cctx c.routine))
                      (* Outside the boundary: the seam is not a candidate failure to classify. *)
                      |> Result.map
                           ~f:
                             (apply_measurement_seam ~label:(spec_label spec) ~digest:c.digest_after)
                    with
                    | Ok timing_result when Option.is_none (admitted_timing_ms timing_result) ->
                        (* The family counters answer whether a candidate compiled and reached a
                           timing window, independently of whether that window yielded a usable
                           verdict. Keep that accounting stable under refusal; the historical
                           [timings_contended] counter covers every unusable timing result. *)
                        (match spec with
                        | Fiss (F_sketch _) when !in_playoff -> Int.incr n_playoff_timed
                        | Fiss (F_sketch { entries; fine }) -> (
                            Int.incr n_fiss_sketch_timed;
                            if not fine then
                              match entries with
                              | [ (key, _) ] -> Hash_set.add coarse_single_refused key
                              | _ :: _ :: _ -> fs_composite := `Refused
                              | [] -> ())
                        | Fiss (F_split { sites }) ->
                            Int.incr n_sr_timed;
                            if List.length sites >= 2 then sr_composite_timed := true
                        | _ -> ());
                        if spec_expects_mma spec then Int.incr n_mma_timed;
                        Int.incr n_timings_contended;
                        if timing_result.unbatched then Int.incr n_timings_unbatched;
                        Hash_set.add contended_digests c.digest_after;
                        (* Unlike a launch/compile decline, contention is not a property of this
                           schedule. An equivalent later seed is a useful retry, not a dedup. *)
                        Hash_set.remove seen c.digest_after;
                        if timing_result.unbatched then
                          logf
                            "%s: NOT TIMED, queued calibration measured no batch within the target \
                             (digest %s)"
                            (spec_label spec) (dshort c.digest_after)
                        else
                          logf
                            "%s: NOT TIMED, contention or a degenerate clock reading made the \
                             sample window unusable (digest %s)"
                            (spec_label spec) (dshort c.digest_after);
                        release_candidate c;
                        None
                    | Ok timing_result ->
                        let ms = Option.value_exn (admitted_timing_ms timing_result) in
                        Int.incr n_timed;
                        (* Publish window accounting before the post-admission injection seam: it
                           can raise, and the partial report still owns this completed window. *)
                        (match spec with
                        | Fiss (F_sketch _) when !in_playoff -> Int.incr n_playoff_timed
                        | Fiss (F_sketch { entries; fine }) -> (
                            Int.incr n_fiss_sketch_timed;
                            if not fine then
                              match entries with
                              | [ (key, _) ] -> Hash_set.add coarse_single_measured key
                              | _ :: _ :: _ -> fs_composite := `Timed
                              | [] -> ())
                        | Fiss (F_split { sites }) -> (
                            Int.incr n_sr_timed;
                            match sites with
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
                          let previous = fst !best_so_far in
                          best_so_far := (Some c, ms);
                          best_steps := (!n_timed, ms) :: !best_steps;
                          (* A timing tie can evict the old best from the beam while the best
                             reference retains it. Replacing that last owner must release it. *)
                          Option.iter previous ~f:release_candidate);
                        !on_candidate_callback `Timed ~candidate_ms:ms
                          ~incumbent_ms:(snd !best_so_far);
                        !on_candidate_timed c.routine.Context.name ~timed_so_far:!n_timed;
                        logf "%s: %.4f ms (digest %s)" (spec_label spec) ms (dshort c.digest_after);
                        !on_candidate_callback `Calibration ~candidate_ms:ms
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
          let try_spec spec =
            let result = try_spec spec in
            (* Gated explicitly, not just by [logf]: [logf]'s arguments are evaluated whether or not
               the flag is on, and both readings here fold a hashtable. *)
            if Lazy.force log_enabled then
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
          let try_spec spec =
            check_abandon ();
            Int.incr progress_attempts;
            Int.incr progress_tried;
            (* Guarded here, not only inside [progress_line]: its arguments render a label and read
               the best-so-far table on every attempt of a search that prints nothing. *)
            if progress_enabled () then
              progress_line "candidate"
                (Printf.sprintf "%s attempt=%S" (progress_where ()) (spec_label spec));
            try_spec spec
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
             and the baseline are never pruned. *)
          let model_prefilter_params ~seg_opt ~family params =
            if Float.(keep_fraction >= 1.) || List.length params <= 1 then params
            else
              let scored =
                List.map params ~f:(fun p ->
                    let score =
                      model_score ~static_indices ~limits seg_opt
                        (sketch_schedule ~accum_prec ~p seg_opt)
                    in
                    (p, score))
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
            model_prefilter_params ~seg_opt:base_opt ~family:"whole-routine"
              (sketch_seed_params ~is_gpu ~is_cpu ~limits base_opt)
          in
          n_sketch_candidates := List.length sketch_params;
          n_epilogue_sketch_candidates := List.count sketch_params ~f:(fun p -> p.sk_epilogue);
          (* Per-fission-segment sketch seeds (the [F_sketch] flavor): heavily fissioned graphs tune
             per segment, where the whole-routine sketches never apply. Enumerate the fission
             segmentation once, on a hermetic copy of the base lowering with the same pipeline
             settings the candidate transform uses ([preset_sched]'s defaults), and detect a matmul
             site per [`Normal] segment — keyed by the segment's structural pre-schedule digest,
             like [F_saved]. *)
          let enum_fiss_entries ~arity_cuts =
            if not (is_gpu || is_cpu) then []
            else
              let scratch =
                {
                  base_opt with
                  LL.traced_store = Hashtbl.copy base_opt.LL.traced_store;
                  LL.optimize_ctx = LL.copy_optimize_ctx base_opt.LL.optimize_ctx;
                }
              in
              let preset seg =
                if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits seg
                else Sched.default_cpu ~min_parallel:1 seg
              in
              let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
              match
                Sched.fission_scheduled ~promote_locals:is_gpu
                  ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
                  ~arity_cuts ~preset ~zero_sched ~static_indices scratch
              with
              | exception Outcome.Cause_at _ -> []
              | [] | [ _ ] -> [] (* Unfissioned: the whole-routine sketches cover the site. *)
              | tuples ->
                  List.filter_map tuples ~f:(fun (kind, pre, _, _) ->
                      match kind with
                      | `Zeros | `Solo -> None
                      | `Normal -> (
                          match sketch_seed_params ~is_gpu ~is_cpu ~limits pre with
                          | [] -> None
                          | params ->
                              Some
                                ( SC.digest
                                    (SC.canonicalize ~static_indices ~with_placements:false pre),
                                  pre,
                                  params )))
          in
          let dedup_by_key entries =
            (* Structurally identical segments share a digest — and thus, at apply time, a schedule
               — so keep one entry per digest. *)
            List.fold entries ~init:[] ~f:(fun acc ((key, _, _) as e) ->
                if List.exists acc ~f:(fun (k, _, _) -> String.equal k key) then acc else e :: acc)
            |> List.rev
          in
          let prefilter_entries ~tag entries =
            (* Per-segment pre-filtering: each segment's sketches are their own family —
               cross-segment scores are incomparable (different code volumes), and the singles below
               are also ranked per segment by the recombination step. *)
            List.map entries ~f:(fun (key, pre, ps) ->
                (key, model_prefilter_params ~seg_opt:pre ~family:(tag ^ dshort key) ps))
          in
          let fiss_sketch_entries =
            prefilter_entries ~tag:"segment " (dedup_by_key (enum_fiss_entries ~arity_cuts:false))
          in
          (* The finer ([arity_cuts]) segmentation (gh-ocannl-574): cutting apart a companion that
             cannot follow its site's full arity — the lm_head's max-logits reduction and that
             reduction's initialization nest — frees the site's kernel to seed at full arity, where
             the shared segment's every seed declines on companion coverage. GPU-only: the
             constraint the cut relieves is the GPU sketches' kernel-global launch geometry. Only
             segments whose digest is {e new} versus the coarse segmentation seed singles — an
             unchanged segment's parameters are already timed by its coarse single, and the extra
             cuts elsewhere in a fine twin could only add launches — but the full fine key list is
             kept so the fine recombination below can staff unchanged segments from coarse-timed
             bests. *)
          let fine_all_entries =
            if not is_gpu then []
            else
              let fine = dedup_by_key (enum_fiss_entries ~arity_cuts:true) in
              let coarse_keys = List.map fiss_sketch_entries ~f:fst in
              if List.for_all fine ~f:(fun (k, _, _) -> List.mem coarse_keys k ~equal:String.equal)
              then [] (* The finer mode cut nothing new: the coarse seeds cover every segment. *)
              else prefilter_entries ~tag:"fine segment " fine
          in
          let fine_new_entries =
            let coarse_keys = List.map fiss_sketch_entries ~f:fst in
            List.filter fine_all_entries ~f:(fun (k, _) ->
                not (List.mem coarse_keys k ~equal:String.equal))
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
            List.concat_map fiss_sketch_entries ~f:(fun (key, ps) ->
                List.map ps ~f:(fun p -> Fiss (F_sketch { entries = [ (key, p) ]; fine = false })))
            @ List.concat_map fine_new_entries ~f:(fun (key, ps) ->
                List.map ps ~f:(fun p -> Fiss (F_sketch { entries = [ (key, p) ]; fine = true })))
          in
          n_fiss_sketch_candidates := List.length fiss_sketch_specs;
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
                  "site %s red%d out%d cost%d%s evicted by autotune_split_reduce_max_sites=%d"
                  (Ir.Tnode.debug_name s.sr_target) s.sr_red s.sr_out s.sr_cost
                  (match List.length s.sr_swaps with 0 -> "" | n -> Printf.sprintf " swap%d" n)
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
                    if 2 * b <= s.sr_red then Some (Fiss (F_split { sites = [ (s, b) ] })) else None))
          in
          n_split_reduce_candidates := List.length sr_specs;
          let seed_specs =
            block_size_presets (fun block_size -> Whole (W_preset { block_size }))
            @ (if is_gpu || is_cpu then
                 (* Each fissioned preset is seeded plain and privatized (the latter dedups away by
                    digest when no accumulator is eligible). The [config_thresholds] seeds reproduce
                    the untuned default pipeline exactly (plus its privatized variant), so the
                    winner is never worse than not tuning — the aggressive [min_parallel:1] presets
                    can all lose to it on launch-overhead-bound workloads. *)
                 List.concat_map [ false; true ] ~f:(fun privatize ->
                     Fiss (F_preset { block_size = None; privatize; config_thresholds = true })
                     :: block_size_presets (fun block_size ->
                         Fiss (F_preset { block_size; privatize; config_thresholds = false })))
               else [])
            @ List.map sketch_params ~f:(fun p -> Whole (W_sketch p))
            @ fiss_sketch_specs @ sr_specs
          in
          let fiss_single_results = ref [] in
          progress_phase_begin "seeds" (Some (List.length seed_specs));
          List.iter seed_specs ~f:(fun spec ->
              let result = try_spec spec in
              (match (spec, result) with
              | Fiss (F_sketch { entries = [ (key, p) ]; fine }), Some (_, ms) ->
                  fiss_single_results := (key, fine, (p, ms)) :: !fiss_single_results
              | _ -> ());
              Option.iter result ~f:admit);
          (* At most three composites, each proposed only if its singles justify it: no total to
             announce, so no phase line either; a composite's attempt still counts and can print. *)
          progress_phase_begin "recombine" None;
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
                String.equal k key && (fine_ok || not fine))
            |> List.min_elt ~compare:(fun (_, _, (_, a)) (_, _, (_, b)) -> Float.compare a b)
            |> Option.map ~f:(fun (_, _, (p, _)) -> (key, p))
          in
          let recombined =
            List.filter_map fiss_sketch_entries ~f:(fun (key, _) ->
                best_single_for ~fine_ok:false key)
          in
          (* Missing singles justify undecided only when every missing coarse key had a viable
             window refused and no later equivalent single supplied a measurement. A refusal from an
             unrelated segment or family cannot excuse a missing proposal. *)
          fs_composite :=
            if List.length recombined >= 2 then `Proposed
            else if
              List.length fiss_sketch_entries >= 2
              && List.for_all fiss_sketch_entries ~f:(fun (key, _) ->
                  List.Assoc.mem recombined key ~equal:String.equal
                  || Hash_set.mem coarse_single_refused key
                     && not (Hash_set.mem coarse_single_measured key))
            then `Singles_refused
            else `Ineligible;
          (* A timed composite, as the playoff's starting incumbent: its entries, its segmentation
             and its time. *)
          let try_composite ~fine entries =
            Option.map
              (try_spec (Fiss (F_sketch { entries; fine })))
              ~f:(fun ((_, ms) as r) ->
                admit r;
                (entries, fine, ms))
          in
          let coarse_composite =
            if Poly.equal !fs_composite `Proposed then try_composite ~fine:false recombined
            else None
          in
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
          let fine_composite =
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
          sr_composite_eligible := List.length recombined >= 2;
          if !sr_composite_eligible then
            Option.iter (try_spec (Fiss (F_split { sites = recombined }))) ~f:admit;
          (* The composite playoff (gh-ocannl-1166; see [playoff_margin]), after every recombination
             composite so the [playoff] progress phase holds only its windows: starting from the
             faster timed composite, each key in turn tries its near-tie singles in the composite's
             place, and a faster alternate becomes the incumbent the later keys build on. Coordinate
             descent over the keys, at most [playoff_width] alternates each, so the cost is linear
             in the keyed segments rather than the cartesian product the recombination avoids. *)
          let playoff_contenders ~fine_ok ~current key =
            let singles =
              List.filter_map !fiss_single_results ~f:(fun (k, fine, (p, ms)) ->
                  if String.equal k key && (fine_ok || not fine) then Some (p, ms) else None)
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
          (match
             List.min_elt
               (List.filter_opt [ coarse_composite; fine_composite ])
               ~compare:(fun (_, _, a) (_, _, b) -> Float.compare a b)
           with
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
                         let current = List.Assoc.find_exn incumbent ~equal:String.equal key in
                         List.fold (playoff_contenders ~fine_ok:fine ~current key)
                           ~init:(incumbent, incumbent_ms) ~f:(fun (incumbent, incumbent_ms) p ->
                             (* What [p] would replace: [current] until an earlier alternate of this
                                key swapped in. *)
                             let replaced = List.Assoc.find_exn incumbent ~equal:String.equal key in
                             let alternate =
                               List.map incumbent ~f:(fun (k, q) ->
                                   if String.equal k key then (k, p) else (k, q))
                             in
                             match try_composite ~fine alternate with
                             | Some (_, _, alt_ms) when Float.(alt_ms < incumbent_ms) ->
                                 Int.incr n_playoff_swaps;
                                 logf "playoff: segment %s: %s replaces %s (%.4f ms < %.4f ms)"
                                   (dshort key) (spec_label (Whole (W_sketch p)))
                                   (spec_label (Whole (W_sketch replaced)))
                                   alt_ms incumbent_ms;
                                 (alternate, alt_ms)
                             | Some _ | None -> (incumbent, incumbent_ms)))
                      : (string * sketch_params) list * float)));
          (* [None] iff the beam is empty: no candidate timed and the baseline was not eligible (an
             undispatched GPU baseline never enters the beam with a finite rank; a declined one does
             not enter it at all). *)
          let best = ref (List.hd !beam) in
          let continue_ = ref true in
          while !continue_ && !rounds_run < rounds do
            Int.incr rounds_run;
            let per_entry =
              List.map !beam ~f:(fun (elem, ms) ->
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
                  ( elem,
                    ms,
                    List.concat_map elem.units ~f:(fun u ->
                        List.filter_map
                          (menu ~admits ~is_cpu ~is_gpu ~limits ~registry:u.u_registry u.u_opt)
                          ~f:(fun op -> extend_spec elem u op)) ))
            in
            let cands = List.concat_map per_entry ~f:(fun (_, _, c) -> c) in
            (* gh-ocannl-1137: what the round expands. The beam is the [beam_width] fastest timed
               candidates, and a near-tie sweep (the gh-ocannl-1166 composite playoff, a block-size
               ladder) can fill it with variants of one schedule; [units differing] counts each
               entry's units whose saved schedule differs from the incumbent's, so a beam of
               near-copies reads as a column of small numbers. *)
            (match per_entry with
            | [] -> ()
            | (head, _, _) :: _ ->
                logf "round %d: %d candidate(s) from a beam of %d: %s" !rounds_run
                  (List.length cands) (List.length per_entry)
                  (String.concat ~sep:"; "
                     (List.map per_entry ~f:(fun (elem, ms, c) ->
                          Printf.sprintf "%s %.4f ms, %s units differing, %d move(s)"
                            (dshort elem.digest_after) ms
                            (Option.value_map (units_differing head elem) ~default:"resegmented"
                               ~f:Int.to_string)
                            (List.length c)))));
            (* gh-ocannl-550: bounded like the seed pass, but in a SECOND accumulator, because a
               round's decision compares its own best against the incumbent and, if it wins,
               replaces the beam wholesale — so the previous beam has to stay alive until that
               decision is taken, and this round's also-rans must not (16 compiles per round on the
               gh-ocannl-543 chain). An evicted entry is provably outside [!round] by the time it is
               released, so [release_candidate]'s beam/best check is the whole guard it needs. *)
            round := [];
            let round_admit entry =
              let kept, evicted =
                List.split_n (List.sort (entry :: !round) ~compare:by_time) beam_width
              in
              round := kept;
              pending := None;
              List.iter evicted ~f:(fun (c, _) -> release_candidate c)
            in
            progress_phase_begin (Printf.sprintf "round%d" !rounds_run) (Some (List.length cands));
            List.iter cands ~f:(fun spec -> Option.iter (try_spec spec) ~f:round_admit);
            match !round with
            | [] -> continue_ := false
            | (_, round_best_ms) :: _ ->
                let incumbent_ms = Option.value_map !best ~default:Float.infinity ~f:snd in
                let previous = !beam in
                if Float.(round_best_ms < incumbent_ms *. (1. -. min_progress)) then (
                  beam := !round;
                  best := List.hd !beam;
                  (* The displaced incumbents are dead. *)
                  List.iter previous ~f:(fun (c, _) -> release_candidate c))
                else (
                  continue_ := false;
                  (* The round did not beat the incumbent by enough: the beam is unchanged, so
                     everything this round produced is dead — except a sub-threshold improvement
                     that became [best_so_far], which [release_candidate] keeps and the exit cleanup
                     releases. *)
                  let produced = !round in
                  round := [];
                  List.iter produced ~f:(fun (c, _) -> release_candidate c))
          done;
          let best_c, best_ms =
            match !best with Some (c, ms) -> (Some c, ms) | None -> (None, Float.infinity)
          in
          (* Nothing was timed exactly when every candidate failed and (on GPU) the serial baseline
             was never run — or, since gh-ocannl-533, was itself declined. Nothing measured means
             nothing to cache: a stored entry would pin future processes to a never-timed
             schedule. *)
          let nothing_timed = Float.is_inf best_ms in
          (if use_cache then
             if
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
                   default_ms = default_ms ();
                   default_fingerprint =
                     Option.map (default_ms ()) ~f:(fun _ ->
                         Sched.default_schedule_fingerprint ~backend_name:backend);
                   best_steps = Some { SC.search_shape; steps = List.rev !best_steps };
                 });
          (* Diagnostic control (config [autotune_log]): compile and time the UNTUNED default
             pipeline in this very process, on the search context — discriminates a genuinely slow
             winner from process-state effects when the winner's code nominally equals the untuned
             program yet a separately-run untuned process measures faster (PR #140 round 6: same
             digest, 3.4x runtime difference across processes on cuda). *)
          (if Lazy.force log_enabled then
             match
               progress_stage "untuned_control";
               Context.compile ?name search_ctx comp bindings
             with
             | cctx, croutine ->
                 (match time_routine ~timing ~repeats cctx croutine with
                 | timing_result -> (
                     match admitted_timing_ms timing_result with
                     | Some ms -> logf "untuned-default in-process control: %.4f ms" ms
                     | None ->
                         logf
                           "untuned-default in-process control: refused (contention or degenerate \
                            clock reading)")
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
              candidates_contended = candidates_contended ();
              default_refused = default_refused ();
              candidates_failed = failed_count declines;
              baseline_declined = Option.is_some baseline_decline;
              declines = decline_summaries declines;
              rounds_run = !rounds_run;
              beam_width;
              sketch_candidates = List.length sketch_params;
              epilogue_sketch_candidates = List.count sketch_params ~f:(fun p -> p.sk_epilogue);
              fiss_sketch_candidates = List.length fiss_sketch_specs;
              fiss_sketch_timed = !n_fiss_sketch_timed;
              fiss_sketch_composite = !fs_composite;
              fiss_sketch_playoff_timed = !n_playoff_timed;
              fiss_sketch_playoff_swaps = !n_playoff_swaps;
              split_reduce_candidates = List.length sr_specs;
              split_reduce_timed = !n_sr_timed;
              split_reduce_composite_eligible = !sr_composite_eligible;
              split_reduce_composite_timed = !sr_composite_timed;
              mma_candidates = !n_mma_proposed;
              fiss_mma_candidates = !n_fiss_mma_proposed;
              mma_timed = !n_mma_timed;
              model_scored = !n_model_scored;
              model_pruned = !n_model_pruned;
              bound_pruned = !n_bound_pruned;
              fissioned = Option.exists best_c ~f:(fun c -> is_fissioned c.form);
              baseline_ms;
              default_ms = default_ms ();
              best_ms;
              best_label = winner_label best_c;
              best_tensorized = winner_tensorized best_c;
              best_tensorization =
                Option.map best_c ~f:(fun c -> (mma_summary c).Ir.C_syntax.tensorization);
              best_mma_statements = Option.value_map best_c ~default:0 ~f:mma_statements;
              best_mma_scalar_fallbacks = Option.value_map best_c ~default:0 ~f:mma_scalar_fallbacks;
              mma_best_ms = !mma_best_ms;
              best_schedule = Option.value_map best_c ~default:[] ~f:(fun c -> flat_schedule c.form);
              source_digest = base_digest;
              best_steps = List.rev !best_steps;
            }
          in
          let result =
            if nothing_timed then (
              (* Returning the incumbent here would hand the caller the very serial routine this
                 search refused to dispatch (gh-ocannl-532) — slower than not tuning at all, and on
                 GPU unbounded. The untuned default pipeline is the honest fallback: the same code
                 the caller would have compiled without the tuner. *)
              logf "nothing was timed: falling back to the untuned default compile (gh-ocannl-532)";
              release_all_candidates ~keep:[] ();
              untuned_default_or_raise ())
            else
              (* [nothing_timed] is false, so the beam holds a timed winner. *)
              let best_c = Option.value_exn best_c ~message:timed_winner_exists in
              if Option.is_none timing_ctx then (
                (* The winner's own artifacts ARE the return value here; every other candidate is
                   dead. *)
                release_all_candidates ~keep:[ best_c ] ();
                (best_c.cctx, best_c.routine))
              else
                (* The search ran against the scratch lineage; compile the winner from the caller's
                   context (like the cache-hit path). Digest mismatch or replay failure falls back
                   to the production default schedule. *)
                let spec =
                  match best_c.form with
                  | Whole_saved saved -> Whole (W_saved saved)
                  | Fiss_saved segs -> Fiss (F_saved segs)
                  | Split_saved (prelude, segs) -> Fiss (F_split_saved (prelude, segs))
                in
                (* Nothing the replay needs is an artifact — [spec] above is the winner's saved
                   schedule — so the whole beam goes before the compile that reproduces it
                   (gh-ocannl-550). *)
                release_all_candidates ~keep:[] ();
                progress_stage "winner_compile";
                match compile_spec_real Outcome.Candidate spec with
                | Ok c when not (dispatchable ~is_gpu c.all_opts) ->
                    (* Completes the invariant rather than fixing an observed bug: the winner was
                       timed, so it was dispatchable when measured, and the replay is
                       digest-guarded. But this is the last of the three ways [tune] hands back a
                       routine, and none of them may return an unparallelized GPU routine
                       (gh-ocannl-532). The default compile is the same fallback a failed replay
                       takes. *)
                    logf "winner replay produced an unparallelized routine, falling back: %s"
                      (spec_label spec);
                    (* gh-ocannl-550: rejected, so dead — and the fallback compile below wants the
                       memory. Same one-liner as the rejected cache replay above; the pre-replay
                       sweep could not cover this context, which did not exist yet. *)
                    release_quietly ~what:"the rejected winner replay" c.cctx;
                    untuned_default_or_raise ()
                | Ok c ->
                    logf "winner replay ok: %s" (spec_label spec);
                    (c.cctx, c.routine)
                | Error (Outcome.Classified classified) ->
                    logf "winner replay FAILED (%s), falling back to the default compile: %s"
                      (spec_label spec)
                      (Outcome.detail_of_cause classified.cause);
                    untuned_default_or_raise ()
                | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
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
              release_all_candidates ~keep:[] ();
              if
                use_cache
                && search_measurements_cacheable ~nothing_timed:false
                     ~timings_contended:r.timings_contended
              then
                SC.store_abandonment ~dir:cache_dir ~key
                  {
                    SC.version = SC.entry_version;
                    source_digest = base_digest;
                    trajectory = { SC.search_shape; steps = r.best_steps };
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
        report_or_release completed_report ~result;
        result
