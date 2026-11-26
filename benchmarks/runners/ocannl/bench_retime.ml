(** Re-timing a shipped routine with a stated, measured spread (gh-ocannl-1524).

    A runner that compiles a routine (a tuned winner, a model pick, an untuned default) and then
    reports how fast it runs used to time it min-of-N synchronized runs, once. On a kernel under ~50
    us that single minimum is noise-dominated: on cc, gh-ocannl-1397's unchanged control site read
    0.69x between two builds and a changed site 1.44x the wrong way, while the search's own
    [best_ms] (the tuner's queued protocol) agreed with itself to a few percent. Two things were
    missing, and this module supplies both:

    - The tuner's protocol. Each reading is one {!Autotune.time_routine} call under the configured
      [autotune_timing] and [autotune_repeats] -- warmup, calibration to a ~10 ms batch (on cc and
      Metal up to 200 launches per synchronization), and a minimum over a top-up window -- so a
      re-timing measures the quantity a search ranked its candidates by, with the submit/sync round
      trip amortized rather than paid per reading.
    - A spread. A routine is read [calls] times, each call independent (its own warmup, calibration
      and window), and the report is the median of those readings with their range. Several routines
      are read in one session, interleaved in {!Bench_design}'s spread mirrored rotation, so a drift
      over the session lands on every routine alike; their ratio is PAIRED, call by call, and
      reported as the median paired ratio with its range. A ratio is [resolved] only when every pair
      agrees on its side of 1 -- with the default six calls, the exact two-sided sign-test p of that
      outcome is 2/64 = 0.031.

    Below {!min_readings} admitted readings a spread is not stated: its bounds are NaN and a ratio
    over it is never resolved. A reading the tuner's admission gate refuses
    ({!Autotune.admitted_timing_ms}: host contention, an unbatched calibration, a clock that
    resolved nothing) is counted as refused and left out, never ranked.

    Boundary: the spread is the session's -- what this process measured about routines it compiled.
    Two builds compared across processes (a base and a head) carry, on top, whatever differs between
    the processes (code placement, the tuned arm's crown); pair them through [benchmarks/ab.py] or
    [benchmarks/bin_cells.py], round by round, and read each process's own spread beside the paired
    ratio. *)

open Base

let min_readings = 4

(** [calls_of_env ()]: [BENCH_RETIME_CALLS] (defaults to 6), the timing calls per routine. Even,
    because {!Bench_design} mirrors every batch, and at least {!min_readings}. *)
let calls_of_env ?(default = 6) () =
  match Stdlib.Sys.getenv_opt "BENCH_RETIME_CALLS" with
  | None -> default
  | Some s -> (
      match Int.of_string_opt (String.strip s) with
      | Some n when n >= min_readings && n % 2 = 0 -> n
      | _ ->
          failwith
            (Printf.sprintf
               "BENCH_RETIME_CALLS=%s: an even count of at least %d timing calls per routine \
                (gh-ocannl-1524)"
               s min_readings))

type spread = {
  ms : float;  (** The median admitted reading; NaN with none. *)
  lo_ms : float;  (** The least admitted reading; NaN below {!min_readings}. *)
  hi_ms : float;  (** The greatest admitted reading; NaN below {!min_readings}. *)
  admitted : int;
  refused : int;
}
(** One routine's re-timing: per-launch milliseconds, each reading one {!Autotune.time_routine}
    call. *)

let median sorted =
  let n = Array.length sorted in
  if n = 0 then Float.nan
  else if n % 2 = 1 then sorted.(n / 2)
  else (sorted.((n / 2) - 1) +. sorted.(n / 2)) /. 2.

let sorted_admitted readings =
  let xs = Array.filter_map readings ~f:Fn.id in
  Array.sort xs ~compare:Float.compare;
  xs

(** [spread_of readings]: the summary of one routine's readings, [None] for a refused call. *)
let spread_of (readings : float option array) =
  let xs = sorted_admitted readings in
  let n = Array.length xs in
  let stated = n >= min_readings in
  {
    ms = median xs;
    lo_ms = (if stated then xs.(0) else Float.nan);
    hi_ms = (if stated then xs.(n - 1) else Float.nan);
    admitted = n;
    refused = Array.length readings - n;
  }

(** [rel_spread s]: [(hi - lo) / median], the spread as a fraction of the reading; NaN unstated. *)
let rel_spread s = (s.hi_ms -. s.lo_ms) /. s.ms

type ratio = {
  ratio : float;  (** The median of the paired ratios; NaN with no pair. *)
  ratio_lo : float;  (** The least paired ratio; NaN below {!min_readings} pairs. *)
  ratio_hi : float;  (** The greatest paired ratio; NaN below {!min_readings} pairs. *)
  pairs : int;  (** The calls at which both routines' readings were admitted. *)
  resolved : bool;
      (** At least {!min_readings} pairs and every paired ratio on one side of 1: the two routines
          differ by more than the session's noise. Unresolved is "within the spread", not "equal".
      *)
}

(** [paired_ratio ~num ~den]: [num]'s readings over [den]'s, call by call. The two arrays are two
    routines' readings from one {!retime} session, index [i] of each taken in the same mirrored
    batch, so a drift common to the batch divides out. *)
let paired_ratio ~(num : float option array) ~(den : float option array) =
  if Array.length num <> Array.length den then
    invalid_arg "Bench_retime.paired_ratio: readings from sessions of different lengths";
  let rs =
    Array.filter_mapi num ~f:(fun i n ->
        match (n, den.(i)) with Some n, Some d -> Some (n /. d) | _ -> None)
  in
  let sorted = Array.copy rs in
  Array.sort sorted ~compare:Float.compare;
  let pairs = Array.length sorted in
  let stated = pairs >= min_readings in
  let ratio_lo = if stated then sorted.(0) else Float.nan
  and ratio_hi = if stated then sorted.(pairs - 1) else Float.nan in
  {
    ratio = median sorted;
    ratio_lo;
    ratio_hi;
    pairs;
    resolved = (stated && Float.(ratio_hi < 1. || ratio_lo > 1.));
  }

type controlled_ratio = {
  treatment : ratio;
  base_control : ratio;
  variant_control : ratio;
  effect_lo : float;
  effect_hi : float;
  control_factor : float;
}

(** Compare two variants with one repeat each. The caller discards incomplete rounds from all four
    columns. Either linked copy can carry a bias, so every cross pairing has to clear the repeat
    envelope in the same direction. *)
let controlled_ratio ~base ~base_repeat ~variant ~variant_repeat =
  let treatment = paired_ratio ~num:variant ~den:base in
  let base_control = paired_ratio ~num:base_repeat ~den:base in
  let variant_control = paired_ratio ~num:variant_repeat ~den:variant in
  let cross =
    [
      treatment;
      paired_ratio ~num:variant ~den:base_repeat;
      paired_ratio ~num:variant_repeat ~den:base;
      paired_ratio ~num:variant_repeat ~den:base_repeat;
    ]
  in
  let effect_lo = List.fold cross ~init:Float.infinity ~f:(fun lo r -> Float.min lo r.ratio_lo) in
  let effect_hi =
    List.fold cross ~init:Float.neg_infinity ~f:(fun hi r -> Float.max hi r.ratio_hi)
  in
  let factor r = Float.max r.ratio_hi (1. /. r.ratio_lo) in
  let control_factor = Float.max 1. (Float.max (factor base_control) (factor variant_control)) in
  let resolved =
    treatment.pairs >= min_readings
    && Float.(effect_hi < 1. /. control_factor || effect_lo > control_factor)
  in
  {
    treatment = { treatment with resolved };
    base_control;
    variant_control;
    effect_lo;
    effect_hi;
    control_factor;
  }

(** [bounds_fields ~name s]: [name_lo=<ms> name_hi=<ms> name_retimed=<admitted>/<calls>], for a line
    that prints the median [name=<ms>] elsewhere. *)
let bounds_fields ~name s =
  Printf.sprintf "%s_lo=%.6f %s_hi=%.6f %s_retimed=%d/%d" name s.lo_ms name s.hi_ms name s.admitted
    (s.admitted + s.refused)

(** [spread_fields ~name s]: [name=<ms>] then {!bounds_fields}, the one spelling every runner's line
    uses. *)
let spread_fields ~name s = Printf.sprintf "%s=%.6f %s" name s.ms (bounds_fields ~name s)

(** [ratio_fields ~name r]: [name=<r> name_lo=<r> name_hi=<r> name_resolved=<bool>]. *)
let ratio_fields ~name r =
  Printf.sprintf "%s=%.3f %s_lo=%.3f %s_hi=%.3f %s_resolved=%b" name r.ratio name r.ratio_lo name
    r.ratio_hi name r.resolved

(** The configured objective, read as {!Autotune.tune} reads it. *)
let configured_timing () =
  Autotune.timing_of_setting (Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued")

(** [retime ?calls routines]: every routine's readings, [calls] (default {!calls_of_env}) per
    routine, in {!Bench_design.orders}' visiting order, each reading an independent
    {!Autotune.time_routine} call under {!configured_timing} and the configured [autotune_repeats],
    admitted through {!Autotune.admitted_timing_ms}. The result is in the input's order, one array
    of [calls] readings per routine, index [i] taken in batch [i].

    Timing dispatches each routine many times against its live buffers ({!Autotune.time_routine}
    restores nothing), so a caller checking a routine's output runs it once more afterwards from
    inputs that are not its own outputs, as an exactness oracle must anyway. *)
let configured_repeats () = Utils.registered_int ~arg_name:"autotune_repeats"

(** [screen_reading ms]: what a screen ranks one timing call by, given the call's minimum: the
    minimum whenever the clock resolved one, and [None] for a non-positive or non-finite one. It
    reads no admission verdict: a screen ranks a refused window by its minimum too -- see the
    screen's section below. *)
let screen_reading ms = if Float.is_finite ms && Float.is_positive ms then Some ms else None

(** [read_screen ~timing (ctx, routine)]: one screen reading under [timing] and the configured
    [autotune_repeats]: [(screen_reading ms, refused)], [refused] when
    {!Autotune.admitted_timing_ms} refused the window -- which {!rescreen}'s cut and the
    instrument's count read, and the reading does not. *)
let read_screen ~timing ((ctx, routine) : Context.t * Context.routine) =
  let result = Autotune.time_routine ~timing ~repeats:(configured_repeats ()) ctx routine in
  (screen_reading result.Autotune.ms, Option.is_none (Autotune.admitted_timing_ms result))

(** [read_once (ctx, routine)]: one reading -- one {!Autotune.time_routine} call under
    {!configured_timing} and the configured [autotune_repeats], admitted through
    {!Autotune.admitted_timing_ms} ([None] when refused). {!retime} is made of these. *)
let read_once ?(timing = configured_timing ()) ((ctx, routine) : Context.t * Context.routine) =
  Autotune.admitted_timing_ms
    (Autotune.time_routine ~timing ~repeats:(configured_repeats ()) ctx routine)

let retime ?calls (routines : (Context.t * Context.routine) list) =
  let calls = Option.value_or_thunk calls ~default:(fun () -> calls_of_env ()) in
  let timing = configured_timing () in
  let routines = Array.of_list routines in
  let arms = Array.length routines in
  let readings = Array.init arms ~f:(fun _ -> Array.create ~len:calls None) in
  if arms > 0 then
    Array.iteri (Bench_design.orders ~arms ~batches:calls) ~f:(fun batch order ->
        Array.iter order ~f:(fun a -> readings.(a).(batch) <- read_once ~timing routines.(a)));
  (timing, Array.to_list readings)

(** {1 Screening many routines for the slowest (gh-ocannl-1763, gh-ocannl-1818)}

    Re-timing every one of a step's kernels with a spread costs [calls] timing calls each, about
    half a second a call whatever the kernel's size: on a 178-kernel [gpt2_mini_train] step on
    Metal, about 2.1 s a kernel and 371-381 s of a 377-386 s cell. Only the slowest kernel's time is
    reported, so the many kernels that are nowhere near it need no spread: each is SCREENED with one
    reading ({!read_once}), and only the kernels whose screen reading comes within {!screen_ratio}
    of the slowest one -- at most {!screen_at_most} of them -- are re-timed, together, in one
    {!retime} session, where the slowest is decided by median with its range as before.

    The screen runs in two stages (gh-ocannl-1818). One reading under the configured objective --
    [queued] by default -- still cost 1.1-1.5 s on CUDA/HIP, where it calibrates a batch depth of up
    to 2048 launches with probes and confirmations, so a 270-kernel HIP step spent ~410 s of its 600
    s budget on the screen alone. The first stage reads every kernel once under {!screen_timing},
    [isolated] -- one launch and one host synchronization a sample, no calibration -- and keeps the
    {!screen_wide_at_most} slowest isolated readings, plus any kernel with no reading
    ({!screen_keep_wide}); it applies no ratio, since a ratio between two isolated readings says
    nothing reliable about the configured objective's (two kernels of 8 and 7 us under queued timing
    can read 48 and 97 us isolated, their round trips differing). The second reads only those, once
    each, under the configured objective, and keeps the re-timing's candidates from THOSE readings
    ({!rescreen}). An isolated reading carries a launch-and-sync round trip the queued objective
    amortizes (~40-90 us on gfx1151, varying by kernel), so it does not order kernels whose times
    are within that round trip of one another, and on a step whose dominant kernel is a few times
    the round trip the floor-bound kernels crowd the slowest isolated readings. The wide cap is what
    absorbs that: a kernel outranks the queued-slowest one in the first stage only by an isolated
    reading above it, and the final cap of {!screen_at_most} is ranked by queued readings, which the
    round trip does not inflate.

    A reading is a minimum over a window, so noise only raises it above the kernel's floor. The
    truly slowest kernel is therefore left out only when, in the first stage, {!screen_wide_at_most}
    other kernels' isolated readings came out above its own, or, in the second, some other kernel's
    admitted reading came out more than twice its own -- an inflated window -- or [screen_at_most]
    others above it; and then a wrong one can be named.

    A screen reading is ranked by its minimum even when {!Autotune.admitted_timing_ms} refuses the
    window ({!screen_reading}): the screen only decides what is re-measured, and a refused window's
    minimum can only over-state the kernel -- it is not dropped on a reading noise lowered, it may
    only crowd the cap as any inflated window does. It does not set the second stage's cut, though,
    while any reading there was admitted ({!rescreen}): on CUDA/HIP a queued reading refused as
    [unbatched] is a depth-1, round-trip-inclusive time, and as the slowest it would cut a kernel
    whose admitted queued reading is the larger true time. Before gh-ocannl-1818 a refused reading
    was kept unconditionally, outside the cap, and on a shared box that made the screen's cost
    unbounded: a Metal [gpt2_mini_train] step under co-tenant load had 38 of 195 queued screen
    readings refused, and re-timed all 46 candidates at four calls each, 136 s; isolated, 105 of 195
    were refused. Only a kernel with no reading at all -- a clock that resolved nothing -- is still
    kept outside the cap, since nothing is known of it. The re-timing's readings, which decide the
    dominant kernel and state its range, go through the admission gate as before. *)

(** The first stage's objective: [isolated], whatever [autotune_timing] says, because it needs no
    calibration. *)
let screen_timing = Autotune.Isolated

(** A kernel is a candidate when its screen reading is at least the slowest one over this. *)
let screen_ratio = 2.

(** The most screened kernels re-timed, the slowest readings first: on a step whose many kernels all
    read within {!screen_ratio} of one another (identical layers), the cap keeps the re-timing
    bounded. Those left out read below every candidate, so the cap can mis-name the slowest only
    among kernels a single reading did not order -- ones whose re-timed ranges would most likely
    overlap anyway. *)
let screen_at_most = 8

(** [screen_keep ?ratio ?at_most ?refused entries]: [(kept, dropped)], each in [entries]' order, of
    entries paired with their screen reading ([None]: no reading). Kept: every entry with none, and
    of the others those reading at least [slowest /. ratio] ([ratio] default {!screen_ratio}), at
    most [at_most] (default {!screen_at_most}) of them, the slowest first, ties to the earlier
    entry. [slowest] is the slowest reading of an entry [refused] (default: none) does not hold, or,
    when it holds of every entry with a reading, the slowest of those.

    Without [refused], applied after each new reading to the entries kept so far, it keeps exactly
    what it keeps applied once to all of them: the slowest reading only grows, so an entry it drops
    stays below the final cut, and one dropped by the cap has [at_most] entries above it that are
    dropped, if ever, only for something that drops it too. So a caller may release what it drops as
    it goes. With [refused] the slowest can fall (an admitted reading arriving after refused ones
    only), so apply it once. *)
let screen_keep ?(ratio = screen_ratio) ?(at_most = screen_at_most) ?(refused = fun _ -> false)
    entries =
  let indexed = List.mapi entries ~f:(fun k (x, r) -> (k, x, r)) in
  let slowest_of ~admitted_only =
    List.fold indexed ~init:Float.neg_infinity ~f:(fun m (_, x, r) ->
        match r with Some r when (not admitted_only) || not (refused x) -> Float.max m r | _ -> m)
  in
  let slowest =
    let admitted = slowest_of ~admitted_only:true in
    if Float.is_finite admitted then admitted else slowest_of ~admitted_only:false
  in
  let read = List.filter_map indexed ~f:(fun (k, _, r) -> Option.map r ~f:(fun r -> (k, r))) in
  let ranked =
    List.filter read ~f:(fun (_, r) -> Float.(r >= slowest /. ratio))
    |> List.stable_sort ~compare:(fun (_, a) (_, b) -> Float.compare b a)
  in
  let top = List.map (List.take ranked at_most) ~f:fst in
  let kept_index k = List.mem top k ~equal:Int.equal in
  let kept, dropped =
    List.partition_tf indexed ~f:(fun (k, _, r) -> Option.is_none r || kept_index k)
  in
  let strip = List.map ~f:(fun (_, x, _) -> x) in
  (strip kept, strip dropped)

(** The most kernels the first stage keeps (not counting those with no reading), the slowest
    isolated readings first: three times {!screen_at_most}, so that the queued-slowest kernel stays
    in while fewer than that many others read above it isolated -- sixteen more than the final cap
    allows. Each costs one reading under the configured objective in the second stage: on minix
    (HIP, unified memory, gpt2_mini_train) the 24 queued calls took 25.8-25.9 s, about 1.1 s a call.
*)
let screen_wide_at_most = 3 * screen_at_most

(** [rank_among r readings]: [r]'s rank among [readings], 1 for the slowest, ties sharing the better
    rank -- [1 +] the readings above it. The instrument's closing line states the dominant kernel's
    rank among the re-screened kernels' isolated readings by it: how close the first stage came to
    leaving it out. *)
let rank_among r readings = 1 + List.count readings ~f:(fun x -> Float.(x > r))

(** [screen_keep_wide entries]: the first stage's cut, {!screen_keep} at {!screen_wide_at_most} and
    no ratio, over [entries] paired with their {!screen_timing} reading. Like {!screen_keep} without
    [refused], it may be applied after each reading. *)
let screen_keep_wide entries =
  screen_keep ~ratio:Float.infinity ~at_most:screen_wide_at_most entries

(** [rescreen ~read entries]: the second stage. Reads each of [entries] -- the first stage's kept
    set -- once with [read] (the configured objective's {!read_screen}: the reading and whether the
    admission gate refused it), in [entries]' order, and returns [(kept, dropped)] of
    [(entry, reading)] by {!screen_keep} at its defaults over those readings, the cut set by the
    admitted ones while there are any: what the re-timing gets, and what keeps its second-stage
    reading. *)
let rescreen ~read entries =
  let read = List.map entries ~f:(fun e -> (e, read e)) in
  let kept, dropped =
    screen_keep
      ~refused:(fun (_, (_, refused)) -> refused)
      (List.map read ~f:(fun ((_, (r, _)) as er) -> (er, r)))
  in
  let strip = List.map ~f:(fun (e, (r, _)) -> (e, r)) in
  (strip kept, strip dropped)
