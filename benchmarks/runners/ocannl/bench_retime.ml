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

(** [calls_of_env ()]: [BENCH_RETIME_CALLS] (default 6), the timing calls per routine. Even, because
    {!Bench_design} mirrors every batch, and at least {!min_readings}. *)
let calls_of_env () =
  match Stdlib.Sys.getenv_opt "BENCH_RETIME_CALLS" with
  | None -> 6
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
let retime ?calls (routines : (Context.t * Context.routine) list) =
  let calls = Option.value_or_thunk calls ~default:calls_of_env in
  let timing = configured_timing () in
  let repeats = Utils.registered_int ~arg_name:"autotune_repeats" in
  let routines = Array.of_list routines in
  let arms = Array.length routines in
  let readings = Array.init arms ~f:(fun _ -> Array.create ~len:calls None) in
  if arms > 0 then
    Array.iteri (Bench_design.orders ~arms ~batches:calls) ~f:(fun batch order ->
        Array.iter order ~f:(fun a ->
            let ctx, routine = routines.(a) in
            readings.(a).(batch) <-
              Autotune.admitted_timing_ms (Autotune.time_routine ~timing ~repeats ctx routine)));
  (timing, Array.to_list readings)
