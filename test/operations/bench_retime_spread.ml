(* gh-ocannl-1524: the summaries [Bench_retime] reports a re-timing session by, on SCRIPTED readings
   -- no clock is read, no backend runs.

   A runner's re-timing used to be one min-of-N reading per routine, so a ratio between two routines
   had no stated spread, and on a sub-50 us kernel an unchanged control read 0.69x. [Bench_retime]
   reads each routine several times, interleaved, and reports the median with its range and the
   PAIRED ratio with its range. Claimed here:

   - the median, range and admission counts of one routine's readings, odd and even counts, with a
   refused call left out of every statistic; - below [min_readings] admitted readings the range is
   not stated (NaN), so no line can print a spread of one reading as if it were measured; - the
   paired ratio divides out a drift common to a batch: two routines whose true ratio is 3/4, read
   under a drift that moves both by up to 3x, give a paired ratio of exactly 3/4 at every call --
   resolved -- while their unpaired ranges overlap, which is why the ratio is paired and not taken
   across the two ranges; - a ratio is resolved only when every pair falls on one side of 1 and
   there are at least [min_readings] pairs: pairs straddling 1 are within the spread, and a pair
   with a refused side does not count; - [BENCH_RETIME_CALLS] refuses a count the mirrored design
   cannot use or that states no spread.

   Every reading is a small dyadic, so each printed value is exact. *)

open Base
open Stdio
open Verdict.Claims
module R = Bench_retime

let some = Array.map ~f:Option.some

let () =
  let s = R.spread_of [| Some 3.; None; Some 1.; Some 4.; Some 2.; Some 5. |] in
  printf "odd: %s\n" (R.spread_fields ~name:"ms" s);
  p "odd: the median is the middle admitted reading" Float.(s.R.ms = 3.);
  p "odd: the range is the admitted extremes" Float.(s.R.lo_ms = 1. && s.R.hi_ms = 5.);
  p "odd: the refused call is counted apart" (s.R.admitted = 5 && s.R.refused = 1);
  p "odd: the relative spread is (hi - lo) / median" Float.(R.rel_spread s = 4. /. 3.);
  let e = R.spread_of (some [| 4.; 1.; 3.; 2. |]) in
  printf "even: %s\n" (R.spread_fields ~name:"ms" e);
  p "even: the median is the mean of the middle two" Float.(e.R.ms = 2.5);
  let u = R.spread_of [| Some 2.; None; Some 1.; None; Some 3.; None |] in
  printf "unstated: median %g, admitted %d of %d\n" u.R.ms u.R.admitted (u.R.admitted + u.R.refused);
  p "unstated: below min_readings the range is NaN"
    (u.R.admitted < R.min_readings && Float.is_nan u.R.lo_ms && Float.is_nan u.R.hi_ms);
  p "unstated: so is the relative spread" (Float.is_nan (R.rel_spread u));
  p "unstated: the median is still the admitted readings'" Float.(u.R.ms = 2.)

(* Six batches of a mirrored session; each batch's drift moves both routines alike. *)
let drift = [| 1.; 1.5; 2.; 1.25; 3.; 1. |]

let () =
  let den = Array.map drift ~f:(fun d -> Some d) in
  let num = Array.map drift ~f:(fun d -> Some (0.75 *. d)) in
  let r = R.paired_ratio ~num ~den in
  printf "drift: %s\n" (R.ratio_fields ~name:"num_over_den" r);
  p "drift: every paired ratio is the true 3/4"
    Float.(r.R.ratio = 0.75 && r.R.ratio_lo = 0.75 && r.R.ratio_hi = 0.75);
  p "drift: resolved, every pair below 1" r.R.resolved;
  let sn = R.spread_of num and sd = R.spread_of den in
  p "drift: the unpaired ranges overlap, so a range-against-range reading could not resolve it"
    Float.(sn.R.hi_ms >= sd.R.lo_ms && sd.R.hi_ms >= sn.R.lo_ms)

let () =
  let den = some [| 1.; 1.; 1.; 1.; 1.; 1. |] in
  let straddle = R.paired_ratio ~num:(some [| 0.75; 1.25; 0.875; 1.125; 0.5; 1.5 |]) ~den in
  printf "straddle: %s\n" (R.ratio_fields ~name:"r" straddle);
  p "straddle: pairs on both sides of 1 are within the spread" (not straddle.R.resolved);
  p "straddle: the ratio is the paired median" Float.(straddle.R.ratio = 1.);
  let few =
    R.paired_ratio
      ~num:[| Some 0.5; None; Some 0.5; Some 0.5; None; Some 0.5 |]
      ~den:[| Some 1.; Some 1.; None; Some 1.; Some 1.; Some 1. |]
  in
  printf "few: pairs %d, median %g\n" few.R.pairs few.R.ratio;
  p "few: a pair with a refused side does not count" (few.R.pairs = 3);
  p "few: below min_readings pairs no ratio is resolved, all below 1 or not"
    ((not few.R.resolved) && Float.is_nan few.R.ratio_lo && Float.is_nan few.R.ratio_hi)

let () =
  let calls v =
    Unix.putenv "BENCH_RETIME_CALLS" v;
    match R.calls_of_env () with n -> Some n | exception Failure _ -> None
  in
  p "env: an even count of at least min_readings is taken"
    (Option.equal Int.equal (calls "8") (Some 8));
  p "env: an odd count is refused (the design mirrors every batch)" (Option.is_none (calls "5"));
  p "env: a count below min_readings is refused" (Option.is_none (calls "2"));
  p "env: a non-number is refused" (Option.is_none (calls "six"))
