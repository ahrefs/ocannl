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

(* The test's own floats go through the portable printer (AGENTS.md); the runner-format lines
   ([spread_fields], [ratio_fields]) print the module's fixed-point spelling, which is what they pin
   -- every scripted value is a small dyadic, exact at those precisions, so no rounding tie reaches
   the golden. *)
let fl v = Test_utils.concise_float ~prec:6 v
let () = Test_utils.set_binary_stdout ()
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
  printf "unstated: median %s, admitted %d of %d\n" (fl u.R.ms) u.R.admitted
    (u.R.admitted + u.R.refused);
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
  printf "few: pairs %d, median %s\n" few.R.pairs (fl few.R.ratio);
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

(* gh-ocannl-1763: the dominant-kernel instrument's screen. [screen_keep] picks which screened
   kernels are re-timed: every refused one, and of the rest those within [screen_ratio] of the
   slowest reading, at most [at_most], the slowest first. The instrument applies it after EACH
   reading and releases what it drops as it goes, which is sound only if that incremental use keeps
   exactly what one application to every reading keeps -- claimed below over scripted sequences. *)
let () =
  let names l = String.concat ~sep:" " l in
  let kept, dropped =
    R.screen_keep
      [ ("a", Some 1.); ("b", Some 4.); ("c", Some 2.5); ("d", None); ("e", Some 1.875) ]
  in
  printf "screen: kept [%s] dropped [%s]\n" (names kept) (names dropped);
  p "screen: within 2x of the slowest is kept, below it dropped, a refused reading kept"
    (List.equal String.equal kept [ "b"; "c"; "d" ] && List.equal String.equal dropped [ "a"; "e" ]);
  let kept, _ = R.screen_keep [ ("half", Some 2.); ("slowest", Some 4.) ] in
  p "screen: a reading exactly at the cut is kept"
    (List.equal String.equal kept [ "half"; "slowest" ]);
  let kept, dropped =
    R.screen_keep ~at_most:3
      [ ("p", Some 1.5); ("q", Some 1.75); ("r", Some 2.); ("s", Some 1.25); ("t", Some 1.875) ]
  in
  p "screen: the cap keeps the slowest readings, in input order"
    (List.equal String.equal kept [ "q"; "r"; "t" ] && List.equal String.equal dropped [ "p"; "s" ]);
  let kept, _ =
    R.screen_keep ~at_most:2 [ ("w", Some 2.); ("x", Some 2.); ("y", Some 2.); ("z", None) ]
  in
  p "screen: a tie at the cap goes to the earlier entry; refused readings are outside the cap"
    (List.equal String.equal kept [ "w"; "x"; "z" ]);
  p "screen: with every reading refused, everything is kept"
    (Poly.equal (R.screen_keep [ ("u", None); ("v", None) ]) ([ "u"; "v" ], []));
  p "screen: nothing to screen keeps nothing, drops nothing"
    (Poly.equal (R.screen_keep ([] : (string * float option) list)) ([], []));
  (* Scripted sequences: a linear congruential stream of dyadic readings in [1/8, 8), every seventh
     refused, screened at several caps and ratios. *)
  let sequence ~seed ~len =
    let state = ref seed in
    List.init len ~f:(fun k ->
        state := ((!state * 1103515245) + 12345) % 2147483648;
        let r =
          if !state % 7 = 0 then None else Some (Float.of_int (1 + (!state / 7 % 63)) /. 8.)
        in
        (k, r))
  in
  let cases =
    List.concat_map [ 1; 2; 3; 8 ] ~f:(fun at_most ->
        List.concat_map [ 1.5; 2.; 4.; Float.infinity ] ~f:(fun ratio ->
            List.map [ 11; 29; 47; 1763 ] ~f:(fun seed -> (at_most, ratio, sequence ~seed ~len:40))))
  in
  p_all "screen: applied after each reading, it keeps what one application to all of them keeps"
    cases ~f:(fun (at_most, ratio, entries) ->
      let keep held =
        fst (R.screen_keep ~ratio ~at_most (List.map held ~f:(fun e -> (e, snd e))))
      in
      let incremental = List.fold entries ~init:[] ~f:(fun held e -> keep (held @ [ e ])) in
      List.equal (fun (a, _) (b, _) -> Int.equal a b) incremental (keep entries))

(* gh-ocannl-1818: the screen's two stages. The first reads every kernel once under [isolated]
   timing, whose reading carries a launch-and-sync round trip the queued objective amortizes, and
   keeps the [screen_wide_at_most] slowest readings, with no ratio ([screen_keep_wide]); the second
   reads only those once under the configured objective and keeps the re-timing's candidates from
   those readings ([rescreen]). The scenario is the crowding the wide cap is for, after
   gpt2_mini_train on HIP, whose dominant kernel is a few round trips long: twelve kernels a little
   faster than the dominant one under queued timing but with a longer round trip read above it
   isolated, so a single isolated stage capped at [screen_at_most] would leave it out. Kernels are
   [(name, isolated, queued)]; "unread" has no isolated reading. *)
let () =
  let names l = String.concat ~sep:" " l in
  let kernels =
    ("dom", Some 0.34375, Some 0.3125)
    :: List.init 12 ~f:(fun k -> (Printf.sprintf "f%d" k, Some 0.375, Some 0.25))
    @ List.init 20 ~f:(fun k -> (Printf.sprintf "s%d" k, Some 0.125, Some 0.0625))
    @ [ ("unread", None, Some 0.125) ]
  in
  let name (n, _, _) = n in
  let isolated (_, i, _) = i in
  let starts_s k = String.is_prefix (name k) ~prefix:"s" in
  (* The first stage as the instrument applies it: after each reading, to what it holds so far. *)
  let wide =
    List.fold kernels ~init:[] ~f:(fun held k ->
        fst (R.screen_keep_wide (List.map (held @ [ k ]) ~f:(fun k -> (k, isolated k)))))
  in
  let _, wide_dropped = R.screen_keep_wide (List.map kernels ~f:(fun k -> (k, isolated k))) in
  printf "two-stage screen: the first stage kept %d of %d\n" (List.length wide)
    (List.length kernels);
  p_none "two-stage screen: the scenario crowds the dominant kernel out of a single isolated stage"
    (fst (R.screen_keep (List.map kernels ~f:(fun k -> (k, isolated k)))))
    ~f:(fun k -> String.equal (name k) "dom");
  p_exists "two-stage screen: the first stage keeps the dominant kernel" wide ~f:(fun k ->
      String.equal (name k) "dom");
  p "two-stage screen: the dominant kernel ranks thirteenth among the kept isolated readings"
    (R.rank_among 0.34375 (List.filter_map wide ~f:isolated) = 13);
  p "two-stage screen: the first stage keeps screen_wide_at_most readings and the unread kernel"
    (List.length wide = R.screen_wide_at_most + 1);
  p_all "two-stage screen: what the first stage drops are the fastest isolated readings"
    wide_dropped ~f:starts_s;
  p_exists
    "two-stage screen: the first stage applies no ratio: a reading past 2x of the slowest is kept \
     while the cap has room"
    wide ~f:starts_s;
  p_exists "two-stage screen: the first stage keeps a kernel with no isolated reading" wide
    ~f:(fun k -> String.equal (name k) "unread");
  let reads = ref [] in
  let kept, dropped =
    R.rescreen wide ~read:(fun (n, _, q) ->
        reads := n :: !reads;
        (q, false))
  in
  let kept_names = List.map kept ~f:(fun (k, _) -> name k) in
  printf "two-stage screen: re-timing [%s], dropped %d\n" (names kept_names) (List.length dropped);
  p "two-stage screen: the second stage reads the first stage's kept set once each, in order"
    (List.equal String.equal (List.rev !reads) (List.map wide ~f:name));
  p "two-stage screen: the re-timing's candidates are capped by the second-stage readings"
    (List.length kept = R.screen_at_most && String.equal (List.hd_exn kept_names) "dom");
  p_exists "two-stage screen: a second-stage reading past 2x of the slowest one is dropped" dropped
    ~f:(fun (k, _) -> String.equal (name k) "unread");
  p_all "two-stage screen: what the second stage drops keeps its second-stage reading" dropped
    ~f:(fun ((_, _, q), r) -> Option.equal Float.equal q r);
  let kept_of read entries = List.map (fst (R.rescreen entries ~read)) ~f:(fun ((n, _), _) -> n) in
  let refused_a (n, r) = (r, String.equal n "a") in
  p "two-stage screen: a refused second-stage reading does not set the cut while one is admitted"
    (List.equal String.equal
       (kept_of refused_a [ ("a", Some 0.40625); ("b", Some 0.15625) ])
       [ "a"; "b" ]);
  p "two-stage screen: with every second-stage reading refused, the refused ones set the cut"
    (List.equal String.equal
       (kept_of (fun (_, r) -> (r, true)) [ ("a", Some 0.40625); ("b", Some 0.15625) ])
       [ "a" ]);
  p "two-stage screen: a second-stage call with no reading is kept for the re-timing"
    (List.equal String.equal
       (kept_of
          (fun (n, r) -> if String.equal n "a" then (None, true) else (r, false))
          [ ("a", Some 1.); ("b", Some 4.) ])
       [ "a"; "b" ])

(* gh-ocannl-1818: a screen ranks a timing call by its minimum -- whether or not the admission gate
   admitted the window, since [screen_reading] reads no verdict -- and keeps a kernel
   unconditionally only when the clock resolved nothing. *)
let () =
  p "screen reading: a resolved minimum reads as itself"
    (Option.equal Float.equal (R.screen_reading 0.25) (Some 0.25));
  p "screen reading: a clock that resolved nothing gives no reading"
    (Option.is_none (R.screen_reading 0.)
    && Option.is_none (R.screen_reading (-0.5))
    && Option.is_none (R.screen_reading Float.nan)
    && Option.is_none (R.screen_reading Float.infinity))
