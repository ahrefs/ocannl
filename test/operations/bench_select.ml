(* The two-stage finalist selection projection_shape_bench applies (gh-ocannl-1531,
   [bin/bench_select.ml]), driven by SCRIPTED noise -- no clock is read.

   The noise model is the bench's: every seed of a class is a different geometry, timed in that
   geometry's own round, so each reading carries its ROUND's offset (drift, or contention that only
   ever slows a round down) on top of a small per-arm jitter. The single-shot rule the bench used to
   apply -- send the round-best -- compares readings across those offsets. The two-stage rule
   re-times a shortlist in one round, where every contender shares the offset.

   Claimed: a scripted case where the single-shot pick is wrong and the two-stage pick is the true
   best; over a seeded population of such trials, the two-stage rule picks the true best more often
   than single-shot does, and EVERY time the shortlist held it (a round slowdown that drops the true
   best below third place is the rule's one remaining miss, and the price of a bounded shortlist),
   while with no round offset at all the two rules tie -- the gain is the shared round, not a second
   draw; and the shapes the bench relies on -- a shortlist contains every single-shot pick, both
   rules keep input order and break ties to the earlier arm, a NaN never wins, a class selects
   nothing exactly when it is empty. *)

open Base
open Stdio
open Verdict.Claims
module S = Bench_select

let id x = x

(* The single-shot rule on one statistic: the arm with the best reading. *)
let pick xs = fst (List.hd_exn (S.best ~scores:[ snd ] xs))

(* The scripted case. True throughputs, best first; arm 0 draws a 3% round slowdown, so it reads
   below arm 1, whose round was clean -- single-shot crowns arm 1. Arm 0 still reads second, so a
   shortlist of three holds it, and the shortlist round's one shared slowdown (4%) preserves the
   true order. *)
let () =
  let truth = [ 100.; 98.5; 97.; 95.; 93. ] in
  let round_slowdown = [ 0.03; 0.; 0.01; 0.; 0. ] in
  let first = List.map2_exn truth round_slowdown ~f:(fun t d -> t /. (1. +. d)) in
  let arms = List.mapi first ~f:(fun i r -> (i, r)) in
  let single = pick arms in
  let short = S.shortlist ~k:3 ~primary:snd ~others:[] arms |> List.map ~f:fst in
  let retimed = List.map short ~f:(fun i -> (i, List.nth_exn truth i /. 1.04)) in
  let two_stage = pick retimed in
  printf "scripted case: single-shot picks arm %d; shortlist %s; two-stage picks arm %d\n" single
    (String.concat ~sep:"," (List.map short ~f:Int.to_string))
    two_stage;
  p "scripted: the single-shot pick is not the true best" (single <> 0);
  p "scripted: the shortlist holds the true best" (List.mem short 0 ~equal:Int.equal);
  p "scripted: the two-stage pick is the true best" (two_stage = 0);
  p "scripted: a shortlist of one is the single-shot pick"
    (List.equal Int.equal
       (List.map (S.shortlist ~k:1 ~primary:snd ~others:[] arms) ~f:fst)
       [ single ])

(* A 31-bit linear congruential generator: the population is scripted, identical on every platform
   and OCaml version, unlike [Random]. *)
let lcg = ref 20261024

let uniform () =
  lcg := ((1103515245 * !lcg) + 12345) land 0x7fffffff;
  Float.of_int !lcg /. 2147483648.

(* One reading: the arm's throughput under a round slowdown [d] and a per-arm jitter of
   +-[jitter]/2. *)
let read ~jitter t d = t *. (1. +. ((uniform () -. 0.5) *. jitter)) /. (1. +. d)

(* A round's slowdown: contended with probability 0.35, by up to 15%; under [~offsets:false] there
   is none, and the per-arm jitter is the whole noise. *)
let round_slowdown ~offsets =
  if offsets && Float.(uniform () < 0.35) then uniform () *. 0.15 else 0.

type outcome = { single_right : bool; held : bool; two_stage_right : bool }

(* One trial over eight seeds 1.5% apart, the true best first: each seed read in its own round, then
   the shortlist re-timed in one shared round. *)
let trial ~offsets ~jitter =
  let truth = List.init 8 ~f:(fun i -> 100. *. (1. -. (0.015 *. Float.of_int i))) in
  let first = List.mapi truth ~f:(fun i t -> (i, read ~jitter t (round_slowdown ~offsets))) in
  let short = S.shortlist ~k:3 ~primary:snd ~others:[] first in
  let shared = round_slowdown ~offsets in
  let retimed = List.map short ~f:(fun (i, _) -> (i, read ~jitter (List.nth_exn truth i) shared)) in
  {
    single_right = pick first = 0;
    held = List.exists short ~f:(fun (i, _) -> i = 0);
    two_stage_right = pick retimed = 0;
  }

let () =
  let n = 2000 in
  let population ~offsets ~jitter = List.init n ~f:(fun _ -> trial ~offsets ~jitter) in
  let count l f = List.count l ~f in
  (* The bench's regime: round offsets dominate a small per-arm jitter. *)
  let off = population ~offsets:true ~jitter:0.01 in
  let s_off = count off (fun o -> o.single_right)
  and t_off = count off (fun o -> o.two_stage_right) in
  let held = List.filter off ~f:(fun o -> o.held) in
  (* The control: no round offsets, and a jitter wide enough that single-shot does err. *)
  let iid = population ~offsets:false ~jitter:0.05 in
  let s_iid = count iid (fun o -> o.single_right)
  and t_iid = count iid (fun o -> o.two_stage_right) in
  printf "seeded population, %d trials of 8 seeds 1.5%% apart, shortlist of 3:\n" n;
  printf
    "  round offsets, +-0.5%% jitter: single-shot %d, two-stage %d picked the true best (shortlist \
     held it %d times)\n"
    s_off t_off (List.length held);
  printf "  no offsets, +-2.5%% jitter:    single-shot %d, two-stage %d picked the true best\n"
    s_iid t_iid;
  p "with round offsets, single-shot misses the true best in over a fifth of trials"
    (s_off * 5 < n * 4);
  p "the two-stage rule picks it in at least a twentieth more of them"
    (t_off * 20 >= (s_off * 20) + n);
  p_all "and whenever the shortlist held the true best, the re-timing picked it" held ~f:(fun o ->
      o.two_stage_right);
  p "with no round offsets, single-shot misses the true best sometimes" (s_iid < n);
  (* A second independent draw is no better than the first: the gain above is the SHARED round, not
     the re-timing as such -- and the extra round costs no accuracy where it buys none. *)
  p "and the two rules agree to within a fiftieth of the trials" (abs (t_iid - s_iid) * 50 <= n)

(* The shapes the bench relies on, over a seeded population of classes up to 9 arms with two
   statistics -- the second often disagreeing with the first, and both with ties; every tenth class
   is empty. *)
let () =
  let quant x = Float.round_down (x *. 8.) in
  let classes =
    List.init 300 ~f:(fun c ->
        List.init (c % 10) ~f:(fun i -> (i, quant (uniform ()), quant (uniform ()))))
  in
  let s1 (_, a, _) = a and s2 (_, _, b) = b and idx (i, _, _) = i in
  let mem l x = List.mem (List.map l ~f:idx) (idx x) ~equal:Int.equal in
  let ordered l = List.is_sorted_strictly (List.map l ~f:idx) ~compare:Int.compare in
  p_all "every arm the single-shot rule picks is on the shortlist" classes ~f:(fun xs ->
      let short = S.shortlist ~k:3 ~primary:s1 ~others:[ s2 ] xs in
      List.for_all (S.best ~scores:[ s1; s2 ] xs) ~f:(mem short));
  p_all "a shortlist holds at most k plus one per other statistic, and all of a smaller class"
    classes ~f:(fun xs ->
      let n = List.length (S.shortlist ~k:3 ~primary:s1 ~others:[ s2 ] xs) in
      n <= 4 && (List.length xs > 3 || n = List.length xs));
  p_all "both rules keep the input order, without duplicates" classes ~f:(fun xs ->
      ordered (S.shortlist ~k:3 ~primary:s1 ~others:[ s2 ] xs)
      && ordered (S.best ~scores:[ s1; s2 ] xs));
  p_all "a tie goes to the earlier arm" classes ~f:(fun xs ->
      match S.best ~scores:[ s1 ] xs with
      | [] -> List.is_empty xs
      | [ w ] -> not (List.exists xs ~f:(fun x -> idx x < idx w && Float.(s1 x >= s1 w)))
      | _ -> false);
  p_exists "the population holds empty classes" classes ~f:List.is_empty;
  p_all "a class selects nothing exactly when it is empty, under both rules" classes ~f:(fun xs ->
      Bool.equal (List.is_empty xs) (List.is_empty (S.best ~scores:[ s1; s2 ] xs))
      && Bool.equal (List.is_empty xs)
           (List.is_empty (S.shortlist ~k:3 ~primary:s1 ~others:[ s2 ] xs)));
  p "a NaN never wins over a number"
    (List.equal Float.equal (S.best ~scores:[ id ] [ Float.nan; 1.; Float.nan ]) [ 1. ]);
  p "a shortlist of fewer than one is refused"
    (match S.shortlist ~k:0 ~primary:id ~others:[] [ 1. ] with
    | _ -> false
    | exception Invalid_argument _ -> true)
