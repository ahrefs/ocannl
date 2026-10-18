(* gh-ocannl-1253: the cache-capacity calibration behind tools/calibrate_cache_capacity.exe.

   [Calibrate.fit_capacity] places the ranking score's capacity between the working sets whose
   operand re-reads are served at the cached rate and those served at the DRAM rate. Its readings
   are exercised on synthetic sweeps -- a clean step, a flat sweep that never crosses, a sweep whose
   rates do not split by working set, too few points -- and [Calibrate.working_set] is run on the
   configured backend at toy extents, where only structural facts are claimed: every point is timed
   and requests at least the operands' distinct bytes. No time reaches the golden. *)

open Base
open Verdict.Claims
module Cal = Ocannl.Calibrate

let point ~ws ~rate =
  (* [rate] bytes per millisecond, so a 1 ms point requesting [rate] bytes. *)
  {
    Cal.wp_k = 1;
    wp_working_set = ws;
    wp_requested = rate;
    wp_distinct = 1;
    wp_ms = 1.;
    wp_rank_ms = None;
    wp_bound_ms = None;
  }

let () =
  let step =
    [
      point ~ws:100_000 ~rate:2_000_000;
      point ~ws:1_000_000 ~rate:1_900_000;
      point ~ws:10_000_000 ~rate:300_000;
      point ~ws:100_000_000 ~rate:250_000;
    ]
  in
  p "a clean step fits a monotone capacity between the last cached and first spilled working set"
    (match Cal.fit_capacity step with
    | Cal.Transition { cached_ws; spilled_ws; capacity; monotone } ->
        cached_ws = 1_000_000 && spilled_ws = 10_000_000 && monotone && capacity > cached_ws
        && capacity < spilled_ws
    | Cal.No_transition _ | Cal.Too_few_points -> false);
  p "a sweep that never crosses fits no capacity"
    (match
       Cal.fit_capacity [ point ~ws:100_000 ~rate:1_000_000; point ~ws:100_000_000 ~rate:900_000 ]
     with
    | Cal.No_transition { contrast } -> Float.(contrast < 2.)
    | Cal.Transition _ | Cal.Too_few_points -> false);
  p "rates that do not split by working set fit a capacity flagged non-monotone"
    (match
       Cal.fit_capacity
         [
           point ~ws:100_000 ~rate:250_000;
           point ~ws:1_000_000 ~rate:2_000_000;
           point ~ws:10_000_000 ~rate:300_000;
           point ~ws:100_000_000 ~rate:1_900_000;
         ]
     with
    | Cal.Transition { monotone; _ } -> not monotone
    | Cal.No_transition _ | Cal.Too_few_points -> false);
  p "one timed point fits nothing"
    (match Cal.fit_capacity [ List.hd_exn step ] with Cal.Too_few_points -> true | _ -> false)

let () =
  let m = 16 and n = 32 in
  let ks = [ 16; 64 ] in
  let points = Cal.working_set ~m ~n ~ks ~repeats:1 (Context.auto ()) in
  p_all ~min:(List.length ks)
    "every swept extent is timed and requests at least its operands' distinct bytes" points
    ~f:(fun pt ->
      Float.(pt.Cal.wp_ms > 0.)
      && pt.Cal.wp_working_set > 0
      && pt.Cal.wp_requested >= pt.Cal.wp_k * (m + n) * 4)
