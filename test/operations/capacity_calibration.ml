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
  (* [rate] bytes per millisecond: a 1 ms point requesting [rate] bytes. *)
  {
    Cal.wp_k = 1;
    wp_working_set = ws;
    wp_requested = rate;
    wp_distinct = 1;
    wp_ms = 1.;
    wp_rank_ms = None;
    wp_bound_ms = None;
  }

(* The gh1253_cells sweeps of 2026-10-18 at 6867efcf, (working set, requested GB/s), with each box's
   STREAM-calibrated DRAM bandwidth: tuf-amd-linux (discrete) falls to DRAM between 1.08 and 2.16
   GB; minix-amd-linux (unified) never does within the sweep. *)
let recorded rows =
  List.map rows ~f:(fun (ws, gbps) -> point ~ws ~rate:(Float.to_int (gbps *. 1e6)))

let tuf =
  recorded
    [
      (33816576, 697.42);
      (67502080, 1985.10);
      (134873088, 2193.69);
      (269615104, 1997.08);
      (539099136, 1568.74);
      (1078067200, 1128.88);
      (2156003328, 667.55);
      (4311875584, 342.31);
    ]

let minix =
  recorded
    [
      (42270720, 1866.28);
      (84377600, 2594.49);
      (168591360, 2773.03);
      (337018880, 2805.20);
      (673873920, 2871.75);
      (1347584000, 1843.63);
      (2695004160, 1706.17);
      (5389844480, 1709.02);
    ]

let () =
  let step =
    [
      point ~ws:100_000 ~rate:2_000_000;
      point ~ws:1_000_000 ~rate:1_900_000;
      point ~ws:10_000_000 ~rate:300_000;
      point ~ws:100_000_000 ~rate:250_000;
    ]
  in
  let transition_at ~cached ~spilled = function
    | Cal.Transition { cached_ws; spilled_ws; capacity; monotone; _ } ->
        cached_ws = cached && spilled_ws = spilled && monotone && capacity > cached_ws
        && capacity < spilled_ws
    | Cal.All_cached _ | Cal.All_spilled _ | Cal.No_transition _ | Cal.Too_few_points -> false
  in
  p "a clean step fits a monotone capacity between the last cached and first spilled working set"
    (transition_at ~cached:1_000_000 ~spilled:10_000_000 (Cal.fit_capacity step));
  p "a launch-bound point below the fastest one does not break the fit"
    (transition_at ~cached:1_000_000 ~spilled:10_000_000
       (Cal.fit_capacity (point ~ws:10_000 ~rate:200_000 :: step)));
  p "the recorded tuf sweep fits a capacity between 1.08 GB and 2.16 GB, against its DRAM rate"
    (transition_at ~cached:1078067200 ~spilled:2156003328
       (Cal.fit_capacity ~dram_rate:2.606882e11 tuf));
  p "the recorded tuf sweep's cache bandwidth is its cached plateau, 1.1 to 2.2 TB/s"
    (match Cal.fit_capacity ~dram_rate:2.606882e11 tuf with
    | Cal.Transition { cache_rate; _ } -> Float.(cache_rate >= 1.1e12 && cache_rate <= 2.2e12)
    | _ -> false);
  p "the recorded minix sweep never reaches DRAM: the capacity is at least its largest working set"
    (match Cal.fit_capacity ~dram_rate:2.350682e11 minix with
    | Cal.All_cached { largest_ws; cache_rate } ->
        largest_ws = 5389844480 && Float.(cache_rate >= 1.7e12 && cache_rate <= 2.9e12)
    | _ -> false);
  p "without a DRAM reference, the minix sweep's 1.7x contrast fits nothing"
    (match Cal.fit_capacity minix with
    | Cal.No_transition { contrast } -> Float.(contrast < 2.)
    | _ -> false);
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
    | _ -> false);
  p "a sweep entirely below the DRAM reference is all spilled"
    (match Cal.fit_capacity ~dram_rate:1e12 step with Cal.All_spilled _ -> true | _ -> false);
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
