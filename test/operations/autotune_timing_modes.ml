(* gh-ocannl-755: [Autotune.time_routine] times a candidate against one of two objectives, and they
   do not crown the same candidate. [Isolated] is one launch plus one host synchronization -- the
   latency of a lone dispatch. [Queued] dispatches a calibrated batch back to back, synchronizes
   once and divides, so what is left is what the kernel sustains inside a stream that already has
   work in it, which is what a training step presents to every kernel of a layer.

   What is pinned here is the MECHANISM, not the ranking (the ranking is a device measurement; the
   tables are in the issue and the harness that produced them is [bin/projection_shape_bench.ml]).
   Two failures would silently undo the change while every timing still looked plausible: a batch
   depth that collapses to 1, which turns a queued search back into an isolated one, and a reading
   that is per BATCH rather than per launch, which inflates every candidate by the same factor and
   so leaves the ranking -- and only the ranking -- looking right.

   The dispatch counts are read off the computation itself: the routine is [n[0] += 1] on a
   materialized node, so after a timing call [n[0]] IS the number of launches that call made. That
   is an exact count of the thing under test, not a proxy for it. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module SC = Ir.Schedule_cache

let timing_identity =
  Some
    {
      Ir.Backend_intf.device_signature = "synthetic-device";
      toolchain_signature = Some "synthetic-compiler";
    }

open Verdict.Claims

let backend () = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* {1 Sampling policy, with an injected clock} *)

let sample_from_samples values fallback =
  let rest = ref values and calls = ref 0 in
  let sample () =
    Int.incr calls;
    match !rest with
    | x :: xs ->
        rest := xs;
        x
    | [] -> fallback
  in
  (sample, calls)

let same_sample ms : Autotune.timing_sample = { per_launch_ms = ms; contention_ms = ms }

let sample_from values fallback =
  sample_from_samples (List.map values ~f:same_sample) (same_sample fallback)

let () =
  Stdio.printf "== contention-robust sample budgeting ==\n";
  let sample, calls = sample_from [] 0.08 in
  let fast = Autotune.sample_min ~repeats:3 ~sample in
  p "a fast routine reaches the 64-sample cap"
    (!calls = 64 && fast.samples = 64 && Float.equal fast.ms 0.08 && not fast.contended);
  (* The first 15 samples model the 16-28 ms host stalls from gh-ocannl-855 and the sixteenth the
     routine's ~0.08 ms idle cost. A wall budget would stop after two samples; the minimum-sample
     floor must reach the clean one, and the population must say that the minimum came from a
     contended window rather than silently presenting it as ordinary calibration. *)
  let sample, calls = sample_from (List.init 15 ~f:(fun _ -> 20.)) 0.08 in
  let burst = Autotune.sample_min ~repeats:3 ~sample in
  p "a stall burst cannot spend the budget before the 16-sample floor"
    (!calls = 16 && burst.samples = 16 && Float.equal burst.ms 0.08);
  p "a mostly stalled sample window reports contention" burst.contended;
  (* The mirror error a depth-1 mode can make -- a window reported as its sum, or its mean, instead
     of its minimum -- pinned where it needs no device. The executed leg below cannot refuse it
     without comparing two separately sampled windows, and a uniformly delayed window is not
     [contended] (dispersion is measured WITHIN it), so such a comparison fails valid readings on an
     oversubscribed host (Codex P2, round 4 on PR #735). Its own window, rather than a second
     assertion on the burst above: one clean sample among fifteen slower ones, none of them stalled
     enough to be called contention, so the three statistics are 2, 2.9375 and 47 and one equality
     separates all three. *)
  let sample, _ = sample_from [ 2. ] 3. in
  let spread = Autotune.sample_min ~repeats:3 ~sample in
  p "a window's reading is its minimum, not its sum or its mean"
    (Float.equal spread.ms 2. && spread.samples = 16 && not spread.contended);
  p "a refused positive timing cannot enter ranking or calibration"
    (Option.is_none
       (Autotune.admitted_timing_ms
          { ms = 0.0002; contended = true; unbatched = false; samples = 16 }));
  (* gh-ocannl-888: the contention verdict judged single dispatches, whose dispersion on a GPU is
     the round trip's own tail. Refusing a depth on it starved every search on both GPU backends,
     and a deeper batch is the remedy for that dispersion rather than a casualty of it. Pinned both
     as the depth this estimate is owed and as the invariant that the verdict never enters it. *)
  p "a contended calibration still gets the depth its estimate is owed"
    (Autotune.queued_batch_depth burst = 125
    && Autotune.queued_batch_depth burst
       = Autotune.queued_batch_depth { burst with contended = false });
  let sample, _ = sample_from [] 0. in
  let unresolved = Autotune.sample_min ~repeats:3 ~sample in
  (* The separation itself: a clock that resolved nothing is refused by the admission gate on its
     own number, and is NOT reported as host contention. *)
  p "an unresolved zero clock window is refused by ranking without being called contention"
    ((not unresolved.contended) && Option.is_none (Autotune.admitted_timing_ms unresolved));
  let stalled : Autotune.timing_sample = { per_launch_ms = 0.05; contention_ms = 30. } in
  let clean : Autotune.timing_sample = { per_launch_ms = 0.05; contention_ms = 10. } in
  let sample, _ = sample_from_samples (List.init 63 ~f:(fun _ -> stalled)) clean in
  let queued_burst = Autotune.sample_min ~repeats:3 ~sample in
  p "queued contention is detected on raw batch wall before per-launch division"
    queued_burst.contended;
  let sample, _ = sample_from [] 20. in
  let slow = Autotune.sample_min ~repeats:3 ~sample in
  p "a consistently slow routine is not mistaken for host contention"
    ((not slow.contended) && Autotune.queued_batch_depth slow = 1);
  let sample, calls = sample_from [] 0.5 in
  let budgeted = Autotune.sample_min ~repeats:3 ~sample in
  p "the top-up budget accumulates per-sample time"
    (!calls = 50 && budgeted.samples = 50 && Float.equal budgeted.ms 0.5 && not budgeted.contended);
  p "a complete measurement set with a winner is cacheable"
    (Autotune.search_measurements_cacheable ~nothing_timed:false ~timings_contended:0);
  p "a contention-refused window prevents caching an incomplete winner"
    (not (Autotune.search_measurements_cacheable ~nothing_timed:false ~timings_contended:1));
  p "a search with no measured winner remains uncacheable"
    (not (Autotune.search_measurements_cacheable ~nothing_timed:true ~timings_contended:0))

(* {1 The calibration policy, without a device} *)

(* [queued_batch_depth] is what decides whether queued timing queues anything at all, and its two
   boundaries are the ones a regression crosses silently. Written as the estimate a caller could
   plausibly measure, paired with the depth the policy owes it -- a routine at or above the batch
   target must batch at 1 (there is nothing to amortize, and the two modes then agree by
   construction), a microsecond routine must be capped, and everything between must scale. *)
let depth_cases =
  [
    ("a routine far slower than the batch target", 100., 1);
    ("a routine at the batch target", 10., 1);
    ("a routine at half the batch target", 5., 2);
    ("a 0.1 ms routine", 0.1, 100);
    ("a 0.0048828125 ms routine, exactly at the cap", 0.0048828125, 2048);
    ("a 1 us routine, past the cap", 0.001, 2048);
    (* Saturates rather than raising: the ratio here is past the integer range, so a cap applied
       after the float-to-int conversion would raise instead of capping. *)
    ("a subnormal estimate", Float.min_positive_subnormal_value, 2048);
  ]

(* The estimates ranking refuses. The depth policy still owes each of them one -- batching is what a
   sub-resolution reading CALLS for (gh-ocannl-888), and an unboundedly slow routine is the far end
   of the scale the policy is for, not a reading it failed to take. *)
let degenerate_depth_cases =
  [
    ("an infinitely slow routine", Float.infinity, 1);
    ("a clock that resolved nothing (zero)", 0., 2048);
    ("a clock that resolved nothing (nan)", Float.nan, 2048);
  ]

let refinement_cases =
  [
    (* A 6 ms fixed synchronization around a 1 ms launch. Dividing the depth-2 probe by two would
       select depth 3; separating the fixed term selects the depth-4, 10 ms batch. *)
    ("a shallow probe with dominant fixed synchronization", 7., 2, 8., 4, Some 10.);
    (* An unresolved marginal observation retries deeper rather than relabeling a shallow wall as
       the cap wall. A second batch point can then separate genuine work from an inflated single. *)
    ("a probe with unresolved marginal cost", 6., 2, 6., 4, None);
    ("a clean probe after an inflated single", 6., 2, 0.25, 4, None);
    ("a probe already at the target", 1., 10, 10., 10, Some 10.);
  ]

let confirmation_cases =
  [
    (* The depth-2 probe used for an initially depth-1 calibration confirms a genuinely slow routine
       without changing its timed depth. *)
    ("a genuinely slow depth-one routine", 1, 10., 2, 20., 1, Some 10.);
    (* Metal-like steady work: the provisional batch is already target-sized, and a 25% deeper batch
       grows proportionally. Retaining the base preserves the historical Metal depth. *)
    ("a target-sized batch with confirmed marginal work", 59, 10., 73, 12.5, 59, Some 10.);
    (* A stalled base followed by a clean deeper batch has negative apparent marginal cost. It must
       retry deeper, never accept the stalled base solely because its wall crossed the target. *)
    ("an inflated target-sized batch", 2, 12., 3, 0.4, 6, None);
    (* Two nearby depths inside one shared stall still have a positive slope, but its marginal work
       is nowhere near the target. Target a full batch wall of marginal work rather than accepting
       the shallow stalled base or jumping to the cap. Dyadic inputs keep the expected wall
       exact. *)
    ("two target-sized batches dominated by a shared stall", 2, 12.25, 3, 12.5, 40, Some 21.75);
    (* A clean fit whose real fixed synchronization already exceeds the whole-wall target cannot
       make a 10 ms batch. Its marginal slope says one launch is sufficient; it must not jump to the
       cap and turn a slow candidate into seconds of uninterruptible timing. *)
    ("a batch whose fixed synchronization exceeds the target", 1, 21., 2, 31., 1, Some 21.);
    (* A deeper stalled window can manufacture a steep positive slope and an impossible negative
       fixed component. Beyond the bounded noise tolerance, that fit is unresolved too. *)
    ("a confirmation with impossible negative fixed overhead", 2, 12.2, 3, 20.3, 6, None);
    (* Legitimate fixed synchronization is part of batch wall. A stable affine pair with a small
       fixed component retains its already-target-sized base. *)
    ("a target-sized batch with fixed synchronization", 199, 10.01, 248, 12.46, 199, Some 10.01);
    (* A resolved overshoot brackets the target. Interpolation should reduce it rather than
       preserving a batch substantially longer than the contention rule's stated scale. *)
    ("a batch probe that overshoots the target", 512, 8., 1024, 16., 640, Some 10.);
    (* gh-ocannl-1098: the tolerance is a quarter of the base batch's wall, not of the target. A
       slow candidate 3% superlinear at depth 2 reads a -4 ms fixed term -- under 7% of its 61 ms
       base, over the old absolute 2.5 ms, which sent its calibration doubling. *)
    ("a slow pair superlinear by under a quarter of its base", 1, 61., 2, 126., 1, Some 61.);
    (* A resolved pair between two over-target batches: the base is not the model's first target
       crossing, and keeping it would time 256 ms batches. The crossing is one launch. *)
    ("an over-target base that is not the first crossing", 4, 256., 8, 512., 1, Some 64.);
    ("an over-target superlinear pair", 4, 260., 8, 528., 1, Some 59.);
  ]

let () =
  Stdio.printf "== queued batch depth ==\n";
  Verdict.p_all "CUDA and HIP use the raised queue-depth cap" [ "cuda"; "hip" ] ~f:(fun backend ->
      Autotune.queue_depth_cap_for_backend backend = 2048);
  Verdict.p_all "cc and Metal retain the historical queue-depth cap"
    [ "cc"; "multidev_cc"; "metal" ] ~f:(fun backend ->
      Autotune.queue_depth_cap_for_backend backend = 200);
  Verdict.p_all "every calibration estimate gets the depth the policy owes it" depth_cases
    ~f:(fun (what, est_ms, want) ->
      let got =
        Autotune.queued_batch_depth
          { ms = est_ms; contended = false; unbatched = false; samples = 0 }
      in
      if got <> want then
        Stdio.eprintf "  %s: est %g ms -> depth %d, expected %d\n%!" what est_ms got want;
      got = want);
  Verdict.p_all "every degenerate calibration estimate is refused by ranking but still batched"
    degenerate_depth_cases ~f:(fun (what, est_ms, want) ->
      let result : Autotune.timing_result =
        { ms = est_ms; contended = false; unbatched = false; samples = 16 }
      in
      let admitted = Autotune.admitted_timing_ms result in
      let depth = Autotune.queued_batch_depth result in
      if Option.is_some admitted || depth <> want then
        Stdio.eprintf "  %s: est %g ms admitted=%b, depth %d, expected depth %d\n%!" what est_ms
          (Option.is_some admitted) depth want;
      Option.is_none admitted && depth = want);
  (* The floor and the cap are the two claims a scaling-only implementation would still pass, so
     they are also asserted as the properties they are, over the same population. *)
  let all_depth_cases = depth_cases @ degenerate_depth_cases in
  Verdict.p_all "no calibration estimate ever yields a depth below 1" all_depth_cases
    ~f:(fun (_, est_ms, _) ->
      Autotune.queued_batch_depth { ms = est_ms; contended = false; unbatched = false; samples = 0 }
      >= 1);
  Verdict.p_all "no calibration estimate ever yields a depth above the cap" all_depth_cases
    ~f:(fun (_, est_ms, _) ->
      Autotune.queued_batch_depth { ms = est_ms; contended = false; unbatched = false; samples = 0 }
      <= 2048);
  Verdict.p_all "depth refinement removes fixed synchronization cost from launch scaling"
    refinement_cases ~f:(fun (what, single_ms, probe_depth, probe_ms, want_depth, want_wall) ->
      let depth, wall = Autotune.refine_queued_batch_depth ~single_ms ~probe_depth ~probe_ms in
      let wall_matches =
        match want_wall with None -> Float.is_nan wall | Some want -> Float.equal wall want
      in
      if depth <> want_depth || not wall_matches then
        Stdio.eprintf "  %s: depth %d, wall %g ms; expected depth %d, wall %s\n%!" what depth wall
          want_depth
          (Option.value_map want_wall ~default:"unresolved" ~f:Float.to_string);
      depth = want_depth && wall_matches);
  Verdict.p_all "over-target calibration requires a depth-separated marginal confirmation"
    confirmation_cases
    ~f:(fun (what, base_depth, base_ms, probe_depth, probe_ms, want_depth, want_wall) ->
      let depth, wall =
        Autotune.refine_queued_batch_depth_between ~base_depth ~base_ms ~probe_depth ~probe_ms
      in
      let wall_matches =
        match want_wall with None -> Float.is_nan wall | Some want -> Float.equal wall want
      in
      if depth <> want_depth || not wall_matches then
        Stdio.eprintf "  %s: depth %d, wall %g ms; expected depth %d, wall %s\n%!" what depth wall
          want_depth
          (Option.value_map want_wall ~default:"unresolved" ~f:Float.to_string);
      depth = want_depth && wall_matches)

(* {1 A depth-1 settle reuses its calibration (gh-ocannl-1074)} *)

(* The timed window is [sample_window] resumed from the calibration's singles, so the resumption
   itself is pinned first, as the relationship it has to keep: over any sample sequence, a window
   resumed after ANY prefix of an uninterrupted one is that uninterrupted window -- no sample more,
   none fewer. A resumption that restarted the floor, or re-counted the budget from zero, differs
   from it at some split. *)
let () =
  Stdio.printf "\n== a depth-1 settle reuses its calibration ==\n";
  let per_launch (w : Autotune.timing_sample list) = List.map w ~f:(fun s -> s.per_launch_ms) in
  let sequences =
    [
      ("slow", 3, List.init 80 ~f:(fun i -> 20. +. Float.of_int (i % 3)));
      ("fast", 3, List.init 80 ~f:(fun i -> 0.5 +. Float.of_int (i % 5)));
      ("slow, repeats above the floor", 20, List.init 80 ~f:(fun _ -> 20.));
      ("stalled start", 3, List.init 80 ~f:(fun i -> if i < 12 then 30. else 0.25));
    ]
  in
  let splits =
    List.concat_map sequences ~f:(fun (what, repeats, seq) ->
        let sample, _ = sample_from seq 1. in
        let full = Autotune.sample_window ~repeats ~sample () in
        List.init (List.length full + 1) ~f:(fun k -> (what, repeats, seq, full, k)))
  in
  Verdict.p_all "a window resumed after any prefix is the uninterrupted window" splits
    ~f:(fun (what, repeats, seq, full, k) ->
      let prior = List.take full k in
      let sample, _ = sample_from (List.drop seq k) 1. in
      let resumed = Autotune.sample_window ~prior ~repeats ~sample () in
      let same = List.equal Float.equal (per_launch resumed) (per_launch full) in
      if not same then
        Stdio.eprintf "  %s, resumed after %d: %d samples, uninterrupted %d\n%!" what k
          (List.length resumed) (List.length full);
      same)

(* The whole timing policy on a synthetic device: [batch d] dispatches [d] launches of [launch_ms]
   behind [fixed_ms] of synchronization and counts them, so every launch a call makes is counted
   exactly, and the launches after the depth decision -- the ones the timed loop dispatched itself
   -- are read off the seams rather than inferred. Dyadic costs keep the arithmetic exact. *)
type synthetic_call = {
  settled_depth : int;
  calibration_launches : int;
  window_batches : int;
  reused_batches : int;
  fresh_launches : int;
  all_launches : int;
  probes : Autotune.calibration_probe list;
      (** The calibration's batch probes in dispatch order, as [Autotune.on_calibration_probe]
          reported them (gh-ocannl-1119). *)
  probe_batches : (int * float) list list;
      (** For each reported probe, the [(depth, wall)] batches the device dispatched since the
          previous report, the first probe's segment starting at the call's first batch. *)
  unreported_batches : (int * float) list;
      (** The batches dispatched after the last probe report and before the depth decision. *)
  reading : Autotune.timing_result;
  cap : int;
  repeats : int;
}

(* Every synthetic call this file describes, as [(what, call)], for the dispatch bound checked over
   all of them at the end of the budget section. *)
let synthetic_calls : (string * synthetic_call) list ref = ref []

let synthetic_call ?(repeats = 3) ?walls ~timing ~cap ~fixed_ms ~launch_ms () =
  let launches = ref 0 and batches = ref 0 and decided = ref None in
  (* [segment] holds the batches since the last probe report, newest first. *)
  let segment = ref [] and probes = ref [] in
  let batch d =
    launches := !launches + d;
    Int.incr batches;
    let wall =
      match walls with Some f -> f !batches d | None -> fixed_ms +. (launch_ms *. Float.of_int d)
    in
    if Option.is_none !decided then segment := (d, wall) :: !segment;
    wall
  in
  let window = ref None and unreported = ref [] in
  let old_depth = !Autotune.on_batch_depth
  and old_window = !Autotune.on_timed_window
  and old_probe = !Autotune.on_calibration_probe in
  Exn.protect
    ~finally:(fun () ->
      Autotune.on_batch_depth := old_depth;
      Autotune.on_timed_window := old_window;
      Autotune.on_calibration_probe := old_probe)
    ~f:(fun () ->
      (Autotune.on_calibration_probe :=
         fun probe ->
           probes := (probe, List.rev !segment) :: !probes;
           segment := []);
      (Autotune.on_batch_depth :=
         fun d ~calibration_samples ->
           decided := Some (d, calibration_samples, !launches);
           unreported := List.rev !segment);
      (Autotune.on_timed_window :=
         fun ~samples ~reused ~wall_ms:_ ~median_wall_ms:_ ->
           window := Some (samples, reused, !launches));
      let reading = Autotune.calibrate_and_time ~timing ~repeats ~queue_depth_cap:cap ~batch in
      let settled_depth, calibration_launches, at_decision = Option.value_exn !decided in
      let window_batches, reused_batches, at_window = Option.value_exn !window in
      let probes, probe_batches = List.unzip (List.rev !probes) in
      {
        settled_depth;
        calibration_launches;
        window_batches;
        reused_batches;
        fresh_launches = at_window - at_decision;
        all_launches = !launches;
        probes;
        probe_batches;
        unreported_batches = !unreported;
        reading;
        cap;
        repeats;
      })

let describe what c =
  synthetic_calls := (what, c) :: !synthetic_calls;
  Stdio.eprintf
    "  (not part of the golden) %s: depth %d, %d calibration launches, window %d batches (%d \
     reused), %d fresh launches, %d in all, reading %g ms%s\n\
     %!"
    what c.settled_depth c.calibration_launches c.window_batches c.reused_batches c.fresh_launches
    c.all_launches c.reading.ms
    (String.concat
       [
         Printf.sprintf ", %d probes of %d batches in %g ms" (List.length c.probes)
           (List.sum (module Int) c.probes ~f:(fun pr -> pr.runs))
           (List.sum (module Float) c.probes ~f:(fun pr -> pr.wall_ms));
         (if c.reading.contended then " (contended)" else "");
         (if c.reading.unbatched then " (unbatched)" else "");
       ])

let () =
  let gpu_cap = Autotune.queue_depth_cap_for_backend "hip"
  and cc_cap = Autotune.queue_depth_cap_for_backend "cc" in
  (* 16 ms a launch: over the 10 ms batch target, so there is nothing to amortize. *)
  let slow ?repeats ?walls timing cap =
    synthetic_call ?repeats ?walls ~timing ~cap ~fixed_ms:0. ~launch_ms:16. ()
  in
  let slow_calls =
    [
      ("CUDA/HIP calibration", slow Autotune.Queued gpu_cap);
      ("cc/Metal calibration", slow Autotune.Queued cc_cap);
    ]
  in
  List.iter slow_calls ~f:(fun (what, c) -> describe ("slow, " ^ what) c);
  p_all "a slow candidate settles at depth 1 under either calibration" slow_calls ~f:(fun (_, c) ->
      c.settled_depth = 1);
  p_all "a depth-1 settle times no fresh window: its window is the calibration's singles" slow_calls
    ~f:(fun (_, c) ->
      c.fresh_launches = 0 && c.reused_batches = 16 && c.window_batches = 16
      && c.all_launches = c.calibration_launches);
  p_all "a reused window's reading is the per-launch time" slow_calls ~f:(fun (_, c) ->
      Float.equal c.reading.ms 16. && c.reading.samples = 16 && not c.reading.contended);
  (* The negative control, on the same synthetic routine: [Isolated] has no calibration to reuse, so
     the same instrument must count its whole 16-launch window after the depth decision. A counter
     that never counted would pass the claim above and fail here. *)
  let iso = slow Autotune.Isolated gpu_cap in
  describe "slow, isolated" iso;
  p "the instrument counts a fresh window where one is timed: isolated dispatches its 16"
    (iso.settled_depth = 1 && iso.calibration_launches = 0 && iso.reused_batches = 0
   && iso.fresh_launches = 16 && iso.window_batches = 16 && iso.all_launches = 16);
  p_all "at depth 1 the queued reading is the isolated one" slow_calls ~f:(fun (_, c) ->
      Float.equal c.reading.ms iso.reading.ms);
  (* Topping up: a caller floor above the calibration's sixteen singles is met by fresh launches,
     exactly as many as it asks beyond them. *)
  let topped = slow ~repeats:20 Autotune.Queued gpu_cap in
  describe "slow, repeats 20" topped;
  p "a caller floor above the calibration's is topped up with exactly the missing launches"
    (topped.settled_depth = 1 && topped.reused_batches = 16 && topped.window_batches = 20
   && topped.fresh_launches = 4
    && topped.all_launches = topped.calibration_launches + 4);
  (* The reused window is judged for contention whole, as a fresh window would be: a majority of
     stalled singles (9 of the 16 at 2.5x) refuses the reading rather than being dropped from the
     verdict because calibration took them. The probe and every later batch are clean. *)
  let stalled =
    slow Autotune.Queued gpu_cap ~walls:(fun nth d ->
        if nth <= 9 then 40. *. Float.of_int d else 16. *. Float.of_int d)
  in
  describe "slow, stalled singles" stalled;
  p "a reused window is judged for contention whole"
    (stalled.settled_depth = 1 && stalled.reused_batches = 16 && stalled.reading.contended);
  (* Depth > 1 is unchanged: the singles are a different quantity from the batch and are left out,
     so every one of the window's batches is dispatched by the loop at the settled depth. *)
  let fast timing cap = synthetic_call ~timing ~cap ~fixed_ms:0.0625 ~launch_ms:0.0078125 () in
  let fast_calls =
    [
      ("CUDA/HIP calibration", fast Autotune.Queued gpu_cap);
      ("cc/Metal calibration", fast Autotune.Queued cc_cap);
    ]
  in
  List.iter fast_calls ~f:(fun (what, c) -> describe ("fast, " ^ what) c);
  p_all "a batching depth reuses nothing and dispatches its whole window" fast_calls
    ~f:(fun (_, c) ->
      c.settled_depth > 1 && c.reused_batches = 0
      && c.fresh_launches = c.window_batches * c.settled_depth
      && c.all_launches = c.calibration_launches + c.fresh_launches)

(* {1 The no-verdict fallback is wall-bounded (gh-ocannl-1096)} *)

(* When the CUDA/HIP calibration's affine fits never resolve, it falls back to a depth the fits did
   not choose. That fallback used to be the 2048 cap unconditionally: a bet that the per-launch cost
   is negligible, which the cap bounds in launches but not in wall. On gfx1151 a ~61 ms gpt2_mini
   candidate lost that bet: 126 s batches, 33,529 launches and 2016 s on one timing call. The
   synthetic devices below reproduce the ways to lose it, and the claim is the wall bound itself,
   read off each device's own clean cost model rather than off the policy: the launch work of the
   settled batch fits the 10 ms target, or the batch is a single launch.

   The slow device is 64 ms a launch whose batches grow slightly faster than linearly (1.5% at depth
   4, 12.5% at depth 32), and whose depth-2 probe reads non-monotone. Drift proportional to the wall
   is all it takes: the fit's 2.5 ms noise tolerance is under 1% of a slow candidate's batch, so
   every depth-separated pair reads as an impossible negative fixed term, and each unresolved pair
   doubles the depth until the validation loop runs out.

   The two threshold devices are the cases a bound extrapolated through a per-launch cost gets wrong
   (Codex P1, rounds 1 and 2 on PR #846). Each is cheap at shallow depth and then grows
   quadratically from depth 80. The first reaches the threshold at its provisional depth (400 ms
   there), so its cheapest [wall / depth] is the single launch's and would put the fallback right
   back at 80. The second has a 0.1 ms fixed term, so its clean provisional probe (depth 50, 5.1 ms)
   projects a target depth one rounding step above where such a bound would land, and that
   validation reads ~600 ms.

   The fast device is the negative control: a ~8 us launch behind a 62.5 us round trip whose
   validation and confirmation batches, the ones deeper than its provisional probe, are stalled to a
   flat 40 ms, which also leaves the fits unresolved. A stall and a queue threshold read the same,
   so its fallback cannot go past what it measured within the target either -- but it must still
   batch: a fix that returned depth 1 on every unresolved calibration would turn its queued reading
   back into an isolated one.

   Where no batch at all was measured within the target for a candidate whose single launch owed it
   one -- a 0.1 ms kernel whose every batched probe is stalled to 40 ms (Codex P1, round 3 on PR
   #846) -- the depth-1 reading left would be the isolated objective. Such a call is refused, not
   ranked, exactly as a stalled window is. A slow candidate, owed depth 1 by its own single launch,
   is not. The first threshold device used to be refused the same way, on every search; since
   gh-ocannl-1098 a rescue probe below its shallowest over-target batch times it (below). *)
let () =
  Stdio.printf "\n== the no-verdict fallback is wall-bounded ==\n";
  let gpu_cap = Autotune.queue_depth_cap_for_backend "hip" in
  let unresolved what ?(fixed_ms = 0.) ~launch_ms clean =
    let c =
      synthetic_call ~timing:Autotune.Queued ~cap:gpu_cap ~fixed_ms ~launch_ms
        ~walls:(fun _nth d -> clean d)
        ()
    in
    describe (what ^ ", unresolved fits") c;
    c
  in
  let slow_clean d = (64. *. Float.of_int d) +. (Float.of_int (d * d) /. 4.) in
  let slow = unresolved "slow" ~launch_ms:64. (fun d -> if d = 2 then 60. else slow_clean d) in
  let threshold_clean d =
    if d < 80 then 0.125 *. Float.of_int d else 0.0625 *. Float.of_int (d * d)
  in
  let threshold = unresolved "queue threshold" ~launch_ms:0.125 threshold_clean in
  let offset_launch_work d =
    if d < 80 then 0.1 *. Float.of_int d else 0.0625 *. Float.of_int (d * d)
  in
  let offset =
    unresolved "queue threshold above a fixed term" ~fixed_ms:0.1 ~launch_ms:0.1 (fun d ->
        if d < 80 then 0.1 +. offset_launch_work d else offset_launch_work d)
  in
  let fast_fixed_ms = 0.0625 and fast_launch_ms = 0.0078125 in
  let fast_launch_work d = fast_launch_ms *. Float.of_int d in
  let fast =
    unresolved "fast, stalled validation" ~fixed_ms:fast_fixed_ms ~launch_ms:fast_launch_ms
      (fun d -> if d >= 1272 then 40. else fast_fixed_ms +. fast_launch_work d)
  in
  let stalled_clean d = if d = 1 then 0.1 else 40. in
  let stalled = unresolved "every batch stalled" ~launch_ms:0.1 stalled_clean in
  let cases =
    [
      ("slow", slow, slow_clean);
      ("every batch stalled", stalled, stalled_clean);
      ("queue threshold", threshold, threshold_clean);
      ("queue threshold above a fixed term", offset, offset_launch_work);
      ("fast", fast, fast_launch_work);
    ]
  in
  Verdict.p_all
    "an unresolved calibration never settles on a batch whose launch work exceeds the target" cases
    ~f:(fun (what, c, launch_work) ->
      let ok =
        c.settled_depth = 1 || Float.(launch_work c.settled_depth <= Autotune.queued_batch_ms)
      in
      if not ok then
        Stdio.eprintf "  %s: depth %d carries %g ms of launch work\n%!" what c.settled_depth
          (launch_work c.settled_depth);
      ok);
  p "a slow candidate whose fits never resolve is timed at depth 1, as isolated times it"
    (slow.settled_depth = 1
    && Option.equal Float.equal (Autotune.admitted_timing_ms slow.reading) (Some (slow_clean 1)));
  if Option.is_some (Autotune.admitted_timing_ms stalled.reading) then
    Stdio.eprintf "  every batch stalled: depth %d reading %g ms was admitted\n%!"
      stalled.settled_depth stalled.reading.ms;
  p "a candidate owed a batch that measured none within the target is refused, not timed isolated"
    (Option.is_none (Autotune.admitted_timing_ms stalled.reading));
  Verdict.p_all "a fallback that kept a batch measured within the target is admitted"
    [ ("queue threshold above a fixed term", offset); ("fast", fast) ]
    ~f:(fun (what, c) ->
      let admitted =
        c.settled_depth > 1 && Option.is_some (Autotune.admitted_timing_ms c.reading)
      in
      if not admitted then
        Stdio.eprintf "  %s: depth %d reading %g ms%s\n%!" what c.settled_depth c.reading.ms
          (if c.reading.contended then " (refused)" else "");
      admitted);
  (* The cost the issue is about, in launches: sixteen singles, twelve probes at each of depths 2
     through 32, and a timed window that reuses the singles. The cap fallback spent 16 x 2048
     more. *)
  p "a slow candidate's unresolved calibration costs no more than its probes"
    (slow.fresh_launches = 0
    && slow.all_launches = slow.calibration_launches
    && slow.all_launches <= 16 + (12 * (2 + 4 + 8 + 16 + 32)));
  (* {2 The probes themselves are wall-budgeted (gh-ocannl-1098)}

     The fallback above bounds the depth a call settles on, not what the calibration spends getting
     there. Its probes were budgeted in probes and launches only, and a launch can cost anything:
     the slow device above spent 760 launches (~49 s) doubling before the fallback, and the first
     threshold device ~1600 s doubling toward the cap. Four fixes, each pinned by a claim that
     failed before it:

     - the probes' summed wall has a budget, after which only the rescue probe may start, and each
     probe stops at three minima once its own wall passes two target-sized probes' -- the same
     threshold devices a per-launch bound was defeated by are the fixtures here, since a wall budget
     makes no per-launch extrapolation for them to defeat; - the fit's noise tolerance is a quarter
     of the base batch's wall rather than a quarter of the target, so a slow candidate's slightly
     superlinear first pair fits at once; - a resolved fit whose base is over the target but not its
     first crossing projects to the crossing, rather than keeping a batch of any length; - before
     refusing a candidate that measured no batch within the target, one rescue probe below its
     shallowest over-target batch, and a refusal that stands carries a reason of its own.

     The first threshold device, and the three below, are the fixtures. A 64 ms linear candidate is
     the depth-1 settle every slow gpt2_mini step takes (gh-ocannl-834). A 61 ms one whose depth-2
     batch is 3% superlinear is the pair the absolute 2.5 ms tolerance refused. A 16 ms one whose
     depth-2 probe reads low sends its calibration doubling from an unresolved pair to a resolved
     one between two over-target batches, depths 4 and 8. The clean fast kernel is the negative
     control: a converging calibration's probes are target-sized, and the budget must not touch
     them. *)
  Stdio.printf "\n== calibration probes are wall-budgeted ==\n";
  let device what ?(fixed_ms = 0.) ~launch_ms clean =
    let c =
      synthetic_call ~timing:Autotune.Queued ~cap:gpu_cap ~fixed_ms ~launch_ms
        ~walls:(fun _nth d -> clean d)
        ()
    in
    describe what c;
    c
  in
  let linear_clean d = 64. *. Float.of_int d in
  let linear = device "64 ms a launch" ~launch_ms:64. linear_clean in
  let superlinear_clean d = (59. *. Float.of_int d) +. (2. *. Float.of_int (d * d)) in
  let superlinear =
    device "61 ms a launch, 3% superlinear at depth 2" ~launch_ms:61. superlinear_clean
  in
  let dip_clean d = 16. *. Float.of_int d in
  let dip =
    device "16 ms a launch, its depth-2 probe reading low" ~launch_ms:16. (fun d ->
        if d = 2 then 15. else dip_clean d)
  in
  (* Codex P1, round 2 on PR #847: a crossing sampled below two over-target batches is refitted
     against the upper one, and a fixed-dominated refit projects a batch of marginal work far past
     both. Singles at 12 ms, a low depth-2 probe (11 ms) and a pair (4, 40) / (8, 48) whose fixed
     term fills the target: the crossing samples depth 5 at 47 ms, and the refit (5, 47) / (8, 48)
     wants depth 30 -- past which this device's queue cost jumps. *)
  let jump_clean d =
    match d with
    | 1 -> 12.
    | 2 -> 11.
    | 4 -> 40.
    | 5 -> 47.
    | d when d <= 8 -> 48.
    | d -> 400. *. Float.of_int d
  in
  let jump =
    device "fixed-dominated, its queue cost jumping past depth 8" ~launch_ms:12. jump_clean
  in
  (* gh-ocannl-1100: two more exits settle on a depth no probe measured. The first is the last
     validation's affine projection. Singles at 2.5 ms and a concave curve walk the four validations
     through depths 6, 8, 10 and 14 (the longest path's), each reading under the target, and the
     last pair (10, 9.5) / (14, 9.5625) is nearly flat: its fit wants depth 42, three times the
     deepest probe, past which this device's queue cost jumps. *)
  let projected_clean d =
    match d with
    | 1 -> 2.5
    | 4 -> 7.
    | 6 -> 8.5
    | 8 -> 9.25
    | 10 -> 9.5
    | 14 -> 9.5625
    | d -> 400. *. Float.of_int d
  in
  let projected =
    device "the last validation's projection, its queue cost jumping past depth 14" ~launch_ms:2.5
      projected_clean
  in
  (* The second is a confirmation that reads below the target, which the confirmation branch scales
     from linearly. Singles at 2.5 ms, a provisional probe at depth 4 reading exactly the target,
     and a confirmation at depth 5 reading a quarter of that -- a clock ramping up between the two
     -- scale to depth 20, four times the deepest probe, past which the queue cost jumps. *)
  let scaled_clean d =
    match d with 1 -> 2.5 | 4 -> 10. | 5 -> 2.5 | d -> 400. *. Float.of_int d
  in
  let scaled =
    device "a confirmation scaled past depth 5, where its queue cost jumps" ~launch_ms:2.5
      scaled_clean
  in
  let converging =
    device "fast, clean" ~fixed_ms:fast_fixed_ms ~launch_ms:fast_launch_ms (fun d ->
        fast_fixed_ms +. fast_launch_work d)
  in
  (* The longest path [queue_calibration_max_probes] counts, walked by one device: the provisional
     probe, four validations, a confirmation, its stall retry, a sampled shallower crossing, and the
     rescue. Singles at 2.5 ms give provisional depth 4. Its probe (7 ms) and three validations read
     under the target on a concave curve, so each fit projects one step deeper (6, 8, 10, 14). Past
     depth 10 the cost jumps to a flat ~10.1 ms, so the fourth validation brackets the target at its
     own depth and hands 14 to [confirm_or_scale]. The confirmation at 17 lands in a twelve-batch
     stall over twice that base, so it is retried. The clean retry's fit puts the crossing at 12,
     which is sampled there. Against 14 that sample fits a fixed term just under the target, a
     crossing of one launch, and a settle at depth 1 the singles contradict. The rescue below the
     shallowest over-target batch (12) reads within the target at 11. Every batch is under two
     target-sized batches, so every probe takes all twelve minima, and the seven probes before the
     sampled crossing spend ~898 ms: the wall budget ends nothing early. *)
  let longest_clean d =
    match d with
    | 1 -> 2.5
    | 4 -> 7.
    | 6 -> 8.5
    | 8 -> 9.25
    | 10 -> 9.5
    | 11 -> 9.75
    | 12 -> 10.0859375
    | 14 -> 10.1015625
    | 17 -> 10.2421875
    | d -> 16. *. Float.of_int d
  in
  let longest =
    let stalled_batches = ref Autotune.queue_batch_probe_runs in
    device "the longest calibration path" ~launch_ms:2.5 (fun d ->
        if d = 17 && !stalled_batches > 0 then (
          Int.decr stalled_batches;
          20.25)
        else longest_clean d)
  in
  (* A confirmation that reads as a stall over twice its target-sized base is retried once at the
     same depth -- unless the stall itself spent the wall budget. Singles at 2.5 ms give provisional
     depth 4, whose probe reads exactly the target, so the confirmation at 5 is the second probe.
     Its batches are stalled to 320 ms, so it stops at three minima, and with the provisional
     probe's twelve target-sized batches the two have charged 1080 ms, past the 960 ms budget: the
     retry must not start. Before gh-ocannl-1119 the test grouped a retry with the confirmation it
     repeats, so no device's retry was ever held against the budget. *)
  let stall_at_budget =
    device "a confirmation stall that spends the budget" ~launch_ms:2.5 (fun d ->
        match d with 1 -> 2.5 | 4 -> 10. | 5 -> 320. | d -> 2.5 *. Float.of_int d)
  in
  let every_device =
    [
      ("slow, unresolved fits", slow);
      ("every batch stalled", stalled);
      ("queue threshold", threshold);
      ("queue threshold above a fixed term", offset);
      ("fast, stalled validation", fast);
      ("64 ms a launch", linear);
      ("61 ms a launch, superlinear", superlinear);
      ("16 ms a launch, low depth-2 probe", dip);
      ("fixed-dominated, jumping past depth 8", jump);
      ("the last validation's projection", projected);
      ("a confirmation scaled from below the target", scaled);
      ("fast, clean", converging);
      ("the longest calibration path", longest);
      ("a confirmation stall that spends the budget", stall_at_budget);
    ]
  in
  let is_rescue (pr : Autotune.calibration_probe) =
    match pr.role with Rescue_probe -> true | _ -> false
  in
  (* The rescue is exempt: it is charged to no budget, because its depth bounds its own wall. *)
  Verdict.p_all
    "no calibration probe but the rescue starts once the probes' wall has reached the budget"
    every_device ~f:(fun (what, c) ->
      let late =
        List.fold c.probes ~init:(0., 0) ~f:(fun (spent, late) (pr : Autotune.calibration_probe) ->
            ( spent +. pr.wall_ms,
              if Float.(spent >= Autotune.queue_calibration_wall_ms) && not (is_rescue pr) then
                late + 1
              else late ))
        |> snd
      in
      if late > 0 then Stdio.eprintf "  %s: %d probes started past the budget\n%!" what late;
      late = 0);
  Verdict.p_all "a rescue probe is the calibration's last" every_device ~f:(fun (_, c) ->
      not (List.exists (Option.value ~default:[] (List.drop_last c.probes)) ~f:is_rescue));
  p "a stall retry is not started once the stalled confirmation has spent the budget"
    (List.equal Poly.equal
       (List.map stall_at_budget.probes ~f:(fun pr -> pr.role))
       [ Provisional_probe; Confirmation_probe ]);
  Verdict.p_all
    "a converging calibration is untouched by the budgets: every probe takes all twelve minima"
    converging.probes ~f:(fun pr ->
      if pr.runs <> Autotune.queue_batch_probe_runs then
        Stdio.eprintf "  fast, clean: depth %d took %d minima\n%!" pr.depth pr.runs;
      pr.runs = Autotune.queue_batch_probe_runs);
  p "a converging calibration still reaches its affine target depth"
    (converging.settled_depth = 1272);
  (* gh-ocannl-834's depth-1 settle: sixteen singles, then the depth-2 confirmation, whose 128 ms
     batches pass the per-probe budget at the second -- three minima rather than twelve, 6 launches
     rather than 24. *)
  p "a 64 ms candidate's depth-2 confirmation stops at three minima"
    (linear.settled_depth = 1 && linear.fresh_launches = 0
    && linear.calibration_launches = 16 + (3 * 2));
  p "a slow candidate's superlinear first pair fits within the wall-relative tolerance"
    (superlinear.settled_depth = 1
    && List.equal Int.equal (List.map superlinear.probes ~f:(fun pr -> pr.depth)) [ 2 ]);
  Verdict.p_all
    "a resolved calibration never settles on a batch whose launch work exceeds the target"
    [
      ("64 ms a launch", linear, linear_clean);
      ("61 ms a launch, superlinear", superlinear, superlinear_clean);
      ("16 ms a launch, low depth-2 probe", dip, dip_clean);
      ("fast, clean", converging, fast_launch_work);
    ]
    ~f:(fun (what, c, launch_work) ->
      let ok =
        c.settled_depth = 1 || Float.(launch_work c.settled_depth <= Autotune.queued_batch_ms)
      in
      if not ok then
        Stdio.eprintf "  %s: depth %d carries %g ms of launch work\n%!" what c.settled_depth
          (launch_work c.settled_depth);
      ok);
  (* The escaped depth is unmeasured and beyond every depth the calibration measured: neither the
     wall budget nor the fallback would ever check it. *)
  p "a refitted crossing never settles past the batches it was checked against"
    (let deepest = List.fold jump.probes ~init:1 ~f:(fun m pr -> Int.max m pr.depth) in
     if jump.settled_depth > deepest then
       Stdio.eprintf "  fixed-dominated: settled %d past the deepest probe %d\n%!"
         jump.settled_depth deepest;
     jump.settled_depth <= deepest && jump.settled_depth > 1);
  (* The bound over the probe record (gh-ocannl-1100), on every CUDA/HIP device in this section.
     Each of the two fixtures above walks its own exit -- read off its probes' roles -- and settles
     exactly at the bound, where its projection wanted deeper. *)
  let deepest_probe (c : synthetic_call) =
    List.fold c.probes ~init:1 ~f:(fun m (pr : Autotune.calibration_probe) -> Int.max m pr.depth)
  in
  let within_bound what c =
    let ok = c.settled_depth <= Autotune.queue_depth_projection_factor * deepest_probe c in
    if not ok then
      Stdio.eprintf "  %s: settled %d, past %d times the deepest probe %d\n%!" what c.settled_depth
        Autotune.queue_depth_projection_factor (deepest_probe c);
    ok
  in
  Verdict.p_all
    "no calibration settles deeper than queue_depth_projection_factor times the deepest batch it \
     probed"
    every_device ~f:(fun (what, c) -> within_bound what c);
  let roles (c : synthetic_call) = List.map c.probes ~f:(fun pr -> pr.role) in
  p "the last validation's projection past its deepest probe is capped at the bound"
    (List.equal Poly.equal (roles projected)
       [ Provisional_probe; Validation_probe; Validation_probe; Validation_probe; Validation_probe ]
    && projected.settled_depth = Autotune.queue_depth_projection_factor * deepest_probe projected);
  p "a confirmation scaled from below the target is capped at the bound"
    (List.equal Poly.equal (roles scaled) [ Provisional_probe; Confirmation_probe ]
    && scaled.settled_depth = Autotune.queue_depth_projection_factor * deepest_probe scaled);
  p
    "a kernel with a queue threshold below its provisional depth is rescued, timed within the \
     target"
    (threshold.settled_depth > 1
    && Float.(threshold_clean threshold.settled_depth <= Autotune.queued_batch_ms)
    && Option.is_some (Autotune.admitted_timing_ms threshold.reading));
  p "a refusal no rescue could lift carries its own reason, apart from the contention verdict"
    (stalled.reading.unbatched && not stalled.reading.contended);
  (* The bound is reached, not exceeded: the control flow's longest path is exactly
     [queue_calibration_max_probes] long, so enforcing the count cuts no path short. The path is the
     one the comment above [longest_clean] walks, branch by branch. *)
  let longest_roles = List.map longest.probes ~f:(fun pr -> pr.role) in
  if List.length longest.probes <> Autotune.queue_calibration_max_probes then
    Stdio.eprintf "  the longest calibration path: %d probes at depths %s\n%!"
      (List.length longest.probes)
      (String.concat ~sep:", " (List.map longest.probes ~f:(fun pr -> Int.to_string pr.depth)));
  p "the longest calibration path dispatches queue_calibration_max_probes probes"
    (List.length longest.probes = Autotune.queue_calibration_max_probes);
  p
    "the longest calibration path is the provisional probe, four validations, a confirmation, its \
     stall retry, a sampled crossing and the rescue"
    (List.equal Poly.equal longest_roles
       [
         Provisional_probe;
         Validation_probe;
         Validation_probe;
         Validation_probe;
         Validation_probe;
         Confirmation_probe;
         Stall_retry_probe;
         Crossing_probe;
         Rescue_probe;
       ]);
  p "the longest calibration path's rescue probes rescue_depth and times the candidate there"
    (match List.split_n longest.probes (List.length longest.probes - 1) with
    | earlier, [ ({ role = Rescue_probe; _ } as rescue) ] ->
        Option.equal Int.equal
          (Autotune.rescue_depth ~observed:(List.map earlier ~f:(fun pr -> (pr.depth, pr.min_ms))))
          (Some rescue.depth)
        && longest.settled_depth = rescue.depth
        && Option.is_some (Autotune.admitted_timing_ms longest.reading)
    | _ -> false);
  (* [time_routine]'s documented queued maximum, over every synthetic device in this file: a warmup
     and at most 64 synchronized singles, at most [queue_calibration_max_probes] probes of at most
     [queue_batch_probe_runs] batches, and a timed window of at most [max 64 repeats] batches, every
     batch at most the cap deep. [calibrate_and_time] dispatches no warmup, so the 65 is slack of
     one. *)
  let max_probe_batches = Autotune.queue_calibration_max_probes * Autotune.queue_batch_probe_runs in
  Verdict.p_all "every synthetic device's dispatches stay within time_routine's documented maximum"
    !synthetic_calls ~f:(fun (what, c) ->
      let bound = 65 + (c.cap * (max_probe_batches + Int.max 64 c.repeats)) in
      if c.all_launches > bound then
        Stdio.eprintf "  %s: %d launches past the documented %d\n%!" what c.all_launches bound;
      c.all_launches <= bound);
  (* The seam against the device's own batch log, so a probe that dispatched without reporting, or
     misreported what it measured, fails here rather than thinning every claim above: each report
     ends exactly the batches dispatched since the previous one, at its depth, with their minimum
     and their finite positive sum; only the first report's segment starts earlier, with the
     synchronized singles; and nothing but singles precedes the depth decision unreported. *)
  let singles = List.for_all ~f:(fun (d, _) -> d = 1) in
  Verdict.p_all "every reported probe is exactly the batches the device dispatched for it"
    !synthetic_calls ~f:(fun (what, c) ->
      let matches nth (pr : Autotune.calibration_probe) segment =
        let before, own = List.split_n segment (List.length segment - pr.runs) in
        pr.runs >= 1
        && List.length own = pr.runs
        && List.for_all own ~f:(fun (d, _) -> d = pr.depth)
        && Float.equal pr.min_ms
             (List.fold own ~init:Float.infinity ~f:(fun m (_, w) -> Float.min m w))
        && Float.equal pr.wall_ms
             (List.sum
                (module Float)
                own
                ~f:(fun (_, w) -> if Float.is_finite w && Float.is_positive w then w else 0.))
        && if nth = 0 then singles before else List.is_empty before
      in
      let ok =
        List.for_alli (List.zip_exn c.probes c.probe_batches) ~f:(fun nth (pr, segment) ->
            matches nth pr segment)
        &&
        if List.is_empty c.probes then singles c.unreported_batches
        else List.is_empty c.unreported_batches
      in
      if not ok then
        Stdio.eprintf "  %s: the reported probes disagree with the dispatched batches\n%!" what;
      ok);
  Stdio.eprintf
    "  (not part of the golden) most dispatches of any synthetic device: %d, probes %d\n%!"
    (List.fold !synthetic_calls ~init:0 ~f:(fun m (_, c) -> Int.max m c.all_launches))
    (List.fold !synthetic_calls ~init:0 ~f:(fun m (_, c) -> Int.max m (List.length c.probes)))

(* {1 The setting's spelling} *)

let () =
  Stdio.printf "\n== autotune_timing spelling ==\n";
  let reads s want =
    match Autotune.timing_of_setting s with got -> Poly.equal got want | exception _ -> false
  in
  p "\"queued\" selects the queued objective" (reads "queued" Autotune.Queued);
  p "\"isolated\" selects the isolated objective" (reads "isolated" Autotune.Isolated);
  p "the spelling is case- and space-insensitive" (reads " ISOLATED\n" Autotune.Isolated);
  (* The negative control: a misspelling must be refused rather than falling back to a mode the
     caller did not ask for -- silently timing under the other objective is the failure this whole
     issue is about. *)
  p "a misspelling is refused rather than defaulted"
    (match Autotune.timing_of_setting "batched" with
    | _ -> false
    | exception Invalid_argument _ -> true
    | exception _ -> false)

(* {1 The instrument, on a routine that counts its own launches} *)

(* [n[0] += 1] over a one-element materialized node: every dispatch adds exactly one, and f32 counts
   integers exactly far past any dispatch count these loops can reach. Deliberately trivial, so that
   the same source is a scalar kernel on a GPU backend too -- this test runs on whichever backend is
   configured, and a nest heavy enough to be interesting would run as a single work item there. *)
let counter_node = Ll_test.node_factory ~first_id:9900 ~dims:[| 1 |] () "gh755_counter"

let counter_routine () =
  Ll_test.materialize counter_node;
  let idx = Ll_test.fixed 0 in
  let bump =
    Ll_test.set_at counter_node idx
      (Ll_test.add (Ll_test.get counter_node [| idx |]) (Ll_test.c 1.))
  in
  let o = Ll_test.optimize ~materialized:[ counter_node ] ~name:"gh755_counter" bump in
  let ctx, routine = Ll_test.link ~name:"gh755_counter" o in
  (o, ctx, routine)

type reading = {
  ms : float;
  contended : bool;
  samples : int;
  wall_ms : float;
  dispatches : int;
  depth : int;
  calibration_dispatches : int;
  timed_wall_ms : float;
  timed_median_ms : float;
  timed_batches : int;
  reused : int;
}

(* Held so the cache-key section below asks about the SAME lowering the instrument measured, rather
   than minting a second one whose canonical form it would have to argue is equivalent. *)
let measured : (LL.optimized * Context.t) option ref = ref None

let () =
  Stdio.printf "\n== the instrument's two modes ==\n";
  let opt, ctx, routine = counter_routine () in
  let ctx = Context.set_values ctx counter_node [| 0. |] in
  measured := Some (opt, ctx);
  let count () = Int.of_float (Context.get_values ctx counter_node).(0) in
  (* The depth each call settles on, off the instrument's own observation seam: the queued call
     calibrates independently, so nothing derived from a reading taken outside it (gh-ocannl-851,
     Codex round 2 on PR #521) stands in for what it actually used. *)
  let depth_seen = ref 0 and calibration_dispatches_seen = ref 0 in
  (Autotune.on_batch_depth :=
     fun d ~calibration_samples ->
       depth_seen := d;
       calibration_dispatches_seen := calibration_samples);
  (* The window the returned minimum was taken over (gh-ocannl-994), off the same kind of seam: the
     batches the timed loop ran and their summed wall. The per-launch envelope below is written
     against THIS window rather than against the call the test wraps a clock around, whose wall also
     holds the warmup and the calibration's synchronized singles. *)
  let timed_wall_seen = ref 0. and timed_batches_seen = ref 0 and timed_median_seen = ref 0. in
  let reused_seen = ref 0 in
  (Autotune.on_timed_window :=
     fun ~samples ~reused ~wall_ms ~median_wall_ms ->
       timed_batches_seen := samples;
       reused_seen := reused;
       timed_wall_seen := wall_ms;
       timed_median_seen := median_wall_ms);
  let measure timing =
    let before = count () in
    let c0 = Mtime_clock.counter () in
    let result = Autotune.time_routine ~repeats:3 ~timing ctx routine in
    let wall_ms = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
    {
      ms = result.ms;
      contended = result.contended || result.unbatched;
      samples = result.samples;
      wall_ms;
      dispatches = count () - before;
      depth = !depth_seen;
      calibration_dispatches = !calibration_dispatches_seen;
      timed_wall_ms = !timed_wall_seen;
      timed_median_ms = !timed_median_seen;
      timed_batches = !timed_batches_seen;
      reused = !reused_seen;
    }
  in
  (* The anchor the low side of the per-launch envelope below is written against: one launch plus
     one host synchronization, hand-rolled off the same two primitives the instrument uses and
     minimized over [ref_launch_samples]. This is the quantity [Isolated] is DEFINED as, so it is
     what a reading of it owes agreement with -- and, being a minimum, it is not the call's wall
     mean, which is where contention lands. Taken three times, on either side of each reading, so
     the anchor is a minimum over the whole window the two readings were taken in rather than a
     snapshot of whatever the host was doing before them. *)
  let ref_launch_samples = 16 in
  let ref_ctx = ref ctx in
  let ref_round_trip () =
    let best = ref Float.infinity in
    for _ = 1 to ref_launch_samples do
      let c0 = Mtime_clock.counter () in
      ref_ctx := Context.run !ref_ctx routine;
      Context.sync !ref_ctx;
      let dt = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
      if Float.(dt < !best) then best := dt
    done;
    !best
  in
  let before = ref_round_trip () in
  let iso = measure Autotune.Isolated in
  let que = measure Autotune.Queued in
  let between = ref_round_trip () in
  let que2 = measure Autotune.Queued in
  let iso2 = measure Autotune.Isolated in
  let after = ref_round_trip () in
  let floor_ms = Float.min before (Float.min between after) in
  (* [contended] is reported here because it is the escape hatch every executed claim below is
     written around: a reading that took it is outside the envelope those claims calibrate, so a
     calibration run has to be able to tell the two apart. *)
  let contention r = if r.contended then " (contended)" else "" in
  Stdio.eprintf
    "  (not part of the golden) isolated %.6f ms%s over %d dispatches in %.1f ms wall; queued %.6f \
     ms%s over %d dispatches (batch depth %d) in %.1f ms wall; second round isolated %.6f ms%s, \
     queued %.6f ms%s (batch depth %d); round trip %.6f ms (%.6f/%.6f/%.6f); queued/round-trip %.4f\n\
     %!"
    iso.ms (contention iso) iso.dispatches iso.wall_ms que.ms (contention que) que.dispatches
    que.depth que.wall_ms iso2.ms (contention iso2) que2.ms (contention que2) que2.depth floor_ms
    before between after (que.ms /. floor_ms);
  let finite r = Float.is_finite r.ms && Float.is_positive r.ms in
  p "both modes returned a positive finite per-launch time or reported contention"
    ((finite iso || iso.contended) && (finite que || que.contended));
  (* One launch per timed run, at least [repeats] runs and at most the 64-run top-up cap, plus the
     warmup. Two-sided: the upper bound would also admit a loop that stopped at the warmup. *)
  p "isolated timing either reports contention or dispatches one launch per timed run"
    (iso.contended || (iso.samples >= 16 && iso.samples <= 64 && iso.dispatches = 1 + iso.samples));
  p "isolated timing reports batch depth 1" (iso.depth = 1);
  (* The seam's report is not taken on faith: past the warmup (1) and the calibration dispatches,
     the dispatch counter must decompose into whole batches of the reported depth, between the 16
     guaranteed timed samples and the 64-run top-up cap. A loop batching at some depth other than
     the one it reported fails this on any count the reported depth does not divide. The batches a
     depth-1 settle reused from the calibration (gh-ocannl-1074) are already in its count, and only
     a depth-1 settle may reuse any. *)
  p "queued timing either refuses contention or dispatches whole batches at the reported depth"
    (que.contended
    || que.depth >= 1 && que.calibration_dispatches >= 16 && que.samples >= 16 && que.samples <= 64
       && (que.reused = 0 || que.depth = 1)
       && que.dispatches = 1 + que.calibration_dispatches + ((que.samples - que.reused) * que.depth)
    );
  p "isolated timing reuses no calibration" (iso.reused = 0);
  (* Depth > 1 is what queued mode IS. Gated on the depth the queued call itself reported: on a
     machine where one dispatch already costs a whole batch target the claim is vacuously true, and
     a vacuous [true] must not read like a verified one. *)
  let batches_here = que.depth > 1 in
  gated ~aggregation:`Environment ~on:(backend ())
    ~when_:(que.contended || iso.contended || batches_here)
    ~detail:(fun () ->
      Printf.sprintf "queued %d dispatches at depth %d vs isolated %d" que.dispatches que.depth
        iso.dispatches)
    "queued timing either reports contention or dispatches more launches than isolated"
    (que.contended || iso.contended || que.dispatches > iso.dispatches);
  (* Per launch, not per batch. The two sides refuse mirror errors: a reading that forgot to divide
     by the depth is about [depth] times the cost of a launch in that call, and a reading divided by
     the depth TWICE is that cost's [1/depth]. Both errors are claims about the same quantity the
     reading is -- the cost of one launch inside THIS call -- so both sides are written against the
     window the reading is a MINIMUM over, and against that window's MEDIAN batch, per launch.
     [Autotune.on_timed_window] reports the window: the batches the timed loop ran, their summed
     wall, and their median. A minimum cannot exceed the median of the same samples, so a correct
     reading satisfies the upper side by construction and a reading left per-batch overshoots it by
     [depth / 2]; the low side is below, and the depth from which either discriminates is derived
     further down from the same contention rule.

     The window, rather than the whole call the test clocks around: that wall also holds the warmup
     and the calibration's synchronized singles, and on a backend whose host round trip is two
     orders of magnitude above an amortized launch those few dozen singles are about 40% of the call
     -- so a whole-call mean is diluted by construction, and a host stall landing in the untimed
     part moves the anchor without moving the reading (Codex P2, round 1 on PR #735).

     The median, rather than that window's mean: [contended] declares the reading bypassed when a
     MAJORITY of the window's batches exceed twice its floor, so the regime these claims must
     survive is exactly the one a median is unmoved over, while one arbitrarily long batch among 64
     moves a mean without limit (Codex P2, round 2 on PR #735). The mean is still printed on stderr,
     since the dilution argument above is about it, and because the two together say how stalled a
     window was.

     And neither, rather than [floor_ms], where the low side used to be anchored at a fixed fraction
     (1/16) on the argument that both are minima and so face the same noise. They are -- but they
     are minima of DIFFERENT quantities: [floor_ms] is a launch plus the host synchronization that
     queueing exists to amortize away, and a queued reading is a launch without it. Their ratio is
     the backend's sync cost over its launch cost, which no fraction calibrated on one family of
     machines describes on another; three widenings of that divisor (gh-ocannl-839, -841, -851) were
     measuring it one machine at a time. gh-ocannl-994 measured it on purpose -- 534 runs of this
     instrument over five hosts, four backends and loads from idle to 6x oversubscription -- and
     found it spanning 0.0077 (idle multidev_cc on a 32-core Linux box: a 57 us worker-domain round
     trip against a 0.44 us amortized launch) to 1.2 (cc on the same boxes, where the round trip IS
     about one launch). That 160x spread is wider than the factor of [depth] the check has to
     resolve (200 on the cc, multidev_cc and Metal cap), so no constant fraction of [floor_ms] both
     admits every legitimate reading and refuses a twice-divided one -- the window is empty, not
     mis-centred. The 1/16 in force refused 23% of that sweep's non-contended readings, all of them
     multidev_cc under Linux (30 of 30 on one box), where the claim had been surviving on its
     contention hatch rather than on its envelope.

     [Isolated] is the one reading still compared to that round trip, and only on its low side: it
     IS that quantity by definition, so the two are one quantity's two implementations, and the
     factor of 3 stood at 1.8x clear of the worst measured ratio (0.61) across the same sweep. *)
  let timed_mean r = r.timed_wall_ms /. Float.of_int (max 1 (r.timed_batches * r.depth)) in
  let median_per_launch r = r.timed_median_ms /. Float.of_int (max 1 r.depth) in
  (* The seam's window is the one the reading summarizes: the loop counts its own batches, so an
     anchor taken from a window other than the one [samples] describes fails here rather than
     silently rescaling both sides of the envelope. *)
  p "both readings are anchored on the timed window they reported"
    (iso.timed_batches = iso.samples && que.timed_batches = que.samples);
  (* [Isolated] is compared to the round trip, which is measured independently and by hand off the
     same two primitives, so this is one quantity's two implementations against each other. One side
     only, and deliberately: at depth 1 a batch IS a launch, so a window-anchored upper side would
     be [min <= median] of the same samples -- a theorem, not a check -- while an upper side against
     the round trip compares two SEPARATELY sampled windows, and a uniformly delayed window is not
     [contended], so it fails valid readings on an oversubscribed host (Codex P2, round 4 on PR
     #735). The error such an upper side would refuse, a window summed or averaged instead of
     minimized, is refused on the injected clock above, where it needs no device and no second
     window. *)
  Verdict.pass_fail "isolated reading is a per-launch time or reports contention"
    (iso.contended || Float.(iso.ms >= floor_ms / 3.))
    ~detail:(fun () ->
      Printf.sprintf "%.6f ms vs round trip %.6f ms (median batch %.6f ms)" iso.ms floor_ms
        (median_per_launch iso));
  (* The depth from which either side of the queued envelope discriminates, derived rather than
     measured -- and it is a statement about the DEPTH, never about the reading. Round 2 gated the
     upper side on the error formed from the reading in hand, which decides whether to check using
     the very number in question: at depth 2 or 3 that ran the claim on exactly the per-batch
     reading it cannot refuse (Codex P2, round 3 on PR #735).

     The derivation is [sample_min]'s own contention rule, which is the condition under which these
     claims are made at all. It declares [contended] when at least half the window's batches exceed
     twice its minimum, so in a window these claims judge, FEWER than half do -- and the median,
     which sits at or below the middle of that majority at either parity, is therefore at most TWICE
     the minimum. Both sides follow from that bound. A per-batch reading IS the window's minimum, so
     the upper side refuses it when [minimum > 2 * median / depth], which [median <= 2 * minimum]
     turns into [depth > 4]. The low side admits a correct reading when [minimum >= max(1 / 64, 2 /
     depth) * median], which the same bound turns into [depth >= 4]. So each side takes the
     threshold its own arithmetic gives -- 5 and 4, not one shared 5, since a host settling on
     exactly depth 4 would otherwise skip a control provably safe and discriminating there (Codex
     P2, round 6 on PR #735) -- and both are structural on every window that is not already bypassed
     -- no constant from a sweep stands between a regression and this leg. Fleet-derived constants
     stood here for one round and were 32, which would have left depths 5 through 31 unchecked on
     slower backend/host pairs (Codex P2, round 4 on PR #735).

     The sweep below then reads as corroboration rather than as justification, and its own tail
     reads back: every window it measured at below half its median WAS a contended one, because the
     invariant leaves no other possibility. *)
  let upper_discriminating_depth = 5 and low_discriminating_depth = 4 in
  gated ~aggregation:`Environment ~on:(backend ())
    ~when_:(que.depth >= upper_discriminating_depth)
    ~detail:(fun () ->
      Printf.sprintf "%.6f ms vs twice the median per launch %.6f ms at depth %d" que.ms
        (2. *. median_per_launch que)
        que.depth)
    "queued reading is a per-launch time rather than a per-batch one, or reports contention"
    (que.contended || Float.(que.ms <= 2. * median_per_launch que));
  (* The low side, as the larger of two terms that refuse on different grounds. [2 / depth] is
     structural: the reading is a minimum over the batches this median is a middle of, so it cannot
     exceed that median, a twice-divided one cannot exceed [median / depth], and a bound at twice
     that refuses it at EVERY depth with nothing measured. The 64 of the other term is not a round
     number: it is [Autotune.max_timing_runs], the top-up cap the sample-count claims above pin, so
     at that cap -- where the budget lands for any routine fast enough to batch at all -- the same
     structural argument refuses a reading divided by the RUN count on top of the launch count --
     but only as far as the window's own spread reaches, since such a reading sits at the minimum
     over 64 and the bound at the median over 64. That spread was 1.35x at the sweep's median and as
     little as 1.01x, so THAT refusal is opportunistic where the batch-depth one is guaranteed, and
     the anchor is not traded back for it: a statistic with a longer tail above the minimum would
     refuse a run-count division more often and false-fail on the stalls this bound exists to
     survive. The term's own job is to keep the bound from going slack where [2 / depth] falls far
     below any real reading, at CUDA's depth 2048, and to refuse any reading below a 64th of its
     window's middle. What the constants have to clear is the spread between a window's minimum and
     that middle, which the contention rule caps at 2 for every window these claims judge: over 342
     runs of the load ladder that produced the numbers above, re-read against the median, the
     minimum stayed at 0.19 of its window's median at worst -- a window the invariant says was
     contended, and so bypassed -- and every reading cleared the bound -- contended ones included,
     since the sweep did not sort them out -- the tightest by 7.4x, Metal at 6x oversubscription on
     a 16-core host where a depth of 39 makes [2 / depth] the binding term. The same runs put the
     minimum at 0.064 of their window's MEAN, which is the tail the median is here to cut. *)
  let queued_low_bound =
    Float.max
      (median_per_launch que /. 64.)
      (2. *. median_per_launch que /. Float.of_int (max 1 que.depth))
  in
  Stdio.eprintf
    "  (not part of the golden) timed windows: isolated %d batches in %.3f ms wall (mean %.6f ms, \
     median batch %.6f ms); queued %d batches of %d in %.3f ms wall (mean %.6f ms, median batch \
     %.6f ms, min/mean %.4f, min/median %.4f); queued low bound %.6f ms\n\
     %!"
    iso.timed_batches iso.timed_wall_ms (timed_mean iso) iso.timed_median_ms que.timed_batches
    que.depth que.timed_wall_ms (timed_mean que) que.timed_median_ms
    (que.ms /. timed_mean que)
    (que.ms /. median_per_launch que)
    queued_low_bound;
  (* The gated claims report through [gated] rather than [Verdict.pass_fail]: a skip prints
     [<claim>: true], so only [p]'s dialect keeps a legitimate skip's stdout identical to an
     evaluated pass and the golden intact (Codex P2, round 1 on PR #735; gh-ocannl-997 made
     [Verdict.gated] pick that dialect itself). The numbers a calibration run wants are on the
     stderr line above, on a passing run too; a failure also names its own in the claim's detail.

     The low side's own threshold, one shallower than the upper side's: at depth 4 the bound is half
     the window's median, which [median <= 2 * minimum] puts at or below a correct reading, while a
     twice-divided one -- a quarter of the minimum -- falls under it. Below that, a double division
     stops being separable from the window's ordinary spread and the leg says so rather than passing
     vacuously. *)
  gated ~aggregation:`Environment ~on:(backend ())
    ~when_:(que.depth >= low_discriminating_depth)
    ~detail:(fun () ->
      Printf.sprintf "%.6f ms vs low bound %.6f ms at depth %d" que.ms queued_low_bound que.depth)
    "queued reading is not that per-launch time divided by the batch depth as well, or reports \
     contention"
    (que.contended || Float.(que.ms >= queued_low_bound));
  (* Amortizing a round trip can only remove time, so a queued reading above the isolated one is the
     instrument reporting the wrong quantity, not a slow machine. The factor absorbs the noise a
     min-of-N leaves; the point of the claim is the direction.

     Both sides are minima, and a minimum only means what it says if one of the samples under it was
     taken while the host was not stalling. One reading each cannot promise that: the two calls
     occupy DISJOINT windows, so a burst landing in the queued one inverts the direction with
     neither reading wrong -- 6 of 28 runs at 3-4x oversubscription. Normalizing each reading by the
     round trip measured beside it does not repair it (the worst inversions survive at 99x, 3.5x and
     2.7x), because the stall is inside the batch rather than in the anchor. What does repair it is
     giving each mode more than one window: each is read twice and the claim is made on the min of
     its two, with the four calls ordered as a palindrome -- isolated, queued, queued, isolated --
     so that the two modes have the SAME mean sample time and no ordering bias is left for the
     direction to inherit. The other claims stay on the first round: they are about one call's own
     decomposition, not about a quantity two calls can be minimized over. *)
  let iso_min = Float.min iso.ms iso2.ms and que_min = Float.min que.ms que2.ms in
  Verdict.pass_fail "queued timing does not read above isolated timing or reports contention"
    (iso.contended || iso2.contended || que.contended || que2.contended
    || Float.(que_min <= iso_min * 2.))
    ~detail:(fun () ->
      Printf.sprintf "queued %.6f ms (%.6f, %.6f) vs isolated %.6f ms (%.6f, %.6f)" que_min que.ms
        que2.ms iso_min iso.ms iso2.ms);
  (* The old wall-budget claim is intentionally gone: accumulating per-launch samples means a fast
     queued routine can run all 64 batches. The pure injected-clock claims above pin the budget;
     this executed leg pins only the absolute timed-batch dispatch cap, after the separately
     accounted calibration work. *)
  p "queued timing either reports contention or stays within the 64-batch dispatch cap"
    (que.contended || que.dispatches <= 1 + que.calibration_dispatches + (64 * que.depth))

(* {1 The objective is part of the cache identity} *)

(* gh-ocannl-755, Codex P1 on PR #512: the two objectives crown different candidates, so an entry
   stored under one is not the answer to a search asking the other -- and its stored times are
   readings of a different quantity that a replay would copy into the reading process's report under
   that process's label. Keying on the objective is what keeps the regimes apart, and it is also
   what stops a warm cache from replaying isolated-crowned winners forever and defeating the new
   default outright. Pinned at the key rather than by driving a search: the key is the mechanism --
   a mismatched entry lives in a different file and is never looked up at all. *)
let () =
  Stdio.printf "\n== the objective in the cache key ==\n";
  let opt, ctx = Option.value_exn !measured in
  let canon = SC.canonicalize ~static_indices:[] opt in
  let limits = Context.hardware_limits ctx and backend = Context.backend_name ctx in
  let capabilities = Context.codegen_capabilities ctx in
  let key objective =
    Option.value_exn (SC.cache_key ~timing_identity ~objective ~limits ~capabilities canon ~backend)
  in
  p "the cache key is stable within one objective" (String.equal (key "queued") (key "queued"));
  p "the cache key separates the two timing objectives"
    (not (String.equal (key "isolated") (key "queued")));
  Verdict.p_all "CUDA and HIP queued keys carry the new timing-policy generation" [ "cuda"; "hip" ]
    ~f:(fun backend ->
      String.is_suffix
        (Option.value_exn
           (SC.cache_key ~timing_identity ~objective:"queued" ~limits ~capabilities canon ~backend))
        ~suffix:"-tqueued-v2");
  Verdict.p_all "cc and Metal queued keys retain their unchanged timing generation"
    [ "cc"; "multidev_cc"; "metal" ] ~f:(fun backend ->
      String.is_suffix
        (Option.value_exn
           (SC.cache_key ~timing_identity ~objective:"queued" ~limits ~capabilities canon ~backend))
        ~suffix:"-tqueued");
  (* Derived, not restated: a caller that resolved no mode of its own must key exactly as one that
     resolved the configured mode, or a test's hand-built entry would sit under a key no search
     looks up. *)
  p "an omitted objective keys as the configured one"
    (String.equal
       (Option.value_exn (SC.cache_key ~timing_identity ~limits ~capabilities canon ~backend))
       (key (SC.objective_tag ())));
  (* The tag a key carries is the mode's own spelling, so a report's objective and the entry that
     stored its times name the same thing. *)
  Verdict.p_all "the key's objective spelling round-trips through the mode"
    [ Autotune.Isolated; Autotune.Queued ] ~f:(fun m ->
      Poly.equal (Autotune.timing_of_setting (Autotune.timing_string m)) m)
