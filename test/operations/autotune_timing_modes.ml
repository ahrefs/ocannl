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
  p "a refused positive timing cannot enter ranking or calibration"
    (Option.is_none (Autotune.admitted_timing_ms { ms = 0.0002; contended = true; samples = 16 }));
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
      let got = Autotune.queued_batch_depth { ms = est_ms; contended = false; samples = 0 } in
      if got <> want then
        Stdio.eprintf "  %s: est %g ms -> depth %d, expected %d\n%!" what est_ms got want;
      got = want);
  Verdict.p_all "every degenerate calibration estimate is refused by ranking but still batched"
    degenerate_depth_cases ~f:(fun (what, est_ms, want) ->
      let result : Autotune.timing_result = { ms = est_ms; contended = false; samples = 16 } in
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
      Autotune.queued_batch_depth { ms = est_ms; contended = false; samples = 0 } >= 1);
  Verdict.p_all "no calibration estimate ever yields a depth above the cap" all_depth_cases
    ~f:(fun (_, est_ms, _) ->
      Autotune.queued_batch_depth { ms = est_ms; contended = false; samples = 0 } <= 2048);
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
  let measure timing =
    let before = count () in
    let c0 = Mtime_clock.counter () in
    let result = Autotune.time_routine ~repeats:3 ~timing ctx routine in
    let wall_ms = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
    {
      ms = result.ms;
      contended = result.contended;
      samples = result.samples;
      wall_ms;
      dispatches = count () - before;
      depth = !depth_seen;
      calibration_dispatches = !calibration_dispatches_seen;
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
  Stdio.eprintf
    "  (not part of the golden) isolated %.6f ms over %d dispatches in %.1f ms wall; queued %.6f \
     ms over %d dispatches (batch depth %d) in %.1f ms wall; second round isolated %.6f ms, queued \
     %.6f ms (batch depth %d); round trip %.6f ms (%.6f/%.6f/%.6f)\n\
     %!"
    iso.ms iso.dispatches iso.wall_ms que.ms que.dispatches que.depth que.wall_ms iso2.ms que2.ms
    que2.depth floor_ms before between after;
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
     the one it reported fails this on any count the reported depth does not divide. *)
  p "queued timing either refuses contention or dispatches whole batches at the reported depth"
    (que.contended
    || que.depth >= 1 && que.calibration_dispatches >= 16 && que.samples >= 16 && que.samples <= 64
       && que.dispatches = 1 + que.calibration_dispatches + (que.samples * que.depth));
  (* Depth > 1 is what queued mode IS. Gated on the depth the queued call itself reported: on a
     machine where one dispatch already costs a whole batch target the claim is vacuously true, and
     a vacuous [true] must not read like a verified one. *)
  let batches_here = que.depth > 1 in
  let claim = "queued timing either reports contention or dispatches more launches than isolated" in
  if que.contended || iso.contended || batches_here then
    p claim (que.contended || iso.contended || que.dispatches > iso.dispatches)
  else Verdict.skipped ~aggregation:`Environment ~backend:(backend ()) claim;
  (* Per launch, not per batch. The two sides refuse mirror errors, and they are anchored on
     different quantities because of it.

     The upper side refuses a reading that forgot to divide by the depth: such a reading is about
     [depth] times the mean cost of a launch in that call -- up to 2048x -- which the whole call's
     wall mean bounds with a factor of 3. Contention only inflates that mean, so a busy runner makes
     this side stricter rather than looser, and it stays written against it.

     The low side refuses the reading divided by the depth TWICE, and it cannot use that mean,
     because the mean is exactly where contention lands. [ms] is a MINIMUM over the call's batches;
     the mean is the call's average with the warmup, the calibration and every host stall folded in.
     On a busy runner the two part company without either being wrong -- gh-ocannl-851 widened this
     divisor for the 4.3x gap CI showed, and the HIP sweep then produced 22x (a 0.342 ms launch
     against a 7.48 ms mean, the call cut to 6 dispatches by stalls). No fraction of a mean survives
     that, so the fix is not a wider fraction: it is an anchor that is a minimum too, [floor_ms], so
     that both sides of the comparison face the same noise.

     [Isolated] IS that round trip, so its reading and the anchor are two minima of the SAME
     quantity: the factor of 3 is a bound rather than an envelope, and from an idle box to one at 6x
     oversubscription the measured ratio stayed above 0.86. What it refuses is the low-side error a
     depth-1 mode can still make -- a reading divided by the run count on top of the launch count.
     [Queued] amortizes the round trip away, so its reading sits legitimately BELOW the anchor (6x
     below on an idle HIP box, 0.16 of it at worst across the same sweep of loads), and its divisor
     is 16: clear of that worst legitimate ratio by 2.6x, while a twice-divided reading, sitting at
     [1/depth] of a correct one, is refused across the deep-batch regime the divisor exists for. At
     small depths a double division is a small under-read inside the noise floor and out of this
     instrument's reach, as it already was under the pre-widening factor of 3. *)
  let mean r = r.wall_ms /. Float.of_int (max 1 r.dispatches) in
  let per_launch ~low_div r = Float.(r.ms <= 3. * mean r && r.ms >= floor_ms / low_div) in
  Verdict.pass_fail "isolated reading is a per-launch time or reports contention"
    (iso.contended || per_launch ~low_div:3. iso)
    ~detail:(fun () ->
      Printf.sprintf "%.6f ms vs mean %.6f ms, round trip %.6f ms" iso.ms (mean iso) floor_ms);
  Verdict.pass_fail "queued reading is a per-launch time or reports contention"
    (que.contended || per_launch ~low_div:16. que)
    ~detail:(fun () ->
      Printf.sprintf "%.6f ms vs mean %.6f ms, round trip %.6f ms (depth %d)" que.ms (mean que)
        floor_ms que.depth);
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
  let key objective = SC.cache_key ~objective ~limits canon ~backend in
  p "the cache key is stable within one objective" (String.equal (key "queued") (key "queued"));
  p "the cache key separates the two timing objectives"
    (not (String.equal (key "isolated") (key "queued")));
  Verdict.p_all "CUDA and HIP queued keys carry the new timing-policy generation" [ "cuda"; "hip" ]
    ~f:(fun backend ->
      String.is_suffix
        (SC.cache_key ~objective:"queued" ~limits canon ~backend)
        ~suffix:"-tqueued-v2");
  Verdict.p_all "cc and Metal queued keys retain their unchanged timing generation"
    [ "cc"; "multidev_cc"; "metal" ] ~f:(fun backend ->
      String.is_suffix (SC.cache_key ~objective:"queued" ~limits canon ~backend) ~suffix:"-tqueued");
  (* Derived, not restated: a caller that resolved no mode of its own must key exactly as one that
     resolved the configured mode, or a test's hand-built entry would sit under a key no search
     looks up. *)
  p "an omitted objective keys as the configured one"
    (String.equal (SC.cache_key ~limits canon ~backend) (key (SC.objective_tag ())));
  (* The tag a key carries is the mode's own spelling, so a report's objective and the entry that
     stored its times name the same thing. *)
  Verdict.p_all "the key's objective spelling round-trips through the mode"
    [ Autotune.Isolated; Autotune.Queued ] ~f:(fun m ->
      Poly.equal (Autotune.timing_of_setting (Autotune.timing_string m)) m)
