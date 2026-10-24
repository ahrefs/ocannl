(* gh-ocannl-1527: a timing window ends early once it proves its candidate out of the running.

   A candidate far slower than what its reading will be ranked against used to take its whole window
   anyway: at least sixteen samples, and on CUDA/HIP a depth-2 calibration probe besides (arm B of
   gh-ocannl-1314: ~4.4 s per candidate on tuf). [Autotune.hopeless_window] cuts a window once its
   first three or more samples have a minimum over four times the reference and are not dispersed;
   [Autotune.time_routine ?hopeless_against] applies it to the calibration's singles (gated at the
   batch target, where they are the reading) and to every timed window; and the search passes a
   reference only where the reading feeds a ranking that cannot be moved by a reading that far over
   it -- a whole-routine seed against the beam, a round candidate against its round -- and releases
   a cut candidate unranked.

   Every leg runs on a scripted clock: the pure rule, one timing call per objective shape through
   [calibrate_and_time_with_retry_observer], and an executed cc search through
   [Autotune.on_batch_wall]. Each leg pins a cut window AND a near window that is not cut. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let sample ms : Autotune.timing_sample = { per_launch_ms = ms; contention_ms = ms }
let samples n ms = List.init n ~f:(fun _ -> sample ms)

(* --- The rule, pure. --- *)

let () =
  Stdio.printf "== the rule, pure ==\n";
  let cut ?at_least_ms ~reference_ms window =
    Autotune.hopeless_window ?at_least_ms ~reference_ms window
  in
  p "the rule cuts at three samples, over four times the reference"
    (Autotune.hopeless_min_samples = 3 && Float.equal Autotune.hopeless_ratio 4.);
  p "a steady window five times the reference is cut once it has three samples"
    (Option.is_none (cut ~reference_ms:12. (samples 2 60.))
    &&
    match cut ~reference_ms:12. (samples 3 60.) with
    | Some h ->
        Float.equal h.hopeless_ms 60.
        && Float.equal h.hopeless_threshold_ms 48.
        && h.hopeless_samples = 3
    | None -> false);
  p_all "a window one and a half times the reference is never cut" [ 3; 16; 64 ] ~f:(fun n ->
      Option.is_none (cut ~reference_ms:12. (samples n 18.)));
  p "a window exactly at four times the reference is not cut: the bound is strict"
    (Option.is_none (cut ~reference_ms:12. (samples 3 48.)));
  (* One clean 60 ms sample and two stalled at 200 ms: two of three past twice the floor. *)
  p "a dispersed window is not cut, however slow its minimum"
    (Autotune.window_result [ sample 60.; sample 200.; sample 200. ]).dispersed;
  p "... and the rule leaves it alone"
    (Option.is_none (cut ~reference_ms:1. [ sample 60.; sample 200.; sample 200. ]));
  p "one stall among clean samples moves neither the minimum nor the verdict"
    (match cut ~reference_ms:12. [ sample 60.; sample 500.; sample 60. ] with
    | Some h -> Float.equal h.hopeless_ms 60.
    | None -> false);
  p "the gate keeps a minimum under it from cutting, and admits one at it"
    (Option.is_none (cut ~at_least_ms:10. ~reference_ms:1. (samples 3 8.))
    && Option.is_some (cut ~reference_ms:1. (samples 3 8.))
    && Option.is_some (cut ~at_least_ms:10. ~reference_ms:1. (samples 3 10.)));
  p_all "no finite positive reference, no cut" [ Float.infinity; 0.; -1.; Float.nan ]
    ~f:(fun reference_ms -> Option.is_none (cut ~reference_ms (samples 16 1e6)))

(* --- One timing call on a scripted clock. --- *)

(* A device whose every launch costs [per_launch_ms]: [batch depth] reads [depth] times that. *)
let call ?hopeless_against ~timing ~cap ~per_launch_ms () =
  let batches = ref [] in
  let batch depth =
    batches := depth :: !batches;
    per_launch_ms *. Float.of_int depth
  in
  let outcome =
    match
      Autotune.calibrate_and_time_with_retry_observer ~hopeless_against
        ~on_retry:(fun () -> ())
        ~on_call:(fun _ -> ())
        ~phase:None ~retry_contended:false ~dispersion:Autotune.Refuse_dispersed ~timing ~repeats:3
        ~queue_depth_cap:cap ~batch
    with
    | r -> `Timed r
    | exception Autotune.Timing_hopeless h -> `Hopeless h
  in
  (outcome, List.rev !batches)

let cc_cap = Autotune.queue_depth_cap_for_backend "cc"
let hip_cap = Autotune.queue_depth_cap_for_backend "hip"

let () =
  Stdio.printf "== one timing call ==\n";
  let shapes =
    [
      ("isolated", Autotune.Isolated, cc_cap);
      ("queued, cc calibration", Autotune.Queued, cc_cap);
      ("queued, CUDA/HIP calibration", Autotune.Queued, hip_cap);
    ]
  in
  List.iter shapes ~f:(fun (name, timing, cap) ->
      (* A routine slower than the batch target: depth 1, where the singles are the reading. *)
      let slow, slow_batches = call ~hopeless_against:12. ~timing ~cap ~per_launch_ms:60. () in
      let near, near_batches = call ~hopeless_against:12. ~timing ~cap ~per_launch_ms:18. () in
      let full, full_batches = call ~timing ~cap ~per_launch_ms:60. () in
      Stdio.eprintf "%s (not part of the golden): cut after %d batches, near %d, uncut %d\n%!" name
        (List.length slow_batches) (List.length near_batches) (List.length full_batches);
      p
        (name ^ ": five times the reference is cut after three single launches")
        ((match slow with `Hopeless h -> Float.equal h.hopeless_ms 60. | `Timed _ -> false)
        && List.equal Int.equal slow_batches [ 1; 1; 1 ]);
      (* The whole window: the call with no reference, at least the sixteen-sample floor. *)
      p
        (name ^ ": with no reference the same routine takes its whole window")
        ((match full with `Timed _ -> true | `Hopeless _ -> false)
        && List.length full_batches >= 16);
      p
        (name ^ ": one and a half times the reference takes exactly that whole window")
        ((match near with
           | `Timed r -> Option.equal Float.equal (Autotune.admitted_timing_ms r) (Some 18.)
           | `Hopeless _ -> false)
        && List.equal Int.equal near_batches (snd (call ~timing ~cap ~per_launch_ms:18. ()))));
  (* A fast routine on cc: its singles (2 ms) are under the 10 ms target, so they owe a batch and
     are not the reading -- the gate keeps them from cutting, however far over the reference -- and
     the timed window at the calibrated depth is cut after three batches. *)
  let fast, fast_batches =
    call ~hopeless_against:0.1 ~timing:Autotune.Queued ~cap:cc_cap ~per_launch_ms:2. ()
  in
  let singles, timed = List.partition_tf fast_batches ~f:(fun d -> d = 1) in
  Stdio.eprintf "fast queued (not part of the golden): %d singles, timed batches at depths %s\n%!"
    (List.length singles)
    (String.concat ~sep:"," (List.map timed ~f:Int.to_string));
  let uncut_singles =
    List.count
      (snd (call ~timing:Autotune.Queued ~cap:cc_cap ~per_launch_ms:2. ()))
      ~f:(fun d -> d = 1)
  in
  p "fast queued: the singles under the batch target are all taken"
    (List.length singles = uncut_singles && uncut_singles >= 16);
  p "fast queued: the timed window at its depth is cut after three batches"
    ((match fast with `Hopeless h -> Float.equal h.hopeless_ms 2. | `Timed _ -> false)
    && List.length timed = 3);
  p_all "fast queued: ... each of them a batch deeper than one" timed ~f:(fun d -> d > 1)

(* --- An executed cc search on the scripted clock. --- *)

module SC = Ir.Schedule_cache
module Cache_plant = Test_utils.Cache_plant

(* What the scripted clock reads for the call in flight, decided at its first batch so that only
   attempts reaching a window take a place in the cycle: the dispatched baseline and the floors read
   [floor_ms]; every other attempt cycles through [hopeless_ms] (over four times the floor and over
   four times [near_ms] too), [mid_ms] (over four times the floor, under four times [near_ms]) and
   [near_ms]. *)
let floor_ms = 12.
let hopeless_ms = 100.
let mid_ms = 60.
let near_ms = 18.

type attempt = { label : string; mutable reads : float option; mutable batches : int }

(* A per-segment single, by its label: ranked among its key's singles, never cut. *)
let is_single a =
  String.is_prefix a.label ~prefix:"F_split[" || String.is_prefix a.label ~prefix:"F_sketch["

let is_whole_seed a =
  String.is_prefix a.label ~prefix:"F_preset[" || String.is_prefix a.label ~prefix:"W_preset["

let cache_dir = "autotune_cache_hopeless_window"

let search ~rounds comp =
  let report = ref None in
  let attempts = ref [] in
  let floor_label = Autotune.default_seed_label ~backend_name:"cc" in
  let is_floor label =
    String.equal label "baseline"
    || Option.exists floor_label ~f:(String.equal label)
    || String.is_prefix label ~prefix:"F_seeded"
  in
  let cycled = ref 0 in
  let on_attempt label = attempts := { label; reads = None; batches = 0 } :: !attempts in
  let clock ~depth ~ordinal:_ _wall =
    match !attempts with
    | a :: _ ->
        let reads =
          match a.reads with
          | Some ms -> ms
          | None ->
              let ms =
                if is_floor a.label then floor_ms
                else (
                  Int.incr cycled;
                  List.nth_exn [ hopeless_ms; mid_ms; near_ms ] ((!cycled - 1) % 3))
              in
              a.reads <- Some ms;
              ms
        in
        a.batches <- a.batches + 1;
        reads *. Float.of_int depth
    | [] -> floor_ms *. Float.of_int depth
  in
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let ctx = Context.cpu () in
  let identity = Option.is_some (Context.timing_identity ctx) in
  let ctx, ios =
    SC.recording_cache_io (fun () ->
        Autotune.Seam.with_value Autotune.on_candidate_attempt on_attempt (fun () ->
            Autotune.Seam.with_value Autotune.on_batch_wall clock (fun () ->
                fst
                  (Autotune.tune ~search:true ~rounds ~repeats:1 ~timing:Autotune.Isolated
                     ~cache_dir
                     ~report:(fun r -> report := Some r)
                     ctx comp Ir.Indexing.Empty))))
  in
  Context.release ctx;
  let r = Option.value_exn ~here:[%here] !report in
  let attempts = List.rev !attempts in
  List.iter attempts ~f:(fun a ->
      Stdio.eprintf "rounds=%d (not part of the golden): %s reads %s, %d batches\n%!" rounds a.label
        (Option.value_map a.reads ~default:"nothing" ~f:(Printf.sprintf "%g ms"))
        a.batches);
  Stdio.eprintf
    "rounds=%d (not part of the golden): %s, timed %d, hopeless %d, contended %d, best %g ms\n%!"
    rounds (Autotune.outcome_name r.outcome) r.candidates_timed r.timings_hopeless
    r.timings_contended r.best_ms;
  let timed = List.filter attempts ~f:(fun a -> a.batches > 0) in
  let cut = List.filter timed ~f:(fun a -> a.batches < 16) in
  let reading ms a = Option.equal Float.equal a.reads (Some ms) in
  let tag = Printf.sprintf "rounds=%d: " rounds in
  p_all (tag ^ "every window ran in full or was cut after three samples") timed ~f:(fun a ->
      a.batches >= 16 || a.batches = Autotune.hopeless_min_samples);
  p (tag ^ "the report counts the cut windows") (r.timings_hopeless = List.length cut);
  (* The baseline reads the floor, so this population is never empty. *)
  p_all
    (tag ^ "every floor and near window ran in full")
    (List.filter timed ~f:(fun a -> reading floor_ms a || reading near_ms a))
    ~f:(fun a -> a.batches >= 16);
  p_all (tag ^ "every per-segment single ran in full, however slow")
    (List.filter timed ~f:is_single) ~f:(fun a -> a.batches >= 16);
  p_exists (tag ^ "a single slower than four times the floor was timed") timed ~f:(fun a ->
      is_single a && Float.(Option.value a.reads ~default:0. > 4. *. floor_ms));
  p
    (tag ^ "the search crowned at the floor, refusing nothing")
    (Poly.equal r.outcome Autotune.Searched
    && r.timings_contended = 0 && Float.equal r.best_ms floor_ms);
  Cache_plant.unless_unkeyed_or_refused ~identity ios
    (tag ^ "a search with cut windows still commits its schedule entry")
    (List.exists ios ~f:(fun (io : SC.cache_io) ->
         (match io.op with SC.Store -> true | SC.Lookup -> false)
         && String.equal (Cache_plant.store_kind io) "schedule"
         && not (Cache_plant.refused io)));
  (timed, cut, reading)

let () =
  Stdio.printf "== an executed cc search ==\n";
  let av =
    Array.init (64 * 128)
      ~f:(Ll_test.cycle_flat ~dims:[| 64; 128 |] ~modulus:7 ~offset:(-3.) ~stride:1.)
  in
  let a = TDSL.ndarray av ~label:[ "hw_a" ] ~output_dims:[ 64; 128 ] () in
  let%op y = (a *. a) + a ++ "ij => i" in
  let comp = Train.forward y in
  (* No rounds: only the beam's head is ever read, so a whole-routine seed over four times it -- the
     baseline, at the floor -- is cut. *)
  let timed, cut, reading = search ~rounds:0 comp in
  p_exists "rounds=0: a whole-routine seed over four times the head was cut" cut ~f:is_whole_seed;
  p_all "rounds=0: ... and every whole-routine seed over four times the head was"
    (List.filter timed ~f:(fun a -> is_whole_seed a && (reading hopeless_ms a || reading mid_ms a)))
    ~f:(fun a -> List.mem cut a ~equal:phys_equal);
  (* With a round to run, every beam entry is expanded: the reference is the beam's slowest entry,
     and only once it is full -- so the first whole-routine seed after the baseline, however slow,
     takes its place in the beam in full. *)
  let timed, _, reading = search ~rounds:1 comp in
  p "rounds=1: the first whole-routine seed timed is the hopeless one, and it ran in full"
    (match List.find timed ~f:is_whole_seed with
    | Some a -> reading hopeless_ms a && a.batches >= 16
    | None -> false)
