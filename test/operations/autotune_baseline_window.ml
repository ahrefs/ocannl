(* gh-ocannl-1666: the serial baseline's timing window is bounded.

   Every window takes at least sixteen samples, a floor set for sub-millisecond kernels whose
   min-of-3 is launch jitter. On cc a seconds-long serial baseline paid it in full -- sixteen
   synchronized singles of a [gpt2_mini_train] b8 step plus the warmup, 56-80 s per placement arm.
   [Autotune.window_bounded] ends a window, ADMITTED, once it holds three or more samples that are
   not dispersed and sum to at least the bound; [Autotune.tune] passes it, at
   [Autotune.baseline_window_ms], to the baseline's window alone.

   Every leg runs on a scripted clock: the rule, pure; the sampling loop it bounds; and executed cc
   searches through [Autotune.on_batch_wall], under both objectives. Each leg pins both branches: a
   window the bound ends early, and one under it that keeps the floor. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let sample ms : Autotune.timing_sample = { per_launch_ms = ms; contention_ms = ms }
let samples n ms = List.init n ~f:(fun _ -> sample ms)
let bound_ms = Autotune.baseline_window_ms

(* A sample a third of the bound: three of them reach it exactly. *)
let slow_ms = bound_ms /. 3.

(* A sample whose sixteen-sample floor stays under the bound. *)
let fast_ms = bound_ms /. 64.

(* --- The rule, pure. --- *)

let () =
  Stdio.printf "== the rule, pure ==\n";
  let bounded ?at_least_ms window = Autotune.window_bounded ?at_least_ms ~bound_ms window in
  p "the bound is reached at three samples of a third of it, not at two"
    (bounded (samples 3 slow_ms) && not (bounded (samples 2 (2. *. slow_ms))));
  p "a window just short of the bound is not ended" (not (bounded (samples 3 (slow_ms *. 0.99))));
  p "a sixteen-sample window under the bound is not ended" (not (bounded (samples 16 fast_ms)));
  (* One clean sample and two stalled at over twice it: two of three past the stall ratio. *)
  p "a dispersed window is not ended, however long"
    ((Autotune.window_result [ sample slow_ms; sample (3. *. slow_ms); sample (3. *. slow_ms) ])
       .dispersed
    && not (bounded [ sample slow_ms; sample (3. *. slow_ms); sample (3. *. slow_ms) ]));
  p "the gate keeps a minimum under it from ending the window, and admits one at it"
    (bounded (samples 3 slow_ms)
    && (not (bounded ~at_least_ms:(slow_ms +. 1.) (samples 3 slow_ms)))
    && bounded ~at_least_ms:slow_ms (samples 3 slow_ms));
  p_all "no finite positive minimum, no early end" [ 0.; Float.nan; Float.infinity ] ~f:(fun ms ->
      not (bounded (samples 3 ms)))

(* --- The sampling loop it bounds. --- *)

let window ?bounded ?prior ~repeats ms =
  let taken = ref 0 in
  let w =
    Autotune.sample_window ?prior ?bounded ~repeats
      ~sample:(fun () ->
        Int.incr taken;
        sample ms)
      ()
  in
  (List.length w, !taken)

let () =
  Stdio.printf "== the sampling loop ==\n";
  let bounded = Autotune.window_bounded ~bound_ms in
  p "a slow window ends at three samples under the bound"
    (Poly.equal (window ~bounded ~repeats:1 slow_ms) (3, 3));
  p "... and takes its whole sixteen-sample floor without it"
    (Poly.equal (window ~repeats:1 slow_ms) (16, 16));
  p "a fast window keeps its floor under the bound"
    (Poly.equal (window ~bounded ~repeats:1 fast_ms) (window ~repeats:1 fast_ms)
    && fst (window ~repeats:1 fast_ms) >= 16);
  p "the caller's repeats still wins: five repeats take five slow samples"
    (Poly.equal (window ~bounded ~repeats:5 slow_ms) (5, 5));
  p "a resumed window that already meets the bound takes no fresh sample"
    (Poly.equal (window ~bounded ~prior:(samples 3 slow_ms) ~repeats:3 slow_ms) (3, 0))

(* --- Executed cc searches on the scripted clock. --- *)

type attempt = { label : string; mutable batches : int }

let search ~timing ~baseline_ms comp =
  let report = ref None in
  let attempts = ref [] in
  let on_attempt label = attempts := { label; batches = 0 } :: !attempts in
  (* The attempt in flight reads [baseline_ms] if it is the baseline, [fast_ms] otherwise: every
     candidate reads alike, so no reference cuts a window as hopeless. *)
  let clock ~depth ~ordinal:_ _wall =
    match !attempts with
    | a :: _ ->
        a.batches <- a.batches + 1;
        (if String.equal a.label "baseline" then baseline_ms else fast_ms) *. Float.of_int depth
    | [] -> fast_ms *. Float.of_int depth
  in
  let ctx = Context.cpu () in
  let ctx =
    Autotune.Seam.with_value Autotune.on_candidate_attempt on_attempt (fun () ->
        Autotune.Seam.with_value Autotune.on_batch_wall clock (fun () ->
            fst
              (Autotune.tune ~search:true ~rounds:0 ~repeats:1 ~timing ~cache_dir:""
                 ~report:(fun r -> report := Some r)
                 ctx comp Ir.Indexing.Empty)))
  in
  Context.release ctx;
  let r = Option.value_exn ~here:[%here] !report in
  let attempts = List.rev !attempts in
  let tag = Printf.sprintf "%s, baseline at %g ms" (Autotune.timing_string timing) baseline_ms in
  List.iter attempts ~f:(fun a ->
      Stdio.eprintf "%s (not part of the golden): %s, %d batches\n%!" tag a.label a.batches);
  Stdio.eprintf "%s (not part of the golden): %s, timed %d, baseline %g ms, best %g ms\n%!" tag
    (Autotune.outcome_name r.outcome) r.candidates_timed r.baseline_ms r.best_ms;
  (r, attempts)

let () =
  Stdio.printf "== executed cc searches ==\n";
  let av =
    Array.init (64 * 128)
      ~f:(Ll_test.cycle_flat ~dims:[| 64; 128 |] ~modulus:7 ~offset:(-3.) ~stride:1.)
  in
  let a = TDSL.ndarray av ~label:[ "bw_a" ] ~output_dims:[ 64; 128 ] () in
  let%op y = (a *. a) + a ++ "ij => i" in
  let comp = Train.forward y in
  let baseline_window attempts =
    match attempts with { label = "baseline"; batches } :: _ -> Some batches | _ -> None
  in
  let others attempts =
    List.filter attempts ~f:(fun a -> (not (String.equal a.label "baseline")) && a.batches > 0)
  in
  List.iter [ Autotune.Isolated; Autotune.Queued ] ~f:(fun timing ->
      let name = Autotune.timing_string timing in
      let r, attempts = search ~timing ~baseline_ms:slow_ms comp in
      p
        (name ^ ": a slow baseline's window ended at three samples")
        (Option.equal Int.equal (baseline_window attempts) (Some Autotune.hopeless_min_samples));
      p
        (name ^ ": ... admitted and ranked at its reading")
        (Float.equal r.Autotune.baseline_ms slow_ms
        && Poly.equal r.Autotune.outcome Autotune.Searched);
      p_all ~min:1 (name ^ ": every other window, timed beside it, took its whole floor")
        (others attempts) ~f:(fun a -> a.batches >= 16);
      let r, attempts = search ~timing ~baseline_ms:fast_ms comp in
      p
        (name ^ ": a fast baseline's window took its whole floor")
        (Option.exists (baseline_window attempts) ~f:(fun n -> n >= 16)
        && Float.equal r.Autotune.baseline_ms fast_ms))

let () = Stdio.printf "autotune_baseline_window: done\n"
