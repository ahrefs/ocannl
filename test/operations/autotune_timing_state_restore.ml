(* gh-ocannl-1621: every timing window of a search starts from the training state its lineage held
   before the search, and the search leaves that state as it found it.

   Every timed run of a training step is a training step: before the fix, each candidate was timed
   on the parameters and optimizer state the previous candidates' runs had trained, so a candidate
   timed late in a search ran on other values than one timed early (gh-ocannl-1557's residual), and
   the lineage ended the search trained by it -- the caller's own context, when the search had no
   [timing_ctx].

   The step is an SGD step with momentum over a matmul, so the search seeds several candidates and
   the lineage carries two kinds of state: the parameter [w] and the momentum buffer, warmed by one
   step before the search so that the lineage holds it. Every value is exact by construction: [x]
   holds small integers, [w] eighths, the learning rate and the momentum are powers of two, and the
   gradient of the summed product is a sum of entries of [x]. Two runs from the same state therefore
   agree bitwise whatever schedule computed them, which is what lets the claims compare a step run
   by the shipped routine with one run by an untuned reference.

   The measurements are scripted (the k-th admitted window measures k ms) and the windows opted out
   of host contention, as the other scripted-measurement searches do, so the windows the claims
   observe are there on any host. What the claims read is the lineage's state, never a time. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module IDX = Train.IDX

exception Stop_search

let n = 16

let () =
  Tensor.unsafe_reinitialize ();
  let x =
    TDSL.ndarray
      (Array.init (n * n)
         ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:(-3.) ~stride:1.))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let w_init =
    Array.init (n * n)
      ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:9 ~offset:(-4.) ~stride:0.125)
  in
  let w = TDSL.param ~values:w_init "w" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op y = w * x in
  let%op loss = y ++ "...|...->... => 0" in
  let%op learning_rate = 0.0078125 in
  Train.set_materialized w.Tensor.value;
  Train.set_materialized learning_rate.Tensor.value;
  let comp =
    Ir.Assignments.sequence
      [ Train.grad_update loss; Train.sgd_update ~learning_rate ~momentum:0.5 loss ]
  in
  (* A lineage holding the trained state: [w] from its initialization, the momentum buffer from one
     warm-up step (an embedded node, allocated by the first compile that sees it). *)
  let warm_lineage () =
    let ctx = Train.init_params (Context.auto ()) IDX.empty loss in
    let ctx, routine = Context.compile ctx comp IDX.empty in
    Context.run ctx routine
  in
  let read ctx = Context.get_values ctx w.Tensor.value in
  (* The state a warm lineage holds, and what one more step from it gives: the reference every
     restored lineage is held to. *)
  let reference = warm_lineage () in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name reference);
  let snapshot = read reference in
  let stepped_once =
    let ctx, routine = Context.compile reference comp IDX.empty in
    read (Context.run ctx routine)
  in
  (* One untuned step on the lineage, after the search: equal to [stepped_once] exactly when the
     lineage's parameter AND momentum are the ones the search found. *)
  let steps_like_the_reference ctx =
    let ctx, routine = Context.compile ctx comp IDX.empty in
    Poly.equal (read (Context.run ctx routine)) stepped_once
  in
  p "the step changes the parameter" (not (Poly.equal snapshot stepped_once));
  (* One search, observing the lineage it times on at the start of every window (the preflight seam
     runs inside [time_routine], before its warmup) and at the end of every admitted one. *)
  let search ?stop_after ~lineage ~timing_ctx ctx =
    let at_start = ref [] and at_end = ref [] and admitted = ref 0 in
    let observe acc _ = acc := read lineage :: !acc in
    let seams =
      [
        Autotune.Seam.(fun f -> with_value Autotune.on_candidate_preflight (observe at_start) f);
        Autotune.Seam.(
          fun f ->
            with_value Autotune.on_candidate_timed (fun l ~timed_so_far:_ -> observe at_end l) f);
        Autotune.Seam.(
          fun f ->
            with_value Autotune.on_candidate_measured
              (fun ~label:_ ~digest:_ _ ->
                Int.incr admitted;
                Float.of_int !admitted)
              f);
        Autotune.Seam.(
          fun f ->
            with_value Autotune.on_candidate_attempt
              (fun _ ->
                if Option.exists stop_after ~f:(fun k -> List.length !at_start >= k) then
                  raise Stop_search)
              f);
      ]
    in
    let run () =
      Autotune.with_uncontended_test_windows (fun () ->
          Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1 ~timing:Autotune.Isolated
            ~cache_dir:"" ~log:true ?timing_ctx ctx comp IDX.empty)
    in
    let result =
      match List.fold_right seams ~init:run ~f:(fun seam k () -> seam k) () with
      | shipped -> `Shipped shipped
      | exception Stop_search -> `Stopped
    in
    Stdio.eprintf "windows started: %d, admitted: %d (not part of the golden)\n%!"
      (List.length !at_start) (List.length !at_end);
    (result, List.rev !at_start, List.rev !at_end)
  in
  let window_claims name ~at_start ~at_end =
    p (name ^ ": the search timed more than one window") (List.length at_start >= 2);
    p_all
      (name ^ ": every window starts from the state the lineage held before the search")
      at_start ~f:(Poly.equal snapshot);
    p_exists
      (name ^ ": a window's runs still train (the optimizer update is part of what is timed)")
      at_end ~f:(fun v -> not (Poly.equal v snapshot))
  in
  (* 1. A scratch lineage: the windows, then the scratch after the search, which must step like the
     reference -- its momentum buffer restored along with its parameter. *)
  let scratch = warm_lineage () in
  let target = warm_lineage () in
  let result, at_start, at_end = search ~lineage:scratch ~timing_ctx:(Some scratch) target in
  window_claims "timing_ctx" ~at_start ~at_end;
  p "timing_ctx: the search completed" (match result with `Shipped _ -> true | `Stopped -> false);
  p "timing_ctx: the scratch lineage ends the search holding its state"
    (Poly.equal (read scratch) snapshot);
  p "timing_ctx: the scratch lineage's next step is the reference step"
    (steps_like_the_reference scratch);
  (* 2. No scratch lineage: the search times on the caller's context, and the routine it ships takes
     its first step from the caller's state. *)
  let caller = warm_lineage () in
  let result, at_start, at_end = search ~lineage:caller ~timing_ctx:None caller in
  window_claims "caller's context" ~at_start ~at_end;
  p "caller's context: the caller's state is the one it had before the search"
    (Poly.equal (read caller) snapshot);
  p "caller's context: the shipped routine's first step is the reference step"
    (match result with
    | `Shipped (ctx, routine) -> Poly.equal (read (Context.run ctx routine)) stepped_once
    | `Stopped -> false);
  (* 3. A search that dies after two windows: the lineage is restored on the way out too. *)
  let killed = warm_lineage () in
  let result, at_start, _ = search ~stop_after:2 ~lineage:killed ~timing_ctx:(Some killed) target in
  p "killed search: it died after timing two windows"
    ((match result with `Stopped -> true | `Shipped _ -> false) && List.length at_start = 2);
  p "killed search: the lineage ends holding its state" (Poly.equal (read killed) snapshot);
  p "killed search: the lineage's next step is the reference step" (steps_like_the_reference killed)
