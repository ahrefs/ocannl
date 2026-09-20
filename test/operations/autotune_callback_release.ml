(* gh-ocannl-975: both post-admission callback boundaries must release a losing candidate. Select
   using actual admitted times, not attempt order; working pools count retained backend allocations
   separately from the intentionally persistent constant cache.

   Whether any admitted candidate measures strictly slower than the incumbent is a property of the
   machine's timings: a short search whose samples happen to fall monotonically (warm-up, a loaded
   GPU) admits only winners, and the injection never fires (rog-nv/cuda, sweep 2026-09-20). No seam
   lets a test slow a candidate down, and injecting at a winner instead would prove nothing about
   the pending-owner release this test exists for: a winner already sits in [best_so_far], which the
   exit sweep released before the fix. So a leg whose search completes uninjected is re-rolled a few
   times -- each attempt is a full search with fresh timings -- and only when every roll is monotone
   are the three injection claims reported as a backend-scoped skip, with the census claims still
   asserted over all attempts. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

exception Callback_failure

let () =
  let n = 16 in
  let a =
    TDSL.ndarray
      (Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7)))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let b =
    TDSL.ndarray
      (Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5)))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let%op product = a * b in
  let comp = Train.forward product in
  let parent = Context.auto () in
  Stdio.eprintf "callback cleanup backend: %s\n%!" (Context.backend_name parent);
  let ref_ctx, ref_routine = Context.compile parent comp Ir.Indexing.Empty in
  let ref_ctx = Context.run ref_ctx ref_routine in
  let expected = Context.get_values ref_ctx product.Tensor.value in
  Context.release ref_ctx;
  let backend = Context.backend_name parent in
  let attempt ~scratch site =
    let injected = ref false and report = ref None in
    let old = !Autotune.on_candidate_callback and old_timed = !Autotune.on_candidate_timed in
    let result =
      Exn.protect
        ~finally:(fun () ->
          Autotune.on_candidate_callback := old;
          Autotune.on_candidate_timed := old_timed)
        ~f:(fun () ->
          (Autotune.on_candidate_callback :=
             fun boundary ~candidate_ms ~incumbent_ms ->
               if
                 Option.exists site ~f:(fun s -> Poly.equal s boundary)
                 && Float.(candidate_ms > incumbent_ms)
               then (
                 Stdio.eprintf "nonwinner callback: candidate %.9g ms > incumbent %.9g ms\n%!"
                   candidate_ms incumbent_ms;
                 let inject () =
                   injected := true;
                   raise Callback_failure
                 in
                 match boundary with
                 | `Timed -> Autotune.on_candidate_timed := fun _ ~timed_so_far:_ -> inject ()
                 | `Calibration -> inject ()));
          try
            let ctx, routine =
              Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1
                ?timing_ctx:(if scratch then Some parent else None)
                ~timing:Autotune.Isolated ~cache_dir:""
                ~report:(fun r -> report := Some r)
                parent comp Ir.Indexing.Empty
            in
            let ctx = Context.run ctx routine in
            let got = Context.get_values ctx product.Tensor.value in
            (* Only the control makes this claim on stdout: an injecting leg that completes has
               nothing to verify here and is about to be re-rolled. *)
            if Option.is_none site then
              p_all2 "ordinary completion computes the reference" got expected ~f:(fun x y ->
                  Float.(abs (x - y) < 1e-4));
            Context.release ctx;
            `Returned
          with Callback_failure -> `Injected)
    in
    (result, !injected, !report)
  in
  let run ~scratch site =
    let before = Ir.Alloc_census.snapshot () in
    let name =
      match site with
      | None -> "ordinary"
      | Some `Timed -> "timed"
      | Some `Calibration -> "calibration"
    in
    let name = (if scratch then "scratch " else "direct ") ^ name in
    let max_rolls = 4 in
    let rec roll k =
      match attempt ~scratch site with
      | `Returned, _, _ when Option.is_some site && k < max_rolls ->
          (* Every admitted candidate was a new best, so nothing qualified. The control run
             completed and released everything it made, which the census below still covers; a new
             search draws new timings. *)
          Stdio.eprintf "%s: roll %d admitted no strictly slower candidate; re-rolling\n%!" name k;
          roll (k + 1)
      | outcome -> (outcome, k)
    in
    let (result, injected, report), rolls = roll 1 in
    let after = Ir.Alloc_census.snapshot () in
    Stdio.eprintf "%s before: %s\n%s after: %s\n%!" name
      (Ir.Alloc_census.to_string before)
      name (Ir.Alloc_census.to_string after);
    Stdio.eprintf
      "%s deltas: working pools=%d bytes=%d; contexts created=%d released=%d; constants=%d\n%!" name
      (after.live_working_pools - before.live_working_pools)
      (after.live_working_bytes - before.live_working_bytes)
      (after.contexts_created - before.contexts_created)
      (after.contexts_released - before.contexts_released)
      (after.live_constant_pools - before.live_constant_pools);
    (match site with
    | None -> p "ordinary search completed" (Poly.equal result `Returned)
    | Some _ when not injected ->
        (* [max_rolls] full searches, none with a nonwinner: this backend's timings gave the leg
           nothing to inject at. Not a cleanup failure, and not coverage either. Backend-scoped on
           purpose: another backend on the same box finding a nonwinner says nothing about whether
           THIS backend's release paths ran. *)
        Stdio.eprintf "%s: %d searches admitted no strictly slower candidate\n%!" name rolls;
        List.iter
          [
            "injected at a strictly slower admitted candidate";
            "callback exception propagates";
            "partial report retains a measured incumbent";
          ] ~f:(fun claim -> skipped ~backend (name ^ ": " ^ claim))
    | Some _ ->
        Stdio.eprintf "%s: injected on roll %d\n%!" name rolls;
        p (name ^ ": injected at a strictly slower admitted candidate") injected;
        p (name ^ ": callback exception propagates") (Poly.equal result `Injected);
        p
          (name ^ ": partial report retains a measured incumbent")
          (Option.exists report ~f:(fun r ->
               match r.Autotune.outcome with
               | Autotune.Search_died _ -> r.candidates_timed >= 2 && Float.is_finite r.best_ms
               | _ -> false)));
    p
      (name ^ ": working pools return to their starting count")
      (after.live_working_pools = before.live_working_pools);
    p
      (name ^ ": working bytes return to their starting count")
      (after.live_working_bytes = before.live_working_bytes);
    p
      (name ^ ": every created context was explicitly released")
      (after.contexts_created - before.contexts_created
      = after.contexts_released - before.contexts_released)
  in
  List.iter [ false; true ] ~f:(fun scratch ->
      run ~scratch None;
      run ~scratch (Some `Timed);
      run ~scratch (Some `Calibration))
