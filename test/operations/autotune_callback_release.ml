(* gh-ocannl-975: both post-admission callback boundaries must release a losing candidate. Select
   using actual admitted times, not attempt order; working pools count retained backend allocations
   separately from the intentionally persistent constant cache.

   Whether any admitted candidate measures strictly slower than the incumbent is a property of the
   machine's timings: a short search whose samples happen to fall monotonically (warm-up, a fast
   GPU) admits only winners, and the injection never fires (rog-nv/cuda, sweep 2026-09-20). The
   preferred selector therefore has a deterministic fallback -- the first candidate admitted after
   some other measurement, which exists whenever the partial report can retain a measured incumbent
   -- so the leg always injects; which selector fired is reported on stderr. Admissions are counted
   through [on_candidate_timed], the tuner's own counter, because the timed serial baseline grows
   [candidates_timed] without ever reaching [on_candidate_callback]. *)
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
  let attempt ~scratch ~select site =
    let injected = ref None and report = ref None and timed_seen = ref 0 in
    let old = !Autotune.on_candidate_callback and old_timed = !Autotune.on_candidate_timed in
    let result =
      Exn.protect
        ~finally:(fun () ->
          Autotune.on_candidate_callback := old;
          Autotune.on_candidate_timed := old_timed)
        ~f:(fun () ->
          (Autotune.on_candidate_timed := fun _ ~timed_so_far -> timed_seen := timed_so_far);
          (Autotune.on_candidate_callback :=
             fun boundary ~candidate_ms ~incumbent_ms ->
               if Option.exists site ~f:(fun s -> Poly.equal s boundary) then
                 let nonwinner = Float.(candidate_ms > incumbent_ms) in
                 (* Admissions measured before this candidate: the [`Timed] boundary precedes this
                    candidate's own [on_candidate_timed], the [`Calibration] boundary follows it. *)
                 let preceding =
                   match boundary with `Timed -> !timed_seen | `Calibration -> !timed_seen - 1
                 in
                 let selected =
                   match select with `Nonwinner -> nonwinner | `Over_incumbent -> preceding >= 1
                 in
                 if selected then (
                   Stdio.eprintf "%s callback: candidate %.9g ms vs incumbent %.9g ms\n%!"
                     (if nonwinner then "nonwinner" else "winner")
                     candidate_ms incumbent_ms;
                   let inject () =
                     injected := Some nonwinner;
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
               nothing to verify here and is about to be retried with the fallback selector. *)
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
    let result, injected, report =
      match attempt ~scratch ~select:`Nonwinner site with
      | `Returned, _, _ when Option.is_some site ->
          (* Every admitted candidate was a new best, so the preferred selector had nothing to pick.
             The control run completed and released everything it made, which the census below still
             covers; inject at the first candidate admitted over an existing measurement. *)
          Stdio.eprintf
            "%s: no strictly slower candidate was admitted; injecting at the first candidate \
             admitted over a measured incumbent instead\n\
             %!"
            name;
          attempt ~scratch ~select:`Over_incumbent site
      | outcome -> outcome
    in
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
    | Some _ ->
        Stdio.eprintf "%s: injected at a %s\n%!" name
          (match injected with
          | Some true -> "strictly slower admitted candidate"
          | Some false -> "winning admitted candidate (fallback selector)"
          | None -> "no candidate");
        p (name ^ ": injected at an admitted candidate") (Option.is_some injected);
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
