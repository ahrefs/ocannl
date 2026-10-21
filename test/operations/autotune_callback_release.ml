(* gh-ocannl-975: both post-admission callback boundaries must release a losing candidate. Working
   pools count retained backend allocations separately from the intentionally persistent constant
   cache.

   The injection must land on a NONWINNING admitted candidate: a winner already sits in
   [best_so_far], which the exit sweep released before the fix, so injecting there proves nothing
   about the pending-owner release this test exists for. Whether a real search ever admits a
   candidate slower than its incumbent is a property of the machine's timings, so every leg pins the
   ranking through [Autotune.on_candidate_measured] (gh-ocannl-1027): the first admitted window
   measures 1 ms and the k-th k ms. The second admitted window is then the first nonwinner, on every
   backend and every run, and the claims below name it exactly.

   That second admitted window has to EXIST (gh-ocannl-1465). A loaded host refused every window but
   one of a [direct timed] search on tuf/hip, so nothing was injected and the search completed.
   Every reading here is scripted, so the windows are opted out of host contention
   ([with_uncontended_test_windows], gh-ocannl-1156), as the other scripted-measurement searches do.
   The admission gate's other refusals (a degenerate clock reading) remain, and the search records
   them: a leg whose search recorded a refused window and admitted fewer than two is re-run a
   bounded number of times, and only then are the claims that need the second window an environment
   skip -- gated on the search's own refusal record, never on the admitted count alone, so a search
   that simply stops producing a second candidate still fails. The cleanup claims are never gated.
   The last leg forces the refusal on every host and pins that the gate opens on it and that the
   claim it gates would have failed. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

exception Callback_failure

(* How many times a leg whose second admitted window was refused is run before the claims needing
   that window are skipped, and the skip's reason. *)
let attempts = 3
let starved_on = "timing windows refused before a second was admitted"

(* The claims needing the second admitted window are skipped exactly when this holds: fewer than two
   windows admitted, AND the search's own record says a window was refused. *)
let starved ~admitted (report : Autotune.report option) =
  List.length admitted < 2 && Option.exists report ~f:(fun r -> r.Autotune.timings_contended > 0)

let () =
  let n = 16 in
  let a =
    TDSL.ndarray
      (Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:1.))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let b =
    TDSL.ndarray
      (Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:0. ~stride:1.))
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
  (* [refuse_after]: the windows after that many admitted ones are refused at the admission gate, as
     a degenerate clock reading is (the seam's non-finite result). *)
  let attempt ~scratch ~refuse_after site =
    (* Labels of the admitted windows, newest first; the k-th admitted window measures k ms. *)
    let admitted = ref [] and injected_at = ref None and report = ref None in
    let old = Autotune.Seam.get Autotune.on_candidate_callback
    and old_timed = Autotune.Seam.get Autotune.on_candidate_timed
    and old_measured = Autotune.Seam.get Autotune.on_candidate_measured in
    let result =
      Exn.protect
        ~finally:(fun () ->
          Autotune.Seam.set Autotune.on_candidate_callback old;
          Autotune.Seam.set Autotune.on_candidate_timed old_timed;
          Autotune.Seam.set Autotune.on_candidate_measured old_measured)
        ~f:(fun () ->
          ( Autotune.Seam.set Autotune.on_candidate_measured @@ fun ~label ~digest:_ _ms ->
            if Option.exists refuse_after ~f:(fun k -> List.length !admitted >= k) then Float.nan
            else (
              admitted := label :: !admitted;
              Float.of_int (List.length !admitted)) );
          ( Autotune.Seam.set Autotune.on_candidate_callback
          @@ fun boundary ~candidate_ms ~incumbent_ms ->
            if
              Option.exists site ~f:(fun s -> Poly.equal s boundary)
              && Float.(candidate_ms > incumbent_ms)
            then (
              Stdio.eprintf "nonwinner callback: candidate %.9g ms > incumbent %.9g ms\n%!"
                candidate_ms incumbent_ms;
              (* The callback follows the seam for the same window, so the newest admitted label is
                 the candidate this callback is about. *)
              let inject () =
                injected_at := List.hd !admitted;
                raise Callback_failure
              in
              match boundary with
              | `Timed ->
                  Autotune.Seam.set Autotune.on_candidate_timed @@ fun _ ~timed_so_far:_ ->
                  inject ()
              | `Calibration -> inject ()) );
          try
            let ctx, routine =
              Autotune.with_uncontended_test_windows (fun () ->
                  Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1
                    ?timing_ctx:(if scratch then Some parent else None)
                    ~timing:Autotune.Isolated ~cache_dir:""
                    ~report:(fun r -> report := Some r)
                    parent comp Ir.Indexing.Empty)
            in
            let ctx = Context.run ctx routine in
            let got = Context.get_values ctx product.Tensor.value in
            Context.release ctx;
            `Returned got
          with Callback_failure -> `Injected)
    in
    (result, List.rev !admitted, !injected_at, !report)
  in
  let run ?(attempts = attempts) ?refuse_after ~scratch site =
    let before = Ir.Alloc_census.snapshot () in
    let name =
      match site with
      | None -> "ordinary"
      | Some `Timed -> "timed"
      | Some `Calibration -> "calibration"
    in
    let name =
      (if Option.is_some refuse_after then "forced refusal " else "")
      ^ (if scratch then "scratch " else "direct ")
      ^ name
    in
    let rec attempt_until k =
      let ((_, admitted, injected_at, report) as outcome) = attempt ~scratch ~refuse_after site in
      Stdio.eprintf
        "%s attempt %d: admitted windows: [%s]; refused windows: %s; injected at: %s\n%!" name k
        (String.concat ~sep:"; " admitted)
        (Option.value_map report ~default:"no report" ~f:(fun r ->
             Int.to_string r.Autotune.timings_contended))
        (Option.value injected_at ~default:"-");
      if starved ~admitted report && k < attempts then (
        Stdio.eprintf
          "%s attempt %d of %d (not part of the golden): no second admitted window; retrying\n%!"
          name k attempts;
        attempt_until (k + 1))
      else outcome
    in
    let result, admitted, injected_at, report = attempt_until 1 in
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
    let starved = starved ~admitted report in
    let needs_second label b =
      gated ~aggregation:`Environment ~when_:(not starved) ~on:starved_on label b
    in
    let first_admitted = List.hd admitted in
    (* The report's own accounting, its winner and its best time against what the seam pinned. *)
    let report_pins r =
      r.Autotune.candidates_timed = List.length admitted
      && Option.exists first_admitted ~f:(String.equal r.Autotune.best_label)
      && Float.equal r.best_ms 1.0
    in
    (match (refuse_after, site) with
    | Some _, _ ->
        (* The control: one window admitted, the rest refused. The gate the legs above read opens on
           that record, and what it gates would have failed: nothing was injected, and the search
           completed. *)
        p
          (name ^ ": the search records its refused windows and admitted only the first")
          (List.length admitted = 1
          && Option.exists report ~f:(fun r -> r.Autotune.timings_contended > 0));
        p (name ^ ": the second-window claims' gate opens on that record") starved;
        p
          (name ^ ": with no second admitted window, nothing was injected")
          (Option.is_none injected_at
          && match result with `Returned _ -> true | `Injected -> false)
    | None, None ->
        (* Nothing injects on this leg; an injected result would compare an empty readback. *)
        let got = match result with `Returned got -> got | `Injected -> [||] in
        p_close (name ^ ": completed search computes the reference") got expected ~tol:(fun _ ->
            1e-4);
        p (name ^ ": search completed")
          (match result with `Returned _ -> true | `Injected -> false);
        needs_second
          (name ^ ": the pinned-fastest first admitted window wins")
          (Option.exists report ~f:(fun r ->
               (match r.Autotune.outcome with Autotune.Searched -> true | _ -> false)
               && List.length admitted >= 2
               && report_pins r))
    | None, Some _ ->
        needs_second
          (name ^ ": injected at the second admitted window, the first nonwinner")
          (List.length admitted = 2 && Option.equal String.equal injected_at (List.nth admitted 1));
        needs_second
          (name ^ ": callback exception propagates")
          (match result with `Injected -> true | `Returned _ -> false);
        needs_second
          (name ^ ": partial report retains the pinned incumbent")
          (Option.exists report ~f:(fun r ->
               (match r.Autotune.outcome with Autotune.Search_died _ -> true | _ -> false)
               && report_pins r)));
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
      run ~scratch (Some `Calibration));
  run ~attempts:1 ~refuse_after:1 ~scratch:false (Some `Timed)
