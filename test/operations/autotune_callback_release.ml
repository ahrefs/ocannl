(* gh-ocannl-975: both post-admission callback boundaries must release a losing candidate. Working
   pools count retained backend allocations separately from the intentionally persistent constant
   cache.

   The injection must land on a NONWINNING admitted candidate: a winner already sits in
   [best_so_far], which the exit sweep released before the fix, so injecting there proves nothing
   about the pending-owner release this test exists for. Whether a real search ever admits a
   candidate slower than its incumbent is a property of the machine's timings, so every leg pins the
   ranking through [Autotune.on_candidate_measured] (gh-ocannl-1027): the first admitted window
   measures 1 ms and the k-th k ms. The second admitted window is then the first nonwinner, on every
   backend and every run, and the claims below name it exactly. *)
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
  let attempt ~name ~scratch site =
    (* Labels of the admitted windows, newest first; the k-th admitted window measures k ms. *)
    let admitted = ref [] and injected_at = ref None and report = ref None in
    let old = !Autotune.on_candidate_callback
    and old_timed = !Autotune.on_candidate_timed
    and old_measured = !Autotune.on_candidate_measured in
    let result =
      Exn.protect
        ~finally:(fun () ->
          Autotune.on_candidate_callback := old;
          Autotune.on_candidate_timed := old_timed;
          Autotune.on_candidate_measured := old_measured)
        ~f:(fun () ->
          (Autotune.on_candidate_measured :=
             fun ~label ~digest:_ _ms ->
               admitted := label :: !admitted;
               Float.of_int (List.length !admitted));
          (Autotune.on_candidate_callback :=
             fun boundary ~candidate_ms ~incumbent_ms ->
               if
                 Option.exists site ~f:(fun s -> Poly.equal s boundary)
                 && Float.(candidate_ms > incumbent_ms)
               then (
                 Stdio.eprintf "nonwinner callback: candidate %.9g ms > incumbent %.9g ms\n%!"
                   candidate_ms incumbent_ms;
                 (* The callback follows the seam for the same window, so the newest admitted label
                    is the candidate this callback is about. *)
                 let inject () =
                   injected_at := List.hd !admitted;
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
            p_all2 (name ^ ": completed search computes the reference") got expected ~f:(fun x y ->
                Float.(abs (x - y) < 1e-4));
            Context.release ctx;
            `Returned
          with Callback_failure -> `Injected)
    in
    (result, List.rev !admitted, !injected_at, !report)
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
    let result, admitted, injected_at, report = attempt ~name ~scratch site in
    let after = Ir.Alloc_census.snapshot () in
    Stdio.eprintf "%s admitted windows: [%s]; injected at: %s\n%!" name
      (String.concat ~sep:"; " admitted)
      (Option.value injected_at ~default:"-");
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
    let first_admitted = List.hd admitted in
    (* The report's own accounting, its winner and its best time against what the seam pinned. *)
    let report_pins r =
      r.Autotune.candidates_timed = List.length admitted
      && Option.exists first_admitted ~f:(String.equal r.Autotune.best_label)
      && Float.equal r.best_ms 1.0
    in
    (match site with
    | None ->
        p (name ^ ": search completed") (Poly.equal result `Returned);
        p
          (name ^ ": the pinned-fastest first admitted window wins")
          (Option.exists report ~f:(fun r ->
               (match r.Autotune.outcome with Autotune.Searched -> true | _ -> false)
               && List.length admitted >= 2
               && report_pins r))
    | Some _ ->
        p
          (name ^ ": injected at the second admitted window, the first nonwinner")
          (List.length admitted = 2 && Option.equal String.equal injected_at (List.nth admitted 1));
        p (name ^ ": callback exception propagates") (Poly.equal result `Injected);
        p
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
      run ~scratch (Some `Calibration))
