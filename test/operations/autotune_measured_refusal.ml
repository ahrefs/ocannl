(* gh-ocannl-1027's reject path: a value [Autotune.on_candidate_measured] returns passes the same
   admission gate as a real clock reading, so one the gate refuses (not finite and positive) refuses
   that window the way a degenerate reading would -- never ranked, and counted as a degenerate clock
   reading ([timings_unreadable]), not as contention (gh-ocannl-1517). The seam here refuses the
   untuned-default seed's window and leaves every other reading unchanged, so the refusal lands on
   the one candidate the report singles out by name: [default_refused].

   The seed is recognized by its label, and its digest is then refused on every window: a refused
   digest is dropped from the dedup set so an equivalent later seed can retry it (on both cc and
   Metal the next [F_preset] reproduces the same code), and that retry is the same measurement of
   the same code. So exactly one distinct candidate stays unmeasured while the refused windows
   number more.

   The host can refuse windows too, and on a loaded Metal device it routinely does: those never
   reach the seam, so the test counts them itself -- every timing call reports its record through
   [Autotune.with_timing_call_observer], named by the [Autotune.on_candidate_attempt] before it, and
   one the seam does not follow was refused by the host, for the cause its record's [admission]
   names. The report's refusal count must equal the two sources together on every run, and so must
   each cause's count; the claims only a quiet host can decide are gated on there having been no
   such refusal. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let () =
  (* 128 x 128 output cells reach [cpu_schedule_min_work], so on cc the untuned default is parallel
     and distinct from the serial baseline; below it, the default seed dedups against the baseline
     and is timed only under the label ["baseline"]. *)
  let n = 128 in
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
  let backend_name = Context.backend_name parent in
  Stdio.eprintf "measured refusal backend: %s\n%!" backend_name;
  (* The test config leaves automatic scheduling and fission at their defaults, so the default is a
     seed; [None] would mean no candidate reproduces it, and the claims below would have no
     target. *)
  let default_seed_label =
    Option.value (Autotune.default_seed_label ~backend_name) ~default:"<no default seed>"
  in
  let default_digest = ref None and seam_refused = ref 0 and report = ref None in
  (* The label of the latest attempt, the label and admission of a timed window the seam has not yet
     followed, and the labels and causes of the windows the host refused. *)
  let attempt = ref "" and unfollowed = ref None and host_refused = ref [] in
  let retried = ref [] in
  let settle_window () =
    Option.iter !unfollowed ~f:(fun label -> host_refused := label :: !host_refused);
    unfollowed := None
  in
  let old_measured = Autotune.Seam.get Autotune.on_candidate_measured
  and old_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  Exn.protect
    ~finally:(fun () ->
      Autotune.Seam.set Autotune.on_candidate_measured old_measured;
      Autotune.Seam.set Autotune.on_candidate_attempt old_attempt)
    ~f:(fun () ->
      Autotune.with_timing_call_observer (fun (c : Autotune.timing_call) ->
          let depth = c.decision.depth in
          Option.iter c.discarded_window ~f:(fun w ->
              retried := !attempt :: !retried;
              Stdio.eprintf "retrying %s: host contention over %d samples at depth %d\n%!" !attempt
                w.samples depth);
          Stdio.eprintf
            "  (not part of the golden) %s: depth %d, samples %d, median batch wall %.4f ms\n%!"
            !attempt depth c.window.samples c.window.median_wall_ms;
          settle_window ();
          unfollowed := Some (!attempt, c.decision.admission))
      @@ fun () ->
      (Autotune.Seam.set Autotune.on_candidate_attempt @@ fun label -> attempt := label);
      ( Autotune.Seam.set Autotune.on_candidate_measured @@ fun ~label ~digest ms ->
        unfollowed := None;
        if String.equal label default_seed_label then default_digest := Some digest;
        if Option.exists !default_digest ~f:(String.equal digest) then (
          Int.incr seam_refused;
          Stdio.eprintf "refusing %s (digest %s)\n%!" label digest;
          Float.nan)
        else ms );
      let ctx, routine =
        Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1 ~timing:Autotune.Queued
          ~cache_dir:""
          ~report:(fun r -> report := Some r)
          parent comp Ir.Indexing.Empty
      in
      Context.release (Context.run ctx routine));
  settle_window ();
  let host_causes = List.rev !host_refused in
  let host_refused = List.map host_causes ~f:fst in
  let host_refused_for cause =
    List.count host_causes ~f:(fun (_, admission) -> Poly.equal admission cause)
  in
  Stdio.eprintf "refused by the seam: %d; by the host: %d [%s]\n%!" !seam_refused
    (List.length host_refused)
    (String.concat ~sep:"; "
       (List.map host_causes ~f:(fun (label, admission) ->
            label ^ " " ^ Autotune.batch_admission_name admission)));
  Stdio.eprintf "windows retried after host contention: %d [%s]\n%!" (List.length !retried)
    (String.concat ~sep:"; " (List.rev !retried));
  let quiet_host = List.is_empty host_refused in
  let default_reached_seam = not (List.mem host_refused default_seed_label ~equal:String.equal) in
  gated ~aggregation:`Environment ~when_:default_reached_seam
    ~on:"a host that refused the default seed's window" "the seam saw the default seed's window"
    (Option.is_some !default_digest);
  match !report with
  | None -> p "the search reported" false
  | Some r ->
      Stdio.eprintf
        "timings_contended=%d (unreadable %d, unbatched %d, contention %d) candidates_contended=%d \
         candidates_timed=%d\n\
         %!"
        r.Autotune.timings_contended r.timings_unreadable r.timings_unbatched
        (Autotune.contention_refusals r) r.candidates_contended r.candidates_timed;
      p "the search completed" (match r.outcome with Autotune.Searched -> true | _ -> false);
      p "default_refused" r.default_refused;
      p "every started retry is retained in the search report"
        (r.timings_retried = List.length !retried);
      p "at least one window was refused" (r.timings_contended >= 1);
      p "every refused window is one the seam or the host refused"
        (r.timings_contended = !seam_refused + List.length host_refused);
      (* The seam's refusals are degenerate readings; the host's are counted under the cause its own
         timing call named. *)
      p "every window the seam refused counts as a degenerate clock reading, the host's by cause"
        (r.timings_unreadable = !seam_refused + host_refused_for Autotune.Refused_unreadable
        && r.timings_unbatched = host_refused_for Autotune.Refused_unbatched
        && Autotune.contention_refusals r = host_refused_for Autotune.Refused_contended);
      (* Where the host refused the default's own window first, the seam never learned its digest,
         and an equivalent later seed it admitted measured the default after all. *)
      p "the unmeasured candidates are the default and at most the host's refusals"
        ((if default_reached_seam then 1 else 0) <= r.candidates_contended
        && r.candidates_contended <= 1 + List.length host_refused);
      gated ~aggregation:`Environment ~when_:quiet_host ~on:"a host that refused a window"
        "exactly one distinct candidate stayed unmeasured" (r.candidates_contended = 1);
      gated ~aggregation:`Environment ~when_:default_reached_seam
        ~on:"a host that refused the default seed's window" "the default has no reference time"
        (Option.is_none r.default_ms);
      p "the refused seed did not win" (not (String.equal r.best_label default_seed_label))
