(* gh-ocannl-2209: finite kernels only. The domain-local seam changes the host's status
   observations, never GPU execution. Children isolate the runner's permanent failure/abandonment
   latches. This tests bounded waits after event signalling, not the OS watchdog or real hangs. *)
open Base
open Verdict.Claims
module MF = Ir.Metal_failure

let elapsed clock = Mtime.Span.to_float_ns (Mtime_clock.count clock) /. 1e9

let submit ctx =
  let mk = Ll_test.node_factory ~first_id:9_500 ~dims:[| 8 |] () in
  let x = mk "bounded_output" in
  Ll_test.materialize x;
  let i = Ll_test.sym () in
  let llc = Ll_test.loop_n i 8 (Ll_test.set_at x (Ll_test.iter i) (Ll_test.c 7.)) in
  let name = "bounded_finite" in
  let o = Ll_test.optimize ~materialized:[ x ] ~name llc in
  fst (Ll_test.run_routine ~ctx ~name o ~seed:[ (x, Ll_test.blank 8) ])

let child mode =
  let ctx = Context.auto () in
  if String.equal mode "healthy" then Context.await_within (submit ctx) ~seconds:4.
  else
    let waiting = ref None in
    let reads = ref 0 in
    MF.For_testing.with_reap_status
      (fun ~what ->
        if String.is_substring what ~substring:"kernel bounded_finite" then (
          Stdlib.incr reads;
          let fail =
            String.equal mode "failure"
            && Option.value_map !waiting ~default:false ~f:(fun clock ->
                !reads > 4 && Float.(elapsed clock >= 0.1))
          in
          Some (if fail then MF.For_testing.Error (Some "scripted bounded failure") else Pending))
        else None)
      ~f:(fun () ->
        let ctx = submit ctx in
        let clock = Mtime_clock.counter () in
        waiting := Some clock;
        let bound = if String.equal mode "failure" then 4. else 0.25 in
        let result =
          try `Returned (Context.await_within ctx ~seconds:bound) with exn -> `Raised exn
        in
        let seconds = elapsed clock in
        Stdio.eprintf
          "bounded %s: %.6f s, bound %.2f s, status reads %d (not part of the golden)\n%!" mode
          seconds bound !reads;
        match (mode, result) with
        | ( "failure",
            `Raised (MF.Command_buffer_failed { first_report = true; runs_kernels = true; _ }) ) ->
            Float.(seconds < bound /. 2.)
        | "timeout", `Returned false ->
            let latched =
              try
                ignore (Context.await_within ctx ~seconds:4. : bool);
                false
              with Ir.Backend_intf.Device_wedged _ -> true
            in
            latched && Float.(seconds >= bound && seconds < 2.)
        | _ -> false)

let run mode =
  let exe = Stdlib.Sys.executable_name in
  let pid = Unix.create_process exe [| exe; mode |] Unix.stdin Unix.stdout Unix.stderr in
  match Unix.waitpid [] pid with _, Unix.WEXITED 0 -> true | _ -> false

let () =
  match Stdlib.Sys.argv with
  | [| _; mode |] -> Stdlib.exit (if child mode then 0 else 1)
  | _ ->
      let ctx = Context.auto () in
      let backend = Context.backend_name ctx in
      let leg = Verdict.leg ~when_:(String.equal backend "metal") ~on:backend () in
      List.iter
        [
          ("failure", "scripted failure raises first typed report before half the four-second bound");
          ("timeout", "pending buffer abandons at the bound and latches Device_wedged");
          ("healthy", "healthy bounded wait returns true");
        ]
        ~f:(fun (mode, label) -> Verdict.in_leg ~leg label (fun label -> p label (run mode)))
