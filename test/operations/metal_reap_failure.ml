(* gh-ocannl-2139: exercise the actual [reap] -> [raise_on_failure] -> backend classifier path
   without relying on macOS's watchdog. A finite kernel is submitted normally; only its observed
   status and error text are replaced. The healthy control runs the same computation first. *)
open Base
open Verdict.Claims
module MF = Ir.Metal_failure
module SO = Ir.Schedule_outcome

let ctx = Context.auto ()
let backend_name = Context.backend_name ctx
let leg = Verdict.leg ~when_:(String.equal backend_name "metal") ~on:backend_name ()
let driver_error = "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)"

let submit ~first_id ~name ctx =
  let mk = Ll_test.node_factory ~first_id ~dims:[| 8 |] () in
  let a = mk (name ^ "_src") and x = mk name in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let llc =
    Ll_test.loop_n i 8 (Ll_test.set_at x at (Ll_test.add (Ll_test.get a [| at |]) (Ll_test.c 1.)))
  in
  let o = Ll_test.optimize ~materialized:[ a; x ] ~name llc in
  let av = Array.init 8 ~f:Float.of_int in
  let ctx, _ = Ll_test.run_routine ~ctx ~name o ~seed:[ (a, av); (x, Ll_test.blank 8) ] in
  (ctx, x, Array.map av ~f:(fun v -> v +. 1.))

let raised f =
  try
    f ();
    None
  with exn -> Some exn

let report = Option.map ~f:Exn.to_string

let outcome =
  lazy
    (let healthy, x, expected = submit ~first_id:9_300 ~name:"reap_healthy" ctx in
     Context.sync healthy;
     let before = Array.equal Float.equal (Context.get_values healthy x) expected in
     let failed, output, _ = submit ~first_id:9_400 ~name:"reap_injected" ctx in
     let observed = ref None in
     let first =
       MF.For_testing.with_reap_error
         (fun ~what ->
           if String.is_substring what ~substring:"kernel reap_injected" then (
             observed := Some what;
             Some (Some driver_error))
           else None)
         ~f:(fun () -> raised (fun () -> Context.sync failed))
     in
     (* Outside the override scope: subsequent reads and fresh lineages must still refuse the failed
        runner, even though the real finite kernel completed successfully. *)
     let read = raised (fun () -> ignore (Context.get_values failed output : float array)) in
     let fresh = raised (fun () -> Context.sync (Context.auto ())) in
     (before, !observed, first, read, fresh))

let claim label f = Verdict.in_leg ~leg label (fun label -> p label (f (Lazy.force outcome)))
let classified exn phase = Option.bind exn ~f:(Context.failure_classifier ctx phase)

let () =
  claim "the uninjected routine ran and read back" (fun (before, _, _, _, _) -> before);
  claim "reap observed the submitted kernel" (fun (_, observed, _, _, _) -> Option.is_some observed);
  claim "the wait raised the first typed report for a kernel buffer" (fun (_, _, first, _, _) ->
      match first with
      | Some (MF.Command_buffer_failed { first_report = true; runs_kernels = true; _ }) -> true
      | _ -> false);
  claim "the report names the observed kernel and carries the driver text"
    (fun (_, observed, first, _, _) ->
      match (observed, report first) with
      | Some what, Some text ->
          String.is_substring text ~substring:what
          && String.is_substring text ~substring:driver_error
      | _ -> false);
  List.iter [ SO.Sync; SO.Launch ] ~f:(fun phase ->
      claim
        ("the first report is an Expected gpu_hang decline at "
        ^ Sexp.to_string (SO.sexp_of_phase phase))
        (fun (_, _, first, _, _) ->
          match classified first phase with
          | Some
              {
                phase = got;
                cause = Backend_rejected { backend = "metal"; stage; severity = Expected; _ };
                execution_effect = Writes_may_have_occurred;
              } ->
              SO.equal_phase phase got && String.equal stage MF.gpu_hang_stage
          | _ -> false));
  claim "reading the output raised the same report" (fun (_, _, first, read, _) ->
      Option.is_some read && Option.equal String.equal (report first) (report read));
  claim "a fresh lineage on the same device raised the same report" (fun (_, _, first, _, fresh) ->
      Option.is_some fresh && Option.equal String.equal (report first) (report fresh));
  claim "later reports are marked as repeats and stay unclassified" (fun (_, _, _, read, fresh) ->
      let repeat = function
        | Some (MF.Command_buffer_failed { first_report = false; _ }) as exn ->
            Option.is_none (classified exn SO.Sync)
        | _ -> false
      in
      repeat read && repeat fresh)
