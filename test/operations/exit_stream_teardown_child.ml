(* The child processes [exit_stream_teardown] runs and watches from outside (gh-ocannl-1036): what
   happens at process exit can only be observed by a parent. The first argument picks the mode.

   [device] runs a small computation on the configured backend and exits normally. It prints the
   backend's name, so the parent reports which backend the run actually reached, and reads the
   result back from an [at_exit] handler registered before the device opened.

   [device_busy] stages the hung case as far as real hardware can: seconds of work queued on the
   configured backend and never synced, then a normal exit. The backend's teardown reports how long
   it waited on stderr, since nothing registered in here can run after it.

   [device_graph] (gh-ocannl-1445) runs a fissioned routine twice, so a CUDA or HIP backend
   graph-captures its segments and caches the instantiated graph, prints how many graphs the backend
   caches, and exits normally without releasing the routine: the backend's exit teardown must
   destroy that graph too.

   [never_idle] and [never_idle_raise] open no device. They run [Utils.bounded_exit_teardown] from
   an [at_exit] handler over a stand-in stream that never becomes idle and whose teardown would hang
   for an hour; the second then ends on an uncaught exception, which OCaml runs [at_exit] for too.
   [two_never_idle] runs two such stand-ins, which share one bound. [idle] runs one that is idle at
   once, so its teardown is due. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

(* A routine that fissions into several segments: the materialized intermediates cut the chain of
   two matmuls, so a CUDA or HIP backend graph-captures its segment launches. *)
let fissioned_chain () =
  let q = 32 in
  let matrix label ~modulus =
    TDSL.ndarray
      (Array.init (q * q)
         ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus ~offset:(-2.) ~stride:1.))
      ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()
  in
  let qa = matrix "fc_a" ~modulus:11 and qb = matrix "fc_b" ~modulus:7 in
  let qc = matrix "fc_c" ~modulus:5 in
  let%op qd = qa + qb in
  Train.set_materialized qd.Tensor.value;
  let%op qe = qd * qc in
  Train.set_materialized qe.Tensor.value;
  let%op qg = qc * qe in
  Train.forward qg

let keep_alive : (Context.t * Context.routine) option ref = ref None

let outcome_name : Utils.exit_teardown_outcome -> string = function
  | Torn_down -> "torn_down"
  | Disabled -> "disabled"
  | Still_busy -> "still_busy"
  | Failed _ -> "failed"

let stand_in ?(count = 1) ~idle () =
  Stdlib.at_exit (fun () ->
      let started = Mtime_clock.counter () in
      Utils.bounded_exit_teardown
        (List.init count ~f:(fun k : Utils.exit_teardown_resource ->
             {
               what = "stand-in stream " ^ Int.to_string k;
               is_idle = (fun () -> idle);
               teardown =
                 (fun () ->
                   Stdio.printf "%s\n%!" Exit_stream_teardown_marker.teardown;
                   if not idle then Unix.sleepf 3600.);
             }))
      |> List.iter ~f:(fun outcome -> Stdio.printf "outcome: %s\n%!" (outcome_name outcome));
      (* The teardown call alone, without process startup, for claims too tight to absorb it. *)
      Stdio.printf "teardown seconds: %.3f\n%!"
        (Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9))

let () =
  (* Binary stdout: the driver matches "read at exit: 15\n" byte for byte, and Windows' text mode
     would end the line "\r\n". *)
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  match Array.to_list (Sys.get_argv ()) |> List.tl_exn |> List.hd with
  | Some "device" ->
      (* The program's own [at_exit], registered before the device opens, reads the device at exit:
         the backend's teardown must not have destroyed the stream under it. *)
      let read_at_exit = ref (fun () -> ()) in
      Stdlib.at_exit (fun () -> !read_at_exit ());
      Tensor.unsafe_reinitialize ();
      let ctx = Context.auto () in
      let%op y = ({ hey = 7.0 } * ([ 2.0 ] : q)) + ([ 1.0 ] : p) in
      let ctx = Train.forward_once ctx y in
      Stdio.printf "backend: %s\n%!" (Context.backend_name ctx);
      read_at_exit :=
        fun () -> Stdio.printf "read at exit: %g\n%!" (Context.get_values ctx y.Tensor.value).(0)
  | Some "device_busy" ->
      Tensor.unsafe_reinitialize ();
      let ctx = Context.auto () in
      (* Large enough that the device, not the host's submission of it, bounds a run's time. *)
      let n = 2048 in
      let a = TDSL.range_of_shape ~output_dims:[ n ] ~input_dims:[ n ] () in
      let b = TDSL.range_of_shape ~output_dims:[ n ] ~input_dims:[ n ] () in
      let%op c = a * b in
      (* Only on hip, the backend that tears its stream down at exit: elsewhere the driver skips the
         claim, and a synchronous backend would spend the seconds below right here. *)
      let runs, one_run =
        if not (String.equal (Context.backend_name ctx) "hip") then (0, 0.)
        else
          let ctx, routine = Train.to_routine ctx Train.IDX.empty (Train.forward c) in
          let ctx = Context.run ctx routine in
          Context.sync ctx;
          (* Amortized over a batch, so one sync's latency does not pass for device time. *)
          let batch = 20 in
          let started = Mtime_clock.counter () in
          for _ = 1 to batch do
            ignore (Context.run ctx routine : Context.t)
          done;
          Context.sync ctx;
          let one_run =
            Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9 /. Float.of_int batch
          in
          (* Seconds of queued work, far past the bound the driver passes, left unsynced. *)
          let runs = Int.of_float (Float.round_up (3. /. Float.max one_run 1e-4)) in
          for _ = 1 to runs do
            ignore (Context.run ctx routine : Context.t)
          done;
          (runs, one_run)
      in
      Stdio.printf "backend: %s\n%!" (Context.backend_name ctx);
      Stdio.eprintf "queued %d runs of %.4fs each without a sync (not part of the golden)\n%!" runs
        one_run
  | Some "device_graph" ->
      Tensor.unsafe_reinitialize ();
      let ctx = Context.auto () in
      let ctx, routine = Context.compile ctx (fissioned_chain ()) Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      let ctx = Context.run ctx routine in
      Context.sync ctx;
      let cached =
        match Context.global_debug_info ctx with
        | Sexp.List (_ :: fields) ->
            List.find_map fields ~f:(function
              | Sexp.List [ Sexp.Atom "live_graph_execs"; Sexp.Atom n ] -> Some n
              | _ -> None)
        | _ -> None
      in
      (* Rooted to the end of the process: the backend's registry holds the cache weakly, and a
         collection before its exit teardown would let the binding's finalizer destroy the graph
         instead, which the claim would then credit to the teardown. *)
      keep_alive := Some (ctx, routine);
      Stdio.printf "backend: %s\n%!" (Context.backend_name ctx);
      Stdio.printf "cached graphs at exit: %s\n%!" (Option.value cached ~default:"none")
  | Some "never_idle" -> stand_in ~idle:false ()
  | Some "two_never_idle" -> stand_in ~count:2 ~idle:false ()
  | Some "never_idle_raise" ->
      stand_in ~idle:false ();
      failwith "exit_stream_teardown_child: deliberate uncaught exception"
  | Some "idle" -> stand_in ~idle:true ()
  | _ ->
      failwith
        "exit_stream_teardown_child: expected \
         device|device_busy|device_graph|never_idle|two_never_idle|never_idle_raise|idle"
