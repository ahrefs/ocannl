(* The child processes [exit_stream_teardown] runs and watches from outside (gh-ocannl-1036): what
   happens at process exit can only be observed by a parent. One mode per argument:

   - [device]: a small computation on the configured backend, then a normal exit; prints the
   backend's name, so the parent reports which backend the run actually reached. - [never_idle] /
   [never_idle_raise]: no device at all. [Utils.bounded_exit_teardown] from an [at_exit] handler
   over a stand-in stream that never becomes idle and whose teardown would hang for an hour -- the
   hung-device case, which real hardware cannot stage on demand. The second then ends on an uncaught
   exception, which OCaml runs [at_exit] for too. - [idle]: the same stand-in, idle at once, so its
   teardown is due. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

let outcome_name : Utils.exit_teardown_outcome -> string = function
  | Torn_down -> "torn_down"
  | Disabled -> "disabled"
  | Still_busy -> "still_busy"
  | Failed _ -> "failed"

let stand_in ~idle =
  Stdlib.at_exit (fun () ->
      let outcome =
        Utils.bounded_exit_teardown ~what:"a stand-in stream"
          ~is_idle:(fun () -> idle)
          ~teardown:(fun () ->
            Stdio.printf "%s\n%!" Exit_stream_teardown_marker.teardown;
            if not idle then Unix.sleepf 3600.)
      in
      Stdio.printf "outcome: %s\n%!" (outcome_name outcome))

let () =
  match Array.to_list (Sys.get_argv ()) |> List.tl_exn |> List.hd with
  | Some "device" ->
      Tensor.unsafe_reinitialize ();
      let ctx = Context.auto () in
      let%op y = ({ hey = 7.0 } * ([ 2.0 ] : q)) + ([ 1.0 ] : p) in
      let ctx = Train.forward_once ctx y in
      Stdio.printf "backend: %s\n%!" (Context.backend_name ctx)
  | Some "device_busy" ->
      (* Registered before the device is opened, so it runs AFTER the backend's own teardown
         ([at_exit] handlers run most-recent first) and times it. *)
      let exit_began = ref None in
      Stdlib.at_exit (fun () ->
          Option.iter !exit_began ~f:(fun began ->
              Stdio.printf "exit teardown seconds: %.3f\n%!"
                (Mtime.Span.to_float_ns (Mtime_clock.count began) /. 1e9)));
      Tensor.unsafe_reinitialize ();
      let ctx = Context.auto () in
      let n = 1024 in
      let a = TDSL.range_of_shape ~output_dims:[ n ] ~input_dims:[ n ] () in
      let b = TDSL.range_of_shape ~output_dims:[ n ] ~input_dims:[ n ] () in
      let%op c = a * b in
      let ctx, routine = Train.to_routine ctx Train.IDX.empty (Train.forward c) in
      let ctx = Context.run ctx routine in
      Context.sync ctx;
      let started = Mtime_clock.counter () in
      let ctx = Context.run ctx routine in
      Context.sync ctx;
      let one_run = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9 in
      (* Seconds of queued work, far past the bound the driver passes, left unsynced -- on hip, the
         backend that tears its stream down at exit. Elsewhere the driver skips the claim, and a
         synchronous backend would spend those seconds right here. *)
      let runs =
        if String.equal (Context.backend_name ctx) "hip" then
          Int.of_float (Float.round_up (3. /. Float.max one_run 1e-4))
        else 0
      in
      for _ = 1 to runs do
        ignore (Context.run ctx routine : Context.t)
      done;
      Stdio.printf "backend: %s\n%!" (Context.backend_name ctx);
      Stdio.eprintf "queued %d runs of %.4fs each without a sync (not part of the golden)\n%!" runs
        one_run;
      exit_began := Some (Mtime_clock.counter ())
  | Some "never_idle" -> stand_in ~idle:false
  | Some "never_idle_raise" ->
      stand_in ~idle:false;
      failwith "exit_stream_teardown_child: deliberate uncaught exception"
  | Some "idle" -> stand_in ~idle:true
  | _ ->
      failwith
        "exit_stream_teardown_child: expected device|device_busy|never_idle|never_idle_raise|idle"
