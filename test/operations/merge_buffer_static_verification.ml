open Base
open Ocannl
open Operation.DSL_modules
module Backends = Context.Backends_deprecated
module Idx = Ir.Indexing
module Task = Ir.Task

(* gh-ocannl-288: static verification of merge-buffer nodes "in the right direction".

   [Backend.device_to_device] now returns a transfer *routine* instead of scheduling the copy
   directly. The routine's context statically records the merge-buffer node it produces (in
   [context.merge_buffer_node]). Linking a consumer of the merge buffer against that context
   verifies the node *at link time* -- before any schedule runs -- raising [Utils.User_error] on a
   mismatch. The chaining direction is natural: transfer -> consumer. *)

let () =
  Tensor.unsafe_reinitialize ();
  let backend = Backends.fresh_backend ~backend_name:"sync_cc" () in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:0 in
  let stream = Backend.new_stream device in
  let root = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) stream in

  (* Two tensors with identical shapes, so a shape mismatch cannot mask a node mismatch. *)
  let%op a = [ 1.0; 2.0 ] in
  let%op b = [ 3.0; 4.0 ] in
  Train.set_hosted a.value;
  Train.set_hosted b.value;
  (* Consumer reads b's merge buffer and writes b: its lowered code has merge_node = Some b.value. *)
  let%cd consumer = b =: b.merge in
  let consumer_code = Backend.compile root.optimize_ctx Idx.Empty consumer in

  (* [src] holds both a.value and b.value, initialized from host. *)
  let src = Backend.init_from_host root a.value in
  let src = Backend.init_from_host src b.value in

  (* --- Mismatch path: producer transfers a.value, consumer expects b.value. --- *)
  let ran_schedule = ref false in
  (match Backend.device_to_device a.value ~into_merge_buffer:Copy ~dst:src ~src with
  | None -> Stdio.printf "UNEXPECTED: device_to_device a.value returned None\n"
  | Some transfer_a -> (
      let _observe : Task.t =
        Task.append transfer_a.schedule ~work:(fun () -> ran_schedule := true)
      in
      try
        let _ = Backend.link transfer_a.context consumer_code in
        Stdio.printf "UNEXPECTED: mismatched link did not raise\n"
      with Utils.User_error _ ->
        Stdio.printf "mismatch: link raised at link time, schedule not run = %b\n"
          (not !ran_schedule)));

  (* --- Matched path: producer transfers b.value, consumer expects b.value. --- *)
  (match Backend.device_to_device b.value ~into_merge_buffer:Copy ~dst:src ~src with
  | None -> Stdio.printf "UNEXPECTED: device_to_device b.value returned None\n"
  | Some transfer_b ->
      let consumer_routine = Backend.link transfer_b.context consumer_code in
      Task.run transfer_b.schedule;
      Task.run consumer_routine.schedule;
      Stdio.printf "matched: link succeeded and transfer + consumer schedules ran\n");

  Stdio.printf "done\n"
