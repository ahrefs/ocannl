open Base
open Ocannl
open Operation.DSL_modules
module Backends = Context.Backends_deprecated
module Idx = Ir.Indexing
module Task = Ir.Task
module Tn = Ir.Tnode

(* gh-ocannl-288: static verification of merge-buffer nodes "in the right direction".

   [Backend.device_to_device] now returns a transfer *routine* instead of scheduling the copy
   directly. The routine's context statically records the merge-buffer node it produces (in
   [context.merge_buffer_node]). Linking a consumer of the merge buffer against that context
   verifies the node *at link time* -- before any schedule runs -- raising [Utils.User_error] on a
   mismatch. The chaining direction is natural: transfer -> consumer. *)

(* A hosted, gradient-free tensor of shape [n] initialized from [vals]. *)
let make_tensor label vals =
  let open Bigarray in
  let n = Array.length vals in
  let ga = Genarray.create Float32 c_layout [| n |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[] ~input_dims:[] ~output_dims:[ n ] ()

let () =
  Tensor.unsafe_reinitialize ();
  let backend = Backends.fresh_backend ~backend_name:"sync_cc" () in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:0 in
  let stream = Backend.new_stream device in
  let root = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) stream in

  (* Three hosted tensors with identical shapes. [b] carries the data to transfer; [out] starts at
     a distinct value so an inert (non-copying) pipeline would be observable. *)
  let a = make_tensor "a" [| 1.0; 2.0 |] in
  let b = make_tensor "b" [| 3.0; 4.0 |] in
  let out = make_tensor "out" [| 9.0; 9.0 |] in
  (* Consumer reads b's merge buffer and writes out: its lowered code has merge_node = Some
     b.value. *)
  let%cd consumer = out =: b.merge in
  let consumer_code = Backend.compile root.optimize_ctx Idx.Empty consumer in

  (* [src] holds a.value, b.value and out.value, initialized from host. After gh-ocannl-333 the host
     buffer is supplied explicitly; here it comes from each literal's registered init data. *)
  let host_of (tn : Tn.t) = Lazy.force (Option.value_exn ~here:[%here] (Ir.Host_inits.find tn)) in
  let src = Backend.init_from_host root a.value (host_of a.value) in
  let src = Backend.init_from_host src b.value (host_of b.value) in
  let src = Backend.init_from_host src out.value (host_of out.value) in

  (* --- Mismatch path: producer transfers a.value, consumer expects b.value. --- *)
  (match Backend.device_to_device a.value ~into_merge_buffer:Copy ~dst:src ~src with
  | None -> Stdio.printf "UNEXPECTED: device_to_device a.value returned None\n"
  | Some transfer_a -> (
      (* device_to_device must NOT have eagerly scheduled the copy: the old implementation set the
         stream's [updating_for_merge_buffer] immediately via [update_writer_event]. With the
         routine-returning form it stays unset until [transfer_a.schedule] is run. *)
      Stdio.printf "no eager side effect (updating_for_merge_buffer still None) = %b\n"
        (Option.is_none src.stream.updating_for_merge_buffer);
      try
        let _ = Backend.link transfer_a.context consumer_code in
        Stdio.printf "UNEXPECTED: mismatched link did not raise\n"
      with Utils.User_error _ -> Stdio.printf "mismatch: link raised at link time\n"));

  (* --- Matched path: producer transfers b.value, consumer expects b.value. --- *)
  (match Backend.device_to_device b.value ~into_merge_buffer:Copy ~dst:src ~src with
  | None -> Stdio.printf "UNEXPECTED: device_to_device b.value returned None\n"
  | Some transfer_b ->
      let consumer_routine = Backend.link transfer_b.context consumer_code in
      (* End to end: transfer copies b.value ([3 4]) into the merge buffer, the consumer copies it
         back out into out.value (initialized to [9 9]). *)
      Task.run transfer_b.schedule;
      Task.run consumer_routine.schedule;
      Backend.await stream;
      let out_nd = host_of out.value in
      ignore (Backend.to_host consumer_routine.context out.value out_nd : bool);
      Backend.await stream;
      let vals = Ir.Ndarray.retrieve_flat_values out_nd in
      Stdio.printf "matched: out (init [9 9]) after transfer + consumer = [%s]\n"
        (String.concat ~sep:" " (Array.to_list (Array.map vals ~f:(Printf.sprintf "%g")))));

  Stdio.printf "done\n"
