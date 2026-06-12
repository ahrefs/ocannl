open Base
open Ocannl
open Operation.DSL_modules
module Backends = Context.Backends_deprecated
module Idx = Ir.Indexing
module Task = Ir.Task
module Tn = Ir.Tnode

(* Regression test for the data-parallel sharding primitives (task-2445dd1c).

   Part 1 exercises the host-level [Parallel.shard_along] / [Parallel.gather] round-trip: splitting a
   batch tensor into shards and gathering must reconstruct the original row order exactly, and
   invalid configurations must raise.

   Part 2 exercises the raw-backend merge-buffer all-reduce that [Parallel.grad_sync] is built on:
   two streams hold distinct values of the same tnode; a [device_to_device ~into_merge_buffer:Copy]
   transfer routine plus an accumulating consumer ([g =+ g.merge]) must sum them on the owner
   stream. This is the cross-stream channel that survived gh-ocannl-341. *)

let make_batch label vals =
  (* A batch-major tensor of shape [n] along the batch axis, values from [vals]. *)
  let open Bigarray in
  let n = Array.length vals in
  let ga = Genarray.create Float32 c_layout [| n |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  TDSL.rebatch ~l:label nd ()

let make_vec label vals =
  (* A gradient-free tensor with a single output axis of size [n] (matches the merge-buffer
     static-verification idiom). *)
  let open Bigarray in
  let n = Array.length vals in
  let ga = Genarray.create Float32 c_layout [| n |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[] ~input_dims:[] ~output_dims:[ n ] ()

let () =
  Tensor.unsafe_reinitialize ();

  (* --- Part 1: shard_along / gather round-trip --- *)
  let batch = make_batch "batch" [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  ignore (Tn.get_values batch.Tensor.value : float array);
  let shards = Parallel.shard_along ~axis:0 ~n_shards:3 batch in
  Stdio.printf "n_shards=%d\n" (Array.length shards);
  Array.iteri shards ~f:(fun i s ->
      let vs = Tn.get_values s.Tensor.value in
      Stdio.printf "shard %d = [%s]\n" i
        (String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%g")))));
  let gathered = Parallel.gather ~axis:0 shards in
  let gv = Tn.get_values gathered.Tensor.value in
  Stdio.printf "gathered = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map gv ~f:(Printf.sprintf "%g"))));
  let original = Tn.get_values batch.Tensor.value in
  Stdio.printf "round-trip identity = %b\n" (Array.equal Float.equal original gv);

  (* Invalid configurations must raise. *)
  let raises f = try f (); false with Invalid_argument _ -> true in
  Stdio.printf "n_shards<=0 raises = %b\n"
    (raises (fun () -> ignore (Parallel.shard_along ~axis:0 ~n_shards:0 batch)));
  Stdio.printf "uneven batch raises = %b\n"
    (raises (fun () -> ignore (Parallel.shard_along ~axis:0 ~n_shards:4 batch)));
  Stdio.printf "axis<>0 raises = %b\n"
    (raises (fun () -> ignore (Parallel.shard_along ~axis:1 ~n_shards:2 batch)));

  (* --- Part 2: raw-backend merge-buffer all-reduce (grad_sync core) --- *)
  let backend = Backends.fresh_backend ~backend_name:"sync_cc" () in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:0 in
  let stream0 = Backend.new_stream device in
  let stream1 = Backend.new_stream device in
  (* Note: [g]/[m] are reserved shorthands in %cd, so use descriptive names. *)
  let owner_g = make_vec "owner_g" [| 0.; 0. |] in
  let tmp = make_vec "tmp" [| 0.; 0. |] in
  Train.set_hosted owner_g.Tensor.value;
  Train.set_materialized tmp.Tensor.value;
  (* Owner (stream0) starts at [1 2]; source (stream1) holds [3 4]. *)
  Tn.set_values owner_g.Tensor.value [| 3.; 4. |];
  let ctx1 = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) stream1 in
  let ctx1 = Backend.init_from_host ctx1 owner_g.Tensor.value in
  Tn.set_values owner_g.Tensor.value [| 1.; 2. |];
  let ctx0 = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) stream0 in
  let ctx0 = Backend.init_from_host ctx0 owner_g.Tensor.value in
  (* All-reduce stream1's value into stream0 via the merge buffer: owner = [1 2] + [3 4] = [4 6].
     The merge buffer is array-level (no shape inference), so it is copied into a pre-shaped temp
     [tmp] and then accumulated into the owner. *)
  let%cd copy_merge = tmp =: owner_g.merge in
  let%cd add_temp = owner_g =+ tmp in
  let copy_code = Backend.compile (Backend.empty_optimize_ctx ()) Idx.Empty copy_merge in
  let add_code = Backend.compile (Backend.empty_optimize_ctx ()) Idx.Empty add_temp in
  (match
     Backend.device_to_device owner_g.Tensor.value ~into_merge_buffer:Copy ~dst:ctx0 ~src:ctx1
   with
  | None -> Stdio.printf "UNEXPECTED: device_to_device returned None\n"
  | Some transfer ->
      Task.run transfer.schedule;
      (* [copy_merge] consumes the merge buffer recorded on [transfer.context]; [add_temp]
         accumulates the temp into the owner gradient. *)
      let copy_routine = Backend.link transfer.context copy_code in
      Task.run copy_routine.schedule;
      let add_routine = Backend.link copy_routine.context add_code in
      Task.run add_routine.schedule);
  Backend.await stream0;
  ignore (Backend.to_host ctx0 owner_g.Tensor.value : bool);
  Backend.await stream0;
  let summed = Tn.get_values owner_g.Tensor.value in
  Stdio.printf "all-reduce sum = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map summed ~f:(Printf.sprintf "%g"))));
  Stdio.printf "done\n"
