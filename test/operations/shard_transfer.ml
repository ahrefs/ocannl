open Base
open Ocannl
open Operation.DSL_modules
module Backends = Context.Backends
module Idx = Ir.Indexing
module Task = Ir.Task
module Tn = Ir.Tnode

(* Regression test for the data-parallel sharding primitives (task-2445dd1c).

   Part 1 exercises the host-level [Parallel.shard_along] / [Parallel.gather] round-trip: splitting
   a batch tensor into shards and gathering must reconstruct the original row order exactly, and
   invalid configurations must raise; a shard's label, like [Train.forward]'s block comment, spells
   its source's [label_name] even once a repeated label has moved the source's debug name.

   Part 2 exercises the raw-backend merge-buffer all-reduce that [Parallel.grad_sync] is built on:
   two context trees hold distinct values of the same tnode; a [device_to_device
   ~into_merge_buffer:Copy] transfer routine plus an accumulating consumer ([g =+ g.merge]) must sum
   them into the owner context. This is the cross-context channel that survived gh-ocannl-341. *)

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
  ignore (Parallel.host_values batch : float array);
  let shards = Parallel.shard_along ~axis:0 ~n_shards:3 batch in
  Stdio.printf "n_shards=%d\n" (Array.length shards);
  Array.iteri shards ~f:(fun i s ->
      let vs = Parallel.host_values s in
      Stdio.printf "shard %d = [%s]\n" i
        (String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%g")))));
  let gathered = Parallel.gather ~axis:0 shards in
  let gv = Parallel.host_values gathered in
  Stdio.printf "gathered = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map gv ~f:(Printf.sprintf "%g"))));
  let original = Parallel.host_values batch in
  Verdict.p "round-trip identity" (Array.equal Float.equal original gv);

  (* Invalid configurations must raise. *)
  let raises f =
    try
      f ();
      false
    with Invalid_argument _ -> true
  in
  Verdict.p "n_shards<=0 raises"
    (raises (fun () -> ignore (Parallel.shard_along ~axis:0 ~n_shards:0 batch)));
  Verdict.p "uneven batch raises"
    (raises (fun () -> ignore (Parallel.shard_along ~axis:0 ~n_shards:4 batch)));
  Verdict.p "axis<>0 raises"
    (raises (fun () -> ignore (Parallel.shard_along ~axis:1 ~n_shards:2 batch)));

  (* Names stored, not shown, spell [label_name] (post-wave task post-1-task-6, after
     gh-ocannl-1686): a node whose label its routine repeats is minted the [n<id>_<label>] code
     name, and the debug name follows it -- but [shard_along]'s shard labels and [Train.forward]'s
     block comment are what an unminted twin with the same label gets, so they do not depend on what
     a renderer minted first. The minting is forced as the renderer does it for a repeated label
     ([Tn.styled_ident] under a repeating-idents table naming it). *)
  let minted = make_batch "rows" [| 0.; 1.; 2.; 3. |] in
  let twin = make_batch "rows" [| 0.; 1.; 2.; 3. |] in
  let label_name = Tn.label_name minted.Tensor.value in
  let repeating = Hashtbl.create (module String) in
  Hashtbl.set repeating ~key:label_name ~data:();
  Tn.update_code_name minted.Tensor.value
    (Tn.styled_ident ~repeating_nograd_idents:repeating
       ~repeating_grad_idents:(Hashtbl.create (module String))
       (`Heuristic_ocannl `Under_grad) minted.Tensor.value);
  let shown =
    (Tn.debug_name minted.Tensor.value
     [@rendered_name "the precondition compares the minted display name with label_name"])
  in
  Verdict.p "precondition: the repeated label moved the debug name away from label_name"
    (String.equal (Tn.label_name twin.Tensor.value) label_name
    && not (String.equal shown label_name));
  let shard_labels t =
    Array.to_list
      (Array.map (Parallel.shard_along ~axis:0 ~n_shards:2 t) ~f:(fun s ->
           Tn.label_name s.Tensor.value))
  in
  Verdict.p "shard_along labels a minted node's shards as it labels its unminted twin's"
    (List.equal String.equal (shard_labels minted) (shard_labels twin));
  Verdict.p_all "each shard label of the minted node starts with label_name, not the debug name"
    (shard_labels minted) ~f:(fun l ->
      String.is_prefix l ~prefix:label_name && not (String.is_substring l ~substring:shown));
  let fwd_comment t =
    match (Train.forward t).Ir.Assignments.asgns with
    | Ir.Assignments.Block_comment (s, _) -> Some s
    | _ -> None
  in
  let minted_comment = fwd_comment minted in
  Verdict.p "Train.forward comments a minted node's block as its unminted twin's, from label_name"
    (Option.is_some minted_comment
    && Option.equal String.equal minted_comment (fwd_comment twin)
    && Option.for_all minted_comment ~f:(fun s ->
        String.is_prefix s ~prefix:label_name && not (String.is_substring s ~substring:shown)));

  (* --- Part 2: raw-backend merge-buffer all-reduce (grad_sync core) --- *)
  let backend = Backends.backend_module (Backends.get_backend ~backend_name:"cc" ()) in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:0 in
  (* Note: [g]/[m] are reserved shorthands in %cd, so use descriptive names. *)
  let owner_g = make_vec "owner_g" [| 0.; 0. |] in
  let tmp = make_vec "tmp" [| 0.; 0. |] in
  Train.set_materialized owner_g.Tensor.value;
  Train.set_materialized tmp.Tensor.value;
  (* After gh-ocannl-333 host data is supplied explicitly to the backend transfers. *)
  let host_vec vals =
    let nd =
      Ir.Ndarray.create_array ~debug:"shard_transfer host" Ir.Ops.single
        ~dims:[| Array.length vals |]
        ~padding:None
    in
    Ir.Ndarray.set_flat_values nd vals;
    nd
  in
  (* Owner (ctx0) starts at [1 2]; source (ctx1) holds [3 4] -- two independent context trees on the
     same device. *)
  let ctx1 = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) device in
  let ctx1 = Backend.init_from_host ctx1 owner_g.Tensor.value (host_vec [| 3.; 4. |]) in
  let ctx0 = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) device in
  let ctx0 = Backend.init_from_host ctx0 owner_g.Tensor.value (host_vec [| 1.; 2. |]) in
  (* All-reduce ctx1's value into ctx0 via the merge buffer: owner = [1 2] + [3 4] = [4 6]. The
     merge buffer is array-level (no shape inference), so it is copied into a pre-shaped temp [tmp]
     and then accumulated into the owner. *)
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
  Backend.await device;
  let out_nd = host_vec [| 0.; 0. |] in
  ignore (Backend.to_host ctx0 owner_g.Tensor.value out_nd : bool);
  Backend.await device;
  let summed = Ir.Ndarray.retrieve_flat_values out_nd in
  Stdio.printf "all-reduce sum = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map summed ~f:(Printf.sprintf "%g"))));
  Stdio.printf "done\n"
