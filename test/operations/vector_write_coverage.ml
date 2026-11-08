(* gh-ocannl-2049: the downsample kernel initializer in resnet_block reduces to a uniform of shape
   [3; 1; 1; 2], centered and scaled. Its packed stores cross the minor axis: the four-lane body and
   two-lane tail together fill the node. *)
open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode

let build channels =
  Tensor.unsafe_reinitialize ();
  let r = TDSL.uniform () ~output_dims:[ channels; 1; 1 ] ~input_dims:[ 2 ] () in
  let%op y = 0.5 *. (r - 0.5) in
  Ocannl.Train.set_materialized y.value;
  (r, y)

let run channels ~materialized =
  let r, y = build channels in
  if materialized then Ocannl.Train.set_materialized r.value;
  let ctx = Context.auto () in
  let comp = Ocannl.Train.forward y in
  let opt = Context.lowered_for_decisions ctx comp Ir.Indexing.Empty in
  let lowered = Context.lowered_transform_input opt in
  let delta = Context.link_delta ctx lowered [ lowered ] in
  Stdio.eprintf "channels=%d materialized=%b requested-pool-bytes=%d\n%!" channels materialized
    (delta.ld_working_bytes + delta.ld_constant_bytes);
  let traced = Hashtbl.find_exn opt.LL.traced_store r.value in
  let mode = Tn.Placements.get opt.optimize_ctx.placements r.value in
  Stdio.eprintf "channels=%d materialized=%b uniform=%s read_before_write=%b mode=%s\n%!" channels
    materialized (Tn.debug_name r.value) traced.read_before_write (Tn.debug_memory_mode mode);
  let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx y.value in
  Context.release ctx;
  (values, traced.read_before_write, Tn.Placements.known_virtual opt.optimize_ctx.placements r.value)

let () =
  List.iter [ 2; 3; 4; 5; 7 ] ~f:(fun channels ->
      let reference, _, _ = run channels ~materialized:true in
      let actual, rbw, virtual_ = run channels ~materialized:false in
      pf "%d channels: every uniform cell is written before its read" channels (not rbw);
      pf "%d channels: the uniform leaves device storage" channels virtual_;
      pf "%d channels: the reference distinguishes every cell" channels
        (List.length (List.dedup_and_sort (Array.to_list reference) ~compare:Float.compare)
        = 2 * channels);
      p_all
        (Printf.sprintf "%d channels: reference cannot hide a zeroed producer or output" channels)
        (Array.to_list reference) ~f:(fun v -> Float.(v <> 0. && v <> -0.25));
      p_close (Printf.sprintf "%d channels: executed parity with materialized uniform" channels)
        actual reference ~tol:(fun _ -> 1e-6))
