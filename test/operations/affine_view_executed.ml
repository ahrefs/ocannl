(* gh-ocannl-1162, the executed half of [affine_coordinate_view]: a lowering-produced flattened
   store and ordinary accesses to the same node, compiled and run through the whole pipeline,
   against a materialized reference.

   A packed [uniform] of shape [2; 9] -> [1] is stored flat ([Row]'s strided projection, lowered as
   [Sub_axis; 4·c; 0]) in [Set_from_vec] runs of four lanes, the last one peeled to the two lanes of
   cells 16-17 — the trailing-lane tail store the per-axis reading answered [Disjoint] against [[h;
   e; 0]]: "mask the last component" masked the trailing unit axis and left the flattened [4·c]
   interpretable. Here it is consumed in the same routine by row-major reads ([y = r + x], [z =
   2r]), so the analyses the coordinate view feeds — coverage, placement, the CPU pool's
   shared-write rule, the GPU schedule's conflict queries — see the flattened store and the ordinary
   reads of one node together. The reference run computes [r] alone, materialized; the value stream
   depends only on the flat element index and the node id, which [Tensor.unsafe_reinitialize] lines
   up, so every [y] and [z] cell must equal the host's arithmetic on the reference [r] — and every
   cell of [x] is distinct, so a read of the wrong [r] cell cannot pass. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

let n_cells = 18

(* [r] is built first, so its node id — and with it the value stream — is the same in both runs. *)
let make_r () =
  Tensor.unsafe_reinitialize ();
  TDSL.uniform () ~input_dims:[ 1 ] ~output_dims:[ 2; 9 ] ()

let build () =
  let r = make_r () in
  let xv = Array.init n_cells ~f:(fun i -> Float.of_int (i + 1) *. 10.) in
  let x = TDSL.ndarray xv ~label:[ "ave_x" ] ~input_dims:[ 1 ] ~output_dims:[ 2; 9 ] () in
  let%op y = r + x in
  let%op z = r *. 2. in
  let%op w = y + z in
  (r, xv, y, z, w)

let () =
  let backend_name = Utils.get_global_arg ~arg_name:"backend" ~default:"cc" in
  Stdio.eprintf "affine_view_executed: backend=%s (not part of the golden)\n%!" backend_name;
  (* The reference: [r] materialized on its own. *)
  let r_ref =
    let r = make_r () in
    let ctx = Context.auto () in
    Ocannl.Train.set_materialized r.value;
    let ctx = Ocannl.Train.forward_once ctx r in
    Context.get_values ctx r.value
  in
  p_all "the reference holds values in [0, 1)" (Array.to_list r_ref) ~f:(fun v ->
      Float.(v >= 0. && v < 1.));
  p "the reference holds 18 pairwise distinct values"
    (List.length (List.dedup_and_sort (Array.to_list r_ref) ~compare:Float.compare) = n_cells);
  let close a b = Float.(abs (a - b) <= 1e-5 * (1. + abs b)) in
  (* Two subjects. With [r] routine-local the optimizer turns the reads into lane extracts of the
     random bits (gh-509), so the flat store meets the row-major reads in the analyses over the raw
     code only; with [r] materialized the store and the reads of one node survive together into the
     optimized kernel, where the schedule's and renderer's conflict queries see them. *)
  let subject ~leg ~materialize_r =
    let r, xv, y, z, w = build () in
    let ctx = Context.auto () in
    if materialize_r then Ocannl.Train.set_materialized r.value;
    Ocannl.Train.set_materialized y.value;
    Ocannl.Train.set_materialized z.value;
    let ctx = Ocannl.Train.forward_once ctx w in
    let yv = Context.get_values ctx y.value and zv = Context.get_values ctx z.value in
    pf "%s: the uniform node is 2x9x1 (a trailing unit axis beside the flattened run)" leg
      (Array.equal Int.equal (Lazy.force r.value.dims) [| 2; 9; 1 |]);
    p_all (leg ^ ": y = r + x in every cell, against the materialized reference")
      (List.range 0 n_cells) ~f:(fun i ->
        Array.length yv = n_cells && close yv.(i) (r_ref.(i) +. xv.(i)));
    p_all (leg ^ ": z = 2r in every cell, against the materialized reference")
      (List.range 0 n_cells) ~f:(fun i ->
        Array.length zv = n_cells && close zv.(i) (2. *. r_ref.(i)));
    p_all (leg ^ ": the tail store's two cells (16, 17) reach both readers") (List.range 16 n_cells)
      ~f:(fun i ->
        Array.length yv = n_cells
        && Array.length zv = n_cells
        && close yv.(i) (r_ref.(i) +. xv.(i))
        && close zv.(i) (2. *. r_ref.(i)))
  in
  subject ~leg:"routine-local r" ~materialize_r:false;
  subject ~leg:"materialized r" ~materialize_r:true
