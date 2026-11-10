(* gh-ocannl-2024: the row GLB's two-hop reach, pinned as policy on executed programs. A draw
   reaches its use site through a chain. Each case prints the draw's shape and how many distinct
   values it holds against how many cells its use site has. The row-level stores, where the reach
   shows, are test/einsum/test_row_glb_reach.ml.

   What these pin: an executed repeat is gh-544's, not the two-hop reach. An operation result closes
   down to its arguments' shapes and its use site broadcasts it in, so a draw behind [u + 1] or [(u
   *. 2) *. 2] meets a result that never widened. Making a row's GLB travel transitively, or into
   use-site rows, was built and compared under gh-ocannl-2024 and changed none of these draws. A
   chain of [stretch] results, each resolving at its use site, fills at any depth. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train

let context () =
  let ctx = Context.auto () in
  Stdio.eprintf "(not part of the golden) backend: %s\n%!" (Context.backend_name ctx);
  ctx

let values n =
  Ir.Ndarray.init_array ~debug:"c" Ir.Ops.single ~dims:[| n |] ~padding:None ~f:(fun idx ->
      Float.of_int idx.(0))

let unique v = List.length (List.dedup_and_sort ~compare:Float.compare (Array.to_list v))
let cells t = Array.fold (Shape.to_dims t.Tensor.shape) ~init:1 ~f:( * )

(* The draw's shape, its distinct values, and its use site's cells. [build] returns the draw, its
   use site, and the tensor to run. *)
let observe build =
  Tensor.unsafe_reinitialize ();
  let u, use, root = build () in
  Train.set_materialized u.Tensor.value;
  let ctx = Train.forward_once (context ()) root in
  (Shape.to_string_hum u.Tensor.shape, unique (Context.get_values ctx u.Tensor.value), cells use)

let case name ~fills build =
  Verdict.case name @@ fun () ->
  let shape, n, total = observe build in
  Stdio.printf "%s\n  u = %s, %d distinct values for %d cells\n" name shape n total;
  if fills then pf "%s: the draw fills its use site" name (n = total)
  else pf "%s: the draw repeats along its use site (gh-544's close-down)" name (n < total)

let plus1 r =
  let%op r2 = r + !.1. in
  r2

let times2 r =
  let%op r2 = r *. !.2. in
  r2

let stretched r =
  let%op s = stretch r in
  s

let c53 () = NTDSL.reshape ~l:"c" ~b:[ 5 ] ~o:[ 3 ] (values 15) ()

let () =
  case "u *. c[5|3]" ~fills:true (fun () ->
      let u = TDSL.uniform () () in
      let%op y = u *. c53 () in
      (u, y, y));
  case "(u + 1) *. c[5|3]" ~fills:false (fun () ->
      let u = TDSL.uniform () () in
      let%op y = plus1 u *. c53 () in
      (u, y, y));
  case "stretch (stretch (stretch u)) *. c[5|3]" ~fills:true (fun () ->
      let u = TDSL.uniform () () in
      let%op y = stretched (stretched (stretched u)) *. c53 () in
      (u, y, y));
  (* gh-ocannl-1961's executed shape: the use site's bound arrives at the final stage, through a
     fixed index, and a chain of two operation results stands between it and the draw. *)
  case "(u *. 2) *. 2, then a fixed index, batch 4" ~fills:false (fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
      let m = times2 (times2 u) in
      let%op y = m ++ "...|3 => ...|0" in
      (u, m, y));
  case "u *. 2, then a fixed index, batch 4" ~fills:true (fun () ->
      let u = TDSL.uniform () ~batch_dims:[ 4 ] () in
      let m = times2 u in
      let%op y = m ++ "...|3 => ...|0" in
      (u, m, y))
