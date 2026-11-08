(* gh-ocannl-2024: executed chains, compared across [Row.row_glb_transport]. A draw reaches its use
   site through a chain; each case prints the draw's shape and how many distinct values it holds
   against how many cells its use site has, under each transport. The row-level comparison, where
   the transports differ, is test/einsum/test_row_glb_transport.ml.

   What these pin: the repeats are gh-544's, not the two-hop reach. An operation result closes down
   to its arguments' shapes and its use site broadcasts it in, so a draw behind [u + 1] or [(u *. 2)
   *. 2] meets a result that never widened, under every transport. A chain of [stretch] results,
   each resolving at its use site, fills at any depth under every transport. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module Row = Ocannl.Row

let context () =
  let ctx = Context.auto () in
  Stdio.eprintf "(not part of the golden) backend: %s\n%!" (Context.backend_name ctx);
  ctx

let values n =
  Ir.Ndarray.init_array ~debug:"c" Ir.Ops.single ~dims:[| n |] ~padding:None ~f:(fun idx ->
      Float.of_int idx.(0))

let unique v = List.length (List.dedup_and_sort ~compare:Float.compare (Array.to_list v))
let cells t = Array.fold (Shape.to_dims t.Tensor.shape) ~init:1 ~f:( * )

let transports : (string * Row.row_glb_transport) list =
  [ ("two hops", Two_hops); ("use site", Use_site); ("transitive", Transitive) ]

(* The draw's shape, its distinct values, and its use site's cells, under one transport. [build]
   returns the draw, its use site, and the tensor to run. *)
let observe transport build =
  Row.with_row_glb_transport transport @@ fun () ->
  Tensor.unsafe_reinitialize ();
  let u, use, root = build () in
  Train.set_materialized u.Tensor.value;
  let ctx = Train.forward_once (context ()) root in
  (Shape.to_string_hum u.Tensor.shape, unique (Context.get_values ctx u.Tensor.value), cells use)

let case name ~fills build =
  Verdict.case name @@ fun () ->
  let seen = List.map transports ~f:(fun (t, transport) -> (t, observe transport build)) in
  Stdio.printf "%s\n" name;
  List.iter seen ~f:(fun (t, (shape, n, total)) ->
      Stdio.printf "  %-10s u = %s, %d distinct values for %d cells\n" t shape n total);
  let outcomes = List.map seen ~f:snd in
  p_all (name ^ ": every transport draws the same") outcomes ~f:(Poly.equal (List.hd_exn outcomes));
  let _, n, total = List.hd_exn outcomes in
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
