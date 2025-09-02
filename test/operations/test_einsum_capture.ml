open Base
open Ocannl

let () =
  let module TDSL = Operation.TDSL in
  let module PDSL = Operation.PDSL in
  let module Backend = (val Backends.fresh_backend ()) in
  let%op x = { x = uniform1 (); o = [ 2; 3 ] } in
  let%op y = { y = uniform1 (); o = [ 3; 4 ] } in
  let%op z = x *+ "ab;bc=>ac" [ "a"; "b"; "c" ] y in

  (* Trigger shape inference by accessing the tensor node *)
  let ctx = Train.forward_once (module Backend) z in

  (* Check if dimensions were captured *)
  Stdio.printf "Dimension a: %s\n"
    (match a.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "Dimension b: %s\n"
    (match b.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "Dimension c: %s\n"
    (match c.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  let%op x2 = { x2 = uniform1 (); o = [ 5; 7 ] } in
  (* Manually call einsum1 with capture_dims for now *)
  let%op y2 = x2 ++ "ij=>ji" [ "i"; "j" ] in

  (* Trigger shape inference by accessing the tensor node *)
  let ctx = Train.forward_once (module Backend) ~ctx y2 in

  (* Check if dimensions were captured *)
  Stdio.printf "Dimension i: %s\n"
    (match i.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "Dimension j: %s\n"
    (match j.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  (* Test capturing row variables *)
  let%op x3 = { x3 = uniform1 (); o = [ 2; 3; 4 ] } in
  let%op y3 = { y3 = uniform1 (); o = [ 3; 4; 5 ] } in
  let%op z3 = x3 *+ "a..r..;..r..b=>ab" [ "r" ] y3 in
  
  (* Trigger shape inference *)
  let ctx = Train.forward_once (module Backend) ~ctx z3 in
  
  (* Check if row variable was captured *)
  Stdio.printf "Row variable r (product of dims): %s\n"
    (match r.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  
  let%op dim_calc = dim a + dim j + dim r in
  let _ctx = Train.forward_once (module Backend) ~ctx dim_calc in

  Train.printf ~here:[%here] ~with_code:false ~with_grad:false dim_calc
