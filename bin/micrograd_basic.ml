open Base
open Ocannl
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

let _suspended () =
  CDSL.with_debug := true;
  CDSL.keep_files_in_run_directory := true;
  Random.init 0;
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  (* let%op c = c + c + 1 in
     let%op c = c + 1 + c + ~-a in *)
  (* Train.set_fully_on_host g.value;
     Train.set_fully_on_host a.value;
     Train.set_fully_on_host b.value; *)
  Train.every_non_literal_fully_on_host c;
  (* refresh_session ~verbose:true (); *)
  Tensor.print_tree ~with_grad:true ~depth:9 c;
  Stdio.print_endline "\n";
  Tensor.print ~with_code:false ~with_grad:false `Default @@ c;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ b

let () =
  CDSL.with_debug := true;
  CDSL.keep_files_in_run_directory := true;
  Random.init 0;
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  let%op d = (a *. b) + (b **. 3) in
  let%op c = c + c + 1 in
  let%op c = c + 1 + c + ~-a in
  let%op d = d + (d *. 2) + !/(b + a) in
  let%op d = d + (3 *. d) + !/(b - a) in
  let%op e = c - d in
  let%op f = e *. e in
  let%op g = f /. 2 in
  let%op g = g + (10. /. f) in
  (* *
     Train.set_fully_on_host g.value;
     Train.set_fully_on_host a.value;
     Train.set_fully_on_host b.value;
     * *)
  Train.every_non_literal_fully_on_host g;
  (* refresh_session ~verbose:true (); *)
  Tensor.print_tree ~with_grad:true ~depth:9 g;
  Stdio.print_endline "\n";
  Tensor.print ~with_code:false ~with_grad:false `Default @@ g;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ b
