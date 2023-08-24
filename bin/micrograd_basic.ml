open Base
open Ocannl
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL




let _suspended () =
  (* SDSL.drop_all_sessions (); *)

  CDSL.with_debug := true;
  CDSL.keep_files_in_run_directory := true;
  Random.init 0;
  let%nn_op c = "a" [ -4 ] + "b" [ 2 ] in
  (* let%nn_op c = c + c + 1 in
     let%nn_op c = c + 1 + c + ~-a in *)
  (* Tensor.set_fully_on_host g;
     Tensor.set_fully_on_host a;
     Tensor.set_fully_on_host b; *)
  (* everything_fully_on_host (); *)
  (* refresh_session ~verbose:true (); *)
  Tensor.print_tree ~with_grad:true ~depth:9 c;
  Stdio.print_endline "\n";
  Tensor.print ~with_code:false ~with_grad:false `Default @@ c;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ b

let () =
  (* SDSL.drop_all_sessions (); *)

  CDSL.with_debug := true;
  CDSL.keep_files_in_run_directory := true;
  Random.init 0;
  let%nn_op c = "a" [ -4 ] + "b" [ 2 ] in
  let%nn_op d = (a *. b) + (b **. 3) in
  let%nn_op c = c + c + 1 in
  let%nn_op c = c + 1 + c + ~-a in
  let%nn_op d = d + (d *. 2) + !/(b + a) in
  let%nn_op d = d + (3 *. d) + !/(b - a) in
  let%nn_op e = c - d in
  let%nn_op f = e *. e in
  let%nn_op g = f /. 2 in
  let%nn_op g = g + (10. /. f) in
  (* *)
  Tensor.set_fully_on_host g;
  Tensor.set_fully_on_host a;
  Tensor.set_fully_on_host b;
  (* *)
  (* (* everything_fully_on_host (); *) *)
  (* refresh_session ~verbose:true (); *)
  Tensor.print_tree ~with_grad:true ~depth:9 g;
  Stdio.print_endline "\n";
  Tensor.print ~with_code:false ~with_grad:false `Default @@ g;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ b
