open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module SDSL = Session.SDSL

let () = SDSL.set_executor Gccjit

let () =
  SDSL.drop_all_sessions ();
  SDSL.set_executor Cuda;
  Code.with_debug := true;
  Code.keep_files_in_run_directory := true;
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
  SDSL.set_fully_on_host g;
  SDSL.set_fully_on_host a;
  SDSL.set_fully_on_host b;
  SDSL.refresh_session ();
  SDSL.print_node_tree ~with_grad:true ~depth:9 g.id;
  SDSL.print_formula ~with_code:false ~with_grad:false `Default @@ g;
  SDSL.print_formula ~with_code:false ~with_grad:true `Default @@ a;
  SDSL.print_formula ~with_code:false ~with_grad:true `Default @@ b
