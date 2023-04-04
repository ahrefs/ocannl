open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL

let () = Session.SDSL.set_executor OCaml

let () =
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let%nn_op n = "w" [-3, 1] * "x" [2; 0] + "b" [6.7] in
  refresh_session ();
  Stdio.printf "\n%!";
  print_preamble ();
  Stdio.printf "\n%!";
  print_node_tree ~with_grad:true ~depth:9 n.id;
  Stdio.printf "\n%!";
  print_formula ~with_grad:false ~with_code:true `Default n;
  Stdio.printf "\n%!"
