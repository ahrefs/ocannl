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
  let%nn_op f = "a" [2] *. "b" [-3] + "c" [10] in
  refresh_session ();
  Stdio.printf "\n%!";
  print_node_tree ~with_grad:true ~depth:9 f.id;
  Stdio.printf "\n%!"
