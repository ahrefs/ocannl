open Base
open Ocannl

module FDSL = Operation.FDSL

let () = Session.SDSL.set_executor OCaml

let () =
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let a =
    FDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[2] () in
  let b =
    FDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[3] () in
  let%nn_op c = (a + 1) *+"i; j => i->j" b in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ a;
  print_formula ~with_code:false ~with_grad:false `Default @@ b;
  print_formula ~with_code:false ~with_grad:false `Default @@ c
