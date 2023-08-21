open Base
open Ocannl
module CDSL = Code.CDSL
module TDSL = Operation.TDSL

let () = Session.SDSL.set_executor Gccjit

let () =
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let a =
    TDSL.range_of_shape ~batch_dims:[ CDSL.dim 3 ] ~input_dims:[ CDSL.dim 4 ] ~output_dims:[ CDSL.dim 2 ] ()
  in
  let b =
    TDSL.range_of_shape ~batch_dims:[ CDSL.dim 3 ] ~input_dims:[ CDSL.dim 1 ] ~output_dims:[ CDSL.dim 4 ] ()
  in
  let%nn_op c = a *+ "...|i->1; ...|...->i => ...|i" b in
  refresh_session ();
  print_tensor ~with_code:false ~with_grad:false `Default @@ a;
  print_tensor ~with_code:false ~with_grad:false `Default @@ b;
  print_tensor ~with_code:false ~with_grad:false `Default @@ c
