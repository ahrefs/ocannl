open Base
open Ocannl
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL



let () =

  (* drop_all_sessions (); *)
  Random.init 0;
  let a =
    TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 4 ] ~output_dims:[ 2 ] ()
  in
  let b =
    TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 1 ] ~output_dims:[ 4 ] ()
  in
  let%nn_op c = a *+ "...|i->1; ...|...->i => ...|i" b in
  (* refresh_session (); *)
  Tensor.print ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ c
