open! Base
open Ocannl
open Nn_blocks.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in

  (* Hey is inferred to be a matrix because of matrix multiplication [*]. *)
  let%op y = ({ hey = 7.0 } * 'q' 2.0) + 'p' 1.0 in
  let _ctx = Train.forward_once ctx y in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey
