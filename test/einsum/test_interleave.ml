open! Base
open! Ocannl
open! Nn_blocks.DSL_modules

let () =
  (* FIXME: NOT IMPLEMENTED YET *)
  Tensor.unsafe_reinitialize ();

  (* let%op t1 = [ 1.0; 2.0; 3.0 ] in
  let%op t2 = [ 4.0; 5.0; 6.0 ] in
  let t3 = Operation.interleave t1 t2 () in *)
  (* let%op t3 = interleave [ 1.0; 2.0; 3.0 ] [ 4.0; 5.0; 6.0 ] in *)

  (* t3 should be [1.0; 4.0; 2.0; 5.0; 3.0; 6.0] *)
  (* let ctx = Context.auto () in *)

  (* let _ctx = Train.forward_once ctx t3 in *)
  (* Train.printf ~here:[%here] ~with_code:false ~with_grad:false t3 *)
  ()
