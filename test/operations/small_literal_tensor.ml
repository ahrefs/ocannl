open! Base
open! Ocannl
open! Nn_blocks.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op hey = [ (1, 2, 3); (4, 5, 6) ] in
  let ctx = Train.forward_once ctx hey in
  (* ignore (failwith @@ Tn.debug_memory_mode hey.value.memory_mode); *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline hey;
  let%op hoo = [| [ 1; 2; 3 ]; [ 4; 5; 6 ] |] in
  let _ctx = Train.forward_once ctx hoo in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline hoo
