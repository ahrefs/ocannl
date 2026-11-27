open! Base
open Ocannl.Nn_blocks.DSL_modules

(* Regression test for applying layer_norm to a plain 1-D tensor with no batch axes (a term_init
   tensor whose axis placement is decided only by the Total_elems constraint). Two solver bugs used
   to make this silently compute garbage: [compute_row_product] latched the captured row variable
   [d]'s product to 1 while its tail was still an open row variable (so both [/. dim d] divisions
   vanished), and the singleton-bounds equality heuristic equated the mean's fixed-index output axis
   ([=> ... | 0], At_least_dim 1) with its consumer's axis, resolving it to 4 instead of the
   guess-to-1 broadcast (so the mean was only subtracted at index 0). *)
let () =
  let ctx = Context.auto () in
  let ln = Ocannl.Nn_blocks.layer_norm ~label:[ "ln" ] () in
  let y = Tensor.term_init [| 1.; 2.; 3.; 4. |] ~label:[ "y" ] () in
  let normed = ln y in
  let ctx = Ocannl.Train.forward_once ctx normed in
  Stdio.printf "1-D layer_norm output shape:\n%s\n%!" (Shape.to_string_hum normed.Tensor.shape);
  (* layer_norm([1;2;3;4]) with gamma=1, beta=0: (x - 2.5) / sqrt(1.25 + eps) = [-1.342; -0.447;
     0.447; 1.342]. *)
  Ocannl.Train.printf ~here:[%here] ~with_grad:false ctx normed
