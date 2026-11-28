open! Base
open Ocannl.Nn_blocks.DSL_modules

(* Regression test for the "divided einsum result" LayerNorm formulation: dividing the reduction
   *result* by the reduced dimension (rather than scaling the reduction operand) interposes a
   pointwise-division broadcast hop between the einsum output and its consumer. The fixed-index
   output axis (`=> ... | 0`) then has to be guessed to 1 at the final inference stage; the solver
   used to drop the mean's Shape_row constraint when its row variable was closed via a GLB at stage
   6, leaving the axis variable unresolved ("Not enough shape information: unresolved variable").

   The failing context needs inline params both inside the norm (gamma, beta) and around it, with
   two chained layer norms. *)
let%op layer_norm ~label ?(epsilon = 1e-5) () x =
  let mean = (x ++ " ... | ..d..  => ... | 0 " [ "d" ]) /. dim d in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ " ... | ... => ... |  0 ") /. dim d in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  ({ gamma = 1. } *. normalized) + { beta = 0. }

let () =
  let ctx = Context.auto () in
  (* Chained layer norms with inline params between them: the minimal context that used to fail. *)
  let ln1 = layer_norm ~label:[ "ln1" ] () in
  let ln2 = layer_norm ~label:[ "ln2" ] () in
  let%op block x =
    let x1 = ln1 (x + ({ w_o } * x)) in
    ln2 (x1 + ({ w_2 } * x1))
  in
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ 2; 8 ] ~input_dims:[] ~output_dims:[ 16 ] ()
  in
  let output = block x in
  let ctx = Ocannl.Train.forward_once ctx output in
  Stdio.printf "Chained layer norms output shape:\n%s\n%!" (Shape.to_string_hum output.Tensor.shape);
  (* Numeric check of the formulation itself: layer_norm of the row [1;2;3;4] (and of [5;6;7;8])
     with gamma=1, beta=0 is (x - mean) / sqrt(1.25 + eps) = [-1.342; -0.447; 0.447; 1.342]. *)
  let ln3 = layer_norm ~label:[ "ln3" ] () in
  let y =
    NTDSL.init ~l:"y" ~prec:Ir.Ops.single ~b:[ 2 ] ~i:[] ~o:[ 4 ]
      ~f:(function
        | [| b; o |] -> Float.of_int ((b * 4) + o + 1)
        | idcs ->
            failwith @@ "unexpected indices " ^ Sexp.to_string_hum ([%sexp_of: int array] idcs))
      ()
  in
  let normed = ln3 y in
  let ctx = Ocannl.Train.forward_once ctx normed in
  Ocannl.Train.printf ~here:[%here] ~with_grad:false ctx normed
