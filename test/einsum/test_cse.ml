open Base
open Ocannl
open Nn_blocks.DSL_modules

(* Verifies CSE eliminates duplicate Local_scope computations when the same
   virtualized tensor is consumed multiple times in one expression.
   Uses a reduction (einsum) so the inlined computation involves a loop,
   which simplify_llc preserves as a Local_scope rather than collapsing. *)

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* a is 3x4, b is 4: c = a * b summed over the inner dim => 3-element vector *)
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 4 ] ~output_dims:[ 3 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~output_dims:[ 4 ] () in
  (* c involves a reduction loop (einsum), so its Local_scope won't be simplified away *)
  let%op c = a * b in
  (* d uses c twice => two inline copies => CSE should deduplicate *)
  let%op cse_d = c *. c in
  Train.set_hosted a.value;
  Train.set_hosted b.value;
  ignore (Train.forward_once ctx cse_d);
  (* Print numerical result to verify correctness *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false cse_d;
  Stdio.printf "\n%!"
