open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

let () =
  let inputs = [| 0.0; 0.5; 1.0 |] in
  let x =
    Operation.init ~l:"x" ~prec:Ir.Ops.single ~o:[ 3 ]
      ~f:(fun i -> inputs.(i.(0)))
      ~grad_spec:Tensor.Require_grad ()
  in
  (* A nonzero payload makes an open gate at either endpoint visible in the gradient. *)
  let%op loss = (7.0 *. sat01 x) ++ "a => 0" in
  let grad = (Option.value_exn ~here:[%here] x.Tensor.diff).grad in
  Train.set_materialized x.value;
  Train.set_materialized grad;
  let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
  p_all2 "sat01 backward passes the upstream gradient only inside (0, 1)"
    (Context.get_values ctx grad) [| 0.0; 7.0; 0.0 |] ~f:Float.equal
