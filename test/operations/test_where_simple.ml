open Base
open Ocannl
open Operation.At
open Nn_blocks.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Simple test: where(true, x, y) should have gradient flow to x only *)
  let x = Tensor.number ~grad_spec:Require_grad 2.0 in
  let y = Tensor.number ~grad_spec:Require_grad 3.0 in
  let cond = Tensor.number 1.0 in
  (* true *)
  let result = Operation.where ~grad_spec:If_needed cond x y () in

  Train.set_materialized x.value;
  Train.set_materialized y.value;
  Train.set_materialized cond.value;
  Train.set_materialized result.value;
  Train.set_materialized (Option.value_exn ~here:[%here] x.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] y.diff).grad;

  let ctx = Train.init_params ctx Train.IDX.empty result in
  let update = Train.grad_update result in
  let routine = Train.to_routine ctx Train.IDX.empty update in

  Train.run ctx routine;

  Stdio.printf "x = %.4g, gradient = %.4g\n" (ctx, x).@[0] (ctx, x).@%[0];
  Stdio.printf "y = %.4g, gradient = %.4g\n" (ctx, y).@[0] (ctx, y).@%[0];
  Stdio.printf "result = %.4g\n" (ctx, result).@[0];
  Stdio.printf "Expected: x gradient = 1.0, y gradient = 0.0\n";

  (* Now test with condition false *)
  Stdio.printf "\nTest 2: where(false, x, y)\n";
  let x2 = Tensor.number ~grad_spec:Require_grad 2.0 in
  let y2 = Tensor.number ~grad_spec:Require_grad 3.0 in
  let cond2 = Tensor.number 0.0 in
  (* false *)
  let result2 = Operation.where ~grad_spec:If_needed cond2 x2 y2 () in

  Train.set_materialized x2.value;
  Train.set_materialized y2.value;
  Train.set_materialized cond2.value;
  Train.set_materialized result2.value;
  Train.set_materialized (Option.value_exn ~here:[%here] x2.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] y2.diff).grad;

  let ctx2 = Train.init_params ctx Train.IDX.empty result2 in
  let update2 = Train.grad_update result2 in
  let routine2 = Train.to_routine ctx2 Train.IDX.empty update2 in

  Train.run ctx routine2;

  Stdio.printf "x = %.4g, gradient = %.4g\n" (ctx, x2).@[0] (ctx, x2).@%[0];
  Stdio.printf "y = %.4g, gradient = %.4g\n" (ctx, y2).@[0] (ctx, y2).@%[0];
  Stdio.printf "result = %.4g\n" (ctx, result2).@[0];
  Stdio.printf "Expected: x gradient = 0.0, y gradient = 1.0\n"
