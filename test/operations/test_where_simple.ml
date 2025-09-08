open Base
open Ocannl
open Operation.At
open Operation.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* Simple test: where(true, x, y) should have gradient flow to x only *)
  let x = Tensor.term_init [| 2.0 |] ~grad_spec:Require_grad () in
  let y = Tensor.term_init [| 3.0 |] ~grad_spec:Require_grad () in
  let cond = Tensor.number 1.0 in
  (* true *)
  let result = Operation.where ~grad_spec:If_needed cond x y () in

  Train.set_hosted x.value;
  Train.set_hosted y.value;
  Train.set_hosted cond.value;
  Train.set_hosted result.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] y.diff).grad;

  let ctx = Train.init_params ctx Train.IDX.empty result in
  let update = Train.grad_update result in
  let routine = Train.to_routine ctx Train.IDX.empty update in

  Train.run ctx routine;

  Stdio.printf "x = %.4g, gradient = %.4g\n" x.@[0] x.@%[0];
  Stdio.printf "y = %.4g, gradient = %.4g\n" y.@[0] y.@%[0];
  Stdio.printf "result = %.4g\n" result.@[0];
  Stdio.printf "Expected: x gradient = 1.0, y gradient = 0.0\n";

  (* Now test with condition false *)
  Stdio.printf "\nTest 2: where(false, x, y)\n";
  let x2 = Tensor.term_init [| 2.0 |] ~grad_spec:Require_grad () in
  let y2 = Tensor.term_init [| 3.0 |] ~grad_spec:Require_grad () in
  let cond2 = Tensor.number 0.0 in
  (* false *)
  let result2 = Operation.where ~grad_spec:If_needed cond2 x2 y2 () in

  Train.set_hosted x2.value;
  Train.set_hosted y2.value;
  Train.set_hosted cond2.value;
  Train.set_hosted result2.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x2.diff).grad;
  Train.set_hosted (Option.value_exn ~here:[%here] y2.diff).grad;

  let ctx2 = Train.init_params ctx Train.IDX.empty result2 in
  let update2 = Train.grad_update result2 in
  let routine2 = Train.to_routine ctx2 Train.IDX.empty update2 in

  Train.run ctx routine2;

  Stdio.printf "x = %.4g, gradient = %.4g\n" x2.@[0] x2.@%[0];
  Stdio.printf "y = %.4g, gradient = %.4g\n" y2.@[0] y2.@%[0];
  Stdio.printf "result = %.4g\n" result2.@[0];
  Stdio.printf "Expected: x gradient = 0.0, y gradient = 1.0\n"
