open Base
open Stdio
open Ocannl
open Operation.DSL_modules
module IDX = Train.IDX

(* Test 1: RAW dependency — sgd_routine reads gradients written by grad_routine.
   Pattern from zero2hero_1of7_exec.ml simple_gradients_hosted. *)
let test_raw_dependency () =
  printf "=== Test 1: RAW dependency ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_on_host l;
  Train.every_non_literal_on_host learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  let grad_id = Context.routine_id grad_routine in
  let sgd_deps = Context.execution_deps sgd_routine in
  printf "sgd depends on grad: %b\n" (List.mem sgd_deps grad_id ~equal:Int.equal);
  printf "sgd has deps: %b\n" (not (List.is_empty sgd_deps));
  (* Correct order: grad then sgd *)
  let ctx' = Context.run ctx grad_routine in
  let _ctx' = Context.run ctx' sgd_routine in
  printf "Correct order (grad then sgd): OK\n"

(* Test 2: Disjoint routines — sibling compiles from same context produce no deps *)
let test_disjoint () =
  printf "\n=== Test 2: Disjoint routines ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op loss_x = { x_param = [ 5 ] } *. { x_in = [ 3 ] } in
  let%op loss_y = { y_param = [ 7 ] } *. { y_in = [ 2 ] } in
  Train.every_non_literal_on_host loss_x;
  Train.every_non_literal_on_host loss_y;
  let grad_x = Train.grad_update loss_x in
  let grad_y = Train.grad_update loss_y in
  let ctx = Train.init_params ctx IDX.empty loss_x in
  let ctx = Train.init_params ctx IDX.empty loss_y in
  (* Compile from same context — sibling branches, should be independent *)
  let routine_x = Train.to_routine ctx IDX.empty grad_x in
  let routine_y = Train.to_routine ctx IDX.empty grad_y in
  let x_id = Context.routine_id routine_x in
  let y_id = Context.routine_id routine_y in
  let x_deps = Context.execution_deps routine_x in
  let y_deps = Context.execution_deps routine_y in
  (* Neither should depend on the other — they may have deps on init_params routines *)
  printf "x depends on y: %b\n" (List.mem x_deps y_id ~equal:Int.equal);
  printf "y depends on x: %b\n" (List.mem y_deps x_id ~equal:Int.equal);
  (* Both should be runnable since init_params already executed *)
  printf "can_run x: %b\n" (Context.can_run ctx routine_x);
  printf "can_run y: %b\n" (Context.can_run ctx routine_y);
  (* Either order should work *)
  let ctx' = Context.run ctx routine_y in
  let _ctx' = Context.run ctx' routine_x in
  printf "Reverse order (y then x): OK\n"

(* Test 3: can_run reflects execution state *)
let test_can_run () =
  printf "\n=== Test 3: can_run ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  printf "can_run grad (before execution): %b\n" (Context.can_run ctx grad_routine);
  printf "can_run sgd (before grad): %b\n" (Context.can_run ctx sgd_routine);
  let ctx' = Context.run ctx grad_routine in
  printf "can_run sgd (after grad): %b\n" (Context.can_run ctx' sgd_routine)

(* Test 4: Negative test — running a dependent routine out of order raises Failure *)
let test_wrong_order_raises () =
  printf "\n=== Test 4: Wrong order raises ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_on_host l;
  Train.every_non_literal_on_host learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  (* sgd depends on grad — running sgd first must fail *)
  (try
     ignore (Context.run ctx sgd_routine);
     printf "Wrong order (sgd before grad): no error (BUG)\n"
   with Failure msg ->
     let is_enforcement = String.is_substring msg ~substring:"Context.run:" in
     printf "Wrong order raises Failure from Context.run: %b\n" is_enforcement)

(* Test 5: Re-execution pattern — grad -> sgd -> grad succeeds without reset *)
let test_reexecution () =
  printf "\n=== Test 5: Re-execution (grad -> sgd -> grad) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  let ctx' = Context.run ctx grad_routine in
  let ctx' = Context.run ctx' sgd_routine in
  let _ctx' = Context.run ctx' grad_routine in
  printf "grad -> sgd -> grad: OK\n"

let () =
  test_raw_dependency ();
  test_disjoint ();
  test_can_run ();
  test_wrong_order_raises ();
  test_reexecution ()
