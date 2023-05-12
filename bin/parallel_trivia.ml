open Base
open Ocannl
module CDSL = Code.CDSL
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module SDSL = Session.SDSL

let () = SDSL.set_executor Gccjit

let () =
  SDSL.drop_all_sessions ();
  SDSL.enable_all_debugs ();
  SDSL.set_executor SDSL.Interpreter;
  Random.init 0;
  let num_tasks = 10 in
  SDSL.num_parallel_tasks := num_tasks;
  let a = FDSL.init_param ~l:"a" ~o:[ Dim 1 ] [| 2.0 |] in
  (* let a = FDSL.init_param ~l:"a" ~o:[ Parallel ] @@ Array.create ~len:num_tasks 2.0 in *)
  let b = FDSL.init_param ~l:"b" ~o:[ Parallel ] @@ Array.create ~len:num_tasks (-3.0) in
  let c = FDSL.init_param ~l:"c" ~o:[ Parallel ] @@ Array.create ~len:num_tasks 10.0 in
  let f = FDSL.init_param ~l:"f" ~o:[ Parallel ] @@ Array.create ~len:num_tasks (-2.0) in
  let%nn_op e = a *. b in
  let%nn_op d = e + c in
  let%nn_op l = d *. f in
  SDSL.minus_learning_rate := Some (FDSL.init_const ~l:"minus_lr" ~o:[ Dim 1 ] [| 0.1 |]);
  SDSL.refresh_session ~update_params:false ();
  (* We did not update the params: all values and gradients will be at initial points, which are
     specified in the formula in the brackets. *)
  Stdio.printf "\n%!";
  SDSL.print_node_tree ~with_grad:true ~depth:9 l.id;
  Stdio.printf "\n%!";
  SDSL.refresh_session ~update_params:true ();
  (* Now we updated the params, but after the forward and backward passes: only params values
     will change, compared to the above. *)
  Stdio.printf "\n%!";
  SDSL.print_node_tree ~with_grad:true ~depth:9 l.id;
  Stdio.printf "\n%!";
  SDSL.refresh_session ~update_params:false ();
  (* Now again we did not update the params, they will remain as above, but both param gradients
     and the values and gradients of other nodes will change thanks to the forward and backward passes. *)
  Stdio.printf "\n%!";
  SDSL.print_node_tree ~with_grad:true ~depth:9 l.id;
  Stdio.printf "\n%!"

let _suspended () =
  SDSL.drop_all_sessions ();
  SDSL.enable_all_debugs ();
  Random.init 0;
  let%nn_op n = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  SDSL.refresh_session ();
  Stdio.printf "\n%!";
  SDSL.print_node_tree ~with_id:true ~with_grad:true ~depth:9 n.id;
  Stdio.printf "\nHigh-level code:\n%!";
  SDSL.print_session_code ();
  Stdio.printf "\nCompiled code:\n%!";
  SDSL.print_session_code ~compiled:true ();
  Stdio.printf "\n%!"
