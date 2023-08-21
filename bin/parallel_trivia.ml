open Base
open Ocannl
module CDSL = Code.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module SDSL = Session.SDSL

let () = SDSL.set_executor Gccjit

let () =
  SDSL.drop_all_sessions ();
  SDSL.enable_all_debugs ();
  SDSL.set_executor SDSL.Interpreter;
  Random.init 0;
  let num_tasks = 10 in
  let a = TDSL.init_param ~l:"a" ~o:[ CDSL.dim 1 ] [| 2.0 |] in
  (* let a = TDSL.init_param ~l:"a" ~o:[ CDSL.parallel num_tasks ] @@ Array.create ~len:num_tasks 2.0 in *)
  let b = TDSL.init_param ~l:"b" ~o:[ CDSL.parallel num_tasks ] @@ Array.create ~len:num_tasks (-3.0) in
  let c = TDSL.init_param ~l:"c" ~o:[ CDSL.parallel num_tasks ] @@ Array.create ~len:num_tasks 10.0 in
  let f = TDSL.init_param ~l:"f" ~o:[ CDSL.parallel num_tasks ] @@ Array.create ~len:num_tasks (-2.0) in
  let%nn_op e = a *. b in
  let%nn_op d = e + c in
  let%nn_op l = d *. f in
  SDSL.minus_learning_rate := Some (TDSL.init_const ~l:"minus_lr" ~o:[ CDSL.dim 1 ] [| 0.1 |]);
  SDSL.refresh_session ~update_params:false ();
  (* We did not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
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
  let%nn_op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  SDSL.refresh_session ();
  Stdio.printf "\n%!";
  SDSL.print_node_tree ~with_id:true ~with_grad:true ~depth:9 v.id;
  Stdio.printf "\nHigh-level code:\n%!";
  SDSL.print_session_code ();
  Stdio.printf "\nCompiled code:\n%!";
  SDSL.print_session_code ~compiled:true ();
  Stdio.printf "\n%!"
