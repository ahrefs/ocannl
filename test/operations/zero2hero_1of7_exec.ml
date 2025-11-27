open Base
open Ocannl
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module IDX = Train.IDX
module CDSL = Train.CDSL
open Nn_blocks.DSL_modules

module type Backend = Ir.Backend_intf.Backend

let graph_drawing_recompile () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let open Operation.At in
  let%op f_nd = (3 *. ({ x = [ 5 ] } **. 2)) - (4 *. x) + 5 in
  Train.set_hosted x.value;
  ignore (Train.forward_once ctx f_nd);
  Train.printf_tree ~with_grad:true ~depth:9 f_nd;
  let%op f = (3 *. ({ x = [ 5 ] } **. 2)) - (4 *. x) + 5 in
  Train.every_non_literal_on_host f;
  let f_upd = Train.grad_update f in
  let ctx = Train.init_params ctx IDX.empty f in
  let f_bprop = Train.to_routine ctx IDX.empty f_upd in
  Train.run ctx f_bprop;
  Train.printf_tree ~with_grad:true ~depth:9 f;
  let xs = Array.init 10 ~f:Float.(fun i -> of_int i - 5.) in
  let ys =
    Array.map xs ~f:(fun v ->
        (* This is inefficient because it compiles the argument update inside the loop. *)
        let assign_x =
          Train.to_routine (Context.context f_bprop) IDX.empty
            [%cd
              ~~("assign_x";
                 x =: !.v)]
        in
        Train.run ctx assign_x;
        Train.run ctx f_bprop;
        f.@[0])
  in
  let plot_box =
    PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" } ]
  in
  PrintBox_text.output Stdio.stdout plot_box

let graph_drawing_fetch () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  Train.every_non_literal_on_host f5;
  ignore (Train.forward_once ctx f5);
  Train.printf_tree ~with_grad:false ~depth:9 f5;
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  (* Yay, the whole shape gets inferred! *)
  let x_flat = Tensor.term_init xs ~label:[ "x_flat" ] ~grad_spec:Require_grad () in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op x = x_flat @| step_sym in
  let%op fx = f x in
  Train.set_hosted x.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x.diff).grad;
  let update = Train.grad_update fx in
  let fx_routine = Train.to_routine ctx bindings update in
  let step_ref = IDX.find_exn (Context.bindings fx_routine) step_sym in
  let ys, dys =
    Array.unzip
    @@ Array.mapi xs ~f:(fun i _ ->
        step_ref := i;
        Train.run ctx fx_routine;
        (fx.@[0], x.@%[0]))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let plot_box =
    PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" };
        Scatterplot { points = Array.zip_exn xs dys; content = PrintBox.line "*" };
        Line_plot { points = Array.create ~len:20 0.; content = PrintBox.line "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box

let simple_gradients_hosted () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  (* We need to either call `grad_update` before introducing `learning_rate`, or disable the
     rootness check. *)
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_on_host l;
  Train.every_non_literal_on_host learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  (* Note the initial state without running an init or forward pass can contain garbage. *)
  (* Train.printf_tree ~with_grad:true ~depth:9 l; *)
  (* Do not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  let ctx = Context.run ctx grad_routine in
  Train.printf_tree ~with_grad:true ~depth:9 l;
  (* Now we update the params, but we are not doing the forward and backward passes: only params
     values will change, compared to the above. The update is in the opposite direction of the
     gradient. *)
  let ctx = Context.run ctx sgd_routine in
  Train.printf_tree ~with_grad:true ~depth:9 l;

  (* Now the params will remain as above, but both param gradients and the values and gradients of
     other nodes will change thanks to the forward and backward passes. *)
  let _ctx = Context.run ctx grad_routine in
  Train.printf_tree ~with_grad:true ~depth:9 l

let simple_gradients_virtual () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  (* We pretend this is for parallel updates, to force materializing gradients, because our SGD
     update is compiled separately from our gradient update. Alternatively we could compile
     grad_update and sgd_update together.*)
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  (* Note the state without running initialization can contain garbage. *)
  (* Train.printf_tree ~with_grad:true ~depth:9 l; *)
  (* Do not update the params: all values and gradients will be at initial points, which are
     specified in the tensor in the brackets. *)
  let ctx = Context.run ctx grad_routine in
  Train.printf_tree ~with_grad:true ~depth:9 l;
  (* Only now compile the SGD update. *)
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  (* Now we update the params, but are not doing the forward and backward passes: only params values
     will change, compared to the above. Since virtual tensors are computed by-need, they will
     always be recomputed using the latest parameter state. *)
  let ctx = Context.run ctx sgd_routine in
  Train.printf_tree ~with_grad:true ~depth:9 l;
  (* Now the params will remain as above, but both param gradients and the values and gradients of
     other nodes will change thanks to the forward and backward passes. *)
  let _ctx = Context.run ctx grad_routine in
  Train.printf_tree ~with_grad:true ~depth:9 l

let two_d_neuron_hosted () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op v = ({ w = [ (-3, 1) ] } * { x = [ 2; 0 ] }) + { b = [ 6.7 ] } in
  Train.every_non_literal_on_host v;
  let update = Train.grad_update v in
  let ctx = Train.init_params ctx IDX.empty v in
  let routine = Train.to_routine ctx IDX.empty update in
  Train.run ctx routine;
  Train.printf_tree ~with_grad:true ~depth:9 v

let two_d_neuron_virtual () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op v = ({ w = [ (-3, 1) ] } * { x = [ 2; 0 ] }) + { b = [ 6.7 ] } in
  let update = Train.grad_update v in
  let ctx = Train.init_params ctx IDX.empty v in
  let routine = Train.to_routine ctx IDX.empty update in
  Train.run ctx routine;
  Train.printf_tree ~with_grad:true ~depth:9 v

let main () =
  let () = graph_drawing_recompile () in
  let () = graph_drawing_fetch () in
  let () = simple_gradients_hosted () in
  let () = simple_gradients_virtual () in
  let () = two_d_neuron_hosted () in
  let () = two_d_neuron_virtual () in
  ()

let () = main ()
