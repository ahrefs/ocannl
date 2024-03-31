open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Arrayjit.Indexing.IDX
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Utils = Arrayjit.Utils

let _suspended () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  Train.every_non_literal_on_host v;
  let code = Train.grad_update v in
  let routine = Backend.jit ctx IDX.empty code.fwd_bprop in
  Train.sync_run (module Backend) routine v;
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_id:true ~with_grad:true ~depth:9 v;
  Stdlib.Format.printf "\nHigh-level code:\n%!";
  Stdlib.Format.printf "%a\n%!" (Arrayjit.Assignments.fprint_hum ()) code.fwd_bprop

let _suspended () =
  Random.init 0;
  CDSL.enable_all_debugs ();
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  let module Backend = (val Train.fresh_backend ()) in
  Train.every_non_literal_on_host f5;
  Train.forward_and_forget (module Backend) Backend.(init @@ get_device ~ordinal:0) f5;
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_grad:false ~depth:9 f5;
  Stdio.printf "\n%!"

let () =
  Utils.settings.output_debug_files_in_run_directory <- true;
  Random.init 0;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let size = 100 in
  let values = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  (* Test that the batch axis dimensions will be inferred. *)
  let x_flat =
    Tensor.term ~grad_spec:Tensor.Require_grad ~label:[ "x_flat" ] ~input_dims:[] ~output_dims:[ 1 ]
      ~init_op:(Constant_fill { values; strict = true })
      ()
  in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  (* The [let x =] line is the same as this except [let%op x =] uses [~grad_spec:If_needed]. *)
  (* let%op x = x_flat @| step_sym in *)
  let x = Operation.slice ~label:[ "x" ] ~grad_spec:Require_grad step_sym x_flat in
  Train.set_hosted (Option.value_exn x.diff).grad;
  let%op fx = f x in
  Stdio.print_endline "\n";
  Tensor.print_tree ~with_id:true ~with_value:false ~with_grad:false ~depth:9 fx;
  Stdio.print_endline "\n";
  let module Backend = (val Train.fresh_backend ()) in
  let ctx = Backend.init @@ Backend.get_device ~ordinal:0 in
  let update = Train.grad_update fx in
  let routine = Backend.jit ctx bindings update.fwd_bprop in
  let step_ref = IDX.find_exn routine.bindings step_sym in
  let ys = Array.create ~len:size 0. and dys = Array.create ~len:size 0. in
  let open Tensor.O in
  let looping () =
    assert (Backend.to_host routine.context fx.value);
    assert (Backend.to_host routine.context (Option.value_exn x.diff).grad);
    ys.(!step_ref) <- fx.@[0];
    dys.(!step_ref) <- x.@%[0]
  in
  Train.sync_run ~looping (module Backend) routine fx;
  Tensor.print ~with_grad:true ~with_code:true `Default fx;
  Stdio.print_endline "\n";
  Tensor.print_tree ~with_id:true ~with_value:true ~with_grad:true ~depth:9 fx;
  Stdio.print_endline "\n";
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn values ys; pixel = "#" };
        Scatterplot { points = Array.zip_exn values dys; pixel = "*" };
        Line_plot { points = Array.create ~len:20 0.; pixel = "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box

let _suspended () =
  Random.init 0;
  Utils.settings.with_debug <- true;
  Utils.settings.output_debug_files_in_run_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  Random.init 0;
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  Train.every_non_literal_on_host l;
  let open (val Train.fresh_backend ()) in
  let device = get_device ~ordinal:0 in
  let update = Train.grad_update l in
  let routine = jit (init device) IDX.empty @@ update.fwd_bprop in
  Tensor.iter_embedded_arrays l ~f:(fun a ->
      if from_host routine.context a then Stdio.printf "Sent array %s.\n%!" @@ Tn.name a);
  Train.run routine;
  await device;
  Tensor.iter_embedded_arrays l ~f:(fun a ->
      if to_host routine.context a then Stdio.printf "Retrieved array %s.\n%!" @@ Tn.name a);
  Stdio.print_endline
    {|
      We did not update the params: all values and gradients will be at initial points,
      which are specified in the tensor in the brackets.|};
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  let%op learning_rate = 0.1 in
  let routine = jit routine.context IDX.empty @@ Train.sgd_update ~learning_rate update in
  (* learning_rate is virtual so this will not print anything. *)
  Tensor.iter_embedded_arrays learning_rate ~f:(fun a ->
      if from_host routine.context a then Stdio.printf "Sent array %s.\n%!" @@ Tn.name a);
  Stdio.print_endline
    {|
      Due to how the gccjit backend works, since the parameters were constant in the grad_update
      computation, they did not exist on the device before. Now they do. This would not be needed
      on the cuda backend.|};
  List.iter [ a.value; b.value; c.value; f.value ] ~f:(fun a ->
      if from_host routine.context a then Stdio.printf "Sent array %s.\n%!" @@ Tn.name a);
  Train.run routine;
  await device;
  Tensor.iter_embedded_arrays l ~f:(fun a ->
      if to_host routine.context a then Stdio.printf "Retrieved array %s.\n%!" @@ Tn.name a);
  Stdio.print_endline
    {|
      Now we updated the params, but after the forward and backward passes:
      only params values will change, compared to the above.|};
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  (* We could reuse the jitted code if we did not use `jit_and_run`. *)
  let update = Train.grad_update l in
  let routine = jit routine.context IDX.empty update.fwd_bprop in
  Train.run routine;
  await device;
  Tensor.iter_embedded_arrays l ~f:(fun a ->
      if to_host routine.context a then Stdio.printf "Retrieved array %s.\n%!" @@ Tn.name a);
  Stdio.print_endline
    {|
      Now again we did not update the params, they will remain as above, but both param
      gradients and the values and gradients of other nodes will change thanks to the forward and
      backward passes.|};
  Tensor.print_tree ~with_grad:true ~depth:9 l
