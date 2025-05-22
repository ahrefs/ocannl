open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Rand = Ir.Rand.Lib

module type Backend = Ir.Backend_intf.Backend

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let _suspended () =
  Rand.init 0;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  Train.every_non_literal_on_host v;
  let code = Train.grad_update v in
  let routine = Train.to_routine (module Backend) ctx IDX.empty code.fwd_bprop in
  Train.run routine;
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_id:true ~with_grad:true ~depth:9 v;
  Stdio.printf "\nHigh-level code:\n%!";
  Ir.Assignments.doc_hum () code.fwd_bprop.asgns |> PPrint.ToChannel.pretty 0.7 100 Stdio.stdout;
  Stdio.printf "\n%!"

let _suspended () =
  Rand.init 0;
  CDSL.enable_all_debugs ();
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  let module Backend = (val Backends.fresh_backend ()) in
  Train.every_non_literal_on_host f5;
  Train.forward_and_forget
    (module Backend)
    Backend.(make_context @@ new_stream @@ get_device ~ordinal:0)
    f5;
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_grad:false ~depth:9 f5;
  Stdio.printf "\n%!"

let _suspended () =
  (* FIXME: why is this toplevel example broken and the next one working? *)
  Utils.settings.output_debug_files_in_build_directory <- true;
  Rand.init 0;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let size = 100 in
  let values = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  (* Test that the batch axis dimensions will be inferred. *)
  let x_flat =
    Tensor.term ~grad_spec:Tensor.Require_grad
      ~label:[ "x_flat" ] (* ~input_dims:[] ~output_dims:[ 1 ] *)
      ~init_op:(Constant_fill { values; strict = true })
      ()
  in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  (* The [let x =] line is the same as this except [let%op x =] uses [~grad_spec:If_needed]. *)
  let%op x = x_flat @| step_sym in
  (* let x = Operation.slice ~label:[ "x" ] ~grad_spec:Require_grad step_sym x_flat in *)
  Train.set_hosted (Option.value_exn ~here:[%here] x.diff).grad;
  let%op fx = f x in
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let update = Train.grad_update fx in
  let routine = Train.to_routine (module Backend) ctx bindings update.fwd_bprop in
  let step_ref = IDX.find_exn routine.bindings step_sym in
  let ys = Array.create ~len:size 0. and dys = Array.create ~len:size 0. in
  let open Operation.At in
  let f () =
    Train.run routine;
    ys.(!step_ref) <- fx.@[0];
    dys.(!step_ref) <- x.@%[0]
  in
  Train.sequential_loop routine.bindings ~f;
  let plot_box =
    PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn values ys; content = PrintBox.line "#" };
        Scatterplot { points = Array.zip_exn values dys; content = PrintBox.line "*" };
        Line_plot { points = Array.create ~len:20 0.; content = PrintBox.line "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box;
  Stdio.print_endline ""

let _suspended () =
  (* Utils.set_log_level 2; *)
  Utils.settings.output_debug_files_in_build_directory <- true;
  (* Utils.settings.debug_log_from_routines <- true; *)
  Rand.init 0;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  Train.every_non_literal_on_host f5;
  Train.forward_and_forget (module Backend) ctx f5;
  Tensor.print_tree ~with_grad:false ~depth:9 f5;
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  (* Yay, the whole shape gets inferred! *)
  let x_flat =
    Tensor.term ~grad_spec:Require_grad ~label:[ "x_flat" ]
      ~init_op:(Constant_fill { values = xs; strict = true })
      ()
  in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op x = x_flat @| step_sym in
  let%op fx = f x in
  Train.set_hosted x.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x.diff).grad;
  let update = Train.grad_update fx in
  let fx_routine = Train.to_routine (module Backend) ctx bindings update.fwd_bprop in
  let step_ref = IDX.find_exn fx_routine.bindings step_sym in
  let%track_sexp () =
    let ys, dys =
      Array.unzip
      @@ Array.mapi xs ~f:(fun i _ ->
             step_ref := i;
             Train.run fx_routine;
             (fx.@[0], x.@%[0]))
    in
    (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
    let plot_box =
      PrintBox_utils.plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
        [
          Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" };
          Scatterplot { points = Array.zip_exn xs dys; content = PrintBox.line "*" };
          Line_plot { points = Array.create ~len:20 0.; content = PrintBox.line "-" };
        ]
    in
    PrintBox_text.output Stdio.stdout plot_box
  in
  ()

let () =
  Rand.init 0;
  Utils.set_log_level 2;
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  Train.every_non_literal_on_host l;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let update = Train.grad_update l in
  let routine =
    Train.to_routine (module Backend) (Backend.make_context stream) IDX.empty update.fwd_bprop
  in
  Train.run routine;
  (* Tensor.iter_embedded l ~f:(fun a -> ignore (Backend.to_host routine.context a : bool));
     Backend.await stream; *)
  Stdio.print_endline
    {|
      We did not update the params: all values and gradients will be at initial points,
      which are specified in the tensor in the brackets.|};
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  let%op learning_rate = 0.1 in
  let routine =
    Train.to_routine (module Backend) routine.context IDX.empty
    @@ Train.sgd_update ~learning_rate update
  in
  (* learning_rate is virtual so this will not print anything. *)
  Stdio.print_endline
    {|
      Due to how the gccjit backend works, since the parameters were constant in the grad_update
      computation, they did not exist on the device before. Now they do. This would not be needed
      on the cuda backend.|};
  Train.run routine;
  Stdio.print_endline
    {|
      Now we updated the params, but after the forward and backward passes:
      only params values will change, compared to the above.|};
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  (* We could reuse the jitted code if we did not use `jit_and_run`. *)
  let update = Train.grad_update l in
  let routine = Train.to_routine (module Backend) routine.context IDX.empty update.fwd_bprop in
  Train.run routine;
  (* Tensor.iter_embedded l ~f:(fun a -> ignore (Backend.to_host routine.context a : bool));
     Backend.await stream; *)
  Stdio.print_endline
    {|
      Now again we did not update the params, they will remain as above, but both param
      gradients and the values and gradients of other nodes will change thanks to the forward and
      backward passes.|};
  Tensor.print_tree ~with_grad:true ~depth:9 l
