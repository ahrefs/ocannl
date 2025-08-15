open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

module type Backend = Ir.Backend_intf.Backend

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let graph_t () : unit =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = where (x < !.0.) (sin x) (cos x) in
  (* let%op f x = sin x in *)
  (* let%op f x = sin x in *)
  let size = 10 in
  let x_min = -5. in
  let x_max = 5. in
  let xs =
    Array.init size ~f:Float.(fun i -> x_min + (of_int i * (x_max - x_min) / (of_int size - 1.)))
  in
  let x_flat = Tensor.term_init xs ~label:[ "x_flat" ] ~grad_spec:Require_grad () in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op xkcd = x_flat @| step_sym in
  let%op fx = f xkcd in
  Train.set_hosted xkcd.value;
  Train.set_hosted x_flat.value;
  Train.set_hosted (Option.value_exn ~here:[%here] xkcd.diff).grad;
  let ctx = Train.init_params (module Backend) IDX.empty fx in
  let update = Train.grad_update fx in
  let fx_routine = Train.to_routine (module Backend) ctx bindings update in
  Train.run fx_routine;
  let step_ref = IDX.find_exn fx_routine.bindings step_sym in
  Train.printf_tree ~with_grad:true ~depth:9 xkcd;
  let ys, dys =
    Array.unzip
    @@ Array.mapi xs ~f:(fun i _ ->
           step_ref := i;
           Train.run fx_routine;
           (fx.@[0], xkcd.@%[0]))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  let plot_box =
    PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn xs dys; content = PrintBox.line "*" };
        Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" };
        Line_plot { points = Array.create ~len:20 0.; content = PrintBox.line "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box

let () = graph_t ()
