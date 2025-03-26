open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

module type Backend = Arrayjit.Backend_intf.Backend

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let%debug_sexp graph_t () : unit =
  Tensor.unsafe_reinitialize ();
  Rand.init 0;
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = recip x in
  let size = 50 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) + 0.1) in
  let x_flat =
    Tensor.term ~grad_spec:Require_grad ~label:[ "x_flat" ]
      ~init_op:(Constant_fill { values = xs; strict = true })
      ()
  in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op xkcd = x_flat @| step_sym in
  let%op fx = f xkcd in
  Train.set_hosted xkcd.value;
  Train.set_hosted x_flat.value;
  Train.set_hosted (Option.value_exn ~here:[%here] xkcd.diff).grad;
  let update = Train.grad_update fx in
  let fx_routine = Train.to_routine (module Backend) ctx bindings update.fwd_bprop in
  let step_ref = IDX.find_exn fx_routine.bindings step_sym in
  Tensor.print_tree ~with_shape:true ~with_grad:true ~depth:9 xkcd;
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
