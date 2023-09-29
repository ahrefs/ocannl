open Base
open Ocannl
module LA = Arrayjit.Lazy_array
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
  let jitted = Backend.jit ctx IDX.empty code in
  (* jitted.run (); *)
  Train.sync_run (module Backend) jitted v;
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_id:true ~with_grad:true ~depth:9 v;
  Stdlib.Format.printf "\nHigh-level code:\n%!";
  Stdlib.Format.printf "%a\n%!" Arrayjit.Assignments.fprint_code code

let _suspended () =
  Random.init 0;
  CDSL.enable_all_debugs ();
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  let module Backend = (val Train.fresh_backend ()) in
  Train.every_non_literal_on_host f5;
  let jitted = Backend.(jit (init @@ get_device ~ordinal:0) IDX.empty @@ Train.forward f5) in
  Train.sync_run (module Backend) jitted f5;
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_grad:false ~depth:9 f5;
  Stdio.printf "\n%!"

let () =
  Utils.settings.with_debug <- true;
  Utils.settings.keep_files_in_run_directory <- true;
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
  let step_sym, step_ref, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  (* The [let x =] line is the same as this except [let%op x =] uses [~grad_spec:If_needed]. *)
  (* let%op x = x_flat @| step_sym in *)
  let x = Operation.slice ~label:[ "x" ] ~grad_spec:Require_grad step_sym x_flat in
  Train.set_on_host (Option.value_exn x.diff).grad;
  let%op fx = f x in
  Stdio.print_endline "\n";
  Tensor.print_tree ~with_id:true ~with_value:false ~with_grad:false ~depth:9 fx;
  Stdio.print_endline "\n";
  let module Backend = (val Train.fresh_backend ()) in
  let ctx = Backend.init @@ Backend.get_device ~ordinal:0 in
  let jitted = Backend.jit ctx bindings @@ Train.grad_update fx in
  let ys = Array.create ~len:size 0. and dys = Array.create ~len:size 0. in
  let open Tensor.O in
  let looping () =
    assert (Backend.to_host jitted.context fx.value);
    assert (Backend.to_host jitted.context (Option.value_exn x.diff).grad);
    ys.(!step_ref) <- fx.@[0];
    dys.(!step_ref) <- x.@%[0]
  in
  Train.sync_run ~looping (module Backend) jitted fx;
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
  Utils.settings.keep_files_in_run_directory <- true;
  Utils.settings.debug_log_jitted <- true;
  Random.init 0;
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  Train.every_non_literal_on_host l;
  let open (val Train.fresh_backend ()) in
  let device = get_device ~ordinal:0 in
  let jitted = jit (init device) IDX.empty @@ Train.grad_update l in
  Tensor.iter_embedded_arrays l ~f:(fun a ->
      if from_host jitted.context a then Stdio.printf "Sent array %s.\n%!" @@ LA.name a);
  jitted.run ();
  await device;
  Tensor.iter_embedded_arrays l ~f:(fun a ->
      if to_host jitted.context a then Stdio.printf "Retrieved array %s.\n%!" @@ LA.name a);
  Stdio.print_endline
    "\n\
     We did not update the params: all values and gradients will be at initial points,\n\
    \    which are specified in the tensor in the brackets.";
  Tensor.print_tree ~with_grad:true ~depth:9 l
(*;
  let jitted = jit jitted.context IDX.empty @@ Train.sgd_update l in
  jitted.run ();
  Stdio.print_endline
    "\n\
     Now we updated the params, but after the forward and backward passes:\n\
    \    only params values will change, compared to the above.";
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  (* We could reuse the jitted code if we did not use `jit_and_run`. *)
  let jitted = jit jitted.context IDX.empty @@ Train.grad_update l in
  jitted.run ();
  Stdio.print_endline
    "\n\
     Now again we did not update the params, they will remain as above, but both param\n\
    \    gradients and the values and gradients of other nodes will change thanks to the forward and \
     backward passes.";
  Tensor.print_tree ~with_grad:true ~depth:9 l
*)
