open Base
open Ocannl
module IDX = Arrayjit.Indexing.IDX
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

let _suspended () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let%op v = ("w" [ (-3, 1) ] * "x" [ 2; 0 ]) + "b" [ 6.7 ] in
  let code = Train.grad_update v in
  let jitted = Backend.jit Backend.active_context IDX.empty code in
  jitted.run ();
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_id:true ~with_grad:true ~depth:9 v;
  Caml.Format.printf "\nHigh-level code:\n%a\n%!" Sexp.pp_hum @@ [%sexp_of: Arrayjit.Assignments.t] code

let _suspended () =
  Random.init 0;
  CDSL.enable_all_debugs ();
  CDSL.virtualize_settings.enable_device_only <- false;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let%op f5 = f 5 in
  let module Backend = (val Train.fresh_backend ()) in
  let ctx = Backend.init @@ Backend.get_device ~ordinal:0 in
  let jitted = Backend.jit ctx IDX.empty @@ Train.grad_update f5 in
  jitted.run ();
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_grad:false ~depth:9 f5;
  Stdio.printf "\n%!"

let _suspended () =
  Random.init 0;
  let%op f x = (3 *. (x **. 2)) - (4 *. x) + 5 in
  let size = 100 in
  let values = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  (* Test that the batch axis dimensions will be inferred. *)
  let x_flat =
    Tensor.term ~grad_spec:Tensor.Require_grad ~label:"x_flat" ~input_dims:[] ~output_dims:[ 1 ]
      ~init_op:(Constant_fill { values; strict = true })
      ()
  in
  let session_step, _, bindings = IDX.get_static_symbol IDX.empty in
  let%op x = x_flat @| session_step in
  let%op fx = f x in
  Stdio.print_endline "\n";
  Tensor.print_tree ~with_id:true ~with_value:false ~with_grad:false ~depth:9 fx;
  Stdio.print_endline "\n";
  let module Backend = (val Train.fresh_backend ()) in
  let ctx = Backend.init @@ Backend.get_device ~ordinal:0 in
  let jitted = Backend.jit ctx IDX.empty @@ Train.grad_update fx in
  jitted.run ();
  Tensor.print ~with_grad:true ~with_code:true ~with_low_level:true `Default fx;
  Stdio.print_endline "\n";
  Tensor.print_tree ~with_id:true ~with_value:true ~with_grad:true ~depth:9 fx;
  Stdio.print_endline "\n";
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [
        Scatterplot { points = Array.zip_exn values (Tensor.value_1d_points ~xdim:0 fx); pixel = "#" };
        Scatterplot { points = Array.zip_exn values (Tensor.grad_1d_points ~xdim:0 fx); pixel = "*" };
        Line_plot { points = Array.create ~len:20 0.; pixel = "-" };
      ]
  in
  PrintBox_text.output Stdio.stdout plot_box

let () =
  Random.init 0;
  CDSL.with_debug := true;
  CDSL.keep_files_in_run_directory := true;
  Random.init 0;
  let%op e = "a" [ 2 ] *. "b" [ -3 ] in
  let%op d = e + "c" [ 10 ] in
  let%op l = d *. "f" [ -2 ] in
  List.iter ~f:Tensor.set_fully_on_host [ e; d; l ];
  let module Backend = (val Train.fresh_backend ()) in
  let jitted = Backend.(jit (init @@ get_device ~ordinal:0) IDX.empty @@ Train.grad_update l) in
  jitted.run ();
  Stdio.print_endline
    "\n\
     We did not update the params: all values and gradients will be at initial points,\n\
    \    which are specified in the tensor in the brackets.";
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  let jitted = Backend.jit jitted.context IDX.empty @@ Train.sgd_update l in
  jitted.run ();
  Stdio.print_endline
    "\n\
     Now we updated the params, but after the forward and backward passes:\n\
    \    only params values will change, compared to the above.";
  Tensor.print_tree ~with_grad:true ~depth:9 l;
  (* We could reuse the jitted code if we did not use `jit_and_run`. *)
  let jitted = Backend.jit jitted.context IDX.empty @@ Train.grad_update l in
  jitted.run ();
  Stdio.print_endline
    "\n\
     Now again we did not update the params, they will remain as above, but both param\n\
    \    gradients and the values and gradients of other nodes will change thanks to the forward and \
     backward passes.";
  Tensor.print_tree ~with_grad:true ~depth:9 l
