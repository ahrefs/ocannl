open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Rand = Ir.Rand.Lib

let _get_local_debug_runtime = Utils.get_local_debug_runtime

let%diagn_sexp _suspended () =
  let module Backend = (val Backends.fresh_backend ~backend_name:"multicore_cc" ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  Utils.set_log_level 2;
  Rand.init 0;
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  let%op d = c + c + 1 in
  (* let%op c = c + 1 + c + ~-a in *)
  (* Uncomment just the first "fully on host" line to see which arrays can be virtual, and just the
     second line to see the intermediate computation values. *)
  Train.every_non_literal_on_host d;
  (* List.iter ~f:(function Some diff -> Train.set_hosted diff.grad | None -> ()) [ a.diff; b.diff
     ]; *)
  let update = Train.grad_update d in
  let routine = Train.to_routine (module Backend) ctx IDX.empty update in
  Train.run routine;
  Tensor.print_tree ~with_grad:true ~depth:9 d;
  Stdio.print_endline "\n";
  Tensor.print ~with_code:false ~with_grad:false `Default @@ d;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ b

let%diagn_sexp () : unit =
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  (* Utils.settings.output_debug_files_in_build_directory <- true; *)
  Rand.init 0;
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  let%op d = (a *. b) + (b **. 3) in
  let%op c = c + c + 1 in
  let%op c = c + 1 + c + ~-a in
  let%op d = d + (d *. 2) + relu (b + a) in
  let%op d = d + (3 *. d) + relu (b - a) in
  let%op e = c - d in
  let%op f = e *. e in
  let%op g = f /. 2 in
  let%op g = g + (10. /. f) in
  List.iter ~f:(function Some diff -> Train.set_hosted diff.grad | None -> ()) [ a.diff; b.diff ];
  (* Train.every_non_literal_on_host g; *)
  let init_params = Tensor.init_params g in
  let update = Train.grad_update g in
  let init = Backend.link ctx @@ Backend.compile ctx.optimize_ctx IDX.empty init_params in
  let routine = Train.to_routine (module Backend) init.context IDX.empty update in
  Utils.capture_stdout_logs @@ fun () ->
  Train.run init;
  Train.run routine;
  (* Tensor.print_tree ~with_grad:true ~depth:9 g; *)
  Tensor.print ~with_code:false ~with_grad:false `Default @@ g;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:true `Default @@ b
