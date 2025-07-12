open Base
open Ocannl
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Ir.Assignments
module Rand = Ir.Rand.Lib
module Tn = Ir.Tnode

module type Backend = Ir.Backend_intf.Backend

let () =
  Tensor.unsafe_reinitialize ();
  Rand.init 0;
  Utils.set_log_level 2;
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  let%op d = (a *. b) + (b **. 3) in
  let%op c = c + c + 1 in
  let%op c = c + 1 + c + ~-a in
  let%op d = d + (d *. 2) + relu (b + a) in
  let%op d = d + (3 *. d) + relu (b - a) in
  let%op e = c - d in
  let%op f = e **. 2 in
  let%op g = f /. 2 in
  let%op g = g + (10. /. f) in
  List.iter ~f:(Option.iter ~f:(fun diff -> Train.set_hosted diff.Tensor.grad)) [ a.diff; b.diff ];
  let init_params = Tensor.init_params g in
  let update = Train.grad_update g in
  let init = Train.to_routine (module Backend) ctx IDX.empty init_params in
  let step = Train.to_routine (module Backend) init.context IDX.empty update in
  Utils.capture_stdout_logs @@ fun () ->
  Train.run init;
  Train.run step;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default g;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:true `Default a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:true `Default b
