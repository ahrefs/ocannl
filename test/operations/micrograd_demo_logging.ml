open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module CDSL = Train.CDSL
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

module type Backend = Ir.Backend_intf.Backend

let () =
  Tensor.unsafe_reinitialize ();
  Utils.set_log_level 2;
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  if
    String.equal (Utils.get_global_arg ~default:"sync_cc" ~arg_name:"backend") "metal"
    && Utils.get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files"
  then (
    Utils.log_debug_routine_logs
      ~log_contents:
        [
          "Metal routine stream logging is skipped: the Metal backend emits routine debug logs via \
           os_log rather than the captured stream-file logger.";
        ]
      ~stream_name:"metal-0-0";
    Utils.restore_settings ();
    Stdlib.exit 0);
  let ctx = Context.auto () in
  let%op c = { a = [ -4 ] } + { b = [ 2 ] } in
  let%op d = (a *. b) + (b **. 3) in
  let%op c = c + c + 1 in
  let%op c = c + 1 + c + ~-a in
  let%op d = d + (d *. 2) + relu (b + a) in
  let%op d = d + (3 *. d) + relu (b - a) in
  let%op e = c - d in
  let%op f = e **. 2 in
  let%op g = f /. 2 in
  let%op g = g + (10. /. f) in
  List.iter
    ~f:(Option.iter ~f:(fun diff -> Train.set_materialized diff.Tensor.grad))
    [ a.diff; b.diff ];
  (* FIXME(#351): this is a good test for common subexpression elimination. *)
  Utils.capture_stdout_logs @@ fun () ->
  let ctx = Train.update_once ctx g in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx g;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx a;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx b;
  Utils.restore_settings ()
