open Base
open Ocannl
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

module type Backend = Ir.Backend_intf.Backend

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  Tn.print_accessible_headers ();
  Utils.capture_stdout_logs @@ fun () ->
  ignore (Train.update_once (module Backend) g);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false g;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true a;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true b

let _suspended () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  Tn.print_accessible_headers ();
  Utils.capture_stdout_logs @@ fun () ->
  ignore (Train.update_once (module Backend) g);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false g;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true a;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true b
