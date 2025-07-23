open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

let _get_local_debug_runtime = Utils.get_local_debug_runtime

let%diagn_sexp _suspended () =
  let module Backend = (val Backends.fresh_backend ~backend_name:"multicore_cc" ()) in
  let%op c = "a" [ -4 ] + "b" [ 2 ] in
  let%op d = c + c + 1 in
  (* let%op c = c + 1 + c + ~-a in *)
  (* Uncomment just the first "fully on host" line to see which arrays can be virtual, and just the
     second line to see the intermediate computation values. *)
  Train.every_non_literal_on_host d;
  (* List.iter ~f:(function Some diff -> Train.set_hosted diff.grad | None -> ()) [ a.diff; b.diff
     ]; *)
  ignore (Train.update_once ~hosted:true (module Backend) d);
  Train.printf_tree ~with_grad:true ~depth:9 d;
  Stdio.print_endline "\n";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false d;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true a;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true b

let%diagn_sexp () : unit =
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
  let module Backend = (val Backends.fresh_backend ~backend_name:"multicore_cc" ()) in
  Utils.capture_stdout_logs @@ fun () ->
  ignore (Train.update_once ~hosted:true (module Backend) g);
  (* Train.printf_tree ~with_grad:true ~depth:9 g; *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false g;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true a;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true b
