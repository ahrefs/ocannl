open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL

module type Backend = Ir.Backend_intf.Backend

let _suspended () =
  let module Backend = (val Backends.fresh_backend ()) in
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~input_dims:[ 2; 3; 4 ] ~output_dims:[ 2 ] () in
  let%op c = a *+ "i->1; ij...->0 => ...->ji" b in
  ignore (Train.forward_once (module Backend) c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  Stdio.printf "\n%!"

let _suspended () =
  let module Backend = (val Backends.fresh_backend ~backend_name:"cuda" ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op _ho = hey ++ "b|i->o => o|b->i" in

  let hey2 =
    TDSL.range_of_shape ~batch_dims:[ 2; 3 ] ~input_dims:[ 4; 5 ] ~output_dims:[ 6; 7 ] ()
  in
  let%op ho2 = hey2 ++ "ab|cd->ef => cf|ae->db" in
  Utils.capture_stdout_logs @@ fun () ->
  ignore (Train.forward_once backend ho2);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ho2

let () =
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  let a = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 4 ] ~output_dims:[ 5 ] () in
  let%op _ = a *+ "b|i->o; b|i->o => b|i->o" a in
  let%op c = b *+ "b|h->o; b|i->h => b|i->o" a in
  Utils.capture_stdout_logs (fun () -> ignore (Train.forward_once backend c));
  (* let%op d = a *+ "a|i->h; b|h->o => ab|i->o" b in Utils.capture_stdout_logs (fun () ->
     ignore (Train.forward_once backend d)); let%op e = a *+ "b|i->h; b|h->o => i->o" b in
     Utils.capture_stdout_logs (fun () -> ignore (Train.forward_once backend e)); let%op f = a *+
     "a|i->h; b|h->o => i->o" b in Utils.capture_stdout_logs (fun () -> ignore (Train.forward_once backend f)); *)
  (* Train.printf ~here:[%here] ~with_code:false ~with_grad:false a2; *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c
