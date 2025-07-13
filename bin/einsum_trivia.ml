open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module Rand = Ir.Rand.Lib

module type Backend = Ir.Backend_intf.Backend

let _suspended () =
  Rand.init 0;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Utils.settings.output_debug_files_in_build_directory <- true;
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~input_dims:[ 2; 3; 4 ] ~output_dims:[ 2 ] () in
  let%op c = a *+ "i->1; ij...->0 => ...->ji" b in
  Train.forward_and_forget (module Backend) ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c;
  Stdio.printf "\n%!"

let _suspended () =
  (* Utils.set_log_level 2; *)
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  let module Backend = (val Backends.fresh_backend ~backend_name:"cuda" ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "b|i->o => o|b->i" in
  let ctx = Utils.capture_stdout_logs (fun () -> Train.forward_and_ctx backend ctx ho) in
  let hey2 =
    TDSL.range_of_shape ~batch_dims:[ 2; 3 ] ~input_dims:[ 4; 5 ] ~output_dims:[ 6; 7 ] ()
  in
  let%op ho2 = hey2 ++ "ab|cd->ef => cf|ae->db" in
  Utils.capture_stdout_logs @@ fun () ->
  Train.forward_and_forget backend ctx ho2;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho2

let () =
  (* Utils.set_log_level 2; *)
  Utils.settings.output_debug_files_in_build_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  let a = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 4 ] ~output_dims:[ 5 ] () in
  let%op a2 = a *+ "b|i->o; b|i->o => b|i->o" a in
  let ctx = Utils.capture_stdout_logs (fun () -> Train.forward_and_ctx backend ctx a2) in
  let%op c = b *+ "b|h->o; b|i->h => b|i->o" a in
  Utils.capture_stdout_logs (fun () -> Train.forward_and_forget backend ctx c);
  (* let%op d = a *+ "a|i->h; b|h->o => ab|i->o" b in Utils.capture_stdout_logs (fun () ->
     Train.forward_and_forget backend ctx d); let%op e = a *+ "b|i->h; b|h->o => i->o" b in
     Utils.capture_stdout_logs (fun () -> Train.forward_and_forget backend ctx e); let%op f = a *+
     "a|i->h; b|h->o => i->o" b in Utils.capture_stdout_logs (fun () -> Train.forward_and_forget
     backend ctx f); *)
  (* Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a2; *)
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c
