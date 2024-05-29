open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

let _suspended () =
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.(new_virtual_device @@ get_device ~ordinal:0) in
  let ctx = Backend.init device in
  Utils.settings.output_debug_files_in_run_directory <- true;
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~input_dims:[ 2; 3; 4 ] ~output_dims:[ 2 ] () in
  let%op c = a *+ "i->1; ij...->0 => ...->ji" b in
  Train.forward_and_forget (module Backend) ctx c;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ c;
  Stdlib.Format.force_newline ()

let () =
  Utils.settings.with_debug_level <- 2;
  Utils.settings.output_debug_files_in_run_directory <- true;
  Utils.settings.debug_log_from_routines <- true;
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.(new_virtual_device @@ get_device ~ordinal:0) in
  let ctx = Backend.init device in
  Rand.init 0;
  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "b|i->o => o|b->i" in
  Train.forward_and_forget backend ctx ho;
  let hey2 = TDSL.range_of_shape ~batch_dims:[ 2; 3 ] ~input_dims:[ 4; 5 ] ~output_dims:[ 6; 7 ] () in
  let%op ho2 = hey2 ++ "ab|cd->ef => cf|ae->db" in
  Train.forward_and_forget backend ctx ho2;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ ho2
