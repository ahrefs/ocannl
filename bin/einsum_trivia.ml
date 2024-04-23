open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

let () =
  Rand.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  let device = Backend.get_device ~ordinal:0 in
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
