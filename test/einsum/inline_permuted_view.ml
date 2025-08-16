open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

module type Backend = Ir.Backend_intf.Backend

let () =
  let module Backend = (val Backends.fresh_backend ()) in
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 5 ] ~output_dims:[ 3 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~input_dims:[ 4 ] ~output_dims:[ 5 ] () in
  let%op c = (a * b) ++ "...|i->j => ...|ij" in
  (* FIXME(#351): this setting will become the default once we eliminate common subexpressions. *)
  Ir.Low_level.virtualize_settings.inline_complex_computations <- true;
  Train.set_hosted a.value;
  Train.set_hosted b.value;
  ignore (Train.forward_once (module Backend) c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  Stdio.printf "\n%!"
