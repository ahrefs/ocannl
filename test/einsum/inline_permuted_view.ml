open Base
open Ocannl
module IDX = Train.IDX
module CDSL = Train.CDSL
open Operation.DSL_modules

module type Backend = Ir.Backend_intf.Backend

let inline_inner () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 5 ] ~output_dims:[ 3 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~input_dims:[ 4 ] ~output_dims:[ 5 ] () in
  let%op c = (a * b) ++ "...|i->j => ...|ij" in
  (* FIXME(#351): this setting will become the default once we eliminate common subexpressions. *)
  Ir.Low_level.virtualize_settings.inline_complex_computations <- true;
  Train.set_hosted a.value;
  Train.set_hosted b.value;
  ignore (Train.forward_once ctx c);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false c;
  Stdio.printf "\n%!"

let inline_view () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 5 ] ~output_dims:[ 3 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~input_dims:[ 4 ] ~output_dims:[ 5 ] () in
  let%op c1 = (a * b) ++ "...|i->j => ...|ij" in
  let%op d = c1 + 1 in
  Ir.Low_level.virtualize_settings.inline_complex_computations <- false;
  Train.set_hosted a.value;
  Train.set_hosted b.value;
  ignore (Train.forward_once ctx d);
  Train.printf_tree ~here:[%here] ~with_grad:false d;
  Stdio.printf "\n%!"

let () =
  inline_inner ();
  inline_view ()
