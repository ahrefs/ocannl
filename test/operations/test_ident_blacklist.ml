open Base
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

(* Falsifier test: tensors whose labels match C keywords must get disambiguated code names
   (n<id>_<keyword>) rather than the bare keyword, which would produce ill-formed C declarations
   like [float float[1] = ...] or [return return[1] = ...]. *)

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let mk label =
    Tensor.term ~label:[ label ] ~grad_spec:Prohibit_grad ~output_dims:[ 1 ] ()
  in
  let t_return = mk "return" in
  let t_int = mk "int" in
  let t_float = mk "float" in
  let sum = NTDSL.add t_return t_int () in
  let result = NTDSL.add sum t_float () in
  Train.set_materialized t_return.value;
  Train.set_materialized t_int.value;
  Train.set_materialized t_float.value;
  Train.set_materialized result.value;
  let _ctx = Train.forward_once ctx result in
  let print_code_name label tn =
    match tn.Tn.code_name with
    | Some name ->
        let bare = String.equal name label in
        Stdio.printf "%s -> %s%s\n" label name (if bare then " [FAIL: bare keyword used]" else "")
    | None -> Stdio.printf "%s -> <not compiled>\n" label
  in
  print_code_name "return" t_return.value;
  print_code_name "int" t_int.value;
  print_code_name "float" t_float.value
