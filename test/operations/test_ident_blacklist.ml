open Base
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

(* Falsifier: tensors whose labels match C keywords or C math function names must get
   disambiguated code names (n<id>_<label>) rather than the bare name.  A bare keyword
   would produce ill-formed C like [float float[1] = ...] or [return return[1] = ...].
   A bare math-function name would shadow the callee in the same generated routine. *)

let mk label =
  Tensor.term ~label:[ label ] ~grad_spec:Prohibit_grad ~output_dims:[ 1 ] ()

let print_code_name label tn =
  match tn.Tn.code_name with
  | Some name ->
      let bare = String.equal name label in
      Stdio.printf "%s -> %s%s\n" label name (if bare then " [FAIL: bare name used]" else "")
  | None -> Stdio.printf "%s -> <not compiled>\n" label

let () =
  (* ── Test 1: C keyword labels ── *)
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let t_return = mk "return" in
  (* id=0 *)
  let t_int = mk "int" in
  (* id=1 *)
  let t_float = mk "float" in
  (* id=2 *)
  let sum = NTDSL.add t_return t_int () in
  let result = NTDSL.add sum t_float () in
  Train.set_materialized t_return.value;
  Train.set_materialized t_int.value;
  Train.set_materialized t_float.value;
  Train.set_materialized result.value;
  let _ctx = Train.forward_once ctx result in
  print_code_name "return" t_return.value;
  print_code_name "int" t_int.value;
  print_code_name "float" t_float.value;

  (* ── Test 2: C math function name labels ──
     exp/log exercise the direct (non-nested) unop extraction path.
     floorf/fabsf exercise the Satur01_gate binop path where the old remove_paren approach
     produced the wrong combined string "fabsffloorf" instead of two entries "fabsf" and "floorf". *)
  Tensor.unsafe_reinitialize ();
  let ctx2 = Context.auto () in
  let t_exp = mk "exp" in
  (* id=0 *)
  let t_log = mk "log" in
  (* id=1 *)
  let t_floorf = mk "floorf" in
  (* id=2 *)
  let t_fabsf = mk "fabsf" in
  (* id=3 *)
  let sum2 = NTDSL.add t_exp t_log () in
  let sum3 = NTDSL.add t_floorf t_fabsf () in
  let result2 = NTDSL.add sum2 sum3 () in
  Train.set_materialized t_exp.value;
  Train.set_materialized t_log.value;
  Train.set_materialized t_floorf.value;
  Train.set_materialized t_fabsf.value;
  Train.set_materialized result2.value;
  let _ctx2 = Train.forward_once ctx2 result2 in
  print_code_name "exp" t_exp.value;
  print_code_name "log" t_log.value;
  print_code_name "floorf" t_floorf.value;
  print_code_name "fabsf" t_fabsf.value
