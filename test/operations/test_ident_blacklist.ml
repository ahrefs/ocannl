open Base
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

(* Falsifier: tensors whose labels match C keywords or C math function names must get disambiguated
   code names (n<id>_<label>) rather than the bare name. A bare keyword would produce ill-formed C
   like [float float[1] = ...] or [return return[1] = ...]. A bare math-function name would shadow
   the callee in the same generated routine. *)

let mk label = Tensor.term ~label:[ label ] ~grad_spec:Prohibit_grad ~output_dims:[ 1 ] ()

let print_code_name label tn =
  match tn.Tn.code_name with
  | Some name ->
      Stdio.printf "%s -> %s\n" label name;
      if String.equal name label then
        Verdict.fail (Printf.sprintf "%s -> %s: bare name used" label name)
  | None ->
      Stdio.printf "%s -> <not compiled>\n" label;
      Verdict.fail (Printf.sprintf "%s: no code name -- the tensor was not compiled" label)

let () =
  Verdict.case "C keywords" (fun () ->
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
      print_code_name "float" t_float.value);

  Verdict.case "C math functions" (fun () ->
      (* ── Test 2: C math function name labels ── exp/log exercise the direct (non-nested) unop
         extraction path. floorf/fabsf exercise the Satur01_gate binop path where the old
         remove_paren approach produced the wrong combined string "fabsffloorf" instead of two
         entries "fabsf" and "floorf". *)
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
      print_code_name "fabsf" t_fabsf.value);

  Verdict.case "backend builtins" (fun () ->
      (* ── Test 3: names the backend's own rendering emits, colliding inside one kernel ── A
         reserved name only bites when the kernel that declares the node also *calls* the function,
         so each case below puts the two in the same routine. The reserved set is per-backend by
         construction (C spells [Tanh_approx] "tanhf", MSL spells it "tanh"), so the code name each
         node ends up with is backend-dependent and deliberately not printed; what is uniform, and
         what this section pins, is that the kernel compiles at all.

         [tanh] covers the GPU backends. It is not a contrived label: {!Tensor.unop}'s [~op_label]
         makes it the label of the node every [Operation.tanh] mints, so the gelu in a GPT-2 forward
         pass produces one. When the reserved names were read off the C spellings instead of the
         backend's own syntax functions, Metal declared [device float *__restrict tanh] for that
         node and the [tanh(...)] call on the next line resolved to the pointer -- "called object
         type 'device float *' is not a function or function pointer" (gh-ocannl-553).

         [bfloat16_to_single] covers the C-family backends, which bridge bfloat16 arithmetic through
         builtins of that name ({!Cc_backend.CC_syntax_config}); those spellings live in the
         backend's overrides, so they were likewise invisible to a list derived from {!Ir.Ops}. *)
      Tensor.unsafe_reinitialize ();
      let ctx3 = Context.auto () in
      (* The operand is unlabelled so the destination node's label is exactly ["tanh"] -- with a
         labelled operand it would be "tanh_<operand>" and never collide. *)
      let t_tanh = NTDSL.tanh (NTDSL.add (mk "a") (mk "b") ()) () in
      let bf_a = mk "bfloat16_to_single" in
      let bf_b = mk "single_to_bfloat16" in
      let bf_sum = NTDSL.add bf_a bf_b () in
      List.iter [ bf_a; bf_b; bf_sum ] ~f:(fun t -> Tn.update_prec t.Tensor.value Ir.Ops.bfloat16);
      let result3 = NTDSL.add t_tanh bf_sum () in
      List.iter [ t_tanh; bf_a; bf_b; bf_sum; result3 ] ~f:(fun t ->
          Train.set_materialized t.Tensor.value);
      let _ctx3 = Train.forward_once ctx3 result3 in
      if Option.is_some t_tanh.value.Tn.code_name then Stdio.printf "tanh -> compiled\n"
      else
        Verdict.fail
          "tanh -> no code name: the kernel that collides with the callee did not compile";
      List.iter [ bf_a; bf_b ] ~f:(fun t ->
          Stdio.printf "%s -> compiled\n" (List.hd_exn t.Tensor.value.Tn.label)))
