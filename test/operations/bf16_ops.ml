(* Regression test for bfloat16 code generation, in particular on the HIP backend.

   [__hip_bfloat16] carries implicit conversion operators to float, double, int, char, ... , so
   emissions that call an overloaded libm function on bf16 operands ([fma]) or that mix a bf16
   operand with a literal of another type ([fmax(0.0, x)], [1 / x], [x == 0.0]) are ambiguous under
   hiprtc and fail to compile -- the kernel never runs. This test drives one such emission per case:
   FMA (the [+*] accumulation), Relu, Satur01, Recip and Not, all at bfloat16 precision.

   The inputs are chosen so every value is exactly representable in bfloat16 (7 mantissa bits),
   keeping the printed numbers backend-uniform. *)

open Base
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = Tensor.term_init [| 1.; 2.; -3.; 4. |] ~label:[ "a" ] () in
  let b = Tensor.term_init [| 2.; 0.5; 1.; 4. |] ~label:[ "b" ] () in
  (* The reduction over [i] lowers to an FMA accumulation: 2 + 1 - 3 + 16 = 16. *)
  let%op dot = a +* "i;i=>0" b in
  let%op relued = relu (dot - 20.) in
  let%op reciped = recip dot in
  let%op saturated = sat01 dot in
  let%op notted = not dot in
  let%op combined = relued + reciped + saturated + notted in
  let intermediates = [ dot; relued; reciped; saturated; notted ] in
  List.iter (a :: b :: intermediates) ~f:(fun t -> Tn.update_prec t.Tensor.value Ir.Ops.bfloat16);
  (* Materializing keeps each result in its own bfloat16 node, so the ops above are emitted at
     bfloat16 rather than being inlined into a single-precision temporary. *)
  List.iter intermediates ~f:(fun t -> Train.set_materialized t.Tensor.value);
  Train.set_materialized combined.value;
  let ctx = Train.forward_once ctx combined in
  (* dot = 16, relu (16 - 20) = 0, recip 16 = 0.0625, sat01 16 = 1, not 16 = 0. *)
  Train.printf_tree ctx combined;
  (* The bfloat16 uniform builtins must yield bfloat16 *values*: on the GPU backends they used to
     return the bit pattern as an [unsigned short], which the assignment to a bfloat16 cell then
     converted by value -- 0x3F80 arriving as 16256.0 rather than 1.0. A range check catches that
     where a materialized-vs-virtual parity check cannot (both paths share the builtin). *)
  let u = TDSL.uniform () ~output_dims:[ 8 ] () in
  Tn.update_prec u.value Ir.Ops.bfloat16;
  Train.set_materialized u.value;
  let ctx = Train.forward_once ctx u in
  let values = Context.get_values ctx u.value in
  Stdio.printf "\n";
  Verdict.p_all
    (Printf.sprintf "uniform bfloat16: %d values, all in [0,1)" (Array.length values))
    (Array.to_list values)
    ~f:(fun v -> Float.(v >= 0. && v < 1.));
  (* Keep the uniform virtual so its consumer reads each packed block through the lane-extract
     builtin rather than [Set_from_vec]. This covers the other route by which a raw ushort return
     used to be converted numerically into a bfloat cell. *)
  let virtual_u = TDSL.uniform () ~output_dims:[ 9 ] () in
  Tn.update_prec virtual_u.value Ir.Ops.bfloat16;
  let%op virtual_values = virtual_u *. 1. in
  Tn.update_prec virtual_values.value Ir.Ops.bfloat16;
  Train.set_materialized virtual_values.value;
  let ctx = Train.forward_once ctx virtual_values in
  let values = Context.get_values ctx virtual_values.value in
  Verdict.p_all
    (Printf.sprintf "virtual uniform bfloat16: %d values, all in [0,1)" (Array.length values))
    (Array.to_list values)
    ~f:(fun v -> Float.(v >= 0. && v < 1.))
