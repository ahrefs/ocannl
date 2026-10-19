(* Regression test for half-precision code generation on the GPU backends -- the f16 counterpart of
   [bf16_ops].

   [Relu_gate] at [Half_prec] used to emit the literal [0.0h] on the CUDA backend (gh-ocannl-518).
   The [h] suffix is a clang extension (and valid MSL), but not CUDA C++, so nvrtc rejected every
   kernel containing a half relu backward with "user-defined literal operator not found" -- blocking
   half-precision training on CUDA outright. The HIP backend had already sidestepped the literal via
   [__hgt] against a bitcast zero; CUDA now does the same.

   The gate only appears in the *backward* pass of [relu], so a forward-only test cannot reach it:
   this one runs fwd+bwd with the relu input's value and gradient, and the relu output's gradient,
   all at half precision. Materializing the intermediates keeps them in their own half nodes rather
   than being inlined into a single-precision temporary, which is what forces the half emission.

   Every value is exactly representable in half (10 mantissa bits), keeping the printed numbers
   backend-uniform. The zero entry pins the gate's boundary: [relu_gate] is a strict [> 0] test, so
   it must pass no gradient there. *)

open Base
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

(* A zero's sign is printed normalized, and its raw bits go to stderr. The sign of an exact zero is
   the one thing a backend's own floating-point licence may change here: HIP compiles with clang's
   [-ffast-math], whose [-fno-signed-zeros] lets it fold the [+0 + x * g] a gradient accumulation
   opens with once its covering zero is elided ([gpu_elide_covered_zeros]) down to [x * g] -- which
   is [-0] for [x = -2], [g = +0], where IEEE gives [+0] (observed on gfx1102: the kernel's
   [__hadd(__float2half(0.0), __hmul(x, z_grad))] stored [-0]). What this test pins is the half
   emission and the gate's boundary (a closed gate passes zero), which hold under either sign;
   OCANNL's own signed-zero licence is [fp_finite_math] (gh-ocannl-998), and it is off. *)
let show ctx label t =
  let values = Context.get_values ctx t in
  if Array.exists values ~f:(fun x -> Float.(x = 0.) && Float.ieee_negative x) then
    Stdio.eprintf "%s holds a negative zero: %s (not part of the golden)\n" label
      (String.concat ~sep:" " (Array.to_list (Array.map values ~f:(Printf.sprintf "%h"))));
  Stdio.printf "%s:" label;
  Stdio.printf " ";
  Test_utils.print_floats ~prec:4
    (Array.to_list (Array.map values ~f:(fun x -> if Float.(x = 0.) then 0. else x)));
  Stdio.printf "\n"

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let x = Tensor.term_init [| 1.; -2.; 0.; 4. |] ~label:[ "x" ] ~grad_spec:Require_grad () in
  let w = Tensor.term_init [| 0.5; 0.5; 0.5; 0.5 |] ~label:[ "w" ] ~grad_spec:Require_grad () in
  (* [z] is the relu input: its value is [relu_gate]'s first operand, its gradient the assignment's
     target, and [y]'s gradient the second operand -- all three at half is the emission under
     test. *)
  let%op z = x *. w in
  let%op y = relu z in
  let%op loss = y ++ "i=>0" in
  let grad t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad in
  List.iter [ x; w; z; y ] ~f:(fun t ->
      Tn.update_prec t.Tensor.value Ir.Ops.half;
      Tn.update_prec (grad t) Ir.Ops.half);
  List.iter [ x; w; z; y ] ~f:(fun t ->
      Train.set_materialized t.Tensor.value;
      Train.set_materialized (grad t));
  let ctx = Train.update_once ctx loss in
  (* z = [0.5; -1; 0; 2], y = relu z = [0.5; 0; 0; 2], loss = 2.5, y.grad = 1 everywhere. The gate
     is strict, so z.grad = [1; 0; 0; 1] -- the zero entry is closed. *)
  show ctx "z" z.Tensor.value;
  show ctx "y" y.Tensor.value;
  show ctx "loss" loss.Tensor.value;
  show ctx "z.grad (relu_gate at half)" (grad z);
  show ctx "x.grad" (grad x);
  show ctx "w.grad" (grad w);
  Stdio.printf "z prec: %s, z.grad prec: %s, y.grad prec: %s\n"
    (Ir.Ops.prec_string (Lazy.force z.Tensor.value.Tn.storage_prec))
    (Ir.Ops.prec_string (Lazy.force (grad z).Tn.storage_prec))
    (Ir.Ops.prec_string (Lazy.force (grad y).Tn.storage_prec))
