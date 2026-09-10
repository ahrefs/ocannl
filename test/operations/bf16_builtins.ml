(* Regression test for bfloat16 math-builtin code generation on the GPU backends -- the
   builtin-return-type counterpart of [bf16_ops], which covers the operand-ambiguity family.

   MSL's math library has [float] and [half] overloads but no [bfloat] ones, so a builtin called on
   bfloat operands promotes them and returns [float]; unlike C, MSL then rejects the narrowing
   assignment back to a [bfloat] destination ("assigning to 'bfloat' from incompatible type
   'float'"). Every math builtin is affected, not just the few whose ambiguity the GPU compilers
   report: [sqrt] (Nn_blocks.layer_norm) and [fmax] (the softmax max-reduction) are the two sites
   that block gpt2_mini at bf16 (gh-ocannl-549).

   The inputs are powers of two, so every printed result is exact in bfloat16 (7 mantissa bits) and
   backend-uniform. The transcendentals are checked by range instead: a reduced-precision golden
   cannot pin a result whose last bits depend on the backend's math library. *)

open Base
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

let bf16 t =
  Tn.update_prec t.Tensor.value Ir.Ops.bfloat16;
  (* Materializing keeps the result in its own bfloat16 node, so the op is emitted at bfloat16
     rather than being inlined into a single-precision temporary. *)
  Train.set_materialized t.Tensor.value;
  t

(* Precision without materialization: the builtin's result stays virtual and is inlined into
   whatever consumes it. *)
let bf16_virtual t =
  Tn.update_prec t.Tensor.value Ir.Ops.bfloat16;
  t

let show ctx label t =
  Stdio.printf "%s: " label;
  Test_utils.print_floats ~prec:4 (Array.to_list (Context.get_values ctx t.Tensor.value));
  Stdio.printf "\n"

let finite ctx label t ~lo ~hi =
  let values = Context.get_values ctx t.Tensor.value in
  Verdict.p_all
    (Printf.sprintf "%s: %d values, all in [%g, %g]" label (Array.length values) lo hi)
    (Array.to_list values)
    ~f:(fun v -> Float.(v >= lo && v <= hi))

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let x = bf16 @@ Tensor.term_init [| 4.; 16.; 0.25; 1. |] ~label:[ "x" ] () in
  let y = bf16 @@ Tensor.term_init [| 2.; 0.5; 8.; 3. |] ~label:[ "y" ] () in
  (* [sqrt] -- the layer_norm site. *)
  let%op sqrt_x = sqrt x in
  (* [fmax] / [fmin] -- the softmax max-reduction site, and its dual. *)
  let%op max_x = x @^^ "i=>0" in
  let%op max_xy = (x @^^ "i=>0") + (y @^^ "i=>0") in
  (* [pow], [1/x], [rsqrt] and the transcendentals: same builtin-overload gap, different names. *)
  let%op pow_xy = x **. 2 in
  let%op recip_x = recip x in
  let%op rsqrt_x = recip_sqrt x in
  let%op exp_y = exp y in
  let%op log_x = log x in
  let%op log2_x = log2 x in
  let%op sin_y = sin y in
  let%op cos_y = cos y in
  let%op tanh_y = tanh y in
  (* [Satur01] and [Relu] were already bridged; keep them in the same emission to pin them. *)
  let%op sat_x = sat01 (x - 8.) in
  let%op relu_x = relu (x - 8.) in
  (* The placement half of the defect. Everything above stores the builtin's result in its own
     bfloat16 node, which CUDA and HIP accept even unbridged (their converting constructor from
     float is implicit) -- only Metal rejects it. Leaving the builtin virtual instead inlines it
     into the consuming bfloat16 binop, where the float operand is what nvrtc reports as a
     mixed-operand [__hadd] and hiprtc as an ambiguous [operator '+']. That is why gpt2_mini at bf16
     compiles in the materialized placement on those two backends and not in the default one:
     nothing introduces a float, inlining just moves the builtin's own float result from an
     assignment into an operand.

     sqrt x = [2; 4; 0.5; 1] and exp (x - x) = 1, both exact, so the sums and products are too. *)
  let%op sqrt_inline = sqrt x in
  let%op inlined_sqrt = y + sqrt_inline in
  let%op zeroed = x - x in
  let%op exp_inline = exp zeroed in
  let%op inlined_exp = y *. exp_inline in
  List.iter [ sqrt_inline; zeroed; exp_inline ] ~f:(fun t -> ignore (bf16_virtual t));
  let exact = [ ("sqrt", bf16 sqrt_x); ("max", bf16 max_x); ("max sum", bf16 max_xy) ] in
  let exact =
    exact
    @ [
        ("pow 2", bf16 pow_xy);
        ("recip", bf16 recip_x);
        ("rsqrt", bf16 rsqrt_x);
        ("sat01", bf16 sat_x);
        ("relu", bf16 relu_x);
        ("inlined sqrt", bf16 inlined_sqrt);
        ("inlined exp", bf16 inlined_exp);
      ]
  in
  let ranged =
    [
      ("exp", bf16 exp_y, 0., 3000.);
      ("log", bf16 log_x, -2., 3.);
      ("log2", bf16 log2_x, -2., 4.);
      ("sin", bf16 sin_y, -1., 1.);
      ("cos", bf16 cos_y, -1., 1.);
      ("tanh", bf16 tanh_y, -1., 1.);
    ]
  in
  let ctx =
    List.fold
      (List.map exact ~f:snd @ List.map ranged ~f:(fun (_, t, _, _) -> t))
      ~init:ctx
      ~f:(fun ctx t -> Train.forward_once ctx t)
  in
  List.iter exact ~f:(fun (label, t) -> show ctx label t);
  List.iter ranged ~f:(fun (label, t, lo, hi) -> finite ctx label t ~lo ~hi)
