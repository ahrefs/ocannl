(* gh-ocannl-1324: the seeded untuned default gives large GPU contractions operand reuse without
   changing a result.

   One dense layer with a bias, [y = w * x + b], backpropagated through [loss = sum (y *. dy)], so
   its weight gradient lowers as backprop lowers every weight gradient of a training step -- [for b,
   s, o, i: w.grad[o,i] += dy[b,s,o] * x[b,s,i]], the contraction loops outermost -- with a
   reduction extent of [nb * ns] = 256, while the forward product reduces over [ni] = 96. The bias
   gradient [b.grad[o] += dy[b,s,o]] reduces over the same rows, which is what lets the default
   fission keep it in the weight gradient's kernel as a companion no sketch covers: the seeding has
   to cut that segment finer to reach the site.

   The routine is compiled twice from fresh contexts, on the default pipeline and through
   [Autotune.seeded_default] at a threshold of 128 (between the two reduction extents), and both
   gradients are compared bitwise. The inputs are NOT dyadic: every partial sum rounds, so equality
   holds only if the seeded schedule keeps each output's summation order -- the property that keeps
   a training run's losses unchanged. On CUDA and HIP the seeding must seed, refine and not fall
   back; elsewhere it must leave the default alone. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* gh-ocannl-998: the bitwise claims are about summation order, so every product accumulation is an
   explicit FMA -- the contraction licence, pinned on here. With the licence off (the default) the
   update is a plain [acc + a * b], and whether it fuses is the GPU compiler's call, kernel by
   kernel: on HIP (gfx1102) clang's [-ffast-math] contracted the seeded tiled kernel and the default
   kernel differently, which is a different rounding, not a different order, and failed the
   claims. *)
let () = Unix.putenv "OCANNL_FP_CONTRACT" "true"

let seeds_here =
  String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

let nb = 4
and ns = 64
and no = 64
and ni = 96

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"sd_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"sd_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x_values = values ~n:(nb * ns * ni) ~stride:389
let dy_values = values ~n:(nb * ns * no) ~stride:577
let x = NTDSL.ndarray x_values ~label:[ "sd_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()
let dy = NTDSL.ndarray dy_values ~label:[ "sd_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

(* The fixture's order sensitivity, the control that gives the bitwise claims below their teeth:
   [w.grad] reduced on the host in f32, over the rows in REVERSE order. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)

let wgrad_reversed =
  Array.init (no * ni) ~f:(fun c ->
      let o = c / ni and i = c % ni in
      let acc = ref 0. in
      for r = (nb * ns) - 1 downto 0 do
        acc := f32 (!acc +. f32 (f32 dy_values.((r * no) + o) *. f32 x_values.((r * ni) + i)))
      done;
      !acc)

let%op y = (w * x) + b
let%op loss = (y *. dy) ++ "... => 0"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let wgrad = grad_of w
let bgrad = grad_of b

let () =
  Train.set_materialized wgrad;
  Train.set_materialized bgrad

let comp = Train.grad_update loss
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* One compile and run from a fresh context: the two gradients' values, and the seeding report. *)
let run compile =
  let ctx, routine = compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx wgrad, Context.get_values ctx bgrad)

let seeded ~min_reduction =
  let report = ref None in
  let values =
    run (fun ctx comp bindings ->
        Autotune.seeded_default ~min_reduction ~report:(fun r -> report := Some r) ctx comp bindings)
  in
  (values, Option.value_exn ~here:[%here] !report)

let () =
  let w_default, b_default = run Context.compile in
  let (w_seeded, b_seeded), r = seeded ~min_reduction:128 in
  List.iter r.Autotune.sd_seeded ~f:(fun (at, p) ->
      Stdio.eprintf "seeded: unit %d %s (not part of the golden)\n" at
        (Autotune.sketch_params_label p));
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the seeded default seeds a segment at a threshold below the weight gradient's \
     reduction"
    (not (List.is_empty r.sd_seeded));
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP it cuts the bias gradient's companion away from the weight gradient to seed it"
    (r.sd_refined >= 1);
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the seeded routine compiles without falling back to the default"
    (not r.sd_fell_back);
  Verdict.gated ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the seeded default seeds nothing"
    (List.is_empty r.sd_seeded && r.sd_refined = 0);
  p_exists "the weight gradient is not trivially zero" (Array.to_list w_default) ~f:(fun v ->
      Float.(abs v > 1e-3));
  p_exists "the fixture is order-sensitive: a reversed f32 reduction differs from it in some cell"
    (List.zip_exn (Array.to_list wgrad_reversed) (Array.to_list w_default))
    ~f:(fun (a, b) -> not (bitwise a b));
  p_all2 "the seeded weight gradient is bitwise the default pipeline's" w_seeded w_default
    ~f:bitwise;
  p_all2 "the seeded bias gradient is bitwise the default pipeline's" b_seeded b_default ~f:bitwise;
  (* The threshold: above every reduction extent nothing qualifies, and 0 disables the seeding. *)
  let (w_high, _), r_high = seeded ~min_reduction:(nb * ns * 2) in
  p "a threshold above every reduction extent seeds nothing"
    (List.is_empty r_high.sd_seeded && r_high.sd_refined = 0);
  p_all2 "that routine computes the default pipeline's weight gradient" w_high w_default ~f:bitwise;
  let _, r_off = seeded ~min_reduction:0 in
  p "a threshold of 0 disables the seeding" (List.is_empty r_off.sd_seeded)
