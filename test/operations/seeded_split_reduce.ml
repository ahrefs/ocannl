(* gh-ocannl-1325: the seeded untuned default splits a long, narrow reduction.

   A bias added across [nb * ns] rows, backpropagated through [loss = sum ((x + b) *. dy)], so its
   gradient lowers as a training step's bias and layer-norm parameter gradients do -- [for b, s, o:
   b.grad[o] += dy[b,s,o]], one thread per output cell walking all [nb * ns] = 8192 rows in sequence
   under the default plan. [Autotune.seeded_default] splits it before segmenting: its serial chain
   per cell (8192) is above the default threshold of 4096, and the split reduction is detected on
   the batch loop through the enabling interchange, as the search detects it.

   The split reassociates the sum, so it is held to an f64 host reference within an f32 tolerance,
   beside the default pipeline's result held to the same; the inputs are NOT dyadic, so a split
   result bitwise equal to the default's would mean no split ran. A threshold above the serial chain
   splits nothing and must then be bitwise the default's, and a threshold of 0 disables the split.
   On CUDA and HIP the split must happen; elsewhere it must not. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let splits_here =
  String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

let nb = 128
and ns = 64
and no = 64

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let b =
  Operation.init ~l:"ssr_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x_values = values ~n:(nb * ns * no) ~stride:389
let dy_values = values ~n:(nb * ns * no) ~stride:577
let x = NTDSL.ndarray x_values ~label:[ "ssr_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
let dy = NTDSL.ndarray dy_values ~label:[ "ssr_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

(* The bias gradient in f64 on the host: every cell sums the f32-rounded [dy] over the rows. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)

let bgrad_exact =
  Array.init no ~f:(fun o ->
      let acc = ref 0. in
      for r = 0 to (nb * ns) - 1 do
        acc := !acc +. f32 dy_values.((r * no) + o)
      done;
      !acc)

let%op y = x + b
let%op loss = (y *. dy) ++ "... => 0"
let bgrad = (Option.value_exn ~here:[%here] b.Tensor.diff).Tensor.grad
let () = Train.set_materialized bgrad
let comp = Train.grad_update loss
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* Either summation order lands within 5e-5 of the f64 sum on this fixture (the stderr line below
   prints both errors); 1e-3 leaves a margin of twenty, while one dropped or doubled block of 64
   rows moves a cell by orders of magnitude more. *)
let tolerance = 1e-3

let within name (got : float array) =
  p_alli name (Array.to_list got) ~f:(fun o v -> Float.(abs (v - bgrad_exact.(o)) <= tolerance))

let max_error (got : float array) =
  Array.foldi got ~init:0. ~f:(fun o m v -> Float.max m (Float.abs (v -. bgrad_exact.(o))))

let run compile =
  let ctx, routine = compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  Context.get_values ctx bgrad

let seeded ~min_serial =
  let report = ref None in
  let values =
    run (fun ctx comp bindings ->
        Autotune.seeded_default ~min_serial ~report:(fun r -> report := Some r) ctx comp bindings)
  in
  (values, Option.value_exn ~here:[%here] !report)

let split_bgrad (r : Autotune.seeded_choice) =
  List.exists r.sd_split ~f:(fun (tn, _) -> Ir.Tnode.equal tn bgrad)

let () =
  let default = run Context.compile in
  let split, r = seeded ~min_serial:4096 in
  Stdio.eprintf "max error vs f64: default %g, seeded %g (not part of the golden)\n"
    (max_error default) (max_error split);
  List.iter r.Autotune.sd_split ~f:(fun (tn, blocks) ->
      Stdio.eprintf "split: %s into %d blocks (not part of the golden)\n" (Ir.Tnode.debug_name tn)
        blocks);
  Verdict.gated ~when_:splits_here ~on:backend_name
    "on CUDA/HIP the seeded default splits the bias gradient at a threshold below its serial chain"
    (split_bgrad r);
  Verdict.gated ~when_:splits_here ~on:backend_name
    "on CUDA/HIP the split routine compiles without falling back to the default"
    (not r.sd_fell_back);
  Verdict.gated ~when_:(not splits_here) ~on:backend_name
    "off CUDA/HIP the seeded default splits nothing" (List.is_empty r.sd_split);
  p_exists "the bias gradient is not trivially zero" (Array.to_list default) ~f:(fun v ->
      Float.(abs v > 1.));
  within "the default pipeline's bias gradient is within the f32 tolerance of the f64 reference"
    default;
  within "the seeded default's bias gradient is within the f32 tolerance of the f64 reference" split;
  Verdict.gated ~when_:splits_here ~on:backend_name
    "on CUDA/HIP the split changed the summation order: some cell differs from the default's"
    (Array.exists2_exn split default ~f:(fun a b -> not (bitwise a b)));
  (* The threshold: above the serial chain nothing qualifies, and 0 disables the split. *)
  let high, r_high = seeded ~min_serial:(nb * ns * 2) in
  p "a threshold above the serial chain splits nothing" (List.is_empty r_high.sd_split);
  p_all2 "that routine computes the default pipeline's bias gradient bitwise" high default
    ~f:bitwise;
  let _, r_off = seeded ~min_serial:0 in
  p "a threshold of 0 disables the split" (List.is_empty r_off.sd_split)
