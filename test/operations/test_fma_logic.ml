(* Regression test for the ternary ~logic branch of ppx_cd: [fma ... ~logic:"."] and
   [fma ... ~logic:"@"] inside [%cd] used to expand to [Shape.Pointwise_bin] / [Shape.Compose]
   (compose_type constructors) where [Tensor.raw_ternop] expects [Shape.ternary_type], i.e.
   [Shape.Pointwise_tern] / [Shape.Compose_accumulate] -- producing ill-typed OCaml code. *)

open Base
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
module Tn = Ir.Tnode

(* A hosted, gradient-free tensor initialized from [vals]. *)
let make_tensor ?(input_dims = []) ~output_dims label vals =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| Array.length vals |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[] ~input_dims ~output_dims ()

let print_values ctx prefix t =
  let vals = Context.get_values ctx t.Tensor.value in
  Stdio.printf "%s = [%s]\n" prefix
    (String.concat ~sep:" " (Array.to_list (Array.map vals ~f:(Printf.sprintf "%g"))))

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in

  (* Pointwise ternary logic: out.(i) = a.(i) * x.(i) + b.(i). *)
  let a = make_tensor ~output_dims:[ 4 ] "a" [| 1.0; 2.0; 3.0; 4.0 |] in
  let x = make_tensor ~output_dims:[ 4 ] "x" [| 10.0; 20.0; 30.0; 40.0 |] in
  let b = make_tensor ~output_dims:[ 4 ] "b" [| 0.5; 0.5; 0.5; 0.5 |] in
  let%cd fwd = { out } =: fma a x b ~logic:"." in
  Train.set_materialized out.value;
  let routine = Train.to_routine ctx Train.IDX.empty fwd in
  let ctx = Context.context routine in  Train.run ctx routine;
  print_values ctx "fma pointwise (a*x + b)" out;
  Stdio.printf "expected = [10.5 40.5 90.5 160.5]\n";

  (* Compose-accumulate logic: out = p @ q + r, with p a 1x1 matrix. *)
  let p = make_tensor ~input_dims:[ 1 ] ~output_dims:[ 1 ] "p" [| 3.0 |] in
  let q = make_tensor ~output_dims:[ 1 ] "q" [| 2.0 |] in
  let r = make_tensor ~output_dims:[ 1 ] "r" [| 10.0 |] in
  let%cd fwd2 = { out2 } =: fma p q r ~logic:"@" in
  Train.set_materialized out2.value;
  let routine2 = Train.to_routine ctx Train.IDX.empty fwd2 in
  let ctx = Context.context routine2 in  Train.run ctx routine2;
  print_values ctx "fma compose-accumulate (p @ q + r)" out2;
  Stdio.printf "expected = [16]\n"
