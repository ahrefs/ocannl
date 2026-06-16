open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* Positive test: ~logic:"@" with * (matrix multiply) must still work after the
   prohibition of ~logic:"@" with / and **. *)

let make_tensor ?(input_dims = []) ~output_dims label vals =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| Array.length vals |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[] ~input_dims ~output_dims ()

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  (* a is 2x3 (output=2, input=3); b is 3x1 (output=3, input=1).
     c[i] = sum_k a[i,k] * b[k] = matrix-vector multiply. *)
  let a = make_tensor ~input_dims:[ 3 ] ~output_dims:[ 2 ] "a" [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let b = make_tensor ~input_dims:[ 1 ] ~output_dims:[ 3 ] "b" [| 1.0; 0.0; 1.0 |] in
  let%cd fwd = { c } =:+ a * b ~logic:"@" in
  Train.set_materialized c.value;
  let routine = Train.to_routine ctx Train.IDX.empty fwd in
  let ctx = Context.context routine in
  Train.run ctx routine;
  let vals = Context.get_values ctx c.value in
  (* a = [[1,2,3],[4,5,6]], b col-vec [1;0;1], c = [1*1+2*0+3*1, 4*1+5*0+6*1] = [4, 10] *)
  Stdio.printf "c = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map vals ~f:(Printf.sprintf "%g"))));
  Stdio.printf "expected = [4 10]\n"
