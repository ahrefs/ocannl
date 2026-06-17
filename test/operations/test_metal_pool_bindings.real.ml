(* Regression test for gh-ocannl-344 AC 4: the Metal backend reaches a kernel's materialized tensor
   nodes through a small fixed set of bound pool buffers + a slot table, not one argument binding per
   node. This sums 40 distinct ndarray-backed constants in one expression, so the forward kernel
   materializes 40 constant reads plus the output -- well past Metal's ~31 argument-buffer binding
   limit.

   The invariant pinned: a kernel that reads > 31 materialized nodes still launches and computes the
   correct result. Under the old one-buffer-per-tnode binding this kernel would issue 40+ [set_buffer]
   calls and could not be encoded; with pooled binding the 40 constants share a constant pool reached
   via the in-shader pool array, so only a handful of pools are bound. If Metal regressed to
   O(num_tnodes) bindings, encoding this kernel would fail (or silently mis-bind) and the
   "correct sum" assertion below would not hold. The harness condition that instantiates the AC is
   [n = 40 > 31]: a small-graph variant would pass even under the old per-tnode binding. *)

open! Base
open Ocannl
open Operation.DSL_modules

let make_const label v =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| 2 |] in
  Genarray.set ga [| 0 |] v;
  Genarray.set ga [| 1 |] (v +. 0.5);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ] ~batch_dims:[]
    ~input_dims:[] ~output_dims:[ 2 ] ()

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.metal () in
  let n = 40 in
  let consts =
    List.init n ~f:(fun i -> make_const (Printf.sprintf "c%d" i) (Float.of_int (i + 1)))
  in
  let sum = List.reduce_exn consts ~f:(fun a b -> TDSL.O.(a + b)) in
  let ctx = Train.forward_once ctx sum in
  let got = Context.get_values ctx sum.Tensor.value in
  (* sum_i (i+1) = 820 ; sum_i (i+1.5) = 820 + 40*0.5 = 840 *)
  let expected = [| 820.; 840. |] in
  Stdio.printf "materialized nodes read by one kernel = %d\n" n;
  Stdio.printf "exceeds Metal ~31 binding limit = %b\n" (n > 31);
  Stdio.printf "pooled binding computed the correct sum = %b\n"
    (Array.length got = 2 && Array.for_all2_exn got expected ~f:(fun a b -> Float.(abs (a - b) < 1e-2)))
