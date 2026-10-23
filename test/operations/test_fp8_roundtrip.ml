open Base
open Ocannl

(* Regression test for fp8 tensors on the configured backend (gh: fp8 kernels failed NVRTC
   compilation on CUDA). Exercises the scalar and vectorized uniform builtins, which draw in [0, 1)
   as every precision does (gh-ocannl-1486; they used to return the random byte as a raw e5m2
   pattern) by assembling the e5m2 byte themselves -- a numeric conversion of that byte to the
   vendor's fp8 type would print values of 1 and above -- and fp8 arithmetic, which bridges through
   float on every backend. Values are printed in hex float notation to stay platform-independent;
   the arithmetic part uses only exactly-representable fp8 values so all backends agree
   bit-for-bit. *)

let print_values header values =
  Stdio.printf "%s (%d values):\n" header (Array.length values);
  Array.iteri values ~f:(fun i v -> Stdio.printf "  [%d]: %h\n" i v)

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let open Nn_blocks.DSL_modules in
  let module O = TDSL.O in
  (* Random fp8 draws through the scalar uniform builtin. *)
  let counter1 = TDSL.range 4 in
  let fp8_bits1 = O.threefry4x32 (O.embed_self_id ()) counter1 |> O.uint4x32_to_prec_uniform1 in
  Ir.Tnode.update_prec fp8_bits1.value Ir.Ops.fp8;
  Train.set_materialized fp8_bits1.value;
  let ctx = Train.forward_once ctx fp8_bits1 in
  print_values "fp8 uniform1" (Context.get_values ctx fp8_bits1.value);

  (* Random fp8 draws through the vectorized uniform builtin (16 fp8s per uint4x32). *)
  let counter2 = TDSL.range 4 in
  let fp8_bits_vec = O.threefry4x32 (O.embed_self_id ()) counter2 |> O.uint4x32_to_prec_uniform in
  Ir.Tnode.update_prec fp8_bits_vec.value Ir.Ops.fp8;
  Train.set_materialized fp8_bits_vec.value;
  let ctx = Train.forward_once ctx fp8_bits_vec in
  let vec_values = Context.get_values ctx fp8_bits_vec.value in
  print_values "fp8 uniform vec, first 16" (Array.sub vec_values ~pos:0 ~len:16);
  Stdio.printf "fp8 uniform vec total: %d values\n" (Array.length vec_values);

  (* fp8 arithmetic round-trip: relu(range * 2 + (-3)); every intermediate is forced to fp8 and
     every result is exactly representable in E5M2 (no rounding, no overflow). *)
  let base = TDSL.range 4 in
  Ir.Tnode.update_prec base.value Ir.Ops.fp8;
  let doubled = O.( *. ) base (TDSL.number 2.0) in
  Ir.Tnode.update_prec doubled.value Ir.Ops.fp8;
  let shifted = O.( + ) doubled (TDSL.number (-3.0)) in
  Ir.Tnode.update_prec shifted.value Ir.Ops.fp8;
  let result = O.relu shifted in
  Ir.Tnode.update_prec result.value Ir.Ops.fp8;
  Train.set_materialized base.value;
  Train.set_materialized doubled.value;
  Train.set_materialized shifted.value;
  Train.set_materialized result.value;
  let ctx = Train.forward_once ctx result in
  print_values "base" (Context.get_values ctx base.value);
  print_values "doubled" (Context.get_values ctx doubled.value);
  print_values "shifted" (Context.get_values ctx shifted.value);
  print_values "fp8 relu(range * 2 - 3)" (Context.get_values ctx result.value)
