(* gh-ocannl-616, the end-to-end form of test/operations/footprint_materialization: a matrix product
   with a reduction extent above [virtualize_max_inline_reduction], read only through the diagonal
   extraction [a ++ "ii => i"]. The reduction cap used to materialize the whole [n×n] product; the
   footprint form leaves it virtual and computes the [n] diagonal cells into an [n]-sized scratch
   right after the product's own nest.

   Unlike the hand-built rows, this goes through the whole default pipeline of the ambient backend —
   [Assignments] lowering, the schedule annotator (on a GPU the scratch crosses the statement
   boundary between the prologue and the reader, so [promote_statement_crossing_locals] lifts it to
   [On_device] and fission cuts a kernel there: the two-launch form the issue ships first), codegen
   and linking. The structure is read off the hermetic [Context.lowered_for_decisions] record; the
   values are compared with an OCaml oracle and with the same program compiled with the product
   pre-decided materialized. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let n = 8

(* Above the reduction cap's default of 16. *)
let k = 32
let () = assert (LL.virtualize_settings.max_inline_reduction = 16)
let () = assert LL.virtualize_settings.footprint_materialization

let () =
  Tensor.unsafe_reinitialize ();
  let x = TDSL.range_of_shape ~output_dims:[ n; k ] () in
  let y = TDSL.range_of_shape ~output_dims:[ k; n ] () in
  let%op a = x +* "ik; kj => ij" y in
  let%op d = a ++ "ii => i" in
  let comp = Train.forward d in
  let ctx = Context.auto () in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name ctx);
  let opt = Context.lowered_for_decisions ctx comp Ir.Indexing.Empty in
  let plc = opt.LL.optimize_ctx.LL.placements in
  p "einsum: the product stays virtual" (Tn.Placements.known_virtual plc a.Tensor.value);
  p "einsum: no buffer read of the product survives" (Ll_test.count_get opt a.Tensor.value = 0);
  let scratches =
    Hashtbl.keys opt.LL.traced_store
    |> List.filter ~f:(fun tn -> String.equal tn.Tn.namespace LL.footprint_namespace)
  in
  p "einsum: exactly one scratch, of the diagonal's extent"
    (List.equal (List.equal Int.equal)
       (List.map scratches ~f:(fun tn -> Array.to_list (Lazy.force tn.Tn.dims)))
       [ [ n ] ]);
  (* x[i, kk] = k i + kk and y[kk, j] = n kk + j, both row-major ranges. *)
  let expected =
    Array.init n ~f:(fun i ->
        Float.of_int
          (List.sum
             (module Int)
             (List.init k ~f:Fn.id)
             ~f:(fun kk -> ((k * i) + kk) * ((n * kk) + i))))
  in
  let run ctx =
    let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    Context.get_values ctx d.Tensor.value
  in
  let got = run ctx in
  let mat = run (Context.decide_materialized ctx [ a.Tensor.value ]) in
  let close a b = Float.(abs (a -. b) <= 1e-6 *. Float.max 1. (abs b)) in
  p_all2 "einsum: executed values are the diagonal of the product" got expected ~f:close;
  p_all2 "einsum: footprint and materialized readings agree" got mat ~f:close
