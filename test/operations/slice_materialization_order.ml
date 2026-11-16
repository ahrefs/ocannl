(* Ordinary slices have no parameter-imposed storage intent. The helper is the only setup that
   materializes this parent and view, so a copying slice cannot inline into its later consumer and
   hide a reader emitted before its producer. All values are small exact integers. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

let () =
  let index, bindings = Train.IDX.get_static_symbol ~static_range:2 Train.IDX.empty in
  let data =
    NTDSL.init ~l:"smo_data" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 4 ]
      ~f:(function [| b; j |] -> Float.of_int ((10 * b) + j + 1) | _ -> assert false)
      ()
  in
  let parent = NTDSL.pointmul data (NTDSL.number 2.) () in
  let owner = NTDSL.pointmul parent (NTDSL.number 3.) () in
  let reader = NTDSL.slice index parent () in
  let later = NTDSL.add owner (NTDSL.number 1.) () in
  let out = NTDSL.add reader later () in
  Ll_test.materialize_copying_slice ~copy_prec:Ir.Ops.half ~array:reader.Tensor.value
    ~sliced:parent.Tensor.value;
  Train.set_materialized out.Tensor.value;
  let base = Context.auto () in
  let ctx, routine =
    Context.compile ~name:"smo_order" base (Tensor.consume_forward_code out) bindings
  in
  Train.IDX.find_exn routine.bindings index := 1;
  let ctx = Context.run ctx routine in
  p "the ordinary slice copies its parent"
    (not
       (Ir.Assignments.slice_alias_eligible ~array:reader.Tensor.value ~sliced:parent.Tensor.value));
  p "the materialized slice reads its producer in order"
    (Array.equal Float.equal
       (Context.get_values ctx out.Tensor.value)
       (Array.init 8 ~f:(fun i -> Float.of_int (29 + (60 * (i / 4)) + (8 * (i % 4))))));
  Context.release ctx;
  Context.release base
