(* gh-ocannl-540: a mixed-precision cast twin must have its master's shape.

   [Operation.cast] is a shape-INFERRED pointwise op, so the twin's batch row starts as an open row
   variable. Read as the weight operand of a BATCHED matmul, that row variable is resolved by the
   use site: the batch axis broadcasts into the twin and it materializes as [batch, out, in] — a
   per-batch-row copy of a weight that has one value for the whole batch.

   Nothing downstream is wrong numerically (every slice holds the same value), which is why the
   loss-trajectory oracle in test/training/mixed_prec_parity.ml never saw it: that model's input
   carries no batch axis, so the twin had nothing to broadcast. The costs are (a) [batch]x the
   twin's memory and cast work every step, and (b) the row symbol appears in the matmul's weight
   operand, which made EVERY tensorized candidate decline on a backend whose only tensor-core route
   is a reduced input format — unstaged seeds fail [Schedule.Tensorize]'s unit-coefficient role
   check (the third micro symbol occurs in an operand), and staged seeds pin the cooperative [Stage]
   load nest inside the row loop (the row symbol is an outer-part symbol of the staged source),
   breaking the perfect nest Tensorize requires. On HIP that was 16 of 28 mma candidates, and
   mma_timed = 0.

   So this pins the shape itself, at every reduced precision and with a batch axis present. *)

open! Base
open Ocannl.Operation.DSL_modules
module Train = Ocannl.Train
module MP = Ocannl.Mixed_prec
module Tn = Ir.Tnode

let batch = 8
let din = 4
let dout = 3

(* The cast twin is the unique direct consumer of the master parameter. *)
let find_consumer root master =
  let visited = Hashtbl.create (module Int) in
  let rec walk t =
    if not (Hashtbl.mem visited t.Tensor.value.Tn.id) then (
      Hashtbl.set visited ~key:t.Tensor.value.Tn.id ~data:();
      if
        List.exists t.Tensor.children ~f:(fun c ->
            c.Tensor.subtensor.Tensor.value.Tn.id = master.Tensor.value.Tn.id)
      then Some t
      else List.find_map t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
    else None
  in
  Option.value_exn ~here:[%here] (walk root)

let dims_of t = Lazy.force t.Tensor.value.Tn.dims
let show_dims d = "[" ^ String.concat ~sep:"x" (Array.to_list d |> List.map ~f:Int.to_string) ^ "]"

let leg ~label ~prec =
  (* A batched input: [batch] rows of [din], so the matmul's weight operand is the thing whose batch
     row could be resolved by the use site. *)
  let x =
    NTDSL.init ~l:("x_" ^ label) ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ din ]
      ~f:(function [| _; i |] -> Float.of_int i *. 0.25 | _ -> assert false)
      ()
  in
  let make () =
    let%op f x = { w = 0.5; o = [ dout ]; i = [ din ] } * x in
    f
  in
  let f = MP.with_master_weights ~prec make in
  let y = f x in
  let%op loss = y ++ "...|i=>0" in
  Train.set_materialized y.Tensor.value;
  (* Shape inference is only forced by lowering. *)
  ignore (Train.forward_once (Context.auto ()) loss : Context.t);
  let master =
    List.find_exn (Set.to_list loss.Tensor.params) ~f:(fun t ->
        String.is_prefix (Tn.label_name t.Tensor.value) ~prefix:"w")
  in
  let twin = find_consumer loss master in
  let m_dims = dims_of master and t_dims = dims_of twin in
  Verdict.pf "%s: master %s | twin %s | equal" label (show_dims m_dims) (show_dims t_dims)
    (Array.equal Int.equal m_dims t_dims)

let () =
  leg ~label:"bf16" ~prec:Ir.Ops.bfloat16;
  leg ~label:"f16" ~prec:Ir.Ops.half
