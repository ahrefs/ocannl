(* gh-ocannl-544: a use site cannot silently widen an operation result's row beyond its arguments' —
   the widening must be requested by name, with [stretch].

   Leg 1 (the gh-ocannl-540 failure shape, generalized): [relu w] read as the weight operand of a
   BATCHED matmul. Before gh-ocannl-544 the relu result's open batch row was resolved by the use
   site, materializing a [batch, out, in] per-batch-row copy of the weight — numerically invisible
   (every slice equal), caught by no parity gate. Now it closes down to the weight's shape and the
   matmul broadcasts it in.

   Leg 2: the same construction with [stretch] opts back into use-site resolution — the stretched
   weight acquires the batch axis, demonstrating the explicit escape hatch.

   Leg 3: the shape-inferred-constant idiom (formerly [0.5 + 0.5]): [stretch 1.0] as an einsum
   operand acquires the axes the spec demands, so projections can iterate them. *)

open! Base
open Ocannl.Operation.DSL_modules
module Train = Ocannl.Train
module Tn = Ir.Tnode

let batch = 8
let din = 4
let dout = 3
let dims_of t = Lazy.force t.Tensor.value.Tn.dims
let show_dims d = "[" ^ String.concat ~sep:"x" (Array.to_list d |> List.map ~f:Int.to_string) ^ "]"

(* Find the unique tensor in [root]'s subgraph whose debug name contains [sub]. *)
let find_by_name root sub =
  let visited = Hashtbl.create (module Int) in
  let rec walk t =
    if Hashtbl.mem visited t.Tensor.value.Tn.id then None
    else (
      Hashtbl.set visited ~key:t.Tensor.value.Tn.id ~data:();
      if String.is_substring (Tn.label_name t.Tensor.value) ~substring:sub then Some t
      else List.find_map t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
  in
  Option.value_exn ~here:[%here] (walk root)

let batched_input l =
  NTDSL.init ~l ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ din ]
    ~f:(function [| _; i |] -> Float.of_int i *. 0.25 | _ -> assert false)
    ()

let () =
  let x = batched_input "x1" in
  let%op y = relu { w = 0.5; o = [ dout ]; i = [ din ] } * x in
  let%op loss = y ++ "...|i=>0" in
  ignore (Train.forward_once (Context.auto ()) loss : Context.t);
  let h = find_by_name loss "relu" in
  let d = dims_of h in
  Verdict.pf "close-down: relu(w) %s | batch-free" (show_dims d) (Array.length d = 2)

let () =
  let x = batched_input "x2" in
  let%op y2 = stretch { w2 = 0.5; o = [ dout ]; i = [ din ] } * x in
  let%op loss2 = y2 ++ "...|i=>0" in
  ignore (Train.forward_once (Context.auto ()) loss2 : Context.t);
  let s = find_by_name loss2 "stretch" in
  let d = dims_of s in
  Verdict.pf "stretch: stretch(w2) %s | widened-to-batch" (show_dims d)
    (Array.length d = 3 && d.(0) = batch)

let () =
  let x3 = batched_input "x3" in
  let%op total = x3 +* "...|i; |i => 0" (stretch 1.0) in
  ignore (Train.forward_once (Context.auto ()) total : Context.t);
  let ones = find_by_name total "stretch" in
  Stdio.printf "kernel: stretch(1.0) %s\n" (show_dims (dims_of ones))

(* Dim-level close-down: the sum of two fixed scalars has one open axis variable; consumed pointwise
   by a [din]-wide tensor, that axis used to widen to the use site's dim. Now an unmarked,
   unconstrained axis is guessed minimal and broadcasts — the pre-544 [0.5 + 0.5] result no longer
   silently inherits its consumer's width (e.g. a learning-rate expression meeting a single
   parameter shape stays scalar). *)
let () =
  let x4 = batched_input "x4" in
  let one = TDSL.O.( + ) (TDSL.number 0.5) (TDSL.number 0.5) in
  let%op y4 = one *. x4 in
  let%op loss4 = y4 ++ "...|i=>0" in
  ignore (Train.forward_once (Context.auto ()) loss4 : Context.t);
  let d = dims_of one in
  Verdict.pf "dim close-down: 0.5+0.5 %s | scalar" (show_dims d) (Array.fold d ~init:1 ~f:( * ) = 1)
