(* gh-ocannl-1769: does a mixed-precision cast twin round its master, whichever placement it gets?

   The master holds [1 + 2^-12], which f16 rounds to [1] (its ulp at 1 is [2^-10]), and the model
   reads the twin once, pointwise, into an f32 output: [y = twin *. 1]. A twin that rounds gives [y
   = 1]; one whose cast was dropped gives [y = 1 + 2^-12]. Both values are exact, so they print.

   The materialized twin is stored at f16, so it rounds by construction. A virtual twin is inlined
   as a scope of its own precision, and [Low_level.simplify_llc] collapses a single-assignment scope
   into its reader -- dropping the scope's conversion: the twin's reads become reads of the f32
   master. This census pins what each placement does today ([0x1p+0] rounded, [0x1.001p+0] not);
   [Twin_auto] is whichever of the two the virtualization heuristics chose, and the row says
   which. *)

open! Base
open Ocannl.Operation.DSL_modules
module Train = Ocannl.Train
module MP = Ocannl.Mixed_prec
module Tn = Ir.Tnode

(* [1 + 2^-12]: the parameter initializer below takes it as a literal. *)
let master_value = 1. +. (2. **. -12.)

let leg ~label placement =
  Tensor.unsafe_reinitialize ();
  let make () =
    let%op f x = { w = 1.000244140625; o = [ 1 ] } *. x in
    f
  in
  let f = MP.with_master_weights ~placement ~prec:Ir.Ops.half make in
  let%op y = f 1.0 in
  Tn.update_prec y.Tensor.value Ir.Ops.single;
  Train.set_materialized y.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) y in
  let v = (Context.get_values ctx y.Tensor.value).(0) in
  Stdio.printf "%s: y = %h\n" label v;
  v

open Verdict.Claims

let () =
  let materialized = leg ~label:"Twin_materialized" MP.Twin_materialized in
  p "the materialized twin rounds the master to f16" Float.(materialized = 1.);
  (* Census rows, not claims: the virtual twin's value is the finding the golden records. Either
     value is one of the two exact candidates, which is what keeps the rows honest. *)
  List.iter
    [ ("Twin_virtual", MP.Twin_virtual); ("Twin_auto", MP.Twin_auto) ]
    ~f:(fun (label, placement) ->
      let v = leg ~label placement in
      p (label ^ ": y is the rounded or the unrounded master") Float.(v = 1. || v = master_value))
