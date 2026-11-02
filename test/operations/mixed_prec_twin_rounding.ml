(* gh-ocannl-1769, gh-ocannl-1817: does a mixed-precision cast twin round its master, whichever
   placement it gets?

   The master holds [1 + 2^-12], which f16 rounds to [1] (its ulp at 1 is [2^-10]), and the model
   reads the twin once, pointwise, into an f32 output: [y = twin *. 1]. A twin that rounds gives [y
   = 1]; one whose rounding is skipped gives [y = 1 + 2^-12]. Both values are exact.

   The materialized twin is stored at f16, so it rounds by construction, on every backend. A virtual
   twin is inlined as an f16 scope over a load of the f32 master, read in an f32 slot; since
   gh-ocannl-1788 [Low_level.simplify_llc] keeps that scope (collapsing it would skip a rounding f16
   does not widen exactly), and a kept scope rounds to the precision its local is declared at: the
   backend's compute precision of f16 ([Context.codegen_capabilities]'s [compute_prec]). That is f16
   on the GPU backends, and on cc under [narrow_compute_f32=false]; single on cc's default, where
   the twin reads its master unrounded. So the virtual twin's value is a code-generation fact, and
   the claim derives it from that fact rather than printing a value that differs by backend (its
   digits go to stderr). [Twin_auto] is whichever placement the virtualization heuristics chose,
   which the placement settles per compilation and the test does not read back: its value is the
   materialized twin's or the virtual twin's. *)

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
  let half_computes_at =
    (Context.codegen_capabilities ctx).Ir.Backend_intf.compute_prec Ir.Ops.half
  in
  Stdio.eprintf "%s: y = %h, f16 computes at %s (not part of the golden)\n" label v
    (Ir.Ops.prec_string half_computes_at);
  (v, half_computes_at)

open Verdict.Claims

let () =
  let materialized, _ = leg ~label:"Twin_materialized" MP.Twin_materialized in
  Stdio.printf "Twin_materialized: y = %h\n" materialized;
  p "the materialized twin rounds the master to f16" Float.(materialized = 1.);
  let virtual_, half_computes_at = leg ~label:"Twin_virtual" MP.Twin_virtual in
  (* The kept f16 scope rounds exactly where its local is declared f16. *)
  let virtual_expected =
    if Ir.Ops.equal_prec half_computes_at Ir.Ops.half then 1. else master_value
  in
  p "the virtual twin rounds the master exactly where f16 computes at f16"
    Float.(virtual_ = virtual_expected);
  let auto, _ = leg ~label:"Twin_auto" MP.Twin_auto in
  p "the auto twin gives the materialized twin's value or the virtual twin's"
    Float.(auto = materialized || auto = virtual_expected)
