(* Load-time precision conversion for data-backed tensors (gh-ocannl-492, the forward-only leg).

   Data-backed tensors are precision-[Specified] at creation from their ndarray, so a storage policy
   cannot re-assign them; and with no optimizer there is no master copy for a cast twin to preserve.
   [TDSL.wrap ~prec] (via [Ir.Ndarray.convert]) re-precisions the data at ingestion instead —
   torch's [model.half ()] for inference. Pinned here:

   - The wrapped tensor's storage precision is the requested one, and its values are the
   half-rounded images of the f32 data (readback matches an explicit host-side rounding). - A matvec
   forward over bf16-converted weights tracks the f32 forward within bf16 tolerance — the end-to-end
   shape of bench_gpt's BENCH_PRECISION leg. - Same-precision wrap is the identity (values
   bitwise-equal to the plain wrap). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Stdio
module Tn = Ir.Tnode
open Verdict.Claims

let n = 12
let k = 16

let wv =
  Array.init (n * k) ~f:(fun i ->
      (Float.sin (Float.of_int i) *. 1.7) +. (Float.of_int (i % 5) *. 0.31))

let xv = Array.init k ~f:(fun i -> Float.cos (Float.of_int (3 * i)))

let nd_of values ~dims =
  Ir.Ndarray.init_array ~debug:"wpc" Ir.Ops.single ~dims ~padding:None ~f:(fun idcs ->
      let flat = Array.foldi idcs ~init:0 ~f:(fun ax acc i -> (acc * dims.(ax)) + i) in
      ignore flat;
      values.((idcs.(0) * dims.(1)) + idcs.(1)))

let () =
  Tensor.unsafe_reinitialize ();
  (* -- Precision and value pins of the conversion itself. -- *)
  let w_nd = nd_of wv ~dims:[| n; k |] in
  let w_half = TDSL.wrap ~l:"wpc_w_half" ~prec:Ir.Ops.half ~i:[ k ] ~o:[ n ] w_nd () in
  let ctx = Context.auto () in
  let ctx = Train.forward_once ctx w_half in
  p "wrap ~prec pins the storage precision"
    (Ir.Ops.equal_prec (Lazy.force w_half.Tensor.value.Tn.storage_prec) Ir.Ops.half);
  let got = Context.get_values ctx w_half.Tensor.value in
  (* Readback equals an explicit host-side half-rounding of the same data. *)
  let rounded =
    let tmp = Ir.Ndarray.convert Ir.Ops.half (nd_of wv ~dims:[| n; k |]) in
    Ir.Ndarray.retrieve_flat_values tmp
  in
  p "converted values equal the host-side half rounding" (Array.equal Float.equal got rounded);
  p "conversion actually rounds (data is not half-exact)" (not (Array.equal Float.equal got wv));

  (* -- Same-precision wrap is the identity. -- *)
  Tensor.unsafe_reinitialize ();
  let w_same =
    TDSL.wrap ~l:"wpc_w_same" ~prec:Ir.Ops.single ~i:[ k ] ~o:[ n ] (nd_of wv ~dims:[| n; k |]) ()
  in
  let ctx = Context.auto () in
  let ctx = Train.forward_once ctx w_same in
  let wv_f32 = Ir.Ndarray.retrieve_flat_values (nd_of wv ~dims:[| n; k |]) in
  p "same-precision wrap keeps the values bitwise"
    (Array.equal Float.equal (Context.get_values ctx w_same.Tensor.value) wv_f32);

  (* -- End-to-end: bf16-converted weights, f32 activations elsewhere — forward parity. -- *)
  Tensor.unsafe_reinitialize ();
  let make ~prec =
    let w =
      match prec with
      | Some prec -> TDSL.wrap ~l:"wpc_w" ~prec ~i:[ k ] ~o:[ n ] (nd_of wv ~dims:[| n; k |]) ()
      | None -> TDSL.wrap ~l:"wpc_w" ~i:[ k ] ~o:[ n ] (nd_of wv ~dims:[| n; k |]) ()
    in
    let x = TDSL.ndarray xv ~label:[ "wpc_x" ] ~output_dims:[ k ] () in
    let%op y = w * x in
    y
  in
  let y32 = make ~prec:None in
  let ctx = Context.auto () in
  let ctx = Train.forward_once ctx y32 in
  let want = Context.get_values ctx y32.Tensor.value in
  Tensor.unsafe_reinitialize ();
  let ybf = make ~prec:(Some Ir.Ops.bfloat16) in
  let ctx = Context.auto () in
  let ctx = Train.forward_once ctx ybf in
  let got = Context.get_values ctx ybf.Tensor.value in
  let close_tol b = Float.(0.05 *. (1. +. abs b)) in
  p_close "bf16-converted weights: forward tracks f32 within bf16 tolerance" got want ~tol:close_tol;
  printf "\nDone.\n%!"
