open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* The issue's ordinary compose closes the unconstrained packed draw to rank zero. *)
let compose () =
  Tensor.unsafe_reinitialize ();
  let u = TDSL.uniform () () in
  let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * u) in
  let ctx = Train.forward_once (Context.auto ()) w_o in
  Verdict.p "compose closes the packed draw to rank zero"
    (Array.length (Lazy.force u.value.dims) = 0);
  Verdict.claim "compose produces two values" (Array.length (Context.get_values ctx w_o.value) = 2)

let draw ~packed ~materialized ~singleton =
  Tensor.unsafe_reinitialize ();
  let output_dims = if singleton then [ 1 ] else [] in
  let u =
    (if packed then TDSL.uniform () else TDSL.uniform1 ())
      ~batch_dims:[] ~input_dims:[] ~output_dims ()
  in
  if materialized then Train.set_materialized u.value;
  let%op urz = u *. 1. in
  let ctx = Train.forward_once (Context.auto ()) urz in
  let placements = Context.placements ctx in
  Verdict.pf "packed=%b materialized=%b singleton=%b uses the intended placement" packed
    materialized singleton
    (if materialized then Ir.Tnode.Placements.known_non_virtual placements u.value
     else Ir.Tnode.Placements.known_virtual placements u.value);
  Context.get_values ctx urz.value

let einsum () =
  Tensor.unsafe_reinitialize ();
  let st = TDSL.uniform () () in
  let%op urz_einsum = (st *. 2.) +* "...|h,w,..c..; ...|..c.. => ...|h" st in
  let ctx = Train.forward_once (Context.auto ()) urz_einsum in
  Verdict.p "einsum closes the packed draw to rank zero"
    (Array.length (Lazy.force st.value.dims) = 0);
  Context.get_values ctx urz_einsum.value

let () =
  Verdict.case "compose" compose;
  let reference = draw ~packed:false ~materialized:true ~singleton:false in
  Verdict.claim "pointwise rank-zero reference has one value" (Array.length reference = 1);
  Verdict.p_all "reference lies in [0,1)" (Array.to_list reference) ~f:(fun x ->
      Float.(x >= 0. && x < 1.));
  List.iter [ false; true ] ~f:(fun materialized ->
      List.iter [ false; true ] ~f:(fun singleton ->
          let values = draw ~packed:true ~materialized ~singleton in
          Verdict.pf_all2 "packed materialized=%b singleton=%b equals the first lane" materialized
            singleton values reference ~f:Float.equal));
  Verdict.case "einsum" @@ fun () ->
  (* All inferred axes have size one, so the contraction is 2*u*u. The pointwise draw above has the
     same self ids and seed. Allow the device's single-precision rounding. *)
  let expected = Array.map reference ~f:(fun u -> 2. *. u *. u) in
  Verdict.p_close "packed einsum equals 2*u*u" (einsum ()) expected ~tol:(fun _ -> 1e-6)
