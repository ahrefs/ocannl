open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* The issue's ordinary compose closes the unconstrained packed draw to rank zero. *)
let compose () =
  Tensor.unsafe_reinitialize ();
  let u = TDSL.uniform () () in
  let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * u) in
  let ctx = Train.forward_once (Context.auto ()) w_o in
  Verdict.p_all "compose closes the packed draw to rank zero" [ u ] ~f:(fun t ->
      Array.is_empty (Lazy.force t.value.dims));
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
  Context.get_values ctx urz.value

let () =
  Verdict.case "compose" compose;
  let reference = draw ~packed:false ~materialized:true ~singleton:false in
  Verdict.claim "pointwise rank-zero reference has one value" (Array.length reference = 1);
  Verdict.p_all "reference lies in [0,1)" (Array.to_list reference) ~f:(fun x ->
      Float.(x >= 0. && x < 1.));
  List.iter [ false; true ] ~f:(fun materialized ->
      List.iter [ false; true ] ~f:(fun singleton ->
          let values = draw ~packed:true ~materialized ~singleton in
          Verdict.claimf "packed materialized=%b singleton=%b equals the first lane" materialized
            singleton
            (Array.equal Float.equal values reference)))
