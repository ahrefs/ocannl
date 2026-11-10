open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

(* A capture's size claims no basis in projection derivation (gh-ocannl-2033). A capture reference
   publishes a bare size, and projection derivation re-solves each operation with it, as a
   [default]-tagged dimension. So a capture whose dimension solved to any other tag was refused
   there with "different bases": a free one, which closes to the broadcast top [1_(bcast_if_1)], and
   one an operand named, here [rgb]. The size now stands only where the operation's closed axes
   leave the dimension open. A genuine conflict, a named dimension met with a [default] one, is
   still refused, by the inference itself. *)

let open_leaf name = NTDSL.term ~label:[ name ] ~fetch_op:(Constant 2.) ()
let rgb3 () = NTDSL.ndarray [| 1.; 2.; 3. |] ~output_axes:[ ("rgb", 3) ] ()
let plain d = NTDSL.init ~l:"plain" ~prec:Ir.Ops.single ~b:[] ~o:[ d ] ~f:(fun _ -> 1.) ()
let dims t = Array.to_list (Lazy.force t.Tensor.value.dims)
let dims_str t = String.concat ~sep:"x" (List.map (dims t) ~f:Int.to_string)
let has_dims t expected = List.equal Int.equal (dims t) expected
let capture n t = NTDSL.einsum1 ~capture_dims:[ n ] "n => n" t ()

let solved_to n d =
  Option.equal Int.equal n.Shape.var_ref.solved_dim (Some d)
  && Option.is_none n.Shape.var_ref.solved_sym

(* A case's own verdict: the rewind keeps a refusal from reaching the cases after it. *)
let case label f =
  Verdict.case label (fun () ->
      try f ()
      with Row.Shape_error _ as e ->
        Tensor.rewind_to_last_closure ~reason:"gh-2033 refused case" ();
        raise e)

let refused_with name ~sub f =
  let message =
    match f () with
    | () -> None
    | exception Row.Shape_error (msg, _) ->
        Tensor.rewind_to_last_closure ~reason:"gh-2033 refusal case" ();
        Some msg
  in
  Stdio.printf "refusal: %s\n" (Option.value message ~default:"none");
  p name (Option.exists message ~f:(fun msg -> String.is_substring msg ~substring:sub))

(* The values [y] computes, in a context of its own. *)
let values y =
  Train.set_materialized y.Tensor.value;
  Ir.Tnode.set_observable y.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) y in
  let vs = Array.to_list (Context.get_values ctx y.Tensor.value) in
  Context.release ctx;
  vs

let () =
  case "a free captured dimension" (fun () ->
      let n = Shape.get_variable_ref "n" in
      let y = capture n (open_leaf "q") in
      Stdio.printf "free: %s\n" (dims_str y);
      p "a free captured dimension closes to 1" (has_dims y [ 1 ] && solved_to n 1);
      p_all ~min:1 "the free capture computes its operand" (values y) ~f:(Float.equal 2.));
  case "a captured named dimension" (fun () ->
      let n = Shape.get_variable_ref "n" in
      let y = capture n (rgb3 ()) in
      Stdio.printf "named: %s\n" (dims_str y);
      p "a captured rgb dimension is the operand's" (has_dims y [ 3 ] && solved_to n 3);
      p "the named capture computes its operand" (List.equal Float.equal (values y) [ 1.; 2.; 3. ]));
  (* Controls: a named dimension against a [default] one is a conflict of its own. *)
  let bases = "different bases" in
  refused_with "a named and a default dimension are refused as the second registers" ~sub:bases
    (fun () ->
      let n = Shape.get_variable_ref "n" in
      let _y = capture n (rgb3 ()) in
      ignore (capture n (plain 3) : Tensor.t));
  case "a free and a plain 1 capture in one inference" (fun () ->
      let n = Shape.get_variable_ref "n" in
      let y = capture n (open_leaf "q") and z = capture n (plain 1) in
      p "a free and a plain 1 capture in one inference agree" (has_dims y [ 1 ] && has_dims z [ 1 ]))
