open! Base
open Ocannl.Nn_blocks.DSL_modules
module Row = Ocannl.Row

(* gh-ocannl-2154 is a known inference gap, not an unsatisfiable spec. Add the concrete witness to
   the ORIGINAL registered store with [infer_equal]: these are additional equations, not a
   replacement of the operations or their constraints. The second case pins the current false
   refusal, so a design fix can replace it with acceptance and an executed value check. *)
let build () =
  let st = TDSL.uniform () () in
  let x = TDSL.range_of_shape ~output_dims:[ 3; 4; 5 ] () in
  let%op p = st *. !.2. in
  let%op y = p +* "...|h,w,..c..; ...|..c.. => ...|h" st in
  let%op q = p *. x in
  let%op z = y + q in
  (st, p, y, q, z)

let () =
  Verdict.case "concrete witness" @@ fun () ->
  Tensor.unsafe_reinitialize ();
  let witness =
    try
      let st, p, y, q, z = build () in
      let scalar = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[] () in
      let wide =
        TDSL.range_of_shape ~batch_dims:[] ~input_dims:[]
          ~output_axes:[ (Row.bcast_if_1, 1); (Row.default_basis, 5) ]
          ()
      in
      let one =
        TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_axes:[ (Row.bcast_if_1, 1) ] ()
      in
      let full = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3; 4; 5 ] () in
      Shape.infer_equal st.Tensor.shape scalar.Tensor.shape;
      Shape.infer_equal p.Tensor.shape wide.Tensor.shape;
      Shape.infer_equal y.Tensor.shape one.Tensor.shape;
      Shape.infer_equal q.Tensor.shape full.Tensor.shape;
      Shape.infer_equal z.Tensor.shape full.Tensor.shape;
      ignore (Shape.to_dims z.Tensor.shape : int array);
      List.map
        [ (st, []); (p, [ 1; 5 ]); (y, [ 1 ]); (q, [ 3; 4; 5 ]); (z, [ 3; 4; 5 ]) ]
        ~f:(fun (t, want) -> (Array.to_list (Shape.to_dims t.Tensor.shape), want))
    with Row.Shape_error (msg, trace) ->
      Stdio.eprintf "witness refused: %s\n%!" (Shape.error_to_string_hum msg trace);
      []
  in
  Verdict.p_all ~min:5 "the original store accepts all five witness shapes" witness
    ~f:(fun (got, want) -> List.equal Int.equal got want)

let has_shape id row = List.mem (Row.row_shapes row) id ~equal:Int.equal

let is_gap_trace ~p_id ~q_id trace =
  let remaining_equality =
    List.exists trace ~f:(function
      | Row.Constraint_failed (Row.Row_eq { r1; r2; origin }) ->
          has_shape p_id r1 && has_shape q_id r2
          && List.exists origin ~f:(fun o ->
              Poly.equal o.lhs_kind `Output && Poly.equal o.rhs_kind `Output
              && Option.equal String.equal o.operation (Some "remaining rows"))
      | _ -> false)
  in
  let conflicting_remainders =
    List.exists trace ~f:(function
      | Row.Row_mismatch
          [
            ({ beg_dims = []; dims = [ Dim { d = 4; _ } ]; bcast = Row_var _; _ } as required);
            ({ beg_dims = []; dims = []; bcast = Broadcastable; _ } as closed);
          ] ->
          has_shape q_id required && has_shape p_id closed
      | _ -> false)
  in
  remaining_equality && conflicting_remainders

let () =
  Verdict.case "known inference gap" @@ fun () ->
  Tensor.unsafe_reinitialize ();
  let _, p, _, q, z = build () in
  let refusal =
    try
      ignore (Shape.to_dims z.Tensor.shape : int array);
      None
    with Row.Shape_error (msg, trace) -> Some (msg, trace)
  in
  Verdict.p "known gap: the inferred program is still refused"
    (Option.exists refusal ~f:(fun (msg, _) ->
         String.is_prefix msg ~prefix:"Number of axes mismatch"));
  Verdict.p "known gap: p's empty remainder conflicts with q's axis-4 remainder"
    (Option.exists refusal ~f:(fun (_, trace) ->
         is_gap_trace ~p_id:p.Tensor.shape.id ~q_id:q.Tensor.shape.id trace));
  (* A different rank mismatch must not satisfy the gap's trace oracle. *)
  Tensor.unsafe_reinitialize ();
  let leaf = TDSL.range_of_shape ~output_dims:[ 3 ] () in
  let unrelated =
    try
      let%op bad = leaf ++ "...|a,b,..r.. => ...|a" in
      ignore (Shape.to_dims bad.Tensor.shape : int array);
      None
    with Row.Shape_error (msg, trace) -> Some (msg, trace)
  in
  Verdict.p "control: an unrelated rank refusal does not match the gap"
    (Option.exists unrelated ~f:(fun (msg, trace) ->
         String.is_prefix msg ~prefix:"Number of axes mismatch"
         && not (is_gap_trace ~p_id:leaf.Tensor.shape.id ~q_id:leaf.Tensor.shape.id trace)))
