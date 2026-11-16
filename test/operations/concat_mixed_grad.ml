(* Mixed-gradient concatenation: each upstream weight is distinct, so dropping a constant's operand
   position changes the gradient oracle. All values are exact integers. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims

let tensor ~grad values =
  Tensor.ndarray
    ~grad_spec:(if grad then Tensor.Require_grad else Tensor.Prohibit_grad)
    values ~batch_dims:[] ~input_dims:[]
    ~output_dims:[ Array.length values ]
    ()

let grad_of t = (Option.value_exn t.Tensor.diff).Tensor.grad

let execute ~constant_grad ~left_values ~make ~width =
  Tensor.unsafe_reinitialize ();
  let constant = tensor ~grad:constant_grad [| 10. |] in
  let left = tensor ~grad:true left_values in
  let right = tensor ~grad:true [| 3.; 4.; 5. |] in
  let cat, parameters = make constant left right in
  let weights = tensor ~grad:false (Array.sub [| 2.; 3.; 5.; 7.; 11.; 13. |] ~pos:0 ~len:width) in
  let%op loss = (cat *. weights) ++ "...|... => |->0" in
  (* The reference materializes every operand, result and gradient, and gives the fixed operand a
     gradient too: it uses the existing scatter with no skipped target positions. *)
  List.iter [ constant; left; right; cat ] ~f:(fun t ->
      Train.set_materialized t.Tensor.value;
      Option.iter t.diff ~f:(fun diff -> Train.set_materialized diff.Tensor.grad));
  let ctx = Train.update_once ~output_cd_file:false (Context.auto ()) loss in
  ( Context.get_values ctx cat.value,
    List.map parameters ~f:(fun t -> Context.get_values ctx (grad_of t)),
    Option.is_none constant.diff )

let run ?(left_values = [| 1.; 2. |]) ~label ~make ~expected_forward ~expected_grads () =
  Verdict.case label (fun () ->
      let forward, grads, no_grad =
        execute ~constant_grad:false ~left_values ~make ~width:(Array.length expected_forward)
      in
      pf_all2 "%s: forward values" label forward expected_forward ~f:Float.equal;
      List.iter2_exn grads expected_grads ~f:(fun actual expected ->
          pf_all2 "%s: parameter gradient" label actual expected ~f:Float.equal);
      pf "%s: constant has no gradient" label no_grad;
      let reference_forward, reference_grads, _ =
        execute ~constant_grad:true ~left_values ~make ~width:(Array.length expected_forward)
      in
      pf_all2 "%s: forward matches materialized reference" label forward reference_forward
        ~f:Float.equal;
      List.iter2_exn grads reference_grads ~f:(fun actual expected ->
          pf_all2 "%s: gradient matches materialized reference" label actual expected ~f:Float.equal))

let () =
  run ~label:"constant first"
    ~make:(fun c p _ ->
      let%op cat = (c, p) ++^ "a; b => a^b" in
      (cat, [ p ]))
    ~expected_forward:[| 10.; 1.; 2. |]
    ~expected_grads:[ [| 3.; 5. |] ]
    ();
  run ~label:"constant last"
    ~make:(fun c p _ ->
      let%op cat = (p, c) ++^ "a; b => a^b" in
      (cat, [ p ]))
    ~expected_forward:[| 1.; 2.; 10. |]
    ~expected_grads:[ [| 2.; 3. |] ]
    ();
  run ~label:"constant middle"
    ~make:(fun c p q ->
      let%op cat = (p, c, q) ++^ "a; b; c => a^b^c" in
      (cat, [ p; q ]))
    ~expected_forward:[| 1.; 2.; 10.; 3.; 4.; 5. |]
    ~expected_grads:[ [| 2.; 3. |]; [| 7.; 11.; 13. |] ]
    ();
  run ~label:"constant then width-1 parameter" ~left_values:[| 1. |]
    ~make:(fun c p _ ->
      let%op cat = (c, p) ++^ "a; b => a^b" in
      (cat, [ p ]))
    ~expected_forward:[| 10.; 1. |] ~expected_grads:[ [| 3. |] ] ();
  Tensor.unsafe_reinitialize ();
  let a = tensor ~grad:false [| 1.; 2. |] in
  let b = tensor ~grad:false [| 10. |] in
  let%op cat = (a, b) ++^ "a; b => a^b" in
  p "all-constant concat has no gradient" (Option.is_none cat.Tensor.diff);
  Train.set_materialized cat.value;
  let ctx = Train.forward_once ~output_cd_file:false (Context.auto ()) cat in
  p_all2 "all-constant concat forward values"
    (Context.get_values ctx cat.value)
    [| 1.; 2.; 10. |] ~f:Float.equal
