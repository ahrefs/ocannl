(* Repeated handout preserves comp identity; other non-root routes retain their diagnostics. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let leaf name =
  NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ 2 ]
    ~f:(function [| i |] -> Float.of_int (i + 1) | _ -> assert false)
    ()

let rejection ~name f =
  match f () with
  | exception Tensor.Session_error (msg, _) -> msg
  | _ ->
      fail (name ^ ": consume did not raise");
      ""

let has msg ~substring = String.is_substring msg ~substring

let () =
  (* Leg 1: the forward code was consumed already. *)
  let x = leaf "cfr_x1" in
  let y = NTDSL.O.relu x in
  let first = Tensor.consume_forward_code y in
  p "second consume: returns the same comp" (phys_equal first (Tensor.consume_forward_code y));
  (* Leg 2: a parameter never owns forward code. *)
  let w = TDSL.param ~value:0.5 "cfr_w" () in
  let msg = rejection ~name:"param" (fun () -> Tensor.consume_forward_code w) in
  p "parameter: says it is a parameter" (has msg ~substring:"is a parameter");
  p "parameter: does not claim a prior consumption" (not (has msg ~substring:"already consumed"));
  (* Leg 3: a subterm whose forward code a consumer embedded. *)
  let x = leaf "cfr_x3" in
  let _y = NTDSL.O.relu x in
  let msg = rejection ~name:"subterm" (fun () -> Tensor.consume_forward_code x) in
  p "subterm: says the code is embedded in a consumer" (has msg ~substring:"embedded in a tensor");
  p "subterm: does not claim a prior consumption" (not (has msg ~substring:"already consumed"));
  (* Leg 4: the backprop side records consumption the same way. *)
  let v = Tensor.term_init [| 1.; 2. |] ~label:[ "cfr_v4" ] ~grad_spec:Require_grad () in
  let l = TDSL.O.relu v in
  let first = Tensor.consume_backprop_code l in
  p "second backprop consume: returns the same comp"
    (phys_equal first (Tensor.consume_backprop_code l));
  let msg = rejection ~name:"backprop subterm" (fun () -> Tensor.consume_backprop_code v) in
  p "backprop subterm: says the code is embedded in a consumer"
    (has msg ~substring:"embedded in a tensor");
  (* A [%cd] embedding uses the same handout marker. *)
  let x = leaf "cfr_x6" in
  let y = NTDSL.O.relu x in
  let acc = leaf "cfr_acc6" in
  let _embedding : Ir.Assignments.comp = [%cd acc =+ y] in
  p "after a %cd embedding: returns the tensor's comp"
    (phys_equal y.Tensor.forward (Tensor.consume_forward_code y));
  (* [with_unchanged_roots] restores the consumed marks with the roots: an [ignore]d [%cd] block's
     consumption must not later be reported as a prior consumption. *)
  let x = leaf "cfr_x5" in
  let y = NTDSL.O.relu x in
  Tensor.with_unchanged_roots ~f:(fun () ->
      ignore (Tensor.consume_forward_code y : Ir.Assignments.comp));
  p "consumption inside with_unchanged_roots is undone" (Tensor.is_fwd_root y);
  ignore (Tensor.consume_forward_code y : Ir.Assignments.comp);
  p "and the root can then be consumed once" (not (Tensor.is_fwd_root y))

let () =
  let x = leaf "conflict_leaf" in
  let shared = NTDSL.O.relu x in
  let owner = NTDSL.O.relu shared in
  let sibling = NTDSL.O.relu shared in
  let msg =
    rejection ~name:"first handout conflict" (fun () -> Tensor.consume_forward_code sibling)
  in
  p "first handout still rejects a conflicting root" (has msg ~substring:"conflicting roots");
  p "failed first handout leaves the root intact" (Tensor.is_fwd_root sibling);
  ignore (Tensor.consume_forward_code owner : Ir.Assignments.comp);
  ignore (Tensor.consume_forward_code sibling : Ir.Assignments.comp);
  let msg = rejection ~name:"non-differentiable" (fun () -> Tensor.consume_backprop_code sibling) in
  p "non-differentiable backprop remains rejected" (has msg ~substring:"not differentiable")

let () =
  let v = Tensor.term_init [| -1.; 3. |] ~grad_spec:Require_grad () in
  let init_tensor = TDSL.O.relu v in
  let forward = Tensor.consume_forward_code init_tensor in
  let backprop = Tensor.consume_backprop_code init_tensor in
  let reuse ?label:_ ?top_down_prec:_ ?batch_dims:_ ?batch_axes:_ ?input_dims:_ ?output_dims:_
      ?input_axes:_ ?output_axes:_ ?deduced:_ () =
    init_tensor
  in
  let parameter = Tensor.param ~t:reuse "taken_initializer_param" () in
  let msg =
    rejection ~name:"taken initializer parameter" (fun () -> Tensor.consume_forward_code parameter)
  in
  p "parameter from a taken initializer still rejects forward" (has msg ~substring:"is a parameter");
  p "parameterization preserves the initializer forward handout"
    (phys_equal forward (Tensor.consume_forward_code init_tensor));
  p "parameterization preserves the initializer backprop handout"
    (phys_equal backprop (Tensor.consume_backprop_code init_tensor));
  p "new parameter owns its fresh backprop root" (Tensor.is_bprop_root parameter);
  let parameter_backprop = Tensor.consume_backprop_code parameter in
  p "first parameter backprop handout removes its root" (not (Tensor.is_bprop_root parameter));
  p "parameter backprop can then replay its fresh comp"
    (phys_equal parameter_backprop (Tensor.consume_backprop_code parameter));
  let parameter = Tensor.param ~t:reuse "embedded_initializer_param" () in
  let _consumer = TDSL.O.relu parameter in
  let msg =
    rejection ~name:"embedded parameter backprop" (fun () -> Tensor.consume_backprop_code parameter)
  in
  p "parameter backprop embedded after parameterization remains rejected"
    (has msg ~substring:"embedded in a tensor")
