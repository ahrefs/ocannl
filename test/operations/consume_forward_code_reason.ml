(* The rejection [Tensor.consume_forward_code] raises for a non-root used to hint only at "maybe
   you're trying to forward a param?", which misled in the common test-authoring case: one tensor
   [Train.forward]ed twice (e.g. compiled under different [?lowered_transform]s, as
   [tile_mma_geometry.ml] does across four compiles) -- its forward code was consumed by the first
   call, so it is no longer a root, and nothing about it is a parameter. A tensor leaves the root
   map by three distinct routes, and the tensor now carries a consumption marker so the message
   names the route that applies. Each leg pins its sentence's discriminating words, never the old
   hint. *)

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
  ignore (Tensor.consume_forward_code y : Ir.Assignments.comp);
  let msg = rejection ~name:"twice" (fun () -> Tensor.consume_forward_code y) in
  p "second consume: names the earlier consumption" (has msg ~substring:"already consumed");
  p "second consume: does not blame a parameter" (not (has msg ~substring:"is a parameter"));
  p "second consume: names the tensor" (has msg ~substring:(Ir.Tnode.debug_name y.Tensor.value));
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
  ignore (Tensor.consume_backprop_code l : Ir.Assignments.comp);
  let msg = rejection ~name:"bprop twice" (fun () -> Tensor.consume_backprop_code l) in
  p "second backprop consume: names the earlier consumption" (has msg ~substring:"already consumed");
  p "second backprop consume: names the backprop code" (has msg ~substring:"backprop code");
  (* Leg 5: a [%cd] block that reads a tensor embeds its forward code through the same handout, so
     the rejection names the consumption rather than a consumer that does not exist. *)
  let x = leaf "cfr_x6" in
  let y = NTDSL.O.relu x in
  let acc = leaf "cfr_acc6" in
  let _embedding : Ir.Assignments.comp = [%cd acc =+ y] in
  let msg = rejection ~name:"cd" (fun () -> Tensor.consume_forward_code y) in
  p "after a %cd embedding: names the consumption" (has msg ~substring:"already consumed");
  p "after a %cd embedding: does not blame a consumer" (not (has msg ~substring:"consume that"));
  (* [with_unchanged_roots] restores the consumed marks with the roots: an [ignore]d [%cd] block's
     consumption must not later be reported as a prior consumption. *)
  let x = leaf "cfr_x5" in
  let y = NTDSL.O.relu x in
  Tensor.with_unchanged_roots ~f:(fun () ->
      ignore (Tensor.consume_forward_code y : Ir.Assignments.comp));
  p "consumption inside with_unchanged_roots is undone" (Tensor.is_fwd_root y);
  ignore (Tensor.consume_forward_code y : Ir.Assignments.comp);
  p "and the root can then be consumed once" (not (Tensor.is_fwd_root y))
