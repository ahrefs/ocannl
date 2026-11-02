(* gh-ocannl-1744: two routines of one lineage accumulating into one gradient with no common
   zeroing. [Tensor.op] carries a shared operand's gradient zeroing (and backprop) in its
   first-constructed consumer only, while every consumer's own gradient code accumulates into it. So
   two [Train.grad_update]s over a shared [Require_grad] input, parameter or trunk, compiled in one
   lineage, give neither objective's gradient: [Context.compile] refuses the second, naming the
   gradient and both routines. The supported spellings run here too, with exact values: one routine
   over the objectives' sum, and a common reset compiled first ([Train.zero_params_grads]) followed
   by accumulating routines that do not zero the gradient ([~accum_steps:1]).

   The style-transfer leg is the maintainer's motivating case: a content and a style objective over
   one optimized image [x = [1; 2; 3]] -- [content = x *. x] (d/dx = 2x = [2; 4; 6]) and [style = x
   *. 3] (d/dx = 3). Every value is exact by construction (small integer and dyadic sums), compared
   bitwise. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Asgns = Ir.Assignments

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let has msg substring = String.is_substring msg ~substring
let exact name got want = p_close name got want ~tol:(fun _ -> 0.)
let names_grad msg t = has msg (Ir.Tnode.label_name (grad_of t))
let one t = Set.singleton (module Tensor) t

(* The text of the [User_error] [f] raises; a failed claim when it raises none. *)
let refusal name f =
  match f () with
  | _ ->
      fail (name ^ ": was accepted");
      ""
  | exception Utils.User_error msg ->
      Stdio.eprintf "%s: %s (not part of the golden)\n%!" name msg;
      msg

let accepted name f =
  match f () with
  | x ->
      p name true;
      Some x
  | exception Utils.User_error msg ->
      p (name ^ " (refused: " ^ msg ^ ")") false;
      None

let compile ctx comp = Train.to_routine ctx IDX.empty comp

(* The image's values, uploaded: for a routine reading [x] whose initialization another consumer's
   code embeds. *)
let upload ctx x = Context.set_values ctx x.Tensor.value [| 1.; 2.; 3. |]
let run_all ctx routines = List.fold routines ~init:ctx ~f:Context.run

(* A fresh image and its two objectives, [content] constructed first: it carries [x]'s zeroing. *)
let objectives ?(data_backed = false) prefix =
  let x =
    if data_backed then
      (* An image the optimizer steps must be data-backed: [term_init]'s values are a fetch the
         forward re-establishes every run, which [Train.sgd_update] refuses (gh-ocannl-1776). *)
      Operation.init ~l:(prefix ^ "_x") ~prec:Ir.Ops.single ~o:[ 3 ]
        ~f:(function [| i |] -> Float.of_int (i + 1) | _ -> assert false)
        ~grad_spec:Require_grad ()
    else
      Tensor.term_init [| 1.; 2.; 3. |]
        ~label:[ prefix ^ "_x" ]
        ~output_dims:[ 3 ] ~grad_spec:Require_grad ()
  in
  let%op content = x *. x in
  let%op style = x *. 3. in
  (x, content, style)

let summed = [| 5.; 7.; 9. |]

let () =
  case "metadata" (fun () ->
      let x, content, style = objectives "sg_md" in
      let gx = grad_of x in
      let c_update = Train.grad_update content and s_update = Train.grad_update style in
      p "the first consumer zeroes and accumulates into the shared gradient"
        (Set.mem c_update.zeroes gx && Set.mem c_update.accumulates gx);
      p "the second only accumulates into it"
        (Set.mem s_update.accumulates gx && not (Set.mem s_update.zeroes gx));
      p "the term owns no backprop fragment, and the second loss requires nothing of it"
        (Set.is_empty (Option.value_exn ~here:[%here] x.diff).backprop.owned
        && Set.is_empty s_update.requires);
      p "the first consumer zeroes exactly its own and the image's gradient"
        (Set.equal c_update.zeroes (Set.of_list (module Ir.Tnode) [ gx; grad_of content ]));
      let x, content, style = objectives "sg_md2" in
      let gx = grad_of x in
      let zero = Train.zero_params_grads ~params:(one x) content in
      let a1 = Train.grad_update ~accum_steps:1 content in
      let a2 = Train.grad_update ~accum_steps:1 style in
      p "zero_params_grads zeroes the leaf and accumulates nothing"
        (Set.mem zero.zeroes gx && Set.is_empty zero.accumulates);
      p "grad_update ~accum_steps strips the leaf's zeroing from the first consumer"
        (Set.mem a1.accumulates gx && not (Set.mem a1.zeroes gx));
      p "grad_update ~accum_steps over the second consumer has no zeroing of it to strip"
        (Set.mem a2.accumulates gx && not (Set.mem a2.zeroes gx));
      (* That the stated zeroings are the emitted ones is [Train.filter_out_grad_zeroing]'s
         two-sided guard, which both [~accum_steps] updates above passed: it raises on a stated
         zeroing it finds no [Fetch] for, and on a [Fetch] of a leaf the metadata does not state. *)
      p "with accum_steps, the first consumer zeroes only its own gradient"
        (Set.equal a1.zeroes (Set.singleton (module Ir.Tnode) (grad_of content))));
  case "shared input, two plain updates" (fun () ->
      let x, content, style = objectives "sg_in" in
      let ctx = Train.update_once (Context.auto ()) content in
      exact "content alone: x.grad = 2x" (Context.get_values ctx (grad_of x)) [| 2.; 4.; 6. |];
      let msg = refusal "style after content" (fun () -> Train.update_once ctx style) in
      p "style compiled after content is refused, naming x's gradient and both routines"
        (names_grad msg x && has msg "style" && has msg "content" && has msg "zeroes it itself");
      exact "the refusal left content's gradient"
        (Context.get_values ctx (grad_of x))
        [| 2.; 4.; 6. |];
      ignore
        (accepted "the same context still compiles a common reset" (fun () ->
             compile ctx (Train.zero_params_grads ~params:(one x) style)));
      (* The reverse compile order. [content]'s code is handed out first (it embeds [x]'s forward,
         which [style] then reads from the context), and [x.grad] -- which [style]'s backprop
         accumulates into without embedding it -- is allocated by a reset compiled first: a reset
         before both does not excuse a contributor that zeroes the gradient itself. *)
      let x, content, style = objectives "sg_in2" in
      let c_update = Train.grad_update content in
      let ctx, zero =
        compile (upload (Context.auto ()) x) (Train.zero_params_grads ~params:(one x) style)
      in
      let ctx, s_update = compile ctx (Train.grad_update style) in
      let ctx = run_all ctx [ zero; s_update ] in
      exact "style alone: x.grad = 3" (Context.get_values ctx (grad_of x)) [| 3.; 3.; 3. |];
      let msg = refusal "content after style" (fun () -> compile ctx c_update) in
      p "content compiled after style is refused: it zeroes the gradient itself"
        (names_grad msg x && has msg "style" && has msg "content" && has msg "zeroes it itself"));
  case "shared input, one routine over the sum" (fun () ->
      let x, content, style = objectives ~data_backed:true "sg_sum" in
      let%op total = content + style in
      let ctx, update = compile (Context.auto ()) (Train.grad_update total) in
      let ctx = run_all ctx [ update; update ] in
      exact "two runs: x.grad = 2x + 3, no carry-over" (Context.get_values ctx (grad_of x)) summed;
      let ctx, step =
        compile ctx (Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.5) ~params:(one x) total)
      in
      let ctx = Context.run ctx step in
      exact "one input-optimization step: x - 0.5 (2x + 3)" (Context.get_values ctx x.value)
        [| -1.5; -1.5; -1.5 |]);
  case "shared input, common reset" (fun () ->
      let x, content, style = objectives "sg_cr" in
      let ctx, zero = compile (Context.auto ()) (Train.zero_params_grads ~params:(one x) content) in
      let ctx, a1 = compile ctx (Train.grad_update ~accum_steps:1 content) in
      match
        accepted "a reset compiled before both, carried by neither" (fun () ->
            compile ctx (Train.grad_update ~accum_steps:1 style))
      with
      | None -> ()
      | Some (ctx, a2) ->
          let ctx = run_all ctx [ zero; a1; a2 ] in
          exact "reset, content, style: x.grad = 2x + 3" (Context.get_values ctx (grad_of x)) summed;
          let ctx = run_all ctx [ zero; a1; a2 ] in
          exact "a second cycle: the same" (Context.get_values ctx (grad_of x)) summed;
          let ctx = run_all ctx [ a1; a2 ] in
          exact "a cycle without the reset carries over (the control)"
            (Context.get_values ctx (grad_of x))
            [| 10.; 14.; 18. |]);
  case "shared input, reset order" (fun () ->
      let x, content, style = objectives "sg_ro" in
      let reset () = Train.zero_params_grads ~params:(one x) content in
      let ctx, _ = compile (Context.auto ()) (reset ()) in
      let ctx, _ = compile ctx (Train.grad_update ~accum_steps:1 content) in
      let ctx, _ = compile ctx (reset ()) in
      let msg =
        refusal "a reset between the two" (fun () ->
            compile ctx (Train.grad_update ~accum_steps:1 style))
      in
      p "a zeroing compiled after the first accumulation is no common reset"
        (names_grad msg x && has msg "zeroes it after the first accumulation");
      let x, content, style = objectives "sg_ro2" in
      let ctx, _ = compile (Context.auto ()) (Train.grad_update ~accum_steps:1 content) in
      let msg =
        refusal "no reset" (fun () -> compile ctx (Train.grad_update ~accum_steps:1 style))
      in
      p "two accumulations with no reset at all are refused"
        (names_grad msg x && has msg "no zeroing of it is compiled in this lineage before both");
      let x, content, style = objectives "sg_ro3" in
      let ctx0 = Context.auto () in
      let _sibling = compile ctx0 (Train.zero_params_grads ~params:(one x) content) in
      let ctx, _ = compile ctx0 (Train.grad_update ~accum_steps:1 content) in
      let msg =
        refusal "a sibling's reset" (fun () -> compile ctx (Train.grad_update ~accum_steps:1 style))
      in
      p "a reset compiled in a sibling does not count"
        (names_grad msg x && has msg "no zeroing of it is compiled in this lineage before both"));
  case "shared input, siblings" (fun () ->
      (* Siblings never collide -- but that is compilation independence only: [style] still lacks
         the zeroing [content] carries, and both siblings bind the [x.grad] the common ancestor's
         reset allocated, so each value claim runs that reset first. *)
      let x, content, style = objectives "sg_sib" in
      let ctx0, zero =
        compile (upload (Context.auto ()) x) (Train.zero_params_grads ~params:(one x) content)
      in
      let c_update = Train.grad_update content in
      ignore
        (accepted "style's plain update, as a sibling of content's" (fun () ->
             compile ctx0 (Train.grad_update style)));
      match
        ( accepted "content's update" (fun () -> compile ctx0 c_update),
          accepted "style's accumulating update, as a sibling of content's" (fun () ->
              compile ctx0 (Train.grad_update ~accum_steps:1 style)) )
      with
      | Some (ctx_c, uc), Some (ctx_s, us) ->
          let ctx_c = run_all ctx_c [ zero; uc ] in
          exact "content's sibling: x.grad = 2x"
            (Context.get_values ctx_c (grad_of x))
            [| 2.; 4.; 6. |];
          let ctx_s = run_all ctx_s [ zero; us ] in
          exact "style's sibling, over the reset: x.grad = 3"
            (Context.get_values ctx_s (grad_of x))
            [| 3.; 3.; 3. |]
      | _ -> ());
  case "shared parameter" (fun () ->
      let prefix = "sg_par" in
      let build () =
        let w = TDSL.param ~values:[| 1.; 2. |] prefix ~output_dims:[ 2 ] () in
        let%op a = w *. 2. in
        let%op b = w *. w in
        (w, a, b)
      in
      let w, a, b = build () in
      p "a parameter owns no backprop fragment"
        (Set.is_empty (Option.value_exn ~here:[%here] w.diff).backprop.owned);
      let ctx = Train.update_once (Context.auto ()) a in
      let msg =
        refusal "two plain updates over one parameter" (fun () -> Train.update_once ctx b)
      in
      p "the second update is refused, naming the parameter's gradient"
        (names_grad msg w && has msg "zeroes it itself");
      exact "the refusal left a's gradient" (Context.get_values ctx (grad_of w)) [| 2.; 2. |];
      let w, a, b = build () in
      let ctx = Train.init_params (Context.auto ()) IDX.empty a in
      let ctx, zero = compile ctx (Train.zero_params_grads a) in
      let ctx, ua = compile ctx (Train.grad_update ~accum_steps:1 a) in
      match
        accepted "a common reset, then both" (fun () ->
            compile ctx (Train.grad_update ~accum_steps:1 b))
      with
      | None -> ()
      | Some (ctx, ub) ->
          let ctx = run_all ctx [ zero; ua; ub; zero; ua; ub ] in
          exact "two cycles: w.grad = 2 + 2w" (Context.get_values ctx (grad_of w)) [| 4.; 6. |]);
  (* A shared trunk [h = w *. w] under [l1 = h *. 2] (constructed first: it carries [h]'s zeroing
     and backprop) and [l2 = h *. 3]. [h]'s forward and a zeroing of [h.grad] and [w.grad] are
     routines of their own, so both losses read a materialized [h] and [h.grad] is allocated. *)
  let trunk prefix =
    let w = TDSL.param ~values:[| 1.; 2.; 3. |] (prefix ^ "_w") ~output_dims:[ 3 ] () in
    let%op h = w *. w in
    let h_fwd = Train.forward h in
    let%op l1 = h *. 2. in
    let%op l2 = h *. 3. in
    let h_zero =
      let z = (Option.value_exn ~here:[%here] h.diff).zero_grads in
      { z with asgns = Asgns.Block_comment (prefix ^ "_h_zero", z.asgns) }
    in
    let base () =
      let ctx = Train.init_params (Context.auto ()) IDX.empty l1 in
      let ctx, r = compile ctx h_fwd in
      let ctx = Context.run ctx r in
      let ctx, zero = compile ctx h_zero in
      (Context.run ctx zero, zero)
    in
    (* [l1]'s propagation without [grad_update]'s zeroing of [h.grad]. *)
    let l1_propagation () =
      [%cd
        ~~(l1 "propagation only";
           l1.forward;
           l1.grad =: 1;
           l1.backprop)]
    in
    (w, h, l1, l2, base, l1_propagation)
  in
  case "shared trunk" (fun () ->
      let _w, h, l1, l2, base, _ = trunk "sg_tr" in
      let ctx, _ = base () in
      let ctx = Train.update_once ctx l1 in
      let msg = refusal "trunk, owner first" (fun () -> Train.update_once ctx l2) in
      p "l2 after l1's update is refused: l1 already propagates from h's gradient"
        (names_grad msg h && has msg "already propagates from it");
      let _w, h, l1, l2, base, _ = trunk "sg_tr2" in
      let ctx, _ = base () in
      let ctx, _ = compile ctx (Train.grad_update l2) in
      let msg = refusal "trunk, contributor first" (fun () -> compile ctx (Train.grad_update l1)) in
      p "l1's update after l2's is refused: it zeroes h's gradient itself"
        (names_grad msg h && has msg "zeroes it itself");
      let _w, h, _l1, l2, base, l1_propagation = trunk "sg_tr3" in
      let ctx, _ = base () in
      let ctx, _ = compile ctx (l1_propagation ()) in
      let msg =
        refusal "trunk, zero-free owner first" (fun () -> compile ctx (Train.grad_update l2))
      in
      p "a contribution compiled after the zero-free propagation is refused"
        (names_grad msg h && has msg "already propagates from it");
      let w, _h, _l1, l2, base, l1_propagation = trunk "sg_tr4" in
      let ctx, zero = base () in
      let ctx, r2 = compile ctx (Train.grad_update l2) in
      match
        accepted "contributor first, zero-free owner last, over a common reset" (fun () ->
            compile ctx (l1_propagation ()))
      with
      | None -> ()
      | Some (ctx, r1) ->
          let ctx = run_all ctx [ zero; r2; r1; zero; r2; r1 ] in
          (* d (2h + 3h) / d w = 5 * 2w. *)
          exact "two cycles: w.grad = 10 w" (Context.get_values ctx (grad_of w)) [| 10.; 20.; 30. |])
