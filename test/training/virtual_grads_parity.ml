(* Regression test for the backward stale-local inlining bug (cross-framework parity finding): with
   a linear layer whose pre-activation (logits = b + w*x) stays Virtual, the executed backward pass
   of Nn_blocks.cross_entropy_loss computed wrong parameter gradients. The per-statement CSE pass
   ([Low_level.eliminate_common_subexpressions]) judged an inner-loop recomputation of a virtual
   node alpha-equivalent to the enclosing iteration's recomputation (free loop symbols were renamed
   as if bound), replacing it with a [Get_local] of the outer scope's local — so the backward's
   exp-sum used one class's logit for every class.

   Checks parameter gradients (w.grad, b.grad) against a double-precision softmax-CE oracle: dlogits
   = (softmax(logits) - targets) / batch; dw[c,d] = sum_b dlogits[b,c] * x[b,d]; db[c] = sum_b
   dlogits[b,c]. Also checks the same gradients through the standalone Nn_blocks.softmax formulation
   neg((targets *. log(softmax logits)) ++ ...), whose max-shift contribution cancels
   mathematically, so it must produce the same parameter gradients. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules

let batch = 64
let din = 2
let classes = 2

(* Fixed pseudo-random but hand-picked-scale data: moderate values so both the log-sum-exp and the
   naive log(softmax) formulations are finite and well-conditioned. *)
let x_val b d = (Float.of_int (((b * 13) + (d * 7)) % 11) /. 5.5) -. 1.0
let target_idx b = ((b * 5) + 2) % classes
let w_val c d = (Float.of_int (((c * 17) + (d * 29)) % 13) /. 6.5) -. 1.0
let b_val c = (Float.of_int (c * 7 % 5) /. 5.0) -. 0.4

(* Double-precision oracle. *)
let logit b c =
  b_val c
  +. Array.fold (Array.init din ~f:Fn.id) ~init:0. ~f:(fun acc d -> acc +. (w_val c d *. x_val b d))

let probs b =
  let row = Array.init classes ~f:(logit b) in
  let m = Array.fold row ~init:Float.neg_infinity ~f:Float.max in
  let e = Array.map row ~f:(fun v -> Float.exp (v -. m)) in
  let s = Array.fold e ~init:0. ~f:( +. ) in
  Array.map e ~f:(fun v -> v /. s)

let dlogit b c = ((probs b).(c) -. if c = target_idx b then 1. else 0.) /. Float.of_int batch

let expected_db c =
  Array.fold (Array.init batch ~f:Fn.id) ~init:0. ~f:(fun acc b -> acc +. dlogit b c)

let expected_dw c d =
  Array.fold (Array.init batch ~f:Fn.id) ~init:0. ~f:(fun acc b -> acc +. (dlogit b c *. x_val b d))

let check_grads ~label ctx w b_p =
  let open Operation.At in
  let gradients bias weight =
    Array.init
      (classes * (din + 1))
      ~f:(fun i ->
        let c = i / (din + 1) and d = i % (din + 1) in
        if d = 0 then bias c else weight c (d - 1))
  in
  let actual = gradients (fun c -> (ctx, b_p).@%{[| c |]}) (fun c d -> (ctx, w).@%{[| c; d |]}) in
  let expected = gradients expected_db expected_dw in
  Verdict.pf_close ~difference_only:true "%s gradients match softmax-CE oracle (max abs err < 1e-6)"
    label actual expected ~tol:(fun _ -> Stdlib.Float.pred 1e-6)

let run_case ~label make_loss =
  Tensor.unsafe_reinitialize ();
  let normalize_by = TDSL.O.(!.(Float.of_int batch)) in
  let x =
    NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ din ]
      ~f:(function [| b; d |] -> x_val b d | _ -> assert false)
      ()
  in
  let targets =
    NTDSL.init ~l:"targets" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ classes ]
      ~f:(function [| b; c |] -> if c = target_idx b then 1. else 0. | _ -> assert false)
      ()
  in
  let w =
    Operation.init ~l:"w" ~prec:Ir.Ops.single ~i:[ din ] ~o:[ classes ]
      ~f:(function [| c; d |] -> w_val c d | _ -> assert false)
      ~grad_spec:Tensor.Require_grad ()
  in
  let b_p =
    Operation.init ~l:"b" ~prec:Ir.Ops.single ~o:[ classes ]
      ~f:(function [| c |] -> b_val c | _ -> assert false)
      ~grad_spec:Tensor.Require_grad ()
  in
  let logits = TDSL.O.(b_p + (w * x)) in
  let loss = make_loss ~normalize_by ~logits ~targets in
  Train.set_materialized (Option.value_exn ~here:[%here] w.Tensor.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] b_p.Tensor.diff).grad;
  let update = Train.grad_update loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, routine = Context.compile ctx update IDX.empty in
  let ctx = Context.run ctx routine in
  check_grads ~label ctx w b_p

let () =
  Utils.settings.fixed_state_for_init <- Some 42;
  run_case ~label:"cross_entropy_loss (virtual logits)" (fun ~normalize_by ~logits ~targets ->
      Nn_blocks.cross_entropy_loss ~spec:"... | v" ~normalize_by () ~logits ~targets);
  run_case ~label:"neg log softmax (virtual logits)" (fun ~normalize_by ~logits ~targets ->
      let%op l =
        neg ((targets *. log (Nn_blocks.softmax ~spec:"... | v" () logits)) ++ "...|... => |->0")
      in
      TDSL.O.(l /. normalize_by))
