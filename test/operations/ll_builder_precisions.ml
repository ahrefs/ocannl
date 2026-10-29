(* The hand-built-IR builders of [Ll_builders] build what the operand-precision check accepts
   (gh-ocannl-1709's post-wave task, lukstafi/ocannl-staging#1514).

   [Low_level.optimize] refuses an operand annotation a homogeneous operator would ignore: one that
   is neither the evaluation precision nor the operand's own. Most of the hand-built tests that
   tripped it did so through two builder shapes: an [if_] that annotated its condition [single],
   handed an index comparison whose [Constant] operands the [cmp] family annotates at index
   precision, and the [single]-annotated [binop]/[where_] arms under a [half] or [bfloat16] store.
   [if_] now reads its condition at its own precision, and the own-precision builders ([binop_own],
   [add_own], [mul_own], [where_own]) annotate as the assignments lowering does. Each is checked
   here under [single], [half] and [bfloat16] destinations, beside the shapes the check still
   refuses: the old [if_]'s form ([if_at single]), and [binop]/[where_] arms under a narrow store. A
   point-in-time check of the builders this file names, through the check itself, not through
   [optimize]. *)
open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module B = Ll_builders
open Verdict.Claims

let n = 4
let precs = [ Ops.single; Ops.half; Ops.bfloat16 ]
let narrow = [ Ops.half; Ops.bfloat16 ]

let check label llc =
  match LL.validate_operand_precisions ~site:"ll_builder_precisions" llc with
  | () -> Ok ()
  | exception Invalid_argument msg ->
      Stdio.eprintf "(not part of the golden) %s: %s\n" label msg;
      Error msg

(* One fresh node set per destination precision: [d] the destination, [x] a source and [flag] a
   one-cell guard value, all at [prec]. *)
let next_id = ref 23700

let nodes prec =
  let first_id = !next_id in
  next_id := !next_id + 10;
  let node = B.node_factory ~prec ~first_id ~dims:[| n |] () in
  (node "lbp_d", node "lbp_x", node ~dims:[| 1 |] "lbp_flag")

(* [build ~d ~x ~flag s] is a statement over the loop symbol [s]; the case wraps it in the loop. *)
let routine prec build =
  let d, x, flag = nodes prec in
  let s = B.sym () in
  B.loop_n s n (build ~d ~x ~flag s)

let accepted_under label ps build =
  p_all label ps ~f:(fun prec ->
      Result.is_ok (check (label ^ " @" ^ Ops.prec_string prec) (routine prec build)))

let refused_under label ps build =
  p_all label ps ~f:(fun prec ->
      Result.is_error (check (label ^ " @" ^ Ops.prec_string prec) (routine prec build)))

let at s = [| B.iter s |]
let copy ~d ~x s = B.set_at d (B.iter s) (B.get x (at s))
let flag_value flag = B.get flag [| B.fixed 0 |]

let () =
  Verdict.case "guard builders" (fun () ->
      accepted_under "if_ over a cmp comparison with a constant operand is accepted" precs
        (fun ~d ~x ~flag:_ s -> B.if_ (B.lt (B.embed s) (B.ic 2)) (copy ~d ~x s));
      accepted_under "if_ over a conjunction of comparisons is accepted" precs
        (fun ~d ~x ~flag:_ s ->
          B.if_ (B.conj (B.le (B.ic 1) (B.embed s)) (B.lt (B.embed s) (B.ic 3))) (copy ~d ~x s));
      accepted_under "if_ over a value-read flag is accepted" precs (fun ~d ~x ~flag s ->
          B.if_ (flag_value flag) (copy ~d ~x s));
      accepted_under "if_idx over a cmp comparison is accepted" precs (fun ~d ~x ~flag:_ s ->
          B.if_idx (B.eq (B.embed s) (B.ic 0)) (copy ~d ~x s));
      (let s = B.sym () and d, x, _ = nodes Ops.single in
       let cond = B.lt (B.embed s) (B.ic 2) in
       p "if_ and if_idx build the same guard over a cmp comparison"
         (LL.equal (B.if_ cond (copy ~d ~x s)) (B.if_idx cond (copy ~d ~x s))));
      refused_under
        "the old if_'s form -- a cmp comparison annotated single -- is refused under every store"
        precs (fun ~d ~x ~flag:_ s -> B.if_at Ops.single (B.lt (B.embed s) (B.ic 2)) (copy ~d ~x s)))

let () =
  Verdict.case "value builders" (fun () ->
      accepted_under "add_own accumulating a load is accepted" precs (fun ~d ~x ~flag:_ s ->
          B.set_at d (B.iter s) (B.add_own (B.get d (at s)) (B.get x (at s))));
      accepted_under "mul_own of a load and a constant is accepted" precs (fun ~d ~x ~flag:_ s ->
          B.set_at d (B.iter s) (B.mul_own (B.get x (at s)) (B.c 0.5)));
      accepted_under "where_own over a cmp comparison with a constant arm is accepted" precs
        (fun ~d ~x ~flag:_ s ->
          B.set_at d (B.iter s) (B.where_own (B.lt (B.embed s) (B.ic 2)) (B.get x (at s)) (B.c 0.)));
      accepted_under "where_own over a value-read flag with nested own-precision arms is accepted"
        precs (fun ~d ~x ~flag s ->
          B.set_at d (B.iter s)
            (B.where_own (flag_value flag)
               (B.add_own (B.mul_own (B.get x (at s)) (B.get x (at s))) (B.c 1.))
               (B.get d (at s))));
      accepted_under "where_own's condition guards an if_ the same way" precs
        (fun ~d ~x ~flag:_ s ->
          let cond = B.lt (B.embed s) (B.ic 3) in
          B.if_ cond (B.set_at d (B.iter s) (B.where_own cond (B.get x (at s)) (B.get d (at s)))));
      accepted_under "binop, add, mul and where_ (single arms) are accepted under a single store"
        [ Ops.single ] (fun ~d ~x ~flag:_ s ->
          B.set_at d (B.iter s)
            (B.where_
               (B.lt (B.embed s) (B.ic 2))
               (B.add
                  (B.mul (B.get x (at s)) (B.get x (at s)))
                  (B.binop Ops.Sub (B.get d (at s)) (B.c 1.)))
               (B.get x (at s))));
      refused_under "binop over narrow loads (single annotations) is refused under a narrow store"
        narrow (fun ~d ~x ~flag:_ s ->
          B.set_at d (B.iter s) (B.binop Ops.Add (B.get d (at s)) (B.get x (at s))));
      refused_under "where_'s single arms over narrow loads are refused under a narrow store" narrow
        (fun ~d ~x ~flag:_ s ->
          B.set_at d (B.iter s) (B.where_ (B.lt (B.embed s) (B.ic 2)) (B.get x (at s)) (B.c 0.))))
