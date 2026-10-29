(* A homogeneous operator's operand annotation that it would ignore is refused (gh-ocannl-1709).

   Every binop but [Uint4x32_to_prec_uniform_lane] evaluates its operands at its result precision,
   so the precision an operand carries there is never semantics. [bf16_subnormal_storage]'s leg "f32
   addition of bfloat loads, narrowed" stored [Add ((bf16 load, single), (bf16 load, single))] to a
   bfloat node and claimed an f32 addition; every backend computed a bfloat one.
   [Low_level.optimize] now refuses that form, naming the destination, both precisions and the
   remedy. An annotation may still describe: the evaluation precision itself, or the operand's own
   precision -- what [Low_level.mk_binop] and the assignments lowering write, so a mixed-precision
   store built that way is accepted. Heterogeneous slots (a [Where] condition) keep their
   annotation, unchecked. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let bf = Ops.bfloat16
let f32 = Ops.single
let half = Ops.half
let n = 4
let bnode = Ll_test.node_factory ~prec:bf ~first_id:23400 ~dims:[| n |] ()
let fnode = Ll_test.node_factory ~prec:f32 ~first_id:23500 ~dims:[| n |] ()

(* [Ok ()] when [optimize] accepts the routine, [Error msg] when it refuses it; with the
   destination's name. *)
let optimize name body =
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let in_bf = bnode (name ^ "_in_bf") and in_f = fnode (name ^ "_in_f") in
  let out_bf = bnode (name ^ "_out_bf") and out_f = fnode (name ^ "_out_f") in
  List.iter [ in_bf; in_f; out_bf; out_f ] ~f:Ll_test.materialize;
  let get t = Ll_test.get t i in
  let store, out = body ~get ~in_bf ~in_f ~out_bf ~out_f in
  let llc = Ll_test.loop_n s n (Ll_test.set store i out) in
  match Ll_test.optimize ~name llc with
  | (_ : LL.optimized) -> (Ok (), Ir.Tnode.label_name store)
  | exception Invalid_argument msg -> (Error msg, Ir.Tnode.label_name store)

let double t p ~get = LL.Binop (Ops.Add, (get t, p), (get t, p))

let refused label ~needles result =
  let outcome, dest = result in
  match outcome with
  | Ok () -> p label false
  | Error msg ->
      Stdio.eprintf "(not part of the golden) %s: %s\n" label msg;
      p_all label (dest :: needles) ~f:(fun needle -> String.is_substring msg ~substring:needle)

let accepted label (outcome, _) =
  (match outcome with
  | Error msg -> Stdio.eprintf "(not part of the golden) %s refused: %s\n" label msg
  | Ok () -> ());
  p label (Result.is_ok outcome)

let () =
  Verdict.case "operand precision annotations" (fun () ->
      refused
        "the old leg's form -- a single-annotated addition of bfloat loads stored to a bfloat node \
         -- is refused, naming the destination, both precisions and the remedy"
        ~needles:[ "Operand 1 of Add"; "annotated single"; "here bfloat16"; "make the destination" ]
        (optimize "opa_old" (fun ~get ~in_bf ~in_f:_ ~out_bf ~out_f:_ ->
             (out_bf, double in_bf f32 ~get)));
      refused "a unary operand annotated half under a bfloat store is refused"
        ~needles:[ "The operand of Neg"; "annotated half"; "here bfloat16" ]
        (optimize "opa_neg" (fun ~get ~in_bf ~in_f:_ ~out_bf ~out_f:_ ->
             (out_bf, LL.Unop (Ops.Neg, (get in_bf, half)))));
      refused "a Where arm annotated half under a single store is refused"
        ~needles:[ "The else-arm of Where"; "annotated half"; "here single" ]
        (optimize "opa_where" (fun ~get ~in_bf:_ ~in_f ~out_bf:_ ~out_f ->
             (out_f, LL.Ternop (Ops.Where, (get in_f, f32), (get in_f, f32), (LL.Constant 0., half)))));
      accepted "the fixed leg -- the same addition stored to a single node -- is accepted"
        (optimize "opa_fixed" (fun ~get ~in_bf ~in_f:_ ~out_bf:_ ~out_f ->
             (out_f, double in_bf f32 ~get)));
      accepted
        "operands annotated with their own precision (apply_op's form) under a wider store are \
         accepted"
        (optimize "opa_natural" (fun ~get ~in_bf ~in_f:_ ~out_bf:_ ~out_f ->
             (out_f, LL.apply_op (Ops.Binop Ops.Add) [| get in_bf; get in_bf |])));
      accepted
        "single loads annotated single under a bfloat store (own precision, narrowed) are accepted"
        (optimize "opa_narrow" (fun ~get ~in_bf:_ ~in_f ~out_bf ~out_f:_ ->
             (out_bf, double in_f f32 ~get)));
      accepted "a Where condition keeps any annotation (a heterogeneous slot)"
        (optimize "opa_cond" (fun ~get ~in_bf ~in_f:_ ~out_bf ~out_f:_ ->
             ( out_bf,
               LL.Ternop (Ops.Where, (get in_bf, half), (get in_bf, bf), (LL.Constant 0., bf)) ))))
