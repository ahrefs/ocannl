(* A homogeneous operator's operand annotation that it would ignore is refused (gh-ocannl-1709).

   Every binop but [Uint4x32_to_prec_uniform_lane] evaluates its operands at its result precision,
   so the precision an operand carries there is never semantics. [bf16_subnormal_storage]'s leg "f32
   addition of bfloat loads, narrowed" stored [Add ((bf16 load, single), (bf16 load, single))] to a
   bfloat node and claimed an f32 addition; every backend computed a bfloat one.
   [Low_level.optimize] now refuses that form, naming the destination, both precisions and the
   remedy. An annotation may still describe: the evaluation precision itself, or the operand's own
   precision -- what [Low_level.mk_binop] and the assignments lowering write, so a mixed-precision
   store built that way is accepted. Heterogeneous slots (a [Where] condition) keep their
   annotation, unchecked.

   gh-ocannl-1739: [C_syntax.compile_proc] runs the same check on the way out. A [?prelowered]
   routine skips [optimize], so the exit gate is the only one it meets. The optimizer's own output
   passes it because the optimizer re-describes the annotations its rewrites leave stale (a
   collapsed scope of another precision, an elided arm) -- its input passed the entry gate, so none
   of them is a builder's request; [Low_level.describe_operand_precisions] does it. [Schedule.apply]
   applies it to a schedule's result when the program the schedule started from passes the check, so
   a [?prelowered] request still reaches the gate through a schedule.

   gh-ocannl-1788: that includes a request on a load. [simplify] used to re-annotate every load at
   its storage precision, which erased such a request before the gate could judge it (and, in a
   heterogeneous slot, changed the precision the renderer evaluates the load at); it now keeps a
   leaf's annotation. It also collapses a scope only where that is exact, so the collapse cases
   below collapse a widening scope over a load. *)
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

(* [Ok ()] when code generation compiles the routine as written, [Error msg] when it refuses it. *)
let link_outcome name o =
  match Ll_test.link ~name o with
  | (_ : Context.t * Context.routine) -> (Ok (), "")
  | exception Invalid_argument msg -> (Error msg, "")

(* A bfloat store of a bfloat sum whose operands are annotated [annotate], reaching code generation
   through the [?prelowered] seam: the optimizer sees only the descriptive twin. *)
let prelowered ?(scheduled = false) ?(value = fun ~x p -> LL.Binop (Ops.Add, (x, p), (x, p))) name
    ~annotate =
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let in_bf = bnode (name ^ "_in_bf") and out_bf = bnode (name ^ "_out_bf") in
  List.iter [ in_bf; out_bf ] ~f:Ll_test.materialize;
  let llc p = Ll_test.loop_n s n (Ll_test.set out_bf i (value ~x:(Ll_test.get in_bf i) p)) in
  let o = Ll_test.optimize_scoped ~name ~raw:(llc bf) (llc annotate) in
  let o =
    if scheduled then Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty = LL.Serial } ] o else o
  in
  (fst (link_outcome name o), Ir.Tnode.label_name out_bf)

(* A virtual single node copying a bfloat input, read in a bfloat sum, annotated with its own
   precision there (what the lowering writes). Inlining makes the read a single scope over a bfloat
   load, which simplification collapses into the sum -- exactly, since bfloat16 widens to single
   without loss (gh-ocannl-1788): the operand is now a bfloat load still annotated single, which
   describes neither the slot nor the load. *)
let optimized_collapse name =
  let s1 = Ll_test.sym () and s2 = Ll_test.sym () in
  let i1 = [| Ll_test.iter s1 |] and i2 = [| Ll_test.iter s2 |] in
  let in_bf = bnode (name ^ "_in_bf") and mid_f = fnode (name ^ "_mid_f") in
  let out_bf = bnode (name ^ "_out_bf") in
  List.iter [ in_bf; out_bf ] ~f:Ll_test.materialize;
  Ll_test.virtualize mid_f;
  let llc =
    LL.unflat_lines
      [
        Ll_test.loop_n s1 n (Ll_test.set mid_f i1 (Ll_test.get in_bf i1));
        Ll_test.loop_n s2 n
          (Ll_test.set out_bf i2
             (LL.Binop (Ops.Add, (Ll_test.get in_bf i2, bf), (Ll_test.get mid_f i2, f32))));
      ]
  in
  let o = Ll_test.optimize ~materialized:[ in_bf; out_bf ] ~name llc in
  (o, mid_f)

(* A single store of [x + (x + x) * x] whose product's first operand is annotated bfloat16: stale,
   since that operand is a single sum. The product, annotated bfloat16 in turn, describes it as
   written -- a product's own precision is its first operand's annotation -- and stops describing it
   once that annotation is repaired. *)
let stale_nested () =
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let in_f = fnode "opx_nest_in_f" and out_f = fnode "opx_nest_out_f" in
  let x = (Ll_test.get in_f i, f32) in
  let sum = LL.Binop (Ops.Add, x, x) in
  let product = LL.Binop (Ops.Mul, (sum, bf), x) in
  Ll_test.loop_n s n (Ll_test.set out_f i (LL.Binop (Ops.Add, x, (product, bf))))

(* A [?prelowered] bfloat sum whose second operand is a single scope over a bfloat load, annotated
   single: the scope's own precision, so descriptive as written. A schedule's trailing simplify
   collapses the scope into the sum (exactly: the widening loses nothing), leaving the load
   annotated single. *)
let scheduled_collapse name =
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let in_bf = bnode (name ^ "_in_bf") and mid_f = fnode (name ^ "_mid_f") in
  let out_bf = bnode (name ^ "_out_bf") in
  List.iter [ in_bf; out_bf ] ~f:Ll_test.materialize;
  let x = (Ll_test.get in_bf i, bf) in
  let scope = Ll_test.scope mid_f ~orig_indices:i (fst x) in
  let sum operand = Ll_test.loop_n s n (Ll_test.set out_bf i (LL.Binop (Ops.Add, x, operand))) in
  let o = Ll_test.optimize_scoped ~name ~raw:(sum x) (sum (scope, f32)) in
  let o = Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty = LL.Serial } ] o in
  (o, Ll_test.count_scopes o.LL.llc)

(* A [?prelowered] bfloat store selecting on a bfloat load whose condition slot -- a heterogeneous
   one, evaluated at its own annotation -- is annotated half, through a schedule. Returns the
   condition annotations left in the scheduled program. *)
let scheduled_condition name =
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let in_bf = bnode (name ^ "_in_bf") and out_bf = bnode (name ^ "_out_bf") in
  List.iter [ in_bf; out_bf ] ~f:Ll_test.materialize;
  let x = Ll_test.get in_bf i in
  let llc cond_prec =
    Ll_test.loop_n s n
      (Ll_test.set out_bf i (LL.Ternop (Ops.Where, (x, cond_prec), (x, bf), (LL.Constant 0., bf))))
  in
  let o = Ll_test.optimize_scoped ~name ~raw:(llc bf) (llc half) in
  let o = Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty = LL.Serial } ] o in
  let conds = ref [] in
  Ll_test.walk o.LL.llc ~on_scalar:(function
    | LL.Ternop (Ops.Where, (_, cprec), _, _) -> conds := cprec :: !conds
    | _ -> ());
  !conds

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
               LL.Ternop (Ops.Where, (get in_bf, half), (get in_bf, bf), (LL.Constant 0., bf)) ))));
  Verdict.case "the exit gate (gh-ocannl-1739)" (fun () ->
      refused
        "a ?prelowered routine carrying the old leg's form is refused at codegen's exit gate, \
         naming the destination, both precisions and the remedy"
        ~needles:
          [ "C_syntax.compile_proc"; "Operand 1 of Add"; "annotated single"; "here bfloat16" ]
        (prelowered "opx_req" ~annotate:f32);
      accepted "the same ?prelowered routine annotated bfloat16 compiles"
        (prelowered "opx_ok" ~annotate:bf);
      refused
        "a load's request reaches the exit gate through a schedule too: simplification keeps a \
         load's annotation (gh-ocannl-1788)"
        ~needles:
          [ "C_syntax.compile_proc"; "Operand 1 of Add"; "annotated single"; "here bfloat16" ]
        (prelowered ~scheduled:true "opx_req_ld_sch" ~annotate:f32);
      p "a schedule keeps a load's annotation in a heterogeneous slot (a Where condition)"
        (match scheduled_condition "opx_cond_sch" with
        | [ cprec ] -> Ops.equal_prec cprec half
        | _ -> false);
      let plus_one ~x p = LL.Binop (Ops.Add, (x, bf), (LL.Constant 1., p)) in
      refused
        "a request still reaches the exit gate through a schedule, which re-describes only a \
         program that starts descriptive"
        ~needles:[ "C_syntax.compile_proc"; "Operand 2 of Add"; "annotated half" ]
        (prelowered ~scheduled:true ~value:plus_one "opx_req_sch" ~annotate:half);
      accepted "its descriptive twin compiles through the same schedule"
        (prelowered ~scheduled:true ~value:plus_one "opx_ok_sch" ~annotate:bf);
      let o, mid = optimized_collapse "opx_col" in
      p
        "the single node read in the bfloat sum is inlined and its scope collapsed, so the case \
         reaches the stale annotation"
        (Ll_test.known_virtual o mid && Ll_test.count_scopes o.LL.llc = 0);
      p "the optimizer's output re-describes the collapsed scope's stale bfloat16 annotation"
        (Option.is_none (LL.operand_precision_violation o.LL.llc));
      accepted "the optimized routine compiles through the exit gate" (link_outcome "opx_col" o);
      let o, scopes = scheduled_collapse "opx_sch" in
      p
        "a descriptive ?prelowered routine stays descriptive through a schedule whose simplify \
         collapses a single scope"
        (scopes = 0 && Option.is_none (LL.operand_precision_violation o.LL.llc));
      accepted "the scheduled routine compiles through the exit gate" (link_outcome "opx_sch" o);
      p
        "describing repairs a nested operand before its parent, whose operand's own precision the \
         repair changes"
        (Option.is_none
           (LL.operand_precision_violation (LL.describe_operand_precisions (stale_nested ())))))
