(* gh-ocannl-2130: large RNG inputs under a half store are legal. This already passed before sharing
   the slot rule: the heterogeneous uniform conversion takes its block at uint4x32, so Threefry
   never evaluates its operands at half. Compare a nested block with an explicitly materialized
   block, for both RNGs, both input conversion paths (double bits versus an integer index), and both
   uniform conversions. Each counter varies with the output cell. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let n = 8
let big = 70000

let check ~ordinal ~op ~indexed ~lane =
  let label =
    Printf.sprintf "%s_%s_%s" (Ops.binop_cd_fallback_syntax op)
      (if indexed then "index" else "constant")
      (if lane then "lane" else "scalar")
  in
  let first_id = 2130000 + (ordinal * 10) in
  let mk = Ll_test.node_factory ~prec:Ops.half ~first_id ~dims:[| n |] () in
  let nested = mk "oep_nested" and stored = mk "oep_stored" in
  let block =
    Ll_test.node_factory ~prec:Ops.uint4x32 ~first_id:(first_id + 2) ~dims:[| n |] () "oep_block"
  in
  List.iter [ nested; stored; block ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let idcs = [| Ll_test.iter i |] in
  let key =
    if indexed then LL.Embed_index (Ll_test.fixed big) else LL.Constant (Float.of_int big)
  in
  let rng = LL.Binop (op, (key, LL.scalar_precision key), (Ll_test.embed i, Ll_test.iprec ())) in
  let draw block =
    if lane then
      LL.Binop
        ( Ops.Uint4x32_to_prec_uniform_lane,
          (block, Ops.uint4x32),
          (LL.Embed_index (Ll_test.fixed 3), Ll_test.iprec ()) )
    else LL.Unop (Ops.Uint4x32_to_prec_uniform1, (block, Ops.uint4x32))
  in
  let llc =
    Ll_test.loop_n i n
      (LL.unflat_lines
         [
           Ll_test.set block idcs rng;
           Ll_test.set nested idcs (draw rng);
           Ll_test.set stored idcs (draw (Ll_test.get block idcs));
         ])
  in
  let name = "oep_" ^ label in
  let optimized = Ll_test.optimize ~materialized:[ nested; stored; block ] ~name llc in
  p
    (label ^ ": a uniform conversion still directly contains the nested RNG")
    (Ll_test.count_scalar optimized.LL.llc ~f:(function
       | LL.Unop
           ( Ops.Uint4x32_to_prec_uniform1,
             (LL.Binop ((Ops.Threefry4x32_crypto | Ops.Threefry4x32_light), _, _), _) )
       | LL.Binop
           ( Ops.Uint4x32_to_prec_uniform_lane,
             (LL.Binop ((Ops.Threefry4x32_crypto | Ops.Threefry4x32_light), _, _), _),
             _ ) ->
           true
       | _ -> false)
    > 0);
  let got = Ll_test.execute ~name optimized ~seed:[] ~read:[ nested; stored ] in
  let a = List.nth_exn got 0 and b = List.nth_exn got 1 in
  p
    (label ^ ": large RNG input is accepted under a half store and matches materialized")
    (Ll_test.same ~tol:0. [ a ] [ b ]);
  p_all (label ^ ": draws are finite and in [0, 1)") (Array.to_list a) ~f:(fun x ->
      Float.is_finite x && Float.(x >= 0. && x < 1.));
  p
    (label ^ ": counters discriminate output cells")
    (List.length (List.dedup_and_sort (Array.to_list a) ~compare:Float.compare) > 1)

let check_slots () =
  let check label op annotations expected =
    p label
      (Array.equal Ops.equal_prec
         (LL.operand_eval_precisions ~eval:Ops.half op annotations)
         expected)
  in
  check "homogeneous arithmetic inherits the slot, ignoring operand annotations" (Ops.Binop Ops.Add)
    [| Ops.single; Ops.double |] [| Ops.half; Ops.half |];
  check "Where keeps its condition's precision and inherits the slot in both arms"
    (Ops.Ternop Ops.Where)
    [| Ops.int32; Ops.single; Ops.double |]
    [| Ops.int32; Ops.half; Ops.half |];
  check "Round_prec keeps its exact narrow annotation" (Ops.Unop Ops.Round_prec) [| Ops.bfloat16 |]
    [| Ops.bfloat16 |];
  check "uniform lane keeps the block and lane precisions under a half slot"
    (Ops.Binop Ops.Uint4x32_to_prec_uniform_lane) [| Ops.uint4x32; Ops.int32 |]
    [| Ops.uint4x32; Ops.int32 |]

(* Internal rewrites can leave stale operator annotations before the closing describe walk. The
   Where is annotated packed but evaluated in a single slot. Its condition must never go through the
   packed at_slot rewrite: re-annotating it there and restoring its own precision afterwards made
   the simplifier recurse forever instead of reaching a fixed point. *)
let check_stale_where () =
  let mk = Ll_test.node_factory ~first_id:2130900 ~dims:[| 1 |] () in
  let flag = mk "oep_flag" and dst = mk "oep_stale" in
  let at0 = [| Ll_test.fixed 0 |] in
  let conditional =
    LL.Ternop
      ( Ops.Where,
        (Ll_test.get flag at0, Ops.single),
        (LL.Constant 1., Ops.single),
        (LL.Constant 2., Ops.single) )
  in
  let raw =
    Ll_test.set dst at0
      (LL.Binop (Ops.Add, (conditional, Ops.uint4x32), (Ll_test.get flag at0, Ops.single)))
  in
  let simplified = LL.simplify_llc [] raw in
  p "a stale packed Where annotation terminates with its condition at its own single precision"
    (Ll_test.count_scalar simplified ~f:(function
       | LL.Ternop (Ops.Where, (_, p), _, _) -> Ops.equal_prec p Ops.single
       | _ -> false)
    = 1)

let () =
  check_slots ();
  check_stale_where ();
  let ctx = Context.auto () in
  Stdio.eprintf "rng_operand_precision: backend %s (not part of the golden)\n"
    (Context.backend_name ctx);
  let ordinal = ref 0 in
  List.iter [ Ops.Threefry4x32_crypto; Ops.Threefry4x32_light ] ~f:(fun op ->
      List.iter [ false; true ] ~f:(fun indexed ->
          List.iter [ false; true ] ~f:(fun lane ->
              check ~ordinal:!ordinal ~op ~indexed ~lane;
              Int.incr ordinal)))
