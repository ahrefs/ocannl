(* gh-ocannl-2046: a constant is a value of its slot's precision. At a narrow float evaluation
   precision (f16, bf16, fp8) the simplifier's closing walk rounds it, after every fold, so a
   backend computing the format wider (cc under [narrow_compute_f32], Metal for fp8) reads the value
   every other backend and a materialized node read, at no run-time cost.

   Each case is hand-built and EXECUTED on the selected backend; the expectations are exact grid
   values written out here, not the codec the walk calls. Three shapes discriminate a rounded from
   an unrounded constant where the format computes wider: - a virtual narrow node holding the
   literal, read by an f32 node: it reads the literal itself; - a literal factor in narrow
   arithmetic: [3 * (1 + 3/8 u)] stores [3 + 2 u] unrounded, [3] rounded ([u] the narrow ulp at 1);
   - a folded literal: [3 * ((1 + 3/8 u) + (1 + 11/16 u))]. The fold is in double and rounded once,
   to [2 + 2 u], so the product [6 + 6 u] ties to [6 + 8 u]. Unrounded the store reads [6 + 4 u];
   rounding each operand before adding (to [1] and [1 + u], summing to the tie [2 + u], even [2])
   would read [6]. The three readings differ, so this pins the fold-once policy too. And one
   exemption: an integer exponent renders as repeated multiplication, so it is no value of the slot.
   An fp8 [2 ** 9] reads [512]; rounded to e5m2 the exponent would be [8], reading [256].

   Where the backend computes the format at storage width the first three read the same either way;
   the case still runs, and checks the value. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Tn = Ir.Tnode
open Verdict.Claims

let ulp prec = 2. **. Float.of_int (1 - Ops.significand_bits prec)
let next_id = ref 2046000

let fresh ~prec label =
  Int.incr next_id;
  Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy [| 1 |])
    ~padding:(lazy None)
    ()

let at0 = [| Ll_test.fixed 0 |]

(* Routine and node names are distinct across the cases (AGENTS.md: a same-named routine overwrites
   the earlier one's build_files/ artifacts), so each carries its precision, spaces made
   underscores. *)
let ident ~prec role label =
  String.concat ~sep:"_"
    [ "ncr"; role; Ops.prec_string prec; String.tr ~target:' ' ~replacement:'_' label ]

(* Bitwise, so a signed zero and a NaN of either sign are told apart from their neighbours. *)
let same_value got expected =
  (Float.is_nan got && Float.is_nan expected)
  || Int64.equal (Int64.bits_of_float got) (Int64.bits_of_float expected)

let run ~name ~materialized ~seed ~read llc =
  let o = Ll_test.optimize ~materialized ~name llc in
  (o, (List.hd_exn (Ll_test.execute ~name o ~seed ~read:[ read ])).(0))

let report label got expected =
  Stdio.eprintf "%s: reads %h, expected %h (not part of the golden)\n" label got expected;
  pf "%s: reads its rounded value" label (same_value got expected)

(* A virtual narrow node [v] holding [c], read into a materialized f32 node: the inlined scope
   evaluates [c] at [v]'s precision. *)
let virtual_literal ~prec ~label c ~expected =
  let v = fresh ~prec (ident ~prec "v" label)
  and dst = fresh ~prec:Ops.single (ident ~prec "d" label) in
  Ll_test.materialize dst;
  let llc =
    Ll_test.seq (Ll_test.set v at0 (LL.Constant c)) (Ll_test.set dst at0 (Ll_test.get v at0))
  in
  let o, got = run ~name:(ident ~prec "r" label) ~materialized:[ dst ] ~seed:[] ~read:dst llc in
  let label = Ops.prec_string prec ^ " virtual literal " ^ label in
  pf "%s: the node is inlined" label (Ll_test.known_virtual o v);
  report label got expected

(* [dst = src * llsc] at [prec], with [src] seeded to [3]. *)
let times_three ~prec ~label llsc ~expected =
  let src = fresh ~prec (ident ~prec "s" label) and dst = fresh ~prec (ident ~prec "d" label) in
  Ll_test.materialize src;
  Ll_test.materialize dst;
  let llc = Ll_test.set dst at0 (LL.Binop (Ops.Mul, (Ll_test.get src at0, prec), (llsc, prec))) in
  let _, got =
    run ~name:(ident ~prec "r" label) ~materialized:[ src; dst ]
      ~seed:[ (src, [| 3. |]) ]
      ~read:dst llc
  in
  report (Ops.prec_string prec ^ " " ^ label) got expected

let literal_operand prec =
  let u = ulp prec in
  times_three ~prec ~label:"literal factor" (LL.Constant (1. +. (3. /. 8. *. u))) ~expected:3.

let folded_operand prec =
  let u = ulp prec in
  let a = 1. +. (3. /. 8. *. u) and b = 1. +. (11. /. 16. *. u) in
  times_three ~prec ~label:"folded factor"
    (LL.Binop (Ops.Add, (LL.Constant a, prec), (LL.Constant b, prec)))
    ~expected:(6. +. (8. *. u))

let integer_exponent () =
  let prec = Ops.fp8 in
  let src = fresh ~prec (ident ~prec "s" "pow") and dst = fresh ~prec (ident ~prec "d" "pow") in
  Ll_test.materialize src;
  Ll_test.materialize dst;
  let llc =
    Ll_test.set dst at0
      (LL.Binop (Ops.ToPowOf, (Ll_test.get src at0, prec), (LL.Constant 9., prec)))
  in
  let o, got =
    run ~name:(ident ~prec "r" "pow") ~materialized:[ src; dst ]
      ~seed:[ (src, [| 2. |]) ]
      ~read:dst llc
  in
  pf "fp8 integer power: the exponent stays 9"
    (Ll_test.count_scalar o.LL.llc ~f:(function
       | LL.Binop (Ops.ToPowOf, _, (LL.Constant 9., _)) -> true
       | _ -> false)
    = 1);
  report "fp8 2 ** 9" got 512.

let () =
  let h = ulp Ops.half in
  List.iter
    ~f:(fun (label, c, expected) -> virtual_literal ~prec:Ops.half ~label c ~expected)
    [
      ("below a tie", 1. +. (3. /. 8. *. h), 1.);
      ("tie to even below", 1. +. (h /. 2.), 1.);
      ("tie to even above", 1. +. (3. /. 2. *. h), 1. +. (2. *. h));
      ("negative", -.(1. +. (5. /. 8. *. h)), -.(1. +. h));
      ("negative zero", -0., -0.);
      ("subnormal tie", 1.5 *. (2. **. -24.), 2. **. -23.);
      ("underflow", 2. **. -26., 0.);
      ("nan", Float.nan, Float.nan);
    ];
  virtual_literal ~prec:Ops.bfloat16 ~label:"below a tie"
    (1. +. (3. /. 8. *. ulp Ops.bfloat16))
    ~expected:1.;
  virtual_literal ~prec:Ops.bfloat16 ~label:"0.1" 0.1 ~expected:0x1.9ap-4;
  virtual_literal ~prec:Ops.fp8 ~label:"below a tie" (1. +. (3. /. 8. *. ulp Ops.fp8)) ~expected:1.;
  virtual_literal ~prec:Ops.fp8 ~label:"0.1" 0.1 ~expected:0x1.8p-4;
  List.iter [ Ops.half; Ops.bfloat16; Ops.fp8 ] ~f:(fun prec ->
      literal_operand prec;
      folded_operand prec);
  integer_exponent ()
