(* gh-ocannl-514 phase 3: [Ir.Cost_model.completion_floor] — the dual (lower-bound) extraction over
   hand-built programs where every number is checkable by hand (single precision, 4 bytes/cell).
   Each case also asserts the duality invariant on the same code: the floor never exceeds the upper
   extraction ([analyze]).

   - pointwise map: all accesses exact, floor = upper on both legs; - matmul with an rmw
   accumulator: exact when closed; opening the accumulator's placement attributes the whole producer
   — its ops and operand reads — to the open level, and committing (re-evaluation with the narrowed
   open set) recovers the closed floor, monotone in refinement; - guarded write: the floor counts
   guards-never-taken (condition ops only, no write traffic) where the upper counts guards-taken —
   [fr_exact] false; - two reads of one node: pairwise provably-disjoint exact images sum in both
   extractions (gh-ocannl-578); possibly-overlapping ones keep the asymmetry — the floor takes the
   larger exact image (a union is at least its largest member, flagged loose) where the upper takes
   the capped sum; - short-circuiting forms: Where arms, And/Or right operands (conditional —
   cheaper-arm / left-operand floors, conditional reads zeroed) and Arg1/Arg2 discarded operands
   (never rendered at all, so absent from both extractions); - the over-producing open producer,
   dead loops, and the dynamic-gather fallback flooring to zero where the upper falls back to the
   whole node.

   The last section pins the classifier those short-circuiting cases now read from
   ([Ops.binop_conditionality] / [Ops.ternop_conditionality], gh-ocannl-582) and its agreement with
   the plain-C renderings. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops
module CM = Ir.Cost_model

let fresh_tn =
  let c = ref 980_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ops.single

let for_over ?(extent = 4) sym body =
  LL.For_loop { index = sym; from_ = 0; to_ = extent - 1; body; axis = LL.Serial }

let get tn idcs = LL.Get (tn, idcs)
let it s = Idx.Iterator s

let show name ?open_placement code =
  let s = CM.analyze code in
  let f = CM.completion_floor ?open_placement code in
  Stdio.printf "== %s ==\n  floor: flops=%d bytes=%d %s\n  upper: flops=%d bytes=%d\n" name
    f.CM.fr_flops f.CM.fr_bytes
    (if f.CM.fr_exact then "exact" else "inexact")
    s.CM.flops (CM.total_bytes s);
  Verdict.p "  floor <= upper" (f.CM.fr_flops <= s.CM.flops && f.CM.fr_bytes <= CM.total_bytes s);
  f

let () =
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  (* Pointwise 4x5 map: C[i][j] = A[i][j] + B[j]. All accesses exact: floor = upper. A rd 80 B, B rd
     20 B, C wr 80 B; 20 adds. *)
  let a = fresh_tn "A" [| 4; 5 |] in
  let b = fresh_tn "B" [| 5 |] in
  let c = fresh_tn "C" [| 4; 5 |] in
  let pointwise =
    for_over i
      (for_over ~extent:5 j
         (LL.Set
            {
              tn = c;
              idcs = [| it i; it j |];
              llsc = LL.Binop (Ops.Add, (get a [| it i; it j |], sp), (get b [| it j |], sp));
              debug = "";
            }))
  in
  let _ = show "pointwise map" pointwise in
  (* 4x5x6 matmul, rmw accumulator: D[i][j] += A2[i][k] * B2[k][j]. Exact both ways: flops = 2*4*5*6
     = 240; bytes = A2 96 + B2 120 + D rd 80 + D wr 80 = 376. Opening D's placement drops its 160
     bytes; node_floor_bytes D restores them — the Materialize delta. *)
  let a2 = fresh_tn "A2" [| 4; 6 |] in
  let b2 = fresh_tn "B2" [| 6; 5 |] in
  let d = fresh_tn "D" [| 4; 5 |] in
  let matmul =
    for_over i
      (for_over ~extent:5 j
         (for_over ~extent:6 k
            (LL.Set
               {
                 tn = d;
                 idcs = [| it i; it j |];
                 llsc =
                   LL.Binop
                     ( Ops.Add,
                       (get d [| it i; it j |], sp),
                       ( LL.Binop
                           (Ops.Mul, (get a2 [| it i; it k |], sp), (get b2 [| it k; it j |], sp)),
                         sp ) );
                 debug = "";
               })))
  in
  let closed = show "matmul (rmw accumulator)" matmul in
  let opened = show "matmul, D's placement open" ~open_placement:(fun tn -> Tn.equal tn d) matmul in
  (* Committing a placement is re-evaluation with the narrowed open set: suppression only shrinks,
     so the floor is monotone in refinement — here, committing D (the only open node) recovers the
     closed floor exactly, and both legs are monotone. *)
  let committed = CM.completion_floor ~open_placement:(fun _ -> false) matmul in
  Verdict.p "  commitment (narrowed open set) recovers the closed floor"
    (committed.CM.fr_bytes = closed.CM.fr_bytes && committed.CM.fr_flops = closed.CM.fr_flops);
  Verdict.p "  floors are monotone in refinement"
    (opened.CM.fr_bytes <= committed.CM.fr_bytes && opened.CM.fr_flops <= committed.CM.fr_flops);
  (* Guarded write: if (E[i] > 0) F[i] = E[i] * 2. Upper counts guards-taken (cmp + mul per
     iteration, F written); the floor counts only the certain condition ops and no F traffic. *)
  let e = fresh_tn "E" [| 4 |] in
  let fq = fresh_tn "F" [| 4 |] in
  let guarded =
    for_over i
      (LL.If
         {
           cond = (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get e [| it i |], sp)), sp);
           body =
             LL.Set
               {
                 tn = fq;
                 idcs = [| it i |];
                 llsc = LL.Binop (Ops.Mul, (get e [| it i |], sp), (LL.Constant 2., sp));
                 debug = "";
               };
         })
  in
  let _ = show "guarded write" guarded in
  (* Two exact reads of one node with provably disjoint images: G[i][j] = H[0][j] + H[1][j]. Both
     extractions agree the union is the sum (gh-ocannl-578): 10 cells = 40 rd bytes, G writes 80
     bytes both ways — floor = upper = 120, exact. *)
  let g = fresh_tn "G" [| 4; 5 |] in
  let h = fresh_tn "H" [| 4; 5 |] in
  let two_reads =
    for_over i
      (for_over ~extent:5 j
         (LL.Set
            {
              tn = g;
              idcs = [| it i; it j |];
              llsc =
                LL.Binop
                  ( Ops.Add,
                    (get h [| Idx.Fixed_idx 0; it j |], sp),
                    (get h [| Idx.Fixed_idx 1; it j |], sp) );
              debug = "";
            }))
  in
  let _ = show "two disjoint reads, both extractions sum" two_reads in
  (* Two exact reads that may overlap: G2[i] = H8[i] + H8[i+1]. The upper sums (8 cells = 32 rd
     bytes, a union bound); the floor takes only the larger exact image (4 cells = 16 bytes) — a
     union is at least its largest member, no more is certain. G2 writes 16 both ways: floor 32 vs
     upper 48. *)
  let g2 = fresh_tn "G2" [| 4 |] in
  let h8 = fresh_tn "H8" [| 8 |] in
  let shift1 s = Idx.Affine { symbols = [ (1, s) ]; offset = 1 } in
  let overlapping_reads =
    for_over i
      (LL.Set
         {
           tn = g2;
           idcs = [| it i |];
           llsc = LL.Binop (Ops.Add, (get h8 [| it i |], sp), (get h8 [| shift1 i |], sp));
           debug = "";
         })
  in
  let _ = show "two overlapping reads, union floor = larger image" overlapping_reads in
  (* Dynamic gather: P[i] = Q[R[i]]. Q's access is uninterpretable — the upper falls back to the
     whole node, the floor to zero (inexact). *)
  let p = fresh_tn "P" [| 4 |] in
  let q = fresh_tn "Q" [| 8 |] in
  let r = fresh_tn "R" [| 4 |] in
  let gather =
    for_over i
      (LL.Set
         {
           tn = p;
           idcs = [| it i |];
           llsc =
             LL.Get_dynamic
               {
                 tn = q;
                 idcs = [| Idx.Fixed_idx 0 |];
                 dyn_axis = 0;
                 dyn_value = (get r [| it i |], sp);
               };
           debug = "";
         })
  in
  let _ = show "dynamic gather" gather in
  (* Where short-circuits (?: in every renderer): K[i] = where(E>0, E*2, E+M[i]+1). The floor counts
     cond + select + the cheaper arm (1+1+1 = 3/iter vs the upper's both-arms 5/iter), and M — read
     only inside an arm — floors its read to zero; E is also read in an arm, so its read floors too
     (node-granular certainty, conservative). *)
  let kq = fresh_tn "K" [| 4 |] in
  let e2 = fresh_tn "E2" [| 4 |] in
  let m = fresh_tn "M" [| 4 |] in
  let where_case =
    for_over i
      (LL.Set
         {
           tn = kq;
           idcs = [| it i |];
           llsc =
             LL.Ternop
               ( Ops.Where,
                 (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get e2 [| it i |], sp)), sp),
                 (LL.Binop (Ops.Mul, (get e2 [| it i |], sp), (LL.Constant 2., sp)), sp),
                 ( LL.Binop
                     ( Ops.Add,
                       (get e2 [| it i |], sp),
                       (LL.Binop (Ops.Add, (get m [| it i |], sp), (LL.Constant 1., sp)), sp) ),
                   sp ) );
           debug = "";
         })
  in
  let _ = show "where short-circuit" where_case in
  (* An open producer computing a larger domain than consumed: P2[0..7] = A3[0..7] * 2, then C3[0] =
     P2[0]. The inline completion instantiates one multiply and one A3 read and drops the setter
     loop, so with P2 open the floor keeps only C3's certain write — the producer's ops AND its A3
     reads attribute to the open placement. *)
  let p2 = fresh_tn "P2" [| 8 |] in
  let a3 = fresh_tn "A3" [| 8 |] in
  let c3 = fresh_tn "C3" [| 1 |] in
  let over_produce =
    LL.Seq
      ( for_over ~extent:8 j
          (LL.Set
             {
               tn = p2;
               idcs = [| it j |];
               llsc = LL.Binop (Ops.Mul, (get a3 [| it j |], sp), (LL.Constant 2., sp));
               debug = "";
             }),
        LL.Set
          { tn = c3; idcs = [| Idx.Fixed_idx 0 |]; llsc = get p2 [| Idx.Fixed_idx 0 |]; debug = "" }
      )
  in
  let _ = show "over-producing closed" over_produce in
  let _ =
    show "over-producing, P2's placement open"
      ~open_placement:(fun tn -> Tn.equal tn p2)
      over_produce
  in
  (* And renders as the short-circuiting &&: the certain floor is the op plus the left operand; the
     right comparison (reading M) is conditional — its op floors away and M's read floors to zero.
     Arg1 renders only its selected operand: the discarded expensive right operand contributes to
     neither extraction — not its ops, not its M read. *)
  let l = fresh_tn "L" [| 4 |] in
  let and_case =
    for_over i
      (LL.Set
         {
           tn = l;
           idcs = [| it i |];
           llsc =
             LL.Binop
               ( Ops.And,
                 (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get e2 [| it i |], sp)), sp),
                 (LL.Binop (Ops.Cmplt, (get m [| it i |], sp), (LL.Constant 1., sp)), sp) );
           debug = "";
         })
  in
  let _ = show "and short-circuit" and_case in
  let arg1_case =
    for_over i
      (LL.Set
         {
           tn = l;
           idcs = [| it i |];
           llsc =
             LL.Binop
               ( Ops.Arg1,
                 (get e2 [| it i |], sp),
                 (LL.Binop (Ops.Mul, (get m [| it i |], sp), (LL.Constant 3., sp)), sp) );
           debug = "";
         })
  in
  let _ = show "arg1 discarded operand" arg1_case in
  (* Relu_gate also renders with ?: — its right operand (the expensive product reading M) is
     conditional: floor = gate + left operand per iteration, M's read zeroed. *)
  let gate_case =
    for_over i
      (LL.Set
         {
           tn = l;
           idcs = [| it i |];
           llsc =
             LL.Binop
               ( Ops.Relu_gate,
                 (get e2 [| it i |], sp),
                 (LL.Binop (Ops.Mul, (get m [| it i |], sp), (LL.Constant 3., sp)), sp) );
           debug = "";
         })
  in
  let _ = show "relu-gate short-circuit" gate_case in
  (* A dead loop's body never executes: its whole-node access must not reach the floor. *)
  let d2 = fresh_tn "D2" [| 4 |] in
  let dead =
    LL.For_loop { index = k; from_ = 0; to_ = -1; body = LL.Zero_out d2; axis = LL.Serial }
  in
  let _ = show "dead loop" dead in
  ()

(* gh-ocannl-582: the classifier the cases above read from. The table is printed over [Ops]' derived
   enumeration, so adding an operator is a visible promotion diff here whether or not anyone
   remembers this test; the second half checks the plain-C renderings against the classifier — the
   same check every backend runs on itself at [C_syntax] functor application, where the GPU
   spellings (which shadow these) get covered on their own hardware. *)
module C_config = Ir.C_syntax.Pure_C_config (struct
  let procs = [||]
  let full_printf_support = true
end)

let () =
  Stdio.printf "\n== operand conditionality ==\n";
  List.iter Ops.all_of_binop ~f:(fun op ->
      Stdio.printf "  %-28s %s\n" (Ops.binop_cd_fallback_syntax op)
        (Sexp.to_string @@ Ops.sexp_of_binop_conditionality @@ Ops.binop_conditionality op));
  List.iter Ops.all_of_ternop ~f:(fun op ->
      Stdio.printf "  %-28s %s\n" (Ops.ternop_cd_syntax op)
        (Sexp.to_string @@ Ops.sexp_of_ternop_conditionality @@ Ops.ternop_conditionality op));
  let violations =
    Ir.C_syntax.operand_conditionality_violations ~ternop_syntax:C_config.ternop_syntax
      ~binop_syntax:C_config.binop_syntax
  in
  (* Over the operators the table above enumerates, which is the sweep the violations came from. *)
  let swept =
    List.map Ops.all_of_binop ~f:Ops.binop_cd_fallback_syntax
    @ List.map Ops.all_of_ternop ~f:Ops.ternop_cd_syntax
  in
  Verdict.p_empty "  plain-C renderings agree" ~over:swept violations;
  List.iter violations ~f:(fun v -> Stdio.printf "  VIOLATION %s\n" v)
