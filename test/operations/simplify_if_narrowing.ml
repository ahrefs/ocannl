(* gh-ocannl-566: [simplify_llc] narrows its interval environment with the bounds an enclosing [If]
   condition implies for its body, so a guard the condition proves is folded away.

   Exposed by gh-ocannl-487 phase 1: the depth-2 pipelined prefetch loads block [k+1] under [If (k <
   to_)], and the [k := k+1] substitution widens its zero-fringe edge guard's index interval past
   the extent -- the loop range alone cannot decide it, the enclosing condition can. That end-to-end
   case is pinned (structurally and by executed bitwise parity) in [schedule_pipelined_matmul]; here
   each narrowing rule is exercised in isolation against a discriminating control that must NOT
   fold, since a folded guard silently changes the value a cell holds.

   High-level lowering does not produce these shapes on demand, so the cases are built directly as
   [Ir.Low_level.t] and run through [Ir.Low_level.simplify_llc] -- the pass under test, called with
   no static indices, exactly as [Schedule.apply]'s trailing simplify calls it. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Asgns = Ir.Assignments
open Verdict.Claims

let single = Ops.single
let iprec = Ops.index_prec ()

(* Builders and structural counters come from [Ll_test] (gh-ocannl-600, gh-ocannl-608): the
   index-precision comparison and connective builders this file needed are now part of that shared
   set, so a guard is spelled the same way here and in every other hand-built-IR test. The names
   below are this file's vocabulary for them, not a second implementation. *)
let mk = Ll_test.node_factory ~first_id:7000 ~dims:[| 16 |] ()
let sym = Ll_test.sym
let iter = Ll_test.iter
let embed = Ll_test.embed_idx
let ivar = Ll_test.embed
let ic = Ll_test.ic

(* [Σ coeff*sym + offset] as an index expression. *)
let affine_idx = Ll_test.aff
let affine symbols offset = embed (affine_idx symbols offset)
let lt = Ll_test.lt
let le = Ll_test.le
let eq = Ll_test.eq
let ne = Ll_test.ne
let conj = Ll_test.conj
let disj = Ll_test.disj

(* The zero-fringe shape: [cond ? src[idx] : 0]. *)
let where_zero cond src idx : LL.scalar_t =
  Ll_test.where_ cond (Ll_test.get src [| idx |]) (Ll_test.c 0.)

let set_at = Ll_test.set_at
let set dst s llsc : LL.t = set_at dst (iter s) llsc
let loop ?from_ ~to_ s body : LL.t = Ll_test.loop ?from_ ~upto:to_ s body
let if_ = Ll_test.if_idx

(* --- structural probes on the simplified form --- *)
let wheres llc =
  Ll_test.count_scalar llc ~f:(function LL.Ternop (Ops.Where, _, _, _) -> true | _ -> false)

let ifs llc = Ll_test.count_stmt llc ~f:(function LL.If _ -> true | _ -> false)
let zeros llc = Ll_test.count_scalar llc ~f:(function LL.Constant 0. -> true | _ -> false)
let simplify llc = LL.simplify_llc [] llc

(* The motivating shape, parameterized by the guard wrapping the inner nest: two k-blocks of 16 over
   an extent of 32, the prefetch reading a block ahead ([i + 16*k + 16], the real prefetch's [mb]
   read) into a k-dependent slot ([i + 16*k]), edge-guarded against the extent. Under [k <= 0] the
   read index stays in [16, 31] and the guard folds; over the full loop range it reaches 47 and must
   not. Returns the tnodes for the executed leg below. *)
let block_case ~guard_of =
  let dst = mk ~dims:[| 32 |] "sin_dst" and src = mk ~dims:[| 32 |] "sin_src" in
  let k = sym () and i = sym () in
  let read_idx = affine_idx [ (1, i); (16, k) ] 16 in
  let body =
    loop ~to_:15 i
      (set_at dst
         (affine_idx [ (1, i); (16, k) ] 0)
         (where_zero (lt (embed read_idx) (ic 32)) src read_idx))
  in
  (loop ~to_:1 k (match guard_of k with None -> body | Some cond -> if_ cond body), dst, src)

let block_llc ~guard_of =
  let llc, _, _ = block_case ~guard_of in
  llc

(* Mixed-radix index [4*k + m] over [0, 3] x [0, 3], reaching 15; narrowing only [k] leaves 7,
   narrowing both leaves 5, so the edge guard [< 6] discriminates the [And] rule from either half.
   Returns the tnodes for the executed leg below. *)
let mixed_case ~guard_of =
  let dst = mk "sin_dst2" and src = mk ~dims:[| 16 |] "sin_src2" in
  let k = sym () and m = sym () in
  let idx = affine_idx [ (4, k); (1, m) ] 0 in
  ( loop ~to_:3 k
      (loop ~to_:3 m (if_ (guard_of k m) (set dst m (where_zero (lt (embed idx) (ic 6)) src idx)))),
    dst,
    src )

let mixed_llc ~guard_of =
  let llc, _, _ = mixed_case ~guard_of in
  llc

let () =
  (* --- Cmplt: the pipelined prefetch's own shape. --- *)
  let folded = simplify (block_llc ~guard_of:(fun k -> Some (lt (ivar k) (ic 1)))) in
  let unguarded = simplify (block_llc ~guard_of:(fun _ -> None)) in
  let weak = simplify (block_llc ~guard_of:(fun k -> Some (lt (ivar k) (ic 2)))) in
  p "Cmplt guard folds the edge guard it proves" (wheres folded = 0 && zeros folded = 0);
  p "without the guard the same edge guard survives" (wheres unguarded = 1);
  p "a guard weaker than the loop range narrows nothing" (wheres weak = 1);

  (* --- Cmple. --- *)
  let folded_le = simplify (block_llc ~guard_of:(fun k -> Some (le (ivar k) (ic 0)))) in
  let weak_le = simplify (block_llc ~guard_of:(fun k -> Some (le (ivar k) (ic 1)))) in
  p "Cmple guard folds the edge guard it proves" (wheres folded_le = 0);
  p "a Cmple guard at the loop's top narrows nothing" (wheres weak_le = 1);

  (* --- Cmpeq narrows both ways; Cmpne (its negation) is not a narrowing rule. --- *)
  let folded_eq = simplify (block_llc ~guard_of:(fun k -> Some (eq (ivar k) (ic 0)))) in
  let unfolded_ne = simplify (block_llc ~guard_of:(fun k -> Some (ne (ivar k) (ic 0)))) in
  p "Cmpeq guard folds the edge guard it proves" (wheres folded_eq = 0);
  p "Cmpne guard narrows nothing" (wheres unfolded_ne = 1);

  (* --- Conjunction: both conjuncts contribute, and neither alone suffices ([mixed_case] above). A
     disjunction implies neither and must narrow nothing. --- *)
  let both =
    simplify (mixed_llc ~guard_of:(fun k m -> conj (lt (ivar k) (ic 2)) (lt (ivar m) (ic 2))))
  in
  let one = simplify (mixed_llc ~guard_of:(fun k _ -> lt (ivar k) (ic 2))) in
  let either =
    simplify (mixed_llc ~guard_of:(fun k m -> disj (lt (ivar k) (ic 2)) (lt (ivar m) (ic 2))))
  in
  p "And guard narrows both conjuncts' symbols" (wheres both = 0);
  p "only one conjunct's narrowing does not decide it" (wheres one = 1);
  p "Or guard narrows nothing" (wheres either = 1);

  (* --- The reversed operand order narrows the symbol's LOWER bound (a negative coefficient in the
     difference), and the decided guard folds to its else-branch. --- *)
  let low_case ~guarded =
    let dst = mk "sin_dst3" and src = mk ~dims:[| 16 |] "sin_src3" in
    let k = sym () in
    let body = set dst k (where_zero (lt (ivar k) (ic 6)) src (iter k)) in
    loop ~to_:9 k (if guarded then if_ (lt (ic 5) (ivar k)) body else body)
  in
  let low = simplify (low_case ~guarded:true) in
  let low_ref = simplify (low_case ~guarded:false) in
  p "a lower-bound guard folds a refuted edge guard to its else-branch"
    (wheres low = 0 && zeros low = 1);
  p "without it the same edge guard survives" (wheres low_ref = 1);

  (* --- An inner guard the outer one implies is erased entirely (statement-level). --- *)
  let nested =
    let dst = mk "sin_dst4" and src = mk ~dims:[| 16 |] "sin_src4" in
    let k = sym () in
    simplify
      (loop ~to_:9 k
         (if_
            (lt (ivar k) (ic 3))
            (if_ (lt (ivar k) (ic 5)) (set dst k (LL.Get (src, [| iter k |]))))))
  in
  p "an inner guard implied by the outer one is erased" (ifs nested = 1);

  (* --- Negative controls on soundness: a condition that is not an integer-affine index comparison
     contributes no facts, and a guard that is unsatisfiable in conjunction (a dead body) is left
     alone rather than fabricating an empty interval. --- *)
  let data_dep =
    let dst = mk "sin_dst5" and src = mk ~dims:[| 32 |] "sin_src5" and sel = mk "sin_sel" in
    let k = sym () and i = sym () in
    let idx = affine [ (1, i); (16, k) ] 16 in
    simplify
      (loop ~to_:1 k
         (if_
            (LL.Binop
               (Ops.Cmplt, (LL.Get (sel, [| Idx.Fixed_idx 0 |]), single), (LL.Constant 1., single)))
            (loop ~to_:15 i (set dst i (where_zero (lt idx (ic 32)) src (iter i))))))
  in
  p "a data-dependent guard narrows nothing" (wheres data_dep = 1 && ifs data_dep = 1);
  let dead =
    let dst = mk "sin_dst6" and src = mk ~dims:[| 16 |] "sin_src6" in
    let k = sym () in
    simplify
      (loop ~to_:1 k
         (if_
            (conj (lt (ivar k) (ic 1)) (lt (ic 0) (ivar k)))
            (set dst k (LL.Get (src, [| iter k |])))))
  in
  p "an unsatisfiable conjunction is left alone" (ifs dead = 1);

  (* --- Evaluation-precision soundness: past a float precision's exact-integer range the machine
     comparison is not the mathematical one -- single represents [2^24 + 1] as [2^24], so a
     single-precision guard [k <= 2^24] is machine-true at [k = 2^24 + 1] too, and narrowing from it
     would fold the (index-precision) edge guard to the wrong branch there. The identical guard
     evaluated at the index precision is exact and narrows. --- *)
  let lim =
    16777216
    (* 2^24, [Interval.float_exact_int_limit] of single *)
  in
  let prec_case ~cond_prec =
    let dst = mk "sin_dst7" and src = mk "sin_src7" in
    let k = sym () in
    simplify
      (loop ~to_:(lim + 1) k
         (LL.If
            {
              cond = (LL.Binop (Ops.Cmple, (ivar k, cond_prec), (ic lim, cond_prec)), cond_prec);
              body = set dst k (where_zero (le (ivar k) (ic lim)) src (iter k));
            }))
  in
  p "a float-precision guard past its exact-integer range narrows nothing"
    (wheres (prec_case ~cond_prec:single) = 1);
  p "the same guard at the index precision narrows" (wheres (prec_case ~cond_prec:iprec) = 0)

(* --- Executed leg (the structural-vs-executable rule): a folded guard changes the emitted code but
   must not change any value, and an over-narrowing defect would -- by folding a [Where] to the
   wrong branch. Each representative case runs raw and simplified through the configured backend and
   both results are compared bitwise to a hand-computed reference. The carrier comp (a plain copy)
   only pins the tnodes' read/write roles and placement; [lowered_transform] replaces its lowered
   code with the case, [Zero_out]-prefixed so unwritten cells are defined. --- *)

let dbg : Idx.projections_debug = { spec = ""; derived_for = Sexp.Atom ""; trace = [] }

let copy_proj t ~n : Idx.projections =
  {
    components = [| [ (n, t) ] |];
    lhs_dims = [| n |];
    rhs_dims = [| [| n |] |];
    project_lhs = [| iter t |];
    project_rhs = [| [| iter t |] |];
    extent_syms = [];
    debug_info = dbg;
  }

let exec ~name ~n (llc, dst, src) ~src_vals =
  Tn.update_memory_mode dst Tn.On_device "99:test-setup";
  let carrier =
    Asgns.Accum_op
      {
        initialize_neutral = true;
        accum = Ops.Add;
        lhs = dst;
        rhs = Asgns.Unop { op = Ops.Identity; rhs = Asgns.Node src };
        projections = lazy (copy_proj (sym ()) ~n);
        projections_debug = "carrier";
      }
  in
  let comp = { Asgns.asgns = carrier; embedded_nodes = Set.of_list (module Tn) [ dst ] } in
  let ctx = Context.auto () in
  let ctx = Context.set_values ctx src src_vals in
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun o -> [ { o with LL.llc = LL.Seq (LL.Zero_out dst, llc) } ])
      ctx comp Idx.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx dst

let run_case ~name ~routine ~n ((llc, dst, src) as case) ~expected =
  let src_vals = Array.init n ~f:(fun j -> 100. +. Float.of_int j) in
  let raw = exec ~name:(routine ^ "_raw") ~n case ~src_vals in
  let simp = exec ~name:(routine ^ "_simp") ~n (simplify llc, dst, src) ~src_vals in
  let expected = expected src_vals in
  p (name ^ " raw and simplified execute bitwise-identically") (Array.equal Float.equal raw simp);
  p
    (name ^ " executed values match a nonzero hand-computed reference")
    (Array.equal Float.equal raw expected
    && Array.exists expected ~f:(fun v -> not (Float.equal v 0.)))

let () =
  (* The motivating fold: only the k = 0 block runs, prefetching src[16..31] into dst[0..15]. *)
  run_case ~name:"exec Cmplt block" ~routine:"sin_exec_cmplt" ~n:32
    (block_case ~guard_of:(fun k -> Some (lt (ivar k) (ic 1))))
    ~expected:(fun sv -> Array.init 32 ~f:(fun j -> if j < 16 then sv.(j + 16) else 0.));
  (* Unguarded control: the k = 1 block's edge guard survives and writes zeros over zeros. *)
  run_case ~name:"exec unguarded block" ~routine:"sin_exec_unguarded" ~n:32
    (block_case ~guard_of:(fun _ -> None))
    ~expected:(fun sv -> Array.init 32 ~f:(fun j -> if j < 16 then sv.(j + 16) else 0.));
  (* The And rule: both conjuncts narrow, the k = 1 pass overwrites the k = 0 values. *)
  run_case ~name:"exec And mixed" ~routine:"sin_exec_and" ~n:16
    (mixed_case ~guard_of:(fun k m -> conj (lt (ivar k) (ic 2)) (lt (ivar m) (ic 2))))
    ~expected:(fun sv -> Array.init 16 ~f:(function 0 -> sv.(4) | 1 -> sv.(5) | _ -> 0.));
  (* The lower-bound (reversed-operand) rule folding to the THEN branch: under [5 < k] the edge
     guard [4 < k] is proved, so cells 6..9 carry src values. *)
  let low_then_case =
    let dst = mk "sin_dst8" and src = mk "sin_src8" in
    let k = sym () in
    ( loop ~to_:9 k
        (if_ (lt (ic 5) (ivar k)) (set dst k (where_zero (lt (ic 4) (ivar k)) src (iter k)))),
      dst,
      src )
  in
  let low_then_llc, _, _ = low_then_case in
  p "a lower-bound guard folds a proven edge guard to its then-branch"
    (wheres (simplify low_then_llc) = 0 && wheres low_then_llc = 1);
  run_case ~name:"exec lower-bound" ~routine:"sin_exec_lower" ~n:16 low_then_case
    ~expected:(fun sv -> Array.init 16 ~f:(fun j -> if 6 <= j && j <= 9 then sv.(j) else 0.))
