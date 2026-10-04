(* gh-ocannl-637 Part 2, gh-ocannl-1011: [Ir.Cost_model]'s account of ONE read of a node — the
   recompute cost the virtualizer used to price with a traced proxy (reduction extent × read
   multiplicity × transitive fan-in). Since gh-ocannl-1011 the priced code is the inliner's own
   instantiation at a synthetic read site, after the pipeline the emitted code receives
   ([Low_level.instantiate_at_synthetic_read]), so the relationship this test pins is PRICE =
   EMITTED READ: the price equals [Cost_model.analyze] over the reader statement the optimizer
   actually emits when the node is inlined at a reader spanning its axes — counts and per-leg
   exactness alike.

   - Hand-built computations through [instantiation_cost]: a reduction, a [Where] whose arm is a
   hoisted scope (exact), an arm the simplifier collapses (a bound), a shared-loop sibling the
   inliner filters, the CSE of two alpha-equivalent scopes, a broadcast operand read by every cell,
   and the lane-extract form of a packed-uniform producer. - The guard-emitting shapes the hand
   rewrite could only mark as bounds (review rounds 1-4 of staging#744): an affine write position, a
   diagonal producer, a two-component concat — guards now priced, each checked against the emitted
   read: the diagonal and pure range guards have exact op counts. The latter use an eager 0/1
   conjunction; ordinary short-circuit guards remain bounds. - [recompute_cost] through a real
   [optimize]: a two-link chain, the [`Materialize] flip's modeled price, and the proxy-vs-model
   ordering witness. - The [`Inline] flip of a node a heuristic cap materialized, through the
   production seam: priced from the computation the virtualizer's own walk stores once the node is
   not materialized, checked against the read emitted after [prefer_inline] — for a reduction, a
   packed-uniform producer over a virtual counter, and a consumer whose default setter hosts a
   footprint scratch. - Pricing is invisible: the lineage's placements and the numbering of symbols
   and scope ids generated code prints are untouched. - A flip the store refuses prices by the
   proxy. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
open Ll_test
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops
module CM = Ir.Cost_model

let mk = node_factory ~first_id:990_000_000 ~dims:[| 4 |] ()

let show name (r : CM.recompute) =
  Stdio.printf "  %-46s flops=%d%s bytes=%d%s%s\n" name r.CM.rc_flops
    (if r.CM.rc_flops_approx then " (bound)" else " (exact)")
    r.CM.rc_bytes
    (if r.CM.rc_bytes_approx then " (bound)" else " (exact)")
    (if r.CM.rc_opaque then " OPAQUE" else "")

let show_opt name = function None -> Stdio.printf "  %-46s refused\n" name | Some r -> show name r

let price ?static_indices ~self computations =
  CM.instantiation_cost ?static_indices ~self computations

(* The read the optimizer EMITS: the reader's setter statement in the optimized code, analyzed on
   its own — its loop index free, as the synthetic read's is — with the reader's and the inlined
   node's own traffic excluded, as [Cost_model] excludes the node's. *)
let emitted_read (o : LL.optimized) ~reader ~self : CM.recompute option =
  let found = ref None in
  walk o.LL.llc ~on_stmt:(function
    | LL.Set { tn; _ } as stmt when Tn.equal tn reader && Option.is_none !found ->
        found := Some stmt
    | _ -> ());
  Option.map !found ~f:(fun stmt ->
      let s = CM.analyze stmt in
      let others =
        List.filter s.CM.per_node ~f:(fun (tn, _) -> not (Tn.equal tn self || Tn.equal tn reader))
      in
      {
        CM.rc_flops = s.CM.flops;
        rc_bytes = List.sum (module Int) others ~f:(fun (_, fp) -> fp.CM.fp_read_bytes);
        rc_flops_approx = s.CM.flops_approx;
        rc_bytes_approx = List.exists others ~f:(fun (_, fp) -> fp.CM.fp_approx);
        rc_opaque = s.CM.opaque;
      })

let same_cost (a : CM.recompute option) (b : CM.recompute option) =
  match (a, b) with
  | Some a, Some b ->
      a.CM.rc_flops = b.CM.rc_flops && a.CM.rc_bytes = b.CM.rc_bytes
      && Bool.equal a.CM.rc_flops_approx b.CM.rc_flops_approx
      && Bool.equal a.CM.rc_bytes_approx b.CM.rc_bytes_approx
      && Bool.equal a.CM.rc_opaque b.CM.rc_opaque
  | _ -> false

let flops_exact = function Some r -> not r.CM.rc_flops_approx | None -> false

let () =
  Stdio.printf "== instantiation_cost on hand-built computations ==\n";
  let i = sym () and k = sym () in
  (* Reduction template as [virtual_llc] stores it — the loop over the projected symbol i is the
     root, the reduction loop over k inside: for i: for k: S[i] = S[i] + A[i][k]. As a kernel: 20
     adds. One read at a fresh index: the i loop binds away, 5 adds, A's row read (20 B), S's own
     traffic a scope local. *)
  let s = mk "S" and a = mk ~dims:[| 4; 5 |] "A" in
  let reduction =
    loop_n i 4
      (loop_n k 5 (set s [| iter i |] (add (get s [| iter i |]) (get a [| iter i; iter k |]))))
  in
  let kernel = CM.analyze reduction in
  let one = price ~self:s [ (Some [| iter i |], reduction) ] in
  show_opt "reduction, one read" one;
  p "reduction: kernel flops = read flops x projected extent, 20 bytes, exact"
    (match one with
    | Some r ->
        kernel.CM.flops = 4 * r.CM.rc_flops
        && r.CM.rc_flops = 5 && r.CM.rc_bytes = 20 && (not r.CM.rc_flops_approx)
        && not r.CM.rc_bytes_approx
    | None -> false);
  (* Where body with a hoisted scope arm: W[i] = where(K[i], { lv := 0; for k: lv += A[i][k] }, 0).
     The reduction body stays hoisted through the simplifier: select + 5 adds per read, A's row and
     K's cell read — exact, per gh-ocannl-637 Part 1. *)
  let w = mk "W" and kk = mk "K" and a4 = mk "A4" and lv = mk ~dims:[||] "lv" in
  virtualize lv;
  let scope_with ?(id = LL.get_scope lv) body =
    LL.Local_scope { id; body; orig_indices = [| iter i |]; mint = LL.Inlined_computation }
  in
  let row_sum id =
    seq
      (LL.Set_local (id, c 0.))
      (loop_n k 5 (LL.Set_local (id, add (LL.Get_local id) (get a [| iter i; iter k |]))))
  in
  let reduction_scope () =
    let id = LL.get_scope lv in
    scope_with ~id (row_sum id)
  in
  let where_body =
    loop_n i 4 (set w [| iter i |] (where_ (get kk [| iter i |]) (reduction_scope ()) (c 0.)))
  in
  let where_one = price ~self:w [ (Some [| iter i |], where_body) ] in
  show_opt "where, hoisted reduction arm" where_one;
  p "where: exact, 6 ops and 24 bytes per read"
    (match where_one with
    | Some r -> flops_exact where_one && r.CM.rc_flops = 6 && r.CM.rc_bytes = 24
    | None -> false);
  (* A single-assignment scope under the arm is what the simplifier collapses into the arm's
     expression, where it IS conditional: the op count is a guards-taken bound. *)
  let where_collapsed =
    let id = LL.get_scope lv in
    loop_n i 4
      (set w
         [| iter i |]
         (where_
            (get kk [| iter i |])
            (scope_with ~id (LL.Set_local (id, mul (get a4 [| iter i |]) (c 2.))))
            (c 0.)))
  in
  let collapsed = price ~self:w [ (Some [| iter i |], where_collapsed) ] in
  show_opt "where, arm the simplifier collapses" collapsed;
  p "where: a collapsible arm prices as the simplified, conditional form"
    (match collapsed with Some r -> r.CM.rc_flops_approx | None -> false);
  (* A shared-loop template carries a sibling setter the inliner filters out: for i: (S[i] := 3
     A[i][0]; T[i] := A[i][1] + A[i][2] + A[i][3]) read as S is one multiply. *)
  let t = mk "T" in
  let shared =
    loop_n i 4
      (seq
         (set s [| iter i |] (mul (get a [| iter i; fixed 0 |]) (c 3.)))
         (set t
            [| iter i |]
            (add
               (add (get a [| iter i; fixed 1 |]) (get a [| iter i; fixed 2 |]))
               (get a [| iter i; fixed 3 |]))))
  in
  let s_only = price ~self:s [ (Some [| iter i |], shared) ] in
  show_opt "shared loop read as S (sibling filtered)" s_only;
  p "shared loop: the sibling's ops and reads do not price"
    (match s_only with Some r -> r.CM.rc_flops = 1 && r.CM.rc_bytes = 4 | None -> false);
  (* The scalar CSE the emitted code receives: a consumer of [x + x] with a virtual [x] carries two
     alpha-equivalent scope bodies that execute once — one row sum and one add, A's row read
     once. *)
  let y = mk "Y" in
  let twice = loop_n i 4 (set y [| iter i |] (add (reduction_scope ()) (reduction_scope ()))) in
  let cse = price ~self:y [ (Some [| iter i |], twice) ] in
  show_opt "two alpha-equivalent scopes (CSE'd)" cse;
  p "alpha-equivalent scope bodies price once"
    (flops_exact cse && match cse with Some r -> r.CM.rc_flops = 6 | None -> false);
  (* An operand read at a fixed position is read by every read's instantiation: B[i] = P[0] + Q[0]
     is one add and both operands' cells per read. *)
  let b = mk "B" and pp = mk "P" and q = mk "Q" in
  let broadcast =
    loop_n i 4 (set b [| iter i |] (add (get pp [| fixed 0 |]) (get q [| fixed 0 |])))
  in
  let bc = price ~self:b [ (Some [| iter i |], broadcast) ] in
  show_opt "broadcast operands, per read" bc;
  p "broadcast operands: one add and both cells per read, exact"
    (match bc with
    | Some r -> r.CM.rc_flops = 1 && r.CM.rc_bytes = 8 && flops_exact bc && not r.CM.rc_bytes_approx
    | None -> false);
  (* A packed-uniform producer inlines as the lane-extract form (gh-509 task 4), not its vector
     store: the counter's block read at a runtime index (a dynamic read, whose bytes are the whole
     node's bound) and the lane select — its op count, index arithmetic included, is exact. *)
  let v = mk "V" and u = mk "U" in
  let vec_body =
    loop_n i 1
      (LL.Set_from_vec
         {
           tn = v;
           idcs = [| aff [ (4, i) ] 0 |];
           length = 4;
           vec_unop = Ops.Uint4x32_to_prec_uniform;
           arg = (get u [| iter i |], single);
           debug = "";
         })
  in
  let vec = price ~self:v [ (Some [| aff [ (4, i) ] 0 |], vec_body) ] in
  show_opt "packed-uniform producer (lane extract)" vec;
  p "packed-uniform producer: the lane-extract form, exact ops, dynamic-read bytes a bound"
    (match vec with
    | Some r -> (not r.CM.rc_flops_approx) && r.CM.rc_flops > 0 && r.CM.rc_bytes_approx
    | None -> false)

(* The shapes whose binding depends on the reader's index — the ones the hand rewrite marked as
   bounds. Each is optimized with a virtual producer and an identity reader spanning its axes; the
   price must be exactly what the reader statement the optimizer emits costs. *)
let () =
  Stdio.printf "== guard-emitting shapes: price = emitted read ==\n";
  let case ?expected ~name ~self ~reader ~materialized llc =
    let o = optimize ~materialized ~name llc in
    let priced = CM.recompute_cost o.LL.optimize_ctx self in
    let emitted = emitted_read o ~reader ~self in
    show_opt (name ^ ", priced") priced;
    show_opt (name ^ ", emitted") emitted;
    p (name ^ ": the producer is inlined at the reader") (known_virtual o self);
    p (name ^ ": the price is the emitted read, counts and exactness") (same_cost priced emitted);
    p_exists (name ^ ": the Materialize flip uses the exact modeled price") o.LL.flip_candidates
      ~f:(fun fc ->
        Tn.equal fc.fc_tn self
        && List.exists fc.fc_alternatives ~f:(fun fa ->
            LL.equal_reading fa.fa_flip `Materialize
            && fa.fa_modeled
            && Option.exists priced ~f:(fun r -> fa.fa_recompute_cost = r.CM.rc_flops)));
    List.iter materialized ~f:materialize;
    Tn.set_observable reader;
    let seed =
      List.mapi materialized ~f:(fun k tn ->
          ( tn,
            if Tn.equal tn reader then blank (Tn.num_elems tn)
            else Array.init (Tn.num_elems tn) ~f:(fun i -> Float.of_int (1 + (10 * k) + i)) ))
    in
    let virt = execute ~name:("cmt_guard_" ^ Tn.debug_name self) o ~seed ~read:[ reader ] in
    let mat =
      execute
        ~name:("cmt_guard_mat_" ^ Tn.debug_name self)
        (optimize ~materialized:(self :: materialized) ~name:(name ^ " materialized") llc)
        ~seed ~read:[ reader ]
    in
    p (name ^ ": executed virtual and materialized values agree") (same virt mat);
    Option.iter expected ~f:(fun values ->
        p (name ^ ": executed values match the independent reference") (same virt [ values ]));
    priced
  in
  (* Affine write position: T2[2*oh + wh] = A[oh][wh] for oh, wh < 2, read at out[x]. Unit solving
     binds wh := x - 2*oh', keeps the oh loop and range-guards it. The arm only reads; both pure
     comparisons execute eagerly, so the two comparisons, conjunction and select execute twice:
     eight exact operations per read. *)
  let t2 = mk "T2" and a = mk ~dims:[| 2; 2 |] "Aff" and out = mk "outA" in
  let oh = sym () and wh = sym () and x = sym () in
  let affine =
    seq
      (loop_n oh 2
         (loop_n wh 2 (set t2 [| aff [ (2, oh); (1, wh) ] 0 |] (get a [| iter oh; iter wh |]))))
      (loop_n x 4 (set out [| iter x |] (get t2 [| iter x |])))
  in
  let affine_price =
    case ~expected:[| 1.; 2.; 3.; 4. |] ~name:"affine position" ~self:t2 ~reader:out
      ~materialized:[ a; out ] affine
  in
  p "affine position: two eager range guards cost eight exact operations per read"
    (match affine_price with
    | Some r -> r.CM.rc_flops = 8 && flops_exact affine_price
    | None -> false);
  (* Diagonal producer: D[j, j] = A4[j] over a zeroed D, read at out[x, y]. The first occurrence of
     j binds, the second turns into the consistency guard x = y. *)
  let d = mk ~dims:[| 4; 4 |] "D" and a4 = mk "A4d" and out2 = mk ~dims:[| 4; 4 |] "outD" in
  let j = sym () and x2 = sym () and y2 = sym () in
  let diag =
    seq (zero d)
      (seq
         (loop_n j 4 (set d [| iter j; iter j |] (get a4 [| iter j |])))
         (loop_n x2 4
            (loop_n y2 4 (set out2 [| iter x2; iter y2 |] (get d [| iter x2; iter y2 |])))))
  in
  let diag_price =
    case ~name:"diagonal producer" ~self:d ~reader:out2 ~materialized:[ a4; out2 ] diag
  in
  p "diagonal producer: the consistency guard is counted, the op count exact"
    (flops_exact diag_price && match diag_price with Some r -> r.CM.rc_flops > 0 | None -> false);
  (* Two-component concat: B[i] = P[i] for i < 2, B[2 + i] = Q[i] for i < 2, read at out[x]. Every
     component replays at the read, each under its range guard and select — what the hand rewrite
     summed without (the raw setters have no op at all). The pure comparisons now execute eagerly,
     giving four exact operations after the simplifier folds known comparisons and eliminates
     multiplication by 1. *)
  let bc = mk "Bc" and pp = mk ~dims:[| 2 |] "Pc" and q = mk ~dims:[| 2 |] "Qc" in
  let out3 = mk "outC" in
  let i1 = sym () and i2 = sym () and x3 = sym () in
  let concat =
    seq
      (loop_n i1 2 (set bc [| iter i1 |] (get pp [| iter i1 |])))
      (seq
         (loop_n i2 2 (set bc [| aff [ (1, i2) ] 2 |] (get q [| iter i2 |])))
         (loop_n x3 4 (set out3 [| iter x3 |] (get bc [| iter x3 |]))))
  in
  let concat_price =
    case ~expected:[| 1.; 2.; 11.; 12. |] ~name:"two-component concat" ~self:bc ~reader:out3
      ~materialized:[ pp; q; out3 ] concat
  in
  p "two-component concat: simplified eager guards cost four exact operations per read"
    (match concat_price with
    | Some r -> r.CM.rc_flops = 4 && flops_exact concat_price
    | None -> false);
  (* Unmatched residual iterations form negative or too-large operand indices. The select must still
     gate those reads; zero init covers unwritten boundary cells. *)
  let boundary ~name ~offset ~sign ~expected =
    let producer = mk ~dims:[| 6 |] (name ^ "_producer") in
    let input = mk ~dims:[| 2; 2 |] (name ^ "_input") in
    let reader = mk ~dims:[| 6 |] (name ^ "_reader") in
    let oh = sym () and wh = sym () and x = sym () in
    let code =
      seq (zero producer)
        (seq
           (loop_n oh 2
              (loop_n wh 2
                 (set producer
                    [| aff [ (2, oh); (sign, wh) ] offset |]
                    (get input [| iter oh; iter wh |]))))
           (loop_n x 6 (set reader [| iter x |] (get producer [| iter x |]))))
    in
    ignore
      (case ~expected ~name ~self:producer ~reader ~materialized:[ input; reader ] code
        : CM.recompute option)
  in
  boundary ~name:"shifted affine boundary" ~offset:1 ~sign:1 ~expected:[| 0.; 1.; 2.; 3.; 4.; 0. |];
  boundary ~name:"reflected affine boundary" ~offset:2 ~sign:(-1)
    ~expected:[| 0.; 2.; 1.; 4.; 3.; 0. |]

(* The eager representation is reserved for the inliner's pure index guards. General [&&] and
   conditional arithmetic retain the cost model's bound contract. *)
let () =
  Stdio.printf "== eager index guards and simplifier narrowing ==\n";
  let x = sym () and target = mk "narrow_target" and input = mk "narrow_input" in
  let ip = Ops.index_prec () in
  let cmp op a b = LL.Binop (op, (LL.Embed_index a, ip), (LL.Embed_index b, ip)) in
  let lower = cmp Ops.Cmple (fixed 1) (iter x) in
  let upper = cmp Ops.Cmplt (iter x) (fixed 3) in
  let combine op = LL.Binop (op, (lower, ip), (upper, ip)) in
  let guarded cond value = set target [| iter x |] (where_ cond value (c 0.)) in
  let eager = CM.analyze (guarded (combine Ops.Mul) (get input [| iter x |])) in
  p "pure eager comparisons: two comparisons, conjunction, select = four exact ops"
    (eager.CM.flops = 4 && not eager.CM.flops_approx);
  let short = CM.analyze (guarded (combine Ops.And) (get input [| iter x |])) in
  p "ordinary short-circuit conjunction retains its operation bound" short.CM.flops_approx;
  let arithmetic = CM.analyze (guarded (combine Ops.Mul) (mul (get input [| iter x |]) (c 2.))) in
  p "eager guard does not make conditional arm arithmetic exact" arithmetic.CM.flops_approx;
  let nested cond = loop_n x 4 (if_idx cond (guarded upper (get input [| iter x |]))) in
  let narrowed = LL.simplify_llc [] (nested (combine Ops.Mul)) in
  let wheres code =
    count_scalar code ~f:(function LL.Ternop (Ops.Where, _, _, _) -> true | _ -> false)
  in
  p "true eager index conjunction narrows both comparisons in its body" (wheres narrowed = 0);
  let numeric = LL.Binop (Ops.Mul, (LL.Embed_index (iter x), ip), (upper, ip)) in
  let unchanged = LL.simplify_llc [] (nested numeric) in
  p "arbitrary numeric multiplication is not treated as a comparison conjunction"
    (wheres unchanged > 0)

(* This parameter's domain reaches a signed overflow in the RHS affine expression. The false lower
   comparison must continue to skip it. Only two one-cell arrays are allocated: the large range is a
   launch parameter domain, not a tensor extent or an executed loop. *)
let () =
  let ip = Ops.index_prec () in
  let limit, coeff =
    match ip with Ops.Int32_prec _ -> (2147483647, 1) | _ -> (Int.max_value, 2)
  in
  let parameter, bindings =
    (Idx.get_static_symbol ~static_range:limit Idx.Empty : Idx.static_symbol * Idx.unit_bindings)
  in
  let x = parameter.Idx.static_symbol in
  let input = mk ~dims:[| 1 |] "boundary_input" and out = mk ~dims:[| 1 |] "boundary_out" in
  let cmp op a b = LL.Binop (op, (LL.Embed_index a, ip), (LL.Embed_index b, ip)) in
  let lower = cmp Ops.Cmple (iter x) (fixed 5) in
  let upper = cmp Ops.Cmplt (fixed 5) (aff [ (coeff, x) ] 8) in
  let cond = LL.Binop (Ops.And, (lower, ip), (upper, ip)) in
  let code = set out [| fixed 0 |] (where_ cond (get input [| fixed 0 |]) (c 0.)) in
  let o =
    optimize ~materialized:[ input; out ] ~static_indices:(Idx.bound_symbols bindings)
      ~name:"cmt_index_boundary" code
  in
  p "near-limit domain retains a short-circuit RHS and bounded operation price"
    (count_scalar o.LL.llc ~f:(function LL.Binop (Ops.And, _, _) -> true | _ -> false) = 1
    && (CM.analyze o.LL.llc).CM.flops_approx);
  let run value name =
    execute ~bindings
      ~launch:[ (parameter, value) ]
      ~name o
      ~seed:[ (input, [| 37. |]); (out, blank 1) ]
      ~read:[ out ]
  in
  p "near-limit false lower guard skips overflowing RHS and returns the init value"
    (same (run (limit - 1) "cmt_index_boundary_high") [ [| 0. |] ]);
  p "the same boundary guard executes its valid matching arm"
    (same (run 0 "cmt_index_boundary_low") [ [| 37. |] ])

(* Pin literal conversion at signed-32 precision even in a large-model build. The invalid literal is
   inspected structurally; the preceding executed control covers skipped affine overflow. *)
let () =
  let ip = Ops.int32 in
  let parameter, bindings =
    (Idx.get_static_symbol ~static_range:9 Idx.Empty : Idx.static_symbol * Idx.unit_bindings)
  in
  let x = parameter.Idx.static_symbol in
  let input = mk ~dims:[| 1 |] "literal_input" and out = mk ~dims:[| 1 |] "literal_out" in
  let cmp op a b = LL.Binop (op, (a, ip), (b, ip)) in
  let lower = cmp Ops.Cmple (c 5.) (LL.Embed_index (iter x)) in
  let simplify bound =
    let upper = cmp Ops.Cmplt (LL.Embed_index (iter x)) (c bound) in
    let cond = LL.Binop (Ops.And, (lower, ip), (upper, ip)) in
    LL.simplify_llc (Idx.bound_symbols bindings)
      (set out
         [| fixed 0 |]
         (LL.Ternop (Ops.Where, (cond, ip), (get input [| fixed 0 |], single), (c 0., single))))
  in
  let limit = (Ir.Interval.dtype_range ip).hi in
  let safe = simplify limit and unsafe = simplify (limit +. 1.) in
  let ands code = count_scalar code ~f:(function LL.Binop (Ops.And, _, _) -> true | _ -> false) in
  p "valid boundary literal permits an exact simplified guard price"
    (ands safe = 0 && (CM.analyze safe).CM.flops = 2 && not (CM.analyze safe).CM.flops_approx);
  p "out-of-range boundary literal retains short-circuit evaluation and a price bound"
    (ands unsafe = 1 && (CM.analyze unsafe).CM.flops_approx);
  let o =
    optimize ~materialized:[ input; out ] ~static_indices:(Idx.bound_symbols bindings)
      ~name:"cmt_literal_boundary" safe
  in
  p "valid boundary literal executes its matching arm"
    (same
       (execute ~bindings
          ~launch:[ (parameter, 5) ]
          ~name:"cmt_literal_boundary" o
          ~seed:[ (input, [| 37. |]); (out, blank 1) ]
          ~read:[ out ])
       [ [| 37. |] ])

(* A chain through a real optimization: x1 = x0 + w1 (virtual), x2 = sin(x1) (virtual), out = x2 *
   x2. x2's stored computation already carries x1 inlined as a nested scope: 1 + 1 ops. *)
let () =
  Stdio.printf "== recompute_cost through optimize ==\n";
  let x0 = mk "x0" and w1 = mk "w1" and x1 = mk "x1" and x2 = mk "x2" and out = mk "out" in
  List.iter [ x0; w1; out ] ~f:materialize;
  let i = sym () and j = sym () and l = sym () in
  let llc =
    seq
      (loop_n i 4 (set x1 [| iter i |] (add (get x0 [| iter i |]) (get w1 [| iter i |]))))
      (seq
         (loop_n j 4 (set x2 [| iter j |] (LL.Unop (Ops.Sin, (get x1 [| iter j |], single)))))
         (loop_n l 4 (set out [| iter l |] (mul (get x2 [| iter l |]) (get x2 [| iter l |])))))
  in
  let o = optimize ~name:"cmt_chain" llc in
  p "chain: both links virtual" (known_virtual o x1 && known_virtual o x2);
  let cost = CM.recompute_cost o.LL.optimize_ctx in
  let c1 = cost x1 and c2 = cost x2 in
  show_opt "x1 = x0 + w1" c1;
  show_opt "x2 = sin(x1), x1 nested" c2;
  p "chain: x2's recompute is transitive (2 ops, x0 and w1's cells)"
    (match c2 with
    | Some r -> r.CM.rc_flops = 2 && r.CM.rc_bytes = 8 && flops_exact c2 && not r.CM.rc_bytes_approx
    | None -> false);
  p "chain: a materialized leaf has no stored computation to price" (Option.is_none (cost x0));
  Stdio.printf "  flip candidates:\n";
  List.iter o.LL.flip_candidates ~f:(fun fc ->
      List.iter fc.LL.fc_alternatives ~f:(fun fa ->
          Stdio.printf "    %-4s %-11s -> %-11s cost %d %s\n" (Tn.debug_name fc.LL.fc_tn)
            (LL.reading_to_string fc.LL.fc_default)
            (LL.reading_to_string fa.LL.fa_flip)
            fa.LL.fa_recompute_cost
            (if fa.LL.fa_modeled then "(modeled)" else "(proxy)")));
  let find tn r =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn tn then
          List.find fc.LL.fc_alternatives ~f:(fun fa -> LL.equal_reading fa.LL.fa_flip r)
        else None)
  in
  p "chain: x2's Materialize flip is priced by the model (2 ops x multiplicity 1, one reader)"
    (match find x2 `Materialize with
    | Some fa -> fa.LL.fa_modeled && fa.LL.fa_recompute_cost = 2
    | None -> false);
  (* Pricing is side-effect free: the lineage's placements are untouched, and no symbol is drawn
     from the counter generated code is numbered by. *)
  let placements_before = Sexp.to_string (Tn.Placements.sexp_of_t o.LL.optimize_ctx.placements) in
  let (Idx.Symbol before) = Idx.get_symbol () in
  ignore (CM.recompute_cost o.LL.optimize_ctx x2 : CM.recompute option);
  let (Idx.Symbol after) = Idx.get_symbol () in
  p "pricing draws no symbol from the global counter" (after = before + 1);
  p "pricing leaves the lineage's placements untouched"
    (String.equal placements_before
       (Sexp.to_string (Tn.Placements.sexp_of_t o.LL.optimize_ctx.placements)))

(* The ordering witness: a = p + q + r (fan-in 3, two adds) and b = sin(sin(sin(sin(p)))) (fan-in 1,
   four ops), both consumed once. The proxy ranks a above b (3 > 1), the model b above a (4 > 2). *)
let () =
  Stdio.printf "== ordering witness: proxy vs model ==\n";
  let pp = mk "p" and q = mk "q" and r = mk "r" and a = mk "a" and b = mk "b" and out = mk "out2" in
  List.iter [ pp; q; r; out ] ~f:materialize;
  let i = sym () and j = sym () and l = sym () in
  let llc =
    seq
      (loop_n i 4
         (set a
            [| iter i |]
            (add (add (get pp [| iter i |]) (get q [| iter i |])) (get r [| iter i |]))))
      (seq
         (loop_n j 4
            (set b
               [| iter j |]
               (let e x = LL.Unop (Ops.Sin, (x, single)) in
                e (e (e (e (get pp [| iter j |])))))))
         (loop_n l 4 (set out [| iter l |] (add (get a [| iter l |]) (get b [| iter l |])))))
  in
  let o = optimize ~name:"cmt_witness" llc in
  let proxy tn =
    let tr = Hashtbl.find_exn o.LL.traced_store tn in
    tr.LL.inline_reduction_extent * tr.LL.inline_fanin
  in
  let modeled tn =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn tn then
          List.find_map fc.LL.fc_alternatives ~f:(fun fa ->
              Option.some_if fa.LL.fa_modeled fa.LL.fa_recompute_cost)
        else None)
  in
  Stdio.printf "  %-4s proxy (extent x fan-in) %d  modeled %s\n" "a" (proxy a)
    (Option.value_map (modeled a) ~default:"none" ~f:Int.to_string);
  Stdio.printf "  %-4s proxy (extent x fan-in) %d  modeled %s\n" "b" (proxy b)
    (Option.value_map (modeled b) ~default:"none" ~f:Int.to_string);
  p "witness: the proxy ranks the wide sum above the deep chain" (proxy a > proxy b);
  p "witness: the model ranks the deep chain above the wide sum"
    (match (modeled a, modeled b) with Some ca, Some cb -> cb > ca | _ -> false);
  let seed =
    [
      (pp, [| 1.; 2.; 3.; 4. |]);
      (q, [| 5.; 6.; 7.; 8. |]);
      (r, [| 9.; 10.; 11.; 12. |]);
      (out, blank 4);
    ]
  in
  let got = execute ~name:"cmt_witness" o ~seed ~read:[ out ] in
  let expected =
    Array.init 4 ~f:(fun i ->
        let x = Float.of_int (i + 1) in
        x
        +. Float.of_int (i + 5)
        +. Float.of_int (i + 9)
        +. Float.sin (Float.sin (Float.sin (Float.sin x))))
  in
  p "witness: executed values match the reference" (same got [ expected ])

(* The [`Inline] flip of a node a heuristic cap materialized. Its computation was never stored (the
   cap fires before the virtualizer's walk), so the pricer re-runs that walk over the node's raw
   setter statements in a scratch copy of the lineage with the node undecided, and instantiates what
   it stores in the world that re-run leaves. The oracle is the read the optimizer emits once the
   node is preferred inline: the modeled flip cost must be that read's op count times the node's
   read multiplicity. *)
let inline_flip_vs_emitted ~name ~self ~reader ~materialized ~mult ?(prior = fun _ -> ()) llc =
  let ctx = LL.empty_optimize_ctx () in
  prior ctx;
  let o = optimize_in ctx ~materialized ~name llc in
  p (name ^ ": a heuristic cap materialized the node") (known_non_virtual o self);
  let inline_flip =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn self then
          List.find fc.LL.fc_alternatives ~f:(fun fa -> LL.equal_reading fa.LL.fa_flip `Inline)
        else None)
  in
  (match inline_flip with
  | Some fa ->
      Stdio.printf "  %-46s cost %d %s\n" (name ^ ", Inline flip") fa.LL.fa_recompute_cost
        (if fa.LL.fa_modeled then "(modeled)" else "(proxy)")
  | None -> Stdio.printf "  %-46s no Inline flip\n" name);
  let ctx_pref = LL.empty_optimize_ctx () in
  prior ctx_pref;
  LL.prefer_inline ctx_pref [ self ];
  let o_pref = optimize_in ctx_pref ~materialized ~name:(name ^ "_pref") llc in
  p (name ^ ": preferred inline, the node is inlined at the reader") (known_virtual o_pref self);
  let emitted = emitted_read o_pref ~reader ~self in
  show_opt (name ^ ", emitted read once inlined") emitted;
  p
    (name ^ ": the Inline flip is modeled and prices the emitted read's ops per instantiation")
    (match (inline_flip, emitted) with
    | Some fa, Some e ->
        fa.LL.fa_modeled && (not e.CM.rc_flops_approx)
        && fa.LL.fa_recompute_cost = mult * e.CM.rc_flops
    | _ -> false);
  o

let () =
  Stdio.printf "== the Inline flip of a cap-materialized node ==\n";
  (* R[i] = sum_k A[i][k] over k < 20: the inline-reduction cap (16) materializes it. *)
  let rr = mk "R" and a = mk ~dims:[| 4; 20 |] "Ar" and out = mk "outR" in
  let i = sym () and k = sym () and x = sym () in
  let llc =
    seq (zero rr)
      (seq
         (loop_n i 4
            (loop_n k 20
               (set rr [| iter i |] (add (get rr [| iter i |]) (get a [| iter i; iter k |])))))
         (loop_n x 4 (set out [| iter x |] (get rr [| iter x |]))))
  in
  ignore
    (inline_flip_vs_emitted ~name:"reduction cap" ~self:rr ~reader:out ~materialized:[ a; out ]
       ~mult:1 llc
      : LL.optimized);
  (* A packed-uniform producer read twice per cell (the visit cap materializes it) whose counter is
     a virtual producer: materialized, V's setter inlines the counter, and cleanup commits the
     counter virtual — yet once V is inlined, the lane-extract form reads the counter as a buffer
     and commits it materialized (review round 2). The walk re-run stores V's setter raw, as the
     store does, and the read is instantiated against the placements the walk left, where the
     counter is still undecided. *)
  let v = mk "Vp" and u = mk ~dims:[| 1 |] "Up" and wp = mk ~dims:[| 1 |] "Wp" in
  let o = mk ~dims:[| 4; 2 |] "op" in
  let i = sym () and j = sym () and x = sym () and y = sym () in
  let llc =
    seq
      (loop_n j 1 (set u [| iter j |] (add (get wp [| iter j |]) (c 1.))))
      (seq
         (loop_n i 1
            (LL.Set_from_vec
               {
                 tn = v;
                 idcs = [| aff [ (4, i) ] 0 |];
                 length = 4;
                 vec_unop = Ops.Uint4x32_to_prec_uniform;
                 arg = (get u [| iter i |], single);
                 debug = "";
               }))
         (loop_n x 4 (loop_n y 2 (set o [| iter x; iter y |] (get v [| iter x |])))))
  in
  ignore
    (inline_flip_vs_emitted ~name:"packed-uniform, virtual counter" ~self:v ~reader:o
       ~materialized:[ wp; o ] ~mult:2 llc
      : LL.optimized);
  (* A cap-materialized node sharing its captured loop with a virtual sibling it reads: for i: (P[i]
     = 2 A[i]; for k < 20: R[i] += P[i] B[i][k]). The walk re-run stores only the priced node — the
     sibling keeps the computation the routine's walk stored, so a read of R inlines one copy of P,
     as the preferred-inline routine does (review round 3). *)
  let ps = mk "Ps" and rs = mk "Rs" and a_s = mk "As2" and bs = mk ~dims:[| 4; 20 |] "Bs" in
  let os = mk "os" in
  let i = sym () and k = sym () and x = sym () in
  let llc =
    seq (zero rs)
      (seq
         (loop_n i 4
            (seq
               (set ps [| iter i |] (mul (get a_s [| iter i |]) (c 2.)))
               (loop_n k 20
                  (set rs
                     [| iter i |]
                     (add
                        (get rs [| iter i |])
                        (mul (get ps [| iter i |]) (get bs [| iter i; iter k |])))))))
         (loop_n x 4 (set os [| iter x |] (get rs [| iter x |]))))
  in
  ignore
    (inline_flip_vs_emitted ~name:"shared loop with a virtual sibling" ~self:rs ~reader:os
       ~materialized:[ a_s; bs; os ] ~mult:1 llc
      : LL.optimized);
  (* A consumer read twice per cell (the visit cap materializes it), reading an inherited virtual
     producer diagonally: by default the consumer's setter hosts the producer's footprint scratch.
     Once the consumer is inlined its setter is no longer a materialized one, the footprint decision
     retracts to ordinary inlining, and the producer's reduction is part of every read (review round
     2). *)
  let ai = mk ~dims:[| 4; 4 |] "Ai" and cc = mk "Ci" and xi = mk "Xi" in
  let oi = mk ~dims:[| 4; 2 |] "oi" in
  let producer =
    let i = sym () and j = sym () and k = sym () in
    seq (zero ai)
      (loop_n i 4
         (loop_n j 4
            (loop_n k 20
               (set ai
                  [| iter i; iter j |]
                  (add (get ai [| iter i; iter j |]) (add (tag i j) (embed k)))))))
  in
  let prior ctx =
    let o = optimize_in ctx ~name:"cmt_inherited_producer" producer in
    p "footprint: the producer routine leaves the node virtual" (known_virtual o ai)
  in
  let i = sym () and x = sym () and y = sym () in
  let llc =
    seq
      (loop_n i 4 (set cc [| iter i |] (mul (get ai [| iter i; iter i |]) (get xi [| iter i |]))))
      (loop_n x 4 (loop_n y 2 (set oi [| iter x; iter y |] (get cc [| iter x |]))))
  in
  let o =
    inline_flip_vs_emitted ~name:"consumer of a footprint-scoped producer" ~self:cc ~reader:oi
      ~materialized:[ xi; oi ] ~mult:2 ~prior llc
  in
  p "footprint: by default the consumer reads the producer through a scratch"
    (Hashtbl.existsi o.LL.traced_store ~f:(fun ~key ~data:_ ->
         String.equal key.Tn.namespace LL.footprint_namespace))

(* The flip pricer draws nothing from the counters generated code is numbered by, however it prices
   — including the walk it re-runs for a cap-materialized node: optimizing a program with such a
   candidate consumes exactly as many symbols and scope ids as optimizing it with the pricer off. *)
let () =
  Stdio.printf "== pricing is invisible to the generated code's numbering ==\n";
  let rr = mk "Rn" and a = mk ~dims:[| 4; 20 |] "An" and out = mk "outN" and lvn = mk "lvn" in
  let i = sym () and k = sym () and x = sym () in
  let llc =
    seq (zero rr)
      (seq
         (loop_n i 4
            (loop_n k 20
               (set rr [| iter i |] (add (get rr [| iter i |]) (get a [| iter i; iter k |])))))
         (loop_n x 4 (set out [| iter x |] (get rr [| iter x |]))))
  in
  let consumed () =
    (* Both runs analyze from scratch: a cache hit would skip the analysis' own minting. *)
    LL.clear_analysis_cache ();
    let (Idx.Symbol s0) = Idx.get_symbol () and { LL.scope_id = c0; _ } = LL.get_scope lvn in
    let o = optimize ~materialized:[ a; out ] ~name:"cmt_numbering" llc in
    let (Idx.Symbol s1) = Idx.get_symbol () and { LL.scope_id = c1; _ } = LL.get_scope lvn in
    (o, s1 - s0, c1 - c0)
  in
  let o, symbols, scopes = consumed () in
  let modeled =
    List.exists o.LL.flip_candidates ~f:(fun fc ->
        List.exists fc.LL.fc_alternatives ~f:(fun fa -> fa.LL.fa_modeled))
  in
  let pricer = !LL.recompute_pricer in
  (LL.recompute_pricer := fun ~static_indices:_ _ _ -> None);
  let _, symbols_off, scopes_off =
    Exn.protect ~f:consumed ~finally:(fun () -> LL.recompute_pricer := pricer)
  in
  p "numbering: the priced compile did price a candidate" modeled;
  p "numbering: pricing draws no symbol and no scope id"
    (symbols = symbols_off && scopes = scopes_off)

(* An [`Inline] flip a READER refuses (review round 4): over a zeroed F, F[0] = 2 X[0], read at F[0]
   and at F[1] by every cell of out[x] (the visit cap materializes F). The synthetic read at the
   written slice would be served, but replaying the flip meets the read at F[1], which the inliner
   cannot serve (13): the pricer instantiates at every read site the routine has, and prices the
   flip by the proxy. *)
let () =
  Stdio.printf "== an Inline flip a read site refuses ==\n";
  let f = mk ~dims:[| 2 |] "Fr" and x = mk ~dims:[| 1 |] "Xr" and out = mk ~dims:[| 2 |] "outF" in
  let j = sym () in
  let llc =
    seq (zero f)
      (seq
         (set f [| fixed 0 |] (mul (get x [| fixed 0 |]) (c 2.)))
         (loop_n j 2 (set out [| iter j |] (add (get f [| fixed 0 |]) (get f [| fixed 1 |])))))
  in
  let o = optimize ~materialized:[ x; out ] ~name:"cmt_reader_refused" llc in
  p "reader-refused flip: a cap materialized the node" (known_non_virtual o f);
  let inline_flip =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn f then
          List.find fc.LL.fc_alternatives ~f:(fun fa -> LL.equal_reading fa.LL.fa_flip `Inline)
        else None)
  in
  p "reader-refused flip: offered, and priced by the proxy rather than modeled"
    (match inline_flip with Some fa -> not fa.LL.fa_modeled | None -> false);
  let ctx = LL.empty_optimize_ctx () in
  LL.prefer_inline ctx [ f ];
  let o_pref = optimize_in ctx ~materialized:[ x; out ] ~name:"cmt_reader_refused_pref" llc in
  Stdio.printf "  preferred inline, the virtualizer's verdict: %s\n"
    (Option.value_map (rejection_code o_pref f) ~default:"none" ~f:Tn.provenance_to_string);
  p "reader-refused flip: preferred inline, the virtualizer refuses it all the same"
    (known_non_virtual o_pref f);
  (* gh-ocannl-1093: the pricer's world carries the refusal onto the alternative, and it is the
     virtualizer's own verdict. *)
  p "reader-refused flip: the alternative carries the virtualizer's verdict as its refusal"
    (match inline_flip with
    | Some fa ->
        Option.is_some fa.LL.fa_refused
        && Option.equal String.equal fa.LL.fa_refused
             (Option.map (rejection_code o_pref f) ~f:Tn.provenance_to_string)
    | None -> false)

(* An [`Inline] flip the store itself refuses: a scalar reduction S[0] = sum_i A[i] over i < 20 (the
   cap materializes it). Captured at its setter, the read of A escapes the reduction loop the
   capture leaves outside, so [virtual_llc] refuses the node however it is preferred — and so does
   the walk the pricer re-runs, which prices the flip by the proxy instead of modeling a reading
   that cannot happen (the hand rewrite priced such a flip as the whole nest per read). *)
let () =
  Stdio.printf "== an Inline flip the store refuses ==\n";
  let s = mk ~dims:[| 1 |] "Ssum"
  and a = mk ~dims:[| 20 |] "As"
  and out = mk ~dims:[| 1 |] "outS" in
  let i = sym () in
  let llc =
    seq (zero s)
      (seq
         (loop_n i 20 (set s [| fixed 0 |] (add (get s [| fixed 0 |]) (get a [| iter i |]))))
         (set out [| fixed 0 |] (get s [| fixed 0 |])))
  in
  let o = optimize ~materialized:[ a; out ] ~name:"cmt_refused" llc in
  let inline_flip =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn s then
          List.find fc.LL.fc_alternatives ~f:(fun fa -> LL.equal_reading fa.LL.fa_flip `Inline)
        else None)
  in
  p "refused flip: offered, and priced by the proxy rather than modeled"
    (match inline_flip with Some fa -> not fa.LL.fa_modeled | None -> false);
  let ctx = LL.empty_optimize_ctx () in
  LL.prefer_inline ctx [ s ];
  let o_pref = optimize_in ctx ~materialized:[ a; out ] ~name:"cmt_refused_pref" llc in
  Stdio.printf "  preferred inline, the virtualizer's verdict: %s\n"
    (Option.value_map (rejection_code o_pref s) ~default:"none" ~f:Tn.provenance_to_string);
  p "refused flip: preferred inline, the virtualizer refuses it all the same"
    (known_non_virtual o_pref s);
  p "refused flip: the alternative carries the virtualizer's verdict as its refusal"
    (match inline_flip with
    | Some fa ->
        Option.is_some fa.LL.fa_refused
        && Option.equal String.equal fa.LL.fa_refused
             (Option.map (rejection_code o_pref s) ~f:Tn.provenance_to_string)
    | None -> false)
