(* gh-ocannl-637 Part 2: [Ir.Cost_model]'s account of ONE inlined computation, the recompute cost
   the virtualizer used to price with a traced proxy (reduction extent × read multiplicity ×
   transitive fan-in).

   - [template_cost] on hand-built templates, checked against the counts of the same body analyzed
   as a kernel: a reduction body (the projected loop collapses, the reduction loop is the trip
   count), a [Where] body whose arm is a hoisted scope (exact, per Part 1), and a shared-loop body
   whose sibling setter instantiation drops. - [recompute_cost] through a real [optimize]: a
   two-link chain prices transitively, and the [`Materialize] flip candidates carry the modeled cost
   ([fc_modeled]). - The ordering witness: two virtual nodes the proxy and the model rank in
   opposite orders — a three-operand sum (fan-in 3, two ops) against a four-deep unary chain (fan-in
   1, four ops). - [producer_cost] on the setter nest of a node the fan-in cap materialized (the
   [`Inline] flip), where no template was stored: the chain prefix's adds per cell. *)

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
  Stdio.printf "  %-44s flops=%d bytes=%d%s%s\n" name r.CM.rc_flops r.CM.rc_bytes
    (if r.CM.rc_approx then " approx" else " exact")
    (if r.CM.rc_opaque then " OPAQUE" else "")

let () =
  Stdio.printf "== template_cost on hand-built templates ==\n";
  let i = sym () and k = sym () in
  (* Reduction template as [virtual_llc] stores it — the loop over the projected symbol i is the
     root, the reduction loop over k inside: for i: for k: S[i] = S[i] + A[i][k]. As a kernel: 20
     adds, A rd 80 B, S rd/wr 16 B. As one instantiation at i: the i loop collapses, 5 adds, A rd 20
     B, S's own traffic excluded. *)
  let s = mk "S" and a = mk ~dims:[| 4; 5 |] "A" in
  let reduction =
    loop_n i 4
      (loop_n k 5 (set s [| iter i |] (add (get s [| iter i |]) (get a [| iter i; iter k |]))))
  in
  let kernel = CM.analyze reduction in
  let one = CM.template_cost ~self:s ~at:[| iter i |] reduction in
  show "reduction body, one instantiation" one;
  p "reduction: kernel flops = instantiation flops x projected extent"
    (kernel.CM.flops = 4 * one.CM.rc_flops && one.CM.rc_flops = 5);
  p "reduction: bytes are the operand's cells of one instantiation, self excluded"
    (one.CM.rc_bytes = 20 && not one.CM.rc_approx);
  (* Without [~at] nothing collapses: the whole nest is the instantiation. *)
  let whole = CM.template_cost ~self:s reduction in
  p "reduction: no index vector, no collapse" (whole.CM.rc_flops = kernel.CM.flops);
  (* Where body with a hoisted scope arm: W[i] = where(K[i], { lv := A4[i] * 2 }, 0). Exact by Part
     1 — 2 ops per instantiation, A4's cell read. *)
  let w = mk "W" and kk = mk "K" and a4 = mk "A4" and lv = mk ~dims:[||] "lv" in
  virtualize lv;
  let id = LL.get_scope lv in
  let where_body =
    loop_n i 4
      (set w
         [| iter i |]
         (where_
            (get kk [| iter i |])
            (LL.Local_scope
               {
                 id;
                 body = LL.Set_local (id, mul (get a4 [| iter i |]) (c 2.));
                 orig_indices = [| iter i |];
                 mint = LL.Inlined_computation;
               })
            (c 0.)))
  in
  let where_one = CM.template_cost ~self:w ~at:[| iter i |] where_body in
  show "where body, hoisted arm" where_one;
  p "where: exact, 2 ops and 8 bytes per instantiation"
    ((not where_one.CM.rc_approx) && where_one.CM.rc_flops = 2 && where_one.CM.rc_bytes = 8);
  (* A shared-loop template carries a sibling setter that instantiation filters out: for i: (S[i] =
     A[i][0] * 3; T[i] = A[i][1] + A[i][2] + A[i][3]) priced for S is one multiply. *)
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
  let s_only = CM.template_cost ~self:s ~at:[| iter i |] shared in
  show "shared-loop body priced for S (sibling dropped)" s_only;
  p "shared loop: the sibling's ops and reads do not price"
    (s_only.CM.rc_flops = 1 && s_only.CM.rc_bytes = 4)

(* A chain through a real optimization: x1 = x0 + w1 (virtual), x2 = sin(x1) (virtual), out = x2 *
   x2. [recompute_cost] of x2 expands x1's template: 1 + 1 ops, x0 and w1 read. *)
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
  let show_opt name = function
    | None -> Stdio.printf "  %-44s none\n" name
    | Some r -> show name r
  in
  let c1 = cost x1 and c2 = cost x2 in
  show_opt "x1 = x0 + w1" c1;
  show_opt "x2 = sin(x1), x1 expanded" c2;
  p "chain: x2's recompute is transitive (2 ops, x0 and w1's cells)"
    (match c2 with
    | Some r -> r.CM.rc_flops = 2 && r.CM.rc_bytes = 8 && not r.CM.rc_approx
    | None -> false);
  p "chain: a materialized leaf has no template cost" (Option.is_none (cost x0));
  Stdio.printf "  flip candidates:\n";
  List.iter o.LL.flip_candidates ~f:(fun fc ->
      Stdio.printf "    %-11s %-4s cost %d %s\n"
        (match fc.LL.fc_flip with `Materialize -> "materialize" | `Inline -> "inline")
        (Tn.debug_name fc.LL.fc_tn) fc.LL.fc_recompute_cost
        (if fc.LL.fc_modeled then "(modeled)" else "(proxy)"));
  let find tn =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn tn then Some fc else None)
  in
  p "chain: x2's Materialize flip is priced by the model (2 ops x multiplicity 1, one reader)"
    (match find x2 with
    | Some fc -> fc.LL.fc_modeled && fc.LL.fc_recompute_cost = 2
    | None -> false)

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
        if Tn.equal fc.LL.fc_tn tn && fc.LL.fc_modeled then Some fc.LL.fc_recompute_cost else None)
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

(* [producer_cost]: the setter nest of a materialized node in optimized code, per written cell. The
   chain x1..x3 with x3 consumed: materializing x3 by declaration keeps x1, x2 inlined into its
   setter, so its per-cell cost is the three adds of the prefix. *)
let () =
  Stdio.printf "== producer_cost on a materialized setter nest ==\n";
  let x0 = mk "y0" and w1 = mk "v1" and w2 = mk "v2" and w3 = mk "v3" in
  let x1 = mk "y1" and x2 = mk "y2" and x3 = mk "y3" and out = mk "out3" in
  List.iter [ x0; w1; w2; w3; x3; out ] ~f:materialize;
  let syms = Array.init 4 ~f:(fun _ -> sym ()) in
  let link tn prev w s =
    loop_n s 4 (set tn [| iter s |] (add (get prev [| iter s |]) (get w [| iter s |])))
  in
  let llc =
    seq
      (link x1 x0 w1 syms.(0))
      (seq
         (link x2 x1 w2 syms.(1))
         (seq
            (link x3 x2 w3 syms.(2))
            (loop_n syms.(3) 4 (set out [| iter syms.(3) |] (get x3 [| iter syms.(3) |])))))
  in
  let o = optimize ~name:"cmt_producer" llc in
  p "producer: x1 and x2 inlined into x3's setter" (known_virtual o x1 && known_virtual o x2);
  (match CM.producer_cost ~self:x3 o.LL.llc with
  | None -> Stdio.printf "  none\n"
  | Some r -> show "x3's setter nest, per cell" r);
  p "producer: three adds and four leaf cells per written cell"
    (match CM.producer_cost ~self:x3 o.LL.llc with
    | Some r -> r.CM.rc_flops = 3 && r.CM.rc_bytes = 16 && not r.CM.rc_approx
    | None -> false);
  p "producer: a node the code never sets has no producer cost"
    (Option.is_none (CM.producer_cost ~self:x1 o.LL.llc))
