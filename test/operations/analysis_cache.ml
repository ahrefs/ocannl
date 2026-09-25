(* gh-560: reuse one [Low_level] analysis across sibling candidate compiles.

   Phase 1 exercises the analysis cache directly on hand-built [Ir.Low_level.t] (like
   inline_decision_replay.ml): an alpha-variant re-lowering of the same routine (fresh loop symbols,
   same tensor nodes) hits the cache and reproduces the first compile's result exactly; the same
   structure over different tensor nodes, over a different static symbol, or after mutating a static
   symbol's range, does not share an entry (tensor nodes and statics key by identity — a cache hit
   reuses the stored code verbatim).

   Phase 2 exercises the seam end-to-end: sibling [Context.compile]s of one comp share the analysis,
   and the analyze-only [Context.decision_surface] reports the same flip candidates as a capture
   compile's [lowered_transform] — without advancing the context.

   Printed facts are booleans/PASS lines so the expected output stays backend-stable. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing

(* Nodes and builders come from [Ll_test] (gh-ocannl-600, gh-ocannl-608), which carries the same set
   this file used to spell for itself — in one of the three mutually incompatible argument
   conventions the hand-built-IR tests had drifted into. *)
let mk = Ll_test.node_factory ~first_id:3000 ~dims:[| 3 |] ()
let materialize = Ll_test.materialize
let sym = Ll_test.sym
let iter = Ll_test.iter
let set s tn llsc = Ll_test.set_at tn (iter s) llsc
let get s tn = Ll_test.get tn [| iter s |]
let mul = Ll_test.mul
let c = Ll_test.c
let loop s body = Ll_test.loop ~upto:2 s body
let seq = Ll_test.seq

open Verdict.Claims

(* One "lowering" of the two-consumer routine: fresh loop symbols each call, tensor nodes fixed by
   the caller — the shape sibling candidate compiles produce. *)
let build_llc (x, prod, oa, ob) =
  let i = sym () and j = sym () and k = sym () in
  let producer = loop i (set i prod (mul (get i x) (c 2.))) in
  let use_a = loop j (set j oa (get j prod)) in
  let use_b = loop k (set k ob (get k prod)) in
  seq producer (seq use_a use_b)

let optimize llc ~static_indices = Ll_test.optimize ~static_indices ~name:"ac_test" llc

(* Runs [f] and reports whether the analysis cache registered a hit (and no miss) for it. *)
let delta f =
  let h0, m0 = LL.analysis_cache_stats () in
  let result = f () in
  let h1, m1 = LL.analysis_cache_stats () in
  (result, `Hits (h1 - h0), `Misses (m1 - m0))

let phase1 () =
  let mk_tns () =
    let x = mk "ac_x" and prod = mk "ac_prod" and oa = mk "ac_oa" and ob = mk "ac_ob" in
    materialize x;
    materialize oa;
    materialize ob;
    (x, prod, oa, ob)
  in
  let tns = mk_tns () in
  let o1, h1, m1 = delta (fun () -> optimize (build_llc tns) ~static_indices:[]) in
  p "first compile misses" (Poly.equal (h1, m1) (`Hits 0, `Misses 1));
  let o2, h2, m2 = delta (fun () -> optimize (build_llc tns) ~static_indices:[]) in
  p "alpha-variant sibling lowering hits" (Poly.equal (h2, m2) (`Hits 1, `Misses 0));
  p "hit reproduces the first result exactly"
    (Sexp.equal (LL.sexp_of_t o1.LL.llc) (LL.sexp_of_t o2.LL.llc));
  let _o3, h3, m3 = delta (fun () -> optimize (build_llc (mk_tns ())) ~static_indices:[]) in
  p "same structure over different tensor nodes misses" (Poly.equal (h3, m3) (`Hits 0, `Misses 1));
  (* Static symbols key by identity and by their (mutable) range facts. *)
  let mk_static () =
    {
      Idx.static_symbol = sym ();
      static_range = Some 3;
      used_as_extent = false;
      used_as_slice = false;
    }
  in
  let stns = mk_tns () in
  let sx, _, soa, _ = stns in
  let static = mk_static () in
  let static_llc (ss : Idx.static_symbol) =
    let s = ss.Idx.static_symbol in
    let i = sym () in
    loop i (set i soa (mul (get i sx) (LL.Get (sx, [| iter s |]))))
  in
  let _, h4, m4 = delta (fun () -> optimize (static_llc static) ~static_indices:[ static ]) in
  p "static-indexed first compile misses" (Poly.equal (h4, m4) (`Hits 0, `Misses 1));
  let _, h5, m5 = delta (fun () -> optimize (static_llc static) ~static_indices:[ static ]) in
  p "same static symbol hits" (Poly.equal (h5, m5) (`Hits 1, `Misses 0));
  let static' = mk_static () in
  let _, h6, m6 = delta (fun () -> optimize (static_llc static') ~static_indices:[ static' ]) in
  p "a fresh static symbol misses" (Poly.equal (h6, m6) (`Hits 0, `Misses 1));
  static'.Idx.static_range <- Some 2;
  let _, h7, m7 = delta (fun () -> optimize (static_llc static') ~static_indices:[ static' ]) in
  p "mutating the static's range misses" (Poly.equal (h7, m7) (`Hits 0, `Misses 1))

open Ocannl
open Ocannl.Operation.DSL_modules

let phase2 () =
  Utils.settings.fixed_state_for_init <- Some 42;
  Tensor.unsafe_reinitialize ();
  (* Matmul-plus-relu: the matmul intermediate is policy-virtual (it inlines into the pointwise
     consumer), so the decision surface reports at least one [`Materialize] flip candidate (as in
     inline_flip_tune.ml). *)
  let n = 8 in
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ac2_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "ac2_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = Train.forward t2 in
  let ctx = Context.auto () in
  let (ctx, _routine), _, _ = delta (fun () -> Context.compile ctx comp Idx.Empty) in
  let captured = ref [] in
  let (_ctx2, _routine2), h2, m2 =
    delta (fun () ->
        Context.compile
          ~lowered_transform:(fun o ->
            captured := o.LL.flip_candidates;
            [ o ])
          ctx comp Idx.Empty)
  in
  p "sibling Context.compile shares the analysis" (Poly.equal (h2, m2) (`Hits 1, `Misses 0));
  let surface, h3, m3 = delta (fun () -> Context.decision_surface ctx comp Idx.Empty) in
  p "decision_surface shares the analysis too" (Poly.equal (h3, m3) (`Hits 1, `Misses 0));
  let render fc =
    ( fc.LL.fc_tn.Tn.id,
      (match fc.LL.fc_flip with
      | `Inline -> "inline"
      | `Materialize -> "materialize"
      | `Footprint -> "footprint"),
      fc.LL.fc_recompute_cost )
  in
  p "decision_surface matches the capture compile's flip candidates"
    (Poly.equal (List.map surface ~f:render) (List.map !captured ~f:render));
  p "decision_surface reports a nonempty surface here" (not (List.is_empty surface))

let () =
  phase1 ();
  phase2 ()
