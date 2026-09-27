(* gh-555: inlining as a first-class, searchable schedule decision.

   Phase 1 exercises the analysis/specialization split directly on hand-built [Ir.Low_level.t] (like
   virtual_shared_loop.ml): [analyze_proc] runs once, [specialize_proc] replays per candidate over
   the shared analysis. Pinned: - the default policy materializes a complex producer read twice (the
   [virtualize_max_visits] cap); - an [inline_preferences] entry (the Inline half of the decision
   vector) exempts the producer from the cap, and the virtualizer inlines it — same analysis,
   different decisions, hermetic results; - replaying with a fresh default context reproduces the
   first result exactly (candidate hermeticity of the shared traced store).

   Phase 2 exercises the [Context.decide_inline] API end-to-end on a tensor graph: executed values
   must agree between the default (materialized intermediate) and the inline-forced compile.

   Printed facts are booleans/PASS lines so the expected output stays backend-stable. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode

(* Nodes, builders, traversal and counters come from [Ll_test] (gh-ocannl-600, gh-ocannl-608): this
   file carried a verbatim copy of the exhaustive [Low_level.t] / [scalar_t] traversal pair and the
   [count_set] / [count_get] derived from it, which is the artifact that has to be edited for every
   new IR constructor and miscounts silently when it is not. *)
let mk = Ll_test.node_factory ~first_id:2000 ~dims:[| 3 |] ()
let materialize = Ll_test.materialize
let sym = Ll_test.sym
let iter = Ll_test.iter
let set s tn llsc = Ll_test.set_at tn (iter s) llsc
let get s tn = Ll_test.get tn [| iter s |]
let mul = Ll_test.mul
let c = Ll_test.c
let loop s body = Ll_test.loop ~upto:2 s body
let seq = Ll_test.seq
let count_set = Ll_test.count_set
let count_get = Ll_test.count_get

open Verdict.Claims

(* A reversed read [prod[2-j]]: position differs from the enclosing statement's write position, so
   unlike a plain copy it is not read-modify-write-exempt and counts toward the multiplicity. *)
let get_rev s tn : LL.scalar_t = Ll_test.get tn [| Ll_test.aff [ (-1, s) ] 2 |]

let phase1 () =
  let x = mk "x" and prod = mk "prod" and oa = mk "oa" and ob = mk "ob" in
  materialize x;
  materialize oa;
  materialize ob;
  let i = sym () and j = sym () and k = sym () in
  (* [prod] is complex (reads the materialized [x]) and read by two overlapping consumer sites: the
     default policy trips the visit cap (multiplicity 2 > virtualize_max_visits = 1) and
     materializes it. *)
  let producer = loop i (set i prod (mul (get i x) (c 2.))) in
  let use_a = loop j (set j oa (get_rev j prod)) in
  let use_b = loop k (set k ob (get_rev k prod)) in
  let llc = seq producer (seq use_a use_b) in
  assert (LL.virtualize_settings.max_visits = 1);
  let an = LL.analyze_proc [] llc in
  let o_default = LL.specialize_proc (LL.empty_optimize_ctx ()) an in
  let ctx_inline = LL.empty_optimize_ctx () in
  Hash_set.add ctx_inline.LL.inline_preferences prod;
  let o_inline = LL.specialize_proc ctx_inline an in
  let o_replay = LL.specialize_proc (LL.empty_optimize_ctx ()) an in
  let plc o = o.LL.optimize_ctx.LL.placements in
  p "default: producer materialized by the visit cap"
    (Tn.Placements.known_non_virtual (plc o_default) prod);
  p "default: producer setter survives, both consumers read it"
    (count_set o_default prod = 1 && count_get o_default prod = 2);
  p "inline-preferred: producer virtual" (Tn.Placements.known_virtual (plc o_inline) prod);
  p "inline-preferred: setter dropped, no residual reads"
    (count_set o_inline prod = 0 && count_get o_inline prod = 0);
  p "replay over shared analysis reproduces the default result"
    (Sexp.equal (LL.sexp_of_t o_default.LL.llc) (LL.sexp_of_t o_replay.LL.llc));
  p "hermeticity: inline candidate did not leak into the default traced store"
    (Tn.Placements.known_non_virtual (plc o_replay) prod);
  (* The searchable decision surface (gh-555): each arm reports the producer as flippable in the
     opposite direction, with the recompute-cost bound (extent 1 × multiplicity 2). *)
  let find_flip o tn =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        match fc.LL.fc_alternatives with
        | [ fa ] when fc.LL.fc_tn.Tn.id = tn.Tn.id -> Some (fa.LL.fa_flip, fa.LL.fa_recompute_cost)
        | _ -> None)
  in
  p "default arm reports the producer as an Inline flip of cost 2"
    (match find_flip o_default prod with Some (`Inline, 2) -> true | _ -> false);
  p "inline arm reports the producer as a Materialize flip"
    (match find_flip o_inline prod with Some (`Materialize, 2) -> true | _ -> false)

open Ocannl
open Ocannl.Operation.DSL_modules

let phase2 () =
  (* [t] is complex (reads [x]) and read by two reduction sites (reductions read at positions
     differing from their write position, so unlike copies they count toward the multiplicity):
     default-materialized by the visit cap; [Context.decide_inline] keeps it virtual. *)
  let build () =
    Utils.settings.fixed_state_for_init <- Some 42;
    Tensor.unsafe_reinitialize ();
    let x =
      NTDSL.init ~l:"idr_x" ~prec:Ir.Ops.single ~o:[ 4 ]
        ~f:(fun idcs -> Float.of_int (idcs.(0) + 1) *. 0.5)
        ()
    in
    let%op t = x *. x in
    let%op y = t ++ "i=>0" + (t ++ "i=>0") in
    Train.set_materialized y.Tensor.value;
    Tn.set_observable y.Tensor.value;
    (t, y)
  in
  let run ~inline =
    let t, y = build () in
    let ctx = Context.auto () in
    let ctx = if inline then Context.decide_inline ctx [ t.Tensor.value ] else ctx in
    let ctx, routine = Context.compile ctx (Train.forward y) Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    (Context.get_values ctx y.Tensor.value, Context.placements ctx, t.Tensor.value)
  in
  let yv_default, plc_default, t_default = run ~inline:false in
  let yv_inline, plc_inline, t_inline = run ~inline:true in
  p "phase2 default: intermediate is non-virtual"
    (Tn.Placements.known_non_virtual plc_default t_default);
  p "phase2 inline-forced: intermediate is virtual"
    (Tn.Placements.known_virtual plc_inline t_inline);
  p "phase2 parity: same result length" (Array.length yv_default = Array.length yv_inline);
  let max_diff =
    Array.foldi yv_default ~init:0.0 ~f:(fun i acc v ->
        Float.max acc (Float.abs (v -. yv_inline.(i))))
  in
  Verdict.pass_fail "phase2 parity"
    Float.(max_diff <= 1e-6)
    ~detail:(fun () -> Printf.sprintf "max diff %.8f" max_diff)

let () =
  phase1 ();
  phase2 ()
