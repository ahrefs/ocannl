(* Regression test for gh-ocannl-133 Stage B: virtualize injective affine producers (multi-symbol
   affine LHS indices).

   High-level lowering never produces these shapes directly in a controllable way, so -- like
   [virtual_shared_loop.ml] -- the cases are built directly as [Ir.Low_level.t] and run through
   [Ir.Low_level.optimize] (the same visit_llc -> virtual_llc -> cleanup -> simplify pipeline the
   backends use). We assert structurally on the optimized form (which producers virtualize, that no
   intermediate array read/setter survives, and which stay materialized).

   Cases:
   - structural affine match: producer [2*oh+wh] consumed at the same affine structure inlines with
     no intermediate buffer;
   - unit-coefficient solving at a plain iterator: producer [2*oh+wh] consumed at [t] inlines (the
     residual [oh] loop is kept and range-guarded);
   - triangular [(s1, s1+s2)]: unit-coefficient solving after pinning s1;
   - non-injective [i+j] (both ranges > 1) stays non-virtual, preserving a producer array read;
   - Stage A diagonal [i;i] still virtualizes (no regression).

   End-to-end numeric correctness of the injective-scatter lowering payoff and affine inlining is
   covered by the high-level [test/einsum/test_max_pool2d.ml] suite. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing

let single = Ir.Ops.single
let next_id = ref 2000

let mk ?(dims = [| 6 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ] ~unpadded_dims:(lazy dims)
    ~padding:(lazy None) ()

let materialize tn = Tn.update_memory_mode tn Tn.Materialized 99

(* --- low-level builders --- *)
let sym () = Idx.get_symbol ()
let iter s = Idx.Iterator s
let aff terms offset = Idx.Affine { symbols = terms; offset }
let set tn idcs llsc : LL.t = LL.Set { tn; idcs; llsc; debug = "" }
let get tn idcs : LL.scalar_t = LL.Get (tn, idcs)
let c x : LL.scalar_t = LL.Constant x
let zero tn : LL.t = LL.Zero_out tn

(* [from_ = 0, to_ = n - 1] gives a loop of width [n]. *)
let loop_r s n body : LL.t = LL.For_loop { index = s; from_ = 0; to_ = n - 1; body; trace_it = true }
let seq a b : LL.t = LL.Seq (a, b)

let optimize llc : LL.optimized =
  let ctx : LL.optimize_ctx = { computations = Hashtbl.create (module Tn) } in
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"virtual_affine" [] llc

(* --- structural probes on the optimized form --- *)
let rec walk_t ~on_set ~on_get (llc : LL.t) =
  match llc with
  | LL.Noop | LL.Declare_local _ | LL.Comment _ | LL.Staged_compilation _ -> ()
  | LL.Seq (a, b) ->
      walk_t ~on_set ~on_get a;
      walk_t ~on_set ~on_get b
  | LL.For_loop { body; _ } -> walk_t ~on_set ~on_get body
  | LL.Zero_out tn -> on_set tn
  | LL.Set { tn; llsc; _ } ->
      on_set tn;
      walk_s ~on_set ~on_get llsc
  | LL.Set_from_vec { tn; arg = s, _; _ } ->
      on_set tn;
      walk_s ~on_set ~on_get s
  | LL.Set_local (_, s) -> walk_s ~on_set ~on_get s

and walk_s ~on_set ~on_get (s : LL.scalar_t) =
  match s with
  | LL.Constant _ | LL.Constant_bits _ | LL.Get_local _ | LL.Embed_index _ | LL.Get_merge_buffer _ ->
      ()
  | LL.Get (tn, _) -> on_get tn
  | LL.Get_dynamic { tn; dyn_value = v, _; _ } ->
      on_get tn;
      walk_s ~on_set ~on_get v
  | LL.Local_scope { body; _ } -> walk_t ~on_set ~on_get body
  | LL.Ternop (_, (a, _), (b, _), (d, _)) ->
      walk_s ~on_set ~on_get a;
      walk_s ~on_set ~on_get b;
      walk_s ~on_set ~on_get d
  | LL.Binop (_, (a, _), (b, _)) ->
      walk_s ~on_set ~on_get a;
      walk_s ~on_set ~on_get b
  | LL.Unop (_, (a, _)) -> walk_s ~on_set ~on_get a

let count_set (o : LL.optimized) tn =
  let n = ref 0 in
  walk_t ~on_set:(fun t -> if t.Tn.id = tn.Tn.id then Int.incr n) ~on_get:(fun _ -> ()) o.llc;
  !n

let count_get (o : LL.optimized) tn =
  let n = ref 0 in
  walk_t ~on_set:(fun _ -> ()) ~on_get:(fun t -> if t.Tn.id = tn.Tn.id then Int.incr n) o.llc;
  !n

(* Count [Where] guards and [Cmplt] comparisons in the optimized form: range guards emitted by
   unit-coefficient solving render as [Where (And (Cmplt _, Cmplt _), value, Get_local)], whereas a
   pure structural affine match introduces neither. *)
let count_guard_ops (o : LL.optimized) =
  let wh = ref 0 and lt = ref 0 in
  let rec t (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        t a;
        t b
    | LL.For_loop { body; _ } -> t body
    | LL.Set { llsc; _ } -> s llsc
    | LL.Set_from_vec { arg = a, _; _ } -> s a
    | LL.Set_local (_, x) -> s x
    | _ -> ()
  and s (sc : LL.scalar_t) =
    match sc with
    | LL.Ternop (op, (a, _), (b, _), (d, _)) ->
        (match op with Ops.Where -> Int.incr wh | _ -> ());
        s a;
        s b;
        s d
    | LL.Binop (op, (a, _), (b, _)) ->
        (match op with Ops.Cmplt -> Int.incr lt | _ -> ());
        s a;
        s b
    | LL.Unop (_, (a, _)) -> s a
    | LL.Local_scope { body; _ } -> t body
    | _ -> ()
  in
  t o.LL.llc;
  (!wh, !lt)

let p name b = Stdio.printf "%s: %b\n" name b

(* === Case 1: structural affine match === *)
let case_structural_match () =
  let tgt = mk ~dims:[| 6 |] "smatch" and out = mk ~dims:[| 6 |] "out1" in
  materialize out;
  let oh = sym () and wh = sym () and a = sym () and b = sym () in
  (* Injective + surjective scatter over [0, 6): no zero-init (lowering payoff). *)
  let prod = loop_r oh 3 (loop_r wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (c 5.))) in
  let cons =
    loop_r a 3
      (loop_r b 2 (set out [| aff [ (2, a); (1, b) ] 0 |] (get tgt [| aff [ (2, a); (1, b) ] 0 |])))
  in
  let o = optimize (seq prod cons) in
  p "structural-match producer virtual" (Tn.known_virtual tgt);
  p "structural-match producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "structural-match producer setter dropped" (count_set o tgt = 0);
  p "structural-match consumer setter kept" (count_set o out = 1);
  (* Same affine structure on both sides: bound pairwise, no range/equality guard. *)
  let wh, lt = count_guard_ops o in
  p "structural-match has no guard ops" (wh = 0 && lt = 0)

(* === Case 2: unit-coefficient solving at a plain iterator === *)
let case_unit_solve_plain () =
  let tgt = mk ~dims:[| 6 |] "usolve" and out = mk ~dims:[| 6 |] "out2" in
  materialize out;
  let oh = sym () and wh = sym () and t = sym () in
  let prod = loop_r oh 3 (loop_r wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (c 7.))) in
  let cons = loop_r t 6 (set out [| iter t |] (get tgt [| iter t |])) in
  let o = optimize (seq prod cons) in
  p "unit-solve(plain) producer virtual" (Tn.known_virtual tgt);
  p "unit-solve(plain) producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "unit-solve(plain) producer setter dropped" (count_set o tgt = 0);
  p "unit-solve(plain) consumer setter kept" (count_set o out = 1);
  (* Solving [wh = t - 2*oh] keeps the [oh] loop and range-guards [0 <= t-2*oh < 2]: a [Where] over an
     [And] of two [Cmplt] bounds. *)
  let wh, lt = count_guard_ops o in
  p "unit-solve(plain) emits a range guard (Where + 2 Cmplt)" (wh >= 1 && lt >= 2)

(* === Case 3: triangular (s1, s1 + s2), unit-coefficient solving after pinning s1 === *)
let case_triangular () =
  let tgt = mk ~dims:[| 3; 4 |] "tri" and out = mk ~dims:[| 3; 4 |] "out3" in
  materialize out;
  let s1 = sym () and s2 = sym () and a = sym () and b = sym () in
  (* Triangular map is injective but not surjective, so it carries a zero-init. *)
  let prod =
    seq (zero tgt) (loop_r s1 3 (loop_r s2 2 (set tgt [| iter s1; aff [ (1, s1); (1, s2) ] 0 |] (c 9.))))
  in
  let cons =
    loop_r a 3 (loop_r b 4 (set out [| iter a; iter b |] (get tgt [| iter a; iter b |])))
  in
  let o = optimize (seq prod cons) in
  p "triangular producer virtual" (Tn.known_virtual tgt);
  p "triangular producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "triangular consumer setter kept" (count_set o out = 1);
  (* s2 = b - a is solved and range-guarded [0 <= b-a < 2]. *)
  let wh, lt = count_guard_ops o in
  p "triangular emits a range guard (Where + 2 Cmplt)" (wh >= 1 && lt >= 2)

(* === Case 4: non-injective i+j (both ranges > 1) stays non-virtual === *)
let case_noninjective () =
  let tgt = mk ~dims:[| 5 |] "ni" and out = mk ~dims:[| 5 |] "out4" in
  materialize out;
  let i = sym () and j = sym () and a = sym () and b = sym () in
  let prod = loop_r i 3 (loop_r j 3 (set tgt [| aff [ (1, i); (1, j) ] 0 |] (c 1.))) in
  let cons =
    loop_r a 3 (loop_r b 3 (set out [| aff [ (1, a); (1, b) ] 0 |] (get tgt [| aff [ (1, a); (1, b) ] 0 |])))
  in
  let o = optimize (seq prod cons) in
  (* i+j with both ranges > 1 is not injective: the dropped producer loops fold over a fiber, so the
     producer must stay materialized (the reason is the injectivity soundness line). *)
  p "non-injective producer stays non-virtual" (Tn.known_non_virtual tgt);
  p "non-injective producer array read preserved" (count_get o tgt >= 1)

(* === Case 5: Stage A diagonal [i;i] still virtualizes (no regression) === *)
let case_stage_a_diagonal () =
  let d = mk ~dims:[| 3; 3 |] "diag" and out = mk ~dims:[| 3; 3 |] "out5" in
  materialize out;
  let i = sym () and a = sym () and b = sym () in
  let prod = seq (zero d) (loop_r i 3 (set d [| iter i; iter i |] (c 4.))) in
  let cons =
    loop_r a 3 (loop_r b 3 (set out [| iter a; iter b |] (get d [| iter a; iter b |])))
  in
  let o = optimize (seq prod cons) in
  p "stage-a diagonal producer virtual" (Tn.known_virtual d);
  p "stage-a diagonal inlined (no array reads survive)" (count_get o d = 0)

let () =
  case_structural_match ();
  case_unit_solve_plain ();
  case_triangular ();
  case_noninjective ();
  case_stage_a_diagonal ();
  Stdio.printf "%!"
