(* Regression test for gh-ocannl-134: multiple virtual tensors sharing one traced for-loop.

   High-level lowering never places two distinct tensors in the same for-loop (each assignment
   lowers to its own [loop_over_dims]), so these cases are built directly as [Ir.Low_level.t] and
   run through [Ir.Low_level.optimize] -- the same pipeline (visit_llc -> virtual_llc ->
   cleanup_virtual_llc -> simplify -> CSE -> hoist) the backends use. We assert structurally on the
   optimized form and on the resulting [traced_array]/memory-mode facts, which precisely pin the
   #134 invariants:

   - shared loop symbols no longer force [is_complex]; - each candidate tensor in a shared loop gets
   its own stored computation and inlines downstream; - a surviving (materialized) sibling setter
   inlines a virtualized provider from the same loop; - a forward virtual->virtual chain is fully
   inlined, leaving no read of a dropped virtual node; - a reverse/read-before-write sibling read
   stays materialized (existing safety mechanism); - cleanup keeps non-virtual residual setters
   instead of dropping the whole loop.

   End-to-end numeric correctness of virtual inlining is covered by the existing suite that still
   passes (test_cse, test_block_tensor, test_concat_graph, primitive_ops). *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops

let single = Ir.Ops.single
let next_id = ref 1000

let mk ?(dims = [| 3 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

let materialize tn = Tn.update_memory_mode tn Tn.Materialized 99

(* --- low-level builders --- *)
let sym () = Ir.Indexing.get_symbol ()
let iter s = Ir.Indexing.Iterator s
let set s tn llsc : LL.t = LL.Set { tn; idcs = [| iter s |]; llsc; debug = "" }
let get s tn : LL.scalar_t = LL.Get (tn, [| iter s |])
let add a b : LL.scalar_t = LL.Binop (Ops.Add, (a, single), (b, single))
let mul a b : LL.scalar_t = LL.Binop (Ops.Mul, (a, single), (b, single))
let c x : LL.scalar_t = LL.Constant x
let loop s body : LL.t = LL.For_loop { index = s; from_ = 0; to_ = 2; body; trace_it = true }
let seq a b : LL.t = LL.Seq (a, b)

let optimize llc : LL.optimized =
  let ctx : LL.optimize_ctx = { computations = Hashtbl.create (module Tn) } in
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"shared_loop" [] llc

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
  | LL.Constant _ | LL.Constant_bits _ | LL.Get_local _ | LL.Embed_index _ | LL.Get_merge_buffer _
    ->
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

let is_complex (o : LL.optimized) tn = (Hashtbl.find_exn o.traced_store tn).LL.is_complex
let p name b = Stdio.printf "%s: %b\n" name b

(* === Case 1: two independent virtual siblings in one loop, read downstream === *)
let case_independent () =
  let a = mk "a" and b = mk "b" and oa = mk "oa" and ob = mk "ob" in
  materialize oa;
  materialize ob;
  let i = sym () and j = sym () and k = sym () in
  let shared = loop i (seq (set i a (c 2.)) (set i b (c 3.))) in
  let use_a = loop j (set j oa (get j a)) in
  let use_b = loop k (set k ob (get k b)) in
  let o = optimize (seq shared (seq use_a use_b)) in
  p "independent siblings both virtual" (Tn.known_virtual a && Tn.known_virtual b);
  p "independent siblings setters dropped" (count_set o a = 0 && count_set o b = 0);
  p "independent siblings inlined at use sites (no array reads survive)"
    (count_get o a = 0 && count_get o b = 0);
  (* Sharing symbol [i] alone must not make either sibling complex. *)
  p "is_complex from sharing alone" (is_complex o a || is_complex o b)

(* === Case 2: mixed loop -- one sibling virtual, one materialized === *)
let case_mixed () =
  let a = mk "a" and b = mk "b" and oa = mk "oa" in
  materialize b;
  materialize oa;
  let i = sym () and j = sym () in
  let shared = loop i (seq (set i a (c 2.)) (set i b (c 3.))) in
  let use_a = loop j (set j oa (get j a)) in
  let o = optimize (seq shared use_a) in
  p "mixed cleanup keeps b setter" (count_set o b = 1);
  p "mixed drops virtual a setter" (count_set o a = 0);
  p "mixed a virtual, b non-virtual" (Tn.known_virtual a && Tn.known_non_virtual b)

(* === Case 3: forward sibling provider inlined into a surviving materialized reader === *)
let case_forward_provider () =
  let a = mk "a" and b = mk "b" in
  materialize b;
  let i = sym () in
  (* a written, then b reads a -- both in one loop; b survives (materialized), a virtual. *)
  let shared = loop i (seq (set i a (c 2.)) (set i b (add (get i a) (c 1.)))) in
  let o = optimize shared in
  p "forward provider inlined into materialized reader"
    (Tn.known_virtual a && count_set o a = 0 && count_get o a = 0 && count_set o b = 1)

(* === Case 4: forward virtual->virtual chain consumed downstream === *)
let case_chain () =
  let a = mk "a" and b = mk "b" and out = mk "out" in
  materialize out;
  let i = sym () and j = sym () in
  (* a = f; b = g(a); both virtual. out = h(b), materialized, read downstream. *)
  let shared = loop i (seq (set i a (c 2.)) (set i b (add (get i a) (c 1.)))) in
  let use_b = loop j (set j out (mul (get j b) (c 2.))) in
  let o = optimize (seq shared use_b) in
  p "forward virtual-to-virtual chain both virtual" (Tn.known_virtual a && Tn.known_virtual b);
  p "forward virtual-to-virtual chain fully inlined"
    (count_set o a = 0
    && count_set o b = 0
    && count_get o a = 0
    && count_get o b = 0
    && Tn.known_non_virtual out)

(* === Case 5: loop-carried / read-before-write sibling read stays materialized === [a] is written
   at [i] but read at [i+1] in the same loop, so the read of [a[i+1]] precedes its write in trace
   order (read-before-write). The existing access analysis records this as a recurrent access and
   forces [a] materialized; the later writer must NOT be used to rewrite the earlier read. This is
   the safety mechanism the proposal relies on (#134). *)
let case_reverse () =
  let a = mk "a" and b = mk "b" in
  materialize b;
  let i = sym () in
  let read_ahead = LL.Get (a, [| Ir.Indexing.Affine { symbols = [ (1, i) ]; offset = 1 } |]) in
  let shared = loop i (seq (set i a (c 2.)) (set i b (add read_ahead (c 1.)))) in
  let o = optimize shared in
  p "loop-carried provider kept materialized" (Tn.known_non_virtual a);
  p "loop-carried provider read NOT rewritten (array read preserved)" (count_get o a >= 1)

(* === Case 6: is_complex still set by a genuine complex scalar computation === *)
let case_complex () =
  let x = mk "x" and y = mk "y" and z = mk "z" in
  materialize x;
  materialize y;
  materialize z;
  let i = sym () in
  let l = loop i (set i z (mul (get i x) (get i y))) in
  let o = optimize l in
  p "is_complex from genuine complex scalar" (is_complex o z)

(* === Case 7: two virtual providers + an in-loop materialized consumer (Codex P1) === c
   (materialized) reads BOTH a and b in the same loop. The storage pass for the first candidate (a)
   walks c's setter, which reads the not-yet-stored b; it must not call inline_computation on b
   before b is stored (that raised a stale optimize_ctx error). Both providers must virtualize and
   inline into c, and c's setter must survive. *)
let case_inloop_consumer () =
  let a = mk "a" and b = mk "b" and cons = mk "cons" in
  materialize cons;
  let i = sym () in
  let shared =
    loop i (seq (set i a (c 2.)) (seq (set i b (c 3.)) (set i cons (add (get i a) (get i b)))))
  in
  let o = optimize shared in
  p "in-loop consumer: both providers virtual" (Tn.known_virtual a && Tn.known_virtual b);
  p "in-loop consumer: providers inlined (no array reads survive)"
    (count_get o a = 0 && count_get o b = 0);
  p "in-loop consumer: consumer setter kept" (count_set o cons = 1)

let () =
  case_independent ();
  case_mixed ();
  case_forward_provider ();
  case_chain ();
  case_reverse ();
  case_complex ();
  case_inloop_consumer ();
  Stdio.printf "%!"
