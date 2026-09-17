(* Recompute-cost guard for virtualization ([virtualize_max_inline_reduction]): a node whose setter
   is enclosed by reduction loops (loops not appearing in the setter's indices) with a large total
   trip count must not be virtualized -- inlining it replays the whole reduction at every read site,
   and the cost multiplies through chains of virtual consumers. This is what made untuned
   transformer/MLP graphs pathologically slow: with all reads at the assigned cell (exempt from the
   visit cap under [inline_complex_computations]), attention out-projections and FFN matmuls stayed
   virtual and were recomputed per consumer element.

   A small reduction (K=4, under the default cap of 16) stays virtual and inlines into its same-cell
   consumer; a large one (K=64) is forced non-virtual. Both execute and match hand-computed sums. *)

open Base
module Idx = Ir.Indexing
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Ops = Ir.Ops

let single = Ops.single
let next_id = ref 7000

let mk ~dims label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

let dbg : Idx.projections_debug = { spec = ""; derived_for = Sexp.Atom ""; trace = [] }

(* Row-sum reduce: prod[i] = sum_k a[i, k], product axes i (dim n), k (dim kdim). *)
let reduce_proj i k ~n ~kdim : Idx.projections =
  {
    components = [| [ (n, i) ]; [ (kdim, k) ] |];
    lhs_dims = [| n |];
    rhs_dims = [| [| n; kdim |] |];
    project_lhs = [| Idx.Iterator i |];
    project_rhs = [| [| Idx.Iterator i; Idx.Iterator k |] |];
    extent_syms = [];
    debug_info = dbg;
  }

let reduce_asgn ~dst ~src proj =
  Asgns.Accum_op
    {
      initialize_neutral = true;
      accum = Ops.Add;
      lhs = dst;
      rhs = Asgns.Unop { op = Ops.Identity; rhs = Asgns.Node src };
      projections = lazy proj;
      projections_debug = "row_sum";
    }

(* Same-cell consumer: out[i] = prod[i]. Reading only the assigned cell keeps the producer under the
   visit cap, so before the recompute-cost guard it was always virtualized. *)
let copy_proj t ~n : Idx.projections =
  {
    components = [| [ (n, t) ] |];
    lhs_dims = [| n |];
    rhs_dims = [| [| n |] |];
    project_lhs = [| Idx.Iterator t |];
    project_rhs = [| [| Idx.Iterator t |] |];
    extent_syms = [];
    debug_info = dbg;
  }

let copy_asgn ~dst ~src proj =
  Asgns.Accum_op
    {
      initialize_neutral = true;
      accum = Ops.Add;
      lhs = dst;
      rhs = Asgns.Unop { op = Ops.Identity; rhs = Asgns.Node src };
      projections = lazy proj;
      projections_debug = "copy";
    }

let n = 2

(* A large reduction that nothing reads must NOT be materialized by the recompute-cost guard: with
   no read site there is no inlining cost, and forcing it non-virtual would turn dead
   virtual-eligible work into executed work. It stays a committed virtual computation. *)
let run_dead ~kdim =
  let i = Idx.get_symbol () and k = Idx.get_symbol () in
  let a = mk ~dims:[| n; kdim |] "a_dead" in
  let prod = mk ~dims:[| n |] "prod_dead" in
  let reduce = reduce_asgn ~dst:prod ~src:a (reduce_proj i k ~n ~kdim) in
  let asgns = Asgns.Block_comment ("reduction_dead", reduce) in
  let comp = { Asgns.asgns; embedded_nodes = Set.of_list (module Tn) [ prod ] } in
  let ctx = Context.auto () in
  let ctx = Context.set_values ctx a (Array.create ~len:(n * kdim) 1.) in
  let ctx, _routine = Context.compile ctx comp Idx.Empty in
  let plc = Context.placements ctx in
  (Tn.Placements.known_virtual plc prod, Tn.Placements.known_non_virtual plc prod)

(* Returns (out values, prod known-virtual, prod known-non-virtual). *)
let run ~kdim =
  let i = Idx.get_symbol () and k = Idx.get_symbol () and t = Idx.get_symbol () in
  let a = mk ~dims:[| n; kdim |] "a" in
  let prod = mk ~dims:[| n |] "prod" in
  let out = mk ~dims:[| n |] "out" in
  Tn.update_memory_mode out Tn.On_device "99:test-setup";
  let reduce = reduce_asgn ~dst:prod ~src:a (reduce_proj i k ~n ~kdim) in
  let copy = copy_asgn ~dst:out ~src:prod (copy_proj t ~n) in
  let asgns = Asgns.Block_comment ("reduction_inline_guard", Asgns.Seq (reduce, copy)) in
  let comp = { Asgns.asgns; embedded_nodes = Set.of_list (module Tn) [ prod; out ] } in
  let ctx = Context.auto () in
  (* a[i, k] = (i+1) * (k+1), so prod[i] = (i+1) * kdim*(kdim+1)/2. *)
  let values =
    Array.init (n * kdim) ~f:(fun p -> Float.of_int (((p / kdim) + 1) * ((p % kdim) + 1)))
  in
  let ctx = Context.set_values ctx a values in
  let ctx, routine = Context.compile ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  let plc = Context.placements ctx in
  ( Context.get_values ctx out,
    Tn.Placements.known_virtual plc prod,
    Tn.Placements.known_non_virtual plc prod )

let show vals = String.concat ~sep:"; " (Array.to_list (Array.map vals ~f:(Printf.sprintf "%g")))

let () =
  let expected ~kdim =
    show (Array.init n ~f:(fun i -> Float.of_int ((i + 1) * (kdim * (kdim + 1) / 2))))
  in
  let out_small, virt_small, nonvirt_small = run ~kdim:4 in
  Stdio.printf "small reduction (K=4): virtual=%b non-virtual=%b\n" virt_small nonvirt_small;
  (* The pair is one tri-state placement report, so the row keeps its shape and the claims sit
     beside it, each on the same [let]-bound boolean the row prints. *)
  Verdict.claim "small reduction (K=4) stays virtual" virt_small;
  Verdict.claim "small reduction (K=4) is not forced non-virtual" (not nonvirt_small);
  Stdio.printf "small reduction (K=4): out=[%s] expected=[%s]\n" (show out_small) (expected ~kdim:4);
  let out_large, virt_large, nonvirt_large = run ~kdim:64 in
  Stdio.printf "large reduction (K=64): virtual=%b non-virtual=%b\n" virt_large nonvirt_large;
  Verdict.claim "large reduction (K=64) is not virtual" (not virt_large);
  Verdict.claim "large reduction (K=64) is forced non-virtual" nonvirt_large;
  Stdio.printf "large reduction (K=64): out=[%s] expected=[%s]\n" (show out_large)
    (expected ~kdim:64);
  let virt_dead, nonvirt_dead = run_dead ~kdim:64 in
  Stdio.printf "dead large reduction (K=64): virtual=%b non-virtual=%b\n" virt_dead nonvirt_dead;
  Verdict.claim "dead large reduction (K=64) stays virtual" virt_dead;
  Verdict.claim "dead large reduction (K=64) is not forced non-virtual" (not nonvirt_dead);
  Stdio.printf "%!"
