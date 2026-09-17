(* Context-scoped memory modes (docs/proposals/context-scoped-memory-modes.md): sibling compiles
   from the SAME context must be hermetic, each resolving placements in its own lineage. The same
   shared node [x] is compiled by two sibling routines: lineage A consumes it once (virtualizable),
   lineage B consumes it at two visits of the same cell (over the visit cap, so B must keep it
   non-virtual). Under the retired global [Tnode.memory_mode_intent] settlement this divergence was
   impossible: whichever sibling compiled first pinned the mode for the other. Both siblings also
   execute, pinning that the divergent placements produce correct values, and the tnode's declared
   intent stays untouched throughout. *)

open Base
module Idx = Ir.Indexing
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Ops = Ir.Ops
open Verdict.Claims

let single = Ops.single
let next_id = ref 5000

let mk ~dims label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

let dbg : Idx.projections_debug = { spec = ""; derived_for = Sexp.Atom ""; trace = [] }

(* Projection lhs[i] = f(rhs1[i], rhs2[i or 0]) over one axis of dim [n]. With [rhs2_fixed0] the
   second operand is read at the fixed cell 0: reading a cell different from the assigned cell is
   what the visit counter counts (under [inline_complex_computations]), so cell 0 of rhs2
   accumulates one visit per loop iteration and trips the [virtualize_max_visits = 1] cap. *)
let elementwise i ~n ~num_rhs ~rhs2_fixed0 : Idx.projections =
  {
    components = [| [ (n, i) ] |];
    lhs_dims = [| n |];
    rhs_dims = Array.init num_rhs ~f:(fun _ -> [| n |]);
    project_lhs = [| Idx.Iterator i |];
    project_rhs =
      Array.init num_rhs ~f:(fun k ->
          if rhs2_fixed0 && k = 1 then [| Idx.Fixed_idx 0 |] else [| Idx.Iterator i |]);
    extent_syms = [];
    debug_info = dbg;
  }

(* lhs[i] = rhs1[i] + rhs2[i], overwriting (accumulated from the Add neutral). *)
let add_asgn ~lhs ~rhs1 ~rhs2 proj =
  Asgns.Accum_op
    {
      initialize_neutral = true;
      accum = Ops.Add;
      lhs;
      rhs = Asgns.Binop { op = Ops.Add; rhs1 = Asgns.Node rhs1; rhs2 = Asgns.Node rhs2 };
      projections = lazy proj;
      projections_debug = "elementwise_add";
    }

let n = 4

(* One comp: x[i] = src[i] + src[i]; out[i] = x[i] + x[i or 0]. With [repeat_cell0] the consumer
   re-reads x's cell 0 at every iteration (over the visit cap: lineage keeps x non-virtual);
   otherwise each cell of x is read only at its own assignment position (virtualizable). *)
let make_comp ~name ~src ~x ~out ~repeat_cell0 =
  let i1 = Idx.get_symbol () and i2 = Idx.get_symbol () in
  let produce =
    add_asgn ~lhs:x ~rhs1:src ~rhs2:src (elementwise i1 ~n ~num_rhs:2 ~rhs2_fixed0:false)
  in
  let consume =
    add_asgn ~lhs:out ~rhs1:x ~rhs2:x (elementwise i2 ~n ~num_rhs:2 ~rhs2_fixed0:repeat_cell0)
  in
  let asgns = Asgns.Block_comment (name, Asgns.Seq (produce, consume)) in
  let embedded_nodes = Set.of_list (module Tn) [ x; out ] in
  { Asgns.asgns; embedded_nodes }

let () =
  let src = mk ~dims:[| n |] "src" in
  let x = mk ~dims:[| n |] "x_shared" in
  let out1 = mk ~dims:[| n |] "out1" in
  let out2 = mk ~dims:[| n |] "out2" in
  Tn.update_memory_mode out1 Tn.On_device (Site "99:test-setup");
  Tn.update_memory_mode out2 Tn.On_device (Site "99:test-setup");
  (* Lineage A: x consumed at one visit per cell -> virtualizable. *)
  let comp_a = make_comp ~name:"sibling_a" ~src ~x ~out:out1 ~repeat_cell0:false in
  (* Lineage B: x's cell 0 re-read at every iteration -> over the visit cap, kept non-virtual. *)
  let comp_b = make_comp ~name:"sibling_b" ~src ~x ~out:out2 ~repeat_cell0:true in
  let ctx0 = Context.auto () in
  let ctx0 = Context.set_values ctx0 src [| 1.; 2.; 3.; 4. |] in
  (* Sibling compiles from the SAME context. *)
  let ctx_a, routine_a = Context.compile ctx0 comp_a Idx.Empty in
  let ctx_b, routine_b = Context.compile ctx0 comp_b Idx.Empty in
  let ctx_a = Context.run ctx_a routine_a in
  let ctx_b = Context.run ctx_b routine_b in
  p "lineage A virtualized x" (Tn.Placements.known_virtual (Context.placements ctx_a) x);
  p "lineage B kept x non-virtual" (Tn.Placements.known_non_virtual (Context.placements ctx_b) x);
  p "x declared intent untouched" (Option.is_none x.Tn.memory_mode_intent);
  let show name values =
    Stdio.printf "%s:" name;
    Array.iter values ~f:(fun v -> Stdio.printf " %.1f" v);
    Stdio.printf "\n"
  in
  show "out1 (= 4*src)" (Context.get_values ctx_a out1);
  show "out2 (= 2*src + 2*src[0])" (Context.get_values ctx_b out2)
