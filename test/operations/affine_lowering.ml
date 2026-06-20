(* gh-ocannl-133 Stage B: lowering payoff (AC4) and executable virtual-vs-materialized parity (AC6).

   AC4 -- an injective + surjective affine scatter (pool-backward [2*oh+wh] with stride = window)
   lowers to a plain setter with no neutral-init [Zero_out] and no read-modify-write accumulation. We
   build the [Accum_op] directly (the einsum grammar has no result-side affine scatter) and inspect
   [Ir.Assignments.to_low_level]. A non-injective [i+j] scatter is the contrast: it DOES emit a
   [Zero_out] and a read-modify-write, proving the AC4 assertion is non-vacuous.

   AC6 -- the same [2*oh+wh] scatter, consumed at a plain iterator, virtualizes (Stage B
   unit-coefficient solving) and, executed on the configured backend, matches both the materialized
   execution and the hand-computed expected values. *)

open Base
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Ops = Ir.Ops

let single = Ops.single
let next_id = ref 3000

let mk ~dims label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ] ~unpadded_dims:(lazy dims)
    ~padding:(lazy None) ()

let dbg : Idx.projections_debug = { spec = ""; derived_for = Sexp.Atom ""; trace = [] }

(* Scatter projection: lhs[c1*s1 + c2*s2] = rhs[s1, s2], product axes s1 (dim n1), s2 (dim n2). *)
let scatter_proj s1 s2 ~n1 ~n2 ~c1 ~c2 ~lhs_dim : Idx.projections =
  {
    product_space = [| [ n1 ]; [ n2 ] |];
    lhs_dims = [| lhs_dim |];
    rhs_dims = [| [| n1; n2 |] |];
    product_iterators = [| [ s1 ]; [ s2 ] |];
    project_lhs = [| Idx.Affine { symbols = [ (c1, s1); (c2, s2) ]; offset = 0 } |];
    project_rhs = [| [| Idx.Iterator s1; Idx.Iterator s2 |] |];
    debug_info = dbg;
  }

let scatter_asgn ~dst ~src proj =
  Asgns.Accum_op
    {
      initialize_neutral = true;
      accum = Ops.Add;
      lhs = dst;
      rhs = Asgns.Unop { op = Ops.Identity; rhs = Asgns.Node src };
      projections = lazy proj;
      projections_debug = "scatter";
    }

(* --- AC4: inspect the lowered Low_level.t --- *)
let count_zero_out (llc : LL.t) tn =
  let n = ref 0 in
  let rec t = function
    | LL.Seq (a, b) -> t a; t b
    | LL.For_loop { body; _ } -> t body
    | LL.Zero_out z -> if z.Tn.id = tn.Tn.id then Int.incr n
    | LL.Comment _ | LL.Noop -> ()
    | LL.Set { llsc; _ } -> ignore llsc
    | _ -> ()
  in
  t llc;
  !n

(* Does the setter of [tn] read [tn] (read-modify-write accumulation)? *)
let setter_reads_self (llc : LL.t) tn =
  let found = ref false in
  let rec scal (s : LL.scalar_t) =
    match s with
    | LL.Get (g, _) -> if g.Tn.id = tn.Tn.id then found := true
    | LL.Get_dynamic { tn = g; dyn_value = v, _; _ } ->
        if g.Tn.id = tn.Tn.id then found := true;
        scal v
    | LL.Local_scope { body; _ } -> t body
    | LL.Ternop (_, (a, _), (b, _), (c, _)) -> scal a; scal b; scal c
    | LL.Binop (_, (a, _), (b, _)) -> scal a; scal b
    | LL.Unop (_, (a, _)) -> scal a
    | _ -> ()
  and t = function
    | LL.Seq (a, b) -> t a; t b
    | LL.For_loop { body; _ } -> t body
    | LL.Set { tn = stn; llsc; _ } -> if stn.Tn.id = tn.Tn.id then scal llsc
    | LL.Set_local (_, s) -> scal s
    | _ -> ()
  in
  t llc;
  !found

let p name b = Stdio.printf "%s: %b\n" name b

let ac4 () =
  (* Injective + surjective: 2*oh + wh, oh in [0,3), wh in [0,2), covers [0,6). *)
  let oh = Idx.get_symbol () and wh = Idx.get_symbol () in
  let dst_i = mk ~dims:[| 6 |] "dst_inj" and src_i = mk ~dims:[| 3; 2 |] "src_inj" in
  let llc_inj =
    Asgns.to_low_level
      (scatter_asgn ~dst:dst_i ~src:src_i
         (scatter_proj oh wh ~n1:3 ~n2:2 ~c1:2 ~c2:1 ~lhs_dim:6))
  in
  p "injective scatter: no neutral-init Zero_out" (count_zero_out llc_inj dst_i = 0);
  p "injective scatter: plain setter (no read-modify-write)"
    (not (setter_reads_self llc_inj dst_i));

  (* Non-injective contrast: i + j, both in [0,3) -- the assertion above is non-vacuous. *)
  let i = Idx.get_symbol () and j = Idx.get_symbol () in
  let dst_n = mk ~dims:[| 5 |] "dst_ni" and src_n = mk ~dims:[| 3; 3 |] "src_ni" in
  let llc_ni =
    Asgns.to_low_level
      (scatter_asgn ~dst:dst_n ~src:src_n
         (scatter_proj i j ~n1:3 ~n2:3 ~c1:1 ~c2:1 ~lhs_dim:5))
  in
  p "non-injective scatter: emits neutral-init Zero_out" (count_zero_out llc_ni dst_n >= 1);
  p "non-injective scatter: read-modify-write accumulation" (setter_reads_self llc_ni dst_n)

(* --- AC6: execute [out[t] = dst[t]] where [dst[2*oh+wh] = src[oh,wh]], comparing the scatter
   virtual (inlined via Stage B unit-coefficient solving) vs materialized. --- *)
let copy_proj t ~n : Idx.projections =
  {
    product_space = [| [ n ] |];
    lhs_dims = [| n |];
    rhs_dims = [| [| n |] |];
    product_iterators = [| [ t ] |];
    project_lhs = [| Idx.Iterator t |];
    project_rhs = [| [| Idx.Iterator t |] |];
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

(* Returns (out values, dst known_virtual). [materialize_dst] forces the intermediate to a buffer. *)
let run_scatter_then_copy ~materialize_dst =
  let oh = Idx.get_symbol () and wh = Idx.get_symbol () and t = Idx.get_symbol () in
  let src = mk ~dims:[| 3; 2 |] "src" in
  let dst = mk ~dims:[| 6 |] "dst" in
  let out = mk ~dims:[| 6 |] "out" in
  Tn.update_memory_mode out Tn.Materialized 99;
  if materialize_dst then Tn.update_memory_mode dst Tn.Materialized 99;
  let scatter = scatter_asgn ~dst ~src (scatter_proj oh wh ~n1:3 ~n2:2 ~c1:2 ~c2:1 ~lhs_dim:6) in
  let copy = copy_asgn ~dst:out ~src:dst (copy_proj t ~n:6) in
  (* The backend derives the routine name from a block comment. *)
  let asgns = Asgns.Block_comment ("affine_scatter_copy", Asgns.Seq (scatter, copy)) in
  (* [dst] and [out] are produced by the comp (embedded); [src] is the only external input. *)
  let comp =
    { Asgns.asgns; embedded_nodes = Set.of_list (module Tn) [ dst; out ] }
  in
  let ctx = Context.auto () in
  let ctx = Context.set_values ctx src [| 10.; 11.; 12.; 13.; 14.; 15. |] in
  let ctx, routine = Context.compile ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx out, Tn.known_virtual dst)

(* Triangular scatter dst[s1, s1+s2] = src[s1,s2], s1 in [0,3), s2 in [0,2), consumed at plain
   [out[a,b] = dst[a,b]]. Unit-coefficient solving binds s1<-a then solves s2 = b - a with range guard
   [0 <= b-a < 2]; off-region cells fall back to the prepended init (0). Exercises a non-trivial [rest]
   (= s1) and the init fallback. *)
let tri_scatter_proj s1 s2 : Idx.projections =
  {
    product_space = [| [ 3 ]; [ 2 ] |];
    lhs_dims = [| 3; 4 |];
    rhs_dims = [| [| 3; 2 |] |];
    product_iterators = [| [ s1 ]; [ s2 ] |];
    project_lhs = [| Idx.Iterator s1; Idx.Affine { symbols = [ (1, s1); (1, s2) ]; offset = 0 } |];
    project_rhs = [| [| Idx.Iterator s1; Idx.Iterator s2 |] |];
    debug_info = dbg;
  }

let tri_copy_proj a b : Idx.projections =
  {
    product_space = [| [ 3 ]; [ 4 ] |];
    lhs_dims = [| 3; 4 |];
    rhs_dims = [| [| 3; 4 |] |];
    product_iterators = [| [ a ]; [ b ] |];
    project_lhs = [| Idx.Iterator a; Idx.Iterator b |];
    project_rhs = [| [| Idx.Iterator a; Idx.Iterator b |] |];
    debug_info = dbg;
  }

let run_triangular ~materialize_dst =
  let s1 = Idx.get_symbol () and s2 = Idx.get_symbol () in
  let a = Idx.get_symbol () and b = Idx.get_symbol () in
  let src = mk ~dims:[| 3; 2 |] "tsrc" in
  let dst = mk ~dims:[| 3; 4 |] "tdst" in
  let out = mk ~dims:[| 3; 4 |] "tout" in
  Tn.update_memory_mode out Tn.Materialized 99;
  if materialize_dst then Tn.update_memory_mode dst Tn.Materialized 99;
  let scatter = scatter_asgn ~dst ~src (tri_scatter_proj s1 s2) in
  let copy = copy_asgn ~dst:out ~src:dst (tri_copy_proj a b) in
  let asgns = Asgns.Block_comment ("tri_scatter_copy", Asgns.Seq (scatter, copy)) in
  let comp = { Asgns.asgns; embedded_nodes = Set.of_list (module Tn) [ dst; out ] } in
  let ctx = Context.auto () in
  let ctx = Context.set_values ctx src [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let ctx, routine = Context.compile ctx comp Idx.Empty in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx out, Tn.known_virtual dst)

let show a = String.concat ~sep:" " (Array.to_list (Array.map a ~f:Float.to_string))

let ac6 () =
  (* Fresh tnodes (unique ids) per run isolate memory-mode decisions between the two runs. *)
  let out_virtual, dst_virtual = run_scatter_then_copy ~materialize_dst:false in
  let out_mat, _ = run_scatter_then_copy ~materialize_dst:true in
  (* dst index 2*oh+wh and src row-major index oh*2+wh coincide, so out = src flat. *)
  let expected = [| 10.; 11.; 12.; 13.; 14.; 15. |] in
  Stdio.printf "AC6 unit-solve(plain): virtual=[%s] expected=[%s]\n" (show out_virtual) (show expected);
  p "AC6 unit-solve(plain) dst virtualized (Stage B path exercised)" dst_virtual;
  p "AC6 unit-solve(plain) virtual matches expected" (Array.equal Float.equal out_virtual expected);
  p "AC6 unit-solve(plain) virtual matches materialized" (Array.equal Float.equal out_virtual out_mat);

  (* Triangular: out[a,b] = src[a, b-a] when 0 <= b-a < 2, else 0 (init). *)
  let tri_virtual, tri_dst_virtual = run_triangular ~materialize_dst:false in
  let tri_mat, _ = run_triangular ~materialize_dst:true in
  let tri_expected = [| 0.; 1.; 0.; 0.; 0.; 2.; 3.; 0.; 0.; 0.; 4.; 5. |] in
  Stdio.printf "AC6 triangular: virtual=[%s] expected=[%s]\n" (show tri_virtual) (show tri_expected);
  p "AC6 triangular dst virtualized (Stage B path exercised)" tri_dst_virtual;
  p "AC6 triangular virtual matches expected" (Array.equal Float.equal tri_virtual tri_expected);
  p "AC6 triangular virtual matches materialized" (Array.equal Float.equal tri_virtual tri_mat)

(* --- task-9658aac9: reuse the Stage B unit-solve scatter->copy under debug value-logging. The
   unit-solve [Where (range_cond, Get (src, solved_idx), Get_local id)] is emitted inside the
   virtualized producer's [Local_scope] body, which is rendered with [log_set_locals:false], so it is
   never passed to [debug_float] -- only the runtime [pp_scalar] ternary (which short-circuits)
   appears. We assert the kernel still virtualizes (Stage B path) and computes correct results with
   debug value-logging enabled (the config that would surface the OOB read), confirming no fault.
   The non-vacuous coverage of the [debug_float] [Where] guard itself -- the actual fix site -- is in
   [test/operations/debug_where_guard.ml], which exercises a directly-logged top-level [Where]. --- *)
(* Route stdout to /dev/null around [f] so the (backend-specific) config-retrieval noise that
   [log_level > 1] turns on stays out of the deterministic test output. *)
let with_stdout_to_devnull f =
  Stdlib.flush Stdlib.stdout;
  let saved = Unix.dup Unix.stdout in
  let dn = Unix.openfile "/dev/null" [ Unix.O_WRONLY ] 0o600 in
  Unix.dup2 dn Unix.stdout;
  Unix.close dn;
  Exn.protect ~f ~finally:(fun () ->
      Stdlib.flush Stdlib.stdout;
      Unix.dup2 saved Unix.stdout;
      Unix.close saved)

let ac_debug_guard () =
  Utils.set_log_level 2;
  Utils.settings.debug_log_from_routines <- true;
  Utils.settings.output_debug_files_in_build_directory <- true;
  let out, dst_virtual =
    with_stdout_to_devnull (fun () -> run_scatter_then_copy ~materialize_dst:false)
  in
  let expected = [| 10.; 11.; 12.; 13.; 14.; 15. |] in
  p "AC4 debug-logging: Stage B unit-solve virtualized" dst_virtual;
  p "AC4 debug-logging: parity holds under debug value-logging"
    (Array.equal Float.equal out expected)

let () =
  ac4 ();
  ac6 ();
  ac_debug_guard ();
  Stdio.printf "%!"
