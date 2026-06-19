open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module LL = Ir.Low_level

(* Builders shared across scenarios. *)
let make_tn ~id ~label ~dim =
  Tn.create (Tn.Default Ops.single) ~id ~label:[ label ]
    ~unpadded_dims:(lazy [| dim |])
    ~padding:(lazy None)
    ()

(* Structural summary of a lowered program: how many Declare_local hoists, surviving Local_scope
   nodes, and Get_local references it contains. A cross-statement hoist (legitimate CSE firing)
   shows up as [declares >= 1] with the shared computation referenced via [Get_local]; a rejected
   (unsound) merge leaves the Local_scope nodes in place with no Declare_local. *)
let summarize (llc : LL.t) : int * int * int =
  let rec count_proc ((dl, ls, gl) as acc) (llc : LL.t) =
    match llc with
    | LL.Noop | LL.Comment _ | LL.Staged_compilation _ | LL.Zero_out _ -> acc
    | LL.Declare_local _ -> (dl + 1, ls, gl)
    | LL.Seq (a, b) -> count_proc (count_proc acc a) b
    | LL.For_loop { body; _ } -> count_proc acc body
    | LL.Set { llsc; _ } -> count_scalar acc llsc
    | LL.Set_from_vec { arg = s, _; _ } -> count_scalar acc s
    | LL.Set_local (_, s) -> count_scalar acc s
  and count_scalar ((dl, ls, gl) as acc) (s : LL.scalar_t) =
    match s with
    | LL.Local_scope { body; _ } -> count_proc (dl, ls + 1, gl) body
    | LL.Get_local _ -> (dl, ls, gl + 1)
    | LL.Get_dynamic { dyn_value = v, _; _ } -> count_scalar acc v
    | LL.Get _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _ | LL.Embed_index _ ->
        acc
    | LL.Ternop (_, (a, _), (b, _), (c, _)) -> count_scalar (count_scalar (count_scalar acc a) b) c
    | LL.Binop (_, (a, _), (b, _)) -> count_scalar (count_scalar acc a) b
    | LL.Unop (_, (a, _)) -> count_scalar acc a
  in
  count_proc (0, 0, 0) llc

let report name ~hoisted_expected llc =
  let declares, scopes, gets = summarize llc in
  let hoisted = declares >= 1 in
  let verdict = if Bool.equal hoisted hoisted_expected then "OK" else "FAIL" in
  Stdio.printf "[%s] declares=%d local_scopes=%d get_locals=%d hoisted=%b expected=%b -> %s\n" name
    declares scopes gets hoisted hoisted_expected verdict

(* A Local_scope over [tn_src] whose body reads src at a fixed index, parameterized by its scope id
   and orig_indices. Bodies are structurally identical across instances modulo the scope id, so the
   only thing that can distinguish two instances under alpha-equivalence is [orig_indices]. *)
let make_scope ~tn_src scope_id orig_indices =
  LL.Local_scope
    {
      id = scope_id;
      body = LL.Set_local (scope_id, LL.Get (tn_src, [| Idx.Fixed_idx 0 |]));
      orig_indices;
    }

let make_set ~tn_out scalar =
  LL.Set { tn = tn_out; idcs = [| Idx.Fixed_idx 0 |]; llsc = scalar; debug = "" }

let () =
  let tn_src = make_tn ~id:1 ~label:"src" ~dim:4 in
  let tn_out1 = make_tn ~id:2 ~label:"out1" ~dim:1 in
  let tn_out2 = make_tn ~id:3 ~label:"out2" ~dim:1 in

  (* ===================================================================== *)
  (* Scenario B (soundness, Bug 1): distinct-vs-repeated orig_indices must NOT be CSE'd, in BOTH
     statement orderings. off = t[a; b] (a != b), diag = t[c; c]. The bodies are alpha-equivalent,
     so the only blocker is the orig_indices bijection check. *)
  Stdio.printf "=== Scenario B: diagonal vs off-diagonal (must NOT merge, both orders) ===\n";
  let mk_off () =
    let a = Idx.get_symbol () and b = Idx.get_symbol () in
    make_scope ~tn_src { tn = tn_src; scope_id = 100 } [| Idx.Iterator a; Idx.Iterator b |]
  in
  let mk_diag () =
    let c = Idx.get_symbol () in
    make_scope ~tn_src { tn = tn_src; scope_id = 200 } [| Idx.Iterator c; Idx.Iterator c |]
  in
  (* Representative-first: off-diagonal seen first (the order in which the old code mis-merged). *)
  let b_off_first =
    LL.Seq (make_set ~tn_out:tn_out1 (mk_off ()), make_set ~tn_out:tn_out2 (mk_diag ()))
  in
  report "B.off-first" ~hoisted_expected:false (LL.hoist_cross_statement_cse b_off_first);
  (* Candidate-first: diagonal seen first. *)
  let b_diag_first =
    LL.Seq (make_set ~tn_out:tn_out1 (mk_diag ()), make_set ~tn_out:tn_out2 (mk_off ()))
  in
  report "B.diag-first" ~hoisted_expected:false (LL.hoist_cross_statement_cse b_diag_first);
  Stdio.printf "\n";

  (* ===================================================================== *)
  (* Scenario C (does-not-disable, Bug 1): legitimate alpha-equivalent pair with distinct, but
     consistently-renamed, fresh symbols in orig_indices ([a; b] vs [c; d], all distinct, a!=b,
     c!=d) MUST still be CSE'd/hoisted. *)
  Stdio.printf "=== Scenario C: legitimate renamed pair (must merge) ===\n";
  let mk_pair scope_id =
    let x = Idx.get_symbol () and y = Idx.get_symbol () in
    make_scope ~tn_src { tn = tn_src; scope_id } [| Idx.Iterator x; Idx.Iterator y |]
  in
  let c_prog =
    LL.Seq (make_set ~tn_out:tn_out1 (mk_pair 300), make_set ~tn_out:tn_out2 (mk_pair 400))
  in
  report "C.renamed-pair" ~hoisted_expected:true (LL.hoist_cross_statement_cse c_prog);
  Stdio.printf "\n";

  (* ===================================================================== *)
  (* Scenario D (hoist hazard, Bug 2): two alpha-equivalent Local_scopes reading [src] separated by
     a sibling For_loop that WRITES [src] must NOT be hoisted above the loop. With the old
     non-recursive writes_of_stmt the For_loop reported no writes and the hoist fired (unsound);
     with the recursive version the write to src is seen and the hoist is blocked. *)
  Stdio.printf "=== Scenario D: For_loop write hazard blocks hoist (must NOT merge) ===\n";
  let d_scope0 = make_scope ~tn_src { tn = tn_src; scope_id = 500 } [||] in
  let d_scope2 = make_scope ~tn_src { tn = tn_src; scope_id = 600 } [||] in
  let k = Idx.get_symbol () in
  let d_loop =
    LL.For_loop
      {
        index = k;
        from_ = 0;
        to_ = 3;
        body =
          LL.Set
            { tn = tn_src; idcs = [| Idx.Iterator k |]; llsc = LL.Constant 1.0; debug = "" };
        trace_it = false;
      }
  in
  let d_prog =
    LL.Seq
      ( make_set ~tn_out:tn_out1 d_scope0,
        LL.Seq (d_loop, make_set ~tn_out:tn_out2 d_scope2) )
  in
  report "D.forloop-hazard" ~hoisted_expected:false (LL.hoist_cross_statement_cse d_prog);
  Stdio.printf "\n";

  (* ===================================================================== *)
  (* Scenario A (legitimate cross-statement hoist, full pipeline demo): two sibling Set statements
     with structurally-equivalent reduction Local_scopes share a computation, which CSE hoists.
     Preserved from the original test to keep end-to-end codegen coverage of a firing hoist. *)
  let scope1 : LL.scope_id = { tn = tn_src; scope_id = 700 } in
  let scope2 : LL.scope_id = { tn = tn_src; scope_id = 800 } in
  let idx1 = Idx.get_symbol () in
  let idx2 = Idx.get_symbol () in
  let make_local_scope scope_id idx =
    LL.Local_scope
      {
        id = scope_id;
        body =
          LL.For_loop
            {
              index = idx;
              from_ = 0;
              to_ = 3;
              body =
                LL.Set_local
                  ( scope_id,
                    LL.Binop
                      ( Ops.Add,
                        (LL.Get_local scope_id, Ops.single),
                        (LL.Get (tn_src, [| Idx.Iterator idx |]), Ops.single) ) );
              trace_it = true;
            };
        orig_indices = [||];
      }
  in
  let stmt1 =
    LL.Set
      {
        tn = tn_out1;
        idcs = [| Idx.Fixed_idx 0 |];
        llsc = make_local_scope scope1 idx1;
        debug = "out1 := sum(src)";
      }
  in
  let stmt2 =
    LL.Set
      {
        tn = tn_out2;
        idcs = [| Idx.Fixed_idx 0 |];
        llsc = make_local_scope scope2 idx2;
        debug = "out2 := sum(src)";
      }
  in
  let llc = LL.Seq (stmt1, stmt2) in

  Stdio.printf "=== Scenario A: legitimate hoist (to_doc, before) ===\n";
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (LL.to_doc () llc);
  Stdio.printf "\n\n";

  let result = LL.hoist_cross_statement_cse llc in
  report "A.legit-hoist" ~hoisted_expected:true result;

  Stdio.printf "=== Scenario A: legitimate hoist (to_doc, after) ===\n";
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (LL.to_doc () result);
  Stdio.printf "\n\n";

  Stdio.printf "=== Scenario A: legitimate hoist (to_doc_cstyle, after) ===\n";
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (LL.to_doc_cstyle () result);
  Stdio.printf "\n\n";

  let optimized : LL.optimized =
    {
      traced_store = Hashtbl.create (module Ir.Tnode);
      optimize_ctx = { computations = Hashtbl.create (module Ir.Tnode) };
      llc = result;
      merge_node = None;
    }
  in
  let module Syntax = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    type buffer_ptr = unit Ctypes.ptr

    let procs = [| optimized |]
    let full_printf_support = true
  end))
  in
  Utils.set_log_level 2;
  Utils.settings.debug_log_from_routines <- true;
  Stdio.printf "=== Scenario A: legitimate hoist (c_syntax pp_ll, after) ===\n";
  PPrint.ToChannel.pretty 0.9 110 Stdio.stdout (Syntax.compile_main result);
  Stdio.printf "\n%!"
