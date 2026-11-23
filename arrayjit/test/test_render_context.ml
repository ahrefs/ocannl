(* gh-ocannl-769: a backend syntax callback may render another procedure before the outer renderer
   resumes. The inner compile must not replace the outer traversal's state. *)
open Base
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Cs = Ir.C_syntax
module B = Ll_builders

let node = B.node_factory ~first_id:76900 ~dims:[| 2 |] ()
let scratch = node "reentrant_scratch"
let output = node "reentrant_output"
let source = node "reentrant_source"

let () =
  Tn.update_memory_mode scratch Tn.Local (Site "769:test-setup");
  B.materialize output;
  B.materialize source

let index = B.sym ()

let outer =
  B.seq (B.zero scratch)
    (B.seq
       (B.loop ~upto:1 index
          (B.seq
             (B.set output
                [| B.fixed 0 |]
                (B.add (B.get output [| B.fixed 0 |]) (B.get source [| B.iter index |])))
             (B.set scratch [| B.iter index |] (LL.Constant 7.))))
       (B.zero scratch))

let optimized llc nodes =
  let traced_store = Hashtbl.create (module Tn) in
  List.iter nodes ~f:(fun tn ->
      let n = LL.get_node traced_store tn in
      n.LL.zero_initialized_by_code <- Tn.equal tn scratch);
  {
    LL.traced_store;
    llc;
    optimize_ctx = LL.empty_optimize_ctx ();
    merge_node = None;
    workgroup_shared = Set.empty (module Tn);
    simdgroup_fragments = Set.empty (module Tn);
    masked_fragments = Set.empty (module Tn);
    swizzled = Map.empty (module Tn);
    pipelined = Map.empty (module Tn);
    zero_fringe = Set.empty (module Tn);
    flip_candidates = [];
    spliced_rbw = Set.empty (module Tn);
    source = llc;
  }

let outer_proc = optimized outer [ scratch; output; source ]
let inner_proc = optimized LL.Noop []
let on_binop = ref (fun () -> ())

module Config = struct
  include Cs.Pure_C_config (struct
    let procs = [| outer; LL.Noop |]
    let full_printf_support = true
  end)

  let volatile_serial_accumulation = true

  let binop_syntax prec op a b =
    !on_binop ();
    Cs.default_binop_syntax prec op a b
end

module Syntax = Cs.C_syntax (Config)

let render name proc =
  let _, doc, _ = Syntax.compile_proc ~name [] proc in
  Syntax.doc_to_string doc

let zero_stores text =
  List.count (String.split_lines text) ~f:(fun line ->
      String.is_substring line ~substring:"reentrant_scratch["
      && String.is_substring line ~substring:"] = (float)(0.0)")

let () =
  let baseline, baseline_census = Cs.with_volatility_census (fun () -> render "outer" outer_proc) in
  p "baseline retains the genuine second whole-node zero" (zero_stores baseline = 1);
  p "baseline exercises volatile device RMW rendering" (baseline_census.volatile_rmw_reads = 1);
  let nested_calls = ref 0 in
  (on_binop :=
     fun () ->
       (on_binop := fun () -> ());
       Int.incr nested_calls;
       ignore (render "inner" inner_proc : string));
  let nested, census = Cs.with_volatility_census (fun () -> render "outer" outer_proc) in
  p "the nested compile ran while rendering the outer expression" (!nested_calls = 1);
  p "nested rendering retains the outer genuine second whole-node zero" (zero_stores nested = 1);
  p "nested rendering preserves the outer volatility decision" (census.volatile_rmw_reads = 1);
  p_all "nested rendering attributes every outer volatility site to its own kernel" census.entries
    ~f:(fun (name, _) -> String.equal name "outer");
  (* A callback fails after the inner compile has finished: a fresh compile still starts clean. *)
  (on_binop :=
     fun () ->
       (on_binop := fun () -> ());
       ignore (render "inner_failure" inner_proc : string);
       raise Stdlib.Exit);
  let failed =
    try
      ignore (render "interrupted" outer_proc : string);
      false
    with Stdlib.Exit -> true
  in
  p "the exception control interrupts emission" failed;
  let recovered, recovered_census =
    Cs.with_volatility_census (fun () -> render "recovered" outer_proc)
  in
  p "rendering after an interrupted compile starts with fresh traversal state"
    (zero_stores recovered = 1 && recovered_census.volatile_rmw_reads = 1);
  p_all "recovery records only the recovered kernel" recovered_census.entries ~f:(fun (name, _) ->
      String.equal name "recovered")

(* Negative control: reproduce the old inner-compile reset against an explicitly shared outer cell.
   The same emitted-code observation must see the missing re-zero. *)
let () =
  let ctx = Syntax.create_render_ctx ~name:"shared_control" outer_proc in
  (on_binop :=
     fun () ->
       (on_binop := fun () -> ());
       Hash_set.clear (Syntax.zero_out_seen ctx));
  let broken = Syntax.doc_to_string (Syntax.compile_main ctx outer) in
  p "the negative control detects the old shared zero-state reset" (zero_stores broken = 0)

(* No compiler/global fresh-id allocation is needed to overlap collection brackets. The main domain
   holds its bracket open while a child completes its own, deterministically exposing a
   process-global collector without relying on scheduling sleeps. *)
let () =
  let (), outer =
    Cs.with_census (fun () ->
        Cs.mma_census () := [ ("main", Cs.Mma_scalar_fallback) ];
        Cs.mma_scope_census () := [ "main" ];
        let inner =
          Stdlib.Domain.spawn (fun () ->
              let (), summary =
                Cs.with_census (fun () ->
                    Cs.mma_census () := [ ("child", Cs.Mma_register_tiled) ];
                    Cs.mma_scope_census () := [ "child"; "child" ])
              in
              summary)
        in
        let child = Stdlib.Domain.join inner in
        p "a child domain summarizes only its own renderings"
          (child.statements = 1 && child.scalar_fallbacks = 0 && child.fragment_scopes = 2))
  in
  p "a main-domain census excludes a child domain's renderings"
    (outer.statements = 1 && outer.scalar_fallbacks = 1 && outer.fragment_scopes = 1);
  p_all "main-domain attribution survives overlapping collection brackets" outer.renderings
    ~f:(fun (name, _) -> String.equal name "main")

(* gh-ocannl-1707: the persistent-fragment scope census nests as the statement census does: an inner
   bracket summarizes its own scopes, and they count toward the enclosing one too. *)
let () =
  let inner = ref Cs.empty_mma_summary in
  let (), outer =
    Cs.with_census (fun () ->
        Cs.mma_scope_census () := [ "outer" ];
        let (), summary = Cs.with_census (fun () -> Cs.mma_scope_census () := [ "inner" ]) in
        inner := summary)
  in
  p
    "a nested census counts its own persistent-fragment scopes, and the enclosing one counts them \
     too"
    (!inner.fragment_scopes = 1 && outer.fragment_scopes = 2)

let () =
  let (), _ =
    Cs.with_peel_census (fun () ->
        let outer = Syntax.create_render_ctx ~name:"suspended" outer_proc in
        Syntax.peel_census_enabled outer := false;
        let inner = Syntax.create_render_ctx ~name:"inner_collecting" inner_proc in
        p "suspending the outer peel census does not suspend a nested compile"
          (!(Syntax.peel_census_enabled inner) && not !(Syntax.peel_census_enabled outer)))
  in
  ()

(* gh-ocannl-724: emission-level invariants of {!Cs.render_summary} — every tensor-write emitter
   records its store, discarded projection operands are excluded from reads, unconsumed zero-seeds
   preserve emission order, discarded [debug_float] renders do not duplicate loops, peel provenance
   is scoped to the closing store occurrence, and [render_census_enabled] is read live at emission
   time for body-only [compile_main] callers. *)
let () =
  let module Ops = Ir.Ops in
  let prec = Ops.single in
  let render_pure ?(vec_bytes = 0) ~name ~nodes llc =
    let opt = optimized llc nodes in
    let module S = Cs.C_syntax (struct
      include Cs.Pure_C_config (struct
        let procs = [| opt.LL.llc |]
        let full_printf_support = true
      end)

      let vector_file = { Ir.Vector_file.bytes = vec_bytes; registers = 16 }
    end) in
    let _, summary = Cs.with_render_census (fun () -> ignore (S.compile_proc ~name [] opt)) in
    summary
  in
  let rnode = B.node_factory ~first_id:724_100_000 ~dims:[| 16 |] () in
  let r_out = rnode "rc_out"
  and r_src = rnode "rc_src"
  and r_idx = B.node_factory ~first_id:724_100_100 ~dims:[| 4 |] () "rc_idx" in
  List.iter [ r_out; r_src; r_idx ] ~f:B.materialize;
  let ri = B.sym () in
  (* 1. Set_dynamic, Set_from_vec, and try_vectorize packed stores all record store entries. *)
  let dyn_summary =
    render_pure ~name:"rc_set_dynamic" ~nodes:[ r_out; r_src ]
      (B.scatter ~tn:r_out
         ~idcs:[| B.fixed 0 |]
         ~dyn_axis:0
         ~dyn_value:(LL.Embed_index (B.fixed 1), B.iprec ())
         (B.get r_src [| B.fixed 0 |]))
  in
  p "Set_dynamic records its store entry and source read"
    (match dyn_summary.Cs.stores with
    | [ { Cs.tn_uid; rendering = Cs.Store_plain; reads; _ } ] ->
        tn_uid = r_out.Tn.uid
        && List.equal (fun (u1, c1) (u2, c2) -> u1 = u2 && c1 = c2) reads [ (r_src.Tn.uid, 1) ]
    | _ -> false);
  let vec_unop_summary =
    render_pure ~name:"rc_set_from_vec" ~nodes:[ r_out ]
      (B.set_from_vec ~length:4 ~vec_unop:Ops.Uint4x32_to_prec_uniform r_out
         [| B.fixed 0 |]
         (LL.Constant_bits 0L, Ops.uint4x32))
  in
  p "Set_from_vec records its store entry"
    (match vec_unop_summary.Cs.stores with
    | [ { Cs.tn_uid; rendering = Cs.Store_plain; _ } ] -> tn_uid = r_out.Tn.uid
    | _ -> false);
  let packed_summary =
    render_pure ~vec_bytes:16 ~name:"rc_try_vectorize" ~nodes:[ r_out; r_src ]
      (B.loop ~upto:15 ~axis:LL.Vectorized ri
         (B.set r_out [| B.iter ri |] (B.get r_src [| B.iter ri |])))
  in
  p "try_vectorize records its packed main-loop store beside Render_simd_elementwise"
    ((match packed_summary.Cs.loops with
       | [ { Cs.rendering = Cs.Render_simd_elementwise { lanes = 4 }; _ } ] -> true
       | _ -> false)
    && List.count packed_summary.Cs.stores ~f:(fun e ->
           e.Cs.tn_uid = r_out.Tn.uid && Cs.equal_store_rendering e.Cs.rendering Cs.Store_plain)
       >= 2);
  (* 2. Discarded projection operand (Arg1) does not turn a plain store into Store_rmw. *)
  let proj_summary =
    render_pure ~name:"rc_arg1_proj" ~nodes:[ r_out ]
      (B.set r_out
         [| B.fixed 0 |]
         (LL.Binop (Ops.Arg1, (LL.Constant 1.0, prec), (B.get r_out [| B.fixed 0 |], prec))))
  in
  p "discarded Arg1 operand is excluded from store read counts"
    (match proj_summary.Cs.stores with
    | [ { Cs.tn_uid; rendering = Cs.Store_plain; reads = []; _ } ] -> tn_uid = r_out.Tn.uid
    | _ -> false);
  (* 3. Unconsumed zero seed preserves emission order: zero store appears before SIMD epilogue. *)
  let z_out = B.node_factory ~first_id:724_100_200 ~dims:[| 1 |] () "rc_zout" in
  B.materialize z_out;
  let z_i = B.sym () in
  let z_summary =
    render_pure ~vec_bytes:16 ~name:"rc_unconsumed_zero" ~nodes:[ z_out; r_src ]
      (B.seq (B.zero z_out)
         (B.loop ~upto:15 ~axis:LL.Vectorized z_i
            (B.set z_out
               [| B.fixed 0 |]
               (B.add (B.get z_out [| B.fixed 0 |]) (B.get r_src [| B.iter z_i |])))))
  in
  p "unconsumed zero seed records the zero clear before the SIMD reduction epilogue"
    (match z_summary.Cs.stores with
    | [
     { Cs.tn_uid = u1; rendering = Cs.Store_plain; _ };
     { Cs.tn_uid = u2; rendering = Cs.Store_simd_epilogue; _ };
    ] ->
        u1 = z_out.Tn.uid && u2 = z_out.Tn.uid
    | _ -> false);
  (* 4. Discarded debug_float rendering of a Get_dynamic index Local_scope does not duplicate
     loops. *)
  let dbg_i = B.sym () in
  let dbg_sid = { LL.tn = r_idx; scope_id = 72401 } in
  let dbg_idx_scope =
    B.scope_of dbg_sid
      (B.seq
         (LL.Set_local (dbg_sid, LL.Constant 0.0))
         (B.loop ~upto:1 dbg_i (LL.Set_local (dbg_sid, LL.Constant 1.0))))
  in
  let dbg_llc =
    B.set r_out
      [| B.fixed 0 |]
      (B.gather ~tn:r_src ~idcs:[| B.fixed 0 |] ~dyn_axis:0 ~dyn_value:(dbg_idx_scope, prec))
  in
  let saved_dbg = Utils.settings.debug_log_from_routines and saved_lvl = Utils.settings.log_level in
  let dbg_summary =
    Exn.protect
      ~f:(fun () ->
        Utils.settings.debug_log_from_routines <- true;
        Utils.settings.log_level <- 2;
        render_pure ~name:"rc_debug_gather_scope" ~nodes:[ r_out; r_src ] dbg_llc)
      ~finally:(fun () ->
        Utils.settings.debug_log_from_routines <- saved_dbg;
        Utils.settings.log_level <- saved_lvl)
  in
  p "discarded debug_float Get_dynamic definitions do not duplicate loop census entries"
    (List.length dbg_summary.Cs.loops = 1);
  (* 5. Peel provenance is scoped to the closing store occurrence: a later unpeeled Local_scope
     reusing the same scope_id reports peeled = false. *)
  let p_out = B.node_factory ~first_id:724_100_300 ~dims:[| 2 |] () "rc_pout" in
  B.materialize p_out;
  let p_k = B.sym () in
  let shared_sid = { LL.tn = p_out; scope_id = 72402 } in
  let hoisted_scope =
    B.scope_of
      ~orig_indices:[| B.fixed 0 |]
      ~mint:LL.Schedule_minted shared_sid
      (B.seq
         (LL.Set_local (shared_sid, B.get p_out [| B.fixed 0 |]))
         (LL.Set_local (shared_sid, B.add (LL.Get_local shared_sid) (B.get r_src [| B.iter p_k |]))))
  in
  let unpeeled_scope =
    B.scope_of
      ~orig_indices:[| B.fixed 1 |]
      ~mint:LL.Schedule_minted shared_sid
      (LL.Set_local (shared_sid, LL.Constant 2.0))
  in
  let reuse_summary =
    render_pure ~name:"rc_scope_reuse" ~nodes:[ p_out; r_src ]
      (B.seq
         (B.loop ~upto:3 p_k (B.set p_out [| B.fixed 0 |] hoisted_scope))
         (B.set p_out [| B.fixed 1 |] unpeeled_scope))
  in
  p "peel provenance is scoped to the closing store, not leaked to later uses of the same scope_id"
    (match reuse_summary.Cs.stores with
    | [
     { Cs.rendering = Cs.Store_from_scope { peeled = true; _ }; _ };
     { Cs.rendering = Cs.Store_from_scope { peeled = false; _ }; _ };
    ] ->
        true
    | _ -> false);
  (* 6. Body-only compile_main reads render_census_enabled live at emission time. *)
  let live_llc = B.loop ~upto:1 ri (B.set r_out [| B.iter ri |] (LL.Constant 3.0)) in
  let live_opt = optimized live_llc [ r_out ] in
  let module Live_syntax = Cs.C_syntax (Cs.Pure_C_config (struct
    let procs = [| live_opt.LL.llc |]
    let full_printf_support = true
  end))
  in
  let pre_ctx = Live_syntax.create_render_ctx ~name:"rc_live_switch" live_opt in
  let (), inside_summary =
    Cs.with_render_census (fun () -> ignore (Live_syntax.compile_main pre_ctx live_opt.LL.llc))
  in
  ignore (Live_syntax.compile_main pre_ctx live_opt.LL.llc);
  p "create_render_ctx before with_render_census still records compile_main inside the bracket"
    (List.length inside_summary.Cs.loops = 1 && List.length inside_summary.Cs.stores = 1);
  Verdict.p_empty "compile_main after with_render_census closes appends nothing to the loop global"
    ~over:inside_summary.Cs.loops
    !(Cs.render_loop_census ());
  Verdict.p_empty "compile_main after with_render_census closes appends nothing to the store global"
    ~over:inside_summary.Cs.stores
    !(Cs.render_store_census ());
  (* 7. Sibling Local_scope operands emitted in Scope_order.plan order (sorted by scope_id) report
     their loop census entries in emission order, not operand traversal order. *)
  let sid_hi = { LL.tn = r_idx; scope_id = 72412 }
  and sid_lo = { LL.tn = r_idx; scope_id = 72411 } in
  let l_hi = B.sym () and l_lo = B.sym () in
  let scope_hi =
    B.scope_of sid_hi
      (B.seq
         (LL.Set_local (sid_hi, LL.Constant 0.0))
         (B.loop ~upto:1 ~axis:LL.Unrolled l_hi
            (LL.Set_local (sid_hi, B.add (LL.Get_local sid_hi) (B.get r_src [| B.iter l_hi |])))))
  in
  let scope_lo =
    B.scope_of sid_lo
      (B.seq
         (LL.Set_local (sid_lo, LL.Constant 0.0))
         (B.loop ~upto:2 ~axis:LL.Serial l_lo
            (LL.Set_local (sid_lo, B.add (LL.Get_local sid_lo) (B.get r_src [| B.iter l_lo |])))))
  in
  let sibling_summary =
    render_pure ~name:"rc_sibling_scope_order" ~nodes:[ r_out; r_src ]
      (B.set r_out [| B.fixed 0 |] (B.add scope_hi scope_lo))
  in
  p "sibling Local_scope operands record loop census entries in Scope_order emission order"
    (match sibling_summary.Cs.loops with
    | [
     { Cs.rendering = Cs.Render_serial; _ }; { Cs.rendering = Cs.Render_unrolled { copies = 2 }; _ };
    ] ->
        true
    | _ -> false);
  (* 8. A shared Local_scope referenced twice in one statement is emitted once by Scope_order.plan,
     so its reads are counted once rather than per operand occurrence. *)
  let dup_summary =
    render_pure ~name:"rc_shared_scope_reads" ~nodes:[ r_out; r_src ]
      (B.set r_out [| B.fixed 0 |] (B.add scope_lo scope_lo))
  in
  p "shared Local_scope referenced twice in one statement counts its reads once"
    (match dup_summary.Cs.stores with
    | [ { Cs.tn_uid; rendering = Cs.Store_plain; reads; _ } ] ->
        tn_uid = r_out.Tn.uid
        && List.equal (fun (u1, c1) (u2, c2) -> u1 = u2 && c1 = c2) reads [ (r_src.Tn.uid, 1) ]
    | _ -> false)
