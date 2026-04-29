open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module LL = Ir.Low_level

let () =
  (* Create tensor nodes using the real Tnode.create API *)
  let tn_src =
    Tn.create (Tn.Default Ops.single) ~id:1 ~label:[ "src" ] ~unpadded_dims:(lazy [| 4 |])
      ~padding:(lazy None) ()
  in
  let tn_out1 =
    Tn.create (Tn.Default Ops.single) ~id:2 ~label:[ "out1" ] ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None) ()
  in
  let tn_out2 =
    Tn.create (Tn.Default Ops.single) ~id:3 ~label:[ "out2" ] ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None) ()
  in

  (* Build scope_id records directly (type exposed in .mli) *)
  let scope1 : LL.scope_id = { tn = tn_src; scope_id = 100 } in
  let scope2 : LL.scope_id = { tn = tn_src; scope_id = 200 } in

  (* Use separate iterator symbols so cse_equal_scalar sees them as alpha-equivalent *)
  let idx1 = Idx.get_symbol () in
  let idx2 = Idx.get_symbol () in

  (* Two alpha-equivalent Local_scope computations (same structure, different ids/symbols) *)
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

  (* Two sibling Set statements with equivalent Local_scope nodes *)
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

  (* 1. Print before: to_doc (%cd-style printer in low_level.ml) *)
  Stdio.printf "=== Before hoist (to_doc) ===\n";
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (LL.to_doc () llc);
  Stdio.printf "\n\n";

  let result = LL.hoist_cross_statement_cse llc in

  (* 2. Print after: to_doc *)
  Stdio.printf "=== After hoist (to_doc) ===\n";
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (LL.to_doc () result);
  Stdio.printf "\n\n";

  (* 3. Print after: to_doc_cstyle (C-like printer in low_level.ml) *)
  Stdio.printf "=== After hoist (to_doc_cstyle) ===\n";
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (LL.to_doc_cstyle () result);
  Stdio.printf "\n\n";

  (* 4. Print after: c_syntax.ml pp_ll (actual backend codegen path) *)
  let optimized : LL.optimized =
    {
      traced_store = Hashtbl.create (module Ir.Tnode);
      optimize_ctx = { computations = Hashtbl.create (module Ir.Tnode) };
      llc = result;
      merge_node = None;
    }
  in
  let module Syntax =
    Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
      type buffer_ptr = unit Ctypes.ptr

      let use_host_memory = None
      let procs = [| optimized |]
      let full_printf_support = true
    end))
  in
  Stdio.printf "=== After hoist (c_syntax pp_ll) ===\n";
  PPrint.ToChannel.pretty 0.9 110 Stdio.stdout (Syntax.compile_main result);
  Stdio.printf "\n%!"
