(* Regression test for gh-ocannl-340: Local_scope and Declare_local declarations should omit the [=
   0] initializer when the body writes the local before any read (non-recurrent case), and keep it
   when the body reads the local before the first write (recurrent/accumulator case).

   This exercises [reads_scope_before_set] directly and checks generated C via [compile_main]. *)

open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module LL = Ir.Low_level

let make_tn ~id ~label =
  Tn.create (Tn.Default Ops.single) ~id ~label:[ label ]
    ~unpadded_dims:(lazy [| 1 |])
    ~padding:(lazy None)
    ()

let make_optimized llc =
  LL.
    {
      traced_store = Hashtbl.create (module Tn);
      optimize_ctx = { computations = Hashtbl.create (module Tn) };
      llc;
      merge_node = None;
    }

let pp llc =
  let optimized = make_optimized llc in
  let module Syntax = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    type buffer_ptr = unit Ctypes.ptr

    let procs = [| optimized |]
    let full_printf_support = true
  end))
  in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout (Syntax.compile_main llc)

(* ===== reads_scope_before_set unit tests ===== *)

let () =
  let tn = make_tn ~id:1 ~label:"t" in
  let id : LL.scope_id = { tn; scope_id = 1 } in

  (* write-before-read: Set_local(id, constant) -- target written first, no read needed *)
  let body_write_first = LL.Set_local (id, LL.Constant 0.) in
  assert (not (LL.reads_scope_before_set id body_write_first));

  (* read-before-write: acc = acc + 1 -- target read in value expression *)
  let body_acc =
    LL.Set_local
      (id, LL.Binop (Ops.Add, (LL.Get_local id, Ops.single), (LL.Constant 1., Ops.single)))
  in
  assert (LL.reads_scope_before_set id body_acc);

  (* Seq: write first, then read-via-get_local -- the first Set_local dominates *)
  let body_write_then_use =
    LL.Seq
      ( LL.Set_local (id, LL.Constant 0.),
        LL.Set_local
          (id, LL.Binop (Ops.Add, (LL.Get_local id, Ops.single), (LL.Constant 1., Ops.single))) )
  in
  assert (not (LL.reads_scope_before_set id body_write_then_use));

  (* Loop with write body -- write is definite when from_ <= to_ *)
  let idx = Idx.get_symbol () in
  let loop_with_write =
    LL.For_loop
      {
        index = idx;
        from_ = 0;
        to_ = 3;
        body = LL.Set_local (id, LL.Constant 1.);
        trace_it = false;
      }
  in
  assert (not (LL.reads_scope_before_set id loop_with_write));

  (* Loop with read body -- reads always count *)
  let loop_with_read =
    LL.For_loop
      {
        index = idx;
        from_ = 0;
        to_ = 3;
        body =
          LL.Set_local
            (id, LL.Binop (Ops.Add, (LL.Get_local id, Ops.single), (LL.Constant 1., Ops.single)));
        trace_it = false;
      }
  in
  assert (LL.reads_scope_before_set id loop_with_read);

  (* Noop body: the local is never written -- needs initialization so the Local_scope expression
     value is not an uninitialized C local. scan returns `Neither`, which must map to true (needs
     init). *)
  assert (LL.reads_scope_before_set id LL.Noop);

  (* Empty loop (from_ > to_): write is NOT definite (loop never runs), so the local may be
     uninitialized after the body -- needs initialization. scan returns `Neither` for the same
     reason as Noop, so this must also return true. *)
  let empty_loop_write =
    LL.For_loop
      {
        index = idx;
        from_ = 5;
        to_ = 0;
        body = LL.Set_local (id, LL.Constant 1.);
        trace_it = false;
      }
  in
  assert (LL.reads_scope_before_set id empty_loop_write);

  (* Empty loop followed by accumulator read: scan returns `Read` (the accumulator step reads the
     local after the non-definite empty-loop write), which also maps to true. A mutation that treats
     the empty-loop write as definite would make scan return `Written` after the Seq's first
     component and suppress this `Read`, failing here. *)
  let empty_loop_then_acc =
    LL.Seq
      ( empty_loop_write,
        LL.Set_local
          (id, LL.Binop (Ops.Add, (LL.Get_local id, Ops.single), (LL.Constant 1., Ops.single))) )
  in
  assert (LL.reads_scope_before_set id empty_loop_then_acc);

  Stdio.printf "reads_scope_before_set: all unit assertions passed\n%!"

(* ===== C codegen: inline Local_scope, write-before-read (no init) ===== *)

let () =
  let tn_src = make_tn ~id:2 ~label:"src" in
  let tn_out = make_tn ~id:3 ~label:"out" in
  let id : LL.scope_id = { tn = tn_src; scope_id = 10 } in
  (* Body: unconditional assignment from tensor -- local is written before any read. *)
  let body = LL.Set_local (id, LL.Get (tn_src, [| Idx.Fixed_idx 0 |])) in
  let local_scope = LL.Local_scope { id; body; orig_indices = [||] } in
  let llc =
    LL.Set { tn = tn_out; idcs = [| Idx.Fixed_idx 0 |]; llsc = local_scope; debug = "write-first" }
  in
  Stdio.printf "=== Local_scope write-before-read (no init): ===\n";
  pp llc;
  Stdio.printf "\n%!"

(* ===== C codegen: inline Local_scope, read-before-write (needs init) ===== *)

let () =
  let tn_src = make_tn ~id:4 ~label:"src2" in
  let tn_out = make_tn ~id:5 ~label:"out2" in
  let id : LL.scope_id = { tn = tn_src; scope_id = 20 } in
  (* Body: accumulator loop -- local is read before written. *)
  let idx = Idx.get_symbol () in
  let body =
    LL.For_loop
      {
        index = idx;
        from_ = 0;
        to_ = 3;
        body =
          LL.Set_local
            ( id,
              LL.Binop
                ( Ops.Add,
                  (LL.Get_local id, Ops.single),
                  (LL.Get (tn_src, [| Idx.Iterator idx |]), Ops.single) ) );
        trace_it = false;
      }
  in
  let local_scope = LL.Local_scope { id; body; orig_indices = [||] } in
  let llc =
    LL.Set { tn = tn_out; idcs = [| Idx.Fixed_idx 0 |]; llsc = local_scope; debug = "accumulator" }
  in
  Stdio.printf "=== Local_scope read-before-write (needs init): ===\n";
  pp llc;
  Stdio.printf "\n%!"

(* ===== C codegen: Noop body (needs init -- Neither case) ===== *)
(* A Local_scope whose body is Noop never writes the local, so the expression value
   would be an uninitialized C local without the initializer. *)

let () =
  let tn_src = make_tn ~id:14 ~label:"src_noop" in
  let tn_out = make_tn ~id:15 ~label:"out_noop" in
  let id : LL.scope_id = { tn = tn_src; scope_id = 60 } in
  let local_scope = LL.Local_scope { id; body = LL.Noop; orig_indices = [||] } in
  let llc =
    LL.Set { tn = tn_out; idcs = [| Idx.Fixed_idx 0 |]; llsc = local_scope; debug = "noop-body" }
  in
  Stdio.printf "=== Local_scope Noop body (needs init, Neither case): ===\n";
  pp llc;
  Stdio.printf "\n%!"

(* ===== C codegen: empty-loop write then accumulator (needs init) ===== *)
(* The empty loop body write is not definite, so the following accumulator read is
   read-before-write -> declaration must keep the = 0 initializer. *)

let () =
  let tn_src = make_tn ~id:12 ~label:"src_el" in
  let tn_out = make_tn ~id:13 ~label:"out_el" in
  let id : LL.scope_id = { tn = tn_src; scope_id = 50 } in
  let idx = Idx.get_symbol () in
  let empty_loop =
    LL.For_loop
      {
        index = idx;
        from_ = 5;
        to_ = 0;
        body = LL.Set_local (id, LL.Constant 0.);
        trace_it = false;
      }
  in
  let acc_step =
    LL.Set_local
      (id, LL.Binop (Ops.Add, (LL.Get_local id, Ops.single), (LL.Constant 1., Ops.single)))
  in
  let body = LL.Seq (empty_loop, acc_step) in
  let local_scope = LL.Local_scope { id; body; orig_indices = [||] } in
  let llc =
    LL.Set
      {
        tn = tn_out;
        idcs = [| Idx.Fixed_idx 0 |];
        llsc = local_scope;
        debug = "empty-loop-then-acc";
      }
  in
  Stdio.printf "=== Local_scope empty-loop-write then accumulator (needs init): ===\n";
  pp llc;
  Stdio.printf "\n%!"

(* ===== C codegen: hoisted Declare_local, write-before-read (no init) ===== *)

let () =
  let tn_src = make_tn ~id:6 ~label:"src3" in
  let tn_out1 = make_tn ~id:7 ~label:"out3" in
  let tn_out2 = make_tn ~id:8 ~label:"out4" in
  let scope1 : LL.scope_id = { tn = tn_src; scope_id = 30 } in
  let scope2 : LL.scope_id = { tn = tn_src; scope_id = 31 } in
  (* Simple write-before-read local scope (shared between two statements) *)
  let make_scope sid orig_indices =
    LL.Local_scope
      { id = sid; body = LL.Set_local (sid, LL.Get (tn_src, [| Idx.Fixed_idx 0 |])); orig_indices }
  in
  let stmt1 =
    LL.Set { tn = tn_out1; idcs = [| Idx.Fixed_idx 0 |]; llsc = make_scope scope1 [||]; debug = "" }
  in
  let stmt2 =
    LL.Set { tn = tn_out2; idcs = [| Idx.Fixed_idx 0 |]; llsc = make_scope scope2 [||]; debug = "" }
  in
  let hoisted = LL.hoist_cross_statement_cse (LL.Seq (stmt1, stmt2)) in
  Stdio.printf "=== Hoisted Declare_local write-before-read (no init): ===\n";
  pp hoisted;
  Stdio.printf "\n%!"

(* ===== C codegen: hoisted Declare_local, read-before-write (needs init) ===== *)

let () =
  let tn_src = make_tn ~id:9 ~label:"src4" in
  let tn_out1 = make_tn ~id:10 ~label:"out5" in
  let tn_out2 = make_tn ~id:11 ~label:"out6" in
  let scope1 : LL.scope_id = { tn = tn_src; scope_id = 40 } in
  let scope2 : LL.scope_id = { tn = tn_src; scope_id = 41 } in
  let idx1 = Idx.get_symbol () in
  let idx2 = Idx.get_symbol () in
  (* Accumulator-style local scope (shared between two statements) *)
  let make_acc_scope sid idx =
    LL.Local_scope
      {
        id = sid;
        body =
          LL.For_loop
            {
              index = idx;
              from_ = 0;
              to_ = 3;
              body =
                LL.Set_local
                  ( sid,
                    LL.Binop
                      ( Ops.Add,
                        (LL.Get_local sid, Ops.single),
                        (LL.Get (tn_src, [| Idx.Iterator idx |]), Ops.single) ) );
              trace_it = false;
            };
        orig_indices = [||];
      }
  in
  let stmt1 =
    LL.Set
      { tn = tn_out1; idcs = [| Idx.Fixed_idx 0 |]; llsc = make_acc_scope scope1 idx1; debug = "" }
  in
  let stmt2 =
    LL.Set
      { tn = tn_out2; idcs = [| Idx.Fixed_idx 0 |]; llsc = make_acc_scope scope2 idx2; debug = "" }
  in
  let hoisted = LL.hoist_cross_statement_cse (LL.Seq (stmt1, stmt2)) in
  Stdio.printf "=== Hoisted Declare_local read-before-write (needs init): ===\n";
  pp hoisted;
  Stdio.printf "\n%!"
