(* task-9658aac9: non-vacuous regression for the Stage B unit-solve [Where] debug-OOB guard.

   Stage B unit-coefficient solving folds the producer read into
     [Ternop (Ops.Where, (range_cond, index_prec), (Get producer solved_idx, value_prec),
              (Get_local id, value_prec))]
   (see [Low_level.inline_computation]). For a non-matching kept-loop iteration [solved_idx] can be
   out of bounds; the runtime C ternary short-circuits, but debug value-logging
   ([C_syntax.debug_float]) used to emit all three branch reads as unconditional printf arguments,
   dereferencing the producer out of bounds. In the real virtualized flow that [Where] sits inside a
   non-logged [Local_scope] ([log_set_locals:false]), so it never reaches [debug_float]; here we
   build the identical shape in a *logged top-level [Set]* and run it through the actual backend
   codegen ([C_syntax.compile_proc]) with debug value-logging enabled, then assert the producer read
   is gated by the range condition. Reverting the [debug_float] guard makes the assertion fail. *)

open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module LL = Ir.Low_level

let p name b = Stdio.printf "%s: %b\n" name b

let doc_to_string doc =
  let b = Buffer.create 4096 in
  PPrint.ToBuffer.pretty 0.9 100 b doc;
  Buffer.contents b

let mk ~id ~label ~dims =
  let tn =
    Tn.create (Tn.Default Ops.single) ~id ~label:[ label ] ~unpadded_dims:(lazy dims)
      ~padding:(lazy None) ()
  in
  Tn.update_memory_mode tn Tn.Materialized 999;
  tn

let () =
  Utils.set_log_level 2;
  Utils.settings.debug_log_from_routines <- true;
  let producer = mk ~id:1 ~label:"producer" ~dims:[| 6 |] in
  let out = mk ~id:2 ~label:"outc" ~dims:[| 6 |] in
  let index_prec = Ops.index_prec () in
  let t = Idx.get_symbol () in
  (* Else-branch local: the init fallback the producer's [Zero_out]/prepended-init emits. *)
  let id : LL.scope_id = { tn = mk ~id:3 ~label:"init_local" ~dims:[| 1 |]; scope_id = 1 } in
  let lt a b = LL.Binop (Ops.Cmplt, (LL.Embed_index a, index_prec), (LL.Embed_index b, index_prec)) in
  (* A range guard like the [And] of two [Cmplt] bounds [try_unit_solve] emits. *)
  let range_cond =
    LL.Binop
      ( Ops.And,
        (lt (Idx.Iterator t) (Idx.Fixed_idx 4), index_prec),
        (lt (Idx.Fixed_idx 0) (Idx.Fixed_idx 5), index_prec) )
  in
  (* The producer read at the unit-solved index; here [producer[t]] standing in for the solved index
     that uses the kept-loop symbol. *)
  let producer_read = LL.Get (producer, [| Idx.Iterator t |]) in
  let guarded =
    LL.Ternop
      ( Ops.Where,
        (range_cond, index_prec),
        (producer_read, Ops.single),
        (LL.Get_local id, Ops.single) )
  in
  let body = LL.Set { tn = out; idcs = [| Idx.Iterator t |]; llsc = guarded; debug = "unit-solve guard" } in
  let llc =
    LL.Seq
      ( LL.Declare_local { id; needs_init = true },
        LL.For_loop { index = t; from_ = 0; to_ = 5; body; trace_it = false } )
  in
  let optimized : LL.optimized =
    {
      traced_store = Hashtbl.create (module Tn);
      optimize_ctx = { computations = Hashtbl.create (module Tn) };
      llc;
      merge_node = None;
    }
  in
  let module Syntax =
    Ir.C_syntax.C_syntax
      (Ir.C_syntax.Pure_C_config (struct
        type buffer_ptr = unit Ctypes.ptr

        let procs = [| optimized |]
        let full_printf_support = true
      end))
  in
  let _kparams, doc = Syntax.compile_proc ~name:"stage_b_where_debug" [] optimized in
  let c = doc_to_string doc in
  (* The debug value-log [fprintf] for [outc] renders the [Where] value. After the fix the
     then-branch producer read is short-circuited on the range condition:
       (((i1 < 4) && (0 < 5)) ? producer[i1] : 0)
     The " ? producer[i1] : 0)" form is produced only by the guard wrapper -- the runtime ternary's
     else branch is the init local [v1_init_local], never a literal [0], so this substring is a
     precise signal that the producer read is gated. Before the fix the producer read was emitted as
     a bare unconditional printf argument ([producer[i1]] with no [? ... : 0)] wrapper), which is the
     out-of-bounds dereference this task closes. *)
  p "Stage B unit-solve producer read is gated by the range condition in debug log"
    (String.is_substring c ~substring:"? producer[i1] : 0)");
  (* AC2 fidelity: the annotated display still shows both branch sub-values; the else-branch
     [Get_local] init fallback is logged directly (it is not an array dereference, so it is not -- and
     need not be -- guarded). *)
  p "Stage B debug log preserves the annotated Where branch displays"
    (String.is_substring c ~substring:"producer[%u]{=%g}"
    && String.is_substring c ~substring:"v1_init_local{=%g}")
