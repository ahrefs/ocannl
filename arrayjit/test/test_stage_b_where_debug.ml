(* task-9658aac9: non-vacuous regression for the Stage B unit-solve [Where] debug-OOB guard.

   Stage B unit-coefficient solving folds the producer read into
     [Ternop (Ops.Where, (range_cond, index_prec), (Get producer solved_idx, value_prec),
              (Get_local id, value_prec))]
   (see [Low_level.inline_computation]). For a non-matching kept-loop iteration [solved_idx] can be
   out of bounds; the runtime C ternary short-circuits, but debug value-logging
   ([C_syntax.debug_float]) used to emit all three branch reads as unconditional printf arguments,
   dereferencing the producer out of bounds. In the real virtualized flow that [Where] sits inside a
   non-logged [Local_scope] ([log_set_locals:false]), so it never reaches [debug_float]; here we
   build the identical shapes in a *logged top-level [Set]* and run them through the actual backend
   codegen ([C_syntax.compile_proc]) -- a pure C-syntax path that emits the same C on every backend,
   so the test is backend-independent -- with debug value-logging enabled, then assert each
   conditionally-evaluated array read is gated by its (negated) condition. Reverting the
   [debug_float] guard makes the assertions fail.

   The guard wrapper emits each gated read as [(<cond> ? ident[..] : 0)]; the runtime [pp_scalar]
   ternary instead renders [? ( ident[..])] / [: ( ident[..])] (note the [( ] after [?]/[:]). So the
   substring ["? ident["] (no inner paren) matches ONLY the guarded debug printf argument, never the
   runtime ternary -- a precise, loop-symbol-independent signal that the read short-circuits. *)

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

(* Run a hand-built [Low_level.t] through the real backend codegen and return the generated C. Pure
   [C_syntax] -- no [Context]/backend selection -- so the output is identical on every backend. *)
let compile_to_c ~name llc =
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
  let _kparams, doc = Syntax.compile_proc ~name [] optimized in
  doc_to_string doc

let index_prec = Ops.index_prec ()

let lt a b =
  LL.Binop (Ops.Cmplt, (LL.Embed_index a, index_prec), (LL.Embed_index b, index_prec))

let () =
  Utils.set_log_level 2;
  Utils.settings.debug_log_from_routines <- true

(* === Case 1: the Stage B unit-solve guard shape itself === *)
let () =
  let producer = mk ~id:1 ~label:"producer" ~dims:[| 6 |] in
  let out = mk ~id:2 ~label:"outc" ~dims:[| 6 |] in
  let t = Idx.get_symbol () in
  (* Else-branch local: the init fallback the producer's [Zero_out]/prepended-init emits. *)
  let id : LL.scope_id = { tn = mk ~id:3 ~label:"init_local" ~dims:[| 1 |]; scope_id = 1 } in
  (* A range guard like the [And] of two [Cmplt] bounds [try_unit_solve] emits. *)
  let range_cond =
    LL.Binop
      ( Ops.And,
        (lt (Idx.Iterator t) (Idx.Fixed_idx 4), index_prec),
        (lt (Idx.Fixed_idx 0) (Idx.Fixed_idx 5), index_prec) )
  in
  (* The producer read at the unit-solved index; [producer[t]] stands in for the solved index. *)
  let guarded =
    LL.Ternop
      ( Ops.Where,
        (range_cond, index_prec),
        (LL.Get (producer, [| Idx.Iterator t |]), Ops.single),
        (LL.Get_local id, Ops.single) )
  in
  let body =
    LL.Set { tn = out; idcs = [| Idx.Iterator t |]; llsc = guarded; debug = "unit-solve guard" }
  in
  let llc =
    LL.Seq
      ( LL.Declare_local { id; needs_init = true },
        LL.For_loop { index = t; from_ = 0; to_ = 5; body; trace_it = false } )
  in
  let c = compile_to_c ~name:"stage_b_where_debug" llc in
  (* then-branch producer read short-circuited on the range condition: [(<cond> ? producer[t] : 0)].
     Mutation: reverting the [debug_float] guard wrap drops the [? producer[] arg form. *)
  p "Stage B unit-solve producer read is gated by the range condition in debug log"
    (String.is_substring c ~substring:"? producer[");
  (* AC2 fidelity: the annotated display still shows both branch sub-values; the else-branch
     [Get_local] init fallback is logged directly (not an array dereference, so not -- and need not be
     -- guarded). *)
  p "Stage B debug log preserves the annotated Where branch displays"
    (String.is_substring c ~substring:"producer[%u]{=%g}"
    && String.is_substring c ~substring:"init_local{=%g}")

(* === Case 2: symmetric guarding -- array reads in BOTH branches === *)
let () =
  let athen = mk ~id:11 ~label:"athen2" ~dims:[| 4 |] in
  let belse = mk ~id:12 ~label:"belse2" ~dims:[| 4 |] in
  let out = mk ~id:13 ~label:"out2" ~dims:[| 4 |] in
  let t = Idx.get_symbol () in
  let where =
    LL.Ternop
      ( Ops.Where,
        (lt (Idx.Iterator t) (Idx.Fixed_idx 2), index_prec),
        (LL.Get (athen, [| Idx.Iterator t |]), Ops.single),
        (LL.Get (belse, [| Idx.Iterator t |]), Ops.single) )
  in
  let body = LL.Set { tn = out; idcs = [| Idx.Iterator t |]; llsc = where; debug = "symmetric" } in
  let llc = LL.For_loop { index = t; from_ = 0; to_ = 3; body; trace_it = false } in
  let c = compile_to_c ~name:"symmetric_where" llc in
  (* then read gated by [cond]; else read gated by [!cond]. Both branch reads are array
     dereferences, so both must short-circuit. *)
  p "symmetric Where then-branch read is gated by the condition"
    (String.is_substring c ~substring:"? athen2[");
  p "symmetric Where else-branch read is gated by the negated condition"
    (String.is_substring c ~substring:"(!(" && String.is_substring c ~substring:"? belse2[")

(* === Case 3: a nested Where whose inner CONDITION contains an array read, under an outer guard ===
   The inner condition's [Get] is reached only when the outer guard holds, so its dereference must be
   gated by the outer guard (Codex P2: the condition must inherit the enclosing guard). *)
let () =
  let athen = mk ~id:21 ~label:"athen3" ~dims:[| 4 |] in
  let condrd = mk ~id:22 ~label:"condrd3" ~dims:[| 4 |] in
  let out = mk ~id:23 ~label:"out3" ~dims:[| 4 |] in
  let id1 : LL.scope_id = { tn = mk ~id:24 ~label:"loc3a" ~dims:[| 1 |]; scope_id = 31 } in
  let id2 : LL.scope_id = { tn = mk ~id:25 ~label:"loc3b" ~dims:[| 1 |]; scope_id = 32 } in
  let t = Idx.get_symbol () in
  let outer_cond = lt (Idx.Iterator t) (Idx.Fixed_idx 2) in
  (* Inner condition is a value comparison containing an array read [condrd3[t]]. *)
  let inner_cond =
    LL.Binop (Ops.Cmplt, (LL.Get (condrd, [| Idx.Iterator t |]), Ops.single), (LL.Constant 1., Ops.single))
  in
  let inner_where =
    LL.Ternop
      ( Ops.Where,
        (inner_cond, Ops.single),
        (LL.Get (athen, [| Idx.Iterator t |]), Ops.single),
        (LL.Get_local id2, Ops.single) )
  in
  let outer_where =
    LL.Ternop
      (Ops.Where, (outer_cond, index_prec), (inner_where, Ops.single), (LL.Get_local id1, Ops.single))
  in
  let body = LL.Set { tn = out; idcs = [| Idx.Iterator t |]; llsc = outer_where; debug = "nested" } in
  let llc =
    LL.Seq
      ( LL.Declare_local { id = id1; needs_init = true },
        LL.Seq
          ( LL.Declare_local { id = id2; needs_init = true },
            LL.For_loop { index = t; from_ = 0; to_ = 3; body; trace_it = false } ) )
  in
  let c = compile_to_c ~name:"nested_where" llc in
  (* The inner condition's read [condrd3[t]] is gated by the OUTER guard: [(<outer_cond> ? condrd3[t]
     : 0)]. Without passing the enclosing guard into the condition this would be a bare unconditional
     printf argument (no [? condrd3[] form). Mutation: reverting the [debug_float] condition guard
     drops this. *)
  p "nested Where inner-condition array read is gated by the outer guard"
    (String.is_substring c ~substring:"? condrd3[")
