(* gh-ocannl-343: structural unit tests for [Low_level.rewrite_one_hot_reductions].

   We hand-build the lowered low-level IR for the one-hot embedding reduction [result[b,d] = sum_k
   (k == ids[b]) * C[k,d]] in both the scalar-local (virtualized) and the materialized accumulator
   shapes, and assert the pass collapses the vocabulary loop into a guarded [Get_dynamic]. Negative
   cases (loop variable used twice / under an affine index, partial loop bounds) must be left
   unchanged. *)

open Base
module Idx = Ir.Indexing
module LL = Ir.Low_level
module B = Ll_builders
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

(* Count [Get_dynamic] occurrences and surviving [For_loop] nodes in a proc tree. *)
let summarize (llc : LL.t) : int * int =
  let dyn = ref 0 and loops = ref 0 in
  let rec proc (llc : LL.t) =
    match llc with
    | LL.Noop | LL.Comment _ | LL.Staged_compilation _ | LL.Zero_out _ | LL.Declare_local _
    | LL.Workgroup_barrier | LL.Tile_mma _ ->
        ()
    | LL.Seq (a, b) ->
        proc a;
        proc b
    | LL.For_loop { body; _ } ->
        Int.incr loops;
        proc body
    | LL.Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> scal c.LL.init);
        proc body
    | LL.Set { llsc; _ } -> scal llsc
    | LL.Set_dynamic { dyn_value = v, _; llsc; _ } ->
        scal v;
        scal llsc
    | LL.Set_from_vec { arg = s, _; _ } -> scal s
    | LL.Set_local (_, s) -> scal s
    | LL.If { cond = c, _; body } ->
        scal c;
        proc body
  and scal (s : LL.scalar_t) =
    match s with
    | LL.Get_dynamic { dyn_value = v, _; _ } ->
        Int.incr dyn;
        scal v
    | LL.Local_scope { body; _ } -> proc body
    | LL.Ternop (_, (a, _), (b, _), (c, _)) ->
        scal a;
        scal b;
        scal c
    | LL.Binop (_, (a, _), (b, _)) ->
        scal a;
        scal b
    | LL.Unop (_, (a, _)) -> scal a
    | LL.Get _ | LL.Get_local _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _
    | LL.Embed_index _ ->
        ()
  in
  proc llc;
  (!dyn, !loops)

open Verdict.Claims

(* Build [result[b,d]] as a [Set] whose scalar is a scalar-local one-hot reduction over [k].
   [table_idcs] selects the table read; [bounds] is the loop [(from_, to_)]; [reversed] flips the
   equality operand order; [use_mul] selects the multiply form instead of the [Where] form. *)
let make_local_scope_reduction ~table ~ids ~result ~table_idcs ~vocab ~bounds ~reversed ~use_mul =
  let b = Idx.get_symbol () and d = Idx.get_symbol () and k = Idx.get_symbol () in
  let table_idcs = table_idcs k d in
  let from_, to_ = bounds in
  let iprec = Lazy.force ids.Tn.storage_prec in
  let vprec = Lazy.force table.Tn.storage_prec in
  let cmpeq =
    let kside = (LL.Embed_index (Idx.Iterator k), iprec) in
    let idside = (LL.Get (ids, [| Idx.Iterator b |]), iprec) in
    if reversed then LL.Binop (Ops.Cmpeq, idside, kside) else LL.Binop (Ops.Cmpeq, kside, idside)
  in
  let table_get = LL.Get (table, table_idcs) in
  let contribution =
    if use_mul then LL.Binop (Ops.Mul, (cmpeq, iprec), (table_get, vprec))
    else LL.Ternop (Ops.Where, (cmpeq, iprec), (table_get, vprec), (LL.Constant 0., vprec))
  in
  let id : LL.scope_id = { tn = result; scope_id = !next_id * 10 } in
  let acc = LL.Binop (Ops.Add, (LL.Get_local id, vprec), (contribution, vprec)) in
  let body =
    LL.Seq (LL.Set_local (id, LL.Constant 0.), B.loop ~from_ ~upto:to_ k (LL.Set_local (id, acc)))
  in
  ignore vocab;
  B.set result
    [| Idx.Iterator b; Idx.Iterator d |]
    (LL.Local_scope
       {
         id;
         body;
         orig_indices = [| Idx.Iterator b; Idx.Iterator d |];
         mint = LL.Inlined_computation;
       })

let plain_table_idcs k d = [| Idx.Iterator k; Idx.Iterator d |]

let () =
  let vocab = 4 and embed = 3 in
  let table = mk ~dims:[| vocab; embed |] "C" in
  let ids = mk ~dims:[| 2 |] "ids" in
  let result = mk ~dims:[| 2; embed |] "emb" in

  (* Positive: Where form. *)
  let pos_where =
    make_local_scope_reduction ~table ~ids ~result ~table_idcs:plain_table_idcs ~vocab
      ~bounds:(0, vocab - 1)
      ~reversed:false ~use_mul:false
  in
  let dyn, loops = summarize (LL.rewrite_one_hot_reductions pos_where) in
  p "scalar-local Where form rewrites to Get_dynamic" (dyn = 1);
  p "scalar-local Where form removes the vocab loop" (loops = 0);

  (* Positive: multiply form, reversed equality operands. *)
  let pos_mul =
    make_local_scope_reduction ~table ~ids ~result ~table_idcs:plain_table_idcs ~vocab
      ~bounds:(0, vocab - 1)
      ~reversed:true ~use_mul:true
  in
  let dyn, loops = summarize (LL.rewrite_one_hot_reductions pos_mul) in
  p "scalar-local multiply form (reversed Cmpeq) rewrites to Get_dynamic" (dyn = 1 && loops = 0);

  (* Negative: loop variable used twice in the table access. *)
  let neg_twice =
    make_local_scope_reduction
      ~table:(mk ~dims:[| vocab; vocab |] "Csq")
      ~ids ~result
      ~table_idcs:(fun k _d -> [| Idx.Iterator k; Idx.Iterator k |])
      ~vocab
      ~bounds:(0, vocab - 1)
      ~reversed:false ~use_mul:false
  in
  let dyn, loops = summarize (LL.rewrite_one_hot_reductions neg_twice) in
  p "double-use of loop var is not rewritten" (dyn = 0 && loops = 1);

  (* Negative: partial loop bounds (does not span the full vocabulary axis). *)
  let neg_partial =
    make_local_scope_reduction ~table ~ids ~result ~table_idcs:plain_table_idcs ~vocab
      ~bounds:(0, vocab - 2)
      ~reversed:false ~use_mul:false
  in
  let dyn, loops = summarize (LL.rewrite_one_hot_reductions neg_partial) in
  p "partial loop bounds are not rewritten" (dyn = 0 && loops = 1);

  (* Negative: affine table index at the gathered axis. *)
  let neg_affine =
    make_local_scope_reduction ~table ~ids ~result
      ~table_idcs:(fun k d -> [| Idx.Affine { symbols = [ (1, k) ]; offset = 0 }; Idx.Iterator d |])
      ~vocab
      ~bounds:(0, vocab - 1)
      ~reversed:false ~use_mul:false
  in
  let dyn, loops = summarize (LL.rewrite_one_hot_reductions neg_affine) in
  p "affine table index is not rewritten" (dyn = 0 && loops = 1)

(* gh-ocannl-466: the transposed (scatter) form — the embedding-table gradient. *)

(* Count [Set_dynamic] statements and surviving [For_loop] nodes. *)
let count_scatters (llc : LL.t) : int * int =
  let scatters = ref 0 and loops = ref 0 in
  let rec proc (llc : LL.t) =
    match llc with
    | LL.Set_dynamic _ -> Int.incr scatters
    | LL.Seq (a, b) ->
        proc a;
        proc b
    | LL.For_loop { body; _ } ->
        Int.incr loops;
        proc body
    | LL.If { body; _ } -> proc body
    | _ -> ()
  in
  proc llc;
  (!scatters, !loops)

(* Build the transposed reduction [for k in bounds: d_table[d, k] += (k == ids[b]) * g] where [g] is
   the incoming-gradient scalar. [fma] selects the fused accumulator shape; otherwise [Add (acc,
   Where (cmpeq, g, 0))]. *)
let make_transposed_loop ~d_table ~ids ~g ~lhs_idcs ~bounds ~fma =
  let b = Idx.get_symbol () and d = Idx.get_symbol () and k = Idx.get_symbol () in
  let lhs_idcs = lhs_idcs k d in
  let from_, to_ = bounds in
  let iprec = Lazy.force ids.Tn.storage_prec in
  let vprec = Lazy.force d_table.Tn.storage_prec in
  let cmpeq =
    LL.Binop
      ( Ops.Cmpeq,
        (LL.Embed_index (Idx.Iterator k), iprec),
        (LL.Get (ids, [| Idx.Iterator b |]), iprec) )
  in
  let g_scalar = g ~b ~d ~k in
  let acc_read = LL.Get (d_table, lhs_idcs) in
  let acc =
    if fma then LL.Ternop (Ops.FMA, (g_scalar, vprec), (cmpeq, iprec), (acc_read, vprec))
    else
      LL.Binop
        ( Ops.Add,
          (acc_read, vprec),
          (LL.Ternop (Ops.Where, (cmpeq, iprec), (g_scalar, vprec), (LL.Constant 0., vprec)), vprec)
        )
  in
  B.loop ~from_ ~upto:to_ k (B.set d_table lhs_idcs acc)

let () =
  let vocab = 4 and embed = 3 in
  let d_table = mk ~dims:[| embed; vocab |] "dC" in
  let ids = mk ~dims:[| 2 |] "ids2" in
  let demb = mk ~dims:[| 2; embed |] "demb" in
  let plain_lhs k d = [| Idx.Iterator d; Idx.Iterator k |] in
  let g_demb ~b ~d ~k:_ = LL.Get (demb, [| Idx.Iterator b; Idx.Iterator d |]) in

  (* Positive: fused FMA accumulator (the shape actual lowering produces). *)
  let pos_fma =
    make_transposed_loop ~d_table ~ids ~g:g_demb ~lhs_idcs:plain_lhs
      ~bounds:(0, vocab - 1)
      ~fma:true
  in
  let scatters, loops = count_scatters (LL.rewrite_one_hot_reductions pos_fma) in
  p "transposed FMA form rewrites to Set_dynamic" (scatters = 1);
  p "transposed FMA form removes the vocab loop" (loops = 0);

  (* Positive: Add-of-Where accumulator. *)
  let pos_where =
    make_transposed_loop ~d_table ~ids ~g:g_demb ~lhs_idcs:plain_lhs
      ~bounds:(0, vocab - 1)
      ~fma:false
  in
  let scatters, loops = count_scatters (LL.rewrite_one_hot_reductions pos_where) in
  p "transposed Add-of-Where form rewrites to Set_dynamic" (scatters = 1 && loops = 0);

  (* Negative: the contribution reads the scattered tensor itself — dropping the per-row iterations
     could change what it observes. *)
  let neg_reads_target =
    make_transposed_loop ~d_table ~ids
      ~g:(fun ~b:_ ~d ~k:_ -> LL.Get (d_table, [| Idx.Iterator d; Idx.Fixed_idx 0 |]))
      ~lhs_idcs:plain_lhs
      ~bounds:(0, vocab - 1)
      ~fma:true
  in
  let scatters, loops = count_scatters (LL.rewrite_one_hot_reductions neg_reads_target) in
  p "contribution reading the scattered tensor is not rewritten" (scatters = 0 && loops = 1);

  (* Negative: the contribution mentions the reduction variable. *)
  let neg_g_mentions_k =
    make_transposed_loop ~d_table ~ids
      ~g:(fun ~b ~d:_ ~k -> LL.Get (demb, [| Idx.Iterator b; Idx.Iterator k |]))
      ~lhs_idcs:plain_lhs
      ~bounds:(0, vocab - 1)
      ~fma:true
  in
  let scatters, loops = count_scatters (LL.rewrite_one_hot_reductions neg_g_mentions_k) in
  p "contribution mentioning the loop var is not rewritten" (scatters = 0 && loops = 1);

  (* Negative: partial loop bounds (does not span the full written axis). *)
  let neg_partial =
    make_transposed_loop ~d_table ~ids ~g:g_demb ~lhs_idcs:plain_lhs
      ~bounds:(0, vocab - 2)
      ~fma:true
  in
  let scatters, loops = count_scatters (LL.rewrite_one_hot_reductions neg_partial) in
  p "transposed partial bounds are not rewritten" (scatters = 0 && loops = 1);

  (* Negative: the loop variable indexes the written tensor twice. *)
  let neg_twice =
    make_transposed_loop
      ~d_table:(mk ~dims:[| vocab; vocab |] "dCsq")
      ~ids ~g:g_demb
      ~lhs_idcs:(fun k _d -> [| Idx.Iterator k; Idx.Iterator k |])
      ~bounds:(0, vocab - 1)
      ~fma:true
  in
  let scatters, loops = count_scatters (LL.rewrite_one_hot_reductions neg_twice) in
  p "double-use of loop var in the written indices is not rewritten" (scatters = 0 && loops = 1)
