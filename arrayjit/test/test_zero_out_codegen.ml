(* Regression test for gh-ocannl-420 (round 2): the c_syntax guard that elides a [Zero_out]
   loop made redundant by a declaration's [= {0}] must suppress ONLY the first-touch,
   function-scope [Zero_out] of a local node -- never a genuine re-zero.

   The [zero_initialized_by_code] flag is node-level: once the first first-touch [Zero_out]
   sets it, it stays set even after the node is written again. So a guard keyed solely on that
   flag would also drop:
     - a later [Zero_out tn] in a [Zero_out tn; Set tn; Zero_out tn] sequence, and
     - a [Zero_out tn] reached inside an iterated loop,
   both of which are real re-zeros that [= {0}] (entry-time only) does not cover.

   This builds the low-level IR directly and runs it through the actual backend codegen path
   ([C_syntax.compile_proc]), so the generated C is checked exactly. *)

open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module LL = Ir.Low_level

let () =
  (* Two local (non-virtual, non-materialized) tensor nodes. Local nodes are the ones whose
     declaration gets [= {0}] when [zero_initialized_by_code] is set. *)
  let make_local id label =
    let tn =
      Tn.create (Tn.Default Ops.single) ~id ~label:[ label ]
        ~unpadded_dims:(lazy [| 2 |])
        ~padding:(lazy None)
        ()
    in
    Tn.update_memory_mode tn Tn.Local 999;
    tn
  in
  let tn_a = make_local 1 "acc_a" in
  let tn_b = make_local 2 "acc_b" in

  let set tn v =
    LL.Set { tn; idcs = [| Idx.Fixed_idx 0 |]; llsc = LL.Constant v; debug = "" }
  in

  (* Scenario A: Zero_out; Set; Zero_out -- the FIRST Zero_out is redundant with [= {0}] and
     should be elided; the SECOND is a genuine re-zero (it discards the value just written)
     and must still emit its zeroing loop. *)
  let scenario_a = LL.Seq (LL.Zero_out tn_a, LL.Seq (set tn_a 7.0, LL.Zero_out tn_a)) in

  (* Scenario B: a first-touch Zero_out reached inside an iterated loop. Even though it is the
     first (and only) Zero_out of [tn_b], it re-runs every iteration, so [= {0}] (entry-time
     only) does not make it redundant -- the zeroing loop must be emitted. *)
  let scenario_b =
    LL.For_loop
      { index = Idx.get_symbol (); from_ = 0; to_ = 3; body = LL.Zero_out tn_b; trace_it = false }
  in

  let llc = LL.Seq (scenario_a, scenario_b) in

  (* Both nodes carry [zero_initialized_by_code = true], i.e. their declarations emit [= {0}]. *)
  let traced_store = Hashtbl.create (module Tn) in
  List.iter [ tn_a; tn_b ] ~f:(fun tn ->
      let node = LL.get_node traced_store tn in
      node.LL.zero_initialized_by_code <- true);

  let optimized : LL.optimized =
    {
      traced_store;
      optimize_ctx = { computations = Hashtbl.create (module Tn) };
      llc;
      merge_node = None;
    }
  in
  let module Syntax = Ir.C_syntax.C_syntax (Ir.C_syntax.Pure_C_config (struct
    type buffer_ptr = unit Ctypes.ptr

    let use_host_memory = None
    let procs = [| optimized |]
    let full_printf_support = true
  end)) in
  let _kparams, doc = Syntax.compile_proc ~name:"zero_out_codegen" [] optimized in
  PPrint.ToChannel.pretty 0.9 100 Stdio.stdout doc;
  Stdio.printf "\n%!"
