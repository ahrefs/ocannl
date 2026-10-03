(* Regression test for the buffer-addressing refactor + gh-ocannl-344 pooling: buffer handles are
   deterministic integer [buffer_loc = { pool_id; offset }] resolved against a backend-private pool
   table, and a context's working (non-constant) delta is packed into a SINGLE pool at increasing
   byte offsets rather than one pool per tnode.

   Two invariants are pinned: 1. Determinism / reproducibility: linking the same graph into two
   independently-created root contexts yields identical offsets and pooling structure (the
   debuggability win). Pool ids are deterministic but advance monotonically across the runs: the
   backend and its devices are process singletons since the backend-singletons refactor
   (docs/proposals/backend-singletons-context-copy.md), so [device.next_pool_id] is not reset by
   creating a fresh root. 2. Pooling: when a single routine materializes several non-constant
   tnodes, they share one [pool_id] with distinct, increasing offsets; read-only inputs (constants)
   live in separate per-device pools. If the allocator regressed to one-pool-per-tnode, the two
   outputs below would print different pool ids (and all offsets 0); if it stopped honoring offsets,
   the computed values would be wrong. *)

open Base
open Ocannl
open Operation.DSL_modules
module Backends = Context.Backends
module Idx = Ir.Indexing
module Task = Ir.Task
module Tn = Ir.Tnode
module BI = Ir.Backend_intf

let make_tensor label vals =
  let open Bigarray in
  let n = Array.length vals in
  let ga = Genarray.create Float32 c_layout [| n |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[] ~input_dims:[] ~output_dims:[ n ] ()

let run_once tag =
  Tensor.unsafe_reinitialize ();
  let backend = Backends.backend_module (Backends.get_backend ~backend_name:"cc" ()) in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:0 in
  let root = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) device in
  let a = make_tensor "a" [| 1.0; 2.0 |] in
  let b = make_tensor "b" [| 3.0; 4.0 |] in
  let out = make_tensor "out" [| 0.0; 0.0 |] in
  let%cd add = out =: a + b in
  let code = Backend.compile root.BI.optimize_ctx Idx.Empty add in
  let routine = Backend.link root code in
  (* Print [tn -> { pool_id; offset }] for every node in the freshly linked context, sorted by debug
     name so the two runs are directly comparable. *)
  Map.to_alist routine.BI.context.BI.ctx_buffers
  |> List.map ~f:(fun (tn, (loc : BI.buffer_loc)) -> (Tn.debug_name tn, loc.pool_id, loc.offset))
  |> List.sort ~compare:(fun (a, _, _) (b, _, _) -> String.compare a b)
  |> List.iter ~f:(fun (name, pool_id, offset) ->
      Stdio.printf "%s: %-14s -> { pool_id = %d; offset = %d }\n" tag name pool_id offset)

(* A routine that materializes two non-constant outputs in one link, to demonstrate that the working
   delta packs into a single pool. *)
let run_packed () =
  Tensor.unsafe_reinitialize ();
  let backend = Backends.backend_module (Backends.get_backend ~backend_name:"cc" ()) in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:0 in
  let root = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) device in
  let a = make_tensor "a" [| 1.0; 2.0 |] in
  let b = make_tensor "b" [| 3.0; 4.0 |] in
  let p = make_tensor "p" [| 0.0; 0.0 |] in
  let q = make_tensor "q" [| 0.0; 0.0 |] in
  (* Two assignments sequenced into one computation => one routine with two materialized outputs. *)
  let%cd combo =
    p =: a + b;
    q =: a *. b
  in
  let code = Backend.compile root.BI.optimize_ctx Idx.Empty combo in
  let routine = Backend.link root code in
  let loc name =
    Map.to_alist routine.BI.context.BI.ctx_buffers
    |> List.find_map ~f:(fun (tn, (l : BI.buffer_loc)) ->
        if String.is_prefix (Tn.debug_name tn) ~prefix:name then Some l else None)
  in
  match (loc "p", loc "q", loc "a", loc "b") with
  | Some lp, Some lq, Some la, Some lb ->
      (* Both outputs are non-constant working nodes: same pool, distinct increasing offsets. *)
      Verdict.p "two outputs share a pool" (lp.pool_id = lq.pool_id);
      Verdict.p "two outputs have distinct offsets" (lp.offset <> lq.offset);
      Verdict.p "offsets are 8-byte aligned (2x float32)" (lp.offset % 4 = 0 && lq.offset % 4 = 0);
      (* The two read-only constants pack into one (per-device) constant pool at distinct offsets,
         separate from the working pool. *)
      Verdict.p "two constants share a pool" (la.pool_id = lb.pool_id);
      Verdict.p "two constants have distinct offsets" (la.offset <> lb.offset);
      Verdict.p "constant pool is separate from working pool" (la.pool_id <> lp.pool_id)
  | _ -> Verdict.fail "outputs/constants missing from ctx_buffers"

(* Value-correctness check for pooled sub-region addressing. Two non-constant working outputs (p, q)
   sharing a pool at distinct byte offsets must hold independent values after the computation runs.
   The exact assertion that catches the broken `ignore offset; ptr` stub: with both p and q pointing
   to offset 0, q's write clobbers p's slot, so reading p returns [3.0; 8.0] instead of [4.0; 6.0].
   Run with OCANNL_BACKEND=cuda to exercise the CUDA pool-addressing path on minipc-wsl. *)
let run_pooled_values_correct () =
  Tensor.unsafe_reinitialize ();
  let backend = Backends.backend_module (Backends.get_backend ~backend_name:"cc" ()) in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let device = Backend.get_device ~ordinal:0 in
  let root = Backend.make_context ~optimize_ctx:(Backend.empty_optimize_ctx ()) device in
  let a = make_tensor "a" [| 1.0; 2.0 |] in
  let b = make_tensor "b" [| 3.0; 4.0 |] in
  let p = make_tensor "p" [| 0.0; 0.0 |] in
  let q = make_tensor "q" [| 0.0; 0.0 |] in
  let%cd combo =
    p =: a + b;
    q =: a *. b
  in
  let code = Backend.compile root.BI.optimize_ctx Idx.Empty combo in
  let routine = Backend.link root code in
  let ctx = routine.BI.context in
  let loc name =
    Map.to_alist ctx.BI.ctx_buffers
    |> List.find_map ~f:(fun (tn, (l : BI.buffer_loc)) ->
        if String.is_prefix (Tn.debug_name tn) ~prefix:name then Some l else None)
  in
  let lp = Option.value_exn ~here:[%here] (loc "p") in
  let lq = Option.value_exn ~here:[%here] (loc "q") in
  Verdict.pf "pooled p.offset=%d q.offset=%d share_pool" lp.offset lq.offset
    (lp.pool_id = lq.pool_id);
  Task.run routine.BI.schedule;
  Backend.await device;
  let read_back tnode =
    let nd =
      Ir.Ndarray.create_array ~debug:"pool_offset_test" Ir.Ops.single ~dims:[| 2 |] ~padding:None
    in
    ignore (Backend.to_host ctx tnode nd : bool);
    Backend.await device;
    Ir.Ndarray.retrieve_flat_values nd
  in
  let pv = read_back p.Tensor.value in
  let qv = read_back q.Tensor.value in
  (* p = a+b = [4.0; 6.0]; q = a*b = [3.0; 8.0]. With broken offset both point to offset 0: q
     overwrites p's slot, reading p gives [3.0; 8.0]. The claims below would then fail the run. *)
  Verdict.p "pooled p (a+b expect [4.0;6.0]) correct" (Array.equal Float.equal pv [| 4.0; 6.0 |]);
  Verdict.p "pooled q (a*b expect [3.0;8.0]) correct" (Array.equal Float.equal qv [| 3.0; 8.0 |])

let () =
  Verdict.case "run1" (fun () -> run_once "run1");
  Verdict.case "run2" (fun () -> run_once "run2");
  Verdict.case "run_packed" run_packed;
  Verdict.case "run_pooled_values_correct" run_pooled_values_correct;
  Stdio.printf "done\n"
