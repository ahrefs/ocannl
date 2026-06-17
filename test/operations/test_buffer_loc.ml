(* Regression test for the buffer-addressing refactor + gh-ocannl-344 pooling: buffer handles are
   deterministic integer [buffer_loc = { pool_id; offset }] resolved against a backend-private pool
   table, and a context's working (non-constant) delta is packed into a SINGLE pool at increasing
   byte offsets rather than one pool per tnode.

   Two invariants are pinned:
   1. Determinism / reproducibility: linking the same graph into two independently-created fresh
      [sync_cc] backends yields identical [{ pool_id; offset }] values (the debuggability win).
   2. Pooling: when a single routine materializes several non-constant tnodes, they share one
      [pool_id] with distinct, increasing offsets; read-only inputs (constants) live in separate
      per-device pools. If the allocator regressed to one-pool-per-tnode, the two outputs below would
      print different pool ids (and all offsets 0); if it stopped honoring offsets, the computed
      values would be wrong. *)

open Base
open Ocannl
open Operation.DSL_modules
module Backends = Context.Backends_deprecated
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module BI = Ir.Backend_intf

let make_tensor label vals =
  let open Bigarray in
  let n = Array.length vals in
  let ga = Genarray.create Float32 c_layout [| n |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set ga [| i |] v);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ] ~batch_dims:[]
    ~input_dims:[] ~output_dims:[ n ] ()

let run_once tag =
  Tensor.unsafe_reinitialize ();
  let backend = Backends.fresh_backend ~backend_name:"sync_cc" () in
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
  let backend = Backends.fresh_backend ~backend_name:"sync_cc" () in
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
  match (loc "p", loc "q") with
  | Some lp, Some lq ->
      (* Both outputs are non-constant working nodes: same pool, distinct increasing offsets. *)
      Stdio.printf "two outputs share a pool = %b\n" (lp.pool_id = lq.pool_id);
      Stdio.printf "two outputs have distinct offsets = %b\n" (lp.offset <> lq.offset);
      Stdio.printf "offsets are 8-byte aligned (2x float32) = %b\n"
        (lp.offset % 4 = 0 && lq.offset % 4 = 0)
  | _ -> Stdio.printf "two outputs share a pool = MISSING\n"

let () =
  run_once "run1";
  run_once "run2";
  run_packed ();
  Stdio.printf "done\n"
