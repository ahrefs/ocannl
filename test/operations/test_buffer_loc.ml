(* Regression test for the buffer-addressing refactor (backend-buffer-addressing proposal): buffer
   handles are deterministic integer [buffer_loc = { pool_id; offset }] resolved against a
   backend-private pool table, not opaque per-run pointers.

   It compiles+links the same tiny graph into two independently-created fresh [sync_cc] backends and
   prints the resulting [ctx_buffers] locations. The invariant pinned here: locations are
   deterministic and reproducible across fresh backends (the debuggability win), one pool per tnode
   at offset 0 (phase-1 policy), and the two runs print identical [{ pool_id; offset }] values. If
   the allocator stopped minting pool ids in deterministic tnode-iteration order, or reintroduced
   opaque pointers, the two blocks below would differ or print noise. *)

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

let () =
  run_once "run1";
  run_once "run2";
  Stdio.printf "done\n"
