(* CUDA pool-allocator sub-region addressing regression test. Exercises the AC from
   docs/proposals/cuda-pool-allocator-region-addressing.md: two non-constant working outputs (p, q)
   sharing a bump-packed pool slab at distinct byte offsets must hold independent values after the
   CUDA kernel runs.

   The invariant under test (failure condition): with the broken `ignore offset; ptr` stub in
   Slab.ptr_at, both p and q map to offset 0 in the pool slab — q's write overwrites p's slot, so
   reading p returns q's value [3.0; 8.0] instead of the expected [4.0; 6.0].

   Harness condition: p and q must be bump-packed into the same pool at DISTINCT non-zero offsets
   (confirmed by the assert below); a test where they happen to be at the same offset would be
   vacuous. The bump-packing is guaranteed by having both appear as written nodes in the same %cd
   combo (one routine), triggering the allocate_delta path in backends.ml. *)
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

let () =
  Tensor.unsafe_reinitialize ();
  let backend = Backends.backend_module (Backends.get_backend ~backend_name:"cuda" ()) in
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
        if String.is_prefix (Tn.label_name tn) ~prefix:name then Some l else None)
  in
  let lp = Option.value_exn ~here:[%here] (loc "p") in
  let lq = Option.value_exn ~here:[%here] (loc "q") in
  (* Harness condition: p and q must share a pool at DISTINCT offsets for the test to be
     non-vacuous, so it is claimed rather than merely reported. *)
  Verdict.p "p and q share pool" (lp.pool_id = lq.pool_id);
  Verdict.pf "p.offset=%d q.offset=%d distinct" lp.offset lq.offset (lp.offset <> lq.offset);
  Task.run routine.BI.schedule;
  Backend.await device;
  let read_back tnode =
    let nd =
      Ir.Ndarray.create_array ~debug:"cuda_pool_offset" Ir.Ops.single ~dims:[| 2 |] ~padding:None
    in
    ignore (Backend.to_host ctx tnode nd : bool);
    Backend.await device;
    Ir.Ndarray.retrieve_flat_values nd
  in
  let pv = read_back p.Tensor.value in
  let qv = read_back q.Tensor.value in
  (* p = a+b = [4.0; 6.0]; q = a*b = [3.0; 8.0]. With broken Slab.ptr_at (ignore offset), both p and
     q address offset 0: q's write clobbers p, so reading p gives [3.0; 8.0] — the claim below then
     fails the run. *)
  Verdict.p "CUDA pooled p (a+b expect [4.0;6.0]) correct"
    (Array.equal Float.equal pv [| 4.0; 6.0 |]);
  Verdict.p "CUDA pooled q (a*b expect [3.0;8.0]) correct"
    (Array.equal Float.equal qv [| 3.0; 8.0 |])
