(* gh-ocannl-1391: every kernel a training step ships compiles, and runs, as a routine of its own --
   the contract of the isolated per-kernel instrument ([Bench_harness.time_shipped_kernels], behind
   [dominant_kernel], [BENCH_KERNEL_TABLE] and the kprof attribution, which refuses a cell with a
   declined kernel). The step is driven through that instrument itself, so what is checked is the
   seam a measurement uses, not a reconstruction of it.

   The model is two transformer encoder blocks, so their layer norms' and attention's internal nodes
   ([mean], [sqrt_std_dev], ...) share labels across layers. On a GPU backend the step's gradient
   zeros fold into one zero segment, and [gpu_elide_covered_zeros] drops the zeros an accumulation
   covers. Before gh-ocannl-1391 the dropped nodes stayed in that segment's store, and every
   materialized store entry was a kernel parameter, while parameter names are disambiguated over the
   code alone: two same-labelled dropped nodes became two parameters named alike, which the shipped
   batch tolerates (its other kernels touch them) and a compile of the segment alone does not (CUDA:
   "duplicate parameter name", on kernel 44 of 92 on rog; ahrefs/ocannl#1387 on HIP and Metal too).
   Both halves changed: the elision prunes what it left untouched from the store, and a kernel
   declares only the store entries its code touches. A CPU backend keeps its zero segments, so there
   the claim holds without exercising the elision path. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module H = Bench_harness

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let batch = 4
let seq = 16
let d = 32

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[] ~output_dims:[ d ]
      ()
  in
  let block l =
    Nn_blocks.transformer_encoder_block ~label:[ l ] ~num_heads:2 ~d_k:16 ~d_v:16 ~d_ff:64 ()
  in
  let b0 = block "l0" and b1 = block "l1" in
  let%op x = x /. 2048. in
  let h = b1 ~train_step:None (b0 ~train_step:None x) in
  let%op loss = (h *. h) ++ "... | ... => 0" in
  let parts = H.train_step_parts ~leg:H.self_test_leg ~learning_rate:(TDSL.O.( !. ) 0.01) loss in
  let bindings = Ir.Indexing.Empty in
  let ctx = Train.init_params (Context.auto ()) bindings loss in
  let ctx, routines =
    H.compile_train_step ~tune:false
      ~tuned:(fun _ _ -> failwith "shipped_kernels_alone does not autotune")
      ctx bindings parts
  in
  let ctx_ref = ref ctx in
  H.run_train_step routines ctx_ref ~step:0;
  Context.sync !ctx_ref;
  let kernels =
    H.time_shipped_kernels ~out:Stdio.stderr ~repeats:1 ~ctx:!ctx_ref ~bindings
      (H.step_routines routines)
  in
  p_all "every kernel the step ships compiles and runs on its own" kernels ~f:(fun (_, _, result) ->
      Result.is_ok result)

(* The parameter rule's one blind spot, kept open on purpose: a [Staged_compilation] callback's
   accesses are invisible to [Low_level.code_touches_tn], so a kernel holding one keeps every stored
   entry. Here the record's store is built by a raw twin that writes the node, and the shipped body
   is only a staged statement writing it, so dropping the untouched entry would emit a reference to
   an undeclared pointer. *)
let () =
  let tn = (Ll_test.node_factory ~first_id:9950 ~dims:[| 1 |] ()) "sk_staged" in
  Ll_test.materialize tn;
  let cell = [| Ll_test.fixed 0 |] in
  let o =
    Ll_test.optimize_scoped ~materialized:[ tn ] ~name:"sk_staged"
      ~raw:(Ll_test.set tn cell (Ir.Low_level.Constant 9.0))
      (* Staged text is emitted verbatim, so it spells the node as the dialect emits it
         (gh-ocannl-1338: [t_sk_staged] on a GPU backend). *)
      (Ir.Low_level.Staged_compilation
         (fun () ->
           PPrint.string
             (Ir.C_syntax.node_ident_prefix_for_backend backend_name ^ "sk_staged[0] = 2.0f;")))
  in
  let got = Ll_test.execute ~name:"sk_staged" o ~seed:[ (tn, [| -1.0 |]) ] ~read:[ tn ] in
  p "a kernel whose only access to a node is staged code still declares it, and the write lands"
    (Ll_test.close (List.hd_exn got) [| 2.0 |])
