(* gh-ocannl-1391: every kernel a training step ships compiles, and runs, as a routine of its own --
   the contract of the isolated per-kernel instrument ([Bench_harness.time_shipped_kernels], behind
   [dominant_kernel], [BENCH_KERNEL_TABLE] and the kprof attribution, which refuses a cell with a
   declined kernel). The step is driven through that instrument itself, so what is checked is the
   seam a measurement uses, not a reconstruction of it.

   The model is two transformer encoder blocks, so their layer norms' and attention's internal nodes
   ([mean], [sqrt_std_dev], ...) share labels across layers. On a GPU backend the step's gradient
   zeros fold into one zero segment, and [gpu_elide_covered_zeros] drops the zeros an accumulation
   covers. Before gh-ocannl-1391 the dropped nodes stayed in that segment's store, hence among its
   kernel parameters, while parameter names are disambiguated over the code alone: two same-labelled
   dropped nodes became two parameters named alike, which the shipped batch tolerates (its other
   kernels touch them) and a compile of the segment alone does not (CUDA: "duplicate parameter
   name", on kernel 44 of 92 on rog). A CPU backend keeps its zero segments, so there the claim
   holds without exercising that path. *)

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
  p_all ~min:2 "the step ships several kernels, and every one compiles and runs on its own" kernels
    ~f:(fun (_, _, result) -> Result.is_ok result)
