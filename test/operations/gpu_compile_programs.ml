(* gh-ocannl-1603: a backend that compiles a batch's kernel code cache misses as several programs at
   once (CUDA's [cuda_compile_programs]) computes exactly what the one joint program computes.

   The routine is a small MLP's training step -- forward, backprop, SGD -- which the GPU backends'
   default pipeline fissions into several kernels. It is compiled twice in this process from fresh
   contexts, under two routine names so that the second compile misses every kernel too: once with
   the split forced off ([Kernel_code_cache.with_split], at most one program) and once with it
   forced on (up to four programs of at least one kernel each). The step runs once after each, and
   the loss and both updated weights must agree bit for bit: the same kernels' code, only grouped
   into other programs. The inputs are not dyadic, so a kernel compiled differently in its smaller
   program would show.

   The split claims are gated on [Kernel_code_cache.split_capable_compiles] (a backend that can
   split recorded the compile), never on a backend name; that record stays with the kernel cache
   disabled, which compiles one program as before, so a run with [OCANNL_CUDA_KERNEL_CACHE_MIB=0] on
   CUDA must fail them -- the negative control. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module K = Ir.Kernel_code_cache
module CC = Ir.Compile_cost
module Asgns = Ir.Assignments

let batch = 16
let d_in = 8
let d_hid = 16
let classes = 4
let w1_init = Ll_test.weighted ~weights:[| 5; 3 |] ~modulus:11 ~offset:(-5.) ~stride:0.03

let step () =
  let xs =
    NTDSL.init ~l:"gcp_xs" ~prec:Ir.Ops.single ~b:[ batch ] ~i:[] ~o:[ d_in ]
      ~f:(Ll_test.weighted ~weights:[| 3; 5 |] ~modulus:7 ~offset:(-3.) ~stride:0.1)
      ()
  in
  let ys =
    NTDSL.init ~l:"gcp_ys" ~prec:Ir.Ops.single ~b:[ batch ] ~i:[] ~o:[ classes ]
      ~f:(fun idcs -> if idcs.(0) % classes = idcs.(1) then 1. else 0.)
      ()
  in
  let w1 =
    Operation.init ~l:"gcp_w1" ~prec:Ir.Ops.single ~b:[] ~i:[ d_in ] ~o:[ d_hid ] ~f:w1_init
      ~grad_spec:Tensor.Require_grad ()
  in
  let w2 =
    Operation.init ~l:"gcp_w2" ~prec:Ir.Ops.single ~b:[] ~i:[ d_hid ] ~o:[ classes ]
      ~f:(Ll_test.weighted ~weights:[| 7; 1 |] ~modulus:13 ~offset:(-6.) ~stride:0.02)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op logits = w2 * relu (w1 * xs) in
  let%op loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch () ~logits ~targets:ys
  in
  let update = Train.grad_update loss in
  (* [w1]/[w2] are [Require_grad] leaves, which do not join [loss.params]: pass them explicitly. *)
  let sgd =
    Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.1)
      ~params:(Set.of_list (module Tensor) [ w1; w2 ])
      loss
  in
  (Asgns.sequence [ update; sgd ], [ loss; w1; w2 ])

let comp, read = step ()

let compile_and_run ~name split =
  let capable = K.split_capable_compiles () and programs = K.programs_compiled_concurrently () in
  let ctx = Context.auto () in
  let (ctx, routine), cost =
    K.with_split split (fun () ->
        CC.with_census (fun () -> Context.compile ~name ctx comp Ir.Indexing.Empty))
  in
  let ctx = Context.run ctx routine in
  ( Context.backend_name ctx,
    List.map read ~f:(fun t -> Context.get_values ctx t.Tensor.value),
    cost,
    K.split_capable_compiles () > capable,
    K.programs_compiled_concurrently () - programs )

let bitwise_equal a e =
  Array.equal (fun x y -> Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)) a e

let () =
  let backend, joint, cj, _, joint_programs =
    compile_and_run ~name:"gcp_joint" { K.max_programs = 1; min_kernels = 1 }
  in
  let _, split, cs, capable, split_programs =
    compile_and_run ~name:"gcp_split" { K.max_programs = 4; min_kernels = 1 }
  in
  List.iter2_exn read (List.zip_exn joint split) ~f:(fun t (joint, split) ->
      p
        ("split and joint agree bit for bit on " ^ Ir.Tnode.debug_name t.Tensor.value)
        (bitwise_equal split joint));
  (* [w1] is [d_hid] outputs by [d_in] inputs, row-major; its single-precision values are within
     1e-6 of the initializer's, so a move past 1e-4 is the step's. *)
  let w1_after = List.nth_exn joint 1 in
  p_exists "the step moved w1 (the parity is not over untouched values)"
    (List.init (d_hid * d_in) ~f:Fn.id)
    ~f:(fun j ->
      let before = w1_init [| j / d_in; j % d_in |] in
      Float.(abs (w1_after.(j) - before) > 1e-4));
  let on = backend and when_ = capable in
  gated ~when_ ~on "the step rendered at least two kernels" (cs.kernels >= 2);
  gated ~when_ ~on "the joint compile compiled no program concurrently" (joint_programs = 0);
  gated ~when_ ~on "the split compile compiled every kernel it rendered"
    (cs.kernels_compiled = cs.kernels);
  gated ~when_ ~on "the split compile compiled at least two programs at once" (split_programs >= 2);
  gated ~when_ ~on "and timed them as device time" (Option.is_some cs.device_s);
  Stdio.eprintf
    "gpu_compile_programs (not part of the golden): joint %d of %d kernels, %d concurrent \
     programs, device %s s; split %d of %d kernels, %d concurrent programs, device %s s\n\
     %!"
    cj.kernels_compiled cj.kernels joint_programs
    (Option.value_map cj.device_s ~default:"none" ~f:(Printf.sprintf "%.3f"))
    cs.kernels_compiled cs.kernels split_programs
    (Option.value_map cs.device_s ~default:"none" ~f:(Printf.sprintf "%.3f"))
