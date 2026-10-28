(* gh-ocannl-1674: the same kernels compiled at the shipping floor PTX and at the device's own arch
   compute the same values, bit for bit.

   CUDA compiles a kernel for the lowest arch its instructions need ([Cuda_mma.gpu_arch_options]:
   compute_80 for tf32 wmma, nvrtc's default for an unmarked kernel), and the driver JIT-compiles
   that PTX up to the device. lukstafi/ocannl-staging#1430's lane-0 fragment transfer was wrong only
   through that route: nvrtc's PTX for compute_80 to compute_90 kept a lane's zeroing store in a
   register across [__syncthreads()], while compute_120 PTX of the same source was right. A
   comparison of the two targets needs no reference of its own, so it catches that class wherever
   the kernel under test is exact, whatever else the kernel computes.

   The kernels are [conv_merged_gpu_seeds]' [cm8w] graph (conv over 8 rows and 64 out-channels, its
   relu and scale companions materialized): the natural form, and every whole-routine GPU sketch
   seed, the pad-masked shared-fragment flavor of #1430 among them, under [tf32_matmuls] (set here,
   so the CUDA run reaches the tf32 tile without a rerun rule). The operands are small integers, so
   every sum is exact in f32 and in tf32: the comparison is bitwise. Each is compiled under
   [Cuda_mma.with_ptx_target Floor] and [Native] and run once; the arch each compile reached is read
   from its PTX's own [.target] directive (what [with_ptx_target] returns), never from the options
   asked for. The claims are gated on a compile having recorded a PTX program, never on a backend
   name: a backend that compiles no PTX skips them.

   Negative control: with #1430's [C_syntax.lane_restrict] reverted, the pad-masked seed's floor
   compile differs from its native one on rog-nv-linux (sm_120). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Mma = Ir.Cuda_mma

let () = Ir.Numerics.set_policy { (Ir.Numerics.get ()) with Ir.Numerics.tf32_matmuls = true }
let accum_prec = Ll_test.backend_accum_prec
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
let limits = Context.hardware_limits (Context.auto ())

(* The seeder's own judgment that a GPU conv leg can seed an f32 site here, tf32 being on. *)
let gpu_capable =
  (not on_cpu)
  &&
  match
    Ll_test.tensorized_capability ~is_gpu:true ~is_cpu:false ~limits ~a_prec:Ir.Ops.single
      ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single
  with
  | `Advertised -> true
  | `Withheld _ -> false

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let rows = 8
let oc = 64

let make_merged tag =
  let x =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ]
      ~o:[ rows + 2; rows + 2; 8 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ oc ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op pr =
    x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  Ir.Tnode.update_memory_mode pr.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
  let%op y2 = relu pr in
  Ir.Tnode.update_memory_mode y2.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
  let%op y3 = y2 *. 0.5 in
  y3

let whole_lowering () =
  let captured = ref None in
  ignore
    (Context.compile
       ~lowered_transform:(fun opt ->
         captured := Some (scratch opt);
         [ opt ])
       (Context.auto ())
       (named "ptp_probe" (Train.forward (make_merged "ptp_p")))
       Ir.Indexing.Empty);
  Option.value_exn ~here:[%here] !captured

(* One compile and run of a fresh graph at [target]: its values, its mma census, and the PTX
   programs it compiled. *)
let run ~target ~name ~transform =
  let y = make_merged name in
  let (ctx, routine), compiles =
    Mma.with_ptx_target target (fun () ->
        Context.compile ~lowered_transform:transform (Context.auto ())
          (named name (Train.forward y))
          Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx y.Tensor.value, routine.Context.mma, compiles)

let bitwise_equal a e =
  (not (Array.is_empty a))
  && Array.length a = Array.length e
  && Array.for_all2_exn a e ~f:(fun x y ->
      Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y))

let targets (compiles : Mma.ptx_compile list) =
  String.concat ~sep:","
    (List.map compiles ~f:(fun c -> Option.value c.Mma.target ~default:"(no .target)"))

let native_arch (c : Mma.ptx_compile) = "sm_" ^ Int.to_string c.Mma.device_cc

let claims =
  [
    "a whole-routine GPU seed pads the row to the mma tile";
    "every variant's floor and native compiles agree bit for bit";
    "every variant with floor PTX for another arch compiled native PTX, all for the device's arch";
    "some variant's floor compile compiled PTX for another arch";
    "every seed's routine is labeled tensorized at both targets";
  ]

let () =
  (* Compiled only where an f32 tensor-core tile can seed at all. *)
  let natural_floor =
    if gpu_capable then
      Some (run ~target:Mma.Floor ~name:"ptp_nat_f" ~transform:(fun opt -> [ opt ]))
    else None
  in
  match natural_floor with
  | None | Some (_, _, []) ->
      (* No f32 tensor-core tile to seed, or no PTX compiled: the backend has no floor/native
         distinction to compare. *)
      List.iter claims ~f:(Verdict.skipped ~backend:backend_name)
  | Some ((_, _, natural_compiles) as natural_floor) ->
      let device_cc = (List.hd_exn natural_compiles).Mma.device_cc in
      Stdio.eprintf "ptx_target_parity (not part of the golden): backend %s, device compute %d\n%!"
        backend_name device_cc;
      let seeds =
        Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits (whole_lowering ())
      in
      p_exists (List.nth_exn claims 0) seeds ~f:(fun sp ->
          sp.Autotune.sk_bm = 0 && sp.Autotune.sk_tm > 0);
      let variants =
        ("natural", fun opt -> [ opt ])
        :: List.map seeds ~f:(fun sp ->
            ( Autotune.sketch_params_label sp,
              fun opt -> [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp opt) opt ] ))
      in
      let results =
        List.mapi variants ~f:(fun i (label, transform) ->
            let floor =
              if i = 0 then natural_floor
              else run ~target:Mma.Floor ~name:(Printf.sprintf "ptp_v%d_f" i) ~transform
            in
            let native = run ~target:Mma.Native ~name:(Printf.sprintf "ptp_v%d_n" i) ~transform in
            let (fv, _, fc), (nv, _, nc) = (floor, native) in
            let agree = bitwise_equal fv nv in
            Stdio.eprintf
              "ptx_target_parity (not part of the golden): %s: floor PTX %s, native PTX %s, %s\n%!"
              label (targets fc) (targets nc)
              (if agree then "bitwise equal"
               else if Array.length fv <> Array.length nv then "shape mismatch"
               else
                 Printf.sprintf "%d of %d cells differ"
                   (Array.counti fv ~f:(fun j x -> not (Float.equal x nv.(j))))
                   (Array.length fv));
            (i, agree, floor, native))
      in
      let _, _, (reference, _, _), _ = List.hd_exn results in
      if not (Array.exists reference ~f:(fun v -> Float.(v <> 0.))) then
        failwith "ptx_target_parity: the natural form is all zeros, so the comparison is vacuous";
      p_all (List.nth_exn claims 1) results ~f:(fun (_, agree, _, _) -> agree);
      (* A variant whose floor compiles already targeted the device's arch (an sm_80 device's tf32
         seeds) has the same option list at both targets, so its native compile may be served by the
         kernel code cache and record nothing: only the others must compile at native. *)
      let at_native (c : Mma.ptx_compile) =
        Option.equal String.equal c.Mma.target (Some (native_arch c))
      in
      p_all (List.nth_exn claims 2)
        (List.filter results ~f:(fun (_, _, (_, _, fc), _) -> not (List.for_all fc ~f:at_native)))
        ~f:(fun (_, _, _, (_, _, nc)) ->
          (not (List.is_empty nc))
          && List.for_all nc ~f:(fun c ->
              Option.equal String.equal c.Mma.target (Some (native_arch c))));
      p_exists (List.nth_exn claims 3) results ~f:(fun (_, _, (_, _, fc), _) ->
          List.exists fc ~f:(fun c ->
              not (Option.equal String.equal c.Mma.target (Some (native_arch c)))));
      p_all (List.nth_exn claims 4)
        (List.filter results ~f:(fun (i, _, _, _) -> i > 0))
        ~f:(fun (_, _, (_, fm, _), (_, nm, _)) ->
          Ll_test.rendered_tensorized fm && Ll_test.rendered_tensorized nm)
