(* gh-ocannl-912: where [Autotune.detect_conv] stops, and why — the refusal witness for the
   convolution family's recognition boundary.

   The boundary is NOT the spatial rank or the batch rank. The matcher is rank-generic: it reads the
   output's minor axis as the out-channel and its second-to-last as the implicit-GEMM row, and every
   other output axis — any number of batch axes, any number of further spatial axes — becomes an
   outer loop. 3-D, 1-D, batchless and two-batch-axis convolutions are detected AND seeded on both
   legs (claims below). What the matcher refuses is a SINGLETON axis: lowering drops every extent-1
   loop and indexes the axis with a literal [Fixed_idx 0], while the matcher demands plain iterators
   on the output and on the kernel. So:

   (1) A singleton OUTPUT axis (a batch of one, an output spatial extent of one) leaves the output
   written at a fixed index: refused. (2) A singleton KERNEL axis (a single input channel, as in
   LeNet's first conv on grayscale images; a k-by-1 or 1-by-k window) leaves the kernel read at a
   fixed index: refused. With one input channel there is moreover no reduction-channel loop at all,
   and the implicit-GEMM pipelines' [Tensorize (row, oc, ic)] needs one. (3) A channel ROW of
   several axes ([..ic..] spanning two axes) gives two reduction-channel symbols where the pipelines
   contract exactly one: refused. (4) An all-singleton window (1x1, any stride) is a plain GEMM once
   its window loops are gone: the MATMUL family detects and seeds it, which is its correct family
   (the matmul pipelines already own strided operands and the per-statement vs fragment MMA scope
   rule).

   Each refusal is pinned together with its reason, read off the lowered accumulation: the
   singleton's fixed index on the output or kernel map, or the count of channel symbols both
   operands read. The same reason predicates are evaluated on the accepted control, where they must
   be false — so a lowering change that stops eliding extent-1 loops fails a "because" claim here
   rather than silently leaving this documentation stale. A refused site gets no matmul or conv
   sketch seeds; a search can still time the preset (whole-routine and fissioned) and split-reduce
   candidates over it, which this file does not exercise.

   Detection for the refused classes is deliberately not added (the wave decision on the issue: add
   it when a benchmark leg wants it). The one benchmark leg that owns such a site is lenet's conv1
   (one input channel, benchmarks/README.md), and no measurement says the implicit-GEMM formulation
   — whose GEMM reduction is the channel axis alone — pays at a reduction extent of one; the window
   would have to join the reduction for it to. Everything here is structural: graphs are lowered and
   optimized but never compiled or dispatched, and seeding runs against MOCKED limits, so the claims
   are backend-independent. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module A = Ir.Affine
module BI = Ir.Backend_intf
module Idx = Ir.Indexing
open Verdict.Claims

(* An mma capability with 8x8x8 f32 tiles: enough for the GPU conv and matmul families to seed,
   whatever device (if any) the run has. *)
let mma_limits =
  {
    BI.no_hardware_limits with
    mma =
      Some
        {
          BI.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((BI.Mma_f32, BI.Mma_f32, BI.Mma_f32), (8, 8, 8)) ];
        };
  }

(* The C-backend leg's limits: a pinned 16-byte vector width, no mma. *)
let cpu_limits = { BI.no_hardware_limits with simd_vector_bytes = 16 }

type observed = {
  conv : Autotune.conv_site option;  (** On the whole lowered routine. *)
  matmul : bool;
  cpu : Autotune.sketch_params list;  (** Seeds on the whole routine (C-backend leg). *)
  gpu : Autotune.sketch_params list;
      (** Seeds on the pre-schedule fission segments (GPU leg: a conv's [Zero_out] must sit in its
          own kernel first). *)
  out_map : Idx.axis_index array;  (** The accumulation's write map. *)
  in_map : Idx.axis_index array;  (** Its read map on the conv input. *)
  kern_map : Idx.axis_index array;  (** Its read map on the other operand, the kernel. *)
}

(* The accumulation statement reading [x]: its rmw write, the read of [x], and the one remaining
   operand read. *)
let accumulation name (x : Ir.Tnode.t) (llc : LL.t) =
  let accs = LL.affine_accesses llc in
  let reads_x w =
    List.exists accs ~f:(fun a ->
        (not a.A.a_write) && phys_equal a.A.a_tn x && A.same_statement a.A.a_path w.A.a_path)
  in
  match
    List.filter accs ~f:(fun a -> a.A.a_write && a.A.a_rmw && (not a.A.a_whole) && reads_x a)
  with
  | [ w ] -> (
      let operands =
        List.filter accs ~f:(fun a ->
            (not a.A.a_write)
            && A.same_statement a.A.a_path w.A.a_path
            && not (phys_equal a.A.a_tn w.A.a_tn))
      in
      match List.partition_tf operands ~f:(fun a -> phys_equal a.A.a_tn x) with
      | [ xr ], [ kr ] -> (w.A.a_map, xr.A.a_map, kr.A.a_map)
      | _ -> failwith (name ^ ": the accumulation does not read exactly the input and one kernel"))
  | l ->
      failwith
        (Printf.sprintf "%s: expected one accumulation reading the input, found %d" name
           (List.length l))

let observe name ~(x : Tensor.t) (y : Tensor.t) =
  (* The optimized lowering exactly as [Context.compile] would hand it to a [lowered_transform], by
     lowering and optimization alone: no codegen, no linking, and so no [Train.init_params] (which
     would compile AND run the parameter initialization). Nothing here is dispatched. *)
  let opt = Context.lowered_for_decisions ~name (Context.auto ()) (Train.forward y) Idx.Empty in
  let preset seg = Sched.default_gpu ~min_parallel:1 ~limits:mma_limits seg in
  let zero_sched tns = Sched.zero_expansion ~limits:mma_limits tns in
  let segments = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt in
  let out_map, in_map, kern_map = accumulation name x.Tensor.value opt.LL.llc in
  {
    conv = Autotune.detect_conv opt.LL.llc;
    matmul = Option.is_some (Autotune.detect_matmul opt.LL.llc);
    cpu = Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt;
    gpu =
      List.concat_map segments ~f:(fun (_, pre, _, _) ->
          Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:mma_limits pre);
    out_map;
    in_map;
    kern_map;
  }

(* The reasons, read off the lowered accumulation. *)
let has_fixed map = Array.exists map ~f:(function Idx.Fixed_idx _ -> true | _ -> false)

let plain map =
  Array.to_list map |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)

(* Channel symbols: plain iterators read by both operands. Kernel-window symbols are plain in the
   kernel but sit inside an affine component of the input, so they do not count. *)
let n_channels o =
  List.count (plain o.in_map) ~f:(fun s -> List.mem (plain o.kern_map) s ~equal:Idx.equal_symbol)

let singleton_output o = has_fixed o.out_map
let singleton_kernel o = has_fixed o.kern_map
let seeds_conv l = List.exists l ~f:(fun q -> q.Autotune.sk_conv)

(* An einsum conv over constant operands (only the lowered structure is inspected); the spec's axis
   list decides the rank. *)
let conv name ~b ~o ~i ~spec =
  let x = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.single ~b ~o ~f:(fun _ -> 1.) () in
  let k = NTDSL.init ~l:(name ^ "_k") ~prec:Ir.Ops.single ~i ~o:[ 8 ] ~f:(fun _ -> 1.) () in
  observe name ~x (NTDSL.O.einsum ~label:[ name ] spec x k)

let s1d = "...| 1*ow<+kw, ..ic..; |kw, ..ic.. -> ..oc.. => ...| ow, ..oc.."
let s2 = "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."
let s2s = "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."

let s3 =
  "...| 1*od<+kd, 1*oh<+kh, 1*ow<+kw, ..ic..; |kd, kh, kw, ..ic.. -> ..oc.. => ...| od, oh, ow, \
   ..oc.."

let () =
  (* === Accepted: the boundary is not the rank === *)
  let accepted ~what ~spatial o =
    p_exists (what ^ ": detected, one conv axis per spatial axis") (Option.to_list o.conv)
      ~f:(fun s -> List.length s.Autotune.c_axes = spatial);
    p (what ^ ": the C-backend leg seeds a conv candidate") (seeds_conv o.cpu);
    p (what ^ ": the GPU leg seeds a conv candidate") (seeds_conv o.gpu)
  in
  let control = conv "cdb_2d" ~b:[ 2 ] ~o:[ 6; 6; 4 ] ~i:[ 3; 3; 4 ] ~spec:s2 in
  accepted ~what:"2-D control" ~spatial:2 control;
  (* The reason predicates below are false on the control: they discriminate. *)
  p_all "2-D control: the output and the kernel are indexed at plain iterators only"
    (Array.to_list control.out_map @ Array.to_list control.kern_map)
    ~f:(function Idx.Iterator _ -> true | _ -> false);
  p "2-D control: both operands read exactly one channel symbol" (n_channels control = 1);
  accepted ~what:"3-D spatial" ~spatial:3
    (conv "cdb_3d" ~b:[ 2 ] ~o:[ 5; 6; 6; 4 ] ~i:[ 2; 3; 3; 4 ] ~spec:s3);
  accepted ~what:"two batch axes" ~spatial:2
    (conv "cdb_2b" ~b:[ 2; 3 ] ~o:[ 6; 6; 4 ] ~i:[ 3; 3; 4 ] ~spec:s2);
  accepted ~what:"no batch axis" ~spatial:2
    (conv "cdb_0b" ~b:[] ~o:[ 6; 6; 4 ] ~i:[ 3; 3; 4 ] ~spec:s2);
  accepted ~what:"1-D" ~spatial:1 (conv "cdb_1d" ~b:[ 2 ] ~o:[ 6; 4 ] ~i:[ 3; 4 ] ~spec:s1d);

  (* === Refused, with the reason === *)
  let refused ~what ~why ~reason o =
    p
      (what ^ ": not a conv site, and no sketch family seeds it")
      (Option.is_none o.conv && (not o.matmul) && List.is_empty o.cpu && List.is_empty o.gpu);
    p (what ^ ": because " ^ why) (reason o)
  in
  let fixed_output =
    "lowering elided an extent-1 loop and the output is written at a fixed index"
  in
  let fixed_kernel = "lowering elided an extent-1 loop and the kernel is read at a fixed index" in
  refused ~what:"batch of one" ~why:fixed_output ~reason:singleton_output
    (conv "cdb_b1" ~b:[ 1 ] ~o:[ 6; 6; 4 ] ~i:[ 3; 3; 4 ] ~spec:s2);
  refused ~what:"output row extent one" ~why:fixed_output ~reason:singleton_output
    (conv "cdb_ow1" ~b:[ 2 ] ~o:[ 6; 3; 4 ] ~i:[ 3; 3; 4 ] ~spec:s2);
  refused ~what:"outer output spatial extent one" ~why:fixed_output ~reason:singleton_output
    (conv "cdb_oh1" ~b:[ 2 ] ~o:[ 3; 6; 4 ] ~i:[ 3; 3; 4 ] ~spec:s2);
  refused ~what:"3x1 window" ~why:fixed_kernel ~reason:singleton_kernel
    (conv "cdb_k31" ~b:[ 2 ] ~o:[ 6; 6; 4 ] ~i:[ 3; 1; 4 ] ~spec:s2);
  refused ~what:"1x3 window" ~why:fixed_kernel ~reason:singleton_kernel
    (conv "cdb_k13" ~b:[ 2 ] ~o:[ 6; 6; 4 ] ~i:[ 1; 3; 4 ] ~spec:s2);
  (* lenet's conv1, through the block the benchmark builds it with: 32x32 grayscale images, a valid
     5x5 window, 6 out-channels. *)
  (let x =
     NTDSL.init ~l:"cdb_lenet_x" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 32; 32; 1 ] ~f:(fun _ -> 1.) ()
   in
   let conv1 =
     Nn_blocks.conv2d ~label:[ "cdb_lenet" ] ~kernel_size:5 ~use_padding:false ~out_channels:6 ()
   in
   refused ~what:"lenet conv1 (one input channel)"
     ~why:(fixed_kernel ^ ", and no reduction-channel loop is left")
     ~reason:(fun o -> singleton_kernel o && n_channels o = 0)
     (observe "cdb_lenet" ~x (conv1 x)));
  refused ~what:"two input-channel axes"
    ~why:"both operands read two channel symbols, and the pipelines contract exactly one"
    ~reason:(fun o -> n_channels o = 2)
    (conv "cdb_ic2" ~b:[ 2 ] ~o:[ 6; 6; 2; 3 ] ~i:[ 3; 3; 2; 3 ] ~spec:s2);

  (* === Rerouted: an all-singleton window is a GEMM, and the matmul family owns it === *)
  let rerouted ~what o =
    p (what ^ ": not a conv site, but a matmul site") (Option.is_none o.conv && o.matmul);
    p
      (what ^ ": the matmul family, not the conv family, seeds it on both legs")
      ((not (List.is_empty o.cpu))
      && (not (List.is_empty o.gpu))
      && (not (seeds_conv o.cpu))
      && not (seeds_conv o.gpu));
    p
      (what ^ ": because the window loops are gone and the kernel is read at a fixed index")
      (singleton_kernel o && n_channels o = 1)
  in
  rerouted ~what:"1x1 window" (conv "cdb_k11" ~b:[ 2 ] ~o:[ 6; 6; 4 ] ~i:[ 1; 1; 4 ] ~spec:s2);
  rerouted ~what:"1x1 window, stride 2"
    (conv "cdb_k11s" ~b:[ 2 ] ~o:[ 7; 7; 4 ] ~i:[ 1; 1; 4 ] ~spec:s2s);
  rerouted ~what:"1-D width-1 window" (conv "cdb_k1" ~b:[ 2 ] ~o:[ 6; 4 ] ~i:[ 1; 4 ] ~spec:s1d)
