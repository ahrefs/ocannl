(* gh-ocannl-1457: under a compile that pins contraction off, a tensorized matmul's [Tile_mma]
   scalar fallback rounds exactly as the serial loop does -- twice per k step, in serial order -- on
   NON-DYADIC f32 operands, where one rounding per k step and two differ.

   [Schedule.Tensorize] used to write every floating fallback as an explicit [FMA] (gh-ocannl-998),
   while the serial loop's [acc + a * b] is the C compiler's to contract. Under
   [cc_backend_fp_contract=off], which [profile=reproducible] pins, nothing contracts the serial
   loop, so a packed seed and the untuned kernel differed by an ulp wherever the seed won the timing
   (gh-ocannl-1682's residue). The renderer now fuses the fallback only where the compile leaves
   contraction to the compiler ([C_syntax_config.compiler_contraction_off]).

   The register tile itself is held to the same equality by [schedule_pad]'s
   [runtest-schedule_pad-contract_off] rule; this test pins the fallback, which is what the
   reproducible profile renders ([cc_vector_bytes=0] makes the register tile decline). Both pins are
   stated in-process before the first compile. Values go through the bits, and the kernel is held to
   a host emulation of the serial f32 chain, so the claim says which arithmetic ran, not only that
   two kernels agree; the data are shown to separate one rounding from two by the same emulation
   fused.

   Two legs. The cc leg holds packed seeds (found through the CPU seeder) on 33x40x37 data. The GPU
   leg (gh-ocannl-1834), open on CUDA and HIP, holds a hand-applied [Tensorize] on 48x32x37 data
   under [cuda_fast_math=false] / [hip_fast_math=false], also stated in-process before the first
   compile: nvrtc then compiles with [--fmad=false] and hiprtc with [-ffp-contract=off]. The seeder
   cannot supply those seeds -- it withholds an f32 tensorized capability on CUDA (wmma needs tf32,
   off here) and on HIP (rocWMMA has no f32 shape) -- and for the same reason the statement declines
   to the lane-0 scalar fallback there, which the leg claims from the routine's census rather than
   assuming. Its schedule is [schedule_mma_matmul]'s: the zeroing's column loop and the lane loop
   are both Workgroup(32), hence the GPU data's 32 columns. Every claim of a leg is skipped
   elsewhere; the digits of any disagreement go to stderr. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
open Verdict.Claims

(* The pins, set as the first thing the executable does. [cc_backend_fp_contract=off] is the
   environment pin [fp_contract_pin_scan] reads. The rest are stated in process, ranked above every
   source a run is given (a contradicting commandline argument is refused here, naming the key):
   [fp_contract=false] keeps lowering from writing the serial loop's [acc + a * b] as an explicit
   FMA, which no compile pin would undo; [tf32_matmuls=false] keeps CUDA's wmma off the f32
   statement on tf32 hardware. *)
let () = Unix.putenv "OCANNL_CC_BACKEND_FP_CONTRACT" "off"
let () = Utils.state_config ~key:"fp_contract" ~data:"false"
let () = Utils.state_config ~key:"cc_vector_bytes" ~data:"0"
let () = Utils.state_config ~key:"cuda_fast_math" ~data:"false"
let () = Utils.state_config ~key:"hip_fast_math" ~data:"false"
let () = Utils.state_config ~key:"tf32_matmuls" ~data:"false"
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cc = List.mem [ "cc"; "multidev_cc" ] backend_name ~equal:String.equal
let on_cuda_or_hip = List.mem [ "cuda"; "hip" ] backend_name ~equal:String.equal
let accum_prec = Ll_test.backend_accum_prec

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* One matmul problem [c[i,j] = sum_k a[i,k] * b[k,j]]: its operands, and the host emulation of its
   f32 serial chain, k ascending from +0, rounded twice per k step ([unfused]) and once ([fused]).
   An f32 product is exact in a double and an f32 sum's double rounding is innocuous (53 >= 2 * 24 +
   2), so [unfused] is the correctly rounded two-rounding chain. [fused] rounds the double [fma]
   once more to f32: not guaranteed correctly rounded, which is fine for its one use -- showing the
   data move under a single rounding. *)
type problem = {
  m : int;
  n : int;
  k : int;
  a_vals : float array;
  b_vals : float array;
  unfused : float array;
  fused : float array;
}

let r32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Steps of a tenth and of a third, cycled over prime moduli, as in [schedule_pad]: every product
   and partial sum rounds at f32. *)
let problem ~m ~n ~k =
  let a_vals =
    Array.init (m * k)
      ~f:(Ll_test.cycle_flat ~dims:[| m; k |] ~modulus:97 ~offset:(-1.3) ~stride:0.1)
  in
  let b_vals =
    Array.init (k * n)
      ~f:(Ll_test.cycle_flat ~dims:[| k; n |] ~modulus:89 ~offset:(-0.7) ~stride:(1. /. 3.))
  in
  let chain ~step =
    Array.init (m * n) ~f:(fun c ->
        let i = c / n and j = c % n in
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc := step !acc (r32 a_vals.((i * k) + kk)) (r32 b_vals.((kk * n) + j))
        done;
        !acc)
  in
  {
    m;
    n;
    k;
    a_vals;
    b_vals;
    unfused = chain ~step:(fun acc a b -> r32 (acc +. r32 (a *. b)));
    fused = chain ~step:(fun acc a b -> r32 (Stdlib.Float.fma a b acc));
  }

let cpu = problem ~m:33 ~n:40 ~k:37
let bm = 16
let simd_width = 32
let gpu = problem ~m:(3 * bm) ~n:simd_width ~k:37
let bits x = Int64.bits_of_float x
let bitwise a b = Array.equal (fun x y -> Int64.equal (bits x) (bits y)) a b

let mk pb name =
  let ma =
    TDSL.ndarray ~prec:Ir.Ops.single pb.a_vals
      ~label:[ name ^ "a" ]
      ~input_dims:[ pb.k ] ~output_dims:[ pb.m ] ()
  in
  let mb =
    TDSL.ndarray ~prec:Ir.Ops.single pb.b_vals
      ~label:[ name ^ "b" ]
      ~input_dims:[ pb.n ] ~output_dims:[ pb.k ] ()
  in
  let%op nc = ma * mb in
  nc

let max_diff pb got =
  Array.foldi got ~init:0. ~f:(fun c acc g -> Float.max acc (Float.abs (g -. pb.unfused.(c))))

let cpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    simd_vector = { Ir.Vector_file.bytes = 32; registers = 16 };
  }

(* The claims, each leg open on its backends and skipped by name elsewhere, each written once
   (gh-ocannl-1703). *)
let cc_leg = Verdict.leg ~when_:on_cc ~on:backend_name ()
let gpu_leg = Verdict.leg ~when_:on_cuda_or_hip ~on:backend_name ()

let results =
  lazy
    (let seeds = ref [] in
     let serial =
       let nc = mk cpu "tfu0" in
       let ctx, routine =
         Context.compile
           ~lowered_transform:(fun opt ->
             seeds :=
               List.filter
                 (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt)
                 ~f:(fun p ->
                   p.Autotune.sk_mma && (not p.Autotune.sk_gpu) && (not p.Autotune.sk_epilogue)
                   && Option.is_none p.Autotune.sk_tile);
             [ opt ])
           (Context.auto ())
           (named "tfu_serial" (Train.forward nc))
           Ir.Indexing.Empty
       in
       Context.get_values (Context.run ctx routine) nc.Tensor.value
     in
     let runs =
       List.mapi (List.take !seeds 3) ~f:(fun i (p : Autotune.sketch_params) ->
           let nc = mk cpu (Printf.sprintf "tfu%d" (i + 1)) in
           let transform opt = Sched.apply (Autotune.sketch_schedule ~accum_prec ~p opt) opt in
           let ctx, routine =
             Context.compile
               ~lowered_transform:(fun o -> [ transform o ])
               (Context.auto ())
               (named (Printf.sprintf "tfu_seed_%d" i) (Train.forward nc))
               Ir.Indexing.Empty
           in
           let got = Context.get_values (Context.run ctx routine) nc.Tensor.value in
           Stdio.eprintf
             "packed seed %d hoist=%b grid=%b: max |fallback - unfused chain| = %h (not part of \
              the golden)\n\
              %!"
             i p.sk_hoist p.sk_grid (max_diff cpu got);
           (routine.Context.mma.Ir.C_syntax.tensorization, got))
     in
     (serial, runs))

(* [schedule_mma_matmul]'s tensorized schedule: expand the zeroing, split its rows into Grid x
   Serial(16) and retype its columns to Workgroup (each lane zeroes its own column; the extent
   matches the lane loop's, as barrier-strength uniformity requires once a [Tile_mma] is present),
   split the accumulation's rows the same way, then [Tensorize] the serial 16 x 32 x 37 micro-kernel
   inside a fresh Workgroup(32) lane loop. *)
let gpu_schedule ~out (opt : LL.optimized) : Sched.schedule =
  let i, j, k =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k ~simd_width () in
  [ ez; sp_zi; rz; sp_i; tz ]

(* What the GPU leg's claims read, each its own field. *)
type gpu_results = {
  contraction_off : bool;  (** The backend's compile pins contraction off. *)
  serial : float array;  (** The untuned serial loop's values. *)
  tensorization : Ir.C_syntax.tensorization;  (** The tensorized routine's mma census. *)
  got : float array;  (** The tensorized routine's values. *)
}

let gpu_results =
  lazy
    (let contraction_off =
       (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.compiler_contraction_off ()
     in
     let serial =
       let nc = mk gpu "tfg0" in
       let ctx, routine =
         Context.compile
           ~lowered_transform:(fun opt -> [ opt ])
           (Context.auto ())
           (named "tfg_serial" (Train.forward nc))
           Ir.Indexing.Empty
       in
       Context.get_values (Context.run ctx routine) nc.Tensor.value
     in
     let nc = mk gpu "tfg1" in
     let tn = nc.Tensor.value in
     let ctx, routine =
       Context.compile
         ~lowered_transform:(fun o -> [ Sched.apply (gpu_schedule ~out:tn o) o ])
         (Context.auto ())
         (named "tfg_tensorized" (Train.forward nc))
         Ir.Indexing.Empty
     in
     (* The schedule expands the target's zero ahead of the [Tile_mma]: a fresh buffer is zero and
        would hide a dropped expansion, so the target starts at [1000 + c], off every value the
        chain reaches, and the bitwise claim then needs every cell overwritten. *)
     let numel = Array.fold (Tn.dims_without_padding tn) ~init:1 ~f:( * ) in
     let ctx = Context.set_values ctx tn (Array.init numel ~f:(fun c -> Float.of_int (1000 + c))) in
     let got = Context.get_values (Context.run ctx routine) tn in
     Stdio.eprintf
       "%s: serial max |loop - unfused chain| = %h, tensorized max |fallback - unfused chain| = %h \
        (not part of the golden)\n\
        %!"
       backend_name (max_diff gpu serial) (max_diff gpu got);
     { contraction_off; serial; tensorization = routine.Context.mma.Ir.C_syntax.tensorization; got })

let () =
  p "the host emulation separates one rounding per k step from two"
    (not (bitwise cpu.unfused cpu.fused));
  p "the GPU leg's data separate one rounding per k step from two"
    (not (bitwise gpu.unfused gpu.fused));
  in_leg ~leg:cc_leg "the serial loop computes the two-rounding f32 chain bitwise" (fun label ->
      p label (bitwise (fst (Lazy.force results)) cpu.unfused));
  in_leg ~leg:cc_leg "every packed seed renders its Tile_mma through the scalar fallback"
    (fun label ->
      p_all ~min:2 label
        (snd (Lazy.force results))
        ~f:(fun (t, _) -> Ir.C_syntax.equal_tensorization t Ir.C_syntax.Scalar_fallback));
  in_leg ~leg:cc_leg "every packed seed's fallback computes the two-rounding f32 chain bitwise"
    (fun label ->
      p_all ~min:2 label (snd (Lazy.force results)) ~f:(fun (_, got) -> bitwise got cpu.unfused));
  in_leg ~leg:gpu_leg "the CUDA/HIP kernel compile pins contraction off under precise math"
    (fun label -> p label (Lazy.force gpu_results).contraction_off);
  in_leg ~leg:gpu_leg "the CUDA/HIP serial loop computes the two-rounding f32 chain bitwise"
    (fun label -> p label (bitwise (Lazy.force gpu_results).serial gpu.unfused));
  in_leg ~leg:gpu_leg
    "the CUDA/HIP tensorized f32 matmul renders its Tile_mma through the scalar fallback"
    (fun label ->
      p label
        (Ir.C_syntax.equal_tensorization (Lazy.force gpu_results).tensorization
           Ir.C_syntax.Scalar_fallback));
  in_leg ~leg:gpu_leg "the CUDA/HIP fallback computes the two-rounding f32 chain bitwise"
    (fun label -> p label (bitwise (Lazy.force gpu_results).got gpu.unfused))
