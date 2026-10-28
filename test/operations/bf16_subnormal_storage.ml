(* Where bfloat16 subnormals survive, and where Metal flushes them (gh-ocannl-1516).

   bfloat16 has f32's exponent range, so its subnormals (|x| < 2^-126) are f32 subnormals too. The
   issue's report (staging#1316) was that on Metal "2^-130 reads back 0 through every op", taken
   then for a property of [bfloat] storage. It is not one. Measured on M4 Max / macOS 26, a
   subnormal survives every leg that moves bits: the host round trip (Metal's host transfer is a
   memcpy), a bfloat-to-bfloat copy kernel, and the bfloat-to-float widening load. It is flushed by
   every leg that runs the float ALU: the float-to-bfloat narrowing store, f32 addition, and bfloat
   addition (computed in float). Each such leg gives the zero of the input's sign. The MSL
   specification permits flushing f32 denormal operands and results to zero even with fast math off,
   and Metal has no compile option to keep them, so this is the boundary, not a defect of ours. A
   bit-level narrowing helper would only rescue an f32 subnormal copied from storage into a bfloat
   tensor: the arithmetic upstream of nearly every store has flushed it already.

   CUDA and HIP compile with fast math, and CUDA's [--use_fast_math] implies [--ftz=true]; measured
   for gh-ocannl-1680, that flushes less than its name suggests. On CUDA (RTX 5070 Ti, sm_120) only
   the f32 additions flush: they are [add.ftz.f32]. The widening load is a bit move ([mov.b32
   {0,h}]), the narrowing store a [cvt.rn.bf16.f32] (emitted without [.ftz]), and bfloat addition
   the native [fma.rn.bf16], all of which keep subnormals. HIP's [-ffast-math] leaves AMDGPU's f32
   denormal mode alone (that is [-fgpu-flush-denormals-to-zero], which we do not pass), so every HIP
   leg is IEEE: measured on gfx1151 (minix, unified memory) and gfx1102 (tuf, discrete).

   The claims are per backend, each the IEEE result except where that backend's [flushes] says the
   leg flushes a subnormal input to the zero of its sign. On the C backends (without
   [cc_backend_fast_math]) and HIP no leg flushes; every value is exactly representable in both
   formats, so those legs are bitwise exact. On Metal every leg but the moving ones flushes, on CUDA
   only the f32 arithmetic. The normal controls (+-2^-126, 1.5) are exact everywhere. Each claim
   pins its hardware's measured behaviour, so it fails if that backend starts or stops flushing. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "bf16-subnormal-storage backend: %s\n" backend_name
let is_metal = String.equal backend_name "metal"
let is_cuda = String.equal backend_name "cuda"
let is_hip = String.equal backend_name "hip"

(* [-ffast-math] may link a startup file that sets flush-to-zero (GCC's [crtfastmath.o]), so the C
   claim holds only for the default IEEE compile. *)
let ieee_cpu =
  Ir.Schedule.backend_is_cpu backend_name
  && not (Utils.get_global_flag ~default:false ~arg_name:"cc_backend_fast_math")

let bf = Ops.bfloat16
let f32 = Ops.single

(* bfloat16 subnormals: the smallest (code 0x0001), the issue's 2^-130, a negative one, the largest
   (code 0x007F). Then the normal controls. *)
let inputs = [ 0x1p-133; 0x1p-130; -0x1p-127; 0x1.fcp-127; 0x1p-126; -0x1p-126; 1.5 ]
let n = List.length inputs

(* What a leg does to its value: only moves bits ([`Moves]), narrows an f32 to bfloat ([`Narrows]),
   or adds at f32 ([`F32_alu]) or at bfloat ([`Bf16_alu]) precision; and its IEEE result. *)
type kind = [ `Moves | `Narrows | `F32_alu | `Bf16_alu ]
type leg = { label : string; kind : kind; ieee : float -> float }

let run () =
  let bnode = Ll_test.node_factory ~prec:bf ~first_id:23100 ~dims:[| n |] () in
  let fnode = Ll_test.node_factory ~prec:f32 ~first_id:23200 ~dims:[| n |] () in
  let in_bf = bnode "bsn_in_bf" and in_f = fnode "bsn_in_f" in
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let get t = Ll_test.get t i in
  let double t p = LL.Binop (Ops.Add, (get t, p), (get t, p)) in
  let twice x = 2. *. x in
  (* Each leg: its output node and the scalar it stores. *)
  let legs =
    [
      ({ label = "bfloat copy"; kind = `Moves; ieee = Fn.id }, bnode "bsn_copy", get in_bf);
      ( { label = "bfloat-to-float widening"; kind = `Moves; ieee = Fn.id },
        fnode "bsn_widen",
        get in_bf );
      ( { label = "float-to-bfloat narrowing"; kind = `Narrows; ieee = Fn.id },
        bnode "bsn_narrow",
        get in_f );
      ({ label = "f32 addition"; kind = `F32_alu; ieee = twice }, fnode "bsn_add_f", double in_f f32);
      ( { label = "bfloat addition"; kind = `Bf16_alu; ieee = twice },
        bnode "bsn_add_bf",
        double in_bf bf );
      (* Binary operations compute at their result's precision (every binop is homogeneous), so the
         f32 addition of widened bfloat loads needs an f32 destination: stored to a bfloat node, the
         same operands render as the bfloat addition above. *)
      ( { label = "f32 addition of bfloat loads"; kind = `F32_alu; ieee = twice },
        fnode "bsn_add_w",
        double in_bf f32 );
    ]
  in
  let outs = List.map legs ~f:(fun (_, t, _) -> t) in
  List.iter (in_bf :: in_f :: outs) ~f:Ll_test.materialize;
  let body =
    List.fold legs ~init:LL.Noop ~f:(fun acc (_, t, v) -> Ll_test.seq acc (Ll_test.set t i v))
  in
  let o = Ll_test.optimize ~name:"bsn" (Ll_test.loop_n s n body) in
  let vs = Array.of_list inputs in
  let got =
    Ll_test.execute ~name:"bsn" o
      ~seed:([ (in_bf, vs); (in_f, vs) ] @ List.map outs ~f:(fun t -> (t, Ll_test.blank n)))
      ~read:(in_bf :: in_f :: outs)
  in
  let host label = { label = label ^ " host round trip"; kind = `Moves; ieee = Fn.id } in
  (* The two inputs, read back after the kernel ran, are the host round trip. *)
  List.zip_exn ([ host "bfloat"; host "f32" ] @ List.map legs ~f:(fun (l, _, _) -> l)) got

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let subnormal x = Float.(abs x < 0x1p-126)

(* Which legs flush a subnormal input to the zero of its sign, per backend (measured). *)
let no_leg_flushes (_ : kind) = false

let metal_flushes : kind -> bool = function
  | `Moves -> false
  | `Narrows | `F32_alu | `Bf16_alu -> true

(* [--ftz=true] reaches the f32 instructions only. *)
let cuda_flushes : kind -> bool = function
  | `F32_alu -> true
  | `Moves | `Narrows | `Bf16_alu -> false

let expected ~flushes leg x =
  if flushes leg.kind && subnormal x then Float.copysign 0. x else leg.ieee x

let () =
  Verdict.case "bfloat16 subnormal storage" (fun () ->
      let results = run () in
      List.iter results ~f:(fun (leg, got) ->
          Stdio.eprintf "(not part of the golden) %s %s:" backend_name leg.label;
          Array.iter got ~f:(fun v -> Stdio.eprintf " %h" v);
          Stdio.eprintf "\n");
      let cells =
        List.concat_map results ~f:(fun (leg, got) ->
            List.mapi inputs ~f:(fun j x -> (leg, x, got.(j))))
      in
      let report ok (leg, x, v) =
        if not ok then
          Stdio.eprintf "(not part of the golden) %s %s(%h) = %h\n" backend_name leg.label x v;
        ok
      in
      let claim ~when_ ~flushes label =
        gated_all ~when_ ~on:backend_name label cells ~f:(fun ((leg, x, v) as c) ->
            report (bits_equal v (expected ~flushes leg x)) c)
      in
      claim ~when_:ieee_cpu ~flushes:no_leg_flushes
        "C backends: every leg keeps bfloat16 subnormals, bitwise IEEE";
      claim ~when_:is_metal ~flushes:metal_flushes
        "Metal: storage, copies and widening keep bfloat16 subnormals bitwise; narrowing and \
         arithmetic flush them to the signed zero, exact on normals";
      claim ~when_:is_cuda ~flushes:cuda_flushes
        "CUDA: storage, copies, widening, narrowing and bfloat arithmetic keep bfloat16 subnormals \
         bitwise; f32 arithmetic flushes them to the signed zero, exact on normals";
      claim ~when_:is_hip ~flushes:no_leg_flushes
        "HIP: every leg keeps bfloat16 subnormals, bitwise IEEE")
