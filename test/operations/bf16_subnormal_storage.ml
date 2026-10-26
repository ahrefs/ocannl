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

   The claims are per backend family. On the C backends (without [cc_backend_fast_math]) every leg
   is IEEE and exact, since each value is exactly representable in both formats. On Metal the moving
   legs are bitwise and the ALU legs flush to the signed zero, with the normal controls (+-2^-126,
   1.5) exact. That claim pins the hardware's behaviour, so it fails if a future Metal stops
   flushing. CUDA and HIP compile with fast math, and CUDA's [--use_fast_math] implies [--ftz=true].
   They print their table to stderr and make no claim. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "bf16-subnormal-storage backend: %s\n" backend_name
let is_metal = String.equal backend_name "metal"

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

(* Whether a leg only moves bits ([`Moves]) or runs the float ALU ([`Computes]), and the IEEE result
   for an input. *)
type leg = { label : string; kind : [ `Moves | `Computes ]; ieee : float -> float }

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
      ( { label = "float-to-bfloat narrowing"; kind = `Computes; ieee = Fn.id },
        bnode "bsn_narrow",
        get in_f );
      ( { label = "f32 addition"; kind = `Computes; ieee = twice },
        fnode "bsn_add_f",
        double in_f f32 );
      ( { label = "bfloat addition"; kind = `Computes; ieee = twice },
        bnode "bsn_add_bf",
        double in_bf bf );
      ( { label = "f32 addition of bfloat loads, narrowed"; kind = `Computes; ieee = twice },
        bnode "bsn_add_w",
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

(* What Metal gives: the IEEE result where the leg moves bits or the input is normal, and the zero
   of the input's sign where the leg computes on a subnormal. *)
let metal_expected leg x =
  match leg.kind with
  | `Computes when subnormal x -> Float.copysign 0. x
  | `Moves | `Computes -> leg.ieee x

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
      gated_all ~when_:ieee_cpu ~on:backend_name
        "C backends: every leg keeps bfloat16 subnormals, bitwise IEEE" cells
        ~f:(fun ((leg, x, v) as c) -> report (bits_equal v (leg.ieee x)) c);
      gated_all ~when_:is_metal ~on:backend_name
        "Metal: storage, copies and widening keep bfloat16 subnormals bitwise; narrowing and \
         arithmetic flush them to the signed zero, exact on normals"
        cells ~f:(fun ((leg, x, v) as c) -> report (bits_equal v (metal_expected leg x)) c))
