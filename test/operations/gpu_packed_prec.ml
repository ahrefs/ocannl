(* The GPU packed path's own precision gate (gh-ocannl-1295). The [`Packed_struct] rendering of a
   [Vectorized] loop moves whole 128-bit packs and computes every lane as a scalar, so it admits
   every float precision with a full-width pack type -- half and bf16 as well as f32 and f64 -- but
   only where that precision is the storage precision of every node it packs, so a pack is always a
   run of the node's own cells.

   Every leg runs one [Vectorized] loop and its [Serial] twin on the same backend and compares them
   bitwise: packing moves memory, never a value. The extent, 27, is not a lane multiple, so the
   packed rendering's serial remainder runs too. The operands are exact in the narrowest precision
   used and nonzero, so a lane mapped to the wrong cell changes the value and no signed zero is in
   play; the twin is in turn checked against an f64 reference, so the two cannot agree by both being
   wrong (a sentinel stays in any cell nobody wrote). The products are single operations ([Mul], an
   explicit [FMA]), so no compiler contraction can treat the two kernel shapes differently.

   Legs: 1. half and bf16 [Mul]/[FMA] loops render packed on the GPU backends ([half8_t],
   [bfloat16x8_t] loads and stores), equal to the twin; 2. an fp8 store fed from f32 declines the
   packed path: Metal computes fp8 in f32 and stores it as a byte, and before the gate checked the
   store's storage precision it reinterpreted the byte buffer as [float4_t] packs (wrong values,
   writes past the cells); 3. every type a kernel of this test casts a pointer to is declared
   16-byte aligned in the same source -- the narrow pack types were not, so a packed load would have
   been compiled as a 2-byte-aligned access. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Ops = Ir.Ops
module Generated = Test_utils.Generated

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "gpu-packed-prec backend: %s (not part of the golden)\n%!" backend_name
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let is_gpu = List.mem [ "cuda"; "hip"; "metal" ] backend_name ~equal:String.equal
let len = 27

(* Exact in fp16 and bf16 (at most five fraction bits over [1, 2)), positive, and distinct per
   cell. *)
let a_of i = 1. +. (Float.of_int i /. 32.)
let b_of i = 2. -. (Float.of_int i /. 64.)
let c_of i = 0.25 +. (Float.of_int i /. 128.)

type shape = Mul | Fma

let shape_name = function Mul -> "mul" | Fma -> "fma"

let reference shape i =
  match shape with Mul -> a_of i *. b_of i | Fma -> (a_of i *. b_of i) +. c_of i

(* One elementwise loop over [len] cells under routine [name]: [out = a * b] or [out = fma(a, b,
   c)], every node at [prec], the loop [axis]-annotated. *)
let run ~prec ~shape ~axis ~name ~first_id =
  let node = Ll_test.node_factory ~prec ~first_id ~dims:[| len |] () in
  let a = node (name ^ "_a") and b = node (name ^ "_b") and c = node (name ^ "_c") in
  let out = node (name ^ "_out") in
  List.iter [ a; b; c; out ] ~f:Ll_test.materialize;
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let get tn = (Ll_test.get tn at, prec) in
  let value =
    match shape with
    | Mul -> LL.Binop (Ops.Mul, get a, get b)
    | Fma -> LL.Ternop (Ops.FMA, get a, get b, get c)
  in
  let body = Ll_test.loop_n ~axis s len (Ll_test.set out at value) in
  let o = Ll_test.optimize ~name body in
  let seed f = Array.init len ~f in
  List.hd_exn
    (Ll_test.execute ~name o
       ~seed:[ (a, seed a_of); (b, seed b_of); (c, seed c_of); (out, Ll_test.blank len) ]
       ~read:[ out ])

let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)

(* The packed and twin runs of one leg, with the parity and reference claims every leg makes. *)
let leg ~label ~prec ~shape ~tol ~name ~first_id =
  let packed = run ~prec ~shape ~axis:LL.Vectorized ~name ~first_id in
  let twin = run ~prec ~shape ~axis:LL.Serial ~name:(name ^ "_twin") ~first_id:(first_id + 50) in
  p_alli (label ^ ": the Serial twin is within the precision's rounding of the f64 reference")
    (Array.to_list twin) ~f:(fun i v ->
      let want = reference shape i in
      Float.(abs (v - want) <= tol * abs want));
  p_alli (label ^ ": the Vectorized loop equals its Serial twin bitwise") (Array.to_list packed)
    ~f:(fun i v ->
      let ok = same_bits v twin.(i) in
      if not ok then
        Stdio.eprintf "(not part of the golden) %s cell %d: packed %h, twin %h\n" label i v twin.(i);
      ok);
  p
    (label ^ ": the Serial twin renders no pack")
    (not (String.is_substring (Generated.read (name ^ "_twin")) ~substring:"reinterpret_cast"))

(* Every routine this test generated, for the source-wide alignment claim. *)
let routines = ref []

let packed_leg ~prec ~prec_name ~pack ~tol ~first_id shape =
  let name = Printf.sprintf "pk_%s_%s" prec_name (shape_name shape) in
  let label = Printf.sprintf "%s %s" prec_name (shape_name shape) in
  leg ~label ~prec ~shape ~tol ~name ~first_id;
  routines := name :: (name ^ "_twin") :: !routines;
  let src = Generated.read name in
  gated ~when_:is_gpu ~on:backend_name
    (Printf.sprintf "%s: the Vectorized loop loads and stores whole %s packs" label pack)
    (String.is_substring src ~substring:(Printf.sprintf "const %s*>(&" pack)
    && String.is_substring src ~substring:(Printf.sprintf " %s*>(&" pack))

let () =
  Verdict.case "narrow float packs" (fun () ->
      (* Rounding of the twin against f64: one rounding of a product (and one of the sum, for [FMA]
         at the fused path's single rounding or the unfused path's two) at 11 and 8 significant
         bits. *)
      List.iteri [ Mul; Fma ] ~f:(fun k shape ->
          packed_leg ~prec:Ops.half ~prec_name:"half" ~pack:"half8_t" ~tol:0x1p-10
            ~first_id:(29_000 + (100 * k))
            shape;
          packed_leg ~prec:Ops.bfloat16 ~prec_name:"bf16" ~pack:"bfloat16x8_t" ~tol:0x1p-7
            ~first_id:(29_200 + (100 * k))
            shape);
      (* The f32 pack the gate always admitted, so the alignment claim below also covers the
         existing [float4_t]. *)
      packed_leg ~prec:Ops.single ~prec_name:"f32" ~pack:"float4_t" ~tol:0x1p-23 ~first_id:29_400
        Mul)

(* [out_fp8 = in_f32 * 0.5]: the store's compute precision (f32 on Metal, where fp8 is a stored
   byte) differs from its storage precision, so a pack of compute values is not a pack of the node's
   cells. *)
let run_fp8 ~axis ~name ~first_id =
  let inp = Ll_test.node_factory ~prec:Ops.single ~first_id ~dims:[| len |] () (name ^ "_in") in
  let out =
    Ll_test.node_factory ~prec:Ops.fp8 ~first_id:(first_id + 1) ~dims:[| len |] () (name ^ "_out")
  in
  List.iter [ inp; out ] ~f:Ll_test.materialize;
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let value = LL.Binop (Ops.Mul, (Ll_test.get inp at, Ops.single), (Ll_test.c 0.5, Ops.single)) in
  let body = Ll_test.loop_n ~axis s len (Ll_test.set out at value) in
  let o = Ll_test.optimize ~name body in
  List.hd_exn
    (Ll_test.execute ~name o
       ~seed:
         [
           (inp, Array.init len ~f:(fun i -> 1. +. (Float.of_int i /. 4.))); (out, Ll_test.blank len);
         ]
       ~read:[ out ])

let () =
  Verdict.case "fp8 stores computed at f32" (fun () ->
      let packed = run_fp8 ~axis:LL.Vectorized ~name:"pk_fp8" ~first_id:29_500 in
      let twin = run_fp8 ~axis:LL.Serial ~name:"pk_fp8_twin" ~first_id:29_550 in
      routines := "pk_fp8" :: "pk_fp8_twin" :: !routines;
      (* e5m2 keeps two fraction bits: one rounding is within 2^-3 relative. *)
      p_alli "fp8: the Serial twin is within e5m2 rounding of the f64 reference"
        (Array.to_list twin) ~f:(fun i v ->
          let want = (1. +. (Float.of_int i /. 4.)) *. 0.5 in
          Float.(abs (v - want) <= 0x1p-3 * abs want));
      p_alli "fp8: the Vectorized loop equals its Serial twin bitwise" (Array.to_list packed)
        ~f:(fun i v ->
          let ok = same_bits v twin.(i) in
          if not ok then
            Stdio.eprintf "(not part of the golden) fp8 cell %d: packed %h, twin %h\n" i v twin.(i);
          ok);
      p "fp8: a store whose storage is not its compute precision renders no pack"
        (not (String.is_substring (Generated.read "pk_fp8") ~substring:"reinterpret_cast")))

(* The type named inside each [reinterpret_cast<...>] of [src]: the last word before the pointer
   star, past any address-space and [const] qualifiers. *)
let cast_types src =
  let marker = "reinterpret_cast<" in
  List.filter_map (String.substr_index_all src ~may_overlap:false ~pattern:marker) ~f:(fun at ->
      let start = at + String.length marker in
      String.index_from src start '>'
      |> Option.map ~f:(fun stop ->
          String.sub src ~pos:start ~len:(stop - start)
          |> String.rstrip ~drop:(fun c -> Char.equal c '*' || Char.is_whitespace c)
          |> String.split ~on:' ' |> List.last_exn))

(* [ty]'s definition in [src] declares 16-byte alignment: CUDA/HIP's [__align__(16)] or C++'s
   [alignas(16)], or MSL's one-member wrapper of the 16-byte [float4] vector type, aligned by its
   member. *)
let declared_16_aligned src ty =
  String.split_lines src
  |> List.exists ~f:(fun line ->
      let defines =
        String.is_substring line ~substring:"struct"
        && (String.is_suffix (String.rstrip line) ~suffix:("} " ^ ty ^ ";")
           || String.is_substring line ~substring:("struct " ^ ty ^ " {")
           || String.is_substring line ~substring:("struct alignas(16) " ^ ty ^ " {"))
      in
      defines
      && (String.is_substring line ~substring:"__align__(16)"
         || String.is_substring line ~substring:"alignas(16)"
         || String.is_substring line ~substring:("struct " ^ ty ^ " { float4 v; };")))

let () =
  Verdict.case "aligned packs" (fun () ->
      let casts =
        List.concat_map (List.rev !routines) ~f:(fun routine ->
            let src = Generated.read routine in
            List.map (cast_types src) ~f:(fun ty -> (routine, ty, src)))
      in
      gated_all ~when_:is_gpu ~on:backend_name
        "every type a pointer is cast to is declared 16-byte aligned in the same source" casts
        ~f:(fun (routine, ty, src) ->
          let ok = declared_16_aligned src ty in
          if not ok then
            Stdio.eprintf "(not part of the golden) %s casts to %s, declared without alignment\n"
              routine ty;
          ok))
