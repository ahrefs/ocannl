(* The GPU packed path's own precision gate (gh-ocannl-1295). The [`Packed_struct] rendering of a
   [Vectorized] loop moves whole 128-bit packs and computes every lane as a scalar, so it admits
   every float precision with a full-width pack type -- half and bf16 as well as f32 and f64 -- but
   only where that precision is the storage precision of every node it packs, so a pack is always a
   run of the node's own cells.

   Every leg runs one [Vectorized] loop and its [Serial] twin on the same backend and compares them
   bitwise: packing moves memory, never a value. The extent, 27, is not a lane multiple, so the
   packed rendering's serial remainder runs too. The operands are exact in the narrowest precision
   used and nonzero, so a lane mapped to the wrong cell changes the value (the only zeros are a
   gate's constant, on a loop that declines packing). The twin is in turn checked against an f64
   reference, so the two cannot agree by both being wrong, and a sentinel stays in any cell nobody
   wrote. Each lane's arithmetic is one operation (an explicit [FMA] at most), so no compiler
   contraction can treat the two kernel shapes differently.

   Legs: 1. half and bf16 [Mul]/[FMA] loops render packed on the GPU backends ([half8_t],
   [bfloat16x8_t] loads and stores), equal to the twin. 2. Operators the per-lane renderer used to
   decline -- [exp], [max], [pow], a constant integer power over negative bases -- render packed at
   f32 and half through the same syntax hooks and integer-power helper as the serial path; a [Where]
   arm or a [Relu_gate]'s gated operand mentioning the loop declines instead (a pack load is
   unconditional). 3. An fp8 store fed from f32 declines the packed path: Metal computes fp8 in f32
   and stores it as a byte, and before the gate checked the store's storage precision it
   reinterpreted the byte buffer as [float4_t] packs (wrong values, writes past the cells). 4. Every
   type a kernel of this test casts a pointer to is declared 16-byte aligned in the same source --
   the narrow pack types were not, so a packed load would have been compiled as a 2-byte-aligned
   access. *)

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
let is_metal = String.equal backend_name "metal"
let len = 27

(* Exact in fp16 and bf16 (at most five fraction bits over [1, 2)), positive, and distinct per
   cell. *)
let a_of i = 1. +. (Float.of_int i /. 32.)
let b_of i = 2. -. (Float.of_int i /. 64.)
let c_of i = 0.25 +. (Float.of_int i /. 128.)

(* The loop bodies, [out[i] = ...] over operands [a], [b], [c] read at [i]:

   - [Mul], [Fma]: [a * b], [fma(a, b, c)]; - [Exp], [Max], [Pow]: [exp(a)], [max(a, b)], [pow(a,
   b)] -- operators the per-lane renderer used to decline, each rendered through the backend's
   syntax hooks (Metal's half [exp] and [pow] widen to f32 and narrow once); - [Powi]: [(-a)^3], a
   constant integer exponent over negative bases, which takes the integer-power helper in the packed
   lanes as in the serial remainder (an f32 [pow] of a negative base is NaN on Metal); -
   [Where_arm]: [c[0] ? a : b], a conditional arm mentioning the loop: a pack load would read it
   unconditionally, so the loop declines the packed path; - [Relu_gate]: [relu_gate(a - 1.4, b)],
   whose second operand is read only where the first is positive: declines likewise. *)
type shape = Mul | Fma | Exp | Max | Pow | Powi | Where_arm | Relu_gate

let shape_name = function
  | Mul -> "mul"
  | Fma -> "fma"
  | Exp -> "exp"
  | Max -> "max"
  | Pow -> "pow"
  | Powi -> "powi"
  | Where_arm -> "where"
  | Relu_gate -> "relu_gate"

let reference shape i =
  let a = a_of i and b = b_of i and c = c_of i in
  match shape with
  | Mul -> a *. b
  | Fma -> (a *. b) +. c
  | Exp -> Float.exp a
  | Max -> Float.max a b
  | Pow -> Float.( ** ) a b
  | Powi -> Float.( ** ) (-.a) 3.
  | Where_arm -> a
  | Relu_gate -> if Float.(a -. 1.4 > 0.) then b else 0.

(* One elementwise loop over [len] cells under routine [name], every node at [prec], the loop
   [axis]-annotated. *)
let run ~prec ~shape ~axis ~name ~first_id =
  let node = Ll_test.node_factory ~prec ~first_id ~dims:[| len |] () in
  let a = node (name ^ "_a") and b = node (name ^ "_b") and c = node (name ^ "_c") in
  let out = node (name ^ "_out") in
  List.iter [ a; b; c; out ] ~f:Ll_test.materialize;
  let s = Ll_test.sym () in
  let at = [| Ll_test.iter s |] in
  let get tn = (Ll_test.get tn at, prec) in
  let const x = (Ll_test.c x, prec) in
  let value =
    match shape with
    | Mul -> LL.Binop (Ops.Mul, get a, get b)
    | Fma -> LL.Ternop (Ops.FMA, get a, get b, get c)
    | Exp -> LL.Unop (Ops.Exp, get a)
    | Max -> LL.Binop (Ops.Max, get a, get b)
    | Pow -> LL.Binop (Ops.ToPowOf, get a, get b)
    | Powi -> LL.Binop (Ops.ToPowOf, (LL.Unop (Ops.Neg, get a), prec), const 3.)
    | Where_arm -> LL.Ternop (Ops.Where, (Ll_test.get c [| Ll_test.fixed 0 |], prec), get a, get b)
    | Relu_gate -> LL.Binop (Ops.Relu_gate, (LL.Binop (Ops.Sub, get a, const 1.4), prec), get b)
  in
  let body = Ll_test.loop_n ~axis s len (Ll_test.set out at value) in
  let o = Ll_test.optimize ~name body in
  let seed f = Array.init len ~f in
  List.hd_exn
    (Ll_test.execute ~name o
       ~seed:[ (a, seed a_of); (b, seed b_of); (c, seed c_of); (out, Ll_test.blank len) ]
       ~read:[ out ])

let same_bits x y = Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)

(* Every routine this test generated, for the source-wide alignment claim. *)
let routines = ref []

(* The packed and twin runs of one leg, with the parity and reference claims every leg makes; the
   [Vectorized] loop's source is returned for the leg's structural claims. *)
let leg ~prec ~prec_name ~tol ~first_id shape =
  let name = Printf.sprintf "pk_%s_%s" prec_name (shape_name shape) in
  let label = Printf.sprintf "%s %s" prec_name (shape_name shape) in
  let packed = run ~prec ~shape ~axis:LL.Vectorized ~name ~first_id in
  let twin = run ~prec ~shape ~axis:LL.Serial ~name:(name ^ "_twin") ~first_id:(first_id + 50) in
  routines := name :: (name ^ "_twin") :: !routines;
  p_alli (label ^ ": the Serial twin is within the precision's rounding of the f64 reference")
    (Array.to_list twin) ~f:(fun i v ->
      let want = reference shape i in
      let ok = Float.(abs (v - want) <= tol * abs want) in
      if not ok then
        Stdio.eprintf "(not part of the golden) %s cell %d: twin %h, reference %h\n" label i v want;
      ok);
  p_alli (label ^ ": the Vectorized loop equals its Serial twin bitwise") (Array.to_list packed)
    ~f:(fun i v ->
      let ok = same_bits v twin.(i) in
      if not ok then
        Stdio.eprintf "(not part of the golden) %s cell %d: packed %h, twin %h\n" label i v twin.(i);
      ok);
  p
    (label ^ ": the Serial twin renders no pack")
    (not (String.is_substring (Generated.read (name ^ "_twin")) ~substring:"reinterpret_cast"));
  (label, Generated.read name)

let renders_packed ~pack src =
  String.is_substring src ~substring:(Printf.sprintf "const %s*>(&" pack)
  && String.is_substring src ~substring:(Printf.sprintf " %s*>(&" pack)

let packed_leg ~prec ~prec_name ~pack ~tol ~first_id shape =
  let label, src = leg ~prec ~prec_name ~tol ~first_id shape in
  gated ~when_:is_gpu ~on:backend_name
    (Printf.sprintf "%s: the Vectorized loop loads and stores whole %s packs" label pack)
    (renders_packed ~pack src);
  src

let () =
  Verdict.case "narrow float packs" (fun () ->
      (* Rounding of the twin against f64: one rounding of a product (and one of the sum, for [FMA]
         at the fused path's single rounding or the unfused path's two) at 11 and 8 significant
         bits. *)
      List.iteri [ Mul; Fma ] ~f:(fun k shape ->
          ignore
            (packed_leg ~prec:Ops.half ~prec_name:"half" ~pack:"half8_t" ~tol:0x1p-10
               ~first_id:(29_000 + (100 * k))
               shape
              : string);
          ignore
            (packed_leg ~prec:Ops.bfloat16 ~prec_name:"bf16" ~pack:"bfloat16x8_t" ~tol:0x1p-7
               ~first_id:(29_200 + (100 * k))
               shape
              : string));
      (* The f32 pack the gate always admitted, so the alignment claim below also covers the
         existing [float4_t]. *)
      ignore
        (packed_leg ~prec:Ops.single ~prec_name:"f32" ~pack:"float4_t" ~tol:0x1p-23 ~first_id:29_400
           Mul
          : string))

let count src ~pattern = List.length (String.substr_index_all src ~may_overlap:false ~pattern)

(* Calls of the function [fn] in [src]: [fn(] not preceded by an identifier character, so that a
   kernel named [pk_half_exp] or a helper named [ocannl_powi_f32] is not one. *)
let calls src ~fn =
  String.substr_index_all src ~may_overlap:false ~pattern:(fn ^ "(")
  |> List.count ~f:(fun at ->
      at = 0
      ||
      let c = src.[at - 1] in
      not (Char.is_alphanum c || Char.equal c '_'))

let () =
  Verdict.case "per-lane operators" (fun () ->
      (* Tolerances against f64: f32 [exp] and [pow] under the backends' fast math functions are
         within a few f32 ulps, the rest one f32 rounding; at half every result is one rounding of
         an f32 value at 11 significant bits. *)
      List.iteri [ Exp; Max; Pow; Powi ] ~f:(fun k shape ->
          let f32_tol = match shape with Exp | Pow -> 0x1p-18 | _ -> 0x1p-22 in
          let f32_src =
            packed_leg ~prec:Ops.single ~prec_name:"f32" ~pack:"float4_t" ~tol:f32_tol
              ~first_id:(30_000 + (100 * k))
              shape
          in
          let half_src =
            packed_leg ~prec:Ops.half ~prec_name:"half" ~pack:"half8_t" ~tol:0x1p-10
              ~first_id:(30_500 + (100 * k))
              shape
          in
          match shape with
          | Powi ->
              gated_all ~when_:is_gpu ~on:backend_name
                "a constant integer power renders through the integer-power helper in the packed \
                 lanes, never through pow"
                [ f32_src; half_src ] ~f:(fun src ->
                  String.is_substring src ~substring:"ocannl_powi_f32(" && calls src ~fn:"pow" = 0)
          | Exp ->
              let widened = count half_src ~pattern:"(half)exp((float)("
              and calls = calls half_src ~fn:"exp" in
              gated ~when_:is_metal ~on:backend_name
                ~detail:(fun () -> Printf.sprintf "%d of %d exp calls widened" widened calls)
                "every exp call in a packed half loop on Metal is the f32 builtin narrowed once"
                (widened > 0 && widened = calls)
          | Mul | Fma | Max | Pow | Where_arm | Relu_gate -> ());
      (* Conditionally evaluated operands mentioning the loop decline: the loop geometry is the
         [Mul] legs', which pack, so the operator is what declines. *)
      List.iteri [ Where_arm; Relu_gate ] ~f:(fun k shape ->
          List.iteri
            [ (Ops.single, "f32", 0x1p-22); (Ops.half, "half", 0x1p-10) ]
            ~f:(fun j (prec, prec_name, tol) ->
              let label, src =
                leg ~prec ~prec_name ~tol ~first_id:(31_000 + (200 * k) + (100 * j)) shape
              in
              p
                (label
               ^ ": an operand evaluated only conditionally keeps the loop off the packed path")
                (not (String.is_substring src ~substring:"reinterpret_cast")))))

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
