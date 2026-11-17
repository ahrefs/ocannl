(* The register tile's bf16 and fp16 A columns, widened a group of rows at a time.

   At 4 lanes of f32 the tile's A rows cross the narrow -> f32 boundary through
   [Ir.C_syntax.vec_widen_rows_macros]'s pair of builtins: on aarch64 one vector widening per group
   of four rows, read back one lane per row; elsewhere each row's own scalar conversion, the per-row
   rendering after preprocessing. (Why aarch64 differs: [cc_march_census]'s "no known defects"
   note.) Either arm must widen every code exactly as the scalar codec does, and the census only
   compiles the aarch64 arm. This test EXECUTES whichever arm the host compiler takes -- the packed
   one on an arm64 host (fp16 under clang) -- on every code of both formats, against the tile's own
   serial micro-kernel.

   A is [m x 1] narrow storage holding every code once (and three more rows, so the last group is a
   padded partial one), B is [1 x 4] of 1.0, and D is f32 seeded with -0.0: each cell is fma(a, 1,
   -0), which is [a] exactly for every code, a signed zero included, and a NaN for a NaN code. *)

(* cc-fp-contract: exact-operands -- an explicit fma by one and minus zero for every bf16/fp16 code,
   exact for every input *)

open Base
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
open Verdict.Claims
module Generated = Test_utils.Generated

(* Before any compile: the renderer reads the vector width per kernel, and 16 bytes is the width
   whose f32 tile the grouped widening renders at, whatever this host's own is. *)
let () = Unix.putenv "OCANNL_CC_VECTOR_BYTES" "16"
let codes = 65536
let m = codes + 3
let n = 4
let bits x = Int64.bits_of_float x
let same_bits a b = Int64.equal (bits a) (bits b)

let bf16_value c =
  if c land 0x7F80 = 0x7F80 && c land 0x7F <> 0 then Float.nan
  else Int32.float_of_bits (Int32.of_int_trunc (c lsl 16))

(* Decode the IEEE binary16 fields arithmetically, independently of the codecs under test.
   Subnormals are multiples of 2^-24; normal significands have the implicit 1024 bit. *)
let fp16_value c =
  let exponent = (c lsr 10) land 31 and fraction = c land 1023 in
  let magnitude =
    if exponent = 31 then if fraction = 0 then Float.infinity else Float.nan
    else if exponent = 0 then Stdlib.Float.ldexp (Float.of_int fraction) (-24)
    else Stdlib.Float.ldexp (Float.of_int (1024 + fraction)) (exponent - 25)
  in
  if c land 0x8000 = 0 then magnitude else -.magnitude

(* One leg's nodes, its [Tile_mma] and the serial micro-kernel it is equivalent to; [tiled] picks
   which of the two the routine runs. Each leg has its own nodes, as the two routines are compiled
   and executed separately. *)
let leg ~store_prec ~a_values ~name ~first_id ~tiled =
  let mk ~prec ~dims label =
    let tn = Ll_builders.node_factory ~prec ~first_id ~dims () (name ^ "_" ^ label) in
    Ll_builders.materialize tn;
    tn
  in
  let d = mk ~prec:Ir.Ops.single ~dims:[| m; n |] "d" in
  let a = mk ~prec:store_prec ~dims:[| m; 1 |] "a" in
  let b = mk ~prec:store_prec ~dims:[| 1; n |] "b" in
  let i = Idx.get_symbol () and j = Idx.get_symbol () and l = Idx.get_symbol () in
  let it s = Idx.Iterator s in
  let fallback =
    let serial index to_ body = LL.For_loop { index; from_ = 0; to_; axis = LL.Serial; body } in
    serial i (m - 1)
    @@ serial j (n - 1)
    @@ serial l 0
    @@ LL.Set
         {
           tn = d;
           idcs = [| it i; it j |];
           llsc =
             LL.Ternop
               ( Ir.Ops.FMA,
                 (LL.Get (a, [| it i; it l |]), Ir.Ops.single),
                 (LL.Get (b, [| it l; it j |]), Ir.Ops.single),
                 (LL.Get (d, [| it i; it j |]), Ir.Ops.single) );
           debug = "";
         }
  in
  let o =
    if tiled then
      let lane = Idx.get_symbol () in
      let origin = [| Idx.Fixed_idx 0; Idx.Fixed_idx 0 |] in
      Ll_test.optimize_scoped ~materialized:[ d; a; b ] ~name ~raw:fallback
        (LL.For_loop
           {
             index = lane;
             from_ = 0;
             to_ = 0;
             axis = LL.Workgroup;
             body =
               Ll_builders.tile_mma ~m ~n ~k:1 ~lane ~d:(d, origin) ~a:(a, origin) ~b:(b, origin)
                 fallback;
           })
    else Ll_test.optimize ~materialized:[ d; a; b ] ~name fallback
  in
  match
    Ll_test.execute ~ctx:(Context.cpu ()) ~name o
      ~seed:[ (d, Array.create ~len:(m * n) (-0.)); (a, a_values); (b, Array.create ~len:n 1.) ]
      ~read:[ d ]
  with
  | [ values ] -> values
  | _ -> assert false

let check ~tag ~store_prec ~decode ~first_id =
  let a_values = Array.init m ~f:(fun i -> decode (i % codes)) in
  let serial_name = "rowgrp_" ^ tag ^ "_serial" in
  let tile_name = "rowgrp_" ^ tag ^ "_tiled" in
  let twin = leg ~store_prec ~a_values ~name:serial_name ~first_id ~tiled:false in
  let tile = leg ~store_prec ~a_values ~name:tile_name ~first_id:(first_id + 100) ~tiled:true in
  let src = Generated.read tile_name in
  (* Without this the parity could hold vacuously: a declined tile runs the serial micro-kernel, and
     a tile without a row tail never pads a group. An absent pack/row pair fails the guard but still
     runs the value claims, including in a negative control. *)
  p
    (tag
   ^ ": the tile renders at 4 lanes, widening its A column by row groups, with a partial last group"
    )
    (match Ir.C_syntax.vec_widen_rows_macros ~store_prec ~prec:Ir.Ops.single ~lanes:4 with
    | None -> false
    | Some (pack, row) ->
        String.is_substring src ~substring:"4-lane float"
        && String.is_substring src ~substring:(pack ^ "(")
        && String.is_substring src ~substring:(row ^ "(")
        && String.is_substring src ~substring:"row tail 3");
  p_all2
    (tag ^ ": the tiled widening of every code is bitwise identical to the serial twin")
    tile twin ~f:same_bits;
  p_all2
    (tag ^ ": the tiled widening returns every code's exact value, and a NaN for a NaN code")
    tile
    (Array.init (m * n) ~f:(fun c -> a_values.(c / n)))
    ~f:(fun got want -> if Float.is_nan want then Float.is_nan got else same_bits got want)

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  Generated.init ~backend_name:"cc";
  check ~tag:"bf16" ~store_prec:Ir.Ops.bfloat16 ~decode:bf16_value ~first_id:16400;
  check ~tag:"fp16" ~store_prec:Ir.Ops.half ~decode:fp16_value ~first_id:16600
