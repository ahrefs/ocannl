(* The register tile's bf16 A column, widened a group of rows at a time.

   At 4 lanes of f32 the tile's A rows cross the bf16 -> f32 boundary through
   [Ir.C_syntax.vec_widen_rows_macros]'s pair of builtins: on aarch64 one vector widening per group
   of four rows, read back one lane per row; elsewhere each row's own scalar conversion, the per-row
   rendering after preprocessing. (Why aarch64 differs: [cc_march_census]'s "no known defects"
   note.) Either arm must widen every code exactly as the scalar codec does, and the census only
   compiles the aarch64 arm. This test EXECUTES whichever arm the host compiler takes -- the packed
   one on an arm64 host -- on every bf16 code, against the tile's own serial micro-kernel.

   A is [m x 1] bf16 holding every code once (and three more rows, so the last group is a padded
   partial one), B is [1 x 4] of 1.0, and D is f32 seeded with -0.0: each cell is fma(a, 1, -0),
   which is [a] exactly for every code, a signed zero included, and a NaN for a NaN code. *)

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

let a_values = Array.init m ~f:(fun i -> bf16_value (i % codes))

(* One leg's nodes, its [Tile_mma] and the serial micro-kernel it is equivalent to; [tiled] picks
   which of the two the routine runs. Each leg has its own nodes, as the two routines are compiled
   and executed separately. *)
let leg ~name ~first_id ~tiled =
  let mk ~prec ~dims label =
    let tn = Ll_builders.node_factory ~prec ~first_id ~dims () (name ^ "_" ^ label) in
    Ll_builders.materialize tn;
    tn
  in
  let d = mk ~prec:Ir.Ops.single ~dims:[| m; n |] "d" in
  let a = mk ~prec:Ir.Ops.bfloat16 ~dims:[| m; 1 |] "a" in
  let b = mk ~prec:Ir.Ops.bfloat16 ~dims:[| 1; n |] "b" in
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

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  Generated.init ~backend_name:"cc";
  let twin = leg ~name:"rowgrp_serial" ~first_id:16400 ~tiled:false in
  let tile = leg ~name:"rowgrp_tiled" ~first_id:16500 ~tiled:true in
  let src = Generated.read "rowgrp_tiled" in
  let pack, row =
    Option.value_exn
      (Ir.C_syntax.vec_widen_rows_macros ~store_prec:Ir.Ops.bfloat16 ~prec:Ir.Ops.single ~lanes:4)
  in
  (* Without these the parity below could hold vacuously: a declined tile runs the serial
     micro-kernel, and a tile without a row tail never pads a group. *)
  p "the tile renders at 4 lanes, widening its A column by row groups, with a partial last group"
    (String.is_substring src ~substring:"4-lane float"
    && String.is_substring src ~substring:(pack ^ "(")
    && String.is_substring src ~substring:(row ^ "(")
    && String.is_substring src ~substring:"row tail 3");
  p_all2 "the tiled widening of every bf16 code is bitwise identical to the serial twin" tile twin
    ~f:same_bits;
  p_all2 "the tiled widening returns every code's exact value, and a NaN for a NaN code" tile
    (Array.init (m * n) ~f:(fun c -> a_values.(c / n)))
    ~f:(fun got want -> if Float.is_nan want then Float.is_nan got else same_bits got want)
