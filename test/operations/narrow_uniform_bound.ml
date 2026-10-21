(* The half and bf16 uniform draws stay below 1 on the backend that runs this (gh-ocannl-1434).

   A narrow draw used to be the f32 draw of its bits ROUNDED to the narrow format, which is exactly
   1.0 for every f32 draw within half a narrow ulp of 1 -- one half draw in 4096 and one bf16 draw
   in 512 -- breaking every consumer that relies on [u < 1] (a keep mask [u < p] at [p = 1], [log (1
   - u)]). Each is now the f32 draw truncated toward zero, so the largest draw is the format's
   greatest value below 1.

   Sampling would meet the boundary only by luck, so the conversions are fed the worst-case bits
   directly: the key [Constant_bits (-1)], whose [uint4x32] widening is two all-ones words and two
   zero words. Every form a kernel can render is executed -- the scalar conversion, the packed
   [_vec] store and the per-cell [_lane] extraction the virtualizer mints -- and each must draw,
   from an all-ones word, exactly the format's greatest value below 1, and from a zero word exactly
   0. The values are exact by construction (a bit selection and one exact narrowing), so they are
   golden lines; the f32 row is the unchanged reference. The host conversions over [builtins.c]
   ([Ops.uint4x32_to_*_uniform]) are held to the same draw. *)

open Base
open Ll_test
open Verdict.Claims

let backend_name = Context.backend_name (Lazy.force base_ctx)
let () = Stdio.eprintf "(not part of the golden) narrow_uniform_bound: backend=%s\n%!" backend_name

(* The all-ones key: [int64_to_uint4x32 (-1)] is [0xFFFFFFFF; 0xFFFFFFFF; 0; 0]. *)
let key = LL.Constant_bits (-1L)
let key_lanes = Ops.int64_to_uint4x32 (-1L)

(* The packed forms' lanes: 8 at a 16-bit format, lane [j] drawing from 16 bits of word [j / 2]; 4
   at f32, one word each. Either way the first half come from the all-ones words and the second half
   from the zero words. *)
let lanes prec = Ops.vec_unop_lanes prec

(* The greatest value below 1, read off the host codec: 1.0's encoding, one step down. *)
let below_one (prec : Ops.prec) =
  match prec with
  | Ops.Bfloat16_prec _ -> Ops.bfloat16_to_single (Ops.single_to_bfloat16 1.0 - 1)
  | Ops.Half_prec _ -> Ops.half_to_single (Ops.single_to_half 1.0 - 1)
  | _ -> 1.0 -. 0x1p-24

let host_scalar_draw (prec : Ops.prec) =
  match prec with
  | Ops.Bfloat16_prec _ -> Ops.bfloat16_to_single (Ops.uint4x32_to_bfloat16_uniform key_lanes)
  | Ops.Half_prec _ -> Ops.half_to_single (Ops.uint4x32_to_half_uniform key_lanes)
  | _ -> Ops.uint4x32_to_single_uniform key_lanes

let next_id = ref 3400

let node prec label n =
  Int.incr next_id;
  let tn =
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy [| n |])
      ~padding:(lazy None)
      ()
  in
  materialize tn;
  tn

let run ~name llc tn n =
  let o = optimize ~name llc in
  match execute ~name o ~seed:[ (tn, blank n) ] ~read:[ tn ] with
  | [ values ] -> values
  | _ -> assert false

let scalar prec prec_name =
  let tn = node prec "nub_scalar" 1 in
  let llc = set_at tn (fixed 0) (LL.Unop (Ops.Uint4x32_to_prec_uniform1, (key, Ops.uint4x32))) in
  run ~name:("nub_scalar_" ^ prec_name) llc tn 1

let packed prec prec_name =
  let lanes = lanes prec in
  let tn = node prec "nub_vec" lanes in
  let llc =
    set_from_vec ~length:lanes ~vec_unop:Ops.Uint4x32_to_prec_uniform tn
      [| fixed 0 |]
      (key, Ops.uint4x32)
  in
  run ~name:("nub_vec_" ^ prec_name) llc tn lanes

let per_lane prec prec_name =
  let lanes = lanes prec in
  let tn = node prec "nub_lane" lanes in
  let i = sym () in
  let llc =
    loop_n i lanes
      (set_at tn (iter i)
         (LL.Binop
            ( Ops.Uint4x32_to_prec_uniform_lane,
              (key, Ops.uint4x32),
              (LL.Embed_index (iter i), Ops.index_prec ()) )))
  in
  run ~name:("nub_lane_" ^ prec_name) llc tn lanes

let () =
  List.iter
    [ ("f32", Ops.single); ("bf16", Ops.bfloat16); ("f16", Ops.half) ]
    ~f:(fun (prec_name, prec) ->
      let top = below_one prec in
      let s = scalar prec prec_name and v = packed prec prec_name and l = per_lane prec prec_name in
      Stdio.printf "%s scalar draw of the all-ones key: %s\n" prec_name (Test_utils.hex_float s.(0));
      Stdio.printf "%s packed draws of the all-ones key: %s\n" prec_name
        (String.concat_array ~sep:" " (Array.map v ~f:Test_utils.hex_float));
      p
        (prec_name ^ ": the scalar draw from an all-ones word is the greatest value below 1")
        Float.(s.(0) = top && s.(0) < 1.0);
      p
        (prec_name ^ ": the host conversion draws what the backend draws from the all-ones key")
        Float.(host_scalar_draw prec = s.(0));
      let lanes = lanes prec in
      let want = Array.init lanes ~f:(fun j -> if j < lanes / 2 then top else 0.) in
      p_all2
        (prec_name
       ^ ": the packed draws are the greatest value below 1 from all-ones bits and 0 from zero bits"
        )
        v want ~f:Float.equal;
      p_all2
        (prec_name ^ ": each per-lane extraction draws what the packed store draws in that lane")
        l v ~f:Float.equal;
      p_all
        (prec_name ^ ": no form draws 1 or more")
        (Array.to_list (Array.concat [ s; v; l ]))
        ~f:(fun x -> Float.(x < 1.0)))
