(* gh-ocannl-1498: [simplify_llc] simplifies the operands of the RNG binops -- the two threefry
   variants and the per-lane uniform conversion -- like any others; only the RNG binop itself is
   never folded. So a producer inlined beneath a threefry goes through the same rewrites as its
   materialized setter, by construction rather than fold by fold (gh-ocannl-1468 was one such fold).

   Structural legs, on [simplify_llc] directly: - the single-assignment [uint4x32] scopes an inlined
   key and counter leave beneath a threefry collapse into its operands; - a fixed embedded index
   beneath a threefry stays an index even when annotated at the index precision (what
   [Low_level.mk_binop] annotates it with): a threefry renders its operands at its own [uint4x32]
   precision, so the gh-ocannl-1468 gate is keyed on that slot, and a fold to a [Constant] would
   re-key the conversion from [int32_to_uint4x32] to [double_to_uint4x32]; - the lane conversion's
   lane operand, an index-precision scalar, folds like any scalar; - the threefry itself is never
   folded, even over constant operands.

   Executed legs: one draw computed three ways -- key and counter inlined as scopes, spelled as
   leaves directly beneath the threefry, and materialized in nodes the draw reads back -- must agree
   bit for bit, for an iterator counter and for a fixed one. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let u4 = Ops.uint4x32
let iprec = Ops.index_prec ()
let n = 6
let self_bits = 7L
let seed_bits = 42L
let threefry a b = LL.Binop (Ops.Threefry4x32_light, a, b)
let uniform1 bits = LL.Unop (Ops.Uint4x32_to_prec_uniform1, (bits, u4))

(* The scope an inliner leaves for a virtual node: a fresh id, one [Set_local]. *)
let scope tn ~orig v =
  let id = LL.get_scope tn in
  LL.Local_scope
    { id; body = LL.Set_local (id, v); orig_indices = orig; mint = LL.Inlined_computation }

let threefries llc =
  Ll_test.count_scalar llc ~f:(function
    | LL.Binop ((Ops.Threefry4x32_light | Ops.Threefry4x32_crypto), _, _) -> true
    | _ -> false)

let scopes llc = Ll_test.count_scalar llc ~f:(function LL.Local_scope _ -> true | _ -> false)

let fixed_index k llc =
  Ll_test.count_scalar llc ~f:(function LL.Embed_index (Idx.Fixed_idx j) -> j = k | _ -> false)

let constants llc = Ll_test.count_scalar llc ~f:(function LL.Constant _ -> true | _ -> false)
let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* [counter i] is the counter's index at cell [i]: the iterator itself, or the fixed index
   [Fixed_idx k] for [fixed = Some k]. *)
let case ~first_id ~name ~(counter : Idx.symbol -> Idx.axis_index) ~fixed =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let mk_u4 = Ll_test.node_factory ~prec:u4 ~first_id:(first_id + 50) ~dims:[| n |] () in
  let self_v = mk_u4 ~dims:[| 1 |] (name ^ "_self") in
  let key_v = mk_u4 ~dims:[| 1 |] (name ^ "_key") in
  let ctr_v = mk_u4 (name ^ "_ctr") in
  let key_m = mk_u4 ~dims:[| 1 |] (name ^ "_key_m") in
  let ctr_m = mk_u4 (name ^ "_ctr_m") in
  let out_scoped = mk (name ^ "_scoped") in
  let out_direct = mk (name ^ "_direct") in
  let out_mat = mk (name ^ "_mat") in
  let out_double = mk (name ^ "_double") in
  List.iter [ self_v; key_v; ctr_v ] ~f:Ll_test.virtualize;
  let controls = if Option.is_some fixed then [ out_double ] else [] in
  let materialized = [ key_m; ctr_m; out_scoped; out_direct; out_mat ] @ controls in
  List.iter materialized ~f:Ll_test.materialize;
  let key_direct () = threefry (LL.Constant_bits self_bits, u4) (LL.Constant_bits seed_bits, u4) in
  let at0 = [| Idx.Fixed_idx 0 |] in
  let scoped_stmt i =
    let self = scope self_v ~orig:at0 (LL.Constant_bits self_bits) in
    let key = scope key_v ~orig:at0 (threefry (self, u4) (LL.Constant_bits seed_bits, u4)) in
    let ctr = scope ctr_v ~orig:[| Ll_test.iter i |] (LL.Embed_index (counter i)) in
    Ll_test.set_at out_scoped (Ll_test.iter i) (uniform1 (threefry (key, u4) (ctr, u4)))
  in
  let direct_stmt i =
    (* The index annotated at the index precision, as [Low_level.mk_binop] would. *)
    Ll_test.set_at out_direct (Ll_test.iter i)
      (uniform1 (threefry (key_direct (), u4) (LL.Embed_index (counter i), iprec)))
  in
  let loop body =
    let i = Ll_test.sym () in
    Ll_test.loop_n i n (body i)
  in
  let scoped = loop scoped_stmt and direct = loop direct_stmt in
  let materialized_stmts =
    [
      Ll_test.set key_m at0 (key_direct ());
      loop (fun i -> Ll_test.set_at ctr_m (Ll_test.iter i) (LL.Embed_index (counter i)));
      loop (fun i ->
          Ll_test.set_at out_mat (Ll_test.iter i)
            (uniform1
               (threefry (LL.Get (key_m, at0), u4) (LL.Get (ctr_m, [| Ll_test.iter i |]), u4))));
    ]
  in
  (* The control: the same draw from the counter's value as a [Constant] -- what the fold would
     leave -- which renders through [double_to_uint4x32]. Only a fixed counter has one. *)
  let double_stmt =
    match fixed with
    | Some k ->
        loop (fun i ->
            Ll_test.set_at out_double (Ll_test.iter i)
              (uniform1 (threefry (key_direct (), u4) (LL.Constant (Float.of_int k), u4))))
    | None -> LL.Noop
  in
  ( scoped,
    direct,
    LL.unflat_lines (materialized_stmts @ [ double_stmt ]),
    materialized,
    [ out_scoped; out_direct; out_mat ] @ controls )

let check ~first_id ~name ~counter ~fixed =
  let scoped, direct, mat, materialized, outputs = case ~first_id ~name ~counter ~fixed in
  let s_scoped = LL.simplify_llc [] scoped and s_direct = LL.simplify_llc [] direct in
  p
    (name ^ ": the hand-built scoped form has three scopes beneath its threefries")
    (scopes scoped = 3);
  p (name ^ ": simplify collapses the uint4x32 scopes beneath the threefries") (scopes s_scoped = 0);
  p (name ^ ": simplify never folds a threefry") (threefries s_scoped = 2 && threefries s_direct = 2);
  Option.iter fixed ~f:(fun k ->
      p
        (name ^ ": the fixed counter stays an embedded index beneath the threefry, in both forms")
        (fixed_index k s_scoped = 1 && fixed_index k s_direct = 1);
      p (name ^ ": no constant replaces it") (constants s_scoped = 0 && constants s_direct = 0));
  let program = LL.unflat_lines [ scoped; direct; mat ] in
  let routine = "rngop_" ^ name in
  let optimized = Ll_test.optimize ~materialized ~name:routine program in
  match Ll_test.execute ~name:routine optimized ~seed:[] ~read:outputs with
  | v_scoped :: v_direct :: v_mat :: controls ->
      Stdio.printf "%s: %d cells drawn three ways\n" name (Array.length v_mat);
      p_all (name ^ ": every draw lies in [0, 1)") (Array.to_list v_mat) ~f:(fun x ->
          Float.(0. <= x && x < 1.));
      (match controls with
      | [ v_double ] ->
          p_all2 (name ^ ": every draw differs from the double-converted counter's (the control)")
            v_mat v_double ~f:(fun a b -> not (bits_equal a b))
      | _ ->
          p_pairwise_distinct
            (name ^ ": the draws differ from cell to cell")
            (Array.to_list v_mat) ~equal:bits_equal ~to_string:Float.to_string);
      p_all2
        (name ^ ": inlined scopes and materialized nodes draw bitwise-equal values")
        v_scoped v_mat ~f:bits_equal;
      p_all2
        (name ^ ": direct leaves and materialized nodes draw bitwise-equal values")
        v_direct v_mat ~f:bits_equal
  | _ -> assert false

(* The lane conversion: the block operand is the threefry's, the lane operand an index-precision
   scalar the renderer reads at its own precision, so it folds like one. *)
let check_lane ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let out = mk "lane_out" in
  Ll_test.materialize out;
  let i = Ll_test.sym () in
  let bits = threefry (LL.Constant_bits self_bits, u4) (LL.Embed_index (Ll_test.iter i), u4) in
  let lane =
    LL.Binop (Ops.Add, (LL.Embed_index (Idx.Fixed_idx 1), iprec), (LL.Constant 1., iprec))
  in
  let llc =
    Ll_test.loop_n i n
      (Ll_test.set_at out (Ll_test.iter i)
         (LL.Binop (Ops.Uint4x32_to_prec_uniform_lane, (bits, u4), (lane, iprec))))
  in
  let simplified = LL.simplify_llc [] llc in
  p "lane: the lane operand folds to the constant 2"
    (Ll_test.count_scalar simplified ~f:(function
       | LL.Binop (Ops.Uint4x32_to_prec_uniform_lane, _, (LL.Constant c, _)) -> Float.equal c 2.
       | _ -> false)
    = 1);
  p "lane: the counter beneath the threefry stays the iterator"
    (Ll_test.count_scalar simplified ~f:(function
       | LL.Embed_index (Idx.Iterator s) -> Idx.equal_symbol s i
       | _ -> false)
    = 1)

let () =
  check ~first_id:1498000 ~name:"iter" ~counter:Ll_test.iter ~fixed:None;
  check ~first_id:1498100 ~name:"fixed" ~counter:(fun _ -> Idx.Fixed_idx 3) ~fixed:(Some 3);
  check_lane ~first_id:1498200
