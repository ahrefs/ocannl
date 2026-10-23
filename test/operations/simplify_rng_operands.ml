(* gh-ocannl-1498: [simplify_llc] simplifies the operands of the RNG binops -- the two threefry
   variants and the per-lane uniform conversion -- like any others; only the RNG binop itself is
   never folded. So a producer inlined beneath a threefry goes through the same rewrites as its
   materialized setter, by construction rather than fold by fold (gh-ocannl-1468 was one such fold).

   In a [uint4x32] slot the source precision of what renders there picks the conversion: an index
   goes through [int32_to_uint4x32], a [Constant] through [double_to_uint4x32], and the two draw
   different bits. So every operand the renderer reads at that slot is simplified AT the slot, not
   at its own annotation ([Low_level.mk_binop] annotates an embedded index at the index precision):
   the threefry's operands, and the child a transparent rewrite hands the wrapper's place to (an
   [Identity], a projection, a decided [Where]). A scope of another precision in that slot is not
   collapsed, since its own conversion is part of the value.

   Structural legs, on [simplify_llc] directly. The single-assignment [uint4x32] scopes an inlined
   key and counter leave beneath a threefry collapse into its operands. A fixed counter stays an
   index beneath a threefry, spelled directly, inside a collapsed scope, behind the transparent
   wrappers, and in a materialized setter behind them. A single-precision scope beneath a threefry
   stays a scope. The lane conversion's lane operand, an index-precision scalar, folds like any
   scalar. A threefry is never folded, even over constant operands.

   Executed legs: one draw computed several ways -- key and counter inlined as scopes, spelled as
   leaves directly beneath the threefry, materialized in nodes the draw reads back, and (for the
   fixed counter) behind the wrappers both inlined and materialized -- must agree bit for bit. The
   controls: iterator draws are pairwise distinct, and fixed draws differ from the draw of the
   counter as a double-converted [Constant]. *)

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

(* [x] behind three transparent rewrites, each annotated at the index precision: an [Identity] of
   the projection [Arg1] of a [Where] whose condition is the constant 1 and whose other arm is a
   different fixed index. Simplify reduces it to [x]. *)
let wrapped ~other x =
  let where =
    LL.Ternop (Ops.Where, (LL.Constant 1., iprec), (x, iprec), (LL.Embed_index other, iprec))
  in
  let arg1 = LL.Binop (Ops.Arg1, (where, iprec), (LL.Constant 0., Ops.single)) in
  LL.Unop (Ops.Identity, (arg1, iprec))

let threefries llc =
  Ll_test.count_scalar llc ~f:(function
    | LL.Binop ((Ops.Threefry4x32_light | Ops.Threefry4x32_crypto), _, _) -> true
    | _ -> false)

let scopes llc = Ll_test.count_scalar llc ~f:(function LL.Local_scope _ -> true | _ -> false)

let fixed_index k llc =
  Ll_test.count_scalar llc ~f:(function LL.Embed_index (Idx.Fixed_idx j) -> j = k | _ -> false)

let constants llc = Ll_test.count_scalar llc ~f:(function LL.Constant _ -> true | _ -> false)
let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

type forms = {
  scoped : LL.t;
  direct : LL.t;
  wrapped_direct : LL.t;  (** [Noop] for the iterator counter. *)
  wrapped_setter : LL.t;  (** [Noop] for the iterator counter. *)
  rest : LL.t;  (** The materialized forms and the control. *)
  materialized : Ir.Tnode.t list;
  outputs : Ir.Tnode.t list;
}

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
  let ctr_w = mk_u4 (name ^ "_ctr_w") in
  let out_scoped = mk (name ^ "_scoped") in
  let out_direct = mk (name ^ "_direct") in
  let out_mat = mk (name ^ "_mat") in
  let out_double = mk (name ^ "_double") in
  let out_wrapped = mk (name ^ "_wrapped") in
  let out_wrapped_mat = mk (name ^ "_wrapped_mat") in
  List.iter [ self_v; key_v; ctr_v ] ~f:Ll_test.virtualize;
  let fixed_outputs =
    if Option.is_some fixed then [ out_double; out_wrapped; out_wrapped_mat ] else []
  in
  let fixed_nodes = if Option.is_some fixed then [ ctr_w ] else [] in
  let outputs = [ out_scoped; out_direct; out_mat ] @ fixed_outputs in
  let materialized = [ key_m; ctr_m ] @ fixed_nodes @ outputs in
  List.iter materialized ~f:Ll_test.materialize;
  let key_direct () = threefry (LL.Constant_bits self_bits, u4) (LL.Constant_bits seed_bits, u4) in
  let at0 = [| Idx.Fixed_idx 0 |] in
  let loop body =
    let i = Ll_test.sym () in
    Ll_test.loop_n i n (body i)
  in
  let scoped =
    loop (fun i ->
        let self = scope self_v ~orig:at0 (LL.Constant_bits self_bits) in
        let key = scope key_v ~orig:at0 (threefry (self, u4) (LL.Constant_bits seed_bits, u4)) in
        let ctr = scope ctr_v ~orig:[| Ll_test.iter i |] (LL.Embed_index (counter i)) in
        Ll_test.set_at out_scoped (Ll_test.iter i) (uniform1 (threefry (key, u4) (ctr, u4))))
  in
  (* The index annotated at the index precision, as [Low_level.mk_binop] would. *)
  let direct =
    loop (fun i ->
        Ll_test.set_at out_direct (Ll_test.iter i)
          (uniform1 (threefry (key_direct (), u4) (LL.Embed_index (counter i), iprec))))
  in
  let read_back ctr out =
    loop (fun i ->
        Ll_test.set_at out (Ll_test.iter i)
          (uniform1 (threefry (LL.Get (key_m, at0), u4) (LL.Get (ctr, [| Ll_test.iter i |]), u4))))
  in
  let materialized_stmts =
    [
      Ll_test.set key_m at0 (key_direct ());
      loop (fun i -> Ll_test.set_at ctr_m (Ll_test.iter i) (LL.Embed_index (counter i)));
      read_back ctr_m out_mat;
    ]
  in
  let wrapped_direct, wrapped_setter, fixed_stmts =
    match fixed with
    | None -> (LL.Noop, LL.Noop, [])
    | Some k ->
        let w () = wrapped ~other:(Idx.Fixed_idx (k + 2)) (LL.Embed_index (Idx.Fixed_idx k)) in
        let wrapped_direct =
          loop (fun i ->
              Ll_test.set_at out_wrapped (Ll_test.iter i)
                (uniform1 (threefry (key_direct (), u4) (w (), iprec))))
        in
        let wrapped_setter = loop (fun i -> Ll_test.set_at ctr_w (Ll_test.iter i) (w ())) in
        (* The control: the same draw from the counter's value as a [Constant] -- what a fold would
           leave -- which renders through [double_to_uint4x32]. *)
        let double =
          loop (fun i ->
              Ll_test.set_at out_double (Ll_test.iter i)
                (uniform1 (threefry (key_direct (), u4) (LL.Constant (Float.of_int k), u4))))
        in
        (wrapped_direct, wrapped_setter, [ double; read_back ctr_w out_wrapped_mat ])
  in
  {
    scoped;
    direct;
    wrapped_direct;
    wrapped_setter;
    rest = LL.unflat_lines (materialized_stmts @ fixed_stmts);
    materialized;
    outputs;
  }

let check ~first_id ~name ~counter ~fixed =
  let f = case ~first_id ~name ~counter ~fixed in
  let s_scoped = LL.simplify_llc [] f.scoped and s_direct = LL.simplify_llc [] f.direct in
  p
    (name ^ ": the hand-built scoped form has three scopes beneath its threefries")
    (scopes f.scoped = 3);
  p (name ^ ": simplify collapses the uint4x32 scopes beneath the threefries") (scopes s_scoped = 0);
  p (name ^ ": simplify never folds a threefry") (threefries s_scoped = 2 && threefries s_direct = 2);
  Option.iter fixed ~f:(fun k ->
      let s_wrapped = LL.simplify_llc [] f.wrapped_direct in
      let s_setter = LL.simplify_llc [] f.wrapped_setter in
      p
        (name ^ ": the fixed counter stays an embedded index beneath the threefry, in both forms")
        (fixed_index k s_scoped = 1 && fixed_index k s_direct = 1);
      p (name ^ ": no constant replaces it") (constants s_scoped = 0 && constants s_direct = 0);
      p
        (name ^ ": behind the wrappers, beneath a threefry, the fixed counter stays an index")
        (fixed_index k s_wrapped = 1 && fixed_index (k + 2) s_wrapped = 0 && constants s_wrapped = 0);
      p
        (name ^ ": behind the wrappers, in a uint4x32 setter, the fixed counter stays an index")
        (fixed_index k s_setter = 1 && fixed_index (k + 2) s_setter = 0 && constants s_setter = 0));
  let program =
    LL.unflat_lines [ f.scoped; f.direct; f.wrapped_direct; f.wrapped_setter; f.rest ]
  in
  let routine = "rngop_" ^ name in
  let optimized = Ll_test.optimize ~materialized:f.materialized ~name:routine program in
  match Ll_test.execute ~name:routine optimized ~seed:[] ~read:f.outputs with
  | v_scoped :: v_direct :: v_mat :: fixed_values -> (
      Stdio.printf "%s: %d cells drawn %d ways\n" name (Array.length v_mat)
        (List.length f.outputs - if Option.is_some fixed then 1 else 0);
      p_all (name ^ ": every draw lies in [0, 1)") (Array.to_list v_mat) ~f:(fun x ->
          Float.(0. <= x && x < 1.));
      p_all2
        (name ^ ": inlined scopes and materialized nodes draw bitwise-equal values")
        v_scoped v_mat ~f:bits_equal;
      p_all2
        (name ^ ": direct leaves and materialized nodes draw bitwise-equal values")
        v_direct v_mat ~f:bits_equal;
      match fixed_values with
      | [ v_double; v_wrapped; v_wrapped_mat ] ->
          p_all2 (name ^ ": every draw differs from the double-converted counter's (the control)")
            v_mat v_double ~f:(fun a b -> not (bits_equal a b));
          p_all2
            (name ^ ": the wrapped counter beneath the threefry draws bitwise-equal values")
            v_wrapped v_mat ~f:bits_equal;
          p_all2
            (name ^ ": the wrapped counter materialized draws bitwise-equal values")
            v_wrapped_mat v_mat ~f:bits_equal
      | _ ->
          p_pairwise_distinct
            (name ^ ": the draws differ from cell to cell")
            (Array.to_list v_mat) ~equal:bits_equal ~to_string:Float.to_string)
  | _ -> assert false

(* A scope of another precision converts its value into its own precision before the [uint4x32] slot
   converts it again; collapsing it would drop that step, so it stays. *)
let check_foreign_scope ~first_id =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let out = mk "foreign_out" and inner = mk "foreign_inner" in
  Ll_test.materialize out;
  Ll_test.virtualize inner;
  let i = Ll_test.sym () in
  let ctr = scope inner ~orig:[| Ll_test.iter i |] (LL.Embed_index (Idx.Fixed_idx 3)) in
  let llc =
    Ll_test.loop_n i n
      (Ll_test.set_at out (Ll_test.iter i)
         (uniform1 (threefry (LL.Constant_bits self_bits, u4) (ctr, Ops.single))))
  in
  p "foreign scope: a single-precision scope beneath a threefry is not collapsed"
    (scopes (LL.simplify_llc [] llc) = 1)

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
  check_foreign_scope ~first_id:1498300;
  check_lane ~first_id:1498200
