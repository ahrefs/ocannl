(* cc-fp-contract: no-contraction -- Threefry uniform draws and plain addition, no product feeding
   an add *)

(* gh-ocannl-2412: static twin of [debug_check_lane_bounds]. At the shared [Raise_backend.compile]
   site before rendering, [Low_level.validate_lane_bounds] refuses any
   [Uint4x32_to_prec_uniform_lane] whose lane operand has the form [Mod (_, K)] with fixed integer
   [K > Ops.vec_unop_lanes prec] at the precision the call renders at, on every backend and with no
   config key.

   This test runs on the backend selected by the environment ([Context.auto]): - Across every scalar
   precision in [Ops.scalar_precs], [Mod (i, Fixed_idx (lanes + 1))] is refused before rendering
   with a message naming the node, [K], the rendered precision and its lane count (including
   [double] on Metal, since the check precedes backend rendering). - At the boundary [Mod (i,
   Fixed_idx lanes)], every precision compiles and executes cleanly (with [double] gated by
   [supports_f64]). - Unsimplified [Embed_index (Fixed_idx K)] (bypassing [simplify_llc] via
   [Ll_test.optimize_scoped]), [Local_scope] boundaries (valid narrow scope inside wider store vs
   invalid narrow-lane scope inside wider-lane store), homogeneous arithmetic around an RNG lane
   extract at narrow storage, dynamic buffer-read lanes (not refused statically), discarded [Arg1]
   projection operands, multi-segment fissioned batches, and backend [compute_prec] in [Where]/[If]
   conditions. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Tn = Ir.Tnode
open Verdict.Claims

let u4 = Ops.uint4x32
let iprec = Ops.index_prec ()
let base_ctx = Lazy.force Ll_test.base_ctx
let backend_name = Context.backend_name base_ctx
let supports_f64 = (Context.codegen_capabilities base_ctx).Ir.Backend_intf.supports_f64
let f64_leg = Verdict.leg ~when_:supports_f64 ~on:backend_name ()
let () = Stdio.eprintf "lane_bounds_static: backend %s (not part of the golden)\n%!" backend_name

let bits =
  LL.Binop
    (Ops.Threefry4x32_light, (LL.Constant_bits 7L, u4), (LL.Embed_index (Idx.Fixed_idx 3), u4))

let mod_lane i k =
  LL.Binop
    (Ops.Mod, (LL.Embed_index (Ll_test.iter i), iprec), (LL.Embed_index (Idx.Fixed_idx k), iprec))

let lane_extract lane = LL.Binop (Ops.Uint4x32_to_prec_uniform_lane, (bits, u4), (lane, iprec))

let compile_outcome ?lowered_transform ~name (o : LL.optimized) =
  match
    Context.compile ~name ?lowered_transform ~prelowered:o base_ctx Ir.Assignments.empty_comp
      Idx.Empty
  with
  | _ -> `Accepted
  | exception Invalid_argument msg -> `Refused msg
  | exception exn -> `Other (Exn.to_string exn)

let check_refused label ~needles outcome =
  let refused_msg =
    match outcome with
    | `Refused msg -> Some msg
    | `Accepted ->
        Stdio.eprintf "%s: unexpectedly accepted (not part of the golden)\n%!" label;
        None
    | `Other msg ->
        Stdio.eprintf "%s: raised unexpected %s (not part of the golden)\n%!" label msg;
        None
  in
  p_all ~min:1 label needles ~f:(fun needle ->
      match refused_msg with
      | Some msg ->
          let ok = String.is_substring msg ~substring:needle in
          if not ok then
            Stdio.eprintf "%s: refused with %S, missing %S (not part of the golden)\n%!" label msg
              needle;
          ok
      | None -> false)

let next_id = ref 2412000

let mk_node prec dims label =
  let id = !next_id in
  next_id := id + 10;
  let tn = Ll_test.node_factory ~prec ~first_id:id ~dims () label in
  Ll_test.materialize tn;
  tn

let mk_virtual prec dims label =
  let id = !next_id in
  next_id := id + 10;
  let tn = Ll_test.node_factory ~prec ~first_id:id ~dims () label in
  Ll_test.virtualize tn;
  tn

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* 1. Every scalar precision in [Ops.scalar_precs]: K = lanes + 1 refused, K = lanes accepted. *)
let () =
  List.iter Ops.scalar_precs ~f:(fun prec ->
      let ps = Ops.prec_string prec in
      let lanes = Ops.vec_unop_lanes prec in
      let bad_k = lanes + 1 in
      let out_bad = mk_node prec [| bad_k |] ("lbs_bad_" ^ ps) in
      let i_bad = Ll_test.sym () in
      let llc_bad =
        Ll_test.loop_n i_bad bad_k
          (Ll_test.set_at out_bad (Ll_test.iter i_bad) (lane_extract (mod_lane i_bad bad_k)))
      in
      let opt_bad = Ll_test.optimize ~materialized:[ out_bad ] ~name:("lbs_bad_" ^ ps) llc_bad in
      check_refused
        (Printf.sprintf "%s (%d lanes): fixed modulus %d is refused statically" ps lanes bad_k)
        ~needles:
          [
            "Raise_backend.compile";
            Tn.label_name out_bad;
            Printf.sprintf "renders at %s (%d lanes)" ps lanes;
            Printf.sprintf "fixed modulus %d > %d" bad_k lanes;
          ]
        (compile_outcome ~name:("lbs_bad_" ^ ps) opt_bad);
      let run_good () =
        let out_good = mk_node prec [| lanes |] ("lbs_good_" ^ ps) in
        let ref_out = mk_node prec [| lanes |] ("lbs_ref_" ^ ps) in
        let i_good = Ll_test.sym () in
        let llc_good =
          LL.unflat_lines
            [
              Ll_test.set_from_vec ~length:lanes ~vec_unop:Ops.Uint4x32_to_prec_uniform ref_out
                [| Idx.Fixed_idx 0 |] (bits, u4);
              Ll_test.loop_n i_good lanes
                (Ll_test.set_at out_good (Ll_test.iter i_good)
                   (lane_extract (mod_lane i_good lanes)));
            ]
        in
        let name = "lbs_good_" ^ ps in
        let opt_good = Ll_test.optimize ~materialized:[ out_good; ref_out ] ~name llc_good in
        (* Distinct seeds on the two buffers so any skipped store shows up as a mismatch. *)
        let seed =
          [ (out_good, Array.create ~len:lanes 0.); (ref_out, Array.create ~len:lanes 1.) ]
        in
        match Ll_test.execute ~name opt_good ~seed ~read:[ out_good; ref_out ] with
        | [ values; ref_values ] ->
            Array.length values = lanes
            && Array.equal bits_equal values ref_values
            && List.length (List.dedup_and_sort (Array.to_list values) ~compare:Float.compare) > 1
        | _ -> false
      in
      let good_label =
        Printf.sprintf "%s (%d lanes): in-range modulus %d compiles and executes" ps lanes lanes
      in
      if Ops.equal_prec prec Ops.double then
        in_leg ~leg:f64_leg good_label (fun label -> p label (run_good ()))
      else p good_label (run_good ()))

(* 2. Unsimplified [Embed_index (Fixed_idx K)] bypassing [simplify_llc]. *)
let () =
  let out = mk_node Ops.single [| 8 |] "lbs_unsimplified" in
  let i = Ll_test.sym () in
  let llc k =
    Ll_test.loop_n i 8 (Ll_test.set_at out (Ll_test.iter i) (lane_extract (mod_lane i k)))
  in
  let raw =
    Ll_test.optimize_scoped ~materialized:[ out ] ~name:"lbs_unsimplified" ~raw:(llc 4) (llc 8)
  in
  check_refused "unsimplified Mod (_, Fixed_idx 8) at single is refused statically"
    ~needles:
      [
        "Raise_backend.compile";
        Tn.label_name out;
        "renders at single (4 lanes)";
        "fixed modulus 8 > 4";
      ]
    (compile_outcome ~name:"lbs_unsimplified" raw)

(* 3. Scope boundary: the rendered precision is the scope's, not the outer store's. *)
let () =
  (* Valid: half draw (8 lanes) inside a Local_scope stored to a single node (4 lanes). *)
  let out = mk_node Ops.single [| 8 |] "lbs_scope_ok_out" in
  let u = mk_virtual Ops.half [| 8 |] "lbs_scope_ok_u" in
  let i = Ll_test.sym () in
  let scoped = Ll_test.scope ~orig_indices:[| Ll_test.iter i |] u (lane_extract (mod_lane i 8)) in
  let llc = Ll_test.loop_n i 8 (Ll_test.set_at out (Ll_test.iter i) scoped) in
  let opt = Ll_test.optimize ~materialized:[ out ] ~name:"lbs_scope_ok" llc in
  let values =
    match Ll_test.execute ~name:"lbs_scope_ok" opt ~seed:[] ~read:[ out ] with
    | [ vs ] -> vs
    | _ -> assert false
  in
  p "half Local_scope with modulus 8 inside single store compiles and executes"
    (Array.length values = 8
    && Array.for_all values ~f:(fun x -> Float.(0. <= x && x < 1.))
    && List.length (List.dedup_and_sort (Array.to_list values) ~compare:Float.compare) > 1);
  (* Invalid: single Local_scope (4 lanes) with modulus 8 inside a half store (8 lanes). *)
  let out_bad = mk_node Ops.half [| 8 |] "lbs_scope_bad_out" in
  let u_bad = mk_virtual Ops.single [| 8 |] "lbs_scope_bad_u" in
  let i_bad = Ll_test.sym () in
  let scoped_bad =
    Ll_test.scope ~orig_indices:[| Ll_test.iter i_bad |] u_bad (lane_extract (mod_lane i_bad 8))
  in
  let llc_bad = Ll_test.loop_n i_bad 8 (Ll_test.set_at out_bad (Ll_test.iter i_bad) scoped_bad) in
  let opt_bad = Ll_test.optimize ~materialized:[ out_bad ] ~name:"lbs_scope_bad" llc_bad in
  check_refused "single Local_scope with modulus 8 inside half store is refused statically"
    ~needles:
      [
        "Raise_backend.compile";
        Tn.label_name u_bad;
        Tn.label_name out_bad;
        "renders at single (4 lanes)";
        "fixed modulus 8 > 4";
      ]
    (compile_outcome ~name:"lbs_scope_bad" opt_bad)

(* 4. Homogeneous arithmetic over an RNG lane extract at narrow storage (pinned to storage prec). *)
let () =
  let out_ok = mk_node Ops.half [| 8 |] "lbs_arith_half_ok" in
  let i_ok = Ll_test.sym () in
  let llc_ok =
    Ll_test.loop_n i_ok 8
      (Ll_test.set_at out_ok (Ll_test.iter i_ok)
         (LL.Binop (Ops.Add, (lane_extract (mod_lane i_ok 8), Ops.half), (LL.Constant 1., Ops.half))))
  in
  let opt_ok = Ll_test.optimize ~materialized:[ out_ok ] ~name:"lbs_arith_half_ok" llc_ok in
  let values =
    match Ll_test.execute ~name:"lbs_arith_half_ok" opt_ok ~seed:[] ~read:[ out_ok ] with
    | [ vs ] -> vs
    | _ -> assert false
  in
  p "half store arithmetic with modulus 8 renders at half (8 lanes) and executes"
    (Array.length values = 8 && Array.for_all values ~f:(fun x -> Float.(1. <= x && x < 2.)));
  let out_bad = mk_node Ops.half [| 9 |] "lbs_arith_half_bad" in
  let i_bad = Ll_test.sym () in
  let llc_bad =
    Ll_test.loop_n i_bad 9
      (Ll_test.set_at out_bad (Ll_test.iter i_bad)
         (LL.Binop (Ops.Add, (lane_extract (mod_lane i_bad 9), Ops.half), (LL.Constant 1., Ops.half))))
  in
  let opt_bad = Ll_test.optimize ~materialized:[ out_bad ] ~name:"lbs_arith_half_bad" llc_bad in
  check_refused "half store arithmetic with modulus 9 is refused statically"
    ~needles:
      [
        "Raise_backend.compile";
        Tn.label_name out_bad;
        "renders at half (8 lanes)";
        "fixed modulus 9 > 8";
      ]
    (compile_outcome ~name:"lbs_arith_half_bad" opt_bad)

(* 5. Dynamic lane operand (materialized index buffer) is NOT refused statically. *)
let () =
  let out = mk_node Ops.single [| 4 |] "lbs_dyn_out" in
  let idx_buf = mk_node iprec [| 4 |] "lbs_dyn_idx" in
  let i = Ll_test.sym () in
  let llc =
    Ll_test.loop_n i 4
      (Ll_test.set_at out (Ll_test.iter i) (lane_extract (LL.Get (idx_buf, [| Ll_test.iter i |]))))
  in
  let opt = Ll_test.optimize ~materialized:[ out; idx_buf ] ~name:"lbs_dyn" llc in
  let values =
    match
      Ll_test.execute ~name:"lbs_dyn" opt ~seed:[ (idx_buf, [| 0.; 1.; 2.; 3. |]) ] ~read:[ out ]
    with
    | [ vs ] -> vs
    | _ -> assert false
  in
  p "dynamic buffer-read lane operand is accepted statically and executes"
    (Array.length values = 4 && Array.for_all values ~f:(fun x -> Float.(0. <= x && x < 1.)))

(* 6. Discarded projection operand ([Arg1]) is not rendered and not refused. *)
let () =
  let out = mk_node Ops.single [| 4 |] "lbs_proj_out" in
  let i = Ll_test.sym () in
  let safe =
    Ll_test.loop_n i 4 (Ll_test.set_at out (Ll_test.iter i) (lane_extract (mod_lane i 4)))
  in
  let proj =
    Ll_test.loop_n i 4
      (Ll_test.set_at out (Ll_test.iter i)
         (LL.Binop
            ( Ops.Arg1,
              (lane_extract (mod_lane i 4), Ops.single),
              (lane_extract (mod_lane i 8), Ops.single) )))
  in
  let raw = Ll_test.optimize_scoped ~materialized:[ out ] ~name:"lbs_proj" ~raw:safe proj in
  let values =
    match Ll_test.execute ~name:"lbs_proj" raw ~seed:[] ~read:[ out ] with
    | [ vs ] -> vs
    | _ -> assert false
  in
  p "discarded Arg1 operand with modulus 8 is not refused and executes"
    (Array.length values = 4 && Array.for_all values ~f:(fun x -> Float.(0. <= x && x < 1.)))

(* 7. Fissioned multi-segment batch: a violation in a later segment is still refused. *)
let () =
  let out_ok = mk_node Ops.single [| 4 |] "lbs_fission_ok" in
  let out_bad = mk_node Ops.single [| 8 |] "lbs_fission_bad" in
  let i1 = Ll_test.sym () and i2 = Ll_test.sym () in
  let seg_ok =
    Ll_test.optimize ~materialized:[ out_ok ] ~name:"lbs_fission_seg0"
      (Ll_test.loop_n i1 4 (Ll_test.set_at out_ok (Ll_test.iter i1) (lane_extract (mod_lane i1 4))))
  in
  let seg_bad =
    Ll_test.optimize ~materialized:[ out_bad ] ~name:"lbs_fission_seg1"
      (Ll_test.loop_n i2 8
         (Ll_test.set_at out_bad (Ll_test.iter i2) (lane_extract (mod_lane i2 8))))
  in
  check_refused "fissioned batch with modulus 8 in second segment is refused statically"
    ~needles:
      [
        "Raise_backend.compile";
        Tn.label_name out_bad;
        "renders at single (4 lanes)";
        "fixed modulus 8 > 4";
      ]
    (compile_outcome ~name:"lbs_fission" ~lowered_transform:(fun _ -> [ seg_ok; seg_bad ]) seg_ok)

(* 8. Pure predicate: [comp_prec] governs heterogeneous [Where] / [If] condition slots. *)
let () =
  let out = mk_node Ops.single [| 8 |] "lbs_cond_out" in
  let i = Ll_test.sym () in
  let cond = (lane_extract (mod_lane i 8), Ops.half) in
  let llc_where =
    Ll_test.loop_n i 8
      (Ll_test.set_at out (Ll_test.iter i)
         (LL.Ternop (Ops.Where, cond, (LL.Constant 1., Ops.single), (LL.Constant 2., Ops.single))))
  in
  let llc_if =
    Ll_test.loop_n i 8
      (Ll_test.if_at Ops.half (fst cond) (Ll_test.set_at out (Ll_test.iter i) (LL.Constant 1.)))
  in
  let widen_half = function Ops.Half_prec _ -> Ops.single | p -> p in
  p "Where condition annotated half with modulus 8: accepted when comp_prec keeps half"
    (Option.is_none (LL.lane_bounds_violation ~comp_prec:Fn.id llc_where));
  p "Where condition annotated half with modulus 8: refused when comp_prec widens half to single"
    (Option.exists (LL.lane_bounds_violation ~comp_prec:widen_half llc_where) ~f:(fun msg ->
         String.is_substring msg ~substring:(Tn.label_name out)
         && String.is_substring msg ~substring:"renders at single (4 lanes)"
         && String.is_substring msg ~substring:"fixed modulus 8 > 4"));
  p "If condition annotated half with modulus 8: accepted when comp_prec keeps half"
    (Option.is_none (LL.lane_bounds_violation ~comp_prec:Fn.id llc_if));
  p "If condition annotated half with modulus 8: refused when comp_prec widens half to single"
    (Option.exists (LL.lane_bounds_violation ~comp_prec:widen_half llc_if) ~f:(fun msg ->
         String.is_substring msg ~substring:"a guard condition"
         && String.is_substring msg ~substring:"renders at single (4 lanes)"
         && String.is_substring msg ~substring:"fixed modulus 8 > 4"))

(* 9. Modulus conversion at the lane operand's evaluation precision (unsigned wrap). *)
let () =
  let out = mk_node Ops.single [| 4 |] "lbs_byte_mod_out" in
  let i = Ll_test.sym () in
  let byte_mod_lane m =
    LL.Binop
      ( Ops.Uint4x32_to_prec_uniform_lane,
        (bits, u4),
        (LL.Binop (Ops.Mod, (LL.Embed_index (Ll_test.iter i), Ops.byte), (m, Ops.byte)), Ops.byte)
      )
  in
  let llc_of m = Ll_test.loop_n i 4 (Ll_test.set_at out (Ll_test.iter i) (byte_mod_lane m)) in
  p "byte lane modulus Constant_bits 260 wraps to 4 and is accepted at single"
    (Option.is_none (LL.lane_bounds_violation (llc_of (LL.Constant_bits 260L))));
  p "byte lane modulus Constant_bits (-1) wraps to 255 and is refused at single"
    (Option.exists
       (LL.lane_bounds_violation (llc_of (LL.Constant_bits (-1L))))
       ~f:(fun msg ->
         String.is_substring msg ~substring:(Tn.label_name out)
         && String.is_substring msg ~substring:"renders at single (4 lanes)"
         && String.is_substring msg ~substring:"fixed modulus 255 > 4"));
  p "byte lane modulus Fixed_idx 260 wraps to 4 and is accepted at single"
    (Option.is_none (LL.lane_bounds_violation (llc_of (LL.Embed_index (Idx.Fixed_idx 260)))));
  p "byte lane modulus Fixed_idx (-1) wraps to 255 and is refused at single"
    (Option.exists
       (LL.lane_bounds_violation (llc_of (LL.Embed_index (Idx.Fixed_idx (-1)))))
       ~f:(fun msg ->
         String.is_substring msg ~substring:(Tn.label_name out)
         && String.is_substring msg ~substring:"renders at single (4 lanes)"
         && String.is_substring msg ~substring:"fixed modulus 255 > 4"))

(* 10. Vectorized inner loop over narrow storage with a loop-invariant lane extract (modulus 8 at
   bf16): [C_syntax.try_vectorize] must not splat the RNG extract at widened f32 (4 lanes), which
   would read lanes 4..7 out of bounds on cc. *)
let () =
  let x = mk_node Ops.bfloat16 [| 8 |] "lbs_vec_x" in
  let out_vec = mk_node Ops.bfloat16 [| 8; 8 |] "lbs_vec_out" in
  let out_ser = mk_node Ops.bfloat16 [| 8; 8 |] "lbs_ser_out" in
  let build axis out =
    let row = Ll_test.sym () and col = Ll_test.sym () in
    let draw = lane_extract (mod_lane row 8) in
    let rhs =
      LL.Binop (Ops.Add, (Ll_test.get x [| Ll_test.iter col |], Ops.bfloat16), (draw, Ops.bfloat16))
    in
    Ll_test.loop_n row 8
      (Ll_test.loop_n ~axis col 8 (Ll_test.set out [| Ll_test.iter row; Ll_test.iter col |] rhs))
  in
  let llc = LL.unflat_lines [ build LL.Vectorized out_vec; build LL.Serial out_ser ] in
  let opt = Ll_test.optimize ~materialized:[ x; out_vec; out_ser ] ~name:"lbs_vec_bf16" llc in
  let seed =
    [
      (x, Array.init 8 ~f:(fun j -> Float.of_int (j + 1)));
      (out_vec, Array.create ~len:64 (-1.));
      (out_ser, Array.create ~len:64 (-2.));
    ]
  in
  match Ll_test.execute ~name:"lbs_vec_bf16" opt ~seed ~read:[ out_vec; out_ser ] with
  | [ vec_vals; ser_vals ] ->
      p "Vectorized bf16 loop with loop-invariant mod-8 lane extract matches serial twin bitwise"
        (Array.length vec_vals = 64
        && Array.equal bits_equal vec_vals ser_vals
        && Array.for_all vec_vals ~f:(fun v -> Float.(v >= 1. && v < 10.)))
  | _ -> assert false
