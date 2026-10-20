(* gh-ocannl-928: the contract a routine with a BOUND gh-490 symbolic extent keeps about the cells
   past that extent, as [docs/shape_inference.md] ("Symbolic extents: cells past a bound extent")
   states it, pinned on executed values.

   Two accumulations over one symbolic-extent operand [x : s a] ([s] declared with maximum [range],
   bound as a launch parameter): [onto = x ++ "s a => s"] reduces ONTO the symbolic axis, so its
   target carries the extent; [over = x ++ "s a => a"] reduces OVER it, so every one of its cells is
   in extent and depends on the bound value. Each is lowered twice from the one record shape
   inference produced: as emitted (initializing, [=:+]) and with [initialize_neutral] cleared (the
   non-initializing [=+] every backprop accumulation is) -- same projections, so the two flavours
   differ only in the initialization the contract is about. They are lowered through
   [Assignments.to_low_level ~static_indices] and executed through [Ll_test]'s [?prelowered] seam
   (the gh-817/gh-878 pattern), which keeps the default schedule out of the nest: this file pins the
   lowering's contract, and the real-graph leg at the end plus [symbolic_extent_backward.ml] run it
   through the backends' own schedules.

   The doctrines of [Ll_test] decide the values. Every in-extent [x] cell is [1 + 10 s + a], so a
   cell read from the wrong row or column changes the sum; every cell PAST the extent holds a poison
   two orders of magnitude larger, so a read past the extent is a visible jump in an in-extent
   result; and the target's prior content is a per-cell sentinel distinct from every value the
   accumulation produces, so a write past the extent -- or a misplaced one -- fails exactly.

   The initializing record's own initialization pass is guarded like its body, so the sentinel
   survives past the extent there too; a third program spells the unguarded initialization as an
   explicit [Fetch] fill ahead of the non-initializing record, and a fill covers the materialized
   buffer whatever the extent -- the two must stay distinguishable.

   The last leg does the same for the other initializing writer, the [Rev_sides] scatter a
   concatenation's gradient lowers to: its per-target initialization is guarded by the target's own
   bound extents, so a sentinel past the extent survives in every target -- including one whose
   concatenation segment has width 1, which the scatter indexes at [Fixed_idx 0]. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Tn = Ir.Tnode
open Verdict.Claims

let range = 4
let width = 3
let extents = [ 0; 2; range ]

let rec find_accum ?(f = fun _ -> true) = function
  | Asgns.Accum_op { rhs; _ } as a when f rhs -> Some a
  | Asgns.Seq (a, b) -> Option.first_some (find_accum ~f a) (find_accum ~f b)
  | Asgns.Block_comment (_, body) -> find_accum ~f body
  | Asgns.Accum_op _ | Asgns.Noop | Asgns.Set_vec_unop _ | Asgns.Fetch _ -> None

let accum_of t = Option.value_exn ~here:[%here] (find_accum (Train.forward t).Asgns.asgns)

let non_initializing = function
  | Asgns.Accum_op r -> Asgns.Accum_op { r with initialize_neutral = false }
  | _ -> failwith "expected an Accum_op"

(* [x]'s cells at extent [e], row-major over [s; a]. *)
let in_extent_x s a = Float.of_int (1 + (10 * s) + a)
let poison s a = Float.of_int (1000 + (10 * s) + a)

let x_cells e =
  Array.init (range * width) ~f:(fun i ->
      let s = i / width and a = i % width in
      if s < e then in_extent_x s a else poison s a)

(* The prior content of a target: per-cell, distinct from every result (results are positive). *)
let prior n = Array.init n ~f:(fun i -> -1. -. Float.of_int i)
let row_sum s = List.sum (module Float) (List.init width ~f:(in_extent_x s)) ~f:Fn.id
let column_sum e a = List.sum (module Float) (List.init e ~f:(fun s -> in_extent_x s a)) ~f:Fn.id

let () =
  Tensor.unsafe_reinitialize ();
  let extent, bindings = Idx.get_static_symbol ~static_range:range Idx.Empty in
  let static_indices = Idx.bound_symbols bindings in
  let%op x = { x = 0.5 } in
  let%op onto = x ++ "s a => s" [ "s"; "a" ] in
  Shape.set_sym_dim s extent;
  Shape.set_dim a width;
  let%op over = x ++ "s a => a" in
  Train.set_materialized onto.Tensor.value;
  Train.set_materialized over.Tensor.value;
  let x_tn = x.Tensor.value and onto_tn = onto.Tensor.value and over_tn = over.Tensor.value in
  p "the reduction onto the symbolic axis is sized at the declared maximum"
    (Array.equal Int.equal (Lazy.force onto_tn.Tn.dims) [| range |]);
  let onto_accum = accum_of onto and over_accum = accum_of over in
  List.iter [ x_tn; onto_tn; over_tn ] ~f:Ll_test.materialize;
  let link ~name accum =
    let llc = Asgns.to_low_level ~static_indices accum in
    let o = Ll_test.optimize ~materialized:[ x_tn; onto_tn; over_tn ] ~static_indices ~name llc in
    Ll_test.link ~bindings ~name o
  in
  (* One compile per program; each extent is a launch of it, as gh-490 promises. *)
  let run (ctx, routine) ~target ~e =
    Idx.find_exn routine.Context.bindings extent := e;
    let ctx =
      Ll_test.run_linked (ctx, routine)
        ~seed:[ (x_tn, x_cells e); (target, prior (Tn.num_elems target)) ]
    in
    Context.get_values ctx target
  in
  let onto_init = link ~name:"sym_contract_onto_init" onto_accum in
  let onto_accum_only = link ~name:"sym_contract_onto_accum" (non_initializing onto_accum) in
  let over_init = link ~name:"sym_contract_over_init" over_accum in
  let over_accum_only = link ~name:"sym_contract_over_accum" (non_initializing over_accum) in
  let onto_fetch_then_accum =
    link ~name:"sym_contract_onto_fetch"
      (Asgns.Seq
         ( Asgns.Fetch { array = onto_tn; fetch_op = Asgns.Constant 0.; dims = onto_tn.Tn.dims },
           non_initializing onto_accum ))
  in
  List.iter extents ~f:(fun e ->
      let before = prior range in
      (* {1 Reads: an in-extent result never depends on a cell past the extent.} *)
      let got = run onto_init ~target:onto_tn ~e in
      if e > 0 then
        pf_all2 "extent %d, initializing onto: in-extent cells are the row sums of in-extent x" e
          (Array.sub got ~pos:0 ~len:e) (Array.init e ~f:row_sum) ~f:Float.equal;
      (* Past the extent an initializing accumulation leaves UNSPECIFIED content by contract; the
         lowering guards its initialization pass like the body, so a planted sentinel survives
         there. Pinned as what the lowering does, not as a promise. *)
      if e < range then
        pf_all2 "extent %d, initializing onto: every cell past the extent keeps its sentinel" e
          (Array.sub got ~pos:e ~len:(range - e))
          (Array.sub before ~pos:e ~len:(range - e))
          ~f:Float.equal;
      (* A [Fetch] in the same routine is a fill, and a fill covers the materialized buffer (rule 4)
         whatever the extent: the initialization the lowering emitted before it was guarded, spelled
         as that fill ahead of the non-initializing record, still overwrites every sentinel. *)
      let got = run onto_fetch_then_accum ~target:onto_tn ~e in
      if e > 0 then
        pf_all2 "extent %d, Fetch fill then accumulation onto: in-extent cells are the row sums" e
          (Array.sub got ~pos:0 ~len:e) (Array.init e ~f:row_sum) ~f:Float.equal;
      if e < range then
        p_all
          (Printf.sprintf
             "extent %d, Fetch fill then accumulation onto: every cell past the extent is filled" e)
          (Array.to_list (Array.sub got ~pos:e ~len:(range - e)))
          ~f:(Float.equal 0.);
      let got = run over_init ~target:over_tn ~e in
      pf_all2 "extent %d, initializing over: every cell sums exactly the in-extent rows" e got
        (Array.init width ~f:(column_sum e))
        ~f:Float.equal;
      (* {1 Writes: a non-initializing accumulation leaves cells past the extent untouched.} *)
      let got = run onto_accum_only ~target:onto_tn ~e in
      if e > 0 then
        pf_all2 "extent %d, non-initializing onto: in-extent cells add the row sums to their prior"
          e (Array.sub got ~pos:0 ~len:e)
          (Array.init e ~f:(fun s -> before.(s) +. row_sum s))
          ~f:Float.equal;
      if e < range then
        pf_all2 "extent %d, non-initializing onto: every cell past the extent keeps its prior" e
          (Array.sub got ~pos:e ~len:(range - e))
          (Array.sub before ~pos:e ~len:(range - e))
          ~f:Float.equal;
      let got = run over_accum_only ~target:over_tn ~e in
      let over_before = prior width in
      pf_all2 "extent %d, non-initializing over: every cell adds exactly the in-extent rows" e got
        (Array.init width ~f:(fun a -> over_before.(a) +. column_sum e a))
        ~f:Float.equal);

  (* {1 The control: the claims above discriminate.} The same two records lowered with the extent
     UNBOUND carry no guard -- gh-490's maximum-shape semantics -- so launched with the extent bound
     to 2 they read the poison and write past the extent. Were the guard ever dropped from the bound
     lowering, the claims above would fail the same way. *)
  let link_unguarded ~name accum =
    let o =
      Ll_test.optimize ~materialized:[ x_tn; onto_tn; over_tn ] ~name (Asgns.to_low_level accum)
    in
    Ll_test.link ~bindings ~name o
  in
  let e = 2 in
  let got =
    run
      (link_unguarded ~name:"sym_contract_onto_unguarded" (non_initializing onto_accum))
      ~target:onto_tn ~e
  in
  p "control: the unguarded non-initializing onto writes the cells past the extent"
    (not
       (Array.equal Float.equal
          (Array.sub got ~pos:e ~len:(range - e))
          (Array.sub (prior range) ~pos:e ~len:(range - e))));
  let got =
    run (link_unguarded ~name:"sym_contract_over_unguarded" over_accum) ~target:over_tn ~e
  in
  p_all "control: every cell of the unguarded over has read the poison past the extent"
    (Array.to_list got) ~f:(fun v -> Float.(v > 1000.));

  (* {1 The same reads through the real compile path and the backend's default schedule.} *)
  let ctx = Train.init_params (Context.auto ()) bindings over in
  let fwd = Asgns.sequence [ Train.forward onto; Train.forward over ] in
  let ctx, routine = Train.to_routine ctx bindings fwd in
  let e = 2 in
  Idx.find_exn routine.Context.bindings extent := e;
  let ctx = Context.set_values ctx x_tn (x_cells e) in
  let ctx = Context.run ctx routine in
  pf_all2 "real compile at extent %d: onto's in-extent cells are the in-extent row sums" e
    (Array.sub (Context.get_values ctx onto_tn) ~pos:0 ~len:e)
    (Array.init e ~f:row_sum) ~f:Float.equal;
  pf_all2 "real compile at extent %d: over sums exactly the in-extent rows" e
    (Context.get_values ctx over_tn)
    (Array.init width ~f:(column_sum e))
    ~f:Float.equal;

  (* {1 The [Rev_sides] scatter's initialization is guarded the same way.} A concatenation's
     gradient writes each operand's gradient through [Rev_sides] -- one record, one target per
     operand. The front end only emits it NON-initializing (a backprop [=+]), so the initializing
     flavour is that record with [initialize_neutral] set: same projections. No projection carrying
     a symbolic extent is proven surjective ([Affine.is_surjective]), so every target gets an
     initialization pass. *)
  let%op u = { u = 0.5 } in
  let%op v = { v = 0.5 } in
  let%op cat = (u, v) ++^ "s, b; s, c => s, b^c" [ "s"; "b"; "c" ] in
  Shape.set_sym_dim s extent;
  Shape.set_dim b 2;
  Shape.set_dim c 1;
  let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad in
  let rev_sides =
    match
      find_accum
        ~f:(function Asgns.Rev_sides _ -> true | _ -> false)
        (Option.value_exn ~here:[%here] cat.Tensor.diff).Tensor.backprop.Asgns.asgns
    with
    | Some (Asgns.Accum_op r) ->
        p "the concatenation's gradient scatter is emitted non-initializing"
          (not r.initialize_neutral);
        Asgns.Accum_op { r with initialize_neutral = true }
    | _ -> failwith "expected the concatenation's Rev_sides gradient"
  in
  let gu = grad_of u and gv = grad_of v and gcat = grad_of cat in
  let cat_width = 3 in
  List.iter [ gu; gv; gcat ] ~f:Ll_test.materialize;
  let rev_init =
    let name = "sym_contract_rev_init" in
    let llc = Asgns.to_low_level ~static_indices rev_sides in
    Ll_test.link ~bindings ~name
      (Ll_test.optimize ~materialized:[ gu; gv; gcat ] ~static_indices ~name llc)
  in
  (* [cat]'s gradient as [x] is seeded: [1 + 10 s + j] in extent, poison past it. *)
  let gcat_cells e =
    Array.init (range * cat_width) ~f:(fun i ->
        let s = i / cat_width and j = i % cat_width in
        if s < e then in_extent_x s j else poison s j)
  in
  List.iter extents ~f:(fun e ->
      let ctx, routine = rev_init in
      Idx.find_exn routine.Context.bindings extent := e;
      let ctx =
        Ll_test.run_linked (ctx, routine)
          ~seed:[ (gcat, gcat_cells e); (gu, prior (range * 2)); (gv, prior range) ]
      in
      (* [u]'s gradient is columns [0, 2) of [cat]'s, [v]'s is column 2. *)
      List.iter
        [ ("u", gu, 2, 0); ("v", gv, 1, 2) ]
        ~f:(fun (name, g, g_width, offset) ->
          let got = Context.get_values ctx g in
          let in_extent = e * g_width in
          if e > 0 then
            pf_all2
              "extent %d, initializing Rev_sides: %s's in-extent gradient is its slice of cat's" e
              name
              (Array.sub got ~pos:0 ~len:in_extent)
              (Array.init in_extent ~f:(fun i -> in_extent_x (i / g_width) (offset + (i % g_width))))
              ~f:Float.equal;
          if e < range then
            pf_all2
              "extent %d, initializing Rev_sides: every cell of %s's gradient past the extent \
               keeps its sentinel"
              e name
              (Array.sub got ~pos:in_extent ~len:((range - e) * g_width))
              (Array.sub (prior (range * g_width)) ~pos:in_extent ~len:((range - e) * g_width))
              ~f:Float.equal))
