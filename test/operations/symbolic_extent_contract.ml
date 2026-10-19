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
   accumulation produces, so a write past the extent -- or a misplaced one -- fails exactly. *)

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

let rec find_accum = function
  | Asgns.Accum_op _ as a -> Some a
  | Asgns.Seq (a, b) -> Option.first_some (find_accum a) (find_accum b)
  | Asgns.Block_comment (_, body) -> find_accum body
  | Asgns.Noop | Asgns.Set_vec_unop _ | Asgns.Fetch _ -> None

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
  List.iter extents ~f:(fun e ->
      let before = prior range in
      (* {1 Reads: an in-extent result never depends on a cell past the extent.} *)
      let got = run onto_init ~target:onto_tn ~e in
      if e > 0 then
        pf_all2 "extent %d, initializing onto: in-extent cells are the row sums of in-extent x" e
          (Array.sub got ~pos:0 ~len:e) (Array.init e ~f:row_sum) ~f:Float.equal;
      (* Past the extent an initializing accumulation leaves UNSPECIFIED content: shown, not
         pinned. *)
      Stdio.eprintf
        "extent %d, initializing onto, cells past the extent (not part of the golden): %s\n" e
        (String.concat ~sep:" "
           (List.map (Array.to_list (Array.sub got ~pos:e ~len:(range - e))) ~f:Float.to_string));
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
    ~f:Float.equal
