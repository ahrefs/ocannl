(* Collapsing a scope keeps the scope's rounding (gh-ocannl-1788).

   A [Local_scope] converts its value into its own precision before its slot reads it. [simplify]
   collapses a single-assignment scope into the expression around it, which drops that step, so the
   collapse is exact only when the slot evaluates at the scope's precision, or when the value is a
   load whose storage precision the scope's widens exactly. Before gh-ocannl-1788 it collapsed every
   float scope: a bfloat16 scope over a single product, read in a single sum, lost its rounding
   wherever the scope's local is declared at bfloat16 -- Metal, CUDA and HIP, and cc under
   [narrow_compute_f32=false]. (Under cc's default [narrow_compute_f32=true] the local is single
   anyway, so the collapse happened to be exact there.)

   Each shape is executed in four forms: the scoped form through the [?prelowered] seam as written
   (the uncollapsed oracle); the same through a schedule whose trailing [simplify] decides the
   collapse; a virtual node read in the sum, inlined by [optimize], whose own [simplify] decides it;
   and the value written straight into the sum, which shows that the oracle's rounding is observable
   where the claim needs it to be. All run under each narrow-compute policy
   ([Ir.Numerics.set_policy] in-process, distinct routine names per policy). The single inputs and
   their squares lie off the bfloat16 grid; the bfloat16 inputs lie on it, so seeding them is
   exact. *)
open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Ops = Ir.Ops
open Verdict.Claims

let bf = Ops.bfloat16
let f32 = Ops.single
let n = 4
let bnode = Ll_test.node_factory ~prec:bf ~first_id:24700 ~dims:[| n |] ()
let fnode = Ll_test.node_factory ~prec:f32 ~first_id:24800 ~dims:[| n |] ()

(* 1 + (2k + 1/4) * 2^-10: off the bfloat16 grid (8 significant bits), and so are their squares. *)
let f32_inputs = Array.init n ~f:(fun k -> 1. +. ((Float.of_int (2 * k) +. 0.25) *. 0.0009765625))

(* Exactly representable in bfloat16, so seeding a bfloat16 node is exact. *)
let bf_inputs = Array.init n ~f:(fun k -> 1. +. (Float.of_int (k + 1) *. 0.0078125))

type shape = {
  label : string;
  slot : Ops.prec;  (** the destination's, hence the sum's evaluation precision *)
  scope_prec : Ops.prec;
  value : x:LL.scalar_arg -> y:LL.scalar_arg -> LL.scalar_t;  (** the scope's value *)
}

(* [out[i] = x[i] + scope(value)], with [x] at the slot's precision and [y] at the other one.
   [`Oracle] runs it as written, [`Scheduled] through [Schedule.apply]'s trailing simplify, and
   [`Unscoped] runs the value written straight into the sum -- the collapse done by hand, without
   the scope's rounding. [`Inlined] reaches the scope the ordinary way: a virtual node computing the
   value, read in the sum and inlined by [optimize], whose own simplify decides the collapse.
   Returns the values, the number of scopes left in the program that ran, and whether the scope's
   node is virtual there. *)
let run shape ~form ~policy =
  let tag =
    match form with
    | `Oracle -> "pre"
    | `Scheduled -> "sch"
    | `Unscoped -> "raw"
    | `Inlined -> "inl"
  in
  let name = Printf.sprintf "scr_%s_%s_%s" shape.label tag policy in
  let s = Ll_test.sym () in
  let i = [| Ll_test.iter s |] in
  let node prec suffix = (if Ops.equal_prec prec bf then bnode else fnode) (name ^ suffix) in
  let other = if Ops.equal_prec shape.slot bf then f32 else bf in
  let x_tn = node shape.slot "_x" and y_tn = node other "_y" in
  let mid = node shape.scope_prec "_mid" and out = node shape.slot "_out" in
  List.iter [ x_tn; y_tn; out ] ~f:Ll_test.materialize;
  let x = (Ll_test.get x_tn i, shape.slot) and y = (Ll_test.get y_tn i, other) in
  let v = shape.value ~x ~y in
  let sum operand = Ll_test.loop_n s n (Ll_test.set out i (LL.Binop (Ops.Add, x, operand))) in
  let raw = sum (v, shape.slot) in
  let o =
    match form with
    | `Unscoped -> Ll_test.optimize_scoped ~name ~raw raw
    | `Inlined ->
        Ll_test.virtualize mid;
        let s1 = Ll_test.sym () in
        let i1 = [| Ll_test.iter s1 |] in
        let v1 = shape.value ~x:(Ll_test.get x_tn i1, shape.slot) ~y:(Ll_test.get y_tn i1, other) in
        Ll_test.optimize ~materialized:[ x_tn; y_tn; out ] ~name
          (LL.unflat_lines
             [
               Ll_test.loop_n s1 n (Ll_test.set mid i1 v1); sum (Ll_test.get mid i, shape.scope_prec);
             ])
    | `Oracle | `Scheduled ->
        Ll_test.optimize_scoped ~name ~raw
          (sum (Ll_test.scope mid ~orig_indices:i v, shape.scope_prec))
  in
  let o =
    match form with
    | `Scheduled -> Ir.Schedule.apply [ Ir.Schedule.Retype { axis = s; ty = LL.Serial } ] o
    | `Oracle | `Unscoped | `Inlined -> o
  in
  let seed tn =
    (tn, if Ops.equal_prec (Lazy.force tn.Ir.Tnode.storage_prec) bf then bf_inputs else f32_inputs)
  in
  let values = Ll_test.execute ~name o ~seed:[ seed x_tn; seed y_tn ] ~read:[ out ] in
  (List.hd_exn values, Ll_test.count_scopes o.LL.llc, Ll_test.known_virtual o mid)

let bits vs = Array.map vs ~f:(fun v -> Int64.bits_of_float v)
let same_bits a b = Array.equal Int64.equal (bits a) (bits b)

let narrow_product =
  (* A bfloat16 scope over a single product of single loads, in a single sum. *)
  {
    label = "narrow";
    slot = f32;
    scope_prec = bf;
    value = (fun ~x ~y:_ -> LL.Binop (Ops.Mul, x, x));
  }

let wide_product =
  (* A single scope over a single product of single loads, in a bfloat16 sum. *)
  { label = "wide"; slot = bf; scope_prec = f32; value = (fun ~x:_ ~y -> LL.Binop (Ops.Mul, y, y)) }

let narrow_load =
  (* A bfloat16 scope over a bfloat16 load, in a single sum: the scope's rounding is the
     identity. *)
  { label = "nload"; slot = f32; scope_prec = bf; value = (fun ~x:_ ~y -> fst y) }

let widening_load =
  (* A single scope over a bfloat16 load, in a bfloat16 sum: the widening is exact. *)
  { label = "wload"; slot = bf; scope_prec = f32; value = (fun ~x ~y:_ -> fst x) }

let () =
  let saved = Ir.Numerics.get () in
  List.iter [ true; false ] ~f:(fun narrow_compute_f32 ->
      Ir.Numerics.set_policy { saved with Ir.Numerics.narrow_compute_f32 };
      let policy = if narrow_compute_f32 then "nc_on" else "nc_off" in
      Verdict.case
        (if narrow_compute_f32 then "narrow floats computed in single"
         else "narrow floats computed at storage width")
        (fun () ->
          List.iter [ narrow_product; wide_product ] ~f:(fun shape ->
              let pre, pre_scopes, _ = run shape ~form:`Oracle ~policy in
              let sch, sch_scopes, _ = run shape ~form:`Scheduled ~policy in
              let raw, _, _ = run shape ~form:`Unscoped ~policy in
              let inl, inl_scopes, inlined = run shape ~form:`Inlined ~policy in
              (* The rounding is observable wherever the scope's local is declared at the scope's
                 own precision, which every backend does under [narrow_compute_f32=false]; cc's
                 default declares the local single, so there the two may agree. *)
              if not narrow_compute_f32 then
                p
                  (Printf.sprintf
                     "%s: the oracle's rounding is observable -- the value written straight into \
                      the sum computes different values"
                     shape.label)
                  (not (same_bits pre raw));
              p
                (Printf.sprintf
                   "%s: the uncollapsed oracle runs its scope, and the schedule's simplify keeps it"
                   shape.label)
                (pre_scopes = 1 && sch_scopes = 1);
              p
                (Printf.sprintf "%s: the scheduled form computes the oracle's values bitwise"
                   shape.label)
                (same_bits pre sch);
              p
                (Printf.sprintf
                   "%s: optimize inlines the virtual node and keeps its scope, computing the \
                    oracle's values bitwise"
                   shape.label)
                (inlined && inl_scopes = 1 && same_bits pre inl);
              let show vs =
                String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%h")))
              in
              Stdio.eprintf "(not part of the golden) %s %s: oracle %s; unscoped %s\n" policy
                shape.label (show pre) (show raw));
          List.iter [ narrow_load; widening_load ] ~f:(fun shape ->
              let pre, _, _ = run shape ~form:`Oracle ~policy in
              let sch, sch_scopes, _ = run shape ~form:`Scheduled ~policy in
              let inl, inl_scopes, inlined = run shape ~form:`Inlined ~policy in
              p
                (Printf.sprintf "%s: an exact scope over a load is still collapsed" shape.label)
                (sch_scopes = 0);
              p
                (Printf.sprintf "%s: the collapsed form computes the oracle's values bitwise"
                   shape.label)
                (same_bits pre sch);
              p
                (Printf.sprintf
                   "%s: optimize inlines the virtual node and collapses its scope, computing the \
                    oracle's values bitwise"
                   shape.label)
                (inlined && inl_scopes = 0 && same_bits pre inl))));
  Ir.Numerics.set_policy saved
