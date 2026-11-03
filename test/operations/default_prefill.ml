(* gh-ocannl-1774: discriminating by default.

   A fresh device buffer is already zero, so a routine that wrongly drops a zero computes the same
   values from it and its parity check passes. gh-ocannl-1625 made each zero-transform test prefill
   what it zeroes, opt-in; [Ll_test.run_linked] now prefills every node the routine writes before
   reading, unless the test names it in [~seed] or opts it out with [~fresh_zero] and a reason.

   Part 1 pins the harness's contract on hand-built IR, on any backend: which nodes the default
   reaches (a node written before it is read, never a routine input), that it is per-cell
   {!Ll_test.prefill} and not a zero, that [~fresh_zero] keeps the fresh zero, that a node the
   context already holds keeps its value (a second run starts from the first run's results), and
   that a stale or unexplained opt-out raises.

   Part 2 is the oracle gh-ocannl-1625 left missing: a matmul's localized-accumulator zero seed on
   the backend's own default pipeline, checked against a CLOSED FORM from a prefilled output. On a
   GPU backend, [gpu_elide_covered_zeros] drops the output's zero and [Low_level.seed_first_write]
   opens the localized accumulator from [0.0] itself; the unscheduled twin [schedule_ops] compares
   against takes the same seed, so a seed that read the node instead computed the same values from
   fresh buffers in both twins and survived. Against a closed form, from a prefilled buffer, it
   cannot: the prefill reaches the result. On cc the same leg discriminates
   [C_syntax.try_localize_serial_reduce]'s localized seed (gh-ocannl-1625's A04). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Asgns = Ir.Assignments

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let raises f = match f () with _ -> false | exception Invalid_argument _ -> true

(* {1 Part 1: the harness contract} *)

let node = L.node_factory ~first_id:1774000 ~dims:[| 4 |] ()
let x = node "dp_x"
let y = node "dp_y"
let () = List.iter [ x; y ] ~f:L.materialize
let xv = [| 10.; 20.; 30.; 40. |]

(* [y[i] = x[i] + 1] over the first two cells only: [y] is written and never read, [x] is read and
   never written, and [y]'s cells 2 and 3 are covered by no writer, so what they read back is
   exactly what the buffer held before the run. *)
let partial_write () =
  let i = L.sym () in
  L.optimize ~name:"dp_partial"
    (L.loop ~upto:1 i (L.set_at y (L.iter i) (L.add (L.get x [| L.iter i |]) (L.c 1.))))

let covered = [| 11.; 21. |]
let uncovered vs = Array.sub vs ~pos:2 ~len:2
let written vs = Array.sub vs ~pos:0 ~len:2

let () =
  let o = partial_write () in
  let ctx, routine = L.link ~name:"dp_partial" o in
  let wbr = L.written_before_read routine in
  p "the default reaches a node the routine writes before reading" (Set.mem wbr y);
  p "the default leaves a routine input alone" (not (Set.mem wbr x));
  let prefill = Option.value_exn ~here:[%here] (L.prefill y) in
  p_all "the prefill is clear of zero and of the sentinel" (Array.to_list prefill) ~f:(fun v ->
      Float.(v <> 0. && v <> L.sentinel));
  (* Default: no seed for [y], so its uncovered cells read back the prefill, not a zero. *)
  let got = Context.get_values (L.run_linked (ctx, routine) ~seed:[ (x, xv) ]) y in
  p_all2 "the routine's own writes land over the default prefill" (written got) covered
    ~f:Float.equal;
  p_all2 "cells no writer covers read back the default prefill" (uncovered got) (uncovered prefill)
    ~f:Float.equal;
  (* [~fresh_zero]: the same program linked in a fresh root, whose buffers no run has written yet,
     opted out. (Not the context above: its buffers already hold the first run's values, which is
     what a context value cannot show — the device buffer is shared by every context of a root.) *)
  let got =
    Context.get_values
      (L.run_linked
         ~fresh_zero:[ (y, "this leg pins the opt-out itself") ]
         (L.link ~ctx:(Context.auto ()) ~name:"dp_partial" o)
         ~seed:[ (x, xv) ])
      y
  in
  p_all2 "an opted-out node keeps the fresh zero" (uncovered got) [| 0.; 0. |] ~f:Float.equal;
  (* A held value is kept: run once from an explicit seed, then again on the returned context with
     no seed for [y]. The second run must not prefill over what the first one left. *)
  let ctx1 = L.run_linked (ctx, routine) ~seed:[ (x, xv); (y, L.blank 4) ] in
  let got = Context.get_values (L.run_linked (ctx1, routine) ~seed:[ (x, xv) ]) y in
  p_all2 "a second run starts from the first run's values, not a new prefill" (uncovered got)
    [| L.sentinel; L.sentinel |] ~f:Float.equal;
  p "an opt-out without a reason raises"
    (raises (fun () -> L.run_linked ~fresh_zero:[ (y, " ") ] (ctx, routine) ~seed:[ (x, xv) ]));
  p "an opt-out of a node the routine reads first raises"
    (raises (fun () ->
         L.run_linked ~fresh_zero:[ (x, "stale on purpose") ] (ctx, routine) ~seed:[ (x, xv) ]))

(* {1 Part 2: the localized-accumulator zero seed against a closed form} *)

(* 96 output cells, past the GPU pipeline's [min_parallel] of 64, so the default pipeline maps the
   output loops onto the grid on a GPU backend, and the reduction of 5 stays a serial loop each
   thread localizes. *)
let m, k, n = (8, 5, 12)
let a_init idcs = L.cycle ~dims:[| m; k |] ~modulus:7 ~offset:1. ~stride:1. idcs
let b_init idcs = L.cycle ~dims:[| k; n |] ~modulus:11 ~offset:(-4.) ~stride:1. idcs

(* Integer operands with integer products well inside f32's exact range: the closed form is exact,
   so the claims are bitwise. *)
let products = L.matmul_products ~m ~n ~k a_init b_init
let closed_form = Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id))

(* A fresh matmul per leg, so no leg's compile can reuse another's lowering. *)
let matmul ~name ?lowered_transform () =
  let ma = NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.single ~o:[ m ] ~i:[ k ] ~f:a_init () in
  let mb = NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.single ~o:[ k ] ~i:[ n ] ~f:b_init () in
  let%op mc = ma * mb in
  let comp = Train.forward mc in
  let comp = { comp with Asgns.asgns = Asgns.Block_comment (name, comp.Asgns.asgns) } in
  let linked = Context.compile ?lowered_transform (Context.auto ()) comp Ir.Indexing.Empty in
  (linked, mc.Tensor.value)

let gpu_annotated o = [ Ir.Schedule.apply (Ir.Schedule.default_gpu ~min_parallel:1 o) o ]

let () =
  p_all "the operands' products sum exactly in f32" products
    ~f:(L.exact_partials ~any_sign:true ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single));
  (* The explicit prefill, and its own distance from the result: a seed that read the node instead
     of opening from zero adds a prefill cell to every result cell. *)
  let pre = Array.init (m * n) ~f:(fun i -> 1000.5 +. Float.of_int i) in
  p_all2 "the prefill differs from the closed form in every cell" pre closed_form ~f:(fun a b ->
      Float.(a <> b));
  let seeded ~name ?lowered_transform () =
    let linked, out = matmul ~name ?lowered_transform () in
    p
      (name ^ ": the output is written before it is read")
      (Set.mem (L.written_before_read (snd linked)) out);
    Context.get_values (L.run_linked linked ~seed:[ (out, pre) ]) out
  in
  p_all2 "the default pipeline's matmul from a prefilled output is the closed form"
    (seeded ~name:"dp_mm_default" ()) closed_form ~f:Float.equal;
  (* As [schedule_ops]'s twins compile it: the GPU annotator, at a [min_parallel] of 1. *)
  p_all2 "the GPU-annotated matmul from a prefilled output is the closed form"
    (seeded ~name:"dp_mm_gpu" ~lowered_transform:gpu_annotated ())
    closed_form ~f:Float.equal;
  (* The default prefill alone, with no seed for the output: what a test that forgets one gets. *)
  let linked, out = matmul ~name:"dp_mm_unseeded" () in
  p_all2 "the same matmul from the default prefill is the closed form"
    (Context.get_values (L.run_linked linked ~seed:[]) out)
    closed_form ~f:Float.equal;
  Stdio.eprintf "default_prefill: backend %s (not part of the golden)\n%!" backend_name
