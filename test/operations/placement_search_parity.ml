(* gh-ocannl-1878: the placement pre-search of [Autotune.model_default] (config
   [model_default_placements]) chooses on the lowering the compile prepares. A compile sinks zeros
   under [buffer_aliasing] before scheduling ([Context.lowered_transform_input]), which moves the
   fission boundaries; a leaf scored on the bare hermetic lowering was ranked on a fission that
   never ships, and the compile then ran a selection the search had never priced.

   The relationship pinned, under both settings of the gate: the rank the search gave the leaf it
   won with ([mc_placement_rank_ms]) is the rank the selection then reports on that leaf
   ([mc_rank_ms]), exactly, since one is a rerun of the other. A rank sums each kernel's larger leg,
   so it shows the zeros' position where they change the kernels and a kernel can be compute- or
   memory-bound by turns: the envelope in the dune rule, on a segmented GPU routine. So [cuda
   stand-in] ([Ll_test.cuda_model_target]) is the leg that discriminates on every host, and on a GPU
   backend the backend's own legs do too; cc's one kernel does not. The own legs hold the claim
   against a real compile, and check what that compile computes.

   Two graphs cover both kinds of winner on the stand-in: [picked], where a flip beats the default
   placements, and [kept], where the default placements win. Both are training steps:
   [Train.grad_update] zeroes the gradients in one up-front block, which is what sinks.

   The floor claim is what keeps fathoming sound on the sunk form: the search's bound is read off
   the unprepared lowering, so it must not depend on the statement order. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module CM = Ir.Cost_model
module LL = Ir.Low_level
open Verdict.Claims

let bound = Ir.Indexing.Empty

let inputs ~l ~n =
  NTDSL.init ~l ~prec:Ir.Ops.single ~b:[ n ] ~o:[ n ]
    ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:9 ~offset:0. ~stride:0.125)
    ()

let picked () =
  let n = 32 in
  let xs = inputs ~l:"pp_xs" ~n in
  let%op h1 = relu ({ w1; o = [ n ] } * xs) in
  let%op y = exp ({ w2; o = [ n ] } * h1) in
  let%op err = (y *. y) + relu y in
  let%op loss = err ++ "...|... => |->0" in
  loss

let kept () =
  let n = 128 in
  let xs = inputs ~l:"pk_xs" ~n in
  let%op h1 = relu ({ b1; o = [ n ] } + ({ w1; o = [ n ] } * xs)) in
  let%op h2 = relu ({ b2; o = [ n ] } + ({ w2; o = [ n ] } * h1)) in
  let%op y = exp ({ w3; o = [ n ] } * h2) in
  let%op err = (y *. y) + relu y in
  let%op loss = err ++ "...|... => |->0" in
  loss

let is_placement_pick (r : Autotune.model_choice) =
  String.is_prefix r.mc_label ~prefix:"placements["

let show_rank = Option.value_map ~default:"none" ~f:(Printf.sprintf "%.6f")

(* The search's rank is the selection's: both present and the same number. An exact comparison: the
   selection reruns on the same lowering, so any difference is a different program. *)
let rank_parity name (r : Autotune.model_choice) =
  Stdio.eprintf "%s: search %s ms, selection %s ms, %s (not part of the golden)\n%!" name
    (show_rank r.mc_placement_rank_ms)
    (show_rank r.mc_rank_ms) r.mc_label;
  p name
    (match (r.mc_placement_rank_ms, r.mc_rank_ms) with
    | Some searched, Some selected -> Float.equal searched selected
    | _ -> false)

let scenario ~aliasing ~gname ~graph ~stand_in_picks =
  Unix.putenv "OCANNL_BUFFER_ALIASING" (Bool.to_string aliasing);
  let tag s = Printf.sprintf "[aliasing %s] %s: %s" (if aliasing then "on" else "off") gname s in
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 5;
  let loss = graph () in
  let name = Printf.sprintf "psp_%s_%b" gname aliasing in
  let comp : Asgns.comp =
    let c = Train.grad_update loss in
    { c with asgns = Asgns.Block_comment (name, c.asgns) }
  in
  let with_params = Train.init_params (Context.auto ()) bound loss in
  (* The control: the graph's zeros move, exactly under the gate. *)
  let raw = Context.lowered_for_decisions with_params comp bound in
  let prepared = Context.lowered_transform_input raw in
  p
    (tag "the compile's preparation moves the zeros exactly under the gate")
    (Bool.equal aliasing (not (phys_equal raw.LL.llc prepared.LL.llc)));
  p
    (tag "the floor fathoming reads is the same on the prepared lowering")
    (Sexp.equal
       (CM.sexp_of_floor (CM.completion_floor raw.LL.llc))
       (CM.sexp_of_floor (CM.completion_floor prepared.LL.llc)));
  let stand_in =
    Autotune.model_pick ~target:(Ll_test.cuda_model_target ()) with_params comp bound
  in
  p
    (tag "cuda stand-in: the winner is the expected kind")
    (Bool.equal stand_in_picks (is_placement_pick stand_in));
  rank_parity (tag "cuda stand-in: the search's rank is the selection's") stand_in;
  let own = Autotune.model_pick with_params comp bound in
  rank_parity (tag "own target, uncompiled: the search's rank is the selection's") own;
  (* Against a real compile: the selection it ran, on the pick it made, computing the same
     values. *)
  let plain_ctx, plain = Context.compile with_params comp bound in
  let plain_ctx = Context.run plain_ctx plain in
  let report = ref None in
  let ctx, routine =
    Autotune.model_default ~report:(fun r -> report := Some r) with_params comp bound
  in
  let ctx = Context.run ctx routine in
  (match !report with
  | None -> p (tag "model_default reports its choice") false
  | Some r ->
      rank_parity (tag "compiled: the search's rank is the selection's") r;
      p
        (tag "compiled: model_pick names model_default's pick")
        (String.equal r.mc_label own.mc_label));
  (* A schedule reorders the f32 reductions, so a cell cancelling to near zero moves by an ulp of
     the terms it sums: the admitted difference scales with the array, not the cell. *)
  let same_values name tn =
    let got = Context.get_values ctx tn and want = Context.get_values plain_ctx tn in
    let scale = Array.fold want ~init:1. ~f:(fun m w -> Float.max m (Float.abs w)) in
    p_close
      (tag ("compiled: the pick computes the plain compile's " ^ name))
      got want
      ~tol:(fun _ -> 1e-5 *. scale)
  in
  same_values "loss" loss.Tensor.value;
  List.iteri (Set.to_list loss.Tensor.params) ~f:(fun i param ->
      same_values
        (Printf.sprintf "parameter %d's gradient" i)
        (Option.value_exn ~here:[%here] param.Tensor.diff).grad)

let () =
  List.iter [ false; true ] ~f:(fun aliasing ->
      scenario ~aliasing ~gname:"picked" ~graph:picked ~stand_in_picks:true;
      scenario ~aliasing ~gname:"kept" ~graph:kept ~stand_in_picks:false)
