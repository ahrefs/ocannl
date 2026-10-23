(* gh-ocannl-1481: [Train.tune_placements] on a routine with a footprint-scoped node.

   The routine is footprint_diagonal_einsum's: an 8x32 by 32x8 matrix product, read only through its
   diagonal (here over hosted-constant inputs), whose reduction extent is above
   [virtualize_max_inline_reduction]. The default policy footprint-scopes the product -- it stays
   virtual and the 8 diagonal cells are computed into a scratch node of the [footprint] namespace,
   minted by the lowering.

   Every autotune candidate compiles the ONE base lowering's code rebased onto its own compile's
   fresh [optimize_ctx] (see [Autotune.compile_candidate_gated]), and that fresh lowering mints a
   scratch of its own. The scratch's [Never_virtual] used to live only in the lowering's lineage
   table, so the base's scratch had no placement in the candidate's fork, and arm A's first
   candidate tripped [Tnode.Placements.is_materialized_peek]'s [None] assertion in
   [Schedule.analyze_parallel_chains]. Now the scratch's placement is its declared intent, which
   travels with the node.

   Leg 1 (the issue): the call completes, arm A timed a rebased candidate with none failing, and the
   shipped routine computes the untuned output bit for bit (integer inputs, every sum exact in
   single precision). Its premise is checked, not assumed: the default lowering footprint-scopes the
   product.

   Leg 2 (gh-ocannl-1440's flip-group commit rule, which needs a multi-alternative group on cc -- a
   footprint-scoped node offers one, and this issue's assertion stopped every search over it): its
   flip group is [`Materialize] and [`Inline], measured against one incumbent. Timings are scripted
   (arm A 4 ms, arm B 50 ms, every flip 1 ms) so the flips beat the incumbent. Unbudgeted, both
   alternatives are measured and the group commits: a flip ships. With the call's virtual clock
   jumping past its budget once the group's first alternative has reported, the group is cut after
   one of its two alternatives, and a cut group commits nothing -- arm A ships, though the measured
   alternative beat it. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

let n = 8

(* Above the reduction cap's default of 16. *)
let k = 32
let () = assert (LL.virtualize_settings.max_inline_reduction = 16)
let () = assert LL.virtualize_settings.footprint_materialization

(* Hosted constants rather than [range_of_shape]: a computed input would be a flip candidate of its
   own. Small integers, so every sum of products is exact in single precision. *)
let ints label ~rows ~cols modulus =
  TDSL.ndarray
    (Array.init (rows * cols)
       ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus ~offset:(-2.) ~stride:1.))
    ~label:[ label ] ~output_dims:[ rows; cols ] ()

let x = ints "x" ~rows:n ~cols:k 7
let y = ints "y" ~rows:k ~cols:n 5
let%op a = x +* "ik; kj => ij" y
let%op d = a ++ "ii => i"
let comp = Train.forward d

let reference =
  let ctx, routine = Context.compile (Context.cpu ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx d.Tensor.value in
  Context.release ctx;
  values

let p_bitwise name got =
  p_all2 ~min:n name got reference ~f:(fun g r ->
      Int64.equal (Int64.bits_of_float g) (Int64.bits_of_float r))

(* The premise, read off the default lowering and the decision surface. *)
let () =
  let opt = Context.lowered_for_decisions (Context.cpu ()) comp Ir.Indexing.Empty in
  p "premise: the default lowering reads the product through a footprint scratch"
    (Hashtbl.existsi opt.LL.traced_store ~f:(fun ~key ~data:_ ->
         String.equal key.Tn.namespace LL.footprint_namespace));
  let surface = Autotune.placement_surface (Context.cpu ()) comp Ir.Indexing.Empty in
  let groups =
    List.map surface.Autotune.ps_candidates ~f:(fun fc ->
        ( Tn.debug_name fc.LL.fc_tn,
          LL.reading_to_string fc.LL.fc_default,
          List.map fc.LL.fc_alternatives ~f:(fun fa -> LL.reading_to_string fa.LL.fa_flip) ))
  in
  Stdio.eprintf "surface (not part of the golden): %s\n%!"
    (String.concat ~sep:"; "
       (List.map groups ~f:(fun (tn, def, alts) ->
            Printf.sprintf "%s %s -> %s" tn def (String.concat ~sep:"," alts))));
  p
    "premise: the surface's only flip group is the footprint-scoped product's, with two \
     alternatives"
    (match surface.Autotune.ps_candidates with
    | [ fc ] ->
        Tn.equal fc.LL.fc_tn a.Tensor.value
        && LL.equal_reading fc.LL.fc_default `Footprint
        && List.length fc.LL.fc_alternatives = 2
    | _ -> false)

type call = {
  arm_reports : Autotune.report list;
  flip_reports : Autotune.report list;
  shipped : Train.shipped option;
  output : float array;
}

(* [~cut_after_flips:f]: the call's clock reads 0 until [f] flip searches have reported, then far
   past the [call_budget] given. *)
let run ?call_budget ?cut_after_flips ~script tag =
  let arm_reports = ref [] and flip_reports = ref [] and shipped = ref None in
  let clock _real =
    match cut_after_flips with Some f when List.length !flip_reports >= f -> 1e9 | _ -> 0.
  in
  let (ctx, routine), _record =
    Autotune.Seam.with_value Autotune.call_budget_elapsed clock @@ fun () ->
    Scripted_search.scripted
      ~measure:(fun _ -> script ~arm:(List.length !arm_reports))
      (fun () ->
        Train.tune_placements ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
          ~placement_store:false ~inline_flips:2 ?call_budget
          ~on_ship:(fun s -> shipped := Some s)
          ~report:(fun r -> arm_reports := r :: !arm_reports)
          ~flip_report:(fun r -> flip_reports := r :: !flip_reports)
          (Context.cpu ()) d comp Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  let output = Context.get_values ctx d.Tensor.value in
  Context.release ctx;
  let call =
    {
      arm_reports = List.rev !arm_reports;
      flip_reports = List.rev !flip_reports;
      shipped = !shipped;
      output;
    }
  in
  Stdio.eprintf "%s (not part of the golden): shipped %s; arm A timed %s, failed %s; %d flip(s)\n%!"
    tag
    (Option.value_map call.shipped ~default:"-" ~f:(fun s -> s.Train.ship_arm))
    (Option.value_map (List.hd call.arm_reports) ~default:"-" ~f:(fun r ->
         Int.to_string r.Autotune.candidates_timed))
    (Option.value_map (List.hd call.arm_reports) ~default:"-" ~f:(fun r ->
         Int.to_string r.Autotune.candidates_failed))
    (List.length call.flip_reports);
  call

let ship_arm call = Option.map call.shipped ~f:(fun s -> s.Train.ship_arm)

(* Arm A beats arm B; every flip beats both. *)
let script ~arm = match arm with 0 -> 4.0 | 1 -> 50.0 | _ -> 1.0

(* Leg 1, and leg 2's control: unbudgeted. *)
let () =
  let call = run ~script "unbudgeted" in
  p_bitwise "the search completes and ships values bit-identical to the untuned compile" call.output;
  p "both arms report, in position" (List.length call.arm_reports = 2);
  let arm_a = List.hd_exn call.arm_reports in
  (* Beside the baseline (the base compile itself), every candidate is rebased base code: the
     default seed, the candidate that asserted, is one. *)
  p "arm A timed a rebased candidate beside its baseline" (arm_a.Autotune.candidates_timed >= 2);
  p "no candidate of arm A failed" (arm_a.Autotune.candidates_failed = 0);
  p "control: both alternatives of the flip group are measured" (List.length call.flip_reports = 2);
  p "control: the group commits and a flip ships"
    (Option.equal String.equal (ship_arm call) (Some "flip"))

(* Leg 2: the group cut after its first alternative. *)
let () =
  let call = run ~call_budget:1000. ~cut_after_flips:1 ~script "cut" in
  p "cut: exactly one alternative of the group is measured" (List.length call.flip_reports = 1);
  p "cut: the group commits nothing -- arm A ships though the flip beat it"
    (Option.equal String.equal (ship_arm call) (Some "A"));
  p_bitwise "cut: the shipped values are bit-identical to the untuned compile" call.output

let () = Stdio.printf "tune_footprint_default: done\n"
