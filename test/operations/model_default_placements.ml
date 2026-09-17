(* gh-ocannl-514: the placement levels of the untuned regime — [Autotune.model_default] under config
   [model_default_placements] = N > 0 branch-and-bounds over the top-N flip candidates of the
   decision surface before compiling, scoring each vector's hermetic lowering with the same
   selection that scores the pipelines, and applying the winning vector via the context-level
   placement decisions. The dune rule pins the cc backend, [model_default_placements=4] (the whole
   surface: the modeled recompute cost of gh-ocannl-637 ranks [y]'s one exp per read last, below the
   reductions that replay it), and a compute-bound envelope (peak_flops 1e9, peak_bandwidth 1e12).

   The graph makes a placement flip the model-argmin deterministically: [y = exp u] is read by two
   consumer statements yet stays policy-virtual, so both consumers replay the exp and the surface
   reports the [`Materialize] flip; materializing trades y's buffer traffic (one write, one read per
   consumer statement) for the 64 duplicated exp evaluations, which the compute-bound envelope
   prices as a strict win — the placement argmin reverses the greedy default in exactly the
   direction the recompute-cost bound cannot see. The pick must fire, the emitted label must carry
   the placement decision, and the executed values must match a plain compile (the
   structural-vs-executable rule: a placement pick that computed different values would be a
   miscompile, not a schedule choice). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
open Verdict.Claims

let approx a b = Float.(abs (a -. b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let m = 64

let () =
  let uv = Array.init m ~f:(fun i -> Float.of_int (i % 9) *. 0.25) in
  let u = TDSL.ndarray uv ~label:[ "u" ] ~output_dims:[ m ] () in
  let%op y = exp u in
  let%op s1 = relu y in
  let%op total = s1 ++ "i => 0" + (y ++ "i => 0") in
  ignore (y, s1);
  let comp = named "mdp" (Train.forward total) in
  (let module LL = Ir.Low_level in
   let surface = Autotune.placement_surface (Context.auto ()) comp Ir.Indexing.Empty in
   Stdio.printf "decision surface:\n";
   List.iter surface.Autotune.ps_candidates ~f:(fun fc ->
       Stdio.printf "  %-11s %-8s cost %d\n"
         (match fc.LL.fc_flip with `Materialize -> "materialize" | `Inline -> "inline")
         (Ir.Tnode.debug_name fc.LL.fc_tn) fc.LL.fc_recompute_cost));
  (* Reference values from a plain compile. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref total.Tensor.value in
  let choice = ref None in
  let ctx, routine =
    Autotune.model_default
      ~report:(fun r -> choice := Some r)
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx total.Tensor.value in
  p_all2 "model_default with placement levels returns a routine with correct values" got expected
    ~f:approx;
  match !choice with
  | None -> Stdio.printf "expected a model_choice report\n"
  | Some r ->
      p "selection ran (scored the default pipeline and the placement leaves)"
        (r.Autotune.mc_scored >= 2);
      p "the placement pick fired and the label carries it"
        (String.is_prefix r.Autotune.mc_label ~prefix:"placements[");
      p "the pick materializes y" (String.is_substring r.Autotune.mc_label ~substring:"mat:exp_y")
