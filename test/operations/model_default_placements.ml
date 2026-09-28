(* gh-ocannl-514: the placement levels of the untuned regime — [Autotune.model_default] under config
   [model_default_placements] = N > 0 branch-and-bounds over the top-N flip candidates of the
   decision surface before compiling, scoring each vector's hermetic lowering with the same
   selection that scores the pipelines, and applying the winning vector via the context-level
   placement decisions. The dune rule pins the cc backend, [model_default_placements=2], and a
   compute-bound envelope (peak_flops 1e9, peak_bandwidth 1e12).

   gh-ocannl-1093: the lowered surface also carries the two scalar reductions' [`Inline] flips,
   which the virtualizer refuses (their operand reads escape the setter a store captures) and which
   therefore carry the traced proxy, the 64-cell extent — priced above [y]'s modeled one exp per
   read. They are marked refused ([fa_refused], the virtualizer's own verdict, checked here against
   a compile that prefers them inline) and never ranked, so a cut of 2 is the whole ranked surface:
   before, the two refused flips took both of its slots and the pick could not reach [y].

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
   let module Tn = Ir.Tnode in
   let show title candidates =
     Stdio.printf "%s:\n" title;
     List.iter candidates ~f:(fun fc ->
         Stdio.printf "  %-8s %-11s -> %s\n" (Tn.debug_name fc.LL.fc_tn)
           (LL.reading_to_string fc.LL.fc_default)
           (String.concat ~sep:", "
              (List.map fc.LL.fc_alternatives ~f:(fun fa ->
                   Printf.sprintf "%s cost %d%s"
                     (LL.reading_to_string fa.LL.fa_flip)
                     fa.LL.fa_recompute_cost
                     (Option.value_map fa.LL.fa_refused ~default:"" ~f:(fun code ->
                          " refused " ^ code))))))
   in
   let lowered = Context.decision_surface (Context.auto ()) comp Ir.Indexing.Empty in
   show "decision surface (as lowered)" lowered;
   let surface = Autotune.placement_surface (Context.auto ()) comp Ir.Indexing.Empty in
   show "ranked surface" surface.Autotune.ps_candidates;
   let flips cands =
     List.concat_map cands ~f:(fun fc -> List.map fc.LL.fc_alternatives ~f:(fun fa -> (fc, fa)))
   in
   let refused = List.filter (flips lowered) ~f:(fun (_, fa) -> Option.is_some fa.LL.fa_refused) in
   p "the lowered surface carries the two scalar reductions' refused flips" (List.length refused = 2);
   p_all "each refused flip is an Inline flip" refused ~f:(fun (_, fa) ->
       LL.equal_reading fa.LL.fa_flip `Inline);
   (* The refusal is the virtualizer's verdict: preferring the flip inline, the compile still
      materializes the node, under the provenance the alternative carries. *)
   p_all "each refusal is the virtualizer's verdict on the flip preferred inline" refused
     ~f:(fun (fc, fa) ->
       let o =
         Context.lowered_for_decisions ~inline:[ fc.LL.fc_tn ] (Context.auto ()) comp
           Ir.Indexing.Empty
       in
       Tn.Placements.known_non_virtual o.LL.optimize_ctx.placements fc.LL.fc_tn
       && Option.equal String.equal fa.LL.fa_refused
            (Option.map (Tn.Placements.get o.LL.optimize_ctx.placements fc.LL.fc_tn)
               ~f:(fun (_, prov) -> Tn.provenance_to_string (Tn.leading_provenance prov))));
   p_none "no flip on the ranked surface is refused" (flips surface.Autotune.ps_candidates)
     ~f:(fun (_, fa) -> Option.is_some fa.LL.fa_refused);
   p "the ranked surface keeps every flip not refused"
     (List.length (flips surface.Autotune.ps_candidates)
     = List.length (flips lowered) - List.length refused);
   p "the ranked surface drops the nodes whose only flip is refused"
     (List.length surface.Autotune.ps_candidates
     = List.count lowered ~f:(fun fc ->
         List.exists fc.LL.fc_alternatives ~f:(fun fa -> Option.is_none fa.LL.fa_refused))));
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
