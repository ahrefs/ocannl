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

let approx_tol _ = 1e-4

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
  p_close "model_default with placement levels returns a routine with correct values" got expected
    ~tol:approx_tol;
  match !choice with
  | None -> Stdio.printf "expected a model_choice report\n"
  | Some r ->
      p "selection ran (scored the default pipeline and the placement leaves)"
        (r.Autotune.mc_scored >= 2);
      p "the placement pick fired and the label carries it"
        (String.is_prefix r.Autotune.mc_label ~prefix:"placements[");
      p "the pick materializes y" (String.is_substring r.Autotune.mc_label ~substring:"mat:exp_y")

(* gh-ocannl-1390: the pick's fallback recompiles from the default placements on a CLASSIFIED
   rejection only. It used to catch every [Invalid_argument] out of the compile under the picked
   placements, so a transform bug there -- fatal under [strict_failure_classification], the default
   -- was retried from the default placements and the routine came back as though nothing had
   happened. The injection is [Schedule.with_transform_fault], firing at the LAST op application a
   clean run makes: the placement search scores hermetic lowerings first and the compile under the
   picked placements applies its pipeline last, so that call is inside the picked compile (the pick
   must fire, as above). A clean run's count is checked stable first, since the injected run is
   located by it. The graph keeps the flip -- [y = exp u] read twice, now within one elementwise
   statement -- at 16384 cells, each costing at least a store, so it reaches
   [cpu_schedule_min_work]'s default and the default CPU pipeline is not empty: it retypes the
   parallel loop to [Grid], which is what gives the compile op applications to fault (the graph
   above, whose consumers are scalar reductions, applies none). *)
let () =
  let m = 16384 in
  let uv = Array.init m ~f:(fun i -> Float.of_int (i % 9) *. 0.25) in
  let u = TDSL.ndarray uv ~label:[ "u_fault" ] ~output_dims:[ m ] () in
  let%op y = exp u in
  let%op total = relu y + y in
  ignore y;
  let comp = named "mdp_fault" (Train.forward total) in
  let picked = ref false in
  let run fault =
    Ir.Schedule.with_transform_fault fault (fun () ->
        Autotune.model_default
          ~report:(fun r -> picked := String.is_prefix r.Autotune.mc_label ~prefix:"placements[")
          (Context.auto ()) comp Ir.Indexing.Empty)
  in
  let count () =
    let n = ref 0 in
    ignore (run (fun _ -> Int.incr n) : Context.t * Context.routine);
    !n
  in
  let calls = count () in
  let picked_clean = !picked in
  Stdio.eprintf "clean run: %d op application(s), pick fired: %b (not part of the golden)\n" calls
    picked_clean;
  p "a clean run's op applications are counted, and the pick fires there" (calls > 0 && picked_clean);
  p "the count is stable across clean runs" (count () = calls);
  let injected = "injected transform bug in the picked compile (gh-ocannl-1390)" in
  let n = ref 0 in
  let fired = ref false in
  let outcome =
    match
      run (fun _ ->
          Int.incr n;
          if !n = calls then (
            fired := true;
            invalid_arg injected))
    with
    | (_ : Context.t * Context.routine) -> `Returned
    | exception Invalid_argument detail when String.equal detail injected -> `Propagated
    | exception exn -> `Other (Exn.to_string exn)
  in
  p "the injected transform bug fired" !fired;
  p
    "an untyped transform exception in the picked compile propagates, never retried from the \
     default placements"
    (match outcome with
    | `Propagated -> true
    | `Returned -> false
    | `Other e ->
        Stdio.eprintf "unexpected exception (not part of the golden): %s\n" e;
        false)
