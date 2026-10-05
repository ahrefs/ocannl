(* CUDA's tensor-core cutoffs on both sides of every boundary (gh-ocannl-1214).

   Two tables own the compute-capability cutoffs of CUDA's tile-MMA support: the capability
   descriptor ([Cuda_mma.capability], the [entry ~min_cc] literals autotune seeds from) and the arm
   resolver ([Cuda_mma.mma_arm], whose [arm_floor] is the [wc_min_cc] / [mma16_min_cc] each emitting
   hook checks against the attached devices). A seed the descriptor admits but the hook declines is
   a candidate timed as scalar code under a tensorized label (gh-ocannl-545); the reverse withholds
   a rendering the device has. One fleet device (sm_120) sits above every cutoff and can vary
   neither side, so the agreement is checked here as pure functions of the compute capability.

   Nothing below names a cutoff. The descriptor's cutoffs are OBSERVED, as the capabilities where
   its record changes over a sweep, and the resolver's are its floors tabulated over every storage
   triple, emission scope and numerics policy. The two sets are compared, and then both owners are
   evaluated just below and at every cutoff of either: since neither side changes between its own
   cutoffs, those capabilities cover every interval. The descriptor is read through the consumer's
   own lookup ([Autotune.mma_tile_for_precisions_in_scope], the policy-resolved format triple and
   wide-accumulator scope gate autotune seeds with), so a storage triple counts as advertised
   exactly when a tensorized seed would be proposed for it. *)

open Base
open Verdict.Claims
module BI = Ir.Backend_intf
module N = Ir.Numerics
module M = Ir.Cuda_mma

(* The domain of the sweep: CUDA compute capabilities are [major * 10 + minor], and 0 is what the
   backend reports with no device attached. The top is a capability no cutoff will approach for
   years; nothing depends on its value but the sweep's reach. *)
let top_cc = 200
let ccs = List.range 0 top_cc ~stop:`inclusive

let policies =
  List.concat_map [ N.Fp16_auto; N.Fp16_narrow; N.Fp16_wide ] ~f:(fun fp16_arithmetic ->
      List.concat_map [ N.Bf16_auto; N.Bf16_narrow; N.Bf16_wide ] ~f:(fun bf16_arithmetic ->
          List.concat_map [ false; true ] ~f:(fun tf32_matmuls ->
              List.map [ false; true ] ~f:(fun narrow_compute_f32 ->
                  { N.tf32_matmuls; narrow_compute_f32; fp16_arithmetic; bf16_arithmetic }))))

let triples =
  List.concat_map Ir.Ops.all_precs ~f:(fun a_prec ->
      List.concat_map Ir.Ops.all_precs ~f:(fun b_prec ->
          List.map Ir.Ops.all_precs ~f:(fun d_prec -> (a_prec, b_prec, d_prec))))

let scopes = BI.all_of_mma_emission_scope

(* Both tables read the current numerics policy (the tf32 gate, the wide-accumulator modes), so each
   evaluation installs its policy first. *)
let under policy f =
  N.set_policy policy;
  f ()

(* The descriptor reads no policy, so one evaluation per capability serves every row. *)
let capabilities = Array.of_list_map ccs ~f:(fun cc -> M.capability ~cc)
let capability cc = capabilities.(cc)

(* The resolver's verdict at [cc]: the arm the hooks select for these storage precisions, if any,
   and whether the device meets its floor. A hook that declines an arm on capability falls through
   to the next table in its order (gh-ocannl-1153), and today no table it can fall through to has a
   lower floor, so the selected arm's floor is the hooks' whole capability gate. That ordering is
   [mma_arm]'s own premise, pinned against rendered CUDA by [schedule_mma_matmul]; it is assumed
   here, not re-derived. *)
let resolver_admits ~cc ~scope (a_prec, b_prec, d_prec) =
  match M.mma_arm ~a_prec ~b_prec ~d_prec ~scope with
  | None -> false
  | Some { BI.arm_floor; _ } -> Option.for_all arm_floor ~f:(fun floor -> cc >= floor)

let descriptor_admits ~cc ~scope (a_prec, b_prec, d_prec) =
  match capability cc with
  | None -> false
  | Some mma ->
      Option.is_some (Autotune.mma_tile_for_precisions_in_scope mma ~scope ~a_prec ~b_prec ~d_prec)

let descriptor_stages ~cc (a_prec, b_prec, d_prec) =
  match capability cc with
  | None -> false
  | Some mma ->
      List.exists (Autotune.mma_format_triples ~a_prec ~b_prec ~d_prec) ~f:(fun key ->
          List.Assoc.mem mma.BI.mma_staged_layouts key ~equal:BI.equal_mma_format_triple)

let show_triple (a, b, d) =
  String.concat ~sep:"*" (List.map [ a; b ] ~f:Ir.Ops.prec_string) ^ ">" ^ Ir.Ops.prec_string d

let show_policy policy = Sexp.to_string (N.sexp_of_t policy)
let dedup xs = List.dedup_and_sort xs ~compare:Int.compare
let ints xs = String.concat ~sep:" " (List.map xs ~f:Int.to_string)

let () =
  let descriptor_cutoffs =
    List.filter ccs ~f:(fun cc ->
        cc > 0 && not (Option.equal BI.equal_mma_capability (capability cc) (capability (cc - 1))))
  in
  let resolver_floors =
    List.concat_map policies ~f:(fun policy ->
        under policy (fun () ->
            List.concat_map scopes ~f:(fun scope ->
                List.filter_map triples ~f:(fun (a_prec, b_prec, d_prec) ->
                    Option.bind (M.mma_arm ~a_prec ~b_prec ~d_prec ~scope) ~f:(fun arm ->
                        arm.BI.arm_floor)))))
    |> dedup
  in
  Stdio.eprintf
    "descriptor cutoffs: %s; resolver floors: %s; cp.async floor: %d (not part of the golden)\n"
    (ints descriptor_cutoffs) (ints resolver_floors) M.async_copy_floor;
  p_all "every resolver floor is a cutoff of the descriptor" resolver_floors ~f:(fun floor ->
      List.mem descriptor_cutoffs floor ~equal:Int.equal);
  p_all "the descriptor changes only at a resolver floor or the cp.async floor" descriptor_cutoffs
    ~f:(fun cc -> List.mem (M.async_copy_floor :: resolver_floors) cc ~equal:Int.equal);
  (* Just below and at every cutoff of either owner, plus the ends of the domain. *)
  let probes =
    dedup
      (0 :: top_cc
       :: List.concat_map (descriptor_cutoffs @ resolver_floors) ~f:(fun c -> [ c - 1; c ])
      |> List.filter ~f:(fun cc -> cc >= 0 && cc <= top_cc))
  in
  let rows =
    List.concat_map policies ~f:(fun policy ->
        under policy (fun () ->
            List.concat_map probes ~f:(fun cc ->
                List.concat_map scopes ~f:(fun scope ->
                    List.map triples ~f:(fun triple ->
                        ( (policy, cc, scope, triple),
                          descriptor_admits ~cc ~scope triple,
                          resolver_admits ~cc ~scope triple ))))))
  in
  let report what ((policy, cc, scope, triple), descriptor, resolver) =
    Stdio.eprintf "%s: sm_%d %s %s under %s: descriptor %b, resolver %b\n" what cc
      (Sexp.to_string (BI.sexp_of_mma_emission_scope scope))
      (show_triple triple) (show_policy policy) descriptor resolver
  in
  p_all ~min:100_000
    "the descriptor admits a storage triple, scope and policy exactly where the resolver's arm \
     floor does, just below and at every cutoff"
    rows ~f:(fun ((_, descriptor, resolver) as row) ->
      Bool.equal descriptor resolver
      ||
      (report "disagreement" row;
       false));
  (* The controls: each floor is a boundary some admission actually crosses, so the agreement above
     was tested on both of its sides rather than on two identical ones. *)
  let admitted ~cc =
    List.filter_map rows ~f:(fun ((policy, at, scope, triple), descriptor, resolver) ->
        if at = cc && descriptor && resolver then
          Some
            (String.concat ~sep:" "
               [
                 show_policy policy;
                 Sexp.to_string (BI.sexp_of_mma_emission_scope scope);
                 show_triple triple;
               ])
        else None)
    |> Set.of_list (module String)
  in
  p_all "at every resolver floor some admission is refused just below it by both owners"
    resolver_floors ~f:(fun floor ->
      not (Set.is_empty (Set.diff (admitted ~cc:floor) (admitted ~cc:(floor - 1)))));
  (* Staged swizzled layouts are read by the inline-PTX register scope, i.e. by the fragment-scope
     arm: a triple the descriptor stages at all is staged exactly where that arm is admitted. *)
  let staged_rows =
    List.concat_map policies ~f:(fun policy ->
        under policy (fun () ->
            List.filter triples ~f:(descriptor_stages ~cc:top_cc)
            |> List.concat_map ~f:(fun triple ->
                List.map probes ~f:(fun cc ->
                    ( (policy, cc, BI.Mma_fragment_scope, triple),
                      descriptor_stages ~cc triple,
                      resolver_admits ~cc ~scope:BI.Mma_fragment_scope triple )))))
  in
  p_all ~min:100
    "a staged swizzled layout is advertised exactly where the fragment-scope arm is admitted"
    staged_rows ~f:(fun ((_, descriptor, resolver) as row) ->
      Bool.equal descriptor resolver
      ||
      (report "staged disagreement" row;
       false));
  p_all "pipelined depths are proposed exactly at and above the cp.async floor" probes ~f:(fun cc ->
      let depths =
        Option.value_map (capability cc) ~default:[] ~f:(fun mma -> mma.BI.mma_pipeline_depths)
      in
      Bool.equal (not (List.is_empty depths)) (cc >= M.async_copy_floor))
