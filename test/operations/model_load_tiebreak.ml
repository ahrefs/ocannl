(* gh-ocannl-1254: among candidates whose ranking scores tie exactly, fewer operand loads win.

   The plain and [Swizzle_b128] staged fp8 twins of gh-ocannl-1190 touch the same cells with the
   same operations, so the ranking score ([Cost_model.ranking_seconds]) ties them, and the family
   search's first-strictly-better rule kept the plain twin listed first, though on rog's sm_120 the
   swizzled one ran 12% faster, its A operand coming in through [ldmatrix] instead of byte gathers.
   The tie-break ([Schedule_space.search ~tie_break], [Autotune.model_family_search]) orders exact
   ties by [Autotune.model_operand_loads] under the backend's [mma_operand_loads] rule: no priced
   constant, no change to the twins' order. Everything here compiles nothing.

   The search's tie semantics, on hand-built trees: a tied leaf with a smaller key displaces the
   best leaf; equal keys keep the first; a tie never displaces the incumbent; keys are computed only
   for ties; a subtree whose bound equals the threshold is entered once a leaf holds it.

   The twins at the gh-ocannl-1190 shape (an 8192x32x4096 fp8 matmul, f32 accumulator), scheduled
   under a CUDA-shaped [hardware_limits] record whose tile-MMA descriptor is [Ir.Cuda_mma.capability
   ~cc:120]: equal flops, bytes and ranking score, and fewer loads for the swizzled twin under
   CUDA's own rule ([Ir.Cuda_mma.operand_loads], pure, so it stands in off CUDA), equal loads under
   a rule without [ldmatrix], which is every other backend's.

   [Autotune.model_family_search] over that site: the pick has the fewest loads among the leaves
   tied with it under either rule, and both picks score alike. The site's argmin is an unstaged leaf
   with no twin, so the walk also runs over the staged leaves alone, where it crowns the swizzled
   twin under CUDA's rule and keeps the plain one without [ldmatrix].

   The same claims are made over the backend's own record and rule where it advertises the staged
   fp8 layout (CUDA from sm_89), and [Autotune.model_explain]'s shape on every backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module CM = Ir.Cost_model
module Sspace = Ir.Schedule_space
module BI = Ir.Backend_intf
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* {1 The search's tie semantics} *)

let leaf_tree leaves =
  Sspace.Choice
    {
      level = "leaf";
      children =
        List.map leaves ~f:(fun (name, _, _) -> (name, Sspace.Child (lazy (Sspace.Leaf name))));
    }

let () =
  let keys_asked = ref [] in
  let pick ?incumbent ?tie_break leaves =
    let find name = List.find_exn leaves ~f:(fun (n, _, _) -> String.equal n name) in
    let score name = (fun (_, s, _) -> Some s) (find name) in
    let tie_break =
      Option.map tie_break ~f:(fun () name ->
          keys_asked := name :: !keys_asked;
          (fun (_, _, k) -> Some k) (find name))
    in
    Option.map (fst (Sspace.search ?incumbent ?tie_break ~score (leaf_tree leaves))) ~f:fst
  in
  let leaves = [ ("a", 2., 5); ("b", 2., 3); ("c", 2., 3); ("d", 3., 0) ] in
  p "without a tie-break the first of the tied leaves wins"
    (Option.equal String.equal (pick leaves) (Some "a"));
  p "with a tie-break the tied leaf with the smallest key wins, the first among equal keys"
    (Option.equal String.equal (pick ~tie_break:() leaves) (Some "b"));
  p "a tie never displaces the incumbent" (Option.is_none (pick ~incumbent:2. ~tie_break:() leaves));
  keys_asked := [];
  let untied = [ ("a", 3., 5); ("b", 2., 3); ("d", 1., 0) ] in
  ignore (pick ~tie_break:() untied : string option);
  p_empty "keys are computed only for leaves that tie" ~over:untied !keys_asked;
  keys_asked := [];
  ignore (pick ~tie_break:() [ ("a", 2., 5); ("b", 2., 3); ("c", 2., 4) ] : string option);
  p "each best leaf's key is computed once"
    (List.equal String.equal (List.sort !keys_asked ~compare:String.compare) [ "a"; "b"; "c" ]);
  (* A leaf the key does not apply to (a candidate with no tile-MMA site: its operands load by paths
     the count does not see) is no zero: it neither displaces a tied leaf nor is displaced. *)
  let partial = [ ("s", 2., None); ("t", 2., Some 3); ("u", 2., Some 1) ] in
  let pick_partial order =
    let leaves =
      List.map order ~f:(fun n -> List.find_exn partial ~f:(fun (m, _, _) -> String.equal n m))
    in
    let find name = List.find_exn leaves ~f:(fun (n, _, _) -> String.equal n name) in
    Option.map
      (fst
         (Sspace.search
            ~tie_break:(fun name -> (fun (_, _, k) -> k) (find name))
            ~score:(fun name -> (fun (_, s, _) -> Some s) (find name))
            (leaf_tree leaves)))
      ~f:fst
  in
  p "a keyless leaf listed first keeps a tie against keyed leaves"
    (Option.equal String.equal (pick_partial [ "s"; "t"; "u" ]) (Some "s"));
  p "a keyless leaf listed later never displaces a keyed best"
    (Option.equal String.equal (pick_partial [ "t"; "s"; "u" ]) (Some "u"));
  (* A subtree bounded exactly at the threshold: fathomed by the strict rule, entered under a
     tie-break once a leaf holds the threshold, since a completion at the bound could win the
     tie. *)
  let nested =
    Sspace.Choice
      {
        level = "outer";
        children =
          [
            ("x", Sspace.Child (lazy (Sspace.Leaf "x")));
            ("y", Sspace.Child (lazy (Sspace.Leaf "y")));
          ];
      }
  in
  let score2 = function "x" | "y" -> Some 2. | _ -> None in
  let key2 = function "x" -> Some 5 | _ -> Some 1 in
  let bound ~path _ =
    match List.last path with Some (_, "y") -> Some 2. | Some _ | None -> None
  in
  let pick2 ?tie_break () =
    Option.map (fst (Sspace.search ~bound ?tie_break ~score:score2 nested)) ~f:fst
  in
  p "without a tie-break a subtree bounded at the threshold is fathomed"
    (Option.equal String.equal (pick2 ()) (Some "x"));
  p "with a tie-break a subtree bounded at the threshold is entered, and its tie wins"
    (Option.equal String.equal (pick2 ~tie_break:key2 ()) (Some "y"));
  (* A keyless best leaf can lose no tie, so the subtree bounded at its score is fathomed unscored,
     as under the strict rule. *)
  let _, stats = Sspace.search ~bound ~tie_break:(fun _ -> None) ~score:score2 nested in
  p "under a keyless best leaf a subtree bounded at the threshold is fathomed unscored"
    (stats.Sspace.st_scored = 1 && stats.Sspace.st_fathomed = 1)

(* A candidate's kernels combine into one count: an undescribed kernel makes the whole unknown (a
   partial total would read as complete), kernels with no tile-MMA site add nothing. *)
let () =
  let combine = CM.combine_load_counts in
  p "an undescribed kernel makes a candidate's count unknown"
    (CM.equal_load_count (combine [ CM.Loads 3; CM.Undescribed; CM.No_tile_mma ]) CM.Undescribed);
  p "kernels without tile-MMA sites add nothing to the counted ones"
    (CM.equal_load_count (combine [ CM.No_tile_mma; CM.Loads 2; CM.Loads 5 ]) (CM.Loads 7));
  p "a candidate with no tile-MMA site anywhere has nothing to count"
    (CM.equal_load_count (combine [ CM.No_tile_mma; CM.No_tile_mma ]) CM.No_tile_mma)

(* {1 The gh-ocannl-1190 twins} *)

let nm = 8192
and nn = 32
and nk = 4096

let node = Ll_test.node_factory ~prec:Ir.Ops.fp8 ~first_id:125400 ~dims:[| 1 |] ()
let a = node ~dims:[| nm; nk |] "mlt_a"
let b = node ~dims:[| nk; nn |] "mlt_b"
let c = Ll_test.node_factory ~prec:Ir.Ops.single ~first_id:125410 ~dims:[| nm; nn |] () "mlt_c"
let () = List.iter [ a; b; c ] ~f:Ll_test.materialize
let it = Ll_test.iter

(* [c[i,j] += a[i,k] * b[k,j]], the contraction innermost, as the matmul detection reads it. *)
let matmul () =
  let i = Ll_test.sym () and j = Ll_test.sym () and k = Ll_test.sym () in
  Ll_test.loop_n i nm
    (Ll_test.loop_n j nn
       (Ll_test.loop_n k nk
          (Ll_test.set c
             [| it i; it j |]
             (LL.Ternop
                ( Ir.Ops.FMA,
                  (Ll_test.get a [| it i; it k |], Ll_test.single),
                  (Ll_test.get b [| it k; it j |], Ll_test.single),
                  (Ll_test.get c [| it i; it j |], Ll_test.single) )))))

(* rog-nv's RTX 5070 Ti as CUDA's backend describes it: the queried caps, [Cuda_mma.capability] at
   its compute capability, and the advisory envelope, capacity and cache tier of
   [Cuda_backend.hardware_limits]. A stand-in, so the claims run on every backend; on a CUDA device
   that advertises the staged fp8 layout the same claims are made again over the backend's own
   record and load rule. *)
let stand_in =
  {
    BI.no_hardware_limits with
    max_threads_per_workgroup = Some 1024;
    max_workgroup_memory_bytes = Some 49152;
    max_workgroup_dims = Some (1024, 1024, 64);
    max_grid_yz = Some 65535;
    simdgroup_width = Some 32;
    mma = Ir.Cuda_mma.capability ~cc:120;
    peak_flops = Some 1.5e13;
    peak_memory_bandwidth = Some 8.0e12;
    cache_capacity_bytes = Some (12 * 1024 * 1024 * 1024);
    cache_bandwidth = Some 1.8e12;
    max_resident_threads = Some (70 * 1536);
  }

let base = Ll_test.optimize ~name:"mlt_mm" (matmul ())

let hermetic (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* The tile-MMA work the backend's hook accepts: what the score prices at the matrix units' rate
   (gh-ocannl-1442), the rest at the scalar issue rate. *)
let accepted (capabilities : BI.codegen_capabilities) o =
  CM.accepted_tile_mma_flops ~operand_accepts:capabilities.mma_accepts o

(* The ranking score [Autotune.model_default] minimizes, under a record's own constants and load
   rule. *)
let rank ~capabilities (limits : BI.hardware_limits) (o : LL.optimized) =
  CM.ranking_seconds ?peak_flops:limits.peak_flops
    ?peak_memory_bandwidth:limits.peak_memory_bandwidth
    ?cache_capacity_bytes:limits.cache_capacity_bytes ?cache_bandwidth:limits.cache_bandwidth
    ?issue_rate:limits.issue_rate ?dram_bandwidth:limits.dram_bandwidth
    ~mma_flops:(accepted capabilities o) ?max_resident_threads:limits.max_resident_threads
    ~is_global:(CM.global_in o) o.LL.llc

(* The two legs of one kernel's score, under the same constants. *)
let legs_with ~capabilities (limits : BI.hardware_limits) ?(issue_rate = limits.issue_rate)
    (o : LL.optimized) =
  let r =
    CM.reuse_traffic ?max_resident_threads:limits.max_resident_threads ~is_global:(CM.global_in o)
      o.LL.llc
  in
  CM.ranking_legs_of ?peak_flops:limits.peak_flops
    ?peak_memory_bandwidth:limits.peak_memory_bandwidth
    ?cache_capacity_bytes:limits.cache_capacity_bytes ?cache_bandwidth:limits.cache_bandwidth
    ?issue_rate ?dram_bandwidth:limits.dram_bandwidth ~mma_flops:(accepted capabilities o)
    (CM.analyze o.LL.llc) (Some r)

let score ~capabilities limits segs =
  List.fold segs ~init:(Some 0.) ~f:(fun acc o ->
      Option.map2 acc (rank ~capabilities limits o) ~f:Float.( + ))

let no_ldmatrix_rule ~a_prec:_ ~b_prec:_ ~d_prec:_ ~scope:_ ~ta:_ ~tb:_ ~a:_ ~b:_ = (false, false)

let schedule p =
  Ir.Schedule.apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p (hermetic base)) (hermetic base)

let is_staged_plain_twin (p : Autotune.sketch_params) =
  p.Autotune.sk_mma && Option.is_none p.Autotune.sk_swizzle

let swizzled_twin (p : Autotune.sketch_params) =
  { p with Autotune.sk_swizzle = Some LL.Swizzle_b128 }

(* The claims over one record and its [ldmatrix] rule, each label prefixed by [tag]. [report]
   evaluates a claim, or skips it where the record is not this run's to check. *)
let site_claims ~tag ~(limits : BI.hardware_limits) ~(capabilities : BI.codegen_capabilities)
    ~evaluated =
  let rule = capabilities in
  (* Scored under the record's own hook, as [model_default] scores them. *)
  let rank = rank ~capabilities:rule
  and legs = legs_with ~capabilities:rule
  and score = score ~capabilities:rule in
  let no_ldmatrix = { capabilities with mma_operand_loads = no_ldmatrix_rule } in
  let label s = tag ^ ": " ^ s in
  let loads capabilities o =
    match Autotune.model_operand_loads ~capabilities o with
    | CM.Loads n -> Some n
    | CM.No_tile_mma | CM.Undescribed -> None
  in
  let show_loads = Option.value_map ~default:"no tile-MMA" ~f:Int.to_string in
  let fewer a b = match (a, b) with Some a, Some b -> a < b | _ -> false in
  let claim s b =
    if evaluated then p (label s) (Lazy.force b) else skipped ~backend:backend_name (label s)
  in
  let seeds =
    if evaluated then Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits base else []
  in
  let has p = List.mem seeds p ~equal:Poly.equal in
  (* The twin pairs the family seeds: a plain staged tensorized leaf and its swizzled sibling. *)
  let twins =
    List.filter_map seeds ~f:(fun p ->
        if is_staged_plain_twin p && has (swizzled_twin p) then
          Some (p, schedule p, schedule (swizzled_twin p))
        else None)
  in
  List.iter twins ~f:(fun (p, plain, swz) ->
      Stdio.eprintf
        "%s: twin %s: rank %.17g, loads %s plain vs %s swizzled (not part of the golden)\n" tag
        (Autotune.sketch_params_label p)
        (Option.value (rank limits plain) ~default:Float.nan)
        (show_loads (loads rule plain))
        (show_loads (loads rule swz)));
  let all s f = claim s (lazy ((not (List.is_empty twins)) && List.for_all twins ~f)) in
  (* gh-ocannl-1442: the tile-MMA work the hook accepts stays at the [peak_flops] ceiling, so a
     slower scalar issue rate moves only the rest of a tensorized leaf's compute leg; work the hook
     declines renders its scalar fallback and is priced as scalar work. *)
  let tensorized =
    List.filter_map seeds ~f:(fun p ->
        if p.Autotune.sk_mma then Option.try_with (fun () -> schedule p) else None)
  in
  let decline_all =
    {
      capabilities with
      mma_accepts =
        (fun ~d_prec:_
          ~a_prec:_
          ~b_prec:_
          ~ta:_
          ~tb:_
          ~m:_
          ~n:_
          ~k:_
          ~fragment_target:_
          ~d:_
          ~a:_
          ~b:_
        -> false);
    }
  in
  let peak = Option.value limits.peak_flops ~default:1. in
  let slows_by ~capabilities o =
    match
      ( fst (legs_with ~capabilities limits ~issue_rate:(Some (peak /. 10.)) o),
        fst (legs_with ~capabilities limits ~issue_rate:None o) )
    with
    | Some slow, Some fast -> Some (slow /. fast)
    | _ -> None
  in
  claim "every tensorized leaf has tile-MMA work the hook accepts, within its operations"
    (lazy
      ((not (List.is_empty tensorized))
      && List.for_all tensorized ~f:(fun o ->
          let a = accepted rule o in
          a > 0 && a <= (CM.analyze o.LL.llc).CM.flops)));
  claim "a tenfold slower issue rate slows each tensorized leaf's compute leg less than tenfold"
    (lazy
      ((not (List.is_empty tensorized))
      && List.for_all tensorized ~f:(fun o ->
          Option.exists (slows_by ~capabilities:rule o) ~f:(fun r -> Float.(r < 10. -. 1e-9)))));
  claim
    "under a hook that declines every statement, no work is accepted and the whole compute leg \
     slows tenfold"
    (lazy
      ((not (List.is_empty tensorized))
      && List.for_all tensorized ~f:(fun o ->
          accepted decline_all o = 0
          && Option.exists (slows_by ~capabilities:decline_all o) ~f:(fun r ->
              Float.(abs (r -. 10.) < 1e-9)))));
  let same f (_, plain, swz) = f plain = f swz in
  all "the gh-ocannl-1190 site seeds staged fp8 twins, and each pair has equal flops"
    (same (fun o -> (CM.analyze o.LL.llc).CM.flops));
  all "each pair has equal bytes" (same (fun o -> CM.total_bytes (CM.analyze o.LL.llc)));
  all "each pair has exactly equal ranking scores" (fun (_, plain, swz) ->
      match (rank limits plain, rank limits swz) with
      | Some a, Some b -> Float.equal a b
      | _ -> false);
  all "under CUDA's load rule each swizzled twin issues fewer operand loads" (fun (_, plain, swz) ->
      fewer (loads rule swz) (loads rule plain));
  all "under a rule without ldmatrix each pair issues equal operand loads" (fun (_, plain, swz) ->
      Option.is_some (loads no_ldmatrix swz)
      && Option.equal Int.equal (loads no_ldmatrix swz) (loads no_ldmatrix plain));
  (* How often a fragment loads is the arm's: CUDA's inline-PTX fp8 arm reloads each operand for
     every tile of the other's free axis, so on a statement spanning several intrinsic tiles a
     reloading arm issues more loads than one reusing its fragments, as wmma does. *)
  let reusing =
    {
      capabilities with
      mma_operand_fragments =
        (fun ~a_prec ~b_prec ~d_prec ~scope ->
          Option.map (capabilities.mma_operand_fragments ~a_prec ~b_prec ~d_prec ~scope)
            ~f:(fun f -> { f with BI.mf_reloads = false }));
    }
  in
  all "the arm's reloads count: a reusing arm issues fewer loads than the record's own"
    (fun (_, plain, _) -> fewer (loads reusing plain) (loads rule plain));
  (* A staged twin's accumulator is a persistent fragment (gh-ocannl-1063): the hook's update-only
     call in the fragment scope when the hook accepts it there, a statement of its own otherwise.
     With reloads that differ by scope, the count shows which scope it read. *)
  let per_statement_reloads =
    {
      capabilities with
      mma_operand_fragments =
        (fun ~a_prec ~b_prec ~d_prec ~scope ->
          Option.map (capabilities.mma_operand_fragments ~a_prec ~b_prec ~d_prec ~scope)
            ~f:(fun f ->
              { f with BI.mf_reloads = BI.equal_mma_emission_scope scope BI.Mma_per_statement }));
    }
  in
  let per_statement (o : LL.optimized) =
    { o with LL.simdgroup_fragments = Set.empty (module Ir.Tnode) }
  in
  let accepting ~fragment =
    {
      per_statement_reloads with
      mma_accepts =
        (fun ~d_prec:_
          ~a_prec:_
          ~b_prec:_
          ~ta:_
          ~tb:_
          ~m:_
          ~n:_
          ~k:_
          ~fragment_target
          ~d:_
          ~a:_
          ~b:_
        -> Option.is_none fragment_target || fragment);
    }
  in
  let declining =
    {
      capabilities with
      mma_accepts =
        (fun ~d_prec:_
          ~a_prec:_
          ~b_prec:_
          ~ta:_
          ~tb:_
          ~m:_
          ~n:_
          ~k:_
          ~fragment_target:_
          ~d:_
          ~a:_
          ~b:_
        -> false);
    }
  in
  (* An element-granularity swizzle has no intrinsic load form: the renderer declines it before any
     hook, so the count does too, whatever the hook would say. *)
  all "an element-swizzled twin has no count" (fun (p, _, _) ->
      match schedule { p with Autotune.sk_swizzle = Some LL.Swizzle_elem } with
      | o -> CM.equal_load_count (Autotune.model_operand_loads ~capabilities:rule o) CM.Undescribed
      | exception _ -> false);
  all "a statement the hook declines has no count" (fun (_, plain, swz) ->
      CM.equal_load_count
        (Autotune.model_operand_loads ~capabilities:declining plain)
        CM.Undescribed
      && CM.equal_load_count
           (Autotune.model_operand_loads ~capabilities:declining swz)
           CM.Undescribed);
  all "a marked statement the hook accepts as a fragment update is counted in the fragment scope"
    (fun (_, plain, _) ->
      (not (Set.is_empty plain.LL.simdgroup_fragments))
      && fewer
           (loads (accepting ~fragment:true) plain)
           (loads (accepting ~fragment:true) (per_statement plain)));
  all "a marked statement the hook declines as a fragment update is counted per statement"
    (fun (_, plain, _) ->
      Option.is_some (loads (accepting ~fragment:false) plain)
      && Option.equal Int.equal
           (loads (accepting ~fragment:false) plain)
           (loads (accepting ~fragment:false) (per_statement plain)));
  (* The family search over the site, under each rule. Every leaf it could score, scored the way it
     scores them, is the population its pick is checked against. *)
  let scored =
    List.filter_map seeds ~f:(fun p ->
        match schedule p with
        | o -> Option.map (score limits [ o ]) ~f:(fun s -> (p, o, s))
        | exception _ -> None)
  in
  (* The leaves the score ranks lowest, with their legs: what decides the site's argmin. *)
  if evaluated then
    List.sort scored ~compare:(fun (_, _, a) (_, _, b) -> Float.compare a b)
    |> Fn.flip List.take 8
    |> List.iter ~f:(fun (p, o, s) ->
        let ms = Option.value_map ~default:"n/a" ~f:(fun x -> Printf.sprintf "%.6f" (x *. 1e3)) in
        let c, m = legs limits o in
        Stdio.eprintf
          "%s: leaf %s%s: rank_ms %.6f compute_ms %s memory_ms %s (not part of the golden)\n" tag
          (Autotune.sketch_params_label p)
          (if Set.is_empty o.LL.workgroup_shared then "" else " [staged]")
          (s *. 1e3) (ms c) (ms m));
  let pick rule =
    if not evaluated then None
    else
      match
        Autotune.model_family_search ~capabilities:rule ~is_gpu:true ~is_cpu:false ~limits
          ~static_indices:[] ~peak_flops:limits.peak_flops
          ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score:(score limits)
          ~incumbent:Float.infinity base
      with
      | Some (Some (p, s), _, _) -> Some (p, schedule p, s)
      | Some (None, _, _) | None -> None
  in
  let cuda_pick = pick rule and plain_pick = pick no_ldmatrix in
  let show = function
    | Some (p, o, s) ->
        Printf.sprintf "%s @ %.17g, %s loads" (Autotune.sketch_params_label p) s
          (show_loads (loads rule o))
    | None -> "none"
  in
  if evaluated then
    Stdio.eprintf "%s: pick under CUDA's rule: %s; without ldmatrix: %s (not part of the golden)\n"
      tag (show cuda_pick) (show plain_pick);
  let tied_with (_, _, s) = List.filter scored ~f:(fun (_, _, s') -> Float.equal s s') in
  let minimizes rule pick =
    match pick with
    | None -> false
    | Some ((_, o, _) as pk) ->
        (* Leaves without a count are outside the comparison, as in the walk. *)
        List.for_all (tied_with pk) ~f:(fun (_, o', _) ->
            not (fewer (loads rule o') (loads rule o)))
  in
  claim "under CUDA's rule the pick issues the fewest operand loads among the leaves tied with it"
    (lazy (minimizes rule cuda_pick));
  claim "without ldmatrix the pick issues the fewest operand loads among the leaves tied with it"
    (lazy (minimizes no_ldmatrix plain_pick));
  claim "the tie-break moves no score: both rules' picks score exactly alike"
    (lazy
      (match (cuda_pick, plain_pick) with
      | Some (_, _, a), Some (_, _, b) -> Float.equal a b
      | _ -> false));
  (* At this site the argmin is an unstaged leaf, which has no twin. Restricted to the staged leaves
     — the others left unscored, which the walk never crowns — the argmin is a tied twin pair, and
     the loads decide it: the swizzled twin under CUDA's rule, the first-listed plain twin under a
     rule without ldmatrix. *)
  let staged_pick rule =
    if not evaluated then None
    else
      let staged segs =
        if List.for_all segs ~f:(fun (o : LL.optimized) -> Set.is_empty o.LL.workgroup_shared) then
          None
        else score limits segs
      in
      match
        Autotune.model_family_search ~capabilities:rule ~is_gpu:true ~is_cpu:false ~limits
          ~static_indices:[] ~peak_flops:limits.peak_flops
          ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score:staged
          ~incumbent:Float.infinity base
      with
      | Some (Some (p, _), _, _) -> Some p
      | Some (None, _, _) | None -> None
  in
  let swizzled (p : Autotune.sketch_params) = Option.is_some p.Autotune.sk_swizzle in
  claim "among the staged leaves, under CUDA's rule the family search picks a swizzled twin"
    (lazy (Option.exists (staged_pick rule) ~f:swizzled));
  claim "among the staged leaves, without ldmatrix it keeps the first-listed plain twin"
    (lazy (Option.exists (staged_pick no_ldmatrix) ~f:(fun p -> not (swizzled p))))

let stand_in_capabilities =
  {
    BI.no_codegen_capabilities with
    mma_operand_loads = Ir.Cuda_mma.operand_loads;
    mma_operand_fragments = Ir.Cuda_mma.operand_fragments;
    (* CUDA's acceptance is its hook's, which needs the backend; the stand-in accepts every
       statement, and the backend-record leg reads the real hook. *)
    mma_accepts =
      (fun ~d_prec:_
        ~a_prec:_
        ~b_prec:_
        ~ta:_
        ~tb:_
        ~m:_
        ~n:_
        ~k:_
        ~fragment_target:_
        ~d:_
        ~a:_
        ~b:_
      -> true);
  }

let () =
  site_claims ~tag:"stand-in" ~limits:stand_in ~capabilities:stand_in_capabilities ~evaluated:true

(* gh-ocannl-1442: the same site under the ranking score's own rates, rog's measured DRAM rate and
   an issue rate in the range the seeded kernel achieved there: the twins still tie exactly, and the
   loads still decide among them. *)
let () =
  site_claims ~tag:"rated stand-in"
    ~limits:{ stand_in with issue_rate = Some 3.4e12; dram_bandwidth = Some 5.82e11 }
    ~capabilities:stand_in_capabilities ~evaluated:true

(* The backend's own record and rule, where they describe the staged fp8 twins: CUDA from sm_89. *)
let () =
  let ctx = Context.auto () in
  let limits = Context.hardware_limits ctx in
  let fp8 = (BI.Mma_fp8_e5m2, BI.Mma_fp8_e5m2, BI.Mma_f32) in
  let staged_fp8 =
    match limits.BI.mma with
    | Some mma -> List.Assoc.mem mma.BI.mma_staged_layouts fp8 ~equal:BI.equal_mma_format_triple
    | None -> false
  in
  site_claims ~tag:"backend record" ~limits
    ~capabilities:(Context.codegen_capabilities ctx)
    ~evaluated:staged_fp8

(* [Autotune.model_explain] over the same site as a computation: what [model_default] sees of each
   candidate. On every backend: the default first, then distinct labels, each score at least its
   bound. Where the backend describes the staged fp8 twins, some two candidates tie exactly on the
   ranking score and differ in their operand loads — the twins the tie-break separates. *)
let () =
  let ctx = Context.auto () in
  let limits = Context.hardware_limits ctx in
  let fa = NTDSL.init ~l:"mlt_xa" ~prec:Ir.Ops.fp8 ~i:[ nk ] ~o:[ nm ] ~f:(fun _ -> 1.) () in
  let fb = NTDSL.init ~l:"mlt_xb" ~prec:Ir.Ops.fp8 ~i:[ nn ] ~o:[ nk ] ~f:(fun _ -> 1.) () in
  let%op t = fa * fb in
  Ir.Tnode.update_prec t.Tensor.value Ir.Ops.single;
  let comp = Train.forward t in
  let explained = Autotune.model_explain ~name:"mlt_explain" ctx comp Ir.Indexing.Empty in
  List.iter explained ~f:(fun (label, ex) ->
      let ms = Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.9f") in
      Stdio.eprintf
        "explain %s: floor %s rank %s compute %s memory %s spilled %s loads %s (not part of the \
         golden)\n"
        label (ms ex.Autotune.ex_floor_ms) (ms ex.ex_rank_ms) (ms ex.ex_compute_ms)
        (ms ex.ex_memory_ms)
        (Option.value_map ex.ex_spilled ~default:"n/a" ~f:Bool.to_string)
        (Sexp.to_string (CM.sexp_of_load_count ex.ex_operand_loads)));
  p "model_explain lists model_default's admitted incumbent first"
    (match explained with
    | (label, ex) :: _ ->
        List.mem [ "default"; "seeded" ] label ~equal:String.equal && ex.Autotune.ex_admitted
    | [] -> false);
  (* Without envelope constants (the C backends carry none) [model_default] ranks nothing, and
     [model_explain] lists only the default pipeline. *)
  let ranks = match Autotune.envelope_legs ~limits with None, None -> false | _ -> true in
  if ranks then
    p_pairwise_distinct "model_explain's labels are distinct" (List.map explained ~f:fst)
      ~equal:String.equal ~to_string:Fn.id
  else
    p "model_explain's labels are distinct"
      (match explained with [ ("default", _) ] -> true | _ -> false);
  p_all "every explained ranking score is at least its bound" explained ~f:(fun (_, ex) ->
      match (ex.Autotune.ex_rank_ms, ex.ex_floor_ms) with
      | Some r, Some f -> Float.(r >= f)
      | None, None -> true
      | _ -> false);
  let staged_fp8 =
    match limits.BI.mma with
    | Some mma ->
        List.Assoc.mem mma.BI.mma_staged_layouts
          (BI.Mma_fp8_e5m2, BI.Mma_fp8_e5m2, BI.Mma_f32)
          ~equal:BI.equal_mma_format_triple
    | None -> false
  in
  let leaves = List.tl explained |> Option.value ~default:[] in

  (* A backend that describes none of its arms (every backend but CUDA: the CPU register tiles,
     Metal's and HIP's fragments) has no operand-load count for any candidate: none is invented. *)
  let caps = Context.codegen_capabilities ctx in
  let describes =
    List.exists Ir.Ops.all_precs ~f:(fun a_prec ->
        List.exists Ir.Ops.all_precs ~f:(fun b_prec ->
            List.exists Ir.Ops.all_precs ~f:(fun d_prec ->
                List.exists BI.all_of_mma_emission_scope ~f:(fun scope ->
                    Option.is_some (caps.BI.mma_operand_fragments ~a_prec ~b_prec ~d_prec ~scope)))))
  in
  gated_all ~when_:(not describes) ~on:backend_name
    "where the backend describes no mma arm, no candidate gets an operand-load count" explained
    ~f:(fun (_, ex) ->
      match ex.Autotune.ex_operand_loads with
      | CM.Loads _ -> false
      | CM.No_tile_mma | CM.Undescribed -> true);
  gated_exists ~when_:staged_fp8 ~on:backend_name
    "model_explain shows two candidates tied exactly on the ranking score with different operand \
     loads" (List.cartesian_product leaves leaves) ~f:(fun ((_, x), (_, y)) ->
      Option.equal Float.equal x.Autotune.ex_rank_ms y.Autotune.ex_rank_ms
      && Option.is_some x.ex_rank_ms
      &&
      match (x.ex_operand_loads, y.ex_operand_loads) with
      | CM.Loads a, CM.Loads b -> a < b
      | _ -> false);
  (* What [model_default] then ships at this shape, for the record. *)
  if staged_fp8 then
    let choice = ref None in
    let _ctx, _routine =
      Autotune.model_default ~name:"mlt_default"
        ~report:(fun r -> choice := Some r)
        ctx comp Ir.Indexing.Empty
    in
    Option.iter !choice ~f:(fun r ->
        Stdio.eprintf "model_default picks %s (not part of the golden)\n" r.Autotune.mc_label)
