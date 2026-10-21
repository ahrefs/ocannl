(* gh-ocannl-1274: the flip chain skips a structural twin of a flip that lost, without a budget
   slot.

   On rog's approximate CUDA gpt2_mini cell the first 11 slots of the inline-flip chain were the
   per-layer q/k/v projections, and each set's members lost by the same figure: a budget of 2 never
   reached past layer 1's k and v. The rule: an alternative whose local reading is canonically
   identical ([Autotune.placement_surface]'s [ps_twin_identity] -- the flipped node's
   producer/consumer statements under fresh-name normalization, each node with its dims, precision
   and placements, plus the flip's readings and recompute cost) to one already measured and lost in
   the same chain is skipped as [twin-of] that flip, and spends no budget.

   The fixture has three branches of the same shape, [x = a *. b; y = relu x], each summed into the
   loss: the first two over 8x8 inputs, the third over 4x4. So [x1]/[x2] and [y1]/[y2] are twins,
   while [x3]/[y3] share their structure but not their dims, and must not be twinned (the issue's
   negative acceptance). On cc, with every timing from a synthetic clock
   ([Autotune.on_candidate_measured]): - the surface's identities, printed as twin classes; - a
   chain whose every flip is hopeless: each twin is skipped as [twin-of] its first member, the
   dims-differing branch is measured, the budget counts only measured flips, and a budget of 2
   reaches 2 distinct classes past a skipped twin; - a warm rerun of that chain replays the anchors'
   abandonments and skips the same twins; - a flip searched in full that loses anchors its twin just
   as an abandoned one does; - the control: flips that WIN anchor nothing, so their twins are
   measured; - two refusals, each with a control showing the refusal is what keeps the flips apart:
   a computation inherited from an earlier routine, in the caller's lineage and in the timing
   lineage alone (both flips measured; the rule with the refusal lifted twins them), and a
   neighbourhood's statement-level local, defined by a different statement or by none (hand-built:
   no [Assignments] lowering emits the shape). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

(* The abandonment depth (the beam width), and no beam rounds: each search times only its seeds. A
   flip at 100 ms is abandoned when its second attempt starts, so the chain compiles as few
   candidates as the claims allow -- every compiled candidate is a fresh dlopen, which macOS
   scans. *)
let k = 1

(* One measurement script: the [i]-th admitted window measures [script i]. *)
let with_clock script f =
  fst (Scripted_search.scripted ~measure:(fun w -> script w.Scripted_search.ordinal) f)

(* The progress lines of event [name], and one field of each that has it. *)
let events = Stderr_capture.Progress.events
let field = Stderr_capture.Progress.field
let plain lines name key = List.filter_map (events lines name) ~f:(fun l -> field l key)

type chain = {
  flip_reports : Autotune.report list;
  measured_arms : string list;  (** The flips' [arm_start] labels, in chain order. *)
  twins : (string * string) list;  (** [(arm, twin_of)] per [flip_twin] event, in chain order. *)
  done_measured : string list;
  done_twinned : string list;
  shipped : string list;
}

(* [Train.tune_placements] from [ctx] (a fresh [Context.auto ()] by default) with arm A's windows at
   4 ms, arm B's at 50 ms, and each flip's per [flip_script ~flip window] (both 1-based: the flip's
   ordinal among the measured ones, the window's within that flip's search). The searches are told
   apart by how many reports have been delivered when a window is admitted. *)
let run_chain ?(cache_dir = "") ?ctx ?timing_ctx ~budget comp loss ~flip_script =
  let arm_reports = ref [] and flip_reports = ref [] and shipped = ref [] in
  let delivered () = List.length !arm_reports + List.length !flip_reports in
  let flip_window = ref 0 and flip_seen = ref 0 in
  let script _ =
    match delivered () with
    | 0 -> 4.0
    | 1 -> 50.0
    | d ->
        if d <> !flip_seen then (
          flip_seen := d;
          flip_window := 0);
        Int.incr flip_window;
        flip_script ~flip:(d - 1) !flip_window
  in
  let (), lines =
    Stderr_capture.lines (fun () ->
        with_clock script (fun () ->
            let _ctx, _routine =
              Train.tune_placements ~beam_width:k ~rounds:0 ~repeats:1 ~cache_dir
                ~placement_store:false
                ~report:(fun r -> arm_reports := r :: !arm_reports)
                ~flip_report:(fun r -> flip_reports := r :: !flip_reports)
                ~on_ship:(fun s -> shipped := s.Train.ship_arm :: !shipped)
                ?timing_ctx ~inline_flips:budget
                (match ctx with Some ctx -> ctx | None -> Context.auto ())
                loss comp Ir.Indexing.Empty
            in
            ()))
  in
  let flip_label l = Option.filter (field l "arm") ~f:(String.is_prefix ~prefix:"flip ") in
  {
    flip_reports = List.rev !flip_reports;
    measured_arms = List.filter_map (events lines "arm_start") ~f:flip_label;
    twins =
      List.filter_map (events lines "flip_twin") ~f:(fun l ->
          Option.both (field l "arm") (field l "twin_of"));
    done_measured = plain lines "flips_done" "measured";
    done_twinned = plain lines "flips_done" "twinned";
    shipped = !shipped;
  }

let clean_cache = Scripted_search.clear_cache_dir

(* An arm label names its node as [flip <reading> <debug name> (cost ...)]. *)
let names (t : Tensor.t) label =
  String.is_substring label ~substring:(" " ^ Tn.debug_name t.Tensor.value ^ " (")

let () =
  let n = 8 and m = 4 in
  let nd label dims modulus offset =
    let d = dims in
    TDSL.ndarray
      (Array.init (d * d) ~f:(Ll_test.cycle_flat ~dims:[| d; d |] ~modulus ~offset ~stride:0.5))
      ~label:[ label ] ~output_dims:[ d; d ] ()
  in
  (* Distinct values per input: the twins share structure, not data. *)
  let a1 = nd "a1" n 7 0. and b1 = nd "b1" n 5 (-2.) in
  let a2 = nd "a2" n 11 1. and b2 = nd "b2" n 3 (-1.) in
  let a3 = nd "a3" m 7 0. and b3 = nd "b3" m 5 (-2.) in
  let%op x1 = a1 *. b1 in
  let%op y1 = relu x1 in
  let%op x2 = a2 *. b2 in
  let%op y2 = relu x2 in
  let%op x3 = a3 *. b3 in
  let%op y3 = relu x3 in
  let%op loss = y1 ++ "ij => 0" + (y2 ++ "ij => 0") + (y3 ++ "ij => 0") in
  let comp = Train.forward loss in
  (* The surface, as the chain reads it. *)
  let surface = Autotune.placement_surface (Context.auto ()) comp Ir.Indexing.Empty in
  let no_chain _ = None and no_tag _ = Some "" in
  let classes = Hashtbl.create (module String) in
  let class_of id =
    match id with
    | None -> "-"
    | Some id ->
        Hashtbl.find_or_add classes id ~default:(fun () ->
            String.of_char (Char.of_int_exn (Char.to_int 'A' + Hashtbl.length classes)))
  in
  Stdio.printf "ranked surface (node, default -> alternative, cost, twin class):\n";
  List.iter surface.Autotune.ps_candidates ~f:(fun fc ->
      List.iter fc.LL.fc_alternatives ~f:(fun fa ->
          Stdio.printf "  %-10s %-11s -> %-11s cost %-5d class %s\n" (Tn.debug_name fc.LL.fc_tn)
            (LL.reading_to_string fc.LL.fc_default)
            (LL.reading_to_string fa.LL.fa_flip)
            fa.LL.fa_recompute_cost
            (class_of (surface.Autotune.ps_twin_identity ~node_tag:no_tag ~chain:no_chain fc fa))));
  let candidate (t : Tensor.t) =
    List.find surface.Autotune.ps_candidates ~f:(fun fc -> Tn.equal fc.LL.fc_tn t.Tensor.value)
  in
  let identity ?(node_tag = no_tag) ?(chain = no_chain) t =
    Option.bind (candidate t) ~f:(fun fc ->
        Option.bind (List.hd fc.LL.fc_alternatives) ~f:(fun fa ->
            surface.Autotune.ps_twin_identity ~node_tag ~chain fc fa))
  in
  let twins s t =
    Option.is_some (identity s) && Option.equal String.equal (identity s) (identity t)
  in
  p_all "every branch node is a flip candidate with an identity" [ x1; y1; x2; y2; x3; y3 ]
    ~f:(fun t -> Option.is_some (identity t));
  p "x1 and x2 are twins: same structure, same dims, same placements" (twins x1 x2);
  p "y1 and y2 are twins" (twins y1 y2);
  p "x3 shares x1's structure but not its dims, and is not its twin" (not (twins x1 x3));
  p "y3 is not y1's twin either" (not (twins y1 y3));
  p "a producer and its consumer are not twins" (not (twins x1 y1));
  (* A reading the chain has committed for a neighbour changes the flip's world: y1 reads x1, so
     materializing x1 changes y1's identity and leaves y2's, which no longer match. *)
  let x1_committed n = if Tn.equal n x1.Tensor.value then Some `Materialize else None in
  p "a chain-committed neighbour reading separates twins"
    (not
       (Option.equal String.equal (identity ~chain:x1_committed y1)
          (identity ~chain:x1_committed y2)));
  (* The caller's tag -- the flip chain's timing lineage -- enters per node too. *)
  let a2_tagged n = Some (if Tn.equal n a2.Tensor.value then ";t-differs" else "") in
  p "a caller tag on a twin's neighbour separates the twins"
    (not
       (Option.equal String.equal (identity ~node_tag:a2_tagged x2)
          (identity ~node_tag:a2_tagged x1)));
  let a2_refused n = if Tn.equal n a2.Tensor.value then None else Some "" in
  p "a caller tag refusing a neighbour leaves no identity"
    (Option.is_some (identity x2) && Option.is_none (identity ~node_tag:a2_refused x2));
  p "a commitment outside the neighbourhood leaves twins twinned"
    (Option.equal String.equal (identity ~chain:x1_committed x2) (identity x2));
  let total_alternatives =
    List.sum
      (module Int)
      surface.Autotune.ps_candidates
      ~f:(fun fc -> List.length fc.LL.fc_alternatives)
  in
  let big_budget = total_alternatives + 1 in
  let hopeless ~flip:_ _ = 100.0 in
  (* Every flip hopeless, budget past the surface: the whole chain walked. *)
  let cache_dir = "autotune_cache_flip_twins" in
  clean_cache cache_dir;
  let cold = run_chain ~cache_dir ~budget:big_budget comp loss ~flip_script:hopeless in
  let twin_of t c =
    List.find_map c.twins ~f:(fun (arm, anchor) -> Option.some_if (names t arm) anchor)
  in
  let measured t c = List.exists c.measured_arms ~f:(names t) in
  p "x2 is skipped as twin-of x1" (Option.exists (twin_of x2 cold) ~f:(names x1));
  p "y2 is skipped as twin-of y1" (Option.exists (twin_of y2 cold) ~f:(names y1));
  p_all "the first of each twin set and the dims-differing branch are measured" [ x1; y1; x3; y3 ]
    ~f:(fun t -> measured t cold && Option.is_none (twin_of t cold));
  p_none "a skipped twin is never also measured" cold.twins ~f:(fun (arm, _) ->
      List.mem cold.measured_arms arm ~equal:String.equal);
  p_all "every twin names an anchor measured before it" cold.twins ~f:(fun (_, anchor) ->
      List.mem cold.measured_arms anchor ~equal:String.equal);
  p "every alternative is either measured or skipped as a twin"
    (List.length cold.measured_arms + List.length cold.twins = total_alternatives);
  p "flips_done counts the measured flips and the twins apart"
    (List.equal String.equal cold.done_measured [ Int.to_string (List.length cold.measured_arms) ]
    && List.equal String.equal cold.done_twinned [ Int.to_string (List.length cold.twins) ]
    && List.length cold.flip_reports = List.length cold.measured_arms);
  p_all "every measured flip was abandoned" cold.flip_reports ~f:(fun r ->
      match r.Autotune.outcome with Autotune.Abandoned _ -> true | _ -> false);
  p "the A/B winner ships" (List.equal String.equal cold.shipped [ "A" ]);
  (* A warm rerun: the anchors replay their abandonments, and the same twins are skipped. *)
  let warm = run_chain ~cache_dir ~budget:big_budget comp loss ~flip_script:hopeless in
  p_all ~min:1 "the warm chain replays every measured flip's abandonment" warm.flip_reports
    ~f:(fun r -> match r.Autotune.outcome with Autotune.Abandonment_replay _ -> true | _ -> false);
  p "the warm chain measures the same flips and skips the same twins"
    (List.equal String.equal warm.measured_arms cold.measured_arms
    && List.equal (fun (a, b) (c, d) -> String.equal a c && String.equal b d) warm.twins cold.twins
    );
  clean_cache cache_dir;
  (* The flips are measured in the timing lineage, so what it brings to a neighbour separates twins
     there: with [y2] preferred inline in the timing context alone, neither [x2] (whose consumer
     [y2] is) nor [y2] itself is a twin any more, while [y3]'s branch and the other pair are
     unaffected. *)
  let timing_ctx = Context.decide_inline (Context.auto ()) [ y2.Tensor.value ] in
  let timed = run_chain ~timing_ctx ~budget:big_budget comp loss ~flip_script:hopeless in
  p "under a timing lineage that differs on y2, x2 and y2 are measured, not twinned"
    (measured x2 timed && measured y2 timed
    && Option.is_none (twin_of x2 timed)
    && Option.is_none (twin_of y2 timed));
  (* Budget 2, the approximate profile's: the first twin pair's second member is skipped without a
     slot, so the chain reaches two distinct classes and not one class twice. *)
  let first_twin_rank =
    List.find_mapi surface.Autotune.ps_candidates ~f:(fun i fc ->
        Option.some_if
          (List.exists cold.twins ~f:(fun (arm, _) ->
               String.is_substring arm ~substring:(" " ^ Tn.debug_name fc.LL.fc_tn ^ " (")))
          i)
  in
  p "the surface ranks a twin among its first three candidates"
    (Option.exists first_twin_rank ~f:(fun i -> i < 3));
  let small = run_chain ~budget:2 comp loss ~flip_script:hopeless in
  p "a budget-2 chain measures two flips" (List.length small.measured_arms = 2);
  p "it skips at least one twin on the way" (not (List.is_empty small.twins));
  p "the two flips it measures are of distinct twin classes"
    (match small.measured_arms with
    | [ f1; f2 ] ->
        let id label =
          List.find_map surface.Autotune.ps_candidates ~f:(fun fc ->
              Option.some_if
                (String.is_substring label ~substring:(" " ^ Tn.debug_name fc.LL.fc_tn ^ " ("))
                fc)
          |> Option.bind ~f:(fun fc ->
              Option.bind (List.hd fc.LL.fc_alternatives) ~f:(fun fa ->
                  surface.Autotune.ps_twin_identity ~node_tag:no_tag ~chain:no_chain fc fa))
        in
        Option.is_some (id f1)
        && Option.is_some (id f2)
        && not (Option.equal String.equal (id f1) (id f2))
    | _ -> false);
  (* A flip searched in full that loses anchors its twin like an abandoned one: within the ratio at
     depth k (5 ms against the incumbent's 4 ms, ratio 1.5625), never faster than arm A. *)
  let losing ~flip:_ _ = 5.0 in
  let lost = run_chain ~budget:2 comp loss ~flip_script:losing in
  p_all ~min:1 "every measured flip is searched in full" lost.flip_reports ~f:(fun r ->
      Poly.equal r.Autotune.outcome Autotune.Searched);
  p "a flip that lost a full search anchors its twin"
    (Option.exists (twin_of y2 lost) ~f:(names y1));
  (* Only clean evidence anchors: a flip whose every window is refused (a NaN reading, refused as a
     contended window is) measured nothing, so its twin is measured in turn. *)
  let refused ~flip:_ _ = Float.nan in
  let unclean = run_chain ~budget:2 comp loss ~flip_script:refused in
  p "a flip whose windows were all refused timed nothing"
    (match unclean.flip_reports with
    | r :: _ -> r.Autotune.candidates_timed = 0 && r.Autotune.timings_contended > 0
    | [] -> false);
  p_empty "a flip that timed nothing anchors no twin" ~over:unclean.measured_arms unclean.twins;
  p "y2 is measured after y1 timed nothing" (measured y1 unclean && measured y2 unclean);
  (* The control: every flip wins, each beating the one accepted before it. A flip that is accepted
     did not lose, so it anchors nothing and its twin is measured. *)
  let winning ~flip _ = 2.0 /. Float.of_int flip in
  let won = run_chain ~budget:2 comp loss ~flip_script:winning in
  p "the winning chain ships a refined placement" (List.equal String.equal won.shipped [ "flip" ]);
  p_empty "no flip is skipped as a twin when the flips win" ~over:won.measured_arms won.twins;
  p "y2 is measured after y1 won" (measured y1 won && measured y2 won);
  (* An inherited computation refuses an identity, in either lineage. [u1 = c1 + d1] and [u2 = c2 +
     d2] are inlined into an earlier routine [shared = u1 + u2], whose compile stores their
     computations in its lineage; a later routine reading them replays those templates, which its
     source does not show. The second loss sums [v1 = u1 *. e1] and [v2 = u2 *. e2], so the
     neighbourhoods of [v1] and [v2] render [u1] and [u2]. The earlier routine treats [u1] and [u2]
     alike, so every tag the identity renders is symmetric between the two flips, and only the
     refusal keeps them apart. *)
  let c1 = nd "c1" n 7 0. and d1 = nd "d1" n 5 (-2.) in
  let c2 = nd "c2" n 11 1. and d2 = nd "d2" n 3 (-1.) in
  let e1 = nd "e1" n 13 0.5 and e2 = nd "e2" n 9 (-0.5) in
  let%op u1 = c1 + d1 in
  let%op u2 = c2 + d2 in
  let%op shared = u1 + u2 in
  let earlier = Train.forward shared in
  let%op v1 = u1 *. e1 in
  let%op v2 = u2 *. e2 in
  let%op loss2 = v1 ++ "ij => 0" + (v2 ++ "ij => 0") in
  let comp2 = Train.forward loss2 in
  let first_alternative (s : Autotune.placement_surface) (t : Tensor.t) =
    List.find_map s.Autotune.ps_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn t.Tensor.value then
          Option.map (List.hd fc.LL.fc_alternatives) ~f:(fun fa -> (fc, fa))
        else None)
  in
  let alternatives (s : Autotune.placement_surface) =
    List.sum (module Int) s.Autotune.ps_candidates ~f:(fun fc -> List.length fc.LL.fc_alternatives)
  in
  let identity_by rule s t =
    Option.bind (first_alternative s t) ~f:(fun (fc, fa) ->
        rule ~node_tag:no_tag ~chain:no_chain fc fa)
  in
  let surface_identity (s : Autotune.placement_surface) =
    identity_by s.Autotune.ps_twin_identity s
  in
  let measured_untwinned c t = measured t c && Option.is_none (twin_of t c) in
  (* The caller's lineage inherits. *)
  let inheriting, _ = Context.compile (Context.auto ()) earlier Ir.Indexing.Empty in
  let inherits = Context.inherits_computation inheriting in
  p "the earlier routine stored computations for u1 and u2 in its lineage"
    (inherits u1.Tensor.value && inherits u2.Tensor.value);
  let surface2 = Autotune.placement_surface inheriting comp2 Ir.Indexing.Empty in
  p_all "v1 and v2 are flip candidates without an identity" [ v1; v2 ] ~f:(fun t ->
      Option.is_some (first_alternative surface2 t) && Option.is_none (surface_identity surface2 t));
  (* The control: the surface's own rule over the same lowering, once with the lineage's inheritance
     -- which reproduces the surface's verdict -- and once with the refusal lifted, which twins the
     two flips. *)
  let rule ~inherited =
    identity_by
      (Autotune.flip_twin_identity ~static_indices:[] ~lineage:(Context.placements inheriting)
         ~inherited ~enablement:surface2.Autotune.ps_enablement
         ~disablement:surface2.Autotune.ps_disablement
         (Context.lowered_for_decisions inheriting comp2 Ir.Indexing.Empty))
      surface2
  in
  p_all "the rule with the lineage's inheritance gives the surface's verdict" [ v1; v2 ]
    ~f:(fun t -> Option.is_none (rule ~inherited:inherits t));
  let lifted = rule ~inherited:(fun _ -> false) in
  p "with the refusal lifted, the same lowering twins v2 with v1"
    (Option.is_some (lifted v1) && Option.equal String.equal (lifted v1) (lifted v2));
  let inherited_chain =
    run_chain ~ctx:inheriting ~budget:(alternatives surface2) comp2 loss2 ~flip_script:hopeless
  in
  p_all "under an inheriting caller lineage, v1 and v2 are both measured, neither as a twin"
    [ v1; v2 ]
    ~f:(measured_untwinned inherited_chain);
  (* The timing lineage inherits. [held] computed [shared] with [u1] and [u2] materialized: the
     caller's surface reads two buffers and twins [v1] with [v2], while the flips are measured in
     [inheriting], which replays the stored templates. The flip chain's timing tag refuses there. *)
  let held, _ =
    Context.compile
      (Context.decide_materialized (Context.auto ()) [ u1.Tensor.value; u2.Tensor.value ])
      earlier Ir.Indexing.Empty
  in
  p "the held lineage stores no computation for u1 or u2"
    (not
       (Context.inherits_computation held u1.Tensor.value
       || Context.inherits_computation held u2.Tensor.value));
  let surface3 = Autotune.placement_surface held comp2 Ir.Indexing.Empty in
  p "on the held caller's surface, v2 is v1's twin"
    (Option.is_some (surface_identity surface3 v1)
    && Option.equal String.equal (surface_identity surface3 v1) (surface_identity surface3 v2));
  let timed_inherited =
    run_chain ~ctx:held ~timing_ctx:inheriting ~budget:(alternatives surface3) comp2 loss2
      ~flip_script:hopeless
  in
  p_all "under an inheriting timing lineage, v1 and v2 are both measured, neither as a twin"
    [ v1; v2 ]
    ~f:(measured_untwinned timed_inherited);
  (* Two per-node facts, each declared on one member of a still-twinned pair, last since both
     declarations are monotone. The compile-time bounds candidate (the simplifier folds guards on
     it): a proposal on [b2], an operand of [x2]'s producer, separates [x2] from [x1]. *)
  let x_were_twins = twins x1 x2 in
  let b2_unknown = Option.is_none (Tn.bounds_candidate b2.Tensor.value) in
  Tn.propose_bounds ~what:"flip_twins" b2.Tensor.value (Ir.Interval.of_int_range 0 3);
  p "a bounds candidate on x2's operand separates x2 from x1"
    (x_were_twins && b2_unknown && not (twins x1 x2));
  (* Observability (an observable buffer is never aliased): declaring [y2] observable separates it
     from [y1]. *)
  let y_were_twins = twins y1 y2 in
  let was_observable = Tn.is_observable y2.Tensor.value in
  Tn.set_observable y2.Tensor.value;
  p "declaring y2 observable separates it from its twin y1"
    (y_were_twins && (not was_observable) && not (twins y1 y2))

(* Statement-level locals. A neighbourhood is closed over the scope locals it reads: a top-level
   statement defining one joins it, and a local no statement defines refuses the identity. No
   [Assignments] lowering puts a local's definition in a top-level statement apart from its reads --
   even the online-softmax rewrite emits its locals inside the nest that reads them -- so the case
   is hand-built and handed to the surface's rule ([Autotune.flip_twin_identity]) directly. [xi]
   reads the local [pi], defined ahead of it from [ai]; with equal definitions [x1] and [x2] are
   twins, which is the control that the definitions are all that differs between the programs
   below. *)
let () =
  let open Ll_test in
  let node = node_factory ~first_id:1_274_000 ~dims:[| 8 |] () in
  let a1 = node "la1" and a2 = node "la2" and b1 = node "lb1" and b2 = node "lb2" in
  let s1 = node "ls1" and s2 = node "ls2" in
  let p1 = node ~dims:[| 1 |] "lp1" and p2 = node ~dims:[| 1 |] "lp2" in
  let x1 = node "lx1" and x2 = node "lx2" in
  List.iter [ a1; a2; b1; b2; s1; s2 ] ~f:materialize;
  List.iter [ p1; p2 ] ~f:virtualize;
  let p1 = LL.get_scope p1 and p2 = LL.get_scope p2 in
  let produce x p b =
    let i = sym () in
    loop_n i 8 (set_at x (iter i) (mul (LL.Get_local p) (get b [| iter i |])))
  in
  let consume s x =
    let i = sym () in
    loop_n i 8 (set_at s (iter i) (add (get x [| iter i |]) (c 1.)))
  in
  let identities name ~def2 =
    let o =
      optimize ~name
        (LL.unflat_lines
           (define_local p1 (mul (get a1 [| fixed 0 |]) (c 2.))
           @ Option.value_map def2 ~default:[] ~f:(define_local p2)
           @ [ produce x1 p1 b1; produce x2 p2 b2; consume s1 x1; consume s2 x2 ]))
    in
    let rule =
      Autotune.flip_twin_identity ~static_indices:[]
        ~lineage:(LL.empty_optimize_ctx ()).LL.placements
        ~inherited:(fun _ -> false)
        ~enablement:(Set.empty (module Tn))
        ~disablement:(Set.empty (module Tn))
        o
    in
    let identity x =
      List.find_map o.LL.flip_candidates ~f:(fun fc ->
          if Tn.equal fc.LL.fc_tn x then
            Option.bind (List.hd fc.LL.fc_alternatives) ~f:(fun fa ->
                rule ~node_tag:(fun _ -> Some "") ~chain:(fun _ -> None) fc fa)
          else None)
    in
    (identity x1, identity x2)
  in
  let same1, same2 = identities "lt_same" ~def2:(Some (mul (get a2 [| fixed 0 |]) (c 2.))) in
  let diff1, diff2 = identities "lt_differ" ~def2:(Some (add (get a2 [| fixed 0 |]) (c 1.))) in
  let unbound1, unbound2 = identities "lt_unbound" ~def2:None in
  p "control: with equal definitions of their locals, x1 and x2 are twins"
    (Option.is_some same1 && Option.equal String.equal same1 same2);
  p "a local defined by a different statement separates x2 from x1"
    (Option.is_some diff1 && Option.is_some diff2 && not (Option.equal String.equal diff1 diff2));
  p "x1's identity is the same in both programs" (Option.equal String.equal diff1 same1);
  p "a local no statement defines leaves no identity"
    (Option.is_some unbound1 && Option.is_none unbound2)
