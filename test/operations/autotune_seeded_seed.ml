(* gh-ocannl-1388: the autotuner seeds the seeded untuned plan FIRST, so a search cut to one
   candidate ships the step the untuned compile ships.

   Since gh-ocannl-1324 the recipes' untuned compile on CUDA and HIP is [Autotune.seeded_default]:
   the default pipeline with its large contractions seeded with a reuse-tiled sketch, its long
   narrow reductions split (gh-ocannl-1325, CUDA) and the rest of its reductions sunk
   (gh-ocannl-1348). Before this issue the search's pool started from the plain default, so a search
   that timed only its first candidate shipped a plan the untuned compile had already beaten (1720
   ms against 563 ms on gpt2_mini_train b256, rog).

   The fixture is one dense layer with a bias, [y = w * x + b], backpropagated through [loss = sum
   (y *. dy)] over [nb * ns] = 4096 rows: the weight gradient's reduction (4096) and the forward
   product's (96) both reach the seeding threshold (32), and the bias gradient's serial chain (4096
   rows per output cell) and the loss sum's reach the default split threshold (1024), so on CUDA the
   plan seeds AND splits; HIP never splits untuned (gh-ocannl-1325), so there it seeds only. Which
   backend promises what is read off [Autotune.seeded_default_backend] and [seeded_split_backend],
   the predicates the untuned compile consults. Every setting is the configuration's default but the
   seeding gate (below): the search reads them, as the untuned compile does.

   The search's measurements are scripted through [Autotune.on_candidate_measured] (gh-ocannl-1027)
   with the windows opted out of host contention ([with_uncontended_test_windows], gh-ocannl-1156):
   the seeded plan's windows keep their readings and every other one is refused, which is the search
   cut to its first candidate. The claims: the first candidate attempted is the seeded plan, its
   timed code is the untuned compile's segment for segment (the digests the search keys candidates
   by), the search did time later candidates, it crowned the seeded plan, and the crowned routine
   computes bitwise what the untuned compile computes, on inputs whose every partial sum rounds --
   and which tell the plans apart: the split reassociates the bias gradient, so the plain default
   pipeline computes it differently in some cell, which is claimed as the control where the plan
   splits (elsewhere the plan is bitwise the default by construction, and only the digest claim
   tells them apart). Off CUDA/HIP the untuned compile is the plain default and the search seeds no
   plan, which is claimed instead.

   A window the host refuses anyway (queued calibration that measured no batch, routine on a shared
   box) never reaches the seam, so the claims that read the scripted windows are gated on every
   attempt of the seeded plan having reached it; the first-candidate claim is structural and always
   evaluated.

   A second search refuses the seeded plan's first window, as host contention would, and scripts
   every other candidate far slower than any real reading: a refused digest leaves the dedup set for
   an equivalent later seed, but no later seed reproduces the plan, so only the one retry the seed
   pass ends with can measure and crown it. That retry is claimed, with the crown it wins.

   The CUDA/HIP legs need a device that seeds; elsewhere they are skipped. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module SC = Ir.Schedule_cache
module Cache_plant = Test_utils.Cache_plant

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* gh-ocannl-1253: the seeding gate is pinned at threshold 32, the default before [auto]. Every
   claim below needs a plan that seeds, and the ranked [auto] gate seeds nothing on this fixture on
   HIP, even with every working set past the cache: whether a sketch beats the preset is its call,
   while what this test checks -- the search seeds the plan the untuned compile ships, whatever the
   gate resolved it to -- is the gate's to feed, not to decide. The untuned compile and the search
   read the same setting, set before the first compile reads it. *)
let () = Unix.putenv "OCANNL_GPU_DEFAULT_SKETCH_MIN_REDUCTION" "32"

(* What the untuned compile promises here, read off the predicates it consults: seeding on CUDA and
   HIP, the split likewise ([Autotune.seeded_split_backend], gh-ocannl-1325, gh-ocannl-1412). *)
let seeds_here = Autotune.seeded_default_backend backend_name
let splits_here = Autotune.seeded_split_backend backend_name

(* The batch axis outermost and long: the split detector splits a reduction over its outermost loop,
   which at [nb = 4] would be too short to propose (the gpt2_mini_train b8 case). *)
let nb = 1024
and ns = 4
and no = 64
and ni = 96

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"ss_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"ss_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x =
  NTDSL.ndarray
    (values ~n:(nb * ns * ni) ~stride:389)
    ~label:[ "ss_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()

let dy =
  NTDSL.ndarray
    (values ~n:(nb * ns * no) ~stride:577)
    ~label:[ "ss_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

let%op y = (w * x) + b
let%op loss = (y *. dy) ++ "... => 0"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let wgrad = grad_of w
let bgrad = grad_of b

let () =
  Train.set_materialized wgrad;
  Train.set_materialized bgrad

let comp = Train.grad_update loss
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let static_indices = Ir.Indexing.bound_symbols Ir.Indexing.Empty

(* The digest the search keys a compiled candidate by: each compiled segment's canonical digest, in
   order. *)
let digest_of posts =
  String.concat ~sep:"+"
    (List.map posts ~f:(fun post -> SC.digest (SC.canonicalize ~static_indices post)))

let run ctx routine =
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx wgrad, Context.get_values ctx bgrad)

(* The untuned compile, through the transform [Autotune.seeded_default] compiles with, capturing the
   code it hands the backend and what it decided. *)
let untuned () =
  let ctx = Context.auto () in
  let digest = ref None and choice = ref None in
  let ctx, routine =
    match
      Autotune.seeded_transform ~on_choice:(fun c -> choice := Some c) ctx Ir.Indexing.Empty
    with
    | None -> Context.compile ctx comp Ir.Indexing.Empty
    | Some transform ->
        Context.compile
          ~lowered_transform:(fun opt ->
            let posts = transform opt in
            digest := Some (digest_of posts);
            posts)
          ctx comp Ir.Indexing.Empty
  in
  (run ctx routine, !digest, !choice)

let is_seeded label = String.is_prefix label ~prefix:"F_seeded["

(* A search whose admitted windows are scripted: [reading ~index ~label ms] is what the [index]th
   admitted window (from 0) records, [nan] refusing it. *)
let scripted_search ~reading =
  let latest = ref None and timed = ref [] and report = ref None in
  let (ctx, routine), (record : Scripted_search.record) =
    Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
        if not (String.equal label "baseline") then latest := Some label)
    @@ fun () ->
    (* Each timing call that returns, under the attempt it timed: a window the host refused still
       returns one, a candidate that failed before timing never does. *)
    Autotune.with_timing_call_observer (fun (_ : Autotune.timing_call) ->
        Option.iter !latest ~f:(fun label -> timed := label :: !timed))
    @@ fun () ->
    Scripted_search.scripted
      ~measure:(fun w -> reading ~index:(w.ordinal - 1) ~label:w.label w.reading)
      (fun () ->
        Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
          ~cache_dir:"" ~log:false
          ~report:(fun r -> report := Some r)
          (Context.auto ()) comp Ir.Indexing.Empty)
  in
  let report = Option.value_exn ~here:[%here] !report in
  let windows = List.map record.windows ~f:(fun (w, ms) -> (w.label, w.digest, ms)) in
  (* Every attempt of the seeded plan must compile and reach a timing window ([reached], a claim: a
     candidate failure is a failure), and a window the host refuses (queued calibration that
     measured no batch, routine on a shared box) never reaches the seam: what the window claims read
     is the seeded plan's, so they are gated on the host having refused none of its windows
     ([quiet]). *)
  let attempts =
    List.filter record.attempts ~f:(fun label -> not (String.equal label "baseline"))
  in
  let seeded_timed = List.count !timed ~f:is_seeded in
  let reached = List.count attempts ~f:is_seeded = seeded_timed && seeded_timed > 0 in
  let quiet = seeded_timed = List.count windows ~f:(fun (label, _, _) -> is_seeded label) in
  Stdio.eprintf
    "scripted search (not part of the golden): %d windows, %d refused by the script, %d refused in \
     all\n\
     %!"
    (List.length windows)
    (List.count windows ~f:(fun (_, _, ms) -> Float.is_nan ms))
    report.timings_contended;
  (run ctx routine, attempts, windows, report, quiet, reached)

(* The search cut to its first candidate, the seeded plan: every other window refused. *)
let cut_search () =
  scripted_search ~reading:(fun ~index:_ ~label ms -> if is_seeded label then ms else Float.nan)

(* The seeded plan's first window refused, as contention would refuse it, and every other candidate
   scripted far slower than any real reading of this fixture: only a retry of the plan can crown
   it. *)
let refused_seed_search () =
  let seeded_seen = ref 0 in
  scripted_search ~reading:(fun ~index:_ ~label ms ->
      if not (is_seeded label) then 1e6
      else (
        Int.incr seeded_seen;
        if !seeded_seen = 1 then Float.nan else ms))

let () =
  let (w_untuned, b_untuned), untuned_digest, choice = untuned () in
  let claim label evaluate = Verdict.gate ~when_:seeds_here ~on:backend_name label evaluate in
  claim "on CUDA/HIP the untuned plan seeds a contraction" (fun () ->
      p "on CUDA/HIP the untuned plan seeds a contraction"
        (match choice with Some c -> not (List.is_empty c.Autotune.sd_seeded) | None -> false));
  Verdict.gate ~when_:splits_here ~on:backend_name
    "where the untuned compile splits (CUDA and HIP), the plan splits a reduction" (fun () ->
      p "where the untuned compile splits (CUDA and HIP), the plan splits a reduction"
        (match choice with Some c -> not (List.is_empty c.Autotune.sd_split) | None -> false));
  Verdict.gated ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the untuned compile is the plain default" (Option.is_none untuned_digest);
  Verdict.gate ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the search derives no seeded plan" (fun () ->
      (* The search's own derivation, on the lowering the plain default compiles: no plan. *)
      let plan = ref None in
      let ctx = Context.auto () in
      let limits = Context.hardware_limits ctx in
      let accum_prec = (Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec in
      ignore
        (Context.compile
           ~lowered_transform:(fun opt ->
             plan :=
               Some
                 (Autotune.seeded_plan ~backend:backend_name ~limits ~accum_prec ~static_indices opt);
             [ opt ])
           ctx comp Ir.Indexing.Empty
          : Context.t * Context.routine);
      p "off CUDA/HIP the search derives no seeded plan"
        (match !plan with Some None -> true | Some (Some _) | None -> false));
  let quiet_host = "a host that refused a window of the seeded plan"
  and quiet_note label quiet =
    if not quiet then
      Stdio.eprintf "%s: the host refused a seeded window, so the scripted claims skip\n%!" label
  in
  let tuned = if seeds_here then Some (cut_search ()) else None in
  (* [f] gets the search; [quiet] gates the claims that read its windows. *)
  let with_tuned ?(scripted = true) label f =
    claim label (fun () ->
        match tuned with
        | None -> p (label ^ " (no search ran)") false
        | Some ((_, _, _, _, quiet, _) as t) ->
            quiet_note "cut search" quiet;
            gated ~aggregation:`Environment ~when_:((not scripted) || quiet) ~on:quiet_host label
              (f t))
  in
  let seeded_windows windows =
    List.filter windows ~f:(fun (label, _, ms) -> is_seeded label && not (Float.is_nan ms))
  in
  with_tuned ~scripted:false "the search's first candidate is the seeded untuned plan"
    (fun (_, attempts, _, _, _, _) ->
      Option.iter (List.hd attempts) ~f:(fun label ->
          Stdio.eprintf "first candidate: %s (not part of the golden)\n%!" label);
      match attempts with label :: _ :: _ -> is_seeded label | _ -> false);
  with_tuned ~scripted:false "every attempt of the seeded plan compiled and reached a timing window"
    (fun (_, _, _, _, _, reached) -> reached);
  with_tuned "its timed code is the untuned compile's, segment for segment"
    (fun (_, _, windows, _, _, _) ->
      match (seeded_windows windows, untuned_digest) with
      | [ (_, digest, _) ], Some d -> String.equal digest d
      | _ -> false);
  with_tuned "the search went on to time later candidates, every one of them refused"
    (fun (_, _, windows, _, _, _) ->
      List.exists windows ~f:(fun (label, _, _) -> not (is_seeded label))
      && List.for_all windows ~f:(fun (label, _, ms) -> is_seeded label || Float.is_nan ms));
  with_tuned "the search crowned the seeded plan, its reading the report's seeded_ms and best_ms"
    (fun (_, _, windows, r, _, _) ->
      match seeded_windows windows with
      | [ (label, _, ms) ] ->
          String.equal r.Autotune.best_label label
          && Option.equal Float.equal r.seeded_ms (Some ms)
          && Float.equal r.best_ms ms
      | _ -> false);
  (let label = "the crowned gradients are bitwise the untuned compile's" in
   claim label (fun () ->
       match tuned with
       | None -> p (label ^ " (no search ran)") false
       | Some ((w_tuned, b_tuned), _, _, _, quiet, _) ->
           Verdict.gate ~aggregation:`Environment ~when_:quiet ~on:quiet_host label (fun () ->
               p_all2 label (Array.append w_tuned b_tuned)
                 (Array.append w_untuned b_untuned)
                 ~f:bitwise)));
  (* Only the split reassociates anything: where the plan does not split, it is bitwise the default
     pipeline by construction, and the digest claim is what tells the two apart. *)
  Verdict.gate ~when_:splits_here ~on:backend_name
    "the control: the plain default pipeline's bias gradient differs in some cell" (fun () ->
      let ctx, routine = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
      let _, b_default = run ctx routine in
      p_exists "the control: the plain default pipeline's bias gradient differs in some cell"
        (List.zip_exn (Array.to_list b_default) (Array.to_list b_untuned))
        ~f:(fun (a, b) -> not (bitwise a b)));
  let retried = if seeds_here then Some (refused_seed_search ()) else None in
  let with_retried label f =
    claim label (fun () ->
        match retried with
        | None -> p (label ^ " (no search ran)") false
        | Some ((_, _, _, _, quiet, _) as t) ->
            quiet_note "refused-seed search" quiet;
            gated ~aggregation:`Environment ~when_:quiet ~on:quiet_host label (f t))
  in
  claim "in the refused-seed search too, every seeded attempt compiled and was timed" (fun () ->
      p "in the refused-seed search too, every seeded attempt compiled and was timed"
        (match retried with Some (_, _, _, _, _, reached) -> reached | None -> false));
  with_retried "a refused seeded window is retried after the seed pass"
    (fun (_, _, windows, _, _, _) ->
      match List.filter windows ~f:(fun (label, _, _) -> is_seeded label) with
      | [ (_, _, first); (_, _, retry) ] -> Float.is_nan first && Float.(retry < 1e6)
      | _ -> false);
  with_retried "the retried seeded plan is crowned and reported as seeded_ms"
    (fun (_, _, windows, r, _, _) ->
      match seeded_windows windows with
      | [ (label, _, ms) ] ->
          String.equal r.Autotune.best_label label && Option.equal Float.equal r.seeded_ms (Some ms)
      | _ -> false)

(* gh-ocannl-1253: a search that times nothing -- every window refused, the seeded plan's retry
   included -- falls back to the untuned compile the caller would have shipped, which on CUDA/HIP is
   the seeded plan, not the plain default pipeline: its gradients are bitwise the untuned compile's
   (on CUDA the control above shows the plain default differs in the bias gradient). *)
let () =
  let (w_untuned, b_untuned), _, _ = untuned () in
  let label = "a search that times nothing returns the untuned compile's gradients, bitwise" in
  Verdict.gate ~when_:seeds_here ~on:backend_name label (fun () ->
      let (w_fallback, b_fallback), _, windows, r, _, _ =
        scripted_search ~reading:(fun ~index:_ ~label:_ _ -> Float.nan)
      in
      (* One claim, so the golden reads the same on and off CUDA/HIP: its population is empty -- and
         fails the non-emptiness floor -- unless the search refused every window and timed nothing,
         the precondition the fallback claim is about. *)
      let refused = List.count windows ~f:(fun (_, _, ms) -> Float.is_nan ms) in
      Stdio.eprintf
        "all-refused search (not part of the golden): %d of %d windows refused, best_ms %g\n%!"
        refused (List.length windows) r.Autotune.best_ms;
      let nothing_timed =
        refused > 0 && refused = List.length windows && Float.is_inf r.Autotune.best_ms
      in
      p_all label
        (if nothing_timed then
           List.zip_exn
             (Array.to_list (Array.append w_fallback b_fallback))
             (Array.to_list (Array.append w_untuned b_untuned))
         else [])
        ~f:(fun (a, b) -> bitwise a b))

(* gh-ocannl-1441: a [tune] call that does not search ships the untuned compile the recipes ship,
   the seeded plan on CUDA/HIP -- not the plain default pipeline it compiled before. Two of its
   paths: the search off with no cache (no base compile at all), and the search off with a chosen
   cache that misses (after the base compile). Values, bitwise, against the untuned compile: where
   the plan splits, the control above shows the plain default computes the bias gradient
   differently, so a plain-default fallback fails these claims. *)
let () =
  let (w_untuned, b_untuned), _, _ = untuned () in
  let untuned_values = Array.append w_untuned b_untuned in
  let searchless ~cache_dir =
    Scripted_search.clear_cache_dir cache_dir;
    let report = ref None in
    let ctx, routine =
      Autotune.tune ~search:false ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ~log:false
        ~report:(fun r -> report := Some r)
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    let w, b = run ctx routine in
    (Array.append w b, !report)
  in
  let leg label ~cache_dir =
    Verdict.gate ~when_:seeds_here ~on:backend_name label (fun () ->
        let values, report = searchless ~cache_dir in
        let disabled =
          match report with
          | Some r -> Poly.equal r.Autotune.outcome Autotune.Search_disabled
          | None -> false
        in
        p_all label
          (if disabled then List.zip_exn (Array.to_list values) (Array.to_list untuned_values)
           else [])
          ~f:(fun (a, b) -> bitwise a b))
  in
  leg "a search-less tune with no cache ships the untuned compile's gradients, bitwise"
    ~cache_dir:"";
  leg "a search-less tune whose cache misses ships the untuned compile's gradients, bitwise"
    ~cache_dir:"autotune_cache_seeded_searchless"

(* gh-ocannl-1441: two seeds against one trajectory. The seeded plan's digest is in the trajectory
   identity ([trajectory.search_shape], gh-ocannl-1217), so a record a search stored under one seed
   is not an equal-depth record for a search under another: it must neither replay that search's
   abandonment nor carry its steps. No setting changes the seed alone (the keys that steer the plan
   render in the trajectory's settings themselves), so seed B is the plan replaced by none through
   [Autotune.on_seeded_plan] -- a real seed state, the one a search takes where the plan is the
   default pipeline -- on the same computation, cache key and settings as seed A, the plan the
   search derives. The windows are scripted, so the abandonment rule and the stored records do not
   depend on the machine.

   The replay claims rest on the cache's stores and lookups, which the filesystem may refuse
   (gh-ocannl-1040) and which a context without a timing identity never makes: each is waived as an
   environment skip where its sequence's cache I/O was refused or its contexts were unkeyed
   ([Test_utils.Cache_plant]), and one claim no refusal waives says not every store was refused. *)
let () =
  let seed_b f = Autotune.Seam.with_value Autotune.on_seeded_plan (fun _ -> None) f in
  (* One call: what it did, its report, the windows it timed, its cache I/O, and whether its own
     context had a timing identity to key that I/O on. *)
  let tuned ?abandon ~cache_dir ~seed () =
    let report = ref None in
    let ctx = Context.auto () in
    let identity = Option.is_some (Context.timing_identity ctx) in
    let (outcome, ios), record =
      Scripted_search.scripted
        ~measure:(fun w -> match abandon with Some _ -> 100.0 | None -> w.reading)
        (fun () ->
          let body () =
            SC.recording_cache_io (fun () ->
                match
                  Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
                    ~timing:Autotune.Isolated ~cache_dir ?abandon ~log:false
                    ~report:(fun r -> report := Some r)
                    ctx comp Ir.Indexing.Empty
                with
                | ctx, _routine ->
                    Context.release ctx;
                    `Completed
                | exception Autotune.Search_abandoned _ -> `Abandoned)
          in
          match seed with `A -> body () | `B -> seed_b body)
    in
    ((outcome, !report, List.length record.windows), (ios, identity))
  in
  let outcome_is ~f = function Some r -> f r.Autotune.outcome | None -> false in
  let abandon =
    { Autotune.incumbent_steps = [ (1, 1.0) ]; trailing_ratio = Autotune.flip_abandon_ratio () }
  in
  let cache_dir = "autotune_cache_seeded_trajectory" in
  (* Each sequence runs once, where it is first claimed on; off CUDA/HIP never. Each returns its
     three calls, then the cache I/O of all three and whether every one of them was keyed. *)
  let sequence ?abandon () =
    Scripted_search.clear_cache_dir cache_dir;
    let first, (ios1, id1) = tuned ?abandon ~cache_dir ~seed:`A () in
    let again, (ios2, id2) = tuned ?abandon ~cache_dir ~seed:`A () in
    let other, (ios3, id3) = tuned ?abandon ~cache_dir ~seed:`B () in
    let timed (_, _, n) = n in
    Stdio.eprintf "%s legs (not part of the golden): timed %d, %d (same seed), %d (other seed)\n%!"
      (match abandon with Some _ -> "abandonment" | None -> "crown")
      (timed first) (timed again) (timed other);
    ((first, again, other), (ios1 @ ios2 @ ios3, id1 && id2 && id3))
  in
  let abandonments = lazy (sequence ~abandon ()) in
  let crowns = lazy (sequence ()) in
  (* [~cached:true]: the claim rests on the sequence's cache I/O, so it is waived where that I/O was
     refused or unkeyed. *)
  let seed_claim ?(cached = false) label legs f =
    Verdict.gate ~when_:seeds_here ~on:backend_name label (fun () ->
        let calls, (ios, identity) = Lazy.force legs in
        if cached then Cache_plant.unless_unkeyed_or_refused ~identity ios label (f calls)
        else p label (f calls))
  in
  let abandoned_live = function Autotune.Abandoned _ -> true | _ -> false in
  seed_claim "under seed A the search is abandoned after its first timed candidates" abandonments
    (fun ((outcome, r, timed), _, _) ->
      Poly.equal outcome `Abandoned && timed = 2 && outcome_is r ~f:abandoned_live);
  seed_claim ~cached:true "control: seed A again replays that abandonment without timing anything"
    abandonments (fun (_, (outcome, r, timed), _) ->
      Poly.equal outcome `Abandoned && timed = 0
      && outcome_is r ~f:(function Autotune.Abandonment_replay _ -> true | _ -> false));
  seed_claim ~cached:true
    "under seed B the same trajectory searches rather than replay seed A's abandonment" abandonments
    (fun (_, _, (outcome, r, timed)) ->
      Poly.equal outcome `Abandoned && timed = 2 && outcome_is r ~f:abandoned_live);
  let steps r = Option.value_map r ~default:[] ~f:(fun r -> r.Autotune.best_steps) in
  let replayed r = outcome_is r ~f:(Poly.equal Autotune.Cache_replay) in
  let same_step (a, x) (b, y) = a = b && Float.equal x y in
  seed_claim "under seed A the search completes and records its steps" crowns
    (fun ((outcome, r, _), _, _) ->
      Poly.equal outcome `Completed
      && outcome_is r ~f:(Poly.equal Autotune.Searched)
      && not (List.is_empty (steps r)));
  seed_claim ~cached:true "control: seed A again replays the crown with its steps" crowns
    (fun ((_, searched, _), (outcome, r, timed), _) ->
      Poly.equal outcome `Completed && replayed r && timed = 0
      && (not (List.is_empty (steps searched)))
      && List.equal same_step (steps r) (steps searched));
  (* The record a replay carries is seed A's exactly when it equals A's non-empty steps, as in the
     control above; a replay under seed B that did not replay, or a seed A search that recorded
     nothing, fails. *)
  seed_claim ~cached:true "under seed B the crown replays without seed A's steps" crowns
    (fun ((_, searched, _), _, (outcome, r, timed)) ->
      Poly.equal outcome `Completed && replayed r && timed = 0
      && (not (List.is_empty (steps searched)))
      && not (List.equal same_step (steps r) (steps searched)));
  (* Not waived by a refusal: of the stores the sequences attempted, not every one was refused. Each
     keyed sequence is bound to attempt two: seed A's record and seed B's, both under the one
     key. *)
  let label = "not every store of the two-seed sequences was refused" in
  Verdict.gate ~when_:seeds_here ~on:backend_name label (fun () ->
      let _, (ios_a, id_a) = Lazy.force abandonments and _, (ios_c, id_c) = Lazy.force crowns in
      Cache_plant.not_every_store_refused ~expected:(id_a && id_c) label (ios_a @ ios_c))
