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

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* gh-ocannl-1253: the seeding gate is pinned at threshold 32, the default before [auto]. Every
   claim below needs a plan that seeds, and the ranked [auto] gate seeds nothing on this fixture on
   HIP, even with every working set past the cache: whether a sketch beats the preset is its call,
   while what this test checks -- the search seeds the plan the untuned compile ships, whatever the
   gate resolved it to -- is the gate's to feed, not to decide. The untuned compile and the search
   read the same setting, set before the first compile reads it. *)
let () = Unix.putenv "OCANNL_GPU_DEFAULT_SKETCH_MIN_REDUCTION" "32"

(* What the untuned compile promises here, read off the predicates it consults: seeding on CUDA and
   HIP, the split on CUDA alone (gh-ocannl-1325: HIP keeps the unsplit plan). *)
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
  let windows = ref [] and attempts = ref [] and timed = ref [] and report = ref None in
  let old_measured = Autotune.Seam.get Autotune.on_candidate_measured
  and old_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  let ctx, routine =
    Exn.protect
      ~finally:(fun () ->
        Autotune.Seam.set Autotune.on_candidate_measured old_measured;
        Autotune.Seam.set Autotune.on_candidate_attempt old_attempt)
      ~f:(fun () ->
        ( Autotune.Seam.set Autotune.on_candidate_attempt @@ fun label ->
          if not (String.equal label "baseline") then attempts := label :: !attempts );
        (* Each timing call that returns, under the attempt it timed: a window the host refused
           still returns one, a candidate that failed before timing never does. *)
        Autotune.with_timing_call_observer (fun (_ : Autotune.timing_call) ->
            Option.iter (List.hd !attempts) ~f:(fun label -> timed := label :: !timed))
        @@ fun () ->
        ( Autotune.Seam.set Autotune.on_candidate_measured @@ fun ~label ~digest ms ->
          let ms = reading ~index:(List.length !windows) ~label ms in
          windows := (label, digest, ms) :: !windows;
          ms );
        Autotune.with_uncontended_test_windows (fun () ->
            Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~seed_block_sizes:[]
              ~cache_dir:"" ~log:false
              ~report:(fun r -> report := Some r)
              (Context.auto ()) comp Ir.Indexing.Empty))
  in
  let report = Option.value_exn ~here:[%here] !report in
  let windows = List.rev !windows in
  (* Every attempt of the seeded plan must compile and reach a timing window ([reached], a claim: a
     candidate failure is a failure), and a window the host refuses (queued calibration that
     measured no batch, routine on a shared box) never reaches the seam: what the window claims read
     is the seeded plan's, so they are gated on the host having refused none of its windows
     ([quiet]). *)
  let attempts = List.rev !attempts in
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
    "where the untuned compile splits (CUDA), the plan splits a reduction" (fun () ->
      p "where the untuned compile splits (CUDA), the plan splits a reduction"
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
